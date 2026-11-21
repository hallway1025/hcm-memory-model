"""fastpath：部署/规模场景的 HCM 情节存储加速路径（opt-in，非 canonical）。

依据（bench/exp43_inverted_index{,_scale}.json，2026-11-09）：
  Part0  增量倒排镜像 vs canonical 稠密 Mn：write/reject 判定 0 hard mismatch
         （tie 全在 |dot-theta|<1e-5 ulp 平带）、d_a ~1e-9 —— 数值等价。
  Part2  N=1e5, d_h=16384, k=328：写门 197x（3.45ms vs 0.680s/条）、
         内存 7.1x（索引 512MB+CSR 372MB vs 稠密 6250MB）、
         f32 链检索 |Delta a|=2e-9 等价且 L=3 单 query 1.16s（canonical ~213s）。

为什么 opt-in：canonical 链的数值被论文 claims 钉死（check_claims 327/327），
softmax_beta 的 float64 上卷是冻结数值的一部分；本模块不改动 src/hcm 现有文件，
供部署与规模运行（C2 N>=1e6、A3 GPU 栈）选择使用。

限制：append-only（不支持 prune 列压缩——prune 会破坏增量索引，exp43 口径同）；
仅实现 write/max_cos/retrieve/scores 的热路径；元数据表（t/g/r/nu/sup/cid/events）
按需保留最小集（t/sup），不复刻 EpisodicStore 全部接口。
"""
from __future__ import annotations
import numpy as np
from .ops import unit


def softmax32(z, beta):
    """softmax_beta 的 float32 保真版（ops.softmax_beta 强制 f64 上卷，R1 修复对象）。"""
    z = np.asarray(z, dtype=np.float32) * np.float32(beta)
    z = z - z.max()
    e = np.exp(z)
    return e / (e.sum() + np.float32(1e-300))


class FastEpisodic:
    """倒排镜像 + f32 Hopfield 链的情节快存（append-only）。"""

    def __init__(self, d_h, capacity, k, beta=25.0, theta_dup=0.95, row_cap=4096):
        self.d_h, self.capacity, self.k = int(d_h), int(capacity), int(k)
        self.beta = float(beta)
        self.theta_dup = float(theta_dup)
        self.row_cap = int(row_cap)
        self.rowmat = np.zeros((self.d_h, self.row_cap), dtype=np.int32)
        self.rowval = np.zeros((self.d_h, self.row_cap), dtype=np.float32)
        self.rowlen = np.zeros(self.d_h, dtype=np.int64)
        self.t = np.zeros(capacity, dtype=np.float64)
        self.sup = np.ones(capacity, dtype=np.float64)
        self.n = 0
        self.n_write_attempts = 0
        self.n_dup_rejected = 0
        self._csr = None

    # ---------- 索引 ----------
    def _grow(self):
        self.row_cap *= 2
        rm = np.zeros((self.d_h, self.row_cap), dtype=np.int32)
        rv = np.zeros((self.d_h, self.row_cap), dtype=np.float32)
        rm[:, : self.row_cap // 2] = self.rowmat
        rv[:, : self.row_cap // 2] = self.rowval
        self.rowmat, self.rowval = rm, rv

    def _add(self, c, h):
        nz = np.flatnonzero(h)
        if int(self.rowlen[nz].max()) + 1 > self.row_cap:
            self._grow()
        for r in nz:
            j = int(self.rowlen[r])
            self.rowmat[r, j] = c
            self.rowval[r, j] = h[r]
            self.rowlen[r] = j + 1
        self.n = max(self.n, c + 1)
        self._csr = None

    def _max_dot(self, h):
        if self.n == 0:
            return -1.0
        nz = np.flatnonzero(h)
        cs, vs = [], []
        for r in nz:
            L = int(self.rowlen[r])
            if L:
                cs.append(self.rowmat[r, :L])
                vs.append(self.rowval[r, :L] * h[r])
        if not cs:
            return 0.0
        c = np.concatenate(cs)
        v = np.concatenate(vs)
        acc = np.bincount(c, weights=v.astype(np.float64), minlength=self.n)
        return float(acc.max())

    def _build_csr(self):
        if self._csr is not None:
            return self._csr
        n = int(self.rowlen.sum())
        pr = np.empty(n, dtype=np.int32)
        pv = np.empty(n, dtype=np.float32)
        pc = np.empty(n, dtype=np.int32)
        off = 0
        for r in range(self.d_h):
            L = int(self.rowlen[r])
            if L:
                pr[off:off + L] = r
                pv[off:off + L] = self.rowval[r, :L]
                pc[off:off + L] = self.rowmat[r, :L]
                off += L
        self._csr = (pr, pv, pc)
        return self._csr

    # ---------- 热路径 ----------
    def max_cos(self, h):
        return self._max_dot(h)

    def write(self, h, t, sup=1.0):
        self.n_write_attempts += 1
        if self._max_dot(h) >= self.theta_dup:
            self.n_dup_rejected += 1
            return -1
        if self.n >= self.capacity:
            return -1
        i = self.n
        self.t[i] = t
        self.sup[i] = sup
        self._add(i, np.asarray(h, dtype=np.float32))
        return i

    def _mtv(self, v):
        pr, pv, pc = self._build_csr()
        return np.bincount(pc, weights=v[pr] * pv, minlength=self.n)

    def _mt_cols(self, a):
        pr, pv, pc = self._build_csr()
        return np.bincount(pr, weights=a[pc] * pv, minlength=self.d_h)

    def retrieve(self, q=None, L=3, beta=None, dense=None, ebias=None, f32=True):
        """现代 Hopfield 迭代（语义同 EpisodicStore.retrieve）。f32=True 全链 float32。"""
        beta = self.beta if beta is None else beta
        sm = softmax32 if f32 else (lambda z, b: _softmax64(z, b))
        if self.n == 0:
            return q.copy(), np.zeros(0)
        z = unit(np.maximum(dense, 0.0)) if dense is not None else unit(q)
        e = 0.0 if ebias is None else ebias
        a = sm(self._mtv(z) + e, beta)
        for _ in range(max(0, int(L))):
            z = unit(self._mt_cols(a))
            a = sm(self._mtv(z) + e, beta)
            sa = a.sum()
            if sa > 0:
                a = a / sa
        return z, a

    def scores(self, h, sup=True):
        nz = np.flatnonzero(h)
        cs, vs = [], []
        for r in nz:
            L = int(self.rowlen[r])
            if L:
                cs.append(self.rowmat[r, :L])
                vs.append(self.rowval[r, :L] * h[r])
        out = np.zeros(self.n, dtype=np.float64)
        if cs:
            c = np.concatenate(cs)
            v = np.concatenate(vs)
            out += np.bincount(c, weights=v.astype(np.float64), minlength=self.n)
        if sup:
            out *= self.sup[: self.n]
        return out


def _softmax64(z, beta):
    z = np.asarray(z, dtype=np.float64) * beta
    z = z - z.max()
    e = np.exp(z)
    return e / (e.sum() + 1e-300)


def selftest(n=2000, d_h=1024, k=20, seed=0):
    """Part0 口径的等价性自检：与 canonical EpisodicStore 逐项对拍。"""
    from .episodic import EpisodicStore
    from .ops import unit, relu_kwta
    rng = np.random.default_rng(seed)
    E = EpisodicStore(d_h, n + 10, k, beta=25.0, seed=seed)
    F = FastEpisodic(d_h, n + 10, k, beta=25.0)
    n_tie = 0
    for c in range(n):
        rows = rng.choice(d_h, size=k, replace=False)
        h = np.zeros(d_h, dtype=np.float32)
        h[rows] = 1.0 / np.sqrt(k)
        h = unit(h)
        si = E.write(h, t=float(c), g=1.0, theta_dup=0.95)
        fi = F.write(h, t=float(c))
        if si >= 0:
            F.sup[int(si)] = 1.0
        if (si < 0) != (fi < 0):
            de = abs(float(np.max(E.M[:, :E.n].T @ h)) - 0.95)
            dm = abs(F._max_dot(h) - 0.95)
            if min(de, dm) < 1e-5:
                n_tie += 1
            else:
                return {"ok": False, "reason": "hard mismatch at c=%d" % c, "seed": seed}
    d_a = 0.0
    for _ in range(20):
        d0 = np.abs(unit(rng.standard_normal(d_h).astype(np.float32)))
        q = unit(relu_kwta(d0, k))
        _, a_e = E.retrieve(q, L=3, beta=25.0, dense=d0)
        _, a_f = F.retrieve(q, L=3, beta=25.0, dense=d0, f32=False)
        d_a = max(d_a, float(np.max(np.abs(a_e - a_f))))
        _, a_f32 = F.retrieve(q, L=3, beta=25.0, dense=d0, f32=True)
        d_a = max(d_a, float(np.max(np.abs(a_e - a_f32))))
    return {"ok": True, "n": n, "ties": n_tie, "max_da": d_a, "seed": seed,
            "equiv_ok": bool(d_a < 1e-4)}
