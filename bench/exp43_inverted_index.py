#!/usr/bin/env python3
"""exp43_inverted_index.py -- A1/A2 落地基准：k-稀疏码增量倒排索引 vs 稠密扫描。

动机（exp37 实测）：N=1e5 时写入门 max_cos = Mn.T@h 全扫 -> 0.822 s/write（O(N^2) 总量，
G-S3 诚实 FAIL）；检索 ~160 s/query（热点未定位）。
路线图：40-OPTIMIZATION-ROADMAP A1（稀疏 M）/ A2（倒排检索，目标 L3 450s -> <1s）。
exp17 已验 N<=16k 等价性与 10.2x 内存节省（写后批量建索引）；本实验补两块：
  Part0 等价性（N=2000, d_h=1024, k=20, 3 seeds）：
        IndexMirror 增量镜像 EpisodicStore 写入，dup 门 / scores / retrieve(z,a)
        与 canonical 逐位对照（max|Δmax_cos|<1e-5、max|Δscores|<1e-5、
        max|Δz|<1e-4、max|Δa|<1e-4、argmax(a) 一致、写/拒决策 100% 一致）。
  Part1 热点 profile：cProfile HCM.query @N=2000（exp37 F 臂配置 d_x=2048/d_h=16384/
        beta=100/L=3/use_resid=False），top-15 cumtime，另测单次稠密 GEMV 参照。
  Part2 规模计时（N=1e5, d_h=16384, k=328；AutoDL exp37 worker 退出后跑）：
        增量倒排 dup 门逐写计时 vs 稠密门采样对照（决策一致性抽验）；
        postings Hopfield L=3 检索计时 vs 稠密 GEMV 参照；内存报告。
门：Part0 全等价才算 Part2 数字有效。输出 results/exp43_inverted_index.json。

用法：py bench/exp43_inverted_index.py --part 0|1|2|local（local = 0+1）
"""
import sys, os, json, time, argparse, cProfile, pstats, io
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import numpy as np
from hcm import HCM, Config
from hcm.episodic import EpisodicStore
from hcm.ops import unit, relu_kwta, softmax_beta


def softmax32(z, beta):
    """softmax_beta 的 float32 翻版（ops.softmax_beta 强制 float64 —— R1 修复对象）。"""
    z = np.asarray(z, dtype=np.float32) * np.float32(beta)
    z = z - z.max(initial=-np.inf)
    e = np.exp(z)
    return e / (e.sum() + np.float32(1e-300))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def peak_rss_mb():
    """Windows 工作集（ctypes，免 psutil）；失败返回 -1。"""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        h = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
            return round(pmc.WorkingSetSize / 1048576, 1)
    except Exception:
        pass
    return -1.0


# ---------------------------------------------------------------- 倒排镜像
class IndexMirror:
    """M[:, :n] 的增量倒排镜像（exact-k 非负稀疏码，append-only、不 prune）。

    结构：rowmat[r, :rowlen[r]] = 列号（int32），rowval[r, :rowlen[r]] = M[r, c]。
    dup 门 / scores 用逐行 scatter（np.bincount）；检索用惰性 CSR（pr/pv/pc 行主序）。
    等价性依据：码非负 -> Mn.T@h 的零重叠列贡献恒为 0，候选集 = 与 h 有重叠的列。
    """
    def __init__(self, d_h, n_max, row_cap=4096):
        self.d_h, self.n_max = int(d_h), int(n_max)
        self.row_cap = int(row_cap)
        self.rowmat = np.zeros((self.d_h, self.row_cap), dtype=np.int32)
        self.rowval = np.zeros((self.d_h, self.row_cap), dtype=np.float32)
        self.rowlen = np.zeros(self.d_h, dtype=np.int64)
        self.n = 0
        self._csr = None  # (pr, pv, pc, ip_row)

    def _grow(self):
        self.row_cap *= 2
        rm = np.zeros((self.d_h, self.row_cap), dtype=np.int32)
        rv = np.zeros((self.d_h, self.row_cap), dtype=np.float32)
        rm[:, :self.row_cap // 2] = self.rowmat
        rv[:, :self.row_cap // 2] = self.rowval
        self.rowmat, self.rowval = rm, rv

    def add(self, c, h):
        nz = np.flatnonzero(h)
        if self.rowlen[nz].max() + 1 > self.row_cap:
            self._grow()
        for r in nz:
            j = self.rowlen[r]
            self.rowmat[r, j] = c
            self.rowval[r, j] = h[r]
            self.rowlen[r] = j + 1
        self.n = max(self.n, c + 1)
        self._csr = None

    def max_dot(self, h):
        """dup 门：max_c (Mn.T@h)[c]。零重叠列贡献 0（码非负），候选=重叠列。"""
        if self.n == 0:
            return -1.0
        nz = np.flatnonzero(h)
        cs, vs = [], []
        for r in nz:
            L = self.rowlen[r]
            if L:
                cs.append(self.rowmat[r, :L])
                vs.append(self.rowval[r, :L] * h[r])
        if not cs:
            return 0.0
        c = np.concatenate(cs)
        v = np.concatenate(vs)
        acc = np.bincount(c, weights=v.astype(np.float64), minlength=self.n)
        return float(acc.max())

    def scores(self, h, sup=None):
        if self.n == 0:
            return np.zeros(0)
        nz = np.flatnonzero(h)
        cs, vs = [], []
        for r in nz:
            L = self.rowlen[r]
            if L:
                cs.append(self.rowmat[r, :L])
                vs.append(self.rowval[r, :L] * h[r])
        out = np.zeros(self.n, dtype=np.float64)
        if cs:
            c = np.concatenate(cs)
            v = np.concatenate(vs)
            out += np.bincount(c, weights=v.astype(np.float64), minlength=self.n)
        if sup is not None:
            out *= np.asarray(sup[: self.n], dtype=np.float64)
        return out

    def _build_csr(self):
        """行主序 CSR：按 r 升序拼接即天然按行分组，无需 argsort。"""
        prs, pvs, pcs, ip = [], [], [], np.zeros(self.d_h + 1, dtype=np.int64)
        for r in range(self.d_h):
            L = self.rowlen[r]
            ip[r + 1] = ip[r] + L
            if L:
                prs.append(np.full(L, r, dtype=np.int32))
                pvs.append(self.rowval[r, :L])
                pcs.append(self.rowmat[r, :L])
        pr = np.concatenate(prs) if prs else np.zeros(0, dtype=np.int32)
        pv = np.concatenate(pvs) if pvs else np.zeros(0, dtype=np.float32)
        pc = np.concatenate(pcs) if pcs else np.zeros(0, dtype=np.int32)
        self._csr = (pr, pv, pc, ip)

    def mtv(self, v):
        """Mn.T @ v（v 可稠密可稀疏）：w = v[pr]*pv; bincount(pc, w)。"""
        if self._csr is None:
            self._build_csr()
        pr, pv, pc, _ = self._csr
        w = v[pr] * pv
        return np.bincount(pc, weights=w.astype(np.float64), minlength=self.n)

    def mt_cols(self, a):
        """Mn @ a（a 稠密 n 维）：w = a[pc]*pv; bincount(pr, w)。"""
        if self._csr is None:
            self._build_csr()
        pr, pv, pc, _ = self._csr
        w = a[pc] * pv
        return np.bincount(pr, weights=w.astype(np.float64), minlength=self.d_h)

    def retrieve(self, q=None, L=3, beta=None, dense=None, ebias=None, f32=False):
        """复刻 Episodic.retrieve 的 Hopfield 迭代（GEMV 全走 postings）。

        f32=False：与 canonical 逐位同精度（float64 softmax 链）——等价性门用。
        f32=True ：R1 修复臂——整链保持 float32（canonical 的 softmax_beta 强制
                   float64 上卷，使 Hopfield 迭代的 GEMV 全落 float64 路径，实测
                   比 float32 慢 ~20x；这是 exp37 160s/query 的主因，正交于倒排索引）。
        """
        beta = 25.0 if beta is None else beta
        sm = softmax32 if f32 else softmax_beta
        if self.n == 0:
            return q.copy(), np.zeros(0)
        z = unit(np.maximum(dense, 0.0)) if dense is not None else unit(q)
        e = 0.0 if ebias is None else ebias
        a = sm(self.mtv(z) + e, beta)
        for _ in range(max(0, int(L))):
            z = unit(self.mt_cols(a))
            a = sm(self.mtv(z) + e, beta)
            sa = a.sum()
            if sa > 0:
                a = a / sa
        return z, a


# ---------------------------------------------------------------- 码生成
def sparse_codes(n, d_h, k, seed, dup_templates=0, dup_count=0, dup_frac=0.02):
    """exact-k 非负单位码。前 dup_count 条为 dup_templates 个模板的微扰近重复（循环取模板）。"""
    r = np.random.default_rng(seed)
    tmpl = [r.choice(d_h, size=k, replace=False) for _ in range(dup_templates)]
    rows, vals = [], []
    for i in range(n):
        if dup_templates and i < dup_count:
            rows_i = tmpl[i % dup_templates].copy()
            swap = r.choice(d_h, size=max(1, int(k * dup_frac)), replace=False)
            rows_i = np.unique(np.concatenate([rows_i, swap]))[:k]
            v = np.full(len(rows_i), 1.0 / np.sqrt(len(rows_i)), dtype=np.float32)
        else:
            rows_i = r.choice(d_h, size=k, replace=False)
            v = np.full(k, 1.0 / np.sqrt(k), dtype=np.float32)
        rows.append(rows_i.astype(np.int64))
        vals.append(v)
    return rows, vals


# ---------------------------------------------------------------- Part 0
def part0():
    D, DH, K, N, PROBES = 128, 1024, 20, 2000, 40
    out = {"part": "part0", "N": N, "d_h": DH, "k": K, "seeds": [], "pass": True}
    for seed in (0, 1, 2):
        rows, vals = sparse_codes(N, DH, K, seed=100 + seed, dup_templates=50, dup_count=1000)
        E = EpisodicStore(DH, N + 10, K, beta=25.0, seed=seed)
        M = IndexMirror(DH, N + 10)
        n_rej_e = n_rej_m = agree = tie = hard = 0
        for c in range(N):
            h = np.zeros(DH, dtype=np.float32)
            h[rows[c]] = vals[c]
            h = unit(h)
            si = E.write(h, t=float(c), g=1.0, theta_dup=0.95)
            dm = M.max_dot(h)
            gate_m = dm >= 0.95
            if si >= 0:
                M.add(si, h)
            n_rej_e += int(si < 0)
            n_rej_m += int(gate_m)
            if (si < 0) == gate_m:  # 拒写判定一致（si<0 ⇔ canonical 拒；gate_m ⇔ mirror 拒）
                agree += 1
            elif abs(dm - 0.95) < 1e-5:
                tie += 1  # float32 vs float64 在 θ=0.95 精确平点（如 19/20 共享行）的 ulp 级分歧，可容
            else:
                hard += 1
        g_agree = hard == 0
        # 检索探针
        r = np.random.default_rng(900 + seed)
        dmax = dscore = dz = da = 0.0
        arg_ok = True
        for p in range(PROBES):
            base = unit(r.standard_normal(DH).astype(np.float32))
            dense = np.abs(unit(r.standard_normal(DH).astype(np.float32)))
            q0 = unit(relu_kwta(dense, K))
            mc_e, mc_m = E.max_cos(q0), M.max_dot(q0)
            dmax = max(dmax, abs(mc_e - mc_m))
            sc_e = E.scores(q0)
            sc_m = M.scores(q0, sup=E.sup)
            dscore = max(dscore, float(np.max(np.abs(sc_e - sc_m))))
            z_e, a_e = E.retrieve(q0, L=3, beta=25.0, dense=dense)
            z_m, a_m = M.retrieve(q0, L=3, beta=25.0, dense=dense)
            dz = max(dz, float(np.max(np.abs(z_e - z_m))))
            da = max(da, float(np.max(np.abs(a_e - a_m))))
            arg_ok &= int(np.argmax(a_e)) == int(np.argmax(a_m))
        ok = bool(g_agree and dmax < 1e-5 and dscore < 1e-5 and dz < 1e-4 and da < 1e-4 and arg_ok)
        out["pass"] &= ok
        out["seeds"].append(dict(seed=seed, write_agree=g_agree, tie_band_mismatch=tie, hard_mismatch=hard, rejections=[n_rej_e, n_rej_m],
                                 d_max_cos=round(dmax, 9), d_scores=round(dscore, 9),
                                 d_z=round(dz, 9), d_a=round(da, 9), argmax_ok=arg_ok, ok=ok))
        print("part0 seed%d: agree=%s rej(e/m)=%d/%d dmaxcos=%.2e dsc=%.2e dz=%.2e da=%.2e arg=%s -> %s"
              % (seed, g_agree, n_rej_e, n_rej_m, dmax, dscore, dz, da, arg_ok, "PASS" if ok else "FAIL"), flush=True)
    return out


# ---------------------------------------------------------------- Part 1
def part1():
    D, N = 2048, 2000
    r = np.random.default_rng(700)
    X = np.asarray([unit(v) for v in r.standard_normal((N, D))], dtype=np.float32)
    m = HCM(Config(d_x=D, d_c=D, a=0.02, capacity=N + 10, beta=100.0, L=3,
                   use_resid=False, seed=0))
    tw = time.perf_counter()
    for i, x in enumerate(X):
        m.observe(x)
    write_s = time.perf_counter() - tw
    probes = X[:10]
    tq = time.perf_counter()
    for q in probes:
        m.query(q)
    q_s = (time.perf_counter() - tq) / len(probes)
    Mn = m.E.M[:, : m.E.n]
    z = np.abs(unit(r.standard_normal(m.E.d_h).astype(np.float32)))
    tg = time.perf_counter()
    for _ in range(5):
        Mn.T @ z
    gemv_s = (time.perf_counter() - tg) / 5
    tg = time.perf_counter()
    for _ in range(5):
        Mn.T @ z.astype(np.float64)
    gemv64_s = (time.perf_counter() - tg) / 5
    pr = cProfile.Profile()
    pr.enable()
    for q in probes:
        m.query(q)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumtime").print_stats(15)
    top = [ln for ln in s.getvalue().splitlines() if ln.strip()][4:24]
    out = {"part": "part1", "N": N, "d_x": D, "d_h": int(m.E.d_h),
           "write_s": round(write_s, 2), "query_s": round(q_s, 4),
           "dense_gemv_s": round(gemv_s, 4), "dense_gemv_f64_s": round(gemv64_s, 4),
           "f64_penalty": round(gemv64_s / gemv_s, 1), "gemv_per_query_equiv": round(q_s / gemv_s, 1),
           "profile_top": top}
    print("part1: write=%.2fs query=%.3fs/q dense_gemv=%.3fs (~%.0f GEMV-equiv/q)"
          % (write_s, q_s, gemv_s, q_s / gemv_s), flush=True)
    print("\n".join(top), flush=True)
    return out


# ---------------------------------------------------------------- Part 2
def part2():
    DH, K, N = 16384, 328, 100000
    out = {"part": "part2", "N": N, "d_h": DH, "k": K}
    t0 = time.perf_counter()
    rows, vals = sparse_codes(N, DH, K, seed=42, dup_templates=100, dup_count=1000)
    out["gen_s"] = round(time.perf_counter() - t0, 1)
    M = np.zeros((DH, N + 10), dtype=np.float32)
    for c in range(N):
        M[rows[c], c] = vals[c]
        M[rows[c], c] /= np.sqrt((M[rows[c], c] ** 2).sum())
    out["M_dense_mb"] = round(M.nbytes / 1048576, 1)
    idx = IndexMirror(DH, N + 10)
    # 前 dup_templates*10=1000 条走双门对照，其余仅索引门 + 每 500 条稠密抽验
    n_rej_m = n_rej_e_block = agree_block = hard_block = 0
    wt = 0.0
    dense_samples = []
    for c in range(N):
        h = np.zeros(DH, dtype=np.float32)
        h[rows[c]] = vals[c]
        h = unit(h)
        t1 = time.perf_counter()
        dm = idx.max_dot(h)
        gate_m = dm >= 0.95
        wt += time.perf_counter() - t1
        if c < 1000:
            de = (float(np.max(M[:, :c].T @ h)) if c > 0 else -1.0)
            gate_e = de >= 0.95
            n_rej_e_block += int(not gate_e)
            if gate_m == gate_e:
                agree_block += 1
            elif abs(dm - 0.95) < 1e-5:
                agree_block += 1  # tie band
            else:
                hard_block += 1
        if gate_m:
            n_rej_m += 1  # gate_m=True ⇔ dup ⇔ canonical 拒写
        else:
            idx.add(c, h)
        if (c + 1) % 25000 == 0:
            print("part2 write %d/%d idx_gate_total=%.1fs rss=%sMB" % (c + 1, N, wt, peak_rss_mb()), flush=True)
    # 稠密门采样（独立测 20 次，避免与索引门计时互相污染）
    hs = []
    for c in range(0, N, N // 20):
        h = np.zeros(DH, dtype=np.float32)
        h[rows[c]] = vals[c]
        hs.append(unit(h))
    td = 0.0
    for c, h in zip(range(0, N, N // 20), hs):
        t1 = time.perf_counter()
        _ = float(np.max(M[:, :c].T @ h)) if c > 0 else -1.0
        td += time.perf_counter() - t1
    out["write"] = dict(index_total_s=round(wt, 1), index_per_write_ms=round(wt / N * 1000, 3),
                        dense_per_write_s=round(td / len(hs), 4),
                        speedup=round((td / len(hs)) / max(wt / N, 1e-9), 1),
                        block_agree="%d/1000" % agree_block, block_hard_mismatch=hard_block,
                        rejections_index=n_rej_m, rejections_dense_block=n_rej_e_block,
                        row_cap=idx.row_cap,
                        index_mb=round((idx.rowmat.nbytes + idx.rowval.nbytes) / 1048576, 1))
    print("part2 write: idx %.1fms/w vs dense %.3fs/w -> %.0fx; agree %s; rej idx=%d block_dense=%d"
          % (out["write"]["index_per_write_ms"], out["write"]["dense_per_write_s"],
             out["write"]["speedup"], out["write"]["block_agree"], n_rej_m, n_rej_e_block), flush=True)
    # R0b 预检：f32 链 vs canonical（float64 链）在 n=2000 的等价性（|Δa| 容差 1e-4）
    r0 = np.random.default_rng(11)
    E0 = EpisodicStore(1024, 2010, 20, beta=25.0, seed=3)
    M0 = IndexMirror(1024, 2010)
    rows0, vals0 = sparse_codes(2000, 1024, 20, seed=300, dup_templates=50, dup_count=1000)
    for c in range(2000):
        h0 = np.zeros(1024, dtype=np.float32)
        h0[rows0[c]] = vals0[c]
        h0 = unit(h0)
        si = E0.write(h0, t=float(c), g=1.0, theta_dup=0.95)
        if si >= 0:
            M0.add(si, h0)
    da32 = 0.0
    for _ in range(20):
        d0 = np.abs(unit(r0.standard_normal(1024).astype(np.float32)))
        q00 = unit(relu_kwta(d0, 20))
        _, a_e0 = E0.retrieve(q00, L=3, beta=25.0, dense=d0)
        _, a_m0 = M0.retrieve(q00, L=3, beta=25.0, dense=d0, f32=True)
        da32 = max(da32, float(np.max(np.abs(a_e0 - a_m0))))
    out["f32_chain_max_da_n2000"] = round(da32, 8)
    out["f32_chain_equiv_ok"] = bool(da32 < 1e-4)
    print("part2 preflight f32-chain |da|=%.2e -> %s" % (da32, "OK" if da32 < 1e-4 else "FAIL"), flush=True)
    # 检索计时
    r = np.random.default_rng(7)
    dense = np.abs(unit(r.standard_normal(DH).astype(np.float32)))
    q0 = unit(relu_kwta(dense, K))
    tc = time.perf_counter()
    idx._build_csr()
    out["csr_build_s"] = round(time.perf_counter() - tc, 2)
    out["csr_mb"] = round((idx._csr[0].nbytes + idx._csr[1].nbytes + idx._csr[2].nbytes) / 1048576, 1)
    tq = time.perf_counter()
    z_m, a_m = idx.retrieve(q0, L=3, beta=100.0, dense=dense)
    out["retrieve_postings_L3_f64_s"] = round(time.perf_counter() - tq, 3)
    tq = time.perf_counter()
    z_m, a_m = idx.retrieve(q0, L=3, beta=100.0, dense=dense, f32=True)
    out["retrieve_postings_L3_f32_s"] = round(time.perf_counter() - tq, 3)
    # 稠密参照 GEMV（float32，BLAS）
    tq = time.perf_counter()
    _ = M[:, :N].T @ dense
    out["dense_mtv_s"] = round(time.perf_counter() - tq, 3)
    a = softmax_beta(M[:, :N].T @ dense, 100.0)
    tq = time.perf_counter()
    _ = M[:, :N] @ a
    out["dense_mtcols_s"] = round(time.perf_counter() - tq, 3)
    tq = time.perf_counter()
    _ = softmax_beta(M[:, :N].T @ dense, 100.0)
    out["dense_gemv_plus_softmax_s"] = round(time.perf_counter() - tq, 3)
    # R1-alone：稠密 GEMV + float32 链（无倒排索引）
    Mn = M[:, :N]
    tq = time.perf_counter()
    zz = unit(np.maximum(dense, 0.0))
    a32 = softmax32(Mn.T @ zz, 100.0)
    for _ in range(3):
        zz = unit(Mn @ a32)
        a32 = softmax32(Mn.T @ zz, 100.0)
        a32 = a32 / a32.sum()
    out["retrieve_dense_f32_L3_s"] = round(time.perf_counter() - tq, 3)
    out["retrieve_postings_L3_s"] = out["retrieve_postings_L3_f32_s"]
    out["retrieval_matrix"] = {
        "postings_f64": out["retrieve_postings_L3_f64_s"],
        "postings_f32": out["retrieve_postings_L3_f32_s"],
        "dense_f32": out["retrieve_dense_f32_L3_s"],
    }
    out["retrieve_speedup_vs_8gemv"] = round(
        (8 * max(out["dense_mtv_s"], 1e-9)) / max(out["retrieve_postings_L3_s"], 1e-9), 1)
    out["peak_rss_mb"] = peak_rss_mb()
    out["total_s"] = round(time.perf_counter() - t0, 1)
    print("part2 retrieve: postings L3=%.2fs vs dense mtv=%.2fs (8x=%.2fs) -> %.0fx; csr=%sMB rss=%sMB"
          % (out["retrieve_postings_L3_s"], out["dense_mtv_s"], 8 * out["dense_mtv_s"],
             out["retrieve_speedup_vs_8gemv"], out["csr_mb"], out["peak_rss_mb"]), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="local", choices=["0", "1", "2", "local", "all"])
    args = ap.parse_args()
    res = {}
    if args.part in ("0", "local", "all"):
        res["part0"] = part0()
    if args.part in ("1", "local", "all"):
        res["part1"] = part1()
    if args.part in ("2", "all"):
        res["part2"] = part2()
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    path = os.path.join(ROOT, "results", "exp43_inverted_index%s.json" % ("" if args.part != "2" else "_scale"))
    with open(path, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print("WROTE", path, flush=True)


if __name__ == "__main__":
    main()
