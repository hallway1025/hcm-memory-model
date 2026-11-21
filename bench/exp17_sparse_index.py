#!/usr/bin/env python3
"""exp17_sparse_index.py -- A 层工程优化 #1：稀疏 M 存储 + 倒排索引（优化路线 #1）。

主张：
  1. M 的每列是 k-稀疏单位码 -> (行号,值) 对 + 单元倒排表可在**不改检索语义**的
     前提下替代稠密 M 做全部 Mn.T @ v 与 Mn @ a 运算；
  2. 内存节省 ~d_h/(2k) 倍（d_h=1024,k=20 约 25x）；
  3. 在 N=16000 规模上 scatter/gather 的 wall time 与稠密 BLAS 可比或更快（本机），
     真正的收益在 L2/L3（放不进内存的规模）。

验收门：
  - 每条查询的 ev 向量 max|Δ| < 1e-5 且 top-1 idx 与稠密路径 100% 一致；
  - Hopfield 迭代后的 a 向量 max|Δ| < 1e-4。
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit, softmax_beta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = 128
SEEDS = [0, 1, 2]
NS = [2000, 16000]
PROBES = 40


def pats(N, d, seed):
    r = np.random.default_rng(seed)
    return np.asarray([unit(v) for v in r.standard_normal((N, d))], dtype=np.float32)


def degrade(x, frac, seed):
    r = np.random.default_rng(seed)
    y = x.copy()
    nz = np.where(np.abs(y) > 1e-6)[0]
    y[r.choice(nz, size=max(1, int(len(nz) * frac)), replace=False)] = 0.0
    return unit(y)


class SparseM:
    """M[:, :n] 的稀疏镜像：行列式 + 单元倒排表。"""
    def __init__(self, Mn):
        self.dh, self.N = Mn.shape  # Mn: (d_h, n)
        # 列式：每列 nnz 个 (row, val)
        self.rows, self.vals, self.lens = [], [], []
        for c in range(self.N):
            nz = np.flatnonzero(Mn[:, c])
            self.rows.append(nz.astype(np.int32))
            self.vals.append(Mn[nz, c].astype(np.float32))
            self.lens.append(len(nz))
        # 倒排（单元 -> [(col,val)] CSR）
        pairs_r, pairs_v, pairs_c = [], [], []
        for c in range(self.N):
            pairs_r.append(self.rows[c]); pairs_v.append(self.vals[c])
            pairs_c.append(np.full(len(self.rows[c]), c, dtype=np.int32))
        pr = np.concatenate(pairs_r); pv = np.concatenate(pairs_v); pc = np.concatenate(pairs_c)
        order = np.argsort(pr, kind="stable")
        self.pr = pr[order]; self.pv = pv[order].astype(np.float32); self.pc = pc[order]
        cnt = np.bincount(self.pr, minlength=self.dh)
        self.ip = np.zeros(self.dh + 1, dtype=np.int64); self.ip[1:] = np.cumsum(cnt)

    def mtv(self, v):
        """Mn.T @ v via postings."""
        out = np.zeros(self.N, dtype=np.float32)
        nz = np.flatnonzero(v)
        for j in nz:
            s, e = self.ip[j], self.ip[j + 1]
            if e > s:
                np.add.at(out, self.pc[s:e], v[j] * self.pv[s:e])
        return out

    def mtv_rows(self, v):
        """Mn.T @ v via postings + bincount（全向量化）。"""
        w = v[self.pr] * self.pv
        return np.bincount(self.pc, weights=w, minlength=self.N).astype(np.float32)

    def gather(self, a):
        """Mn @ a."""
        out = np.zeros(self.dh, dtype=np.float32)
        nzc = np.flatnonzero(a)
        for c in nzc:
            out[self.rows[c]] += a[c] * self.vals[c]
        return out

    def nbytes(self):
        b = 0
        for r, v, l in zip(self.rows, self.vals, self.lens):
            b += r.nbytes + v.nbytes
        b += self.pr.nbytes + self.pc.nbytes + self.pv.nbytes + self.ip.nbytes
        return b


def hopfield_sparse(SM, z0, L, beta):
    a = softmax_beta(SM.mtv_rows(z0), beta)
    for _ in range(L):
        z = unit(SM.gather(a))
        a = softmax_beta(SM.mtv_rows(z), beta)
        sa = a.sum()
        if sa > 0:
            a = a / sa
    return z, a


def hopfield_dense(Mn, z0, L, beta):
    a = softmax_beta(Mn.T @ z0, beta)
    for _ in range(L):
        z = unit(Mn @ a)
        a = softmax_beta(Mn.T @ z, beta)
        sa = a.sum()
        if sa > 0:
            a = a / sa
    return z, a


def run(N, seed):
    X = pats(N, D, 100 + seed)
    m = HCM(Config(d_x=D, a=0.02, capacity=N, seed=seed,
                   use_resid=True))  # D-052 显式传参：保持翻转前语义（产物早于 2026-11-08 05:04 翻转）
    for x in X:
        m.observe(x)
    Mn = np.ascontiguousarray(m.E.M[:, : m.E.n])
    SM = SparseM(Mn)
    # 查询集：库内退化 + 原始
    Q = [(degrade(X[i], 0.25, 7000 + seed + i), X[i]) for i in range(PROBES)]
    r = np.random.default_rng(9000 + seed)
    Q += [(unit(r.standard_normal(D).astype(np.float32)), None) for _ in range(10)]
    # 编码一条查询的稠密驱动（模拟 query() 的 ed 路径）
    max_dz, max_da, mism = 0.0, 0.0, 0
    t_dense = t_sparse = 0.0
    for q, orig in Q:
        s_ev = m.enc.pre(m.enc.resid_read_all(q)[0])
        qq = m.enc.encode_from_pre(s_ev)
        ed_d = Mn.T @ s_ev; ed_s = SM.mtv_rows(s_ev)
        es_d = Mn.T @ qq;   es_s = SM.mtv_rows(qq)
        z0 = unit(np.maximum(s_ev, 0.0))
        t0 = time.perf_counter(); zd, ad = hopfield_dense(Mn, z0, 3, m.cfg.beta); t_dense += time.perf_counter() - t0
        t0 = time.perf_counter(); zs, asp = hopfield_sparse(SM, z0, 3, m.cfg.beta); t_sparse += time.perf_counter() - t0
        shat = unit(m.sem.forward(q))
        sr_d = Mn.T @ shat; sr_s = SM.mtv_rows(shat)
        for a_, b_ in ((ed_d, ed_s), (es_d, es_s), (sr_d, sr_s)):
            max_dz = max(max_dz, float(np.abs(a_ - b_).max()))
            if int(np.argmax(a_)) != int(np.argmax(b_)):
                mism += 1
        max_da = max(max_da, float(np.abs(ad - asp).max()))
    # 纯算子计时（100 次 s_ev 规模向量）
    v = s_ev
    t0 = time.perf_counter()
    for _ in range(100): Mn.T @ v
    d_op = (time.perf_counter() - t0) / 100
    t0 = time.perf_counter()
    for _ in range(100): SM.mtv_rows(v)
    s_op = (time.perf_counter() - t0) / 100
    nb_d = Mn.nbytes; nb_s = SM.nbytes()
    lens = np.array(SM.lens)
    return dict(N=N, seed=seed, n=m.E.n,
                nnz_per_col_mean=float(lens.mean()), nnz_per_col_min=int(lens.min()), nnz_per_col_max=int(lens.max()),
                max_abs_diff_matvec=max_dz, max_abs_diff_hop_a=max_da,
                argmax_mismatch_matvec=mism, argmax_mismatch_hop=int(np.argmax(ad) != np.argmax(asp)),
                hopfield_dense_ms=float(t_dense / len(Q) * 1000), hopfield_sparse_ms=float(t_sparse / len(Q) * 1000),
                matvec_dense_us=float(d_op * 1e6), matvec_sparse_us=float(s_op * 1e6),
                mem_dense_mb=float(nb_d / 1e6), mem_sparse_mb=float(nb_s / 1e6),
                mem_ratio=float(nb_d / nb_s))


def main():
    if os.environ.get('SMOKE'):
        global SEEDS, NS
        SEEDS, NS = [0], [2000]
    out = {"what": "exp17 sparse M + inverted index: equivalence + cost", "seeds": SEEDS, "N_list": NS, "rows": []}
    for N in NS:
        for s in SEEDS:
            r = run(N, s)
            out["rows"].append(r)
            print("N=%d seed=%d nnz/col=%.1f dVec=%.2e aVec=%.2e mismatch=%d/%d hd=%.2fms hs=%.2fms matvec d=%.0fus s=%.0fus mem %.1fMB->%.1fMB (%.1fx)" % (
                N, s, r["nnz_per_col_mean"], r["max_abs_diff_matvec"], r["max_abs_diff_hop_a"],
                r["argmax_mismatch_matvec"], PROBES + 10, r["hopfield_dense_ms"], r["hopfield_sparse_ms"],
                r["matvec_dense_us"], r["matvec_sparse_us"], r["mem_dense_mb"], r["mem_sparse_mb"], r["mem_ratio"]), flush=True)
    ok = all(r["max_abs_diff_matvec"] < 1e-5 and r["max_abs_diff_hop_a"] < 1e-4
             and r["argmax_mismatch_matvec"] == 0 and r["argmax_mismatch_hop"] == 0 for r in out["rows"])
    out["verdict"] = "EQUIVALENT" if ok else "MISMATCH"
    with open(os.path.join(ROOT, "results", "exp17_sparse_index.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("VERDICT", out["verdict"], flush=True)


if __name__ == "__main__":
    main()
