# -*- coding: utf-8 -*-
"""equiv_sparse.py -- 验证 max_cos/scores 的非零行稀疏化与全量 matvec 等价。"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "bench"))
from hcm import Config, HCM
from gen_overlap_corpus import gen_corpus

N = 2000
X, d, _, _ = gen_corpus(N, 20, 0.5, 0)
m = HCM(Config(d_x=d, d_c=d, a=0.02, capacity=N + 10, beta=100.0, L=3, use_resid=False, seed=0))
for x in X:
    m.observe(x)
E = m.E
print("store n =", E.n, " d_h =", E.M.shape[0])

def dots_dense(h):
    return E.M[:, : E.n].T @ h

def dots_sparse(h):
    h32 = np.asarray(h, dtype=np.float32)
    nz = np.flatnonzero(h32)
    if nz.size == 0:
        return np.zeros(E.n, dtype=np.float32)
    return (E.M[nz, : E.n].T @ h32[nz]).astype(np.float32)

rng = np.random.default_rng(3)
worst_dot = 0.0; worst_mc = 0.0; top_mm = 0; nnzs = []
import time
t_d = t_s = 0.0
for t in range(200):
    x = X[rng.integers(0, N)]
    s_pre = m.enc.pre(m.enc.resid_write(x)) if hasattr(m.enc, 'resid_write') else m.enc.pre(x)
    h = m.enc.encode_from_pre(s_pre)
    nnz = int(np.count_nonzero(h)); nnzs.append(nnz)
    t0 = time.perf_counter(); d1 = dots_dense(h); t_d += time.perf_counter()-t0
    t0 = time.perf_counter(); d2 = dots_sparse(h); t_s += time.perf_counter()-t0
    worst_dot = max(worst_dot, float(np.max(np.abs(d1-d2))))
    mc1 = float(np.max(d1)); mc2 = float(np.max(d2))
    worst_mc = max(worst_mc, abs(mc1-mc2))
    if np.argmax(d1) != np.argmax(d2): top_mm += 1
print("nnz(h): min=%d max=%d mean=%.1f" % (min(nnzs), max(nnzs), sum(nnzs)/len(nnzs)))
print("max|Δdot| = %.3e   max|Δmax_cos| = %.3e   argmax mismatch %d/200" % (worst_dot, worst_mc, top_mm))
print("dense %.4fs  sparse %.4fs  speedup %.1fx" % (t_d, t_s, t_d/t_s))
print("VERDICT:", "EQUIVALENT" if (top_mm==0 and worst_mc < 1e-5) else "DIFFERENT")
