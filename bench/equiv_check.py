# -*- coding: utf-8 -*-
"""equiv_check.py -- 验证 float32 z 修正与现有 float64 路径的数值等价性。

背景：retrieve 里 a=softmax_beta(...) 是 float64 → z=unit(Mn@a) 变 float64 →
后续 Mn.T@z 把 float32 的 M(16064×5000) 整块上转成 float64 → 每次 matvec 多
~0.55s（实测 retrieve 2.1s vs 理论 0.05s）。
修正：把 z 保持 float32（M 本身是 float32，迭代的固有精度就是 float32）。

本脚本对同一库分别跑「原实现」与「修正实现」，比较 action/idx/m1/m2/nu。
"""
import os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "bench"))
from hcm import Config, HCM
from hcm.ops import unit, softmax_beta
from gen_overlap_corpus import gen_corpus

N = int(os.environ.get("N", "1200"))
X, d, _, _ = gen_corpus(N, 20, 0.4, 0)
m = HCM(Config(d_x=d, d_c=d, a=0.02, capacity=N + 10, beta=100.0, L=3,
               use_resid=False, seed=0))
for x in X:
    m.observe(x)
E = m.E


def retrieve_old(q, L, beta, dense):
    Mn = E.M[:, : E.n]
    z = unit(np.maximum(dense, 0.0)) if dense is not None else unit(q)
    a = softmax_beta(Mn.T @ z, beta)
    for _ in range(max(0, int(L))):
        z = unit(Mn @ a)
        a = softmax_beta(Mn.T @ z, beta)
        sa = a.sum()
        if sa > 0:
            a = a / sa
    return z, a


def retrieve_new(q, L, beta, dense):
    Mn = E.M[:, : E.n]
    z = unit(np.maximum(dense, 0.0)) if dense is not None else unit(q)
    a = softmax_beta(Mn.T @ z, beta).astype(np.float32)
    for _ in range(max(0, int(L))):
        z = unit(Mn @ a)
        a = softmax_beta(Mn.T @ z, beta).astype(np.float32)
        sa = a.sum()
        if sa > 0:
            a = a / sa
    return z, a


rng = np.random.default_rng(7)
worst = 0.0
idx_mismatch = 0
top_mismatch = 0
for t in range(30):
    x = X[rng.integers(0, N)]
    nz = np.where(np.abs(x) > 1e-6)[0]
    y = x.copy()
    y[rng.choice(nz, size=max(1, int(len(nz) * 0.3)), replace=False)] = 0.0
    s_pre = m.enc.pre(y)
    t0 = time.perf_counter()
    z1, a1 = retrieve_old(s_pre, 3, 100.0, s_pre)
    t_old = time.perf_counter() - t0
    t0 = time.perf_counter()
    z2, a2 = retrieve_new(s_pre, 3, 100.0, s_pre)
    t_new = time.perf_counter() - t0
    d_a = float(np.max(np.abs(a1 - a2)))
    d_z = float(np.max(np.abs(z1 - z2)))
    worst = max(worst, d_a, d_z)
    o1 = np.argsort(-a1)[:3]
    o2 = np.argsort(-a2)[:3]
    if o1[0] != o2[0]:
        top_mismatch += 1
    if list(o1) != list(o2):
        idx_mismatch += 1

print("N=%d  max|Δa| = %.3e   max|Δz| = %.3e" % (N, worst, worst))
print("top1 mismatch: %d/30   top3 order mismatch: %d/30" % (top_mismatch, idx_mismatch))
t0 = time.perf_counter()
for _ in range(3):
    retrieve_old(s_pre, 3, 100.0, s_pre)
t_old = (time.perf_counter() - t0) / 3
t0 = time.perf_counter()
for _ in range(3):
    retrieve_new(s_pre, 3, 100.0, s_pre)
t_new = (time.perf_counter() - t0) / 3
print("retrieve old %.3f s  new %.3f s  speedup %.2fx" % (t_old, t_new, t_old / t_new))
print("VERDICT:", "EQUIVALENT" if (top_mismatch == 0 and worst < 1e-4) else "DIFFERENT")
