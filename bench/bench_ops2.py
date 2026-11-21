# -*- coding: utf-8 -*-
"""bench_ops2.py -- 复刻 retrieve 内部逐步计时，找出 2.3s/query 的真实热点。

bench_ops.py 显示：单独测 Mn.T@z / Mn@a 各 ~0.008s，但 E.retrieve 整体 2.33s
→ 差距 30 倍，热点不在 matvec 本身。逐行复刻 retrieve 计时。
"""
import os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "bench"))
from hcm import Config, HCM
from hcm.ops import unit, softmax_beta
from gen_overlap_corpus import gen_corpus

N = int(os.environ.get("N", "5000"))
X, d, _, _ = gen_corpus(N, 20, 0.4, 0)
m = HCM(Config(d_x=d, d_c=d, a=0.02, capacity=N + 10, beta=100.0, L=3,
               use_resid=False, seed=0))
t0 = time.perf_counter()
for x in X:
    m.observe(x)
print("write %.1fs" % (time.perf_counter() - t0), flush=True)

E = m.E
Mn = E.M[:, : E.n]
s_pre = m.enc.pre(X[0])
print("M", E.M.shape, "n", E.n, "d_h", E.d_h, "s_pre", s_pre.shape, flush=True)
print("Mn contiguous:", Mn.flags["C_CONTIGUOUS"], Mn.flags["F_CONTIGUOUS"], flush=True)
print("Mn.T contiguous:", Mn.T.flags["C_CONTIGUOUS"], Mn.T.flags["F_CONTIGUOUS"], flush=True)

def clk(label, fn, reps=3):
    r = fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        r = fn()
    print("%-40s %8.4f s" % (label, (time.perf_counter() - t0) / reps), flush=True)
    return r

beta = 100.0
L = 3
z = clk("unit(max(dense,0))", lambda: unit(np.maximum(s_pre, 0.0)))
a = clk("softmax_beta(Mn.T@z, beta)", lambda: softmax_beta(Mn.T @ z, beta))
it = 0
for _ in range(L):
    it += 1
    z = clk("iter%d: Mn @ a" % it, lambda: unit(Mn @ a))
    a = clk("iter%d: softmax_beta(Mn.T@z)" % it, lambda: softmax_beta(Mn.T @ z, beta))
    a = clk("iter%d: a/a.sum()" % it, lambda: a / (a.sum() + 1e-12))

print("--- full retrieve for reference ---", flush=True)
clk("E.retrieve", lambda: E.retrieve(s_pre, L=L, beta=beta, dense=s_pre))
print("--- is the cost in M.n vs capacity? ---", flush=True)
E.capacity = E.n
Mn2 = E.M[:, : E.n]
clk("Mn2.T@z (capacity==n)", lambda: Mn2.T @ z)
