# -*- coding: utf-8 -*-
"""bench_ops.py -- 逐行计时 HCM query/retrieve 的内部算子，定位 O(N) 热点。

profile_g3 显示 retrieve 单次 2.5s @ N=5000，但理论 FLOP 仅 ~0.44 GFLOP/query
→ 存在数量级级别的低效（疑似 strided M 的缓存失效或重复计算）。
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
Mn = E.M[:, :E.n]
print("d_h", E.d_h, "n", E.n, "M shape", E.M.shape, flush=True)
d_h = E.d_h
dense = np.random.default_rng(1).standard_normal(d_h).astype(np.float32)
dense = unit(np.maximum(dense, 0.0))

def t(label, fn, reps=3):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        r = fn()
    dt = (time.perf_counter() - t0) / reps
    print("%-34s %8.4f s" % (label, dt), flush=True)
    return r

print("--- retrieve 内部 ---", flush=True)
z0 = dense
t("Mn.T @ z", lambda: Mn.T @ z0)
t("softmax_beta(Mn.T@z)", lambda: softmax_beta(Mn.T @ z0, 100.0))
t("Mn @ a (a=rand)", lambda: Mn @ np.ones(E.n, dtype=np.float32))
t("unit(Mn @ a)", lambda: unit(Mn @ np.ones(E.n, dtype=np.float32)))
Mt_contig = np.ascontiguousarray(Mn.T)
t("Mt_contig @ z", lambda: Mt_contig @ z0)
t("ascontiguousarray(Mn.T)", lambda: np.ascontiguousarray(Mn.T))

print("--- model.query 内部 ---", flush=True)
x = X[0]
t("enc.pre(x)", lambda: m.enc.pre(x))
s_pre = m.enc.pre(x)
t("Mn.T @ s_pre", lambda: Mn.T @ s_pre)
t("sem.forward(x)", lambda: m.sem.forward(x))
shat = unit(m.sem.forward(x))
t("Mn.T @ shat", lambda: Mn.T @ shat)
t("E.D[:,:n] @ a", lambda: E.D[:, :E.n] @ np.ones(E.n, dtype=np.float32))
t("E.retrieve(dense)", lambda: E.retrieve(s_pre, L=3, beta=100.0, dense=s_pre))
t("full m.query(x)", lambda: m.query(x, do_rif=False), reps=2)
