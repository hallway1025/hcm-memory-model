# -*- coding: utf-8 -*-
"""profile_g3.py -- 定位 B2 β 扫描的耗时热点（cProfile）。

用途：β 扫描在 N=20000 时每 β 约 530s，全网格 30 格算力不够。
先profile出热点，再决定是向量化/降低探针数/精简网格。
"""
import cProfile, io, os, pstats, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "bench"))

from hcm import Config, HCM
from gen_overlap_corpus import gen_corpus

N = int(os.environ.get("PROF_N", "5000"))
K = 20
O = 0.4
SEED = 0
PROBES = int(os.environ.get("PROF_PROBES", "5"))
BETAS = (20.0, 100.0)

def degrade(x, frac, seed):
    rng = np.random.default_rng(seed)
    y = x.copy()
    nz = np.where(np.abs(y) > 1e-6)[0]
    if len(nz):
        k = max(1, int(len(nz) * frac))
        y[rng.choice(nz, size=k, replace=False)] = 0.0
    return y

def run():
    X, d, shared, pool = gen_corpus(N, K, O, SEED)
    print("gen d=%d N=%d" % (d, N), flush=True)
    m = HCM(Config(d_x=d, d_c=d, a=0.02, capacity=N + 10, beta=20.0,
                   L=3, use_resid=False, seed=SEED))
    t0 = time.perf_counter()
    for x in X:
        m.observe(x)
    print("write %.1fs" % (time.perf_counter() - t0), flush=True)
    probe_idx = list(range(0, N, N // PROBES))[:PROBES]
    qs = [degrade(X[i], 0.3, i) for i in probe_idx]
    for b in BETAS:
        m.cfg.beta = b
        t1 = time.perf_counter()
        for q in qs:
            m.query(q, do_rif=False)
        print("beta=%.0f  %.1fs total  %.3fs/query" % (b, time.perf_counter()-t1,
              (time.perf_counter()-t1)/len(qs)), flush=True)

pr = cProfile.Profile()
pr.enable()
run()
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(28)
print(s.getvalue())
