import os

import sys, time, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.data import random_patterns

print("=== 写入/检索的实测耗时（决定规模上限的关键）===")
for N in (2000, 8000, 16000):
    X = random_patterns(N, 128, seed=5)
    m = HCM(Config(d_x=128, a=0.02, capacity=N+10, beta=20.0, L=3, use_resid=True, seed=0))
    t0 = time.time()
    for x in X: m.observe(x)
    t_w = time.time() - t0
    probe = list(range(0, N, max(1, N // 30)))[:30]
    t0 = time.time()
    for i in probe: m.query(X[i])
    t_q = (time.time() - t0) / len(probe)
    print("  N=%-6d 写入 %6.1fs (%d 条)  单次检索 %7.1f ms  库内 %d" % (N, t_w, N, t_q*1000, m.E.n))
