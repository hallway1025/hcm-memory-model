
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.data import overlapping_set

D = 128
print("=== dense_raw 对结果的影响（写入存 s_pre，查询用 raw vs 用 s_pre）===")
print("%-6s %-10s %-12s %-12s %-12s" % ("share", "dense_raw", "无残差召回", "有残差召回", "有残差写入"))
for share in (0.85, 0.90, 0.92):
    X = overlapping_set(30, D, overlap=share, seed=11)
    for dr in (True, False):
        res = {}
        for resid in (False, True):
            m = HCM(Config(d_x=D, a=0.02, capacity=500, beta=20.0, L=3,
                           use_resid=resid, seed=0, dense_raw=dr))
            for x in X: m.observe(x)
            ok = sum(int(m.query(X[i])["idx"] == i) for i in range(30))
            res[resid] = (ok, m.E.n)
        print("%-6.2f %-10s %-12s %-12s %-12d" % (
            share, dr, "%d/30"%res[False][0], "%d/30"%res[True][0], res[True][1]))
