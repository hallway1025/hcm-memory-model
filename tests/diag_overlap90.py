
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import overlapping_set
D=128
X = overlapping_set(30, D, overlap=0.90, seed=11)
for resid in (False, True):
    m = HCM(Config(d_x=D, a=0.02, capacity=500, beta=20.0, L=3, use_resid=resid, seed=0))
    for x in X: m.observe(x)
    print("resid=%s  写入 %d/30" % (resid, m.E.n))
    gs = [r["g"] for r in m.log]; nus = [r["nu"] for r in m.log]
    print("   g mean=%.3f range=[%.3f,%.3f]  nu mean=%.3f range=[%.3f,%.3f]" % (
        np.mean(gs), min(gs), max(gs), np.mean(nus), min(nus), max(nus)))
    # 逐条召回
    ok = sum(int(m.query(X[i])["idx"] == i) for i in range(30))
    print("   精确召回 %d/30" % ok)
