
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import random_patterns
D = 128
def degrade(x, frac, seed):
    r = np.random.default_rng(seed); y = x.copy()
    nz = np.where(np.abs(y) > 1e-6)[0]
    y[r.choice(nz, size=max(1,int(len(nz)*frac)), replace=False)] = 0.0
    return unit(y)
# 扫 theta_raw：risk-coverage 曲线
m = HCM(Config(d_x=D, a=0.02, capacity=600, beta=20.0, L=3, use_resid=True, seed=0, d_z=64))
X = random_patterns(40, D, seed=61); Xn = random_patterns(40, D, seed=62)
for x in X: m.observe(x)
m.ctl.theta_raw = -1.0     # 强制全部 recall，先收集
recs = []
for i, x in enumerate(X):
    q = m.query(degrade(x, 0.35, i)); recs.append((q["raw_evidence"], q["idx"] == i))
for i, x in enumerate(Xn):
    q = m.query(degrade(x, 0.35, 500+i)); recs.append((q["raw_evidence"], False))
recs.sort(key=lambda t: -t[0])
print("theta   coverage  risk    (in-store correct / total)")
for th in (0.0, 0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
    sel = [c for r, c in recs if r >= th]
    cov = len(sel)/len(recs)
    risk = 1 - (sum(sel)/len(sel)) if sel else 0.0
    print("%.2f    %.3f     %.3f" % (th, cov, risk))
