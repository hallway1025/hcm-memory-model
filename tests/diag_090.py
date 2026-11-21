
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import overlapping_set
D=128
X = overlapping_set(30, D, overlap=0.90, seed=11)
m = HCM(Config(d_x=D, a=0.02, capacity=500, beta=20.0, L=3, use_resid=True, seed=0))
for x in X: m.observe(x)
acts = {}
raws = []
for i in range(30):
    q = m.query(X[i])
    acts[q["action"]] = acts.get(q["action"], 0) + 1
    raws.append((q["raw_evidence"], q["idx"], i))
print("actions:", acts)
print("theta_raw =", m.ctl.theta_raw)
raws.sort()
print("raw_evidence: min=%.3f max=%.3f" % (raws[0][0], raws[-1][0]))
print("样本（raw, 预测idx, 真值）:", [(round(a,3), b, c) for a,b,c in raws[:5]])
print()
print("=> 若 raw 全低于 theta_raw=0.50，则全部弃权。检查：")
below = sum(1 for r,_,_ in raws if r < m.ctl.theta_raw)
print("   低于阈值条数:", below, "/30")
