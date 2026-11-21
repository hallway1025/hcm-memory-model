
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import overlapping_set
D=128
X = overlapping_set(30, D, overlap=0.90, seed=11)
print("隔离实验：overlap=0.90, resid=True, L=0")
print("theta_write=0 强制全写 vs 默认自适应门控")
for tw in (0.0, 0.5):
    m = HCM(Config(d_x=D, a=0.02, capacity=500, beta=20.0, L=0, use_resid=True, seed=0, theta_write=tw))
    for x in X: m.observe(x)
    ok = sum(int(m.query(X[i])["idx"] == i) for i in range(30))
    print("  theta_write=%.1f 写入=%d/30  精确召回=%d/30  bank=%d" % (tw, m.E.n, ok, len(m.enc.MU)))
    if m.E.n > 1:
        H = m.E.M[:, :m.E.n]
        print("     码间余弦 mean=%.3f" % np.mean((H.T@H)[np.triu_indices(m.E.n,1)]))
