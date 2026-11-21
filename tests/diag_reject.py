
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import overlapping_set
D=128
X = overlapping_set(30, D, overlap=0.90, seed=11)
m = HCM(Config(d_x=D, a=0.02, capacity=500, beta=20.0, L=0, use_resid=True, seed=0, theta_write=0.5))
for x in X: m.observe(x)
print("26 条写入后，逐条查未写条目为何被拒：")
for i in range(30):
    xe = m.enc.resid_read(X[i]); h = m.enc.encode_from_pre(m.enc.pre(xe))
    sim = float(np.max(m.E.M[:, :m.E.n].T @ h)) if m.E.n else 0.0
    rec = m.log[i]
    print("  i=%2d nu=%.3f (1-maxcos=%.3f) g=%.3f wrote=%s" % (i, rec["nu"], 1-sim, rec["g"], rec["wrote"]))
    if i >= 9: break
print()
print("问题：残差化后的码彼此高度相似（overlap 0.90 时码间余弦 0.255-0.308），")
print("      于是 '1 - maxcos' 很小 -> nu 小 -> g 低于阈值 -> 拒写。")
print("      但被拒写的条目恰恰是**本应被分离存储**的新条目。")
print("      => 新颖性把「相似但不同」误判为「同一个」，这正是模式分离要解决的问题。")
