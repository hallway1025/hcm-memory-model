
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

m = HCM(Config(d_x=D, a=0.02, capacity=600, beta=20.0, L=3, use_resid=True, seed=0, d_z=64))
X = random_patterns(40, D, seed=61); Xn = random_patterns(40, D, seed=62)
for x in X: m.observe(x)

print("--- in-store queries (35% dropout) ---")
m1s, acts = [], {}
for i in range(40):
    q = m.query(degrade(X[i], 0.35, i))
    m1s.append(q["m1"]); acts[q["action"]] = acts.get(q["action"],0)+1
print("action counts:", acts, " m1 range: %.3f..%.3f mean %.3f" % (min(m1s), max(m1s), np.mean(m1s)))
print("theta_rec =", m.ctl.theta_rec, " margin =", m.ctl.margin)

print("--- novel queries ---")
m1n, actsn = [], {}
for i in range(40):
    q = m.query(degrade(Xn[i], 0.35, 500+i))
    m1n.append(q["m1"]); actsn[q["action"]] = actsn.get(q["action"],0)+1
print("action counts:", actsn, " m1 range: %.3f..%.3f mean %.3f" % (min(m1n), max(m1n), np.mean(m1n)))
print()
print("=> theta_rec=0.25 是否足以区分？in-store 最小 m1 = %.3f, novel 最大 m1 = %.3f" % (min(m1s), max(m1n)))
print("=> 结论：a 是 40 维 softmax（峰值 ~0.3），lam=0.6 时 f 的峰值 ~0.2-0.4；阈值 0.25 卡在中间，"
      "in-store 与 novel 的 m1 分布重叠，所以弃权无法区分两者。")
