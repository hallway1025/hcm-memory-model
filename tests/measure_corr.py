
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import random_patterns

D, N, FR = 128, 60, 0.35
def degrade(x, frac, seed):
    r = np.random.default_rng(seed); y = x.copy()
    nz = np.where(np.abs(y) > 1e-6)[0]
    y[r.choice(nz, size=max(1, int(len(nz)*frac)), replace=False)] = 0.0
    return unit(y)

X = random_patterns(N, D, seed=61)
Xn = random_patterns(N, D, seed=62)
m = HCM(Config(d_x=D, a=0.02, capacity=600, beta=20.0, L=3, use_resid=True, seed=0))
for x in X: m.observe(x)

m1s, raws, correct = [], [], []
for i, x in enumerate(X):
    q = m.query(degrade(x, FR, i))
    m1s.append(q["m1"]); raws.append(q["raw_evidence"]); correct.append(int(q["idx"] == i))
for i, x in enumerate(Xn):
    q = m.query(degrade(x, FR, 500+i))
    m1s.append(q["m1"]); raws.append(q["raw_evidence"]); correct.append(0)

m1s = np.array(m1s); raws = np.array(raws); correct = np.array(correct)
print("样本数:", len(correct), " 正确率: %.3f" % correct.mean())
print("corr(m1, correct)          = %+.4f" % np.corrcoef(m1s, correct)[0,1])
print("corr(raw_evidence, correct) = %+.4f" % np.corrcoef(raws, correct)[0,1])
print()
print("=> 这就是 r 的真值；若与 0.222 不符，说明旧数字确实无出处。")
