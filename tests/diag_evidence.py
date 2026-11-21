
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
Mn = m.E.M[:, :m.E.n]; Dn = m.E.D[:, :m.E.n]

def probe(x):
    xe = m.enc.resid_read(x)
    s_pre = m.enc.pre(xe)
    q0 = m.enc.encode_from_pre(s_pre)
    _, a_hop = m.E.retrieve(q0, L=3, beta=20.0, dense=s_pre)
    a = a_hop / (a_hop.sum()+1e-12)
    j = int(np.argmax(a))
    z = unit(Dn @ a)                                 # 吸引子重建的稠密码
    cos_match = float(unit(s_pre) @ z)               # 与条目的稠密码匹配度
    a_peak = float(a.max())
    aft = unit(Dn[:, j])
    cos_to_top = float(unit(s_pre) @ aft)            # 只与 top1 条目比
    return a_peak, cos_match, cos_to_top

rows_in = [probe(degrade(x, 0.35, i)) for i, x in enumerate(X)]
rows_out = [probe(degrade(x, 0.35, 500+i)) for i, x in enumerate(Xn)]
import numpy as np
for name, rows in (("in-store", rows_in), ("novel   ", rows_out)):
    a = np.array(rows)
    print(name, "a_peak %.3f+-%.3f   cos(recon) %.3f+-%.3f [%.3f..%.3f]   cos(top1) %.3f+-%.3f" % (
        a[:,0].mean(), a[:,0].std(), a[:,1].mean(), a[:,1].std(), a[:,1].min(), a[:,1].max(),
        a[:,2].mean(), a[:,2].std()))
