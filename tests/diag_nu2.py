
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

m = HCM(Config(d_x=D, a=0.02, capacity=600, beta=20.0, L=3, use_resid=True, seed=0))
X = random_patterns(40, D, seed=61); Xn = random_patterns(40, D, seed=62)
for x in X: m.observe(x)
Mn = m.E.M[:, :m.E.n]

print("候选新颖性定义（现有 vs 三种替代）：")
print("%-22s %-18s %-18s %-10s" % ("定义", "库内 mean+-std", "库外 mean+-std", "d'"))
def report(name, f):
    a = np.array([f(degrade(x, 0.35, i)) for i, x in enumerate(X)])
    b = np.array([f(degrade(x, 0.35, 500+i)) for i, x in enumerate(Xn)])
    dp = (b.mean() - a.mean()) / np.sqrt(0.5*(a.var()+b.var()))
    print("%-22s %-18s %-18s %-10.2f" % (name, "%.3f+-%.3f" % (a.mean(), a.std()), "%.3f+-%.3f" % (b.mean(), b.std()), dp))

# 现有定义
def f_now(x):
    xe = m.enc.resid_read(x); h = m.enc.encode_from_pre(m.enc.pre(xe))
    return m.enc.novelty(unit(x), h)
report("现有 ||x-xhat||/||x||", f_now)

# 替代 1：与库中最近条目的失配
def f_mismatch(x):
    xe = m.enc.resid_read(x); h = m.enc.encode_from_pre(m.enc.pre(xe))
    return float(1.0 - np.max(Mn.T @ h))
report("1 - max cos(h, stored)", f_mismatch)

# 替代 2：用重建余弦（已验证 d'=7.12）
def f_recon(x):
    xe = m.enc.resid_read(x); s_pre = m.enc.pre(xe); q0 = m.enc.encode_from_pre(s_pre)
    _, a = m.E.retrieve(q0, L=3, beta=20.0, dense=s_pre)
    a = a / (a.sum()+1e-12)
    return float(1.0 - unit(s_pre) @ unit(m.E.D[:, :m.E.n] @ a))
report("1 - recon cosine", f_recon)
