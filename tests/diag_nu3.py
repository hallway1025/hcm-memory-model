
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import random_patterns
D = 128
m = HCM(Config(d_x=D, a=0.02, capacity=600, beta=20.0, L=3, use_resid=True, seed=0))
X = random_patterns(60, D, seed=11)
for x in X: m.observe(x)
nus = np.array([r["nu"] for r in m.log]); gs = np.array([r["g"] for r in m.log])
print("修复后 nu  : mean=%.3f std=%.3f range=[%.3f, %.3f]  动态范围 %.0f%%" % (
    nus.mean(), nus.std(), nus.min(), nus.max(), 100*(nus.max()-nus.min())))
print("修复后 g   : mean=%.3f std=%.3f range=[%.3f, %.3f]" % (gs.mean(), gs.std(), gs.min(), gs.max()))
print("theta_write = %.2f   写入率 = %d/%d" % (m.cfg.theta_write, m.E.n, len(X)))
# 重复呈现同一模式：门控应当关掉
m2 = HCM(Config(d_x=D, a=0.02, capacity=600, beta=20.0, L=3, use_resid=True, seed=0))
x0 = random_patterns(1, D, seed=99)[0]
for _ in range(12): m2.observe(x0)
g2 = [r["g"] for r in m2.log]
print("\n同一模式重复 12 次：g 首=%.3f 末=%.3f  写入条数=%d（理想：仅首次写入）" % (
    g2[0], g2[-1], m2.E.n))
