
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import random_patterns
D = 128
m = HCM(Config(d_x=D, a=0.02, capacity=600, beta=20.0, L=3, use_resid=True, seed=0))
x0 = random_patterns(1, D, seed=99)[0]
print("重复同一模式 12 次（门控应逐次下降并停止写入）：")
for k in range(12):
    r = m.observe(x0)
    print("  t=%2d nu=%.3f g=%.3f wrote=%s 库内=%d" % (k, r["nu"], r["g"], r["wrote"], m.E.n))
print()
print("对照：不同模式 (nu 应保持高)：")
m2 = HCM(Config(d_x=D, a=0.02, capacity=600, beta=20.0, L=3, use_resid=True, seed=0))
for x in random_patterns(8, D, seed=7):
    r = m2.observe(x)
    print("  nu=%.3f g=%.3f wrote=%s" % (r["nu"], r["g"], r["wrote"]))
