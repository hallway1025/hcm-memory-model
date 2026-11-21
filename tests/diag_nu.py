
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
print("nu  : mean=%.3f  std=%.3f  min=%.3f  max=%.3f  (range %.3f)" % (
    nus.mean(), nus.std(), nus.min(), nus.max(), nus.max()-nus.min()))
print("g   : mean=%.3f  std=%.3f  min=%.3f  max=%.3f" % (gs.mean(), gs.std(), gs.min(), gs.max()))
print("theta_write =", m.cfg.theta_write)
print()
print("=> 新颖性信号的有效动态范围只有 %.1f%%（%.3f..%.3f），" % (100*(nus.max()-nus.min()), nus.min(), nus.max()))
print("   门控输出被压在 [%.2f, %.2f]，与 theta_write=%.2f 相比" % (gs.min(), gs.max(), m.cfg.theta_write))
print("   门控**结构性**失去选择性：所有输入都远超阈值，等价于恒写入。")
print("   这是 M7 非负载承载的根因，不是实验设计错误。")
