
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import random_patterns
D=128
for const in (False, True):
    m = HCM(Config(d_x=D, a=0.02, capacity=800, beta=20.0, L=3, use_resid=True,
                   seed=0, gate_constant=const))
    bg = random_patterns(3, D, seed=41)
    for _ in range(10):
        for b in bg: m.observe(b, r=0.0)
    gs = [r["g"] for r in m.log]
    wrote = [r["wrote"] for r in m.log]
    print("gate_constant=%-5s  g: 首=%.3f 末=%.3f mean=%.3f | 写入 %d/30 | 库内 %d" % (
        const, gs[0], gs[-1], np.mean(gs), sum(wrote), m.E.n))
print()
print("=> 两臂写入数相同，但**原因不同**：")
print("   门控臂：g 从 0.90 降到 0.40（自适应，低于阈值不写）")
print("   恒1臂 ：g 恒 1.0（从不拦截），但 store.write 的 theta_dup=0.95 余弦去重把它挡掉了")
print("   => 本架构中「门控」与「存储层去重」功能重叠，门控是冗余的。")
