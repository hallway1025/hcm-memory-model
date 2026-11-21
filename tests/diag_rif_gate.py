
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import random_patterns
D = 128

print("=== RIF 机制诊断 ===")
m = HCM(Config(d_x=D, a=0.02, capacity=600, beta=20.0, L=3, use_resid=True, seed=0, rho_rif=0.15))
X = random_patterns(20, D, seed=51)
for x in X: m.observe(x)
print("库内 20 条，初始 sup 全 1:", m.E.sup[:20].min(), m.E.sup[:20].max())
for i in range(10):
    m.query(X[i], do_rif=True)
sup = m.E.sup[:20]
print("3 轮检索后 sup: 已练[0:10] mean=%.3f  未练[10:20] mean=%.3f" % (sup[:10].mean(), sup[10:].mean()))
print("  -> RIF 是否在改变 sup？", "是" if sup[:10].mean() < 0.99 else "否")
# 未练条目的召回是否受影响
ok = sum(int(m.query(X[i], do_rif=False)["idx"] == i) for i in range(10, 20))
print("  未练条目召回:", ok, "/10")
m2 = HCM(Config(d_x=D, a=0.02, capacity=600, beta=20.0, L=3, use_resid=True, seed=0, rif_on=False))
for x in X: m2.observe(x)
ok2 = sum(int(m2.query(X[i], do_rif=False)["idx"] == i) for i in range(10, 20))
print("  RIF 关闭时未练条目召回:", ok2, "/10")

print()
print("=== M7 门控诊断 ===")
m3 = HCM(Config(d_x=D, a=0.02, capacity=600, beta=20.0, L=3, use_resid=True, seed=0))
r = np.random.default_rng(3)
bg = random_patterns(3, D, seed=41); events = random_patterns(12, D, seed=42)
for _ in range(10):
    for b in bg: m3.observe(unit(b + 0.05*r.normal(size=D)), r=0.0)
for e in events: m3.observe(unit(e), r=1.0)
gs = [rec["g"] for rec in m3.log]
nus = [rec["nu"] for rec in m3.log]
print("门控 g: 背景段 mean=%.3f  事件段 mean=%.3f" % (np.mean(gs[:30]), np.mean(gs[30:])))
print("新颖性 nu: 背景 mean=%.3f 事件 mean=%.3f" % (np.mean(nus[:30]), np.mean(nus[30:])))
print("写入数 =", m3.E.n, "/ 42")
print("  -> 门控能否区分？", "能" if abs(np.mean(gs[:30]) - np.mean(gs[30:])) > 0.05 else "不能（g 已饱和在 0.9 以上）")
