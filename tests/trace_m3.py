
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import random_patterns
D=128
cfg = Config(d_x=D, a=0.02, capacity=600, beta=20.0, L=3, use_resid=True,
             seed=0, use_ctx_bias=True, kappa_ctx=10.0)
m = HCM(cfg)
r = np.random.default_rng(7)
contents = random_patterns(10, D, seed=31)
ctxA, ctxB = unit(r.normal(size=D)), unit(r.normal(size=D))
for c in contents: m.observe(unit(c + 1.2*ctxA))
for c in contents: m.observe(unit(c + 1.2*ctxB))
print("库内", m.E.n, "条；Ctx shape", m.E.Ctx.shape)
print()
i = 0
m.c = (1.2*ctxA).astype(np.float32)
q = m.query(unit(contents[0] + 1.2*ctxA))
print("查询 A0: idx=%d (期望 0)  action=%s  m1=%.3f" % (q["idx"], q["action"], q["m1"]))
print()
print("情境向量是否被正确存储与使用？")
print("  E.Ctx[:,0] 与 ctxA 余弦 = %.3f" % float(unit(m.E.Ctx[:,0]) @ unit(ctxA)))
print("  E.Ctx[:,10] 与 ctxB 余弦 = %.3f" % float(unit(m.E.Ctx[:,10]) @ unit(ctxB)))
print("  m.c 与 ctxA 余弦 = %.3f" % float(unit(m.c) @ unit(ctxA)))
print()
print("  注意 use_ctx_bias 只在 query 的模板循环里加偏置，而 E.Ctx 存的是")
print("  写入时的情境 unit(self.c)——写入时 c 是逐步 EMA，与场景级 ctxA 不完全相同。")
print("  逐条看前 5 条写入时的 |c| 与 c·ctxA：")
m2 = HCM(cfg)
for k, c in enumerate(contents):
    m2.observe(unit(c + 1.2*ctxA))
    if k < 5:
        cc = unit(m2.c)
        print("    k=%d |c|=%.3f  c·ctxA=%.3f" % (k, float(np.linalg.norm(m2.c)), float(cc @ unit(ctxA))))
