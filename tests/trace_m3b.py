
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import random_patterns
D=128
cfg = Config(d_x=D, a=0.02, capacity=800, beta=20.0, L=3, use_resid=True, seed=0,
             use_ctx_bias=True, kappa_ctx=10.0)
m = HCM(cfg)
r = np.random.default_rng(7)
contents = random_patterns(10, D, seed=31)
ctxA, ctxB = unit(r.normal(size=D)), unit(r.normal(size=D))
for _ in range(30): m.c = (0.95*m.c + 0.05*(1.2*ctxA)).astype(np.float32)
for c in contents: m.observe(unit(c + 1.2*ctxA))
for _ in range(30): m.c = (0.95*m.c + 0.05*(1.2*ctxB)).astype(np.float32)
for c in contents: m.observe(unit(c + 1.2*ctxB))
print("库内", m.E.n, "（期望 20，实际少 1 = 去重挡掉了）")
print("期望：B0 应在 index 10（若 A 全写）")
# 找到 B0 的真实 index
for k, c in enumerate(contents):
    b0 = unit(c + 1.2*ctxB)
    h = m.enc.encode(m.enc.resid_read(b0))
    best = int(np.argmax(m.E.M[:, :m.E.n].T @ h))
    print("  B%d -> 最近条目 idx=%d (期望 %d)" % (k, best, 10+k))
    if k >= 2: break
print()
print("检查 E.Ctx 的正确性：")
print("  Ctx[:,0]·ctxA = %.3f  Ctx[:,10]·ctxB = %.3f" % (
    float(unit(m.E.Ctx[:,0]) @ unit(ctxA)), float(unit(m.E.Ctx[:,10]) @ unit(ctxB))))
print("  => 情境存储正确。问题在检索端。")
print()
print("query 的模板循环里 ctx_bias 只在 use_ctx_bias 时加，但它加的是")
print("  E.Ctx[:, :n].T @ cvec ，量级取决于 cvec=unit(self.c) 与 E.Ctx 列的点积。")
m.c = (1.2*ctxB).astype(np.float32)
cvec = unit(m.c)
bias = m.E.Ctx[:, :m.E.n].T @ cvec
print("  bias 前 12 项:", np.round(bias[:12], 3))
print("  bias 后 8 项 :", np.round(bias[12:], 3))
print("  => B 组的 bias 确实更高，但证据项压倒了它（kappa=10 仍不够）。")
