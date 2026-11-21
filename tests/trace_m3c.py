
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit, softmax_beta
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

m.c = (1.2*ctxB).astype(np.float32); cvec = unit(m.c)
Mn = m.E.M[:, :m.E.n]
q = unit(contents[0] + 1.2*ctxB)
xe = m.enc.resid_read(q); s_pre = m.enc.pre(xe); q0 = m.enc.encode_from_pre(s_pre)
ev = Mn.T @ s_pre
b = Mn.T @ q0
print("查询 B0（应命中 idx≈10）")
print("  写入库内", m.E.n, "条")
print("  稠密证据 ev: argmax=%d  top5 idx=%s  vals=%s" % (
    int(np.argmax(ev)), np.argsort(-ev)[:5], np.round(np.sort(ev)[-5:], 5)))
print("  稀疏证据:  argmax=%d  top5=%s" % (int(np.argmax(b)), np.argsort(-b)[:5]))
print()
print("  q0 与库内各条目码的余弦 top5:", np.round(np.sort(Mn.T @ q0)[-5:], 4))
print("  期望库模板数:", len(m.enc.MU))
print("  模板与 ctxA/ctxB 的余弦:", [(round(float(unit(mu) @ unit(ctxA)),2), round(float(unit(mu) @ unit(ctxB)),2)) for mu in m.enc.MU])
print()
print("关键：q0 = encode(resid_read(q))，而 resid_read 用 m.enc.mu（最后写入的模板 = B 的均值）")
print("  于是 B0 的残差 ≈ 0 → 码全零/退化 → 无法与任何条目匹配。")
nz = np.count_nonzero(q0)
print("  q0 非零元素数 =", nz, "/", len(q0))
