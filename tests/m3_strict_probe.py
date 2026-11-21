import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import random_patterns
D = 128
cfg = Config(d_x=D, a=0.02, capacity=800, beta=20.0, L=3, use_resid=True, seed=0,
             use_ctx_bias=True, kappa_ctx=3.0)
m = HCM(cfg)
r = np.random.default_rng(9)
C = random_patterns(8, D, seed=33)
ctxA, ctxB = unit(r.normal(size=D)), unit(r.normal(size=D))
for _ in range(30): m.c = (0.95*m.c + 0.05*(1.2*ctxA)).astype(np.float32)
for c in C: m.observe(unit(c + 1.2*ctxA))
for _ in range(30): m.c = (0.95*m.c + 0.05*(1.2*ctxB)).astype(np.float32)
for c in C: m.observe(unit(c + 1.2*ctxB))
print('store', m.E.n, 'bank', len(m.enc.MU))
print('A ctx cos to ctxA:', [round(float(unit(m.E.Ctx[:,i])@unit(ctxA)),2) for i in range(8)])
print('B ctx cos to ctxB:', [round(float(unit(m.E.Ctx[:,i])@unit(ctxB)),2) for i in range(8,15)])
print()
print('关键问题：内容相同，A0 与 B0 在内容空间完全一样。差別只在情境。')
for tag, ctx in (('A', ctxA), ('B', ctxB)):
    m.c = (1.2*ctx).astype(np.float32)
    q = m.query(unit(C[0] + 1.2*ctx))
    print('  %s0 -> idx=%d  m1=%.3f raw=%.3f' % (tag, q['idx'], q['m1'], q['raw_evidence']))