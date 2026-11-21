import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import random_patterns
D = 128
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
print('store', m.E.n)
print('A ctx cos:', [round(float(unit(m.E.Ctx[:,i])@unit(ctxA)),3) for i in range(4)])
print('B ctx cos:', [round(float(unit(m.E.Ctx[:,i])@unit(ctxB)),3) for i in range(10,14)])
m.c = (1.2*ctxB).astype(np.float32)
q = m.query(unit(contents[0] + 1.2*ctxB))
print('query B0 -> idx=%d (want 10) action=%s m1=%.3f raw=%.3f' % (q['idx'], q['action'], q['m1'], q['raw_evidence']))
m.c = (1.2*ctxA).astype(np.float32)
q2 = m.query(unit(contents[0] + 1.2*ctxA))
print('query A0 -> idx=%d (want 0) action=%s m1=%.3f raw=%.3f' % (q2['idx'], q2['action'], q2['m1'], q2['raw_evidence']))