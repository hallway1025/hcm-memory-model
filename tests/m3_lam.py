
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import random_patterns
D=128
for use_ctx in (False, True):
  for lam in (0.6, 1.0, 0.0):
    cfg = Config(d_x=D, a=0.02, capacity=800, beta=20.0, L=3, use_resid=True, seed=0,
                 use_ctx_bias=use_ctx, kappa_ctx=10.0)
    m = HCM(cfg); m.cfg.lam = lam
    r = np.random.default_rng(7)
    contents = random_patterns(10, D, seed=31)
    ctxA, ctxB = unit(r.normal(size=D)), unit(r.normal(size=D))
    for _ in range(30): m.c = (0.95*m.c + 0.05*(1.2*ctxA)).astype(np.float32)
    for c in contents: m.observe(unit(c + 1.2*ctxA))
    for _ in range(30): m.c = (0.95*m.c + 0.05*(1.2*ctxB)).astype(np.float32)
    for c in contents: m.observe(unit(c + 1.2*ctxB))
    for tag, ctx in (("A", ctxA), ("B", ctxB)):
        m.c = (1.2*ctx).astype(np.float32)
        ok = 0
        for k, c in enumerate(contents):
            q = m.query(unit(c + 1.2*ctx))
            want = k if tag=="A" else 10+k
            ok += int(q["idx"] == want)
        print("ctx_bias=%-5s lam=%.1f  %s: %d/10" % (use_ctx, lam, tag, ok), end="  ")
    print()
