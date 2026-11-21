
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import random_patterns
D=128

def occasion(gamma_c, n_per, use_ctx, kappa=10.0):
    cfg = Config(d_x=D, a=0.02, capacity=800, beta=20.0, L=3, use_resid=True, seed=0,
                 use_ctx_bias=use_ctx, kappa_ctx=kappa, gamma_c=gamma_c)
    m = HCM(cfg)
    r = np.random.default_rng(7)
    contents = random_patterns(n_per, D, seed=31)
    ctxA, ctxB = unit(r.normal(size=D)), unit(r.normal(size=D))
    events = []
    # 每个情境先"预热"——让 EMA 收敛到该情境，再开始写入
    for _ in range(30):
        m.c = (gamma_c*m.c + (1-gamma_c)*(1.2*ctxA)).astype(np.float32)
    for c in contents:
        m.observe(unit(c + 1.2*ctxA)); events.append(("A", c))
    for _ in range(30):
        m.c = (gamma_c*m.c + (1-gamma_c)*(1.2*ctxB)).astype(np.float32)
    for c in contents:
        m.observe(unit(c + 1.2*ctxB)); events.append(("B", c))
    okA = okB = 0
    for k, (tag, c) in enumerate(events):
        ctx = ctxA if tag == "A" else ctxB
        m.c = (1.2*ctx).astype(np.float32)
        hit = int(m.query(unit(c + 1.2*ctx))["idx"] == k)
        if tag == "A": okA += hit
        else: okB += hit
    return okA, okB, n_per, m.E.n

print("M3 情境绑定（加情境预热，让 EMA 收敛到场景）")
print("%-8s %-8s %-8s %-14s %s" % ("gamma_c","ctx_bias","每情境","A命中 / B命中","写入"))
for gamma_c in (0.95, 0.80, 0.60):
    for use in (False, True):
        a, b, n, nw = occasion(gamma_c, 10, use)
        print("%-8.2f %-8s %-8d %-14s %d" % (gamma_c, use, n, "%d/%d  %d/%d" % (a, n, b, n), nw))
