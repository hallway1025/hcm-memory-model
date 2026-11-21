
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import random_patterns
D=128

def occasion(use_ctx, kappa):
    """同一内容出现在两种情境下，测能否按情境取回对应事件。
    情境以偏置项注入内容空间；关键是查询时**必须给出情境线索**，
    否则这个实验测的是内容检索，不是情境绑定（L-022）。"""
    cfg = Config(d_x=D, a=0.02, capacity=600, beta=20.0, L=3, use_resid=True,
                 seed=0, use_ctx_bias=use_ctx, kappa_ctx=kappa)
    m = HCM(cfg)
    r = np.random.default_rng(7)
    contents = random_patterns(10, D, seed=31)
    ctxA, ctxB = unit(r.normal(size=D)), unit(r.normal(size=D))
    events = []
    for c in contents:
        m.observe(unit(c + 1.2 * ctxA)); events.append(("A", c))
    for c in contents:
        m.observe(unit(c + 1.2 * ctxB)); events.append(("B", c))
    ok = 0
    for k, (tag, c) in enumerate(events):
        m.c = (1.2 * (ctxA if tag == "A" else ctxB)).astype(np.float32)
        q = m.query(unit(c + 1.2 * (ctxA if tag == "A" else ctxB)))
        ok += int(q["idx"] == k)
    return ok, len(events), m.E.n

print("M3_CTX 情境绑定（正确设计：查询时给情境线索）")
print("%-10s %-8s %-12s %s" % ("ctx_bias", "kappa", "命中", "写入"))
for use in (False, True):
    for kappa in (0.6, 3.0, 10.0):
        ok, n, nw = occasion(use, kappa)
        print("%-10s %-8.1f %-12s %d" % (use, kappa, "%d/%d" % (ok, n), nw))
