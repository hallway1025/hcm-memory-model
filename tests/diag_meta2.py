
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import random_patterns
D = 128
def degrade(x, frac, seed):
    r = np.random.default_rng(seed); y = x.copy()
    nz = np.where(np.abs(y) > 1e-6)[0]
    y[r.choice(nz, size=max(1,int(len(nz)*frac)), replace=False)] = 0.0
    return unit(y)
for always in (False, True):
    m = HCM(Config(d_x=D, a=0.02, capacity=600, beta=20.0, L=3, use_resid=True, seed=0, d_z=64))
    if always: m.ctl.always_retrieve = True
    X = random_patterns(40, D, seed=61); Xn = random_patterns(40, D, seed=62)
    for x in X: m.observe(x)
    raws_in, raws_out = [], []
    score = 0.0; acts = {}
    for i, x in enumerate(X):
        q = m.query(degrade(x, 0.35, i))
        raws_in.append(q["raw_evidence"]); acts[q["action"]] = acts.get(q["action"],0)+1
        score += (1.0 if q["idx"]==i else -1.0) if q["action"]=="recall" else -0.3
    for i, x in enumerate(Xn):
        q = m.query(degrade(x, 0.35, 500+i))
        raws_out.append(q["raw_evidence"]); acts[q["action"]] = acts.get(q["action"],0)+1
        score += -1.0 if q["action"]=="recall" else -0.3
    print(("always " if always else "meta   ") +
          "score=%7.1f acts=%s" % (score, acts))
    print("        raw in-store %.3f..%.3f (mean %.3f) | novel %.3f..%.3f (mean %.3f)" %
          (min(raws_in), max(raws_in), np.mean(raws_in), min(raws_out), max(raws_out), np.mean(raws_out)))
