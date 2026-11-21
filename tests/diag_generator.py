
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit

D = 128

def make(share, n=30, seed=11):
    r = np.random.default_rng(seed)
    sh = r.standard_normal(D).astype(np.float32)
    return np.asarray([unit(share*sh + (1-share)*r.standard_normal(D).astype(np.float32)) for _ in range(n)], dtype=np.float32)

print("%-7s %-9s %-9s %-8s %-8s %-10s %s" % ("share","输入余弦","无残差码cos","残差码cos","无残差召回","残差召回","无残差支撑集唯一数"))
for share in (0.90, 0.95, 0.98, 0.99, 0.995, 0.999):
    X = make(share)
    inp = float(np.mean((X @ X.T)[np.triu_indices(30,1)]))
    out = {}
    for resid in (False, True):
        m = HCM(Config(d_x=D, a=0.02, capacity=500, beta=20.0, L=0, use_resid=resid, seed=0))
        for x in X: m.observe(x)
        H = m.E.M[:, :m.E.n]
        cc = float(np.mean((H.T@H)[np.triu_indices(m.E.n,1)])) if m.E.n>1 else 0
        ok = sum(int(m.query(X[i])["idx"] == i) for i in range(30))
        sup = len(set(tuple(np.where(H[:,j]>0)[0]) for j in range(m.E.n)))
        out[resid] = (cc, ok, m.E.n, sup)
    print("%-7.3f %-9.3f %-9.3f %-8.3f %-8s %-10s %d/%d" % (
        share, inp, out[False][0], out[True][0],
        "%d/30"%out[False][1], "%d/30"%out[True][1], out[False][3], out[False][2]))
