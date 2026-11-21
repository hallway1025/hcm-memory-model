
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit, softmax_beta
from hcm.data import random_patterns
D=128
def access(m, X, idxs, beta_probe):
    tot = 0.0
    for i in idxs:
        xe = m.enc.resid_read(X[i]); s_pre = m.enc.pre(xe); q0 = m.enc.encode_from_pre(s_pre)
        z, _ = m.E.retrieve(q0, L=3, beta=20.0, dense=s_pre)
        ev = m.E.M[:, :m.E.n].T @ z
        a = softmax_beta(ev, beta_probe) * m.E.sup[:m.E.n]
        a = a / (a.sum() + 1e-12)
        tot += float(a[i])
    return tot/len(idxs)

for beta_probe in (2.0, 5.0, 20.0):
    out = []
    for rif_on in (True, False):
        m = HCM(Config(d_x=D, a=0.02, capacity=800, beta=20.0, L=3, use_resid=True, seed=0,
                       rho_rif=0.30, rif_on=rif_on))
        X = random_patterns(40, D, seed=51)
        for x in X: m.observe(x)
        prac, unprac = list(range(20)), list(range(20,40))
        b = access(m, X, unprac, beta_probe)
        for _ in range(5):
            for i in prac: m.query(X[i], do_rif=True)
        a1 = access(m, X, unprac, beta_probe)
        a2 = access(m, X, prac, beta_probe)
        out.append((rif_on, b, a1, a2))
    print("beta_probe=%.0f" % beta_probe)
    for rif_on, b, a1, a2 in out:
        print("   RIF=%-5s 未练 %.5f -> %.5f (Δ%+.5f)   已练 %.5f" % (rif_on, b, a1, a1-b, a2))
