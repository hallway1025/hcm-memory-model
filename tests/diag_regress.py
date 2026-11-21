
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import overlapping_set
D = 128
X = overlapping_set(60, D, overlap=0.70, seed=77)
m = HCM(Config(d_x=D, a=0.02, capacity=800, beta=20.0, L=3, use_resid=True, seed=0))
for k, x in enumerate(X):
    rec = m.observe(x)
    if k < 6:
        print("t=%d nu=%.3f eps=%.3f g=%.3f wrote=%s |mu|=%.2f" % (
            k, rec["nu"], rec["eps"], rec["g"], rec["wrote"], float(np.linalg.norm(m.enc.mu))))
print("total written:", m.E.n, "/60")
gs = [r["g"] for r in m.log]
print("g stats: mean=%.3f min=%.3f max=%.3f  theta_write=%.2f" % (
    np.mean(gs), min(gs), max(gs), m.cfg.theta_write))
