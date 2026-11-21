
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import overlapping_set
D=128

def run(overlap, tw=0.5, n=30):
    X = overlapping_set(n, D, overlap=overlap, seed=11)
    m = HCM(Config(d_x=D, a=0.02, capacity=500, beta=20.0, L=0, use_resid=True, seed=0, theta_write=tw))
    for x in X:
        # nu = 与期望模板的偏离（CA1 失配的本来含义），而非「是否已存过」
        x_n = unit(x)
        mu = m.enc.mu
        nu = float(np.linalg.norm(x_n - mu)) if np.linalg.norm(mu) > 1e-6 else 1.0
        rec = m.observe(x)
        m.log[-1]["nu_exp"] = nu
    ok = sum(int(m.query(X[i])["idx"] == i) for i in range(n))
    nus = [r["nu_exp"] for r in m.log]
    return ok, m.E.n, np.mean(nus), min(nus), max(nus)

print("nu = ||x - mu||（期望偏离）作为新颖性：")
print("%-8s %-10s %-10s %s" % ("overlap","召回","写入","nu 范围"))
for ov in (0.0, 0.5, 0.75, 0.90):
    ok, nw, mnu, lo, hi = run(ov)
    print("%-8.2f %-10s %-10s [%.3f, %.3f] mean %.3f" % (ov, "%d/30"%ok, "%d/30"%nw, lo, hi, mnu))
