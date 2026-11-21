
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import overlapping_set, random_patterns
D=128

def test(overlap, lam, n=30):
    X = overlapping_set(n, D, overlap=overlap, seed=11)
    m = HCM(Config(d_x=D, a=0.02, capacity=500, beta=20.0, L=3, use_resid=True, seed=0))
    m.cfg.lam = lam
    for x in X: m.observe(x)
    ok = sum(int(m.query(X[i])["idx"] == i) for i in range(n))
    return ok, m.E.n

print("lam 语义权重扫描（精确认出 / 写入数），overlap 0.90 与 0.75：")
print("%-6s %-14s %-14s" % ("lam", "overlap=0.90", "overlap=0.75"))
for lam in (1.0, 0.8, 0.6, 0.4, 0.2, 0.0):
    a90 = test(0.90, lam); a75 = test(0.75, lam)
    print("%-6.1f %-14s %-14s" % (lam, "%d/30 (n=%d)" % a90, "%d/30 (n=%d)" % a75))
