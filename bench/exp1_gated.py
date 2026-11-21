
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import overlapping_set
D = 128
def degrade(x, frac, seed):
    r = np.random.default_rng(seed); y = x.copy()
    nz = np.where(np.abs(y) > 1e-6)[0]
    y[r.choice(nz, size=max(1,int(len(nz)*frac)), replace=False)] = 0.0
    return unit(y)

print("=== 门控修复后（nu=期望偏离）的分离/完成表 ===")
print("%-8s %-6s %-4s | %-8s %-8s %-8s | %s" % ("overlap","resid","L","noise0","noise20","noise35","写入"))
rows = []
rng = np.random.default_rng(11)
RAND = np.asarray([unit(rng.standard_normal(D).astype(np.float32)) for _ in range(30)])
for overlap in (0.0, 0.5, 0.75, 0.9):
    X = RAND if overlap == 0.0 else overlapping_set(30, D, overlap=overlap, seed=11)
    for resid in (False, True):
        for L in (0, 3):
            m = HCM(Config(d_x=D, a=0.02, capacity=500, beta=20.0, L=L, use_resid=resid, seed=0))
            for x in X: m.observe(x)
            v = [sum(int(m.query(degrade(X[i], nz_, i))["idx"] == i) for i in range(30)) for nz_ in (0.0, 0.2, 0.35)]
            rows.append(dict(overlap=overlap, resid=resid, L=L, written=m.E.n, n0=v[0], n20=v[1], n35=v[2]))
            print("%-8.2f %-6s %-4d | %2d/30    %2d/30    %2d/30    | %d/30" % (overlap, resid, L, v[0], v[1], v[2], m.E.n))
json.dump(rows, open(os.path.join(ROOT,"results","exp1_gated.json"),"w"), indent=1)

# 去重验证：同一模式重复不该反复写入
m = HCM(Config(d_x=D, a=0.02, capacity=500, beta=20.0, L=3, use_resid=True, seed=0))
x0 = RAND[0]
for _ in range(10): m.observe(x0)
print("\n重复同一模式 10 次 -> 库内条数 =", m.E.n, "(应为 1)")
