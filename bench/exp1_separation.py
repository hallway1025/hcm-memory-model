#!/usr/bin/env python3
"""exp1_separation.py —— 分离/完成主表（RQ1）。

2x2 设计：期望库残差 on/off × Hopfield 迭代 0/3，跨 overlap 与线索缺失率。
新颖性用 ν = 相对期望偏离（L-020 第 5 代），门控因此是自适应的而非恒开。
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import overlapping_set

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D, N = 128, 30


def degrade(x, frac, seed):
    if frac <= 0:
        return x
    r = np.random.default_rng(seed)
    y = x.copy()
    nz = np.where(np.abs(y) > 1e-6)[0]
    y[r.choice(nz, size=max(1, int(len(nz) * frac)), replace=False)] = 0.0
    return unit(y)


rng = np.random.default_rng(11)
RAND = np.asarray([unit(rng.standard_normal(D).astype(np.float32)) for _ in range(N)])
rows = []

for overlap in (0.0, 0.5, 0.75, 0.90):
    X = RAND if overlap == 0.0 else overlapping_set(N, D, overlap=overlap, seed=11)
    for resid in (False, True):
        for L in (0, 3):
            m = HCM(Config(d_x=D, a=0.02, capacity=500, beta=20.0, L=L, use_resid=resid, seed=0))
            for x in X:
                m.observe(x)
            H = m.E.M[:, : m.E.n]
            code_cos = float(np.mean((H.T @ H)[np.triu_indices(m.E.n, 1)])) if m.E.n > 1 else 0.0
            v = [sum(int(m.query(degrade(X[i], nz_, i))["idx"] == i) for i in range(N))
                 for nz_ in (0.0, 0.2, 0.35)]
            rows.append(dict(overlap=overlap, resid=resid, L=L, written=m.E.n,
                             code_cos=round(code_cos, 3), n0=v[0], n20=v[1], n35=v[2], N=N))

print("%-8s %-6s %-4s | %-8s %-8s %-8s | %-10s %s" %
      ("overlap", "resid", "L", "noise0", "noise20", "noise35", "code_cos", "写入"))
for overlap in (0.0, 0.5, 0.75, 0.90):
    for resid in (False, True):
        for L in (0, 3):
            r = [x for x in rows if (x["overlap"], x["resid"], x["L"]) == (overlap, resid, L)][0]
            print("%-8.2f %-6s %-4d | %2d/30    %2d/30    %2d/30    | %-10.3f %d/30" %
                  (overlap, resid, L, r["n0"], r["n20"], r["n35"], r["code_cos"], r["written"]))

# 去重：同一模式重复不应反复写入
m = HCM(Config(d_x=D, a=0.02, capacity=500, beta=20.0, L=3, use_resid=True, seed=0))
for _ in range(10):
    m.observe(RAND[0])
dedup = m.E.n
print("\n重复同一模式 10 次 -> 库内 %d 条（期望 1）" % dedup)

json.dump(dict(rows=rows, dedup_writes=dedup),
          open(os.path.join(ROOT, "results", "exp1_separation.json"), "w", encoding="utf-8"), indent=1)
print("saved results/exp1_separation.json")
