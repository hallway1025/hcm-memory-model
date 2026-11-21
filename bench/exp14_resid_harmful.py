#!/usr/bin/env python3
"""exp14_resid_harmful.py —— 「残差在随机数据上有害」的多带子检验（回应审查 P0-1）。

审查指出：论文 §4.3 表 3 最后一行（N=16000，无残差 8/20 vs 有残差 3/20）是**单种子**，
而同一篇论文为其他主张补了 12 种子 —— 这是「选择性统计严谨」。

本脚本用 8 个种子重跑 N=8000 / 16000，报告 mean ± 95%CI。
若效应存在，两臂的 CI 应不重叠；若重叠，则论文必须撤回该反主张。
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = 128
SEEDS = list(range(8))


def pats(N, d, seed):
    r = np.random.default_rng(seed)
    return np.asarray([unit(v) for v in r.standard_normal((N, d))], dtype=np.float32)


def degrade(x, frac, seed):
    r = np.random.default_rng(seed)
    y = x.copy()
    nz = np.where(np.abs(y) > 1e-6)[0]
    y[r.choice(nz, size=max(1, int(len(nz) * frac)), replace=False)] = 0.0
    return unit(y)


def ci95(v):
    v = np.asarray(v, float)
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))


rows = []
print("%-8s %-8s %-16s %-16s %s" % ("N", "残差", "精确线索", "25%缺失线索", "写入"))
for N in (4000, 8000, 16000):
    for resid in (False, True):
        exact, deg, wr = [], [], []
        for s in SEEDS:
            X = pats(N, D, seed=100 + s)
            m = HCM(Config(d_x=D, a=0.02, capacity=N + 10, beta=20.0, L=3,
                           use_resid=resid, seed=s))
            for x in X:
                m.observe(x)
            probe = list(range(0, N, max(1, N // 40)))[:40]
            exact.append(sum(int(m.query(X[i])["idx"] == i) for i in probe) / len(probe))
            deg.append(sum(int(m.query(degrade(X[i], 0.25, i))["idx"] == i) for i in probe) / len(probe))
            wr.append(m.E.n)
        em, ec = ci95(exact)
        dm, dc = ci95(deg)
        rows.append(dict(N=N, resid=resid, exact_mean=round(em, 3), exact_ci=round(ec, 3),
                         degraded_mean=round(dm, 3), degraded_ci=round(dc, 3),
                         written=round(float(np.mean(wr)), 1), seeds=len(SEEDS)))
        print("%-8d %-8s %-16s %-16s %.0f" % (
            N, resid, "%.3f±%.3f" % (em, ec), "%.3f±%.3f" % (dm, dc), np.mean(wr)))
        sys.stdout.flush()

json.dump(rows, open(os.path.join(ROOT, "results", "exp14_resid_harmful.json"), "w", encoding="utf-8"), indent=1)
print()
print("判定方式：N=16000 时，若两臂 25%缺失 的 95%CI 不重叠且无残差更高，则「残差有害」成立；否则必须撤回。")
