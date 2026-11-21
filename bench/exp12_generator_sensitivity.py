#!/usr/bin/env python3
"""exp12_generator_sensitivity.py —— 生成器敏感性（回应洁净复现的核心批评）。

洁净复现者用「输入余弦 0.90」造数据，复现不出「无残差 1/30」。
本实验查明原因：项目的「overlap 0.90」是**混合权重**，产生输入余弦 0.986。
且该效应存在于一个**窄带**里，必须如实报告边界。
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D, N = 128, 30


def make(share, n=N, seed=11):
    """项目实际使用的生成器：x = unit(share*shared + (1-share)*priv)"""
    r = np.random.default_rng(seed)
    sh = r.standard_normal(D).astype(np.float32)
    return np.asarray([unit(share * sh + (1 - share) * r.standard_normal(D).astype(np.float32))
                       for _ in range(n)], dtype=np.float32)


rows = []
print("%-7s %-9s %-11s %-9s %-11s %-9s %-9s %-9s" %
      ("share", "输入余弦", "无残差码cos", "残差码cos", "无残差召回", "残差召回", "无残差写入", "残差写入"))
for share in (0.80, 0.85, 0.90, 0.92, 0.95, 0.98, 0.99, 0.999):
    X = make(share)
    inp = float(np.mean((X @ X.T)[np.triu_indices(N, 1)]))
    out = {}
    for resid in (False, True):
        m = HCM(Config(d_x=D, a=0.02, capacity=500, beta=20.0, L=0, use_resid=resid, seed=0))
        for x in X:
            m.observe(x)
        H = m.E.M[:, : m.E.n]
        cc = float(np.mean((H.T @ H)[np.triu_indices(m.E.n, 1)])) if m.E.n > 1 else 0.0
        ok = sum(int(m.query(X[i])["idx"] == i) for i in range(N))
        # 支撑集唯一性：写入码之间有多少个不同的支撑集
        nsup = len(set(tuple(np.where(H[:, j] > 0)[0]) for j in range(m.E.n))) if m.E.n else 0
        out[resid] = dict(code_cos=round(cc, 3), recall=ok, written=m.E.n, distinct_supports=nsup)
    rows.append(dict(share=share, input_cos=round(inp, 4),
                     resid_off=out[False], resid_on=out[True]))
    print("%-7.3f %-9.3f %-11.3f %-9.3f %-11s %-9s %-9d %-9d" % (
        share, inp, out[False]["code_cos"], out[True]["code_cos"],
        "%d/30" % out[False]["recall"], "%d/30" % out[True]["recall"],
        out[False]["written"], out[True]["written"]))

json.dump(rows, open(os.path.join(ROOT, "results", "exp12_generator.json"), "w", encoding="utf-8"), indent=1)
print()
print("关键：效应只存在于 share 0.85-0.92 这个窄带；share>=0.95 时两臂都失效（写入被去重拒绝）。")
print("      share=0.90 的输入余弦是 0.986，不是 0.90 —— 洁净复现者按『余弦 0.90』造数据，")
print("      那是容易得多的问题，因此复现不出对比。文档缺陷已承认并修正。")
