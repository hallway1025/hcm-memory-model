#!/usr/bin/env python3
"""exp13_confidence.py —— 置信度信号的正确度量（回应审查 P0-3）。

旧论文引用 corr(m1, correct)=0.222，该数字无任何脚本产出过（审查发现）。
本脚本给出实测值，并区分两个信号：
  m1          = 融合分（吸引子概率 + 组内标准化语义先验）
  raw_evidence= 吸引子重建匹配度 cos(s_q, D·a)
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import random_patterns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D, N, FR = 128, 60, 0.35


def degrade(x, frac, seed):
    r = np.random.default_rng(seed)
    y = x.copy()
    nz = np.where(np.abs(y) > 1e-6)[0]
    y[r.choice(nz, size=max(1, int(len(nz) * frac)), replace=False)] = 0.0
    return unit(y)


X = random_patterns(N, D, seed=61)
Xn = random_patterns(N, D, seed=62)
m = HCM(Config(d_x=D, a=0.02, capacity=600, beta=20.0, L=3, use_resid=True, seed=0))
for x in X:
    m.observe(x)

rec = []
for i, x in enumerate(X):
    q = m.query(degrade(x, FR, i))
    rec.append(dict(kind="in", m1=q["m1"], raw=q["raw_evidence"], correct=int(q["idx"] == i)))
for i, x in enumerate(Xn):
    q = m.query(degrade(x, FR, 500 + i))
    rec.append(dict(kind="out", m1=q["m1"], raw=q["raw_evidence"], correct=0))

m1 = np.array([r["m1"] for r in rec])
raw = np.array([r["raw"] for r in rec])
cor = np.array([r["correct"] for r in rec])

def auc(score, label):
    """Mann-Whitney AUC：score 区分 label=1 的能力"""
    pos = score[label == 1]
    neg = score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))

out = dict(
    n=len(rec),
    base_rate=float(cor.mean()),
    corr_m1=round(float(np.corrcoef(m1, cor)[0, 1]), 4),
    corr_raw=round(float(np.corrcoef(raw, cor)[0, 1]), 4),
    auc_m1=round(auc(m1, cor), 3),
    auc_raw=round(auc(raw, cor), 3),
    raw_in_mean=round(float(raw[:N].mean()), 3),
    raw_out_mean=round(float(raw[N:].mean()), 3),
)
json.dump(out, open(os.path.join(ROOT, "results", "exp13_confidence.json"), "w", encoding="utf-8"), indent=1)

print("样本 %d（库内 %d + 库外 %d，缺失 %d%%），基线正确率 %.2f" % (out["n"], N, N, int(FR * 100), out["base_rate"]))
print()
print("%-14s %-12s %-10s" % ("信号", "corr(·,正确)", "AUC"))
print("%-14s %-12.4f %-10.3f" % ("m1（融合分）", out["corr_m1"], out["auc_m1"]))
print("%-14s %-12.4f %-10.3f" % ("raw_evidence", out["corr_raw"], out["auc_raw"]))
print()
print("=> 融合分 m1 与正确性几乎无关（甚至负相关）。")
print("   真正可用的是 raw_evidence（引子重建匹配度）：库内 %.3f vs 库外 %.3f。" % (
    out["raw_in_mean"], out["raw_out_mean"]))
print()
print("旧论文声称 corr(m1,correct)=0.222 —— 实测为 %.4f，且无任何脚本产出过 0.222。" % out["corr_m1"])
