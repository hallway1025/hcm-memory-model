#!/usr/bin/env python3
"""exp8_meta.py —— 弃权与风险-覆盖率（预测 P7）。

改造后的弃权依据 = raw_evidence（吸引子重建匹配度，见 D-026）。
本脚本给出：阈值扫描的 risk-coverage 曲线 + 两臂对比（元控制 vs 永不弃权）+ 判别力统计。
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import random_patterns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D, N, FR = 128, 40, 0.35


def degrade(x, frac, seed):
    r = np.random.default_rng(seed)
    y = x.copy()
    nz = np.where(np.abs(y) > 1e-6)[0]
    y[r.choice(nz, size=max(1, int(len(nz) * frac)), replace=False)] = 0.0
    return unit(y)


X = random_patterns(N, D, seed=61)
Xn = random_patterns(N, D, seed=62)


def collect(always):
    m = HCM(Config(d_x=D, a=0.02, capacity=600, beta=20.0, L=3, use_resid=True, seed=0, d_z=64))
    if always:
        m.ctl.always_retrieve = True
    m.ctl.theta_raw = -1.0 if always else m.ctl.theta_raw
    for x in X:
        m.observe(x)
    recs = []
    for i, x in enumerate(X):
        q = m.query(degrade(x, FR, i))
        recs.append(dict(kind="in", raw=q["raw_evidence"], m1=q["m1"],
                         answered=q["action"] == "recall", correct=(q["idx"] == i)))
    for i, x in enumerate(Xn):
        q = m.query(degrade(x, FR, 500 + i))
        recs.append(dict(kind="out", raw=q["raw_evidence"], m1=q["m1"],
                         answered=q["action"] == "recall", correct=False))
    return recs


def score(recs, c_err=1.0, c_abst=0.3):
    """风险调整得分：答对 +1，答错 -1，弃权 -0.3。"""
    s = 0.0
    for r in recs:
        if r["answered"]:
            s += 1.0 if r["correct"] else -1.0
        else:
            s -= c_abst
    return s


meta = collect(False)
always = collect(True)

# 阈值扫描（用 meta 臂的记录重新做离线判定，保证曲线与在线策略一致）
ths = [0.0, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
curve = []
for th in ths:
    sel = [r for r in meta if r["raw"] >= th]
    cov = len(sel) / len(meta)
    risk = 1.0 - (sum(1 for r in sel if r["correct"]) / len(sel)) if sel else 0.0
    curve.append(dict(theta=th, coverage=round(cov, 3), risk=round(risk, 3)))

raw_in = np.array([r["raw"] for r in meta if r["kind"] == "in"])
raw_out = np.array([r["raw"] for r in meta if r["kind"] == "out"])
d_prime = float((raw_in.mean() - raw_out.mean()) / np.sqrt(0.5 * (raw_in.var() + raw_out.var())))

res = dict(
    n_each=N, dropout=FR,
    curve=curve,
    raw_in=dict(mean=round(float(raw_in.mean()), 3), std=round(float(raw_in.std()), 3)),
    raw_out=dict(mean=round(float(raw_out.mean()), 3), std=round(float(raw_out.std()), 3)),
    d_prime=round(d_prime, 2),
    score_meta=round(score(meta), 1),
    score_always=round(score(always), 1),
    n_abstain_meta=sum(1 for r in meta if not r["answered"]),
)
json.dump(res, open(os.path.join(ROOT, "results", "exp8_meta.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

print("raw in-store %.3f+-%.3f  novel %.3f+-%.3f   d'=%.2f" %
      (raw_in.mean(), raw_in.std(), raw_out.mean(), raw_out.std(), d_prime))
for c in curve:
    print("  theta=%.2f  coverage=%.3f  risk=%.3f" % (c["theta"], c["coverage"], c["risk"]))
print("score: meta=%.1f  always=%.1f  (abstain %d/%d)" %
      (res["score_meta"], res["score_always"], res["n_abstain_meta"], 2 * N))
