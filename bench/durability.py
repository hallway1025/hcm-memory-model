#!/usr/bin/env python3
"""check_durability.py —— 抗遗忘/抗干扰的强度扫描（门禁 G4 补充）。

单个 ρ_rif 值测不出 RIF，是因为基线本身已在天花板（10/10）。
本脚本做的是**强度扫描**：把 θ_write 调低使库内充满相互干扰的条目，
或把 ρ_rif 调到 0.5 观察未练条目的可及性是否真的下降。
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import random_patterns, overlapping_set

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = 128
rows = []


def rif_experiment(rho_rif, n_items=60, practice=30, reps=5, overlap=0.85):
    """高重叠库 + 多次练习 + 强抑制，测未练条目可及性。"""
    m = HCM(Config(d_x=D, a=0.02, capacity=800, beta=20.0, L=3, use_resid=True,
                   seed=0, rho_rif=rho_rif, rif_on=True))
    X = overlapping_set(n_items, D, overlap=overlap, seed=77)
    for x in X:
        m.observe(x)
    unpracticed = list(range(practice, n_items))
    base_ok = sum(int(m.query(X[i], do_rif=False)["idx"] == i) for i in unpracticed)
    for _ in range(reps):
        for i in range(practice):
            m.query(X[i], do_rif=True)
    after_ok = sum(int(m.query(X[i], do_rif=False)["idx"] == i) for i in unpracticed)
    return base_ok, after_ok, len(unpracticed)


print("=== RIF 强度扫描（高重叠库 overlap=0.85，60 条，练 30 条 x5 轮）===", flush=True)
for rho in (0.0, 0.15, 0.30, 0.50, 0.70):
    b, a, n = rif_experiment(rho)
    rows.append(dict(exp="rif", rho_rif=rho, base=b, after=a, n=n, drop=a - b))
    print("  rho_rif=%.2f  未练条目 基线 %2d/%d -> 练后 %2d/%d  (Δ%+d)" % (rho, b, n, a, n, a - b), flush=True)


def gate_selectivity(theta_write, n_events=20, n_bg_reps=30):
    """在新颖性修复后，测门控的选择性：显著事件 vs 重复背景。"""
    cfg = Config(d_x=D, a=0.02, capacity=900, beta=20.0, L=3, use_resid=True,
                 seed=0, theta_write=theta_write)
    m = HCM(cfg)
    r = np.random.default_rng(5)
    bg = random_patterns(2, D, seed=41)
    events = random_patterns(n_events, D, seed=42)
    for _ in range(n_bg_reps):
        for b in bg:
            m.observe(unit(b + 0.03 * r.normal(size=D)), r=0.0)
    for e in events:
        m.observe(unit(e), r=1.0)
    g_bg = np.mean([rec["g"] for rec in m.log[:n_bg_reps * 2]])
    g_ev = np.mean([rec["g"] for rec in m.log[n_bg_reps * 2:]])
    n_written = m.E.n
    # 事件召回
    hit = sum(int(m.query(e)["idx"] >= 0) for e in events)
    return g_bg, g_ev, n_written, hit, n_events


print("\n=== 门控选择性 vs 写入门限（新颖性修复后）===", flush=True)
for tw in (0.5, 0.7, 0.8, 0.9):
    gb, ge, nw, hit, ne = gate_selectivity(tw)
    rows.append(dict(exp="gate", theta_write=tw, g_bg=round(float(gb), 3),
                     g_ev=round(float(ge), 3), written=nw, recall=hit, n_ev=ne))
    print("  theta_write=%.1f  g(背景)=%.3f g(事件)=%.3f  写入 %3d/50  事件召回 %2d/%d" %
          (tw, gb, ge, nw, hit, ne), flush=True)

json.dump(rows, open(os.path.join(ROOT, "results", "durability.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\nsaved results/durability.json")
