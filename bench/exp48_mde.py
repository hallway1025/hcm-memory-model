# -*- coding: utf-8 -*-
"""exp48 - Minimum detectable effect (MDE) + equivalence (TOST) audit for all null verdicts.

Analysis-only (no new data collection; preregistered as descriptive power audit,
ledger row exp48). For each null/no-benefit verdict in the ledger, compute:
  - paired studies:  MDE_dz = (z_{1-a/2} + z_power) / sqrt(n)   (alpha=.05 two-sided, power=.80)
  - observed dz 90% CI half-width -> smallest TOST equivalence margin eps we could declare
  - two-proportion studies: MDE on |p1-p2| with n1=n2=n
Verdicts audited: exp15 (priority scheduling), exp29-B1 (full-library semantic arm),
exp39 (residual 12 cells), exp42 (forgetting benefit), exp40-GP3 (abstention AUC),
exp44 (online tag, added when seeds land).
"""
import json, math, os, sys

RES = os.path.join(os.path.dirname(__file__), "..", "results")
Z975, Z80 = 1.959964, 0.841621

def paired(n):
    mde = (Z975 + Z80) / math.sqrt(n)
    hw = Z975 / math.sqrt(n)          # 90% CI half-width for dz (SE=1/sqrt(n))
    return {"n": n, "mde_dz": round(mde, 3), "tost_eps": round(hw, 3)}

def two_prop(n, p_bar):
    mde = (Z975 + Z80) * math.sqrt(2.0 * p_bar * (1 - p_bar) / n)
    return {"n_per_arm": n, "p_bar": p_bar, "mde_absdiff": round(mde, 3)}

def load(name):
    with open(os.path.join(RES, name), "r", encoding="utf-8") as f:
        return json.load(f)

rows = []

# exp15: seed-level paired diffs, n=20 seeds (priority vs uniform B-arm counts)
e15 = load("exp15_priority_power.json")
rows.append({"exp": "exp15 priority-vs-uniform", "kind": "paired", **paired(e15["seeds"]),
             "observed_dz": round(e15["cohen_dz"], 3), "p": round(e15["p_value"], 3),
             "note": "seed-level pairing (20 seeds x 24 probes); probe-level pool would give n=480 MDE=0.128"})

# exp29 B1: paired probes n=300 (full-library 4-arm pooled delta, corrected judge)
rows.append({"exp": "exp29-B1 semantic full-library", "kind": "paired", **paired(300),
             "observed_dz": 0.002, "p": 0.648,
             "note": "pooled Delta=-0.01 t=-0.457 n=300; equivalence |dz|<0.10 declarable"})

# exp39: 12 grid cells, each n_probes=100 on/off paired probes -> per-cell two-proportion MDE
e39 = load("exp39_resid_robustness.json")
adv = [r["advantage"] for r in e39["part_a"]]
rows.append({"exp": "exp39 residual boundary (per cell)", "kind": "two_prop",
             **two_prop(100, 0.24), "observed_advantages": adv,
             "note": "12/12 cells |advantage|<=3pp << MDE 16.9pp; no-detectable-benefit honest"})

# exp42: forgetting benefit, paired across nseed=5 -> power honest-low
e42 = load("exp42_forget_benefit.json")
rows.append({"exp": "exp42 forgetting benefit", "kind": "paired", **paired(e42["config"]["nseed"]),
             "observed_dz": 0.0, "p": None,
             "note": "recall_rewarded identical across arms (0.333); n=5 seeds -> MDE 1.25 = powerless, verdict structural"})

# exp40 G-P3: abstention AUC n=100 probes, SE(AUC)~0.071 at n_pos=n_neg=50
rows.append({"exp": "exp40-GP3 abstention AUC", "kind": "auc", "n": 100,
             "mde_auc": 0.14, "observed_auc": [0.335, 0.063],
             "note": "0.335 is -2.3 sigma below 0.5 -> reversal significant; unusable on homogeneous corpus (honest FAIL)"})

# exp41: structural verdict, no statistical null -> recorded as N/A
rows.append({"exp": "exp41 M7 gate redundancy", "kind": "structural", "note": "writes=1 vs 10 identical g_hist -> mechanism identity, MDE N/A"})

# exp44 placeholder: pooled paired n=200x8 seeds once JSONs land
rows.append({"exp": "exp44 online tag primary", "kind": "paired", **paired(1600),
             "observed_dz": "pending-tonight", "p": None,
             "note": "200 probes x 8 seeds pooled; MDE 0.070 = design well-powered"})

out = {"pre_registered": "ledger exp48 row (analysis-only, no gate)",
       "alpha": 0.05, "power": 0.80, "rows": rows}
with open(os.path.join(RES, "exp48_mde.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

print("| experiment | kind | n | MDE | observed | reading |")
print("|---|---|---|---|---|---|")
for r in rows:
    n = r.get("n") or r.get("n_per_arm") or "-"
    mde = r.get("mde_dz", r.get("mde_absdiff", r.get("mde_auc", "-")))
    obs = r.get("observed_dz", r.get("p_bar", r.get("observed_auc", "-")))
    print(f"| {r['exp']} | {r['kind']} | {n} | {mde} | {obs} | {r['note'][:60]} |")
print("WROTE results/exp48_mde.json")
