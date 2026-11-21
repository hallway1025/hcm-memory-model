#!/usr/bin/env python3
'''exp34b_analyze.py -- frozen analyzer for exp34b (written before full-scale data; smoke-tested).
Merges shards; computes pre-registered co-primary gates G-A'/G-S1/G-S2, recovery gates
G-B'/G-B'', exploratory dose-response. Bootstrap 10000 resamples seed=7, seed-level paired.
'''
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARDS = ["exp34b_shard0.json", "exp34b_shard1.json", "exp34b_shard2.json"]
rows = []
for f in SHARDS:
    p_ = os.path.join(ROOT, "results", f)
    if os.path.exists(p_):
        rows += json.load(open(p_, encoding="utf-8"))["rows"]
if not rows:
    print("NO DATA"); sys.exit(1)

def boot_ci(vals, n=10000, seed=7):
    v = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    m = v[idx].mean(axis=1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]

def mean(v):
    v = [x for x in v if x is not None and not np.isnan(x)]
    return round(float(np.mean(v)), 4) if v else None

def arm(a):
    return sorted([r for r in rows if r["arm"] == a], key=lambda r: r["seed"])

out = {"n_rows": len(rows), "seeds": sorted(set(r["seed"] for r in rows))}

# G-A' below-baseline RIF on suppressed slots
d15 = [r["hit_nrp"] - r["hit_rp_sup"] for r in arm("son015") if not np.isnan(r["hit_rp_sup"])]
gA = {}
if d15:
    ci = boot_ci(d15)
    gA = {"deficit_mean": round(float(np.mean(d15)), 4), "deficit_sd": round(float(np.std(d15, ddof=1)), 4),
          "ci95": ci, "pass_ge_0p10": bool(np.mean(d15) >= 0.10), "ci_excludes_0": bool(ci[0] > 0 or ci[1] < 0)}
pl = [r["hit_nrp"] - r["hit_rp_cos"] for r in arm("soff") if not np.isnan(r["hit_rp_cos"])]
gA["placebo_deficit_mean"] = mean(pl) if pl else None
gA["placebo_lt_0p05"] = bool(pl and np.mean(pl) < 0.05)
gA["verdict"] = "PASS" if (gA.get("pass_ge_0p10") and gA.get("ci_excludes_0") and gA.get("placebo_lt_0p05")) else "FAIL->G-C"
out["G-A"] = gA

# G-S1 practiced-item collapse (paired seeds son030 vs soff)
soff = arm("soff"); s30 = arm("son030")
by_s_off = {r["seed"]: r for r in soff}
dp = [by_s_off[r["seed"]]["hit_p"] - r["hit_p"] for r in s30 if r["seed"] in by_s_off and not np.isnan(r["hit_p"])]
gS1 = {}
if dp:
    ci = boot_ci(dp)
    gS1 = {"collapse_mean": round(float(np.mean(dp)), 4), "collapse_sd": round(float(np.std(dp, ddof=1)), 4),
           "ci95": ci, "pass_ge_0p10": bool(np.mean(dp) >= 0.10), "ci_excludes_0": bool(ci[0] > 0 or ci[1] < 0)}
gS1["verdict"] = "PASS" if gS1.get("pass_ge_0p10") and gS1.get("ci_excludes_0") else "FAIL"
out["G-S1"] = gS1

# G-S2 hub formation
hs = {a: mean([r["hub_share"] for r in arm(a)]) for a in ("soff", "son015", "son030", "son050")}
gS2 = {"hub_share": hs, "threshold": None}
if hs["son030"] is not None and hs["soff"] is not None:
    thr = max(0.15, 2.0 * hs["soff"])
    gS2["threshold"] = round(thr, 4)
    gS2["verdict"] = "PASS" if hs["son030"] >= thr else "FAIL"
out["G-S2"] = gS2

# G-B' exact inverse restores suppressed-slot hits; G-B'' full restore returns hit_p to soff level
rec = [(r["hit_rp_sup_recovered"], r["hit_p"], r["hit_p_all_recovered"], r["seed"]) for r in arm("son015") if "hit_rp_sup_recovered" in r]
gB = {}
if rec:
    gB["recovered_mean"] = mean([x[0] for x in rec])
    gB["gap_rec_hp_mean"] = round(float(np.mean([abs(x[0] - x[1]) for x in rec])), 4)
    gB["pass_le_0p05"] = bool(np.mean([abs(x[0] - x[1]) for x in rec]) <= 0.05)
    off_hp = {r["seed"]: r["hit_p"] for r in soff}
    gB["hp_all_rec_mean"] = mean([x[2] for x in rec])
    gB["gap_to_soff_hp_mean"] = round(float(np.mean([abs(x[2] - off_hp[x[3]]) for x in rec if x[3] in off_hp and not np.isnan(off_hp[x[3]])])), 4)
    gB["pass_fullrestore"] = bool(np.mean([abs(x[2] - off_hp[x[3]]) for x in rec if x[3] in off_hp and not np.isnan(off_hp[x[3]])]) <= 0.05)
out["G-B"] = gB

# dose-response (exploratory)
dose = {}
for a in ("son015", "son030", "son050"):
    d = [r["hit_nrp"] - r["hit_rp_sup"] for r in arm(a) if not np.isnan(r["hit_rp_sup"])]
    dose[a] = {"deficit_mean": mean(d) if d else None, "hit_p_mean": mean([r["hit_p"] for r in arm(a)]),
               "hub_share": mean([r["hub_share"] for r in arm(a)]),
               "att_rp_sup_mean": mean([r["att_rp_sup"] for r in arm(a)])}
out["dose_response"] = dose
defm = [dose[a]["deficit_mean"] for a in ("son015", "son030", "son050")]
hpm = [dose[a]["hit_p_mean"] for a in ("son015", "son030", "son050")]
out["dose_monotone_deficit"] = bool(all(x is not None for x in defm) and defm[0] < defm[1] < defm[2])
out["dose_monotone_hitp_collapse"] = bool(all(x is not None for x in hpm) and hpm[0] > hpm[1] > hpm[2])

verdicts = [out["G-A"]["verdict"], gS1.get("verdict", "?"), gS2.get("verdict", "?")]
out["headline"] = {"G-A": out["G-A"]["verdict"], "G-S1": gS1.get("verdict"), "G-S2": gS2.get("verdict"),
                   "G-B_fullrestore": gB.get("pass_fullrestore")}
json.dump(out, open(os.path.join(ROOT, "results", "exp34b_analyze.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps(out["headline"], indent=1))
print("hub:", json.dumps(hs), "| hit_p collapse mono:", out["dose_monotone_hitp_collapse"])
