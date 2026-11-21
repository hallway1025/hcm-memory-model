#!/usr/bin/env python3
'''exp34_analyze.py -- pre-registered G-A/G-B/G-C verdicts for exp34 (frozen BEFORE shard data arrives).
G-A: on-arm Rp- deficit (hit_nrp - hit_rp_sup) >= 0.10 with 95% CI excluding 0, AND off-arm
     design-faithful deficit (hit_nrp - hit_rp_cos) < 0.05 (placebo).
G-B: reversibility mean hit_rp_sup_recovered >= 0.80 (on arm, 12 seeds).
G-C: if G-A fails -> honest negative (M8 spec rewrite path).
Paired by seed; bootstrap 95% CI (10000, seed=7) + paired t.
Usage: py -3 bench/exp34_analyze.py [--smoke]
'''
import sys, os, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = "--smoke" in sys.argv
names = ["exp34_rif_interference.json"] if SMOKE else ["exp34_shard%d.json" % i for i in range(3)]
rows = []
for n in names:
    p = os.path.join(ROOT, "results", n)
    if not os.path.exists(p):
        print("missing shard:", n); continue
    rows += json.load(open(p, encoding="utf-8")).get("rows", [])
cells = ["peer-02", "syn_o0.3", "syn_o0.6", "syn_o0.9"]
per = {}
for r in rows:
    per.setdefault((r["cell"], r["arm"]), {})[r["seed"]] = r

def boot_ci(x, b=10000, seed=7):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(b, len(x)))
    m = x[idx].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))

def paired_t(d):
    d = np.asarray(d, float); n = len(d)
    m = d.mean(); sd = d.std(ddof=1) if n > 1 else float("nan")
    t = m / (sd / np.sqrt(n)) if sd and sd > 0 else float("inf") if m else float("nan")
    return float(m), float(sd), float(t)

out = {"n_rows": len(rows), "cells": {}, "gates": {}}
for c in cells:
    on = per.get((c, "on"), {}); off = per.get((c, "off"), {})
    seeds = sorted(set(on) & set(off))
    if not seeds:
        continue
    def series(key, arm_map):
        return np.array([arm_map[s][key] for s in seeds], dtype=object)
    # on-arm mechanism deficit (may contain None if n_rp_sup==0 -> skip those seeds)
    d_sup, d_cos, rec = [], [], []
    for s in seeds:
        r_on, r_off = on[s], off[s]
        if r_on.get("hit_rp_sup") is not None and r_on.get("hit_nrp") is not None:
            d_sup.append(r_on["hit_nrp"] - r_on["hit_rp_sup"])
        if r_on.get("hit_rp_cos") is not None and r_on.get("hit_nrp") is not None:
            d_cos.append(("on", s, r_on["hit_nrp"] - r_on["hit_rp_cos"]))
        if r_off.get("hit_rp_cos") is not None and r_off.get("hit_nrp") is not None:
            d_cos.append(("off", s, r_off["hit_nrp"] - r_off["hit_rp_cos"]))
        if r_on.get("hit_rp_sup_recovered") is not None:
            rec.append(r_on["hit_rp_sup_recovered"])
    d_cos_on = [v for t, s, v in d_cos if t == "on"]
    d_cos_off = [v for t, s, v in d_cos if t == "off"]
    def pack(x):
        if not len(x): return None
        lo, hi = boot_ci(x)
        m, sd, t = paired_t(x)
        return dict(n=len(x), mean=round(m, 4), sd=None if np.isnan(sd) else round(sd, 4),
                    t=None if np.isnan(t) else round(t, 3), ci=([round(lo, 4), round(hi, 4)]))
    cellout = dict(seeds=seeds,
                   deficit_sup_on=pack(d_sup), deficit_cos_on=pack(d_cos_on), deficit_cos_off=pack(d_cos_off),
                   recovered=pack(rec),
                   hit_nrp_on=pack([on[s]["hit_nrp"] for s in seeds if on[s].get("hit_nrp") is not None]),
                   hit_nrp_off=pack([off[s]["hit_nrp"] for s in seeds if off[s].get("hit_nrp") is not None]),
                   hit_rp_sup_on=pack([on[s]["hit_rp_sup"] for s in seeds if on[s].get("hit_rp_sup") is not None]),
                   hit_rp_cos_on=pack([on[s]["hit_rp_cos"] for s in seeds if on[s].get("hit_rp_cos") is not None]),
                   hit_rp_cos_off=pack([off[s]["hit_rp_cos"] for s in seeds if off[s].get("hit_rp_cos") is not None]))
    # paired delta (same seed): d_cos_on[s] - d_cos_off[s]
    both = [on[s]["hit_nrp"] - on[s]["hit_rp_cos"] - (off[s]["hit_nrp"] - off[s]["hit_rp_cos"])
            for s in seeds if all(on[s].get(k) is not None for k in ("hit_nrp", "hit_rp_cos")) and all(off[s].get(k) is not None for k in ("hit_nrp", "hit_rp_cos"))]
    cellout["delta_cos_on_minus_off"] = pack(both)
    out["cells"][c] = cellout

# gate verdicts on the primary hard cell (peer-02) + synthetic sweep
ga = {}
for c in cells:
    cc = out["cells"].get(c)
    if not cc: continue
    d_on = cc["deficit_sup_on"]; d_off = cc["deficit_cos_off"]; rb = cc["recovered"]
    ga[c] = dict(
        gA_on=dict(passed=bool(d_on and d_on["mean"] >= 0.10 and d_on["ci"][0] > 0),
                   mean=d_on and d_on["mean"], ci=d_on and d_on["ci"]),
        gA_off_placebo=dict(passed=bool(d_off and d_off["mean"] < 0.05), mean=d_off and d_off["mean"]),
        gB=dict(passed=bool(rb and rb["mean"] >= 0.80), mean=rb and rb["mean"]))
    ga[c]["G-A"] = "PASS" if ga[c]["gA_on"]["passed"] and ga[c]["gA_off_placebo"]["passed"] else "FAIL"
    ga[c]["G-B"] = "PASS" if ga[c]["gB"]["passed"] else "FAIL"
out["gates"] = ga
out["G-C"] = "NEGATIVE-HONEST" if any(v["G-A"] == "FAIL" for v in ga.values()) else "N/A"
json.dump(out, open(os.path.join(ROOT, "results", "exp34_analyze.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps(out, indent=1))
