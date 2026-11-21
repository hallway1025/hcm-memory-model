# -*- coding: utf-8 -*-
"""exp44 聚合分析：三臂配对统计 + 预注册门判决（D-066）。

输入 results/exp44_s{0..7}.json（--smoke 读 exp44_smoke.json）。
primary = intrain200 deg0.3 semantic-only (lam=0) corrected-criterion hit rate。
统计：同 seed 三臂同库配对 -> mean±sd / sign-flip 置换 p / Cohen dz（exp15 同款）。
门：B>A 正结果如实报；B<=A 负结果如实报（两分支都在 JSON 里落字）。
"""
import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ("A_fixed", "B_online", "C_uniform")
PRIMARY = "intrain200|lam0|deg0.3"


def load(seed_files):
    out = []
    for f in seed_files:
        with open(f, encoding="utf-8") as fh:
            out.append(json.load(fh))
    return out


def pair_stats(d):
    d = np.asarray(d, dtype=np.float64)
    n = d.size
    mean, sd = float(d.mean()), float(d.std(ddof=1)) if n > 1 else 0.0
    rng = np.random.default_rng(0)
    signs = rng.choice([-1.0, 1.0], size=(200000, max(n, 1)))
    pv = float((np.abs((signs * d).mean(axis=1)) >= abs(mean) - 1e-12).mean()) if n else None
    dz = float(mean / max(sd, 1e-9)) if n > 1 else None
    return {"n_seeds": int(n), "mean": mean, "sd": sd, "p_value": pv, "cohen_dz": dz}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        files = [os.path.join(ROOT, "results", "exp44_smoke.json")]
    else:
        files = [os.path.join(ROOT, "results", "exp44_s%d.json" % s) for s in range(8)
                 if os.path.exists(os.path.join(ROOT, "results", "exp44_s%d.json" % s))]
    runs = load(files)
    if not runs:
        print("no exp44 results found")
        return
    seeds = [r["seed"] for r in runs]
    prim = {arm: np.array([r["eval"][arm][PRIMARY]["hit"] / max(1, r["eval"][arm][PRIMARY]["n"])
                           for r in runs], dtype=np.float64) for arm in ARMS}
    print("primary (intrain200 deg0.3 lam=0 hit rate):")
    for arm in ARMS:
        v = prim[arm]
        print("  %-9s %s  mean=%.4f sd=%.4f" % (arm, np.round(v, 3), v.mean(), v.std(ddof=1) if v.size > 1 else 0))
    stats = {}
    for tag, x, y in (("B_minus_A", "B_online", "A_fixed"),
                      ("B_minus_C", "B_online", "C_uniform"),
                      ("A_minus_C", "A_fixed", "C_uniform")):
        d = prim[x] - prim[y]
        stats[tag] = pair_stats(d)
        s = stats[tag]
        print("  %-9s %+.4f +/- %.4f  sign-flip p=%s  dz=%s"
              % (tag, s["mean"], s["sd"],
                 "n/a(1 seed)" if s["p_value"] is None else "%.4f" % s["p_value"],
                 "n/a(1 seed)" if s["cohen_dz"] is None else "%.3f" % s["cohen_dz"]))
    sec = {}
    for key in ("intrain200|lam0|deg0.0", "intrain200|lam05|deg0.3", "dup200|lam0|deg0.0"):
        sec[key] = {arm: float(np.mean([r["eval"][arm][key]["hit"] / max(1, r["eval"][arm][key]["n"])
                                        for r in runs])) for arm in ARMS}
    cov = {arm: {"unique_replayed": float(np.mean([r["coverage"][arm]["unique_replayed_slots"] for r in runs])),
                 "target_replayed": float(np.mean([r["coverage"][arm]["eval_target_replayed"] for r in runs]))}
           for arm in ARMS}
    gate = ("POSITIVE: B>A on primary" if stats["B_minus_A"]["mean"] > 0 and stats["B_minus_A"]["p_value"] < 0.05
            else "NEGATIVE (reported verbatim): B<=A on primary; online tag criterion gives no detectable gain over fixed tag")
    out = {"exp": "exp44_analyze", "n_runs": len(runs), "seeds": seeds,
           "primary_definition": runs[0]["pre_registered"]["primary"],
           "primary_rate": {arm: [float(x) for x in prim[arm]] for arm in ARMS},
           "paired_stats": stats, "secondary_rate": sec, "coverage_mean": cov,
           "theta_trajectories": [r["theta_trajectory"] for r in runs],
           "gate_verdict": gate,
           "gate_pre_registered": runs[0]["pre_registered"]["gate"],
           "completed_at": runs[0]["completed_at"]}
    dest = os.path.join(ROOT, "results", "exp44_smoke_analyze.json" if a.smoke else "exp44_analyze.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print("gate:", gate)
    print("saved", dest)


if __name__ == "__main__":
    main()
