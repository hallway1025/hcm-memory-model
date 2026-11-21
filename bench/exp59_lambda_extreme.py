# -*- coding: utf-8 -*-
"""exp59 - lambda full-grid dose-response on the semantic-stress protocol
(session slice), extending exp49 to lam {0, 0.25, 0.5, 0.75, 1.0} x
deg {0.3, 0.5, 0.7, 0.85}.

exp49 (ledger row exp49, D-068, amendment 7728c64) measured paired
Delta(lam0.5 - lam1.0) at deg <= 0.7 and found exact zero - "semantic-channel
retrieval gain unconfirmed". The CLS complementarity hypothesis predicts any
fusion benefit GROWS with episodic corruption; exp49 never entered the
extreme-degradation regime. exp59 completes the dose-response: same protocol,
same seeds, same judge (obs_of[slot]==target, L-035 corrected), full lambda
grid, extended degradation axis. lam only alters query fusion (model.py:251);
the write path is lam-invariant - one library per seed.

PRIMARY (predeclared): paired per-probe Delta(hit(lam0.5) - hit(lam1.0)) at
deg0.85, intrain200, pooled over 3 seeds (n=600), sign-flip permutation
(200k, seed 15), p two-sided < 0.05 -> POSITIVE (crossover located);
otherwise NEGATIVE ("retrieval-side null extends across the full corruption
axis"). Both branches recorded honestly.
G1 replication anchor: lam1.0 deg0.3 intrain200 hit per seed equals the
exp49_s{seed} grid value within |Delta| <= 2/200 (protocol identity check
across machines; BLAS pinned to 1 thread).
G3 sanity: pooled hit(lam1.0, deg0.85) < pooled hit(lam1.0, deg0.3)
(degradation axis actually bites).
Smoke: --smoke -> train 600 / test 200 rows, lam {0, 1.0}, deg {0.3, 0.85},
pipeline check only.
"""
import argparse, json, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "bench"))
import exp29_text_frontend as e29          # noqa: E402
import exp49_semantic_stress as e49        # noqa: E402
from hcm import HCM, Config                # noqa: E402

LAMS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEGS = (0.0, 0.3, 0.5, 0.7, 0.85)
SMOKE_LAMS = (0.0, 1.0)
SMOKE_DEGS = (0.3, 0.85)


def inject_deg85(recs, qclean):
    e29.LEVEL_KEY.update({0.85: "del085"})
    for rec in recs:
        c = qclean[rec["qsrc"]]
        if "del085" not in rec:
            rec["del085"] = e29.corrupt_positions(
                c, 0.85, e29.SEEDS["deg_base"] + rec["pid"])
    return recs


def run_seed(seed, limit_train, limit_test, out_path, smoke):
    lams = SMOKE_LAMS if smoke else LAMS
    degs = SMOKE_DEGS if smoke else DEGS
    train_raw, train_clean, test_raw, test_clean = e49.load_corpus(
        "session", limit_train, limit_test)
    probes = e29.build_full_probes(train_raw, train_clean, test_raw, test_clean)
    intrain = e49.inject_deg_positions(probes["groups"]["intrain200"],
                                       train_clean, need=(0.7,))
    dup = e49.inject_deg_positions(probes["groups"]["dup200"],
                                   test_clean, need=(0.5, 0.7))
    intrain = inject_deg85(intrain, train_clean)
    dup = inject_deg85(dup, test_clean)

    arm, info = e29.make_arm("NEW256", train_raw, train_clean)
    t0 = time.perf_counter()
    arm.fit(train_raw)
    fit_dt = time.perf_counter() - t0
    X = arm.encode_inputs(train_raw, train_clean)
    m, obs_of, slot_of_obs, wdt, wstats = e49.write_library(
        arm, X, len(train_raw), seed)

    grid = {}
    for lam in lams:
        m.cfg.lam = lam
        for gname, qtexts, qclean, recs in (
                ("intrain200", train_raw, train_clean, intrain),
                ("dup200", test_raw, test_clean, dup)):
            for deg in degs:
                r, per_probe = e29.evaluate_group(m, arm, qtexts, qclean, recs,
                                                  deg, obs_of, slot_of_obs, X)
                grid.setdefault(gname, {}).setdefault("%.1f" % deg, {})[
                    "lam%.2f" % lam] = {
                    "n": r["n"], "hit": r["hit"], "content_hit": r["content_hit"],
                    "abstain": r["abstain"], "actions": r["actions"],
                    "hits_by_pid": [int(p["hit"]) for p in per_probe]}

    out = {"exp": "exp59", "seed": seed, "scale": "session", "arm": "NEW256",
           "pre_registered": "ledger exp59 row (this commit)",
           "judge": "obs_of[slot]==target (L-035 corrected)",
           "n_train": len(train_raw), "n_test": len(test_raw),
           "n_written": int(m.E.n), "write_stats": wstats,
           "write_time_s": round(wdt, 1), "fit_time_s": round(fit_dt, 1),
           "lams": list(lams), "degs": list(degs), "grid": grid,
           "smoke": bool(smoke),
           "finished_at": e29.now_iso()}
    e29.save_data(out_path, out)
    summary = {gn: {d: {k: v["hit"] for k, v in dgs.items()}
                    for d, dgs in gd.items()} for gn, gd in grid.items()}
    print("exp59 seed %d done: written=%d write=%.0fs hits=%s"
          % (seed, out["n_written"], wdt, json.dumps(summary)), flush=True)
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="exp59 lambda x extreme-deg grid")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny slice: train 1200 / test 400 (exp49 smoke slice; 600/200 starves build_full_probes rand200)")
    args = ap.parse_args()
    lt, lte = (1200, 400) if args.smoke else (None, None)
    out = args.out or os.path.join(
        ROOT, "results", "exp59_smoke_s%d.json" % args.seed if args.smoke
        else "exp59_s%d.json" % args.seed)
    t0 = time.perf_counter()
    run_seed(args.seed, lt, lte, out, args.smoke)
    print("EXP59_OK seed=%d elapsed=%.0fs out=%s"
          % (args.seed, time.perf_counter() - t0, out), flush=True)


if __name__ == "__main__":
    main()
