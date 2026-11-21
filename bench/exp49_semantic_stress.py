# -*- coding: utf-8 -*-
"""exp49 - semantic-prior complementarity stress test (preregistered: ledger row exp49, D-068).

Hypothesis: CLS complementarity predicts the semantic prior's fusion benefit grows
with episodic corruption. exp29-B1 tested only deg0.3 (null: dz=0.002, p=0.648).

Grid: lam {0, 0.5, 1.0} x deg {0.3, 0.5, 0.7} x groups {intrain200, dup200};
3 independent NEW256 libraries (HCM seed 0/1/2).
Scale: --scale session = peer-02 slice (exp29 v2_m2final/B1 comparable, primary);
       --scale full    = 6-session 92,760 (exp44 slice; pre-registered secondary, no gate).

PRIMARY (preregistered): paired per-probe Delta(lam0.5 - lam1.0) at deg0.7,
intrain200, corrected judge (obs_of[slot]==target), pooled across 3 seeds (n=600),
sign-flip permutation p<0.05. Gate: positive -> complementarity evidence;
negative/not-significant -> "semantic-channel retrieval gain unconfirmed (incl.
damaged regime)" recorded honestly. Claim 2 wording unchanged in both branches.

lam/deg only alter query fusion (model.py:251), never the write path. Deg positions
0.3/0.5 precomputed in probes (del03/del05); 0.7 generated here with the same seed
convention (41000+pid) via e29.corrupt_positions (prereg amendment 7728c64).
"""
import argparse, json, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "bench"))
import exp29_text_frontend as e29          # noqa: E402
from hcm import HCM, Config                # noqa: E402

LAMS = (0.0, 0.5, 1.0)
DEGS = (0.3, 0.5, 0.7)
LEVELS_FULL = (0.0, 0.3, 0.5, 0.7)


def load_corpus(scale, limit_train=None, limit_test=None):
    tf = [e29.TRAIN_FILES[0]] if scale == "session" else list(e29.TRAIN_FILES)
    tt = [e29.TEST_FILES[0]] if scale == "session" else list(e29.TEST_FILES)
    train_raw = []
    for fn in tf:
        train_raw.extend(e29.read_texts(fn, limit=limit_train))
    test_raw = []
    for fn in tt:
        test_raw.extend(e29.read_texts(fn, limit=limit_test))
    train_clean = [e29.clean_text(t) for t in train_raw]
    test_clean = [e29.clean_text(t) for t in test_raw]
    return train_raw, train_clean, test_raw, test_clean


def write_library(arm, X, n, seed):
    """e29.write_all with an HCM seed parameter (independent libraries per seed)."""
    m = HCM(Config(d_x=arm.dim, d_c=arm.dim, a=0.02, capacity=n + 10, beta=e29.BETA,
                   L=e29.L, use_resid=False, lam=1.0, seed=seed))
    wrote = gate_rej = dup_rej = 0
    t0 = time.perf_counter()
    for i in range(n):
        rec = m.observe(X[i])
        if rec["wrote"]:
            wrote += 1
        elif rec["g"] >= m.cfg.theta_write:
            dup_rej += 1
        else:
            gate_rej += 1
    dt = time.perf_counter() - t0
    obs_of = {s: int(m.E.t[s]) for s in range(int(m.E.n))}
    slot_of_obs = {v: s for s, v in obs_of.items()}
    stats = {"wrote": wrote, "gate_rejected": gate_rej, "dup_or_full_rejected": dup_rej}
    return m, obs_of, slot_of_obs, dt, stats


def inject_deg_positions(recs, qclean, need=(0.5, 0.7)):
    """Add del05/del07 position lists with the preregistered seed convention 41000+pid."""
    keys = {0.5: "del05", 0.7: "del07"}
    e29.LEVEL_KEY.update(keys)
    for rec in recs:
        c = qclean[rec["qsrc"]]
        for frac in need:
            k = keys[frac]
            if k not in rec:
                rec[k] = e29.corrupt_positions(c, frac, e29.SEEDS["deg_base"] + rec["pid"])
    return recs


def run_seed(seed, scale, limit_train, limit_test, out_path):
    train_raw, train_clean, test_raw, test_clean = load_corpus(scale, limit_train, limit_test)
    probes = e29.build_full_probes(train_raw, train_clean, test_raw, test_clean)
    intrain = inject_deg_positions(probes["groups"]["intrain200"], train_clean, need=(0.7,))
    dup = inject_deg_positions(probes["groups"]["dup200"], test_clean, need=(0.5, 0.7))

    arm, info = e29.make_arm("NEW256", train_raw, train_clean)
    t0 = time.perf_counter()
    arm.fit(train_raw)
    fit_dt = time.perf_counter() - t0
    X = arm.encode_inputs(train_raw, train_clean)
    m, obs_of, slot_of_obs, wdt, wstats = write_library(arm, X, len(train_raw), seed)

    grid = {}
    for lam in LAMS:
        m.cfg.lam = lam
        for gname, qtexts, qclean, recs in (("intrain200", train_raw, train_clean, intrain),
                                            ("dup200", test_raw, test_clean, dup)):
            for deg in LEVELS_FULL:
                r, per_probe = e29.evaluate_group(m, arm, qtexts, qclean, recs, deg,
                                                  obs_of, slot_of_obs, X)
                grid.setdefault(gname, {}).setdefault("%.1f" % deg, {})["lam%.1f" % lam] = {
                    "n": r["n"], "hit": r["hit"], "content_hit": r["content_hit"],
                    "abstain": r["abstain"], "actions": r["actions"],
                    "hits_by_pid": [int(p["hit"]) for p in per_probe]}

    out = {"exp": "exp49", "seed": seed, "scale": scale, "arm": "NEW256",
           "pre_registered": "ledger exp49 row + D-068 (amendment 7728c64)",
           "judge": "obs_of[slot]==target (L-035 corrected)",
           "n_train": len(train_raw), "n_test": len(test_raw),
           "n_written": int(m.E.n), "write_stats": wstats,
           "write_time_s": round(wdt, 1), "fit_time_s": round(fit_dt, 1),
           "lams": list(LAMS), "degs": list(LEVELS_FULL), "grid": grid,
           "finished_at": e29.now_iso()}
    e29.save_data(out_path, out)
    summary = {gn: {d: {k: v["hit"] for k, v in dgs.items()}
                    for d, dgs in gd.items()} for gn, gd in grid.items()}
    print("exp49 seed %d done: written=%d write=%.0fs hits=%s"
          % (seed, out["n_written"], wdt, json.dumps(summary)), flush=True)
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="exp49 semantic-prior complementarity stress")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--scale", choices=["session", "full"], default="session")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny slice: train 1200 / test 400 rows, pipeline check only")
    args = ap.parse_args()
    lt, lte = (1200, 400) if args.smoke else (None, None)
    out = args.out or os.path.join(ROOT, "results", "exp49_s%d.json" % args.seed)
    t0 = time.perf_counter()
    run_seed(args.seed, args.scale, lt, lte, out)
    print("EXP49_OK seed=%d elapsed=%.0fs out=%s" % (args.seed, time.perf_counter() - t0, out),
          flush=True)


if __name__ == "__main__":
    main()
