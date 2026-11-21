"""exp59 analyzer: lambda readout dose-response, identity task (session slice).
Prereg f5d406f. Schema (bench/exp59_lambda_extreme.py): grid[group]["%.1f" % deg][lam] =
{n, hit, content_hit, abstain, actions, hits_by_pid}; deg 0.85 -> key "0.8".
PRIMARY (prereg): paired per-probe delta lam0.50 - lam1.00 @ intrain200/deg0.8, pooled 3 seeds n=600, sign-flip 200k.
G1 anchor: exp49 lam1.00 deg0.3 s0 hit count = 138 within +-2/200.
G3: pooled mean hits deg0.8 < deg0.3 at lam1.00.
Sanity G0: lam0.00 cells all zero (semantic path alone carries no identity).
Usage: py -X utf8 bench/exp59_analyze.py
"""
import json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

def load(seed):
    p = RES / f"exp59_s{seed}.json"
    if not p.exists():
        sys.exit(f"MISSING {p}")
    return json.loads(p.read_text(encoding="utf-8"))

def signflip(a, b, iters=200000, seed=0):
    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(a, b)]
    obs = sum(diffs) / len(diffs)
    n = len(diffs)
    cnt = 0
    for _ in range(iters):
        s = 0.0
        for d in diffs:
            s += d if rng.random() < 0.5 else -d
        if abs(s / n) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (iters + 1)

def main():
    js = [load(s) for s in (0, 1, 2)]

    def cell(g, d, lam):
        return js[0]["grid"][g][d][lam]  # per-seed access via js[s]

    lam_keys = ["lam0.00", "lam0.25", "lam0.50", "lam0.75", "lam1.00"]
    deg_keys = ["0.3", "0.5", "0.7", "0.8"]

    # G0 sanity: lam0 all zero (all groups, all degs, all seeds)
    g0 = True
    for s in range(3):
        for g in ("intrain200", "dup200"):
            for d in deg_keys:
                if js[s]["grid"][g][d]["lam0.00"]["hit"] != 0:
                    g0 = False
    # G1 anchor vs exp49 (seed 0, lam1.00, deg0.3)
    anchor = js[0]["grid"]["intrain200"]["0.3"]["lam1.00"]["hit"]
    g1 = abs(anchor - 138) <= 2
    # PRIMARY: per-probe pooled sign-flip, intrain200 deg0.8, lam0.50 vs lam1.00
    v05 = [h for s in range(3) for h in js[s]["grid"]["intrain200"]["0.8"]["lam0.50"]["hits_by_pid"]]
    v10 = [h for s in range(3) for h in js[s]["grid"]["intrain200"]["0.8"]["lam1.00"]["hits_by_pid"]]
    if len(v05) != 600 or len(v10) != 600:
        print(f"WARN: pooled n={len(v05)}/{len(v10)} != 600; check grid keys")
    obs, pval = signflip(v05, v10)
    primary = {"delta": round(obs, 6), "p": pval, "n": len(v05),
               "hits_lam050": sum(v05), "hits_lam100": sum(v10)}
    # G3: pooled deg0.8 < deg0.3 at lam1.00
    a85 = sum(js[s]["grid"]["intrain200"]["0.8"]["lam1.00"]["hit"] for s in range(3)) / 3
    a30 = sum(js[s]["grid"]["intrain200"]["0.3"]["lam1.00"]["hit"] for s in range(3)) / 3
    g3 = a85 < a30

    gates = {"G0_lam0_zero": g0, "G1_anchor_exp49": g1, "G3_deg085_below_deg030": g3}
    out = {
        "gates": gates,
        "primary": primary,
        "anchor_hits_seed0": anchor,
        "grid_means_intrain200": {lam: {d: round(sum(js[s]["grid"]["intrain200"][d][lam]["hit"] for s in range(3)) / 3, 1) for d in deg_keys} for lam in lam_keys},
        "grid_means_dup200": {lam: {d: round(sum(js[s]["grid"]["dup200"][d][lam]["hit"] for s in range(3)) / 3, 1) for d in deg_keys} for lam in lam_keys},
        "n_written": [js[s]["n_written"] for s in range(3)],
    }
    verdict = "GATE_PASS" if all(gates.values()) and primary["p"] < 0.05 else "GATE_FAIL"
    out["verdict"] = verdict
    (RES / "exp59_analyze.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    print("EXP59", verdict)

if __name__ == "__main__":
    main()
