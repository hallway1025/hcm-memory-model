# -*- coding: utf-8 -*-
"""exp49 analyze - preregistered gate on pooled per-probe paired differences.

PRIMARY: deg0.7, intrain200, Delta = hit(lam0.5) - hit(lam1.0), pooled over seeds
(n = 200 x n_seeds), sign-flip permutation (200k), p two-sided; dz = mean/sd.
Gate (preregistered): mean > 0 AND p < 0.05 -> POSITIVE (complementarity evidence);
otherwise -> NEGATIVE (semantic-channel retrieval gain unconfirmed incl. damaged
regime) - recorded honestly, Claim 2 wording unchanged.
Secondary (descriptive): mean Delta per deg per group; per-seed deg0.7 Delta.
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load(seed):
    p = os.path.join(ROOT, "results", "exp49_s%d.json" % seed)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def signflip_p(diff, n_perm=200000, seed=15):
    d = np.asarray(diff, dtype=np.float64)
    if len(d) == 0:
        return float("nan"), 0.0
    obs = float(np.mean(d))
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n_perm, len(d)))
    perm = (signs * d).mean(axis=1)
    p = float((np.abs(perm) >= abs(obs) - 1e-12).mean())
    return p, obs

def main():
    seeds = [s for s in (0, 1, 2) if load(s) is not None]
    if not seeds:
        print("NO_DATA"); sys.exit(2)
    rows = []
    for deg in ("0.3", "0.5", "0.7"):
        for grp in ("intrain200", "dup200"):
            pool = []
            per_seed = []
            for s in seeds:
                j = load(s)
                d05 = np.asarray(j["grid"][grp][deg]["lam0.5"]["hits_by_pid"], dtype=np.int8)
                d10 = np.asarray(j["grid"][grp][deg]["lam1.0"]["hits_by_pid"], dtype=np.int8)
                n = min(len(d05), len(d10))
                d = (d05[:n].astype(np.int8) - d10[:n].astype(np.int8)).astype(np.int8)
                pool.append(d)
                per_seed.append({"seed": s, "n": n,
                                 "hit05": int(d05[:n].sum()), "hit10": int(d10[:n].sum()),
                                 "delta": int(d.sum())})
            pooled = np.concatenate(pool)
            p, obs = signflip_p(pooled)
            sd = float(np.std(pooled, ddof=1)) if len(pooled) > 1 else 0.0
            dz = obs / sd if sd > 0 else 0.0
            rows.append({"deg": deg, "group": grp, "n_pooled": int(len(pooled)),
                         "mean_delta": round(obs, 5), "sd": round(sd, 5),
                         "dz": round(dz, 4), "p_signflip": round(p, 5),
                         "per_seed": per_seed})
    primary = [r for r in rows if r["deg"] == "0.7" and r["group"] == "intrain200"][0]
    verdict = "POSITIVE" if (primary["mean_delta"] > 0 and primary["p_signflip"] < 0.05) else "NEGATIVE"
    out = {"exp": "exp49", "seeds": seeds, "primary": primary, "rows": rows,
           "gate": "mean>0 AND p<0.05 -> POSITIVE else NEGATIVE (prereg D-068)",
           "verdict": verdict,
           "note_neg": ("semantic-channel retrieval gain unconfirmed (incl. damaged "
                        "regime); Claim 2 wording unchanged") if verdict == "NEGATIVE" else "",
           "finished_at": __import__("datetime").datetime.now().isoformat()}
    with open(os.path.join(ROOT, "results", "exp49_analyze.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for r in rows:
        print("%s deg%s: n=%d mean_d=%.3f dz=%.3f p=%.4f" %
              (r["group"], r["deg"], r["n_pooled"], r["mean_delta"], r["dz"], r["p_signflip"]))
    print("PRIMARY deg0.7 intrain200: mean_d=%.3f p=%.4f -> %s" %
          (primary["mean_delta"], primary["p_signflip"], verdict))
    print("WROTE results/exp49_analyze.json")

if __name__ == "__main__":
    main()
