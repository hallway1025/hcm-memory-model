#!/usr/bin/env python3
'''exp41_m7_gate_retest.py -- M7 write-gate retest (second review patch #2).
Pre-registered. Part A (functional redundancy vs theta_dup): one pattern x0 shown
10x verbatim, 2x2 arms {gate, const} x {theta_dup 0.95, dup-disabled}. Write counts.
Mechanism hypothesis: either repetition filter alone suffices (gate+no-dup = 1,
const+no-dup = 10); const+no-dup is the only cell that fails.
Part B (reward selectivity, drift-immune metric): prototype p stored once (r=0),
then probes x = unit(alpha*p + sqrt(1-alpha^2)*e) at alpha in {0.0,0.2,0.4,0.6,0.8},
8 per level, half r=1.0 half r=0.0 alternating, each shown once, interleaved within
level. Metric = observed write rate per reward group (behavioral, no stored-code
comparison => immune to code drift, the ablation2 metric failure). Arms gate/const x
resid True/False x 5 seeds. Functional selectivity FS = [P(w|r=1)-P(w|r=0)]_gate -
[same]_const, pooled over marginal levels alpha in {0.2,0.4,0.6}.
Gates: G-SEL  FS >= +0.15 with sign consistency >= 4/5 seeds -> gate has a unique
reward-selectivity function (keep INCONCLUSIVE + note). G-RED  |FS| <= 0.15 ->
accept REDUNDANT-WITH-THETA_DUP verdict (update MODEL.md M7 + paper SS4.8).
Determinism: PYTHONHASHSEED=0 + crc32 seeds + OMP=1 (L-043).
Output: results/exp41_m7_gate.json
'''
import sys, os, json, time, zlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
from hcm import HCM, Config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = "--smoke" in sys.argv
NSEED = 2 if SMOKE else 5
DX = 64 if SMOKE else 256
OUT = os.path.join(ROOT, "results", "exp41_m7_gate.json")
_t0 = time.time()

def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v

def cfg_of(gate_const, resid, seed):
    c = Config(d_x=DX, d_c=DX, a=0.02, capacity=100, beta=100.0, L=3,
               use_resid=resid, seed=seed, gate_constant=gate_const)
    return c

def set_dup(m, theta_dup):
    if theta_dup is None:  # disabled
        orig = m.E.write
        def wrap(h, t, g, r=0.0, nu=0.0, **kw):
            return orig(h, t, g, r, nu, theta_dup=2.0, **kw)
        m.E.write = wrap
    # theta_dup given -> observe() default 0.95 already, nothing to do

# ---------------- Part A ----------------
def part_a(seed):
    rows = []
    for gate_const in (False, True):
        for dup in (0.95, None):
            m = HCM(cfg_of(gate_const, False, seed))
            set_dup(m, dup)
            rr = np.random.default_rng((seed, zlib.crc32(b"x0")))
            x0 = unit(rr.standard_normal(DX).astype(np.float32))
            w = 0
            gs = []
            for _ in range(10):
                rec = m.observe(x0, r=0.0)
                w += int(rec["wrote"])
                gs.append(round(float(rec["g"]), 4))
            rows.append(dict(arm=("gate" if not gate_const else "const"),
                             dup=("0.95" if dup is not None else "off"),
                             writes=w, g_hist=gs))
    return rows

# ---------------- Part B ----------------
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8]

def part_b(seed):
    out = {}
    for resid in (True, False):
        per = {}
        for gate_const in (False, True):
            m = HCM(cfg_of(gate_const, resid, seed))
            rr = np.random.default_rng((seed, zlib.crc32(b"probe")))
            p = unit(rr.standard_normal(DX).astype(np.float32))
            m.observe(p, r=0.0)  # prototype stored, unrewarded
            level = {}
            for al in ALPHAS:
                evs = []
                for i in range(8):
                    e = unit(rr.standard_normal(DX).astype(np.float32))
                    x = unit(al * p + np.sqrt(max(0.0, 1 - al * al)) * e).astype(np.float32)
                    r = 1.0 if i % 2 == 0 else 0.0   # interleaved, matched
                    rec = m.observe(x, r=r)
                    evs.append(dict(r=r, wrote=bool(rec["wrote"]), g=float(rec["g"]), nu=float(rec["nu"])))
                level[al] = evs
            per["gate" if not gate_const else "const"] = level
        out["resid" if resid else "noresid"] = per
    return out

def summarize(pb):
    # FS over marginal levels, pooled seeds outside
    res = {}
    for rk, per in pb.items():
        for arm, level in per.items():
            for al, evs in level.items():
                rw = [e for e in evs if e["r"] == 1.0]
                bg = [e for e in evs if e["r"] == 0.0]
                key = (rk, arm, al)
                res[key] = dict(p_rw=float(np.mean([e["wrote"] for e in rw])),
                                p_bg=float(np.mean([e["wrote"] for e in bg])),
                                g_rw=float(np.mean([e["g"] for e in rw])),
                                g_bg=float(np.mean([e["g"] for e in bg])),
                                nu=float(np.mean([e["nu"] for e in evs])))
    return res

def main():
    A = [part_a(s) for s in range(NSEED)]
    B = [part_b(s) for s in range(NSEED)]
    # aggregate B over seeds
    marg = [0.2, 0.4, 0.6]
    fs_per_seed = []
    for pb in B:
        s = summarize(pb)
        def pooled(rk, arm):
            vals = [s[(rk, arm, al)] for al in marg]
            return (float(np.mean([v["p_rw"] for v in vals])), float(np.mean([v["p_bg"] for v in vals])))
        g_rw, g_bg = pooled("noresid", "gate")
        c_rw, c_bg = pooled("noresid", "const")
        fs_per_seed.append((g_rw - g_bg) - (c_rw - c_bg))
    gates = {}
    fs_mean = float(np.mean(fs_per_seed))
    sign_ok = int(np.mean([f > 0 for f in fs_per_seed]) >= 0.8) if fs_mean > 0 else int(np.mean([f < 0 for f in fs_per_seed]) >= 0.8)
    if fs_mean >= 0.15 and sign_ok:
        gates["G_SEL"] = dict(fs_mean=fs_mean, fs_per_seed=[round(f,4) for f in fs_per_seed], pass_=True)
        gates["G_RED"] = dict(fs_mean=fs_mean, pass_=False)
    else:
        gates["G_SEL"] = dict(fs_mean=fs_mean, fs_per_seed=[round(f,4) for f in fs_per_seed], pass_=False)
        gates["G_RED"] = dict(fs_mean=fs_mean, sign_consistent=sign_ok, pass_=abs(fs_mean) <= 0.15)
    # verdict
    verdict = "GATE_HAS_UNIQUE_REWARD_FUNCTION" if gates["G_SEL"]["pass_"] else (
        "REDUNDANT_WITH_THETA_DUP" if gates["G_RED"]["pass_"] else "INCONCLUSIVE")
    out = dict(config=dict(nseed=NSEED, d_x=DX, alphas=ALPHAS, marginal=marg,
                           theta_write=0.5, pre_registered="docstring"),
               part_a=A, part_b_summary={str(k): v for k, v in summarize(B[0]).items()},
               fs_per_seed=[round(f, 4) for f in fs_per_seed], gates=gates, verdict=verdict,
               elapsed_s=round(time.time() - _t0, 1))
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("PART A (seed0):", json.dumps(A[0], ensure_ascii=False))
    print("FS per seed:", [round(f, 4) for f in fs_per_seed], "mean", round(fs_mean, 4))
    print("GATES:", json.dumps(gates, ensure_ascii=False))
    print("VERDICT:", verdict, "| elapsed", round(time.time() - _t0, 1), "s")

if __name__ == "__main__":
    main()
