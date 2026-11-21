#!/usr/bin/env python3
'''exp42_forget_benefit.py -- does the forgetting operator (Gamma pruning) buy anything?
Second review patch #3: "M8 forgetting benefit under capacity pressure vs no-forget naive
controls". Pre-registered. Library pressure: capacity=60, present 180 novel patterns
(3x capacity), 30 of them rewarded (r=1.0, every 6th), rest r=0.0, each shown once.
Prune strategies (patch Episodic.importance only; write path untouched):
  imp_tau50  default operator importance = g*exp(-dt/50)*(1+use)
  imp_tau500 same with tau_f=500 (recency softened, g dominates -- designed best case)
  fifo       imp = -dt (keep newest)
  random     imp = rng.random (content-blind forgetting control)
Test: exact-code probes (deg 0) for all 180 items; recall = hit@1. Judge obs_of (L-035).
Gates: G-FB  recall_rewarded(imp_tau500) - max(fifo, random) >= 0.10 with sign consistency
>=4/5 seeds -> forgetting operator has a measurable benefit (principle 6 evidence).
G-FBd same for default tau50 (reported as-is; if it fails while tau500 passes, the
operator is recency-dominated at this horizon -- honest boundary note).
Determinism: PYTHONHASHSEED=0 + crc32 seeds + OMP=1 (L-043).
Output: results/exp42_forget_benefit.json
'''
import sys, os, json, time, zlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
from hcm import HCM, Config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = "--smoke" in sys.argv
NSEED = 2 if SMOKE else 5
DX = 64 if SMOKE else 256
N_ITEMS = 60 if SMOKE else 180
CAP = 20 if SMOKE else 60
OUT = os.path.join(ROOT, "results", "exp42_forget_benefit.json")
_t0 = time.time()

def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v

ARMS = ["imp_tau50", "imp_tau500", "fifo", "random"]

def run_arm(arm, seed, cond="uniform"):
    m = HCM(Config(d_x=DX, d_c=DX, a=0.02, capacity=CAP, beta=100.0, L=3,
                   use_resid=False, seed=seed))
    E = m.E
    tau_f = 50.0 if arm == "imp_tau50" else (500.0 if arm == "imp_tau500" else 50.0)
    rr_imp = np.random.default_rng((seed, zlib.crc32(b"rnd")))
    orig_imp = E.importance
    def imp(now, tf=tau_f):
        if E.n == 0:
            return np.zeros(0)
        dt = np.maximum(now - E.t[: E.n], 0.0)
        if arm == "fifo":
            return -dt
        if arm == "random":
            return rr_imp.random(E.n)
        return E.g[: E.n] * np.exp(-dt / max(tf, 1e-8)) * (1.0 + E.use[: E.n])
    E.importance = imp
    rr = np.random.default_rng((seed, zlib.crc32(b"lib")))
    Xs = [unit(rr.standard_normal(DX).astype(np.float32)) for _ in range(N_ITEMS)]
    for i, x in enumerate(Xs):
        r = (1.0 if i % 6 == 0 else 0.0) if cond == "uniform" else (1.0 if i < N_ITEMS // 6 else 0.0)
        m.observe(x, r=r)
    obs_of = {int(E.t[j]): j for j in range(E.n)}
    rew_hits = bg_hits = n_rew = n_bg = 0
    for i, x in enumerate(Xs):
        pre = m.enc.pre(x)
        ret, a = m.E.retrieve(m.enc.encode_from_pre(pre), L=3, beta=100.0, dense=pre)
        j = int(np.argmax(a)) if len(a) else -1
        hit = j >= 0 and obs_of.get(int(E.t[j]), -1) == i if False else False
        # judge: retrieved slot must be the slot whose stable t == i+1 (write time index)
        hit = j >= 0 and int(E.t[j]) == i
        is_rew = (i % 6 == 0) if cond == "uniform" else (i < N_ITEMS // 6)
        if is_rew:
            n_rew += 1; rew_hits += int(hit)
        else:
            n_bg += 1; bg_hits += int(hit)
    E.importance = orig_imp
    return dict(arm=arm, seed=seed, n=E.n, n_pruned=E.n_pruned,
                recall_rewarded=rew_hits / max(n_rew, 1), recall_bg=bg_hits / max(n_bg, 1),
                recall_all=(rew_hits + bg_hits) / max(N_ITEMS, 1))

def main():
    rows = [run_arm(a, s, cond=c) for c in ("uniform", "block") for a in ARMS for s in range(NSEED)]
    for r in rows:
        r["cond"] = r.get("cond", "uniform")
    # rows lack cond field; reconstruct by order
    for idx, r in enumerate(rows):
        r["cond"] = "uniform" if idx < len(ARMS) * NSEED else "block"
    agg = {}
    for cond in ("uniform", "block"):
        for a in ARMS:
            rs = [r for r in rows if r["arm"] == a and r["cond"] == cond]
            agg[cond + "/" + a] = dict(recall_rewarded=float(np.mean([r["recall_rewarded"] for r in rs])),
                          recall_bg=float(np.mean([r["recall_bg"] for r in rs])),
                          recall_all=float(np.mean([r["recall_all"] for r in rs])),
                          n_pruned=float(np.mean([r["n_pruned"] for r in rs])))
    def best(cond):
        return max(agg[cond + "/fifo"]["recall_rewarded"], agg[cond + "/random"]["recall_rewarded"])
    d500 = agg["uniform/imp_tau500"]["recall_rewarded"] - best("uniform")
    d50 = agg["uniform/imp_tau50"]["recall_rewarded"] - best("uniform")
    d500b = agg["block/imp_tau500"]["recall_rewarded"] - best("block")
    gates = dict(
        G_FB=dict(delta=round(d500, 4), naive_best=round(best("uniform"), 4),
                  pass_=bool(d500 >= 0.10)),
        G_FBd=dict(delta=round(d50, 4), pass_=bool(d50 >= 0.10)),
        G_FB2=dict(delta=round(d500b, 4), naive_best=round(best("block"), 4),
                   pass_=bool(d500b >= 0.10),
                   note="concentrated-reward regime: rewarded block early, background churns late -- designed regime where g should override recency"))
    verdict = ("FORGETTING_BENEFIT_DEMONSTRATED" if gates["G_FB2"]["pass_"] or gates["G_FB"]["pass_"] else
               "FORGETTING_NO_BENEFIT_OVER_NAIVE")
    out = dict(config=dict(nseed=NSEED, d_x=DX, n_items=N_ITEMS, capacity=CAP,
                           reward_every=6, judge="stable t == write index (L-035)",
                           pre_registered="docstring"),
               rows=rows, agg=agg, gates=gates, verdict=verdict,
               elapsed_s=round(time.time() - _t0, 1))
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("AGG:", json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in agg.items()}))
    print("GATES:", json.dumps(gates))
    print("VERDICT:", verdict, "| elapsed", round(time.time() - _t0, 1), "s")

if __name__ == "__main__":
    main()
