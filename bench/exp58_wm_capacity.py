# -*- coding: utf-8 -*-
"""exp58 - M6 working-memory operator capacity curve (standalone, preregistered).

K6 (M6_WM) is the thinnest operator in the nine-operator dossier: the in-model
ablation attributes ~3.7% at C=4, but the operator's own capacity semantics
(wm.py docstring P5: accuracy steps down when load exceeds C) has never been
measured directly. exp58 measures the operator in isolation - no HCM, no
corpus - so the capacity curve is attributable to WM alone. Operator-level
characterization complementing the in-model ablation; labeled as such.

Codes follow the L0 encoder geometry: h = unit(relu_kwta(g, k)) with
d_h=1024 (rho=8 x d_x=128), k=20 (a=0.02), continuous nonneg top-k values
unit-normalized (encoder.py encode_from_pre). Distinct-code cosines ~0.02,
so theta_same=0.90 dedupe never fires (verified empirically per cell) and
the probe found-threshold 0.5 sits between cue-self cosine (~0.75 at 25%
support loss) and cross-item cosine (~0.02).

Protocol per C in {2,4,8,16} x seed in {0,1,2}: fresh WorkingMemory; present
L=2C+2 items sequentially; probe each item with cue_i = item with 25% of
support dropped (per-item seed 41000+i). Hit = module criterion (max sim >
0.5) AND argmax slot == slot holding item i (slot tracked by state diff
around present()). Because full-capacity eviction is argmin-weight (FIFO-
like under near-zero cross-cosine drift), the step is expected in RECENCY
space: items within the last C presentations survive, earlier ones fall.

Gates (predeclared, pooled over seeds within each C):
  G0 sanity: dedupe_absorbed == 0 in every cell.
  G1 fidelity: hit rate at recency < C is >= 0.90 (per C, all four).
  G2 ladder (P5): hit rate at recency >= C is <= (recency < C rate) - 0.25
     (per C, all four).
  G3 dose boundary: hit(recency C-1) - hit(recency C) >= 0.25 for >= 3 of
     4 C values.
Verdict: G1 & G2 & G3 all PASS -> OPERATOR_CAPACITY_CONFIRMED; any FAIL
recorded honestly with the failing cell.
Smoke: EXP58_SMOKE=1 -> d_h=256, k=5, C in {2,4}, seeds {0}.
"""
import argparse, json, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hcm.wm import WorkingMemory  # noqa: E402
from hcm.ops import relu_kwta, unit  # noqa: E402

SMOKE = os.environ.get("EXP58_SMOKE") == "1"
D_H = 256 if SMOKE else 1024
K = 5 if SMOKE else 20
CS = (2, 4) if SMOKE else (2, 4, 8, 16)
SEEDS = (0,) if SMOKE else (0, 1, 2)
CUE_FRAC = 0.25
CUE_SEED_BASE = 41000


def make_code(rng):
    return unit(relu_kwta(rng.standard_normal(D_H).astype(np.float32), K))


def make_cue(h, support, i):
    r = np.random.default_rng(CUE_SEED_BASE + i)
    drop = r.choice(support, size=max(1, int(round(len(support) * CUE_FRAC))),
                    replace=False)
    c = h.copy()
    c[drop] = 0.0
    return unit(c)


def run_cell(C, seed):
    wm = WorkingMemory(D_H, C=C)
    rng = np.random.default_rng(3000 + seed * 100 + C)
    L = 2 * C + 2
    items, slot_of = [], []
    for i in range(L):
        h = make_code(rng)
        support = np.where(h > 0)[0]
        before = wm.slots.copy()
        wm.present(h)
        diff = np.where((before != wm.slots).any(axis=1))[0]
        slot_of.append(int(diff[0]) if len(diff) else -1)
        items.append((h, support))
    hits_by_pos, sims_by_pos, agree = [], [], 0
    for i, (h, support) in enumerate(items):
        cue = make_cue(h, support, i)
        occ = np.where(wm.occ)[0]
        if occ.size == 0:
            hits_by_pos.append(0); sims_by_pos.append(0.0); continue
        sims = wm.slots[occ] @ cue
        mi = int(np.argmax(sims))
        m = float(sims[mi])
        found_mod, _ = wm.probe(cue)
        agree += int(bool(m > 0.5) == bool(found_mod))
        hit = bool(m > 0.5 and occ[mi] == slot_of[i])
        hits_by_pos.append(1 if hit else 0)
        sims_by_pos.append(round(m, 4))
    hits_by_rec = list(reversed(hits_by_pos))
    return {"C": C, "seed": seed, "L": L, "slot_of": slot_of,
            "dedupe_absorbed": int(sum(1 for s in slot_of if s < 0)),
            "hits_by_pos": hits_by_pos, "hits_by_recency": hits_by_rec,
            "sims_by_pos": sims_by_pos, "probe_agreement": agree}


def rate(cells, sel):
    n = h = 0
    for c in cells:
        for r, v in enumerate(c["hits_by_recency"]):
            if sel(r, c["C"]):
                n += 1
                h += int(v)
    return (h / n) if n else float("nan"), n


def main():
    t0 = time.perf_counter()
    cells = []
    for C in CS:
        for seed in SEEDS:
            cells.append(run_cell(C, seed))
            print("exp58 C=%d seed=%d dedupe=%d rec_hits=%s"
                  % (C, seed, cells[-1]["dedupe_absorbed"],
                     cells[-1]["hits_by_recency"]), flush=True)
    gates = {}
    gates["G0_dedupe_silent"] = all(c["dedupe_absorbed"] == 0 for c in cells)
    g1, g2, g3 = [], [], []
    per_c = {}
    for C in CS:
        cc = [c for c in cells if c["C"] == C]
        r_in, n_in = rate(cc, lambda r, C_: r < C_)
        r_out, n_out = rate(cc, lambda r, C_: r >= C_)
        step_last_in = sum(c["hits_by_recency"][C - 1] for c in cc) / len(cc)
        step_first_out = sum(c["hits_by_recency"][C] for c in cc) / len(cc)
        per_c[C] = {"hit_within": round(r_in, 4), "n_within": n_in,
                    "hit_beyond": round(r_out, 4), "n_beyond": n_out,
                    "step_Cm1_to_C": round(step_last_in - step_first_out, 4)}
        g1.append(r_in >= 0.90)
        g2.append(r_out <= r_in - 0.25)
        g3.append(step_last_in - step_first_out >= 0.25)
    gates["G1_fidelity_within_capacity"] = all(g1)
    gates["G2_ladder_step_beyond"] = all(g2)
    gates["G3_dose_boundary_tracks_C"] = sum(g3) >= 3
    verdict = ("OPERATOR_CAPACITY_CONFIRMED" if all(
        gates[k] for k in ("G0_dedupe_silent", "G1_fidelity_within_capacity",
                           "G2_ladder_step_beyond", "G3_dose_boundary_tracks_C"))
        else "GATE_FAIL")
    out = {"exp": "exp58", "standalone": True,
           "pre_registered": "ledger exp58 row (this commit)",
           "d_h": D_H, "k": K, "C_values": list(CS), "seeds": list(SEEDS),
           "cue_frac": CUE_FRAC, "cells": cells, "per_C": per_c,
           "gates": gates, "verdict": verdict,
           "smoke": bool(SMOKE),
           "elapsed_s": round(time.perf_counter() - t0, 1)}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out_path = os.path.join(ROOT, "results",
                            "exp58_wm_capacity_smoke.json" if SMOKE
                            else "exp58_wm_capacity.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("EXP58 %s gates=%s per_C=%s elapsed=%.0fs out=%s"
          % (verdict, json.dumps(gates), json.dumps(per_c),
             time.perf_counter() - t0, out_path), flush=True)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
