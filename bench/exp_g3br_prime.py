# -*- coding: utf-8 -*-
'''exp_g3br_prime.py -- G-R3b-prime: knee 定位密扫（v1.4 工单 A 组）。

预注册（本 docstring 先于任何正式数据落盘）：

**目标**：以 ×√2 步长密扫 β 网格，定位 o=0.2、N=2×10^3 的真实拐点 knee
（粗网格 10 点的 beta* = 网格下沿可能高估/低估真实拐点），检验
  knee * gap  >=  ln N   （gap = 1/k，k=20，与 G-R3b 判门同式）
是否成立。如实报告，不设期望方向。

**网格**：beta ∈ {7.07, 10, 14.1, 20, 28.3, 40, 56.6, 80, 113, 160, 226, 320}
（×√2 双向延伸，覆盖粗网格 beta*=20..50 邻域）。
**配置**：o=0.2, N=2000, seeds {0,1,2}, deg0.3, 40 probes —— 与 G-R3 主网格同款。
**产物**：results/g3br_prime.json（含 knee/gap/lnN 判定）。
'''
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "bench"))

import exp_g3_staircase as G3
from gen_overlap_corpus import gen_corpus  # noqa: E402

BETAS_DENSE = (7.0710678, 10.0, 14.1421356, 20.0, 28.2842712, 40.0,
               56.5685425, 80.0, 113.137085, 160.0, 225.853, 320.0)
K_SPARSE = 20
GAP = 1.0 / K_SPARSE
O_LEVEL = 0.2
N_LEVEL = 2000
SEEDS = (0, 1, 2)


def main():
    G3.BETAS = BETAS_DENSE  # monkeypatch dense grid into measure_beta_star
    t00 = time.perf_counter()
    partial_path = os.path.join(ROOT, "results", "g3br_prime_partial.json")
    rows = []
    if os.path.exists(partial_path):
        with open(partial_path, encoding="utf-8") as f:
            rows = json.load(f)
        print("resume: %d seeds already done" % len(rows), flush=True)
    done_seeds = {r["seed"] for r in rows}
    for seed in SEEDS:
        if seed in done_seeds:
            print("[seed %d] SKIP (checkpoint)" % seed, flush=True)
            continue
        key = "o%g_N%d_s%d" % (O_LEVEL, N_LEVEL, seed)
        t0 = time.perf_counter()
        X, d, shared, pool = gen_corpus(N_LEVEL, K_SPARSE, O_LEVEL, seed)
        gen_s = time.perf_counter() - t0
        print("[%s] gen %.1fs d=%d" % (key, gen_s, d), flush=True)
        res = G3.measure_beta_star(X, N_LEVEL, seed)
        res["o"] = O_LEVEL
        res["N"] = N_LEVEL
        res["seed"] = seed
        res["d"] = d
        res["gen_s"] = round(gen_s, 1)
        res["total_s"] = round(time.perf_counter() - t0, 1)
        print("[%s] knee=%g rate=%.3f (%.0fs)" % (
            key, res["beta_star"], res["beta_star_rate"], res["total_s"]),
            flush=True)
        rows.append(res)
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1, ensure_ascii=False)
        del X
        continue_ = None

    # G-R3b-prime 判定：knee * gap >= ln N（逐 seed + 均值）
    lnN = math.log(N_LEVEL)
    verdicts = []
    for r in rows:
        knee_gap = r["beta_star"] * GAP
        verdicts.append({"seed": r["seed"], "knee": r["beta_star"],
                         "knee_x_gap": round(knee_gap, 4),
                         "ln_N": round(lnN, 4),
                         "holds": bool(knee_gap >= lnN)})
    knee_mean = sum(v["knee"] for v in verdicts) / len(verdicts)
    out = {"exp": "g3br_prime", "hypothesis": "knee*gap >= ln N",
           "gap": GAP, "ln_N": round(lnN, 4),
           "betas": list(BETAS_DENSE), "o": O_LEVEL, "N": N_LEVEL,
           "seeds": list(SEEDS), "rows": rows,
           "verdicts": verdicts,
           "knee_mean": round(knee_mean, 2),
           "knee_x_gap_mean": round(knee_mean * GAP, 4),
           "holds_all": all(v["holds"] for v in verdicts),
           "holds_any": any(v["holds"] for v in verdicts),
           "total_s": round(time.perf_counter() - t00, 1)}
    out_path = os.path.join(ROOT, "results", "g3br_prime.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("saved:", out_path, "total %.0fs" % out["total_s"], flush=True)
    print("G-R3b-prime verdicts:", json.dumps(verdicts, ensure_ascii=False))
    print("holds_all=%s holds_any=%s knee_mean=%g (ln N=%.2f)"
          % (out["holds_all"], out["holds_any"], knee_mean, lnN), flush=True)


if __name__ == "__main__":
    main()
