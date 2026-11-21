#!/usr/bin/env python3
"""analyze_g3_staircase.py -- G-R3a/b/c 判门（合并 M2 + M4 网格）。

用法:
  py -X utf8 tools/analyze_g3_staircase.py --inputs results/g3_staircase_m2.json results/g3_staircase_m4.json --out results/g3_analysis.json

判门（预注册于 bench/exp_g3_staircase.py docstring）：
  G-R3a: Spearman rho(o, beta*) >= 0.9  (n = 5 o-levels x 3 seeds)
  G-R3b: beta*(o) 幅度 >= 理论下界 (log(N)/gap 量级，gap = 1/k)
  G-R3c: 效应量可分辨 (MDE 报告)
"""
import argparse, json, math, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench"))

K_SPARSE = 20
GAP = 1.0 / K_SPARSE          # score gap = 1/k


def spearman_rho(x, y):
    """numpy-only Spearman（无 scipy 依赖）。"""
    import numpy as np
    def rank(v):
        # 平均排名（ties 取均值）——并列值必须同秩，否则 rho 被随机打散
        v = np.asarray(v, dtype=np.float64)
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=np.float64)
        sv = v[order]
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and sv[j + 1] == sv[i]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / den) if den > 0 else 0.0


def theoretical_floor(N):
    """beta >= ln(N)/gap —— N 个竞争候选，需 softmax 压过一个 gap。"""
    return math.log(N) / GAP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "g3_analysis.json"))
    args = ap.parse_args()

    import glob as _glob
    files = []
    for pat in args.inputs:
        if os.path.isdir(pat):
            files.extend(sorted(_glob.glob(os.path.join(pat, "g3_cell_*.json"))))
        elif os.path.exists(pat):
            files.append(pat)
        else:
            files.extend(sorted(_glob.glob(pat)))

    rows = []
    provenance = []
    for p in files:
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception as e:
            provenance.append({"file": os.path.basename(p), "status": "ERROR", "err": str(e)[:120]})
            continue
        rows.extend(d.get("rows", []))
        provenance.append({"file": os.path.basename(p), "status": "OK",
                           "rows": len(d.get("rows", [])),
                           "overlaps": d.get("overlaps"), "seeds": d.get("seeds"),
                           "n_grid": d.get("n_grid"), "total_s": d.get("total_s")})

    # 去重（同 (o,N,seed) 只保留一条；后到者覆盖）
    uniq = {}
    for r in rows:
        uniq[(r["o"], r["N"], r["seed"])] = r
    rows = [uniq[k] for k in sorted(uniq)]

    out = {"provenance": provenance, "n_rows": len(rows),
           "beta_grid": sorted({b for r in rows for b in r.get("beta_rates", {})}, key=float),
           "rows": rows}

    # 网格完整性：每个 cell 覆盖的 β 点数（不足则 β* 不可信）
    from collections import defaultdict
    cov = defaultdict(int)
    for r in rows:
        cov[(r["o"], r["N"], r["seed"])] = len(r.get("beta_rates", {}))
    out["beta_coverage"] = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in sorted(cov.items())}
    out["coverage_hist"] = {}
    for v in cov.values():
        out["coverage_hist"][str(v)] = out["coverage_hist"].get(str(v), 0) + 1

    # ---- G-R3a: Spearman rho(o, beta*) ----
    os_ = [r["o"] for r in rows]
    bs = [float(r["beta_star"]) for r in rows]
    rho = spearman_rho(os_, bs)
    out["G-R3a"] = {"rho": round(rho, 4), "n": len(rows),
                    "threshold": 0.9, "verdict": "PASS" if rho >= 0.9 else "FAIL"}

    # ---- 按 N 分层 ----
    out["by_N"] = {}
    for N in sorted({r["N"] for r in rows}):
        sub = [r for r in rows if r["N"] == N]
        rN = spearman_rho([r["o"] for r in sub], [float(r["beta_star"]) for r in sub])
        # 每个 o 层的 beta* 众数
        levels = {}
        for r in sub:
            levels.setdefault(r["o"], []).append(float(r["beta_star"]))
        out["by_N"][str(N)] = {
            "rho": round(rN, 4),
            "beta_star_by_overlap": {str(o): sorted(set(v)) for o, v in sorted(levels.items())},
            "mean_beta_star_by_overlap": {str(o): round(sum(v)/len(v), 2) for o, v in sorted(levels.items())},
            "floor_lnN_over_gap": round(theoretical_floor(N), 1),
        }

    # ---- G-R3b: 幅度 >= 理论下界 ----
    # 在最大 o（污染最强）与最小 o 之间比较 beta* 增幅，并对照 floor
    byN = out["by_N"]
    g_r3b = {}
    for N, blk in byN.items():
        mb = blk["mean_beta_star_by_overlap"]
        ks = sorted(mb, key=float)
        if len(ks) >= 2:
            lo, hi = ks[0], ks[-1]
            floor = blk["floor_lnN_over_gap"]
            per_o_floor = {o: bool(v >= floor) for o, v in mb.items()}
            g_r3b[N] = {"o_low": lo, "o_high": hi,
                        "beta_low": mb[lo], "beta_high": mb[hi],
                        "ratio": round(mb[hi]/mb[lo], 3) if mb[lo] else None,
                        "delta": round(mb[hi]-mb[lo], 2),
                        "floor": floor,
                        "delta_ge_floor": bool((mb[hi]-mb[lo]) >= floor),
                        "per_o_ge_floor": per_o_floor,
                        "all_o_ge_floor": all(per_o_floor.values()) if per_o_floor else None}
    out["G-R3b"] = g_r3b

    # ---- G-R3c: MDE ----
    # 对每个 (o,N)，seeds 间 beta* 的离散度 -> 最小可分辨效应
    import statistics
    spreads = []
    for N in sorted({r["N"] for r in rows}):
        for o in sorted({r["o"] for r in rows}):
            vals = [float(r["beta_star"]) for r in rows if r["o"] == o and r["N"] == N]
            if len(vals) >= 2:
                spreads.append({"N": N, "o": o, "n_seeds": len(vals),
                                "values": vals,
                                "mean": round(statistics.mean(vals), 2),
                                "sd": round(statistics.pstdev(vals), 3)})
    out["G-R3c"] = {"per_cell": spreads,
                    "max_sd": max([s["sd"] for s in spreads], default=None),
                    "note": "beta* 取值受网格分辨率限制（20/100/500/2500），"
                            "sd=0 的格子说明真值落在网格间隙内，MDE 下界受此限制"}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    print("=== G3 analysis ===")
    print("rows:", len(rows), "| provenance:", json.dumps(out["provenance"], ensure_ascii=False))
    print("G-R3a rho = %.4f (n=%d) -> %s" % (rho, len(rows), out["G-R3a"]["verdict"]))
    for N, blk in byN.items():
        print("  N=%s rho=%.4f  beta* by o: %s  floor=%.1f" %
              (N, blk["rho"], blk["mean_beta_star_by_overlap"], blk["floor_lnN_over_gap"]))
    print("G-R3b:", json.dumps(g_r3b, ensure_ascii=False))
    print("saved:", args.out)


if __name__ == "__main__":
    main()
