# -*- coding: utf-8 -*-
'''exp_g3_staircase.py -- β*(overlap) 阶梯测量（G-R3 修复 B2）。

预注册（本 docstring 先于任何正式数据落盘；LEDGER + D-3xx 同步提交）：

**目标**：在合成可控 overlap 语料上测量 β* 随 code overlap o 的位移，
回答 v1.0 中 G-R3 FAIL 的问题（真实语料 ~25% plateau 不可达）。

**网格**：o ∈ {0, 0.2, 0.4, 0.6, 0.8} × N ∈ {2×10³, 2×10⁴} × seeds {0,1,2}
（L0 算力够，机1 笔记本可跑）。

**仪器**（exp45 同款 score-gap，简化版）：
- 每 (o, N, seed) 生成语料 → 全量写入 HCM（无写入门，capacity=N+10）；
- 写完后扫 β 网格 BETAS，每 β 测 deg0.3 召回（40 探针，均匀采样，
  degrade 30% 非零位）；
- β* = argmax 召回，平局取更小 β（exp45 同款 tie rule）；
- 记录 β*、各 β 召回率、m1（margin 读出，描述性）。

**修订 1（2026-11-19，先于正式数据）**：β 网格由旧 4 点 (20,100,500,2500)
加密为 10 点 (20,50,100,150,200,300,500,800,1200,2500)。理由：旧网格
400 倍空隙把 o=0.4 的真值 β*≈50 量到 100（misclassification 已在
机1探针实验中确认，400× gap → 假 FAIL）。加密不改判门方向，只消除
量化伪影；旧 4 点网格的全部点包含在新网格中。
**修订 2（2026-11-19）**：每格独立落盘 results/g3_cells/cell_o{o}_N{N}_s{seed}.json，
断点续跑（已存在的格默认跳过，--force 重跑）；支持 --cells/--overlaps/--n-grid
拆机并行。c-003 性能修正（retrieve float32 + dots 稀疏）使全网格可行。

**判门（预注册，写进稿前定死）**：
- G-R3a：β* 随 o 单调上升（Spearman ρ≥0.9，n=5×2×3=30 点）；
- G-R3b：β*(o) 幅度 ≥ 理论下界（Appendix E hypergeometric 虚警界：
  β ≥ 1/(p_FA) 量级，p_FA 随 o 增大 → β 需求随 o 增大）；
- G-R3c：效应量可分辨（MDE 报告，参照 §4.5 惯例）。

**分支诚实**：若 G-R3a FAIL → 照 v1.0 惯例 verbatim 报负结果
（"合成受控条件下也不单调"本身是新信息，直接回应"描述性拟合外推"质疑）。

**产物**：results/g3_cells/cell_*.json（单格）+ results/g3_staircase.json
（合并全网格）+ tools/analyze_g3_staircase.py（判门）。
'''
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "bench"))

from hcm import Config, HCM  # noqa: E402
from hcm.ops import unit  # noqa: E402
from gen_overlap_corpus import gen_corpus  # noqa: E402

BETAS = (20.0, 50.0, 100.0, 150.0, 200.0, 300.0, 500.0, 800.0, 1200.0, 2500.0)
DEG_LEVEL = 0.3
PROBES_N = 40
OVERLAPS = (0.0, 0.2, 0.4, 0.6, 0.8)
N_GRID = (2000, 20000)
SEEDS = (0, 1, 2)
K_SPARSE = 20
CELL_DIR = os.path.join(ROOT, "results", "g3_cells")


def degrade(x, frac, seed):
    """零化 frac 比例的非零位。"""
    rng = np.random.default_rng(seed)
    y = x.copy()
    nz = np.where(np.abs(y) > 1e-6)[0]
    if len(nz) == 0:
        return y
    k = max(1, int(len(nz) * frac))
    y[rng.choice(nz, size=k, replace=False)] = 0.0
    return y


def cell_path(o, N, seed):
    return os.path.join(CELL_DIR, "cell_o%s_N%d_s%d.json" % (("%g" % o).replace(".", "p"), N, seed))


def measure_beta_star(X, N, seed):
    """写库 + 扫 β 网格，返回 β* 与各 β 召回率。"""
    d = X.shape[1]
    m = HCM(Config(d_x=d, d_c=d, a=0.02, capacity=N + 10, beta=20.0,
                   L=3, use_resid=False, seed=seed))
    t0 = time.perf_counter()
    for x in X:
        m.observe(x)
    write_s = time.perf_counter() - t0
    n_written = int(m.E.n)

    # 探针：均匀采样
    probe_idx = list(range(0, N, N // PROBES_N))[:PROBES_N]

    results = {"write_s": round(write_s, 1), "n_written": n_written,
               "n_probes": len(probe_idx), "beta_rates": {}, "m1_mean": {}}
    for b in BETAS:
        m.cfg.beta = b
        ok = 0
        abst = 0
        m1s = []
        for i in probe_idx:
            q = degrade(X[i], DEG_LEVEL, i)
            o = m.query(q, do_rif=False)
            if o.get("action") == "abstain":
                abst += 1
            if int(o.get("idx", -1)) == i:
                ok += 1
            m1s.append(float(o.get("m1", 0.0)))
        rate = ok / len(probe_idx)
        results["beta_rates"]["%g" % b] = {"top1": ok, "rate": round(rate, 4),
                                           "abstain": abst}
        results["m1_mean"]["%g" % b] = round(float(np.mean(m1s)), 4)
        print("    β=%-6s deg%.1f: %d/%d rate=%.3f abst=%d"
              % (b, DEG_LEVEL, ok, len(probe_idx), rate, abst), flush=True)
    m.cfg.beta = 20.0

    # β* = argmax rate, tie → smaller β
    best_b = min(BETAS, key=lambda b: (-results["beta_rates"]["%g" % b]["rate"], b))
    results["beta_star"] = best_b
    results["beta_star_rate"] = results["beta_rates"]["%g" % best_b]["rate"]
    del m
    return results


def spearman_rho(x, y):
    """Spearman rank correlation (numpy-only, no scipy)."""
    def rank(v):
        order = np.argsort(v, kind="mergesort")
        ranks = np.empty_like(v, dtype=np.float64)
        ranks[order] = np.arange(len(v))
        return ranks
    rx, ry = rank(np.asarray(x, dtype=np.float64)), rank(np.asarray(y, dtype=np.float64))
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def main():
    global OVERLAPS, N_GRID, SEEDS, K_SPARSE, PROBES_N
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=str, default=None, help="comma-separated seeds (default: 0,1,2)")
    ap.add_argument("--overlaps", type=str, default=None, help="comma-separated overlaps")
    ap.add_argument("--n-grid", type=str, default=None, help="comma-separated N values")
    ap.add_argument("--cells", type=str, default=None, help="only run cells matching this substring filter (e.g. 'o0p4_N2000')")
    ap.add_argument("--force", action="store_true", help="re-run cells even if checkpoint exists")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    os.makedirs(CELL_DIR, exist_ok=True)
    cell_filter = args.cells

    if args.smoke:
        OVERLAPS = (0.0, 0.8)
        N_GRID = (500,)
        SEEDS = (0,)
        K_SPARSE = 20
        PROBES_N = 20
    else:
        if args.seeds:
            SEEDS = tuple(int(s) for s in args.seeds.split(","))
        if args.overlaps:
            OVERLAPS = tuple(float(o) for o in args.overlaps.split(","))
        if args.n_grid:
            N_GRID = tuple(int(n) for n in args.n_grid.split(","))

    t00 = time.perf_counter()
    rows = []
    print("exp_g3_staircase: o∈%s N∈%s seeds=%s k=%d betas=%s"
          % (OVERLAPS, N_GRID, SEEDS, K_SPARSE, BETAS), flush=True)

    for o in OVERLAPS:
        for N in N_GRID:
            for seed in SEEDS:
                key = "o%s_N%d_s%d" % (("%g" % o).replace(".", "p"), N, seed)
                if cell_filter and cell_filter not in key:
                    continue
                cpath = cell_path(o, N, seed)
                if os.path.exists(cpath) and not args.force:
                    with open(cpath, encoding="utf-8") as f:
                        rows.append(json.load(f))
                    print("[%s] SKIP (checkpoint exists)" % key, flush=True)
                    continue
                t0 = time.perf_counter()
                X, d, shared, pool = gen_corpus(N, K_SPARSE, o, seed)
                gen_s = time.perf_counter() - t0
                print("[o=%.1f N=%d seed=%d] gen %.1fs d=%d"
                      % (o, N, seed, gen_s, d), flush=True)
                res = measure_beta_star(X, N, seed)
                res["o"] = o
                res["N"] = N
                res["seed"] = seed
                res["d"] = d
                res["gen_s"] = round(gen_s, 1)
                res["total_s"] = round(time.perf_counter() - t0, 1)
                with open(cpath, "w", encoding="utf-8") as f:
                    json.dump(res, f, indent=1, ensure_ascii=False)
                print("[%s] saved %s" % (key, cpath), flush=True)
                rows.append(res)
                print("[o=%.1f N=%d seed=%d] β*=%g rate=%.3f (%.0fs)"
                      % (o, N, seed, res["beta_star"], res["beta_star_rate"], res["total_s"]), flush=True)
                del X

    out = {"exp": "g3_staircase", "betas": list(BETAS), "deg_level": DEG_LEVEL,
           "k": K_SPARSE, "overlaps": list(OVERLAPS), "n_grid": list(N_GRID),
           "seeds": list(SEEDS), "rows": rows,
           "total_s": round(time.perf_counter() - t00, 1)}

    out_path = args.out or os.path.join(ROOT, "results", "g3_staircase.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("\nsaved:", out_path, "total %.0fs" % out["total_s"])

    # Quick analysis: Spearman ρ (numpy-only)
    pts = [(r["o"], r["beta_star"]) for r in rows]
    if len(pts) >= 3:
        rho = spearman_rho([p[0] for p in pts], [p[1] for p in pts])
        print("\nSpearman ρ(o, β*) = %.4f (n=%d)" % (rho, len(pts)))
        print("G-R3a (ρ≥0.9):", "PASS" if rho >= 0.9 else "FAIL")


if __name__ == "__main__":
    main()
