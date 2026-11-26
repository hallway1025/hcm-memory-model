"""实验 A：λ 扫描 —— 检验 dAA/dλ 是否存在内点最优。

W1 工单预实验配置：M=100_000 固定，λ ∈ {0.1,0.3,0.5,0.7,0.9}，seeds {0,1,2}，
layers=1 单层基线锚定端点（同预算同种子）。共 15+3=18 次训练。
全量模式（4 预算 × 9 λ × 10 seeds）为后续，--full 开启。

判定（工单）：曲线先升后降且峰值 λ ∈ [0.15,0.85] → 申请全量；
AA(λ) 单调 → 方向 B 当场死亡；全程平坦 → 无效应撤回。
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b_common import TrialConfig, run_trial, ResultStore, bca_bootstrap


def run_sweep(store_path, M, lambdas, seeds):
    store = ResultStore(store_path)
    done = {r["key"] for r in store.data}
    for lam in lambdas:
        for seed in seeds:
            cfg = TrialConfig(total_bytes=M, lam=lam, layers=2, seed=seed)
            if cfg.key() in done:
                continue
            store.add(cfg, run_trial(cfg))
            aa = store.aa_by(total_bytes=M, lam=lam, layers=2)[-1]
            print(f"[lam={lam:.1f} seed={seed}] AA={aa:.4f}", flush=True)
    for seed in seeds:
        cfg = TrialConfig(total_bytes=M, lam=1.0, layers=1, seed=seed)
        if cfg.key() in done:
            continue
        store.add(cfg, run_trial(cfg))
        aa = store.aa_by(total_bytes=M, layers=1)[-1]
        print(f"[L1 seed={seed}] AA={aa:.4f}", flush=True)
    print(f"sweep done: {len(store.data)} records -> {store_path}", flush=True)


def analyze(store_path, M, lambdas):
    store = ResultStore(store_path)
    curve = {}
    for lam in lambdas:
        aa = store.aa_by(total_bytes=M, lam=lam, layers=2)
        if len(aa) == 0:
            continue
        curve[lam] = {
            "mean": float(aa.mean()),
            "std": float(aa.std(ddof=1)) if len(aa) > 1 else 0.0,
            **bca_bootstrap(aa, np.mean),
        }
    base_aa = store.aa_by(total_bytes=M, layers=1)
    base_stats = None
    if len(base_aa):
        base_stats = {
            "mean": float(base_aa.mean()),
            "std": float(base_aa.std(ddof=1)) if len(base_aa) > 1 else 0.0,
            **bca_bootstrap(base_aa, np.mean),
        }

    if len(curve) < 3:
        return {"M": M, "error": "数据不足", "curve": curve, "baseline": base_stats}

    lams = np.array(sorted(curve))
    means = np.array([curve[l]["mean"] for l in lams])

    # 一阶差分（中心差分）
    d1 = np.gradient(means, lams)

    # 符号变化点（d1 由正转负），线性插值找零点
    sign_change = None
    for i in range(len(d1) - 1):
        if d1[i] > 0 and d1[i + 1] <= 0:
            x0 = lams[i] + (lams[i+1] - lams[i]) * d1[i] / (d1[i] - d1[i+1])
            sign_change = float(x0)
            break

    is_interior = (sign_change is not None and 0.15 < sign_change < 0.85)

    if np.all(d1 > 0):
        mono = "increasing"
    elif np.all(d1 < 0):
        mono = "decreasing"
    else:
        mono = "non-monotone"

    # 平坦判定：相邻均值差全部小于合并标准误
    flat = False
    if len(means) >= 2:
        pooled = float(np.mean([curve[l]["std"] for l in lams]))
        flat = bool(np.all(np.abs(np.diff(means)) < pooled))

    # 功率检查：flat 判据触发（相邻均值差全部小于合并标准误）时，
    # 名义内点最优不可分辨，不得据以申请全量
    if is_interior and flat:
        verdict, why = "UNDERPOWERED_NOMINAL_INTERIOR", (
            f"名义内点最优 λ*={sign_change:.3f}，但 flat 判据触发（相邻均值差均小于合并标准误），"
            "CI 重叠，信号在噪声之下 → 不可申请全量，需增种子/预算档")
    elif is_interior:
        verdict, why = "PASS_INTERIOR", f"内点最优 λ*={sign_change:.3f} ∈ (0.15,0.85) → 申请全量"
    elif mono == "increasing":
        verdict, why = "FAIL_MONOTONIC_UP", "AA(λ) 单调递增：快层越大越好，分层无意义，方向 B 撤回"
    elif mono == "decreasing":
        verdict, why = "FAIL_MONOTONIC_DOWN", "AA(λ) 单调递减：慢层越大越好（等价压缩存储优于保留原样本）"
    elif flat:
        verdict, why = "FAIL_FLAT", "AA(λ) 平坦：分配比例无效应，主张撤回"
    else:
        verdict, why = "WEAK", "无内点最优但非单调非平坦：弱证据，需补预算档/种子"

    # 基线口径检查：L1 基线的秩容量是否计入字节预算（等字节原则）
    rec_bytes = 4 * (1024 + 1024 + 128 + 128 + 32)
    rank_bytes = 4 * (128 + 1024)
    l1_rec = max(1, M // rec_bytes)
    l1_rank = 64
    l1_charged = l1_rec * rec_bytes
    l1_effective = l1_charged + l1_rank * rank_bytes
    best_lam = float(lams[means.argmax()])
    b_rec = max(1, int(best_lam * M) // rec_bytes)
    b_rank = max(1, int((1.0 - best_lam) * M) // rank_bytes)
    two_charged = b_rec * rec_bytes + b_rank * rank_bytes
    baseline_fairness = {
        "L1_n_rec": l1_rec,
        "L1_n_rank": l1_rank,
        "L1_bytes_charged": l1_charged,
        "L1_effective_capacity_bytes": l1_effective,
        "L1_rank_free": True,
        "two_layer_at_argmax": {"lambda": best_lam, "n_rec": b_rec, "n_rank": b_rank,
                                "bytes_charged": two_charged},
        "capacity_ratio_L1_over_two": round(l1_effective / two_charged, 3),
        "note": "L1 基线 64 秩未计入字节预算，等效容量远超等字节双层；"
                "非劣性腿在当前口径下不可判读，全量前须先修正基线口径（等字节强基线）",
    }

    return {
        "M": M,
        "baseline_fairness": baseline_fairness,
        "lambdas": lams.tolist(),
        "AA_mean": means.tolist(),
        "dAA_dlambda": d1.tolist(),
        "interior_optimum_lambda": sign_change,
        "is_interior": bool(is_interior),
        "monotonicity": mono,
        "flat": flat,
        "max_AA": float(means.max()),
        "argmax_lambda": float(lams[means.argmax()]),
        "verdict": verdict,
        "verdict_reason": why,
        "baseline_L1": base_stats,
        "curve": {str(k): v for k, v in curve.items()},
    }


def plot_curve(report, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    lams = report["lambdas"]
    means = report["AA_mean"]
    curve = report["curve"]
    errs = [[means[i] - curve[str(lams[i])]["ci_lo"] for i in range(len(lams))],
            [curve[str(lams[i])]["ci_hi"] - means[i] for i in range(len(lams))]]
    ax.errorbar(lams, means, yerr=errs, fmt="o-", capsize=4, label="AA(lambda) mean +/- BCa 95% CI")
    base = report.get("baseline_L1")
    if base:
        ax.axhline(base["mean"], ls="--", c="gray",
                   label=f"single-layer baseline L1 = {base['mean']:.4f}")
    if report.get("interior_optimum_lambda") is not None:
        ax.axvline(report["interior_optimum_lambda"], ls=":", c="green",
                   label=f"lambda* = {report['interior_optimum_lambda']:.3f}")
    ax.set_xlabel("lambda = |B1|/M (fast-layer budget share)")
    ax.set_ylabel("AA (semantic-readout accuracy)")
    ax.set_title(f"Exp A prelim: AA(lambda) curve (M={report['M']})\nverdict: {report['verdict']} - {report['verdict_reason']}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="results/exp_a_prelim.json")
    ap.add_argument("--png", default="results/exp_a_prelim_curve.png")
    ap.add_argument("--M", type=int, default=100_000)
    ap.add_argument("--lambdas", default="0.1,0.3,0.5,0.7,0.9")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--full", action="store_true",
                    help="全量模式：4 预算 × 9 λ × 10 seeds（默认预实验）")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    M = args.M
    if args.full:
        lambdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        seeds = list(range(10))
        budgets = [50_000, 100_000, 200_000, 400_000]
    else:
        lambdas = [float(x) for x in args.lambdas.split(",")]
        seeds = [int(x) for x in args.seeds.split(",")]
        budgets = [M]

    if not args.analyze_only:
        for Mb in budgets:
            run_sweep(args.store, Mb, lambdas, seeds)

    report_all = {}
    for Mb in budgets:
        rep = analyze(args.store, Mb, lambdas)
        report_all[str(Mb)] = rep
        print(f"\n=== M={Mb} ===")
        print(json.dumps({k: v for k, v in rep.items() if k != "curve"},
                         ensure_ascii=False, indent=1))
        if "lambdas" in rep:
            plot_curve(rep, args.png)
            print(f"curve -> {args.png}")

    with open("results/exp_a_prelim_analysis.json", "w", encoding="utf-8") as f:
        json.dump(report_all, f, ensure_ascii=False, indent=2)
    print("\n分析已写入 results/exp_a_prelim_analysis.json")


if __name__ == "__main__":
    main()
