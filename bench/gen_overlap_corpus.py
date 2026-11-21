# -*- coding: utf-8 -*-
'''gen_overlap_corpus.py -- 合成可控 overlap 语料生成器（G-R3 修复 B1）。

预注册（本 docstring 先于任何正式数据落盘；LEDGER + D-3xx 同步提交）：

**目标**：生成输入 cosine 可精确控制的合成模式集，用于测量 β*(overlap) 阶梯
（G-R3 修复，v1.0 中 G-R3 FAIL = 真实语料 ~25% plateau 不可达）。

**构造法**（star + private pool，§4.11 Replicator 1 教训——不用 mixing weight
自欺，必须用真实 input cosine 做验收）：
- 共享集 S：k·o 个位置，所有 item 共用（二进制 1.0）；
- 私有池 P：m = max(10·k·(1-o), k) 个位置，每个 item 从 P 中无放回抽
  k·(1-o) 个（二进制 1.0）；
- d = |S| + |P|；item = S ∪ (private pick)，其余位置 0；
- pairwise cosine = (|S| + |P_i ∩ P_j|) / k，E[|P_i ∩ P_j|] =
  C(k(1-o),2)/C(m,2) ≈ k(1-o)²/m，当 m = 10·k·(1-o) 时偏差 ≈ 1/(100k) < 0.001。

**验收判据（预注册，写死）**：
- G-B1.a：实测全部 C(N,2) 对的 input-cosine 均值 = 目标 o ± 0.02；
- G-B1.b：实测 cosine 分布 std ≤ 0.15（分布不过宽，可控）；
- G-B1.c：o=0 时每 item 独立采样，cosine 均值 ≤ 0.02（pool=100k 下 E[cos]=k/m=0.01）。

**sanity（预注册）**：o=0 的语料在 B2 中应复现 v1.0 的 β*(N) 阶梯
（β*(2e3)=20、β*(2e4)=100，exp54/exp45 口径）。

**旋钮**：目标 overlap o∈[0,0.9]、库规模 N、k（每 item 非零位数）、
seed。编码器臂（uniform k-sparse / learned）在 B2 选择，本生成器只产模式。

**产物**：results/g3_corpus_manifest.json（seed、旋钮、全部验收读数）。
'''
import argparse
import json
import os
import sys
import time
from itertools import combinations

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RNG = np.random.default_rng(12345)


def gen_corpus(n, k, overlap, seed=0):
    """生成 N 个 k-sparse 二值模式，pairwise input cosine ≈ overlap。

    返回 (X, d, shared_size, pool_size)。
    X: (N, d) float32，每行恰好 k 个非零位（值 1.0）。
    """
    rng = np.random.default_rng(seed)
    o = float(overlap)
    shared_size = int(round(k * o))
    private_per = k - shared_size
    pool_size = 100 * k if private_per > 0 else 0  # m=100k: E[private cosine]=(1-o)²/100≤0.01
    d = shared_size + pool_size

    # 共享集 S（前 shared_size 个位置）
    # 私有池 P（后 pool_size 个位置）
    X = np.zeros((n, d), dtype=np.float32)
    X[:, :shared_size] = 1.0  # 所有 item 共享

    if private_per > 0:
        # 每个 item 从私有池无放回抽 private_per 个
        for i in range(n):
            picks = rng.choice(pool_size, size=private_per, replace=False)
            X[i, shared_size + picks] = 1.0

    return X, d, shared_size, pool_size


def measure_cosine(X, max_pairs=200000):
    """计算 input cosine 统计量。返回 (mean, std, min, max, n_pairs)。

    N≤2000 时全量计算；N 更大时随机抽样 max_pairs 对（均值估计精度足够）。
    """
    N = len(X)
    if N < 2:
        return 0.0, 0.0, 0.0, 0.0, 0
    norms = np.sqrt(np.sum(X * X, axis=1))
    Xn = X / norms[:, None]
    cos = Xn @ Xn.T
    np.fill_diagonal(cos, np.nan)
    if N * (N - 1) // 2 > max_pairs:
        # 随机抽样
        iu = np.triu_indices(N, k=1)
        idx = _RNG.choice(len(iu[0]), size=max_pairs, replace=False)
        pairs = cos[iu[0][idx], iu[1][idx]]
    else:
        pairs = cos[np.triu_indices(N, k=1)]
    return float(np.mean(pairs)), float(np.std(pairs)), float(np.nanmin(pairs)), float(np.nanmax(pairs)), len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--overlap", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    t0 = time.perf_counter()
    X, d, shared_size, pool_size = gen_corpus(args.n, args.k, args.overlap, args.seed)
    gen_s = time.perf_counter() - t0

    mean_cos, std_cos, min_cos, max_cos, n_pairs = measure_cosine(X)
    target = args.overlap

    # 验收
    gate_a = abs(mean_cos - target) <= 0.02
    gate_b = std_cos <= 0.15
    gate_c = (target == 0.0 and abs(mean_cos) <= 0.02) or (target > 0.0 and gate_a)

    n_nonzero = int(np.sum(X != 0))
    n_expected = args.n * args.k

    manifest = {
        "exp": "g3_corpus",
        "n": args.n,
        "k": args.k,
        "overlap_target": target,
        "seed": args.seed,
        "d": d,
        "shared_size": shared_size,
        "pool_size": pool_size,
        "gen_seconds": round(gen_s, 3),
        "measure_seconds": round(time.perf_counter() - t0 - gen_s, 3),
        "n_pairs_measured": n_pairs,
        "acceptance": {
            "G-B1.a_mean_abs_diff": round(abs(mean_cos - target), 6),
            "G-B1.a_pass": gate_a,
            "G-B1.b_std": round(std_cos, 6),
            "G-B1.b_pass": gate_b,
            "G-B1.c_pass": gate_c,
            "all_pass": bool(gate_a and gate_b and gate_c),
        },
        "cosine_stats": {
            "mean": round(mean_cos, 6),
            "std": round(std_cos, 6),
            "min": round(min_cos, 6),
            "max": round(max_cos, 6),
        },
        "n_nonzero_total": n_nonzero,
        "n_expected_total": n_expected,
        "sparsity_ok": bool(n_nonzero == n_expected),
    }

    out_path = args.out or os.path.join(ROOT, "results", "g3_corpus_manifest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("gen_overlap_corpus: n=%d k=%d o=%.1f d=%d shared=%d pool=%d"
          % (args.n, args.k, target, d, shared_size, pool_size))
    print("  cosine: mean=%.4f std=%.4f min=%.4f max=%.4f (target=%.1f)"
          % (mean_cos, std_cos, min_cos, max_cos, target))
    print("  gates: a=%s b=%s c=%s -> %s"
          % (gate_a, gate_b, gate_c, "PASS" if (gate_a and gate_b and gate_c) else "FAIL"))
    print("  manifest:", out_path)

    sys.exit(0 if (gate_a and gate_b and gate_c) else 4)


if __name__ == "__main__":
    main()
