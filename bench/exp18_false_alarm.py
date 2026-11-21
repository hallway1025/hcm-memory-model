#!/usr/bin/env python3
"""exp18_false_alarm.py -- k-稀疏吸引子码的虚警率解析界 + 蒙特卡洛验证。

主张：N 条随机 k-稀疏码（支撑集近似 Binom(k, k/d_h) 重叠）下，
库外查询的最大虚警分 max_j <c_j, h_q> 的解析分布与蒙特卡洛吻合（KS 检验），
且与库内自匹配分（≈1）的间隔随 d_h 增大 —— 这解释了 exp2c 的容量边界。

模型：
  - uniform 码：支撑集均匀随机 k 元，值 1/sqrt(k) -> score = S/k, S~Hypergeom(dh,k,k)
  - encoder 码：真实 kWTA(relu(W_e x)) 路径产生的码（值=截断高斯次序统计量）
解析：S 的精确超几何分布；max over N 用 union bound + 精确分布对比。
输出：results/exp18_false_alarm.json
"""
import sys, os, json, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KS = 20
DHS = [1024, 4096]
NS = [2000, 16000, 200000]
TRIALS = 4000


def hypergeom_pmf(s, dh, k):
    """P(S=s), S = |support(c) ∩ support(h)|, S~Hypergeom(dh, k, k)"""
    from math import comb
    out = np.zeros(s.max() + 1) if hasattr(s, "max") else None
    vals = {}
    for ss in np.unique(s):
        lo, hi = max(0, 2 * k - dh), k
        if ss < lo or ss > k:
            vals[int(ss)] = 0.0
        else:
            vals[int(ss)] = comb(k, ss) * comb(dh - k, k - ss) / comb(dh, k)
    return np.array([vals[int(x)] for x in s])


def sample_max_uniform(dh, k, N, trials, rng):
    """库外查询的 max score 蒙特卡洛（uniform 码）。
    score_j = <c_j, h> = S_j/k，S_j = 支撑重叠数 ~ Hypergeom(dh, k, k)。
    向量化：ids = repeat(arange(N), k)；score = bincount(ids, w=qmap[lib]) / sqrt(k)。"""
    lib = rng.integers(0, dh, size=(N, k))
    ids = np.repeat(np.arange(N, dtype=np.int64), k)
    c = 1.0 / math.sqrt(k)
    mx = np.empty(trials, dtype=np.float32)
    for t in range(trials):
        q = rng.integers(0, dh, size=k)
        qmap = np.zeros(dh, dtype=np.float32); qmap[q] = c
        w = qmap[lib.ravel()] * c
        sc = np.bincount(ids, weights=w.astype(np.float64), minlength=N)
        mx[t] = sc.max() / 1.0  # 已含 c^2 权重 = 1/k per shared unit
    return mx, lib, ids


def sample_max_encoder(dh, k, N, trials, seed):
    """encoder 码：W_e orthonormal, s=W_e x ~ N(0,I), code=top-k of relu(s) 归一化。"""
    from hcm.ops import orthogonal_matrix, unit
    rng = np.random.default_rng(seed)
    W = orthogonal_matrix(dh, dh, rng)  # 正交：s 的每个分量 ~ N(0,1)（scale=1/sqrt(dh) 与 encoder 一致）
    # 库码
    X = rng.standard_normal((N, dh)).astype(np.float32)
    S = X @ W
    idx = np.argsort(-S, axis=1)[:, :k]
    vals = np.take_along_axis(S, idx, axis=1)
    vals = np.maximum(vals, 0).astype(np.float32)
    libn = np.linalg.norm(vals, axis=1, keepdims=True)
    vals = vals / np.maximum(libn, 1e-12)
    mx = np.empty(trials, dtype=np.float32)
    self_sc = np.empty(trials, dtype=np.float32)
    for t in range(trials):
        x = rng.standard_normal(dh).astype(np.float32)
        s = W @ x
        qi = np.argsort(-s)[:k]
        qv = np.maximum(s[qi], 0).astype(np.float32)
        qv = qv / max(float(np.linalg.norm(qv)), 1e-12)
        qmap = np.zeros(dh, dtype=np.float32); qmap[qi] = qv
        w = (qmap[idx.ravel()] * vals.ravel()).astype(np.float64)
        ids = np.repeat(np.arange(N, dtype=np.int64), k)
        sc = np.bincount(ids, weights=w, minlength=N)
        mx[t] = sc.max()
        # 自匹配（查询=库中某条的退化版：用该条原输入加噪）
        j = rng.integers(0, N)
        xj = X[j] + 0.5 * rng.standard_normal(dh).astype(np.float32)
        sj = W @ xj
        qij = np.argsort(-sj)[:k]
        qvj = np.maximum(sj[qij], 0).astype(np.float32)
        qvj = qvj / max(float(np.linalg.norm(qvj)), 1e-12)
        self_sc[t] = float(vals[j] @ qvj)
    return mx, self_sc


def ks_stat(sample, cdf, cdf_left=None):
    """离散分布的 KS：D- 必须用左极限 F(x-)（右连续会在大原子上高估）。"""
    x = np.sort(sample)
    n = len(x)
    F = cdf(x)
    d_plus = (np.arange(1, n + 1) / n - F).max()
    Fl = cdf_left(x) if cdf_left is not None else F
    d_minus = (Fl - np.arange(0, n) / n).max()
    return float(max(d_plus, d_minus))


def main():
    global TRIALS, NS
    if os.environ.get('SMOKE'):
        TRIALS, NS = 400, [2000]
    out = {"what": "k-sparse false-alarm bound: hypergeom overlap model vs Monte Carlo",
           "K": KS, "rows": []}
    for dh in DHS:
        # 解析 S 分布
        s_grid = np.arange(0, KS + 1)
        pmf = hypergeom_pmf(s_grid, dh, KS)
        cdfS = np.cumsum(pmf)
        # 单码 CDF（score 坐标）
        def cdf1(x, cdfS=cdfS, k=KS):
            idx = np.array([min(int(np.floor(xx * k + 1e-6)), len(cdfS) - 1) for xx in x])
            return cdfS[idx]
        tail = {int(s): float(pmf[s:].sum()) for s in s_grid if pmf[s] > 0}
        # N*：union bound = 1 的规模（max score 达到 s/k 的临界库容）
        def capacity_for(score_thresh):
            s_req = int(math.ceil(score_thresh * KS)) - 1  # 严格超过需 S>=ceil
            p = float(pmf[s_req:].sum())
            return 1.0 / p if p > 0 else float("inf")
        for N in NS:
            rng = np.random.default_rng(500 + dh + N % 97)
            # uniform 模型（bincount 路径，N=200000 时 4e6 索引）
            # max 分布 CDF = F_S(x)^N（N 条码独立）——在 N 循环内构造
            def cdf(x, cdfS=cdfS, k=KS, NN=N):
                return cdf1(x, cdfS, k) ** NN
            # max 对照：20 个独立码库 × (TRIALS/20) 次查询 -> 均值/p99 vs F_S^N 分位数
            n_lib, per = 20, TRIALS // 20
            allmx = []
            for lb in range(n_lib):
                mx_lb, lib, ids = sample_max_uniform(dh, KS, N, per, rng)
                allmx.append(mx_lb)
            mx = np.concatenate(allmx)
            ks_u = float("nan")  # max 的 KS 在共享库下无效，改分位数对照
            q50 = float(cdf1(np.array([0.0]), cdfS=None) if False else 0.0)  # 占位防未定义
            # 单对 S 分布直接检验：一个查询 vs 库中全部 N 码（分数坐标喂 cdf1）
            q = rng.integers(0, dh, size=KS)
            qmap = np.zeros(dh, dtype=np.float32); qmap[q] = 1.0 / math.sqrt(KS)
            w = (qmap[lib.ravel()] * (1.0 / math.sqrt(KS))).astype(np.float64)
            sc_all = np.bincount(ids, weights=w, minlength=N)
            # 左极限：P(S < s) = cdfS[idx-1]
            def cdf_left(x, cdfS=cdfS, k=KS):
                idx = np.array([min(int(np.floor(xx * k + 1e-6)), len(cdfS) - 1) for xx in x]) - 1
                return np.where(idx >= 0, cdfS[np.maximum(idx, 0)], 0.0)
            ks_single = ks_stat(sc_all, cdf1, cdf_left)
            emp_p = {int(s): float((np.round(mx * KS).astype(int) == s).mean()) for s in s_grid if pmf[s] > 0}
            # 解析 F_S^N 分位数
            q50a = float(next(s for s in s_grid if cdf1(np.array([s / KS]))[0] ** N >= 0.5) / KS)
            q99a = float(next(s for s in s_grid if cdf1(np.array([s / KS]))[0] ** N >= 0.99) / KS)
            out["rows"].append(dict(model="uniform", dh=dh, N=N,
                                    max_score_mean=float(mx.mean()), max_score_p99=float(np.quantile(mx, 0.99)),
                                    analytic_max_median=q50a, analytic_max_p99=q99a,
                                    analytic_mean_S_over_k=float((s_grid * pmf).sum() / KS),
                                    tail_analytic=tail, ks_single=ks_single,
                                    emp_vs_analytic_pmf=emp_p))
            print("uniform dh=%d N=%d max mean=%.4f (analytic median %.4f) p99=%.4f (analytic %.4f) KSsingle=%.4f" % (
                dh, N, mx.mean(), q50a, np.quantile(mx, 0.99), q99a, ks_single), flush=True)
        # encoder 模型（N 大时 W 乘法 O(N dh^2) 太贵，只在 N=2000 做）
        mx_e, ss_e = sample_max_encoder(dh, KS, 2000, 400, 900 + dh)
        print("encoder dh=%d N=2000 max=%.4f self=%.4f margin=%.2f" % (
            dh, mx_e.mean(), ss_e.mean(), ss_e.mean() - mx_e.mean()), flush=True)
        out["rows"].append(dict(model="encoder", dh=dh, N=2000,
                                max_score_mean=float(mx_e.mean()), self_match_mean=float(ss_e.mean()),
                                margin=float(ss_e.mean() - mx_e.mean()),
                                capacity_score05=capacity_for(0.5), capacity_score1=capacity_for(1.0)))
    with open(os.path.join(ROOT, "results", "exp18_false_alarm.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
