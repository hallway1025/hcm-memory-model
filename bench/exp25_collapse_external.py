# -*- coding: utf-8 -*-
"""exp25_collapse_external.py —— 坍缩律外部复现（五招之一）。

零 HCM 代码：vanilla 现代 Hopfield / Transformer cross-attention 同构的软检索。
  a = softmax(beta * K q)      （注意力权重）
  r = unit(a @ K)              （一步更新：检索模式 = 权重平均）
  pred = argmax(K @ r)         （对检索模式最近邻分类）
低 beta 时权重弥散 -> r 是混合模式 -> 分类失败；高 beta 时 a 集中 -> 正确。
这正是 HCM 吸引子通道的机制原型，用于证明坍缩律不依赖 HCM 实现。

两种码型：
  A dense：K 行 iid 高斯单位向量；q = unit(K_true + 0.62*噪声)（输入余弦 ~0.85）。
  B sparse：d=1024 二值码 k=20/行（超几何原子，附录 E 同族）；q = 真码 25% 位翻转。
测量：hit@1(beta, N)、分数间隙 gap=s1-s2；拟合 beta* 随 N 的标度。
用法：py -X utf8 bench/exp25_collapse_external.py [--NMAX 100000]
输出：results/exp25_collapse_external.json
"""
import json, math, os, sys
import numpy as np

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
BETAS = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
D = 512
D_SP = 1024
K_SP = 20
FLIP = 0.25
RNG0 = 42000


def trials_for(N):
    return 200 if N <= 100000 else 50


def dense_arm(N, rng):
    T = trials_for(N)
    K = rng.standard_normal((N, D)).astype(np.float32)
    K /= np.linalg.norm(K, axis=1, keepdims=True)
    idx = rng.integers(0, N, size=T)
    eps = (0.62 / math.sqrt(D)) * rng.standard_normal((T, D)).astype(np.float32)  # 输入余弦≈1/sqrt(1+0.62^2)≈0.85
    Q = K[idx] + eps
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    S = Q @ K.T                                   # (T, N) 余弦
    srt = np.sort(S, axis=1)
    gap = srt[:, -1] - srt[:, -2]
    hits = []
    for beta in BETAS:
        Z = beta * (S - S.max(axis=1, keepdims=True))
        np.exp(Z, out=Z)
        a = Z / Z.sum(axis=1, keepdims=True)      # (T, N)
        R = a @ K                                  # (T, D) 检索模式
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        pred = (R @ K.T).argmax(axis=1)
        hits.append(int((pred == idx).sum()))
    return {"betas": BETAS, "hits": hits, "n": T,
            "gap_mean": float(gap.mean()), "gap_p10": float(np.quantile(gap, 0.1)),
            "s_true_mean": float(np.mean(S[np.arange(T), idx]))}


def sparse_arm(N, rng):
    T = trials_for(N)
    codes = np.zeros((N, D_SP), dtype=np.float32)
    for i in range(N):
        codes[i, rng.integers(0, D_SP, size=K_SP)] = 1.0
    idx = rng.integers(0, N, size=T)
    Q = codes[idx].copy()
    Q[rng.random(Q.shape) < FLIP] = 0.0
    hits = []
    gaps = []
    S_rows = []
    for t in range(T):
        s = codes @ Q[t]                           # overlap 计数（每 trial 一次，各 beta 复用）
        s /= K_SP
        o = np.argsort(s)[::-1]
        gaps.append(float(s[o[0]] - s[o[1]]))
        S_rows.append((s, float(s[o[0]]), int(idx[t])))
    for beta in BETAS:
        h = 0
        for s, smax, it in S_rows:
            z = beta * (s - smax)
            p = np.exp(z)
            p /= p.sum()
            r = p @ codes
            r /= (np.linalg.norm(r) + 1e-12)
            if int((codes @ r).argmax()) == it:
                h += 1
        hits.append(h)
    return {"betas": BETAS, "hits": hits, "n": T, "gap_mean": float(np.mean(gaps))}




def dense_cluster_arm(N, rng):
    """C：候选共享质心（真实记忆共享语义成分的还原）。16 簇，键=unit(簇心+0.35 噪声)。"""
    T = trials_for(N)
    NC = 16
    C = rng.standard_normal((NC, D)).astype(np.float32)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    cl = rng.integers(0, NC, size=N)
    K = C[cl] + 0.35 * rng.standard_normal((N, D)).astype(np.float32)
    K /= np.linalg.norm(K, axis=1, keepdims=True)
    idx = rng.integers(0, N, size=T)
    Q = K[idx] + (0.62 / math.sqrt(D)) * rng.standard_normal((T, D)).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    S = Q @ K.T
    srt = np.sort(S, axis=1)
    gap = srt[:, -1] - srt[:, -2]
    hits = []
    for beta in BETAS:
        Z = beta * (S - S.max(axis=1, keepdims=True))
        np.exp(Z, out=Z)
        a = Z / Z.sum(axis=1, keepdims=True)
        R = a @ K
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        pred = (R @ K.T).argmax(axis=1)
        hits.append(int((pred == idx).sum()))
    return {"betas": BETAS, "hits": hits, "n": T,
            "gap_mean": float(gap.mean()),
            "s_true_mean": float(np.mean(S[np.arange(T), idx]))}

def beta_star(d):
    for b, h in zip(d["betas"], d["hits"]):
        if h >= d["n"] / 2:
            return b
    return None


def main():
    nmax = int(sys.argv[sys.argv.index("--NMAX") + 1]) if "--NMAX" in sys.argv else 100000
    Ns = [n for n in [1000, 10000, 100000, 1000000] if n <= nmax]
    out = {"protocol": "vanilla Hopfield/attention soft retrieval, zero HCM code",
           "dense": [], "sparse": [], "dense_cluster": [], "fit": []}
    for N in Ns:
        rng = np.random.default_rng(RNG0 + N)
        d = dense_arm(N, rng)
        d["N"] = N; d["beta_star"] = beta_star(d)
        out["dense"].append(d)
        print("dense  N=%8d beta*=%s s_true=%.3f gap=%.4f hits=%s" %
              (N, d["beta_star"], d["s_true_mean"], d["gap_mean"], d["hits"]), flush=True)
    for N in Ns:
        rng = np.random.default_rng(RNG0 + 7 * N)
        d = sparse_arm(N, rng)
        d["N"] = N; d["beta_star"] = beta_star(d)
        out["sparse"].append(d)
        print("sparse N=%8d beta*=%s gap=%.4f hits=%s" %
              (N, d["beta_star"], d["gap_mean"], d["hits"]), flush=True)
    for N in Ns:
        rng = np.random.default_rng(RNG0 + 13 * N)
        d = dense_cluster_arm(N, rng)
        d["N"] = N; d["beta_star"] = beta_star(d)
        out["dense_cluster"].append(d)
        print("clus   N=%8d beta*=%s s_true=%.3f gap=%.4f hits=%s" %
              (N, d["beta_star"], d["s_true_mean"], d["gap_mean"], d["hits"]), flush=True)
    for arm in ("sparse", "dense_cluster"):
        pts = [(d["N"], d["beta_star"]) for d in out[arm] if d["beta_star"]]
        if len(pts) >= 3:
            lnN = np.log([p[0] for p in pts]); lb = np.log([p[1] for p in pts])
            for name, x in (("sqrt(lnN)", np.sqrt(lnN)), ("lnN", lnN), ("const", np.ones_like(lnN))):
                A = np.vstack([x, np.ones_like(x)]).T
                coef, *_ = np.linalg.lstsq(A, lb, rcond=None)
                pred = A @ coef
                r2 = 1 - ((lb - pred) ** 2).sum() / ((lb - lb.mean()) ** 2).sum()
                out["fit"].append({"arm": arm, "form": name, "r2": round(float(r2), 4),
                                   "slope": round(float(coef[0]), 4)})
    fp = os.path.join(RES, "exp25_collapse_external.json")
    json.dump(out, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", fp)
    print("FIT:", json.dumps(out["fit"], ensure_ascii=False))


if __name__ == "__main__":
    main()
