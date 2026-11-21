"""基础算子。所有实现与 docs/ARCHITECTURE-CORE.md 第 2 节严格一致。"""
from __future__ import annotations
import numpy as np

EPS = 1e-8


def unit(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / (n + EPS)


def kwta(x, k):
    """k-WTA：保留绝对值最大的 k 个分量，其余置 0。不回填、不缩放。"""
    x = np.asarray(x)
    flat = x.ravel()
    k = int(min(max(k, 1), flat.size))
    idx = np.argpartition(-np.abs(flat), k - 1)[:k]
    out = np.zeros_like(flat)
    out[idx] = flat[idx]
    return out.reshape(x.shape)


def relu_kwta(x, k):
    return kwta(np.maximum(x, 0.0), k)


def softmax_beta(z, beta):
    z = np.asarray(z, dtype=np.float64) * beta
    z = z - z.max(initial=-np.inf)
    e = np.exp(z)
    return e / (e.sum() + 1e-300)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))


def cosine_rows(M, v):
    """M: (n, d) 行向量, v: (d,) -> (n,) 余弦相似度（M 行已归一化时可省一次除法）"""
    num = M @ v
    den = np.linalg.norm(M, axis=1) * (np.linalg.norm(v) + EPS) + EPS
    return num / den


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def orthogonal_matrix(rows, cols, rng, scale=None):
    """近似正交随机投射：避免初始化带来的行列相关性（DG 的随机扩张投射）"""
    if rows >= cols:
        q, _ = np.linalg.qr(rng.standard_normal((rows, cols)))
        W = q[:, : cols]
    else:
        q, _ = np.linalg.qr(rng.standard_normal((cols, rows)))
        W = q[:, : rows].T
    s = scale if scale is not None else 1.0 / np.sqrt(max(cols, 1))
    return (W * s).astype(np.float32)


def fit_power_law(xs, ys):
    """对 y = a * t^(-alpha) 做 log-log 最小二乘，返回 (a, alpha, r2)。"""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.clip(np.asarray(ys, dtype=np.float64), 1e-6, None)
    lx, ly = np.log(xs), np.log(ys)
    A = np.vstack([np.ones_like(lx), -lx]).T
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    pred = A @ coef
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum()) + EPS
    r2 = 1.0 - ss_res / ss_tot
    return float(np.exp(coef[0])), float(coef[1]), r2


def fit_exponential(xs, ys):
    ys = np.clip(np.asarray(ys, dtype=np.float64), 1e-6, None)
    lx = np.asarray(xs, dtype=np.float64)
    A = np.vstack([np.ones_like(lx), -lx]).T
    coef, *_ = np.linalg.lstsq(A, np.log(ys), rcond=None)
    pred = A @ coef
    ly = np.log(ys)
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum()) + EPS
    return float(np.exp(coef[0])), float(coef[1]), 1.0 - ss_res / ss_tot
