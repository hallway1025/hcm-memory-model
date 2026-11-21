"""合成任务生成器。刻意做成可解析推导的，便于检验模型是否学到机制而不是记住了数据。"""
from __future__ import annotations
import numpy as np
from .ops import unit


def random_patterns(n, d, seed=0, active=0.3):
    """稀疏二值型模式 -> 连续向量。"""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    if active < 1.0:
        mask = rng.random((n, d)) < active
        X = X * mask
    return np.asarray([unit(x) for x in X], dtype=np.float32)


def corrupt(x, frac=0.3, seed=0):
    rng = np.random.default_rng(seed)
    y = x.copy()
    idx = rng.choice(len(y), size=int(len(y) * frac), replace=False)
    y[idx] = 0.0
    return unit(y)


def similar_pair(base, flip_frac=0.15, seed=0, d=None):
    """把一个模式的 flip_frac 分量换成新值 —— 高相似输入对（测模式分离）。"""
    rng = np.random.default_rng(seed)
    y = unit(rng.standard_normal(base.shape).astype(np.float32))
    k = int(len(base) * flip_frac)
    idx = rng.choice(len(base), size=k, replace=False)
    z = base.copy()
    z[idx] = y[idx]
    return unit(z)


def overlapping_set(n, d, overlap=0.75, seed=0):
    """一组两两高度重叠的模式：共享 (1-overlap) 之外的随机成分。"""
    rng = np.random.default_rng(seed)
    shared = rng.standard_normal(d).astype(np.float32)
    out = []
    for _ in range(n):
        priv = rng.standard_normal(d).astype(np.float32)
        out.append(unit(overlap * shared + (1 - overlap) * priv))
    return np.asarray(out, dtype=np.float32)


def sequence(n_steps, d, seed=0):
    """平滑漂移序列：相邻状态高度相关（测时间情境与回放）。"""
    rng = np.random.default_rng(seed)
    x = unit(rng.standard_normal(d).astype(np.float32))
    out = [x]
    for _ in range(n_steps - 1):
        x = unit(0.85 * x + 0.15 * rng.standard_normal(d).astype(np.float32))
        out.append(x)
    return np.asarray(out, dtype=np.float32)


def two_tasks(n_per=32, d=128, seed=0, shift=1.0):
    """两个不重叠流形（CLS 灾难性遗忘用）。返回 (Xa, Xb)。"""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(d).astype(np.float32)
    A = np.asarray([unit(base + 0.6 * rng.standard_normal(d).astype(np.float32)) for _ in range(n_per)])
    base2 = unit(rng.standard_normal(d).astype(np.float32)) * shift
    B = np.asarray([unit(base2 + 0.6 * rng.standard_normal(d).astype(np.float32)) for _ in range(n_per)])
    return A.astype(np.float32), B.astype(np.float32)


def cluster_task(n, d, n_clusters=4, seed=0, spread=0.35):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, d)).astype(np.float32)
    y = rng.integers(0, n_clusters, size=n)
    X = np.asarray([unit(centers[yi] + spread * rng.standard_normal(d).astype(np.float32)) for yi in y],
                   dtype=np.float32)
    return X, y
