# -*- coding: utf-8 -*-
"""bench_mv.py -- 验证 B2 耗时假设：M[:, :n] 非连续切片是否拖慢 matvec。

若 strided matvec 每次都隐式拷贝整块 M（d_h×n），则 query 里 11 次 matvec
= 11 次全量拷贝 → O(N) 拷贝成本 × 11。
对比 contiguous vs strided 的时间，即可判断是否值得做 bench 局部优化。
"""
import time
import numpy as np

d_h = 2008
n = 5000
cap = n + 10
rng = np.random.default_rng(0)
M = np.zeros((d_h, cap), dtype=np.float32)
M[:, :n] = rng.standard_normal((d_h, n)).astype(np.float32)
M[:, :n] /= np.linalg.norm(M[:, :n], axis=0, keepdims=True)
z = rng.standard_normal(d_h).astype(np.float32)
z /= np.linalg.norm(z)

def timeit(fn, reps=5):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps

Mn_strided = M[:, :n]
Mn_contig = np.ascontiguousarray(Mn_strided)

print("M shape", M.shape, "d_h", d_h, "n", n)
print("strided Mv  : %.4f s" % timeit(lambda: Mn_strided.T @ z))
print("contig  Mv  : %.4f s" % timeit(lambda: Mn_contig.T @ z))
print("contig  MM  : %.4f s" % timeit(lambda: Mn_contig @ z))

# 拷贝成本
t0 = time.perf_counter()
for _ in range(5):
    _ = np.ascontiguousarray(Mn_strided)
print("ascontiguousarray copy: %.4f s" % ((time.perf_counter() - t0) / 5))

# 数值等价性
a = Mn_strided.T @ z
b = Mn_contig.T @ z
print("max |diff| strided vs contig: %.3e" % float(np.max(np.abs(a - b))))
print("identical:", bool(np.array_equal(a, b)))

import numpy.core._multiarray_umath as _mu  # noqa
try:
    from numpy import show_config
except Exception:
    pass
print("numpy", np.__version__)
