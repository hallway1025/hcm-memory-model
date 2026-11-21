
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.data import random_patterns

# 检查写入码的实际稀疏结构：能否用 (indices, values) 存储
m = HCM(Config(d_x=128, a=0.02, capacity=100, beta=20.0, L=3, use_resid=True, seed=0))
for x in random_patterns(50, 128, seed=1): m.observe(x)
H = m.E.M[:, :m.E.n]
nz_per_col = (H != 0).sum(axis=0)
print("M 每列非零数: min=%d max=%d mean=%.1f  (k=%d)" % (nz_per_col.min(), nz_per_col.max(), nz_per_col.mean(), m.cfg.k))
print("M 的 dtype:", H.dtype, " 大小 %.2f MB (dense)" % (H.nbytes/1e6))
print("若存 (k 个索引 int32 + k 个值 float32): %.3f MB" % (m.cfg.k*8*m.E.n/1e6))
print()
D = m.E.D[:, :m.E.n]
d_nz = (D != 0).sum(axis=0)
print("D 每列非零数: min=%d max=%d mean=%.1f  -> D 是**全稠密**的（来自 raw 投射）" % (d_nz.min(), d_nz.max(), d_nz.mean()))
print("D 大小 %.2f MB, 若也必须稠密存储则 L3 会是 5.2 TB" % (D.nbytes/1e6))
print()
print("=> 结论：M 天然稀疏（k=20/1024），D 全稠密。")
print("   要让 L3 可行，必须解决 D —— 要么它也稀疏化，要么改成按需重算（存原始输入 x，用时重投射）。")
print("   P 已经存了原始输入 x（%.1f KB/条），所以 D 完全可以按需重算，不必常驻。" % (m.E.P.nbytes/m.E.n/1024))
