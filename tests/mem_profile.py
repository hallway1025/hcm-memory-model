
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.data import random_patterns

print("=== 内存占用拆解：谁是瓶颈？ ===")
N = 4000
m = HCM(Config(d_x=128, a=0.02, capacity=N, beta=20.0, L=3, use_resid=True, seed=0))
for x in random_patterns(N, 128, seed=5): m.observe(x)

d_h = m.cfg.d_h
per_col = {
    "M (稀疏码, 名义 d_h 列)": m.E.M.nbytes,
    "D (稠密证据, d_h 列)": m.E.D.nbytes,
    "P (原始输入, d_x 列)": m.E.P.nbytes,
    "Ctx (情境)": m.E.Ctx.nbytes,
    "Tau (时间基)": m.E.Tau.nbytes,
}
tot = sum(per_col.values())
print("  N=%d, d_h=%d, k=%d" % (N, d_h, m.cfg.k))
for k, v in per_col.items():
    print("    %-26s %8.1f MB  (%.0f%%)" % (k, v/1e6, 100*v/tot))
print("    %-26s %8.1f MB" % ("合计", tot/1e6))
print()
print("  每条目平均 %.1f KB，其中 D 占 %.1f KB" % (tot/N/1024, m.E.D.nbytes/N/1024))
print()
print("=== 外推到 L3（N=2e7, d_h=65536）===")
for name, cols, dt in (("M 稠密存储", 65536, 4), ("D 稠密", 65536, 4), ("P 原始输入", 8192, 4)):
    gb = 2e7 * cols * dt / 1e9
    print("  %-12s %10.1f GB" % (name, gb))
print("  M 稀疏列存 (k=1311, 值+索引 8B)  %10.1f GB" % (2e7*1311*8/1e9))
print("  D 稀疏化（若可行）             %10.1f GB" % (2e7*1311*8/1e9))
