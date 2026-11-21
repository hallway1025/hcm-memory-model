
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit, softmax_beta
from hcm.data import overlapping_set

D = 128
# 验证：检索时用「重算 D」代替「读取常驻 D」，结果是否逐位相同
X = overlapping_set(30, D, overlap=0.90, seed=11)
m = HCM(Config(d_x=D, a=0.02, capacity=500, beta=20.0, L=3, use_resid=True, seed=0))
for x in X: m.observe(x)

Mn = m.E.M[:, :m.E.n]
Dn = m.E.D[:, :m.E.n]

# 路径 A：读常驻 D（当前实现）
# 路径 B：用 P（原始输入）当场重算 D
D_recomputed = np.stack([m.enc.raw(m.enc.resid_read(m.E.P[:, i])) for i in range(m.E.n)], axis=1)

print("D 重算 vs 常驻：最大绝对差 = %.3e" % float(np.abs(Dn - D_recomputed).max()))
print()
print("两种情况下的检索结果对比（30 条精确线索）:")
for name, Dmat in (("常驻 D", Dn), ("重算 D", D_recomputed)):
    ok = 0
    for i in range(30):
        xe = m.enc.resid_read(X[i]); s_pre = m.enc.pre(xe); q0 = m.enc.encode_from_pre(s_pre)
        ev = Dmat.T @ s_pre
        a = softmax_beta(ev, 20.0)
        z = unit(Mn @ a)
        a2 = softmax_beta(Mn.T @ z, 20.0)
        ok += int(int(np.argmax(a2)) == i)
    print("  %-8s recall = %d/30" % (name, ok))

print()
print("内存对比（N=2e7, d_h=65536, k=1311）:")
print("  稠密 M + 稠密 D        : %.0f GB" % (2e7*65536*4*2/1e9))
print("  稀疏 M + 按需重算 D    : %.0f GB  (M 稀疏 %.0f GB + P 原始输入 %.0f GB)" % (
    (2e7*1311*8 + 2e7*8192*4)/1e9, 2e7*1311*8/1e9, 2e7*8192*4/1e9))
