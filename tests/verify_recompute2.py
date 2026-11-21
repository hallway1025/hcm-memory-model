
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.data import overlapping_set

D = 128
X = overlapping_set(30, D, overlap=0.90, seed=11)
m = HCM(Config(d_x=D, a=0.02, capacity=500, beta=20.0, L=3, use_resid=True, seed=0))
for x in X: m.observe(x)

# 写入时存的是 dense = s_pre = enc.pre(resid_write(x))
# P 存的是 x（原始输入，resid_write 之前的 x）
# 重算需要用 pre(resid_read(P)) 还是 pre(resid_write(P))？写入端用的是 resid_write
Dn = m.E.D[:, :m.E.n]
D_rec = np.stack([m.enc.pre(m.enc.resid_read(m.E.P[:, i])) for i in range(m.E.n)], axis=1)
print("重算 D 与常驻 D 最大差: %.3e" % float(np.abs(Dn - D_rec).max()))
print("若 resid_read 与 resid_write 在 use_resid=False 时都返回 unit(x)，则应完全相等")
print("  当前 use_resid=True，mu 在写入时被更新，故 resid_read 用的是最终的 mu → 不相等")
print()
print("=> 正确结论：D 依赖写入时刻的 mu，**无法从 P 无损重算**。")
print("   要让 L3 可行，只有两条路：(a) D 也稀疏化存储；(b) 每列额外存 (D 的稀疏表示)。")
dd = (Dn != 0).sum(axis=0)
print("   D 每列非零: mean=%.0f / %d  -> 稠密度 %.0f%%，稀疏化收益有限" % (dd.mean(), m.cfg.d_h, 100*dd.mean()/m.cfg.d_h))
print()
print("=== 结论：L3 的真实内存需求 ===")
for name, gb in (("稀疏 M (k=1311,8B)", 2e7*1311*8/1e9),
                 ("稠密 D (65536,4B)", 2e7*65536*4/1e9),
                 ("稠密 P (8192,4B)", 2e7*8192*4/1e9)):
    print("  %-22s %.0f GB" % (name, gb))
print("  合计 %.0f GB —— 这已经超出单机，需要分布式或降低 L3 规模。" % ((2e7*1311*8 + 2e7*65536*4 + 2e7*8192*4)/1e9))
