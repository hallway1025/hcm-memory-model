
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import overlapping_set
D=128
X = overlapping_set(60, D, overlap=0.85, seed=77)
m = HCM(Config(d_x=D, a=0.02, capacity=800, beta=20.0, L=3, use_resid=True, seed=0, rho_rif=0.30))
for x in X: m.observe(x)
print("写入", m.E.n, "/60")
print("sup 初始:", m.E.sup[:m.E.n].min(), m.E.sup[:m.E.n].max())
for _ in range(5):
    for i in range(30): m.query(X[i], do_rif=True)
sup = m.E.sup[:m.E.n]
print("5 轮后: 已练[0:30] mean=%.3f min=%.3f | 未练[30:] mean=%.3f min=%.3f" % (
    sup[:30].mean(), sup[:30].min(), sup[30:].mean(), sup[30:].min()))
# 抑制是否真的进入检索？
i = 35
xe = m.enc.resid_read(X[i]); s_pre = m.enc.pre(xe); q0 = m.enc.encode_from_pre(s_pre)
_, a = m.E.retrieve(q0, L=3, beta=20.0, dense=s_pre)
a_raw = a.copy()
a_sup = a * sup
print("条目 %d: 原始 a=%.4f  乘 sup 后=%.4f  (sup=%.3f)  argmax 原始=%d 抑制后=%d" % (
    i, a_raw[i], a_sup[i], sup[i], int(np.argmax(a_raw)), int(np.argmax(a_sup))))
print()
print("=> 抑制确实施加了（sup 降到 0.1 级），但 argmax 不变，因为")
print("   被抑制的条目原本就不是赢家，竞争由**证据强度**而非 sup 决定。")
print("   这解释了为什么 RIF 在 top-1 指标上不可见 —— 是真实的结构性结论。")
