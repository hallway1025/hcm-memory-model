
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit, softmax_beta
from hcm.data import overlapping_set
D=128
X = overlapping_set(30, D, overlap=0.90, seed=11)
m = HCM(Config(d_x=D, a=0.02, capacity=500, beta=20.0, L=3, use_resid=True, seed=0))
for x in X: m.observe(x)
Mn = m.E.M[:, :m.E.n]; Dn = m.E.D[:, :m.E.n]
print("库内 %d 条" % m.E.n)
# 逐层追踪一条查询
for i in (0, 3, 7):
    xe = m.enc.resid_read(X[i]); s_pre = m.enc.pre(xe); q0 = m.enc.encode_from_pre(s_pre)
    ev_dense = Dn.T @ s_pre
    ev_sparse = Mn.T @ q0
    _, a_hop = m.E.retrieve(q0, L=3, beta=20.0, dense=s_pre)
    a = a_hop/(a_hop.sum()+1e-12)
    print("\n查询 %d:" % i)
    print("  证据(稠密) argmax=%d  top3=%s" % (int(np.argmax(ev_dense)), np.round(np.sort(ev_dense)[-3:],3)))
    print("  证据(稀疏) argmax=%d  top3=%s" % (int(np.argmax(ev_sparse)), np.round(np.sort(ev_sparse)[-3:],3)))
    print("  Hopfield a argmax=%d  a_peak=%.3f  top3=%s" % (int(np.argmax(a)), a.max(), np.round(np.sort(a)[-3:],3)))
    shat = unit(m.sem.forward(X[i])); s_raw = Mn.T @ shat
    print("  语义原始 cos: 与真条目 %.4f, 全体 mean %.4f std %.4f" % (s_raw[i], s_raw.mean(), s_raw.std()))
    # 融合分
    s_z = (s_raw - s_raw.mean())/(s_raw.std()+1e-8); s_n = 1/(1+np.exp(-s_z))
    aa = a*m.E.sup[:m.E.n]; aa = aa/(aa.sum()+1e-12)
    f = 0.6*aa + 0.4*s_n
    print("  融合 f argmax=%d  f_top3=%s   语义项 argmax=%d" % (
        int(np.argmax(f)), np.round(np.sort(f)[-3:],3), int(np.argmax(s_n))))
