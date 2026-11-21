
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit, softmax_beta
from hcm.data import random_patterns
D=128
m = HCM(Config(d_x=D, a=0.02, capacity=800, beta=20.0, L=3, use_resid=True, seed=0, rho_rif=0.30))
X = random_patterns(40, D, seed=51)
for x in X: m.observe(x)
print("写入", m.E.n)
prac = list(range(20))
for _ in range(5):
    for i in prac: m.query(X[i], do_rif=True)
sup = m.E.sup[:m.E.n]
print("sup: 已练 mean=%.3f min=%.3f | 未练 mean=%.3f min=%.3f" % (
    sup[:20].mean(), sup[:20].min(), sup[20:].mean(), sup[20:].min()))
print()
print("检索路径中的 sup 是否生效？逐层看：")
i = 25
xe = m.enc.resid_read(X[i]); s_pre = m.enc.pre(xe); q0 = m.enc.encode_from_pre(s_pre)
z, a_raw = m.E.retrieve(q0, L=3, beta=20.0, dense=s_pre)
print("  E.retrieve 返回的 a_raw: argmax=%d  a_raw[%d]=%.4f  sum=%.4f" % (
    int(np.argmax(a_raw)), i, a_raw[i], a_raw.sum()))
print("  sup[i]=%.4f" % sup[i])
print("  a_raw * sup: argmax=%d  a[%d]=%.6f" % (int(np.argmax(a_raw*sup)), (a_raw*sup)[i]))
print("  -> retrieve() 内部**没有**乘 sup！只有 model.query 里乘。")
print()
print("  E.retrieve 源码里 sup 的处理：")
import inspect
src = inspect.getsource(m.E.retrieve)
for line in src.split("\n"):
    if "sup" in line or "return" in line:
        print("   ", line.strip())
