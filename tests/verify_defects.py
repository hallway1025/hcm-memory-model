import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import overlapping_set
D = 128
X = overlapping_set(30, D, overlap=0.90, seed=11)
# 直接量 input cosine，并检查 doc 公式 vs code 公式的 EMA 方向
cs = (X @ X.T)[np.triu_indices(30, 1)]
print('input pairwise cos: mean=%.4f  min=%.4f  max=%.4f' % (cs.mean(), cs.min(), cs.max()))

for resid in (False, True):
    m = HCM(Config(d_x=D, a=0.02, capacity=500, beta=20.0, L=0, use_resid=resid, seed=0))
    for x in X: m.observe(x)
    H = m.E.M[:, :m.E.n]
    cc = (H.T @ H)[np.triu_indices(m.E.n, 1)]
    ok = sum(int(m.query(X[i])['idx'] == i) for i in range(30))
    print('resid=%-5s written=%2d  code_cos=%.4f  recall=%d/30  |mu|=%.3f' % (
          resid, m.E.n, cc.mean(), ok, float(np.linalg.norm(m.enc.mu))))

print()
print('--- doc formula vs code formula for gamma=0.9 ---')
mu, x = 0.0, 1.0
for label, step in (('code: 0.9*mu + 0.1*x', lambda mu: 0.9*mu + 0.1*x),
                    ('doc : 0.1*mu + 0.9*x', lambda mu: 0.1*mu + 0.9*x)):
    v = mu
    for _ in range(5): v = step(v)
    print('  %s  -> 5 步后 mu=%.4f' % (label, v))
print('  => 文档公式 5 步吃掉 59% 新样本，与\'慢变期望\'的语义矛盾。')