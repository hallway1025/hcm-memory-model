import sys, time
sys.path.insert(0, 'src')
import numpy as np
from hcm import HCM, Config, preset
from hcm.data import random_patterns, corrupt, overlapping_set

cfg = Config(d_x=128, rho=8, capacity=500, seed=0, use_resid=True)  # 锚点历史语义：残差开启的 L0 配置
m = HCM(cfg)
print('d_h=%d k=%d' % (cfg.d_h, cfg.k))

X = random_patterns(40, cfg.d_x, seed=1)
t0 = time.time()
for i, x in enumerate(X):
    m.observe(x, r=0.0)
print('observe 40: %.2fs  stats=%s' % (time.time()-t0, m.stats()))

ok = 0
for i, x in enumerate(X[:20]):
    r = m.query(corrupt(x, 0.3, seed=i), L=6)
    hit = r['idx'] == i
    ok += hit
print('completion@30%% noise: %d/20  actions=%s' % (ok, set(m.query(corrupt(X[0],0.3,seed=99))['action'] for _ in range(3))))

t0 = time.time()
m.consolidate(cycles=3)
print('consolidate 3 cycles: %.2fs' % (time.time()-t0))

sim = overlapping_set(6, cfg.d_x, overlap=0.8, seed=3)
m2 = HCM(Config(d_x=128, rho=8, capacity=500, seed=0, use_resid=True))
hs = [m2.enc.encode(m2.enc.resid_write(x)) for x in sim]
import itertools
cs = [float(hs[i] @ hs[j]) for i, j in itertools.combinations(range(len(hs)), 2)]
print('pattern separation: mean cos of overlapping inputs in code space = %.3f' % np.mean(cs))
