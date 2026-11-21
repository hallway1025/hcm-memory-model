import sys; sys.path.insert(0,'src')
import numpy as np, itertools
from hcm import HCM, Config
from hcm.ops import unit

def sparse_patterns(n, d, active=0.15, seed=0):
    rng = np.random.default_rng(seed)
    X = []
    for _ in range(n):
        x = np.zeros(d, dtype=np.float32)
        idx = rng.choice(d, size=int(d*active), replace=False)
        x[idx] = rng.standard_normal(len(idx)).astype(np.float32)
        X.append(unit(x))
    return np.asarray(X)

def drop_active(x, frac, seed):
    rng = np.random.default_rng(seed)
    y = x.copy()
    nz = np.where(y != 0)[0]
    drop = rng.choice(nz, size=max(1,int(len(nz)*frac)), replace=False)
    y[drop] = 0.0
    return unit(y)

X = sparse_patterns(60, 128, 0.15, seed=1)
print('%-6s %-8s %-8s %-8s %-8s %-8s' % ('a','norm','sep_low','cue_cos','top1','k'))
for norm in ('none','global'):
    for a in (0.02, 0.05, 0.10, 0.20):
        cfg = Config(d_x=128, rho=8, capacity=500, a=a, norm=norm, seed=0)
        m = HCM(cfg)
        for x in X: m.observe(x)
        H = m.E.M[:, :m.E.n]
        # 分离：同一模式的两个近似版本（drop 15% active）
        seps = []
        cues = []
        top1 = 0
        for i in range(20):
            c = drop_active(X[i], 0.15, seed=i)
            h1 = m.enc.encode(m._assemble(c)[1])
            seps.append(float(h1 @ H[:, i]))          # 越大越差（分离失败）
            c2 = drop_active(X[i], 0.30, seed=i+100)
            h2 = m.enc.encode(m._assemble(c2)[1])
            cues.append(float(h2 @ H[:, i]))
            top1 += int(np.argmax(H.T @ h2) == i)
        print('%-6.2f %-8s %-8.3f %-8.3f %-8s %d' % (a, norm, np.mean(seps), np.mean(cues), '%d/20'%top1, cfg.k))
