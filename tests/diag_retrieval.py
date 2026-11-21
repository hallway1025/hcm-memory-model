import sys; sys.path.insert(0,'src')
import numpy as np, itertools
from hcm import HCM, Config
from hcm.data import random_patterns, corrupt, overlapping_set
from hcm.ops import unit

cfg = Config(d_x=128, rho=8, capacity=500, seed=0)
m = HCM(cfg)
X = random_patterns(40, cfg.d_x, seed=1)
for x in X: m.observe(x)

H = m.E.M[:, :m.E.n]
ranks, top1ok, finalok = [], 0, 0
for i, x in enumerate(X[:20]):
    cue = corrupt(x, 0.3, seed=i)
    xn, u = m._assemble(cue)
    q0 = m.enc.encode(u)
    sc = H.T @ q0
    ranks.append(int((np.argsort(-sc) == i).nonzero()[0][0]))
    top1ok += int(np.argmax(sc) == i)
    qs, a = m.E.retrieve(q0, L=6)
    finalok += int(np.argmax(H.T @ qs) == i)
print('cue cos(true) mean=%.3f' % np.mean([H[:,i] @ m.enc.encode(m._assemble(corrupt(X[i],0.3,seed=i))[1]) for i in range(20)]))
print('rank of true item: median=%d mean=%.1f' % (int(np.median(ranks)), np.mean(ranks)))
print('top1 correct: %d/20 ; after 6 iters: %d/20' % (top1ok, finalok))
print('inter-item cos (stored codes): mean=%.3f max=%.3f' % (np.mean(H.T@H - np.eye(40)), (H.T@H - np.eye(40)).max()))

# 分离性对照
for ov in (0.8, 0.9):
    sim = overlapping_set(8, cfg.d_x, overlap=ov, seed=3)
    hs = [m.enc.encode(m._assemble(x)[1]) for x in sim]
    cs = [float(hs[i]@hs[j]) for i,j in itertools.combinations(range(8),2)]
    print('overlap=%.1f -> code cos mean=%.3f' % (ov, np.mean(cs)))
