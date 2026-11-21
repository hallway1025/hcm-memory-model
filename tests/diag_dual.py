import sys; sys.path.insert(0,'src')
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit

def sparse_patterns(n,d,active=0.15,seed=0):
    rng=np.random.default_rng(seed); X=[]
    for _ in range(n):
        x=np.zeros(d,dtype=np.float32); idx=rng.choice(d,size=int(d*active),replace=False)
        x[idx]=rng.standard_normal(len(idx)).astype(np.float32); X.append(unit(x))
    return np.asarray(X)

def degrade(x,frac,seed):
    rng=np.random.default_rng(seed); y=x.copy(); nz=np.where(y!=0)[0]
    y[rng.choice(nz,size=max(1,int(len(nz)*frac)),replace=False)]=0.0
    return unit(y)

X = sparse_patterns(60,128,0.15,seed=1)
cfg = Config(d_x=128, rho=8, capacity=500, a=0.10, seed=0)
m = HCM(cfg)
for x in X: m.observe(x)
H = m.E.M[:, :m.E.n]
G = np.stack([unit(m.enc.forward_pre(m._assemble(x)[1])) for x in X], axis=1)  # 稠密投射（EC 直通通路）

for noise in (0.2, 0.3, 0.4):
    sp=dn=0; ranks=[]
    for i in range(20):
        cue = degrade(X[i], noise, seed=i)
        xq,u = m._assemble(cue)
        hq = m.enc.encode(u)
        gq = unit(m.enc.forward_pre(u))
        cs = H.T @ hq; cd = G.T @ gq
        sp += int(np.argmax(cs)==i); dn += int(np.argmax(cd)==i)
        ranks.append(int((np.argsort(-cs)==i).nonzero()[0][0]))
    print('noise=%.1f  sparse_top1=%d/20  dense_top1=%d/20  sparse_rank_med=%d' % (noise, sp, dn, int(np.median(ranks))))
print('dense cos(cue,true) mean=%.3f' % np.mean([float(unit(m.enc.forward_pre(m._assemble(degrade(X[i],0.3,seed=i))[1])) @ G[:,i]) for i in range(20)]))
