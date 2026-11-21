import sys; sys.path.insert(0,'src')
import numpy as np, time
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import similar_pair

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
for a in (0.10,):
  for wd in (0.0,0.3,0.6,1.0):
    cfg = Config(d_x=128,rho=8,capacity=500,a=a,w_dense=wd,seed=0)
    m = HCM(cfg)
    for x in X: m.observe(x)
    out=[]
    for noise in (0.2,0.3,0.4):
        ok=0
        for i in range(20):
            r = m.query(degrade(X[i],noise,seed=i))
            ok += int(r['idx']==i)
        out.append('%d/20'%ok)
    print('a=%.2f w_dense=%.1f  noise20/30/40: %s' % (a, wd, ' '.join(out)))

# 容量曲线：稠密通路应随 N 增长先饱和
for N in (40, 100, 200, 400):
    Xn = sparse_patterns(N,128,0.15,seed=7)
    cfg = Config(d_x=128,rho=8,capacity=2000,a=0.10,seed=0)
    m = HCM(cfg)
    for x in Xn: m.observe(x)
    ok=0; ok0=0
    for i in range(min(40,N)):
        c = degrade(Xn[i],0.3,seed=i)
        ok += int(m.query(c)['idx']==i)
    m2 = HCM(Config(d_x=128,rho=8,capacity=2000,a=0.10,seed=0)); m2.cfg.w_dense=0.0; m2.cfg.w_dense=0.0
    for x in Xn: m2.observe(x)
    m2.cfg.w_dense=0.0
    for i in range(min(40,N)):
        ok0 += int(m2.query(degrade(Xn[i],0.3,seed=i))['idx']==i)
    print('N=%3d stored=%d  dual=%d/40  sparse_only=%d/40' % (N, m.E.n, ok, ok0))
