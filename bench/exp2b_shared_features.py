'''E3b：共享特征容量（干扰受限，而非条数受限）。池中随机组合特征 → 现实记忆。'''
import sys, json; sys.path.insert(0,'src')
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit

D=128; POOL=40
def build(N,seed=7,n_feat=2):
    r=np.random.default_rng(seed)
    F=np.asarray([unit(r.standard_normal(D).astype(np.float32)) for _ in range(POOL)],dtype=np.float32)
    idx=[r.choice(POOL,size=n_feat,replace=False) for _ in range(N)]
    X=np.asarray([unit(F[i].sum(0)+0.6*r.standard_normal(D).astype(np.float32)) for i in idx],dtype=np.float32)
    return X
def degrade(x,frac,seed):
    r=np.random.default_rng(seed); y=x.copy(); nz=np.where(np.abs(y)>1e-6)[0]
    y[r.choice(nz,size=max(1,int(len(nz)*frac)),replace=False)]=0.0; return unit(y)

rows=[]
for N in (30,60,120,300,600,1200):
  X=build(N)
  for resid in (False,True):
    m=HCM(Config(d_x=D,a=0.02,capacity=max(N,1200),beta=20.0,L=3,use_resid=resid,seed=0))
    for x in X: m.observe(x)
    H=m.E.M[:,:m.E.n]
    cos=float(np.mean((H.T@H)[np.triu_indices(m.E.n,1)]))
    probe=list(range(0,N,max(1,N//30)))[:30]
    e=sum(int(m.query(X[i])['idx']==i) for i in probe)
    d=sum(int(m.query(degrade(X[i],0.25,i))['idx']==i) for i in probe)
    rows.append(dict(N=N,resid=resid,exact=e,degraded=d,n=len(probe),code_cos=cos))
    print('N=%-5d resid=%-5s exact=%2d/%2d degraded=%2d/%2d code_cos=%.3f' % (N,resid,e,len(probe),d,len(probe),cos))
json.dump(rows, open('results/exp2b_shared_features.json','w'), indent=1)
