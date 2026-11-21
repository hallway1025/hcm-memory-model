'''E3：容量/干扰曲线。残差 on/off × 记忆条数 N，测精确与退化线索 top1。'''
import sys, json; sys.path.insert(0,'src')
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit

def pats(N,d,seed=5):
    r=np.random.default_rng(seed)
    return np.asarray([unit(r.standard_normal(d).astype(np.float32)) for _ in range(N)],dtype=np.float32)
def degrade(x,frac,seed):
    if frac<=0: return x
    r=np.random.default_rng(seed); y=x.copy(); nz=np.where(np.abs(y)>1e-6)[0]
    y[r.choice(nz,size=max(1,int(len(nz)*frac)),replace=False)]=0.0; return unit(y)

rows=[]
for N in (30,60,120,300,600,1200):
  X=pats(N,128)
  for resid in (False,True):
    m=HCM(Config(d_x=128,a=0.02,capacity=max(N,1200),beta=20.0,L=3,use_resid=resid,seed=0))
    for x in X: m.observe(x)
    H=m.E.M[:,:m.E.n]
    cos=float(np.mean((H.T@H)[np.triu_indices(m.E.n,1)])) if m.E.n>1 else 0.0
    probe=list(range(0,N,max(1,N//30)))[:30]
    e=sum(int(m.query(X[i])['idx']==i) for i in probe)
    d=sum(int(m.query(degrade(X[i],0.25,i))['idx']==i) for i in probe)
    rows.append(dict(N=N,resid=resid,exact=e,degraded=d,n_probe=len(probe),code_cos=cos))
    print('N=%-5d resid=%-5s exact=%2d/%2d degraded=%2d/%2d code_cos=%.3f' % (N,resid,e,len(probe),d,len(probe),cos))
json.dump(rows, open('results/exp2_capacity.json','w'), indent=1)
