'''E3c：推到容量边界。固定 d_h=1024,k=20，N 从 2000 到 16000，量 load=N*k/d_h。'''
import sys, json, time; sys.path.insert(0,'src')
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit

D=128
def pats(N,d,seed=5):
    r=np.random.default_rng(seed)
    R=r.standard_normal((N,d))
    return np.asarray([unit(x) for x in R],dtype=np.float32)
def degrade(x,frac,seed):
    r=np.random.default_rng(seed); y=x.copy(); nz=np.where(np.abs(y)>1e-6)[0]
    y[r.choice(nz,size=max(1,int(len(nz)*frac)),replace=False)]=0.0; return unit(y)

rows=[]
for N in (2000,4000,8000,16000):
  X=pats(N,D)
  for resid in (False,True):
    t0=time.time()
    m=HCM(Config(d_x=D,a=0.02,capacity=N+10,beta=20.0,L=3,use_resid=resid,seed=0))
    for x in X: m.observe(x)
    H=m.E.M[:,:m.E.n]
    cos=float(np.mean((H[:,:2000].T@H[:,:2000])[np.triu_indices(2000,1)]))
    probe=list(range(0,N,N//20))[:20]
    e=sum(int(m.query(X[i])['idx']==i) for i in probe)
    dg=sum(int(m.query(degrade(X[i],0.25,i))['idx']==i) for i in probe)
    dt=time.time()-t0
    rows.append(dict(N=N,resid=resid,exact=e,degraded=dg,n=len(probe),code_cos=cos,sec=round(dt,1)))
    print('N=%-6d load=%5.1f resid=%-5s exact=%2d/%2d deg=%2d/%2d cos=%.3f %.1fs' % (N,N*20/1024,resid,e,len(probe),dg,len(probe),cos,dt))
json.dump(rows, open('results/exp2c_capacity_edge.json','w'), indent=1)
