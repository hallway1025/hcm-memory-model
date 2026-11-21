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
m = HCM(Config(d_x=128,rho=8,capacity=500,a=0.10,seed=0))
for x in X: m.observe(x)
r = m.query(degrade(X[0],0.3,seed=0))
print({k: (round(v,3) if isinstance(v,float) else v) for k,v in r.items() if k!='pred'})
print('n stored', m.E.n)

# 直接看证据
cue = degrade(X[0],0.3,seed=0)
xn,u = m._assemble(cue); gq = unit(m.enc.forward_pre(u)); q0 = m.enc.encode(u)
ev = 0.6*(m.E.D[:, :m.E.n].T@gq) + 0.4*(m.E.M[:, :m.E.n].T@q0)
print('argmax ev', int(np.argmax(ev)), ' ev_top3', np.round(np.sort(ev)[-3:],3))
print('dense argmax', int(np.argmax(m.E.D[:, :m.E.n].T@gq)))
