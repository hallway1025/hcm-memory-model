import sys; sys.path.insert(0,'src')
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import overlapping_set
D=128
XA=overlapping_set(20,D,overlap=0.6,seed=21)
XB=overlapping_set(20,D,overlap=0.6,seed=22)
def degrade(x,frac,seed):
    r=np.random.default_rng(seed); y=x.copy(); nz=np.where(np.abs(y)>1e-6)[0]
    y[r.choice(nz,size=max(1,int(len(nz)*frac)),replace=False)]=0.0; return unit(y)
m=HCM(Config(d_x=D,a=0.02,capacity=200,beta=20.0,L=3,use_resid=True,seed=0))
for x in XA: m.observe(x)
nA=m.E.n
for x in XB: m.observe(x)
n=m.E.n
print('bank',len(m.enc.MU),'n',n)
b0=sum(int(m.query(degrade(XB[i],0.25,i))['idx']==nA+i) for i in range(20))
print('B recall BEFORE any A query:', b0, '/20')
a0=sum(int(m.query(degrade(XA[i],0.25,i))['idx']==i) for i in range(20))
print('A recall:', a0, '/20')
b1=sum(int(m.query(degrade(XB[i],0.25,i))['idx']==nA+i) for i in range(20))
print('B recall AFTER A queries:', b1, '/20')
sup=m.E.sup[:n].copy()
print('sup stats: min %.3f mean %.3f  n_suppressed %d' % (sup.min(), sup.mean(), int((sup<0.999).sum())))
print('M col norms A %.3f  B %.3f' % (np.linalg.norm(m.E.M[:,:nA],axis=0).mean(), np.linalg.norm(m.E.M[:,nA:n],axis=0).mean()))
