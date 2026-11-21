import sys; sys.path.insert(0,'src')
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import overlapping_set
D=128; NA=20; NB=20
def degrade(x,frac,seed):
    r=np.random.default_rng(seed); y=x.copy(); nz=np.where(np.abs(y)>1e-6)[0]
    y[r.choice(nz,size=max(1,int(len(nz)*frac)),replace=False)]=0.0; return unit(y)
def cortical(m,x,n):
    y=unit(np.asarray(m.sem.forward(x),dtype=np.float32)); return int(np.argmax(m.E.M[:,:n].T@y))
XA=overlapping_set(NA,D,overlap=0.6,seed=21)
XB=overlapping_set(NB,D,overlap=0.6,seed=22)
m=HCM(Config(d_x=D,a=0.02,capacity=200,beta=20.0,L=3,use_resid=True,seed=0))
for x in XA: m.observe(x)
nA=m.E.n
for x in XB: m.observe(x)
nAB=m.E.n
print('bank',len(m.enc.MU),'n',nAB)
print('B recall pre-cortex', sum(int(m.query(degrade(XB[i],0.25,i))['idx']==nA+i) for i in range(20)))
_=[cortical(m,degrade(x,0.25,i),nAB) for i,x in enumerate(XA)]
print('B recall post cortex-calls', sum(int(m.query(degrade(XB[i],0.25,i))['idx']==nA+i) for i in range(20)))
print('A recall post cortex-calls', sum(int(m.query(degrade(XA[i],0.25,i))['idx']==i) for i in range(20)))
print('sup<1 count', int((m.E.sup[:nAB]<0.999).sum()))
