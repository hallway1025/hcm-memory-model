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
for frac in (0.0,0.1,0.2,0.25,0.35):
    a=sum(int(m.query(degrade(XA[i],frac,i))['idx']==i) for i in range(20))
    b=sum(int(m.query(degrade(XB[i],frac,i))['idx']==nA+i) for i in range(20))
    print('degrade=%.2f  A=%2d/20  B=%2d/20  bank=%d' % (frac,a,b,len(m.enc.MU)))
print('--- 皮层学习 ---')
for cyc in (5,10,20,40):
    m.consolidate(cycles=cyc)
    ca=sum(int(np.argmax(m.E.M[:,:n].T@unit(m.sem.forward(XA[i])))==i) for i in range(20))
    cb=sum(int(np.argmax(m.E.M[:,:n].T@unit(m.sem.forward(XB[i])))==nA+i) for i in range(20))
    print('总周期=%-3d cortex A=%2d/20 B=%2d/20  rehears=%d loss=%.4f' % (cyc,ca,cb,len(m.cons.rehearsal),
          (m.cons.history[-1] if m.cons.history else -1)))
