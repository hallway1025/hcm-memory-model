import sys, time; sys.path.insert(0,'src')
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import overlapping_set

D=128
XA=overlapping_set(20,D,overlap=0.6,seed=21)
def acc(m,X,n):
    ok=0
    for i,x in enumerate(X):
        y=unit(np.asarray(m.sem.forward(x),dtype=np.float32)); ok+=int(int(np.argmax(m.E.M[:,:n].T@y))==i)
    return ok
for dz in (64,256):
  for eta in (0.05,0.15):
    for ep in (3,10):
      m=HCM(Config(d_x=D,a=0.02,capacity=200,beta=20.0,L=3,use_resid=True,seed=0,d_z=dz))
      m.cons.eta=eta; m.cons.epochs=ep
      for x in XA: m.observe(x)
      n=m.E.n; t0=time.time(); curve=[]
      for k in range(6):
          m.consolidate(cycles=10); curve.append(acc(m,XA,n))
      print('dz=%-3d eta=%.2f ep=%2d 曲线%s  loss=%.3f  %.0fs' % (dz,eta,ep,curve,
            (m.cons.history[-1] if m.cons.history else -1), time.time()-t0))
