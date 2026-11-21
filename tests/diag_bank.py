import sys; sys.path.insert(0,'src')
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import overlapping_set

D=128
XA=overlapping_set(20,D,overlap=0.6,seed=21)
XB=overlapping_set(20,D,overlap=0.6,seed=22)
m=HCM(Config(d_x=D,a=0.02,capacity=200,beta=20.0,L=3,use_resid=True,seed=0))
ds=[]
for i,x in enumerate(np.vstack([XA,XB])):
    mu=m.enc.mu.copy()
    d=float(np.linalg.norm(unit(x)-mu)); ds.append((i,d,m.enc.nu_ema,len(m.enc.MU)))
    m.observe(x)
for i,d,nm,nb in ds[:4]+ds[17:24]+ds[-3:]:
    print('i=%2d d=%.3f nu_ema=%.3f bank=%d' % (i,d,nm or -1,nb))
print('bank size', len(m.enc.MU), 'mus cos A vs B', float(unit(m.enc.MU[0])@unit(m.enc.MU[-1])))
