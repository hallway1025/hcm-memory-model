import sys, os, json; sys.path.insert(0,'src')
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import random_patterns
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT,'src'))
D=128
def degrade(x,frac,seed):
    r=np.random.default_rng(seed); y=x.copy(); nz=np.where(np.abs(y)>1e-6)[0]
    y[r.choice(nz,size=max(1,int(len(nz)*frac)),replace=False)]=0.0; return unit(y)
print('%-6s %-6s %-8s %-10s %-10s %-10s' % ('N','noise','sparseOnly','denseOnce','denseHop','written'))
for N in (20, 60, 150):
  for noise in (0.30, 0.45, 0.60):
    X=random_patterns(N,D,seed=71)
    m=HCM(Config(d_x=D,a=0.02,capacity=400,beta=20.0,L=3,use_resid=True,seed=0))
    for x in X: m.observe(x)
    Mn=m.E.M[:,:m.E.n]
    out={'sparse_only':0,'dense_once':0,'dense_hopfield':0}
    for i in range(min(N,50)):
      cue=degrade(X[i],noise,i)
      xe=m.enc.resid_read(cue); s_pre=m.enc.pre(xe); q0=m.enc.encode_from_pre(s_pre)
      for arm in out:
        if arm=='sparse_only': sc=Mn.T@q0
        elif arm=='dense_once': sc=Mn.T@s_pre
        else:
            z,a=m.E.retrieve(q0,L=3,beta=20.0,dense=s_pre); sc=a*m.E.sup[:m.E.n]
        out[arm]+=int(int(np.argmax(sc))==i)
    k=min(N,50)
    print('%-6d %-6.2f %-8s %-10s %-10s %d' % (N,noise,'%d/%d'%(out['sparse_only'],k),
          '%d/%d'%(out['dense_once'],k),'%d/%d'%(out['dense_hopfield'],k),m.E.n))