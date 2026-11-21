import sys; sys.path.insert(0,'src')
import numpy as np
from hcm.ops import unit, kwta, softmax_beta
from hcm.data import overlapping_set
def degrade(x,frac,seed):
    r=np.random.default_rng(seed); y=x.copy(); nz=np.where(np.abs(y)>1e-6)[0]
    y[r.choice(nz,size=max(1,int(len(nz)*frac)),replace=False)]=0.0; return unit(y)
for overlap in (0.75,0.9):
  X=overlapping_set(30,128,overlap=overlap,seed=11)
  rng=np.random.default_rng(0); W=np.linalg.qr(rng.standard_normal((1024,128)))[0].astype(np.float32)
  mu=np.zeros(128,dtype=np.float32); Hs=[]
  for x in X:
      mu[:]=(0.9*mu+0.1*x).astype(np.float32); Hs.append(unit(kwta(np.maximum(W@unit(x-mu),0.0),20)))
  M=np.stack(Hs,axis=1)
  for beta in (10,20,40):
    for L in (0,2,5,10):
      for init in ('cue','dense'):
        ok20=ok35=0
        for i in range(30):
          for frac,key in ((0.2,'a'),(0.35,'b')):
            c=unit(degrade(X[i],frac,i)-mu)
            s=np.maximum(W@c,0.0)
            z = unit(kwta(s,20)) if init=='cue' else unit(s)
            for it in range(L):
                z=unit(M@softmax_beta(M.T@z,beta))
            hit=int(np.argmax(M.T@z)==i)
            if key=='a': ok20+=hit
            else: ok35+=hit
        print('ov=%.2f beta=%2d L=%2d init=%-5s n20=%2d/30 n35=%2d/30'%(overlap,beta,L,init,ok20,ok35))
