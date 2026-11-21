import sys, itertools; sys.path.insert(0,'src')
import numpy as np
from hcm.ops import unit, kwta, softmax_beta
from hcm.data import overlapping_set
def degrade(x,frac,seed):
    r=np.random.default_rng(seed); y=x.copy(); nz=np.where(np.abs(y)>1e-6)[0]
    y[r.choice(nz,size=max(1,int(len(nz)*frac)),replace=False)]=0.0; return unit(y)
for overlap in (0.75,0.9):
  X=overlapping_set(30,128,overlap=overlap,seed=11)
  rng=np.random.default_rng(0)
  W=np.linalg.qr(rng.standard_normal((1024,128)))[0].astype(np.float32)
  mu=np.zeros(128,dtype=np.float32); Hs=[]
  for x in X:
      mu[:]=(0.9*mu+0.1*x).astype(np.float32)
      Hs.append(unit(kwta(np.maximum(W@unit(x-mu),0.0),20)))
  M=np.stack(Hs,axis=1)
  for beta in (15,30,60):
    for L in (0,1,3,8):
      for ann in (False,True):
        ok20=ok35=0
        for i in range(30):
          for frac,key in ((0.2,'a'),(0.35,'b')):
            cue=unit(degrade(X[i],frac,i)-mu)
            h=unit(kwta(np.maximum(W@cue,0.0),20))
            b=beta if not ann else beta/3.0
            a=softmax_beta(M.T@h,b)
            for it in range(L):
                xr=unit(kwta(M@a,20)); a=softmax_beta(M.T@xr,b)
                if ann: b=min(beta, b*1.8)
            hit=int(np.argmax(a)==i)
            if key=='a': ok20+=hit
            else: ok35+=hit
        print('ov=%.2f beta=%2d L=%d ann=%-5s noise20=%2d/30 noise35=%2d/30'%(overlap,beta,L,ann,ok20,ok35))
