import sys, itertools; sys.path.insert(0,'src')
import numpy as np
from hcm.ops import unit, kwta, softmax_beta
from hcm.data import overlapping_set
def degrade(x,frac,seed):
    r=np.random.default_rng(seed); y=x.copy(); nz=np.where(np.abs(y)>1e-6)[0]
    y[r.choice(nz,size=max(1,int(len(nz)*frac)),replace=False)]=0.0; return unit(y)
def mc(H):
    return float(np.mean([float(H[i]@H[j]) for i,j in itertools.combinations(range(len(H)),2)][:500]))
for overlap in (0.5,0.75,0.9):
  X=overlapping_set(30,128,overlap=overlap,seed=11)
  for mode in ('raw','resid'):
    rng=np.random.default_rng(0)
    W=np.linalg.qr(rng.standard_normal((1024,128)))[0].astype(np.float32)
    mu=np.zeros(128,dtype=np.float32)
    def eff(x): return unit(x) if mode=='raw' else unit(x-mu)
    H=[]
    for x in X:
        mu[:] = (0.9*mu + 0.1*x).astype(np.float32)
        H.append(unit(kwta(np.maximum(W@eff(x),0.0),20)))
    M=np.stack(H,axis=1)
    ok0=ok2=0
    for i in range(30):
        for frac,acc in ((0.0,'a'),(0.2,'b'),(0.35,'c')):
            pass
        h0=unit(kwta(np.maximum(W@eff(X[i]),0.0),20))
        h2=unit(kwta(np.maximum(W@eff(degrade(X[i],0.2,i)),0.0),20))
        ok0+=int(np.argmax(M.T@h0)==i); ok2+=int(np.argmax(M.T@h2)==i)
    print('ov=%.2f %-6s code_cos=%.3f exact=%2d/30 noise20=%2d/30' % (overlap,mode,mc(H),ok0,ok2))
