import sys; sys.path.insert(0,'src')
import numpy as np, itertools
from hcm.ops import unit, kwta, softmax_beta
from hcm.data import overlapping_set
D_X,D_H=128,1024
def degrade(x,frac,seed):
    r=np.random.default_rng(seed); y=x.copy(); nz=np.where(np.abs(y)>1e-6)[0]
    y[r.choice(nz,size=max(1,int(len(nz)*frac)),replace=False)]=0.0; return unit(y)
def run(overlap,k,beta,L,noise):
    X=overlapping_set(30,D_X,overlap=overlap,seed=11)
    rng=np.random.default_rng(0)
    W=np.linalg.qr(rng.standard_normal((D_H,D_X)))[0].astype(np.float32)
    Ws=np.linalg.qr(rng.standard_normal((D_H,D_X)))[0].astype(np.float32)
    thr=np.zeros(D_H,dtype=np.float32)
    for ep in range(6):
        for x in X: thr+=0.35*(kwta(np.maximum(W@x-thr,0.0),k)-k/D_H)
    enc=lambda x: unit(kwta(np.maximum(W@x-thr,0.0),k))
    M=np.stack([enc(x) for x in X],axis=1); G=np.stack([unit(Ws@x) for x in X])
    ok=0
    for i in range(30):
        cue=degrade(X[i],noise,i) if noise>0 else X[i]
        g=unit(Ws@cue)
        a=softmax_beta(M.T@g,beta)
        for _ in range(L):
            xr=unit(kwta(M@a,k)); a=softmax_beta(M.T@xr,beta)
        ok+=int(np.argmax(a)==i)
    return ok
print('%-8s %-4s %-5s %-8s %-8s %-8s'%('overlap','k','beta','L=0','L=3','L=10'))
for overlap in (0.75,0.9):
  for k in (8,20):
    for beta in (25,60):
      row=['%d/30'%run(overlap,k,beta,L,n) for L in (0,3,10) for n in (0.2,)]
      r2=['%d'%run(overlap,k,beta,L,0.35) for L in (0,3,10)]
      print('%-8.2f %-4d %-5d n20: %-8s %-8s %-8s | n35: %s %s %s'%(overlap,k,beta,row[0],row[1],row[2],r2[0],r2[1],r2[2]))
