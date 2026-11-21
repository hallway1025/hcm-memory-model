import sys; sys.path.insert(0,'src')
import numpy as np, itertools
from hcm.ops import unit, kwta, softmax_beta
from hcm.data import overlapping_set
D_X,D_H=128,1024
def degrade(x,frac,seed):
    r=np.random.default_rng(seed); y=x.copy(); nz=np.where(np.abs(y)>1e-6)[0]
    y[r.choice(nz,size=max(1,int(len(nz)*frac)),replace=False)]=0.0; return unit(y)
def run(overlap,k,beta,L,noise,thr_on):
    X=overlapping_set(30,D_X,overlap=overlap,seed=11)
    rng=np.random.default_rng(0)
    W=np.linalg.qr(rng.standard_normal((D_H,D_X)))[0].astype(np.float32)
    thr=np.zeros(D_H,dtype=np.float32)
    if thr_on:
        for ep in range(6):
            for x in X: thr+=0.35*(kwta(np.maximum(W@x-thr,0.0),k)-k/D_H)
    enc=lambda x: unit(kwta(np.maximum(W@x-thr,0.0),k))
    M=np.stack([enc(x) for x in X],axis=1)
    ok=0
    for i in range(30):
        cue=degrade(X[i],noise,i) if noise>0 else X[i]
        s=np.maximum(W@cue-thr,0.0)   # 稠密 EC 前激活（不取 top-k）
        a=softmax_beta(M.T@s,beta)
        for _ in range(L):
            xr=unit(kwta(M@a,k)); a=softmax_beta(M.T@xr,beta)
        ok+=int(np.argmax(a)==i)
    return ok
print('%-8s %-3s %-4s %-6s | L0 L3 L10 (noise .2) | L0 L3 L10 (noise .35)'%('overlap','k','beta','thr'))
for overlap in (0.0,0.5,0.75,0.9):
  for k in (8,20):
    for thr_on in (False,True):
      r1=[run(overlap,k,40,L,0.2,thr_on) for L in (0,3,10)]
      r2=[run(overlap,k,40,L,0.35,thr_on) for L in (0,3,10)]
      print('%-8.2f %-3d %-4d %-6s | %2d %2d %2d | %2d %2d %2d'%(overlap,k,40,thr_on,r1[0],r1[1],r1[2],r2[0],r2[1],r2[2]))
