import sys; sys.path.insert(0,'src')
import numpy as np, itertools
from hcm.ops import unit, kwta
from hcm.data import overlapping_set
X=overlapping_set(30,128,overlap=0.9,seed=11)
D_X,D_H,k=128,1024,20
rng=np.random.default_rng(0)
W=np.linalg.qr(rng.standard_normal((D_H,D_X)))[0].astype(np.float32)
Ws=np.linalg.qr(rng.standard_normal((D_H,D_X)))[1] if False else np.linalg.qr(rng.standard_normal((D_H,D_X)))[0].astype(np.float32)
thr=np.zeros(D_H,dtype=np.float32)
for ep in range(6):
    for x in X: thr+=0.35*(kwta(np.maximum(W@x-thr,0.0),k)-k/D_H)
def enc(x): return unit(kwta(np.maximum(W@x-thr,0.0),k))
H=np.stack([enc(x) for x in X]); D=np.stack([unit(Ws@x) for x in X])
print('code cos mean=%.3f  distinct_supports=%d/%d' % (float(np.mean([float(H[i]@H[j]) for i,j in itertools.combinations(range(30),2)])), len(set([tuple(np.where(h>0)[0]) for h in H])), len(H)))
def degrade(x,frac,seed):
    r=np.random.default_rng(seed); y=x.copy(); nz=np.where(np.abs(y)>1e-6)[0]
    y[r.choice(nz,size=max(1,int(len(nz)*frac)),replace=False)]=0.0; return unit(y)
def top1(cue, w_dense, rerank=0):
    g=unit(Ws@cue); q=enc(cue)
    ev=w_dense*(D@g)+(1-w_dense)*(H@q)
    if rerank:
        cand=np.argsort(-ev)[:rerank]
        ev2=np.array([H[j]@q for j in cand])
        return int(cand[int(np.argmax(ev2))])
    return int(np.argmax(ev))
for noise in (0.0,0.2,0.35):
    line='noise=%.2f ' % noise
    for w in (0.0,0.6,1.0):
        ok=sum(int(top1(degrade(X[i],noise,i),w)==i) for i in range(30)) if noise>0 else sum(int(top1(X[i],w)==i) for i in range(30))
        line+=' w=%.1f:%2d/30' % (w,ok)
    for w in (0.3,0.6):
        ok=sum(int(top1(degrade(X[i],noise,i),w,rerank=8)==i) for i in range(30)) if noise>0 else sum(int(top1(X[i],w,rerank=8)==i) for i in range(30))
        line+=' rerank(w=%.1f):%2d/30' % (w,ok)
    print(line)
