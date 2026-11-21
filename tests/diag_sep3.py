import sys; sys.path.insert(0,'src')
import numpy as np, itertools
from hcm.ops import unit, kwta
from hcm.data import overlapping_set
X=overlapping_set(30,128,overlap=0.9,seed=11)
def mc(H):
    return float(np.mean([float(H[i]@H[j]) for i,j in itertools.combinations(range(30),2)]))
def norm_rows(W):
    return W/(np.linalg.norm(W,axis=1,keepdims=True)+1e-9)
for k in (8,20,40):
    rng=np.random.default_rng(0); W=np.linalg.qr(rng.standard_normal((1024,128)))[0].astype(np.float32)
    thr=np.zeros(1024,dtype=np.float32)
    for ep in range(6):
        for x in X:
            s=W@x-thr; h=kwta(np.maximum(s,0.0),k)
            thr+=0.35*(h-k/1024.0)
    H=[unit(kwta(np.maximum(W@x-thr,0.0),k)) for x in X]
    # 覆盖度：用了多少不同的单元
    used=len(set(np.concatenate([np.where(h>0)[0] for h in H]).tolist()))
    print('conscience k=%2d code_cos=%.3f units_used=%d/1024' % (k,mc(H),used))

# 竞争学习（在线 k-means / Grossberg）
for k in (8,20):
    rng=np.random.default_rng(0); W=norm_rows(np.linalg.qr(rng.standard_normal((1024,128)))[0].astype(np.float32))
    for ep in range(30):
        for x in X:
            s=W@x; h=kwta(np.maximum(s,0.0),k)
            win=np.where(h>0)[0]
            W[win]+=0.15*(x[None,:]-W[win])
            W=norm_rows(W)
    H=[unit(kwta(np.maximum(W@x,0.0),k)) for x in X]
    used=len(set(np.concatenate([np.where(h>0)[0] for h in H]).tolist()))
    print('compet   k=%2d code_cos=%.3f units_used=%d/1024' % (k,mc(H),used))
