import sys; sys.path.insert(0,'src')
import numpy as np, itertools
from hcm.ops import unit, kwta
from hcm.data import overlapping_set
X=overlapping_set(30,128,overlap=0.9,seed=11)
S=np.cov(X.T).astype(np.float32)
w_,V_=np.linalg.eigh(S)
print('eig top5',np.round(w_[-5:],5),'sum',float(w_.sum()))
rng=np.random.default_rng(0); W=np.linalg.qr(rng.standard_normal((1024,128)))[0].astype(np.float32)
def mc(H):
    return float(np.mean([float(H[i]@H[j]) for i,j in itertools.combinations(range(30),2)]))
for eps in (1e-4, 1e-3, 1e-2, 3e-2):
    Wh=(V_/np.sqrt(np.maximum(w_,eps)))@V_.T
    Xw=np.asarray([unit(Wh@x) for x in X],dtype=np.float32)
    H=[unit(kwta(np.maximum(W@x,0.0),20)) for x in Xw]
    print('eps=%.0e input_cos=%.3f code_cos=%.3f' % (eps, float(np.mean([float(Xw[i]@Xw[j]) for i,j in itertools.combinations(range(30),2)])), mc(H)))
# 组合：白化 + conscience
Wh=(V_/np.sqrt(np.maximum(w_,1e-3)))@V_.T
Xw=np.asarray([unit(Wh@x) for x in X],dtype=np.float32)
thr=np.zeros(1024,dtype=np.float32)
for ep in range(6):
    for x in Xw:
        thr+=0.35*(kwta(np.maximum(W@x-thr,0.0),20)-20.0/1024.0)
H=[unit(kwta(np.maximum(W@x-thr,0.0),20)) for x in Xw]
print('whiten+conscience code_cos=%.3f' % mc(H))
