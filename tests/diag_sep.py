import sys; sys.path.insert(0,'src')
import numpy as np, itertools
from hcm.ops import unit, kwta
from hcm.data import overlapping_set
X=overlapping_set(30,128,overlap=0.9,seed=11)
print('X dtype',X.dtype,'shape',X.shape,'norms',np.round(np.linalg.norm(X[:3],axis=1),3))
print('raw cos', float(np.mean([float(X[i]@X[j]) for i,j in itertools.combinations(range(6),2)])))
S=np.cov(X.T).astype(np.float32)
w_,V_=np.linalg.eigh(S+0.1*np.eye(128))
print('eig max/min',float(w_.max()),float(w_.min()))
Wh=(V_/np.sqrt(np.maximum(w_,0.1)))@V_.T
Xw=np.asarray([unit(Wh@x) for x in X],dtype=np.float32)
print('whitened cos', float(np.mean([float(Xw[i]@Xw[j]) for i,j in itertools.combinations(range(6),2)])))
rng=np.random.default_rng(0); W=np.linalg.qr(rng.standard_normal((1024,128)))[0].astype(np.float32)
def mc(H):
    return float(np.mean([float(H[i]@H[j]) for i,j in itertools.combinations(range(30),2)]))
H0=[unit(kwta(np.maximum(W@x,0.0),20)) for x in X]
Hw=[unit(kwta(np.maximum(W@x,0.0),20)) for x in Xw]
print('code cos raw=%.3f  code cos whitened=%.3f' % (mc(H0),mc(Hw)))

# conscience 正确符号：score = W x - thr, thr += eta*(h - target)
thr=np.zeros(1024,dtype=np.float32)
for ep in range(6):
    for x in X:
        h=kwta(np.maximum(W@x-thr,0.0),20)
        thr+=0.35*(h-20.0/1024.0)
Ha=[unit(kwta(np.maximum(W@x-thr,0.0),20)) for x in X]
print('code cos conscience=%.3f  thr range %.2f..%.2f' % (mc(Ha),float(thr.min()),float(thr.max())))
