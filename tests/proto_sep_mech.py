import sys; sys.path.insert(0,'src')
import numpy as np, itertools
from hcm.ops import unit, kwta
from hcm.data import overlapping_set

D_X, D_H, K = 128, 1024, 20

def randproj(seed=0):
    rng=np.random.default_rng(seed)
    return np.linalg.qr(rng.standard_normal((D_H,D_X)))[0].astype(np.float32)

def mean_cos(H):
    idx=list(itertools.combinations(range(len(H)),2))[:400]
    return float(np.mean([float(H[i]@H[j]) for i,j in idx]))

def run_A(X, W, epochs=8, eta=0.5, target=K/D_H):
    b=np.zeros(D_H,dtype=np.float32)
    for ep in range(epochs):
        for x in X:
            h=kwta(np.maximum(W@x+b,0.0),K)
            b+=eta*(h-target)
    return [unit(kwta(np.maximum(W@x+b,0.0),K)) for x in X]

def run_B(X, W, epochs=8, eta=0.02, r=64, seed=3):
    rng=np.random.default_rng(seed)
    U=rng.standard_normal((D_H,r)).astype(np.float32)*0.01
    V=np.zeros((r,D_H),dtype=np.float32)
    b=np.zeros(D_H,dtype=np.float32)
    for ep in range(epochs):
        for x in X:
            h=kwta(np.maximum(W@x+b,0.0),K)
            for it in range(3):
                h=kwta(np.maximum(W@x+b-(U@(V@h)),0.0),K)
            V-=eta*np.outer(U.T@h,h)
            U-=eta*np.outer(h,V@h)
    return [unit(kwta(np.maximum(W@x+b,0.0),K)) for x in X]

def run_C(X, W, eps=1e-1):
    S=np.cov(X.T).astype(np.float32)
    w_,V_=np.linalg.eigh(S+eps*np.eye(D_X))
    Wh=(V_/np.sqrt(np.maximum(w_,eps)))@V_.T
    return [unit(kwta(np.maximum(W@(Wh@x).astype(np.float32),0.0),K)) for x in X]

for overlap in (0.75,0.9):
    X=overlapping_set(30,D_X,overlap=overlap,seed=11)
    inp=float(np.mean([float(X[i]@X[j]) for i,j in itertools.combinations(range(30),2)]))
    W=randproj()
    base=[unit(kwta(np.maximum(W@x,0.0),K)) for x in X]
    print('overlap=%.2f input_cos=%.3f  base=%.3f  A(conscience)=%.3f  B(foldiak)=%.3f  C(zca)=%.3f' % (
        overlap, inp, mean_cos(base), mean_cos(run_A(X,W)), mean_cos(run_B(X,W)), mean_cos(run_C(X,W))))
