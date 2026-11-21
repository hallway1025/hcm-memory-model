import sys; sys.path.insert(0,'src')
import numpy as np, itertools
from hcm.ops import unit, kwta
from hcm.data import overlapping_set

def train_sa(X, d_h=1024, k=20, epochs=300, eta=0.05, lam=0.02, seed=0):
    rng=np.random.default_rng(seed)
    d_x=X.shape[1]
    W_e=(rng.standard_normal((d_h,d_x))*0.05).astype(np.float32)
    b=np.zeros(d_h,dtype=np.float32)
    W_d=np.zeros((d_x,d_h),dtype=np.float32)
    for ep in range(epochs):
        for x in X:
            s=np.maximum(W_e@x+b,0.0); h=kwta(s,k)
            mask=(h!=0).astype(np.float32)
            xhat=W_d@h
            diff=xhat-x
            W_d-=eta*np.outer(diff,h)
            dh=W_d.T@diff+lam*np.sign(h)
            g=dh*mask
            W_e-=eta*np.outer(g,x)
            b-=eta*g
    return W_e,b,W_d

def code(W_e,b,x,k=20,normalize=True):
    h=kwta(np.maximum(W_e@x+b,0.0),k)
    return unit(h) if normalize else h

for overlap in (0.5,0.75,0.9):
    X=overlapping_set(30,128,overlap=overlap,seed=11)
    inp=np.mean([float(X[i]@X[j]) for i,j in itertools.combinations(range(30),2)])
    # 随机投射基线
    rng=np.random.default_rng(0); Wr=np.linalg.qr(rng.standard_normal((1024,128)))[0].astype(np.float32)
    h_rand=[unit(kwta(np.maximum(Wr@x,0.0),20)) for x in X]
    c_rand=np.mean([float(h_rand[i]@h_rand[j]) for i,j in itertools.combinations(range(30),2)])
    W_e,b_,W_d=train_sa(X)
    h_learn=[code(W_e,b_,x) for x in X]
    c_learn=np.mean([float(h_learn[i]@h_learn[j]) for i,j in itertools.combinations(range(30),2)])
    recon=np.mean([float(code(W_e,b_,x,normalize=False)@np.array([1.0])) for x in X[:1]]) if False else 0.0
    print('input_overlap=%.2f  input_cos=%.3f  random_code_cos=%.3f  learned_code_cos=%.3f' % (overlap, inp, c_rand, c_learn))
