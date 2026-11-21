import sys; sys.path.insert(0,'src')
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit

def sparse_patterns(n, d, active=0.15, seed=0):
    rng = np.random.default_rng(seed)
    X=[]
    for _ in range(n):
        x = np.zeros(d, dtype=np.float32)
        idx = rng.choice(d, size=int(d*active), replace=False)
        x[idx] = rng.standard_normal(len(idx)).astype(np.float32)
        X.append(unit(x))
    return np.asarray(X)

def degrade(x, frac, seed):
    rng = np.random.default_rng(seed); y=x.copy()
    nz = np.where(y!=0)[0]
    y[rng.choice(nz, size=max(1,int(len(nz)*frac)), replace=False)] = 0.0
    return unit(y)

def similar_pair(x, share=0.8, seed=0):
    rng = np.random.default_rng(seed); y=x.copy()
    nz = np.where(y!=0)[0]
    n_new = int(len(nz)*(1-share))
    y[rng.choice(nz, size=max(1,n_new), replace=False)] = 0.0
    fresh = np.setdiff1d(np.arange(len(y)), nz)
    pick = rng.choice(fresh, size=max(1,n_new), replace=False)
    y[pick] = rng.standard_normal(len(pick)).astype(np.float32)
    return unit(y)

X = sparse_patterns(60,128,0.15,seed=1)
print('%-5s %-5s %-5s %-7s %-8s %-8s %-8s' % ('a','rel','ann','k','sep','cue30','top1'))
for a in (0.05,0.10,0.20):
  for rel in (None, 0.5):
    for ann in (False, True):
      cfg = Config(d_x=128, rho=8, capacity=500, a=a, rel=rel, seed=0)
      m = HCM(cfg)
      for x in X: m.observe(x)
      m.cfg.rel = rel
      m.enc.rel = rel
      H = m.E.M[:, :m.E.n]
      seps=[]; cues=[]; top1=0; rec=0
      for i in range(20):
        s2 = similar_pair(X[i], 0.8, seed=i)
        seps.append(float(m.enc.encode(m._assemble(s2)[1]) @ H[:,i]))
        c30 = degrade(X[i], 0.30, seed=i+50)
        q = m.enc.encode(m._assemble(c30)[1])
        cues.append(float(q @ H[:,i]))
        top1 += int(np.argmax(H.T @ q) == i)
        qs,_ = m.E.retrieve(q, L=8, anneal=ann)
        rec += int(np.argmax(H.T @ qs) == i)
      print('%-5.2f %-5s %-5s %-7d %-8.3f %-8.3f %-8s' % (a, str(rel), str(ann), cfg.k, np.mean(seps), np.mean(cues), '%d/20|%d'%(top1,rec)))
