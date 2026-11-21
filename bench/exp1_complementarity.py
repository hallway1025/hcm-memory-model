'''E1：互补性实验。overlap x noise x w_dense x 稳态阈值开/关。'''
import sys, json; sys.path.insert(0,'src')
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import overlapping_set

N=30
def make(overlap, seed=11):
    if overlap <= 0.0:
        rng=np.random.default_rng(seed)
        return np.asarray([unit(rng.standard_normal(128).astype(np.float32)) for _ in range(N)], dtype=np.float32)
    return overlapping_set(N, 128, overlap=overlap, seed=seed)

def degrade(x, frac, seed):
    if frac <= 0: return x
    r=np.random.default_rng(seed); y=x.copy(); nz=np.where(np.abs(y)>1e-6)[0]
    y[r.choice(nz, size=max(1,int(len(nz)*frac)), replace=False)]=0.0
    return unit(y)

rows=[]
for overlap in (0.0, 0.5, 0.75, 0.9):
    X=make(overlap)
    for homeo in (True, False):
        for a in (0.02, 0.008):
            for wd in (0.0, 0.3, 0.6, 1.0):
                for noise in (0.0, 0.2, 0.35):
                    m=HCM(Config(d_x=128, a=a, capacity=500, w_dense=wd, homeo=homeo, L=0, seed=0,
                                 use_resid=True))  # D-052 显式传参：保持翻转前语义（产物早于 2026-11-08 05:04 翻转）
                    if homeo: m.develop(X, epochs=6)
                    for x in X: m.observe(x)
                    ok=sum(int(m.query(degrade(X[i],noise,i))['idx']==i) for i in range(N))
                    rows.append(dict(overlap=overlap, homeo=homeo, a=a, w_dense=wd, noise=noise, top1=ok, N=N))

print('%-7s %-6s %-5s | %s' % ('overlap','homeo','a','  '.join('w=%.1f/x%.2f'%(w,n) for w in (0.0,0.6,1.0) for n in (0.0,0.2,0.35))))
for overlap in (0.0,0.5,0.75,0.9):
  for homeo in (True, False):
    for a in (0.02, 0.008):
      cells=[]
      for w in (0.0,0.6,1.0):
        for n in (0.0,0.2,0.35):
          v=[r['top1'] for r in rows if (r['overlap'],r['homeo'],r['a'],r['w_dense'],r['noise'])==(overlap,homeo,a,w,n)][0]
          cells.append('%2d' % v)
      print('%-7.2f %-6s %-5.3f | %s' % (overlap, homeo, a, '  '.join(cells)))
json.dump(rows, open('results/exp1.json','w'), indent=1)
print('saved results/exp1.json', len(rows), 'rows')
