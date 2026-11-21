'''E4b：皮层容量受限时的稳定-可塑权衡扫描。找顺序学习导致旧任务崩塌的参数区。'''
import sys, json; sys.path.insert(0,'src')
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import overlapping_set

D=128; N=20; FR=0.10; CYC=30
def degrade(x,frac,seed):
    r=np.random.default_rng(seed); y=x.copy(); nz=np.where(np.abs(y)>1e-6)[0]
    y[r.choice(nz,size=max(1,int(len(nz)*frac)),replace=False)]=0.0; return unit(y)
def acc_c(m,X,n,off):
    ok=0
    for i,x in enumerate(X):
        y=unit(np.asarray(m.sem.forward(degrade(x,FR,i)),dtype=np.float32))
        ok+=int(int(np.argmax(m.E.M[:,:n].T@y))==off+i)
    return ok
XA=overlapping_set(N,D,overlap=0.6,seed=21)
XB=overlapping_set(N,D,overlap=0.6,seed=22)
rows=[]
for dz in (8,16,32):
  for eta in (0.05,0.20):
    for arm in ('noreplay','priority','uniform'):
      m=HCM(Config(d_x=D,a=0.02,capacity=200,beta=20.0,L=3,use_resid=True,seed=0,d_z=dz))
      m.cons.eta=eta
      if arm=='uniform': m.cons.uniform=True
      for x in XA: m.observe(x)
      nA=m.E.n; m.consolidate(cycles=CYC); a0=acc_c(m,XA,nA,0)
      if arm=='noreplay': m.cons.rehearsal=[]; m.cons.interleave=0.0
      for x in XB: m.observe(x)
      nAB=m.E.n; m.consolidate(cycles=CYC)
      a1=acc_c(m,XA,nAB,0); b1=acc_c(m,XB,nAB,nA)
      rows.append(dict(dz=dz,eta=eta,arm=arm,A_before=a0,A_after=a1,B_after=b1))
      print('dz=%-3d eta=%.2f %-9s  A: %2d -> %2d   B=%2d/20' % (dz,eta,arm,a0,a1,b1))
json.dump(rows, open('results/exp4_cortical_capacity.json','w'), indent=1)
