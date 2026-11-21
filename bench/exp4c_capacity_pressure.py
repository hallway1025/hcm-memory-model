'''E4c：皮层容量压力下的遗忘（每任务 200 条，d_z 与条数比决定干扰）。'''
import sys, json; sys.path.insert(0,'src')
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import overlapping_set

D=128; N=200; FR=0.10; CYC=40
def degrade(x,frac,seed):
    r=np.random.default_rng(seed); y=x.copy(); nz=np.where(np.abs(y)>1e-6)[0]
    y[r.choice(nz,size=max(1,int(len(nz)*frac)),replace=False)]=0.0; return unit(y)
def acc_c(m,X,n,off):
    ok=0
    for i,x in enumerate(X):
        y=unit(np.asarray(m.sem.forward(degrade(x,FR,i)),dtype=np.float32))
        ok+=int(int(np.argmax(m.E.M[:,:n].T@y))==off+i)
    return ok
XA=overlapping_set(N,D,overlap=0.5,seed=31)
XB=overlapping_set(N,D,overlap=0.5,seed=32)
rows=[]
for dz in (64,256):
  for arm in ('noreplay','priority','uniform'):
    m=HCM(Config(d_x=D,a=0.02,capacity=600,beta=20.0,L=3,use_resid=True,seed=0,d_z=dz))
    if arm=='uniform': m.cons.uniform=True
    for x in XA: m.observe(x)
    nA=m.E.n; m.consolidate(cycles=CYC); a0=acc_c(m,XA,nA,0)
    if arm=='noreplay': m.cons.rehearsal=[]; m.cons.interleave=0.0
    for x in XB: m.observe(x)
    nAB=m.E.n; m.consolidate(cycles=CYC)
    a1=acc_c(m,XA,nAB,0); b1=acc_c(m,XB,nAB,nA)
    rows.append(dict(dz=dz,arm=arm,A_before=a0,A_after=a1,B_after=b1))
    print('dz=%-3d %-9s  A: %2d -> %2d   B=%2d/200  (n=%d)' % (dz,arm,a0,a1,b1,nAB))
json.dump(rows, open('results/exp4c_capacity_pressure.json','w'), indent=1)
