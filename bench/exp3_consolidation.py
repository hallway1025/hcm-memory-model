'''E4：互补学习系统——灾难性遗忘与回放巩固（测量偏移 bug 已修）。'''
import sys, json; sys.path.insert(0,'src')
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import overlapping_set

D=128; N=20; FR=0.10
def degrade(x,frac,seed):
    r=np.random.default_rng(seed); y=x.copy(); nz=np.where(np.abs(y)>1e-6)[0]
    y[r.choice(nz,size=max(1,int(len(nz)*frac)),replace=False)]=0.0; return unit(y)
def acc_c(m,X,n,off):
    ok=0
    for i,x in enumerate(X):
        y=unit(np.asarray(m.sem.forward(degrade(x,FR,i)),dtype=np.float32))
        ok += int(int(np.argmax(m.E.M[:,:n].T@y))==off+i)
    return ok
def acc_h(m,X,off):
    return sum(int(m.query(degrade(x,FR,i))['idx']==off+i) for i,x in enumerate(X))

XA=overlapping_set(N,D,overlap=0.6,seed=21)
XB=overlapping_set(N,D,overlap=0.6,seed=22)
rows=[]
for arm in ('hippo_only','cortex_noreplay','hcm_priority','hcm_uniform'):
    m=HCM(Config(d_x=D,a=0.02,capacity=200,beta=20.0,L=3,use_resid=True,seed=0))
    if arm=='hcm_uniform': m.cons.uniform=True
    for x in XA: m.observe(x)
    nA=m.E.n
    curve=[]
    if arm!='hippo_only':
        for k in range(4):
            m.consolidate(cycles=10); curve.append(acc_c(m,XA,nA,0))
    else:
        curve=[acc_c(m,XA,nA,0)]*4
    if arm=='cortex_noreplay': m.cons.rehearsal=[]; m.cons.interleave=0.0
    for x in XB: m.observe(x)
    nAB=m.E.n
    ret=[]
    for k in range(5):
        if arm!='hippo_only': m.consolidate(cycles=10)
        ret.append(dict(cycles=10*(k+1) if arm!='hippo_only' else 0,
                        A=acc_c(m,XA,nAB,0), B=acc_c(m,XB,nAB,nA)))
    hA=acc_h(m,XA,0); hB=acc_h(m,XB,nA)
    rows.append(dict(arm=arm,curve_A=curve,retention=ret,hippo_A=hA,hippo_B=hB,n=nAB))
    print('%-15s 学A期皮层A=%s' % (arm,curve))
    print('%-15s 学B后 A/B 曲线 %s | 海马 A=%2d/20 B=%2d/20' % ('',
          [(r['cycles'],r['A'],r['B']) for r in ret],hA,hB))
json.dump(rows, open('results/exp3_consolidation.json','w'), indent=1)
