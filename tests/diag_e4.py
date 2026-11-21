import sys; sys.path.insert(0,'src')
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import overlapping_set

D=128
XA=overlapping_set(20,D,overlap=0.6,seed=21)
XB=overlapping_set(20,D,overlap=0.6,seed=22)
print('input cos A-A %.3f  B-B %.3f  A-B %.3f' % (
  np.mean((XA@XA.T)[np.triu_indices(20,1)]), np.mean((XB@XB.T)[np.triu_indices(20,1)]),
  np.mean(XA@XB.T)))
for resid in (False,True):
    m=HCM(Config(d_x=D,a=0.02,capacity=200,beta=20.0,L=3,use_resid=resid,seed=0))
    mus=[]
    for x in XA:
        mus.append(m.enc.mu.copy() if resid else None); m.observe(x)
    nA=m.E.n
    for x in XB:
        mus.append(m.enc.mu.copy() if resid else None); m.observe(x)
    print('--- resid=%s  n=%d' % (resid, m.E.n))
    if resid:
        print('   mu drift: cos(mu_afterA, mu_end)=%.3f  |mu_end|=%.3f' % (
          float(unit(mus[nA-1])@unit(mus[-1])), float(np.linalg.norm(mus[-1]))))
    for name,X,lo,hi in (('A',XA,0,nA),('B',XB,nA,m.E.n)):
        ok=0; oks=0; cs=[]
        for i,x in enumerate(X):
            hq=m.enc.encode(m.enc.resid_read(x)) if resid else m.enc.encode(x)
            cs.append(float(unit(hq)@m.E.M[:,lo+i]))
            sc=m.E.M[:,:m.E.n].T@hq; g=int(np.argmax(sc))
            ok+= int(g==lo+i)
            q=m.query(x); oks+= int(q['idx']==lo+i)
        print('   %s: 稀疏码sumtop1=%2d/20  query()top1=%2d/20  cos(查询码,写入码)均值=%.3f' % (name,ok,oks,np.mean(cs)))
