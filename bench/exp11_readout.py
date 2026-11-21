'''E11（正式版）：双读出的**可辩护**版本。
原论文写的「稀疏 8/20 vs 稠密 20/20」没有任何脚本产出过（评审 P1-1）。
本脚本给出真实测量：在**高特征缺失率**下，三种读出方式确实分层。
低缺失率（<=45%）三者无差异；只有 >=60% 缺失时稠密读出才显示出优势。'''
import sys, os, json; sys.path.insert(0,'src')
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import random_patterns

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D=128; N=60; PROBE=50

def degrade(x,frac,seed):
    r=np.random.default_rng(seed); y=x.copy(); nz=np.where(np.abs(y)>1e-6)[0]
    y[r.choice(nz,size=max(1,int(len(nz)*frac)),replace=False)]=0.0; return unit(y)

X=random_patterns(N,D,seed=71)
m=HCM(Config(d_x=D,a=0.02,capacity=400,beta=20.0,L=3,use_resid=True,seed=0))
for x in X: m.observe(x)
Mn=m.E.M[:,:m.E.n]
rows=[]
print('库内 %d 条；每个条件 %d 次查询' % (m.E.n, PROBE))
print('%-8s %-12s %-12s %-12s' % ('特征缺失','稀疏码读出','稠密单步','稠密+Hopfield'))
for noise in (0.30, 0.45, 0.60, 0.70):
    out={'sparse_only':0,'dense_once':0,'dense_hopfield':0}
    for i in range(PROBE):
        cue=degrade(X[i],noise,i)
        xe=m.enc.resid_read(cue); s_pre=m.enc.pre(xe); q0=m.enc.encode_from_pre(s_pre)
        for arm in out:
            if arm=='sparse_only': sc=Mn.T@q0
            elif arm=='dense_once': sc=Mn.T@s_pre
            else:
                z,a=m.E.retrieve(q0,L=3,beta=20.0,dense=s_pre); sc=a*m.E.sup[:m.E.n]
            out[arm]+=int(int(np.argmax(sc))==i)
    rows.append(dict(noise=noise,n=PROBE,**out))
    print('%-8.2f %-12s %-12s %-12s' % (noise, '%d/%d'%(out['sparse_only'],PROBE),
          '%d/%d'%(out['dense_once'],PROBE),'%d/%d'%(out['dense_hopfield'],PROBE)),flush=True)
json.dump(rows,open(os.path.join(ROOT,'results','exp11_readout.json'),'w'),indent=1)
print('saved results/exp11_readout.json')