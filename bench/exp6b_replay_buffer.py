'''E6b：回放缓冲容量有限时，优先级回放 vs 均匀回放。
旧任务 4 类 x 12 = 48 条；缓冲只有 B 条（6/12/24）。priority 取 surprise*recency*(1+use) 最高的 B 条，
uniform 随机取 B 条。每周期只从缓冲抽 8 条与 24 条新任务样本混合。'''
import sys, json; sys.path.insert(0,'src')
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import random_patterns

D=128; K=8; NOISE=0.35; CYC=12; BUDGET=8
protos=random_patterns(K,D,seed=5)
def sample_set(n_per,seed):
    r=np.random.default_rng(seed); X=[]; Y=[]
    for k in range(K):
        for _ in range(n_per):
            X.append(unit(protos[k]+NOISE*unit(r.normal(size=D)))); Y.append(k)
    return np.array(X,dtype=np.float32), np.array(Y)
Xtr,Ytr=sample_set(12,101)
Xte,Yte=sample_set(6,202)
m0=HCM(Config(d_x=D,a=0.02,capacity=200,beta=20.0,L=3,use_resid=True,seed=3,d_z=64))
CATS=np.array([m0.enc.encode(unit(p)) for p in protos],dtype=np.float32)
def acc(m,X,Y,ks):
    ok=0; n=0
    for x,y in zip(X,Y):
        if y not in ks: continue
        h=unit(np.asarray(m.sem.forward(x),dtype=np.float32))
        ok+=int(int(np.argmax(CATS@h))==y); n+=1
    return ok,n
idxA=np.where(Ytr<4)[0]; idxB=np.where(Ytr>=4)[0]
# 旧样本优先级：surprise（与类别码的重建误差）x recency（越晚越高）
sur=np.array([1.0-float(m0.enc.encode(Xtr[i])@CATS[Ytr[i]]) for i in idxA])
rec=np.linspace(0.2,1.0,len(idxA))
prio=sur*rec
order_pri=np.argsort(-prio); order_uni=np.arange(len(idxA))
rows=[]
for buf in (6,12,24):
  for arm in ('uniform','priority'):
    r=np.random.default_rng(7)
    m=HCM(Config(d_x=D,a=0.02,capacity=200,beta=20.0,L=3,use_resid=True,seed=3,d_z=64))
    Htr=np.array([m.enc.encode(x) for x in Xtr],dtype=np.float32)
    pool=idxA[order_pri[:buf]] if arm=='priority' else idxA[r.choice(len(idxA),size=buf,replace=False)]
    for c in range(CYC):
        b=BUDGET//2
        sel=np.concatenate([r.choice(idxB,size=BUDGET- b,replace=False), r.choice(pool,size=b,replace=False)])
        m.sem.batch_fit(Xtr[sel],Htr[sel],epochs=2,eta=0.05)
    aA,nA=acc(m,Xte,Yte,set(range(4))); aB,nB=acc(m,Xte,Yte,set(range(4,8)))
    rows.append(dict(buf=buf,arm=arm,A=aA,B=aB,n=nA))
    print('buf=%-2d %-8s  A(0-3): %2d/%d   B(4-7): %2d/%d' % (buf,arm,aA,nA,aB,nB),flush=True)
json.dump(rows,open('results/exp6b_buffer.json','w'),indent=1)
