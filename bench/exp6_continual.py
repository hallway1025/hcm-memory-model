'''E6：类别式持续学习——**修正版**（v2）。

v1 的严重缺陷（由对抗评审 P0-1 发现）：uniform 与 priority 两臂的代码**逐字相同**，
所谓「priority 不优于 uniform」其实是同一个实验跑了两遍。本版实现真正的优先级抽样：
  p_i ∝ recency^a · surprise^b · (1+reward)^c
并同时控制 v1 的第二个混淆：blocked 臂每周期只用 24 个样本，而 uniform/priority 用 48 个。
本版给 blocked 臂**同样 48 个样本**（B 类重复抽样），使三臂的训练量严格相等。'''
import sys, json, os; sys.path.insert(0,'src')
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import random_patterns

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D=128; K=8; PER=12; NOISE=0.35; CYC=12; BUDGET=24
protos=random_patterns(K,D,seed=5)

def sample_set(n_per,seed,noise=NOISE):
    r=np.random.default_rng(seed); X=[]; Y=[]
    for k in range(K):
        for _ in range(n_per):
            X.append(unit(protos[k]+noise*unit(r.normal(size=D)))); Y.append(k)
    return np.array(X,dtype=np.float32), np.array(Y)

Xtr,Ytr=sample_set(PER,101)
Xte,Yte=sample_set(6,202)
m0=HCM(Config(d_x=D,a=0.02,capacity=200,beta=20.0,L=3,use_resid=True,seed=3,d_z=64))
CATS=np.array([m0.enc.encode(unit(p)) for p in protos],dtype=np.float32)
Htr=np.array([m0.enc.encode(x) for x in Xtr],dtype=np.float32)
idxA=np.where(Ytr<4)[0]; idxB=np.where(Ytr>=4)[0]

def priority_weights(m, idx):
    """recency x surprise x (1+reward)。
    recency：样本写入越晚越高；surprise：类别码重建误差（当前皮层的拟合缺口）；
    reward：本任务无外部奖赏，统一取 0（故该项退化为常数 1，如实说明）。"""
    rec = np.linspace(0.2, 1.0, len(idx))
    sur = np.array([1.0 - float(unit(np.asarray(m.sem.forward(Xtr[i]),dtype=np.float32)) @ CATS[Ytr[i]])
                    for i in idx])
    sur = np.clip(sur, 0.0, 2.0)
    w = rec * sur
    return w / (w.sum() + 1e-12)

def run(arm, seed=0):
    m=HCM(Config(d_x=D,a=0.02,capacity=200,beta=20.0,L=3,use_resid=True,seed=3,d_z=64))
    r=np.random.default_rng(seed)
    for _ in range(CYC):
        b=r.choice(idxB,size=BUDGET,replace=False)
        if arm=='blocked':
            # 控制训练量：B 类重复抽到与另两臂相同的 2*BUDGET
            extra=r.choice(idxB,size=BUDGET,replace=False)
            sel=np.concatenate([b,extra])
        elif arm=='uniform':
            o=r.choice(idxA,size=BUDGET,replace=False); sel=np.concatenate([b,o])
        else:  # priority：按 recency x surprise 加权抽旧样本
            w=priority_weights(m, idxA)
            o=r.choice(idxA,size=BUDGET,replace=False,p=w); sel=np.concatenate([b,o])
        m.sem.batch_fit(Xtr[sel],Htr[sel],epochs=2,eta=0.05)
    return m

def acc(m,ks):
    ok=n=0
    for x,y in zip(Xte,Yte):
        if y not in ks: continue
        h=unit(np.asarray(m.sem.forward(x),dtype=np.float32))
        ok+=int(int(np.argmax(CATS@h))==y); n+=1
    return ok,n

rows=[]
for arm in ('blocked','uniform','priority'):
    aA=aB=0; nA=nB=0
    for s in range(3):          # 三个种子，报告均值
        m=run(arm,seed=s)
        x,y=acc(m,set(range(4))); aA+=x; nA+=y
        x,y=acc(m,set(range(4,8))); aB+=x; nB+=y
    rows.append(dict(arm=arm,A=round(aA/3,2),B=round(aB/3,2),n=nA//3,seeds=3))
    print('%-9s 旧任务(0-3): %5.1f/%d   新任务(4-7): %5.1f/%d' % (arm,aA/3,nA//3,aB/3,nB//3),flush=True)

json.dump(rows,open(os.path.join(ROOT,'results','exp6_continual.json'),'w'),indent=1)
print()
d_un=rows[1]['A']-rows[0]['A']; d_pr=rows[2]['A']-rows[1]['A']
print('交错回放 vs 不回放（旧任务）: %+.1f' % d_un)
print('优先级 vs 均匀（旧任务）    : %+.1f' % d_pr)