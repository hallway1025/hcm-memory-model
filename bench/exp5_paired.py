'''E5：配对联想冲突（同一线索 -> 不同目标）。这是经典 CLS 的灾难性遗忘设计：
  皮层必须用同一批突触表示两套映射，后学的会覆盖先学的；海马存的是具体情节，
  回放（交错）能否把旧映射保住，是本实验要回答的问题。'''
import sys, json; sys.path.insert(0,'src')
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit, cosine_rows
from hcm.data import random_patterns

D=128; N=20; CYC=30; ETA=0.05
def acc_map(m,cues,targets):
    ok=0
    for i,x in enumerate(cues):
        y=unit(np.asarray(m.sem.forward(x),dtype=np.float32))
        ok+=int(int(np.argmax(targets@y))==i)
    return ok
m=HCM(Config(d_x=D,a=0.02,capacity=200,beta=20.0,L=3,use_resid=True,seed=7,d_z=256))
cues=random_patterns(N,D,seed=1)
respA=random_patterns(N,D,seed=101); respB=random_patterns(N,D,seed=202)
hA=np.array([m.enc.encode(unit(r)) for r in respA],dtype=np.float32)
hB=np.array([m.enc.encode(unit(r)) for r in respB],dtype=np.float32)
print('target-code cross-cosine: A-A %.2f  A-B %.2f' % (float((hA@hA.T).mean()), float((hA@hB.T).mean())))

def train(pairs,cycles,replay=None,frac=0.5,seed=0):
    r=np.random.default_rng(seed)
    for c in range(cycles):
        X=pairs[0]; Y=pairs[1]
        if replay is not None:
            k=int(len(X)*frac)
            idx=r.choice(len(replay[0]),size=k,replace=False)
            X=np.vstack([X,replay[0][idx]]); Y=np.vstack([Y,replay[1][idx]])
        m.sem.batch_fit(X,Y,epochs=3,eta=ETA)
rows=[]
for arm in ('blocked','replay50','replay25','replay_uniform'):
    m=HCM(Config(d_x=D,a=0.02,capacity=200,beta=20.0,L=3,use_resid=True,seed=7,d_z=256))
    train((cues,hA),CYC); a_before=acc_map(m,cues,hA)
    if arm=='blocked':
        train((cues,hB),CYC)
    elif arm=='replay50':
        train((cues,hB),CYC,replay=(cues,hA),frac=0.5)
    elif arm=='replay25':
        train((cues,hB),CYC,replay=(cues,hA),frac=0.25)
    else:
        train((cues,hB),CYC,replay=(cues,hA),frac=1.0)
    a_after=acc_map(m,cues,hA); b_after=acc_map(m,cues,hB)
    rows.append(dict(arm=arm,A_before=a_before,A_after=a_after,B_after=b_after))
    print('%-14s A: %2d -> %2d/20   B: %2d/20' % (arm,a_before,a_after,b_after))
json.dump(rows, open('results/exp5_paired.json','w'), indent=1)
