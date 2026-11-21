'''E7：工作记忆容量拐点。固定槽位数 C，依次送入 L=8 个条目（L > C）。
保留判据：条目 i 的码仍在某个槽里（槽内余弦 > 0.9）且探针能命中该槽。
若模型正确，保留条数应近似 min(C, L)；C 是唯一自变量。'''
import sys, json; sys.path.insert(0,'src')
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import random_patterns

D=128; L=8
rows=[]
for C in (1,2,3,4,5,6,8):
    kept=[]
    for trial in range(6):
        r=np.random.default_rng(200+trial)
        m=HCM(Config(d_x=D,a=0.02,capacity=200,beta=20.0,L=3,use_resid=True,seed=trial,C=C))
        X=random_patterns(L,D,seed=700+trial)
        codes=[m.enc.encode(x) for x in X]
        for h in codes: m.wm.present(h)
        sk=m.wm.slots
        ok=0
        for i,h in enumerate(codes):
            held=float((sk@h).max())
            if held<0.9: continue          # 已被挤出槽位
            j=int(np.argmax(sk@h))
            q=unit(h+r.normal(scale=0.05,size=h.shape))
            found,sim=m.wm.probe(q)
            ok+=int(found and int(np.argmax(sk@q))==j)
        kept.append(ok)
    rows.append(dict(C=C,kept=float(np.mean(kept)),L=L))
    print('C=%d  retained=%4.2f / %d   (prediction: %d)' % (C,np.mean(kept),L,min(C,L)),flush=True)
json.dump(rows,open('results/exp7_wm.json','w'),indent=1)
