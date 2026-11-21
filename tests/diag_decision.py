import sys; sys.path.insert(0,'src')
import numpy as np
from hcm import HCM, Config
from hcm.data import random_patterns, corrupt
X = random_patterns(40,128,seed=1)
m = HCM(Config(d_x=128,rho=8,capacity=500,a=0.10,seed=0))
for x in X: m.observe(x)
acts={}; m1s=[]; m2s=[]; nus=[]
for i in range(20):
    r = m.query(corrupt(X[i],0.3,seed=i))
    acts[r['action']]=acts.get(r['action'],0)+1
    m1s.append(r['m1']); m2s.append(r['m2']); nus.append(r['nu'])
print('actions', acts)
print('m1 mean=%.3f min=%.3f max=%.3f' % (np.mean(m1s), np.min(m1s), np.max(m1s)))
print('m2 mean=%.3f' % np.mean(m2s))
print('nu mean=%.3f min=%.3f max=%.3f' % (np.mean(nus), np.min(nus), np.max(nus)))
print('margin mean=%.3f' % np.mean(np.array(m1s)-np.array(m2s)))
