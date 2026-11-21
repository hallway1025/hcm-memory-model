import sys, itertools; sys.path.insert(0,'src')
import numpy as np
from hcm import HCM, Config
from hcm.data import overlapping_set
for overlap in (0.5,0.75):
  for N in (30,100,300):
    X=overlapping_set(N,128,overlap=overlap,seed=11)
    m=HCM(Config(d_x=128,a=0.02,capacity=1500,w_dense=1.0,homeo=True,seed=0))
    m.develop(X,epochs=3)
    for x in X: m.observe(x)
    H=m.E.M[:,:m.E.n]
    sup=set(tuple(np.where(H[:,j]>0)[0]) for j in range(H.shape[1]))
    idx=np.arange(0,min(N,40))
    cs=[float(H[:,i]@H[:,j]) for i,j in itertools.combinations(idx,2)][:300]
    act=(H[:len(idx)]>0).sum(0)
    print('ov=%.2f N=%3d codes=%d distinct=%3d meancos=%.3f maxcos=%.3f active_units=%d theta[min,max]=%.2f,%.2f' % (
      overlap,N,H.shape[1],len(sup),float(np.mean(cs)),float(np.max(cs)),len(np.where(H.any(1))[0]),float(m.enc.theta.min()),float(m.enc.theta.max())))
