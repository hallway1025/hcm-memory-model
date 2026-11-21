import sys; sys.path.insert(0,'src')
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import overlapping_set

D=128
XA=overlapping_set(20,D,overlap=0.6,seed=21)
XB=overlapping_set(20,D,overlap=0.6,seed=22)
m=HCM(Config(d_x=D,a=0.02,capacity=200,beta=20.0,L=3,use_resid=True,seed=0))
for x in XA: m.observe(x)
print('n after A =', m.E.n, ' gate mean %.3f' % np.mean([r['g'] for r in m.log]), 'wrote', sum(r['wrote'] for r in m.log))
base_n=m.E.n
for x in XB: m.observe(x)
print('n after B =', m.E.n, 'wrote in B phase', sum(r['wrote'] for r in m.log[base_n:]), '/20')
print('  gate values B:', ' '.join('%.2f'%r['g'] for r in m.log[base_n:base_n+6]))
print('  nu  values B:', ' '.join('%.2f'%r['nu'] for r in m.log[base_n:base_n+6]))
n=m.E.n
for name,X,lo in (('A',XA,0),('B',XB,base_n)):
  oks=[]
  for i,x in enumerate(X):
    q=m.query(x); oks.append(int(q['idx']==lo+i))
  print('  %s query exact top1 = %d/20' % (name,sum(oks)))
  # 各模板的独立得分
  best=None
  for k,xe in enumerate(m.enc.resid_read_all(X[0])):
    s_pre=m.enc.pre(xe); qq=m.enc.encode_from_pre(s_pre); ev=m.E.M[:,:n].T@m.enc.raw(xe)
    j=int(np.argmax(ev)); print('    template %d: argmax=%2d (期望 %d) score=%.3f' % (k,j,lo,float(ev[j])))
