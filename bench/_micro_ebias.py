
import sys, numpy as np
sys.path.insert(0, 'src')
from hcm.model import Config, HCM
from hcm.data import random_patterns
rng = np.random.default_rng(0)
d_x = 64
ctxs = [random_patterns(1, d_x, seed=100+i)[0] for i in range(6)]
pats = [random_patterns(1, d_x, seed=200+i)[0] for i in range(6)]
def degrade(x, frac, seed):
    r = np.random.default_rng(seed); y = x.copy()
    nz = np.where(y > 0)[0]; kill = r.choice(nz, int(len(nz)*frac), replace=False)
    y[kill] = 0.0
    n = np.linalg.norm(y)
    return y / n if n > 0 else y
d1 = degrade(pats[1], 0.3, 7)
for flag in (False, True):
    cfg = Config(d_x=d_x, capacity=50, use_ctx_bias=True, ctx_in_energy=flag,
                 use_resid=True,  # D-052 显式传参：保持翻转前语义（无产物文件；翻转前行为默认 True）
                 kappa_ctx=16.0, seed=0)
    m = HCM(cfg)
    slots = []
    for i in range(6):
        m.c = ctxs[i].copy()
        r = m.observe(pats[i])
        slots.append(r.get('idx', -1))
    m.c = ctxs[0].copy(); rA = m.query(pats[1])       # exact content + wrong-era ctx
    m.c = ctxs[0].copy(); rB = m.query(d1)            # degraded content + wrong-era ctx
    m.c = ctxs[1].copy(); rC = m.query(d1)            # degraded content + own-era ctx
    print('flag=%-5s slots=%s | exact+ctx0 -> %s (slot1=%d) | deg+ctx0 -> %s | deg+ctx1 -> %s' %
          (flag, slots, rA.get('idx'), slots[1], rB.get('idx'), rC.get('idx')))
print('MICRO OK')
