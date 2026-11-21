# -*- coding: utf-8 -*-
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from hcm import HCM, Config
from hcm.ops import unit

def pats(n, d, seed):
    r = np.random.default_rng(seed)
    return np.asarray([unit(v) for v in r.standard_normal((n, d))], dtype=np.float32)

def degrade(x, frac, seed):
    r = np.random.default_rng(seed)
    y = x.copy()
    nz = np.where(np.abs(y) > 1e-6)[0]
    y[r.choice(nz, size=max(1, int(len(nz) * frac)), replace=False)] = 0.0
    return unit(y)

N, D = 2000, 128
X = pats(N, D, seed=300)
for arm, wa, b in (('baseline', 0.8, -1.2), ('w_a_x0.5', 0.4, -1.2), ('b_neg', 0.8, -1.6)):
    m = HCM(Config(d_x=D, d_c=D, a=0.02, capacity=N + 10, beta=100.0, L=3, use_resid=False, seed=0))
    m.gate.w_a = wa
    m.gate.b = b
    skipped = []
    for i, x in enumerate(X):
        rec = m.observe(x)
        if not rec['wrote']:
            skipped.append(i)
    print(arm, 'skipped_at_write:', skipped[:10], 'n=', m.E.n)
    n_rep = 200
    rw = 0
    for i in range(n_rep):
        rec = m.observe(X[i])
        rw += 1 if rec['wrote'] else 0
    print(arm, 'exact rewrites:', rw, 'n after reobs=', m.E.n)
    probes = list(range(0, N, N // 40))[:40]
    acts = {}
    hits = 0
    detail = []
    for i in probes:
        q = degrade(X[i], 0.25, i)
        o = m.query(q, do_rif=False)
        acts[o['action']] = acts.get(o['action'], 0) + 1
        if int(o['idx']) == i:
            hits += 1
        detail.append((i, int(o['idx'])))
    print(arm, 'deg25:', hits, '/40 actions:', acts)
    print(arm, 'probe->idx detail:', detail[:8], '...', detail[-3:])
