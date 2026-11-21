#!/usr/bin/env python3
'''exp28a_diag.py -- where does a1 mass sit? (instrument before mechanism, L-034)'''
import sys, os, json, csv, re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESS = 'peer-01'
_NANO = re.compile(r'\[[^\]]{1,12}\]')
def _clean(s):
    s = _NANO.sub(' ', str(s))
    return re.sub(r'\s+', ' ', s).strip()
rows = [r for r in list(csv.reader(open(os.path.join(ROOT, 'bench', 'corpus', 'train__%s.csv' % SESS), encoding='utf-8-sig')))[1:] if len(r) >= 3]
msgs = [r[2] for r in rows]
z = np.load(os.path.join(ROOT, 'results', 'exp23_semantic_embed.npz'), allow_pickle=True)
E = z['E']; chars = [str(c) for c in z['chars']]
V = chars + ([] if '\ufffd' in chars else ['\ufffd'])
Vi = {ch: i for i, ch in enumerate(V)}
M = E.T.copy()
D_x = M.shape[0]
def x_of(s):
    idx = [Vi.get(c, Vi['\ufffd']) for c in _clean(s)] or [Vi['\ufffd']]
    v = M[:, idx].sum(1)
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.zeros(D_x, dtype=np.float32)
cfg = Config(d_x=D_x, d_c=D_x, a=0.02, capacity=len(msgs) + 10, beta=100.0, L=3, use_resid=False, seed=0)
m = HCM(cfg)
for i, s in enumerate(msgs):
    m.observe(x_of(s))
nw = int(m.E.n)
obs_of = {int(m.E.t[j]): j for j in range(nw)}
print('written', nw, flush=True)
probe = [i for i in range(len(msgs) - 1) if i in obs_of and (i + 1) in obs_of and len(_clean(msgs[i])) >= 4][:60]
stats = []
for i in probe:
    x = x_of(msgs[i])
    pre = m.enc.raw(x)
    h = m.enc.encode_from_pre(pre)
    slot_i = obs_of[i]
    col = m.E.M[:, slot_i]
    cos_h_col = float(h @ col)
    _, a1 = m.E.retrieve(h, L=3, beta=100.0, dense=pre)
    k = int(np.argmax(a1))
    top = np.argsort(-a1)[:5]
    stats.append(dict(i=i, slot_i=slot_i, a1_argmax=k, argmax_is_slot=k == slot_i,
                      a1_mass_sloti=round(float(a1[slot_i]), 4), a1_top1_mass=round(float(a1[top[0]]), 4),
                      a1_top5_mass=round(float(a1[top].sum()), 4), cos_h_col=round(cos_h_col, 4),
                      score_self=round(float(m.E.scores(h)[slot_i]), 4),
                      score_top1=round(float(m.E.scores(h)[np.argmax(m.E.scores(h))]), 4)))
agree = sum(s['argmax_is_slot'] for s in stats)
print('a1 argmax==slot_i: %d/%d' % (agree, len(stats)), flush=True)
for s in stats[:12]:
    print(json.dumps(s), flush=True)
ms = [s['a1_mass_sloti'] for s in stats]
print('a1[slot_i] mass: mean=%.4f med=%.4f min=%.4f' % (np.mean(ms), np.median(ms), min(ms)), flush=True)
sc = [s['score_self'] for s in stats]
print('score_self: mean=%.4f min=%.4f' % (np.mean(sc), min(sc)), flush=True)
