# -*- coding: utf-8 -*-
import os, sys, csv, json, re
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from hcm.model import HCM, Config
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
_NANO = re.compile(r'\[[^\]]{1,12}\]|@[\w一-龥]{1,16}')
with open(os.path.join(ROOT, 'bench', 'corpus', 'train__peer-01.csv'), encoding='utf-8-sig', newline='') as f:
    rd = csv.reader(f); next(rd)
    rows = [r[2] for r in rd]
msgs = [_NANO.sub('', x).strip() for x in rows[:600]]
vocab = sorted(set(''.join(msgs)) | {'\ufffd'})
rng = np.random.default_rng(7)
V = {c: i for i, c in enumerate(vocab)}
W = rng.standard_normal((256, len(V))).astype(np.float32)
W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-9
def vec(s):
    idx = [V[c] for c in s if c in V] or [V['\ufffd']]
    x = W @ np.bincount(np.asarray(idx, dtype=np.int64), minlength=len(V)).astype(np.float32)
    return (x / (np.linalg.norm(x) + 1e-12)).astype(np.float32)
X = np.asarray([vec(m) for m in msgs], dtype=np.float32)
cfg = Config(d_x=256, d_c=256, capacity=6000, beta=100.0, L=0, lam=0.6,
             use_resid=False, seed=1, use_ctx_bias=True, kappa_ctx=0.6)
m = HCM(cfg)
episodes = []
for i, x in enumerate(X):
    if msgs[i] == 'OK':
        x_u, u = m._assemble(x)
        xe = m.enc.resid_write(x_u)
        s_pre = m.enc.pre(xe)
        h = m.enc.encode_from_pre(s_pre)
        slot = m.E.write(h, m.t, 1.0, 0.0, 1.0, theta_dup=1.1, payload=x_u, dense=s_pre,
                         ctx=m.c.copy(), tau=m._timecells())
        if slot >= 0:
            episodes.append({'obs': i, 'slot': int(slot), 'cue': m.c.copy()})
            m.wm.present(h); m.t += 1
    else:
        m.observe(x)
    if len(episodes) >= 12: break
print('K =', len(episodes), 'E.n =', m.E.n)
e0 = episodes[0]
xq = vec('OK')
m.c = e0['cue'].copy()
r = m.query(xq)
print('query dict:', json.dumps({k: (v if not isinstance(v, np.ndarray) else 'arr') for k, v in r.items()}, ensure_ascii=False))
# internals
Mn = m.E.M[:, :m.E.n]
xe = xq
s_pre = m.enc.pre(xe)
ed = Mn.T @ s_pre
ev = ed / (np.linalg.norm(ed) + 1e-12)
b = m.E.Ctx[:, :m.E.n].T @ m.c
bz = (b - b.mean()) / (b.std() + 1e-8)
ev2 = ev + 0.6 * np.maximum(bz, 0.0)
print('ev   target rank-value:', float(ev[e0['slot']]), '| max other:', float(np.delete(ev, e0['slot']).max()))
print('b    target:', float(b[e0['slot']]), '| b mean:', float(b.mean()), '| b std:', float(b.std()))
print('ev2  target:', float(ev2[e0['slot']]), '| max other:', float(np.delete(ev2, e0['slot']).max()))
a = np.exp(100.0 * (ev2 - ev2.max())); a /= a.sum()
print('a    target mass:', float(a[e0['slot']]))
recon = m.E.D[:, :m.E.n] @ a
print('raw_ev cos:', float(s_pre @ (recon / (np.linalg.norm(recon) + 1e-12))))
print('argmax slot:', int(np.argmax(ev2)), '| target slot:', e0['slot'])
print('--- direct ctx check ---')
for e in episodes[:4]:
    col = m.E.Ctx[:, e['slot']]
    cue = e['cue']
    print('slot', e['slot'], '| col norm', round(float(np.linalg.norm(col)), 4),
          '| cue norm', round(float(np.linalg.norm(cue)), 4),
          '| cos(col, cue)', round(float(col @ cue / (np.linalg.norm(col) * np.linalg.norm(cue) + 1e-12)), 4))
print('Ctx shape', m.E.Ctx.shape, '| Tau shape', m.E.Tau.shape)
print('--- B arm anatomy (bias off) ---')
m.cfg.use_ctx_bias = False
m.c = episodes[1]['cue'].copy()
rB = m.query(xq)
print('dict:', json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rB.items() if not isinstance(v, np.ndarray)}, ensure_ascii=False))
Mn = m.E.M[:, :m.E.n]
s_preB = m.enc.pre(xq)
ed = Mn.T @ s_preB
evB = ed / (np.linalg.norm(ed) + 1e-12)
top = np.argsort(-evB)[:5]
print('ev top5 slots:', top.tolist(), 'vals:', [round(float(evB[i]), 4) for i in top])
print('episode slots:', [e['slot'] for e in episodes[:6]])
aB = np.exp(100.0 * (evB - evB.max())); aB /= aB.sum()
print('a top mass sum over episodes:', round(float(sum(aB[e['slot']] for e in episodes)), 4))
reconB = m.E.D[:, :m.E.n] @ aB
print('raw_ev:', round(float(s_preB @ (reconB / (np.linalg.norm(reconB) + 1e-12))), 4))
