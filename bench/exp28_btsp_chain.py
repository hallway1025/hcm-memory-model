#!/usr/bin/env python3
'''exp28_btsp_chain.py -- M4_CON extension: replay-formed temporal links give forward recall.
Baseline (exp27): pure episodic completion has NO usable forward association (next@5=1.5%,
med rank 2172 vs ctrl 3631). Mechanism: offline pass potentiates one-shot links
Wlink[slot_j, slot_i] += kappa for consecutive pairs (gap<=30s, diff sender) = behavioral-
timescale / replay link (Bittner-Magee BTSP story; Buzsaki CA3 chains). Retrieval:
two-phase score = Hopfield a1 then a2 = softmax(beta*(scores + w_chain*Wlink^T a1)).
Arms: w_chain in {0, 0.3, 1.0, 3.0}. Metrics: self@1 (argmax==slot_i), next@1/@5 (rank of
slot_j excluding self in a2), joint self@1&next@2. Judge obs_of (L-035). beta=100 (D-035).'''
import sys, os, json, time, csv, re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESS = sys.argv[1] if len(sys.argv) > 1 else 'peer-01'
NP = int(sys.argv[2]) if len(sys.argv) > 2 else 200
_NANO = re.compile(r'\[[^\]]{1,12}\]')
def _clean(s):
    s = _NANO.sub(' ', str(s))
    return re.sub(r'\s+', ' ', s).strip()

rows = list(csv.reader(open(os.path.join(ROOT, 'bench', 'corpus', 'train__%s.csv' % SESS), encoding='utf-8-sig')))
rows = [r for r in rows[1:] if len(r) >= 3]
msgs = [r[2] for r in rows]
times = [r[0] for r in rows]
senders = [r[1] for r in rows]
print('corpus %s: %d msgs' % (SESS, len(msgs)), flush=True)

z = np.load(os.path.join(ROOT, 'results', 'exp23_semantic_embed.npz'), allow_pickle=True)
E = z['E']; chars = [str(c) for c in z['chars']]
V = chars + ([] if '\\ufffd' in chars else ['\\ufffd'])
Vi = {ch: i for i, ch in enumerate(V)}
M = E.T.copy()
if '\\ufffd' not in chars:
    r0 = np.random.default_rng(99)
    M = np.hstack([M, r0.standard_normal((M.shape[0], 1))])
    M[:, -1] /= np.linalg.norm(M[:, -1])
D_x = M.shape[0]
def x_of(s):
    idx = [Vi.get(c, Vi['\\ufffd']) for c in _clean(s)] or [Vi['\\ufffd']]
    v = M[:, idx].sum(1)
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.zeros(D_x, dtype=np.float32)

cfg = Config(d_x=D_x, d_c=D_x, a=0.02, capacity=len(msgs) + 10, beta=100.0, L=3, use_resid=False, seed=0)
m = HCM(cfg)
t0 = time.time()
for i, s in enumerate(msgs):
    m.observe(x_of(s))
nw = int(m.E.n)
obs_of = {int(m.E.t[j]): j for j in range(nw)}
slot_of = {i: obs_of[i] for i in obs_of}
print('written %d/%d %.0fs' % (nw, len(msgs), time.time() - t0), flush=True)

from datetime import datetime
def tparse(s):
    try: return datetime.fromisoformat(s.replace('/', '-'))
    except Exception: return None
pairs = []
for i in range(len(msgs) - 1):
    if i not in obs_of or (i + 1) not in obs_of:
        continue
    a, b = _clean(msgs[i]), _clean(msgs[i + 1])
    if len(a) < 4 or len(b) < 4 or senders[i] == senders[i + 1]:
        continue
    ta, tb = tparse(times[i]), tparse(times[i + 1])
    if ta is None or tb is None or (tb - ta).total_seconds() > 30:
        continue
    pairs.append(i)
rng = np.random.default_rng(2700)
rng.shuffle(pairs)
pairs = pairs[:NP]
print('probe pairs: %d' % len(pairs), flush=True)

def degrade(x, frac, seed):
    r = np.random.default_rng(seed)
    y = x.copy()
    nz = np.where(np.abs(y) > 1e-6)[0]
    y[r.choice(nz, size=max(1, int(len(nz) * frac)), replace=False)] = 0.0
    return unit(y)

# codes for every written slot (for Hopfield drive)
codes = {}
for i in pairs:
    codes[i] = m.enc.encode_from_pre(m.enc.raw(x_of(msgs[i])))

def softmaxb(v, beta):
    v = v - v.max()
    w = np.exp(beta * v)
    return w / w.sum()

out = {}
for w_chain in (0.0, 0.5, 1.0, 2.0):
    Wl = np.zeros((nw, nw), dtype=np.float32)  # Wl[slot_j, slot_i] = 1  (link i -> j)
    if w_chain > 0:
        for i in pairs:
            Wl[obs_of[i + 1], obs_of[i]] += 1.0
    for deg in (0.0, 0.3):
        self1 = next1 = next5 = joint = 0
        used = 0
        for i in pairs:
            j = i + 1
            x = x_of(msgs[i])
            if deg > 0:
                x = degrade(x, deg, i)
            pre = m.enc.raw(x)
            h = m.enc.encode_from_pre(pre)
            _, a1 = m.E.retrieve(h, L=3, beta=100.0, dense=pre)  # episodic readout
            s2 = m.E.scores(h) + w_chain * (Wl @ a1)              # chained readout (INCOMING links)
            a2 = softmaxb(s2, 100.0)
            used += 1
            k1 = int(np.argmax(a1))
            k2 = int(np.argmax(a2))
            order = np.argsort(-a2)
            rj = int(np.where(order == obs_of[j])[0][0])
            if k1 == obs_of[i]: self1 += 1
            if k2 == obs_of[j]: next1 += 1
            if rj < 5: next5 += 1
            if k1 == obs_of[i] and k2 == obs_of[j]: joint += 1
        key = 'w%g_deg%g' % (w_chain, deg * 100)
        out[key] = dict(pairs=used, self_hit1=self1, self_rate=round(self1 / used, 4),
                        next_hit1=next1, next_rate1=round(next1 / used, 4),
                        next_hit5=next5, next_rate5=round(next5 / used, 4),
                        joint_self1_next1=joint, joint_rate=round(joint / used, 4))
        print('%s self@1(a1)=%d/%d next@1(a2)=%d next@5=%d joint=%d/%d' % (key, self1, used, next1,
              next5, joint, used), flush=True)
    del Wl


json.dump(dict(config=dict(session=SESS, n_msgs=len(msgs), n_written=nw, d_x=D_x, pairs=NP,
                judge='obs_of', beta=100.0, mechanism='replay one-shot links Wl[j,i]+=1, a2=softmax(beta*(scores+w*Wl^T a1))'), rows=out),
          open(os.path.join(ROOT, 'results', 'exp28_btsp_chain.json'), 'w', encoding='utf-8'), indent=1)
print('saved results/exp28_btsp_chain.json', flush=True)
