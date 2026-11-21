#!/usr/bin/env python3
'''exp27_temporal_chain.py -- temporal chain recall on real corpus (B3 baseline).
Query = message i (written). Measure: (a) self-recall hit@1 (replicates exp22p at larger N);
(b) NEXT-message recall: rank of slot(j), j=i+1 in time, gap<=30s, different senders;
(c) shuffled-pair control. Hypothesis: conversational continuity -> code-space smoothness
-> next message measurably closer than random (CA3 temporal association, Buzsaki 2015).
Judge: obs_of (L-035). beta=100 (D-035). theta pinned 0 (D-038). Random-fallback embed (exp23e).'''
import sys, os, json, time, csv
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESS = sys.argv[1] if len(sys.argv) > 1 else 'peer-01'
NP = int(sys.argv[2]) if len(sys.argv) > 2 else 200

import re
_NANO = re.compile(r'\[[^\]]{1,12}\]')
def _clean(s):
    s = _NANO.sub(' ', str(s))
    return re.sub(r'\s+', ' ', s).strip()

# ---------- load corpus ----------
path = os.path.join(ROOT, 'bench', 'corpus', 'train__%s.csv' % SESS)
rows = list(csv.reader(open(path, encoding='utf-8-sig')))
rows = [r for r in rows[1:] if len(r) >= 3]
msgs = [r[2] for r in rows]
times = [r[0] for r in rows]
senders = [r[1] for r in rows]
print('corpus %s: %d msgs' % (SESS, len(msgs)), flush=True)

# ---------- embedding (exp23e randfallback S256 if available else R512) ----------
emb_path = os.path.join(ROOT, 'results', 'exp23_semantic_embed.npz')
V = sorted({c for m in msgs for c in _clean(m)} | {'\ufffd'})
Vi = {c: i for i, c in enumerate(V)}
Vn = len(V)
def load_emb():
    if os.path.exists(emb_path):
        z = np.load(emb_path, allow_pickle=True)
        if 'E' in z and 'chars' in z:
            E = z['E']; chars = [str(c) for c in z['chars']]
            V = chars + ([] if '\\ufffd' in chars else ['\\ufffd'])
            Vi = {ch: i for i, ch in enumerate(V)}
            M = E.T.copy()
            if '\\ufffd' not in chars:
                r0 = np.random.default_rng(99)
                M = np.hstack([M, r0.standard_normal((M.shape[0], 1))])
                M[:, -1] /= np.linalg.norm(M[:, -1])
            print('embed: S256 rows', M.shape, flush=True)
            return M, Vi
    r = np.random.default_rng(7)
    M = r.standard_normal((512, Vn))
    M /= np.linalg.norm(M, axis=0, keepdims=True)
    print('embed: random R512', M.shape, flush=True)
    return M, Vi
    if os.path.exists(emb_path):
        z = np.load(emb_path, allow_pickle=True)
        if 'M' in z and z['M'].shape[0] == Vn:
            M = z['M']; print('embed: S256 loaded', M.shape, flush=True); return M
    r = np.random.default_rng(7)
    M = r.standard_normal((512, Vn))
    M /= np.linalg.norm(M, axis=0, keepdims=True)
    print('embed: random R512', M.shape, flush=True)
    return M
M, Vi = load_emb()
col = False  # rows now char-indexed via Vi from loader
def x_of(s):
    idx = [Vi.get(c, Vi['\ufffd']) for c in _clean(s)] or [Vi['\ufffd']]
    v = M[:, idx].sum(1)
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.zeros(M.shape[0], dtype=np.float32)

D_x = M.shape[0]
# random-fallback for zero-support chars handled implicitly: \ufffd column exists;
# emoji-only messages fall to \ufffd (nonzero) so nothing is dead (exp23e lesson).

# ---------- store all ----------
cfg = Config(d_x=D_x, d_c=D_x, a=0.02, capacity=len(msgs) + 10, beta=100.0, L=3, use_resid=False, seed=0)
m = HCM(cfg)
t0 = time.time()
for i, s in enumerate(msgs):
    m.observe(x_of(s))
    if (i + 1) % 2000 == 0:
        print('  write %d/%d %.0fs' % (i + 1, len(msgs), time.time() - t0), flush=True)
nw = int(m.E.n)
obs_of = {int(m.E.t[j]): j for j in range(nw)}
print('written %d/%d' % (nw, len(msgs)), flush=True)

# ---------- pairs: consecutive, gap<=30s, diff sender, both writable ----------
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
print('candidate pairs: %d' % len(pairs), flush=True)
rng = np.random.default_rng(2700)
rng.shuffle(pairs)
pairs = pairs[:NP]

def degrade(x, frac, seed):
    r = np.random.default_rng(seed)
    y = x.copy()
    nz = np.where(np.abs(y) > 1e-6)[0]
    y[r.choice(nz, size=max(1, int(len(nz) * frac)), replace=False)] = 0.0
    return unit(y)

out = {}
for beta in (100.0, 500.0):
    for deg in (0.0, 0.3):
        self_hit = next5 = next20 = 0
        cranks = []
        ctrl5 = 0
        ranks = []
        used = 0
        for i in pairs:
            j = i + 1
            q = x_of(msgs[i])
            if deg > 0:
                q = degrade(q, deg, i)
            pre = m.enc.raw(q)
            _, a = m.E.retrieve(m.enc.encode_from_pre(pre), L=3, beta=beta, dense=pre)
            k = int(np.argmax(a))
            used += 1
            if k == obs_of[i]: self_hit += 1
            rk = int(np.where(np.argsort(-a) == obs_of[j])[0][0])
            ranks.append(rk)
            if rk < 5: next5 += 1
            if rk < 20: next20 += 1
            cslot = int(rng.integers(0, nw))
            while cslot in (obs_of[i], obs_of[j]):
                cslot = int(rng.integers(0, nw))
            rkc = int(np.where(np.argsort(-a) == cslot)[0][0])
            cranks.append(rkc)
            if rkc < 5: ctrl5 += 1
        ranks = np.array(ranks)
        key = 'b%g_deg%g' % (beta, deg * 100)
        out[key] = dict(pairs=used, self_hit1=self_hit, self_rate=round(self_hit / used, 4),
                        next_hit5=next5, next_rate5=round(next5 / used, 4),
                        next_hit20=next20, next_rate20=round(next20 / used, 4),
                        ctrl_hit5=ctrl5, ctrl_rate5=round(ctrl5 / used, 4),
                        next_rank_med=float(np.median(ranks)), next_rank_mean=round(float(ranks.mean()), 1),
                        ctrl_rank_med=float(np.median(cranks)), ctrl_rank_mean=round(float(np.mean(cranks)), 1),
                        lift_med=round(float(np.median(cranks)) / max(1.0, float(np.median(ranks))), 2))
        print('%s self=%d/%d next@5=%d (%.3f) ctrl@5=%d (%.3f) med_rank=%g ctrl_med=%g' % (key, self_hit, used,
              next5, out[key]['next_rate5'], ctrl5, out[key]['ctrl_rate5'], out[key]['next_rank_med'],
              out[key]['ctrl_rank_med']), flush=True)

json.dump(dict(config=dict(session=SESS, n_msgs=len(msgs), n_written=nw, d_x=D_x, pairs=NP,
                judge='obs_of', theta='pinned_zero_D-038', embed=('S256' if col else 'R512')), rows=out),
          open(os.path.join(ROOT, 'results', 'exp27_temporal_chain.json'), 'w', encoding='utf-8'), indent=1)
print('saved results/exp27_temporal_chain.json', flush=True)
