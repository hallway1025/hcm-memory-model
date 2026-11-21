# -*- coding: utf-8 -*-
# exp23b v2: 严格 exp23 同口径的 S256 检索失败取证（NANO clean / V1032 / iso 探针 / deg seed 41000+j）
import csv, json, os, re, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from hcm import HCM, Config
from hcm.ops import unit
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 2000; SESS = 'peer-01'
NANO = re.compile(r'\[[^\]]{1,8}\]')
def _clean(t): return NANO.sub('', re.sub(r'\s+', '', t))
with open(os.path.join(ROOT,'bench','corpus','train__%s.csv' % SESS), encoding='utf-8-sig', newline='') as f:
    rd = csv.reader(f); next(rd)
    texts = [r[2] for r in rd][:N]
msgs = [_clean(t) for t in texts]
V = sorted({ch for t in msgs for ch in t} | {'\ufffd'})
Vn = len(V); VOC = {ch: j for j, ch in enumerate(V)}
print('V =', Vn, flush=True)
def learn_embedding_ppmi_svd(msgs_in, r=256, win=3, guard=False):
    C = np.zeros((Vn, Vn), dtype=np.float64)
    for t in msgs_in:
        chs = [VOC[c] for c in t if (c in VOC or not guard)]
        if not chs: continue
        for i, a in enumerate(chs):
            for b in chs[max(0, i-win):i+win+1]:
                C[a, b] += 1.0
    tot = C.sum()
    pw = C.sum(1) / tot
    P = C / tot
    with np.errstate(divide='ignore'):
        pmi = np.log2(P / (pw[:, None] * pw[None, :] + 1e-12) + 1e-12)
    ppmi = np.maximum(0.0, pmi)
    U, S, Vt = np.linalg.svd(ppmi)
    r = min(r, Vn - 1)
    E = (U[:, :r] * np.sqrt(S[:r])).astype(np.float32)
    print('PPMI nnz=%.3f, r=%d' % (float((ppmi > 0).mean()), r), flush=True)
    return E
def random_proj(dr, seed=7):
    W = np.random.default_rng(seed).standard_normal((dr, Vn), dtype=np.float32)
    W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-9
    return W
def x_of(chars, E_or_W):
    idx = [VOC[c] for c in chars] if chars else [VOC['\ufffd']]
    if E_or_W.shape[1] == Vn: return unit(E_or_W[:, idx].sum(1))
    return unit(E_or_W[idx].sum(0).astype(np.float32))
def encode_with(M, t): return x_of(list(_clean(t)), M)
def encode_deg_with(M, t, frac, seed):
    r = np.random.default_rng(seed)
    chs = list(_clean(t))
    if len(chs) > 1:
        for j in r.choice(len(chs), size=max(1, int(len(chs)*frac)), replace=False):
            chs[j] = ''
    return x_of([c for c in chs if c], M)
t0 = time.time()
iso = []
for i in range(N):
    best = 0.0
    sa = set(msgs[i])
    for j in range(N):
        if j != i:
            v = len(sa & set(msgs[j])) / max(1, len(sa)) if sa else 0.0
            if v > best: best = v
    iso.append(best)
print('iso done %.1fs' % (time.time()-t0), flush=True)

co_texts = []
import glob as _glob, csv as _csv
for _p in sorted(_glob.glob(os.path.join(ROOT, 'bench', 'corpus', 'train__*.csv'))):
    with open(_p, encoding='utf-8-sig', newline='') as _f:
        _rd = _csv.reader(_f); next(_rd)
        co_texts.extend(r[2] for r in _rd)
co_msgs = [_clean(x) for x in co_texts]
ARMSF = [('S256_2k', learn_embedding_ppmi_svd(msgs)), ('S256_129k', learn_embedding_ppmi_svd(co_msgs, guard=True))]
rows = []
for tagK, M in ARMSF:
  dr = M.shape[1]
  X = np.asarray([encode_with(M, t) for t in texts], dtype=np.float32)
  m = HCM(Config(d_x=dr, d_c=dr, a=0.02, capacity=N+10, beta=100.0, L=3, use_resid=False, lam=1.0, seed=0))
  for i in range(N): m.observe(X[i])
  nw = int(m.E.n)
  obs_of = {s: int(m.E.t[s]) for s in range(nw)}
  wset = set(obs_of.values())
  slot_of = {o: s for s, o in obs_of.items()}
  cand = [i for i in range(N) if iso[i] < 0.5 and i in wset][:200]
  for j, i in enumerate(cand[:80]):
    o = m.query(encode_deg_with(M, texts[i], 0.3, 41000+j), do_rif=False)
    ti = slot_of.get(i)
    q = encode_deg_with(M, texts[i], 0.3, 41000+j)
    h = m.enc.encode(q)
    hc = float(h @ m.E.M[:, ti]) if ti is not None else float('nan')
    # 真值在手打管线中的名次：ed 通道打分（dense_raw=True 时主导）
    s_raw = m.enc.raw(q)
    ed = m.E.Mn.T @ s_raw if hasattr(m.E, 'Mn') else None
    if ed is not None:
        order = np.argsort(-ed)
        rank_true = int(np.where(order == ti)[0][0]) if ti is not None else -2
        ed_true = float(ed[ti]) if ti is not None else float('nan')
        ed_max_other = float(ed[order[0]] if order[0] != ti else ed[order[1]])
    else:
        rank_true = -9; ed_true = float('nan'); ed_max_other = float('nan')
    rows.append({'arm': tagK, 'probe': i, 'action': o.get('action'), 'idx': o.get('idx'),
                 'm1': round(float(o.get('m1', float('nan'))), 4),
                 'raw_evidence': round(float(o.get('raw_evidence', float('nan'))), 4),
                 'hcos_true': round(hc, 4), 'nnz_h': int((h > 0).sum()),
                 'rank_true_ed': rank_true, 'ed_true': round(ed_true, 4), 'ed_best_other': round(ed_max_other, 4),
                 'msg': texts[i][:12], 'nchars': len(texts[i])})
from collections import Counter
print(Counter([(r['arm'], r['action']) for r in rows]), flush=True)
nf = [r for r in rows if not (isinstance(r['idx'], int) and r['idx'] >= 0 and obs_of.get(r['idx']) == i if False else False)]
bad = [r for r in rows if not (r['idx'] is not None and isinstance(r['idx'], (int, np.integer)) and obs_of.get(int(r['idx']), -1) == [c for c in [r['probe']]][0])]
for r in bad[:30]: print(json.dumps(r, ensure_ascii=False), flush=True)
json.dump(rows, open(os.path.join(ROOT, 'results', 'exp23d_gate_forensics.json'), 'w'), ensure_ascii=False, indent=1)
print('saved results/exp23d_gate_forensics.json', flush=True)