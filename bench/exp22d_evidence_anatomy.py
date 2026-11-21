# -*- coding: utf-8 -*-
'''exp22d_evidence_anatomy.py —— L-034 仪器化：真实语料证据通道逐步解剖。'''
import csv, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from hcm import HCM, Config
from hcm.ops import unit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = 512
NANO = re.compile(r'\[[^\]]{1,8}\]')
def _clean(t): return NANO.sub('', re.sub(r'\s+', '', t))
def load(split, sess):
    p = os.path.join(ROOT, 'bench', 'corpus', '%s__%s.csv' % (split, sess))
    with open(p, encoding='utf-8-sig', newline='') as f:
        rd = csv.reader(f); next(rd)
        return [r[2] for r in rd]
texts = load('train', 'peer-01')[:2000]
V = sorted({ch for t in texts for ch in _clean(t)} | {'\ufffd'})
VOC = {ch: j for j, ch in enumerate(V)}
rng = np.random.default_rng(7)
W = rng.standard_normal((D, len(V)), dtype=np.float32)
W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-9
def enc(chars):
    idx = [VOC[c] for c in chars] if chars else [VOC['\ufffd']]
    return unit(W[:, idx].sum(1))
def encode(t): return enc(list(_clean(t)))
def encode_deg(t, frac, seed):
    r = np.random.default_rng(seed)
    chs = list(_clean(t))
    if len(chs) > 1:
        for j in r.choice(len(chs), size=max(1, int(len(chs)*frac)), replace=False):
            chs[j] = ''
    return enc([c for c in chs if c])
def char_ov(a, b):
    sa, sb = set(_clean(a)), set(_clean(b))
    return len(sa & sb)/max(1, len(sa)) if sa else 0.0
X = np.asarray([encode(t) for t in texts], dtype=np.float32)
N = len(X)
iso = np.zeros(N)
for i in range(N):
    iso[i] = max(char_ov(texts[i], texts[j]) for j in range(N) if j != i)
sel = [i for i in range(N) if iso[i] < 0.5][:1000]
probes = sel[:40]
m = HCM(Config(d_x=D, d_c=D, a=0.02, capacity=N+10, beta=100.0, L=3,
               use_resid=False, lam=0.0, seed=0))
for i in range(N): m.observe(X[i])
enc_ = m.enc
def code_of(x): return enc_.encode(x)
hits = 0
for j, i in enumerate(probes):
    q = encode_deg(texts[i], 0.3, 9000+j)
    hq = code_of(q); hi = code_of(X[i])
    o = m.query(q, do_rif=False)
    w = int(o['idx'])
    ok = w == i
    hits += ok
    xcos = float(X[i] @ q); hcos = float(hi @ hq)
    line = dict(i=i, true=texts[i][:14], win=texts[w][:14], ok=bool(ok),
                xcos=round(xcos,3), hcos=round(hcos,3))
    s_pre = enc_.pre(q)
    qq = enc_.encode_from_pre(s_pre)
    Mn = m.E.M[:, : m.E.n]
    ed = Mn.T @ s_pre; es = Mn.T @ qq
    w = m.cfg.w_dense
    ev = w*ed/(np.linalg.norm(ed)+1e-12) + (1.0-w)*es/(np.linalg.norm(es)+1e-12)
    rank = int((ev > ev[i]).sum())
    line.update(ev_true=round(float(ev[i]),5), ev_max=round(float(ev.max()),5), ev_rank=rank,
                es_self=round(float(es[i]/(np.linalg.norm(es)+1e-12)),5),
                ed_self=round(float(ed[i]/(np.linalg.norm(ed)+1e-12)),5),
                top1=texts[int(np.argmax(ev))][:14])
    print(json.dumps(line, ensure_ascii=False), flush=True)
print('hits', hits, '/', len(probes), flush=True)