# -*- coding: utf-8 -*-
'''exp22m_inline_query_trace.py  逐行内联 m.query（L=3, lam=1.0），对 late 项打印全级联
'''
import csv, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from hcm import HCM, Config
from hcm.ops import unit
from hcm.ops import softmax_beta
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = 512; N = 2000
NANO = re.compile(r'\[[^\]]{1,8}\]')
def _clean(t): return NANO.sub('', re.sub(r'\s+', '', t))
with open(os.path.join(ROOT,'bench','corpus','train__peer-01.csv'), encoding='utf-8-sig', newline='') as f:
    rd = csv.reader(f); next(rd)
    texts = [r[2] for r in rd][:N]
V = sorted({ch for t in texts for ch in _clean(t)} | {'\ufffd'})
VOC = {ch: j for j, ch in enumerate(V)}
rng = np.random.default_rng(7)
W = rng.standard_normal((D, len(V)), dtype=np.float32)
W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-9
def encode(t):
    chars = list(_clean(t)) or ['\ufffd']
    return unit(W[:, [VOC[c] for c in chars]].sum(1))
def encode_deg(t, frac, seed):
    r = np.random.default_rng(seed)
    chs = list(_clean(t))
    if len(chs) > 1:
        for j in r.choice(len(chs), size=max(1, int(len(chs)*frac)), replace=False):
            chs[j] = ''
    chs = [c for c in chs if c] or ['\ufffd']
    return unit(W[:, [VOC[c] for c in chs]].sum(1))
def char_ov(a, b):
    sa, sb = set(_clean(a)), set(_clean(b))
    return len(sa & sb) / max(1, len(sa)) if sa else 0.0
X = np.asarray([encode(t) for t in texts], dtype=np.float32)
m = HCM(Config(d_x=D, d_c=D, a=0.02, capacity=N+10, beta=100.0, L=3, use_resid=False, lam=1.0, seed=0))
for i in range(N): m.observe(X[i])
slot_of = {}
for s in range(int(m.E.n)): slot_of[int(m.E.t[s])] = s
iso = []
for i in range(N):
    best = 0.0
    for j in range(N):
        if j != i:
            v = char_ov(texts[i], texts[j])
            if v > best: best = v
    iso.append(best)
late = [i for i in range(N) if iso[i] < 0.5 and i in slot_of][40:55]
for j, i in enumerate(late):
    ti = slot_of[i]
    q = encode_deg(texts[i], 0.3, 71000 + j)
    x = unit(np.asarray(q, dtype=np.float32))
    Mn = m.E.M[:, : m.E.n]
    s_pre = m.enc.pre(x)
    s_ev = m.enc.raw(x)
    q0 = m.enc.encode_from_pre(s_pre)
    ed = Mn.T @ s_ev
    es = Mn.T @ q0
    ev = m.cfg.w_dense * ed / (np.linalg.norm(ed) + 1e-12) + (1.0 - m.cfg.w_dense) * es / (np.linalg.norm(es) + 1e-12)
    a_sm = softmax_beta(ev, m.cfg.beta)
    z, a_hop = m.E.retrieve(q0, L=3, beta=m.cfg.beta, dense=s_pre)
    a = a_hop * m.E.sup[: m.E.n]
    a = a / (a.sum() + 1e-12)
    shat = unit(m.sem.forward(x))
    s_raw = Mn.T @ shat
    s_z = (s_raw - s_raw.mean()) / (s_raw.std() + 1e-8)
    s_n = 1.0 / (1.0 + np.exp(-s_z))
    f = m.cfg.lam * a + (1.0 - m.cfg.lam) * s_n
    top1 = int(np.argmax(f))
    am = lambda v: int(np.argmax(v))
    print(json.dumps(dict(i=i, true=ti, arg_ev=am(ev), arg_ahop=am(a_hop), arg_a=am(a), arg_sn=am(s_n), arg_f=top1,
      a_true=round(float(a[ti]), 4), sn_true=round(float(s_n[ti]), 4), sn_max=round(float(s_n.max()), 4)), ensure_ascii=False), flush=True)