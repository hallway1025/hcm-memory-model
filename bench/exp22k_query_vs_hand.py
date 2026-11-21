# -*- coding: utf-8 -*-
'''exp22k_query_vs_hand.py  2x2 判决：m.query vs 手工管线 x 前40写项 vs 后段干净200项
'''
import csv, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from hcm import HCM, Config
from hcm.ops import unit
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
Mn = m.E.M[:, : int(m.E.n)]
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
def hand(q, ti):
    s_pre = m.enc.pre(q)
    q0 = m.enc.encode_from_pre(s_pre)
    _, ah = m.E.retrieve(q0, L=3, beta=100.0, dense=s_pre)
    order = np.argsort(-ah)
    return int(np.where(order == ti)[0][0])
early = sorted(slot_of.keys())[:40]
iso = []
for i in range(N):
    best = 0.0
    for j in range(N):
        if j != i:
            v = char_ov(texts[i], texts[j])
            if v > best: best = v
    iso.append(best)
late = [i for i in range(N) if iso[i] < 0.5 and i in slot_of][40:90]
for name, group in (("early40", early), ("late50", late)):
    hq = hh = 0
    for j, i in enumerate(group):
        q = encode_deg(texts[i], 0.3, 51000 + j)
        o = m.query(q, do_rif=False)
        ok_q = int(o["idx"]) in slot_of and slot_of[int(o["idx"])] == i
        ok_h = hand(q, slot_of[i]) == 0
        hq += ok_q; hh += ok_h
    print(json.dumps(dict(group=name, n=len(group), hit_mquery=hq, hit_hand=hh), ensure_ascii=False), flush=True)
print("sup stats:", json.dumps(dict(min=float(m.E.sup[:m.E.n].min()), max=float(m.E.sup[:m.E.n].max()), mean=float(m.E.sup[:m.E.n].mean())), ensure_ascii=False), flush=True)