# -*- coding: utf-8 -*-
'''exp22j_endtoend_fixed.py  判决二：m.query 端到端 + 对齐判据 + lam=0.6
若高命中 → 真实语料"全灭" = 评测错位 + lam=0 语义独裁 两个伪影叠加
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
iso = []
for i in range(N):
    best = 0.0
    for j in range(N):
        if j != i:
            v = char_ov(texts[i], texts[j])
            if v > best: best = v
    iso.append(best)

for LAM in (0.6, 1.0):
    m = HCM(Config(d_x=D, d_c=D, a=0.02, capacity=N+10, beta=100.0, L=3, use_resid=False, lam=LAM, seed=0))
    for i in range(N): m.observe(X[i])
    slot_of = {}
    for s in range(int(m.E.n)): slot_of[int(m.E.t[s])] = s
    cand = [i for i in range(N) if iso[i] < 0.5 and i in slot_of][:200]
    hit = act_ok = 0
    for j, i in enumerate(cand):
        q = encode_deg(texts[i], 0.3, 41000 + j)
        o = m.query(q, do_rif=False)
        if int(o["idx"]) in slot_of and slot_of[int(o["idx"])] == i: hit += 1
        if o.get("action") in ("recall", "separate"): act_ok += 1
    print(json.dumps(dict(lam=LAM, n_slots=int(m.E.n), n_probes=len(cand), hit_aligned=hit, acted=act_ok), ensure_ascii=False), flush=True)
    json.dump(dict(lam=LAM, n_probes=len(cand), hit_aligned=hit, acted=act_ok),
              open(os.path.join(ROOT, "results", "exp22j_lam%s.json" % str(LAM).replace(".", "")), "w", encoding="utf-8"))
print("saved exp22j results", flush=True)