# -*- coding: utf-8 -*-
'''exp22o_rerun200.py  v2 重跑（拍板②）：重跑 exp22j（R512 一臂）+ spy 自检；语料=peer-02 v2 前 2000 条；v1 工件见 git 历史
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
# v2 rerun（拍板② A2，2026-11-09）：v1 peer-01 语料为非现役残留，改 v2 主会话
with open(os.path.join(ROOT,'bench','corpus','train__peer-02.csv'), encoding='utf-8-sig', newline='') as f:
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
cand = [i for i in range(N) if iso[i] < 0.5 and i in slot_of][:200]
hit = miss_examples = 0
bad = []
for j, i in enumerate(cand):
    q = encode_deg(texts[i], 0.3, 41000 + j)
    o = m.query(q, do_rif=False)
    ok = int(o["idx"]) in slot_of and slot_of[int(o["idx"])] == i
    hit += ok
    if not ok and len(bad) < 5: bad.append(dict(i=i, true=slot_of[i], idx=int(o["idx"]), act=o.get("action"), m1=round(float(o["m1"]), 3)))
print(json.dumps(dict(n_probes=len(cand), hit=hit, bad_sample=bad), ensure_ascii=False), flush=True)
json.dump(dict(n_probes=len(cand), hit=hit), open(os.path.join(ROOT, "results", "exp22o_rerun200.json"), "w", encoding="utf-8"))
print("saved", flush=True)