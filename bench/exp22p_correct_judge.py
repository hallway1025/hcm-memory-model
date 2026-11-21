# -*- coding: utf-8 -*-
'''exp22p_correct_judge.py  v2 重跑（拍板②）：正确判据 obs_of[slot]==i，lam 0.6/1.0 x deg 0.3/0.5；语料=peer-02 v2 前 2000 条
exp22j/k 的判据把 slot 当 obs 键查 slot_of —— 判据 bug；exp22g 判据对但 lam=0 打分坏。
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
iso = []
for i in range(N):
    best = 0.0
    for j in range(N):
        if j != i:
            v = char_ov(texts[i], texts[j])
            if v > best: best = v
    iso.append(best)
res = {}
for LAM in (1.0, 0.6):
    m = HCM(Config(d_x=D, d_c=D, a=0.02, capacity=N+10, beta=100.0, L=3, use_resid=False, lam=LAM, seed=0))
    for i in range(N): m.observe(X[i])
    obs_of = {}
    for s in range(int(m.E.n)): obs_of[s] = int(m.E.t[s])
    cand = [i for i in range(N) if iso[i] < 0.5 and i in set(obs_of.values())][:200]
    for DEG in (0.3, 0.5):
        hit = 0
        for j, i in enumerate(cand):
            q = encode_deg(texts[i], DEG, 41000 + j)
            o = m.query(q, do_rif=False)
            if obs_of.get(int(o["idx"])) == i: hit += 1
        res["lam%s_deg%s" % (str(LAM).replace(".", ""), str(DEG).replace(".", ""))] = dict(n=len(cand), hit=hit)
        print(json.dumps(dict(lam=LAM, deg=DEG, n=len(cand), hit=hit), ensure_ascii=False), flush=True)
json.dump(res, open(os.path.join(ROOT, "results", "exp22p_correct_judge.json"), "w", encoding="utf-8"), indent=1)
print("saved results/exp22p_correct_judge.json", flush=True)