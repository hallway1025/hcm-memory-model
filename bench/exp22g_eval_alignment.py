# -*- coding: utf-8 -*-
'''exp22g_eval_alignment.py
idx 是存储槽位还是观察序号？E.t[slot] 记录写入时的观察计数，命中应为 E.t[idx]==probe_i
'''
import csv, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from hcm import HCM, Config
from hcm.ops import unit
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = 512
NANO = re.compile(r'\[[^\]]{1,8}\]')
def _clean(t): return NANO.sub('', re.sub(r'\s+', '', t))
with open(os.path.join(ROOT,'bench','corpus','train__peer-01.csv'), encoding='utf-8-sig', newline='') as f:
    rd = csv.reader(f); next(rd)
    texts = [r[2] for r in rd][:2000]
V = sorted({ch for t in texts for ch in _clean(t)} | {'\ufffd'})
VOC = {ch: j for j, ch in enumerate(V)}
rng = np.random.default_rng(7)
W = rng.standard_normal((D, len(V)), dtype=np.float32)
W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-9
def enc_chars(chars):
    idx = [VOC[c] for c in chars] if chars else [VOC['\ufffd']]
    return unit(W[:, idx].sum(1))
def encode(t): return enc_chars(list(_clean(t)))
def encode_deg(t, frac, seed):
    r = np.random.default_rng(seed)
    chs = list(_clean(t))
    if len(chs) > 1:
        for j in r.choice(len(chs), size=max(1, int(len(chs)*frac)), replace=False):
            chs[j] = ''
    return enc_chars([c for c in chs if c])
X = np.asarray([encode(t) for t in texts], dtype=np.float32)
N = len(X)
m = HCM(Config(d_x=D, d_c=D, a=0.02, capacity=N+10, beta=100.0, L=3, use_resid=False, lam=0.0, seed=0))
for i in range(N): m.observe(X[i])
E = m.E
slot_to_obs = {int(s): int(E.t[s]) for s in range(E.n)}
written_obs = sorted(slot_to_obs.values())
wset = set(written_obs)
print(json.dumps(dict(n_slots=E.n, n_obs=N, written_frac=round(E.n/N,3),
  first5=written_obs[:5]), ensure_ascii=False), flush=True)
def char_ov(a, b):
    sa, sb = set(_clean(a)), set(_clean(b))
    return len(sa & sb)/max(1, len(sa)) if sa else 0.0
iso = np.zeros(N)
for i in range(N):
    iso[i] = max(char_ov(texts[i], texts[j]) for j in range(N) if j != i)
cand = [i for i in range(N) if iso[i] < 0.5 and i in wset][:200]
hit_new = hit_old = 0
for j, i in enumerate(cand):
    q = encode_deg(texts[i], 0.3, 12000+j)
    o = m.query(q, do_rif=False)
    idx = int(o['idx'])
    if idx >= 0 and slot_to_obs.get(idx) == i: hit_new += 1
    if idx == i: hit_old += 1
print(json.dumps(dict(n_probes=len(cand), hit_aligned=hit_new, hit_misaligned=hit_old), ensure_ascii=False), flush=True)
res = dict(n_slots=E.n, n_probes=len(cand), hit_aligned=hit_new, hit_misaligned=hit_old)
json.dump(res, open(os.path.join(ROOT,"results","exp22g_eval_alignment.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved results/exp22g_eval_alignment.json", flush=True)