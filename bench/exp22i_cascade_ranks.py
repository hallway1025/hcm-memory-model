# -*- coding: utf-8 -*-
'''exp22i_cascade_ranks.py  ed -> es -> ev -> a_hop(L=3) -> a_hop0(L=0) -> f
'''
import csv, json, os, re, sys, statistics
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
X = np.asarray([encode(t) for t in texts], dtype=np.float32)
m = HCM(Config(d_x=D, d_c=D, a=0.02, capacity=N+10, beta=100.0, L=3, use_resid=False, lam=0.6, seed=0))
cap = []
orig_write = m.E.write
def spy(h, *a, **k):
    before = int(m.E.n)
    out = orig_write(h, *a, **k)
    if int(m.E.n) == before + 1: cap.append(before)
    return out
m.E.write = spy
for i in range(N): m.observe(X[i])
slot_of = {}
for s in range(int(m.E.n)): slot_of[int(m.E.t[s])] = s
obs_written = set(slot_of.keys())
Mn = m.E.M[:, : int(m.E.n)]
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
def rank_of(scores, ti):
    order = np.argsort(-scores)
    return int(np.where(order == ti)[0][0])
rows = []
sel = sorted(obs_written)[:40]
for j, i in enumerate(sel):
    ti = slot_of[i]
    q = encode_deg(texts[i], 0.3, 31000 + j)
    s_pre = m.enc.pre(q)
    s_ev = m.enc.raw(q)
    q0 = m.enc.encode_from_pre(s_pre)
    ed = Mn.T @ s_ev
    es = Mn.T @ q0
    ev = ed / (np.linalg.norm(ed) + 1e-12) + es / (np.linalg.norm(es) + 1e-12)
    _, a_hop = m.E.retrieve(q0, L=3, beta=100.0, dense=s_pre)
    _, a_hop0 = m.E.retrieve(q0, L=0, beta=100.0, dense=s_pre)
    rows.append(dict(i=i, xcos=round(cos(q, X[i]), 3), hcos=round(cos(q0, Mn[:, ti]), 3),
      r_ed=rank_of(ed, ti), r_es=rank_of(es, ti), r_ev=rank_of(ev, ti),
      r_ahop=rank_of(a_hop, ti), r_ahop0=rank_of(a_hop0, ti)))
print(json.dumps(dict(n_slots=int(m.E.n), n_probes=len(rows)), ensure_ascii=False), flush=True)
for r in rows[:12]: print(json.dumps(r), flush=True)
def med(k): return statistics.median([r[k] for r in rows])
print(json.dumps(dict(median=dict(r_ed=med("r_ed"), r_es=med("r_es"), r_ev=med("r_ev"), r_ahop=med("r_ahop"), r_ahop0=med("r_ahop0")),
  mean_hcos=round(statistics.mean([r["hcos"] for r in rows]), 3)), ensure_ascii=False), flush=True)
json.dump(rows, open(os.path.join(ROOT, "results", "exp22i_cascade_ranks.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved results/exp22i_cascade_ranks.json", flush=True)