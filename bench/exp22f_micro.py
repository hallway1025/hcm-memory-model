# -*- coding: utf-8 -*-
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
def encode(t):
    idx = [VOC[c] for c in _clean(t)] if _clean(t) else [VOC['\ufffd']]
    return unit(W[:, idx].sum(1))
X = np.asarray([encode(t) for t in texts], dtype=np.float32)
N = len(X)
m = HCM(Config(d_x=D, d_c=D, a=0.02, capacity=N+10, beta=100.0, L=3, use_resid=False, lam=0.0, seed=0))
for i in range(N): m.observe(X[i])
enc = m.enc; E = m.E; Mn = E.M[:, :E.n]
print("theta norm", float(np.linalg.norm(enc.theta)), "b_e", float(np.linalg.norm(enc.b_e)), flush=True)
for i in [93, 96, 102, 117, 134]:
    x = X[i]
    h_store = Mn[:, i] if i < E.n else None
    h_rec = enc.encode(x)
    s_raw = enc.raw(x)
    ed0 = Mn.T @ s_raw
    q = encode(texts[i])  # exact same as X[i]
    line = dict(i=i, t=texts[i][:10], written=bool(h_store is not None and np.abs(h_store).sum()>0),
        cos_store_rec=round(float(h_store @ h_rec),4) if h_store is not None else None,
        ed0_self=round(float(ed0[i]),4), ed0_rank=int((ed0>ed0[i]).sum()),
        ed0_max=round(float(ed0.max()),4), argmax=int(np.argmax(ed0)),
        ed0_at_exact_self_if_written=round(float(ed0[i]),4))
    print(json.dumps(line, ensure_ascii=False), flush=True)
# deg 0.3 for i=93
i = 93
r = np.random.default_rng(9000)
chs = list(_clean(texts[i]))
for j in r.choice(len(chs), size=max(1, int(len(chs)*0.3)), replace=False): chs[j]=""
idx = [VOC[c] for c in chs if c] if any(chs) else [VOC['\ufffd']]
q = unit(W[:, idx].sum(1))
ed = Mn.T @ enc.raw(q)
es = Mn.T @ enc.encode(q)
print(json.dumps(dict(i=i, deg=0.3, xcos=round(float(X[i]@q),3),
  ed_self=round(float(ed[i]),4), ed_rank=int((ed>ed[i]).sum()),
  es_self=round(float(es[i]),4), es_rank=int((es>es[i]).sum())), ensure_ascii=False), flush=True)