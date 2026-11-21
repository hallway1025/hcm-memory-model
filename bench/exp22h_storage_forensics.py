# -*- coding: utf-8 -*-
'''exp22h_storage_forensics v2
(a) store fidelity: M[:,slot] vs write-time h  (b) pipeline identity: h vs enc.encode(X[obs])
(c) stepwise pre/encode_from_pre vs encode
'''
import csv, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from hcm import HCM, Config
from hcm.ops import unit
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = 512; N = 120
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
X = np.asarray([encode(t) for t in texts], dtype=np.float32)
m = HCM(Config(d_x=D, d_c=D, a=0.02, capacity=N+10, beta=100.0, L=3, use_resid=False, lam=0.0, seed=0))
cap = []
orig_write = m.E.write
def spy(h, *a, **k):
    before = int(m.E.n)
    out = orig_write(h, *a, **k)
    if int(m.E.n) == before + 1:
        cap.append((before, np.array(h, copy=True)))
    return out
m.E.write = spy
for i in range(N): m.observe(X[i])
M = m.E.M
print(json.dumps(dict(written=int(m.E.n), captured=len(cap), M_shape=list(M.shape)), ensure_ascii=False), flush=True)
cos = lambda a, b: float(np.float32(a) @ np.float32(b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
fid, ident, mm = [], [], []
for s, h in cap:
    col = M[:, s]
    i = int(m.E.t[s])
    cf = cos(col, h)
    ci = cos(h, m.enc.encode(X[i]))
    fid.append(cf); ident.append(ci)
    if cf < 0.9 or ci < 0.9: mm.append([s, i, round(cf, 3), round(ci, 3)])
a = float(np.mean(fid)); b = float(np.mean(ident))
print(json.dumps(dict(mean_store_fidelity=round(a, 4), mean_pipeline_identity=round(b, 4), n_mismatch=len(mm)), ensure_ascii=False), flush=True)
if mm[:5]: print("first mismatches:", mm[:5], flush=True)
x0 = X[0]
pre = m.enc.pre(x0)
step = m.enc.encode_from_pre(pre)
enc0 = m.enc.encode(x0)
print(json.dumps(dict(stepwise_vs_encode=round(cos(step, enc0), 4), step_vs_captured=round(cos(step, cap[0][1]), 4), step_support=int((step > 0).sum()), cap_support=int((cap[0][1] > 0).sum())), ensure_ascii=False), flush=True)
json.dump(dict(mean_store_fidelity=a, mean_pipeline_identity=b, mismatches=mm[:20]),
          open(os.path.join(ROOT, "results", "exp22h_storage_forensics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved results/exp22h_storage_forensics.json", flush=True)