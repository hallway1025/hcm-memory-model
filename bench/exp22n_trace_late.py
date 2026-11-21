# -*- coding: utf-8 -*-
'''exp22n_trace_late.py  真实 m.query 内部 spy（late 项）：ret 实参/a_hop vs 内联
'''
import csv, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from hcm import HCM, Config
from hcm.ops import unit, softmax_beta
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
orig_ret = m.E.retrieve
calls = []
def spy(q0, L=None, beta=None, dense=None, **k):
    z, a = orig_ret(q0, L=L, beta=beta, dense=dense, **k)
    calls.append(dict(L=L, beta=beta, dense=None if dense is None else np.array(dense, copy=True), a=np.array(a, copy=True)))
    return z, a
m.E.retrieve = spy
for j, i in enumerate(late):
    ti = slot_of[i]
    q = encode_deg(texts[i], 0.3, 81000 + j)
    calls.clear()
    o = m.query(q, do_rif=False)
    cc = calls[0]
    # inline reference
    x = unit(np.asarray(q, dtype=np.float32))
    s_pre = m.enc.pre(x)
    q0 = m.enc.encode_from_pre(s_pre)
    z2, a2 = orig_ret(q0, L=3, beta=100.0, dense=s_pre)
    dc = float(cc["dense"] @ s_pre / (np.linalg.norm(cc["dense"]) * np.linalg.norm(s_pre) + 1e-12)) if cc["dense"] is not None else -1.0
    print(json.dumps(dict(i=i, true=ti, act=o.get("action"), idx=int(o["idx"]),
      ret_L=cc["L"], ret_beta=cc["beta"], dense_cos=round(dc, 6),
      ahop_arg_real=int(np.argmax(cc["a"])), ahop_arg_inline=int(np.argmax(a2)),
      ahop_true_real=round(float(cc["a"][ti]), 4), m1=round(float(o["m1"]), 4)), ensure_ascii=False), flush=True)