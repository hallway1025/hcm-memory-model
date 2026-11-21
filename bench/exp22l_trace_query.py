# -*- coding: utf-8 -*-
'''exp22l_trace_query.py  monkeypatch E.retrieve 抓取 m.query 内部实参/输出，与手工对比
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
X = np.asarray([encode(t) for t in texts], dtype=np.float32)
m = HCM(Config(d_x=D, d_c=D, a=0.02, capacity=N+10, beta=100.0, L=3, use_resid=False, lam=1.0, seed=0))
print("cfg:", json.dumps(dict(beta=m.cfg.beta, L=m.cfg.L, lam=m.cfg.lam, use_resid=m.cfg.use_resid, w_dense=m.cfg.w_dense, dense_raw=m.cfg.dense_raw, use_ctx_bias=m.cfg.use_ctx_bias)), flush=True)
for i in range(N): m.observe(X[i])
slot_of = {}
for s in range(int(m.E.n)): slot_of[int(m.E.t[s])] = s
Mn = m.E.M[:, : int(m.E.n)]
cap_calls = []
orig_ret = m.E.retrieve
def spy_ret(q0, L=None, beta=None, dense=None, **k):
    qs, ah = orig_ret(q0, L=L, beta=beta, dense=dense, **k)
    cap_calls.append(dict(L=L, beta=beta, q0=np.array(q0, copy=True), ah=np.array(ah, copy=True), dense=np.array(dense, copy=True)))
    return qs, ah
m.E.retrieve = spy_ret
early = sorted(slot_of.keys())[:10]
for j, i in enumerate(early):
    q = encode_deg(texts[i], 0.3, 61000 + j)
    cap_calls.clear()
    o = m.query(q, do_rif=False)
    ti = slot_of[i]
    s_pre = m.enc.pre(q)
    q0_hand = m.enc.encode_from_pre(s_pre)
    _, ah_hand = orig_ret(q0_hand, L=3, beta=100.0, dense=s_pre)
    cc = cap_calls[0]
    q0_sim = float(cc["q0"] @ q0_hand / (np.linalg.norm(cc["q0"]) * np.linalg.norm(q0_hand) + 1e-12))
    ah_arg = int(np.argmax(cc["ah"][:len(ah_hand)])) if len(cc["ah"]) >= len(ah_hand) else int(np.argmax(cc["ah"]))
    print(json.dumps(dict(i=i, true_slot=ti, n_ret_calls=len(cap_calls), ret_L=cc["L"], ret_beta=cc["beta"],
      cos_q0=round(q0_sim, 4), ah_argmax=ah_arg, ah_hand_argmax=int(np.argmax(ah_hand)),
      query_idx=int(o["idx"]), ah_len=len(cc["ah"]), hand_len=len(ah_hand)), ensure_ascii=False), flush=True)