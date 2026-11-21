# -*- coding: utf-8 -*-
'''exp22e_readout_forensics.py —— 精确匹配查询都输 → 读出通道法证。
1) 实际写入数 m.E.n vs 2000（theta_dup 拒写了多少）
2) M 列范数/支撑数分布 vs 消息长度
3) 3 个精确匹配探针：ed/es 的真值分、top5 及其列范数、kwta 返回值形态
'''
import csv, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from hcm import HCM, Config
from hcm.ops import unit, kwta

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
rejected = 0
for i in range(N):
    rec = m.observe(X[i])
    if not rec.get('wrote', True): rejected += 1
print('written', m.E.n, '/', N, 'rejected', rejected, 'theta_dup', m.cfg.theta_dup if hasattr(m.cfg,'theta_dup') else 'n/a', flush=True)
Mn = m.E.M[:, : m.E.n]
cn = np.linalg.norm(Mn, axis=0); sup = (Mn > 0).sum(0)
lens = np.array([max(1, len(_clean(t))) for t in texts[: m.E.n]])
print("colnorm mean/std/min/max", round(float(cn.mean()),3), round(float(cn.std()),3), round(float(cn.min()),3), round(float(cn.max()),3), flush=True)
print("support mean/std", round(float(sup.mean()),1), round(float(sup.std()),1), flush=True)
q1 = np.corrcoef(cn, lens)[0,1]; print("corr(colnorm, len)", round(float(q1),3), flush=True)
enc = m.enc
for i in [100, 109, 160]:
    x = X[i]; s_pre = enc.pre(x)
    hq = enc.encode_from_pre(s_pre)
    ed = Mn.T @ s_pre; es = Mn.T @ hq
    o5 = np.argsort(-ed)[:5]
    print(json.dumps(dict(i=i, t=texts[i][:12], hnorm=round(float(np.linalg.norm(hq)),3),
        h_support=int((hq>0).sum()),
        ed_self=round(float(ed[i]),4), es_self=round(float(es[i]),4),
        ed_rank=int((ed>ed[i]).sum()), es_rank=int((es>es[i]).sum()),
        top5=[dict(j=int(j), t=texts[j][:8], cn=round(float(cn[j]),3), ed=round(float(ed[j]),4), es=round(float(es[j]),4)) for j in o5],
        ), ensure_ascii=False), flush=True)
print('dense_raw', m.cfg.dense_raw if hasattr(m.cfg,'dense_raw') else 'n/a', 'w_dense', m.cfg.w_dense if hasattr(m.cfg,'w_dense') else 'n/a', flush=True)