# -*- coding: utf-8 -*-
'''exp22c_identifiability.py —— 真实语料检索失败的可识别性解剖。

假说：exp21/22b 的 0-1/200 不是编码器/吸引子坏了，而是任务在近重复家族上不可识别
（"哈哈"退火后匹配所有哈哈变体；真值不一定排第一）。任何相似度检索都有这个极限。
仪器（N=2000 库，lam=0，theta0，200 探针 deg0.3）：
  对每个失败探针记录：真值文本、获胜者文本、字符重叠率、证据分数
  分类：家族混淆（胜者与真值字符重叠>=0.5）/ 真失败 / 弃权
  另报：无近重复子集（真值与其他消息最大字符重叠<0.5）的 hit@1 —— 干净条件下的真实水平'''
import csv, json, os, re, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from hcm import HCM, Config
from hcm.ops import unit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = 512
SESS = "peer-01"
NANO = re.compile(r'\[[^\]]{1,8}\]')
def _clean(t): return NANO.sub('', re.sub(r'\s+', '', t))
def load(split):
    p = os.path.join(ROOT, 'bench', 'corpus', '%s__%s.csv' % (split, SESS))
    with open(p, encoding='utf-8-sig', newline='') as f:
        rd = csv.reader(f); next(rd)
        return [(t, s, c) for t, s, c in rd]
rows = load('train'); texts_all = [r[2] for r in rows]
texts = texts_all[:2000]
rows_te = load('test')
V = sorted({ch for tl in (texts, [r[2] for r in rows_te]) for t in tl for ch in _clean(t)} | {'\ufffd'})
VOC = {ch: j for j, ch in enumerate(V)}
rng = np.random.default_rng(7)
W = rng.standard_normal((D, len(V)), dtype=np.float32)
W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-9
def enc_chars(chars):
    idx = [VOC[c] for c in chars] if chars else [VOC['\ufffd']]
    return unit(W[:, idx].sum(1))
def encode(t): return enc_chars(list(_clean(t)))
def encode_degrade(t, frac, seed):
    r = np.random.default_rng(seed)
    chs = list(_clean(t))
    if len(chs) > 1:
        for j in r.choice(len(chs), size=max(1, int(len(chs)*frac)), replace=False):
            chs[j] = ''
    return enc_chars([c for c in chs if c])
def char_overlap(a, b):
    sa, sb = set(_clean(a)), set(_clean(b))
    if not sa: return 0.0
    return len(sa & sb) / max(1, len(sa))
X = np.asarray([encode(t) for t in texts], dtype=np.float32)
N = len(X)
# 真值的"家族隔离度"：与其他消息的最大字符重叠
tset = [_clean(t) for t in texts]
iso = []
for i, t in enumerate(texts):
    m = 0.0
    for j, u in enumerate(texts):
        if j == i: continue
        o = char_overlap(t, u)
        if o > m: m = o
    iso.append(m)
iso = np.asarray(iso)
probes = [i for i in range(0, N, max(1, N // 200))][:200]
m = HCM(Config(d_x=D, d_c=D, a=0.02, capacity=N + 10, beta=100.0, L=3,
               use_resid=False, lam=0.0, seed=0))
for i in range(N):
    m.observe(X[i])
fam = clean_fail = clean_ok = dirty_ok = dirty_n = 0
examples = []
for j, i in enumerate(probes):
    q = encode_degrade(texts[i], 0.3, 1000 + j)
    o = m.query(q, do_rif=False)
    w = int(o['idx'])
    ok = w == i
    ov = char_overlap(texts[w], texts[i])
    is_fam = ov >= 0.5
    rec = dict(i=i, true=texts[i][:20], winner=texts[w][:20], ov=round(ov, 2),
               isolated=bool(iso[i] < 0.5), ok=bool(ok))
    if ok:
        if iso[i] < 0.5: clean_ok += 1
        else: dirty_ok += 1
    elif is_fam:
        fam += 1
        if len(examples) < 8: examples.append(rec)
    elif iso[i] < 0.5:
        clean_fail += 1
        if len(examples) < 16: examples.append(rec)
    if iso[i] >= 0.5: dirty_n += 1
clean_n = int((iso < 0.5).sum())
sel = [r for r in probes if iso[r] < 0.5]
clean_hit = sum(1 for j, i in enumerate(sel))  # placeholder replaced below
res = dict(N=N, n_probes=len(probes),
           family_confusions=fam,
           clean_subset=dict(n=clean_n, hit=sum(1 for i in sel if int(m.query(encode_degrade(texts[i], 0.3, 7000+i), do_rif=False)['idx']) == i)),
           dirty=dict(n=dirty_n, ok=dirty_ok),
           examples=examples)
print(json.dumps(res['clean_subset'], ensure_ascii=False), flush=True)
print('family:', fam, 'dirty_ok:', dirty_ok, '/', dirty_n, flush=True)
for e in examples: print(json.dumps(e, ensure_ascii=False), flush=True)
json.dump(res, open(os.path.join(ROOT, 'results', 'exp22c_identifiability.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('saved results/exp22c_identifiability.json')
