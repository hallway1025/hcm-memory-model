# -*- coding: utf-8 -*-
# exp23f v2 rerun (2026-11-09): 嵌入层面配对统计（near/dis/rand 码余弦）：2k(peer-02) vs v2 全库 vs R256；协议同 v1 仅换语料；v1 工件备份 exp23f_embed_stats_oldcorpus.json
import csv, json, os, re, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from hcm import HCM, Config
from hcm.ops import unit
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 2000; SESS = 'peer-02'  # v2（D-053）：v1 旧名 peer-01 语料为非现役残留，主会话改 v2
NANO = re.compile(r'\[[^\]]{1,8}\]')
def _clean(t): return NANO.sub('', re.sub(r'\s+', '', t))
with open(os.path.join(ROOT,'bench','corpus','train__%s.csv' % SESS), encoding='utf-8-sig', newline='') as f:
    rd = csv.reader(f); next(rd)
    texts = [r[2] for r in rd][:N]
msgs = [_clean(t) for t in texts]
V = sorted({ch for t in msgs for ch in t} | {'\ufffd'})
Vn = len(V); VOC = {ch: j for j, ch in enumerate(V)}
print('V =', Vn, flush=True)
def learn_embedding_ppmi_svd(msgs_in, r=256, win=3, guard=False):
    C = np.zeros((Vn, Vn), dtype=np.float64)
    for t in msgs_in:
        chs = [VOC[c] for c in t if (c in VOC or not guard)]
        if not chs: continue
        for i, a in enumerate(chs):
            for b in chs[max(0, i-win):i+win+1]:
                C[a, b] += 1.0
    tot = C.sum()
    pw = C.sum(1) / tot
    P = C / tot
    with np.errstate(divide='ignore'):
        pmi = np.log2(P / (pw[:, None] * pw[None, :] + 1e-12) + 1e-12)
    ppmi = np.maximum(0.0, pmi)
    U, S, Vt = np.linalg.svd(ppmi)
    r = min(r, Vn - 1)
    E = (U[:, :r] * np.sqrt(S[:r])).astype(np.float32)
    print('PPMI nnz=%.3f, r=%d' % (float((ppmi > 0).mean()), r), flush=True)
    return E
def random_proj(dr, seed=7):
    W = np.random.default_rng(seed).standard_normal((dr, Vn), dtype=np.float32)
    W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-9
    return W
def x_of(chars, E_or_W):
    idx = [VOC[c] for c in chars] if chars else [VOC['\ufffd']]
    if E_or_W.shape[1] == Vn: return unit(E_or_W[:, idx].sum(1))
    return unit(E_or_W[idx].sum(0).astype(np.float32))
def encode_with(M, t): return x_of(list(_clean(t)), M)
def encode_deg_with(M, t, frac, seed):
    r = np.random.default_rng(seed)
    chs = list(_clean(t))
    if len(chs) > 1:
        for j in r.choice(len(chs), size=max(1, int(len(chs)*frac)), replace=False):
            chs[j] = ''
    return x_of([c for c in chs if c], M)
t0 = time.time()
iso = []
for i in range(N):
    best = 0.0
    sa = set(msgs[i])
    for j in range(N):
        if j != i:
            v = len(sa & set(msgs[j])) / max(1, len(sa)) if sa else 0.0
            if v > best: best = v
    iso.append(best)
print('iso done %.1fs' % (time.time()-t0), flush=True)

E_sem = learn_embedding_ppmi_svd(msgs)
W_r = random_proj(256)
# exp23f: 仅嵌入层面的配对统计（无模型）：near/dis/rand 三组码余弦，2k vs 129k vs R256
import glob, csv
from exp23_pairs import SYN_NEAR, SYN_DISJOINT
TRAIN_FILES = ['peer-02', 'group-02', 'group-01', 'group-04', 'group-03', 'group-05']  # v2 显式列表：禁 glob（同 exp23c，审计 2026-11-09）
co_texts = []
for _p in TRAIN_FILES:
    with open(os.path.join(ROOT, 'bench', 'corpus', 'train__%s.csv' % _p), encoding='utf-8-sig', newline='') as _f:
        _rd = csv.reader(_f); next(_rd)
        co_texts.extend(r[2] for r in _rd)
co_msgs = [_clean(x) for x in co_texts]
E_2k = learn_embedding_ppmi_svd(msgs)
E_full = learn_embedding_ppmi_svd(co_msgs, guard=True)
W_r = random_proj(256)

import math
def build_enc(seed=0):
    m = HCM(Config(d_x=256, d_c=256, a=0.02, capacity=10, beta=100.0, L=0, use_resid=False, lam=1.0, seed=seed))
    return m.enc
enc = build_enc()
def cos_pairs(M, pairs):
    out = []
    for a, b in pairs:
        sa, sb = set(_clean(a)), set(_clean(b))
        if not (sa <= set(V) and sb <= set(V)): continue
        ha = enc.encode(x_of(list(_clean(a)), M))
        hb = enc.encode(x_of(list(_clean(b)), M))
        out.append(float(ha @ hb))
    return out
def cos_rand(M, n=400, seed=3):
    r9 = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i, j = r9.integers(0, N, 2)
        if i == j: continue
        out.append(float(enc.encode(X_[i]) @ enc.encode(X_[j])))
    return out
def paired_t(a, b):
    d = np.asarray(a) - np.asarray(b)
    m = float(d.mean()); sd = float(d.std(ddof=1))
    return round(m, 4), round(m / (sd / math.sqrt(len(d))), 3) if sd > 1e-9 else float('nan')
out = {}
for tag, M in [('R256', W_r), ('S256_2k', E_2k), ('S256_full', E_full)]:
    X_ = np.asarray([x_of(list(_clean(t)), M) for t in texts], dtype=np.float32)
    near = cos_pairs(M, SYN_NEAR)
    dis = cos_pairs(M, SYN_DISJOINT)
    rnd = cos_rand(M)
    out[tag] = {'near_per': [round(v, 4) for v in near], 'near_mean': round(float(np.mean(near)), 4),
                'dis_per': [round(v, 4) for v in dis], 'dis_mean': round(float(np.mean(dis)), 4),
                'rand_mean': round(float(np.mean(rnd)), 4), 'rand_p95': round(float(np.percentile(rnd, 95)), 4)}
    print(tag, json.dumps(out[tag], ensure_ascii=False), flush=True)
m1_, t1_ = paired_t(out['S256_full']['near_per'], out['R256']['near_per'])
m2_, t2_ = paired_t(out['S256_full']['near_per'], out['S256_2k']['near_per'])
m3_, t3_ = paired_t(out['S256_full']['dis_per'], out['S256_2k']['dis_per'])
m4_, t4_ = paired_t(out['S256_full']['dis_per'], out['R256']['dis_per'])
out['PAIRED'] = {'near_full_vs_R256': [m1_, t1_], 'near_full_vs_2k': [m2_, t2_],
                 'dis_full_vs_2k': [m3_, t3_], 'dis_full_vs_R256': [m4_, t4_]}
print('PAIRED', json.dumps(out['PAIRED']), flush=True)
out['meta'] = {'corpus_version': 'v2', 'sess_2k': SESS, 'train_files': TRAIN_FILES, 'n_full': len(co_msgs), 'note': 'v1 工件=exp23f_embed_stats_oldcorpus.json'}
json.dump(out, open(os.path.join(ROOT, 'results', 'exp23f_embed_stats.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)  # L-059: GBK 崩修复
print('saved results/exp23f_embed_stats.json', flush=True)