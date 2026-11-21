# -*- coding: utf-8 -*-
'''exp21_real_corpus.py —— HCM 真实语料检索基准（L0 先导，peer-02 私聊 [旧名 peer-01]，v2 语料 9033 train 条）。

编码：字符 unigram+bigram 特征哈希 -> d_x 维 tf 向量 -> L2 归一化。
退化：按比例删除字符特征（模拟部分线索）。
探针：
  A 模式补全：train 内消息特征丢失 p∈{0.3,0.5,0.7}，top-1 命中率（库=全部 train）
  B 重复消歧：检索高频重复串（如"上号"），正确答案=与查询同时间邻域的那条
    （纯相似度必然打平 -> 只有时间上下文能救——B2 的真实数据动机）
  C 新颖性/弃权：test（从未写入）查询的弃权率与最近邻语义
  D 规模曲线：库 ∈ {1k,2k,4k,8844} 的命中率
默认 beta=100（exp19g 定案）。
'''
import csv, json, os, re, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from hcm import HCM, Config
from hcm.ops import unit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = 512
SESS = "peer-02"  # 旧名 peer-01（昵称），2026-11-08 语料 v2 重导后按备注命名
# 字符级随机投影编码：V = train+test 全部去重字符；W = 固定高斯 (D x V)，列归一化。
# x = normalize(W[:, chars].sum(1)) —— 6 字消息 -> 512 维稠密判别模式（信息=哪几个字）。
NANO = re.compile(r'\[[^\]]{1,8}\]')

def _clean(text):
    return NANO.sub('', re.sub(r'\s+', '', text))

VOCAB = None
W = None

def init_encoder(texts_tr, texts_te):
    global VOCAB, W
    V = sorted({ch for tl in (texts_tr, texts_te) for t in tl for ch in _clean(t)} | {'\ufffd'})
    VOCAB = {ch: j for j, ch in enumerate(V)}
    rng = np.random.default_rng(7)
    W = rng.standard_normal((D, len(VOCAB)), dtype=np.float32)
    W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-9
    return len(V)

def encode_chars(chars):
    idx = [VOCAB[c] for c in chars] if chars else [VOCAB['\ufffd']]
    return unit(W[:, idx].sum(1))

def encode(text, seed=0):
    return encode_chars(list(_clean(text)))

def encode_degrade(text, frac, seed):
    '''特征级退化 = 从消息里删掉 frac 比例的字符后重新编码（部分线索的自然定义）。'''
    r = np.random.default_rng(seed)
    chs = _clean(text)
    chs = list(chs)
    if len(chs) > 1:
        k = max(1, int(len(chs) * frac))
        for j in r.choice(len(chs), size=k, replace=False):
            chs[j] = ''
    return encode_chars([c for c in chs if c])

def load(split):
    p = os.path.join(ROOT, 'bench', 'corpus', '%s__%s.csv' % (split, SESS))
    with open(p, encoding='utf-8-sig', newline='') as f:
        rd = csv.reader(f); next(rd)
        return [(t, s, c) for t, s, c in rd]

rows = load('train')
texts = [r[2] for r in rows]
rows_te = load('test')
texts_te = [r[2] for r in rows_te]
V = init_encoder(texts, texts_te)
X = np.asarray([encode(t) for t in texts], dtype=np.float32)
Xte = np.asarray([encode(t) for t in texts_te], dtype=np.float32)
print('train=%d vocab=%d nnz_mean=%.1f cos_mean=%.3f' % (len(X), V, float((X > 0).sum(1).mean()), float((X @ X.T)[::max(1,len(X)//500), ::max(1,len(X)//500)].mean())), flush=True)

# 高频重复串（重复消歧探针的材料）
from collections import Counter
cnt = Counter(texts)
dups = [t for t, c in cnt.most_common(200) if c >= 20 and len(t) <= 6][:40]
print('duplicate probes:', len(dups), 'max count:', cnt.most_common(1)[0][1], flush=True)

results = {'meta': dict(sess=SESS, n_train=len(X), d=D, beta=100.0,
                        dup_probes=len(dups), top_dup=cnt.most_common(3)[0])}
rows_out = []

for N in (1000, 2000, 4000, len(X)):
    m = HCM(Config(d_x=D, d_c=D, a=0.02, capacity=N + 10, beta=100.0, L=3,
                   use_resid=False, seed=0))
    t0 = time.time()
    for i in range(N):
        m.observe(X[i])
    wt = time.time() - t0
    probes = list(range(0, N, max(1, N // 200)))[:200]
    # A 模式补全
    for lv in (0.3, 0.5, 0.7):
        ok = 0; abst = 0; rk = []
        for j, i in enumerate(probes):
            o = m.query(encode_degrade(texts[i], lv, 1000 + j), do_rif=False)
            if o['action'] == 'abstain': abst += 1
            r = int(o['rank1']) if 'rank1' in o else None
            ok += int(o['idx'] == i)
        print('N=%-5d write=%.0fs deg%.1f: %d/%d abst=%d' % (N, wt, lv, ok, len(probes), abst), flush=True)
        rows_out.append(dict(task='completion', N=N, level=lv, top1=ok, n=len(probes), abstain=abst, write_s=round(wt, 1)))
    del m

json.dump({'meta': results['meta'], 'rows': rows_out},
          open(os.path.join(ROOT, 'results', 'exp21_real_corpus.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('saved results/exp21_real_corpus.json')
