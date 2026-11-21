import os
# -*- coding: utf-8 -*-
'''exp23_semantic_encoder.py -- B1 学习型编码器第一步：共现 PPMI+SVD 字符嵌入 vs 随机投影。
可证伪预测（m03152）：学后近义对（累死了/好累）码余弦显著高于随机投影基线。
臂：R512 随机投影512（=exp22p 复刻）/ R256 维度匹配对照 / S256 PPMI+SVD 语义嵌入。
指标：E1 近义对码余弦（配对 t 检验 vs R256）；E2 检索回归门 exp22p 判据（obs_of）
     hit@1@deg0.3 不降；附输入空间效应归因（效应在 x 还是 h 产生）。
扩展点（给优化者）：learn_embedding() 换成你的字典学习器，接口=返回 (E: Vn x r, VOC)。'''
import csv, json, os, re, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from hcm import HCM, Config
from hcm.ops import unit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 2000; SESS = 'peer-02'  # 旧名 peer-01；v2 语料（D-053）
NANO = re.compile(r'\[[^\]]{1,8}\]')
def _clean(t): return NANO.sub('', re.sub(r'\s+', '', t))

with open(os.path.join(ROOT,'bench','corpus','train__%s.csv' % SESS), encoding='utf-8-sig', newline='') as f:
    rd = csv.reader(f); next(rd)
    texts = [r[2] for r in rd][:N]
msgs = [_clean(t) for t in texts]
V = sorted({ch for t in msgs for ch in t} | {'\ufffd'})
Vn = len(V); VOC = {ch: j for j, ch in enumerate(V)}
print('V =', Vn, flush=True)

SYN_NEAR = [('累死了','好累'), ('好累啊','累死了'), ('哈哈','哈哈哈哈'), ('嗯嗯','嗯'),
       ('好的','好滴'), ('谢谢','多谢'), ('在吗','在么'), ('睡觉','睡了'),
       ('吃饭','吃饭了'), ('上线','上号'), ('晚安','早点休息'), ('谢谢你','感谢'), ('可以吗','行不行')]
SYN_DISJOINT = [('累死了','疲惫'), ('开心','快乐'), ('睡觉','休息'), ('生气','愤怒'),
       ('难过','悲伤'), ('再见','拜拜'), ('漂亮','美丽'), ('聪明','机智'),
       ('对不起','抱歉'), ('加油','努力'), ('好吃','美味'), ('着急','焦急'),
       ('害怕','恐惧'), ('厉害','强大'), ('舒服','舒适'), ('辛苦','劳累')]

# 扩表 2026-11-07：聊天高频、对内零共享的近义词（charset 过滤自动剔除词表外）
SYN_DISJOINT += [
    ("开心", "高兴"), ("难过", "伤心"), ("努力", "加油"), ("害怕", "恐惧"),
    ("好看", "漂亮"), ("真的", "确实"), ("马上", "立刻"), ("明白", "听懂"),
    ("朋友", "兄弟"), ("吃饭", "干饭"), ("谢谢", "感恩"), ("同意", "赞成"),
    ("担心", "忧虑"), ("辛苦", "疲惫"), ("开始", "启动"), ("打扰", "干扰"),
    ("结束", "收尾"), ("生气", "愤怒"), ("聪明", "机智"), ("说话", "聊天"),
]

def learn_embedding_ppmi_svd(r=256, win=3):
    C = np.zeros((Vn, Vn), dtype=np.float64)
    for t in msgs:
        chs = [VOC[c] for c in t]
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
    nnz = float((ppmi > 0).mean())
    print('PPMI nnz=%.3f, r=%d' % (nnz, r), flush=True)
    return E

def random_proj(dr, seed=7):
    W = np.random.default_rng(seed).standard_normal((dr, Vn), dtype=np.float32)
    W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-9
    return W

def x_of(chars, E_or_W, mode):
    idx = [VOC[c] for c in chars] if chars else [VOC['\ufffd']]
    if E_or_W.shape[1] == Vn:  # (d, Vn) 随机投影：列选
        return unit(E_or_W[:, idx].sum(1))
    return unit(E_or_W[idx].sum(0).astype(np.float32))  # (Vn, r) 嵌入：行选

def encode_with(M, t): return x_of(list(_clean(t)), M, None)
def encode_deg_with(M, t, frac, seed):
    r = np.random.default_rng(seed)
    chs = list(_clean(t))
    if len(chs) > 1:
        for j in r.choice(len(chs), size=max(1, int(len(chs)*frac)), replace=False):
            chs[j] = ''
    return x_of([c for c in chs if c], M, None)

t0 = time.time()
iso = []
for i in range(N):
    best = 0.0
    sa = set(msgs[i])
    for j in range(N):
        if j != i:
            sb = set(msgs[j])
            v = len(sa & sb) / max(1, len(sa)) if sa else 0.0
            if v > best: best = v
    iso.append(best)
print('iso done %.1fs' % (time.time()-t0), flush=True)

def charset(s): return set(_clean(s))
syn_near = [(a, b) for a, b in SYN_NEAR if charset(a) <= set(V) and charset(b) <= set(V)]
syn_disj = [(a, b) for a, b in SYN_DISJOINT if charset(a) <= set(V) and charset(b) <= set(V)]
print('syn usable: near %d/%d disjoint %d/%d' % (len(syn_near), len(SYN_NEAR), len(syn_disj), len(SYN_DISJOINT)), flush=True)

E_sem = learn_embedding_ppmi_svd()
_all = [('R512', random_proj(512)), ('R256', random_proj(256)), ('S256', E_sem)]
_only = os.environ.get('EXP23_ARMS', '').split(',') if os.environ.get('EXP23_ARMS') else None
ARMS = [(k, v) for k, v in _all if not _only or k in _only]
res = {}
for tag, M in ARMS:
    dr = M.shape[1] if M.shape[0] == Vn else M.shape[0]  # (Vn,r) 语义嵌入: 维度在列
    dh = dr * 8
    tA = time.time()
    X = np.asarray([encode_with(M, t) for t in texts], dtype=np.float32)
    H = np.asarray([unit(np.asarray(h)) for h in []], dtype=np.float32) if False else None
    m = HCM(Config(d_x=dr, d_c=dr, a=0.02, capacity=N+10, beta=100.0, L=3,
                   use_resid=False, lam=1.0, seed=0))
    for i in range(N): m.observe(X[i])
    n_written = int(m.E.n)
    obs_of = {s: int(m.E.t[s]) for s in range(n_written)}
    wset = set(obs_of.values())
    def measure(pairs):
        cs, ins = [], []
        for a, b in pairs:
            ha = m.enc.encode(x_of(list(_clean(a)), M, None))
            hb = m.enc.encode(x_of(list(_clean(b)), M, None))
            cs.append(float(ha @ hb))
            ins.append(float(x_of(list(_clean(a)), M, None) @ x_of(list(_clean(b)), M, None)))
        return cs, ins
    syn_cos, syn_in = measure(syn_near)
    dis_cos, dis_in = measure(syn_disj)
    rr = np.random.default_rng(11)
    rand_cos = []
    for _ in range(2000):
        i, j = rr.integers(0, N, 2)
        if i == j: continue
        rand_cos.append(float(m.enc.encode(X[i]) @ m.enc.encode(X[j])))
    cand = [i for i in range(N) if iso[i] < 0.5 and i in wset][:200]
    ret = []
    for lv in (0.3, 0.5):
        ok = 0
        for j, i in enumerate(cand):
            o = m.query(encode_deg_with(M, texts[i], lv, 41000+j), do_rif=False)
            ok += int(obs_of.get(int(o['idx']), -1) == i)
        ret.append(dict(level=lv, n=len(cand), hit=ok))
        print('%s deg%.1f hit %d/%d' % (tag, lv, ok, len(cand)), flush=True)
    arm = dict(d_x=dr, d_h=dh, n_written=n_written,
               syn_mean_cos=round(float(np.mean(syn_cos)), 4),
               dis_mean_cos=round(float(np.mean(dis_cos)), 4),
               dis_input_cos=round(float(np.mean(dis_in)), 4),
               rand_mean_cos=round(float(np.mean(rand_cos)), 4),
               syn_input_cos=round(float(np.mean(syn_in)), 4),
               syn_per_pair=[round(v, 4) for v in syn_cos],
               dis_per_pair=[round(v, 4) for v in dis_cos],
               retrieval=ret, write_eval_s=round(time.time()-tA, 1))
    res[tag] = arm
    print(tag, json.dumps({k: v for k, v in arm.items() if k != 'syn_per_pair'}, ensure_ascii=False), flush=True)
    del m

import math
def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    t = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12))
    return dict(mean=round(float(d.mean()), 4), t=round(t, 3), n=len(d))
verdict = dict(
    near=paired(res.get('S256',{}).get('syn_per_pair',[]), res.get('R256',{}).get('syn_per_pair',[])),
    disjoint=paired(res['S256']['dis_per_pair'], res['R256']['dis_per_pair']),
    input_disjoint=paired([x for x in [res['S256']['dis_input_cos']]], [x for x in [res['R256']['dis_input_cos']]]),
    prediction_holds=None)
vd = verdict['disjoint']
verdict['prediction_holds'] = bool(vd['mean'] > 0 and vd['t'] > 2.0)
res['verdict'] = verdict
print('VERDICT', json.dumps(verdict), flush=True)
np.savez(os.path.join(ROOT, 'results', 'exp23_semantic_embed.npz'),
         E=E_sem, chars=np.asarray(V))
json.dump(res, open(os.path.join(ROOT, 'results', 'exp23_semantic_encoder.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('saved results/exp23_semantic_encoder.json', flush=True)