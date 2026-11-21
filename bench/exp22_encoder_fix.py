# -*- coding: utf-8 -*-
'''exp22_encoder_fix.py —— 真实语料编码器修复：theta 发育协议（Turrigiano 稳态）+ 相对阈值。

exp21 判决：theta=0 的 k-WTA(82/4096) 在真实文本码上不保相似度（补全 1-2/200）。
本文档字符串已指认 theta 是"把 input cos 0.986 的码压到 0.626 的唯一有效机制"。
臂（peer-01 语料，N=8844，d_h=4096, k=82, beta=100, L=3, resid off）：
  A0 = theta=0（exp21 基线，重跑一次作对照）
  A1 = homeo + develop(2000 样本, epochs=6, eta=0.05)（§3.2a 认定安全的占空比协议）
  A2 = A1 + rel=0.2（相对阈值：只留强响应维）
  A3 = A1 + k=328（更大码支撑）
指标：
  (1) 相似度保持 d'：退化自码 cos 与他码 cos 的分离度（核心因变量）
  (2) 补全 top1（deg 0.3/0.5/0.7 × 200 探针）
  (3) 合成对照：N=2000 稠密随机库精确召回（容量未毁）'''
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

rows = load('train'); texts = [r[2] for r in rows]
rows_te = load('test'); texts_te = [r[2] for r in rows_te]
V = sorted({ch for tl in (texts, texts_te) for t in tl for ch in _clean(t)} | {'\ufffd'})
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

X = np.asarray([encode(t) for t in texts], dtype=np.float32)
N = len(X)
probes = list(range(0, N, max(1, N // 200)))[:200]

def code_stats(m, tag):
    idx = probes[:120]
    H = np.asarray([m.enc.encode(X[i]) for i in idx], dtype=np.float32)
    C = H @ H.T
    off = C - np.eye(len(idx), dtype=np.float32)
    selfcos, othercos = [], []
    for j, i in enumerate(probes[:80]):
        h0 = m.enc.encode(encode_degrade(texts[i], 0.3, 5000 + j))
        selfcos.append(float(h0 @ H[j]))
    for j in range(0, 80, 4):
        for jj in range(j + 1, min(j + 5, 80)):
            othercos.append(float(H[j] @ H[jj]))
    d = (np.mean(selfcos) - np.mean(othercos)) / (np.std(othercos) + 1e-6)
    st = dict(nnz=float(np.mean((H > 0).sum(1))), code_cos=float(off[np.triu_indices(len(idx), 1)].mean()),
              selfcos_deg30=round(float(np.mean(selfcos)), 4), othercos=round(float(np.mean(othercos)), 4),
              dprime=round(float(d), 2))
    print(tag, json.dumps(st), flush=True)
    return st

def synthetic_control(homeo, rel, k, seed):
    m = HCM(Config(d_x=128, d_c=128, a=k / 4096.0, capacity=2010, beta=100.0, L=3,
                   use_resid=False, homeo=homeo, rel=rel, seed=seed))
    r = np.random.default_rng(9)
    Y = np.asarray([unit(v) for v in r.standard_normal((2000, 128))], dtype=np.float32)
    if homeo: m.develop(Y[:500], epochs=3)
    for y in Y: m.observe(y)
    ok = sum(int(m.query(Y[i], do_rif=False)['idx']) == i for i in range(0, 2000, 50))
    return ok, 40

rows_out = []
ARMS = [("A0_theta0", False, None, 82), ("A1_develop", True, None, 82),
        ("A2_develop_rel.2", True, 0.2, 82), ("A3_develop_k328", True, None, 328)]
for tag, homeo, rel, k in ARMS:
    t0 = time.time()
    m = HCM(Config(d_x=D, d_c=D, a=k / 4096.0, capacity=N + 10, beta=100.0, L=3,
                   use_resid=False, homeo=homeo, rel=rel, seed=0))
    if homeo:
        th = m.develop(X[:2000], epochs=6, eta_override=0.05) if False else None
        # develop 的 eta 用 enc 级参数（model.develop 未透传 eta，直接调 enc）
        import numpy as _np
        U = [unit(X[i]) for i in range(2000)]
        for _ in range(6):
            for u in U:
                h = m.enc.encode(u)
                m.enc.theta += (0.05 * (h - m.enc.target)).astype(np.float32)
        m.enc.homeo = True  # 写入期继续在线稳定
    st = code_stats(m, tag + ' stats')
    for i in range(N):
        m.observe(X[i])
    arm = dict(arm=tag, stats=st, write_s=round(time.time() - t0, 1), rows=[])
    for lv in (0.3, 0.5, 0.7):
        ok = 0; abst = 0
        for j, i in enumerate(probes):
            o = m.query(encode_degrade(texts[i], lv, 1000 + j), do_rif=False)
            ok += int(o['idx'] == i); abst += int(o['action'] == 'abstain')
        print('%s deg%.1f: %d/200 abst=%d' % (tag, lv, ok, abst), flush=True)
        arm['rows'].append(dict(level=lv, top1=ok, abstain=abst))
    sc_ok, sc_n = synthetic_control(homeo, rel, k, 0)
    arm['synthetic_exact'] = '%d/%d' % (sc_ok, sc_n)
    print(tag, 'synthetic exact:', arm['synthetic_exact'], flush=True)
    rows_out.append(arm)
    del m

json.dump(dict(meta=dict(sess=SESS, N=N, d=D), arms=rows_out),
          open(os.path.join(ROOT, 'results', 'exp22_encoder_fix.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('saved results/exp22_encoder_fix.json')
