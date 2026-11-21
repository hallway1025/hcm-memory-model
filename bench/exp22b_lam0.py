# -*- coding: utf-8 -*-
'''exp22b_lam0.py —— 判决性微实验：真实语料补全失败到底是不是语义先验毒化。

背景：exp22 A0 显示稀疏码相似度保持极好（退化自码 cos 0.63 vs 他码 0.04，d'=5.3），
但补全只有 1-2/200。合成实验里 lam=0.6 融合无害的原因：beta=100 下 a 是 one-hot，
f[top]=0.6+s_n*0.4 >= 0.6 > 任何 s_n[j]*0.4。若真实语料上 a 不收敛为 one-hot，
未训练的 sem（exp21/22 从未巩固/训练）就是 0.4 权重的纯噪声。
臂（peer-01 N=2000，快跑）：{theta0, develop} x {lam=0.6, lam=0.0}，deg{0.3,0.5} x 100 探针。'''
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
rows = load('train'); texts = [r[2] for r in rows][:2000]
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
X = np.asarray([encode(t) for t in texts], dtype=np.float32)
N = len(X)
probes = list(range(0, N, max(1, N // 100)))[:100]
rows_out = []
for tag, homeo, lam in [("theta0_lam.6", False, 0.6), ("theta0_lam0", False, 0.0),
                        ("develop_lam.6", True, 0.6), ("develop_lam0", True, 0.0)]:
    t0 = time.time()
    m = HCM(Config(d_x=D, d_c=D, a=0.02, capacity=N + 10, beta=100.0, L=3,
                   use_resid=False, homeo=homeo, lam=lam, seed=0))
    if homeo:
        U = [unit(X[i]) for i in range(min(1500, N))]
        for _ in range(6):
            for u in U:
                h = m.enc.encode(u)
                m.enc.theta += (0.05 * (h - m.enc.target)).astype(np.float32)
        m.enc.homeo = True
    for i in range(N):
        m.observe(X[i])
    arm = dict(arm=tag, write_s=round(time.time() - t0, 1), rows=[])
    # 仪器：a 的 top 质量 + 证据第一名
    tops = []
    for lv in (0.3, 0.5):
        ok = 0; abst = 0
        for j, i in enumerate(probes):
            o = m.query(encode_degrade(texts[i], lv, 1000 + j), do_rif=False)
            ok += int(o['idx'] == i); abst += int(o['action'] == 'abstain')
            if j < 20:
                aa = o.get('a')
                tops.append(float(np.max(aa)) if aa is not None else -1)
        print('%s deg%.1f: %d/100 abst=%d' % (tag, lv, ok, abst), flush=True)
        arm['rows'].append(dict(level=lv, top1=ok, abstain=abst))
    arm['a_top_mean'] = round(float(np.mean(tops)), 3) if tops else None
    rows_out.append(arm)
    del m
json.dump(rows_out, open(os.path.join(ROOT, 'results', 'exp22b_lam0.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('saved results/exp22b_lam0.json')
