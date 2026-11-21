#!/usr/bin/env python3
'''exp19g_beta_scale.py —— 假说：softmax 对比度随库规模坍缩（scale law）。

证据链：oracle 排名完美（mean rank 0.0）但 a 近乎平坦（m1≈0.39-0.46 at deg25/35 vs 0.78 at deg0）；
门控无关（gated=nogate）；sup≡1；init 通道无关。β=20 是小库（几百候选）时代调的——
N=2e4 时 max-vs-runner-up 的相对间隙被 F_S^N 压扁（附录 E 的零分布），20×gap≪1 → a 平坦
→ 迭代把质量摊到均值方向 + 先验噪声接管 argmax。
判别：库固定，扫 β ∈ {20,100,500,2500} 跑完整管线（L=3, lam=0.6, 门开）。
预测：β 增大 → deg25/35 召回恢复。同时记录 m1（应随 β 升向 1）。'''
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.ops import softmax_beta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = 512
N = 20000
SEEDS = [0, 1]
PROBES_N = 40
LEVELS = (0.10, 0.25, 0.35)
BETAS = (20.0, 100.0, 500.0, 2500.0)

def pats(n, d, seed):
    r = np.random.default_rng(seed)
    return np.asarray([unit(v) for v in r.standard_normal((n, d))], dtype=np.float32)

def degrade(x, frac, seed):
    r = np.random.default_rng(seed)
    y = x.copy()
    nz = np.where(np.abs(y) > 1e-6)[0]
    y[r.choice(nz, size=max(1, int(len(nz) * frac)), replace=False)] = 0.0
    return unit(y)

rows = []
t00 = time.time()
print('exp19g d_x=%d N=%d seeds=%s betas=%s' % (D, N, SEEDS, BETAS), flush=True)
for s in SEEDS:
    X = pats(N, D, seed=100 + s)
    m = HCM(Config(d_x=D, d_c=D, a=0.02, capacity=N + 10, beta=20.0, L=3, use_resid=False, seed=s))
    t0 = time.time()
    for x in X:
        m.observe(x)
    n = m.E.n
    # 仪器：a0 平坦度 + oracle 间隙（β=20 时）
    probes = list(range(0, N, N // PROBES_N))[:PROBES_N]
    gaps, a0top = [], []
    for i in probes:
        q = degrade(X[i], 0.25, i)
        xe = m.enc.resid_read_all(q)[0]
        q0 = m.enc.encode_from_pre(m.enc.pre(xe))
        sc = m.E.M[:, :n].T @ q0
        srt = np.sort(sc)[::-1]
        gaps.append(float(srt[0] - srt[1])); a0top.append(float(softmax_beta(sc, 20.0).max()))
    print('seed%d write=%.1fs n=%d | deg25 sc gap mean=%.4f p10=%.4f | a0(beta20) top mean=%.4f'
          % (s, time.time() - t0, n, float(np.mean(gaps)), float(np.percentile(gaps, 10)), float(np.mean(a0top))), flush=True)
    for b in BETAS:
        m.cfg.beta = b
        for lv in LEVELS:
            ok = 0; m1s = []; abst = 0
            for i in probes:
                q = X[i] if lv == 0.0 else degrade(X[i], lv, i)
                o = m.query(q, do_rif=False)
                if o['action'] == 'abstain': abst += 1
                if int(o['idx']) == i: ok += 1
                m1s.append(float(o['m1']))
            print('seed%d beta=%-6s deg%d: %d/40 abst=%d m1=%.3f'
                  % (s, b, int(lv * 100), ok, abst, float(np.mean(m1s))), flush=True)
            rows.append(dict(seed=s, beta=b, level=lv, top1=ok, abstain=abst,
                             m1_mean=round(float(np.mean(m1s)), 4), n_probes=len(probes)))
    m.cfg.beta = 20.0
    del m, X
print('total %.0fs' % (time.time() - t00), flush=True)
json.dump(rows, open(os.path.join(ROOT, 'results', 'exp19g_beta_scale.json'), 'w', encoding='utf-8'), indent=1)
print('saved results/exp19g_beta_scale.json')
