#!/usr/bin/env python3
'''exp19h_l0_beta_regression.py —— L0 先导规模（d_x=128, d_h=1024, N=4000）下 beta 扫描回归检查。

L1 发现：beta=20 在 N=2e4 下对比度坍缩（8/40@deg25），beta=100-500 完全恢复（40/40）。
本脚本在先导规模检查：beta 提升是否会伤害 §4.1-4.8 时代的经典设置。
若 beta=20 在 L0 仍最优 -> beta 需按规模设定（scale law）；若 L0 也偏好更大 beta
-> beta=20 一开始就偏小，L0 数字可能被低估。'''
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = 128
N = 4000
SEEDS = [0, 1]
PROBES_N = 40
LEVELS = (0.0, 0.10, 0.25, 0.35)
BETAS = (20.0, 100.0, 500.0)

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
print('exp19h L0 d_x=%d N=%d seeds=%s' % (D, N, SEEDS), flush=True)
for s in SEEDS:
    X = pats(N, D, seed=100 + s)
    m = HCM(Config(capacity=N + 10, use_resid=False, seed=s))
    t0 = time.time()
    for x in X:
        m.observe(x)
    n = m.E.n
    print('seed%d write=%.1fs n=%d' % (s, time.time() - t0, n), flush=True)
    probes = list(range(0, N, N // PROBES_N))[:PROBES_N]
    for b in BETAS:
        m.cfg.beta = b
        for lv in LEVELS:
            ok = 0; abst = 0; m1s = []
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
json.dump(rows, open(os.path.join(ROOT, 'results', 'exp19h_l0_beta_regression.json'), 'w', encoding='utf-8'), indent=1)
print('saved results/exp19h_l0_beta_regression.json')
