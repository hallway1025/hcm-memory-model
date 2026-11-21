#!/usr/bin/env python3
'''exp20_resid_beta_l1.py —— 残差有害结论在 beta=100 下的重检验（L1 规模，4 种子）。

背景：exp16（beta=20）报告 L1 下 no-resid deg25=0.250 / resid deg25=0.031，
"残差害处随规模增长"（§7 否定性数据点）。exp19g 已证 beta=20 在 L1 是对比度坍缩伪影
（beta=100 全恢复）。本实验在 beta=100 下重跑两臂，并保留 beta=20 的 deg25 锚点。

设计：4 种子 × {resid=False, True} 两库；每库测：
  beta=100: deg {0, 10, 25, 35}
  beta=20 : deg {25}（锚点，应复现 exp16 量级）
记录 top1/abstain/m1 + 写入时间。'''
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = 512
N = 20000
SEEDS = [0, 1, 2, 3]
PROBES_N = 40

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
print('exp20 d_x=%d N=%d seeds=%s' % (D, N, SEEDS), flush=True)
for s in SEEDS:
    X = pats(N, D, seed=100 + s)
    probes = list(range(0, N, N // PROBES_N))[:PROBES_N]
    for resid in (False, True):
        m = HCM(Config(d_x=D, d_c=D, a=0.02, capacity=N + 10, beta=20.0, L=3,
                       use_resid=resid, seed=s))
        t0 = time.time()
        for x in X:
            m.observe(x)
        wt = time.time() - t0
        ex = 0
        for i in probes:
            if int(m.query(X[i], do_rif=False)['idx']) == i:
                ex += 1
        print('seed%d resid=%s write=%.1fs n=%d exact=%d/40' % (s, resid, wt, m.E.n, ex), flush=True)
        rows.append(dict(seed=s, resid=resid, beta=20.0, level=0.0, top1=ex, n_probes=len(probes), write_s=round(wt, 1)))
        plan = [(100.0, 0.0), (100.0, 0.10), (100.0, 0.25), (100.0, 0.35), (20.0, 0.25)]
        for b, lv in plan:
            m.cfg.beta = b
            ok = 0; abst = 0; m1s = []
            for i in probes:
                q = X[i] if lv == 0.0 else degrade(X[i], lv, i)
                o = m.query(q, do_rif=False)
                if o['action'] == 'abstain': abst += 1
                if int(o['idx']) == i: ok += 1
                m1s.append(float(o['m1']))
            print('seed%d resid=%s beta=%-6s deg%d: %d/40 abst=%d m1=%.3f'
                  % (s, resid, b, int(lv * 100), ok, abst, float(np.mean(m1s))), flush=True)
            rows.append(dict(seed=s, resid=resid, beta=b, level=lv, top1=ok, abstain=abst,
                             m1_mean=round(float(np.mean(m1s)), 4), n_probes=len(probes)))
        del m
    del X
print('total %.0fs' % (time.time() - t00), flush=True)
json.dump(rows, open(os.path.join(ROOT, 'results', 'exp20_resid_beta_l1.json'), 'w', encoding='utf-8'), indent=1)
print('saved results/exp20_resid_beta_l1.json')
