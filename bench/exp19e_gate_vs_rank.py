#!/usr/bin/env python3
'''exp19e_gate_vs_rank.py —— 判别「检索崩塌」还是「弃权门误杀」。

证据链：三通道 oracle mean rank 0.0（真条目永远第一）但管线 idx 命中 8/40、0/40；
exp19c 的 lam=1 臂也只有 19/40 —— 与「排序完好、弃权门（raw_evidence<0.50 → abstain
→ idx=-1）把正确检索拒之门外」的假说一致。
本脚本在同一库上对比：
  GATED  : 默认（记录 idx 命中 + action 分布 + raw_evidence 统计）
  NOGATE : ctl.always_retrieve=True（绕过弃权门，只看 f 的 argmax 排序）
同时把缺失率扩展到 {0,10,25,35,50,70}%，为 §4.10 的「可靠缺失上限」提供
门控无关的重测。'''
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = 512
N = 20000
SEEDS = [0, 1]
PROBES_N = 40
LEVELS = (0.0, 0.10, 0.25, 0.35, 0.50, 0.70)

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
print('exp19e d_x=%d N=%d seeds=%s' % (D, N, SEEDS), flush=True)
for s in SEEDS:
    X = pats(N, D, seed=100 + s)
    m = HCM(Config(d_x=D, d_c=D, a=0.02, capacity=N + 10, beta=20.0, L=3, use_resid=False, seed=s))
    t0 = time.time()
    for x in X:
        m.observe(x)
    print('seed%d write=%.1fs' % (s, time.time() - t0), flush=True)
    probes = list(range(0, N, N // PROBES_N))[:PROBES_N]
    for lv in LEVELS:
        stat = dict(gated=0, nogate=0, acts={}, re=[], m1s=[])
        for i in probes:
            q = X[i] if lv == 0.0 else degrade(X[i], lv, i)
            o = m.query(q, do_rif=False)
            stat['acts'][o['action']] = stat['acts'].get(o['action'], 0) + 1
            stat['re'].append(float(o.get('raw_evidence', -1)))
            stat['m1s'].append(float(o.get('m1', -1)))
            if int(o['idx']) == i:
                stat['gated'] += 1
        m.ctl.always_retrieve = True
        for i in probes:
            q = X[i] if lv == 0.0 else degrade(X[i], lv, i)
            if int(m.query(q, do_rif=False)['idx']) == i:
                stat['nogate'] += 1
        m.ctl.always_retrieve = False
        re = np.asarray(stat['re'])
        print('seed%d deg%d: gated=%d/40 nogate=%d/40 acts=%s raw_ev=%.3f±%.3f m1=%.3f'
              % (s, int(lv * 100), stat['gated'], stat['nogate'], stat['acts'],
                 re.mean(), re.std(), float(np.mean(stat['m1s']))), flush=True)
        rows.append(dict(seed=s, level=lv, gated=stat['gated'], nogate=stat['nogate'],
                         acts=stat['acts'], raw_ev_mean=round(float(re.mean()), 4),
                         raw_ev_std=round(float(re.std()), 4), m1_mean=round(float(np.mean(stat['m1s'])), 4),
                         n_probes=len(probes)))
    del m, X
print('total %.0fs' % (time.time() - t00), flush=True)
json.dump(rows, open(os.path.join(ROOT, 'results', 'exp19e_gate_vs_rank.json'), 'w', encoding='utf-8'), indent=1)
print('saved results/exp19e_gate_vs_rank.json')
