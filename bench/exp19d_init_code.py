#!/usr/bin/env python3
'''exp19d_init_code.py —— 吸引子初值通道判别（exp19c 的后继）。

已证（exp19/exp19b/exp19c）：
  - oracle：M.T@(k 稀疏码 of 退化查询) 真条目 40/40 第一、mean rank 0.0（deg25/35 均）。
  - L=3 管线 deg25 8/40、deg35 0/40；init=raw(s_ev) 修复无效（F1 仍 8/40）。
  → 剩余嫌疑：init=z0=unit(max(s_pre,0)) 是**整个非负 pre 图样**（含大量亚阈值单元），
    不是 oracle 所用的 k 稀疏码 → softmax(20·M.T@z0) 提前锁错。
判别臂：
  G1 L3code    : retrieve_init="code"（z0=q0=k 稀疏码）, lam=0.6 → 预测 40/40
  G2 L3code_l1 : 同上, lam=1.0
  G3 L3code_L1 : retrieve_init="code", L=1（迭代一次）'''
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
LEVELS = (0.10, 0.25, 0.35)

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
print('exp19d d_x=%d N=%d seeds=%s' % (D, N, SEEDS), flush=True)
for s in SEEDS:
    X = pats(N, D, seed=100 + s)
    m = HCM(Config(d_x=D, d_c=D, a=0.02, capacity=N + 10, beta=20.0, L=3, use_resid=False, seed=s))
    t0 = time.time()
    for x in X:
        m.observe(x)
    print('seed%d write=%.1fs' % (s, time.time() - t0), flush=True)
    probes = list(range(0, N, N // PROBES_N))[:PROBES_N]
    old = (m.cfg.retrieve_init, m.cfg.softmax_on_raw, m.cfg.lam)
    def run(name, init, raw, lam, lv, L=3):
        m.cfg.retrieve_init, m.cfg.softmax_on_raw, m.cfg.lam = init, raw, lam
        t0 = time.time(); ok = 0
        for i in probes:
            q = X[i] if lv == 0.0 else degrade(X[i], lv, i)
            if int(m.query(q, L=L, do_rif=False)['idx']) == i:
                ok += 1
        print('seed%d %s deg%d: %d/%d (%.0fs)' % (s, name, int(lv * 100), ok, len(probes), time.time() - t0), flush=True)
        rows.append(dict(seed=s, arm=name, level=lv, top1=ok, n_probes=len(probes)))
    for lv in LEVELS:
        run('L3code', 'code', False, 0.6, lv)            # G1
    for lv in LEVELS:
        run('L3code_lam1', 'code', False, 1.0, lv)       # G2
    for lv in LEVELS:
        run('L3code_L1iter', 'code', False, 0.6, lv, L=1) # G3
    m.cfg.retrieve_init, m.cfg.softmax_on_raw, m.cfg.lam = old
    del m, X
print('total %.0fs' % (time.time() - t00), flush=True)
json.dump(rows, open(os.path.join(ROOT, 'results', 'exp19d_init_code.json'), 'w', encoding='utf-8'), indent=1)
print('saved results/exp19d_init_code.json')
