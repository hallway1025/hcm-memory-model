#!/usr/bin/env python3
'''exp19c_fusion_fix2.py —— 验证两个融合缺陷的修复（exp19b 的因果臂 + 修复臂）。

exp19b 已定案：L1 下管线崩塌不在证据（oracle 40/40, mean rank 0.0），在融合：
  缺陷一（L=0）：ev 按候选数归一化 → beta·ev 动态范围坍缩 → a 近均匀 → 0.4*s_n 主导
                f（L0base 全退化 0/40；L0lam1 40/40 证明顺序保持）。
  缺陷二（L=3）：吸引子初值用阈值化 s_pre（脆），而评分/ oracle 用 raw 投射 s_ev（鲁棒）
                ——初值通道错配 → 退化时吸到错误盆地（L3base 8/40, 0/40）。
修复臂（Config 新开关，默认关闭=历史行为）：
  F1 L3raw     : retrieve_init="raw", lam=0.6   → 预测 deg25/35 恢复 ~40/40
  F2 L3raw_l1  : retrieve_init="raw", lam=1.0   → 修复 + 无先验
  F3 L0raw     : softmax_on_raw=True, lam=0.6   → 预测恢复 ~40/40
  F4 L0raw_l1  : softmax_on_raw=True, lam=1.0
对照臂：DEFAULT deg25（应复现 8/40, 12/40）。'''
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
print('exp19c d_x=%d N=%d seeds=%s' % (D, N, SEEDS), flush=True)
for s in SEEDS:
    X = pats(N, D, seed=100 + s)
    m = HCM(Config(d_x=D, d_c=D, a=0.02, capacity=N + 10, beta=20.0, L=3, use_resid=False, seed=s))
    t0 = time.time()
    for x in X:
        m.observe(x)
    print('seed%d write=%.1fs' % (s, time.time() - t0), flush=True)
    probes = list(range(0, N, N // PROBES_N))[:PROBES_N]
    old = (m.cfg.retrieve_init, m.cfg.softmax_on_raw, m.cfg.lam)
    def run(name, init, raw, lam, lv):
        m.cfg.retrieve_init, m.cfg.softmax_on_raw, m.cfg.lam = init, raw, lam
        t0 = time.time(); ok = 0
        for i in probes:
            q = X[i] if lv == 0.0 else degrade(X[i], lv, i)
            if int(m.query(q, do_rif=False)['idx']) == i:
                ok += 1
        print('seed%d %s deg%d: %d/%d (%.0fs)' % (s, name, int(lv * 100), ok, len(probes), time.time() - t0), flush=True)
        rows.append(dict(seed=s, arm=name, level=lv, top1=ok, n_probes=len(probes)))
    run('DEFAULT', 'pre', False, 0.6, 0.25)          # 对照：应复现 8/40, 12/40
    for lv in LEVELS:
        run('L3raw', 'raw', False, 0.6, lv)          # F1
    for lv in LEVELS:
        run('L3raw_lam1', 'raw', False, 1.0, lv)     # F2
    for lv in LEVELS:
        run('L0raw', 'pre', True, 0.6, lv)           # F3
    for lv in LEVELS:
        run('L0raw_lam1', 'pre', True, 1.0, lv)      # F4
    m.cfg.retrieve_init, m.cfg.softmax_on_raw, m.cfg.lam = old
    del m, X
print('total %.0fs' % (time.time() - t00), flush=True)
json.dump(rows, open(os.path.join(ROOT, 'results', 'exp19c_fusion_fix2.json'), 'w', encoding='utf-8'), indent=1)
print('saved results/exp19c_fusion_fix2.json')
