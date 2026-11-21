#!/usr/bin/env python3
'''diag_theta_protocol.py —— 钉死洁净复现 B 组 MISMATCH 的根因：theta 协议敏感性。

背景：第二轮洁净复现（results/replication2/）在 N=8000 无残差臂得 1.000（论文 0.938），
其与我们的实现差异清单里唯一系统性选项是 theta：他们做了批量占空比平衡发育，
而 exp14 的 Config 从未开 homeo、从未调 develop，theta 恒为 0。
本脚本在完全相同的生成器/检索/读出下只切换 theta 条件：
  V0 theta=0（= exp14 原样）
  V1 develop(前 min(N,2000) 条, epochs=6, eta=0.35) 后冻结（= 代码自带发育协议）
  V2 在线 homeo（写入时 theta += 0.35*(h - k/d_h)）
若 V1/V2 把无残差臂推到 ~1.0，则论文 exp14 的绝对数值是 theta 协议的函数，定性方向不变。
'''
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D, N, SEEDS = 128, 8000, [0, 1, 2]

def pats(n, d, seed):
    r = np.random.default_rng(seed)
    return np.asarray([unit(v) for v in r.standard_normal((n, d))], dtype=np.float32)

def degrade(x, frac, seed):
    r = np.random.default_rng(seed)
    y = x.copy()
    nz = np.where(np.abs(y) > 1e-6)[0]
    y[r.choice(nz, size=max(1, int(len(nz) * frac)), replace=False)] = 0.0
    return unit(y)

def run(cond, resid, seed):
    X = pats(N, D, seed=100 + seed)
    m = HCM(Config(d_x=D, a=0.02, capacity=N + 10, beta=20.0, L=3,
                   use_resid=resid, seed=seed))
    if cond == 'V1':
        m.enc.develop(X[:min(N, 2000)], epochs=6, eta=0.35)
    for x in X:
        m.observe(x)
    probe = list(range(0, N, N // 40))[:40]
    ex = [int(m.query(X[i])['idx'] == i) for i in probe]
    dg = [int(m.query(degrade(X[i], 0.25, i))['idx'] == i) for i in probe]
    return sum(ex) / len(ex), sum(dg) / len(dg)

rows = []
for cond in (('V0', 'V1')):
    for resid in (False, True):
        exs, dgs = [], []
        for s in SEEDS:
            e, d = run(cond, resid, s)
            exs.append(e); dgs.append(d)
        rows.append(dict(cond=cond, resid=resid,
                         exact=[round(v, 3) for v in exs],
                         deg25=[round(v, 3) for v in dgs],
                         exact_mean=round(float(np.mean(exs)), 3),
                         deg25_mean=round(float(np.mean(dgs)), 3)))
        print(cond, 'resid=%s' % resid, 'exact', exs, 'deg25', dgs, flush=True)

json.dump(rows, open(os.path.join(ROOT, 'results', 'diag_theta_protocol.json'), 'w', encoding='utf-8'), indent=1)
print('DONE')
