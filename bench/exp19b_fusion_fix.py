#!/usr/bin/env python3
'''exp19b_fusion_fix.py —— 验证 L1 退化检索崩塌的融合缺陷假说（exp19 的干预臂）。

exp19 已证：L1 下 25%/35% 缺失线索的稀疏码原始证据把真条目排在 40/40 第一名
（mean rank 0.0），但完整管线只有 8/40 与 0/40 —— 证据无损，融合有损。
本脚本在同一写入库上比较四个检索变体，定位凶手：
  A L0base  : L=0（不用吸引子），kappa_ctx=0.6（默认）
  B L0noctx : L=0，kappa_ctx=0
  C L0lam1  : L=0，kappa_ctx=0，lam=1.0（纯稀疏证据，去语义先验）
  D L3noctx : L=3（完整吸引子），kappa_ctx=0
预测：若凶手是「情境噪声 × 吸引子跳盆」，则 A/B 恢复、D 仍崩、C 恢复；
若语义先验也是凶手，则只有 C 恢复。探针一律 do_rif=False。'''
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
print('exp19b d_x=%d N=%d seeds=%s' % (D, N, SEEDS), flush=True)
for s in SEEDS:
    X = pats(N, D, seed=100 + s)
    m = HCM(Config(d_x=D, d_c=D, a=0.02, capacity=N + 10, beta=20.0, L=3, use_resid=False, seed=s))
    t0 = time.time()
    for x in X:
        m.observe(x)
    print('seed%d write=%.1fs' % (s, time.time() - t0), flush=True)
    probes = list(range(0, N, N // PROBES_N))[:PROBES_N]
    variants = [('L0base', 0, 0.6, 0.6), ('L0noctx', 0, 0.0, 0.6), ('L0lam1', 0, 0.0, 1.0), ('L3noctx', 3, 0.0, 0.6)]
    for name, Lv, kap, lam in variants:
        lvl_list = LEVELS if name in ('L0base',) else (0.25, 0.35)
        oldk, oldl = m.cfg.kappa_ctx, m.cfg.lam
        m.cfg.kappa_ctx, m.cfg.lam = kap, lam
        for lv in lvl_list:
            t0 = time.time()
            ok = 0
            for i in probes:
                q = X[i] if lv == 0.0 else degrade(X[i], lv, i)
                if int(m.query(q, L=Lv, do_rif=False)['idx']) == i:
                    ok += 1
            print('seed%d %s deg%d: %d/%d (%.0fs)' % (s, name, int(lv * 100), ok, len(probes), time.time() - t0), flush=True)
            rows.append(dict(seed=s, arm=name, level=lv, top1=ok, n_probes=len(probes)))
        m.cfg.kappa_ctx, m.cfg.lam = oldk, oldl
    # 基线对照：完整默认管线（L=3, kappa=0.6）在 deg25 —— 应复现 exp19 的 8/40 量级
    t0 = time.time()
    ok = 0
    for i in probes:
        if int(m.query(degrade(X[i], 0.25, i), do_rif=False)['idx']) == i:
            ok += 1
    print('seed%d DEFAULT(L3,kctx.6) deg25: %d/%d (%.0fs)' % (s, ok, len(probes), time.time() - t0), flush=True)
    rows.append(dict(seed=s, arm='DEFAULT', level=0.25, top1=ok, n_probes=len(probes)))
    del m, X
print('total %.0fs' % (time.time() - t00), flush=True)
json.dump(rows, open(os.path.join(ROOT, 'results', 'exp19b_fusion_fix.json'), 'w', encoding='utf-8'), indent=1)
print('saved results/exp19b_fusion_fix.json')
