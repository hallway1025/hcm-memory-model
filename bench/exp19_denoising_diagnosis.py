#!/usr/bin/env python3
'''exp19_denoising_diagnosis.py —— L1 退化线索崩塌的机理诊断 + 证据通道定位。

背景（exp16）：L1（d_x=512, d_h=4096, k=82, N=2e4）精确检索完美迁移（1.000），
但 25% 缺失线索召回崩到 0.250（L0 同协议 ≈0.95）。本实验回答三个问题：
  Q1 情境噪声是不是凶手？（臂 noctx：kappa_ctx=0）
  Q2 各证据通道在 L1+退化下还有没有分离度？（oracle 通道排序：
     O_sparse 稀疏码 vs O_denseD 稠密直通 D vs O_payload 原始载荷 P）
  Q3 崩塌是渐进还是悬崖？（deg 10/25/35 曲线）
src/ 冻结不动：一切臂都是参数或 bench 内计算。探针一律 do_rif=False。
SMOKE=1 时本机小规模自检（d=128, N=2000, 1 seed）。'''
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = os.environ.get('SMOKE') == '1'
D = 128 if SMOKE else 512
N = 2000 if SMOKE else 20000
SEEDS = [0] if SMOKE else [0, 1]
PROBES_N = 20 if SMOKE else 40
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

def top1_count(m, X, probes, frac):
    ok = 0
    for i in probes:
        q = X[i] if frac == 0.0 else degrade(X[i], frac, i)
        if int(m.query(q, do_rif=False)['idx']) == i:
            ok += 1
    return ok

def oracle_ranks(m, enc, X, probes, frac):
    '''三个通道分别给真条目的排名（0=第一）。'''
    sp, dd, pp_ = [], [], []
    n = m.E.n
    for i in probes:
        q = X[i] if frac == 0.0 else degrade(X[i], frac, i)
        h = enc.encode(q)
        sc_m = m.E.M[:, :n].T @ h
        s_raw = enc.raw(q)
        sc_d = m.E.D[:, :n].T @ s_raw
        sc_p = m.E.P[:, :n].T @ q
        sp.append(int(np.sum(sc_m > sc_m[i])))
        dd.append(int(np.sum(sc_d > sc_d[i])))
        pp_.append(int(np.sum(sc_p > sc_p[i])))
    return (int(np.sum(np.asarray(sp) == 0)), int(np.sum(np.asarray(dd) == 0)),
            int(np.sum(np.asarray(pp_) == 0)),
            float(np.mean(sp)), float(np.mean(dd)), float(np.mean(pp_)))

rows = []
t00 = time.time()
print('exp19 d_x=%d d_h=%d N=%d seeds=%s smoke=%s' % (D, 8 * D, N, SEEDS, SMOKE), flush=True)
for s in SEEDS:
    X = pats(N, D, seed=100 + s)
    m = HCM(Config(d_x=D, d_c=D, a=0.02, capacity=N + 10, beta=20.0, L=3, use_resid=False, seed=s))
    t0 = time.time()
    for x in X:
        m.observe(x)
    wt = time.time() - t0
    probes = list(range(0, N, N // PROBES_N))[:PROBES_N]
    enc = m.enc
    # Q3：基线 deg 曲线（kappa_ctx=0.6 默认）
    ex = top1_count(m, X, probes, 0.0)
    print('seed%d write=%.1fs exact=%d/%d' % (s, wt, ex, len(probes)), flush=True)
    rows.append(dict(seed=s, arm='base', level=0.0, top1=ex, n_probes=len(probes), write_s=round(wt, 1)))
    for lv in LEVELS:
        t0 = time.time()
        ok = top1_count(m, X, probes, lv)
        qs = time.time() - t0
        o1, o2, o3, r1, r2, r3 = oracle_ranks(m, enc, X, probes, lv)
        print('seed%d deg%d: pipeline=%d/%d (%.0fs) | oracle sparse=%d denseD=%d payload=%d | meanrank %.1f/%.1f/%.1f'
              % (s, int(lv * 100), ok, len(probes), qs, o1, o2, o3, r1, r2, r3), flush=True)
        rows.append(dict(seed=s, arm='base', level=lv, top1=ok, n_probes=len(probes),
                         o_sparse=o1, o_denseD=o2, o_payload=o3,
                         mr_sparse=round(r1, 2), mr_denseD=round(r2, 2), mr_payload=round(r3, 2)))
    # Q1：关掉情境证据
    old = m.cfg.kappa_ctx
    m.cfg.kappa_ctx = 0.0
    ok25 = top1_count(m, X, probes, 0.25)
    m.cfg.kappa_ctx = old
    print('seed%d noctx deg25: %d/%d' % (s, ok25, len(probes)), flush=True)
    rows.append(dict(seed=s, arm='noctx', level=0.25, top1=ok25, n_probes=len(probes)))
    del m, X
print('total %.0fs' % (time.time() - t00), flush=True)
json.dump(rows, open(os.path.join(ROOT, 'results', 'exp19_denoising_diagnosis.json'), 'w', encoding='utf-8'), indent=1)
print('saved results/exp19_denoising_diagnosis.json')
