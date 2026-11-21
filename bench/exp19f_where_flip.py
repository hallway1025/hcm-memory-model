#!/usr/bin/env python3
'''exp19f_where_flip.py —— 逐探针仪器化：a0（oracle 级 softmax）到 a_hop 到 f 的 argmax 在哪一步翻转。

已知：三通道 oracle mean rank 0.0；管线（任意 init/lam/门）deg25 = 8/40。
本脚本 init=code（q0 = k 稀疏码 = oracle 通道），lam=1，绕过门，逐步记录：
  H1 L=0        : f = a0 = softmax(20*M.T@q0)          → 预测 40/40（softmax 单调）
  H2 L=3        : f = a_hop*sup                        → 实测
  H3 L=3 sup=1  : 同 H2 但每次查询前 sup[:n]=1         → 分离 sup 的影响
  H4 L=3 lam.6  : f = .6*a_hop + .4*s_n                → 复现 G1（8/40）作对照
每个 deg25 探针记录：a0 argmax 是否真、a_hop argmax 是否真、sup[true]、sup[a_hop winner]、
M.T@q0 true 分数 vs runner-up。'''
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
print('exp19f d_x=%d N=%d seeds=%s' % (D, N, SEEDS), flush=True)
for s in SEEDS:
    X = pats(N, D, seed=100 + s)
    m = HCM(Config(d_x=D, d_c=D, a=0.02, capacity=N + 10, beta=20.0, L=3, use_resid=False, seed=s))
    t0 = time.time()
    for x in X:
        m.observe(x)
    n = m.E.n
    print('seed%d write=%.1fs n=%d sup: mean=%.3f min=%.3f max=%.3f frac<0.9=%.2f'
          % (s, time.time() - t0, n, float(m.E.sup[:n].mean()), float(m.E.sup[:n].min()),
             float(m.E.sup[:n].max()), float((m.E.sup[:n] < 0.9).mean())), flush=True)
    probes = list(range(0, N, N // PROBES_N))[:PROBES_N]
    m.ctl.always_retrieve = True
    m.cfg.lam = 1.0
    for lv in (0.10, 0.25, 0.35):
        cnt = dict(h1=0, h2=0, h3=0, h4=0, a0_ok=0, ahop_ok=0)
        detail = []
        for i in probes:
            q = X[i] if lv == 0.0 else degrade(X[i], lv, i)
            cvec = unit(m.c)
            u = np.concatenate([q, cvec, m._timecells()]).astype(np.float32)
            s_pre = m.enc.pre(u)
            q0 = m.enc.encode_from_pre(s_pre)
            Mn = m.E.M[:, :n]
            sc = Mn.T @ q0
            a0 = softmax_beta(sc, m.cfg.beta)
            a0_ok = int(np.argmax(a0)) == i
            z, a_hop = m.E.retrieve(q0, L=3, beta=m.cfg.beta, dense=None)
            ahop_ok = int(np.argmax(a_hop)) == i
            sup_t = float(m.E.sup[i]); sup_w = float(m.E.sup[int(np.argmax(a_hop))])
            # H1: L=0 equivalent
            o1 = m.query(q, L=0, do_rif=False)
            # H3: sup 强制为 1
            saved = m.E.sup[:n].copy(); m.E.sup[:n] = 1.0
            o3 = m.query(q, do_rif=False)
            m.E.sup[:n] = saved
            # H2: 正常 L3 lam1
            o2 = m.query(q, do_rif=False)
            # H4: lam 0.6
            m.cfg.lam = 0.6
            o4 = m.query(q, do_rif=False)
            m.cfg.lam = 1.0
            cnt['h1'] += int(o1['idx']) == i; cnt['h2'] += int(o2['idx']) == i
            cnt['h3'] += int(o3['idx']) == i; cnt['h4'] += int(o4['idx']) == i
            cnt['a0_ok'] += a0_ok; cnt['ahop_ok'] += ahop_ok
            if lv == 0.25:
                order = np.argsort(-sc)
                detail.append(dict(i=int(i), a0_ok=bool(a0_ok), ahop_ok=bool(ahop_ok),
                                   sc_true=round(float(sc[i]), 4), sc_run=round(float(order[1] if order[0] == i else order[0]) if False else float(np.sort(sc)[-2]), 4),
                                   sup_true=round(sup_t, 4), sup_win=round(sup_w, 4)))
        print('seed%d deg%d: H1_L0=%d H2_L3lam1=%d H3_sup1=%d H4_L3lam.6=%d | a0_ok=%d ahop_ok=%d'
              % (s, int(lv * 100), cnt['h1'], cnt['h2'], cnt['h3'], cnt['h4'], cnt['a0_ok'], cnt['ahop_ok']), flush=True)
        rows.append(dict(seed=s, level=lv, **cnt))
        if lv == 0.25:
            bad = [d for d in detail if not d['ahop_ok']][:6]
            print('  deg25 ahop_bad samples:', json.dumps(bad), flush=True)
            print('  deg25 sup stats: true_mean=%.3f win_mean=%.3f' % (
                float(np.mean([d['sup_true'] for d in detail])), float(np.mean([d['sup_win'] for d in detail]))), flush=True)
    del m, X
print('total %.0fs' % (time.time() - t00), flush=True)
json.dump(rows, open(os.path.join(ROOT, 'results', 'exp19f_where_flip.json'), 'w', encoding='utf-8'), indent=1)
print('saved results/exp19f_where_flip.json')
