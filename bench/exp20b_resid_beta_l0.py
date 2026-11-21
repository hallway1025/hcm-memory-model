#!/usr/bin/env python3
'''exp20b_resid_beta_l0.py -- 用 beta=100 重跑 exp14 全表（残差 x 规模，回应 D-036）。
exp14/论文 4.3 的残差差距是 beta=20 时代产物；本实验同协议 8 种子，beta=100，
另加 beta=20 单种子锚点确认伪影可复现。判据 obs_of（L-035 规范）+ 记录写入数。'''
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = 128
SEEDS = list(range(8))

def pats(N, d, seed):
    r = np.random.default_rng(seed)
    return np.asarray([unit(v) for v in r.standard_normal((N, d))], dtype=np.float32)

def degrade(x, frac, seed):
    r = np.random.default_rng(seed)
    y = x.copy()
    nz = np.where(np.abs(y) > 1e-6)[0]
    y[r.choice(nz, size=max(1, int(len(nz) * frac)), replace=False)] = 0.0
    return unit(y)

def ci95(v):
    v = np.asarray(v, float)
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))

def run_cell(N, resid, beta, seeds, tag):
    exact, deg, wr = [], [], []
    t0 = time.time()
    for s in seeds:
        X = pats(N, D, seed=100 + s)
        m = HCM(Config(d_x=D, d_c=D, a=0.02, capacity=N + 10, beta=beta, L=3,
                       use_resid=resid, seed=s))
        for x in X:
            m.observe(x)
        obs_of = {int(m.E.t[j]): j for j in range(int(m.E.n))}
        wr.append(int(m.E.n))
        probe = list(range(0, N, max(1, N // 40)))[:40]
        e = sum(int(obs_of.get(int(m.query(X[i])['idx']), -1) == i) for i in probe) / len(probe)
        d = sum(int(obs_of.get(int(m.query(degrade(X[i], 0.25, i))['idx']), -1) == i) for i in probe) / len(probe)
        exact.append(e); deg.append(d)
        del m
    em, ec = ci95(exact); dm, dc = ci95(deg)
    row = dict(N=N, resid=resid, beta=beta, tag=tag, seeds=len(seeds),
               exact_mean=round(em, 3), exact_ci=round(ec, 3),
               deg_mean=round(dm, 3), deg_ci=round(dc, 3),
               written=round(float(np.mean(wr)), 1), cell_s=round(time.time() - t0, 1))
    print("%-6s %-6s beta=%-6s exact %.3f+-%.3f  deg25 %.3f+-%.3f  n=%.0f  %.0fs" % (
        N, resid, beta, em, ec, dm, dc, row["written"], row["cell_s"]), flush=True)
    return row

rows = []
for N in (4000, 8000, 16000):
    rows.append(run_cell(N, False, 100.0, SEEDS, 'b100'))
    rows.append(run_cell(N, True, 100.0, SEEDS, 'b100'))
    rows.append(run_cell(N, False, 20.0, [0], 'b20_anchor'))
    rows.append(run_cell(N, True, 20.0, [0], 'b20_anchor'))
    # verdict per N
    b100f = next(r for r in rows if r['N'] == N and r['resid'] is False and r['beta'] == 100.0)
    b100t = next(r for r in rows if r['N'] == N and r['resid'] is True and r['beta'] == 100.0)
    sep_f = (b100f['deg_mean'] - b100f['deg_ci'], b100f['deg_mean'] + b100f['deg_ci'])
    sep_t = (b100t['deg_mean'] - b100t['deg_ci'], b100t['deg_mean'] + b100t['deg_ci'])
    overlap = not (sep_f[1] < sep_t[0] or sep_t[1] < sep_f[0])
    print('N=%d beta=100 CI overlap (F vs T): %s  F=[%.3f,%.3f] T=[%.3f,%.3f]' % (N, overlap, sep_f[0], sep_f[1], sep_t[0], sep_t[1]), flush=True)
json.dump(rows, open(os.path.join(ROOT, 'results', 'exp20b_resid_beta_l0.json'), 'w', encoding='utf-8'), indent=1)
print('saved results/exp20b_resid_beta_l0.json', flush=True)