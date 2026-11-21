# -*- coding: utf-8 -*-
'''exp56_rho_sweep.py —— 编码器扩张率 ρ 剂量反应（K1 命门算子首个参数曲线）。

预注册（先于运行 commit；本文件即预注册本体）：
  动机：九算子档案 K1 记录「编码器=命门」，rho=8 只有理论理由（d_h=d_x·rho, k=round(a·d_h)），
        从未实测扫过。本实验补算子的 dose-response 曲线，回应「参数是不是拍脑袋」审稿质问。
  协议：逐项复制 exp19g（D=512, N=2e4, seeds{0,1}, probes=40, levels{0.10,0.25,0.35},
        degrade seed=探针号 i, X=pats(seed=100+s)），改扫 rho∈{2,4,8,16}；
        β 复用同一 store 查询档 {20,100}（查询便宜）→ 顺带测 rho 是否移动 β 拐点。
        附加记录：n_written/接受率、write_s、码间余弦采样（100 随机对，单位码点积）、g_mean。
  门：
    G1 复现锚：rho=8 臂与 exp19g 同 seed 的 β=100 行逐一对照（|Δrecall|≤2/40；exp19g seed0
       β100 三 level 全 40/40）——协议逐项一致的 rig 复现锚。
    G2 探索性（无方向门）：rho 排序与 recall 的关系如实报，效应量（每 level 每 β 的 4 点曲线）。
    G3 接受率与码余弦随 rho 的变化如实报（预期 rho↑→码余弦↓=分离↑，无方向门）。
  环境：BLAS pin=1（L-069）。冒烟：EXP56_SMOKE=1 → N=500, rho{8}, 1 seed。
  输出 results/exp56_rho_sweep.json。
'''
import os, json, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, os.path.join(ROOT, 'src'))
from hcm import HCM, Config
from hcm.ops import unit

SMOKE = os.environ.get('EXP56_SMOKE') == '1'
D = 512
N = 500 if SMOKE else 20000
SEEDS = [0] if SMOKE else [0, 1]
PROBES_N = 40
LEVELS = (0.10, 0.25, 0.35)
BETAS = (20.0, 100.0)
RHOS = (8,) if SMOKE else (2, 4, 8, 16)
N_COS = 100

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
print('exp56 rho sweep d_x=%d N=%d seeds=%s rhos=%s smoke=%s' % (D, N, SEEDS, RHOS, SMOKE), flush=True)
for s in SEEDS:
    X = pats(N, D, seed=100 + s)
    for rho in RHOS:
        m = HCM(Config(d_x=D, d_c=D, a=0.02, capacity=N + 10, beta=20.0, L=3,
                       use_resid=False, seed=s, rho=rho))
        t0 = time.time()
        n_acc = 0
        gs = []
        for x in X:
            rec = m.observe(x)
            n_acc += 1 if rec['wrote'] else 0
            gs.append(rec['g'])
        write_s = time.time() - t0
        n = m.E.n
        # 码间余弦采样（单位码点积）
        idxs = np.random.default_rng(7).choice(n, size=min(N_COS, n), replace=False)
        cols = m.E.M[:, idxs].T.astype(np.float32)
        cos = []
        for i in range(len(idxs) - 1):
            cos.extend(np.abs(cols[i] @ cols[i+1:].T).tolist())
        cos_mean = float(np.mean(cos)) if cos else 0.0
        print('seed%d rho=%d k=%d write=%.0fs n_written=%d acc=%.3f g_mean=%.3f cos=%.4f'
              % (s, rho, m.cfg.k, write_s, n, n_acc / N, float(np.mean(gs)), cos_mean), flush=True)
        for b in BETAS:
            m.cfg.beta = b
            for lv in LEVELS:
                ok = 0; m1s = []; abst = 0
                for i in range(PROBES_N):
                    pi = list(range(0, N, N // PROBES_N))[:PROBES_N][i]
                    q = X[pi] if lv == 0.0 else degrade(X[pi], lv, pi)
                    o = m.query(q, do_rif=False)
                    if o['action'] == 'abstain': abst += 1
                    if int(o['idx']) == pi: ok += 1
                    m1s.append(float(o['m1']))
                print('  seed%d rho=%d beta=%-5s deg%d: %d/%d abst=%d m1=%.3f'
                      % (s, rho, b, int(lv * 100), ok, PROBES_N, abst, float(np.mean(m1s))), flush=True)
                rows.append(dict(seed=s, rho=rho, beta=b, level=lv, top1=ok, n_probes=PROBES_N,
                                 abstain=abst, m1_mean=round(float(np.mean(m1s)), 4),
                                 n_written=int(n), accept_rate=round(n_acc / N, 4),
                                 g_mean=round(float(np.mean(gs)), 4), cos_mean=round(cos_mean, 4),
                                 write_s=round(write_s, 1), k=int(m.cfg.k)))
            m.cfg.beta = 20.0
        del m
        if SMOKE: break
    if SMOKE: break
print('total %.0fs' % (time.time() - t00), flush=True)
out = {'exp': 'exp56_rho_sweep', 'smoke': SMOKE, 'N': N, 'D': D, 'rows': rows}
json.dump(out, open(os.path.join(ROOT, 'results', 'exp56_rho_sweep%s.json' % ('_smoke' if SMOKE else '')), 'w', encoding='utf-8'), indent=1)
print('saved results/exp56_rho_sweep%s.json' % ('_smoke' if SMOKE else ''))
