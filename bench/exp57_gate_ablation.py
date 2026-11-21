# -*- coding: utf-8 -*-
'''exp57_gate_ablation.py —— 写入门权重一次一因子消融（M7_NEU 参数曲线）。

预注册（先于运行 commit；本文件即预注册本体）：
  动机：门 g=sigmoid(2.2ν+1.4rpe+0.6ε+0.8a−1.2) 的五个数是手工定的；K2 档案点名「参数敏感」。
        本实验做 OFAT（one-factor-at-a-time）±2× 消融 + 常数门 + 偏置两档，量化每个权重的
        可测影响（接受率 / g 均值 / 重复呈现重写率 / 检索召回），回应「拍脑袋」质问。
  协议：L0 档（D=128, N=2e3, seeds{0,1,2}, probes=40, level{0.25}, β=100）。
        臂（13）：baseline；w_nu/w_rho/w_eps/w_a 各 {×0.5, ×2}；b∈{−0.8, −1.6}；
        constant（gate_constant=True，g≡1）。
        重复呈现测试：写完全库后重呈现前 200 条已存 cue → dup_rewrite_rate=wrote=True 比例
        （健康门应拒绝重写；常数门预期全收）。
  门：
    G1 sanity：所有接受率∈[0,1]；baseline 臂跨 seed 接受率极差≤5pp（协议确定性锚）。
    G2 方向门（预声明）：dup_rewrite_rate(constant) > dup_rewrite_rate(baseline)——门的可测含义
       （重复输入关掉）的反向验证。冒烟实测：精确重呈现由 EpisodicStore dup 检查（≥0.95）拦截，
       两臂皆 0 → G2 预期如实 FAIL（发现=去重职责在 store 层，门角色=新颖性/吞吐门控）。
    G2b 探索性修订（amendment，全量前 commit）：dup_degraded_rate=重呈现 frac=0.10 退化变体
       （cos<0.95 过 dup 检查后门才接管）的重写率，逐臂如实报，无方向门。执行顺序=检索先行
       （退化重呈现会真实写入变体条目，冒烟实测 10 条入库即令 deg25 40/40→0/40，故检索测量
       必须在污染发生前完成）。
    AMENDMENT-2（v2，位置判据修正，v1 全量完成前 commit）：低门臂（w_a_x0.5/b_neg）在 t=0 漏写
       item 0（n=0 时 eps=log2 特殊 + 阈值降低 → g<0.5），库位次整体前移 1；v1 判据 idx==X 序号
       是位置敏感的，把内容完美的检索（诊断实probe->idx=i-1，actions 全 recall）全判 miss。
       v2 维护 write_map（写入位次→X 序号），命中判据改内容身份 write_map[idx]==i，双报
       deg25_pos（v1 判据，连续性）与 deg25_id（内容身份，canonical）。G1/G2 不变。
       产出 results/exp57_gate_ablation_v2.json（v1 JSON 保留原样，LEDGER 注记判据缺陷）。
    G3 探索性：各臂 deg25 recall 与 g_mean 如实报，无方向门。
  环境：BLAS pin=1（L-069）。冒烟：EXP57_SMOKE=1 → N=300, 臂{baseline, constant}, 1 seed。
  输出 results/exp57_gate_ablation.json。
'''
import os, json, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, os.path.join(ROOT, 'src'))
from hcm import HCM, Config
from hcm.ops import unit

SMOKE = os.environ.get('EXP57_SMOKE') == '1'
D = 128
N = 300 if SMOKE else 2000
SEEDS = [0] if SMOKE else [0, 1, 2]
PROBES_N = 40
LEVEL = 0.25
BETA = 100.0
BASE_W = (2.2, 1.4, 0.6, 0.8)
BASE_B = -1.2

ARMS = [('baseline', dict(w=BASE_W, b=BASE_B, constant=False))]
for name, i in (('w_nu', 0), ('w_rho', 1), ('w_eps', 2), ('w_a', 3)):
    for f in (0.5, 2.0):
        w = list(BASE_W); w[i] *= f
        ARMS.append(('%s_x%s' % (name, f), dict(w=tuple(w), b=BASE_B, constant=False)))
ARMS.append(('b_pos', dict(w=BASE_W, b=-0.8, constant=False)))
ARMS.append(('b_neg', dict(w=BASE_W, b=-1.6, constant=False)))
ARMS.append(('constant', dict(w=BASE_W, b=BASE_B, constant=True)))
if SMOKE:
    ARMS = [a for a in ARMS if a[0] in ('baseline', 'constant')]

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
print('exp57 gate ablation d_x=%d N=%d seeds=%s arms=%d smoke=%s' % (D, N, SEEDS, len(ARMS), SMOKE), flush=True)
probes = list(range(0, N, max(1, N // PROBES_N)))[:PROBES_N]
for s in SEEDS:
    X = pats(N, D, seed=300 + s)
    for arm, cfg_w in ARMS:
        m = HCM(Config(d_x=D, d_c=D, a=0.02, capacity=N + 10, beta=BETA, L=3,
                       use_resid=False, seed=s, gate_constant=cfg_w['constant']))
        m.gate.w_nu, m.gate.w_rho, m.gate.w_eps, m.gate.w_a = cfg_w['w']
        m.gate.b = cfg_w['b']
        t0 = time.time()
        n_acc = 0; gs = []
        write_map = []  # 存储位次 -> X 序号（AMENDMENT-2 内容身份判据）
        for i, x in enumerate(X):
            rec = m.observe(x)
            n_acc += 1 if rec['wrote'] else 0
            if rec['wrote']:
                write_map.append(i)
            gs.append(rec['g'])
        # 重复呈现：前 min(200,N) 条已存 cue
        n_rep = min(200, N)
        rewrites = 0
        for i in range(n_rep):
            rec = m.observe(X[i])
            rewrites += 1 if rec['wrote'] else 0
        # 检索（AMENDMENT-2：双判据 deg25_pos=idx==i / deg25_id=write_map[idx]==i）
        ok = 0; ok_id = 0; abst = 0; m1s = []
        for i in probes:
            q = degrade(X[i], LEVEL, i)
            o = m.query(q, do_rif=False)
            if o['action'] == 'abstain': abst += 1
            ri = int(o['idx'])
            if ri == i: ok += 1
            if ri < len(write_map) and write_map[ri] == i: ok_id += 1
            m1s.append(float(o['m1']))

        # 退化重呈现（探索性次指标）：在检索之后执行，避免变体写入污染检索测量
        rewrites_d = 0
        for i in range(n_rep):
            q = degrade(X[i], 0.10, 1000 + i)
            rec = m.observe(q)
            rewrites_d += 1 if rec['wrote'] else 0
        row = dict(seed=s, arm=arm, N=N, n_written=int(m.E.n), accept_rate=round(n_acc / N, 4),
                   g_mean=round(float(np.mean(gs)), 4), dup_rewrite_rate=round(rewrites / n_rep, 4),
                   dup_degraded_rate=round(rewrites_d / n_rep, 4),
                   deg25_pos=ok, deg25_id=ok_id, n_probes=PROBES_N, abstain=abst,
                   m1_mean=round(float(np.mean(m1s)), 4), write_s=round(time.time() - t0, 1))
        rows.append(row)
        print('seed%d %-10s n_written=%d acc=%.3f g_mean=%.3f dup_rw=%.3f deg25_pos=%d/%d deg25_id=%d/%d'
              % (s, arm, row['n_written'], row['accept_rate'], row['g_mean'],
                 row['dup_rewrite_rate'], ok, PROBES_N, ok_id, PROBES_N), flush=True)
        del m
print('total %.0fs' % (time.time() - t00), flush=True)
out = {'exp': 'exp57_gate_ablation_v2', 'smoke': SMOKE, 'N': N, 'D': D, 'arms': [a[0] for a in ARMS], 'rows': rows}
json.dump(out, open(os.path.join(ROOT, 'results', 'exp57_gate_ablation_v2%s.json' % ('_smoke' if SMOKE else '')), 'w', encoding='utf-8'), indent=1)
print('saved results/exp57_gate_ablation_v2%s.json' % ('_smoke' if SMOKE else ''))
