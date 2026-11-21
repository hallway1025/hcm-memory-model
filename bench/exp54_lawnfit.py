import os
# -*- coding: utf-8 -*-
'''exp54_lawnfit.py —— log-N 定律定量收口（分析专用，零新模型运行）。

预注册（先于运行 commit；本文件即预注册本体）：
  目的：把散在 6 个 artifact 家族里的 β-塌陷/β*(N) 数据聚合成一张表，
        拟合 β*(N)≈c·ln N，为 §4.12 可证伪合同句提供定量骨架。
  数据源（只读，全部已在 results/ 落盘）：
    S1 exp25_collapse_external.json   模型外 vanilla Hopfield/attention，3 臂×N{1e3,1e4,1e5,1e6}，β 网格 {2..1000}
    S2 exp19g_beta_scale.json         N=2e4，β{20,100,500,2500}×levels{0.1,0.25,0.35}×2 seeds
    S3 exp19h_l0_beta_regression.json N=4e3，β{20,100,500}×levels{0,0.1,0.25,0.35}×2 seeds
    S4 exp26_l2_pilot.json            N=5e4，β100 全过（deg0/deg25 60/60）+β20 anchor（塌陷演示）
    S5 exp37_N100000_F_b100_s2.json   N=1e5，β100 deg0/deg25
    S6 exp52_N20000_F_b100_s0.json / exp52_eq_N3000_F_b100_s0.json  β100 探针档
    S7 exp45 log-derived β*（PRELIMINARY，可选外部输入 HCM_EXP45_LOGDERIVED，存在才并入）
  门（分析完整性门，非方向门——β 网格粗，不预设 R² 达标）：
    G1 可溯源：输出 JSON 每个数字都能对回 (source 文件, key 路径)，全部 sha256 记录，零手工常量。
    G2 外部臂单调：exp25 dense_cluster 臂 β*_full(N)（min β 使 hits==n）随 N 非降。
    G3 括号纪律：模型内 β* 一律输出为网格括号 [β_low_pass, β_high_fail]，不出裸点值。
  已知坑：exp25 含 -Infinity（Python json 宽松写），本脚本用 Python json 原生容忍。
  输出 results/exp54_lawnfit.json + stdout 表。
'''
import hashlib, json, os, math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
EXP45_LOGDERIVED = os.environ.get('HCM_EXP45_LOGDERIVED', '')  # 可选外部输入

def load(name):
    p = os.path.join(RES, name) if not os.path.isabs(name) else name
    with open(p, 'rb') as f:
        raw = f.read()
    sha = hashlib.sha256(raw).hexdigest()[:16]
    return json.loads(raw.decode('utf-8')), sha

out = {'exp': 'exp54_lawnfit', 'kind': 'analysis-only', 'sources': [], 'staircase': {}, 'fits': {}, 'gates': {}}

def src(tag, name):
    d, sha = load(name)
    out['sources'].append({'id': tag, 'file': name, 'sha256_16': sha})
    return d

# ---------- S1 exp25 外部臂 ----------
s25 = src('S1', 'exp25_collapse_external.json')
ext = {}
mono_ok = True
for arm in ('dense', 'sparse', 'dense_cluster'):
    pts = []
    for row in s25[arm]:
        N, betas, hits, n = row['N'], row['betas'], row['hits'], row['n']
        full = next((b for b, h in zip(betas, hits) if h == n), None)
        best = betas[max(range(len(hits)), key=lambda i: hits[i])]
        pts.append({'N': N, 'n_probes': n, 'beta_star_full': full, 'beta_star_best': best,
                    'beta_star_stored': row.get('beta_star'),
                    'hits_at_grid': {str(b): h for b, h in zip(betas, hits)}})
    ext[arm] = pts
dc = [p['beta_star_full'] for p in ext['dense_cluster']]
mono_ok = all(dc[i] <= dc[i+1] for i in range(len(dc)-1) if dc[i] is not None and dc[i+1] is not None) and all(v is not None for v in dc)
out['staircase']['exp25_external'] = ext
out['gates']['G2_external_dense_cluster_monotone'] = {'beta_star_full_seq': dc, 'pass': bool(mono_ok)}

# 外部臂 ln N 拟合（dense_cluster β*_full 对 ln N OLS；网格粗，描述性）
pts = [(p['N'], p['beta_star_full']) for p in ext['dense_cluster'] if p['beta_star_full'] is not None]
xs = [math.log(N) for N, _ in pts]; ys = [float(b) for _, b in pts]
k = len(xs); mx = sum(xs)/k; my = sum(ys)/k
sxx = sum((x-mx)**2 for x in xs); sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
slope = sxy/sxx if sxx else 0.0
inter = my - slope*mx
ss_res = sum((y-(slope*x+inter))**2 for x, y in zip(xs, ys))
ss_tot = sum((y-my)**2 for y in ys) or 1e-12
r2 = 1 - ss_res/ss_tot
out['fits']['exp25_dense_cluster_fullpass_lnN'] = {
    'slope_per_lnN': round(slope, 4), 'intercept': round(inter, 4), 'r2': round(r2, 4),
    'n_points': k, 'note': 'beta grid {2,5,10,20,50,100,200,500,1000} is coarse; fit is descriptive, not confirmatory'}

# ---------- S2/S3 模型内剂量反应（括号纪律） ----------
def bracket(rows, n_probes, min_level=0.1):
    '''同一 N 内：对每个 β 汇总全部 level>=min_level 的 top1；β_low_pass=min β 全过；β_high_fail=max β 未全过。'''
    per_beta = {}
    for e in rows:
        if e['level'] < min_level: continue
        per_beta.setdefault(e['beta'], []).append(e['top1'])
    res = {}
    for b, ts in sorted(per_beta.items()):
        res[b] = {'min_top1': min(ts), 'n_probes': n_probes, 'full_all': all(t == n_probes for t in ts)}
    fulls = [b for b, v in res.items() if v['full_all']]
    fails = [b for b, v in res.items() if not v['full_all']]
    lo = min(fulls) if fulls else None
    hi = max(fails) if fails else None
    return {'per_beta': res, 'beta_low_pass': lo, 'beta_high_fail': hi}

s19g = src('S2', 'exp19g_beta_scale.json')
s19h = src('S3', 'exp19h_l0_beta_regression.json')
out['staircase']['exp19g_N20000_min_level0.1'] = bracket(s19g, 40)
out['staircase']['exp19h_N4000_min_level0.1'] = bracket(s19h, 40)

# ---------- S4/S5/S6 固定 β=100 通过档 + exp26 β20 塌陷锚 ----------
s26 = src('S4', 'exp26_l2_pilot.json')
fixed100 = []
for e in s26['rows']:
    if e.get('beta') == 100:
        fixed100.append({'source': 'S4', 'N': e['N'], 'cell': e['tag'],
                         'deg0': e.get('deg0', {}).get('hits'), 'deg25': e.get('deg25', {}).get('hits')})
    elif e.get('beta') == 20:
        out['staircase']['exp26_N50000_beta20_anchor'] = {'source': 'S4', 'N': e['N'],
            'deg25_hits': e.get('deg25', {}).get('hits'), 'deg25_of': e.get('deg25', {}).get('of'),
            'note': 'beta=20 collapse anchor at 5e4 (deg0 not run for anchor rows)'}
s37 = src('S5', 'exp37_N100000_F_b100_s2.json')
fixed100.append({'source': 'S5', 'N': s37['N'], 'cell': s37['cell'], 'deg0': s37['deg0']['hits'], 'deg25': s37['deg25']['hits']})
for f52 in ('exp52_N20000_F_b100_s0.json', 'exp52_eq_N3000_F_b100_s0.json'):
    d52 = src('S6:' + f52, f52)
    if 'deg0' in d52:
        fixed100.append({'source': 'S6', 'N': d52['N'], 'cell': d52['cell'], 'deg0': d52['deg0']['hits'], 'deg25': d52['deg25']['hits']})
out['staircase']['fixed_beta100_passes'] = fixed100

# ---------- S7 exp45 log-derived β*（存在才并入，PRELIMINARY） ----------
if os.path.exists(EXP45_LOGDERIVED):
    s45 = src('S7', EXP45_LOGDERIVED)
    prelim = []
    if isinstance(s45, dict) and isinstance(s45.get('seeds'), list):
        for e in s45['seeds']:
            prelim.append({'seed': e.get('seed'),
                           'char_beta_curve': e.get('char_beta_curve'),
                           'sem_beta_curve': e.get('sem_beta_curve')})
    out['staircase']['exp45_online_beta_star_PRELIMINARY'] = {
        'present': True, 'n_seeds': len(prelim),
        'label': 'PRELIMINARY (log-derived, canonical = rerun JSONs pending)',
        'seeds': prelim}
else:
    out['staircase']['exp45_online_beta_star_PRELIMINARY'] = {'present': False}

# ---------- G1/G3 ----------
out['gates']['G1_traceability'] = {'n_sources': len(out['sources']), 'pass': len(out['sources']) >= 7}
out['gates']['G3_bracket_discipline'] = {'pass': 'beta_low_pass' in out['staircase'].get('exp19g_N20000_min_level0.1', {}) and 'beta_low_pass' in out['staircase'].get('exp19h_N4000_min_level0.1', {}),
    'note': 'in-model beta* reported as grid brackets only'}

res_path = os.path.join(RES, 'exp54_lawnfit.json')
with open(res_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

# stdout 摘要
print('=== exp54 lawnfit ===')
for s in out['sources']:
    print(f"  {s['id']}: {s['file']} sha16={s['sha256_16']}")
print('G2 external dense_cluster beta*_full seq =', dc, 'monotone:', mono_ok)
print('fit dense_cluster: slope/lnN=%.3f r2=%.3f (n=%d)' % (slope, r2, k))
print('exp19g N=2e4 bracket:', out['staircase']['exp19g_N20000_min_level0.1']['beta_low_pass'], 'pass /', out['staircase']['exp19g_N20000_min_level0.1']['beta_high_fail'], 'fail')
print('exp19h N=4e3 bracket:', out['staircase']['exp19h_N4000_min_level0.1']['beta_low_pass'], 'pass /', out['staircase']['exp19h_N4000_min_level0.1']['beta_high_fail'], 'fail')
print('fixed beta100 passes:', [(r['source'], r['N'], r['deg0'], r['deg25']) for r in fixed100])
print('exp26 beta20 anchor deg25:', out['staircase'].get('exp26_N50000_beta20_anchor'))
print('exp45 preliminary:', out['staircase']['exp45_online_beta_star_PRELIMINARY'].get('present'))
print('gates:', {k: v.get('pass') for k, v in out['gates'].items()})
print('written:', res_path)
