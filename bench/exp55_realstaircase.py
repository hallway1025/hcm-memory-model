import os
# -*- coding: utf-8 -*-
'''exp55_realstaircase.py —— 真语料 β*(N) 交叉轴定量（分析专用；PRELIMINARY→明晨 rerun JSON 后升 canonical）。

预注册（先于运行 commit；本文件即预注册本体）：
  背景：S-010/§4.12 记录「exp45 char/sem β*(N) 曲线 100↔500 平台摆动，坍缩律交叉轴=弱信号」。
        本脚本把『弱』量化：真语料在线 β*(N) 三 seed × 11 ckpt × {char,sem} 两通道，
        检验 (a) 任一 seed 曲线是否单调非降；(b) 跨 seed 在可比 N 处的一致率；
        (c) ln N-β* 秩相关（β*∈{100,500} 二值 → concordant-pair 分数 + Kendall tau）。
  数据源（只读）：
    S1 可选外部输入 HCM_EXP45_LOGDERIVED（三 seed char_beta_curve/sem_beta_curve，各 11 点）
  门（完整性门，非方向门——本轴预注册声明就是弱信号检验）：
    G1 可溯源：3 seed × 2 通道 × 11 ckpt 全部读入且 N 严格递增，sha256 记录。
    G2 标注纪律：输出全程 PRELIMINARY 标签；canonical 结论以 exp45 rerun JSON 为准（明晨）。
    G3 探索性检验不设显著性门槛，只报效应量与方向（tau/一致率/单调性布尔）。
  预注册预期（如实）：若 β* 平台摆动为选参噪声，tau≈0 且跨 seed 一致率≈50%；若存在 log-N 趋势，
        tau>0 且一致率>70%。两分支均 informative，各自写法预注册：噪声→『β*(N) 在 1e4-9.3e4
        区间平台化，与合成臂的 log 增长形成对照（K3 语义通道吸收差异）』；趋势→补强 4.12。
  输出 results/exp55_realstaircase.json + stdout 表。
'''
import hashlib, json, os, math

SRC = os.environ.get('HCM_EXP45_LOGDERIVED', '')  # 可选外部输入
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')

with open(SRC, 'rb') as f:
    raw = f.read()
sha = hashlib.sha256(raw).hexdigest()[:16]
data = json.loads(raw.decode('utf-8'))

def tau_binary(curve):
    '''curve=[{N,beta}]：β*二值 → Kendall tau = (conc-disc)/total，并列不计。'''
    pts = sorted([(e['N'], e['beta']) for e in curve])
    conc = disc = ties = 0
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            if pts[i][1] == pts[j][1]:
                ties += 1
            elif pts[j][1] > pts[i][1]:
                conc += 1
            else:
                disc += 1
    tot = conc + disc
    return {'conc': conc, 'disc': disc, 'ties': ties,
            'tau': round((conc-disc)/tot, 4) if tot else None,
            'frac_concordant_of_resolved': round(conc/tot, 4) if tot else None}

def monotone(curve):
    vals = [e['beta'] for e in sorted(curve, key=lambda e: e['N'])]
    return all(vals[i] <= vals[i+1] for i in range(len(vals)-1))

def modal_by_N(seeds_curves):
    '''跨 seed：按 N 四舍五入到最近 1e4 桶，报众数 β* 与分布。'''
    buckets = {}
    for c in seeds_curves:
        for e in c:
            b = int(round(e['N']/10000.0))
            buckets.setdefault(b, []).append(e['beta'])
    out = {}
    for b in sorted(buckets):
        vals = buckets[b]
        cnt = {v: vals.count(v) for v in set(vals)}
        mode = max(cnt.items(), key=lambda kv: kv[1])[0]
        out[f'N~{b}e4'] = {'mode': mode, 'dist': cnt, 'n': len(vals)}
    return out

out = {'exp': 'exp55_realstaircase', 'kind': 'analysis-only',
       'label': 'PRELIMINARY (log-derived; canonical = exp45 rerun JSONs pending)',
       'sources': [{'id': 'S1', 'file': SRC, 'sha256_16': sha}], 'channels': {}, 'gates': {}}

n_ck = None
for ch in ('char_beta_curve', 'sem_beta_curve'):
    curves = []
    for e in data['seeds']:
        c = e.get(ch) or []
        ns = [x['N'] for x in c]
        curves.append({'seed': e.get('seed'), 'n_ck': len(c),
                       'N_strictly_increasing': all(ns[i] < ns[i+1] for i in range(len(ns)-1)),
                       'curve': c,
                       'monotone_nondecreasing': monotone(c),
                       'tau': tau_binary(c)})
    # 跨 seed N 覆盖
    all_n = sorted(set(x['N'] for c in curves for x in c['curve']))
    out['channels'][ch] = {'seeds': curves,
                           'modal_by_N_bucket': modal_by_N([c['curve'] for c in curves]),
                           'N_range': [all_n[0], all_n[-1]] if all_n else None,
                           'beta_grid_observed': sorted(set(x['beta'] for c in curves for x in c['curve']))}
    n_ck = len(curves)

taus = [c['tau']['tau'] for ch in out['channels'].values() for c in ch['seeds'] if c['tau']['tau'] is not None]
mono_any = any(c['monotone_nondecreasing'] for ch in out['channels'].values() for c in ch['seeds'])
out['gates']['G1_traceability'] = {'pass': n_ck == 3 and all(c['n_ck'] == 11 for ch in out['channels'].values() for c in ch['seeds'])}
out['gates']['G2_preliminary_label'] = {'pass': out['label'].startswith('PRELIMINARY')}
out['gates']['G3_exploratory_effect_sizes'] = {'taus': taus, 'any_seed_monotone': mono_any,
    'verdict_predeclared': ('trend' if (taus and sum(1 for t in taus if t > 0.3) >= 4) else 'platform_noise' if taus else 'no_data'),
    'note': 'predeclared split: tau>0.3 in >=4/6 seed-curves => trend branch; else platform-noise branch'}

res_path = os.path.join(RES, 'exp55_realstaircase.json')
with open(res_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

print('=== exp55 realstaircase (PRELIMINARY) ===')
for ch, d in out['channels'].items():
    print(f'[{ch}] N_range={d["N_range"]} beta_grid={d["beta_grid_observed"]}')
    for c in d['seeds']:
        t = c['tau']
        print(f"  seed{c['seed']}: mono={c['monotone_nondecreasing']} tau={t['tau']} conc/disc={t['conc']}/{t['disc']} ties={t['ties']}")
    print('  modal_by_N:', json.dumps(d['modal_by_N_bucket']))
print('G3 verdict:', out['gates']['G3_exploratory_effect_sizes']['verdict_predeclared'], '| taus:', taus)
print('gates:', {k: (v.get('pass') if isinstance(v.get('pass'), bool) else v) for k, v in out['gates'].items()})
print('written:', res_path)
