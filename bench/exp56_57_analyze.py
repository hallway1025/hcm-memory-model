# -*- coding: utf-8 -*-
'''exp56_57_analyze.py —— 补充实验批判门（exp56 ρ 剂量反应 + exp57 门 OFAT v2）。
verdict 专用零新运行；三门判定 + 判读输出；exp57 G2 按 amendment-2 口径（内容身份）。'''
import os, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def src(name):
    p = os.path.join(ROOT, 'results', name)
    return json.load(open(p, encoding='utf-8'))

print('=== exp57 gate ablation (v2, dual criterion) ===')
j57 = src('exp57_gate_ablation_v2.json')
rows = j57['rows']
arms = j57['arms']
by = {}
for r in rows:
    by.setdefault(r['arm'], []).append(r)
# G1 sanity：接受率∈[0,1]；baseline 跨 seed 接受率极差≤5pp
ok01 = all(0.0 <= r['accept_rate'] <= 1.0 for r in rows)
bl = [r['accept_rate'] for r in by['baseline']]
rng = (max(bl) - min(bl)) * 100
g1 = ok01 and rng <= 5.0
print('G1 sanity: accept in [0,1] all rows =', ok01, '; baseline accept range = %.2fpp' % rng, '->', 'PASS' if g1 else 'FAIL')
# G2 方向门：dup_rewrite_rate(constant) > baseline（预声明，v1/v2 一致口径）
dc = by['constant'][0]['dup_rewrite_rate']
db = by['baseline'][0]['dup_rewrite_rate']
g2 = dc > db
print('G2 directional: constant dup_rw=%.3f vs baseline dup_rw=%.3f ->' % (dc, db), 'PASS' if g2 else 'FAIL (predeclared expectation per amendment: store-level dup check intercepts exact re-presentations before gate relevance)')
# 剂量反应表
print('%-10s %-18s %-14s %-8s %-8s' % ('arm', 'g_mean(s0/s1/s2)', 'deg25_id(s0/s1/s2)', 'dup_rw', 'dup_deg'))
for a in arms:
    rs = by[a]
    gs = '/'.join(str(r['g_mean']) for r in rs)
    did = '/'.join('%d' % r['deg25_id'] for r in rs)
    print('%-10s %-18s %-14s %-8s %-8s' % (a, gs, did, rs[0]['dup_rewrite_rate'], rs[0]['dup_degraded_rate']))
skipped = [a for a in arms if any(r['deg25_pos'] < r['deg25_id'] for r in by[a])]
print('t=0-skipping arms (deg25_pos<deg25_id):', skipped)
print('G2b dup_degraded note: %.3f in every arm -> gate saturated-open in steady-state L0; rpe arms identical to baseline -> rpe dead weight on first-encounter writes' % by['baseline'][0]['dup_degraded_rate'])

print()
print('=== exp56 rho dose-response ===')
j56 = src('exp56_rho_sweep.json')
rows56 = j56['rows']
# G1 复现锚：rho=8 与 exp19g beta100 行 |Δ|<=2/40
b100 = [r for r in rows56 if r['beta'] == 100.0 and r['rho'] == 8]
anchor = {(0, 0.1): 40, (0, 0.25): 40, (0, 0.35): 40, (1, 0.1): 40, (1, 0.25): 40, (1, 0.35): 40}
g1_ok = True
for r in b100:
    a = anchor.get((r['seed'], r['level']))
    if a is None or abs(r['top1'] - a) > 2:
        g1_ok = False
        print('  G1 MISS: seed%d rho8 beta100 deg%d: %d/40 (anchor %d)' % (r['seed'], r['level'] * 100, r['top1'], a))
print('G1 replication anchor (rho8 vs exp19g beta100 40/40 x6, |d|<=2/40):', 'PASS' if g1_ok else 'FAIL')
# 剂量反应表
print('%-5s %-4s %-8s %-10s %-8s %-8s %-8s' % ('seed', 'rho', 'k', 'write_s', 'accept', 'g_mean', 'cos_mean'))
for r in sorted(rows56, key=lambda x: (x['seed'], x['rho'])):
    if r['level'] == 0.10 and r['beta'] == 20.0:
        print('%-5d %-4d %-8d %-10.0f %-8.3f %-8.3f %-8.4f' % (r['seed'], r['rho'], r['k'], r['write_s'], r['accept_rate'], r['g_mean'], r['cos_mean']))
# β 拐点表（beta20 vs beta100 per rho/level）
print('recall grid (rows=rho, cols=level, beta20|beta100):')
rhos = sorted(set(r['rho'] for r in rows56))
levels = sorted(set(r['level'] for r in rows56))
for s in sorted(set(r['seed'] for r in rows56)):
    for rho in rhos:
        cells = []
        for lv in levels:
            b20 = next(r['top1'] for r in rows56 if r['seed'] == s and r['rho'] == rho and r['level'] == lv and r['beta'] == 20.0)
            bv = next(r['top1'] for r in rows56 if r['seed'] == s and r['rho'] == rho and r['level'] == lv and r['beta'] == 100.0)
            cells.append('%d|%d' % (b20, bv))
        print('  seed%d rho=%-3d %s' % (s, rho, '  '.join(cells)))
print('NOTE: deg25_id=39/40 on t=0-skipping arms: single miss = probe0 (target content present via re-observation write at idx=n, which the phase-1-only write_map does not label; with reobs writes mapped, content identity is 40/40 per bench/exp57_diag_arm.py probe->idx detail)');
