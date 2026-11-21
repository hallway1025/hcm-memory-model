'''exp15: P0-2 修复 —— priority vs uniform 的 20 种子配对统计

审查者指出 3 个种子功效不足（他们跑了 20 种子：uniform 15.25+/-3.29 vs
priority 14.35+/-3.31, t=0.77）。本脚本在修正版 exp6 代码上以 n=20 复测，
用配对 sign-flip 置换检验与 Cohen dz 报告效应是否可检出。
Protocol 与 exp6 v2 完全一致：CATS/Htr 沿用 m0 编码（原型种子 5）。'''
import sys, json, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location('e6', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp6_continual.py'))
e6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e6)

SEEDS = 20
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

res = {a: {'A': [], 'B': []} for a in ('blocked', 'uniform', 'priority')}
for arm in res:
    for s in range(SEEDS):
        m = e6.run(arm, seed=1000 + s)
        x, _ = e6.acc(m, set(range(4)));   res[arm]['A'].append(x)
        y, _ = e6.acc(m, set(range(4, 8))); res[arm]['B'].append(y)
    A = np.array(res[arm]['A'], float); B = np.array(res[arm]['B'], float)
    ci = 1.96 * A.std(ddof=1) / np.sqrt(SEEDS)
    print(f"{arm:9s} A: {A.mean():5.2f} +/- {A.std(ddof=1):4.2f} (95%CI +/- {ci:4.2f})   B: {B.mean():5.2f} +/- {B.std(ddof=1):4.2f}", flush=True)

u = np.array(res['uniform']['A'], float); p = np.array(res['priority']['A'], float)
d = u - p
rng = np.random.default_rng(0)
signs = rng.choice([-1.0, 1.0], size=(200000, SEEDS))
pv = float((np.abs((signs * d).mean(axis=1)) >= abs(d.mean()) - 1e-12).mean())
dz = float(d.mean() / max(d.std(ddof=1), 1e-9))
print(f"uniform-priority paired diff: {d.mean():+.2f} +/- {d.std(ddof=1):.2f}, sign-flip p={pv:.4f}, Cohen dz={dz:.3f}")
json.dump(dict(seeds=SEEDS,
               arms={a: dict(A=[float(v) for v in res[a]['A']], B=[float(v) for v in res[a]['B']]) for a in res},
               diff_mean=float(d.mean()), diff_std=float(d.std(ddof=1)), p_value=pv, cohen_dz=dz),
          open(os.path.join(ROOT, 'results', 'exp15_priority_power.json'), 'w'), indent=1)
print('saved results/exp15_priority_power.json')