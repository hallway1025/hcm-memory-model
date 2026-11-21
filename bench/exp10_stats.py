'''E10：统计稳健性——为论文的关键主张补置信区间与多种子方差。
论文里的「x/30」都是单种子单次计数，没有方差。这里对三条最关键的主张
做多种子重复（每个种子重新生成数据与查询扰动），报告 mean ± 95% CI。'''
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import overlapping_set

D, N = 128, 30
SEEDS = list(range(12))

def degrade(x, frac, seed):
    if frac <= 0: return x
    r = np.random.default_rng(seed); y = x.copy()
    nz = np.where(np.abs(y) > 1e-6)[0]
    y[r.choice(nz, size=int(len(nz) * frac), replace=False)] = 0.0
    return unit(y)

def ci95(vals):
    v = np.asarray(vals, dtype=float)
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0

def run_cond(overlap, resid, L, noise, seed):
    X = overlapping_set(N, D, overlap=overlap, seed=1000 + seed)
    m = HCM(Config(d_x=D, a=0.02, capacity=500, beta=20.0, L=L, use_resid=resid, seed=seed))
    for x in X: m.observe(x)
    return sum(int(m.query(degrade(X[i], noise, seed * 100 + i))['idx'] == i) for i in range(N)), m.E.n

res = {}
print('12 个种子重复，报告 mean ± 95%CI (满分 30)', flush=True)
for label, overlap, resid, L_, noise in [
        ('ov0.90 无残差/精确', 0.90, False, 0, 0.0),
        ('ov0.90 有残差/精确', 0.90, True, 0, 0.0),
        ('ov0.75 无残差/精确', 0.75, False, 0, 0.0),
        ('ov0.75 有残差/精确', 0.75, True, 0, 0.0),
        ('ov0.75 有残差/缺失20%', 0.75, True, 0, 0.20),
        ('ov0.75 有残差L3/缺失20%', 0.75, True, 3, 0.20),
]:
    hits, written = [], []
    for s in SEEDS:
        h, w = run_cond(overlap, resid, L_, noise, s)
        hits.append(h); written.append(w)
    mu, hw = ci95(hits)
    wmu, ww = ci95(written)
    res[label] = dict(mean=round(mu, 2), ci95=round(hw, 2), n=len(SEEDS),
                      per_seed=hits, written_mean=round(wmu, 1))
    print('  %-26s %.2f ± %.2f   写入 %.1f' % (label, mu, hw, wmu), flush=True)

json.dump(res, open(os.path.join(ROOT, 'results', 'exp10_stats.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('saved results/exp10_stats.json')