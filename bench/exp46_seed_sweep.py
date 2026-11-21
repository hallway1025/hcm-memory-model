'''E46（R4 清单项）：exp11 读出分层主张的 seed 扫描（20 seeds co-swept）。
预注册（2026-11-12 20:15，commit 5084d99 先于正式数据）：
- 20 seeds，X 与 HCM model 同 seed 联扫；其余参数与 exp11 逐项相同
  （D=128,N=60,PROBE=50,capacity=400,beta=20,L=3,use_resid=True,noise={0.30,0.45,0.60,0.70}）。
- degrade 种子=41000+seed*1000+i 防跨 seed 重合。
- primary：deg0.70 dense_hopfield − sparse_only probe 级配对差（池化 1000 对），
  sign-flip 200k（rng seed=46）→ p + Cohen dz。
- secondary：deg0.60 同款；dense_once 各级描述统计。
- 门：报告式双分支如实报（p<0.05 且 mean>0 ⇒ 分层主张 seed 扫描背书；否则降级为单 seed 观察）。
'''
import sys, os, json; sys.path.insert(0, 'src')
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import random_patterns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = 128; N = 60; PROBE = 50; SEEDS = list(range(20))
NOISES = (0.30, 0.45, 0.60, 0.70)

def degrade(x, frac, seed):
    r = np.random.default_rng(seed); y = x.copy()
    nz = np.where(np.abs(y) > 1e-6)[0]
    y[r.choice(nz, size=max(1, int(len(nz) * frac)), replace=False)] = 0.0
    return unit(y)

per_seed = []   # [{seed, noise, {arm: hits}}]
probe_diffs = {n: {'dh_minus_sp': [], 'do_minus_sp': []} for n in NOISES}
for sd in SEEDS:
    X = random_patterns(N, D, seed=sd)
    m = HCM(Config(d_x=D, a=0.02, capacity=400, beta=20.0, L=3, use_resid=True, seed=sd))
    for x in X:
        m.observe(x)
    Mn = m.E.M[:, :m.E.n]
    for noise in NOISES:
        hits = {'sparse_only': 0, 'dense_once': 0, 'dense_hopfield': 0}
        for i in range(PROBE):
            cue = degrade(X[i], noise, 41000 + sd * 1000 + i)
            xe = m.enc.resid_read(cue); s_pre = m.enc.pre(xe); q0 = m.enc.encode_from_pre(s_pre)
            res = {}
            for arm in hits:
                if arm == 'sparse_only':
                    sc = Mn.T @ q0
                elif arm == 'dense_once':
                    sc = Mn.T @ s_pre
                else:
                    z, a = m.E.retrieve(q0, L=3, beta=20.0, dense=s_pre)
                    sc = a * m.E.sup[:m.E.n]
                hit = int(int(np.argmax(sc)) == i)
                hits[arm] += hit
                res[arm] = hit
            probe_diffs[noise]['dh_minus_sp'].append(res['dense_hopfield'] - res['sparse_only'])
            probe_diffs[noise]['do_minus_sp'].append(res['dense_once'] - res['sparse_only'])
        per_seed.append(dict(seed=sd, noise=noise, **hits))
        print('seed %02d noise %.2f sp=%02d do=%02d dh=%02d' % (sd, noise, hits['sparse_only'], hits['dense_once'], hits['dense_hopfield']), flush=True)

def signflip(diffs, rng_seed=46, n_perm=200000):
    d = np.asarray(diffs, dtype=np.float64)
    obs = float(d.mean())
    sd = float(d.std(ddof=1))
    dz = obs / sd if sd > 0 else float('nan')
    r = np.random.default_rng(rng_seed)
    signs = r.choice(np.array([-1.0, 1.0]), size=(n_perm, len(d)))
    perm = (signs * d).mean(axis=1)
    p = float((np.abs(perm) >= abs(obs) - 1e-12).mean())
    return dict(n=len(d), mean=obs, dz=dz, p=p, n_perm=n_perm)

tests = {('%.2f' % n): dict(
    dh_minus_sp=signflip(probe_diffs[n]['dh_minus_sp']),
    do_minus_sp=signflip(probe_diffs[n]['do_minus_sp'])) for n in NOISES}
agg = {('%.2f' % n): {a: sum(r[a] for r in per_seed if r['noise'] == n) for a in ('sparse_only', 'dense_once', 'dense_hopfield')} for n in NOISES}
out = dict(pre_registered_commit='5084d99', seeds=SEEDS, params=dict(D=D, N=N, PROBE=PROBE, nois_es=list(NOISES)),
           per_seed=per_seed, pooled_hits_per_1000=agg, tests=tests)
json.dump(out, open(os.path.join(ROOT, 'results', 'exp46_seed_sweep.json'), 'w'), indent=1)
print('--- pooled /1000:', json.dumps(agg))
print('--- primary deg0.70 dh-sp:', json.dumps(tests['0.70']['dh_minus_sp']))
print('saved results/exp46_seed_sweep.json')
