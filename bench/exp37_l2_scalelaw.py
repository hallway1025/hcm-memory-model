#!/usr/bin/env python3
'''exp37_l2_scalelaw.py -- L2 scale law on AutoDL (exp37, round 5).
Sweeps N in {1e5, 2e5} x cells {F_b100, T_b100} x seeds {0,1,2} at d_x=2048, d_h=16384.
Synthetic unit patterns only (no corpus). theta==0 pinned (D-038); judge obs_of (L-035).
Purpose: extend the beta=100 scale-law claim (sec 4.12) from N=5e4 (exp26) to 2e5,
with cross-platform replication of exp26 at identical config (machine independence).
Pre-registered gates (docstring-frozen before data):
  G-S1: beta=100 recall@1 at deg25 >= 0.90 for all N (sufficiency holds at scale)
  G-S2: no-resid and with-resid arms remain statistically indistinguishable at deg0
        (|rate diff| < 0.05, 3 seeds each)
  G-S3: write time scales ~linearly with N (Pearson r >= 0.95 over the 3 N points incl. exp26)
Run via: EXP37_N / EXP37_CELL / EXP37_SEED env vars, one JSON per run.
'''
import sys, os, json, time, resource
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = 2048
N = int(os.environ['EXP37_N'])
CELL = os.environ['EXP37_CELL']
PROBES = int(os.environ.get('EXP37_PROBES', '60'))
SEED = int(os.environ['EXP37_SEED'])

def peak_rss_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)  # Linux KB

def pats(n, d, seed):
    r = np.random.default_rng(seed)
    return np.asarray([unit(v) for v in r.standard_normal((n, d))], dtype=np.float32)

def degrade(x, frac, seed):
    r = np.random.default_rng(seed)
    y = x.copy()
    nz = np.where(np.abs(y) > 1e-6)[0]
    y[r.choice(nz, size=max(1, int(len(nz) * frac)), replace=False)] = 0.0
    return unit(y)

def main():
    assert CELL in ('F_b100', 'T_b100')
    resid = CELL.startswith('T')
    t0 = time.time()
    X = pats(N, D, seed=700)
    m = HCM(Config(d_x=D, d_c=D, a=0.02, capacity=N + 10, beta=100.0, L=3,
                   use_resid=resid, seed=SEED))
    tw = time.time(); wt = []
    for i, x in enumerate(X):
        t1 = time.time()
        m.observe(x)
        wt.append(time.time() - t1)
        if (i + 1) % 25000 == 0:
            print('%s s%d write %d/%d %.0fs rss=%sMB' % (CELL, SEED, i + 1, N, time.time() - tw, peak_rss_mb()), flush=True)
    write_s = round(time.time() - tw, 1)
    n_written = int(m.E.n)
    obs_of = {int(m.E.t[j]): j for j in range(n_written)}
    Mn_mb = round(m.E.M[:, :n_written].nbytes / 1048576, 1) if hasattr(m.E, 'M') else -1
    probe = list(range(0, N, max(1, N // PROBES)))[:PROBES]
    out = {}
    for frac in (0.0, 0.25):
        tq = time.time(); hits = 0; abst = 0
        for i in probe:
            q = X[i] if frac == 0 else degrade(X[i], frac, i)
            o = m.query(q)
            hits += int(obs_of.get(int(o['idx']), -1) == i)
            abst += int(o.get('action') == 'abstain')
        dt = (time.time() - tq) / len(probe)
        out['deg%g' % (frac * 100)] = dict(hits=hits, of=len(probe), rate=round(hits / len(probe), 4),
                                           abstain=abst, s_per_query=round(dt, 3))
        print('%s s%d deg%g: %d/%d abst=%d %.3fs/q' % (CELL, SEED, frac * 100, hits, len(probe), abst, dt), flush=True)
    del m, X
    row = dict(exp='exp37', N=N, d_x=D, cell=CELL, resid=resid, seed=SEED,
               write_s=write_s, s_per_write=round(sum(wt) / len(wt), 4),
               n_written=n_written, Mn_dense_mb=Mn_mb, peak_rss_mb=peak_rss_mb(),
               total_s=round(time.time() - t0, 1), **out)
    os.makedirs('results', exist_ok=True)
    with open('results/exp37_N%d_%s_s%d.json' % (N, CELL, SEED), 'w') as f:
        json.dump(row, f, indent=1)
    print('DONE', json.dumps({k: row[k] for k in ('N', 'cell', 'seed', 'write_s', 'peak_rss_mb')}), flush=True)

if __name__ == '__main__':
    main()
