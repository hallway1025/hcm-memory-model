#!/usr/bin/env python3
'''exp26_l2_pilot.py -- L2 pilot (d_x=2048, d_h=16384, k~328, N=5e4).
Questions: does beta=100 still suffice at N=5e4 (scale-law extrapolation, sec 4.12);
is residual neutrality preserved at L2 dims; timing + memory profile (sec 7 five questions 1/2, first step).
Theta protocol pinned: theta==0 (D-038). Judge: obs_of (L-035).'
'''
import sys, os, json, time, ctypes
from ctypes import wintypes
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = 2048
N = int(os.environ.get('EXP26_N', '50000'))
PROBES = int(os.environ.get('EXP26_PROBES', '60'))

class PMC(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
                ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]

def peak_rss_mb():
    pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
    h = ctypes.windll.kernel32.GetCurrentProcess()
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
    return round(pmc.PeakWorkingSetSize / 1048576, 1) if ok else -1

def pats(n, d, seed):
    r = np.random.default_rng(seed)
    return np.asarray([unit(v) for v in r.standard_normal((n, d))], dtype=np.float32)

def degrade(x, frac, seed):
    r = np.random.default_rng(seed)
    y = x.copy()
    nz = np.where(np.abs(y) > 1e-6)[0]
    y[r.choice(nz, size=max(1, int(len(nz) * frac)), replace=False)] = 0.0
    return unit(y)

def run_cell(resid, beta, deg_fracs, tag):
    t0 = time.time()
    X = pats(N, D, seed=700)
    m = HCM(Config(d_x=D, d_c=D, a=0.02, capacity=N + 10, beta=beta, L=3,
                   use_resid=resid, seed=0))
    tw = time.time()
    for i, x in enumerate(X):
        m.observe(x)
        if (i + 1) % 10000 == 0:
            print('  %s write %d/%d  %.0fs  peakRSS=%sMB' % (tag, i + 1, N, time.time() - tw, peak_rss_mb()), flush=True)
    write_s = round(time.time() - tw, 1)
    n_written = int(m.E.n)
    obs_of = {int(m.E.t[j]): j for j in range(n_written)}
    Mn_mb = round(m.E.M[:, :n_written].nbytes / 1048576, 1) if hasattr(m.E, 'M') else -1
    probe = list(range(0, N, max(1, N // PROBES)))[:PROBES]
    out = {}
    for frac in deg_fracs:
        tq = time.time(); hits = 0; abst = 0
        for i in probe:
            q = X[i] if frac == 0 else degrade(X[i], frac, i)
            o = m.query(q)
            hits += int(obs_of.get(int(o['idx']), -1) == i)
            abst += int(o.get('action') == 'abstain')
        dt = (time.time() - tq) / len(probe)
        out['deg%g' % (frac * 100)] = dict(hits=hits, of=len(probe),
            rate=round(hits / len(probe), 4), abstain=abst, s_per_query=round(dt, 2))
        print('%-14s resid=%s beta=%g deg%g: %d/%d  abst=%d  %.2fs/q' % (tag, resid, beta, frac * 100, hits, len(probe), abst, dt), flush=True)
    del m, X
    row = dict(N=N, d_x=D, resid=resid, beta=beta, tag=tag, write_s=write_s,
               n_written=n_written, Mn_dense_mb=Mn_mb, peak_rss_mb=peak_rss_mb(),
               cell_s=round(time.time() - t0, 1), **out)
    return row

if __name__ == '__main__':
    cells = [
        ('F_b100', False, 100.0, [0.0, 0.25]),
        ('T_b100', True, 100.0, [0.0, 0.25]),
        ('F_b20_anchor', False, 20.0, [0.25]),
        ('T_b20_anchor', True, 20.0, [0.25]),
    ]
    only = os.environ.get('EXP26_CELLS', '')
    if only:
        want = set(x for x in only.split(',') if x)
        cells = [c for c in cells if c[0] in want]
        print('EXP26_CELLS filter: %s' % sorted(want), flush=True)
    rows = []
    print('EXP26 L2 pilot: N=%d d_x=%d probes=%d pid=%d' % (N, D, PROBES, os.getpid()), flush=True)
    out_path = os.path.join(ROOT, 'results', 'exp26_l2_pilot.json')
    cfg = dict(N=N, d_x=D, d_h=16384, a=0.02, L=3, theta='pinned_zero_D-038', judge='obs_of',
               cells=[c[0] for c in cells])
    for tag, resid, beta, fr in cells:
        rows.append(run_cell(resid, beta, fr, tag))
        # checkpoint after every cell (2026-11-07: full run lost to harness restart;
        # a kill now loses at most the in-flight cell)
        json.dump(dict(config=cfg, rows=rows),
                  open(out_path, 'w', encoding='utf-8'), indent=1)
        print('checkpointed %d cell(s) -> results/exp26_l2_pilot.json' % len(rows), flush=True)
    print('saved results/exp26_l2_pilot.json', flush=True)