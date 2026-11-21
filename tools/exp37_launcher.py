#!/usr/bin/env python3
'''exp37 launcher: memory-aware job pool. N=1e5 class ~11GB (max 6 concurrent),
N=2e5 class ~21GB (max 3 concurrent). 12 cores, single-thread per job.'''
import os, subprocess, time, itertools

QUEUE = []
SEEDS = {100000: (0, 1, 2), 200000: (0, 1)}
PROBES = {100000: '60', 200000: '30'}
for Nv in (100000, 200000):
    for cell in ('F_b100', 'T_b100'):
        for seed in SEEDS[Nv]:
            if os.path.exists('results/exp37_N%d_%s_s%d.json' % (Nv, cell, seed)):
                print('SKIP done N%d %s s%d' % (Nv, cell, seed), flush=True)
                continue
            QUEUE.append((Nv, cell, seed))
MAXJ = {100000: 5, 200000: 2}
running = []  # (N, popen, logf)
LOGD = 'results/exp37_logs'
os.makedirs(LOGD, exist_ok=True)
env0 = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONHASHSEED='0')
py = '/root/miniconda3/bin/python'
while QUEUE or running:
    for Nv, cell, seed in list(QUEUE):
        if sum(1 for n, _, _ in running if n == Nv) >= MAXJ[Nv]:
            continue
        if len(running) >= 7:
            continue
        logf = open('%s/N%d_%s_s%d.log' % (LOGD, Nv, cell, seed), 'w')
        env = dict(env0, EXP37_N=str(Nv), EXP37_CELL=cell, EXP37_SEED=str(seed), EXP37_PROBES=PROBES[Nv])
        p = subprocess.Popen([py, 'bench/exp37_l2_scalelaw.py'], env=env, stdout=logf, stderr=subprocess.STDOUT, cwd='/root/autodl-tmp/hcm')
        running.append((Nv, p, logf))
        QUEUE.remove((Nv, cell, seed))
        print('START N%d %s s%d pid=%d queue=%d running=%d' % (Nv, cell, seed, p.pid, len(QUEUE), len(running)), flush=True)
        time.sleep(2)
    time.sleep(10)
    for Nv, p, logf in list(running):
        if p.poll() is not None:
            logf.close()
            running.remove((Nv, p, logf))
            print('EXIT  N%d %s rc=%s running=%d' % (Nv, p.args, p.returncode, len(running)), flush=True)
print('ALL DONE', flush=True)
with open('results/exp37_launcher_done.flag', 'w') as f:
    f.write(time.strftime('%Y-%m-%d %H:%M:%S'))
