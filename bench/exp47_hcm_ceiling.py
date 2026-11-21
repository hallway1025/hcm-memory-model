# -*- coding: utf-8 -*-
'''exp47 supplement: exp38 hcm ceiling — how many of the 200 probes were even written by the gated HCM arm (fairness note for baseline comparison).'''
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp38_baselines as e38
import numpy as np

def main():
    t0 = time.time()
    rows = e38.load('train')
    texts = [r[2] for r in rows]
    n_lib = len(texts)
    Xall = np.asarray([e38.enc(t) for t in texts], dtype=np.float32)
    csets = [set(e38._clean(t)) for t in texts]
    probes = []
    for i in range(n_lib):
        lo, hi = max(0, i - 200), min(n_lib, i + 200)
        ov = max((len(csets[i] & csets[j]) / max(1, min(len(csets[i]), len(csets[j]))))
                 for j in range(lo, hi) if j != i)
        if ov < 0.5: probes.append(i)
        if len(probes) >= 200: break
    arm = e38.HCMArm(n_lib)
    for i in range(n_lib): arm.write(Xall[i])
    m = arm.m
    obs_of = {int(m.E.t[j]): j for j in range(int(m.E.n))}
    wp = sum(1 for i in probes if obs_of.get(i, -1) >= 0)
    # verify hit reproduction at deg0.3
    hit = 0
    for i in probes:
        q = e38.degrade_chars(texts[i], 0.3, 101)
        sc = arm.scores(q)
        hit += int(int(np.argmax(sc)) == obs_of.get(i, -1))
    out = dict(n_lib=n_lib, probes=len(probes), written=int(m.E.n),
               probes_written=wp, probes_written_rate=wp / len(probes),
               hit_deg03_repro=hit, note='hcm ceiling for baseline fairness note')
    p = os.path.join(e38.RES, 'exp47_hcm_ceiling.json')
    with open(p, 'w', encoding='utf-8') as f: json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out), flush=True)
    print('CEILING DONE %.0fs' % (time.time() - t0), flush=True)

if __name__ == '__main__':
    main()