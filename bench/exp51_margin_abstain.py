# -*- coding: utf-8 -*-
"""exp51 - margin-based abstention signal on public corpus (preregistered ledger row exp51).

exp40-GP3 honest FAIL: learned-confidence abstention gate reversed on the homogeneous
closed corpus (AUC 0.335/0.063 < 0.5). Simplest untested alternative: the retrieval
margin (top1 - top2 of the episodic score vector).

Protocol: exp40-identical clean probes (100 lowest +/-200-window max-overlap train
indices), NEW256 arm, N=8000, deg {0.3, 0.5} (exp40 T1 seeds: 101), each probe yields
(margin, hit). AUC(margin -> hit) via Mann-Whitney; permutation p by label shuffling
(200k). PREREGISTERED GATE: AUC > 0.5 AND p < 0.05 in either deg -> usable abstention
signal exists; otherwise the FAIL deepens, recorded honestly. Reference: exp40 learned
AUC 0.335/0.063. Power note (exp48): n=100 -> AUC MDE ~0.14.
"""
import csv, json, os, re, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from hcm import HCM, Config
from hcm.text_encoder import CharacterSemanticEncoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NANO = re.compile(r'\[[^\]]{1,8}\]')
def _clean(s): return NANO.sub('', re.sub(r'\s+', ' ', s)).strip()

def load_train():
    p = os.path.join(ROOT, 'bench', 'corpus_public', 'train__dailydialog.csv')
    with open(p, encoding='utf-8-sig', newline='') as f:
        rd = csv.reader(f); next(rd)
        return [r[2] for r in rd]

def auc_rank(ins, outs):
    allv = [(v, 1) for v in ins] + [(v, 0) for v in outs]
    allv.sort(key=lambda x: x[0])
    n_in, n_out = len(ins), len(outs)
    if n_in == 0 or n_out == 0: return None
    rank_sum_in = 0.0; i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]: j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            if allv[k][1] == 1: rank_sum_in += avg_rank
        i = j + 1
    return (rank_sum_in - n_in * (n_in + 1) / 2.0) / (n_in * n_out)

def perm_auc(ins, outs, n_perm=200000, seed=51):
    obs = auc_rank(ins, outs)
    lab = np.asarray([1] * len(ins) + [0] * len(outs))
    vals = np.asarray(list(ins) + list(outs), dtype=np.float64)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        s = rng.permutation(lab)
        a = auc_rank(vals[s == 1].tolist(), vals[s == 0].tolist())
        if a is not None and a >= obs - 1e-12: ge += 1
    return obs, ge / n_perm

def main():
    try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass
    t0 = time.time()
    texts = load_train()[:8000]
    enc = CharacterSemanticEncoder(dim=256, window=3, seed=0)
    enc.fit(texts)
    Xs = np.asarray([enc.encode(t) for t in texts], dtype=np.float32)
    cfg = Config(d_x=256, d_c=256, a=0.02, capacity=len(texts) + 10, beta=100.0, L=3,
                 use_resid=False, seed=0)
    m = HCM(cfg)
    for i in range(len(texts)):
        m.observe(Xs[i])
    nw = int(m.E.n)
    obs_of = {int(m.E.t[j]): j for j in range(nw)}
    print('written %d/%d' % (nw, len(texts)), flush=True)

    csets = [set(_clean(t)) for t in texts]
    ovs = []
    for i in range(len(texts)):
        lo, hi = max(0, i - 200), min(len(texts), i + 200)
        ov = max((len(csets[i] & csets[j]) / max(1, min(len(csets[i]), len(csets[j]))))
                 for j in range(lo, hi) if j != i)
        ovs.append((ov, i))
    ovs.sort()
    probes = sorted(i for _, i in ovs[:100])

    rows = {}
    for deg in (0.3, 0.5):
        margins, hits = [], []
        per = []
        for pid, i in enumerate(probes):
            r = np.random.default_rng(101 if deg == 0.3 else 102)
            chs = list(_clean(texts[i]))
            if len(chs) > 1:
                for j in r.choice(len(chs), size=max(1, int(len(chs) * deg)), replace=False):
                    chs[j] = ''
            q = enc.encode(''.join(c for c in chs if c))
            pre = m.enc.raw(q)
            _, a = m.E.retrieve(m.enc.encode_from_pre(pre), L=3, beta=100.0, dense=pre)
            sc = np.sort(np.asarray(a))[::-1]
            margin = float(sc[0] - sc[1])
            hit = int(int(np.argmax(a)) == obs_of.get(i, -1))
            margins.append(margin); hits.append(hit)
            per.append(dict(pid=pid, src=i, margin=margin, hit=hit))
        auc, p = perm_auc(margins, hits)
        rows['deg%.1f' % deg] = dict(n=len(probes), n_hit=int(sum(hits)),
                                     auc=round(auc, 4) if auc else None, p=round(p, 5),
                                     per_probe=per)
        print('deg%.1f AUC(margin->hit)=%.4f p=%.4f n_hit=%d/%d'
              % (deg, auc if auc else -1, p, sum(hits), len(probes)), flush=True)
    best = max((r['auc'] or 0) for r in rows.values())
    verdict = 'USABLE_SIGNAL' if any(r['auc'] and r['auc'] > 0.5 and r['p'] < 0.05 for r in rows.values()) else 'FAIL_DEEPENS'
    out = dict(exp='exp51', prereg='ledger exp51 row (e41dc15)',
               config=dict(n_rows=len(texts), n_written=nw, probes=100,
                           encoder='NEW256', beta=100.0, judge='obs_of',
                           signal='margin = top1 - top2 of episodic score'),
               reference_exp40_learned_auc=[0.335, 0.063],
               rows=rows, verdict=verdict,
               note=('margin is a usable abstention signal on public corpus'
                     if verdict == 'USABLE_SIGNAL' else
                     'margin also unusable on homogeneous closed corpus; G-P3 FAIL deepens'),
               finished_at=time.strftime('%Y-%m-%dT%H:%M:%S'))
    with open(os.path.join(ROOT, 'results', 'exp51_margin_abstain.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('VERDICT=%s best_auc=%.4f elapsed=%.0fs' % (verdict, best, time.time() - t0), flush=True)

if __name__ == '__main__':
    main()
