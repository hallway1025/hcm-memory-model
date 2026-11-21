# -*- coding: utf-8 -*-
"""exp50 - wlink chain cure on public corpus (DailyDialog), preregistered ledger row exp50.

Private-corpus chain cure (exp28): w_chain>=1.0 lifts next@5 from 1.5% to full marks.
exp40-GP2 replicated the chain NULL on DailyDialog (both arms next@5<=0.05) but did
not run the cure arm. exp50 closes that gap.

Protocol: DailyDialog train first 8000 rows (synthetic 2s turn gaps satisfy the
gap<=30s rule; A/B senders). NEW256 encoder (CharacterSemanticEncoder, fit on the
8000 texts). Chain pairs = adjacent turns, both written, len>=4, sender differs,
gap<=30s, rng(2700) shuffle, first 200. Two-phase retrieval (exp28 mechanism):
a1 = episodic readout; a2 = softmax(beta*(scores + w_chain*Wl@a1)), Wl[j,i]+=1 per
pair (incoming links). Arms w in {0, 1.0, 3.0}; deg in {0, 0.3} (text-level char
deletion, seed 41000+pid). beta=100 (D-035).

PREREGISTERED GATE: Delta_next5(w1.0 - w0) >= 0.20 AND paired sign-flip p<0.05
-> cure generalizes; otherwise "chaining benefit is corpus-specific" recorded
honestly. K3 wording unchanged in both branches.
"""
import csv, json, os, re, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hcm import HCM, Config
from hcm.text_encoder import CharacterSemanticEncoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NANO = re.compile(r'\[[^\]]{1,8}\]')
def _clean(s): return NANO.sub('', re.sub(r'\s+', ' ', s)).strip()

def load_train():
    p = os.path.join(ROOT, 'bench', 'corpus_public', 'train__dailydialog.csv')
    with open(p, encoding='utf-8-sig', newline='') as f:
        rd = csv.reader(f); next(rd)
        return [(r[0], r[1], r[2]) for r in rd]

def degrade_text(text, frac, seed):
    r = np.random.default_rng(seed)
    chs = list(_clean(text))
    if len(chs) > 1:
        for j in r.choice(len(chs), size=max(1, int(len(chs) * frac)), replace=False):
            chs[j] = ''
    return ''.join(c for c in chs if c)

def softmaxb(v, beta):
    v = v - v.max()
    w = np.exp(beta * v)
    return w / w.sum()

def signflip_p(diff, n_perm=200000, seed=50):
    d = np.asarray(diff, dtype=np.float64)
    if len(d) == 0 or np.all(d == 0):
        return 1.0, float(np.mean(d)) if len(d) else 0.0
    obs = float(np.mean(d))
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n_perm, len(d)))
    perm = (signs * d).mean(axis=1)
    return float((np.abs(perm) >= abs(obs) - 1e-12).mean()), obs

def main():
    try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass
    t0 = time.time()
    rows = load_train()[:8000]
    texts = [r[2] for r in rows]
    times = [r[0] for r in rows]
    senders = [r[1] for r in rows]
    print('train rows=%d' % len(texts), flush=True)
    enc = CharacterSemanticEncoder(dim=256, window=3, seed=0)
    enc.fit(texts)
    Xs = np.asarray([enc.encode(t) for t in texts], dtype=np.float32)
    cfg = Config(d_x=256, d_c=256, a=0.02, capacity=len(texts) + 10, beta=100.0, L=3,
                 use_resid=False, seed=0)
    m = HCM(cfg)
    tw = time.time()
    for i in range(len(texts)):
        m.observe(Xs[i])
    nw = int(m.E.n)
    obs_of = {int(m.E.t[j]): j for j in range(nw)}
    print('written %d/%d %.0fs' % (nw, len(texts), time.time() - tw), flush=True)

    from datetime import datetime
    def tparse(s):
        try: return datetime.fromisoformat(s.replace('/', '-'))
        except Exception: return None
    pairs = []
    for i in range(len(texts) - 1):
        a, b = _clean(texts[i]), _clean(texts[i + 1])
        if len(a) < 4 or len(b) < 4 or senders[i] == senders[i + 1]: continue
        ta, tb = tparse(times[i]), tparse(times[i + 1])
        if ta is None or tb is None or (tb - ta).total_seconds() > 30: continue
        if i not in obs_of or (i + 1) not in obs_of: continue
        pairs.append(i)
    rngp = np.random.default_rng(2700)
    rngp.shuffle(pairs)
    pairs = pairs[:200]
    print('probe pairs=%d' % len(pairs), flush=True)

    arms_out = {}
    next5_hits = {}
    for w_chain in (0.0, 1.0, 3.0):
        Wl = np.zeros((nw, nw), dtype=np.float32)
        if w_chain > 0:
            for i in pairs:
                Wl[obs_of[i + 1], obs_of[i]] += 1.0
        for deg in (0.0, 0.3):
            self1 = next1 = next5 = joint = 0
            h5 = []
            for pid, i in enumerate(pairs):
                j = i + 1
                qstr = _clean(texts[i]) if deg == 0 else degrade_text(texts[i], deg, 41000 + pid)
                q = enc.encode(qstr)
                pre = m.enc.raw(q)
                h = m.enc.encode_from_pre(pre)
                _, a1 = m.E.retrieve(h, L=3, beta=100.0, dense=pre)
                s2 = m.E.scores(h) + w_chain * (Wl @ a1)
                a2 = softmaxb(s2, 100.0)
                k1 = int(np.argmax(a1)); k2 = int(np.argmax(a2))
                order = np.argsort(-a2)
                tgt = obs_of[j]
                rj = int(np.where(order == tgt)[0][0])
                hit5 = int(rj < 5)
                if k1 == obs_of[i]: self1 += 1
                if k2 == tgt: next1 += 1
                next5 += hit5
                if k1 == obs_of[i] and k2 == tgt: joint += 1
                h5.append(hit5)
            key = 'w%g_deg%g' % (w_chain, deg * 100)
            arms_out[key] = dict(pairs=len(pairs), self_hit1=self1, next_hit1=next1,
                                 next_hit5=next5,
                                 next_rate5=round(next5 / max(1, len(pairs)), 4),
                                 joint=joint)
            next5_hits[key] = h5
            print('%s self1=%d next1=%d next5=%d/%d' % (key, self1, next1, next5, len(pairs)),
                  flush=True)
        del Wl

    base = next5_hits['w0_deg0']
    d1 = np.asarray(next5_hits['w1_deg0'], dtype=np.int8) - np.asarray(base, dtype=np.int8)
    p1, obs1 = signflip_p(d1)
    d3 = np.asarray(next5_hits['w3_deg0'], dtype=np.int8) - np.asarray(base, dtype=np.int8)
    p3, obs3 = signflip_p(d3)
    r5 = arms_out['w0_deg0']['next_rate5']
    g1 = arms_out['w1_deg0']['next_rate5']
    verdict = 'CURE_GENERALIZES' if (g1 - r5 >= 0.20 and p1 < 0.05) else 'CORPUS_SPECIFIC'
    out = dict(exp='exp50', prereg='ledger exp50 row (e41dc15)',
               config=dict(n_rows=len(texts), n_written=nw, pairs=len(pairs),
                           encoder='NEW256 (CharacterSemanticEncoder)', beta=100.0,
                           judge='obs_of', mechanism='a2=softmax(beta*(scores+w*Wl@a1))'),
               rows=arms_out, next5_hits=next5_hits,
               gate=dict(delta_w1=round(g1 - r5, 4), p_w1=round(p1, 5),
                         delta_w3=round(arms_out['w3_deg0']['next_rate5'] - r5, 4),
                         p_w3=round(p3, 5)),
               verdict=verdict,
               note=('chaining benefit corpus-specific; K3 wording unchanged'
                     if verdict == 'CORPUS_SPECIFIC' else ''),
               finished_at=time.strftime('%Y-%m-%dT%H:%M:%S'))
    with open(os.path.join(ROOT, 'results', 'exp50_wlink_public.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('GATE w1.0: delta=%.3f p=%.4f verdict=%s elapsed=%.0fs'
          % (g1 - r5, p1, verdict, time.time() - t0), flush=True)

if __name__ == '__main__':
    main()
