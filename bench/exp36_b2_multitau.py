# -*- coding: utf-8 -*-
"""exp36 B2-L1: multi-timescale context (pre-registered before full data).

Arms: single (gamma_c=0.95, d_c=d_x) vs multi (context_taus=[1,10,100,1000], d_c=4*d_x).
Content pathway identical: same W_e seed/shape (d_h,d_x), same npz S256 embedding, same stream.

Probes (context injected via m.c, restored after each query):
  P1 era-anchor   : content x_j (cross-era top content neighbor, cos>=0.5, |pos_i-pos_j|>2000) + ctx C_i -> hit slot_i
  P2 shuffle-ctrl : same x_j + ctx C_j (the far era of j itself) -> hit slot_i (expect low both arms)
  P3 ceiling      : content x_i deg0.3 + own ctx C_i -> hit slot_i
  P4 era-swap     : content x_i deg0.3 + ctx C_near (nearest other target) -> hit slot_i

Gates (pre-registered):
  G-T: mean(P1 multi - P1 single) >= 0.15, >=2/3 seeds bootstrap 95% CI excludes 0, all seed deltas > 0
  G-G: P3 multi >= P3 single - 0.02
  G-C: P3 both arms >= 0.85
  P2 both arms <= 0.20
"""
import sys, os, json, csv, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from hcm.model import Config, HCM
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESS = 'peer-01'
NSEED = 3
N_TARGET = 200
MIN_POS = 3000
MIN_SEP = 25
ERA_GAP = 2000
MIN_COS = 0.5

def _clean(s):
    return ''.join(ch for ch in str(s) if ch.isprintable())

def load_emb():
    z = np.load(os.path.join(ROOT, 'results', 'exp23_semantic_embed.npz'), allow_pickle=True)
    E = z['E']; chars = [str(c) for c in z['chars']]
    V = chars + ([] if '\ufffd' in chars else ['\ufffd'])
    Vi = {ch: i for i, ch in enumerate(V)}
    M = E.T.copy()
    if '\ufffd' not in chars:
        r0 = np.random.default_rng(99)
        M = np.hstack([M, r0.standard_normal((M.shape[0], 1))])
        M[:, -1] /= np.linalg.norm(M[:, -1])
    return M, Vi

def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def tparse(s):
    try: return datetime.fromisoformat(str(s).replace('/', '-'))
    except Exception: return None

def degrade(x, frac, seed):
    r = np.random.default_rng(seed)
    y = x.copy()
    nz = np.where(np.abs(y) > 1e-6)[0]
    y[r.choice(nz, size=max(1, int(len(nz) * frac)), replace=False)] = 0.0
    return unit(y)

def run_arm(seed, taus, msgs, senders, X, targets, probe_ctx):
    cfg = Config(d_x=X.shape[1], d_c=X.shape[1] * (len(taus) if taus else 1), a=0.02,
                 capacity=len(msgs) + 10, beta=100.0, L=3, use_resid=False, seed=seed,
                 use_ctx_bias=True, context_taus=taus)
    m = HCM(cfg)
    slot_of = {}
    ctx_snap = {}
    t0 = time.time()
    for i, s in enumerate(msgs):
        n0 = m.E.n
        rec = m.observe(X[i], sender=senders[i])
        if rec.get('wrote') and rec.get('idx', -1) is not None and rec['idx'] >= 0:
            slot_of[i] = n0
        ctx_snap[i] = m.c.copy()
    nw = len(slot_of)
    kept = [i for i, _ in targets.items()] if isinstance(targets, dict) else list(targets)
    res = {p: 0 for p in ('P1', 'P2', 'P3', 'P4')}
    detail = []
    c_save = m.c.copy()
    for (i, j, knear) in kept:
        si = slot_of.get(i)
        if si is None or j not in slot_of:
            continue
        m.c = ctx_snap[i].copy()
        r1 = m.query(X[j])
        m.c = ctx_snap[j].copy()
        r2 = m.query(X[j])
        m.c = ctx_snap[i].copy()
        r3 = m.query(degrade(X[i], 0.3, seed * 100000 + i))
        m.c = ctx_snap[knear].copy()
        r4 = m.query(degrade(X[i], 0.3, seed * 100000 + i))
        m.c = c_save.copy()
        h1 = int(r1.get('idx', -1) == si)
        h2 = int(r2.get('idx', -1) == si)
        h3 = int(r3.get('idx', -1) == si)
        h4 = int(r4.get('idx', -1) == si)
        res['P1'] += h1; res['P2'] += h2; res['P3'] += h3; res['P4'] += h4
        detail.append(dict(i=i, j=j, knear=knear, slot=si,
                           h1=h1, h2=h2, h3=h3, h4=h4,
                           idx1=r1.get('idx', -1), idx3=r3.get('idx', -1)))
    npr = len(detail)
    out = dict(seed=seed, taus=taus, n_written=nw, n_probes=npr, wall=time.time() - t0,
               hits={p: res[p] for p in res},
               rates={p: (res[p] / npr if npr else None) for p in res}, detail=detail)
    print('arm taus=%s seed=%d written=%d probes=%d P1=%d P2=%d P3=%d P4=%d (%.0fs)' %
          (taus, seed, nw, npr, res['P1'], res['P2'], res['P3'], res['P4'], out['wall']), flush=True)
    return out

def boot_ci(diffs, iters=10000, seed=7):
    d = np.asarray(diffs, dtype=np.float64)
    if len(d) == 0:
        return None
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(d), size=(iters, len(d)))
    means = d[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]

def main():
    M, Vi = load_emb()
    def x_of(s):
        idx = [Vi.get(c, Vi['\ufffd']) for c in _clean(s)] or [Vi['\ufffd']]
        v = M[:, idx].sum(1)
        n = np.linalg.norm(v)
        return v / n if n > 0 else np.zeros(M.shape[0], dtype=np.float32)
    path = os.path.join(ROOT, 'bench', 'corpus', 'train__%s.csv' % SESS)
    rows = list(csv.reader(open(path, encoding='utf-8-sig')))
    rows = [r for r in rows[1:] if len(r) >= 3]
    msgs = [r[2] for r in rows]
    senders = [r[1] for r in rows]
    times = [r[0] for r in rows]
    print('stream %s: %d msgs' % (SESS, len(msgs)), flush=True)
    X = np.stack([x_of(s) for s in msgs]).astype(np.float32)
    # probe targets: shared across arms/seeds (content-only selection)
    n = len(msgs)
    written_hint = None  # targets chosen from candidate pool; slot checked at runtime
    cand = [i for i in range(MIN_POS, n)]
    targets = []
    last = -10**9
    sims_cache = {}
    for i in cand:
        if not np.any(np.abs(X[i]) > 1e-9):
            continue
        sims = X @ X[i]
        mask = np.array([abs(i - j2) > ERA_GAP for j2 in range(n)])
        js = np.where(mask)[0]
        if len(js) == 0:
            continue
        jj = js[int(np.argmax(sims[js]))]
        if sims[jj] < MIN_COS:
            continue
        if i - last < MIN_SEP:
            continue
        targets.append(i)
        last = i
        sims_cache[i] = int(jj)
        if len(targets) >= N_TARGET:
            break
    tgt = []
    for a, i in enumerate(targets):
        j = sims_cache[i]
        knear = None
        bestd = 10**9
        for b, i2 in enumerate(targets):
            if i2 == i:
                continue
            d0 = abs(i - i2)
            if d0 < bestd:
                bestd = d0
                knear = i2
        if knear is None:
            continue
        tgt.append((i, j, knear))
    print('targets: %d (min_pos=%d min_cos=%.2f)' % (len(tgt), MIN_POS, MIN_COS), flush=True)
    out = dict(config=dict(session=SESS, n_msgs=n, n_targets=len(tgt), min_pos=MIN_POS,
                           min_sep=MIN_SEP, era_gap=ERA_GAP, min_cos=MIN_COS, nseed=NSEED),
               gates_preset=dict(GT_min_delta=0.15, GG_tol=0.02, GC_min=0.85, P2_max=0.20),
               arms={})
    for taus in (None, [1, 10, 100, 1000]):
        key = 'multi' if taus else 'single'
        out['arms'][key] = [run_arm(s, taus, msgs, senders, X, tgt, None) for s in range(NSEED)]
    # gate evaluation
    ev = {}
    for key in ('single', 'multi'):
        for r in out['arms'][key]:
            if not r['n_probes']:
                raise SystemExit('exp36 FATAL: arm %s seed %s got 0 probes - target selection broken' % (key, r['seed']))
    s1 = {r['seed']: r['rates']['P1'] for r in out['arms']['single']}
    s2 = {r['seed']: r['rates']['P1'] for r in out['arms']['multi']}
    deltas = {s: s2[s] - s1[s] for s in s1}
    cis = {}
    for r_m, r_s in zip(out['arms']['multi'], out['arms']['single']):
        d = [dm['h1'] - ds['h1'] for dm, ds in zip(r_m['detail'], r_s['detail'])]
        cis[r_m['seed']] = boot_ci(d)
    gt = (float(np.mean(list(deltas.values()))) >= 0.15
          and sum(1 for c0 in cis.values() if c0 and (c0[0] > 0 or c0[1] < 0)) >= 2
          and all(v > 0 for v in deltas.values()))
    p3m = {r['seed']: r['rates']['P3'] for r in out['arms']['multi']}
    p3s = {r['seed']: r['rates']['P3'] for r in out['arms']['single']}
    p2m = {r['seed']: r['rates']['P2'] for r in out['arms']['multi']}
    p2s = {r['seed']: r['rates']['P2'] for r in out['arms']['single']}
    ev['G_T_era_anchor'] = dict(deltas=deltas, cis=cis, mean_delta=float(np.mean(list(deltas.values()))), verdict='PASS' if gt else 'FAIL')
    ev['G_G_ceiling_noharm'] = dict(multi=p3m, single=p3s, verdict='PASS' if all(p3m[s] >= p3s[s] - 0.02 for s in p3m) else 'FAIL')
    ev['G_C_ceiling_level'] = dict(multi=p3m, single=p3s, verdict='PASS' if all(v >= 0.85 for v in list(p3m.values()) + list(p3s.values())) else 'FAIL')
    ev['CTRL_P2_shuffle'] = dict(multi=p2m, single=p2s, verdict='PASS' if all(v <= 0.20 for v in list(p2m.values()) + list(p2s.values())) else 'FAIL')
    out['gates'] = ev
    with open(os.path.join(ROOT, 'results', 'exp36_b2_multitau.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print('GATES:', json.dumps({k: v['verdict'] for k, v in ev.items()}), flush=True)

if __name__ == '__main__':
    main()
