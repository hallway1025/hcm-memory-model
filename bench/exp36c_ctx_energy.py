# -*- coding: utf-8 -*-
"""exp36c: context as attractor ENERGY bias (mechanism variant, pre-registered before test data).

Attribution (exp36b dev, done): the pre-attractor context bias added to ev (model.query)
never reaches the readout -- the L>0 branch overwrites the biased softmax a with the
content-pure Hopfield attractor a_hop from E.retrieve. Hence P1=0 at every kappa in BOTH
arms while P3/P4=1.0 (their content matches win the attractor anyway).

exp36c hypothesis: era-anchored retrieval exists IFF context participates in the attractor
ENERGY itself. Mechanism change (only when Config.ctx_in_energy=True, default False):
  episodic.retrieve(ebias=...) adds kappa_ctx * max(z(b_ctx), 0) to the logits of EVERY
  softmax inside the Hopfield iteration (init + L steps). Legacy path (False) is
  bit-identical: ebias=None.
With ctx_in_energy=True the ev-pre-bias is NOT applied (energy path only) -> clean
attribution vs exp36b (ev-only, inert).

Design identical to exp36b (same dev/test split, probes P1-P4, gates):
  Stage A (dev targets, seed 0): kappa grid {1,2,4,8,16,32};
    kappa*_arm = max kappa with P4_dev >= 0.85 AND P2_dev <= 0.20.
  Stage B (held-out test targets, seeds 0/1/2, each arm at its OWN kappa*):
    G-T: mean(P1 multi - P1 single) >= 0.15 AND >=2/3 seeds paired bootstrap 95%CI excludes 0
         AND all seed deltas > 0
    G-G: P3 multi >= P3 single - 0.02;  G-C: both arms P3 >= 0.85;  P2 both arms <= 0.20
  If an arm has no qualifying kappa it runs at 0.6 and calibration is reported FAIL.
Diagnostics: for the first 5 P1 probes per run, record ebias[target_slot], ebias.max()
(recomputed externally with the same formula) to prove the energy knob is live and its
magnitude is non-trivial.
"""
import sys, os, json, csv, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import exp36_b2_multitau as e36
from hcm.model import Config as _RealConfig

ROOT = e36.ROOT
KGRID = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
P4_MIN, P2_MAX = 0.85, 0.20

def set_kappa(k):
    def _f(**kw):
        kw['kappa_ctx'] = k
        kw['ctx_in_energy'] = True
        return _RealConfig(**kw)
    e36.Config = _f

def ebias_stats(m, cvec, kappa, slot):
    if m.E.Ctx.shape[0] == 0 or m.E.n == 0:
        return None
    b = m.E.Ctx[:, : m.E.n].T @ cvec
    b = (b - b.mean()) / (b.std() + 1e-8)
    e = kappa * np.maximum(b, 0.0)
    return dict(ebias_target=float(e[slot]) if 0 <= slot < len(e) else None,
                ebias_max=float(e.max()), ebias_mean=float(e.mean()))

def run_arm_dbg(seed, taus, msgs, senders, X, targets, kappa):
    cfg = e36.Config(d_x=X.shape[1], d_c=X.shape[1] * (len(taus) if taus else 1), a=0.02,
                     capacity=len(msgs) + 10, beta=100.0, L=3, use_resid=False, seed=seed,
                     use_ctx_bias=True, context_taus=taus)
    m = e36.HCM(cfg)
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
    dbg = []
    c_save = m.c.copy()
    for (i, j, knear) in kept:
        si = slot_of.get(i)
        if si is None or j not in slot_of:
            continue
        m.c = ctx_snap[i].copy()
        r1 = m.query(X[j])
        if len(dbg) < 5:
            st = ebias_stats(m, e36.unit(m.c), kappa, si)
            if st:
                st.update(probe='P1', i=i, slot=si)
                dbg.append(st)
        m.c = ctx_snap[j].copy()
        r2 = m.query(X[j])
        m.c = ctx_snap[i].copy()
        r3 = m.query(e36.degrade(X[i], 0.3, seed * 100000 + i))
        m.c = ctx_snap[knear].copy()
        r4 = m.query(e36.degrade(X[i], 0.3, seed * 100000 + i))
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
    out = dict(seed=seed, taus=taus, kappa=kappa, n_written=nw, n_probes=npr,
               wall=time.time() - t0,
               hits={p: res[p] for p in res},
               rates={p: (res[p] / npr if npr else None) for p in res},
               detail=detail, ebias_dbg=dbg)
    print('arm taus=%s seed=%d k=%g written=%d probes=%d P1=%d P2=%d P3=%d P4=%d (%.0fs)' %
          (taus, seed, kappa, nw, npr, res['P1'], res['P2'], res['P3'], res['P4'], out['wall']), flush=True)
    return out

def main():
    M, Vi = e36.load_emb()
    def x_of(s):
        idx = [Vi.get(c, Vi['\ufffd']) for c in e36._clean(s)] or [Vi['\ufffd']]
        v = M[:, idx].sum(1)
        n = np.linalg.norm(v)
        return v / n if n > 0 else np.zeros(M.shape[0], dtype=np.float32)
    path = os.path.join(ROOT, 'bench', 'corpus', 'train__%s.csv' % e36.SESS)
    rows = list(csv.reader(open(path, encoding='utf-8-sig')))
    rows = [r for r in rows[1:] if len(r) >= 3]
    msgs = [r[2] for r in rows]; senders = [r[1] for r in rows]
    X = np.stack([x_of(s) for s in msgs]).astype(np.float32)
    n = len(msgs)
    cand = list(range(e36.MIN_POS, n))
    targets = []; last = -10**9; sims_cache = {}
    for i in cand:
        if not np.any(np.abs(X[i]) > 1e-9):
            continue
        sims = X @ X[i]
        js = np.where(np.abs(np.arange(n) - i) > e36.ERA_GAP)[0]
        if len(js) == 0:
            continue
        jj = js[int(np.argmax(sims[js]))]
        if sims[jj] < e36.MIN_COS or i - last < e36.MIN_SEP:
            continue
        targets.append(i); last = i; sims_cache[i] = int(jj)
        if len(targets) >= e36.N_TARGET:
            break
    tgt = []
    for a, i in enumerate(targets):
        knear = min((x for x in targets if x != i), key=lambda x: abs(i - x), default=None)
        if knear is None:
            continue
        tgt.append((i, sims_cache[i], knear))
    dev = tgt[0::3]; test = [t for a, t in enumerate(tgt) if a % 3 != 0]
    print('targets %d -> dev %d / test %d' % (len(tgt), len(dev), len(test)), flush=True)
    out = dict(config=dict(kgrid=KGRID, p4_min=P4_MIN, p2_max=P2_MAX,
                           n_dev=len(dev), n_test=len(test), mechanism='ctx_in_energy',
                           preset=dict(GT_min_delta=0.15, GG_tol=0.02, GC_min=0.85)),
               dev={}, test={}, calib={}, gates={})
    # ---- Stage A: calibration on dev ----
    for key, taus in (('single', None), ('multi', [1, 10, 100, 1000])):
        curve = []
        for k in KGRID:
            set_kappa(k)
            r = run_arm_dbg(0, taus, msgs, senders, X, dev, k)
            curve.append(dict(kappa=k, P1=r['rates']['P1'], P2=r['rates']['P2'],
                              P3=r['rates']['P3'], P4=r['rates']['P4'], n=r['n_probes']))
            print('DEV %-6s k=%-5g P1=%.3f P2=%.3f P3=%.3f P4=%.3f' %
                  (key, k, r['rates']['P1'], r['rates']['P2'], r['rates']['P3'], r['rates']['P4']), flush=True)
        ok = [c['kappa'] for c in curve if c['P4'] >= P4_MIN and c['P2'] <= P2_MAX]
        ks = max(ok) if ok else None
        out['dev'][key] = curve
        out['calib'][key] = dict(kappa_star=ks, qualifying=ok, calibrated=ks is not None)
        print('CALIB %-6s kappa*=%s qualifying=%s' % (key, ks, ok), flush=True)
    # ---- Stage B: test at each arm's kappa* ----
    arms = {}
    for key, taus in (('single', None), ('multi', [1, 10, 100, 1000])):
        ks = out['calib'][key]['kappa_star'] or 0.6
        set_kappa(ks)
        arms[key] = [run_arm_dbg(s, taus, msgs, senders, X, test, ks) for s in (0, 1, 2)]
        out['test'][key] = [dict(seed=r['seed'], kappa=ks, rates=r['rates'], n=r['n_probes'],
                                 ebias_dbg=r['ebias_dbg'][:2]) for r in arms[key]]
    set_kappa(0.6)
    def by_i(det):
        return {d['i']: d for d in det}
    s1r = {r['seed']: r['rates']['P1'] for r in arms['single']}
    s2r = {r['seed']: r['rates']['P1'] for r in arms['multi']}
    deltas = {s: s2r[s] - s1r[s] for s in s1r}
    cis = {}
    for rm, rs in zip(arms['multi'], arms['single']):
        dm, ds = by_i(rm['detail']), by_i(rs['detail'])
        common = sorted(set(dm) & set(ds))
        d = [dm[i]['h1'] - ds[i]['h1'] for i in common]
        cis[rm['seed']] = e36.boot_ci(d)
    gt = (float(np.mean(list(deltas.values()))) >= 0.15
          and sum(1 for c in cis.values() if c and (c[0] > 0 or c[1] < 0)) >= 2
          and all(v > 0 for v in deltas.values()))
    p3m = {r['seed']: r['rates']['P3'] for r in arms['multi']}
    p3s = {r['seed']: r['rates']['P3'] for r in arms['single']}
    p2m = {r['seed']: r['rates']['P2'] for r in arms['multi']}
    p2s = {r['seed']: r['rates']['P2'] for r in arms['single']}
    ev = out['gates']
    ev['G_T_era_anchor'] = dict(deltas=deltas, cis=cis, mean_delta=float(np.mean(list(deltas.values()))),
                                verdict='PASS' if gt else 'FAIL')
    ev['G_G_ceiling_noharm'] = dict(multi=p3m, single=p3s,
                                    verdict='PASS' if all(p3m[s] >= p3s[s] - 0.02 for s in p3m) else 'FAIL')
    ev['G_C_ceiling_level'] = dict(multi=p3m, single=p3s,
                                   verdict='PASS' if all(v >= 0.85 for v in list(p3m.values()) + list(p3s.values())) else 'FAIL')
    ev['CTRL_P2_shuffle'] = dict(multi=p2m, single=p2s,
                                 verdict='PASS' if all(v <= 0.20 for v in list(p2m.values()) + list(p2s.values())) else 'FAIL')
    out['calib']['summary'] = {k: out['calib'][k]['kappa_star'] for k in ('single', 'multi')}
    with open(os.environ.get('EXP36C_OUT') or os.path.join(ROOT, 'results', 'exp36c_ctx_energy.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print('CALIB kappa*:', out['calib']['summary'], flush=True)
    print('GATES:', json.dumps({k: v['verdict'] for k, v in ev.items()}), flush=True)

if __name__ == '__main__':
    main()
