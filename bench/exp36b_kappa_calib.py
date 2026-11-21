# -*- coding: utf-8 -*-
"""exp36b: kappa calibration sweep for era-anchor retrieval (pre-registered before test data).

Context: exp36 full run (default kappa_ctx=0.6) gave P1=0/135 in BOTH arms -> content-exact
competitor j dominates context bias. This is a knob-calibration question, so we run a
TWO-STAGE design with dev/test probe split (stratified: every 3rd target to dev).

Stage A (calibration, dev targets, seed 0):
  kappa grid {1,2,4,8,16,32}; kappa*_arm = max kappa with P4_dev >= 0.85 AND P2_dev <= 0.20.
  Rationale: P4 (era-swap) forces content to still win under wrong-era context -> prevents
  the degenerate kappa->infinity solution (pure-context lookup). P2 is the shuffle control.
Stage B (test, held-out targets, seeds 0/1/2, each arm at its OWN kappa*):
  G-T: mean(P1 multi - P1 single) >= 0.15 AND >=2/3 seeds paired bootstrap 95%CI excludes 0
       AND all seed deltas > 0
  G-G: P3 multi >= P3 single - 0.02;  G-C: both arms P3 >= 0.85;  P2 both arms <= 0.20
If an arm has no qualifying kappa, it runs at default 0.6 and calibration is reported FAIL.
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
        kw.setdefault('use_resid', False)  # D-052 显式钉死（拍板④ 2026-11-09）：父模块 exp36_b2_multitau.py:66 已显式传 False，此处兜底防默认再变
        return _RealConfig(**kw)
    e36.Config = _f

def main():
    M, Vi = e36.load_emb()
    def x_of(s):
        idx = [Vi.get(c, Vi['\ufffd']) for c in e36._clean(s)] or [Vi['\ufffd']]
        v = M[:, idx].sum(1)
        n = np.linalg.norm(v)
        return v / n if n > 0 else np.zeros(M.shape[0], dtype=np.float32)
    path = os.path.join(ROOT, 'bench', 'corpus', 'train__%s.csv' % e36.SESS)
    if not os.path.exists(path):
        # 真实聊天语料因隐私不随公开仓库分发。此步在无语料时跳过而非失败，
        # 以便公开克隆能跑完整个复现链；论文中该实验的结果数字来自有语料的原始运行。
        print('SKIP: 缺少真实语料 %s（公开仓库不含，属预期）' % os.path.relpath(path, ROOT), flush=True)
        print('      κ 标定实验需要自备同结构语料放入 bench/corpus/ 后重跑。', flush=True)
        return 0
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
                           n_dev=len(dev), n_test=len(test),
                           preset=dict(GT_min_delta=0.15, GG_tol=0.02, GC_min=0.85)),
               dev={}, test={}, calib={})
    # ---- Stage A: calibration on dev ----
    for key, taus in (('single', None), ('multi', [1, 10, 100, 1000])):
        curve = []
        for k in KGRID:
            set_kappa(k)
            r = e36.run_arm(0, taus, msgs, senders, X, dev, None)
            curve.append(dict(kappa=k, P1=r['rates']['P1'], P2=r['rates']['P2'],
                              P3=r['rates']['P3'], P4=r['rates']['P4'], n=r['n_probes']))
            print('DEV %-6s k=%-5g P1=%.3f P2=%.3f P3=%.3f P4=%.3f' %
                  (key, k, r['rates']['P1'], r['rates']['P2'], r['rates']['P3'], r['rates']['P4']), flush=True)
        ok = [c['kappa'] for c in curve if c['P4'] >= P4_MIN and c['P2'] <= P2_MAX]
        ks = max(ok) if ok else None
        out['dev'][key] = curve
        out['calib'][key] = dict(kappa_star=ks, qualifying=ok,
                                 calibrated=ks is not None)
        print('CALIB %-6s kappa*=%s qualifying=%s' % (key, ks, ok), flush=True)
    # ---- Stage B: test at each arm's kappa* ----
    ev = {}
    arms = {}
    for key, taus in (('single', None), ('multi', [1, 10, 100, 1000])):
        ks = out['calib'][key]['kappa_star'] or 0.6
        set_kappa(ks)
        arms[key] = [e36.run_arm(s, taus, msgs, senders, X, test, None) for s in (0, 1, 2)]
        out['test'][key] = [dict(seed=r['seed'], kappa=ks, rates=r['rates'], n=r['n_probes']) for r in arms[key]]
    set_kappa(0.6)  # restore for any future use
    # align details by probe index i
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
    ev['G_T_era_anchor'] = dict(deltas=deltas, cis=cis, mean_delta=float(np.mean(list(deltas.values()))),
                                verdict='PASS' if gt else 'FAIL')
    ev['G_G_ceiling_noharm'] = dict(multi=p3m, single=p3s,
                                    verdict='PASS' if all(p3m[s] >= p3s[s] - 0.02 for s in p3m) else 'FAIL')
    ev['G_C_ceiling_level'] = dict(multi=p3m, single=p3s,
                                   verdict='PASS' if all(v >= 0.85 for v in list(p3m.values()) + list(p3s.values())) else 'FAIL')
    ev['CTRL_P2_shuffle'] = dict(multi=p2m, single=p2s,
                                 verdict='PASS' if all(v <= 0.20 for v in list(p2m.values()) + list(p2s.values())) else 'FAIL')
    out['gates'] = ev
    out['calib']['summary'] = {k: out['calib'][k]['kappa_star'] for k in ('single', 'multi')}
    with open(os.environ.get('EXP36B_OUT') or os.path.join(ROOT, 'results', 'exp36b_kappa_calib.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print('CALIB kappa*:', out['calib']['summary'], flush=True)
    print('GATES:', json.dumps({k: v['verdict'] for k, v in ev.items()}), flush=True)

if __name__ == '__main__':
    main()
