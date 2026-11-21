# -*- coding: utf-8 -*-
"""exp53 Part A - relinquishment test for the wlink chain cure (audit #5, retrieval-side form).

Prereg: ledger exp53 Part A row (2026-11-13, before any data). exp50 rig reproduced
identically (DailyDialog 8k, NEW256 encoder, 200 pairs rng(2700)); PRE-DEL baseline
{w0,w1.0}x{deg0,0.3}; delete the 200 cue columns (np.delete + obs remap); POST-DEL
w0 x {deg0,0.3}. Headline = paired next@5(post_del w0) - next@5(pre_del w1.0): how
much of the chain cure dies with its source rows. Sanity gate: pre_del w1.0 deg0
>= 130/200 else abort (pipeline bug).
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

def delete_slots(E, del_cols):
    """in-place column removal; returns remap {old_col: new_col} for survivors."""
    keep = np.setdiff1d(np.arange(E.n), np.asarray(sorted(set(int(c) for c in del_cols))))
    E.M = np.ascontiguousarray(E.M[:, keep])
    E.D = np.ascontiguousarray(E.D[:, keep])
    E.t = E.t[keep]; E.g = E.g[keep]; E.r = E.r[keep]; E.nu = E.nu[keep]
    E.use = E.use[keep]; E.sup = E.sup[keep]; E.cid = E.cid[keep]
    if E.d_p:
        E.P = np.ascontiguousarray(E.P[:, keep])
    E.n = int(len(keep))
    E._cid2slot = {int(c): i for i, c in enumerate(E.cid[:E.n])}
    return {int(old): int(new) for new, old in enumerate(keep)}

def run_probes(m, enc, texts, pairs, obs_of, Wl, deg):
    h5 = []; self1 = next1 = next5 = 0
    for pid, i in enumerate(pairs):
        j = i + 1
        qstr = _clean(texts[i]) if deg == 0 else degrade_text(texts[i], deg, 41000 + pid)
        q = enc.encode(qstr)
        pre = m.enc.raw(q)
        h = m.enc.encode_from_pre(pre)
        _, a1 = m.E.retrieve(h, L=3, beta=100.0, dense=pre)
        s2 = m.E.scores(h) + (0.0 if Wl is None else 1.0 * (Wl @ a1))
        a2 = softmaxb(s2, 100.0)
        k1 = int(np.argmax(a1)); k2 = int(np.argmax(a2))
        order = np.argsort(-a2)
        tgt = obs_of[j]
        rj = int(np.where(order == tgt)[0][0])
        hit5 = int(rj < 5)
        if k1 == obs_of.get(i, -1): self1 += 1  # post-del: cue col deleted -> structurally 0
        if k2 == tgt: next1 += 1
        next5 += hit5; h5.append(hit5)
    return dict(pairs=len(pairs), self_hit1=self1, next_hit1=next1, next_hit5=next5,
                next_rate5=round(next5 / max(1, len(pairs)), 4), per_probe=h5)

def main():
    smoke = "--smoke" in sys.argv
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    t0 = time.time()
    n_rows = 800 if smoke else 8000
    n_pairs_cap = 40 if smoke else 200
    rows = load_train()[:n_rows]
    texts = [r[2] for r in rows]
    times = [r[0] for r in rows]
    senders = [r[1] for r in rows]
    print('rows=%d' % len(texts), flush=True)
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
        try: return datetime.fromisoformat(s.replace("/", "-"))
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
    # prereg amendment (smoke stage, before formal data): row-role-disjoint greedy
    # selection - no row may serve as cue in one selected pair and target in
    # another, else deleting cue columns destroys a sibling probe's target.
    sel = []; cue_rows = set(); tgt_rows = set()
    for i in pairs:
        if i in cue_rows or i in tgt_rows or (i + 1) in cue_rows or (i + 1) in tgt_rows:
            continue
        sel.append(i); cue_rows.add(i); tgt_rows.add(i + 1)
        if len(sel) >= n_pairs_cap: break
    pairs = sel
    print('probe pairs=%d' % len(pairs), flush=True)

    # ---------- PRE-DEL baselines ----------
    def build_wl(obs_map):
        W = np.zeros((m.E.n, m.E.n), dtype=np.float32)
        for i in pairs:
            W[obs_map[i + 1], obs_map[i]] += 1.0
        return W
    pre = {}
    for w_on, tag in ((False, "w0"), (True, "w1.0")):
        Wl = build_wl(obs_of) if w_on else None
        for deg in (0.0, 0.3):
            r = run_probes(m, enc, texts, pairs, obs_of, Wl, deg)
            pre["%s_deg%d" % (tag, deg * 10)] = r
            print('PRE  %s_deg%d next5=%d/%d self1=%d next1=%d' % (tag, deg * 10, r['next_hit5'], len(pairs), r['self_hit1'], r['next_hit1']), flush=True)
        del Wl
    sanity = pre["w1.0_deg0"]["next_hit5"]
    if sanity < (14 if smoke else 130):
        print('SANITY FAIL pre_del w1.0 deg0=%d < threshold -> pipeline bug, abort' % sanity, flush=True)
        sys.exit(5)

    # ---------- RELINQUISH: delete cue columns ----------
    cue_cols = [obs_of[i] for i in pairs]
    n_before = m.E.n
    remap = delete_slots(m.E, cue_cols)
    obs_new = {}
    for row_time, old_col in obs_of.items():
        if old_col in remap:
            obs_new[row_time] = remap[old_col]
    lost = [i for i in pairs if (i + 1) not in obs_new]
    print('deleted %d cue cols; store %d->%d; targets lost=%d' % (len(cue_cols), n_before, m.E.n, len(lost)), flush=True)
    if lost:
        print('FATAL: %d probe targets lost in deletion (unexpected)' % len(lost), flush=True)
        sys.exit(6)
    # links rebuilt on remapped columns: incoming source (cue col) is gone -> Wl row zero
    Wl_post = np.zeros((m.E.n, m.E.n), dtype=np.float32)
    for i in pairs:
        tc = obs_new[i + 1]
        pass  # incoming link source deleted; Wl@a1 == 0 identically (preregistered disclosure)
    post = {}
    for deg in (0.0, 0.3):
        r = run_probes(m, enc, texts, pairs, obs_new, Wl_post, deg)
        post["w0post_deg%d" % (deg * 10)] = r
        print('POST w0_deg%d next5=%d/%d self1=%d next1=%d' % (deg * 10, r['next_hit5'], len(pairs), r['self_hit1'], r['next_hit1']), flush=True)

    # ---------- verdicts ----------
    d_head = np.asarray(post["w0post_deg0"]["per_probe"], dtype=np.int8) - np.asarray(pre["w1.0_deg0"]["per_probe"], dtype=np.int8)
    p_head, obs_head = signflip_p(d_head)
    d_red = np.asarray(post["w0post_deg0"]["per_probe"], dtype=np.int8) - np.asarray(pre["w0_deg0"]["per_probe"], dtype=np.int8)
    p_red, obs_red = signflip_p(d_red)
    base_cure = pre["w1.0_deg0"]["next_rate5"]; post_r = post["w0post_deg0"]["next_rate5"]
    surv = round(1.0 - (post_r / base_cure) if base_cure > 0 else 0.0, 4)
    alive = (post_r / base_cure) if base_cure > 0 else 0.0
    verdict = "CURE_PARTLY_STORE_REDUNDANT" if (alive > 0.5 and p_head < 0.05) else "CURE_DIES_WITH_SOURCE_C4_FAIL"
    out = dict(exp="exp53", part="A", prereg="ledger exp53 Part A row (2026-11-13, pre-data)",
               smoke=smoke,
               config=dict(n_rows=len(texts), n_written=nw, pairs=len(pairs),
                           encoder="NEW256 (CharacterSemanticEncoder)", beta=100.0,
                           judge="obs_of", deleted_cols=len(cue_cols), store_after=m.E.n),
               pre=pre, post=post,
               headline=dict(delta=round(obs_head, 4), p=round(p_head, 5),
                             interpretation="post_del_w0 minus pre_del_w1.0 (cure source-dependence)"),
               redundancy=dict(delta=round(obs_red, 4), p=round(p_red, 5)),
               survival_rate=round(alive, 4), lost_rate=round(surv, 4),
               verdict=verdict,
               disclosure="post-del Wl@a1==0 identically (incoming link source deleted); strict audit form (targets also deleted) is an adaptation gap",
               finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    fn = 'exp53_smoke.json' if smoke else 'exp53_relinquish.json'
    with open(os.path.join(ROOT, 'results', fn), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('GATE %s survival=%.3f headline_delta=%.3f p=%.4f elapsed=%.0fs' % (verdict, alive, obs_head, p_head, time.time() - t0), flush=True)

if __name__ == '__main__':
    main()