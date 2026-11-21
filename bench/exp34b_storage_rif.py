#!/usr/bin/env python3
'''exp34b_storage_rif.py -- storage-level RIF (M8 dissociation experiment, fourth round).

Pre-registered gates (written before any data, mirroring exp34_analyze.py discipline):
  CO-PRIMARY (both pre-registered before any full-scale data; smoke only):
  G-A' below-baseline storage RIF: son015 deficit = hit_nrp - hit_rp_sup >= 0.10,
       seed-level paired, bootstrap 95% CI (10000, seed=7) excludes 0;
       placebo: soff design-faithful deficit (hit_nrp - hit_rp_cos) < 0.05.
  G-S1 practiced-item collapse: hit_p(son030) < hit_p(soff) - 0.10, paired CI excl 0.
       (Smoke N=60 showed amplitude attenuation creates a hub slot that captures the
       attractor globally; at N=2000 depth ~0.63 events/slot the outcome is open.)
  G-S2 hub formation: hub_share(son030) >= max(0.15, 2x hub_share(soff)).
  G-B' exact-inverse restores suppressed-slot hits: |hit_recovered - hit_p| <= 0.05.
  G-B'' inverse-of-collapse: after restoring ALL attenuated columns, hit_p returns
       to within 0.05 of same-seed soff hit_p (pure amplitude mechanism check).
  G-C' honest negative if G-A' fails.
  Dose-response (exploratory): deficit and hit_p-collapse monotone in rho.
  Analytic note (no arm needed): column renormalization after attenuation restores the
  original column exactly -> suppression invisible; direction change = deletion.
  Trilemma: readout-gain (local) vs amplitude (global) vs deletion.

Design: david cell only (exp34 showed syn cells at ceiling/floor, no dynamic range;
pre-registered choice). Library N=2000 peer-01 S256 per-seed row sample. Practice 100
probes x k=3 retrievals, do_rif=False in ALL arms; storage arms apply column
attenuation M[:,i] *= (1-rho) to top-5 competitors of the settled attractor
(selection mirrors E.suppress: order[0]=winner, others suppressed, use[winner]+=1).
Test do_rif=False, judge L-035 obs_of (hit = action in {recall,separate} AND idx==s).
att[s] tracks cumulative attenuation per slot (mechanism-exact inverse for G-B').
Arms: soff / son015 / son030 / son050 x 12 seeds = 48 runs. Shard EXP34B_SHARD/NSHARD.
'''
import sys, os, json, csv, re, time, zlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
from hcm import HCM, Config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = "--smoke" in sys.argv
SHARD = int(os.environ.get("EXP34B_SHARD", "0"))
NSHARD = int(os.environ.get("EXP34B_NSHARD", "1"))
_t0 = time.time()

_NANO = re.compile(r"\[[^\]]{1,12}\]")
def _clean(s):
    s = _NANO.sub(" ", str(s))
    return re.sub(r"\s+", " ", s).strip()

rows_all = list(csv.reader(open(os.path.join(ROOT, "bench", "corpus", "train__peer-02.csv"), encoding="utf-8-sig")))
rows_all = [r for r in rows_all[1:] if len(r) >= 3]
msgs_all = [_clean(r[2]) for r in rows_all]

z = np.load(os.path.join(ROOT, "results", "exp23_semantic_embed.npz"), allow_pickle=True)
E0 = z["E"]; chars = [str(c) for c in z["chars"]]
V = chars + ([] if "\ufffd" in chars else ["\ufffd"])
Vi = {ch: i for i, ch in enumerate(V)}
Memb = E0.T.copy()
if "\ufffd" not in chars:
    r0 = np.random.default_rng(99)
    Memb = np.hstack([Memb, r0.standard_normal((Memb.shape[0], 1))])
    Memb[:, -1] /= np.linalg.norm(Memb[:, -1])
D_x = Memb.shape[0]
def x_of(s):
    idx = [Vi.get(c, Vi["\ufffd"]) for c in s] or [Vi["\ufffd"]]
    v = Memb[:, idx].sum(1)
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.zeros(D_x, dtype=np.float32)

N_LIB = 60 if SMOKE else 2000
N_P = 20 if SMOKE else 100
N_RP = 10 if SMOKE else 50
K_PRACTICE = 3
SEEDS = [0] if SMOKE else list(range(12))
RHOS = {"soff": 0.0, "son015": 0.15, "son030": 0.30, "son050": 0.50}

def run_once(arm, seed):
    rng = np.random.default_rng([seed, zlib.crc32(b"david")])  # paired: identical lib/P across arms
    cfg = Config(d_x=D_x, d_c=D_x, a=0.02, capacity=N_LIB + 10, beta=100.0, L=3, use_resid=False, seed=seed)
    m = HCM(cfg)
    pick = rng.choice(len(msgs_all), size=N_LIB, replace=False)
    texts = [msgs_all[int(i)] for i in pick]
    for s in texts:
        m.observe(x_of(s))
    obs_of = {int(m.E.t[j]): j for j in range(m.E.n)}
    slot_texts = {obs_of[i]: texts[i] for i in obs_of}
    written_rows = sorted(obs_of.keys())
    if len(written_rows) < N_P * 2:
        return None
    P_rows = list(rng.choice(written_rows, size=min(N_P, len(written_rows)), replace=False))
    P_slots = [obs_of[i] for i in P_rows]
    rho = RHOS[arm]
    att = np.ones(m.E.capacity, dtype=np.float64)
    n_events = 0
    probe_idx = []  # every test-phase retrieval's returned idx (-1 abstain), for hub stats
    for i in P_rows:
        x = x_of(slot_texts[obs_of[i]])
        for _ in range(K_PRACTICE):
            qq = m.query(x, do_rif=False, debug=True)
            if rho > 0.0 and "dbg" in qq and qq.get("action") in ("recall", "separate"):
                a = qq["dbg"]["a"]
                order = np.argsort(-a)[:5]
                if len(order):
                    winner = int(order[0])
                    m.E.use[winner] += 1
                    for j in order[1:]:
                        j = int(j)
                        if j != winner:
                            m.E.M[:, j] *= (1.0 - rho)
                            att[j] *= (1.0 - rho)
                            n_events += 1
    # Rp- mechanism-exact: attenuated slots (att < 0.999), exclude P
    attv = att[: m.E.n]
    suppressed = [s for s in range(m.E.n) if attv[s] < 0.999 and s not in P_slots]
    Rp_sup = list(rng.choice(suppressed, size=min(N_RP, len(suppressed)), replace=False)) if suppressed else []
    # Rp- design-faithful: top code-cosine quartile to practiced codes (direction unchanged by attenuation)
    Cn = m.E.M[:, : m.E.n].T
    Cn = Cn / (np.linalg.norm(Cn, axis=1, keepdims=True) + 1e-12)
    CP = Cn[P_slots]
    cos_to_P = (Cn @ CP.T).mean(axis=1)
    order2 = np.argsort(-cos_to_P)
    cand = [int(s) for s in order2 if int(s) not in P_slots]
    q = max(1, len(cand) // 4)
    Rp_cos = list(rng.choice(cand[:q], size=min(N_RP, q), replace=False)) if cand else []
    # Nrp: never-attenuated, write-time matched (10/90 pct of Rp_sup slots)
    if Rp_sup:
        t_lo, t_hi = np.percentile([s for s in Rp_sup], [10, 90])
        never = [s for s in range(m.E.n) if attv[s] >= 0.999 and s not in P_slots and t_lo <= s <= t_hi]
    else:
        never = [s for s in range(m.E.n) if attv[s] >= 0.999 and s not in P_slots]
    Nrp = list(rng.choice(never, size=min(N_RP, len(never)), replace=False)) if never else []
    def hit(slots):
        h = 0
        for s in slots:
            x = x_of(slot_texts[s])
            r_ = m.query(x, do_rif=False)
            probe_idx.append(int(r_.get("idx", -1)))
            h += int(r_.get("action") in ("recall", "separate") and r_.get("idx", -1) == s)
        return h / len(slots) if slots else float("nan")
    r = dict(arm=arm, seed=seed,
             hit_rp_sup=hit(Rp_sup), hit_rp_cos=hit(Rp_cos), hit_nrp=hit(Nrp),
             hit_p=hit(P_slots[:N_RP]),
             att_rp_sup=float(np.mean(attv[Rp_sup])) if Rp_sup else 1.0,
             att_rp_cos=float(np.mean(attv[Rp_cos])) if Rp_cos else 1.0,
             att_nrp=float(np.mean(attv[Nrp])) if Nrp else 1.0,
             n_attenuated=int(np.sum(attv < 0.999)), n_events=n_events,
             n_rp_sup=len(Rp_sup), n_rp_cos=len(Rp_cos), n_nrp=len(Nrp))
    # hub diagnostics: modal-slot share over all test probes
    from collections import Counter
    cnt = Counter(probe_idx)
    modal, modal_n = (cnt.most_common(1)[0] if cnt else (-1, 0))
    r["hub_share"] = round(modal_n / len(probe_idx), 4) if probe_idx else None
    r["modal_slot"] = int(modal)
    r["n_probes"] = len(probe_idx)
    # G-B' exact inverse restores amplitude
    if rho > 0.0 and Rp_sup:
        for s in Rp_sup:
            m.E.M[:, s] /= att[s]
            att[s] = 1.0
        r["hit_rp_sup_recovered"] = hit(Rp_sup)
        # G-B'': restore ALL attenuated columns -> does practiced-item hit come back?
        hit_p_before = r["hit_p"]
        for s2 in range(m.E.n):
            if att[s2] < 0.999:
                m.E.M[:, s2] /= att[s2]
                att[s2] = 1.0
        probe_idx.clear()
        r["hit_p_all_recovered"] = hit(P_slots[:N_RP])
        r["hit_p_before_fullrestore"] = hit_p_before
        cnt2 = Counter(probe_idx)
        r["hub_share_after_restore"] = round(cnt2.most_common(1)[0][1] / len(probe_idx), 4) if probe_idx else None
    return r

ARMS = list(RHOS.keys())
jobs = [(arm, sd) for arm in ARMS for sd in SEEDS]
mine = [jb for i, jb in enumerate(jobs) if i % NSHARD == SHARD]
print("shard %d/%d runs=%d arms=%s" % (SHARD, NSHARD, len(mine), ARMS), flush=True)
print("pairing: rng=seed+crc32(david) -> identical lib/P rows across arms", flush=True)
res_rows = []
for k, (arm, sd) in enumerate(mine):
    r = run_once(arm, sd)
    if r is not None:
        res_rows.append(r)
        print("[%d/%d] %s s%d hit_rp_sup=%.3f hit_nrp=%.3f n_att=%d %.0fs" % (
            k + 1, len(mine), arm, sd, r["hit_rp_sup"], r["hit_nrp"], r["n_attenuated"], time.time() - _t0), flush=True)
agg = {}
for arm in ARMS:
    rs = [r_ for r_ in res_rows if r_["arm"] == arm]
    if not rs:
        continue
    def mstat(f):
        v = np.array([r_[f] for r_ in rs], dtype=float)
        v = v[~np.isnan(v)]
        return dict(mean=round(float(v.mean()), 4), sd=round(float(v.std(ddof=1)), 4), n=int(len(v))) if len(v) > 1 else dict(mean=round(float(v[0]), 4), sd=None, n=int(len(v))) if len(v) else dict(mean=None, sd=None, n=0)
    d = [r_["hit_nrp"] - r_["hit_rp_sup"] for r_ in rs if not np.isnan(r_["hit_rp_sup"])]
    dc = [r_["hit_nrp"] - r_["hit_rp_cos"] for r_ in rs if not np.isnan(r_["hit_rp_cos"])]
    agg[arm] = dict(hit_rp_sup=mstat("hit_rp_sup"), hit_rp_cos=mstat("hit_rp_cos"), hit_nrp=mstat("hit_nrp"),
        hit_p=mstat("hit_p"), att_rp_sup=mstat("att_rp_sup"), att_nrp=mstat("att_nrp"),
        n_events=mstat("n_events"), n_attenuated=mstat("n_attenuated"),
        deficit_nrp_sup=dict(mean=round(float(np.mean(d)), 4), sd=round(float(np.std(d, ddof=1)), 4)) if d else dict(mean=None, sd=None),
        deficit_nrp_cos=dict(mean=round(float(np.mean(dc)), 4), sd=round(float(np.std(dc, ddof=1)), 4)) if dc else dict(mean=None, sd=None),
        recovered=[r_["hit_rp_sup_recovered"] for r_ in rs if "hit_rp_sup_recovered" in r_])
out = dict(config=dict(N_LIB=N_LIB, N_P=N_P, K=K_PRACTICE, seeds=len(SEEDS), smoke=SMOKE,
                       judge="obs_of_L-035", test_do_rif=False, rho={k: v for k, v in RHOS.items()},
                       cell="david", selection="mirror E.suppress top5"),
           arms=agg, rows=res_rows, elapsed_s=round(time.time() - _t0, 1))
_outf = os.environ.get("EXP34B_OUT", "exp34b_storage_rif.json")
json.dump(out, open(os.path.join(ROOT, "results", _outf), "w", encoding="utf-8"), indent=1)
print("EXP34B DONE %.0fs rows=%d" % (time.time() - _t0, len(res_rows)))
