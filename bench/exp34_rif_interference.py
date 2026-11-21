#!/usr/bin/env python3
'''exp34_rif_interference.py -- M8 RIF interference ablation (retrieval-practice paradigm).
Design: notes/exp34-design.md v0.1 (pre-registered H1-H4, gates G-A/B/C). Library N=2000
(peer-01 S256 per-seed row sample, or synthetic overlap share in {0.3,0.6,0.9}). Practice: 100
probes x k=3 clean retrievals with do_rif=arm. Rp- (suppressed competitors, sup<1, mechanism-
exact) and Rp-cos (top code-cosine quartile, design-faithful); Nrp = never-suppressed matched by
write time. Test with do_rif=False both arms. Judge L-035. Gates: G-A Rp- deficit >=10% with CI
non-overlap (on) and ~0 (off); G-B reversibility after k disinhibits; G-C honest negative path.
Shard via EXP34_SHARD/EXP34_NSHARD. Counts-only experiment.
'''
import sys, os, json, csv, re, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
from hcm import HCM, Config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = "--smoke" in sys.argv
SHARD = int(os.environ.get("EXP34_SHARD", "0"))
NSHARD = int(os.environ.get("EXP34_NSHARD", "1"))
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

def synth_lib(share, n, rng):
    # exp12 semantics: item = normalize(share*base + (1-share)*noise_i); report measured cos
    base = rng.standard_normal(D_x).astype(np.float32); base /= np.linalg.norm(base)
    items = []
    for i in range(n):
        noise = rng.standard_normal(D_x).astype(np.float32)
        v = share * base + (1 - share) * noise
        v /= np.linalg.norm(v)
        items.append(v)
    G = np.array(items) @ np.array(items).T
    iu = np.triu_indices(n, 1)
    return items, float(G[iu].mean())

def run_once(cell, arm, seed):
    import zlib
    rng = np.random.default_rng([seed, zlib.crc32(cell.encode())])
    cfg = Config(d_x=D_x, d_c=D_x, a=0.02, capacity=N_LIB + 10, beta=100.0, L=3, use_resid=False, seed=seed)
    m = HCM(cfg)
    if cell == "peer-02":
        pick = rng.choice(len(msgs_all), size=N_LIB, replace=False)
        texts = [msgs_all[int(i)] for i in pick]
    else:
        share = float(cell.split("_o")[1])
        items, meas_cos = synth_lib(share, N_LIB, rng)
        texts = ["syn%d" % i for i in range(N_LIB)]
        vecs = items
    if cell != "peer-02":
        for i, v in enumerate(vecs):
            m.observe(v)
        obs_of = {int(m.E.t[j]): j for j in range(m.E.n)}
        slot_texts = None
    else:
        for i, s in enumerate(texts):
            m.observe(x_of(s))
        obs_of = {int(m.E.t[j]): j for j in range(m.E.n)}
        slot_texts = {obs_of[i]: texts[i] for i in obs_of}
    # practice set P
    written_rows = sorted(obs_of.keys())
    if len(written_rows) < N_P * 2:
        return None
    P_rows = list(rng.choice(written_rows, size=min(N_P, len(written_rows)), replace=False))
    P_slots = [obs_of[i] for i in P_rows]
    RIF = (arm == "on")
    for i in P_rows:
        x = x_of(texts[i]) if cell == "peer-02" else vecs[i]
        for _ in range(K_PRACTICE):
            m.query(x, do_rif=RIF)
    # Rp- mechanism-exact: suppressed competitors (sup < 1), exclude P slots
    sup = m.E.sup[: m.E.n]
    suppressed = [s for s in range(m.E.n) if sup[s] < 0.999 and s not in P_slots]
    Rp_sup = list(rng.choice(suppressed, size=min(N_RP, len(suppressed)), replace=False)) if suppressed else []
    # Rp- design-faithful: top code-cosine quartile to practiced codes, exclude P
    Cn = m.E.M[:, : m.E.n].T
    Cn = Cn / (np.linalg.norm(Cn, axis=1, keepdims=True) + 1e-12)
    CP = Cn[P_slots]
    cos_to_P = (Cn @ CP.T).mean(axis=1)
    order = np.argsort(-cos_to_P)
    cand = [int(s) for s in order if int(s) not in P_slots]
    q = max(1, len(cand) // 4)
    Rp_cos = cand[:q]
    Rp_cos = list(rng.choice(Rp_cos, size=min(N_RP, len(Rp_cos)), replace=False)) if Rp_cos else []
    # Nrp: never-suppressed, matched by write-time quartile of Rp_sup
    if Rp_sup:
        t_lo, t_hi = np.percentile([s for s in Rp_sup], [10, 90])
        never = [s for s in range(m.E.n) if sup[s] >= 0.999 and s not in P_slots and t_lo <= s <= t_hi]
    else:
        never = [s for s in range(m.E.n) if sup[s] >= 0.999 and s not in P_slots]
    Nrp = list(rng.choice(never, size=min(N_RP, len(never)), replace=False)) if never else []
    def hit(slots):
        h = 0
        for s in slots:
            x = x_of(slot_texts[s]) if cell == "peer-02" else vecs[s]
            qq = m.query(x, do_rif=False)
            h += int(qq.get("action") in ("recall", "separate") and qq.get("idx", -1) == s)
        return h / len(slots) if slots else float("nan")
    r = dict(cell=cell, arm=arm, seed=seed,
             hit_rp_sup=hit(Rp_sup), hit_rp_cos=hit(Rp_cos), hit_nrp=hit(Nrp),
             hit_p=hit(P_slots[:N_RP]),
             sup_rp_sup=float(np.mean(sup[Rp_sup])) if Rp_sup else 1.0,
             sup_rp_cos=float(np.mean(sup[Rp_cos])) if Rp_cos else 1.0,
             sup_nrp=float(np.mean(sup[Nrp])) if Nrp else 1.0,
             n_suppressed=len(suppressed), n_rp_sup=len(Rp_sup), n_rp_cos=len(Rp_cos), n_nrp=len(Nrp))
    # G-B reversibility: k disinhibits on Rp_sup, re-test
    if RIF and Rp_sup:
        for s in Rp_sup:
            for _ in range(K_PRACTICE):
                m.E.disinhibit(s, rho_rif=0.15)
        r["hit_rp_sup_recovered"] = hit(Rp_sup)
    return r

CELLS = []
for arm in ("on", "off"):
    CELLS.append(("peer-02", arm))
    for sh in ("0.3", "0.6", "0.9"):
        CELLS.append(("syn_o" + sh, arm))
jobs = []
for cell, arm in CELLS:
    for sd in SEEDS:
        jobs.append((cell, arm, sd))
mine = [jb for i, jb in enumerate(jobs) if i % NSHARD == SHARD]
print("shard %d/%d runs=%d cells=%s" % (SHARD, NSHARD, len(mine), sorted(set(c for c, _ in CELLS))), flush=True)
res_rows = []
for k, (cell, arm, sd) in enumerate(mine):
    r = run_once(cell, arm, sd)
    if r is not None:
        res_rows.append(r)
        print("[%d/%d] %s %s s%d hit_rp_sup=%.3f hit_nrp=%.3f sup_rp=%.3f %.0fs" % (
            k + 1, len(mine), cell, arm, sd, r["hit_rp_sup"], r["hit_nrp"], r["sup_rp_sup"], time.time() - _t0), flush=True)
# aggregate per cell
agg = {}
for cell, arm in CELLS:
    rs = [r for r in res_rows if r["cell"] == cell and r["arm"] == arm]
    if not rs:
        continue
    def mstat(f):
        v = np.array([r_[f] for r_ in rs], dtype=float)
        v = v[~np.isnan(v)]
        return dict(mean=round(float(v.mean()), 4), sd=round(float(v.std(ddof=1)), 4), n=int(len(v))) if len(v) > 1 else dict(mean=round(float(v[0]), 4), sd=None, n=int(len(v))) if len(v) else dict(mean=None, sd=None, n=0)
    d = [r_["hit_rp_sup"] - r_["hit_nrp"] for r_ in rs if not np.isnan(r_["hit_rp_sup"])]
    agg[cell + "_" + arm] = dict(hit_rp_sup=mstat("hit_rp_sup"), hit_rp_cos=mstat("hit_rp_cos"),
        hit_nrp=mstat("hit_nrp"), hit_p=mstat("hit_p"), sup_rp_sup=mstat("sup_rp_sup"),
        sup_nrp=mstat("sup_nrp"), n_suppressed=mstat("n_suppressed"),
        delta_rp_nrp=dict(mean=round(float(np.mean(d)), 4), sd=round(float(np.std(d, ddof=1)), 4)) if len(d) > 1 else dict(mean=round(float(d[0]), 4), sd=None) if len(d) else dict(mean=None, sd=None),
        recovered=[r_["hit_rp_sup_recovered"] for r_ in rs if "hit_rp_sup_recovered" in r_])
out = dict(config=dict(N_LIB=N_LIB, N_P=N_P, K=K_PRACTICE, seeds=len(SEEDS), smoke=SMOKE,
                       judge="obs_of_L-035", test_do_rif=False),
           cells=agg, rows=res_rows, elapsed_s=round(time.time() - _t0, 1))
_outf = os.environ.get("EXP34_OUT", "exp34_rif_interference.json")
json.dump(out, open(os.path.join(ROOT, "results", _outf), "w", encoding="utf-8"), indent=1)
print("EXP34 DONE %.0fs rows=%d" % (time.time() - _t0, len(res_rows)))
