#!/usr/bin/env python3
'''exp35_calibration.py -- M0 confidence calibration on real corpus (survey I-region C3-prime).
Design: notes/exp35-design.md (pre-registered gates G1-G4). Library = peer-01 train (S256 loader,
exp27 convention: observe() per row in order, E.t = model step = row index, obs_of mapping,
judge L-035: action in {recall,separate} AND idx == obs_of[target_row]). Probes: in-store clean,
in-store degraded 30%, out-of-store test rows, frozen paraphrases. Features: raw_evidence,
perturbation evidence stats (5 char-drop re-queries), top-5 code consistency (mean pairwise cos,
via query(debug=True) additive output). Logistic (IRLS) fit on dev half, report ECE(10-bin)/
Brier/AUC on eval half per feature set A/B/C. Counts-only: no timing claims.
Pre-registered predictions: G-1 real-corpus ECE>0.05 (if <0.05 and AUC~1.0 -> task too easy);
G-2 consistency feature lowers ECE without lowering AUC; G-3 hard-set AUC<0.95 else check leakage.
'''
import sys, os, json, csv, re, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
from hcm import HCM, Config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = "--smoke" in sys.argv
_t0 = time.time()

_NANO = re.compile(r"\[[^\]]{1,12}\]")
def _clean(s):
    s = _NANO.sub(" ", str(s))
    return re.sub(r"\s+", " ", s).strip()

def load(fn):
    rows = list(csv.reader(open(os.path.join(ROOT, "bench", "corpus", fn), encoding="utf-8-sig")))
    return [r for r in rows[1:] if len(r) >= 3]
train = load("train__peer-02.csv")
test = load("test__peer-02.csv")   # L-055: was test__peer-01.csv (v1 leftover)
if SMOKE:
    train = train[:1500]
msgs_tr = [_clean(r[2]) for r in train]
msgs_te = [_clean(r[2]) for r in test]
print("corpus: train %d test %d" % (len(msgs_tr), len(msgs_te)), flush=True)

# ---------- embedding: exp23 S256 (exp27 loader) ----------
z = np.load(os.path.join(ROOT, "results", "exp23_semantic_embed.npz"), allow_pickle=True)
E0 = z["E"]; chars = [str(c) for c in z["chars"]]
V = chars + ([] if "\ufffd" in chars else ["\ufffd"])
Vi = {ch: i for i, ch in enumerate(V)}
M = E0.T.copy()
if "\ufffd" not in chars:
    r0 = np.random.default_rng(99)
    M = np.hstack([M, r0.standard_normal((M.shape[0], 1))])
    M[:, -1] /= np.linalg.norm(M[:, -1])
D_x = M.shape[0]
def x_of(s):
    idx = [Vi.get(c, Vi["\ufffd"]) for c in s] or [Vi["\ufffd"]]
    v = M[:, idx].sum(1)
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.zeros(D_x, dtype=np.float32)

cfg = Config(d_x=D_x, d_c=D_x, a=0.02, capacity=len(msgs_tr) + 10, beta=100.0, L=3, use_resid=False, seed=0)
m = HCM(cfg)
n_written = 0
for i, s in enumerate(msgs_tr):
    rec = m.observe(x_of(s))
    n_written += int(rec.get("wrote", False))
    if (i + 1) % 2000 == 0:
        print("  observed %d/%d  %.0fs" % (i + 1, len(msgs_tr), time.time() - _t0), flush=True)
nw = int(m.E.n)
obs_of = {int(m.E.t[j]): j for j in range(nw)}   # corpus row -> slot (exp27 convention)
print("library: wrote %d/%d  E.n=%d" % (n_written, len(msgs_tr), nw), flush=True)

NIN = 150 if not SMOKE else 30
NOUT = 150 if not SMOKE else 30
NPAR = 100 if not SMOKE else 12
rng = np.random.default_rng(20251107)

def char_drop(s, frac, seed):
    r = np.random.default_rng(seed)
    if len(s) <= 2: return s
    k = max(1, int(len(s) * frac))
    keep = sorted(r.choice(len(s), size=len(s) - k, replace=False))
    return "".join(s[j] for j in keep)

def feat_bundle(text, target_slot, tag, first_drop=0.0):
    x0 = x_of(char_drop(text, first_drop, rng.integers(1 << 30))) if first_drop > 0 else x_of(text)
    q = m.query(x0, debug=True)
    a = q.get("dbg", {}).get("a")
    hit = int(q.get("action") in ("recall", "separate") and q.get("idx", -1) == target_slot)
    raw1 = float(q["raw_evidence"])
    pert = []
    for jj in range(5):
        xp = x_of(char_drop(text, 0.15, 1000 * jj + rng.integers(1 << 29)))
        pert.append(float(m.query(xp)["raw_evidence"]))
    pert = np.array(pert)
    cons = 0.0
    if a is not None and len(a):
        order = np.argsort(-a)[:5]
        Cc = m.E.M[:, order].T
        Cn = Cc / (np.linalg.norm(Cc, axis=1, keepdims=True) + 1e-12)
        G = Cn @ Cn.T
        iu = np.triu_indices(len(order), 1)
        if len(iu[0]): cons = float(np.clip(G[iu].mean(), -1, 1))
    return dict(label=hit, tag=tag, hit=hit, raw1=raw1,
                pert_mean=float(pert.mean()), pert_std=float(pert.std()), cons=cons)

print("building probes...", flush=True)
recs = []
cand_in = [i for i in range(len(msgs_tr)) if i in obs_of]
pick_in = rng.choice(len(cand_in), size=min(NIN, len(cand_in)), replace=False)
for j in pick_in:
    i = int(cand_in[j])
    recs.append(feat_bundle(msgs_tr[i], obs_of[i], "in_clean", 0.0))
for j in pick_in:
    i = int(cand_in[j])
    recs.append(feat_bundle(msgs_tr[i], obs_of[i], "in_deg30", 0.30))
pick_out = rng.choice(len(msgs_te), size=min(NOUT, len(msgs_te)), replace=False)
for j in pick_out:
    recs.append(feat_bundle(msgs_te[int(j)], -1, "out", 0.0))
print("T1-T3 done: %d" % len(recs), flush=True)
# T4 frozen paraphrases (targets = 0-based train rows; skip rows rejected by dedup)
fp = json.load(open(os.path.join(ROOT, "bench", "frozen_paraphrases.json"), encoding="utf-8"))
# L-055 guard (2026-11-08): frozen asset must match the corpus actually on disk.
import hashlib as _hl
for _fn, _want in (fp.get("corpus_sha256") or {}).items():
    _got = _hl.sha256(open(os.path.join(ROOT, "bench", "corpus", _fn), "rb").read()).hexdigest()
    if _got != _want:
        raise SystemExit("FROZEN ASSET MISMATCH: %s disk=%s asset=%s -- regenerate asset (L-055)" % (_fn, _got[:12], _want[:12]))
print("frozen asset v%s, guarded files: %d" % (fp.get("version"), len(fp.get("corpus_sha256") or {})), flush=True)
n_par = n_skip = 0
for q_ in fp["queries"]:
    if q_.get("session") not in ("peer-01", "peer-02"):
        continue
    if n_par >= NPAR:
        break
    tgt = int(q_["row"])
    if _hl.sha256(msgs_tr[tgt].encode("utf-8")).hexdigest()[:16] != q_.get("target_clean_sha"):
        raise SystemExit("FROZEN TARGET MISMATCH row %d pid %s -- asset stale (L-055)" % (tgt, q_.get("pid")))
    if tgt not in obs_of:
        n_skip += 1
        continue
    recs.append(feat_bundle(q_["query_clean"], obs_of[tgt], "para", 0.0))
    n_par += 1
print("T4 paraphrases: %d (skipped unwritten targets: %d)" % (n_par, n_skip), flush=True)

def auc(score, label):
    pos = score[label == 1]; neg = score[label == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum(); eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))

def irls(Xf, y, iters=25, ridge=1e-3):
    w = np.zeros(Xf.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xf @ w, -30, 30)))
        W = np.clip(p * (1 - p), 1e-6, None)
        g = Xf.T @ (y - p) - ridge * w
        H = (Xf * W[:, None]).T @ Xf + ridge * np.eye(Xf.shape[1])
        step = np.linalg.solve(H, g)
        w = w + step
        if np.max(np.abs(step)) < 1e-9: break
    return w

def ece_brier(p, label, bins=10):
    ece = 0.0; n_ = len(p)
    edges = np.linspace(0, 1, bins + 1)
    for b in range(bins):
        msk = (p >= edges[b]) & (p < edges[b + 1] if b < bins - 1 else p <= edges[b + 1])
        if msk.sum() == 0: continue
        ece += msk.sum() / n_ * abs(label[msk].mean() - p[msk].mean())
    return float(ece), float(((p - label) ** 2).mean())

R = np.array([[r["raw1"]] for r in recs])
RB = np.array([[r["raw1"], r["pert_mean"], r["pert_std"]] for r in recs])
RC = np.array([[r["raw1"], r["pert_mean"], r["pert_std"], r["cons"]] for r in recs])
lab = np.array([r["hit"] for r in recs])
tags = [r["tag"] for r in recs]
is_dev = np.zeros(len(lab), dtype=bool)
for tag in sorted(set(tags)):
    ii = [k for k in range(len(lab)) if tags[k] == tag]
    for pos, k in enumerate(ii):
        is_dev[k] = (pos % 2 == 0)
res = dict(n=len(lab), tags={}, feature_sets={})
for tag in sorted(set(tags)):
    msk = np.array([t == tag for t in tags])
    res["tags"][tag] = dict(n=int(msk.sum()), hit_rate=round(float(lab[msk].mean()), 4))
for name, F in [("A_raw", R), ("B_pert", RB), ("C_cons", RC)]:
    mu = F[is_dev].mean(0); sd = F[is_dev].std(0) + 1e-12
    Z = (F - mu) / sd
    w = irls(np.hstack([Z[is_dev], np.ones((int(is_dev.sum()), 1))]), lab[is_dev])
    p = 1.0 / (1.0 + np.exp(-np.clip(np.hstack([Z, np.ones((len(lab), 1))]) @ w, -30, 30)))
    ece, br = ece_brier(p[~is_dev], lab[~is_dev])
    res["feature_sets"][name] = dict(auc_eval=round(auc(p[~is_dev], lab[~is_dev]), 4),
        ece_eval=round(ece, 4), brier_eval=round(br, 4), auc_all=round(auc(p, lab), 4),
        n_dev=int(is_dev.sum()), n_eval=int((~is_dev).sum()))
hard = np.array([t in ("in_deg30", "para", "out") for t in tags])
mu = RC[is_dev].mean(0); sd = RC[is_dev].std(0) + 1e-12
Z = (RC - mu) / sd
w = irls(np.hstack([Z[is_dev], np.ones((int(is_dev.sum()), 1))]), lab[is_dev])
p = 1.0 / (1.0 + np.exp(-np.clip(np.hstack([Z, np.ones((len(lab), 1))]) @ w, -30, 30)))
res["g3_hard_auc"] = round(auc(p[hard], lab[hard]), 4)
res["g3_hard_n"] = int(hard.sum())
res["elapsed_s"] = round(time.time() - _t0, 1)
res["config"] = dict(d_x=int(D_x), library_rows=len(msgs_tr), written=n_written,
                     n_probes=len(recs), smoke=SMOKE, judge="obs_of_L-035")
json.dump(dict(**res, rows=recs), open(os.environ.get("EXP35_ROWS_OUT") or os.path.join(ROOT, "results", "exp35_probe_rows.json"), "w", encoding="utf-8"), indent=1)
json.dump(res, open(os.environ.get("EXP35_OUT") or os.path.join(ROOT, "results", "exp35_calibration.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps(res, indent=1, ensure_ascii=False))
print("EXP35 DONE %.0fs" % (time.time() - _t0))
