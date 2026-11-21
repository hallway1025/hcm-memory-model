# -*- coding: utf-8 -*-
import sys, os, csv, re, zlib
sys.path.insert(0, "src")
import numpy as np
from hcm import HCM, Config
rows_all = list(csv.reader(open("bench/corpus/train__peer-01.csv", encoding="utf-8-sig")))
rows_all = [r for r in rows_all[1:] if len(r) >= 3]
_NANO = re.compile(r"\[[^\]]{1,12}\]")
msgs_all = [re.sub(r"\s+", " ", _NANO.sub(" ", r[2])).strip() for r in rows_all]
z = np.load("results/exp23_semantic_embed.npz", allow_pickle=True)
E0 = z["E"]; chars = [str(c) for c in z["chars"]]
V = chars + ([] if "\ufffd" in chars else ["\ufffd"]); Vi = {ch: i for i, ch in enumerate(V)}
Memb = E0.T.copy()
if "\ufffd" not in chars:
    r0 = np.random.default_rng(99)
    Memb = np.hstack([Memb, r0.standard_normal((Memb.shape[0], 1))]); Memb[:, -1] /= np.linalg.norm(Memb[:, -1])
D_x = Memb.shape[0]
def x_of(s):
    idx = [Vi.get(c, Vi["\ufffd"]) for c in s] or [Vi["\ufffd"]]
    v = Memb[:, idx].sum(1); n = np.linalg.norm(v)
    return v / n if n > 0 else np.zeros(D_x, dtype=np.float32)
seed = 0; rho = 0.30
rng = np.random.default_rng([seed, zlib.crc32(b"david")])
cfg = Config(d_x=D_x, d_c=D_x, a=0.02, capacity=70, beta=100.0, L=3, use_resid=False, seed=seed)
m = HCM(cfg)
pick = rng.choice(len(msgs_all), size=60, replace=False)
texts = [msgs_all[int(i)] for i in pick]
for s in texts: m.observe(x_of(s))
obs_of = {int(m.E.t[j]): j for j in range(m.E.n)}
slot_texts = {obs_of[i]: texts[i] for i in obs_of}
written = sorted(obs_of.keys())
P_rows = list(rng.choice(written, size=20, replace=False))
P_slots = [obs_of[i] for i in P_rows]
att = np.ones(m.E.capacity)
for i in P_rows:
    x = x_of(slot_texts[obs_of[i]])
    for _ in range(3):
        qq = m.query(x, do_rif=False, debug=True)
        a = qq["dbg"]["a"]
        order = np.argsort(-a)[:5]
        m.E.use[int(order[0])] += 1
        for j in order[1:]:
            j = int(j)
            if j != order[0]:
                m.E.M[:, j] *= (1.0 - rho); att[j] *= (1.0 - rho)
print("att of P slots: min %.4f max %.4f" % (min(att[s] for s in P_slots), max(att[s] for s in P_slots)))
def dbg_probe(s):
    x = x_of(slot_texts[s])
    r_ = m.query(x, do_rif=False, debug=True)
    a = r_["dbg"]["a"]; order = r_["dbg"]["order"]
    f_top = [(int(i), round(float(a[int(i)]), 4), round(float(att[int(i)]), 3)) for i in order[:4]]
    print("slot %3d action=%s idx=%d att[idx]=%.3f att[s]=%.3f top4(slot,a,att)=%s" % (
        s, r_["action"], r_["idx"], att[r_["idx"]] if r_["idx"] >= 0 else -1, att[s], f_top))
fails = 0
for s in P_slots[:10]:
    r_ = m.query(x_of(slot_texts[s]), do_rif=False)
    ok = r_["action"] in ("recall", "separate") and r_["idx"] == s
    if not ok:
        fails += 1
        dbg_probe(s)
print("P fails in first 10:", fails)
sup = att[: m.E.n]
Rp = [s for s in range(m.E.n) if sup[s] < 0.999 and s not in P_slots]
t_lo, t_hi = np.percentile([s for s in Rp], [10, 90])
never = [s for s in range(m.E.n) if sup[s] >= 0.999 and s not in P_slots and t_lo <= s <= t_hi]
Nrp = list(rng.choice(never, size=10, replace=False))
fn = 0
for s in Nrp:
    r_ = m.query(x_of(slot_texts[s]), do_rif=False)
    ok = r_["action"] in ("recall", "separate") and r_["idx"] == s
    if not ok:
        fn += 1
        dbg_probe(s)
print("Nrp fails in 10:", fn)
