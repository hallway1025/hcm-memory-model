# -*- coding: utf-8 -*-
r"""
HCM L0 episodic path -- independent clean reimplementation (round 2).
Only documents read: README.md, docs/MODEL.md, paper/PAPER_final.md (+ audit/SHA256SUMS.txt
for checksum verification only). No source code, bench script, or existing result was read.

Implements with numpy only:
  encoder  : expectation-bank residual (MODEL 3.1) -> orthogonal expansion (MODEL 3.2)
             -> develop-once conscience threshold -> k-WTA sparse code -> dense evidence
  episodic : one-shot column store (MODEL 4) + modern-Hopfield retrieval (PAPER 3.3)
  gate     : g = sigma(w.[nu,r,eps,a]+b), theta_write (MODEL 7)

Experiments
  A capacity/exact recall : N in {2000,16000}, arms {no-resid, resid}, 20 exact-cue probes
  B residual effect       : N=8000, 3 seeds, 20 probes with 25%-zeroed cues (paired arms)
  C readout @60% missing  : N=2000 (task spec) and N=60 (paper 4.10 setup), 50 queries,
                            sparse-readout hit count (primary), dense readouts logged too

All choices forced by ambiguous/underspecified doc points are marked  CHOICE G-xx  and are
itemised in replication2_report.md. Results are reported as produced; no tuning to targets.
Run:  py -3 results\replication2\your_impl.py        (from the project root)
      set SMOKE=1 for a 1-minute smoke config.
"""
import os, sys, json, time
import numpy as np

T0 = time.time()
def log(msg):
    print("[%7.1fs] %s" % (time.time() - T0, msg), flush=True)

# ---------------- L0 preset (MODEL 12/14, PAPER 8) ----------------
DX, RHO = 128, 8
DH = RHO * DX            # 1024
K = 20                   # PAPER 3.2/8/9.3-P2-3 "k=20"; MODEL's ceil(0.02*1024)=21 superseded  [G-1]
BETA = 20.0              # Hopfield inverse temperature
LHOP = 3
TH_WRITE = 0.5
GW = np.array([2.2, 1.4, 0.6, 0.8]); GB = -1.2        # MODEL 7
GMU = 0.9                # bank EMA decay, MODEL 3.1
KAPPA_JUMP = 1.5         # MODEL 3.1
COS_NEW, NORM_NEW, BMAX = 0.3, 0.3, 8                 # MODEL 3.1 (at cap: refuse new template [G-7])
NU_DECAY = 0.9           # CHOICE G-7: EMA decay for nu_ema / EMA[1-cos] (docs only give gamma_mu)
NU_INIT  = 1.0           # CHOICE G-7: init for both EMAs (unit input vs zero template)
DEV_ETA = 0.05           # CHOICE G-3: conscience learning rate (docs give rule but not eta)
DEV_EPOCHS = 100         # CHOICE G-3: epoch cap, early stop on max|dtheta| < 1e-4
DEV_MAX = 2000           # CHOICE G-3: dev set = first min(N, 2000) library patterns

SMOKE = os.environ.get("SMOKE", "") == "1"
if SMOKE:
    SEEDS   = [0]
    NS_A    = [300]
    NS_B    = [500]
    NS_C    = [60, 120]
    NPROBE  = 5
    NQ      = 10
else:
    SEEDS   = [0, 1, 2]
    NS_A    = [2000, 16000]
    NS_B    = [8000] + ([16000] if os.environ.get("DIAG_B16K", "") == "1" else [])  # 16000: diagnostic only
    NS_C    = [2000, 60]
    NPROBE  = 20
    NQ      = 50

MISS_B = 0.25            # 32/128 features zeroed
MISS_C = 0.60            # 77/128 features zeroed

# ---------------- helpers ----------------
def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v

def topk_unit(s, k):
    nz = np.flatnonzero(s)
    if nz.size == 0:
        return np.zeros_like(s)
    idx = np.argpartition(s, s.size - k)[s.size - k:] if nz.size > k else nz
    h = np.zeros_like(s)
    h[idx] = s[idx]
    return unit(h)

def softmax(v):
    e = np.exp(v - v.max())
    return e / e.sum()

def sigmoid(x):
    if x >= 0.0:
        return 1.0 / (1.0 + np.exp(-x))
    e = np.exp(x)
    return e / (1.0 + e)

def cos_u(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def develop_theta(We, X):
    """Batched conscience on the task-input subset, developed once then frozen (D-018/D-3).
    CHOICE G-4: h_i in 'theta_i += eta*(h_i - k/d_h)' is read as the BINARY activity
    indicator. With the normalized code value the rule cannot converge (values ~0.05-0.3
    vs target 0.0195 -> theta diverges); with 0/1 activity it balances duty cycle."""
    theta = np.zeros(DH)
    tgt = K / DH
    for ep in range(DEV_EPOCHS):
        S = X @ We.T - theta[None, :]
        idx = np.argpartition(S, -K, axis=1)[:, -K:]
        H = np.zeros_like(S)
        np.put_along_axis(H, idx, 1.0, axis=1)
        d = DEV_ETA * (H.mean(axis=0) - tgt)
        theta += d
        if np.abs(d).max() < 1e-4:
            break
    S = X @ We.T - theta[None, :]
    idx = np.argpartition(S, -K, axis=1)[:, -K:]
    H = np.zeros_like(S)
    np.put_along_axis(H, idx, 1.0, axis=1)
    return theta, ep + 1, float(H.mean())

class Episodic:
    """Expectation-bank encoder + k-WTA + episodic column store + Hopfield retrieval."""

    def __init__(self, We, theta, use_resid):
        self.We, self.theta, self.use_resid = We, theta, use_resid
        self.MU = [np.zeros(DX)]      # MODEL 11 'MU={0}' vs PAPER 3.10 'MU=empty': same first-sample
        self.nu_ema = [NU_INIT]       # behavior (new template from zero either way)  [G-5, cosmetic]
        self.ema_mis = [NU_INIT]
        self.M = None
        self.src = None

    # ---- write-time bank step: select -> residual -> update (update AFTER residual) [G-8]
    def bank_step(self, x):
        d = [float(np.linalg.norm(x - mu)) for mu in self.MU]
        j = int(np.argmin(d))
        mu = self.MU[j]
        c = cos_u(x, mu)
        mis = 1.0 - c
        create = (d[j] > KAPPA_JUMP * self.nu_ema[j]) or (c < COS_NEW and np.linalg.norm(mu) > NORM_NEW)
        if create and len(self.MU) < BMAX:
            self.MU.append(np.zeros(DX)); self.nu_ema.append(NU_INIT); self.ema_mis.append(NU_INIT)
            j = len(self.MU) - 1
            mu = self.MU[j]; c = 0.0; mis = 1.0
        u = unit(x - mu) if self.use_resid else unit(x)
        nu_t = float(np.clip(mis / max(self.ema_mis[j], 1e-12), 0.0, 2.0))
        self.MU[j] = GMU * mu + (1.0 - GMU) * x
        self.nu_ema[j] = NU_DECAY * self.nu_ema[j] + (1.0 - NU_DECAY) * float(np.linalg.norm(x - mu))
        self.ema_mis[j] = NU_DECAY * self.ema_mis[j] + (1.0 - NU_DECAY) * mis
        return u, nu_t

    def bank_readonly(self, q):
        # MODEL 11 step 8: bank is read-only at query time (drifted mu used as-is)  [G-6]
        if not self.use_resid or len(self.MU) == 0:
            return unit(q)
        j = int(np.argmin([float(np.linalg.norm(q - mu)) for mu in self.MU]))
        return unit(q - self.MU[j])

    def dense_evidence(self, q, mode="spec"):
        u = self.bank_readonly(q)
        pre = self.We @ u
        if mode == "spec":     # PRIMARY, MODEL 3.2: s_t = relu(W_e u + b_e - theta)   [G-9]
            return np.maximum(pre - self.theta, 0.0)
        return pre             # SECONDARY, PAPER 3.2/3.3 literal: s_t = W_e u + b_e (raw)

    def encode(self, x):
        u, nu_t = self.bank_step(x)
        s = np.maximum(self.We @ u - self.theta, 0.0)   # b_e = 0  [CHOICE G-2]
        h = topk_unit(s, K)
        a_act = float(np.abs(h).sum()) / np.sqrt(K)
        g = sigmoid(float(GW @ np.array([nu_t, 0.0, 0.0, a_act])) + GB)
        return h, g

    def write_stream(self, X):
        n = X.shape[0]
        cols = np.zeros((n, DH)); src = np.zeros(n, dtype=np.int64)
        nw = 0
        for i in range(n):
            h, g = self.encode(X[i])
            if g >= TH_WRITE:
                cols[nw] = h; src[nw] = i; nw += 1
        self.M = cols[:nw]; self.src = src[:nw]
        return nw

    # ---- readouts ----
    def query_hop(self, q, mode="spec", L=LHOP):
        z = unit(np.maximum(self.dense_evidence(q, mode), 0.0))   # z0 = unit(relu(s_ev))
        a = softmax(BETA * (self.M @ z))
        for _ in range(L):                                        # PAPER 3.3 loop, output a^(L)
            z = unit(self.M.T @ a)
            a = softmax(BETA * (self.M @ z))
        return int(np.argmax(a))

    def query_hop1(self, q, mode="spec"):
        z = unit(np.maximum(self.dense_evidence(q, mode), 0.0))
        return int(np.argmax(softmax(BETA * (self.M @ z))))

    def query_sparse(self, q):
        s = np.maximum(self.We @ self.bank_readonly(q) - self.theta, 0.0)
        return int(np.argmax(self.M @ topk_unit(s, K)))

    def tgt(self, pred):
        return int(self.src[pred])

# ---------------- data ----------------
def make_lib(rng, N):
    X = rng.standard_normal((N, DX))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def apply_mask(x, mask):
    y = x.copy(); y[mask] = 0.0; return y

def build_pair(seed, N, X):
    rngW = np.random.default_rng(1000 + seed)
    Q, _ = np.linalg.qr(rngW.standard_normal((DH, DX)))   # W_e with orthonormal columns (PAPER 3.10)
    We = np.ascontiguousarray(Q)
    theta, ep_used, duty = develop_theta(We, X[: min(N, DEV_MAX)])
    models = {False: Episodic(We, theta, False), True: Episodic(We, theta, True)}
    return models, ep_used, duty

def common_perm(models, X, rngq, n_probe):
    srcs = set(models[False].src.tolist()) & set(models[True].src.tolist())
    common = np.array(sorted(srcs), dtype=np.int64)
    perm = rngq.choice(common, size=min(n_probe, common.size), replace=False)
    return perm

# ---------------- experiments ----------------
def run_C(models, X, seed, N):
    """60%-zeroed cues over the common written set; sparse readout is the primary metric,
    dense single-step / dense+Hopfield (both dense-evidence modes) logged alongside."""
    rngq = np.random.default_rng(10_000_000 * seed + 1000 * N + 33)
    perm = common_perm(models, X, rngq, NQ)
    masks = [rngq.choice(DX, size=int(round(MISS_C * DX)), replace=False) for _ in perm]
    rec = {}
    for arm in (False, True):
        m = models[arm]
        sp = d1 = dhp = d1p = dhpp = 0
        for i, mk in zip(perm, masks):
            q = apply_mask(X[int(i)], mk)
            sp  += int(m.tgt(m.query_sparse(q)) == int(i))
            d1  += int(m.tgt(m.query_hop1(q, "spec")) == int(i))
            dhp += int(m.tgt(m.query_hop(q, "spec")) == int(i))
            d1p += int(m.tgt(m.query_hop1(q, "paper_raw")) == int(i))
            dhpp += int(m.tgt(m.query_hop(q, "paper_raw")) == int(i))
        rec[str(arm)] = {
            "written": int(m.M.shape[0]), "n_query": int(perm.size), "sparse_hits": int(sp),
            "dense1_spec_hits": int(d1), "densehop_spec_hits": int(dhp),
            "dense1_paper_raw_hits": int(d1p), "densehop_paper_raw_hits": int(dhpp),
        }
        log("C seed=%d N=%d arm=%s  sparse %d/%d  dense1 %d  dense+Hop %d  (paper_raw mode: %d / %d)"
            % (seed, N, "resid" if arm else "noresid", sp, perm.size, d1, dhp, d1p, dhpp))
    return rec

res = {"meta": {}, "A": {}, "B": {}, "C": {}}
t_start = time.time()

for seed in SEEDS:
    # ================= A (and C at N=2000, reusing the same stores) =================
    for N in NS_A:
        rngX = np.random.default_rng(100_000 * seed + N)
        X = make_lib(rngX, N)
        models, ep_used, duty = build_pair(seed, N, X)
        log("seed=%d N=%d  theta developed in %d epochs, final duty=%.4f (target %.4f)"
            % (seed, N, ep_used, duty, K / DH))
        keyA = "N%d" % N
        res["A"].setdefault(keyA, {}).setdefault("seed%d" % seed, {})
        nw = {}
        for arm in (False, True):
            t0 = time.time()
            nw[arm] = models[arm].write_stream(X)
            log("A seed=%d N=%d arm=%s written=%d/%d (%.1fs)"
                % (seed, N, "resid" if arm else "noresid", nw[arm], N, time.time() - t0))
        rngq = np.random.default_rng(10_000_000 * seed + 1000 * N + 11)
        perm = common_perm(models, X, rngq, NPROBE)
        for arm in (False, True):
            hits = hits_sp = 0
            for i in perm:
                i = int(i)
                hits += int(models[arm].tgt(models[arm].query_hop(X[i])) == i)
                hits_sp += int(models[arm].tgt(models[arm].query_sparse(X[i])) == i)
            res["A"][keyA]["seed%d" % seed][str(arm)] = {
                "written": nw[arm], "n_probe": int(perm.size),
                "exact_hop_hits": int(hits), "exact_hop_rate": round(hits / max(perm.size, 1), 4),
                "exact_sparse_hits": int(hits_sp),
            }
            log("A seed=%d N=%d arm=%s  exact Hopfield %d/%d  sparse %d/%d"
                % (seed, N, "resid" if arm else "noresid", hits, perm.size, hits_sp, perm.size))
        if N in NS_C:
            res["C"].setdefault("N%d" % N, {})["seed%d" % seed] = run_C(models, X, seed, N)
        models = None; X = None

    # ================= C at library sizes not covered by A (e.g. N=60, paper 4.10) ======
    for N in NS_C:
        if N in NS_A:
            continue
        rngX = np.random.default_rng(100_000 * seed + N)
        X = make_lib(rngX, N)
        models, _, _ = build_pair(seed, N, X)
        for arm in (False, True):
            models[arm].write_stream(X)
        res["C"].setdefault("N%d" % N, {})["seed%d" % seed] = run_C(models, X, seed, N)
        models = None; X = None

    # ================= B =================
    for N in NS_B:
        rngX = np.random.default_rng(100_000 * seed + N)
        X = make_lib(rngX, N)
        models, _, _ = build_pair(seed, N, X)
        nw = {}
        for arm in (False, True):
            nw[arm] = models[arm].write_stream(X)
        rngq = np.random.default_rng(10_000_000 * seed + 1000 * N + 22)
        perm = common_perm(models, X, rngq, NPROBE)
        masks = [rngq.choice(DX, size=int(round(MISS_B * DX)), replace=False) for _ in perm]
        keyB = "N%d" % N
        res["B"].setdefault(keyB, {}).setdefault("seed%d" % seed, {})
        for arm in (False, True):
            m = models[arm]
            ms = mp = msp = ex = 0
            for i, mk in zip(perm, masks):
                i = int(i)
                q = apply_mask(X[i], mk)
                ms  += int(m.tgt(m.query_hop(q, "spec")) == i)
                mp  += int(m.tgt(m.query_hop(q, "paper_raw")) == i)
                msp += int(m.tgt(m.query_sparse(q)) == i)
                ex  += int(m.tgt(m.query_hop(X[i])) == i)
            res["B"][keyB]["seed%d" % seed][str(arm)] = {
                "written": nw[arm], "n_probe": int(perm.size),
                "miss_spec_hits": int(ms), "miss_spec_rate": round(ms / max(perm.size, 1), 4),
                "miss_paper_raw_hits": int(mp),
                "miss_sparse_hits": int(msp), "exact_hits": int(ex),
            }
            log("B seed=%d N=%d arm=%s written=%d  25%%missing spec %d/%d (paper_raw %d, sparse %d)  exact %d/%d"
                % (seed, N, "resid" if arm else "noresid", nw[arm], ms, perm.size, mp, msp, ex, perm.size))
        models = None; X = None

# ---------------- aggregate ----------------
res["meta"] = {
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "smoke": SMOKE,
    "seeds": SEEDS,
    "n_probe_A": NPROBE, "n_probe_B": NPROBE, "n_query_C": NQ,
    "miss_B": MISS_B, "miss_C": MISS_C,
    "wall_seconds": round(time.time() - t_start, 1),
    "choices": {
        "G-1 k": "20 (PAPER 3.2/8/9.3) not ceil(0.02*1024)=21 (MODEL 3.2)",
        "G-2 b_e": "0",
        "G-3 develop": "batched conscience, eta=0.05, <=100 epochs, dev set = first min(N,2000) library patterns, pre-write, then frozen",
        "G-4 h in theta rule": "binary activity indicator (duty-cycle reading)",
        "G-5 bank init": "single zero template (MODEL 11 == PAPER 3.10 in behavior)",
        "G-6 query bank": "read-only, drifted mu used as-is",
        "G-7 EMAs": "decay 0.9, init 1.0; template cap: refuse new when B=8",
        "G-8 update order": "residual with pre-update mu, then mu/EMA update",
        "G-9 dense evidence": "primary=relu(We u - theta) per MODEL 3.2; secondary=raw We u per PAPER 3.2/3.3",
        "G-10 stored column": "unit sparse code (g scaling dropped)",
        "G-11 dedup": "none (theta_dup not in MODEL.md; would not trigger on random data)",
        "G-12 gate aux": "r=0, eps=0",
    },
}

out_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, "replication2_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
log("results written to %s" % out_path)
