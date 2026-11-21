# -*- coding: utf-8 -*-
'''exp45_online_beta.py -- β 自适应跨 encoder 迁移（工单 2 周增量·实验 2）。

预注册（本 docstring 先于任何正式数据落盘；40-EXP-LEDGER + D-066 同步提交）：
- β*(N) 在线学习（char encoder）：NEW256 在 10 个写库检查点（每 CKPT_EVERY=3 chunk
  =12000 行）上按验证命中率选 β：网格 BETAS=(20,100,500,2500)（exp19g 同款），
  N_VAL=64 验证探针（intrain 已写行、排除评测目标；deg0.3；lam=1.0 episodic-only；
  corrected judge）；β*(N)=argmax，平局取更小 β。输出 β*(N) 实测曲线。
  读出选择依据：β 只进入 episodic softmax（model.py:233/235）；lam=0 时 β 在
  构造上无效，故本实验读出 lam=1.0（exp29 主协议同款）。
- 迁移（sem encoder）：S256npz（exp23 PPMI+SVD 批拟合；与 NEW256 同训练语料、
  不同构造。exp29 口径中它带"泄漏参考臂"注记——本实验用它检验 β* 规则的构造
  不变性，不作为无泄漏主张，如实披露）在同一检查点：transferred=用 char 学到的
  β*(N_char)（同 N 对齐；若命中网格值则复用该格查询，节省一次）；oracle=全网格
  扫描取 hand-tuned best；fixed100=β=100（现网基线，网格内格点复用）。
- 预注册门（工单原文"不劣于手工最优 ±10%"）：mean_ck(transferred) >= 0.9 ×
  mean_ck(hand) 为正；< 0.9 为负。两分支均如实报。fixed100 同步记录。
- 次要读出（描述性）：lam=0.5 融合终评；β*(N) 曲线对照 exp19g 坍缩律预测
  （softmax 对比度随 N 坍缩 → β* 应随 N 上升）。
- 终评（两库满）：settings {fixed100, transferred(=char 最终 β*), hand_best} ×
  组 {intrain200 deg0/deg0.3 lam1.0, dup200 deg0 lam1.0, intrain200 deg0.3 lam0.5}。
- seeds {0,1,2}；每 seed 一个进程；探针/验证种子空间与 exp44 同约定
  （VAL_SEED_BASE=9000 避开 41000+pid）。
'''
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "bench"))

import exp29_text_frontend as e29  # noqa: E402
from hcm import Config, HCM  # noqa: E402

CHUNK = 4000
CKPT_EVERY = 3          # 每 3 chunk（12000 行）一个检查点
N_VAL = 64
BETAS = (20.0, 100.0, 500.0, 2500.0)
BASE_BETA = 100.0
VAL_SEED_BASE = 9000
SEEDS = (0, 1, 2)

PRE_REG = {
    "primary": "mean over checkpoints of validation hit rate (deg0.3, lam=1.0, corrected judge, N_VAL=64)",
    "gate": "pre-registered: transferred >= 0.9 * hand-tuned-best (mean_ck) positive; < 0.9 negative; both verbatim",
    "beta_grid": list(BETAS),
    "tie_rule": "argmax hit rate; ties -> smaller beta",
    "reuse_rule": "transferred beta equal to a grid value reuses that grid cell's queries",
    "transfer_def": "char-encoder beta*(N) applied pointwise to sem encoder at same N",
    "sem_encoder_disclosure": "S256npz is batch-fitted (exp23); used as different-construction encoder for transfer invariance, not as a no-leak claim",
    "reason_lam10": "beta only enters episodic softmax (model.py:233/235); lam=0 makes beta a no-op by construction",
}


def make_arm(kind, seed, train_raw, train_clean):
    name = "NEW256" if kind == "char" else "S256npz"
    # NEW256Arm 的 encoder seed 固定 0；跨 seed 差异由 model seed + 验证抽样承担
    arm, _meta = e29.make_arm(name, train_raw, train_clean)
    return arm, name


def val_hits(rows, m, arm, train_clean, obs_of, beta, seed_tag, lv=0.3, lam=1.0):
    hits = []
    m.cfg.lam = lam
    m.cfg.beta = beta
    try:
        for j, i in enumerate(rows):
            c = train_clean[i]
            mask = e29.corrupt_positions(c, lv, VAL_SEED_BASE + seed_tag * N_VAL + j)
            qstr = e29.apply_del(c, mask)
            q = arm.encode_query(qstr, "")
            o = m.query(q, do_rif=False)
            hits.append(bool(e29.judge(o.get("action"), o.get("idx"), obs_of, i)))
    finally:
        m.cfg.beta = BASE_BETA
        m.cfg.lam = e29.LAM
    return hits


def sample_val_rows(written_rows, excl_rows, rng, n=N_VAL):
    pool = np.asarray(sorted(set(written_rows) - set(excl_rows)), dtype=np.int64)
    if pool.size == 0:
        return []
    k = min(n, pool.size)
    return pool[rng.choice(pool.size, size=k, replace=False)].tolist()


def run_build(kind, seed, train_raw, train_clean, excl_rows, val_rng, smoke,
              char_sched=None):
    """写一个 encoder 的库并在检查点扫 β。返回 (m, arm, obs_of, slot_of_obs, curves, X)。"""
    arm, name = make_arm(kind, seed, train_raw, train_clean)
    if hasattr(arm, "fit"):
        arm.fit(train_raw[:2000] if smoke else train_raw)
    X = arm.encode_inputs(train_raw, train_clean)
    n = len(train_raw)
    m = HCM(Config(d_x=arm.dim, d_c=arm.dim, a=0.02, capacity=n + 10,
                   beta=BASE_BETA, L=e29.L, use_resid=False, lam=e29.LAM, seed=seed))
    step = 500 if smoke else CHUNK
    ck_every = 1 if smoke else CKPT_EVERY
    n_val = 24 if smoke else N_VAL
    curves = {"grid": [], "fixed100": [], "beta_star": [], "transferred": []}
    snaps = []
    obs_of = {}
    slot_of_obs = {}
    wstats = {"wrote": 0, "gate_rejected": 0, "dup_or_full_rejected": 0}
    t0 = time.perf_counter()
    n_ckpt = 0
    for ci, start in enumerate(range(0, n, step)):
        end = min(n, start + step)
        for i in range(start, end):
            rec = m.observe(X[i])
            if rec["wrote"]:
                wstats["wrote"] += 1
            else:
                wstats["gate_rejected" if rec["g"] < m.cfg.theta_write else "dup_or_full_rejected"] += 1
        obs_of = {s: int(m.E.t[s]) for s in range(int(m.E.n))}
        slot_of_obs = {v: s for s, v in obs_of.items()}
        written_rows = sorted(obs_of.values())
        if (ci + 1) % ck_every != 0 and end < n:
            continue
        rows = sample_val_rows(written_rows, excl_rows, val_rng, n=n_val)
        if not rows:
            continue
        n_ckpt += 1
        N_now = int(m.E.n)
        snaps.append({"N": N_now, "rows": list(rows), "obs_of": dict(obs_of)})
        rates = {}
        for b in BETAS:
            h = val_hits(rows, m, arm, train_clean, obs_of, b, seed_tag=100 * seed + n_ckpt)
            rates["%g" % b] = float(np.mean(h))
            curves["grid"].append({"N": N_now, "beta": b, "rate": rates["%g" % b]})
        best_b = min((b for b in BETAS), key=lambda b: (-rates["%g" % b], b))
        curves["beta_star"].append({"N": N_now, "beta": best_b, "rates": rates})
        # fixed100 曲线（网格内格点复用）
        curves["fixed100"].append({"N": N_now, "beta": BASE_BETA, "rate": rates["%g" % BASE_BETA]})
        print("  [%s s%d] ckpt %d N=%d wrote=%d rates=%s beta*=%g %.0fs"
              % (kind, seed, n_ckpt, N_now, wstats["wrote"],
                 {k: round(v, 3) for k, v in rates.items()}, best_b,
                 time.perf_counter() - t0), flush=True)
    write_dt = time.perf_counter() - t0
    return {"m": m, "arm": arm, "name": name, "obs_of": obs_of,
            "slot_of_obs": slot_of_obs, "curves": curves, "write_dt": write_dt,
            "wstats": wstats, "X": X, "n_ckpt": n_ckpt, "snaps": snaps}


def final_eval(kind, B, probes, group_qsrc, char_final_beta, hand_final_beta, smoke):
    m, arm = B["m"], B["arm"]
    obs_of, slot_of_obs = B["obs_of"], B["slot_of_obs"]
    settings = {"fixed100": BASE_BETA,
                "transferred": char_final_beta,
                "hand_best": hand_final_beta}
    out = {}
    for sname, bval in settings.items():
        m.cfg.beta = bval
        blocks = {}
        try:
            for gname, levels, lam_tag, lam_v in (
                    ("intrain200", (0.0, 0.3), "lam10", 1.0),
                    ("dup200", (0.0,), "lam10", 1.0),
                    ("intrain200", (0.3,), "lam05", 0.5)):
                qtexts, qclean = group_qsrc[gname]
                for lv in levels:
                    m.cfg.lam = lam_v
                    recs = probes["groups"][gname]
                    r, _pp = e29.evaluate_group(m, arm, qtexts, qclean, recs, lv,
                                                obs_of, slot_of_obs, B["X"])
                    blocks["%s|%s|deg%.1f" % (gname, lam_tag, lv)] = {
                        kk: vv for kk, vv in r.items() if kk != "dprime"}
        finally:
            m.cfg.beta = BASE_BETA
            m.cfg.lam = e29.LAM
        out[sname] = {"beta": bval, "eval": blocks}
    return out


def run_seed(seed, out_path, smoke=False):
    t00 = time.perf_counter()
    files = e29.TRAIN_FILES[:1] if smoke else e29.TRAIN_FILES
    train_raw = []
    for fn in files:
        train_raw.extend(e29.read_texts(fn, limit=2000 if smoke else None))
    train_clean = [e29.clean_text(t) for t in train_raw]
    n = len(train_raw)
    test_raw = []
    for fn in (e29.TEST_FILES[:1] if smoke else e29.TEST_FILES):
        test_raw.extend(e29.read_texts(fn, limit=300 if smoke else None))
    test_clean = [e29.clean_text(t) for t in test_raw]
    probes = e29.build_full_probes(train_raw, train_clean, test_raw, test_clean)
    excl_rows = set(int(rec["qsrc"]) for rec in probes["groups"]["intrain200"])
    excl_rows.update(int(rec["target"]) for rec in probes["groups"]["dup200"])
    group_qsrc = {"intrain200": (train_raw, train_clean), "dup200": (test_raw, test_clean)}
    val_rng = np.random.default_rng(VAL_SEED_BASE + seed)

    Bc = run_build("char", seed, train_raw, train_clean, excl_rows, val_rng, smoke)
    Bs = run_build("sem", seed, train_raw, train_clean, excl_rows, val_rng, smoke)
    # 迁移 pass（检查点序数对齐：第 k 个检查点处理了相同数量的训练行）
    char_bs = Bc["curves"]["beta_star"]
    for k, snap in enumerate(Bs["snaps"]):
        if k >= len(char_bs):
            break
        tb = char_bs[k]["beta"]
        key = "%g" % tb
        sem_rates_k = Bs["curves"]["beta_star"][k]["rates"] if k < len(Bs["curves"]["beta_star"]) else {}
        if key in sem_rates_k:
            Bs["curves"]["transferred"].append({"N": snap["N"], "N_char": char_bs[k]["N"],
                                                "beta": tb, "rate": sem_rates_k[key], "reused": True})
        else:
            h = val_hits(snap["rows"], Bs["m"], Bs["arm"], train_clean, snap["obs_of"], tb,
                         seed_tag=700000 + seed * 100 + k)
            Bs["curves"]["transferred"].append({"N": snap["N"], "N_char": char_bs[k]["N"],
                                                "beta": tb, "rate": float(np.mean(h)), "reused": False})
        print("  [transfer s%d] ckpt %d beta*=%g rate=%.3f %s"
              % (seed, k, tb, Bs["curves"]["transferred"][-1]["rate"],
                 "reused" if Bs["curves"]["transferred"][-1]["reused"] else "queried"), flush=True)

    def last_beta(curve_key, B):
        arr = B["curves"][curve_key]
        return arr[-1]["beta"] if arr else BASE_BETA

    def hand_curve_mean(B):
        xs = [max(v["rates"].values()) for v in B["curves"]["beta_star"]]
        return float(np.mean(xs)) if xs else None

    def curve_mean(B, key):
        xs = [v["rate"] for v in B["curves"][key]]
        return float(np.mean(xs)) if xs else None

    stats = {}
    for tag, B in (("char", Bc), ("sem", Bs)):
        stats[tag] = {"oracle_mean": hand_curve_mean(B),
                      "fixed100_mean": curve_mean(B, "fixed100"),
                      "beta_star_mean_beta": (float(np.mean([v["beta"] for v in B["curves"]["beta_star"]]))
                                              if B["curves"]["beta_star"] else None)}
    stats["sem"]["transferred_mean"] = curve_mean(Bs, "transferred")
    # 预注册门：sem transferred vs sem hand（同检查点均值；对齐 N 后取均值）
    t_pairs = [(v["N"], v["rate"]) for v in Bs["curves"]["transferred"]]
    h_by_N = {v["N"]: max(v["rates"].values()) for v in Bs["curves"]["beta_star"]}
    pairs = [(r, h_by_N[N]) for N, r in t_pairs if N in h_by_N]
    if pairs:
        tr = float(np.mean([p[0] for p in pairs]))
        hd = float(np.mean([p[1] for p in pairs]))
        stats["gate"] = {"transferred_mean": tr, "hand_mean": hd,
                         "ratio": tr / max(hd, 1e-9),
                         "verdict": "POSITIVE" if tr >= 0.9 * hd else "NEGATIVE",
                         "n_checkpoints": len(pairs)}
    else:
        stats["gate"] = {"verdict": "NO_PAIRS", "n_checkpoints": 0}

    evals = {}
    for tag, B in (("char", Bc), ("sem", Bs)):
        hand_final = min(B["curves"]["beta_star"][-1]["rates"].items(), key=lambda kv: (-kv[1], float(kv[0])))[0] if B["curves"]["beta_star"] else "%g" % BASE_BETA
        evals[tag] = final_eval(tag, B, probes, group_qsrc,
                                char_final_beta=last_beta("beta_star", Bc),
                                hand_final_beta=float(hand_final), smoke=smoke)
    out = {
        "exp": "exp45_online_beta", "seed": seed, "smoke": smoke,
        "pre_registered": PRE_REG,
        "config": {"chunk": 500 if smoke else CHUNK,
                   "ckpt_every": 1 if smoke else CKPT_EVERY,
                   "n_val": 24 if smoke else N_VAL,
                   "betas": list(BETAS), "base_beta": BASE_BETA,
                   "n_train": n, "n_test": len(test_raw),
                   "n_written_char": int(Bc["m"].E.n), "n_written_sem": int(Bs["m"].E.n),
                   "wstats_char": Bc["wstats"], "wstats_sem": Bs["wstats"],
                   "L": e29.L, "lam_base": e29.LAM,
                   "corpus_files": files, "probe_rule": probes["rule"]},
        "timing_s": {"write_char": round(Bc["write_dt"], 1), "write_sem": round(Bs["write_dt"], 1),
                     "total": round(time.perf_counter() - t00, 1)},
        "beta_star_curve_char": Bc["curves"]["beta_star"],
        "sem_curves": {"grid": Bs["curves"]["grid"], "transferred": Bs["curves"]["transferred"],
                       "fixed100": Bs["curves"]["fixed100"], "beta_star": Bs["curves"]["beta_star"]},
        "char_grid": Bc["curves"]["grid"],
        "summary_stats": stats,
        "eval": evals,
        "completed_at": e29.now_iso(),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(e29.sanitize(out), f, indent=1, ensure_ascii=False)
    print("seed %d done in %.0fs char_n=%d sem_n=%d gate=%s"
          % (seed, time.perf_counter() - t00, int(Bc["m"].E.n), int(Bs["m"].E.n),
             stats["gate"].get("verdict")), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = args.out or os.path.join(ROOT, "results",
                                   "exp45_smoke.json" if args.smoke else "exp45_s%d.json" % args.seed)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    run_seed(args.seed, out, smoke=args.smoke)


if __name__ == "__main__":
    main()
