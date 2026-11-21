# -*- coding: utf-8 -*-
"""exp44: 在线 tag 三臂实验（2 周增量工单主体 1；D-066 预注册）。

问题：B3 谱系（Spens-Burgess-Behrens NeurIPS 2025）主张「学习型 state-dependent tag
使优先回放有效」；HCM 现网 tag 是固定公式（recency^-alpha * sigmoid(nu)^b * (1+r)^c），
exp15/exp6 在合成任务上 20 种子未检出 priority>uniform。本实验在真实六会话语料上问：
把判据参数 (alpha, b) 在线化（score-gap 驱动）能否让优先回放变得有效？

三臂（写库共享、锁步交错，唯一变量=回放采样策略）：
  A fixed    alpha=1, b=1（现网默认公式）
  B online   (alpha,b) 每 N=CHUNK*EVAL_EVERY 次写入更新一次：在验证池（非评测探针
             目标行、deg0.3、semantic-only 读出）上测命中率，score-gap=当前-最优，
             优于则保持，劣于则移向已试最优/最近未扫格点（网格 {0,.25,.5,1,2,4}^2）
  C uniform  Consolidator(uniform=True)

协议（全部预注册于本 docstring，跑前不变）：
  语料   六会话 train（exp29 v2 同款 TRAIN_FILES，行序不变=保真 recency 结构）
  写库   NEW256(encoder seed=seed)，Config 同 exp29 full（d_x=256,beta=100,L=3,
         use_resid=False,lam=1.0,capacity=n+10,seed=seed）；写库三臂共享
  回放   每 CHUNK=4000 写入后，每臂各跑 CYCLES_PER_CHUNK=8 个 cycle
         （n_r=96, interleave=0.5 -> 每 cycle 48 新样本；总回放 ~12% 已写库）
  探针   intrain200/dup200 = exp29 build_full_probes（seed 2029/41000 系列，
         与 exp29 完全同款；dup200 目标=train 首现 obs）
  判据   L-035 corrected criterion：abstain/idx<0/obs_of[slot]!=target=miss，
         分母固定=探针数（e29.judge 复用）
  读出   primary = intrain200 deg0.3 semantic-only（cfg.lam=0 时 f=s_n）hit率。
         理由（预注册）：exp29 的 lam=1.0 使语义读出权重为 0，回放策略在构造上
         不可测；semantic-only 是回放直接塑形的读出面。
         secondary = intrain deg0 semantic-only、intrain deg0.3 fused lam=0.5、
         dup200 deg0 semantic-only、回放覆盖率（唯一回放槽位数/评测目标覆盖率）
  预注册门 primary 上 B>A 为正结果如实报；B<=A 为负结果同样如实报（两分支都写）。
  统计   seed=0..7（>=8），同 seed 三臂同库配对；sign-flip 置换检验 + Cohen dz
         （exp44_analyze.py 聚合）

隔离保证：每臂独立 SemanticStore（seed=c.seed+2+arm_idx）与 Consolidator（rng
seed=10*seed+arm_idx）；每臂 cycle 前后快照/还原 E.use（use 不进检索/优先级，
还原只为状态纯净）；A/C 不做任何在线更新。

机2 运行（L-044 WMI 分离）：
  py -X utf8 -u bench/exp44_online_tag.py --seed S --out results/exp44_sS.json
冒烟（机1）：--smoke 走 TRAIN_FILES[0] 前 2000 行、CHUNK=500、EVAL_EVERY=1。
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

spec = importlib.util.spec_from_file_location(
    "e29", os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp29_text_frontend.py"))
e29 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e29)

from hcm import HCM, Config            # noqa: E402
from hcm.consolidate import Consolidator  # noqa: E402
from hcm.semantic import SemanticStore    # noqa: E402
from hcm.text_encoder import CharacterSemanticEncoder  # noqa: E402

CHUNK = 4000
CYCLES_PER_CHUNK = 8
N_R = 96
EVAL_EVERY = 2          # 每 2 chunk（8000 写入）做一次在线评估/更新
N_VAL = 64              # 验证池探针数（每次评估重抽）
GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
SEEDS = range(8)
VAL_SEED_BASE = 9000    # 验证池退化种子空间（避开探针 41000+pid）

ARMS = ("A_fixed", "B_online", "C_uniform")
ARM_IDX = {"A_fixed": 0, "B_online": 1, "C_uniform": 2}

PRE_REG = {
    "primary": "intrain200 deg0.3 semantic-only (cfg.lam=0) corrected-criterion hit rate",
    "gate": "pre-registered: B>A positive reported as positive; B<=A reported as negative; both verbatim",
    "stat": "8 seeds paired (shared store per seed); sign-flip permutation + Cohen dz in exp44_analyze.py",
    "reason_semantic_readout": "exp29 lam=1.0 gives semantic weight 0 -> replay policy unmeasurable by construction; lam=0 is the surface replay directly shapes",
}


def make_enc_arm(seed):
    class _Arm:
        """NEW256 同款（exp29.NEW256Arm），encoder seed 可变。"""
        def __init__(self):
            self.enc = CharacterSemanticEncoder(dim=256, window=3, seed=seed)
            self.dim = 256
            self.name = "NEW256"
        def fit(self, raw_texts):
            return self.enc.fit(raw_texts)
        def encode_inputs(self, raw, cleaned):
            return np.asarray(self.enc.transform(raw), dtype=np.float32)
        def encode_query(self, qstr, raw):
            return self.enc.encode(qstr if qstr else raw)
    return _Arm()


def fresh_sem(cfg, arm_key):
    return SemanticStore(cfg.d_x, cfg.d_h, cfg.d_z,
                         seed=cfg.seed + 2 + ARM_IDX[arm_key],
                         kappa=cfg.kappa, u_meta=cfg.u_meta)


def fresh_cons(E, S, arm_key, seed, alpha=1.0, b=1.0):
    return Consolidator(E, S, n_r=N_R, epochs=3, alpha=alpha, b=b, c=1.0,
                        T_p=0.5, interleave=0.5, eta=0.05,
                        uniform=(arm_key == "C_uniform"),
                        seed=10 * seed + ARM_IDX[arm_key])


def replayed_slots(E, use_before):
    """cycle 后 use 计数上升的槽位 = 本批被回放的槽位。"""
    n = int(E.n)
    delta = E.use[:n] - use_before[:n]
    return np.nonzero(delta > 0)[0].tolist()


def val_query_rows(rows, m, arm, train_clean, obs_of, seed_tag):
    """验证池查询（deg0.3、semantic-only）。返回命中率与逐条命中。"""
    hits = []
    m.cfg.lam = 0.0
    try:
        for j, i in enumerate(rows):
            c = train_clean[i]
            mask = e29.corrupt_positions(c, 0.3, VAL_SEED_BASE + seed_tag * N_VAL + j)
            qstr = e29.apply_del(c, mask)
            q = arm.encode_query(qstr, "")
            o = m.query(q, do_rif=False)
            hits.append(bool(e29.judge(o.get("action"), o.get("idx"), obs_of, i)))
    finally:
        m.cfg.lam = 1.0
    return hits


def sample_val_rows(written_rows, excl_rows, rng, n=N_VAL):
    pool = np.asarray(sorted(set(written_rows) - set(excl_rows)), dtype=np.int64)
    if pool.size == 0:
        return []
    k = min(n, pool.size)
    return pool[rng.choice(pool.size, size=k, replace=False)].tolist()


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
    eval_targets = {g: [int(rec["target"]) for rec in probes["groups"][g]]
                    for g in ("intrain200", "dup200")}
    # dup200 查询侧是 test 行（qsrc=test 行号），目标=train 首现行
    group_qsrc = {"intrain200": (train_raw, train_clean), "dup200": (test_raw, test_clean)}

    arm_obj = make_enc_arm(seed)
    t0 = time.perf_counter()
    arm_obj.fit(train_raw)
    fit_dt = time.perf_counter() - t0
    X = arm_obj.encode_inputs(train_raw, train_clean)
    enc_dt = time.perf_counter() - t0 - fit_dt

    m = HCM(Config(d_x=arm_obj.dim, d_c=arm_obj.dim, a=0.02, capacity=n + 10,
                   beta=e29.BETA, L=e29.L, use_resid=False, lam=e29.LAM, seed=seed))
    obs_of = {}
    slot_of_obs = {}
    wstats = {"wrote": 0, "gate_rejected": 0, "dup_or_full_rejected": 0}
    arms = {k: {"sem": fresh_sem(m.cfg, k),
                "cons": None,
                "replayed": set(),
                "val_curve": [],
                "theta_traj": [],
                "theta": (1.0, 1.0)} for k in ARMS}
    for k in ARMS:
        a, b = arms[k]["theta"]
        arms[k]["cons"] = fresh_cons(m.E, arms[k]["sem"], k, seed, alpha=a, b=b)
    val_rng = np.random.default_rng(VAL_SEED_BASE + seed)
    tried = {}
    order = sorted(((a, b) for a in GRID for b in GRID),
                   key=lambda ab: abs(ab[0] - 1.0) + abs(ab[1] - 1.0))
    untried = [ab for ab in order if ab != (1.0, 1.0)]

    t0 = time.perf_counter()
    n_chunks = 0
    for ci, start in enumerate(range(0, n, (500 if smoke else CHUNK))):
        step = 500 if smoke else CHUNK
        end = min(n, start + step)
        for i in range(start, end):
            rec = m.observe(X[i])
            if rec["wrote"]:
                wstats["wrote"] += 1
            else:
                wstats["gate_rejected" if rec["g"] < m.cfg.theta_write else "dup_or_full_rejected"] += 1
        n_chunks += 1
        # 统一重取 obs/slot 映射（write 路径内部决定槽位）
        obs_of = {s: int(m.E.t[s]) for s in range(int(m.E.n))}
        slot_of_obs = {v: s for s, v in obs_of.items()}
        written_rows = sorted(obs_of.values())
        for k in ARMS:
            A = arms[k]
            use0 = m.E.use.copy()
            losses = []
            for _ in range((2 if smoke else CYCLES_PER_CHUNK)):
                loss, _nr = A["cons"].cycle(float(m.t))
                losses.append(loss)
            A["replayed"].update(replayed_slots(m.E, use0))
            if ci % (1 if smoke else EVAL_EVERY) == 0:
                rows = sample_val_rows(written_rows, excl_rows, val_rng)
                sc = val_query_rows(rows, m, arm_obj, train_clean, obs_of,
                                    seed_tag=ci) if rows else []
                rate = float(np.mean(sc)) if sc else None
                A["val_curve"].append({"chunk": ci, "n_val": len(sc), "rate": rate})
                if k == "B_online":
                    # 预注册更新规则：均值最优者 incumbent；连续 2 次评估 incumbent
                    # 无提升（gap<=0）则探索最近未扫格点。gap=当前均值-全局最优均值。
                    key = A["theta"]
                    tried.setdefault(key, []).append(rate)
                    means = {}
                    for kk, v in tried.items():
                        xs = [x for x in v if x is not None]
                        if xs:
                            means[kk] = float(np.mean(xs))
                    best_k = max(means, key=means.get) if means else key
                    gap = (means.get(key, 0.0) - max(means.values())) if means else None
                    if best_k != key:
                        A["theta"] = best_k
                    elif len(tried.get(key, [])) >= 2 and untried:
                        A["theta"] = untried.pop(0)
                    a, b = A["theta"]
                    A["cons"].alpha, A["cons"].b = a, b
                    A["theta_traj"].append({"chunk": ci, "alpha": a, "b": b,
                                            "rate": rate, "gap": gap})
        print("  chunk %d end=%d wrote=%d %.0fs" % (ci, end, wstats["wrote"],
                                                    time.perf_counter() - t0), flush=True)

    write_dt = time.perf_counter() - t0
    evals = {}
    per_probe_primary = {}
    t0 = time.perf_counter()
    for k in ARMS:
        A = arms[k]
        orig_sem = m.sem
        m.sem = A["sem"]
        evals[k] = {}
        try:
            for lam_tag, lam_v in (("lam0", 0.0), ("lam05", 0.5)):
                m.cfg.lam = lam_v
                for gname, levels in (("intrain200", (0.0, 0.3)), ("dup200", (0.0,))):
                    if lam_tag == "lam05" and gname == "dup200":
                        continue  # 次要读出只对 intrain 做，省时
                    qtexts, qclean = group_qsrc[gname]
                    for lv in levels:
                        recs = probes["groups"][gname]
                        r, pp = e29.evaluate_group(m, arm_obj, qtexts, qclean,
                                                   recs, lv, obs_of, slot_of_obs, X)
                        evals[k]["%s|%s|deg%.1f" % (gname, lam_tag, lv)] = {
                            kk: vv for kk, vv in r.items() if kk != "dprime"}
                        if (gname, "%.1f" % lv, lam_tag) == ("intrain200", "0.3", "lam0"):
                            per_probe_primary[k] = pp
        finally:
            m.sem = orig_sem
            m.cfg.lam = e29.LAM
    eval_dt = time.perf_counter() - t0

    cov = {}
    for k in ARMS:
        tgt = [t for t in eval_targets["intrain200"] if t in slot_of_obs]
        cov[k] = {
            "unique_replayed_slots": len(arms[k]["replayed"]),
            "replay_fraction_of_written": round(len(arms[k]["replayed"]) / max(1, int(m.E.n)), 4),
            "eval_target_replayed": int(sum(1 for t in tgt if slot_of_obs.get(t) in arms[k]["replayed"])),
            "eval_target_written": len(tgt),
        }
    out = {
        "exp": "exp44_online_tag", "seed": seed, "smoke": smoke,
        "pre_registered": PRE_REG,
        "config": {"chunk": 500 if smoke else CHUNK,
                   "cycles_per_chunk": 2 if smoke else CYCLES_PER_CHUNK,
                   "n_r": N_R, "eval_every": 1 if smoke else EVAL_EVERY,
                   "grid": list(GRID), "n_train": n, "n_test": len(test_raw),
                   "n_written": int(m.E.n), "write_stats": wstats,
                   "encoder_seed": seed, "model_seed": seed,
                   "beta": e29.BETA, "L": e29.L, "lam_base": e29.LAM,
                   "corpus_files": files, "probe_rule": probes["rule"]},
        "timing_s": {"fit": round(fit_dt, 1), "encode": round(enc_dt, 1),
                     "write": round(write_dt, 1), "eval": round(eval_dt, 1),
                     "total": round(time.perf_counter() - t00, 1)},
        "coverage": cov,
        "validation_curves": {k: arms[k]["val_curve"] for k in ARMS},
        "theta_trajectory": arms["B_online"]["theta_traj"],
        "tried_scores": {"%g,%g" % k: v for k, v in tried.items()},
        "eval": evals,
        "per_probe_primary": per_probe_primary,
        "completed_at": e29.now_iso(),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(e29.sanitize(out), f, indent=1, ensure_ascii=False)
    prim = {k: evals[k].get("intrain200|lam0|deg0.3", {}).get("hit") for k in ARMS}
    print("seed %d done in %.0fs wrote=%d primary(semantic deg0.3 lam0): %s"
          % (seed, time.perf_counter() - t00, int(m.E.n), prim), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = args.out or os.path.join(ROOT, "results",
                                   "exp44_smoke.json" if args.smoke else "exp44_s%d.json" % args.seed)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    run_seed(args.seed, out, smoke=args.smoke)


if __name__ == "__main__":
    main()
