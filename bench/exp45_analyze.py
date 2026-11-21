# -*- coding: utf-8 -*-
"""exp45 聚合分析：β*(N) 在线学习 + 跨 encoder 迁移（D-066）。

输入 results/exp45_s{0,1,2}.json（--smoke 读 exp45_smoke.json）。
预注册门（工单原文±10%）：mean_ck(transferred) >= 0.9 x mean_ck(hand) 为正，
< 0.9 为负，均如实报（对齐检查点序数 k；逐 seed 再跨 seed 平均）。
次要：终评三设置 x 五组命中率表；beta*(N) 单调性描述（Spearman rho，跨 seed 池化）。
"""
import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS = ("fixed100", "transferred", "hand_best")
BLOCKS = ("intrain200|lam10|deg0.0", "intrain200|lam10|deg0.3",
          "dup200|lam10|deg0.0", "intrain200|lam05|deg0.3")


def load(files):
    out = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            out.append(json.load(fh))
    return out


def rate(block):
    return float(block["hit"]) / max(1, int(block["n"]))


def spearman(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.size < 3 or np.std(xs) == 0 or np.std(ys) == 0:
        return None
    rx = np.argsort(np.argsort(xs)).astype(np.float64)
    ry = np.argsort(np.argsort(ys)).astype(np.float64)
    rx = (rx - rx.mean()) / max(rx.std(), 1e-12)
    ry = (ry - ry.mean()) / max(ry.std(), 1e-12)
    return float((rx * ry).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        files = [os.path.join(ROOT, "results", "exp45_smoke.json")]
    else:
        files = [os.path.join(ROOT, "results", "exp45_s%d.json" % s) for s in (0, 1, 2)
                 if os.path.exists(os.path.join(ROOT, "results", "exp45_s%d.json" % s))]
    runs = load(files)
    if not runs:
        print("no exp45 results found")
        return

    # ---- 预注册门：checkpoint 级 transferred vs hand ----
    per_seed = []
    tr_all, hd_all = [], []
    for r in runs:
        g = r["summary_stats"].get("gate", {})
        per_seed.append({"seed": r["seed"], "verdict": g.get("verdict"),
                         "transferred_mean": g.get("transferred_mean"),
                         "hand_mean": g.get("hand_mean"),
                         "ratio": g.get("ratio"),
                         "n_checkpoints": g.get("n_checkpoints")})
        if g.get("transferred_mean") is not None and g.get("hand_mean") is not None:
            tr_all.append(g["transferred_mean"])
            hd_all.append(g["hand_mean"])
    gate = {}
    if tr_all:
        tr_m, hd_m = float(np.mean(tr_all)), float(np.mean(hd_all))
        ratio = tr_m / max(hd_m, 1e-9)
        gate = {"transferred_mean": tr_m, "hand_mean": hd_m, "ratio": ratio,
                "n_seeds": len(tr_all),
                "verdict": "POSITIVE" if ratio >= 0.9 else
                           "NEGATIVE (reported verbatim): transferred < 0.9 x hand",
                "per_seed": per_seed}
    else:
        gate = {"verdict": "NO_PAIRS", "per_seed": per_seed}

    # ---- 终评表 ----
    final = {}
    for tag in ("char", "sem"):
        final[tag] = {}
        for sname in SETTINGS:
            row = {}
            for blk in BLOCKS:
                vals = []
                for r in runs:
                    b = r.get("eval", {}).get(tag, {}).get(sname, {}).get("eval", {}).get(blk)
                    if b is not None:
                        vals.append(rate(b))
                row[blk] = float(np.mean(vals)) if vals else None
            final[tag][sname] = row

    # ---- beta*(N) 单调性（char 主链，跨 seed 池化）----
    mono = {}
    for tag, key in (("char", "beta_star_curve_char"), ("sem", None)):
        Ns, bs = [], []
        for r in runs:
            curve = r[key] if key else r.get("sem_curves", {}).get("beta_star")
            for v in (curve or []):
                Ns.append(v["N"])
                bs.append(v["beta"])
        if Ns:
            mono[tag] = {"spearman_rho_beta_vs_N": spearman(Ns, bs),
                         "n_checkpoints_pooled": len(Ns),
                         "beta_at_min_N": bs[int(np.argmin(Ns))],
                         "beta_at_max_N": bs[int(np.argmax(Ns))]}

    print("gate (preregistered, transferred vs hand at aligned checkpoints):")
    print("  verdict=%s ratio=%s n_seeds=%s" % (gate.get("verdict"), gate.get("ratio"), gate.get("n_seeds")))
    for ps in per_seed:
        print("  seed %s: %s ratio=%s (tr=%s hd=%s)" % (ps["seed"], ps["verdict"],
              None if ps["ratio"] is None else round(ps["ratio"], 3),
              None if ps["transferred_mean"] is None else round(ps["transferred_mean"], 4),
              None if ps["hand_mean"] is None else round(ps["hand_mean"], 4)))
    for tag in ("char", "sem"):
        print("final eval %s (hit rate):" % tag)
        for sname in SETTINGS:
            cells = " ".join("%s=%s" % (b, "n/a" if final[tag][sname][b] is None else "%.3f" % final[tag][sname][b])
                             for b in BLOCKS)
            print("  %-12s %s" % (sname, cells))
    for tag, m in mono.items():
        print("beta*(N) monotonicity %s: rho=%s n=%s (beta %s -> %s)" % (
            tag, m["spearman_rho_beta_vs_N"], m["n_checkpoints_pooled"],
            m["beta_at_min_N"], m["beta_at_max_N"]))

    out = {"exp": "exp45_analyze", "n_runs": len(runs),
           "seeds": [r["seed"] for r in runs],
           "gate_pre_registered": runs[0]["pre_registered"].get("gate"),
           "gate": gate, "final_eval_rate": final,
           "beta_star_monotonicity": mono,
           "beta_star_trajectory_char": [r["beta_star_curve_char"] for r in runs]}
    dest = os.path.join(ROOT, "results", "exp45_smoke_analyze.json" if a.smoke else "exp45_analyze.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print("saved", dest)


if __name__ == "__main__":
    main()
