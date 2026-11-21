# -*- coding: utf-8 -*-
"""exp52 聚合分析：G-EQ（N=2e5 dense-f32 vs postings 逐探针一致）+ G-S1@1e6
（β=100 deg25 recall>=0.90）+ G-S3'（写入线性）+ 接受性前提复核。

预注册（LEDGER 2026-11-12, prereg c7b3847 / 修订 899325a）：
- G-EQ: F_b100+T_b100 两 cell，hard_mismatch==0 且 max_d_a<1e-4。
- G-S1@1e6: β=100 deg25 rate>=0.90（两 cell，单 seed 如实注记）。
- G-S3': 全语义写入点 Pearson r>=0.95 —— 序列取 {exp43 dense 全语义点} ∪
  {exp52 eq@2e5 全语义写时}（LEDGER 口径 :268）；1e6 修订档（audit 路径）
  不入序列，enc/append 分解 + audit sc 包络单列透明报告。
- 接受性前提（1e6）: audit_rows 无 bad_dup(sc_max>=0.95) 无 g_fail(g<0.5)
  且 n_written==N（运行端已 exit 4 兜底，本脚本复核）。
输入 results/exp52_eq_N200000_{cell}_s0.json、exp52_N1000000_{cell}_s0.json、
（如有）exp52_lockstep_*.json；exp43 锚 results/exp43_inverted_index.json
（part1, 机1）与 exp43_inverted_index_scale{,_autodl}.json（part2）。
"""
import argparse
import glob
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THETA_DUP = 0.95
CELLS = ("F_b100", "T_b100")


def load(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8", "gbk"):
        try:
            return json.loads(raw.decode(enc))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("exp52_analyze.load", raw, 0, 1, "neither utf-8 nor gbk")


def pearson(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.size < 3 or xs.std() == 0 or ys.std() == 0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1000000)
    ap.add_argument("--eqN", type=int, default=200000)
    ap.add_argument("--res", default=None, help="override results dir (testing)")
    a = ap.parse_args()
    RES = a.res or os.path.join(ROOT, "results")

    report = {"exp": "exp52", "analyze_version": 1}
    missing = []

    # ---- lockstep（机1 冒烟，如存在）----
    ls = sorted(glob.glob(os.path.join(RES, "exp52_lockstep_*.json")))
    report["lockstep"] = [{"file": os.path.basename(f),
                           "gate_pass": load(f).get("gate_pass")} for f in ls]

    # ---- G-EQ（N=2e5, 两 cell）----
    eq = {}
    for cell in CELLS:
        f = os.path.join(RES, "exp52_eq_N%d_%s_s0.json" % (a.eqN, cell))
        if not os.path.exists(f):
            missing.append(os.path.basename(f))
            continue
        eq[cell] = load(f)
    if eq:
        hard = max(r["hard_mismatch"] for r in eq.values())
        max_da = max(r["max_d_a"] for r in eq.values())
        max_draw = max(r["max_d_raw"] for r in eq.values())
        geq_pass = (len(eq) == len(CELLS) and hard == 0 and max_da < 1e-4)
        report["G_EQ"] = {"cells": sorted(eq), "hard_mismatch": hard,
                          "max_d_a": max_da, "max_d_raw": max_draw,
                          "gate_pass": geq_pass,
                          "per_cell": {c: {k: eq[c][k] for k in
                                           ("n_written", "write_s", "s_per_write",
                                            "peak_rss_mb", "abst_d", "abst_p")}
                                       for c in eq}}
    else:
        geq_pass = None

    # ---- G-S1@1e6 + 接受性前提 ----
    sc = {}
    for cell in CELLS:
        f = os.path.join(RES, "exp52_N%d_%s_s0.json" % (a.N, cell))
        if not os.path.exists(f):
            missing.append(os.path.basename(f))
            continue
        sc[cell] = load(f)
    premise_ok = None
    if sc:
        per = {}
        for c, r in sc.items():
            aud = r.get("audit_rows") or []
            bad_dup = [x for x in aud if x.get("sc_max") is not None
                       and x["sc_max"] >= THETA_DUP]
            g_fail = [x for x in aud if x.get("g") is not None and x["g"] < 0.5]
            ok = (not bad_dup and not g_fail and int(r["n_written"]) == int(r["N"]))
            premise_ok = ok if premise_ok is None else (premise_ok and ok)
            per[c] = {"deg0_rate": r["deg0"]["rate"], "deg0": "%d/%d" % (r["deg0"]["hits"], r["deg0"]["of"]),
                      "deg25_rate": r["deg25"]["rate"], "deg25": "%d/%d" % (r["deg25"]["hits"], r["deg25"]["of"]),
                      "abstain_deg25": r["deg25"]["abstain"],
                      "s_per_query_deg25": r["deg25"]["s_per_query"],
                      "n_written": r["n_written"], "n_dup_rejected": r["n_dup_rejected"],
                      "audit_n": len(aud), "audit_bad_dup": len(bad_dup),
                      "audit_g_fail": len(g_fail), "audit_stride": r.get("audit_stride"),
                      "csr_mb": r["csr_mb"], "peak_rss_mb": r["peak_rss_mb"],
                      "s_per_write": r["s_per_write"],
                      "timing_s": r["timing"],
                      "premise_ok": ok}
        worst = min(per[c]["deg25_rate"] for c in per)
        gs1_pass = bool(len(sc) == len(CELLS) and worst >= 0.90)
        report["G_S1_1e6"] = {"threshold": 0.90, "worst_deg25_rate": worst,
                              "gate_pass": gs1_pass, "per_cell": per}
    else:
        gs1_pass = None

    # ---- G-S3' 写入线性（全语义 dense 点；1e6 audit 档不入序列）----
    series = []  # (N, s_per_write_s, source, machine)
    f1 = os.path.join(RES, "exp43_inverted_index.json")
    if os.path.exists(f1):
        j = load(f1)["part1"]
        n = int(j["N"])
        series.append((n, j["write_s"] / n, "exp43_part1", "machine1"))
    for f2, mach in (("exp43_inverted_index_scale.json", "machine1"),
                     ("exp43_inverted_index_scale_autodl.json", "autodl")):
        p2 = os.path.join(RES, f2)
        if os.path.exists(p2):
            j = load(p2)["part2"]
            series.append((int(j["N"]), float(j["write"]["dense_per_write_s"]),
                           "exp43_part2", mach))
    if "F_b100" in eq:
        series.append((a.eqN, float(eq["F_b100"]["s_per_write"]), "exp52_eq", "autodl"))
    if "T_b100" in eq:
        series.append((a.eqN, float(eq["T_b100"]["s_per_write"]), "exp52_eq_T", "autodl"))
    gs3 = {"points": [{"N": n, "s_per_write_s": round(spw, 6), "source": src, "machine": mach}
                      for n, spw, src, mach in sorted(series)],
           "note": ("LEDGER :268 series = {exp43 dense full-semantic points} ∪ "
                    "{exp52 eq@2e5 full-semantic write time}; 1e6 audit path "
                    "excluded (enc/append split reported per-cell above). "
                    "exp43 has only N=2e3 (machine1) and N=1e5 (machine1+autodl) "
                    "dense points; the 1e5 anchor for the pre-registered series "
                    "is the AutoDL run (same machine as the eq@2e5 point), the "
                    "machine1 run is reported as sensitivity.")}
    pts_pre, pts_alt = [], []
    for n, spw, src, mach in sorted(series):
        if src == "exp43_part1":
            pts_pre.append((n, spw)); pts_alt.append((n, spw))
        elif src == "exp43_part2":
            (pts_alt if mach == "machine1" else pts_pre).append((n, spw))
        elif src == "exp52_eq":
            pts_pre.append((n, spw))
    r_pre = pearson([p[0] for p in pts_pre], [p[1] for p in pts_pre]) if len(pts_pre) >= 3 else None
    r_alt = pearson([p[0] for p in pts_alt], [p[1] for p in pts_alt]) if len(pts_alt) >= 3 else None
    gs3["series_prereg"] = [list(p) for p in pts_pre]
    gs3["series_sensitivity_machine1_1e5"] = [list(p) for p in pts_alt]
    gs3["pearson_prereg"] = r_pre
    gs3["pearson_alt_machine1_anchor"] = r_alt
    gs3["gate"] = ("PASS" if r_pre is not None and r_pre >= 0.95
                   else ("FAIL" if r_pre is not None else "NOT_EVALUABLE_LT3_PTS"))
    report["G_S3p"] = gs3

    report["missing_files"] = missing
    report["verdict"] = {
        "G_EQ": geq_pass, "G_S1_1e6": gs1_pass, "G_S3p": gs3["gate"],
        "premise_ok": premise_ok,
        "overall": ("PASS" if (geq_pass and gs1_pass and premise_ok
                               and gs3["gate"] == "PASS") else
                    ("FAIL" if geq_pass is False or gs1_pass is False
                     or premise_ok is False or gs3["gate"] == "FAIL"
                     else "PENDING")),
    }
    out = os.path.join(RES, "exp52_analyze.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)
    print(json.dumps(report["verdict"]), flush=True)
    if report.get("G_S1_1e6"):
        print("G-S1@1e6 per cell:", json.dumps({c: (v["deg25"], v["deg0"]) for c, v in report["G_S1_1e6"]["per_cell"].items()}), flush=True)
    if missing:
        print("MISSING:", ", ".join(missing), flush=True)
    return 0 if report["verdict"]["overall"] == "PASS" else (4 if report["verdict"]["overall"] == "FAIL" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
