#!/usr/bin/env python3
"""check_claims.py —— claim <-> 引文 <-> 数字 三向比对（门禁 G5）。

A. 引文完整性：正文 [n] 与参考文献表双向闭合，无孤儿、无缺号。
B. 数字溯源：论文里每个定量主张，都用与论文相同的计算路径从 results/*.json 重算。
C. 措辞扫描：禁止超出证据强度的表述。
"""
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper", "PAPER_final.md")
RES = os.path.join(ROOT, "results")


def J(n):
    return json.load(open(os.path.join(RES, n), encoding="utf-8"))


def rows_of(name):
    d = J(name)
    return d.get("rows", d) if isinstance(d, dict) else d


def check_citations(text):
    body, _, tail = text.partition("## 参考文献")
    cited = set(int(x) for x in re.findall(r"\[(\d+)\]", body))
    listed = set(int(x) for x in re.findall(r"^\[(\d+)\]", tail, re.M))
    p = []
    if cited - listed:
        p.append("正文引用但参考文献表缺失：%s" % sorted(cited - listed))
    if listed - cited:
        p.append("参考文献表孤儿（未被正文引用）：%s" % sorted(listed - cited))
    if listed and listed != set(range(1, max(listed) + 1)):
        p.append("参考文献编号不连续（最大 %d）" % max(listed))
    return p, len(cited), len(listed)


def check_numbers():
    problems, checked = [], []

    def expect(label, got, want, tol=0.005):
        if got is None:
            problems.append("%s: 结果文件里找不到对应记录" % label)
            return
        ok = abs(float(got) - float(want)) <= tol
        checked.append((label, got, want, ok))
        if not ok:
            problems.append("%s: 结果=%s 论文=%s" % (label, got, want))

    d = rows_of("exp1_separation.json")

    def sep(ov, resid, L, key):
        v = [r[key] for r in d if r["overlap"] == ov and r["resid"] == resid and r["L"] == L]
        return v[0] if v else None

    def cos(ov, resid):
        v = [r["code_cos"] for r in d if r["overlap"] == ov and r["resid"] == resid and r["L"] == 0]
        return v[0] if v else None

    expect("表2 ov0.75 无残差/精确", sep(0.75, False, 0, "n0"), 3, 0)
    expect("表2 ov0.75 有残差L0/精确", sep(0.75, True, 0, "n0"), 30, 0)
    expect("表2 ov0.75 有残差L0/缺失20%", sep(0.75, True, 0, "n20"), 17, 0)
    expect("表2 ov0.75 有残差L0/缺失35%", sep(0.75, True, 0, "n35"), 6, 0)
    expect("表2 ov0.75 有残差L3/缺失20%", sep(0.75, True, 3, "n20"), 17, 0)
    expect("表2 ov0.90 无残差/精确", sep(0.90, False, 0, "n0"), 1, 0)
    expect("表2 ov0.90 有残差L0/精确", sep(0.90, True, 0, "n0"), 25, 0)
    expect("表2 ov0.90 有残差L3/精确", sep(0.90, True, 3, "n0"), 20, 0)
    expect("表2 码余弦 ov0.90 无残差", cos(0.90, False), 0.872, 0.002)
    expect("表2 码余弦 ov0.90 有残差", cos(0.90, True), 0.308, 0.002)
    expect("表2 码余弦 ov0.75 无残差", cos(0.75, False), 0.581, 0.002)
    expect("表2 码余弦 ov0.75 有残差", cos(0.75, True), 0.084, 0.002)

    e = J("exp2c_capacity_edge.json")

    def cap(N, resid, key):
        v = [r[key] for r in e if r["N"] == N and r["resid"] == resid]
        return v[0] if v else None

    expect("表3 N=16000 精确/无残差", cap(16000, False, "exact"), 20, 0)
    expect("表3 N=16000 精确/有残差", cap(16000, True, "exact"), 20, 0)
    expect("表3 N=16000 缺失25%/无残差", cap(16000, False, "degraded"), 8, 0)
    expect("表3 N=16000 缺失25%/有残差", cap(16000, True, "degraded"), 3, 0)
    expect("表3 N=8000 缺失25%/无残差", cap(8000, False, "degraded"), 19, 0)
    expect("表3 N=8000 缺失25%/有残差", cap(8000, True, "degraded"), 15, 0)

    # 表 4 v3：改用 exp15（20 种子）的配对统计为准；exp6_continual.json 保留为 3 种子历史数据
    import statistics as _st
    e15 = J("exp15_priority_power.json")

    def arm15(n, k):
        v = e15["arms"].get(n, {}).get(k)
        return round(float(_st.mean(v)), 2) if v else None

    expect("表4 blocked 旧(20种子)", arm15("blocked", "A"), 0.0, 0.02)
    expect("表4 blocked 新(20种子)", arm15("blocked", "B"), 21.9, 0.02)
    expect("表4 uniform 旧(20种子)", arm15("uniform", "A"), 13.85, 0.02)
    expect("表4 uniform 新(20种子)", arm15("uniform", "B"), 19.4, 0.02)
    expect("表4 priority 旧(20种子)", arm15("priority", "A"), 14.8, 0.02)
    expect("表4 priority 新(20种子)", arm15("priority", "B"), 18.75, 0.02)
    expect("表4 uniform-priority 配对差", round(e15["diff_mean"], 2), -0.95, 0.02)
    expect("表4 sign-flip p 在所报区间", e15["p_value"], 0.32, 0.05)

    # §4.3 残差代价规模曲线（exp14，8 种子；N=16000 由二号机产出，见 AUDIT 注记）
    e14 = J("exp14_resid_harmful.json")

    def r14(n, resid, k):
        v = [r[k] for r in e14 if r["N"] == n and r["resid"] == resid]
        return round(v[0], 3) if v else None

    expect("4.3 N=4000 无残差 deg25", r14(4000, False, "degraded_mean"), 0.997, 0.002)
    expect("4.3 N=4000 有残差 deg25", r14(4000, True, "degraded_mean"), 0.975, 0.002)
    expect("4.3 N=8000 无残差 deg25", r14(8000, False, "degraded_mean"), 0.938, 0.002)
    expect("4.3 N=8000 有残差 deg25", r14(8000, True, "degraded_mean"), 0.822, 0.002)
    expect("4.3 N=16000 无残差 deg25", r14(16000, False, "degraded_mean"), 0.569, 0.002)
    expect("4.3 N=16000 有残差 deg25", r14(16000, True, "degraded_mean"), 0.291, 0.002)
    expect("4.3 N=16000 有残差 精确", r14(16000, True, "exact_mean"), 0.972, 0.002)

    b = J("exp6b_buffer.json")

    def buf(n, a, k):
        v = [r[k] for r in b if r["buf"] == n and r["arm"] == a]
        return v[0] if v else None

    expect("表5 buf6 uniform 旧", buf(6, "uniform", "A"), 6, 0)
    expect("表5 buf6 priority 旧", buf(6, "priority", "A"), 3, 0)
    expect("表5 buf24 uniform 旧", buf(24, "uniform", "A"), 6, 0)
    expect("表5 buf24 priority 旧", buf(24, "priority", "A"), 8, 0)

    p4 = J("exp4c_capacity_pressure.json")

    def p4v(a, k):
        v = [r[k] for r in p4 if r["arm"] == a]
        return v[0] if v else None

    expect("表6 200条 blocked 旧(后)", p4v("noreplay", "A_after"), 43, 0)
    expect("表6 200条 priority 旧(后)", p4v("priority", "A_after"), 36, 0)
    expect("表6 200条 uniform 旧(后)", p4v("uniform", "A_after"), 108, 0)

    e5 = J("exp5_paired.json")

    def e5v(a, k):
        v = [r[k] for r in e5 if r["arm"] == a]
        return v[0] if v else None

    expect("表6 配对 blocked 旧(前)", e5v("blocked", "A_before"), 20, 0)
    expect("表6 配对 blocked 旧(后)", e5v("blocked", "A_after"), 20, 0)

    w = J("exp7_wm.json")
    for C, want in ((1, 1.00), (2, 2.00), (3, 2.67), (4, 3.67), (5, 4.83), (6, 5.17), (8, 7.33)):
        v = [r["kept"] for r in w if r["C"] == C]
        expect("表7 C=%d" % C, v[0] if v else None, want, 0.006)

    m = J("exp8_meta.json")
    curve = {round(r["coverage"], 2): r["risk"] for r in m["curve"]}
    # 表8：直接用结果文件里的 11 个操作点，只校验单调性 + 端点
    pts = sorted((r["coverage"], r["risk"]) for r in m["curve"])
    # 正确方向：覆盖率上升（接受更难的查询）-> 风险必须**不下降**
    for (c1, r1), (c2, r2) in zip(pts, pts[1:]):
        if r2 < r1 - 1e-9:
            problems.append("表8 风险非单调：覆盖率 %.3f->%.3f 时风险 %.3f->%.3f（风险不应随覆盖率下降）"
                            % (c1, c2, r1, r2))
    expect("表8 最低覆盖率处风险", pts[0][1], 0.0, 0.002)
    expect("表8 全覆盖率处风险", pts[-1][1], 0.500, 0.002)
    expect("表8 操作点数", len(pts), 11, 0)

    expect("弃权 d'", m.get("d_prime"), 7.12, 0.02)
    expect("弃权得分 meta", m.get("score_meta"), 27.3, 0.5)
    expect("弃权得分 always", m.get("score_always"), 0.0, 0.01)


    # 表 9：多种子统计（12 seeds）
    st = J("exp10_stats.json")
    def stv(k, f):
        v = st.get(k, {}).get(f)
        return v
    expect("表9 ov0.90 无残差 mean", stv("ov0.90 无残差/精确", "mean"), 1.08, 0.02)
    expect("表9 ov0.90 有残差 mean", stv("ov0.90 有残差/精确", "mean"), 24.42, 0.02)
    expect("表9 ov0.90 有残差 CI", stv("ov0.90 有残差/精确", "ci95"), 1.61, 0.02)
    expect("表9 ov0.75 有残差 mean", stv("ov0.75 有残差/精确", "mean"), 29.83, 0.02)
    expect("表9 ov0.75 缺失20% mean", stv("ov0.75 有残差/缺失20%", "mean"), 17.25, 0.02)
    expect("表9 ov0.75+L3 缺失20% mean", stv("ov0.75 有残差L3/缺失20%", "mean"), 15.33, 0.02)
    expect("表9 种子数", stv("ov0.90 有残差/精确", "n"), 12, 0)


    # 表 10：双读出（exp11）—— 此前完全未被校验（对抗评审 P1-1）
    ro = J("exp11_readout.json")
    def ro_(noise, k):
        v = [r[k] for r in ro if abs(r["noise"] - noise) < 1e-9]
        return v[0] if v else None
    expect("表10 缺失60% 稀疏读出", ro_(0.60, "sparse_only"), 39, 0)
    expect("表10 缺失60% 稠密单步", ro_(0.60, "dense_once"), 50, 0)
    expect("表10 缺失45% 稀疏读出", ro_(0.45, "sparse_only"), 48, 0)
    expect("表10 缺失70% 稠密读出", ro_(0.70, "dense_once"), 44, 0)


    # 表 11：生成器敏感性（exp12）
    g12 = J("exp12_generator.json")
    def g12v(share, side, key):
        v = [r[side][key] for r in g12 if abs(r["share"] - share) < 1e-6]
        return v[0] if v else None
    expect("表11 share0.90 输入余弦", g12v(0.90, "resid_off", "code_cos"), 0.872, 0.002)
    expect("表11 share0.90 无残差召回", g12v(0.90, "resid_off", "recall"), 1, 0)
    expect("表11 share0.90 残差召回", g12v(0.90, "resid_on", "recall"), 25, 0)
    expect("表11 share0.90 无残差写入", g12v(0.90, "resid_off", "written"), 18, 0)
    expect("表11 share0.85 残差召回", g12v(0.85, "resid_on", "recall"), 30, 0)
    expect("表11 share0.95 残差召回", g12v(0.95, "resid_on", "recall"), 0, 0)
    expect("表11 share0.95 无残差写入", g12v(0.95, "resid_off", "written"), 6, 0)


    # 表 12：置信度信号（exp13）
    cf = J("exp13_confidence.json")
    expect("表12 corr(m1,correct)", cf["corr_m1"], -0.0697, 0.005)
    expect("表12 AUC(m1)", cf["auc_m1"], 0.458, 0.005)
    expect("表12 AUC(raw)", cf["auc_raw"], 1.000, 0.005)

    # 第 7 节 L1 先导实测（exp16，4 种子，二号机）
    e16 = J("exp16_l1_scale.json")

    def a16(resid, key):
        v = [r[key] for r in e16 if r["resid"] == resid]
        return v[0] if v else None

    expect("§7 L1 无残差/精确", a16(False, "exact_mean"), 1.000, 0.001)
    expect("§7 L1 无残差/deg25", a16(False, "deg25_mean"), 0.250, 0.005)
    expect("§7 L1 有残差/精确", a16(True, "exact_mean"), 0.994, 0.002)
    expect("§7 L1 有残差/deg25", a16(True, "deg25_mean"), 0.031, 0.005)
    expect("§7 L1 写入耗时下界 s", a16(True, "write_s"), 448, 6)
    expect("§7 L1 写入耗时上界 s", a16(False, "write_s"), 453, 6)
    expect("§7 L1 单查 ms", a16(False, "query_ms"), 2457, 10)

    # 附录 E 虚警理论（exp18）
    e18 = rows_of("exp18_false_alarm.json")

    def ks18(dh, N):
        v = [r["ks_single"] for r in e18 if r.get("model") != "encoder" and r["dh"] == dh and r["N"] == N]
        return v[0] if v else None

    expect("附E KS dh1024 N2000", ks18(1024, 2000), 0.0140, 0.001)
    expect("附E KS dh1024 N16000", ks18(1024, 16000), 0.0072, 0.001)
    expect("附E KS dh1024 N200000", ks18(1024, 200000), 0.0023, 0.001)
    expect("附E KS dh4096 N2000", ks18(4096, 2000), 0.0030, 0.001)
    expect("附E KS dh4096 N16000", ks18(4096, 16000), 0.0003, 0.001)
    expect("附E KS dh4096 N200000", ks18(4096, 200000), 0.0003, 0.001)
    enc18 = [r for r in e18 if r.get("model") == "encoder"]
    e1024 = [r for r in enc18 if r["dh"] == 1024][0]
    e4096 = [r for r in enc18 if r["dh"] == 4096][0]
    expect("附E margin dh1024", e1024["margin"], 0.81, 0.01)
    expect("附E margin dh4096", e4096["margin"], 0.88, 0.01)
    expect("附E 容量 dh1024", float(e1024["capacity_score05"]), 1.3e11, 1.3e10)
    expect("附E 容量 dh4096", float(e4096["capacity_score05"]), 3.2e16, 3.2e15)

    # ---- 第三轮修订（exp19g/exp20/exp20b/exp22p/exp22q/exp23 系列/exp24）----
    e19g = J("exp19g_beta_scale.json")

    def g19(beta, level, seed=None):
        v = [r["top1"] for r in e19g if r["beta"] == beta and abs(r["level"] - level) < 1e-9 and (seed is None or r["seed"] == seed)]
        return sum(v) / len(v) if v else None

    expect("4.12 beta=20 deg25(seed0)", g19(20, 0.25, 0), 8, 0)
    expect("4.12 beta=20 deg35(seed0)", g19(20, 0.35, 0), 0, 0)
    expect("4.12 beta=100 deg25 mean", g19(100, 0.25), 40, 0)
    expect("4.12 beta=500 deg25 mean", g19(500, 0.25), 40, 0)
    expect("4.12 beta=2500 deg25 mean", g19(2500, 0.25), 40, 0)

    e20 = J("exp20_resid_beta_l1.json")
    b100l1 = [r["top1"] for r in e20 if r["beta"] == 100]
    expect("4.12 L1 beta=100 min cell", min(b100l1), 40, 0)
    a20f = [r["top1"] for r in e20 if r["beta"] == 20 and abs(r["level"] - 0.25) < 1e-9 and not r["resid"]]
    a20t = [r["top1"] for r in e20 if r["beta"] == 20 and abs(r["level"] - 0.25) < 1e-9 and r["resid"]]
    expect("7 L1 beta=20 anchor F deg25", a20f[0], 8, 0)
    expect("7 L1 beta=20 anchor T deg25", a20t[0], 2, 0)

    e20b = J("exp20b_resid_beta_l0.json")
    b100l0 = [r for r in e20b if r["beta"] == 100]
    expect("4.3 L0 beta=100 exact min", min(r["exact_mean"] for r in b100l0), 1.0, 0.001)
    expect("4.3 L0 beta=100 deg25 min", min(r["deg_mean"] for r in b100l0), 1.0, 0.001)

    def b20a(N, resid, key):
        v = [r[key] for r in e20b if r["beta"] == 20 and r["N"] == N and r["resid"] == resid]
        return v[0] if v else None

    expect("4.3 b20 anchor N8000 F deg25", b20a(8000, False, "deg_mean"), 0.900, 0.001)
    expect("4.3 b20 anchor N16000 F deg25", b20a(16000, False, "deg_mean"), 0.575, 0.001)
    expect("4.3 b20 anchor N16000 T deg25", b20a(16000, True, "deg_mean"), 0.275, 0.001)

    e22p = J("exp22p_correct_judge.json")
    expect("4.13.2 lam.6 deg.3", e22p["lam06_deg03"]["hit"], 200, 0)
    expect("4.13.2 lam.6 deg.5", e22p["lam06_deg05"]["hit"], 197, 0)
    expect("4.13.2 lam1.0 deg.3", e22p["lam10_deg03"]["hit"], 200, 0)
    expect("4.13.2 lam1.0 deg.5", e22p["lam10_deg05"]["hit"], 197, 0)

    e22q = J("exp22q_develop_recheck.json")

    def q22(arm, level):
        a = [x for x in e22q["arms"] if x["arm"] == arm][0]
        r = [x for x in a["rows"] if abs(x["level"] - level) < 1e-9][0]
        return r["hit_new"]

    expect("9.9 A0 deg.3", q22("A0_theta0", 0.3), 120, 0)
    expect("9.9 A0 deg.5", q22("A0_theta0", 0.5), 111, 0)
    expect("9.9 A1 develop deg.3", q22("A1_develop", 0.3), 0, 0)
    expect("9.9 A1 synth_written", [x for x in e22q["arms"] if x["arm"] == "A1_develop"][0]["synth_written"], 2000, 0)

    e23c = J("exp23c_data_scale.json")
    expect("4.13.3 full deg.3 (v2)", e23c["S256_full"]["retrieval"][0]["hit"], 199, 0)
    expect("4.13.3 full deg.5 (v2)", e23c["S256_full"]["retrieval"][1]["hit"], 181, 0)
    expect("4.13.3 2k deg.3 (v2)", e23c["S256_2k"]["retrieval"][0]["hit"], 123, 0)
    expect("4.13.3 full syn cos (v2)", e23c["S256_full"]["syn_mean_cos"], 0.4656, 0.001)
    expect("4.13.3 dis scale t (v2)", e23c["VERDICT"]["paired_t"], 2.754, 0.005)

    e23e = J("exp23e_emoji_verdict.json")

    def expect_s(label, got, want):
        ok = (got == want)
        checked.append((label, got, want, ok))
        if not ok:
            problems.append("%s: 结果=%s 论文=%s" % (label, got, want))

    expect_s("4.13.3 randfallback deg.3", e23e["S256_2k_randfallback"]["hit@0.3"], "200/200")
    expect_s("4.13.3 randfallback deg.5", e23e["S256_2k_randfallback"]["hit@0.5"], "200/200")

    e23f = J("exp23f_embed_stats.json")
    expect("4.13.3 near Delta(full-R256)", e23f["PAIRED"]["near_full_vs_R256"][0], 0.1088, 0.001)
    expect("4.13.3 near t(full-R256)", e23f["PAIRED"]["near_full_vs_R256"][1], 2.261, 0.005)
    expect("4.13.3 dis Delta(full-2k)", e23f["PAIRED"]["dis_full_vs_2k"][0], 0.0345, 0.001)
    expect("4.13.3 dis t(full-2k)", e23f["PAIRED"]["dis_full_vs_2k"][1], 2.754, 0.005)

    # ---- v2 工件值级断言（2026-11-08 工单#06 补钉：钉 11-08 v2 判决工件防漂移；
    #      数字以工件实测为准，与《回音-手册审读-20261108.md》§一核对无出入）----
    e21v2 = J("exp21_real_corpus.json")
    c21 = [r for r in e21v2["rows"] if r["task"] == "completion"]
    expect("4.13 v2 exp21 completion cells", len(c21), 12, 0)
    expect("4.13 v2 exp21 top1 max", max(r["top1"] for r in c21), 0, 0)
    expect("4.13 v2 exp21 probes n min", min(r["n"] for r in c21), 200, 0)
    expect("4.13 v2 exp21 n_train", e21v2["meta"]["n_train"], 9033, 0)
    e23v2 = J("exp23_semantic_encoder.json")
    expect("4.13.3 v2 exp23sem prediction_holds", 1 if e23v2["verdict"]["prediction_holds"] else 0, 0, 0)
    expect("4.13.3 v2 exp23sem near t", e23v2["verdict"]["near"]["t"], 0.081, 0.0005)
    expect("4.13.3 v2 exp23sem near n", e23v2["verdict"]["near"]["n"], 12, 0)
    expect("4.13.3 v2 exp23sem disjoint t", e23v2["verdict"]["disjoint"]["t"], 0.918, 0.0005)
    expect("4.13.3 v2 exp23sem disjoint n", e23v2["verdict"]["disjoint"]["n"], 14, 0)
    e22o = J("exp22o_rerun200.json")
    expect("4.13.2 spy hit (artifact live)", e22o["hit"], 0, 0)
    e29 = J("exp29_text_frontend_v2_m2final.json")
    e29n = e29["full"]["arms"]["NEW256"]
    expect("4.13 full-store written", e29n["n_written"], 92760, 0)
    expect("4.13 full-store deg0", e29n["groups"]["intrain200"]["0.0"]["hit"], 156, 0)
    expect("4.13 full-store deg0.3", e29n["groups"]["intrain200"]["0.3"]["hit"], 129, 0)
    expect("4.13 full-store deg0.5", e29n["groups"]["intrain200"]["0.5"]["hit"], 75, 0)
    expect("4.13.3 v2 exp23sem S256 syn cos", e23v2["S256"]["syn_mean_cos"], 0.3598, 0.0005)
    expect("4.13.3 v2 exp23sem R256 syn cos", e23v2["R256"]["syn_mean_cos"], 0.3568, 0.0005)
    e29v2 = J("exp29_text_frontend_v2_m2final.json")
    q29 = e29v2["f3"]["comparisons"]["query_paired_NEW_vs_R256"]["pooled"]["exact_hit(1-0)"]
    expect("4.13.3 v2 exp29m2 pooled mean_diff", q29["mean_diff"], -0.01, 0.0005)
    expect("4.13.3 v2 exp29m2 pooled t", q29["t"], -0.457, 0.0005)
    expect("4.13.3 v2 exp29m2 pooled p", q29["p_two_sided"], 0.648, 0.0005)
    expect("4.13.3 v2 exp29m2 pooled ci lo", q29["ci95"][0], -0.053, 0.0005)
    expect("4.13.3 v2 exp29m2 pooled ci hi", q29["ci95"][1], 0.033, 0.0005)
    z29 = e29v2["f3"]["comparisons"]["zero_shared_synpair_head_to_head"]
    expect("4.13.3 v2 exp29m2 h2h n_pairs", z29["n_pairs"], 14, 0)
    expect("4.13.3 v2 exp29m2 h2h input NEW", z29["summary"]["NEW_wins_input"], 14, 0)
    expect("4.13.3 v2 exp29m2 h2h input R256", z29["summary"]["R256_wins_input"], 0, 0)
    expect("4.13.3 v2 exp29m2 h2h code NEW", z29["summary"]["NEW_wins_code"], 10, 0)
    expect("4.13.3 v2 exp29m2 h2h code R256", z29["summary"]["R256_wins_code"], 4, 0)
    b29 = e29v2["f3"]["baselines_posthoc"]["retrieval_by_overlap_layer"]["bm25"]["all"]
    expect("4.13.3 v2 exp29m2 bm25 hit1", b29["hit1"], 41, 0)
    expect("4.13.3 v2 exp29m2 bm25 hit5", b29["hit5"], 54, 0)

    e24 = J("exp24_b2_disambiguation.json")
    expect("4.13.4 no cue", e24["arms"]["B_no_ctx"]["hit"], 0, 0)
    expect("4.13.4 exact cue", e24["arms"]["A_slow_cue"]["hit"], 40, 0)
    expect("4.13.4 tau cue", e24["arms"]["A_tau"]["hit"], 38, 0)
    expect("4.13.4 partial cue", e24["arms"]["A_partial_cue"]["hit"], 6, 0)
    expect("4.13.4 wrong cue cued rate", e24["arms"]["C_wrong_cue_slow"]["cued_rate"], 1.0, 0.001)
    expect("4.13.4 wrong cue target rate", e24["arms"]["C_wrong_cue_slow"]["target_rate"], 0.0, 0.001)
    expect("4.13.4 dup code cos", e24["diag"]["code_pairwise_cos"]["mean"], 1.0, 0.001)


    # ---- 4.14 时序链（exp27/exp28，第四轮）----
    e27 = J("exp27_temporal_chain.json")
    r27 = e27["rows"]
    def s27(k, f):
        return r27[k][f]
    expect("4.14.1 b100deg0 self", s27("b100_deg0", "self_hit1"), 200, 0)
    expect("4.14.1 b100deg0 next5", s27("b100_deg0", "next_hit5"), 1, 0)
    expect("4.14.1 b100deg0 ctrlmed", s27("b100_deg0", "ctrl_rank_med"), 3169.5, 1)
    expect("4.14.1 b500deg30 self", s27("b500_deg30", "self_hit1"), 200, 0)
    expect("4.14.1 b500deg30 nextmed", s27("b500_deg30", "next_rank_med"), 2665.5, 0.1)

    e28 = J("exp28_btsp_chain.json")["rows"]
    expect("4.14.2 w0deg0 next1", e28["w0_deg0"]["next_hit1"], 0, 0)
    expect("4.14.2 w0.5deg30 next1", e28["w0.5_deg30"]["next_hit1"], 115, 0)
    expect("4.14.2 w1deg0 next1", e28["w1_deg0"]["next_hit1"], 200, 0)
    expect("4.14.2 w1deg0 self", e28["w1_deg0"]["self_hit1"], 200, 0)
    expect("4.14.2 w1deg0 joint", e28["w1_deg0"]["joint_self1_next1"], 200, 0)
    expect("4.14.2 w1deg30 next1", e28["w1_deg30"]["next_hit1"], 200, 0)
    expect("4.14.2 w2deg0 next1", e28["w2_deg0"]["next_hit1"], 200, 0)
    expect("4.14.2 w2deg30 next1", e28["w2_deg30"]["next_hit1"], 200, 0)
    expect("4.14.2 w2deg30 self", e28["w2_deg30"]["self_hit1"], 200, 0)


    # ---- 4.15 校准（exp35，I 区 C3-prime）----
    e35 = J("exp35_calibration.json")
    expect("4.15 A_raw ece", e35["feature_sets"]["A_raw"]["ece_eval"], 0.0684, 0.0005)
    expect("4.15 A_raw auc", e35["feature_sets"]["A_raw"]["auc_eval"], 0.8713, 0.0005)
    expect("4.15 C_cons ece", e35["feature_sets"]["C_cons"]["ece_eval"], 0.0509, 0.0005)
    expect("4.15 C_cons auc", e35["feature_sets"]["C_cons"]["auc_eval"], 0.914, 0.0005)
    expect("4.15 hard auc", e35["g3_hard_auc"], 0.8831, 0.0005)
    expect("4.15 para hit", e35["tags"]["para"]["hit_rate"], 0.36, 0.0005)
    expect("4.15 n_probes", e35["n"], 550, 0)

    # ---- 4.15.1 弃权工作点与 G-4（exp35b）----
    e35b = J("exp35b_operating_points.json")
    expect("4.15.1 ece raw1", e35b["ece_raw_eval"], 0.2830, 0.0005)
    expect("4.15.1 ece cal", e35b["ece_cal_eval"], 0.0466, 0.0005)
    expect("4.15.1 current cov", e35b["current"]["cov"], 0.9091, 0.0005)
    expect("4.15.1 current risk", e35b["current"]["risk"], 0.424, 0.0005)
    expect("4.15.1 current cost", e35b["current"]["cost"], 0.4082, 0.0005)
    expect("4.15.1 best th", e35b["best_cal"]["th"], 0.7681, 0.0005)
    expect("4.15.1 best cov", e35b["best_cal"]["cov"], 0.4182, 0.0005)
    expect("4.15.1 best risk", e35b["best_cal"]["risk"], 0.087, 0.0005)
    expect("4.15.1 best cost", e35b["best_cal"]["cost"], 0.1818, 0.0005)
    expect("4.15.1 g4 pass", 1 if e35b["g4"]["passed"] else 0, 0, 0)
    expect("4.15.1 g4 max delta", max(d["delta"] for d in e35b["g4"]["details"]), 0.0431, 0.0005)
    expect("4.15.1 k7a cost_not_worse", 1 if e35b["k7a"]["cost_not_worse"] else 0, 1, 0)
    expect("4.15.1 k7a ece line", 1 if e35b["k7a"]["ece_below_005"] else 0, 1, 0)

    # ---- 4.16 RIF 因果检验（exp34，预登记否定结果）----
    a34 = J("exp34_analyze.json")
    d34 = a34["cells"]["david"]
    expect("4.16 deficit_sup_on", d34["deficit_sup_on"]["mean"], -0.1567, 0.0005)
    expect("4.16 deficit ci lo", d34["deficit_sup_on"]["ci"][0], -0.1917, 0.0005)
    expect("4.16 deficit ci hi", d34["deficit_sup_on"]["ci"][1], -0.1233, 0.0005)
    expect("4.16 placebo cos_off", d34["deficit_cos_off"]["mean"], -0.15, 0.0005)
    expect("4.16 paired delta", d34["delta_cos_on_minus_off"]["mean"], -0.0067, 0.0005)
    expect("4.16 delta ci lo", d34["delta_cos_on_minus_off"]["ci"][0], -0.05, 0.0005)
    expect("4.16 delta ci hi", d34["delta_cos_on_minus_off"]["ci"][1], 0.0367, 0.0005)
    expect("4.16 hit_rp_sup_on", d34["hit_rp_sup_on"]["mean"], 1.0, 0)
    expect("4.16 hit_nrp_on", d34["hit_nrp_on"]["mean"], 0.8433, 0.0005)
    expect("4.16 hit_nrp_off", d34["hit_nrp_off"]["mean"], 0.85, 0.0005)
    expect("4.16 recovered", d34["recovered"]["mean"], 1.0, 0)
    expect("4.16 G-A verdict", 1 if a34["gates"]["david"]["G-A"] == "FAIL" else 0, 1, 0)
    expect("4.16 G-B verdict", 1 if a34["gates"]["david"]["G-B"] == "PASS" else 0, 1, 0)
    expect("4.16 G-C verdict", 1 if a34["G-C"] == "NEGATIVE-HONEST" else 0, 1, 0)
    expect("4.16 syn_o0.9 floor recovered", a34["cells"]["syn_o0.9"]["recovered"]["mean"], 0.3317, 0.0005)
    # ---- 4.17 存储级抑制三难（exp34b，预登记共主判据）----
    a34b = J("exp34b_analyze.json")
    expect("4.17 n_rows", a34b["n_rows"], 48, 0)
    expect("4.17 GA deficit", a34b["G-A"]["deficit_mean"], 0.1583, 0.0005)
    expect("4.17 GA ci lo", a34b["G-A"]["ci95"][0], 0.1067, 0.0005)
    expect("4.17 GA ci hi", a34b["G-A"]["ci95"][1], 0.21, 0.0005)
    expect("4.17 GA placebo", a34b["G-A"]["placebo_deficit_mean"], -0.15, 0.0005)
    expect("4.17 GA verdict", 1 if a34b["G-A"]["verdict"] == "PASS" else 0, 1, 0)
    expect("4.17 GS1 collapse", a34b["G-S1"]["collapse_mean"], 0.1567, 0.0005)
    expect("4.17 GS1 verdict", 1 if a34b["G-S1"]["verdict"] == "PASS" else 0, 1, 0)
    expect("4.17 GS2 verdict", 1 if a34b["G-S2"]["verdict"] == "FAIL" else 0, 1, 0)
    expect("4.17 GS2 hub soff", a34b["G-S2"]["hub_share"]["soff"], 0.09, 0.0005)
    expect("4.17 GS2 hub son050", a34b["G-S2"]["hub_share"]["son050"], 0.1021, 0.0005)
    expect("4.17 GB fullrestore", 1 if a34b["G-B"]["pass_fullrestore"] else 0, 1, 0)
    expect("4.17 GB hp_all_rec", a34b["G-B"]["hp_all_rec_mean"], 0.88, 0.0005)
    expect("4.17 GB gap", a34b["G-B"]["gap_to_soff_hp_mean"], 0.0, 0.0005)
    expect("4.17 dose deficit r015", a34b["dose_response"]["son015"]["deficit_mean"], 0.1583, 0.0005)
    expect("4.17 dose deficit r030", a34b["dose_response"]["son030"]["deficit_mean"], 0.29, 0.0005)
    expect("4.17 dose deficit r050", a34b["dose_response"]["son050"]["deficit_mean"], 0.705, 0.0005)
    expect("4.17 dose hitp r015", a34b["dose_response"]["son015"]["hit_p_mean"], 0.8, 0.0005)
    expect("4.17 dose hitp r050", a34b["dose_response"]["son050"]["hit_p_mean"], 0.45, 0.0005)
    expect("4.17 monotone deficit", 1 if a34b["dose_monotone_deficit"] else 0, 1, 0)
    expect("4.17 monotone hitp", 1 if a34b["dose_monotone_hitp_collapse"] else 0, 1, 0)
    # ---- 4.18 情境检索剂量标定（exp36d，第五轮）----
    e36d = J("exp36d_ctx_energy_v2.json")
    def dv36d(arm, k, f):
        return [r[f] for r in e36d["dev"][arm] if abs(r["kappa"] - k) < 1e-9][0]
    expect("4.18 dev s k05 P1", dv36d("single", 0.05, "P1"), 0.7647, 0.0005)
    expect("4.18 dev s k05 P4", dv36d("single", 0.05, "P4"), 0.9216, 0.0005)
    expect("4.18 dev s k10 P4", dv36d("single", 0.1, "P4"), 0.3725, 0.0005)
    expect("4.18 dev m k05 P4", dv36d("multi", 0.05, "P4"), 0.3529, 0.0005)
    expect("4.18 dev m k08 P4", dv36d("multi", 0.08, "P4"), 0.0, 0.0005)
    expect("4.18 calib single", e36d["calib"]["single"]["kappa_star"], 0.05, 1e-9)
    expect("4.18 calib multi", e36d["calib"]["multi"]["kappa_star"], 0.03, 1e-9)
    expect("4.18 GT verdict", 1 if e36d["gates"]["G_T_era_anchor"]["verdict"] == "FAIL" else 0, 1, 0)
    expect("4.18 GT mean delta", e36d["gates"]["G_T_era_anchor"]["mean_delta"], -0.5878, 0.0005)
    expect("4.18 GG verdict", 1 if e36d["gates"]["G_G_ceiling_noharm"]["verdict"] == "PASS" else 0, 1, 0)
    expect("4.18 P2 verdict", 1 if e36d["gates"]["CTRL_P2_shuffle"]["verdict"] == "PASS" else 0, 1, 0)
    ts = e36d["test"]["single"]; tm = e36d["test"]["multi"]
    expect("4.18 test s P1 s0", ts[0]["rates"]["P1"], 0.8065, 0.0005)
    expect("4.18 test s P1 s2", ts[2]["rates"]["P1"], 0.7742, 0.0005)
    expect("4.18 test s P4 min", min(x["rates"]["P4"] for x in ts), 0.9462, 0.0005)
    expect("4.18 test m P1 s0", tm[0]["rates"]["P1"], 0.2473, 0.0005)
    expect("4.18 test m P1 s2", tm[2]["rates"]["P1"], 0.1828, 0.0005)
    expect("4.18 test m P4 min", min(x["rates"]["P4"] for x in tm), 0.9677, 0.0005)
    # ---- 4.19 外部基线（exp38，第六轮）----
    e38 = J("exp38_baselines.json")
    expect("4.19 hopf written", e38["rows"]["hopf"]["written"], 9033, 0)
    expect("4.19 sdm written", e38["rows"]["sdm"]["written"], 9033, 0)
    expect("4.19 hcm written", e38["rows"]["hcm"]["written"], 7340, 0)
    expect("4.19 hopf deg03", e38["rows"]["hopf"]["deg0.3"], 2, 0)
    expect("4.19 hopf deg05", e38["rows"]["hopf"]["deg0.5"], 2, 0)
    expect("4.19 sdm deg03", e38["rows"]["sdm"]["deg0.3"], 0, 0)
    expect("4.19 hcm deg03", e38["rows"]["hcm"]["deg0.3"], 48, 0)
    expect("4.19 hcm deg05", e38["rows"]["hcm"]["deg0.5"], 45, 0)
    expect("4.19 hopf self", e38["rows"]["hopf"]["self_hit1"], 0, 0)
    expect("4.19 sdm self", e38["rows"]["sdm"]["self_hit1"], 0, 0)
    expect("4.19 hcm self", e38["rows"]["hcm"]["self_hit1"], 175, 0)
    expect("4.19 G-B1 delta", e38["gates"]["G_B1"]["delta"], 0.23, 0.0005)
    expect("4.19 G-B1 pass", 1 if e38["gates"]["G_B1"]["pass_"] else 0, 1, 0)
    expect("4.19 G-B2 pass", 1 if e38["gates"]["G_B2"]["pass_"] else 0, 1, 0)
    expect("4.19 G-B3 max", e38["gates"]["G_B3"]["max_rate"], 0.0114, 0.0005)
    expect("4.19 G-B3 pass", 1 if e38["gates"]["G_B3"]["pass_"] else 0, 1, 0)
    expect("4.19 hopf nextmed", e38["rows"]["hopf"]["next_rank_med"], 2627.5, 0.0005)
    # ---- 4.8 M7 判决收紧（exp41，第六轮）----
    e41 = J("exp41_m7_gate.json")
    expect("4.8 pa gate dedup", e41["part_a"][0][0]["writes"], 1, 0)
    expect("4.8 pa gate nodup", e41["part_a"][0][1]["writes"], 1, 0)
    expect("4.8 pa const dedup", e41["part_a"][0][2]["writes"], 1, 0)
    expect("4.8 pa const nodup", e41["part_a"][0][3]["writes"], 10, 0)
    expect("4.8 FS mean", e41["gates"]["G_SEL"]["fs_mean"], 0.05, 0.0005)
    expect("4.8 G-SEL pass", 1 if e41["gates"]["G_SEL"]["pass_"] else 0, 0, 0)
    expect("4.8 G-RED pass", 1 if e41["gates"]["G_RED"]["pass_"] else 0, 1, 0)
    expect_s("4.8 exp41 verdict", e41["verdict"], "REDUNDANT_WITH_THETA_DUP")
    # ---- 4.8 M8 遗忘收益（exp42，第六轮）----
    e42 = J("exp42_forget_benefit.json")
    expect("4.8 imp50 uniform rw", e42["agg"]["uniform/imp_tau50"]["recall_rewarded"], 0.3333, 0.0005)
    expect("4.8 fifo uniform rw", e42["agg"]["uniform/fifo"]["recall_rewarded"], 0.3333, 0.0005)
    expect("4.8 imp50 block rw", e42["agg"]["block/imp_tau50"]["recall_rewarded"], 0.0, 0)
    expect("4.8 imp500 block rw", e42["agg"]["block/imp_tau500"]["recall_rewarded"], 0.0, 0)
    expect("4.8 fifo block rw", e42["agg"]["block/fifo"]["recall_rewarded"], 0.0, 0)
    expect("4.8 random block rw", e42["agg"]["block/random"]["recall_rewarded"], 0.1267, 0.0005)
    expect("4.8 G-FB pass", 1 if e42["gates"]["G_FB"]["pass_"] else 0, 0, 0)
    expect("4.8 G-FBd pass", 1 if e42["gates"]["G_FBd"]["pass_"] else 0, 0, 0)
    expect_s("4.8 exp42 verdict", e42["verdict"], "FORGETTING_NO_BENEFIT_OVER_NAIVE")

    # ---- 4.20 公开语料外部检验（exp40，第六轮）----
    e40 = J("exp40_public_corpus.json")
    t1_40 = {(r["arm"], r["N"]): r for r in e40["T1"]}
    expect("4.20 N2k new", t1_40[("NEW256", 2000)]["deg0.3"], 3, 0)
    expect("4.20 N2k r512", t1_40[("R512", 2000)]["deg0.3"], 3, 0)
    expect("4.20 N8k new", t1_40[("NEW256", 8000)]["deg0.3"], 14, 0)
    expect("4.20 N8k r512", t1_40[("R512", 8000)]["deg0.3"], 11, 0)
    expect("4.20 N20k new", t1_40[("NEW256", 20000)]["deg0.3"], 27, 0)
    expect("4.20 N20k r512", t1_40[("R512", 20000)]["deg0.3"], 20, 0)
    expect("4.20 chain new self", e40["T2"]["NEW256"]["self_hit1"], 134, 0)
    expect("4.20 chain r512 self", e40["T2"]["R512"]["self_hit1"], 27, 0)
    expect("4.20 chain new next_rate", e40["T2"]["NEW256"]["next_rate"], 0.005, 0.0005)
    expect("4.20 chain r512 next_rate", e40["T2"]["R512"]["next_rate"], 0.005, 0.0005)
    expect("4.20 chain ctrl", max(e40["T2"][a]["ctrl5"] for a in ("NEW256", "R512")), 0, 0)
    expect("4.20 auc new", e40["T3"]["NEW256"]["auc"], 0.3354, 0.0005)
    expect("4.20 auc r512", e40["T3"]["R512"]["auc"], 0.0627, 0.0005)
    expect("4.20 G-P1 pass", 1 if e40["gates"]["G_P1"]["pass_"] else 0, 1, 0)
    expect("4.20 G-P2 pass", 1 if e40["gates"]["G_P2"]["pass_"] else 0, 1, 0)
    expect("4.20 G-P3 pass", 1 if e40["gates"]["G_P3"]["pass_"] else 0, 0, 0)

    # ---- 4.21 跨架构稳健性（exp39，第六轮）----
    e39 = J("exp39_resid_robustness.json")
    pa39 = {(r["enc"], r["a"], r["N"]): r for r in e39["part_a"]}
    expect("4.21 R512 a01 N9033 adv", pa39[("R512", 0.01, 9033)]["advantage"], -3, 0)
    expect("4.21 R512 a02 N9033 adv", pa39[("R512", 0.02, 9033)]["advantage"], -1, 0)
    expect("4.21 R512 a04 N9033 adv", pa39[("R512", 0.04, 9033)]["advantage"], -2, 0)
    expect("4.21 S256 a01 N9033 adv", pa39[("S256", 0.01, 9033)]["advantage"], -3, 0)
    expect("4.21 S256 a02 N9033 adv", pa39[("S256", 0.02, 9033)]["advantage"], -2, 0)
    expect("4.21 S256 a04 N9033 adv", pa39[("S256", 0.04, 9033)]["advantage"], -1, 0)
    expect("4.21 R512 a02 N4000 adv", pa39[("R512", 0.02, 4000)]["advantage"], -1, 0)
    expect("4.21 G-R1 mean", e39["gates"]["G_R1"]["mean"], -2, 0.0005)
    expect("4.21 G-R1 pass", 1 if e39["gates"]["G_R1"]["pass_"] else 0, 0, 0)
    expect("4.21 G-R2 mean_abs", e39["gates"]["G_R2"]["mean_abs"], 1.8333, 0.0005)
    expect("4.21 G-R2 pass", 1 if e39["gates"]["G_R2"]["pass_"] else 0, 1, 0)
    expect("4.21 G-R3 pass", 1 if e39["gates"]["G_R3"]["pass_"] else 0, 0, 0)
    pb39 = {r["enc"]: r for r in e39["part_b"]}
    expect("4.21 pb R512 b20", pb39["R512"]["curve"]["b20"], 24, 0)
    expect("4.21 pb R512 b100", pb39["R512"]["curve"]["b100"], 25, 0)
    expect("4.21 pb S256 b20", pb39["S256"]["curve"]["b20"], 15, 0)
    expect("4.21 pb S256 b100", pb39["S256"]["curve"]["b100"], 22, 0)
    expect("4.21 pb R512 written", pb39["R512"]["written"], 6387, 0)
    expect("4.21 pb S256 written", pb39["S256"]["written"], 7340, 0)
    expect("4.21 pa R512 on writes", pa39[("R512", 0.02, 9033)]["written_on"], 8922, 0)
    expect("4.21 pa R512 off writes", pa39[("R512", 0.02, 9033)]["written_off"], 6387, 0)
    expect("4.21 pa S256 on writes", pa39[("S256", 0.02, 9033)]["written_on"], 8912, 0)
    expect("4.21 pa S256 off writes", pa39[("S256", 0.02, 9033)]["written_off"], 7340, 0)

    # ---- 附录 F 外部复现（exp25，零 HCM 代码）----
    e25 = J("exp25_collapse_external.json")
    d25 = {d["N"]: d for d in e25["dense_cluster"]}
    i25 = {d["N"]: d for d in e25["dense"]}
    s25 = {d["N"]: d for d in e25["sparse"]}

    def h25(dd, N, b):
        r = dd[N]
        return r["hits"][r["betas"].index(b)]

    def bs25(dd, N):
        return dd[N]["beta_star"]

    expect("附F 聚簇 beta*(1e4)", bs25(d25, 10000), 5, 0)
    expect("附F 聚簇 beta*(1e6)", bs25(d25, 1000000), 5, 0)
    expect("附F 聚簇 beta2 hit@1e3", h25(d25, 1000, 2), 147, 0)
    expect("附F 聚簇 beta2 hit@1e5", h25(d25, 100000, 2), 0, 0)
    expect("附F 聚簇 beta2 hit@1e6", h25(d25, 1000000, 2), 2, 0)
    expect("附F iid beta2 hit@1e6", h25(i25, 1000000, 2), 50, 0)
    expect("附F 稀疏 beta2 hit@1e6", h25(s25, 1000000, 2), 44, 0)
    expect("附F iid beta*(1e6)", bs25(i25, 1000000), 2, 0)

    # ---- 7 L2 先导（exp37）与 10⁶ 门槛（exp52）——2026-11-17 CN 补节同步钉值 ----
    e37a = J("exp37_N100000_F_b100_s1.json")
    e37b = J("exp37_N100000_F_b100_s2.json")
    e37c = J("exp37_N100000_T_b100_s0.json")
    e37s = [e37a["s_per_write"], e37b["s_per_write"], e37c["s_per_write"]]
    expect("7 exp37 s_per_write min", min(e37s), 0.822, 0.0005)
    expect("7 exp37 s_per_write max", max(e37s), 0.962, 0.0005)
    expect("7 exp37 peak_rss max MB", max(e37a["peak_rss_mb"], e37b["peak_rss_mb"], e37c["peak_rss_mb"]), 28223.2, 0.5)
    for nm, e37 in (("F_s1", e37a), ("F_s2", e37b), ("T_s0", e37c)):
        expect("7 exp37 %s n_written" % nm, e37["n_written"], 100000, 0)
        for dk in ("deg0", "deg25"):
            expect("7 exp37 %s %s hits" % (nm, dk), e37[dk]["hits"], 60, 0)
            expect("7 exp37 %s %s of" % (nm, dk), e37[dk]["of"], 60, 0)
            expect("7 exp37 %s %s abstain" % (nm, dk), e37[dk]["abstain"], 0, 0)
    f52 = J("exp52_N1000000_F_b100_s0.json")
    t52 = J("exp52_N1000000_T_b100_s0.json")
    a52 = J("exp52_analyze.json")
    for nm, e52 in (("F", f52), ("T", t52)):
        expect("7 exp52 %s n_written" % nm, e52["n_written"], 1000000, 0)
        expect("7 exp52 %s n_dup_rejected" % nm, e52["n_dup_rejected"], 0, 0)
        for dk in ("deg0", "deg25"):
            expect("7 exp52 %s %s hits" % (nm, dk), e52[dk]["hits"], 60, 0)
            expect("7 exp52 %s %s of" % (nm, dk), e52[dk]["of"], 60, 0)
            expect("7 exp52 %s %s abstain" % (nm, dk), e52[dk]["abstain"], 0, 0)
    expect("7 exp52 F write_s", f52["write_s"], 17107.6, 0.5)
    expect("7 exp52 T write_s", t52["write_s"], 16022.0, 0.5)
    expect("7 exp52 F enc", f52["timing"]["enc"], 12038.4, 0.5)
    expect("7 exp52 F sc", f52["timing"]["sc"], 3519.7, 0.5)
    expect("7 exp52 F app", f52["timing"]["app"], 1226.3, 0.5)
    expect("7 exp52 T enc", t52["timing"]["enc"], 11464.1, 0.5)
    expect("7 exp52 T sc", t52["timing"]["sc"], 3432.3, 0.5)
    expect("7 exp52 T app", t52["timing"]["app"], 849.0, 0.5)
    expect("7 exp52 F peak_rss", f52["peak_rss_mb"], 75170.3, 0.5)
    expect("7 exp52 T peak_rss", t52["peak_rss_mb"], 75169.7, 0.5)
    expect("7 exp52 F csr", f52["csr_mb"], 3753.7, 0.5)
    expect("7 exp52 T csr", t52["csr_mb"], 3753.7, 0.5)
    expect_s("7 exp52 verdict", a52["verdict"]["overall"], "PASS")
    expect("7 exp52 G_EQ pass", 1 if a52["verdict"]["G_EQ"] else 0, 1, 0)
    expect("7 exp52 G_S1_1e6 pass", 1 if a52["verdict"]["G_S1_1e6"] else 0, 1, 0)
    expect("7 exp52 G_S3p pass", 1 if a52["verdict"]["G_S3p"] == "PASS" else 0, 1, 0)
    expect("7 exp52 pearson prereg", a52["G_S3p"]["pearson_prereg"], 0.9779, 0.001)
    expect("7 exp52 series 2e3", [p["s_per_write_s"] for p in a52["G_S3p"]["points"] if p["N"] == 2000][0], 0.008955, 0.0005)
    expect("7 exp52 series 1e5", [p["s_per_write_s"] for p in a52["G_S3p"]["points"] if p["N"] == 100000 and p["source"] == "exp43_part2"][0], 0.4102, 0.0005)
    expect("7 exp52 series 2e5", [p["s_per_write_s"] for p in a52["G_S3p"]["points"] if p["N"] == 200000 and p["source"] == "exp52_eq"][0], 0.5991, 0.0005)
    expect("7 exp52 audit bad_dup", sum(c["audit_bad_dup"] for c in a52["G_S1_1e6"]["per_cell"].values()), 0, 0)
    expect("7 exp52 audit g_fail", sum(c["audit_g_fail"] for c in a52["G_S1_1e6"]["per_cell"].values()), 0, 0)

    # ---- exp45: online β* cross-session transfer (摘要 99.6%) ----
    e45 = J("exp45_analyze.json")
    expect("摘要 exp45 transfer ratio", e45["gate"]["ratio"], 0.9964, 0.001)
    expect_s("摘要 exp45 verdict", e45["gate"]["verdict"], "POSITIVE")
    expect("摘要 exp45 n_seeds", e45["gate"]["n_seeds"], 3, 0)

    # ---- exp47b: six-architecture contrast (摘要三预注册门) ----
    e47b = J("exp47b_collapse_baselines_m2.json")
    expect("摘要 exp47b G-A hopf max tier", max(t["deg0.3"]["rate"] for t in e47b["rows"]["hopf"]), 0.01, 0.005)
    expect("摘要 exp47b G-B modern flat max|slope|", max(abs(e47b["slopes_deg03"][a]) for a in ["bm25_char1","bm25_char2","knn_tfidf","knn_s256"]), 0.0167, 0.001)
    expect("摘要 exp47b G-C hcm ref slope", e47b["hcm_ref_slope_per_decade_beta100"], -0.0212, 0.001)
    expect("摘要 exp47b gate A pass", 1 if e47b["gates"]["g47b_a_hopf_max_tier_lt_005"] else 0, 1, 0)
    expect("摘要 exp47b gate B pass", 1 if e47b["gates"]["g47b_b_modern_flat_abs_slope_lt_002_per_decade"] else 0, 1, 0)
    expect("摘要 exp47b gate C pass", 1 if e47b["gates"]["g47b_c_hcm_ref_slope_negative"] else 0, 1, 0)


    # ---- §4.21 Part C: 受控合成 β*(overlap) 阶梯（G-R3 修复 B2，2026-11-19 三机网格）----
    g3 = J("g3_analysis.json")
    expect("4.21C G-R3a pooled n", g3["n_rows"], 30, 0)
    expect("4.21C G-R3a pooled rho", g3["G-R3a"]["rho"], 0.8381, 0.0002)
    expect_s("4.21C G-R3a verdict", g3["G-R3a"]["verdict"], "FAIL")
    expect("4.21C beta coverage 10 点全格", sum(1 for v in g3["beta_coverage"].values() if v == 10), 30, 0)
    expect("4.21C rho N=2000", g3["by_N"]["2000"]["rho"], 0.9009, 0.0002)
    expect("4.21C rho N=20000", g3["by_N"]["20000"]["rho"], 0.8913, 0.0002)
    mb2k = g3["by_N"]["2000"]["mean_beta_star_by_overlap"]
    mb20k = g3["by_N"]["20000"]["mean_beta_star_by_overlap"]
    expect("4.21C mean b* N2k o0", mb2k["0"], 20, 0)
    expect("4.21C mean b* N2k o0.2", mb2k["0.2"], 40, 0)
    expect("4.21C mean b* N2k o0.4", mb2k["0.4"], 50, 0)
    expect("4.21C mean b* N2k o0.6", mb2k["0.6"], 50, 0)
    expect("4.21C mean b* N2k o0.8", mb2k["0.8"], 200, 0)
    expect("4.21C mean b* N20k o0", mb20k["0"], 50, 0)
    expect("4.21C mean b* N20k o0.2", mb20k["0.2"], 50, 0)
    expect("4.21C mean b* N20k o0.4", mb20k["0.4"], 50, 0)
    expect("4.21C mean b* N20k o0.6", mb20k["0.6"], 100, 0)
    expect("4.21C mean b* N20k o0.8", mb20k["0.8"], 166.67, 0.01)
    expect("4.21C G-R3b N2000 delta>=floor", 1 if g3["G-R3b"]["2000"]["delta_ge_floor"] else 0, 1, 0)
    expect("4.21C G-R3b N20000 delta>=floor", 1 if g3["G-R3b"]["20000"]["delta_ge_floor"] else 0, 0, 0)
    expect("4.21C G-R3c sd=0 格数", sum(1 for s in g3["G-R3c"]["per_cell"] if s["sd"] == 0), 8, 0)
    # Appendix I/K (G-R3b mechanism reading) — ENF/ARX/CN
    def paper_has(fn, needle):
        return needle in open(os.path.join(ROOT, fn), encoding="utf-8").read()
    # 投稿包不随公开仓库分发（paper/submission/ 已排除）；只校验仓库内实际存在的稿本。
    for f in ["paper/PAPER_EN_final.md"]:
        expect_s("appx GR3b heading in " + f, paper_has(f, "why the analytic lower bound overshoots"), True)
        expect_s("appx GR3b ratio2.5 in " + f, paper_has(f, "ratio 2.5 vs ln 10"), True)
        expect_s("appx GR3b knee sentence in " + f, paper_has(f, "saturation knee"), True)
    expect_s("appx GR3b CN heading", paper_has("paper/PAPER_final.md", "解析下界为何高过实测阶梯"), True)
    expect_s("appx GR3b CN ratio", paper_has("paper/PAPER_final.md", "比值 2.5 vs ln 10"), True)

    # ---- §4.21 Part C / Appendix I,K,H: Stage-0 real-memory retrieval cross-check (2026-11-20) ----
    s0 = J("stage0_retrieval_crosscheck.json")
    rp, rh = s0["retrieval_pool"], s0["retrieval_hits"]
    expect("4.21C stage0 pool n", rp["n_entries"], 132, 0)
    expect("4.21C stage0 pool median", rp["age_median_days"], 43, 0)
    expect("4.21C stage0 pool 31-60d", rp["age_buckets"]["31-60d"], 118, 0)
    expect("4.21C stage0 pool 8-30d", rp["age_buckets"]["8-30d"], 0, 0)
    expect("4.21C stage0 hits instances", rh["hit_instances"], 11171, 0)
    expect("4.21C stage0 hits unique", rh["unique_entries_hit"], 44, 0)
    expect("4.21C stage0 coverage", rh["pool_coverage"], 0.3333, 0.0002)
    expect("4.21C stage0 hit median", rh["hit_age_median_days"], 48, 0)
    expect("4.21C stage0 hit oldest", rh["hit_age_oldest_days"], 76, 0)
    expect("4.21C stage0 hit 8-30d", rh["hit_age_buckets"]["8-30d"], 0, 0)
    libs = {l["name"]: l for l in s0["libraries"]}
    expect("4.21C stage0 lib m1 n", libs["machine1-memory"]["n_dated"], 371, 0)
    expect("4.21C stage0 lib m1 median", libs["machine1-memory"]["median_age_days"], 25, 0)
    expect("4.21C stage0 lib m1 oldest", libs["machine1-memory"]["oldest_age_days"], 67, 0)
    expect("4.21C stage0 lib coze n", libs["coze-export-part04"]["n_dated"], 546, 0)
    expect("4.21C stage0 lib coze median", libs["coze-export-part04"]["median_age_days"], 25, 0)
    expect("4.21C stage0 lib coze oldest", libs["coze-export-part04"]["oldest_age_days"], 67, 0)
    # 投稿包不随公开仓库分发（paper/submission/ 已排除）；只校验仓库内实际存在的稿本。
    for f in ["paper/PAPER_EN_final.md"]:
        expect_s("appx stage0 crosscheck in " + f, paper_has(f, "Cross-check against a real agent memory (Stage 0, descriptive)"), True)
        expect_s("appx stage0 pool132 in " + f, paper_has(f, "132 entries with a median age of 43 days"), True)
        expect_s("appx stage0 hits11171 in " + f, paper_has(f, "11,171 recorded retrieval hits"), True)
    expect_s("appx stage0 CN heading", paper_has("paper/PAPER_final.md", "与真实智能体记忆的交叉核对（Stage 0，描述性）"), True)
    expect_s("appx stage0 CN pool", paper_has("paper/PAPER_final.md", "检索池 132 条、中位 43 天"), True)
    expect_s("appx stage0 CN hits", paper_has("paper/PAPER_final.md", "11,171 次检索命中"), True)

    return problems, checked


BANNED = [("已证明", "过强"), ("充分证明", "过强"), ("普适", "过强"),
          ("完全解决", "过强"), ("state-of-the-art", "无 SOTA 主张"),
          ("SOTA", "无 SOTA 主张")]


def check_wording(text):
    body = text.partition("## 参考文献")[0]
    return ["「%s」（%s）" % (w, why) for w, why in BANNED if w in body]


def main():
    text = open(PAPER, encoding="utf-8").read()
    p1, n_cited, n_listed = check_citations(text)
    p2, checked = check_numbers()
    w = check_wording(text)
    ok_n = sum(1 for *_, ok in checked if ok)
    print("A. 引文：正文 %d 个编号 / 参考文献表 %d 条" % (n_cited, n_listed))
    print("B. 数字溯源：%d/%d 通过" % (ok_n, len(checked)))
    for label, got, want, ok in checked:
        if not ok:
            print("   BAD %s: 结果=%s 论文=%s" % (label, got, want))
    print("C. 措辞扫描：命中 %d 处 %s" % (len(w), w if w else ""))
    problems = p1 + p2 + (["超出证据的措辞：%s" % ", ".join(w)] if w else [])
    if problems:
        print("\n[FAIL]")
        for x in problems:
            print("  - " + x)
        return 1
    print("\n[PASS] 三向比对全绿")
    return 0


if __name__ == "__main__":
    sys.exit(main())
