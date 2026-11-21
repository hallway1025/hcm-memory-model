#!/usr/bin/env python3
"""ablation2.py —— 补齐四个 NON-LOAD-BEARING 子系统的**专属**度量。

问题：ablation.py 用统一四指标，导致 M3_CTX / M7_NEU / M8_FOR / M0_META 显不出作用。
根因：这些子系统作用在**别的因变量**上，而不是 top-1 命中。
  M3_CTX 情境 → 影响"同内容不同情境"的可分辨性（occasion setting），而非一般召回
  M7_NEU 门控 → 影响"该记的记、不该记的不记"的选择性（信噪比），而非总命中
  M8_FOR RIF  → 影响"检索后竞争条目的可及性下降"（RIF 范式），而非平均准确率
  M0_META     → 影响"弃权带来的风险调整收益"，而非召回
本脚本为每个子系统设计专属任务，并在开/关两臂上比较。
"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import random_patterns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "ablation2.json")
D = 128
rows = []


def base_cfg(**kw):
    c = Config(d_x=D, a=0.02, capacity=600, beta=20.0, L=3, use_resid=True, seed=0, d_z=64)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


# ---------- M3_CTX：同内容、两种情境的可分辨性 ----------
def m3_occasion_setting(use_ctx):
    """同一组内容在两种情境下反复出现，要求模型按情境给出不同检索结果。
    情境用"全局背景向量"注入（与内容正交），测：切换情境后能否找回该情境下的关联。"""
    cfg = base_cfg(use_ctx_bias=use_ctx, kappa_ctx=3.0)
    m = HCM(cfg)
    r = np.random.default_rng(7)
    contents = random_patterns(12, D, seed=31)
    ctxA, ctxB = unit(r.normal(size=D)), unit(r.normal(size=D))
    # 情境 A 下呈现内容（写入），情境 B 下呈现同样内容但作为"另一个事件"
    for c in contents:
        m.observe(unit(c + 2.0 * ctxA))
    for c in contents:
        m.observe(unit(c + 2.0 * ctxB))
    # 查询：情境 A 下的内容线索 -> 应命中 A 组（下标 0..11）
    ok = 0
    for i, c in enumerate(contents):
        m.c = (2.0 * ctxA).astype(np.float32)
        q = m.query(unit(c + 2.0 * ctxA))
        ok += int(q["idx"] == i)
    return ok


# ---------- M7_NEU：门控的选择性（信噪比） ----------
def m7_signal_selection(constant_gate):
    """流式输入：80% 是重复的背景噪声模式，20% 是应当记住的显著事件（带奖赏）。
    指标 = 写入库中"显著事件"占比（选择性）与显著事件的召回。"""
    cfg = base_cfg(gate_constant=constant_gate)
    m = HCM(cfg)
    r = np.random.default_rng(3)
    bg = random_patterns(3, D, seed=41)          # 背景：逐字重复的模式
    events = random_patterns(12, D, seed=42)     # 显著事件：一次性出现、带奖赏
    # 背景重复 10 轮（每轮同样 3 条），门控应当逐轮降低写入
    for _ in range(10):
        for b in bg:
            m.observe(b, r=0.0)
    for e in events:
        m.observe(e, r=1.0)
    n = m.E.n
    # 显著事件召回
    hit = sum(int(m.query(e)["idx"] >= 0) for e in events)
    # 选择性 = 写入库中**不属于背景**的条目数（期望 ≈ 3 条背景 + 12 条事件 = 15）
    bg_codes = np.array([m.enc.encode(b) for b in bg])
    n_bg_in_store = 0
    for i in range(m.E.n):
        if float(np.max(bg_codes @ m.E.M[:, i])) > 0.95:
            n_bg_in_store += 1
    sel = m.E.n - n_bg_in_store
    return dict(written=n, recall=hit, distinct_items=n_bg_in_store, event_items=sel)

    return n, hit, sel


# ---------- M8_FOR：检索诱发抑制（RIF）----------
def m8_rif(rif_on):
    """M8 的诚实度量：直接报告**抑制因子的差分**，以及它对 top-1 的影响。

    为什么不用 top-1：实测 sup 确实被压到 0.03 量级，但 argmax 不变 ——
    top-1 由证据强度决定，被抑制者本来就不是赢家。这是 RIF 文献的经典结论
    （练习降低未练竞争项的可及性，显现在自由回忆/反应时上，不显现在强制选择上）。
    在本架构里，更关键的是：**40 条库、每轮 top-5 竞争、5 轮练习**会让几乎所有条目
    都被同等抑制（实测已练 0.031 vs 未练 0.027），因此差分极小。
    这个"抑制是全局而非选择性"的事实本身就是结论（L-021）。
    """
    cfg = base_cfg(rif_on=rif_on, rho_rif=0.30, capacity=800)
    m = HCM(cfg)
    X = random_patterns(40, D, seed=51)
    for x in X:
        m.observe(x)
    practiced, unpracticed = list(range(20)), list(range(20, 40))
    ok_before = sum(int(m.query(X[i], do_rif=False)["idx"] == i) for i in unpracticed)
    # 注意：必须传 do_rif=rif_on，不能用 do_rif=True —— 后者会覆盖 cfg.rif_on，
    # 导致"关闭 RIF"那一臂其实也在抑制（这个 bug 让两臂数字完全相同过一次）。
    for _ in range(5):
        for i in practiced:
            m.query(X[i], do_rif=rif_on)
    sup = m.E.sup[: m.E.n]
    ok_after = sum(int(m.query(X[i], do_rif=False)["idx"] == i) for i in unpracticed)
    return dict(sup_practiced=round(float(sup[:20].mean()), 4),
                sup_unpracticed=round(float(sup[20:40].mean()), 4),
                recall_before=ok_before, recall_after=ok_after, n=20)


# ---------- M0_META：弃权的风险调整收益 ----------
def m0_abstain(always):
    """库内 40 条 + 库外 40 条，线索缺失 35%。
    风险调整得分 = 正确数 - 1.0*错误数 - 0.3*弃权数（答错比弃权贵）。"""
    cfg = base_cfg()
    m = HCM(cfg)
    if always:
        m.ctl.always_retrieve = True
    X = random_patterns(40, D, seed=61)
    Xn = random_patterns(40, D, seed=62)
    for x in X:
        m.observe(x)

    def degrade(x, frac, seed):
        r = np.random.default_rng(seed)
        y = x.copy()
        nz = np.where(np.abs(y) > 1e-6)[0]
        y[r.choice(nz, size=max(1, int(len(nz) * frac)), replace=False)] = 0.0
        return unit(y)

    score = 0.0
    for i, x in enumerate(X):
        q = m.query(degrade(x, 0.35, i))
        if q["action"] == "recall":
            score += 1.0 if q["idx"] == i else -1.0
        else:
            score -= 0.3
    for i, x in enumerate(Xn):
        q = m.query(degrade(x, 0.35, 500 + i))
        score += -1.0 if q["action"] == "recall" else -0.3
    return score


print("== M3_CTX 情境（occasion setting）==", flush=True)
t0 = time.time()
a_on, a_off = m3_occasion_setting(True), m3_occasion_setting(False)
print(f"  ctx_bias 开: {a_on}/12   关: {a_off}/12", flush=True)
rows.append(dict(module="M3_CTX", metric="occasion_setting_hits", on=a_on, off=a_off, max=12,
                 sec=round(time.time() - t0, 1)))

print("== M7_NEU 门控选择性 ==", flush=True)
t0 = time.time()
g_on = m7_signal_selection(False)
g_off = m7_signal_selection(True)
print("  门控学习: 写入 %d/42  事件召回 %d/12  库内不同的背景条目 %d（理想 3）" % (
    g_on["written"], g_on["recall"], g_on["distinct_items"]), flush=True)
print("  门控恒1 : 写入 %d/42  事件召回 %d/12  库内不同的背景条目 %d（理想 3）" % (
    g_off["written"], g_off["recall"], g_off["distinct_items"]), flush=True)
rows.append(dict(module="M7_NEU", metric="write_selectivity", on=g_on, off=g_off,
                 sec=round(time.time() - t0, 1)))

print("== M8_FOR 检索诱发抑制 ==", flush=True)
t0 = time.time()
on = m8_rif(True)
off = m8_rif(False)
for tag, r_ in (("RIF 开", on), ("RIF 关", off)):
    print("  %s: sup 已练=%.4f 未练=%.4f (差分 %+.4f) | 未练召回 %d->%d/%d" % (
        tag, r_["sup_practiced"], r_["sup_unpracticed"],
        r_["sup_unpracticed"] - r_["sup_practiced"], r_["recall_before"], r_["recall_after"], r_["n"]), flush=True)
rows.append(dict(module="M8_FOR", metric="suppression_differential", on=on, off=off,
                 sec=round(time.time() - t0, 1)))

print("== M0_META 弃权 ==", flush=True)
t0 = time.time()
s_meta, s_always = m0_abstain(False), m0_abstain(True)
print(f"  元控制: {s_meta:.1f}   永不弃权: {s_always:.1f}", flush=True)
rows.append(dict(module="M0_META", metric="risk_adjusted_score", on=s_meta, off=s_always,
                 sec=round(time.time() - t0, 1)))

json.dump(rows, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved", OUT)
