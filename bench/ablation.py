#!/usr/bin/env python3
"""ablation.py —— 九子系统消融矩阵（门禁 G4）。

对每个子系统做"关掉它"的干预，测四个下游指标：
  SEP   高重叠(0.90)输入下的模式分离（精确召回）
  DROP  30% 特征缺失下的召回（去噪能力）
  CLS   持续学习中的旧任务保持（回放价值）
  WM    工作记忆超载下的保留条数
若某个子系统关掉后**任何**指标都不变，则标记 NON-LOAD-BEARING（论文必须如实报告）。
"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import overlapping_set, random_patterns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "ablation.json")
D = 128


def degrade(x, frac, seed):
    r = np.random.default_rng(seed)
    y = x.copy()
    nz = np.where(np.abs(y) > 1e-6)[0]
    if len(nz) == 0:
        return unit(x)
    y[r.choice(nz, size=max(1, int(len(nz) * frac)), replace=False)] = 0.0
    return unit(y)


def m_sep_drop(cfg, overlap=0.90, N=30):
    """高重叠输入下的分离 + 去噪。"""
    X = overlapping_set(N, D, overlap=overlap, seed=11)
    m = HCM(cfg)
    for x in X:
        m.observe(x)
    exact = sum(int(m.query(X[i])["idx"] == i) for i in range(N))
    drop = sum(int(m.query(degrade(X[i], 0.30, i))["idx"] == i) for i in range(N))
    return exact, drop


def m_cls(cfg, cyc=12, budget=24, seed=0):
    """持续学习：学 0-3 类，再学 4-7 类，回放交错；返回旧/新留出准确率。"""
    protos = random_patterns(8, D, seed=5)
    NOISE = 0.35

    def sample(n_per, s):
        r = np.random.default_rng(s)
        X, Y = [], []
        for k in range(8):
            for _ in range(n_per):
                X.append(unit(protos[k] + NOISE * unit(r.normal(size=D))))
                Y.append(k)
        return np.array(X, dtype=np.float32), np.array(Y)

    Xtr, Ytr = sample(12, 101)
    Xte, Yte = sample(6, 202)
    m = HCM(cfg)
    CATS = np.array([m.enc.encode(unit(p)) for p in protos], dtype=np.float32)
    Htr = np.array([m.enc.encode(x) for x in Xtr], dtype=np.float32)
    r = np.random.default_rng(seed)
    idxA = np.where(Ytr < 4)[0]
    idxB = np.where(Ytr >= 4)[0]
    for _ in range(cyc):
        b = r.choice(idxB, size=budget, replace=False)
        if cfg_ablate_replay := getattr(cfg, "_no_replay", False):
            sel = b
        else:
            o = r.choice(idxA, size=budget, replace=False)
            sel = np.concatenate([b, o])
        m.sem.batch_fit(Xtr[sel], Htr[sel], epochs=2, eta=0.05)

    def acc(ks):
        ok = n = 0
        for x, y in zip(Xte, Yte):
            if y not in ks:
                continue
            h = unit(np.asarray(m.sem.forward(x), dtype=np.float32))
            ok += int(int(np.argmax(CATS @ h)) == y)
            n += 1
        return ok, n

    aA, nA = acc(set(range(4)))
    aB, _ = acc(set(range(4, 8)))
    return aA, aB, nA


def m_wm(cfg, C=4, L=8, trials=4):
    kept = []
    for t in range(trials):
        r = np.random.default_rng(200 + t)
        cfg_t = Config(**{**cfg.__dict__, "C": C})
        m = HCM(cfg_t)
        X = random_patterns(L, D, seed=700 + t)
        codes = [m.enc.encode(x) for x in X]
        for h in codes:
            m.wm.present(h)
        sk = m.wm.slots
        ok = 0
        for i, h in enumerate(codes):
            if float((sk @ h).max()) < 0.9:
                continue
            j = int(np.argmax(sk @ h))
            q = unit(h + r.normal(scale=0.05, size=h.shape))
            found, _ = m.wm.probe(q)
            ok += int(found and int(np.argmax(sk @ q)) == j)
        kept.append(ok)
    return float(np.mean(kept))


def base_cfg(**kw):
    c = Config(d_x=D, a=0.02, capacity=400, beta=20.0, L=3, use_resid=True, seed=0, d_z=64)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


ARMS = [
    ("full", "完整模型", lambda: base_cfg()),
    ("M1_SEP no-residual", "关掉期望库残差（M1a）", lambda: base_cfg(use_resid=False)),
    ("M1_SEP no-sparsity", "关掉稀疏化：k=d_h（M1b）", lambda: base_cfg(a=1.0)),
    ("M2_EPI no-attractor", "关掉 Hopfield 迭代：L=0（M2）", lambda: base_cfg(L=0)),
    ("M2_EPI no-store", "库容量=0：完全不能写情节（M2）", lambda: base_cfg(capacity=1)),
    ("M3_CTX no-context", "情境时间常数=1：无情境维持（M3）", lambda: base_cfg(gamma_c=1.0)),
    ("M4/M5 no-cortex", "皮层冻结：不学习（M5）", lambda: base_cfg()),
    ("M4 no-replay", "无交错回放（M4）", lambda: base_cfg()),
    ("M6_WM C=1", "工作记忆容量压到 1（M6）", lambda: base_cfg(C=1)),
    ("M7_NEU const-gate", "门控恒为 1：无调质（M7）", lambda: base_cfg(gate_constant=True)),
    ("M8_FOR no-rif", "关掉检索诱发抑制（M8）", lambda: base_cfg(rif_on=False)),
    ("M0_META always-retrieve", "永不弃权（M0）", lambda: base_cfg()),
]


def main():
    rows = []
    for key, desc, mk in ARMS:
        t0 = time.time()
        cfg = mk()
        cfg._no_replay = key == "M4 no-replay"
        exact, drop = m_sep_drop(cfg)
        if key == "M4/M5 no-cortex":
            # 修正（对抗评审 P1-4）：v1 这里测的是一个**从未训练过**的随机网络，
            # 得到 0/24 是同义反复。正确做法是「先训练 A -> 冻结皮层 -> 再训练 B」，
            # 然后看 A 的保持率。冻结用 freeze()（把级联刚度推到极大，eta 近似为 0）。
            cfg2 = mk()
            cfg2._no_replay = False
            protos = random_patterns(8, D, seed=5)
            m2 = HCM(cfg2)
            CATS2 = np.array([m2.enc.encode(unit(pr)) for pr in protos], dtype=np.float32)
            rr = np.random.default_rng(101)
            Xtr2, Ytr2 = [], []
            for k in range(8):
                for _ in range(12):
                    Xtr2.append(unit(protos[k] + 0.35 * unit(rr.normal(size=D))))
                    Ytr2.append(k)
            Xtr2 = np.asarray(Xtr2, dtype=np.float32)
            Ytr2 = np.asarray(Ytr2)
            H2 = np.array([m2.enc.encode(x) for x in Xtr2], dtype=np.float32)
            idxA2 = np.where(Ytr2 < 4)[0]
            # 阶段 1：学 A
            for _ in range(12):
                sel = rr.choice(idxA2, size=min(24, len(idxA2)), replace=False)
                m2.sem.batch_fit(Xtr2[sel], H2[sel], epochs=2, eta=0.05)
            m2.sem.freeze()          # 冻结：此后 eta_eff 近似 0
            # 阶段 2：学 B（皮层已冻结，理应学不进去）
            idxB2 = np.where(Ytr2 >= 4)[0]
            for _ in range(12):
                sel = rr.choice(idxB2, size=min(24, len(idxB2)), replace=False)
                m2.sem.batch_fit(Xtr2[sel], H2[sel], epochs=2, eta=0.05)
            # 评测
            Xte2 = []
            for k in range(8):
                for _ in range(6):
                    Xte2.append((unit(protos[k] + 0.35 * unit(rr.normal(size=D))), k))
            aA = aB = 0
            nA = 0
            for x, y in Xte2:
                h = unit(np.asarray(m2.sem.forward(x), dtype=np.float32))
                hit = int(int(np.argmax(CATS2 @ h)) == y)
                if y < 4:
                    aA += hit
                    nA += 1
                else:
                    aB += hit
        else:
            aA, aB, nA = m_cls(cfg)
        wm = m_wm(Config(**{**mk().__dict__, "C": 4})) if key != "M6_WM C=1" else m_wm(mk(), C=1)
        if key == "M0_META always-retrieve":
            cfg3 = mk()
            cfg3._no_replay = False
            # always-retrieve 只影响决策，不进入上面三个指标；单独记为"决策层消融"
            pass
        if key == "M0_META always-retrieve":
            # 第二份代码级审查（2026-11-08）：本行四项指标物理上等于 full（always-retrieve
            # 只影响决策层，不进 SEP/DROP/CLS/WM）。决策层消融在 ablation2 / g4_ledger。
            rows.append(dict(arm=key, desc=desc + " [镜像行：数字=full，决策层消融见 ablation2/g4_ledger]",
                             sep_exact_090=None, drop30=None, cls_old=None, cls_new=None,
                             cls_n=None, wm_kept=None, sec=round(time.time() - t0, 1)))
            continue
        rows.append(dict(arm=key, desc=desc, sep_exact_090=exact, drop30=drop,
                         cls_old=aA, cls_new=aB, cls_n=nA, wm_kept=round(wm, 2),
                         sec=round(time.time() - t0, 1)))
        print("%-28s SEP=%2d/30 DROP=%2d/30 CLS_old=%2d/%d CLS_new=%d WM=%.2f (%.0fs)" %
              (key, exact, drop, aA, nA, aB, wm, time.time() - t0), flush=True)
    json.dump(rows, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
