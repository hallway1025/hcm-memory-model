"""HCM 主模型：把 9 个子系统装配成 docs/ARCHITECTURE-CORE.md 定义的动力学。"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .ops import unit, softmax_beta, cosine
from .encoder import Encoder
from .episodic import EpisodicStore
from .semantic import SemanticStore
from .wm import WorkingMemory
from .gate import WriteGate
from .consolidate import Consolidator
from .control import Metacontroller


@dataclass
class Config:
    d_x: int = 128
    d_c: int = 128
    d_tau: int = 32
    rho: int = 8
    a: float = 0.02
    d_z: int = 64
    capacity: int = 2000
    C: int = 4
    beta: float = 20.0
    L: int = 3
    gamma_c: float = 0.95
    context_taus: object = None    # 多时间尺度情境：None=单 EMA(gamma_c)；否则如 [1,10,100,1000]（B2/exp36）
    T_ctx: int = 200
    theta_write: float = 0.5
    theta_rec: float = 0.25
    margin: float = 0.06
    lam: float = 0.6
    w_dense: float = 1.0
    homeo: bool = False
    use_resid: bool = False
    # S1（外审采纳，D-052）：默认关闭。exp14/exp20b：残差在 β=100 下中性、
    # 在 β=20 下与误设共享根因；生产配置全部显式传参。历史诊断脚本需旧行为的请在
    # Config 里显式 use_resid=True。
    homeo_eta: float = 0.35
    homeo_epochs: int = 6
    use_ctx_bias: bool = False
    homeo_online: bool = False
    dense_raw: bool = True
    kappa_ctx: float = 0.6
    ctx_in_energy: bool = False
    bank_cap: int = 8
    switch_ratio: float = 1.5
    switch_min_gap: int = 4
    kappa: float = 8.0
    u_meta: float = 0.02
    eta_v: float = 0.05
    expand: bool = True
    norm: str = "none"
    rel: float | None = None
    learn_encoder: bool = False
    gate_constant: bool = False
    retrieve_init: str = "pre"   # "pre"=历史行为（阈值化码作吸引子初值）；"raw"=用稠密鲁棒证据 s_ev（exp19c 修复臂）
    softmax_on_raw: bool = False  # True: L=0 分支的 a=softmax(beta·未归一化 ev)（修复 L0 近均匀问题，exp19c）
    rif_on: bool = True
    rho_rif: float = 0.15
    seed: int = 0

    @property
    def d_h(self):
        return int(self.d_x * self.rho) if self.expand else int(self.d_x)

    @property
    def k(self):
        return max(1, int(round(self.a * self.d_h)))


PRESETS = {
    "L0": Config(d_x=128, rho=8, capacity=2000, d_z=64, T_ctx=200),
    "L1": Config(d_x=512, rho=8, capacity=20000, d_z=128, T_ctx=500),
    "L2": Config(d_x=2048, rho=8, capacity=500000, d_z=256, T_ctx=1000),
    "L3": Config(d_x=8192, rho=8, capacity=20000000, d_z=512, T_ctx=2000),
}


def preset(name):
    import copy
    return copy.deepcopy(PRESETS[name])


class HCM:
    """M = <X, Phi, E, Theta, W, Psi, Pi, Gamma, G> 的可执行版本。"""

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else Config()
        c = self.cfg
        self.enc = Encoder(c.d_x, c.d_c, c.d_tau, c.d_h, c.k, seed=c.seed,
                           expand=c.expand, learnable=c.learn_encoder, norm=c.norm, rel=c.rel,
                           homeo=c.homeo, homeo_eta=c.homeo_eta, use_resid=c.use_resid,
                           bank_cap=c.bank_cap, switch_ratio=c.switch_ratio, switch_min_gap=c.switch_min_gap)
        self.E = EpisodicStore(c.d_h, c.capacity, c.k, beta=c.beta, seed=c.seed + 1, d_p=c.d_x)
        self.sem = SemanticStore(c.d_x, c.d_h, c.d_z, seed=c.seed + 2, kappa=c.kappa, u_meta=c.u_meta)
        self.wm = WorkingMemory(c.d_h, C=c.C)
        self.gate = WriteGate(constant=c.gate_constant)
        self.cons = Consolidator(self.E, self.sem, seed=c.seed + 3)
        self.ctl = Metacontroller(theta_rec=c.theta_rec, margin=c.margin)
        self.c = np.zeros(c.d_c, dtype=np.float32)
        self.c_multi = [np.zeros(c.d_x, dtype=np.float32) for _ in (c.context_taus or [])]
        self.v = np.zeros(c.d_h, dtype=np.float32)
        self.t = 0
        self.log = []

    # ---------------- 情境与时间基 ----------------
    def _update_context(self, x):
        taus = self.cfg.context_taus
        if not taus:
            g = self.cfg.gamma_c
            self.c = (g * self.c + (1.0 - g) * x).astype(np.float32)
            return unit(self.c)
        # B2/exp36：多时间尺度 EMA 拼接；self.c 保持与 d_c 同形（查询/存储共用）
        parts = []
        for i, tau in enumerate(taus):
            g = float(np.exp(-1.0 / max(float(tau), 1e-6)))
            self.c_multi[i] = (g * self.c_multi[i] + (1.0 - g) * x).astype(np.float32)
            parts.append(self.c_multi[i])
        self.c = np.concatenate(parts).astype(np.float32)
        return unit(self.c)

    def _timecells(self):
        c = self.cfg
        j = np.arange(c.d_tau, dtype=np.float32)
        mu = (j + 0.5) / c.d_tau * c.T_ctx
        sigma = c.T_ctx / (2.0 * c.d_tau)
        tt = float(self.t % c.T_ctx)
        tau = np.exp(-((tt - mu) ** 2) / (2.0 * sigma ** 2 + 1e-8)).astype(np.float32)
        return unit(tau)

    def _assemble(self, x):
        x = unit(np.asarray(x, dtype=np.float32))
        cvec = self._update_context(x)
        u = np.concatenate([x, cvec, self._timecells()]).astype(np.float32)
        return x, u

    # ---------------- 在线单步 ----------------
    def observe(self, x, r=0.0, write=True, event=False, sender="", session=""):
        """第 0-6 步：输入 -> 编码 -> 新颖性 -> 门控 -> 写入 -> 工作记忆。"""
        c = self.cfg
        x, u = self._assemble(x)
        xe = self.enc.resid_write(x)             # 情境残差：去掉共享期望成分
        s_pre = self.enc.pre(xe)                 # 稠密 EC 驱动（写入与查询共用同一投射）
        h = self.enc.encode_from_pre(s_pre)      # 写入侧稀疏码（DG 分离）
        dense = s_pre
        ctx = unit(self.c)
        tau_b = self._timecells()
        # 新颖性 = 相对期望模板的偏离（L-020 第 4 代）。去重由 store.write 的 theta_dup 负责。
        nu = self.enc.novelty(x, mu=self.enc.mu)
        if self.E.n > 0:
            sc = self.E.M[:, : self.E.n].T @ h
            p = softmax_beta(sc, c.beta)
            eps = float(-np.log(p.max() + 1e-12))
        else:
            eps = float(np.log(2.0))
        vh = float(self.v @ h)
        rpe = float(r) - vh
        self.v += (c.eta_v * rpe * h).astype(np.float32)
        activity = float(np.abs(h).sum() / np.sqrt(max(c.k, 1)))
        g = self.gate(nu, rpe, eps, activity)
        idx = -1
        eid = None
        if write and event:
            # 事件层（D-042/exp30）：事件记录恒写；内容码按 theta_dup 去重复用，
            # 新码创建仍受门控约束（保持 E 的容量语义与默认路径完全一致）。
            eid, idx = self.E.write_event(h, self.t, g, r, nu, allow_new=(g >= c.theta_write),
                                          payload=x, dense=dense, ctx=ctx, tau=tau_b,
                                          sender=sender, session=session)
            if idx is not None and idx >= 0 and self.cfg.learn_encoder:
                self.enc.hebbian_update(u, h)
        elif write and g >= c.theta_write:
            idx = self.E.write(h, self.t, g, r, nu, payload=x, dense=dense, ctx=ctx, tau=tau_b)
            if idx >= 0 and self.cfg.learn_encoder:
                self.enc.hebbian_update(u, h)
        if c.homeo_online:
            self.enc.adapt(h)                    # 在线更新会破坏已写码的自洽性，默认关闭（见 D-018）
        self.wm.present(h)
        self.t += 1
        rec = {"t": self.t, "g": g, "nu": nu, "eps": eps, "rpe": rpe,
               "wrote": idx is not None and idx >= 0, "idx": idx, "eid": eid, "n": self.E.n}
        self.log.append(rec)
        return rec

    # ---------------- 检索 ----------------
    def query(self, x, L=None, do_rif=None, debug=False):
        c = self.cfg
        L = c.L if L is None else int(L)
        x = unit(np.asarray(x, dtype=np.float32))
        cvec = unit(self.c)
        u = np.concatenate([x, cvec, self._timecells()]).astype(np.float32)
        if self.E.n == 0:
            return {"action": "abstain", "reason": "empty store", "m1": 0.0, "m2": 0.0,
                    "nu": 0.0, "idx": -1, "pred": None, "n_considered": 0, "reinstated": False}
        Mn = self.E.M[:, : self.E.n]
        # exp36c：情境作为吸引子能量偏置（与 exp36/36b 的「证据预偏置」相对）。
        # 机制归因：query() 里 ev 的情境项只进入 softmax(ev) 的初始分布 a，但 L>0 分支
        # 随后用 E.retrieve 的**纯内容**吸引子结果 a_hop 覆盖 a，因此预偏置对最终读出无效
        # （exp36b dev 全 κ P1=0 的根因）。ctx_in_energy=True 时把同一偏置作为逐槽能量项
        # 传入 E.retrieve 的每次 softmax——情境参与能量地形本身。默认 False：路径逐位不变。
        ebias = None
        if c.use_ctx_bias and c.ctx_in_energy and self.E.Ctx.shape[0] > 0:
            b_ctx = self.E.Ctx[:, : self.E.n].T @ cvec
            b_ctx = (b_ctx - b_ctx.mean()) / (b_ctx.std() + 1e-8)
            ebias = c.kappa_ctx * np.maximum(b_ctx, 0.0)

        # ---- 期望库：每个模板给出一个编码假设，取证据最强者（多情境记忆，见 L-014）----
        best = None
        for k_te, xe_t in enumerate(self.enc.resid_read_all(x)):
            s_pre_t = self.enc.pre(xe_t)
            s_ev_t = self.enc.raw(xe_t) if c.dense_raw else s_pre_t
            qq_t = self.enc.encode_from_pre(s_pre_t)
            ed = Mn.T @ s_ev_t
            es = Mn.T @ qq_t
            ev_t = (c.w_dense * ed / (np.linalg.norm(ed) + 1e-12)
                    + (1.0 - c.w_dense) * es / (np.linalg.norm(es) + 1e-12))
            if c.use_ctx_bias and self.E.Ctx.shape[0] > 0:
                # 情境项必须与内容证据**同尺度**才可加。内容项 ev_t 是归一化后的
                # 加权和（量级 ~1e-3 到 1e-2），而 Ctx·cvec 是余弦（量级 ~1）。
                # 直接相加等于把内容证据完全淹没（实测 kappa=10 时 B 组 bias 已到 0.98，
                # 仍不足以改变 argmax，因为 A/B 两组内容证据差了三个数量级）。
                # 正确做法：把情境项也做组内标准化，再按 kappa 混合。（D-027）
                b_ctx = self.E.Ctx[:, : self.E.n].T @ cvec
                b_ctx = (b_ctx - b_ctx.mean()) / (b_ctx.std() + 1e-8)
                ev_t = ev_t + c.kappa_ctx * np.maximum(b_ctx, 0.0)
            sc_t = float(ev_t.max())
            if best is None or sc_t > best[0]:
                best = (sc_t, ev_t, s_pre_t, qq_t, k_te, s_ev_t)
        _, ev, s_pre, q0, k_te, s_ev = best
        reinstated = bool(k_te > 0)
        a = softmax_beta(ev, c.beta)
        # 吸引子补全：给出干净的 q_star 以估计新颖性，并作为收敛参考
        q_star, a_hop = self.E.retrieve(q0, L=L, beta=c.beta, dense=s_pre, ebias=ebias)
        if L > 0:
            a = a_hop * self.E.sup[: self.E.n]
            a = a / (a.sum() + 1e-12)
        # 皮层先验：与条目码的余弦。注意这里是**稀疏码之间**的余弦，
        # 这类码的余弦在结构上就接近 1（码近正交但都非负稀疏），因此必须先做**组内标准化**
        # 再融合。早先版本直接 clip(s,0,1) 融合：40 个候选的 s 全部落在 0.99 附近，
        # 于是融合分被语义项主导，吸引子证据被淹没 —— 表现为"库内/库外查询的 m1 完全相同"
        # （实测 in-store m1 ∈ [0.600,0.612]、novel m1 ∈ [0.600,0.616]），弃权永不触发（L-019）。
        shat = unit(self.sem.forward(x))
        s_raw = Mn.T @ shat
        s_z = (s_raw - s_raw.mean()) / (s_raw.std() + 1e-8)
        s_n = 1.0 / (1.0 + np.exp(-s_z))          # 组内 z -> [0,1]
        # 注意：抑制因子 sup 已在上面（L>0 分支）乘过一次，这里不再乘。
        # 早先两处都乘会让抑制变成平方效应（对抗评审 P2-8）。
        a = a / (a.sum() + 1e-12)
        f = c.lam * a + (1.0 - c.lam) * s_n
        order = np.argsort(-f)
        top1 = int(order[0])
        m1 = float(f[top1])
        m2 = float(f[order[1]]) if len(order) > 1 else 0.0
        nu_q = self.enc.novelty(x, q_star)
        recon = unit(self.E.D[:, : self.E.n] @ a) if self.E.n else np.zeros_like(s_pre)
        raw_evidence = float(unit(s_pre) @ recon)
        action, reason = self.ctl.decide_retrieval(m1, m2, nu_q, raw_evidence=raw_evidence)
        # raw_evidence = 吸引子重建匹配度：把收敛后的软分配 a 解码回稠密码，
        # 再看它与本次查询的稠密驱动有多像。这是"这条记忆能否解释当前输入"的直接度量。
        # 实测（40 库内 + 40 库外、35% 特征缺失）：库内 0.802±0.067，库外 0.393±0.046，
        # 分离度 > 3 sigma —— 是目前唯一可靠的弃权依据（D-026）。
        # 反例（都试过、都失败）：a 的峰值恒为 1.000（40 个候选时 softmax 总有一个主导）；
        # 语义先验的原始余弦在 40 个候选上都≈1（必须组内标准化才有排序信息）；
        # 新颖性 nu 在库内 0.983 / 库外 0.997，仅差 1.4%。
        out = {"action": action, "reason": reason, "m1": m1, "m2": m2, "nu": nu_q,
               "raw_evidence": raw_evidence,
               "idx": -1, "pred": None, "n_considered": int(self.E.n),
               "reinstated": reinstated}
        if debug:  # exp35 calibration: additive output only, no change to computation
            out["dbg"] = {"a": a, "ev": ev, "order": order[:16]}

        if action in ("recall", "separate"):
            out["idx"] = top1
            if self.E.d_p:
                out["pred"] = self.E.P[:, top1].copy()
            rif = c.rif_on if do_rif is None else do_rif
            if rif:
                self.E.suppress(a, rho_rif=c.rho_rif)
        return out

    def develop(self, X, epochs=None):
        """发育期：只跑前向、让稳态阈值 theta 收敛（不写入情景）。"""
        c = self.cfg
        epochs = c.homeo_epochs if epochs is None else int(epochs)
        z = np.zeros(c.d_c, dtype=np.float32)
        tb = self._timecells()
        U = [unit(np.asarray(x, dtype=np.float32)) for x in X]
        return self.enc.develop(U, epochs=epochs)

    # ---------------- 巩固 ----------------
    def query_events(self, t0=None, t1=None, sender=None, session=None):
        """事件层检索（D-042）：按时间窗/来源过滤事件记录。"""
        return self.E.query_events(t0=t0, t1=t1, sender=sender, session=session)

    def consolidate(self, cycles=1, epochs=None):
        return self.cons.sleep(float(self.t), cycles=cycles)

    def forget(self, tau_f=50.0):
        return self.E.prune(float(self.t), keep_free=0, tau_f=tau_f)

    # ---------------- 诊断 ----------------
    def stats(self):
        return {"t": self.t, "n_episodes": self.E.n, **self.E.dump(),
                "wm_load": self.wm.load(), "gate_mean": float(np.mean(self.gate.history[-100:])) if self.gate.history else 0.0}
