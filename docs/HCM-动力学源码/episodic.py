"""E：CA3 情节快存（M2_EPI）。稠密联想记忆（现代 Hopfield）+ 元数据表 + 遗忘算子接入点。"""
from __future__ import annotations
import numpy as np
from .ops import unit, kwta, softmax_beta, cosine


class EpisodicStore:
    """M: (d_h, N_max) 列向量为归一化稀疏码；meta 为结构化并行数组。"""

    def __init__(self, d_h, capacity, k, beta=25.0, seed=0, d_p=0):
        self.d_h, self.capacity, self.k = int(d_h), int(capacity), int(k)
        self.d_p = int(d_p)
        self.P = np.zeros((self.d_p, self.capacity), dtype=np.float32)  # 每条情节的原始输入（供解码/巩固）
        self.Ctx = np.zeros((0, self.capacity), dtype=np.float32)  # 情境向量（用于情境复现）
        self.Tau = np.zeros((0, self.capacity), dtype=np.float32)  # 时间细胞基（用于时间复现）
        self.beta = float(beta)
        self.M = np.zeros((self.d_h, self.capacity), dtype=np.float32)  # DG-CA3 稀疏通路
        self.D = np.zeros((self.d_h, self.capacity), dtype=np.float32)  # EC→CA3 直通稠密通路
        self.n = 0
        self.t = np.zeros(self.capacity, dtype=np.float64)
        self.g = np.zeros(self.capacity, dtype=np.float64)
        self.r = np.zeros(self.capacity, dtype=np.float64)
        self.nu = np.zeros(self.capacity, dtype=np.float64)
        self.use = np.zeros(self.capacity, dtype=np.int64)
        self.sup = np.ones(self.capacity, dtype=np.float64)  # 抑制因子（RIF）
        self.rng = np.random.default_rng(seed)
        self.n_pruned = 0
        self.n_write_attempts = 0
        # ---- 事件层（D-042，exp30）：内容码与事件身份分离 ----
        self.cid = np.full(self.capacity, -1, dtype=np.int64)   # 槽位 -> 稳定内容码 id
        self._cid2slot = {}                                     # 稳定码 id -> 当前槽位（prune 后重建）
        self._next_cid = 0
        self.events = []   # 事件记录：{eid,t,sender,session,code_id,is_new_code}，每条输入恒写

    # ---------- 写入 ----------
    def max_cos(self, h):
        if self.n == 0:
            return -1.0
        return float(np.max(self.M[:, : self.n].T @ h))

    def write(self, h, t, g, r=0.0, nu=0.0, theta_dup=0.95, payload=None, dense=None, ctx=None, tau=None):
        self.n_write_attempts += 1
        if self.max_cos(h) >= theta_dup:
            return -1  # 重复，不写
        if self.n >= self.capacity:
            self.prune(t, keep_free=1)
            if self.n >= self.capacity:
                return -1
        i = self.n
        self.M[:, i] = h.astype(np.float32)
        if dense is not None:
            self.D[:, i] = dense.astype(np.float32)
        self.t[i], self.g[i], self.r[i], self.nu[i] = t, g, r, nu
        self.use[i] = 0
        self.sup[i] = 1.0
        self.cid[i] = self._next_cid
        self._cid2slot[self._next_cid] = i
        self._next_cid += 1
        if payload is not None and self.d_p:
            self.P[:, i] = payload
        if ctx is not None:
            if self.Ctx.shape[0] == 0:
                self.Ctx = np.zeros((len(ctx), self.capacity), dtype=np.float32)
                self.Tau = np.zeros((len(tau), self.capacity), dtype=np.float32)
            self.Ctx[:, i] = ctx
            self.Tau[:, i] = tau
        self.n += 1
        return i

    # ---------- 事件层（D-042）----------
    def write_event(self, h, t, g=0.0, r=0.0, nu=0.0, theta_dup=0.95, allow_new=True,
                    payload=None, dense=None, ctx=None, tau=None, sender="", session=""):
        """恒写事件记录；内容码按 theta_dup 去重复用，新码创建受 allow_new（门控）约束。
        返回 (eid, slot)：slot 为 None 表示事件已记录但无内容码（重复码被拒或门控拒写）。"""
        sc = self.M[:, : self.n].T @ h if self.n else np.zeros(0, dtype=np.float32)
        j = int(np.argmax(sc)) if sc.size else -1
        if j >= 0 and float(sc[j]) >= theta_dup:
            eid, slot, is_new = self._add_event(t, sender, session, int(self.cid[j]), False)
            return eid, slot
        i = -1
        if allow_new:
            i = self.write(h, t, g, r, nu, theta_dup=theta_dup, payload=payload,
                           dense=dense, ctx=ctx, tau=tau)
        cid = int(self.cid[i]) if i >= 0 else None
        eid, _slot, _new = self._add_event(t, sender, session, cid, i >= 0)
        return eid, (i if i >= 0 else None)

    def _add_event(self, t, sender, session, code_id, is_new_code):
        eid = len(self.events)
        self.events.append({"eid": eid, "t": float(t), "sender": str(sender),
                            "session": str(session), "code_id": code_id,
                            "is_new_code": bool(is_new_code)})
        return eid, self._cid2slot.get(code_id) if code_id is not None else None, is_new_code

    def query_events(self, t0=None, t1=None, sender=None, session=None):
        """时间/来源检索作用于事件层；slot 为当前槽位（码被 prune 后为 None）。"""
        out = []
        for ev in self.events:
            if t0 is not None and ev["t"] < t0:
                continue
            if t1 is not None and ev["t"] > t1:
                continue
            if sender is not None and ev["sender"] != sender:
                continue
            if session is not None and ev["session"] != session:
                continue
            r2 = dict(ev)
            r2["slot"] = self._cid2slot.get(ev["code_id"]) if ev["code_id"] is not None else None
            out.append(r2)
        return out

    # ---------- 检索（模式完成通道）----------
    def retrieve(self, q=None, L=3, beta=None, anneal=False, dense=None, w_dense=0.0, ebias=None):
        beta = self.beta if beta is None else beta
        if self.n == 0:
            return q.copy(), np.zeros(0)
        Mn = self.M[:, : self.n]
        # 现代 Hopfield 迭代：z <- unit(M softmax(beta M^T z))，全程连续、不重新稀疏化。
        # 初值用 EC→CA3 稠密驱动（鲁棒）；实测以稀疏码作初值显著变差（D-020）。
        z = unit(np.maximum(dense, 0.0)) if dense is not None else unit(q)
        # ebias: 可选的逐槽能量偏置（情境检索，exp36c）。加在每次 softmax 的 logits 上，
        # 使情境参与吸引子能量地形本身；None（默认）时路径与旧实现逐位一致。
        a = softmax_beta(Mn.T @ z + (0.0 if ebias is None else ebias), beta)
        for _ in range(max(0, int(L))):
            z = unit(Mn @ a)
            a = softmax_beta(Mn.T @ z + (0.0 if ebias is None else ebias), beta)
            sa = a.sum()
            if sa > 0:
                a = a / sa
        return z, a

    def scores(self, h):
        if self.n == 0:
            return np.zeros(0)
        return self.M[:, : self.n].T @ h * self.sup[: self.n]

    # ---------- 遗忘（Gamma）----------
    def importance(self, now, tau_f=50.0):
        if self.n == 0:
            return np.zeros(0)
        dt = np.maximum(now - self.t[: self.n], 0.0)
        return self.g[: self.n] * np.exp(-dt / max(tau_f, 1e-8)) * (1.0 + self.use[: self.n])

    def prune(self, now, keep_free=0, tau_f=50.0):
        target = self.capacity - keep_free
        if self.n <= target:
            return 0
        imp = self.importance(now, tau_f)
        keep = np.sort(np.argsort(-imp)[:target])
        lost = self.n - len(keep)
        for newi, oldi in enumerate(keep):
            if newi != oldi:
                self.M[:, newi] = self.M[:, oldi]
                self.t[newi], self.g[newi], self.r[newi] = self.t[oldi], self.g[oldi], self.r[oldi]
                self.nu[newi], self.use[newi], self.sup[newi] = self.nu[oldi], self.use[oldi], self.sup[oldi]
                self.D[:, newi] = self.D[:, oldi]
                if self.Ctx.shape[0]:
                    self.Ctx[:, newi] = self.Ctx[:, oldi]
                    self.Tau[:, newi] = self.Tau[:, oldi]
                if self.d_p:
                    self.P[:, newi] = self.P[:, oldi]
                self.cid[newi] = self.cid[oldi]
        self.n = len(keep)
        self._cid2slot = {int(self.cid[i]): i for i in range(self.n) if self.cid[i] >= 0}
        self.n_pruned += lost
        return lost

    def suppress(self, a, rho_rif=0.15, topk=5):
        """检索诱发抑制：命中之后，压制同时激活的竞争者（不改 use 计数）。"""
        if self.n == 0:
            return 0
        order = np.argsort(-a)[:topk]
        if len(order) == 0:
            return 0
        winner = order[0]
        others = [i for i in order[1:] if i != winner]
        self.use[winner] += 1
        for i in others:
            self.sup[i] *= (1.0 - rho_rif)
        return len(others)

    def disinhibit(self, idx, rho_rif=0.15):
        self.sup[idx] = min(1.0, self.sup[idx] / max(1e-6, 1.0 - rho_rif))

    def dump(self):
        return {
            "n": int(self.n), "capacity": int(self.capacity),
            "n_pruned": int(self.n_pruned), "write_attempts": int(self.n_write_attempts),
            "mean_use": float(self.use[: self.n].mean()) if self.n else 0.0,
            "n_events": len(self.events),
            "n_event_codes_reused": sum(1 for ev in self.events if not ev["is_new_code"]),
        }
