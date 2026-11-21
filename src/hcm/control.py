"""Pi：前额叶-海马 元控制（M0_META）。代价敏感的动作选择。"""
from __future__ import annotations
import numpy as np


ACTIONS = ("encode", "retrieve", "consolidate", "forget", "abstain")


class Metacontroller:
    """ECost(a) = E[error|a] + c_a * c_unit，取 argmin。

    关键可检验点：抬高 c_abst 时，弃权率下降、错误率上升（风险偏好的对应物）。
    """

    def __init__(self, c_unit=1.0, c_err=1.0, c_abst=0.3, theta_rec=0.25, margin=0.06,
                 always_retrieve=False, no_forget=False, theta_raw=0.50):
        self.c_unit, self.c_err, self.c_abst = float(c_unit), float(c_err), float(c_abst)
        self.theta_rec, self.margin = float(theta_rec), float(margin)
        # theta_raw 作用在 raw_evidence = lam*a_max + (1-lam)*exp(-nu) 上，量纲稳定（[0,1]）
        self.theta_raw = float(theta_raw)
        self.always_retrieve = bool(always_retrieve)  # 消融臂：永不弃权
        self.no_forget = bool(no_forget)

    def costs(self, m1=0.0, m2=0.0, nu=0.0, g=0.0, n=0, L=6, d_h=1024, forget_pressure=0.0):
        gap = max(0.0, m1 - m2)
        c_ret = 0.15 + 0.5 * min(1.0, (L * n) / max(1.0, 20.0 * d_h)) + (1.0 - m1) * self.c_err
        c_enc = 0.3 + 0.7 * min(1.0, max(0.0, g))
        c_con = 0.5
        c_for = 0.2 + 0.6 * forget_pressure
        c_abs = self.c_abst
        return {"encode": c_enc, "retrieve": c_ret, "consolidate": c_con,
                "forget": c_for, "abstain": c_abs}, gap

    def decide_retrieval(self, m1, m2, nu, raw_evidence=None):
        """返回 (action, reason)。

        弃权判据优先用 raw_evidence（绝对证据，未经组内标准化）；
        只有当调用方不提供它时，退回用 m1（向后兼容）。
        """
        if self.always_retrieve:
            return "recall", "ablation: always retrieve"
        crit = m1 if raw_evidence is None else raw_evidence
        if crit < self.theta_raw:
            return "abstain", "low absolute evidence raw<theta_raw"
        if (m1 - m2) < self.margin and nu > 0.5:
            return "separate", "near-tie + high novelty -> pattern separation"
        return "recall", "confident"

    def choose_action(self, **kw):
        if self.always_retrieve:
            return "retrieve"
        c, _ = self.costs(**kw)
        if self.no_forget:
            c.pop("forget", None)
        return min(c, key=c.get)
