"""G：VTA / 杏仁核 / LC 的写入门控（M7_NEU）。"""
from __future__ import annotations
import numpy as np
from .ops import sigmoid


class WriteGate:
    """g_t = sigma(w_nu*nu + w_rho*RPE + w_eps*eps + w_a*activity + b_g)

    可测含义：门控把 重复输入 关掉（novelty 低），把 惊奇/奖赏 打开。
    消融臂：constant=True 时 g_t 恒为 1（相当于没有调质系统）。
    """

    def __init__(self, constant=False, w=(2.2, 1.4, 0.6, 0.8), b=-1.2):
        self.constant = bool(constant)
        self.w_nu, self.w_rho, self.w_eps, self.w_a = w
        self.b = float(b)
        self.history = []

    def __call__(self, nu, rpe, eps, activity):
        if self.constant:
            g = 1.0
        else:
            z = self.w_nu * nu + self.w_rho * rpe + self.w_eps * eps + self.w_a * activity + self.b
            g = float(sigmoid(z))
        self.history.append(g)
        return g
