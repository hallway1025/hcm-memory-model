"""W：PFC 工作记忆（M6_WM）。容量受限 k-WTA 槽位。"""
from __future__ import annotations
import numpy as np


class WorkingMemory:
    """C 个槽位，每槽一个稀疏码 + 激活权重。超出容量时权重最低者被替换。

    容量 C 的可测含义：负荷 > C 时探针准确率阶梯式下跌（预测 P5）。
    """

    def __init__(self, d_h, C=4, eta_w=0.35):
        self.d_h, self.C, self.eta_w = int(d_h), int(C), float(eta_w)
        self.slots = np.zeros((self.C, self.d_h), dtype=np.float32)
        self.w = np.zeros(self.C, dtype=np.float64)
        self.occ = np.zeros(self.C, dtype=bool)
        self.theta_same = 0.90

    def reset(self):
        self.slots[:] = 0
        self.w[:] = 0
        self.occ[:] = False

    def load(self):
        return int(self.occ.sum())

    def present(self, h):
        """送入一个条目；返回 True 表示保留在槽中。"""
        if self.occ.any():
            sims = self.slots[self.occ] @ h
            idxs = np.where(self.occ)[0]
            if sims.size and sims.max() >= self.theta_same:
                j = idxs[int(np.argmax(sims))]
                self.w[j] += 0.15
                return True
        free = np.where(~self.occ)[0]
        if free.size and self.load() < self.C:
            j = int(free[0])
        else:
            j = int(np.argmin(self.w))
        self.slots[j] = h
        self.w[j] = 1.0
        self.occ[j] = True
        # 槽位竞争：与本次输入不相似的槽位权重衰减
        sims = self.slots @ h
        self.w = np.maximum(self.w + self.eta_w * (sims - sims.mean()), 0.0)
        self.w[j] = max(self.w[j], 1.0)
        return True

    def probe(self, q):
        """用部分线索在槽内检索；返回 (是否找到, 最大相似度)。"""
        if not self.occ.any():
            return False, 0.0
        sims = self.slots[self.occ] @ q
        m = float(np.max(sims))
        return m > 0.5, m

    def contents(self):
        return self.slots[self.occ].copy()
