"""Psi：SWR 回放与海马-皮层巩固（M4_CON）。"""
from __future__ import annotations
import numpy as np
from .ops import softmax_beta


class Consolidator:
    """优先级采样 + 交错回放 + 皮层 SGD。

    p_i 正比于 (now - t_i + 1)^(-alpha) * sigmoid(nu_i)^b * (1 + r_i)^c
    uniform=True 是控制臂（均匀回放），用来证明优先级不是装饰。
    """

    def __init__(self, episodic, semantic, n_r=32, epochs=3, alpha=1.0, b=1.0, c=1.0,
                 T_p=0.5, interleave=0.5, eta=0.05, uniform=False, seed=0):
        self.E, self.S = episodic, semantic
        self.n_r, self.epochs = int(n_r), int(epochs)
        self.alpha, self.b, self.c = float(alpha), float(b), float(c)
        self.T_p, self.interleave, self.eta = float(T_p), float(interleave), float(eta)
        self.uniform = bool(uniform)
        self.rng = np.random.default_rng(seed)
        self.rehearsal = []  # list of (x, h) 旧/语义回灌缓冲
        self.history = []

    def priority(self, now):
        E = self.E
        n = E.n
        if n == 0:
            return np.zeros(0)
        rec = (now - E.t[:n] + 1.0) ** (-self.alpha)
        sur = (1.0 / (1.0 + np.exp(-E.nu[:n]))) ** self.b
        rew = (1.0 + np.maximum(E.r[:n], 0.0)) ** self.c
        p = rec * sur * rew
        return p / (p.sum() + 1e-12)

    def sample(self, now, n):
        E = self.E
        if E.n == 0:
            return np.zeros(0, dtype=int)
        if self.uniform:
            return self.rng.choice(E.n, size=min(n, E.n), replace=False)
        p = self.priority(now)
        return self.rng.choice(E.n, size=min(n, E.n), replace=False, p=p)

    def cycle(self, now, epochs=None):
        """一个睡眠周期。返回 (重建损失, 被回放的条目数)。"""
        E, S = self.E, self.S
        epochs = self.epochs if epochs is None else epochs
        if E.n == 0:
            return 0.0, 0
        n_new = max(1, int(round(self.n_r * (1.0 - self.interleave))))
        idx = self.sample(now, n_new)
        X = [E.P[:, i].copy() for i in idx] if E.d_p else []
        H = [E.M[:, i].copy() for i in idx]
        # 交错：从再激活缓冲取旧样本（防止覆盖式遗忘）
        n_old = self.n_r - len(idx)
        if self.rehearsal and n_old > 0:
            take = self.rng.choice(len(self.rehearsal), size=min(n_old, len(self.rehearsal)), replace=False)
            for j in take:
                xj, hj = self.rehearsal[j]
                X.append(xj)
                H.append(hj)
        if not X:
            return 0.0, 0
        X = np.asarray(X, dtype=np.float32)
        H = np.asarray(H, dtype=np.float32)
        loss = S.batch_fit(X, H, epochs=epochs, eta=self.eta)
        # 回放过的条目进入再激活缓冲，供后续交错使用
        for x, h in zip(X, H):
            self.rehearsal.append((x.copy(), h.copy()))
        cap = max(self.n_r * 8, 64)
        if len(self.rehearsal) > cap:
            self.rehearsal = self.rehearsal[-cap:]
        for i in idx:
            E.use[i] += 1
        self.history.append(loss)
        return float(loss), int(len(idx))

    def sleep(self, now, cycles=1):
        out = []
        for _ in range(int(cycles)):
            out.append(self.cycle(now))
        return out
