"""Theta：新皮层语义慢存（M5_SEM）。低秩双线性 + 突触成熟度级联代理（多时间尺度）。"""
from __future__ import annotations
import numpy as np
from .ops import unit


class SemanticStore:
    """f(x) = B tanh(A x)，A: (d_z, d_x), B: (d_h, d_z)。

    omega 为逐坐标成熟度 in [0,1]：eta_eff = eta0 / (1 + kappa*omega)。
    新坐标 omega=0 -> 学得快；反复更新后 omega->1 -> 几乎冻结。
    这正是 Benna-Fusi 级联的现象学代理：产生幂律式遗忘（见 docs 第 11 节诚实清单第 3 条）。
    """

    def __init__(self, d_x, d_h, d_z, seed=0, kappa=8.0, u_meta=0.02, theta_turn=0.9):
        rng = np.random.default_rng(seed)
        self.d_x, self.d_h, self.d_z = int(d_x), int(d_h), int(d_z)
        self.A = (rng.standard_normal((self.d_z, self.d_x)) * (1.0 / np.sqrt(self.d_x))).astype(np.float32)
        self.B = (rng.standard_normal((self.d_h, self.d_z)) * (1.0 / np.sqrt(self.d_z))).astype(np.float32)
        self.omega_A = np.zeros_like(self.A)
        self.omega_B = np.zeros_like(self.B)
        self.kappa = float(kappa)
        self.u_meta = float(u_meta)
        self.theta_turn = float(theta_turn)
        self.n_updates = 0

    def forward(self, x, with_hidden=False):
        t = np.tanh(self.A @ x)
        y = self.B @ t
        if with_hidden:
            return y, t
        return y

    def features(self, x):
        return np.tanh(self.A @ x)

    def train_step(self, x, h_target, eta=0.05, wd=1e-5):
        """一次 SGD 步。返回重建损失。"""
        t = np.tanh(self.A @ x)
        y = self.B @ t
        diff = y - h_target
        loss = float(diff @ diff)
        # 梯度裁剪：学习率稍大时 2*diff 会指数放大（实测 eta=0.2 在第 2 个周期溢出到 inf）。
        diff = np.clip(diff, -3.0, 3.0)
        dB = 2.0 * np.outer(diff, t)
        dz = (self.B.T @ (2.0 * diff)) * (1.0 - t * t)
        dA = np.outer(dz, x)
        dB = np.clip(dB, -1.0, 1.0)
        dA = np.clip(dA, -1.0, 1.0)
        etaA = eta / (1.0 + self.kappa * self.omega_A)
        etaB = eta / (1.0 + self.kappa * self.omega_B)
        self.A -= (etaA * dA).astype(np.float32) + wd * self.A
        self.B -= (etaB * dB).astype(np.float32) + wd * self.B
        # 只有真正被更新的坐标才成熟（稀疏的更替 -> 幂律保持）
        bigA = np.abs(dA) > 1e-6
        bigB = np.abs(dB) > 1e-6
        self.omega_A += self.u_meta * bigA * (1.0 - self.omega_A)
        self.omega_B += self.u_meta * bigB * (1.0 - self.omega_B)
        # 突触更替（turnover）：级联成熟的坐标若仍被持续驱动，就"换新突触"——成熟度归零。
        # 没有这一项，级联会把可塑性永久锁死，新任务学不进去（E4 第一轮 B 全灭的根因）。
        self.omega_A = np.where((self.omega_A > self.theta_turn) & bigA, 0.0, self.omega_A)
        self.omega_B = np.where((self.omega_B > self.theta_turn) & bigB, 0.0, self.omega_B)
        self.n_updates += 1
        return loss

    def batch_fit(self, X, H, epochs=4, eta=0.05, verbose=False):
        """X: (n, d_x), H: (n, d_h) 目标稀疏码。交错 SGD（每 epoch 打乱）。"""
        rng = np.random.default_rng(0)
        loss = 0.0
        for ep in range(int(epochs)):
            order = rng.permutation(len(X))
            for i in order:
                loss += self.train_step(X[i], H[i], eta=eta)
        return loss / max(1, len(X) * int(epochs))

    def reconstruction_cos(self, X, H):
        out = []
        for i in range(len(X)):
            y = self.forward(X[i])
            out.append(float(y @ H[i] / (np.linalg.norm(y) * np.linalg.norm(H[i]) + 1e-8)))
        return float(np.mean(out)) if out else 0.0

    def freeze(self):
        self.kappa = 1e9
