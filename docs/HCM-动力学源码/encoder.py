"""Phi：内嗅皮层 II/III -> 齿状回 的扩张稀疏编码器（M1_SEP）。

关键设计（由 tests/diag_common_space.py 的实验决定，见 memory/20-DECISIONS.md D-016）：
同一条 EC->CA3 投射提供两种读出，而不是两条互不相干的通路。
  (a) 写入侧分离：h = unit(kWTW_k(max(W_e u - theta, 0)))，稀疏码存进 CA3。
  (b) 查询侧鲁棒：s = max(W_e u - theta, 0)，稠密前激活直接与稀疏存储码做线性读出。
理由：稠密码对稠密码的读出在输入相似度 0.9 时无法分辨（cos 0.986），
      而稀疏码对稠密驱动的读出同时继承"分离"（码间低重叠）与"鲁棒"（不取 top-k）。
theta 是每单元稳态阈值（内在可塑性 / 稳态缩放，Turrigiano 1998），
由 delta 规则 theta += eta*(h - k/d_h) 在线更新；它是把 input cos 0.986 的码
从 0.883 压到 0.626 的唯一有效机制（白化与反 Hebb 递推在本任务上均无效）。
"""
from __future__ import annotations
import numpy as np
from .ops import unit, relu_kwta, orthogonal_matrix


class Encoder:
    """u = [x ; c ; tau] -> s = [W_e u + b_e - theta]_+ -> h = unit(kWTW_k(s))。"""

    def __init__(self, d_x, d_c, d_tau, d_h, k, seed=0, expand=True, learnable=False,
                 norm="none", rel=None, homeo=False, homeo_eta=0.35, use_resid=True, mu_gamma=0.9,
                 bank_cap=8, switch_ratio=1.5, switch_min_gap=4):
        self.rng = np.random.default_rng(seed)
        self.d_x, self.d_c, self.d_tau = int(d_x), int(d_c), int(d_tau)
        # 码只吃内容 x：情境/时间作为独立证据项参与决策（见 D-017），不混进码里。
        self.d_u = self.d_x
        self.d_h, self.k = int(d_h), int(k)
        self.expand = bool(expand)
        self.learnable = bool(learnable)
        self.norm = str(norm)
        self.rel = rel
        self.homeo = bool(homeo)
        self.homeo_eta = float(homeo_eta)
        self.W_e = orthogonal_matrix(self.d_h, self.d_u, self.rng)
        self.W_s = orthogonal_matrix(self.d_h, self.d_x, self.rng)  # 仅作消融臂
        self.use_resid = bool(use_resid)
        self.mu_gamma = float(mu_gamma)
        # 期望库（expectation bank）：单一 mu 只能在"情境平稳"时成立。情境切换后旧条目的
        # 写入码是用旧 mu 算的，若查询仍用新 mu，写入码与查询码系统性失配（实测 A 阶段 20 条掉到 13/20，
        # 且两任务库中 B 全灭，见 L-014）。故保留多个期望模板，查询时逐模板假设 + 取最优。
        self.MU = [np.zeros(self.d_x, dtype=np.float32)]
        self.mu = self.MU[0]
        self.k_used = 0
        self.bank_cap = int(bank_cap)
        self.switch_ratio = float(switch_ratio)
        self.switch_min_gap = int(switch_min_gap)
        self.nu_ema = None
        self.nu_hist = []
        self._since = 10 ** 6
        self.b_e = np.zeros(self.d_h, dtype=np.float32)
        self.theta = np.zeros(self.d_h, dtype=np.float32)

    # ---- 情境残差阶段（M1a）：减去缓慢期望，突出共享成分之外的部分 ----
    def resid_write(self, x):
        """写入期：**总是**维护期望模板（即使不开残差编码），否则对照组的新颖性信号恒为 1。

        这是一个方法学要点：消融"残差编码"只应移除"用残差去编码"这一步，
        不应连带移除"维护期望"本身 —— 否则对照组拿到的是一个信息量恒为零的 ν，
        对比就变成了"有信号 vs 无信号"，而不是"用残差 vs 不用残差"。
        """
        x = unit(x)
        d = float(np.linalg.norm(x - self.mu))
        self.nu_ema = d if self.nu_ema is None else 0.9 * self.nu_ema + 0.1 * d
        self._since += 1
        # 两种切换信号：(a) 预测误差显著跳升；(b) 期望方向反转（cos < 0.3）——
        # 单一误差阈值会漏检"新情境的残差范数与旧情境相当"的情形（实测 A->B 比为 1.88 < 2.5）。
        reversal = False
        nmu = float(np.linalg.norm(self.mu))
        if nmu > 0.3:
            reversal = float(unit(x) @ unit(self.mu)) < 0.3
        jump = d > self.switch_ratio * (self.nu_ema + 1e-6)
        if (len(self.MU) < self.bank_cap and self._since >= self.switch_min_gap
                and (reversal or jump)):
            # 新模板从零开始（尚不知道新情境均值 = 暂时不减任何东西），随后由 EMA 收敛。
            # 若直接用第一帧作模板，该帧残差恰为 0，会写入一个退化码。
            self.MU.append(np.zeros(self.d_x, dtype=np.float32))
            self.k_used = len(self.MU) - 1
            self.mu = self.MU[self.k_used]
            self._since = 0
        else:
            self.mu = (self.mu_gamma * self.mu + (1.0 - self.mu_gamma) * x).astype(np.float32)
            self.MU[self.k_used] = self.mu
        return unit(x - self.mu) if self.use_resid else x

    def resid_read(self, x):
        """查询期：用当前活跃模板（单模板，兼容旧调用）。"""
        x = unit(x)
        return unit(x - self.mu) if self.use_resid else x

    def resid_read_all(self, x):
        """查询期：对每个期望模板给出一个候选输入，逐个假设再取最优（多情境记忆）。"""
        x = unit(x)
        if not self.use_resid:
            return [x]
        return [unit(x - m) for m in self.MU]

    @property
    def target(self):
        return self.k / float(self.d_h)

    def pre(self, u):
        """稠密 EC 驱动 s（不取 top-k）。写入与查询共用。"""
        s = self.W_e @ u + self.b_e - self.theta
        if self.norm == "global":
            s = s - s.mean()
        s = np.maximum(s, 0.0)
        if self.rel is not None and s.max() > 0:
            s = np.maximum(s - float(self.rel) * s.max(), 0.0)
        return s.astype(np.float32)

    def raw(self, u):
        """未过阈值/整流的线性投射：稠密鲁棒证据用。theta 只塑造稀疏码，不削稠密读出。"""
        return (self.W_e @ u + self.b_e).astype(np.float32)

    def encode(self, u):
        return unit(relu_kwta(self.pre(u), self.k))

    def encode_from_pre(self, s):
        return unit(relu_kwta(s, self.k))

    def adapt(self, h):
        """稳态阈值在线更新：被选中过多的单元阈值上升（不是下降）。"""
        if self.homeo:
            self.theta += (self.homeo_eta * (h - self.target)).astype(np.float32)

    def develop(self, U, epochs=6, eta=None):
        """⚠️ 码值语义 + 大 η 会发散/坍缩（diag_theta_protocol V1: 精确召回 0.000）。
        收敛语义是二值占空比（h_i∈{0,1}，复现者方案，η=0.05）。MODEL.md §3.2a。"""
        """发育期适配：在没有写入的纯前向阶段让 theta 收敛到目标稀疏度。"""
        eta = self.homeo_eta if eta is None else float(eta)
        for _ in range(int(epochs)):
            for u in U:
                h = self.encode(u)
                self.theta += (eta * (h - self.target)).astype(np.float32)
        return float(np.mean(self.theta))

    def sensory(self, x):
        """消融臂：只看内容、与稀疏码无关的独立稠密投射（旧设计，高重叠时无法分辨）。"""
        return unit(np.maximum(self.W_s @ x, 0.0))

    def novelty(self, x, h=None, mu=None, stored=None):
        """CA1 式新颖性 ν = 1 - cos(x, μ)：相对**期望**的偏离，不是"是否已存过"。

        这条信号被四代实现试错定下来（L-020），每次失效都留下了一个可复用的判据：

        1) ||x - W_x h|| / ||x||：动态范围仅 0.2%（0.998..1.000）→ 门控恒开，等于没有。
        2) 1 - cos(x, W_x h)：W_x 是**秩 1** 的 delta 解码器，刚见过的 x 其重建是纯量倍数，
           cos≡1 → ν 塌到 0.1，60 条输入只写进 1 条。
        3) 1 - max_i cos(h, stored_i)：这在**重复检测**的意义上是对的，但它与
           模式分离**结构性冲突**——高重叠输入本来就应该"像已存条目"，正是因此才需要分离。
           用它当门控会在 overlap 0.90 时拒写 4/30，召回从 25/30 掉到 0/30。
        4) 1 - cos(x, μ)（期望偏离，最终采用）：重复呈现 → μ 收敛到 x → ν→0 → 不写；
           相似但不同 → 相对 μ 的偏离依然存在 → 写。去重职责交给 store.write 的
           theta_dup（那才是该管"是不是同一个"的地方）。

        对应解剖：CA1 的新颖信号比较的是**内嗅皮层输入与 CA3 的预期**，不是"海马里有没有"。
        """
        x = np.asarray(x, dtype=np.float32)
        ref = mu if mu is not None else self.mu
        if ref is None or float(np.linalg.norm(ref)) < 1e-6:
            self.nu_hist.append(1.0)
            return 1.0
        cos = float(np.dot(x, ref) / (np.linalg.norm(x) * np.linalg.norm(ref) + 1e-8))
        raw = float(np.clip(1.0 - cos, 0.0, 2.0))

        # ---- 关键：对 ν 做**相对**归一化（近期水平的比值），而不是用绝对值 ----
        # 绝对期待偏离在"一整套新输入"上会整体偏低（μ 追不上整批新分布），
        # 于是门控把整批都判为"不新"，写入 1/30、召回 1/30（实测）。
        # 而生物的新颖信号是**预测误差的跳变**（LC/CA1 对突变的相位式响应），
        # 不是误差的绝对水平。故用 ratio = 当前 ν / 近期 ν 的 EMA：始终有约一半的
        # 输入被判为"比近期更新"，另一半判为"与近期相当"。
        self.nu_hist.append(raw)
        if len(self.nu_hist) > 200:
            self.nu_hist.pop(0)
        ref_level = float(np.mean(self.nu_hist[-20:])) if self.nu_hist else raw
        ratio = raw / (ref_level + 1e-6)
        return float(np.clip(ratio, 0.0, 2.0))

    def hebbian_update(self, u, h, eta=1e-3):
        if not self.learnable:
            return
        self.W_e += (eta * np.outer(h, u)).astype(np.float32)
        self.W_e /= (np.linalg.norm(self.W_e, axis=1, keepdims=True) + 1e-8)
