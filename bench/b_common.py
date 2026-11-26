"""公共模块：配置、BCa bootstrap、结果落盘（方向 B 互补收益验证实验）。

来源：执行方案_互补性验证实验.md 第三节（TrialConfig / bca_bootstrap / ResultStore 原文）。
run_trial 已接入 HCM 训练/评测管线（L0 几何，纯 CPU）。
scipy 不可用时内置正态分布回退（Acklam 逆变换 + erf），口径与 scipy.stats.norm 一致。
"""

import json
import math
import os
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Callable, Dict, List


@dataclass
class TrialConfig:
    """一次训练运行的完整配置"""
    # —— 预算 ——
    total_bytes: int = 100_000        # M：总字节预算
    lam: float = 0.5                  # λ = |B1|/M，快层占总预算比例
    replay_budget: float = 1.0        # r：回放强度（样本数或比例）
    consol_budget: float = 1.0        # c：巩固频率/处理量
    # —— 结构 ——
    layers: int = 2                   # 1=单层（基线）, 2=双层（HCM）
    comp_fidelity: float = 1.0        # d_c/d：慢层压缩保真度
    comp_strategy: str = "prototype"  # prototype | quantization
    # —— 数据流 ——
    event_structure: str = "structured"  # structured | shuffled
    task_order: str = "default"          # 任务顺序标识
    # —— 运行 ——
    seed: int = 0
    extra: dict = field(default_factory=dict)

    def key(self) -> str:
        """唯一标识，用于缓存/去重"""
        return (f"M{self.total_bytes}_lam{self.lam:.2f}_r{self.replay_budget:.2f}"
                f"_c{self.consol_budget:.2f}_L{self.layers}_f{self.comp_fidelity:.3f}"
                f"_{self.comp_strategy}_{self.event_structure}_{self.task_order}_s{self.seed}")


# ============================================================
# 正态分布（scipy 优先，缺失时回退）
# ============================================================
def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_ppf(p):
    """Acklam 逆正态 CDF（无 scipy 回退，相对精度 ~1e-9）。"""
    p = min(max(float(p), 1e-15), 1.0 - 1e-15)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q /            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)


try:
    from scipy.stats import norm as _scipy_norm
    _CDF = _scipy_norm.cdf
    _PPF = _scipy_norm.ppf
except Exception:
    _CDF = _norm_cdf
    _PPF = _norm_ppf


# ============================================================
# BCa Bootstrap
# ============================================================
def bca_bootstrap(data: np.ndarray, stat_fn: Callable, n_boot: int = 5000,
                  alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    """
    BCa (bias-corrected and accelerated) bootstrap 置信区间。

    比 percentile bootstrap 更准，Equal-Byte 用的也是 BCa，
    对齐口径才能直接对话。

    data: 一维数组（如 10 个 seed 的 AA）
    stat_fn: 统计量函数，如 np.mean / np.median
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    theta_hat = stat_fn(data)

    # bootstrap 分布
    boot = np.array([
        stat_fn(rng.choice(data, size=n, replace=True))
        for _ in range(n_boot)
    ])

    # 偏差校正 z0
    prop = np.mean(boot < theta_hat)
    prop = np.clip(prop, 1e-6, 1 - 1e-6)
    z0 = _PPF(prop)

    # 加速 a（jackknife）
    jack = np.array([stat_fn(np.delete(data, i)) for i in range(n)])
    jack_mean = jack.mean()
    num = np.sum((jack_mean - jack) ** 3)
    den = 6 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0

    # 分位点
    z_lo, z_hi = _PPF(alpha / 2), _PPF(1 - alpha / 2)
    def adj(z):
        return _CDF(z0 + (z0 + z) / (1 - a * (z0 + z)))
    p_lo, p_hi = adj(z_lo), adj(z_hi)

    lo = np.percentile(boot, 100 * p_lo)
    hi = np.percentile(boot, 100 * p_hi)

    return {
        "estimate": float(theta_hat),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "ci_level": 1 - alpha,
        "n_boot": n_boot,
        "n_samples": n,
    }


# ============================================================
# 结果落盘
# ============================================================
class ResultStore:
    """增量落盘，跑一个存一个，中断不丢"""

    def __init__(self, path: str):
        self.path = path
        self.data: List[dict] = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.data = json.load(f)

    def add(self, cfg: TrialConfig, metrics: dict):
        self.data.append({"config": asdict(cfg), "key": cfg.key(), "metrics": metrics})
        self._flush()

    def _flush(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def get(self, **filters) -> List[dict]:
        out = []
        for rec in self.data:
            cfg = rec["config"]
            if all(cfg.get(k) == v for k, v in filters.items()):
                out.append(rec)
        return out

    def aa_by(self, **filters) -> np.ndarray:
        return np.array([r["metrics"]["AA"] for r in self.get(**filters)])


# ============================================================
# run_trial：已接入 HCM 训练/评测管线（L0 几何，纯 CPU）
# ============================================================
HCM_ROOT = "C:/Users/huawei/Desktop/Deepseek Harmness测试区/brain-memory-model"

# 预算口径（L0 几何：d_x=128, rho=8 → d_h=1024, d_c=128, d_tau=32）：
# 每条情景记录的实际占用 = M(d_h) + D(d_h) + P(d_x) + Ctx(d_c) + Tau(d_tau) 个 float32
REC_BYTES = 4 * (1024 + 1024 + 128 + 128 + 32)   # = 9344 B/条
# 语义层每个秩单位 = A 行(d_x) + B 列(d_h) 个 float32
RANK_BYTES = 4 * (128 + 1024)                     # = 4608 B/秩

K_CLASSES = 8
D_X = 128
NOISE = 0.35
N_CYCLES = 12                 # 持续覆盖压力：12 个训练周期
SAMPLES_PER_CYCLE = 8         # 每周期每类 1 条新样本
TEST_PER_CLASS = 6


def run_trial(cfg: TrialConfig) -> Dict[str, float]:
    """
    跑一次 HCM 训练 + 评测，返回指标字典。

    预算映射（等字节口径，必须真的按 cfg.total_bytes 限制两层容量）：
      - layers=2：λ = |B1|/M 切给情景层（快层）容量
        n_rec = floor(λ·M / REC_BYTES)
        1-λ 切给语义层（慢层）秩
        n_rank = floor((1-λ)·M / RANK_BYTES)
      - layers=1：单缓冲基线，全部预算给情景层（λ 无意义），
        语义秩固定 L0 默认 64（锚定端点）

    注意：HCM Config.lam 是读出混合权重（固定 0.6），与预算 λ 无关，切勿混淆。

    协议：12 周期持续覆盖流（每周期 8 条新样本 observe + 1 次巩固），
    评测 AA = 语义读出准确率（exp6b 口径：unit(sem.forward(x)) 与类别码余弦 argmax）。
    """
    import sys, time
    hcm_src = os.path.join(HCM_ROOT, "src")
    if hcm_src not in sys.path:
        sys.path.insert(0, hcm_src)
    from hcm import HCM, Config
    from hcm.ops import unit
    from hcm.data import random_patterns

    t0 = time.perf_counter()
    M = int(cfg.total_bytes)

    if int(cfg.layers) == 1:
        n_rec = max(1, M // REC_BYTES)
        n_rank = 64            # 单缓冲基线：慢层不占预算
    else:
        n_rec = max(1, int(cfg.lam * M) // REC_BYTES)
        n_rank = max(1, int((1.0 - cfg.lam) * M) // RANK_BYTES)

    hcm_cfg = Config(d_x=D_X, rho=8, a=0.02, d_z=n_rank, capacity=n_rec,
                     beta=20.0, L=3, use_resid=True, seed=int(cfg.seed), T_ctx=200)
    m = HCM(hcm_cfg)

    # 固定原型（跨 λ 可比），训练流按 seed 派生
    protos = random_patterns(K_CLASSES, D_X, seed=5)

    def sample_batch(n_per_class, seed):
        r = np.random.default_rng(seed)
        X, Y = [], []
        for k in range(K_CLASSES):
            for _ in range(n_per_class):
                X.append(unit(protos[k] + NOISE * unit(r.normal(size=D_X))))
                Y.append(k)
        return np.array(X, dtype=np.float32), np.array(Y)

    # 持续覆盖压力：每周期新样本 observe 入快层，随后巩固 1 次（快层溢出由 prune 承担）
    for cyc in range(N_CYCLES):
        Xb, Yb = sample_batch(SAMPLES_PER_CYCLE // K_CLASSES, 2000 + cyc)
        for x, y in zip(Xb, Yb):
            m.observe(x, r=1.0)
        m.consolidate(cycles=1)

    # 评测：语义读出 AA（exp6b 口径）
    CATS = np.array([m.enc.encode(unit(p)) for p in protos], dtype=np.float32)
    Xte, Yte = sample_batch(TEST_PER_CLASS, 202)
    ok = 0
    for x, y in zip(Xte, Yte):
        h = unit(np.asarray(m.sem.forward(x), dtype=np.float32))
        ok += int(int(np.argmax(CATS @ h)) == y)
    AA = ok / len(Yte)

    bytes_used = n_rec * REC_BYTES + (n_rank * RANK_BYTES if int(cfg.layers) == 2 else 0)
    st = m.stats()
    return {
        "AA": float(AA),
        "bytes_used": int(bytes_used),
        "n_rec_budget": int(n_rec),
        "n_rank_budget": int(n_rank),
        "n_episodes": int(st["n_episodes"]),
        "n_pruned": int(st["n_pruned"]),
        "train_time": float(time.perf_counter() - t0),
    }
