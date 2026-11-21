# HCM 动力学模型

> 源：brain-memory-model/src/hcm/（9 个子系统）
> 提取日期：2026-11-19

---

## 1. 总览：M = <X, Phi, E, Theta, W, Psi, Pi, G, Gamma>

HCM 把海马—皮层互补记忆架构装配为一个可执行的在线动力学系统。每个时间步 t，输入 x_t 经过编码、门控、写入、检索、工作记忆更新，形成完整的状态转移。

### 状态变量

| 符号 | 含义 | 维度 |
|------|------|------|
| **X** | 输入空间（原始模式） | d_x |
| **Phi** | 编码器（EC→DG 扩张稀疏投射） | W_e: (d_h, d_u) |
| **E** | 情景快存（CA3 联想记忆） | M: (d_h, N), D: (d_h, N) |
| **Theta** | 语义慢存（新皮层） | A: (d_z, d_x), B: (d_h, d_z) |
| **W** | 工作记忆（PFC 槽位） | (C, d_h) |
| **Psi** | 巩固（SWR 回放） | — |
| **Pi** | 元控制（前额叶决策） | — |
| **G** | 写入门控（VTA/杏仁核/LC） | — |
| **Gamma** | 遗忘算子（时衰+使用增强） | — |

### 配置（Config）

```
d_x=128, d_c=128, d_tau=32, rho=8
a=0.02, capacity=2000, C=4, beta=20.0
L=3, gamma_c=0.95, theta_write=0.5
theta_rec=0.25, margin=0.06, lam=0.6
```

**关键派生量：**
- **d_h = d_x × rho**（扩张维度，默认 1024）
- **k = round(a × d_h)**（稀疏度，默认 20）

### 预设层级

| 层级 | d_x | capacity | d_z | 用途 |
|------|-----|----------|-----|------|
| L0 | 128 | 2000 | 64 | 冒烟/单元 |
| L1 | 512 | 20000 | 128 | 常规实验 |
| L2 | 2048 | 500000 | 256 | 大库（exp37/52） |
| L3 | 8192 | 20000000 | 512 | 理论极限 |

---

## 2. 算子（ops.py）

### 2.1 归一化
```
unit(v) = v / (||v|| + ε),  ε = 1e-8
```

### 2.2 k-WTA 稀疏化
```
kwta(x, k): 保留 |x| 最大的 k 个分量，其余置 0（不回填、不缩放）
relu_kwta(x, k) = kwta(max(x, 0), k)
```

### 2.3 温度 softmax
```
softmax_beta(z, β) = exp(β·z) / Σ exp(β·z')
```
β 是检索温度——**核心标度参数**。β 越大，读出越尖锐（趋近 argmax）；β 越小，读出越平坦（趋近均匀）。

### 2.4 余弦
```
cosine(a, b) = (a·b) / (||a||·||b|| + ε)
```

### 2.5 Sigmoid
```
sigmoid(z) = 1 / (1 + exp(-clip(z, -60, 60)))
```

### 2.6 正交随机投射
```
orthogonal_matrix(rows, cols, rng): 用 QR 分解生成近似正交的随机矩阵
```
用于编码器初始化，避免行列相关。

### 2.7 幂律拟合（诊断用）
```
fit_power_law(xs, ys): log-log 最小二乘拟合 y = a·t^(-α)，返回 (a, α, r²)
```

---

## 3. 编码器（Phi: encoder.py）— M1_SEP

### 3.1 输入装配
```
u = [x ; c ; τ]   （输入 x，情境 c，时间细胞基 τ 拼接）
```
> **勘误（2026-11-19，对齐代码）**：u 由 `_assemble()` 拼出，但**编码器不消费 u**。按 D-017，码只吃内容 x：`encoder.d_u = d_x`（encoder.py:27）。u 的唯一下游是 `hebbian_update`，而 `learn_encoder` 默认 False → u 在默认配置下全程未用；`query()` 里拼出的 u 更是从未读取。§3.2 公式已同步修正。
- 情境 c：EMA 更新 `c_t = γ_c·c_{t-1} + (1-γ_c)·x_t`，γ_c=0.95（默认单尺度；多尺度见 context_taus）
- 时间细胞基 τ：高斯调谐曲线，`τ_j = exp(-(t - μ_j)² / (2σ²))`，μ_j = (j+0.5)/d_tau · T_ctx，σ = T_ctx/(2·d_tau)

### 3.2 双读出设计（核心）
同一条 EC→CA3 投射提供**两种读出**：

**(a) 写入侧（稀疏码）：**
```
s = max(W_e·x + b_e - θ, 0)          （稠密前激活，ReLU；x 为 d_x 维内容向量）
h = unit(relu_kwta(s, k))            （k-WTA 稀疏化 + 归一化）
```
h 是存入 CA3 的稀疏码。

**(b) 查询侧（稠密驱动）：**
```
s_pre = max(W_e·x + b_e - θ, 0)     （同一投射的稠密读出，不取 top-k）
```
s_pre 直接与稀疏存储码做线性读出（`M^T · s_pre`），继承"分离+鲁棒"。

### 3.3 稳态阈值 θ（内在可塑性）
```
θ += η_θ · (h - k/d_h)               （delta 规则在线更新）
```
这是把 input cos 0.986 的码从 0.883 压到 0.626 的**唯一有效机制**（白化与反 Hebb 均无效）。

### 3.4 期望库（Expectation Bank）
```
MU = [μ_0, μ_1, ...]                  （多个期望模板）
```
- 写入时用当前 mu 算残差：`xe = x - mu`
- 查询时逐模板假设，取证据最强者（处理情境切换后的失配）

### 3.5 残差写入
```
xe = x - mu_write   （use_resid=True 时；False 时 xe = x 原样）
s_pre = pre(xe)
h = encode_from_pre(s_pre)
```
> **勘误**：模型 Config `use_resid` **默认 False**（model.py:37，D-052：残差在 β=100 下中性、在 β=20 下与误设共享根因，故默认关闭、生产配置显式传参）。默认实际运行的是**无残差**分支——encoder.py:84 `return unit(x - self.mu) if self.use_resid else x`。

---

## 4. 情景快存（E: episodic.py）— M2_EPI

### 4.1 存储结构
- **M**: (d_h, N) — 稀疏码列向量，每列是一条情节的码
- **D**: (d_h, N) — EC→CA3 直通稠密通路（写入时存储 s_pre）
- **P**: (d_p, N) — 原始输入（供巩固/解码）
- **Ctx**: (d_c, N) — 情境向量
- **Tau**: (d_tau, N) — 时间细胞基
- **t, g, r, nu, use, sup** — 元数据（时间、门控值、奖赏、新颖性、使用次数、抑制因子）

### 4.2 写入（含去重）
```
max_cos(h) = max(M[:, :n]^T · h)
if max_cos >= theta_dup (0.95): return -1     （重复，不写）
M[:, n] = h; D[:, n] = dense; P[:, n] = payload
n += 1
```

### 4.3 现代 Hopfield 检索（模式完成通道）
```
z = unit(max(dense, 0))          （初值用稠密驱动，鲁棒）
a = softmax_beta(M^T · z + ebias, β)
for _ in range(L):                （L 次迭代，默认 L=3）
    z = unit(M · a)
    a = softmax_beta(M^T · z + ebias, β)
    a = a / a.sum()
```
- 全程连续，**不重新稀疏化**（现代 Hopfield 的关键）
- 初值用稠密驱动（s_pre）而非稀疏码——实测稀疏码作初值显著变差
- ebias：可选的逐槽能量偏置（情境检索），加在每次 softmax 的 logits 上

### 4.4 抑制因子（RIF — 检索诱发抑制）
```
suppress(a, rho_rif=0.15):
    winner = argmax(a)
    for i in top_k(a) if i != winner:
        sup[i] *= (1 - rho_rif)     （竞争者抑制 15%）
    use[winner] += 1
```
命中之后压制同时激活的竞争者，防止后续检索被干扰。

### 4.5 遗忘（Gamma）
```
importance(i) = g_i · exp(-dt_i / tau_f) · (1 + use_i)
prune(now): 按 importance 降序，保留前 (capacity - keep_free) 条
```
- 时衰指数 τ_f=50（默认）
- 使用增强：被检索过的条目重要性更高

### 4.6 事件层（D-042）
内容码与事件身份分离：每条输入恒写事件记录（eid, t, sender, session），内容码按 theta_dup 去重复用。

---

## 5. 语义慢存（Theta: semantic.py）— M5_SEM

### 5.1 前向
```
f(x) = B · tanh(A · x)
```
A: (d_z, d_x)，B: (d_h, d_z)

### 5.2 突触成熟度（Benna-Fusi 级联代理）
```
eta_eff = eta_0 / (1 + κ · omega)
```
- omega ∈ [0,1] 是逐坐标成熟度
- 新坐标 omega=0 → 学得快；反复更新后 omega→1 → 几乎冻结
- **突触更替（turnover，文档原缺）**：omega > theta_turn 且该坐标仍被持续驱动（|dA|>1e-6）→ **成熟度归零、换新突触**（semantic.py:61-62）。没有这一项，级联会把可塑性永久锁死、新任务学不进去（E4 第一轮 B 全灭的根因）。实际是"冻结—更替"循环，不是单向冻结。
- 产生**幂律式遗忘**——语义记忆的慢学习对应物

### 5.3 训练步（SGD + 梯度裁剪）
```
t = tanh(A·x); y = B·t
diff = clip(y - h_target, -3, 3)     （梯度裁剪，防 eta 大时溢出）
dB = 2·diff⊗t; dz = (B^T·2·diff)·(1-t²); dA = dz⊗x
dB = clip(dB, -1, 1); dA = clip(dA, -1, 1)
A -= etaA·dA + wd·A; B -= etaB·dB + wd·B
```

---

## 6. 工作记忆（W: wm.py）— M6_WM

### 6.1 结构
C 个槽位，每槽一个稀疏码 + 激活权重。

### 6.2 写入（present）
```
if occ.any() and max(slots[occ] · h) >= 0.90:
    w[j] += 0.15                     （同码增强）
else:
    if load < C: j = free[0]
    else: j = argmin(w)              （替换最弱槽）
    slots[j] = h; w[j] = 1.0; occ[j] = True
    w = max(w + eta_w·(sims - sims.mean()), 0)   （竞争衰减）
```

### 6.3 探针
```
probe(q) → (sims.max() > 0.5, max_sim)
```

### 6.4 容量含义
负荷 > C 时探针准确率阶梯式下跌（预测 P5）。

---

## 7. 写入门控（G: gate.py）— M7_NEU

### 7.1 公式
```
g_t = σ(w_nu·nu + w_rho·rpe + w_eps·eps + w_a·activity + b_g)
```
默认权重：w_nu=2.2, w_rho=1.4, w_eps=0.6, w_a=0.8, b=-1.2

### 7.2 输入项
| 项 | 含义 | 计算 |
|----|------|------|
| nu | 新颖性 | **相对比值，非绝对偏离**：raw = clip(1-cos(x, μ), 0, 2)，再除以近 20 步 raw 的均值得 ratio，clip 到 [0,2]（encoder.py:167-178）。绝对值会让"一整批新输入"整体被判不新（μ 追不上新分布）→ 写入 1/30；相对化保证门控始终有约一半输入被判为"比近期更新" |
| rpe | 奖赏预测误差 | r - v·h（v 是价值向量，在线更新） |
| eps | 意外度 | -log softmax(M^T·h) 的峰值（对库的意外程度） |
| activity | 激活度 | |h|/√k（稀疏码的活动水平） |

### 7.3 价值向量 v（在线学习）
```
v += eta_v · rpe · h                  （η_v=0.05）
```

### 7.4 消融
constant=True 时 g_t≡1（等价于无调质系统）。

---

## 8. 巩固（Psi: consolidate.py）— M4_CON

### 8.1 优先级采样
```
p_i ∝ (now - t_i + 1)^(-α) · sigmoid(nu_i)^b · (1 + max(r_i, 0))^c   （负奖赏截断到 0，防 NaN）
```
α=b=c=1.0（默认）
- 时间越近越优先（幂律衰减）
- 新颖性越高越优先
- 奖赏越高越优先

### 8.2 睡眠周期
```
cycle(now):
    n_new = round(n_r · (1 - interleave))
    idx = sample(now, n_new)          （优先级采样）
    X = [P[:, i] for i in idx]        （原始输入）
    H = [M[:, i] for i in idx]        （对应码）
    交错回放：从再激活缓冲取旧样本（防覆盖式遗忘）
    for epoch in range(epochs):
        对每条 (x, h)：语义 SGD 步（train_step）
```
- n_r=32（回放样本数），epochs=3，interleave=0.5

### 8.3 控制臂
uniform=True 时均匀采样（证明优先级不是装饰）。

---

## 9. 元控制（Pi: control.py）— M0_META

### 9.1 代价函数
```
ECost(a) = E[error|a] + c_a · c_unit
```
动作空间：{encode, retrieve, consolidate, forget, abstain}

### 9.2 代价分量
```
c_ret = 0.15 + 0.5·min(1, L·n/(20·d_h)) + (1-m1)·c_err    （检索成本随库大+证据弱而升）
c_enc = 0.3 + 0.7·max(0, g)                               （编码成本随门控升高）
c_con = 0.5
c_for = 0.2 + 0.6·forget_pressure
c_abs = c_abst (0.3)
```

### 9.3 检索决策（decide_retrieval）
```
if raw_evidence < theta_raw (0.50):    return abstain     （绝对证据不足）
if (m1 - m2) < margin (0.06) and nu > 0.5: return separate  （近并列+高新颖→分离）
return recall                                  （自信召回）
```

**raw_evidence** 是关键：吸引子重建匹配度，`unit(s_pre) · unit(D^T · a)`。实测库内 0.802±0.067 / 库外 0.393±0.046，分离度 >3σ，是目前**唯一可靠的弃权依据**。

---

## 10. 完整在线步（observe）

```
输入 x_t
1. 装配：u = [x, c, τ]                    （情境 EMA + 时间细胞；u 默认不进编码器，见 §3.1 勘误）
2. 残差：xe = x - mu_write                （use_resid=True 时；默认 False 时 xe = x）
3. 编码：s_pre = pre(xe); h = encode_from_pre(s_pre)
4. 新颖性：nu = novelty(x, mu)
5. 意外度：eps = -log softmax(M^T·h) 的峰值
6. 奖赏误差：rpe = r - v·h; v += η_v·rpe·h
7. 门控：g = σ(w_nu·nu + w_rho·rpe + w_eps·eps + w_a·activity + b)
8. 写入：if g >= theta_write (0.5) and 去重通过:
       M[:, n] = h; D[:, n] = s_pre; n += 1
9. 工作记忆：wm.present(h)
10. 时间步：t += 1
```

---

## 11. 完整检索步（query）

```
输入 x_q
1. 期望库扫描：对每个期望模板 μ_k:
   xe_k = x_q - μ_k
   s_pre_k = pre(xe_k)
   s_ev_k = raw(xe_k)         （稠密鲁棒证据）
   qq_k = encode_from_pre(s_pre_k)
   ev_k = w_dense·ed/||ed|| + (1-w_dense)·es/||es||
   （ed = M^T·s_ev, es = M^T·qq）
2. 取证据最强者 → ev, s_pre, q0
3. 吸引子迭代：q_star, a_hop = E.retrieve(q0, L, β, dense=s_pre)
4. 读出融合：f = λ·a + (1-λ)·s_n   （λ=0.6）
   s_n = sigmoid(z-score(M^T·shat))  （语义先验，组内标准化）
5. 排序：order = argsort(-f); m1 = f[top1]; m2 = f[top2]
6. 重建证据：raw_evidence = unit(s_pre) · unit(D^T·a)
7. 决策：ctl.decide_retrieval(m1, m2, nu_q, raw_evidence)
   → recall / separate / abstain
8. RIF：if recall: E.suppress(a, rho_rif=0.15)
```

---

## 12. 关键标度律

### 12.1 β*(N) ∝ log N（HCM 核心发现）
- 检索退化随 N 增大而加速（固定 β 下召回率坍缩）
- 补偿方式：β* 随 log N 增长（经验标度律，非闭式推导）
- 实测：char ρ=0.645 / sem ρ=0.322（33 点，10³→10⁶）
- 跨会话迁移 99.6%（exp45）
- 跨机 bit-exact（exp58）

### 12.2 写时分解
- 语义连接（sc）占写时 97.7%（O(N²) 相似度扫描）
- postings 后端可 O(1) 追加 + 0.1% 审计（1e6 档）

---

## 13. 设计决策索引（关键 ADR）

| 编号 | 决策 | 理由 |
|------|------|------|
| D-016 | 双读出（同投射既出稀疏码又出稠密驱动） | 稠密码对稠密码读出在 cos 0.9 时无法分辨（cos 0.986） |
| D-017 | 码只吃内容 x，情境/时间作为独立证据项 | 不混进码里，保持码的正交性 |
| D-019 | 检索初值用稠密驱动而非稀疏码 | 稀疏码作初值显著变差 |
| D-020 | 稳态阈值 θ（delta 规则） | 唯一能把高相似度码压到可分辨的机制 |
| D-026 | raw_evidence 作为弃权依据 | 唯一可靠（>3σ 分离）的弃权信号 |
| D-027 | 情境项必须组内标准化再混合 | 直接相加会淹没内容证据（差 3 个数量级） |
| D-042 | 事件层（内容码与事件身份分离） | 事件恒写，内容码按 theta_dup 去重 |
| D-052 | use_resid 默认关闭 | 残差在 β=100 下中性、β=20 下与误设共享根因 |

---

## 14. 与代码对齐勘误（2026-11-19 外部评审核验后修订）

以下为本文件首版与代码的实质不一致，已逐条对照 `HCM-动力学源码/` 修正：

1. **§3.1/§3.2**：u=[x;c;τ] 未被编码器消费（D-017 生效，d_u=d_x）；公式里 W_e·u → W_e·x。
2. **§3.5/§10**：残差写入默认关闭（use_resid=False，D-052），默认跑无残差分支。
3. **§7.2 nu**：是相对比值（近 20 步 EMA 归一化 + clip [0,2]），不是绝对偏离——这是门控能工作的前提。
4. **§5.2**：补突触更替（turnover）反向项——"冻结—更替"循环而非单向冻结。
5. **§8.1**：奖赏项为 (1+max(r,0))^c，负奖赏截断。

**代码侧的陈旧/死项（代码问题，非文档问题）**：
- `model.py:59-60` 的 `retrieve_init`、`softmax_on_raw`：只在声明处出现，`query()` 从不读取——死配置，注释却声称是 exp19c 修复臂。
- `control.py:19` docstring 说 theta_raw 作用在 lam·a_max+(1-lam)·exp(-nu)，与实际不符：model.py:258 的 raw_evidence = unit(s_pre)·unit(Dᵀa)（吸引子重建匹配度），control.py:42-44 直接拿它与 theta_raw 比较。
- `w_dense=1.0` 默认 → §11 步骤 1 的 ev 双通路里 es 项权重恒为 0（名义双通路、实际单通路）。
- `EpisodicStore` 默认 beta=25 恒被 Config beta=20 覆写（model.py:97 显式传参）。

**§12 数字可复核性**：桌面包只有实现层 13 文件；标度律/迁移率/bit-exact/check_claims 等证据数字在主 repo（brain-memory-model/ 的 bench、results、tests、memory/20-DECISIONS.md）里，不在本包。文档中的"check_claims 327/327"为成文时点数字，现已增至 398/398。

---

## 文件清单

| 文件 | 子系统 |
|------|--------|
| model.py | 主模型装配 + observe/query |
| ops.py | 基础算子 |
| encoder.py | Phi: EC→DG 编码器 |
| episodic.py | E: CA3 情景快存 |
| semantic.py | Theta: 新皮层语义慢存 |
| wm.py | W: PFC 工作记忆 |
| gate.py | G: 写入门控 |
| consolidate.py | Psi: SWR 巩固 |
| control.py | Pi: 元控制 |
| fastpath.py | 检索快速路径（可选） |
| text_encoder.py | 文本输入适配 |
| data.py | 数据加载 |