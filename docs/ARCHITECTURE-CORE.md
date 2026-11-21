# HCM 核心架构定义（CANONICAL CONTRACT v1.0）

> **本文档的地位**：全项目**冻结契约**。spec/、src/、bench/、paper/ 必须以本文档为准；任何不一致以本文档为准。只有 Lead 能改；改动必须在文末 CHANGELOG 追加一行并递增 minor 版本。
> 作者：Lead | 状态：ACTIVE | 下游消费者：arch（规格卡）、impl（实现）、exp（基准）

---

## 0. 一句话定义

**HCM（Hippocampal-Cortical Complementary Memory）** 是一个**双系统、多时间尺度、带元控制**的记忆模型：一个**快**的稀疏吸引子情节存储（海马）负责单次写入、模式分离与模式完成；一个**慢**的分布式语义存储（新皮层）负责统计规律的渐进抽取与泛化；两者由**回放巩固**耦合，由**神经调质门控**决定写不写，由**元控制策略**决定编/取/固/忘/弃。

形式化： M = < X, Phi, E, Theta, W, Psi, Pi, Gamma, G >

| 符号 | 名称 | 生物对应 | 一句话职责 |
|---|---|---|---|
| X | 输入空间 | 感觉/联合皮层输入 | 提供模式与外部奖励 |
| Phi | 分离编码器 | 内嗅皮层 II/III -> 齿状回 | 扩张重编码 + 稀疏化，把相似输入推开 |
| E | 情节快存 | CA3 递归侧支 + CA1 输出 | 单次写入、部分线索完成、时间标记 |
| Theta | 语义慢存 | 新皮层锥体细胞突触 | 慢学习、统计压缩、跨情节泛化 |
| W | 工作记忆 | PFC theta-gamma 槽位 | 容量受限的在线维持与操作 |
| Psi | 巩固算子 | SWR / 睡眠慢波 | 按优先级回放，把 E 的知识转移进 Theta |
| Pi | 检索控制 | PFC-海马环路 | 决定何时取、取哪个、何时弃权 |
| Gamma | 遗忘算子 | PFC 抑制 / 突触衰减 | 主动抑制、重要度剪枝、使用度衰减 |
| G | 调质门控 | VTA / 杏仁核 / LC | 新颖性、奖赏预测误差、显著性 -> 写入增益 |

**非目标（明确不做）**：不追求单细胞生物物理保真；不建模皮层柱层内微环路；不声称本模型是大脑的唯一实现。目标是**在行为与系统层面可证伪**，并且在超算规模上可执行。

---

## 1. 设计原则（每条都对应一个可被实验推翻的承诺）

1. **互补性优于单一系统**：快系统单独会过拟合与灾难性遗忘，慢系统单独无法单次记住。若消融掉任一系统后基准指标不下降，本原则被证伪。
2. **稀疏性是容量与分离的共同来源**：同一稀疏度 a 同时控制容量（a 增大容量下降）与分离（a 增大分离增强）。这是**强约束**：不允许为两者设独立旋钮。
3. **写入由惊奇驱动，不由时间驱动**：门控 g 必须能被“同样输入重复呈现”关掉。
4. **巩固由优先级驱动**：回放不是均匀抽样，p 正比于 recency^a * surprise^b * reward^c。均匀回放作为对照臂。
5. **遗忘是功能不是缺陷**：Gamma 必须提升风险调整后的检索准确率，否则应删除该模块。
6. **弃权是合法动作**：Pi 必须能在低置信时输出 abstain，且该动作有明确代价。
7. **规模可外推**：L0 上验证的结论必须给出到 L3 的外推协议，不允许“只在玩具规模成立”。

---

## 2. 符号与算子

~~~
d_x    输入维度            d_c   情境维度(= d_x, 共享空间)
d_tau  时间基维度          d_h   稀疏码维度(= rho * d_x)
rho    扩张率 (5-10)        a     稀疏度 (0.01-0.05), k = ceil(a * d_h)
N      情节槽位数          d_z   语义码维度
C      工作记忆容量 (3-5)   beta  Hopfield 逆温度 (5-40)
L      attractor 迭代步数 (3-12)
~~~

**算子**（实现必须严格一致，测试里有对应单测）：

- [v]_+ = max(v, 0) 逐元素
- unit(v) = v / (||v||_2 + 1e-8)
- kWTW_k(v)：保留绝对值最大的 k 个分量，其余置 0（**不回填、不缩放**）
- sigma(z) = 1/(1+exp(-z))
- softmax_beta(z)_i = exp(beta z_i) / sum_j exp(beta z_j)，实现时减 max 保证数值稳定
- cos(u,v) = <u,v> / (||u|| ||v|| + 1e-8)

---

## 3. 状态

### 3.1 情节快存 E（CA3）

~~~
E = { (h_i, kappa_i) }_{i=1..N}
h_i in R^{d_h},  ||h_i||_0 = k,  ||h_i||_2 = 1
kappa_i = ( t_i, g_i, r_i, nu_i, u_i, omega_i )
          写入时刻 写入增益 奖赏 新颖性 被取用次数 重要度
~~~

- 存储实现为**矩阵 M in R^{d_h x N}**（列 = 归一化稀疏码）+ 元数据表 K（结构化数组）。
- 容量上界：现代 Hopfield（dense associative memory）在稀疏码下约为 N_max 约等于 c * d_h / (k * log d_h)，c 由 bench/B2 实测标定，**不许用理论值假装实测值**。

### 3.2 语义慢存 Theta（新皮层）

~~~
Theta = ( A, B, omega )
A in R^{d_z x d_x}      输入投影（低秩瓶颈）
B in R^{d_h x d_z}      输出映射（对齐到稀疏码空间）
omega in [0,1]^{d_z x d_x}  突触成熟度（cascade 的标量代理，见 11 诚实清单）
f_Theta(x) = B * tanh(A * x)
~~~

- 低秩双线性是为了让“皮层泛化”可测且可在 numpy 上跑到 L1；L2/L3 允许换成多层 MLP（见 10）。
- omega 实现多时间尺度：eta_eff = eta_0 / (1 + kappa * omega)，更新时 omega <- omega + u_meta * (1 - omega)。u_meta 越小越难改，产生幂律遗忘（Benna-Fusi 的现象学代理）。

### 3.3 工作记忆 W（PFC）

~~~
W = { (slot_j, w_j) }_{j=1..C},  slot_j in R^{d_h}, w_j in [0,1]
~~~

### 3.4 上下文 u（EC / 时间细胞）

~~~
c_t = gamma_c * c_{t-1} + (1 - gamma_c) * x_t_norm ,  gamma_c in [0.90, 0.99]
tau_t[j] = exp( -(t - mu_j)^2 / (2 sigma_tau^2) ),  mu_j 均匀铺满 [0, T_ctx],  sigma_tau = T_ctx/(2*d_tau)
u_t = [ x_t_norm ; unit(c_t) ; unit(tau_t) ]
~~~

---

## 4. 在线单步动力学（t = 1..T）

~~~
0. 输入归一     x_t_norm = unit(x_t)
1. 情境更新     c_t = gamma_c*c_{t-1} + (1-gamma_c)*x_t_norm ; u_t = [x_t_norm ; unit(c_t) ; unit(tau_t)]
2. 分离编码     s_t = [ W_e u_t + b_e ]_+ ;  h_t = kWTW_k(s_t)
3. CA1 失配     x_hat_t = W_d h_t
                nu_t = ||x_t_norm - x_hat_t||_2                     新颖性 in [0,2]
                eps_t = -log softmax_beta({cos(h_t,h_i)})[argmax]   惊奇，越小越熟
4. 调质门控     RPE_t = r_t + lambda_v*(v'h_{t+1}) - v'h_t        TD 误差；无下一时刻取 0
                g_t = sigma( w_nu*nu_t + w_rho*RPE_t + w_eps*eps_t + w_a*(||h_t||_1/d_h) + b_g )
5. 情节写入     if g_t >= theta_write and max_i cos(h_t,h_i) < theta_dup:
                    E <- E 并 {(unit(h_t), kappa_t)} ;  if N > N_max: PRUNE(Gamma)
6. 工作记忆     dW_j = eta_w*( cos(h_t,slot_j) - mean_j cos(h_t,slot_j) )
                w <- [w + dW]_+ ; 保留 top-C 权重的槽位，其余清零并释放
7. 检索（若策略=retrieve，见 6）
8. 元控制       a_t = argmin_{a in {encode,retrieve,consolidate,forget,abstain}} ECost(a|state)
9. 慢学习（在线微步，可选）  Theta <- Theta - eta_online * grad L_online，eta_online << eta_sleep
~~~

**编码器参数** W_e in R^{d_h x d_u}：
- L0/L1：**随机固定**（正交初始化后按列归一）。依据：DG 颗粒细胞的扩张重编码主要依赖随机投射 + 稀疏化。
- L2/L3 允许：局部 Hebbian 竞争学习更新 W_e（W_e <- W_e + eta(h u' - diag)，配 Oja 归一）。
- 解码器 W_d in R^{d_x x d_h}：delta 规则在线更新 W_d <- W_d + eta_d (x_t_norm - x_hat_t) h_t'，用于产生可信的新颖性信号。

---

## 5. 离线巩固 Psi（“睡眠”）

每个 sleep cycle（一次调用 consolidate(n_cycles)）：

~~~
1. 优先级     p_i 正比于 (t_now - t_i + 1)^(-alpha) * sigmoid(nu_i)^b * (1+r_i)^c
              alpha in [0.5,1.5], b,c in [0,2]；用 softmax(p/T_p) 抽样，T_p 为温度；
              均匀抽样作为对照臂。
2. 交错回放   batch = 0.5*n_r 来自 E（新） + 0.5*n_r 来自 rehearsal buffer R（旧/语义）
              若无 R，则 100% 来自 E —— 这一臂**必须**在 B4 里出现遗忘，作为阳性对照。
3. 皮层更新   L = ||f_Theta(x_i) - unit(h_i)||^2 + [若有标签] CE(softmax(g_Theta(x_i)), y_i)
              Theta <- Theta - eta_s * grad L ,  eta_s = eta_0/(1 + kappa*omega)
4. 突触成熟   omega <- omega + u_meta*(1 - omega)   （仅被回放触及的坐标）
5. 情节瘦身   E 中 omega_i 已饱和且可被 Theta 正确重建的条目，转入“可弃”集合（不立即删，等 Gamma 判定）
~~~

**关键约束**：巩固必须**只**通过回放改变 Theta，不允许直接拷贝 E 的条目进 Theta（否则 B4 对照失效）。

---

## 6. 检索与元控制（Pi）

### 6.1 双通道检索

~~~
q0 = kWTW_k([ W_e [x_q_norm ; unit(c_q) ; unit(tau_q)] + b_e ]_+)
Repeat L times:
    a^l = softmax_beta( M' q^l )              完成通道：吸引子收敛
    q^{l+1} = kWTW_k( M a^l ) ; q^{l+1} <- unit(q^{l+1})
e_i = a^L_i                                  情节证据
s_i = cos( f_Theta(x_q), h_i )               语义先验（皮层对候选的重建一致性）
f_i = lambda*e_i + (1-lambda)*s_i            lambda in [0,1]，默认 0.6
~~~

### 6.2 决策规则（这是模型“像人”的地方）

~~~
top1 = argmax f ,  m1 = f[top1] ,  m2 = 第二高
if m1 < theta_rec:                     -> ABSTAIN（“不记得”）
elif (m1 - m2) < margin and nu_q 高:   -> SEPARATE（判为相似但不同：分离失败被显式报告）
else:                                  -> RECALL(top1)
~~~

报告时必须同时给出 m1、m1-m2、nu_q，**置信度是模型的输出，不是事后拟合**。

### 6.3 元控制代价

~~~
ECost(a) = E[error | a, state] + c_a * c_unit
   encode      : 写入能量 + 未来干扰风险，正比于 g_t
   retrieve    : 时间（L*N 次内积）+ 若低置信则错误代价 (1-m1)*c_err
   consolidate : 计算（n_r*epochs）+ 能量
   forget      : 丢失有价值信息的风险，正比于 sum p_i
   abstain     : 固定代价 c_abst（由任务给出，例如答错扣 1、弃权扣 0.3）
a* = argmin_a ECost(a)
~~~

**可检验点**：当 c_abst 被人为抬高时，模型应减少弃权并提高错误率 —— 对应行为经济学里的“风险偏好”。

---

## 7. 遗忘 Gamma

~~~
7.1 情节剪枝   I_i = g_i * exp(-dt_i/tau_f) * (1 + u_i)
               N > N_max 时删 I 最小的条目（tau_f 对应“不被回放就衰减”）
7.2 检索诱发抑制  检索命中 j 后，对 i in top-k 且 i != j：
               g_i <- g_i*(1 - rho_rif) ,  u_i 不增加
               rho_rif in [0.05, 0.3]  —— 直接对应 retrieval-induced forgetting 范式
7.3 语义衰减   未被任何回放触及的概念坐标：可用权重 w_eff = w*(1 - delta*dt)
7.4 抑制解除   被抑制条目在下一次被成功检索时，g 恢复（乘 1/(1-rho_rif)），防止永久损伤
~~~

---

## 8. 九个子系统的责任矩阵

| ID | 子模块 | 实现的符号 | 唯一负责 | 消融方式 | 预期被破坏的基准 |
|---|---|---|---|---|---|
| M1_SEP | 内嗅 II/III + 齿状回 | Phi: W_e, kWTW | 扩张 + 稀疏 + 去相关 | rho=1, k=d_h | B1 相似干扰 |
| M2_EPI | CA3 + CA1 | E, M, attractor | 单次写入 + 完成 | N_max=0 | B1 完成, B2 容量 |
| M3_CTX | EC->CA1 + 时间细胞 | c_t, tau_t | 时间情境绑定 | gamma_c=0 且 tau 常量 | B3 顺序/情境依赖 |
| M4_CON | SWR + 海马-皮层 | Psi: 回放采样 | 知识转移 | n_r=0 | B4 迁移 + 遗忘 |
| M5_SEM | 新皮层 + 语义枢纽 | Theta | 慢统计学习 | eta_s=0 | B4 泛化 |
| M6_WM | PFC theta-gamma | W, C | 容量受限维持 | C=无穷 | B5 超载 |
| M7_NEU | VTA/杏仁核/LC | G: g_t | 写入调制 | g_t 恒等于 1 | B6 干扰/信噪 |
| M8_FOR | PFC-海马 | Gamma | 抑制与剪枝 | rho_rif=0 且不剪枝 | B7 RIF + 曲线 |
| M0_META | PFC-海马环路 | Pi | 策略选择 | pi 恒为 retrieve | B8 校准 |

---

## 9. 可证伪预测（论文核心主张，每条都必须在 bench 里被实测）

| # | 预测 | 观测变量 | 通过判据 | 被证伪的样子 |
|---|---|---|---|---|
| P1 | 关闭 DG 分离后，高相似输入对的混淆率显著上升 | 混淆率差值 | >= 15 个百分点 | < 5 pp |
| P2 | 稀疏度 a 与容量正相关、与完成率负相关，存在折中 | 容量-完成曲线 | 单调且拐点存在 | 两者同向变化 |
| P3 | 抑制回放使顺序任务保持下降，但单任务不变 | 保持率 | 顺序 >=10pp，单任务 <3pp | 两者同降 |
| P4 | 无交错回放时出现灾难性遗忘 | 旧任务保持率 | 掉到随机水平附近 | 保持率不变 |
| P5 | 工作记忆超载时性能呈阶梯而非平滑下降 | 准确率-负荷曲线 | 在 C 处出现拐折 | 平滑线性下降 |
| P6 | 遗忘曲线服从幂律，指数 alpha in [0.3,0.6] | 拟合 R^2、alpha | R^2 > 0.9 且 alpha 在区间 | 指数拟合更优 |
| P7 | 弃权机制提升风险调整准确率 | 风险调整得分 | 提升 >= 8% | 无变化或下降 |

---

## 10. 复杂度与规模分级

单步在线：编码 O(d_h*d_u)，写入 O(d_h)，检索 O(L*N*d_h)（稀疏码可用稀疏乘降到 O(L*N*k)）。巩固：O(n_r*epochs*d_x*d_z)。

| 级 | d_x | rho | d_h | N_max | 检索实现 | 预期硬件 | 目的 |
|---|---|---|---|---|---|---|---|
| L0 | 128 | 8 | 1024 | 2k | 稠密矩阵 | 笔记本 CPU | 单元测试 / 机制验证 |
| L1 | 512 | 8 | 4096 | 20k | 稠密矩阵 | 单机多核 | 全部基准（论文主结果） |
| L2 | 2048 | 8 | 16384 | 500k | 分块 + 稀疏 | 单卡 GPU | scale 律的第一点 |
| L3 | 8192 | 8 | 65536 | 20M | 分片 top-k 检索 | 多卡/多节点 | 超算验证 |

**实现约束**：L0/L1 必须**纯 numpy 且结果可复现**（固定种子）；L2/L3 允许后端替换（torch / 分片检索），但**接口签名不得改变**，且必须在 L1 上做过等价性验证（同一随机种子，前 100 步输出向量 L2 误差 < 1e-6）。

---

## 11. 诚实清单（本文档自己的近似，论文必须原样披露）

1. **情境用 EMA + 时间基**近似时间细胞，未建模 theta 相位进动。
2. **回放是事件级抽样**，未建模 SWR 的 100-250 Hz 振荡与压缩时间尺度（真实回放比真实经历快约 10-20 倍）。
3. **cascade 用标量成熟度 omega 近似** Benna-Fusi 多级离散级联；它产生幂律遗忘，但不是原模型的等价实现。
4. **语义存储是低秩双线性**，不是皮层层级；M5 的“语义枢纽”只体现为低秩瓶颈。
5. **未区分抑制性中间神经元**（PV/SST），k-WTA 是它们的现象学替代。
6. **未建模多巴胺的相位/张力二分**，RPE 只有一个标量通道。
7. **L2/L3 允许换后端**，这会削弱“同一模型”的强主张 —— 论文必须列出等价性验证结果。

---

## 12. 与已有工作的边界（避免被指“重新发明”）

- 与 **CLS（McClelland 1995）**：CLS 只有两个系统、无门控、无回放优先级、无元控制、无遗忘算子；HCM 的增量是 (a) 门控写入 (b) 优先级回放 (c) 显式弃权 (d) 参数由文献取值域约束。
- 与 **现代 Hopfield / 稠密联想记忆**：HCM 用稀疏码把容量-分离耦合到同一个参数 a 上。
- 与 **TEM / TCM（Howard & Kahana）**：HCM 保留情境漂移，但把它作为编码器输入而非唯一记忆载体。
- 与 **Benna-Fusi 级联**：HCM 只借其现象学（多时间尺度 -> 幂律遗忘），并明确标注为近似。
- 与 **PBWM（O'Reilly）**：HCM 的 W 只做容量受限维持，门控在 G 而非 W。

---

## CHANGELOG

- v1.0 (2026-11-06) 首版冻结。定义 M 元组、在线 9 步、巩固 5 步、检索 3 分支决策、遗忘 4 条、9 子系统责任矩阵、7 条可证伪预测、L0-L3 分级、7 条诚实清单。


---

# v1.1 修订（2026-11-06，由实验结果强制）

本节**取代** v1.0 中"两条并行通路竞争"的表述。现行机制链（顺序串联，非并行竞争）：

1. **M1a 情境残差**：mu <- 0.9*mu + 0.1*x；x_eff = unit(x - mu)。分离就发生在这里：共享成分被减掉，剩余部分才进入扩张编码。实测把 input overlap 0.75 的码余弦从 0.581 压到 0.084、0.9 从 0.872 压到 0.308。
2. **M1b 扩张稀疏**：s = max(W_e x_eff - theta, 0)，h = unit(kWTW_k(s))，rho=8，a=0.02。theta 只在发育期收敛（develop），不参与在线漂移。
3. **写入**：E 存 h（稀疏码）+ 元数据（t, g, r, nu, use, sup）。
4. **查询**：同一投射给出稠密驱动 s（不取 top-k）——它是鲁棒的；
   a = softmax(beta M^T s) 是单步读出；再迭代 z <- unit(M a) 得到 CA3 吸引子补全（L=3，beta=20）。
5. **初值规则（必须遵守）**：吸引子初值只能是稠密驱动。用稀疏码作初值在 overlap 0.75 + 20% 缺失下 17/30，用稠密驱动 27/30。

被实验否定的表述（v1.0 遗留，勿再引用）：
- "EC 直通独立投射 + DG 稀疏码并行竞争、稀疏臂负责分离"：稀疏臂从未赢过，已在 overlap>=0.75 全面失效。
- "稳态阈值带来分离"：只是缩短码长，distinct support 会塌缩（30 条只剩 6 组支持集）。
- "码余弦越低越分离"：必须同时看 distinct support 与退化线索准确率（L-012）。
