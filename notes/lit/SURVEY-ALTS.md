# SURVEY-ALTS：已完成环节的更优方案审计（调查二，2026-11-07）

调查对象：HCM 已完成且判 LOAD-BEARING / 已定型的 9 个机制（K1-K9）。问题只有一个：**文献与工程实践中是否存在更优替代**。若有 → roadmap 升级；若无 → 「系统级闭环在已知方案中最优」的站位更硬。

方法（与调查一同标）：Crossref 主通道（tools/collect.py --out refs/survey2/refs_survey2.jsonl，--min-sim 0.35 召回导向，tier 只记录不过滤）+ arXiv 通道（tools/collect_arxiv_survey.py）+ web_search 近期动态（只入报告不入库）。与调查一按 doi/arxiv_id/标题规整串去重（refs/survey2/_postprocess.py），已入库条目一律用 S-id 交叉引用，不重复入库。ref_id 前缀 T=调查二库、S=调查一库。引用核验：报告内全部 ref_id/arxiv_id 已跑 refs/survey2/verify_survey2.py（双库解析，D-003）——**100/100 verified, 0 missing**（refs/survey2/verify-report2.md）。

漏斗总览（ref 见 refs/survey2/query-stats2.json；下表数字为最终值）：
| 通道 | 查询数 | 原始命中（raw） | 采集入库（collect 级去重后） |
|---|---|---|---|
| Crossref | 67 | fetched 5046 行（API total-results 合计 1.24 亿，OR 语义无意义，不作漏斗起点，见 query-stats2.json） | 757（主轮 587 + 补轮 170） |
| arXiv | 50 | 36,624 | 237（主轮 163 + 补轮 74） |
| pin-fetch（web 定点 id_list 直取） | 6 | 6 | 6 |
| **合计** | **123** | — | **1000** |

去重：跨库去重 vs 调查一库 130 条 + 库内 27 条 → **终库 843 条（T 前缀）**。筛选（tools/survey_screen.py，threshold=2）：coarse 550/843 → fine 440。分区 total→fine：K1 96→68 · K2 95→46 · K3 90→51 · K4 93→72 · K5 107→44 · K6 68→34 · K7 110→51 · K8 98→40 · K9 86→34。

**样本量声明**：K2/K5/K6/K8/K9 细筛 <60。按 Lead 指示（m00690，循调查一 I 区先例）诚实降级并标注真相关数：K6≈34（补轮已收 bump-attractor/STSP/群体码三条线，剩余为认知行为学噪声）、K9≈34（哈希嵌入与蒸馏两线已收全）、K2≈46（VSA/DAM 主线 + product-key 工程线）、K5≈44（LoRA 家族已收 15+ 变体，其余为超网络线）、K8≈40（事件分割群 11 条 + event-KG 管线）。未放宽阈值凑数。

---

## K1 稀疏编码器（M1_SEP）

**现方案**：期望模板残差 → 线性扩张 ρ=8（d_h=1024）→ relu 阈值 → k-WTA 取 top-k（活跃比 a=0.02，k=ceil(a·d_h)）→ L2 归一。检索不用稀疏码而用同一投射的未取 top-k 版本（稠密证据线性读出，设计原则 2：分离在残差不在扩张）。

**最优已知界（稀疏联想记忆/SDM 谱系）**：Willshaw–Buneman–Longuet-Higgins 外积模型在 a→0 时容量/能耗渐近最优（每突触单 bit、容量 ~N²/log² N 位级检索）；Palm 的 k-WTA 码书容量规则（稀疏度 a≈ln N/N 附近最优）；现代 SDM 分析（Keeler 1988 容量公式；Jaeckel 标记权选择）。SDM 临界距离：地址空间维度高时先突降后平缓（Willshaw 谱系给出解析）。

**检索式**：见 query-stats2.json K1_sparse_encoder 7 条 crossref + 4 条 arxiv。

**替代方案表**：
| 方案 | 一句话机制 | 优势 | 采纳成本 | 成熟度 | 年代 |
|---|---|---|---|---|---|
| WTA-hash（GPU beam search）T597 | k-WTA 相对排序码直接作 LSH 哈希 | 与 M1 同族——证明 k-WTA 码可作子线性检索；beam search 加速解码 | 中：把稀疏码接 ANN 索引（百行级）；L0-L3 规模暂无必要 | 有实现+有基准 | 2018 |
| TopK-SAE 家族：BatchTopK T613 / SoftSAE T610 / Rational SAE T611 / whitening T612 | 固定/动态 top-k 非线性与输入白化的 SAE 训练改进 | 训练稳定性与特征质量提升 | 中：改 W_e 训练目标需重训重标定 a、k | 有实现+有基准 | 2024-2026 |
| 两层 PQ / PQ-routing T599, T601 | 两层量化索引十亿级 ANN | 吞吐/召回 Pareto 优于暴力积 | 高：需量化码本训练管线，破坏确定性投影语义 | 有实现+有基准 | 2014-2023 |
| LSH 家族：survey T588 / cross-polytope T592 / hyperbolic T590 | 随机投影族哈希，(r,cr)-敏感 | 渐近敏感度最优、免训练 | 中：与 k-WTA 码不同构，需双码并存 | 理论+实现 | 2016-2026 |

**判定**：**KEEP**。全部替代都在「密集向量检索」问题设定下求子线性；M1 的 k-WTA 码本身就是 WTA-hash（T597）同族的稀疏哈希，L0-L3 规模（d_h=1024，查询一次 matvec ≈10⁶ 乘加）暴力扫描即常数成本，替代收益要到 10⁸ 向量级才显现。无更优替代；T597 反而证明现方案属于该谱系。

**采纳建议**：不改结构。①检索加速留到 L3+（情节条目 >10⁶ 时引 T597/T588 作 LSH 桥）；②W_e 重训时可用 BatchTopK（T613）与 whitening（T612）作训练技巧，属实验注意事项不属机制变更；③论文 related work 引 LSH survey（T588）定位「k-WTA=稀疏哈希」谱系。

**近期动态（web 2024-2026）**：
- Projection and Quantisation: A Unifying View of Learning to Hash, from Random Projections to the RAG Era（arXiv 2510.04127，2025-10，库内 T753）——学习哈希统一视角，K1 替代检索的近期综述锚点。

---

## K2 情节库（M2_EPI）

**现方案**：单次写入 M ← [M, g·h]（外积 Hopfield 列扩展）；检索 z ← unit(M·softmax(βMᵀz))，β=20、L=3 步吸引子（>6 步实测漂移）；事件层 theta_dup=0.95 去重。容量实证见论文 fig3（d_h/k 同量级）。

**最优已知界**：经典 Hopfield 0.138N；现代 Hopfield（Ramsauer 2020）指数容量 p ~ exp(d) 级、检索一步收敛；k-WTA/稀疏谱系 Palm 容量；SDM 容量公式（Keeler 1988，~N²/log²N 比特级）。

**检索式**：见 query-stats2.json K2_episodic_store 6 条 crossref + 4 条 arxiv。

**替代方案表**：
| 方案 | 一句话机制 | 优势 | 采纳成本 | 成熟度 | 年代 |
|---|---|---|---|---|---|
| Fast-weight Product Key Memory T639 | 外积快权重与 product-key 键值检索的混合层 | 容量-算力权衡优于纯外积堆积（键检索 O(√K)），可在线增长 | **高**：替换 M2 写入/读出语义，破坏 fig3 容量实证与 check_claims 既有键；建议作 L3 升级路径而非正典替换 | 有实现+有基准 | 2026 |
| Kanerva Machine T627 | 生成式分布式记忆，贝叶斯解析写入 | 写入带不确定性量化、容量解析 | 高：架构替换（高斯隐变量记忆矩阵） | 理论+实现 | 2018 |
| VSA 容量分析 T633 / VSA 比较 T635 / 低维 VSA T638 | 绑定/捆绑算子的干扰与容量解析 | 给外积存储的容量公式与算子选择依据 | 零：分析工具 | 理论 | 2020-2025 |
| DAM 理论群：phase kernel T908 / bifurcation T618 / dynamics T906 / free energy T905 / robustness T909 | 现代 Hopfield 的核函数、分岔、动力学与自由能分析 | T908 指出线性核高记忆载下吸引盆重叠——正是 M2 容量极限的机理解释；其余为 β、L 超参的理论辩护 | 低：只加分析脚本，不改代码 | 理论 | 2024-2026 |
| Memory Layers at Scale T754 / PEER T751 | product-key 记忆层规模化（1.2M keys / 百万专家路由） | 工程证据：键值记忆可扩到百万级 | 高：LLM 层内替换，与 M2 情节库不同层 | 有实现+有基准 | 2024-2025 |

**判定**：**KEEP（L0-L2 规模）+ UPGRADE-AVAILABLE（L3 规模）**。T639 是最直接对位（快权重+product-key 混合），但采纳=替换 M2 单次外积写入语义，破坏 fig3 与 check_claims 链，且生物对应（Hopfield 列扩展）是设计核心。理论侧 T908（2026 phase kernel）已给出线性核高载重叠吸引盆的机理，支持「β=20、L=3 在低载最优」的现有实证；VSA 容量公式（T633）可给 fig3 叠加理论曲线。

**采纳建议**：①M2 现机制保持 L0-L2 正典，论文 limitation 注明线性核高载吸引盆重叠（引 T908）；②roadmap L3 增「product-key 情节库」升级路径（引 T639/T754），触发条件=情节条目 >10⁵ 或检索延迟超标；③fig3 重跑时叠加 T633 容量公式作理论对照（只加分析脚本，零代码风险）。

**近期动态（web 2024-2026）**：
- Memory Layers at Scale（arXiv 2412.09764，Meta，2024-12，库内 T754）——product-key 记忆层扩到 1.2M keys、多 GPU， factual QA 超同参 MoE：K2/K1 的工程级替代证据。
- Attention as Binding: A Vector-Symbolic Perspective on Transformer Reasoning（arXiv 2512.14709，库内 T752）——VSA 绑定与注意力统一视角。
- Mixture of A Million Experts（arXiv 2407.04153，PEER，库内 T751）——product-key 路由的极端规模版。

---

## K3 时序链接（exp28 回放一次性链接）

**现方案**：离线回放一次性链接 Wlink[slot_j, slot_i] += κ（相邻对 gap≤30s、异 sender）；检索两阶段 a2 = softmax(β·(scores + w_chain·Wlinkᵀ a1))，w_chain∈{0,0.3,1,3}；动机 Bittner–Magee BTSP + Buzsáki CA3 链。exp27 基线证明纯情节补全无前向联想（next@5=1.5%）。

**最优已知界**：SR/SF 理论给出多步 value 与 transfer 的解析（Dayan 1993；Barreto SF 系）；图上后继传播等价于 discount 次幂邻接；TKG 谱系给时间约束下的链接预测指标。

**检索式**：见 query-stats2.json K3_temporal_links 5 条 crossref + 3 条 arxiv。

**替代方案表**：
| 方案 | 一句话机制 | 优势 | 采纳成本 | 成熟度 | 年代 |
|---|---|---|---|---|---|
| Robust successor features T650 / 神经生物学 SF T208 | 后继特征=折扣幂邻接算子，多步传播有解析形式 | 多步前向联想解析可写；T208 给海马 Successor-like 表征证据 | 中：Wlink 需扩为全状态后继矩阵（内存 ×槽数²） | 理论+实现 | 2020-2026 |
| TKG 链接预测：embedding dynamics T660 / transfer T661 / meta-learning T658 / graph encoder T651 / Hawkes T652 / surveys T655, T656 | 时序图上（实体×关系×时间）四元组的链接预测 | 表达力强、基准成熟 | 高：需实体/关系嵌入基建，与槽位寻址不同构 | 有实现+有基准 | 2020-2026 |
| PLEASING T200 | 历史+潜在事件联合的 TKG 外推 | 插补+外推一体 | 高：生成模型基建 | 有实现 | 2024 |

**判定**：**KEEP**。Wlink 一次性 Hebbian 链接是 BTSP 动机下的最小机制；SR/TKG 表达力更强但采纳是架构级（嵌入/图基建），且 HCM 前向联想需求是槽位级两阶段读出而非全图推理。exp33 链应力若失败再升级。

**采纳建议**：①exp33 跑 3+ 跳链应力；若 Wlink 级联衰减明显，借 SR 形式做对照分析（Wlink 幂 vs discount 幂邻接，只加分析脚本）；②论文站位引 T208 论证「链接=后继表征的低秩特例」；③related work 引 TKG 综述（T655/T656）。

**近期动态（web 2024-2026）**：
- Behavioral Timescale Synaptic Plasticity: A Burst in the Field of Learning and Memory（JNeurosci 2025, e1332252025）——BTSP 领域综述，现方案的生物动机 2025 仍在扩阵。
- Temporal Knowledge Graph Reasoning: A Survey（IEEE TKDE 2025）——时间图链接预测的三个流派（表示/规则/神经符号）。

---

## K4 巩固调度（M4_CON）

**现方案**：p_i ∝ (e^{−Δt/τ_f})^a · g_i^b · (1+r_i)^c，τ_f=50，a=b=c=1；周期内交错新样本与回放（interleave=0.5）；n_r 次加权抽样送入 S 的交错 SGD。**关键负结果**：小缓冲低预算下 priority 不优于 uniform（exp6b）；「有交错」才是关键；无干扰时顺序学习不产生灾难性遗忘（exp5/exp4c，D-023）。

**最优已知界**：EWC 的 Fisher 正则等价于高斯后验近似（Kirkpatrick 2017）；SI（Zenke 2017）路径积分；GEM/A-GEM 约束优化界；生成式回放的记忆容量-保真度权衡。

**检索式**：见 query-stats2.json K4_consolidation_sched 5 条 crossref + 3 条 arxiv。

**替代方案表**：
| 方案 | 一句话机制 | 优势 | 采纳成本 | 成熟度 | 年代 |
|---|---|---|---|---|---|
| EWC Done Right T679 | 修正 Fisher 估计路径的 EWC | 同预算遗忘更低 | 中：S 的 SGD 加 Fisher 对角项（每参数一个状态量，内存 ×2），破坏 exp4/5/6b 对照语义 | 有实现+有基准 | 2026 |
| SYNERgy T664 | 突触巩固（SI）+经验回放协同，general CL | 两机制互补（正则防漂移+回放抗干扰）的证据 | 中：M4 已有回放，补 SI 路径积分≈每参数一个累计量（内存 ×1.5），S 训练循环改 50-100 行 | 有实现+有基准 | 2022 |
| Experience replay 群：batch-level+review T666 / 防可塑性丢失 T667 / 经典 T670 | 回放缓冲策略变体 | T667：回放同时防可塑性丢失（HCM 未测维度） | 低：M4 回放调度已是该族成员 | 有实现+有基准 | 2018-2025 |
| Brain-inspired replay T243 / 海马生成式回放理论 T271 | 睡眠期生成式回放的生物机制→算法 | 与 M4 生物动机同源，支撑现设计 | 零：理论支撑 | 理论+实现 | 2020-2022 |
| Three scenarios T674 | 任务/域/实例增量三分类 | 给 exp 设计的场景分类语言 | 零 | 理论 | 2019 |

**判定**：**KEEP**。D-023 负结果（无干扰不遗忘、小缓冲 priority≤uniform）划定 M4 所处机制区间：主导因素是交错本身，调度优先级是二阶效应。EWC-DR/SYNERgy 针对的是「无回放或有干扰」场景的遗忘防护，与 M4 互补而非更优；但它们暗示 D-023 的边界条件（干扰场景下 priority/正则可能翻案）——这是最值得做的验证实验，见采纳建议。

**采纳建议**：①保持 p_i 调度正典；②exp7/8 若引入干扰场景，加 SYNERgy 臂（SI+回放，T664）与 EWC-DR 臂（T679）对照，检验 D-023 是否在干扰下反转；③T667（回放防可塑性丢失）补进 M4 实验设计检查单；④论文引 T243/T271 作生物机制对照。

**近期动态（web 2024-2026）**：
- Continual Learning of Large Language Models: A Comprehensive Survey（ACM Computing Surveys 2025, 10.1145/3735633）——五大流派分类（replay/regularization/architecture/optimization/data）。

---

## K5 语义慢存（M5_SEM）

**现方案**：f(x)=B·tanh(Ax) 低秩瓶颈 d_z=64，交错 SGD + 级联调制（κ=8 刚度）+ turnover（u_meta=0.02 成熟速率）。语义权重是唯一慢存层。

**最优已知界**：LoRA 秩-遗忘权衡（低秩更新的容量上界 ~ rank×dim）；adapters 的模块化遗忘隔离；fast weights（Ba 2016）的生物对应与容量。

**检索式**：见 query-stats2.json K5_semantic_slow 5 条 crossref + 3 条 arxiv。

**替代方案表**：
| 方案 | 一句话机制 | 优势 | 采纳成本 | 成熟度 | 年代 |
|---|---|---|---|---|---|
| LoRA 家族：Hydra T311 / LoRA+ T691 / LoRA-Pro T694 / SDS-LoRA T693 / DT-LoRA T690 / **Null-LoRA T692** / GraLoRA T308 / La-LoRA T314 / RaLo T331 | 低秩更新的梯度缩放/切空间/零空间/粒度改进 | Null-LoRA 把更新约束在零空间→持续学习防遗忘，直接对位 M5 需求 | 中：M5 的 B·tanh(Ax) 已是 LoRA 同构，换更新规则需改 S 优化器语义 | 有实现+有基准 | 2024-2026 |
| Hypernetwork CL 群 T929, T928, T930, T933 | 超网络生成任务参数 | 遗忘隔离好 | 高：架构替换，无生物对应 | 有实现 | 2019-2026 |
| 结构可塑性 T794 | 结构 vs 突触可塑性的遗忘分工 | 支撑 M5 turnover（u_meta）设计 | 零：理论支撑 | 理论 | 2022 |

**判定**：**KEEP**。B·tanh(Ax) 功能上就是 LoRA 式低秩 adapter（生物 adapter 定位）；LoRA 家族改进集中在梯度缩放与秩分配，属优化效率而非机制更优；Null-LoRA（T692）的零空间约束是唯一触及「防遗忘机制」的替代，值得 roadmap 注记。

**采纳建议**：①M5 保持现状，定位语「M5=生物启发的低秩 adapter，与 LoRA 谱系同构，另加级联调制与 turnover」；②roadmap 注 Null-LoRA 零空间方向（T692）为持续学习防遗忘候选；③不引超网络（架构替换成本高、无生物对应）。

**近期动态（web 2024-2026）**：
- Controlled Low-Rank Adaptation with Subspace Regularization for Continued Training on LLMs（ACL 2025, 2025.acl-long.940）——低秩更新的子空间正则=防遗忘现成件，与 K5 直接对位。

---

## K6 工作记忆（M6_WM）

**现方案**：C=4 槽位（Miller 4±1 / Lisman–Idiart 嵌套动机），槽位权重竞争 w_j ← max(w_j + η_w(cos − mean), 0)；内容由码余弦写入，无 θ-γ 相位复用（roadmap D2 未实现）。

**最优已知界**：槽位容量 4±1（Cowan 2001）；activity-silent 谱系主张短时突触可塑性承载、无持续放电的容量-能耗优势；指针网络/外部栈的符号容量无界但刷新成本高。

**检索式**：见 query-stats2.json K6_working_memory 5 条 crossref + 3 条 arxiv。

**替代方案表**：
| 方案 | 一句话机制 | 优势 | 采纳成本 | 成熟度 | 年代 |
|---|---|---|---|---|---|
| Hebbian fast plasticity & WM T710 | 短时 Hebbian 突触可塑性承载 WM（activity-silent） | 容量-能耗优势、与多时间尺度自然衔接 | 高：M6 需加 STSP 静默迹层（槽位外新状态）并改检索读出 | 理论+实现 | 2023 |
| STSP WM 仿真群 T402, T403, T404 | 短时可塑性解释不规则持续放电/协同维持 | 现成模型可作 bench 对照 | 中：移植 STSP 参数作对照臂 | 有实现 | 2013-2026 |
| Population code PFC T841 / bump attractor certainty T936 | 群体码/吸引子上的不确定性编码 | T936 给 bump attractor 的置信度读出——M0 与 M6 的桥 | 低-中：读出层扩展 | 理论+实现 | 2003-2026 |
| Pointer networks T714 / stack-pointer T716 | 外部栈/指针符号容量 | 容量无界 | 高：符号基建，非生物对应 | 有实现 | 2018-2020 |

**判定**：**KEEP**。activity-silent（T710）是竞争假设而非更优替代——roadmap D2（θ-γ 相位码/activity-silent 路线）已覆盖该方向；槽位竞争 w_j 更新规则无已知更优解。本区真相关数 ≈34（细筛 <60 诚实降级），但四条替代线（Hebbian 静默迹、STSP 仿真、群体码置信、指针容量）均已收全。

**采纳建议**：①M6 保持槽位+竞争；②bench 增「STSP 静默迹臂」（T710/T404）：同 C=4 预算下比槽位持续迹 vs 突触静默迹的容量/保持时长——把 activity-silent 争论可计算化；③T936 bump attractor 置信度读出注记给 M0×M6 桥接实验。

**近期动态（web 2024-2026）**：
- 持续放电 vs activity-silent 的争论 2024-2025 仍在（NatComm 2024 皮层-内嗅环路持续/静默双态证据）。

---

## K7 元控制（M0_META）

**现方案**：证据融合 f_i = λ·a_i + (1−λ)·clip(s_i,0,1)（λ=0.6）；固定规则三分支 RECALL/SEPARATE/ABSTAIN（θ_rec=0.25、margin=0.06）；代价敏感版用 c_unit/c_err/c_abst 期望代价最小化；实验内 AUC=1.000（红旗：任务太易，校准未在真实语料验证）。

**最优已知界**：Bayes-optimal 序贯检验（SPRT）给出给定误差率下最小期望样本/停时；VOI 理论（Howard 1966）给出信息获取决策的规范基；selective prediction 的 coverage-risk 权衡曲线是标准评估面。

**检索式**：见 query-stats2.json K7_meta_control 6 条 crossref + 3 条 arxiv。

**替代方案表**：
| 方案 | 一句话机制 | 优势 | 采纳成本 | 成熟度 | 年代 |
|---|---|---|---|---|---|
| Memory as a Controlled Process T755 | 学习型记忆管理：读/写/忘动作按预期收益决策 | 元控制从固定规则升为可学习策略，长期收益可优化 | 中：作为可选 bench 臂只加一层 policy（~200-400 行）+reward 设计；进正典则破坏三分支透明性 | 有实现 | 2026 |
| Selective prediction 校准 T717 / T944 / T946 | coverage-risk 曲线+后 hoc 校准 | 分布无关的弃权评估；诊断 AUC=1.000 红旗的标准工具 | 低：评估协议扩展，不动 M0 本体 | 有实现+有基准 | 2020-2024 |
| Partial abstention multi-label T718 | 多标签部分弃权 | 比 RECALL/SEPARATE/ABSTAIN 更细的弃权粒度 | 中：分支结构扩展 | 理论+实现 | 2019 |
| Optimal metacognitive control of recall T433 | 回忆元认知控制的最优停时 | 给 RECALL/ABSTAIN 阈值规范基础 | 零：理论辩护 | 理论 | 2024 |
| Spacing/metacognitive control 群 T435, T437, T441, T444 | 学习者自控间隔与线索整合 | 支撑 M0→M4 控制链路的生物对应 | 零 | 理论（行为） | 2004-2026 |

**判定**：**UPGRADE-AVAILABLE（协议先行，策略可选）**。固定阈值三分支透明、代价敏感版已有；真正缺口是**验证协议**（真实语料 coverage-risk + 校准，回应 AUC=1.000 红旗），而非策略本身。T755 学习型管理是 L3 方向，不构成现规模下的替换理由。

**K7 特别对照：Memory as a Controlled Process（T755）采纳成本 + 验证实验设计**
- 采纳成本分级：**A 级（零代码）**——真实语料重跑 M0：θ_rec×margin 扫描画 coverage-risk 曲线族 + ECE/可靠性图；**B 级（~50-100 行）**——用 T433 最优停时给 θ_rec 做解析对照（阈值点的规范辩护）；**C 级（~200-400 行）**——T755 式学习门控：以 7-bit 证据特征（a_i、s_i、码龄、会话密度、码频次、残差能量、链支持）训练 logistic/小型策略，对比固定三分支的期望代价；需新增 reward 语义与 check_claims 键。
- 验证实验设计（B 级先行）：语料=exp23 真实文本库。步骤 1：固定 λ=0.6，扫 θ_rec∈{0.15,0.20,0.25,0.30}×margin∈{0.04,0.06,0.10}，每点报 coverage/risk；步骤 2：对 s_i 做温度缩放或 Platt 校准，报 ECE 前后对比；步骤 3：以现代价参数（c_unit/c_err/c_abst）在 risk-coverage 曲面上选最优工作点，与现 (0.25,0.06) 对照。判据：真实语料 ECE<0.05，且最优点期望代价 ≤ 现点。C 级仅在 B 级显示学习门控期望代价优势 >10% 时启动。

**采纳建议**：①先做 A 级+B 级（真实语料 coverage-risk + ECE + T433 阈值对照）——这是九区里性价比最高的升级；②T755 学习门控列 L3 可选臂（触发条件=B 级实验显示固定阈值次优）；③T718 部分弃权若引入多标签查询再议；④论文引 T433/T435 群作元认知控制的认知对照。

**近期动态（web 2024-2026）**：
- Memory as a Controlled Process: Learned Adaptive Memory Management for LLM Agents（arXiv 2607.13591，库内 T755）——学习型记忆控制策略，K7 最直接的对位新文献。
- Calibrating LLMs for Selective Prediction: Balancing Coverage and Risk（NeurIPS 2025）——coverage-risk 校准协议可直接借用作 M0 的真实语料验证协议。
- Abstention, Refusal, and Calibration in LLMs（UncertainLP 2025 workshop）——弃权分类学。

---

## K8 事件层（D-042/exp30）

**现方案**：事件恒写 {eid,t,sender,session,code_id,is_new_code}；内容码按 theta_dup=0.95 去重复用、新码创建受门控约束；query_events(t0,t1,sender,session) 时间/来源过滤；码被剪枝后 slot=None 保留事件壳。

**最优已知界**：TKG（时间戳四元组）查询表达力强于纯时间窗过滤，代价是实体/关系抽取层；事件本体（schema.org/Event、ASE）给 schema 校验但引入标注负担；episodic buffer（Baddeley）是认知架构对应物。

**检索式**：见 query-stats2.json K8_event_layer 5 条 crossref + 3 条 arxiv。

**替代方案表**：
| 方案 | 一句话机制 | 优势 | 采纳成本 | 成熟度 | 年代 |
|---|---|---|---|---|---|
| Object-centric logs→TEKG T727 | 对象中心事件日志转时间事件知识图谱的完整管线 | schema 化（对象×关系×时间戳），查询表达力强 | 高：需实体/关系抽取与图存储，替换 D-042 恒写事件 | 有实现 | 2024 |
| OEKG T724 | 开放域事件 KG（新闻级抽取） | 覆盖开放域事件 | 高：抽取层依赖 | 有实现 | 2023 |
| Event segmentation 认知群 T516, T510, T511, T512, T518, T515 | 事件分割机制：预测误差驱动边界（T512）、分割提升一月保持（T518）、WM 遗忘率（T511） | 给 D-042「事件=分割单元」的认知依据；T511 连接 M6 | 零：理论支撑 | 理论（行为） | 2013-2025 |
| 无监督视觉事件分割 T960 | 照片流无监督分割 | 边界检测可计算化参考 | 低：可作 is_new_code 门控对照 | 有实现 | 2018 |

**判定**：**KEEP**。事件恒写+码去重复用是查询/存储的最小闭环；TKG 管线（T727）表达力强但要求抽取层与图基建，当前事件源（对话/会话）schema 简单、收益不足。event segmentation 文献群是 D-042 的认知依据弹药（T516/T518/T512/T511）。本区真相关数 ≈40（诚实降级：行为学分割文献占比高，工程替代线仅 3 条但已收全）。

**采纳建议**：①保持恒写层；②L3 若引入多模态/开放域事件，走 T727 对象中心→TEKG 转换路径（schema 扩展而非替换恒写）；③论文引 T516/T518/T512 论证「事件作为记忆单元」的认知基础，引 T511 连接 WM 遗忘率；④is_new_code 门控可借 T960 无监督边界检测做对照臂。

**近期动态（web 2024-2026）**：
- 事件分割（event segmentation）与 LLM 自动回忆评估结合 2025 出现应用层文献（HMM 眼动分割 ↔ 记忆回想损伤）。

---

## K9 文本前端（PPMI+SVD 字符编码器）

**现方案**：字符级共现 PPMI → 截断 SVD → 词符嵌入矩阵 E（本地、确定性、无 GPU、无预训练权重）；exp23 用它做真实语料编码器。

**最优已知界**：TF-IDF+BOW 的稀疏基线（BM25）；feature hashing 的 O(1) 内存、无词表（Weinberger 2009）；小型蒸馏句嵌入（确定性蒸馏）检索质量高但带模型依赖；LSH/字符 n-gram 中间态。

**检索式**：见 query-stats2.json K9_text_frontend 6 条 crossref + 4 条 arxiv。

**替代方案表**：
| 方案 | 一句话机制 | 优势 | 采纳成本 | 成熟度 | 年代 |
|---|---|---|---|---|---|
| **HTP T750** | 谐波投影：无词表、免训练、确定性、**可逆**的文本嵌入 | 与 PPMI+SVD 同设计目标，多出可逆性（重构校验）与无语料依赖（PPMI 需先建共现库） | 中：text_encoder.py 换实现（隔离模块 ~100-200 行）；exp23 嵌入重生成；check_claims 编码器指纹更新（exp23b 新键） | 有实现 | 2025 |
| Hash2Vec T741 / additive feature hashing T740 / hashing trick T743 | 特征哈希 token→固定维 | O(1) 内存、无词表；但纯词袋级无语义结构 | 低-中：实现易，质量预期低于 PPMI+SVD | 有实现 | 2009-2021 |
| byteSteady byte n-gram T972 | 字节级 n-gram 嵌入分类 | 无分词、快速 | 中：分类导向，非检索嵌入 | 有实现 | 2021 |
| 蒸馏句嵌入家族 T745, T746, T747, T744 | 大模型蒸馏小型句嵌入 | 检索质量最高 | 高：带预训练模型依赖（违背 HCM 无预训练权重原则） | 有实现+有基准 | 2019-2021 |
| 共现→嵌入谱系 T566, T562 | PPMI/SVD 理论脉络 | 现方案的谱系定位 | 零 | 理论 | 2012-2025 |

**判定**：**UPGRADE-AVAILABLE（低成本 A/B）**。HTP（T750）与 PPMI+SVD 设计目标重合，且多出可逆性与无语料依赖两优点；采纳是模块级替换而非架构级。本区真相关数 ≈34（诚实降级：蒸馏句嵌入 6 条为「被否决的替代」仍入表作反证）。

**K9 特别对照：HTP（T750）采纳成本 + 验证实验设计**
- 采纳成本：text_encoder.py 是隔离模块（M9 无下游梯度依赖，读出全为余弦），换实现 ~100-200 行；连带成本=exp23 嵌入重生成（E 矩阵指纹变化，check_claims 需新键 exp23b）+基准重跑；风险低（回滚=git revert 单模块）。
- 验证实验设计（A/B 臂）：语料=exp23 同批文本。臂 A=PPMI+SVD（现），臂 B=HTP（T750）。指标：①检索质量——20 组查询 top-5 对人工标注相关集的 nDCG@5；②确定性——同输入两次编码逐位一致；③约束达成——无 GPU/离线/无预训练权重检查；④成本——每千 token 编码时延与嵌入体积。判据：B 的 nDCG@5 ≥ A 且时延 ≤2×，则 B 升正典、A 降对照臂；否则 A 保持并记录 HTP 为候选。

**采纳建议**：①跑 T750 HTP 的 A/B 臂（判据见上）——九区里唯一可能「白捡」的升级；②hashing trick（T743）作下界基线一并跑；③蒸馏句嵌入（T747）明确写进论文「被否决的替代」段（违背无预训练权重原则）；④谱系引 T566。

**近期动态（web 2024-2026）**：
- Harmonic Token Projection（HTP, arXiv 2511.20665，库内 T750）——无词表、免训练、确定性、可逆嵌入：K9 现方案的 2025 直接对位（同设计目标），需收进细筛。

---

---

## 九区判定矩阵

| 区 | 现方案 | 最接近替代 | 判定 | 建议动作 |
|---|---|---|---|---|
| K1 M1_SEP | 残差→扩张 ρ=8→k-WTA a=0.02 | WTA-hash GPU T597 | **KEEP** | 不改；L3 若条目 >10⁶ 引 LSH 桥 |
| K2 M2_EPI | 外积单次写入 + Hopfield β=20 L=3 | Fast-weight PKM T639 | **KEEP@L0-L2 / UPGRADE-AVAILABLE@L3** | limitation 注 T908；roadmap L3 product-key 路径 |
| K3 exp28 | Wlink 一次性 Hebbian 链接 | SR/SF T650, T208 | **KEEP** | exp33 链应力后议升级；SR 作分析对照 |
| K4 M4_CON | p_i 优先回放 + interleave | EWC-DR T679 / SYNERgy T664 | **KEEP** | 干扰场景加 SYNERgy/EWC-DR 臂验证 D-023 边界 |
| K5 M5_SEM | B·tanh(Ax) 低秩瓶颈 | Null-LoRA T692 | **KEEP** | roadmap 注零空间方向；定位=生物 adapter |
| K6 M6_WM | C=4 槽位 + 竞争 | activity-silent T710, T404 | **KEEP** | STSP 静默迹 bench 臂；T936 置信读出注记 |
| K7 M0_META | 固定三分支 + 代价敏感 | Memory-as-Controlled-Process T755 | **UPGRADE-AVAILABLE** | coverage-risk+ECE 协议先行（A/B 级）；学习门控列 L3 可选 |
| K8 D-042 | 事件恒写 + 码去重 | logs→TEKG T727 | **KEEP** | L3 走 schema 扩展路径；引事件分割文献作认知依据 |
| K9 text_encoder | PPMI+SVD 字符嵌入 | HTP T750 | **UPGRADE-AVAILABLE** | A/B bench 臂（nDCG@5+确定性+时延判据） |

**总计：KEEP 6 区（K1/K3/K4/K5/K6/K8）+ 条件 KEEP 1 区（K2，现规模）+ UPGRADE-AVAILABLE 2 区（K7/K9）+ REPLACE 0 区。**

## 论文站位语句

> HCM 九个承重机制经 843 条文献的系统对照审计（Crossref 67 查询 + arXiv 50 查询 + 6 条定点补采，去重后细筛 440 条），六区（M1 稀疏编码、exp28 时序链接、M4 巩固调度、M5 语义慢存、M6 工作记忆、D-042 事件层）与现规模的 M2 情节库在已知文献与工程实践中不存在更优替代：最接近的候选（WTA-hash、后继特征、EWC/SYNERgy、Null-LoRA、activity-silent、事件知识图谱）或属同族谱系、或互补假设、或采纳成本为架构级。两处可升级点均有低成本验证路径——M0 元控制的真实语料 coverage-risk/ECE 协议与文本前端 HTP 的 A/B 臂——已列入 roadmap 而非推翻既有机制。

## Roadmap 修改建议

1. **L3 升级路径（K2）**：product-key 情节库（T639/T754），触发条件=情节条目 >10⁵ 或检索延迟超标。
2. **验证协议（K7，优先级最高）**：真实语料 coverage-risk 曲线 + ECE 校准 + T433 最优停时阈值对照；学习门控（T755）仅在协议显示固定阈值次优时启动。
3. **A/B 臂（K9）**：HTP（T750）对照 PPMI+SVD，判据 nDCG@5 ≥ 且时延 ≤2×。
4. **干扰场景实验（K4）**：SYNERgy 臂（T664）+ EWC-DR 臂（T679）检验 D-023 在干扰下是否反转。
5. **STSP 静默迹臂（K6）**：槽位持续迹 vs 突触静默迹（T710/T404），把 activity-silent 之争可计算化。
6. **注记项**：Null-LoRA 零空间方向（K5/T692）；bump attractor 置信读出（K6×M0/T936）；事件 schema→TEKG 扩展路径（K8/T727）。
