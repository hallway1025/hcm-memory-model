# SURVEY-WEAKPOINTS — 十薄弱环节文献调查（high-recall）

调查代理：lit-surveyor（fresh）· 日期 2026-11-07 · 项目根 brain-memory-model
对象：HCM 记忆模型 10 个已确认薄弱环节（M0/M2/M3/M4/M6/M7/M8/B3/工程/J 对位）。
数据：refs/survey/refs_survey.jsonl（Crossref + arXiv 双通道，S 前缀 ref_id，**1 139 条**，2026-11-07 终）；本报告引用的每一条 ref_id 都在该库中并通过 verify_citations（refs/survey/verify-report.md）。

## 方法与筛选漏斗（可审计）
- 查询定义：refs/survey/queries.jsonl（主轮 48 条：crossref 33 + arxiv 15）+ queries_supp_crossref.jsonl(16) + queries_supp_arxiv.jsonl(27) + queries_fix.jsonl(6, J 定点)。检索式原文与每查询 funnel 数见 refs/survey/query-stats.json 与下文各节。
- Crossref 通道：tools/collect.py --min-sim 0.35（召回导向，主题词与标题 sim 阈值放宽；主库 0.80 门禁不适用于调查库），每查询取 rows=max(n*6,14) 条候选，按 tier/年份/相似度排序后保留 n 条；tier 只记录不过滤；retracted 丢弃。
- arXiv 通道：tools/collect_arxiv_survey.py（export.arxiv.org/api/query，AND-of-words，限速 ≥3.2s，Atom 解析）。
- 去重：DOI → arxiv_id → 规范化标题前 120 字符（跨通道）。
- 粗筛（标题级）→ 细筛（标题+摘要加权关键词，tools/survey_screen.py，core×3/aux×1/abs-core×2，阈值 3.0）→ 人工复核（screen 是排序器不是终审；每节表格为人工确认过的条目）。
- web_search 不入 refs_survey.jsonl，只进各节"近期动态"（2024-2026，带 URL）。
- 最终 funnel：fetch（crossref 行预算 max(n*6,14) + arxiv 实取 min(raw,n)）→ 去重后 → 粗筛 → 细筛；**合计 5 975 → 1 139 → 802 → 685**（分区分查询明细见文末"总览矩阵"附表与 refs/survey/query-stats.json）。查无命中/定点补采如实记录：E 区 supp arxiv "theta gamma nested oscillation memory" raw=0；A 区 "hippocampal time cell sequence memory" raw=2、"multi-timescale memory hierarchy" raw=2；D 区 arxiv "Benna Fusi consolidation" raw=2；J 区 3 条定点查询（Memory Age / LongMemEval / MemOS）AND 词撞不上或 sim 门未过，改由 refs/survey/_fetch_ids.py 按 id_list 直取补采（S1097/S1084/S1089）。


## A_timescale 多时间尺度上下文 / 时间细胞（M3，最薄弱）

**检索式**（主轮 crossref）：(1) "temporal context model" (n=25) (2) "hippocampal time cells" (n=20) (3) "time cell hippocampus memory" (n=25) (4) "temporal context memory drift" (n=20)；arxiv："temporal context model"、"hippocampal time cell sequence memory" (各 n=15)；supp crossref："temporal receptive fields timescales hierarchy"(n=15)、"state space models timescales memory"(n=12)、"multiple timescales neural integration"(n=12)。funnel：fetch 770 → 去重后 120 → 粗筛 61 → 细筛 37。

**HCM 现状对照**：M3 当前实现为单时间尺度 EMA 上下文 \(c_t = \gamma c_{t-1} + (1-\gamma)x_t\)（M3_CTX 消融判 REDUNDANT——能力被 M1 期望库吸收）。roadmap B2（τ∈{1,10,100,1000} 步 EMA 堆叠）未实现。缺：多时间尺度堆叠、时间细胞式位置码、漂移-再激活耦合。

**现成方案表**（均为人工确认条目，完整表见 screen_fine.jsonl _area=A_timescale）：

| 方案 | 一句话机制 | 来源 | 成熟度 | 年代 |
|---|---|---|---|---|
| 时间细胞（综述+理论） | 时间沿记忆流被显式表示为 Cells 的感受野，重置/分裂/拉伸三规则 | S002 | 理论+实证综述 | 2014 |
| TCM / scale-invariant 漂移 | 上下文以 \(c_t=\rho c_{t-1}+x\) 漂移，ρ 可随层深度缩放 → 多时间尺度自然涌现 | S025 | 理论+行为拟合 | 2004 |
| TCM 行为验证（大鼠） | 大鼠序列记忆符合 TCM 漂移预测（primacy/间距效应） | S029 | 实证+拟合 | 2015 |
| OFC 上下文信号 | 情节关联检索由 OFC 携带的时间上下文信号驱动 | S017 | 实证 | 2010 |
| EEG 上下文再激活 | 编码期漂移率预测后续情节区分（ iEEG ） | S021 | 实证 | 2019 |
| RSC 时间细胞 | 屏障任务中 RSC 编码时间间隔的时间细胞（2024 新证据） | S009 | 实证 | 2024 |
| 多时间尺度突触巩固 | 单突触双过程：快 LTP+慢巩固进程 → 时间尺度分层 | S041 | 理论+实证 | 2014 |
| 习惯化的多时间尺度 | 新皮层感觉响应多时间尺度习惯化（fitted 分成分） | S043 | 实证+模型 | 2022 |
| Wilson-Cowan MMOs | 混合模振荡在不同频率尺度上持有多项记忆 → 天然 θ-γ 嵌套载体 | S048 | 理论+仿真 | 2026 |
| SSM 天然时间细胞 | 状态空间模型天然产生时间细胞与行波 | S1128 (2507.13638) | 理论+仿真 | 2025 |
| 层级 Hebbian 多时间尺度 | 分层 Hebbian 记忆实现快速联想适应→持久多时间尺度记忆 | S1122 (2609.00358) | 理论+仿真 | 2026 |
| 间距效应 MTL 模型 | 慢漂移时间上下文 + MTL 交界 → 间距/去上下文化效应 | S023 | 计算+实证桥 | 2024 |

**差距判定**：**已有方案可采纳**（非真空白）。TCM 标准式即多时间尺度；S041/S048 给出生物合理的载具；S023 证明漂移率与情节区分的因果链。
**采纳建议**：
1. M3 实装 roadmap B2：把单 γ EMA 换成 4 通道 EMA 堆叠（τ=1/10/100/1000 步），各通道投影到读出（等价 TCM 的 scale-invariant 层次版）。
2. 借 S048 的 MMO：在 Wilson-Cowan 环节引入 2-3 个耦合振荡模，天然产生 θ-γ 嵌套（同时服务 E 区）。
3. 把 M0 的 raw_evidence 加上"多通道一致性"项：同一片段被多个 τ 通道同时高激活 → 置信度提升（对应时间细胞的时间比例不变性 S015）。
**近期动态（2024-2026）**：SSM 天然时间细胞/行波 [2507.13638]；层级 Hebbian 多时间尺度 [2609.00358]；"Memory Out of Context: spacing effects & decontextualization in MTL model"（2025, psycnet 2025-06339-001）。

## B_forget 遗忘的功能性机制（M8 判 INERT）

**检索式**（主轮 crossref）："retrieval-induced forgetting adaptive function"(25) / "forgetting benefits generalization flexibility"(25) / "adaptive forgetting reinforcement learning"(25) / "directed forgetting motivated neuroscience"(20)；arxiv："principled forgetting neural networks"(20)；supp："machine unlearning"(15) / "fuzzy trace theory gist verbatim"(10)；supp arxiv："machine unlearning LLM agents"(15) / "catastrophic forgetting replay continual learning"(10)。funnel：fetch 765 → 去重后 147 → 粗筛 95 → 细筛 95（细筛阈值下几乎全部保留；B 区关键词命中率最高）。

**HCM 现状对照**：M8 已有抑制机制（剪枝 I_i=g_i·e^{−Δt/τ}(1+use_i)，RIF sup←(1−ρ)sup），但消融判 INERT——施加抑制后 top-1 命中不变。roadmap B5=选择性 RIF+fuzzy-trace（gist 存 S/verbatim 衰减）未实现。缺：遗忘的功能性收益（泛化/容量/干扰减少）未被任何 HCM 实验展示。

**现成方案表**：

| 方案 | 一句话机制 | 来源 | 成熟度 | 年代 |
|---|---|---|---|---|
| RIF 串序版（预注册 4 实验） | 输出干扰来自检索动作本身还是活跃表征——串序记忆中分离 RIF 与 restudy 效应 | S110 | 实证 | 2026 |
| 睡眠样离线回放减灾难遗忘 | ANN 加 sleep 相位（局部 Hebbian+噪声输入）回放新旧冲突记忆 | S186 | 有实现 | 2022 |
| 扩散式调质消除灾难遗忘 | 突触参数受扩散调质→模块化隔离任务信息 | S194 | 有实现 | 2017 |
| MESU：突触不确定性的贝叶斯更新 | 学习率×不确定性 → 无任务边界的"原则性学+忘"+OOD 检测 | S187 | 有实现+基准 | 2025 |
| 生长式双网络的剪枝效应 | 剪枝作为遗忘算子在增量学习中的作用 | S195 | 有实现 | 2019 |
| 遗忘利于运动保持 | 上下文干扰范式：插入遗忘期反而提升保持（functional forgetting 实证） | S128 | 实证 | 1991 |
| 动机性遗忘（lPFC 抑制控制） | 抑制不想要的记忆=前额叶抑制控制，编码/检索两阶段 | S174 | 实证综述 | 2014 |
| RIF 作为动机认知 | 需要闭合(NFC)增强 RIF；内容服务目标时减弱 | S176 | 实证综述 | 2018 |
| "Forgetting in ML and Beyond" 综述 | ML+神经+心理三视角统一遗忘功能分类 | S1129 (2405.20620) | 综述 | 2024 |
| LLM agent 机器遗忘 | Agent 记忆的选择性遗忘算子与评测 | S1038 (2610.17659) | 有实现 | 2026 |

**差距判定**：**已有方案可采纳**。HCM 的问题不是缺机制而是缺"展示收益"的实验设计——S128/S186/S187 证明遗忘收益需在干扰/增量场景测，而非单轮 top-1。
**采纳建议**：
1. 把 M8 从"静默抑制"改成三档功能开关：容量档（剪枝+子线性索引 H 区联动）、干扰档（RIF 只对与正确答案竞争的抑制，ρ 按竞争度调制）、泛化档（fuzzy-trace：gist 保留、verbatim 指数衰减）。
2. 新增消融实验：多轮干扰检索下开/关 RIF 的 top-1 与 top-k 差异（预期 INERT 变 LOAD-BEARING）。
3. MESU 思路可直接映射：HCM 的写入权重 g_i 乘以该码的"被访问方差"（不确定性代理）。
**近期动态（2024-2026）**：Serotonin required for retrieval-induced forgetting（bioRxiv 2025.03.05.641624）；Can Agents Truly Forget? Machine Unlearning for LLM Agents [2610.17659]；"Forgetting in ML and Beyond: A Survey"（引用>500 的统一分类）。

## C_neuro 调质多路分化 + 资格痕迹（M7 判 INCONCLUSIVE）

**检索式**（主轮 crossref）："acetylcholine hippocampal encoding retrieval dynamics"(25) / "norepinephrine locus coeruleus memory encoding"(25) / "dopamine prediction error hippocampal plasticity"(25) / "eligibility traces synaptic plasticity"(20) / "novelty modulated encoding hippocampus"(20)；arxiv："eligibility traces learning synapse"(10) / "neuromodulation gating consolidation"(10)；supp："eligibility traces synaptic plasticity"(10) / "locus coeruleus novelty memory"(10) / "acetylcholine forgetting consolidation hippocampus"(10)。funnel：fetch 890 → 去重后 105 → 粗筛 66 → 细筛 63。

**HCM 现状对照**：M7 单通道门控 g_t=σ(w·[ν,r,ε,a]+b)（RPE 类标量），无 DA/ACh/NE 分化、无资格痕迹、无时序折扣。两臂数字全等（INCONCLUSIVE）。roadmap B6=VOI+盆地几何置信度未实现。

**现成方案表**：

| 方案 | 一句话机制 | 来源 | 成熟度 | 年代 |
|---|---|---|---|---|
| LC-NE 自适应增益理论 | phasic LC=决策/P3 的增益开关，exploit/explore 转换 | S248 | 理论+实证 | 2005 |
| LC 组织学（mini-review） | LC-NE 系统的分化投射=功能分化的解剖基础 | S250 | 综述 | 2015 |
| LC 记忆链接 | LC-NE 选择性调制"链接已有记忆"而非新记忆形成 | S234 | 实证 | 2022 |
| D1/D5 显著性→持久 LTP | DA 受体把"信息显著性"转成持久可塑性使能 | S264 | 实证 | 2012 |
| DA RPE 定量验证 | 中脑 DA 神经元编码定量 RPE（眼动奖赏任务） | S252 | 实证 | 2005 |
| 静默资格痕迹（纹状体） | pre-post 活动留下"silent"痕迹，DA 事后到达才转为强化 | S276 | 实证（全细胞记录） | 2018 |
| 资格痕迹综述（2025） | 多系统资格痕迹证据：调质依赖+平台电位两条腿，LTP/LTD 双向 | S278 | 综述 | 2025 |
| ACh 编码/检索开关 | ACh 高=编码态、低=检索态（Hasselmo 谱系，见细筛列表） | （screen_fine C 区多条） | 理论+实证 | 1990s-2020s |

**差距判定**：**已有方案需适配**。资格痕迹（S276/S278）与 HCM 的 M1 写入天然契合——码写入即留痕，调质信号（DA=RPE、ACh=编码/检索态、NE=意外度）分道到位。
**采纳建议**：
1. 把 M7 的单通道 g_t 拆成三通道向量 (g_da, g_ach, g_ne)：DA 乘写入权重、ACh 控制写/读模式偏置、NE 调制置信度先验（I 区联动）。
2. 实装资格痕迹：每个码存 e_i=指数衰减的活动痕迹；调质信号到达时执行 w_i+=g·e_i（解耦"何时写"与"写多少"，直接回应 roadmap B6）。
3. 消融实验改设计：单通道→三通道对比必须在"延迟奖赏"场景（奖赏滞后 3-5 轮），当前实验奖赏即时所以两臂全等。
**近期动态（2024-2026）**：Eligibility traces as a synaptic substrate for learning（Curr Opin Neurobiol 2025, PMID 39965463）；Fast adaptation to rule switching using neuronal surprise（PLoS Comp Biol 2025）；Post-learning replay biased by RPE（Nat Comm 2025, 10.1038/s41467-025-65354-2）。

## D_consolidation 巩固：Benna-Fusi 级联 / 睡眠分期 / 优先回放（M4）

**检索式**（主轮 crossref）："Benna Fusi cascade synaptic consolidation"(20) / "sharp-wave ripple sleep spindle coordination"(25) / "systems consolidation hippocampal neocortical dynamics"(25) / "prioritized replay hippocampus"(20) / "targeted memory reactivation sleep"(15)；arxiv："experience replay prioritization"(20) / "Benna Fusi consolidation"(10) / "sleep replay consolidation model"(10)；supp："synaptic consolidation cascade model"(10) / "systems consolidation sleep replay hippocampus"(15)。funnel：fetch 820 → 去重后 126 → 粗筛 118 → 细筛 99。

**HCM 现状对照**：M4 优先回放 p_i∝recency^a·surprise^b·reward^c 已 LOAD-BEARING（交错回放）。级联=标量成熟度 ω 代理 Benna-Fusi（诚实清单自认）。无睡眠分期、无 SHY（roadmap B4=回放后突触稳态缩放）、无序列回放（事件级，无反向/前向序列）。

**现成方案表**：

| 方案 | 一句话机制 | 来源 | 成熟度 | 年代 |
|---|---|---|---|---|
| Benna-Fusi 级联 | 突触强度分布在 N 级变量链上扩散，2^N 时间尺度谱 | 主轮查询（screen_fine D 区，含 Fusi 系） | 理论 | 2016 |
| 双稳态巩固模型最优刺激协议 | 巩固效率取决于重复刺激频率（间歇优于持续） | S300 | 理论+仿真 | 2019 |
| 睡眠期离线活动+两类突触可塑性 | 回放期间 co-Hebbian/homeostatic 两类规则并行 | S302 | 实证综述 | 2023 |
| spindles-ripple 耦合（长时长 ripple） | 海马-新皮层信息转移靠 spindle 门控 ripple | S336 | 实证（iEEG） | 2020 |
| swr 内容：rigid vs plastic | 回放既巩固特定记忆也维持全局图式（两模式） | S321 | 实证综述 | 2023 |
| 回放内容综述 | 回放=记忆巩固+前瞻规划双重功能 | S350 | 综述 | 2018 |
| 皮层慢波的角色 | 慢波经验依赖调节、因果操控增强/损害巩固 | S305 | 实证综述 | 2017 |
| 情绪记忆清醒再激活 | 静息期 hipp-neocortical 再激活（情绪强于中性） | S340 | 实证 | 2016 |
| TMR meta 分析 | 91 实验 g=0.29，NREM 两期有效（选择性加强的因果证据） | S376 | meta 分析 | 2020 |
| TMR 选择性加强机制 | 线索上调 cued 记忆再激活，上调量预测后续记忆 | S374 | 实证+机制 | 2019 |
| 优先回放（RL 视角） | 经验回放按 TD 误差优先级采样（工程对应） | 主轮 arxiv kept 7 条 | 有实现 | 2016-2025 |
| recall-gated 选择性巩固 | 只有被回忆的记忆获得巩固（eLife reviewed preprint） | web-only（eLife 90793v2，不入库） | 理论+仿真 | 2024 |

**差距判定**：**已有方案可采纳**（级联与分期回放都有可移植的现成形式；SHY 是唯一需自研适配的部分）。
**采纳建议**：
1. 把标量 ω 换成 5 级离散级联（Benna-Fusi 参数表可直接抄），每级时间尺度 ×2^k——同时给 M3 提供多时间尺度物理载具（A 区联动）。
2. M4 回放调度加"分期"：模拟睡眠相位=批内 Hebbian 重放（S186 的 sleep-like 相位已是现成 ANN 实现）+SHY 缩放（回放后统一乘 β<1 再按使用度恢复）。
3. 序列回放：把当前事件级回放升级为 2-3 事件序列回放（正/反向各半），测试对 G 区转移统计的加速。
**近期动态（2024-2026）**：Selective consolidation via recall-gated plasticity（eLife 90793v2）；Closed-loop TMR slow-wave motor（Nat Comm 2025, 10.1038/s41467-025-57602-2）；Awake replay: off the clock but on the job（TINS 2025）。

## E_wm 工作记忆容量基准 / θ-γ 相位码（M6）

**检索式**（主轮 crossref）："theta gamma coupling working memory"(25) / "working memory capacity slots resource model"(25) / "activity-silent working memory"(20) / "n-back computational model capacity"(15)；arxiv："working memory neural network capacity"(15)；supp："theta gamma coupling working memory"(10)；supp arxiv："theta gamma nested oscillation memory"(10)。funnel：fetch 595 → 去重后 106 → 粗筛 92 → 细筛 92。

**HCM 现状对照**：M6 = 4 槽位+权重竞争 w_j←max(w_j+η_w(cos−mean),0)，LOAD-BEARING。无 θ-γ 嵌套（roadmap D2=相位复用码，Lisman-Idiart 已引但未实现）。容量基准=自造 fig3，无外部基准锚定。

**现成方案表**：

| 方案 | 一句话机制 | 来源 | 成熟度 | 年代 |
|---|---|---|---|---|
| θ-γ 相位码理论（Lisman-Idiart 谱系） | 7±2 项=θ 周期内 γ 子循环的相位复用 | 主轮（screen_fine E 区头部） | 理论 | 1995-2010 |
| 人类海马 θ-γ phase-phase 耦合 | iEEG 证据：γ 子循环精确锁定 θ 相位（多项目 WM） | S406 | 实证（iEEG） | 2015 |
| hipp-mPFC θ-γ 长程耦合 | 跨区耦合在空间 WM 成功试次增强；基因模型+光遗传 | S408 | 实证 | 2017 |
| tACS 驱动 θ-γ 提升老年 WM | 峰耦合 TGCp-tACS 因果证据（DLPFC 多次） | S414 | 实证（因果） | 2024 |
| θ-γ 耦合预测 WM 表现 | 年轻/老年对照：耦合强度预测容量 | S425 | 实证 | 2024 |
| CFC 非线性统一机制 | Class II 群体间歇放电→CFC 是非线性普遍产物 | S415 | 理论+数据 | 2025 |
| persistent vs activity-silent 之争 | PFC 吸引子与静默痕迹交互产生 serial bias | S455 | 实证+建模 | 2020 |
| 细胞基础综述（activity-silent） | 短期突触可塑性作为 WM 载具的细胞证据 | S454 | 综述 | 2019 |
| 快 Hebbian WM 模型 | 快速突触可塑性替代持续放电：抗干扰+多项目 | S487 | 有实现 | 2003 |
| 槽位 vs 资源模型 | 容量 4±1 的槽位/资源之争与混合模型 | 主轮 screen_fine E 区 | 理论+实证 | 2001-2020s |

**差距判定**：**已有方案需适配**。θ-γ 相位码有成熟理论与因果实证，但移植到 HCM 要把"振荡相位"映射为离散槽位循环——roadmap D2 已规划，缺的是把容量基准换成可复现的外部对照（S406/S425 的人类 4±1 数字）。
**采纳建议**：
1. M6 实装 roadmap D2：θ 槽位循环=外层 t 步一循环，γ=循环内 4 子相位，每子相位绑定一个槽位码（相位复用=时间维度堆叠，与 A 区多时间尺度正交）。
2. 容量基准：新增 WM 容量扫描实验（负载 1-8），对照人类 4±1 与 S487 模型曲线；把"容量 4"从假设变实测。
3. activity-silent 联动：M6 的静默期用 M1 短期痕迹（fast Hebbian S487 思路）承载，槽位循环作为重激活机制——统一 S455 的"交互"论。
**近期动态（2024-2026）**：Trained RNNs phase-locked limit cycles in WM（PMC10868787）；tACS θ-γ 系（S414/S425）；θ-γ sequential memory/imagination model（Front Neural Circuits 2025）。

## F_capacity 情节容量解析界（M2）

**检索式**（主轮 crossref）："modern Hopfield network capacity"(25) / "dense associative memory capacity"(20) / "associative memory capacity sparse coding"(25) / "Hopfield storage capacity scaling"(20)；arxiv："associative memory capacity theory"(20)；supp："Hopfield network storage capacity"(10)；supp arxiv："modern Hopfield network capacity"(10)。funnel：fetch 630 → 去重后 133 → 粗筛 124 → 细筛 75。

**HCM 现状对照**：M2=Hopfield 单次写入 LOAD-BEARING；容量实证=自造 fig3；roadmap C1=k-稀疏 WTW 码负载 p 下虚警率解析界（Willshaw 谱系）——纯推导目标未完成。

**现成方案表**：

| 方案 | 一句话机制 | 来源 | 成熟度 | 年代 |
|---|---|---|---|---|
| Hopfield 最大存储容量（热力学极限） | 0.14N 界的推导与对角修正推广 | S562 | 理论 | 2017 |
| 自突触突破 0.14N | autapses+冗余态把容量推过 0.14N（代数增长） | S578 | 理论+仿真 | 2019 |
| 前馈连接增容 | Hopfield+前馈结构提高容量的构造界 | S579 | 理论 | 2022 |
| 稀疏 Hopfield 严格界 | dilute 连接下的严格容量界 | S564 | 理论 | 1992 |
| 极稀 Hopfield 容量 | extremely diluted 模型容量解析 | S569 | 理论 | 2004 |
| 有限尺度 Q-state 容量+盆地 | 有限 N 的容量与吸引盆尺度 | S566 | 理论+仿真 | 1992 |
| 容量-简单性困境 | 小网络+延迟反馈实现大容量的构造 | S586 | 理论 | 2012 |
| MHN 容量（现代谱系） | 指数容量 MHN、连续态 Hopfield、流形假设下的容量 | （主轮 arxiv kept 10 条 + supp） | 理论 | 2016-2025 |
| 流形假设下 MHN 容量 | 数据在低维流形时的容量修正 | S1052 (2503.09518) | 理论 | 2025 |

**差距判定**：**已有方案可采纳**（解析界工具齐全；HCM 缺的只是把自家 k-稀疏 WTW 码代进现成框架）。
**采纳建议**：
1. 完成 roadmap C1：以 S564（稀疏严格界）+ S569（极稀）为模板，对 k-稀疏 WTW 码在负载 p 下推导虚警率上界——这是论文"理论完备性"的最大缺口。
2. fig3 加两条参照线：0.14N（S562）与流形修正（2503.09518），让实证容量有可比锚。
3. autapse 思路（S578）若 HCM 码允许自连接，可作廉价容量增益实验。
**近期动态（2024-2026）**：Yet another exponential Hopfield model（Physica A 2025）；Neuron–astrocyte associative memory（PNAS 2025）；Capacity of MHNs under the Data Manifold Hypothesis [2503.09518]。

## G_structure 组合结构 / 后继表征 / 图（B3，未实现）

**检索式**（主轮 crossref）："successor representation hippocampus"(25) / "structural learning hippocampus graph"(20) / "vector symbolic architecture binding"(25) / "relational binding episodic memory"(20)；arxiv："structured episodic memory graph"(15)；supp："successor representation hippocampus"(10) / "cognitive map relational memory"(10)；supp arxiv："successor representation generalization"(10) / "graph structured memory agent"(10)。funnel：fetch 695 → 去重后 119 → 粗筛 43 → 细筛 41（G 区查询词最"专"，命中率集中）。

**HCM 现状对照**：B3 roadmap=后继链接=码间转移统计（CA3 链动机），未实现；无 schema/图结构。

**现成方案表**：

| 方案 | 一句话机制 | 来源 | 成熟度 | 年代 |
|---|---|---|---|---|
| SR 与时间上下文（TCM≈SR） | TCM 变体=直接学 SR：时间上下文漂移即后继表征学习 | S614 | 理论桥 | 2012 |
| Hebbian 时间联想=unnormalized SR | 无归一化 SR 与纯 Hebbian 时间联想在均匀状态频率下恒等 | S616 | 理论 | 2024 |
| SR 支撑层级抽象 | 社区结构任务中 SR 提供子目标抽象 | S617 | 实证+建模 | 2024 |
| 人类 RL 中的 SR | 行为+fMRI 证据：人类奖励学习用 SR 泛化 | S604 | 实证 | 2017 |
| SR 的计算逻辑与神经基底 | SR 作为预测表征的理论定位综述 | S605 | 综述 | 2018 |
| 概率化 SR（上下文依赖） | 贝叶斯上下文混合下的 SR 推广 | S610 | 理论+行为 | 2024 |
| 海马=SR 智能体（生物模型） | 位置细胞/网格细胞与 SR 的对应综述 | S619 | 综述 | 2022 |
| 结构学习与海马 | 海马对任务统计结构的学习（经典） | S627 | 实证 | 2007 |
| VSA 最优二次 binding | 关系推理的向量符号绑定算子 | S640 | 理论 | 2023 |
| 情节的 relational binding | 界面绑定/磨损（mnemonic discrimination 双过程） | S678 | 实证 | 2019 |
| 图式快速同化 | fast mapping/schema assimilation 的计算模型（见 screen_fine G 区） | （screen_fine G 区） | 理论+实证 | 2010s |

**差距判定**：**已有方案需适配，且有一个漂亮的统一机会**——S614+S616 表明 HCM 的 B3（码间转移统计）与 M3（上下文 EMA）本是同一族：TCM 的漂移即 SR 的另一种学习规则。实现 B3 时应同时服务 M3，而不是两条独立机制。
**采纳建议**：
1. B3 实装为码间转移矩阵 M[i,j]（计数+衰减），归一化行→SR；读出时 query 码做一步 SR 传播（M·a）支持联想泛化。
2. 与 M3 融合：M3 的多时间尺度 EMA 通道直接生成不同折扣率的 SR（γ=0.9/0.99/0.999 对应 τ=10/100/1000）——一处实现两区收益。
3. 图式/结构化：schema 层暂缓，先用 SR 的社区结构实验（S617 范式）做最小验证。
**近期动态（2024-2026）**：DAG-Based Memory Architecture for Structured LLM Agents（zenodo 15347105）；NeuralPlayground（海马模型标准化评测环境，UCL 2024）；Graph-Empowered Agentic AI（KAIST 讲义 2025）；grid-like schema code in humans（Cell 2026）。

## H_scale 稀疏索引 / 子线性检索（工程）

**检索式**（主轮 arxiv）："approximate nearest neighbor search survey"(20) / "sublinear retrieval associative memory"(15) / "memory augmented neural network scaling"(15)；crossref："sparse distributed memory"(15)；supp："sparse distributed memory"(10) / "locality sensitive hashing similarity search"(10)；supp arxiv："HNSW vector search"(10) / "sparse associative memory retrieval"(10)。funnel：fetch 280 → 去重后 109 → 粗筛 70 → 细筛 53。

**HCM 现状对照**：稠密 M 9.1KB/条→CSR≈0.5-1KB 规划中（roadmap A1/A2）；L3 线性扫 450s→倒排+top-C 精排 <1s 目标（roadmap A4）；时间分片库规划中。全部未实现。

**现成方案表**：

| 方案 | 一句话机制 | 来源 | 成熟度 | 年代 |
|---|---|---|---|---|
| ANN 维度缩减综述（2024） | 降维×ANN 检索的系统综述+评测 | S854 | 综述+基准 | 2024 |
| Filtered ANN 综述（2025） | 属性过滤混合查询的向量检索综述 | S852 | 综述 | 2025 |
| 图基 ANN 全面综述+实验对比 | HNSW 系图索引的权威对比（引用>2000） | S853 | 综述+基准 | 2021 |
| Filtered ANN 统一基准 | 过滤 ANN 的系统实验 | S856 | 基准 | 2025 |
| Falconn++ | LSH 家族的现代实现（多探针+自适应） | S714 | 有实现 | 2022 |
| SOAR | 改进图索引排序的 ANN | S715 | 有实现 | 2023 |
| Kanerva SDM 概率解释 | SDM=高维二值空间的条件概率读出 | S733 | 理论 | 1989 |
| SDM 专家记忆解释 | SDM 复现专家提取速度+tip-of-tongue 非线性 | S732 | 有实现 | 2014 |
| Integer SDM | 模块算术整数 SDM：保留 SDM 性质+支持遗忘+序列存储 | S736 | 有实现 | 2013 |
| 稀疏重建哈希 | 大规模 ANN 的稀疏重建哈希 | S702 | 有实现 | 2025 |
| HNSW 工程（d-HNSW） | 分离内存架构上的 HNSW | S1063 (2603.13591) | 有实现 | 2026 |

**差距判定**：**已有方案可采纳**（工程问题无需自研算法，选型+集成即可）。
**采纳建议**：
1. 索引选型：k-稀疏 WTW 码天然适合 inverted index（posting list 按 1-bit 索引，查询交集=top-C 候选）——这是 SDM/Willshaw 的原生寻址方式，比引入 HNSW 更贴近 HCM 动力学内核；保留 HNSW（S853）作语义嵌入层的对照。
2. 时间分片：按 τ 分层（A 区多时间尺度）+ 按会话分片，冷库归档。
3. S736 Integer SDM 的"支持遗忘"思路直接对接 M8：剪枝=posting list 淘汰。
**近期动态（2024-2026）**：d-HNSW: disaggregated memory vector search [2603.13591]；Memory by Design: Probabilistic Sequence Layers [2605.31163]；filtered ANN benchmark S856。

## I_confidence 置信度 / 贝叶斯读出（M0）

**检索式**（主轮 crossref）："confidence recognition memory signal detection"(25) / "metamemory confidence calibration"(20)；arxiv："uncertainty aware retrieval"(15)；supp："confidence memory retrieval metacognition"(15)；supp arxiv："confidence calibrated retrieval"(10)。funnel：fetch 385（含 fix2 同义词轮：feeling of knowing tip of the tongue accuracy / metacognitive monitoring memory accuracy / memory strength belief retrieval fluency，各 n=10）→ 去重后 112 → 粗筛 78 → 细筛 74（细筛 74 中大量是"置信区间校准"类统计噪声，真正相关约 12-15 条——**本区文献密度最低**，符合"最难统一"预期；已用 6 组检索式，如实报告）。

**HCM 现状对照**：M0 = raw_evidence=cos(s_q, D·a)，三分支 RECALL/SEPARATE/ABSTAIN；实验内 AUC=1.000（过于完美）；真实数据校准未做；无贝叶斯读出。

**现成方案表**：

| 方案 | 一句话机制 | 来源 | 成熟度 | 年代 |
|---|---|---|---|---|
| 置信+反应时 OU 扩散模型 | 匹配度→证据分布→置信判据间面积作漂移率，racing OU 过程 | S763 | 理论+拟合 | 2009 |
| 高置信错误（误提取账户） | 元监测失败源于误提取而非阈值问题 | S798 | 实证 | 2007 |
| 挑战高斯 SDT 的 morph 实验 | 等选择率下信心不同 → 信心≠似然比 | S776 | 实证 | 2009 |
| 极值 SDT | 用极值分布替代高斯的参数化路线（选择集不变性） | S768 | 理论+实证 | 2026 |
| 元记忆×语义记忆 | 元记忆判断在语义/情节域的分离 | S800 | 实证 | 2012 |
| 校准研究议程 | confidence calibration 的方向性综述 | S801 | 综述 | 2013 |
| RAG 不确定性感知检索 | UMG-RAG：粒度=可靠性估计，训练自由 | S875 | 有实现 | 2026 |
| 检索几何内生置信 | Resonance Gate：向量符号记忆的检索几何→置信 | web-only（zenodo 21446859，不入库） | 有实现（preprint） | 2026 |
| 贝叶斯读出（一般工具） | 扩散决策/贝叶斯置信的经典引用网 | （S763 及其引用网） | 理论 | 2000s |

**差距判定**：**已有方案需适配 + 部分真空白**。单项目置信有成熟模型（S763），但"检索式记忆系统的端到端校准基准"仍缺（S875/Resonance Gate 是 2025-2026 才出现的早期尝试）。
**采纳建议**：
1. M0 读出升级为两阶段：raw_evidence → 证据分布（同码多次检索的方差）→ S763 式判据面积作置信度；三分支阈值改为校准曲线（reliability diagram）而非固定 0.5。
2. AUC=1.000 是红旗：换真实语料（bench 已有的干扰集）重测；报告 ECE（期望校准误差）而不只 AUC。
3. S776 的教训：置信度应含"检索集内部一致性"项（同码多路径收敛度），避免单点 cos 值的过度自信。
**近期动态（2024-2026）**：The Resonance Gate（zenodo 21446859，向量符号记忆内生置信）；UMG-RAG [S875]；extreme-value SDT [S768]。

## J_llm_memory LLM 记忆系统全景（对位基线）

**检索式**（主轮 arxiv）："large language model agent memory survey"(20) / "memory augmented LLM agent"(20) / "episodic memory large language model"(15) / "retrieval augmented agent long-term memory"(15)；supp arxiv 定点："agent memory survey"(15) / "generative agents memory reflection"(10) / "MemGPT LLM operating system"(5) / "generative agents interactive simulacra"(5) / "Mem0 scalable long-term memory"(5) / "Zep temporal knowledge graph"(5) / "HippoRAG neurobiologically inspired long-term"(5) / "A-MEM agentic memory Zettelkasten"(5) / "LongMemEval long-term interactive memory"(5) / "Larimar episodic memory control"(5) / "MemOS memory operating system"(5) / "MemAgent long-context reinforcement"(5)；fix 轮 6 条定点（Rethinking / Memory Age / LongMemEval / MemOS / MemAgent / MemoryBank）。funnel：fetch 145 → 去重后 62 → 粗筛 55 → 细筛 56。

**HCM 现状对照**：项目定位=他们做符号管理层、HCM 做动力学内核；notes/lit/llm-memory-systems-routes.md 已有六路线（MemGPT/GenAgents/Mem0/Zep-Graphiti/HippoRAG1-2/A-MEM）；对照臂=HippoRAG。

**现成方案表**（系统+基准+综述三类）：

| 方案 | 一句话机制 | 来源 | 成熟度 | 年代 |
|---|---|---|---|---|
| MemGPT | LLM OS 分页记忆（主存/外存中断驱动） | S1076 (2310.08560) | 有实现 | 2023 |
| Generative Agents | 经验流+recency/importance/relevance 三因子检索+反思 | S1077 (2304.03442) | 有实现 | 2023 |
| Mem0 | CRUD 记忆管线（抽取-更新-冲突处理） | S1080 (2504.19413) | 有实现+基准 | 2025 |
| Zep/Graphiti | 双时间轴动态知识图谱（有效时间/事件时间） | S1081 (2501.13956) | 有实现+基准 | 2025 |
| HippoRAG 1/2 | 海马索引理论+PPR 多跳检索；v2 非参数持续学习 | S1082 (2405.14831) / S1098 (2502.14802) | 有实现+基准 | 2024-25 |
| A-MEM | Zettelkasten 卡片盒：笔记式互链记忆 | S1083 (2502.12110) | 有实现 | 2025 |
| MemOS | 记忆操作系统：MAG 三类记忆统一调度 | S1089 (2505.22101) | 有实现 | 2025 |
| LongMemEval | 长程对话记忆基准（500+ 问题，5 能力） | S1084 (2410.10813) | 基准 | 2025 |
| LoCoMo | 超长对话评估（多 session 问答/事件摘要/多模态） | S907 | 基准 | 2024 |
| Larimar | 稀疏分布式记忆体+LLM（情景编辑/泛化） | S896 | 有实现 | 2024 |
| MemoryBank | 遗忘曲线人格化长期记忆 | S1096 (2305.10250) | 有实现 | 2023 |
| MemAgent（RL） | 多轮 RL 学记忆管理策略（长上下文） | S1095 (2507.02259) | 有实现 | 2025 |
| MemInsight | 自主记忆增强（结构化属性标注） | S884 | 有实现 | 2025 |
| Echo | 时间情景记忆的 LLM | S894 | 有实现 | 2025 |
| Agent Memory Survey（权威综述） | 记忆机制四象限分类（写/管/读/用） | S877 | 综述 | 2024 |
| Context Engineering 综述 | 上下文工程全景（含记忆节） | S883 | 综述 | 2025 |
| NIAH-Memory | 记忆型大海捞针基准 | S900 | 基准 | 2024 |
| TA-Mem / DYNA / KARMA / MemFit / CoreMem | 2026 新一代（工具增强/动态网络/具身/高效/黎曼检索） | S902/S897/S903/S905/S904 | 有实现 | 2026 |

**差距判定**：**已有方案可采纳（对位完成度高）**。J 区的"缺口"不是文献而是 HCM 在该版图中的实证站位——对照臂只有 HippoRAG 一个。
**采纳建议**：
1. 对照臂扩到 3：HippoRAG 2（非参数持续学习）+ Mem0（CRUD 管线）+ LongMemEval（基准对齐）；LLM 记忆系统的"写入-管理-读出-使用"四象限（S877 分类）每格给一行 HCM 对应机制，写进论文 §2.7 或附录。
2. LongMemEval 的 5 能力（信息抽取/跨 session 推理/时序推理/拒答/冲突）逐条映射到 HCM 模块——顺便检验 M0 ABSTAIN 分支（拒答能力）。
3. MemOS 的记忆调度视角与 HCM 的"调质三通道"（C 区建议）呼应：调度策略=元认知层，值得在讨论节对位。
**近期动态（2024-2026）**：Memory in the Age of AI Agents [2512.13564]；Rethinking Memory in LLM based Agents [2505.00675]；MemoryTax: Bitter Lesson of Agent Memory [2610.29823]；BMAM: Brain-inspired Multi-Agent Memory [2601.20465]；LangMem SDK（LangChain 2025）；MemAgent: heterogeneous memory providers [2609.32521]；Memory in the LLM Era（GitHub Yanchen398 全景清单）。


## 总览矩阵

| 区 | 最接近的现成方案 | HCM 缺口 | 差距判定 | 建议动作 |
|---|---|---|---|---|
| A_timescale (M3) | TCM 层次漂移 S025 + 时间细胞 S002 + 多尺度突触 S041/S048 | 单尺度 EMA（消融 REDUNDANT）；B2 未实现 | 已有方案可采纳 | 4 通道 EMA 堆叠（τ=1/10/100/1000）+ MMO 振荡载具 + 多通道一致性入 M0 |
| B_forget (M8) | MESU 不确定性遗忘 S187 + functional forgetting 实证 S128 + 统一综述 S1129 | M8 INERT：机制在、收益未展示（缺干扰/增量场景） | 已有方案可采纳 | 三档功能开关（容量/干扰/泛化）+ 多轮干扰消融（预期 INERT→LOAD-BEARING） |
| C_neuro (M7) | 资格痕迹 S276/S278 + LC-NE 增益理论 S248 + DA RPE S252 | 单通道 g_t、无 DA/ACh/NE 分化、无资格痕迹 | 已有方案需适配 | 三通道分化 + 突触资格痕迹 e_i + 延迟奖赏消融场景 |
| D_consolidation (M4) | Benna-Fusi 级联（screen_fine D 区 Fusi 系）+ sleep-like 相位 S186 + TMR S376/S374 | 标量 ω 代理级联；无睡眠分期、无 SHY、无序列回放 | 已有方案可采纳 | 5 级离散级联 + 分期回放（Hebbian 批+SHY 缩放）+ 2-3 事件序列回放 |
| E_wm (M6) | θ-γ 相位码理论（Lisman-Idiart 谱系）+ iEEG 证据 S406/S414/S425 | 无嵌套振荡；容量基准自造 | 已有方案需适配 | roadmap D2 相位复用 + 负载 1-8 扫描对照人类 4±1 + activity-silent 联动 M1 |
| F_capacity (M2) | 稀疏 Hopfield 严格界 S564/S569 + 0.14N 界 S562 + MHN 流形修正 S1052 | roadmap C1 解析界未完成；fig3 无外部锚 | 已有方案可采纳 | k-稀疏 WTW 负载 p 虚警率上界推导 + fig3 加参照线 |
| G_structure (B3) | TCM≈SR 桥 S614 + Hebbian=unnormalized SR S616 + SR 层级抽象 S617 | B3 未实现；与 M3 的同族性未被利用 | 已有方案需适配 | 码间转移矩阵 SR + 与 M3 多尺度 EMA 融合（γ 扫描一处两区） |
| H_scale (工程) | SDM 家族 S733/S732/S736 + ANN 综述 S853/S854/S852/S856 | 倒排/CSR/时间分片全部未实现 | 已有方案可采纳 | inverted index（k-稀疏 WTW 原生寻址）而非 HNSW；HNSW 留作嵌入层对照 |
| I_confidence (M0) | OU 扩散置信模型 S763 + 极值 SDT S768 + UMG-RAG S875 | AUC=1.000 红旗；无真实语料校准；无贝叶斯读出 | 需适配 + 部分真空白（端到端检索记忆校准基准仍缺） | 两阶段读出（证据分布→判据面积）+ ECE 报告 + 检索集内部一致性项 |
| J_llm_memory | HippoRAG 1/2 S1082/S1098 + 四象限综述 S877 + LongMemEval S1084 | 对照臂单一（仅 HippoRAG） | 对位完成度高，缺实证站位 | 对照臂扩到 3 + LongMemEval 五能力逐条映射（含 M0 拒答） |

**总判定**：10 区无一是"真空白"——每个薄弱环节都有可追溯的现成理论/实现/基准；最接近真空白的两处是 I 区（检索式记忆系统的端到端校准基准，2025-2026 才有早期尝试 S875）与 A 区（HCM 式离散码上的多时间尺度——理论齐备但须自行适配到 WTW 码）。G 区存在"一处实现两区收益"的统一机会（S614/S616）。

## 附表：分区分查询 funnel 明细（全量 100 条查询，主轮 48 + supp 43 + fix 6 + fix2 3）

| 区 | fetch（crossref 行预算+arxiv 实取） | 去重后 | 粗筛 | 细筛 |
|---|---|---|---|---|
| A_timescale | 770 | 120 | 61 | 37 |
| B_forget | 765 | 147 | 95 | 95 |
| C_neuro | 890 | 105 | 66 | 63 |
| D_consolidation | 820 | 126 | 118 | 99 |
| E_wm | 595 | 106 | 92 | 92 |
| F_capacity | 630 | 133 | 124 | 75 |
| G_structure | 695 | 119 | 43 | 41 |
| H_scale | 280 | 109 | 70 | 53 |
| I_confidence | 385 | 112 | 78 | 74 |
| J_llm_memory | 145 | 62 | 55 | 56 |
| **合计** | **5 975** | **1 139** | **802** | **685** |

注：crossref 通道 API total-results 为百万级（OR 语义），不作 funnel 起点；fetch=每查询实取行预算 Σmax(n*6,14)。arxiv 通道 raw（opensearch totalResults）见 refs/survey/query-stats.json。去重键：DOI → arxiv_id → 规范化标题前 120 字符。粗筛/细筛规则与阈值见"方法与筛选漏斗"。各节表格为人工复核后的条目，screen 输出（screen_coarse.tsv / screen_fine.jsonl）是排序器而非终审。