# 草稿 · §1 三主张重写（工单三「去日志化重压缩」D+10 执行件；先备稿）

状态：草稿（未动 PAPER_EN_final.md 正文）。约束：只改叙事不动数据；check_claims 327/327 每节护航；可复现锚点保留（当前 25 处）；不动标题/摘要/引文；不动 exp22o 伪影四与 Limitation 7 原文。

## 现状诊断（§1 为何要重写）

- 现 §1 是「四贡献」清单（mechanism re-location / falsifiable tests / engineering scalability / gain scale law），其中：贡献 3 的复杂度行「writing costs O(k), retrieval O(LNk)」已过时——exp43 实锤稠密写门 O(N²)（0.82–0.96 s/条@1e5）、检索被 f64 上卷主导（137× 修复）；贡献 4 把「残差叙事」与「负结果方法论」混在一条，而标题已把 log-N 坍缩律立为头条。
- 工单要求 §1 压缩为三主张框架：**law / dual-readout / verifiability**，与标题、摘要三主张一一对应。

## §1 改写稿（EN，供 D+10 直接落地）

（第 1-2 段保留现文开场张力与「engineering-executable 三缺」段落，微调衔接句；以下替换「Four main contributions」清单为三主张。）

Three claims, each falsifiable and each tested:

1. **A log-N collapse law in attractor memory.** The retrieval softmax gain β is not a free parameter: as library size N grows, retrieval contrast collapses unless β scales as log N. We give the analytical basis (Appendix E), a dose–response map of β across library sizes (Sec. 4.12), a pre-registered online-learning test in which the model itself tracks β*(N) across checkpoints on a 92,760-message corpus (Sec. 4.x, exp45), and a cross-encoder transfer test of the same rule. [数据行一律引用现文数字，不新增。]

2. **A dual-readout architecture for complementary memory.** Pattern separation is re-located: expansion plus k-WTA sparsification does not improve item separability (code cosine 0.872 at overlap 0.90); subtracting a slowly-varying expectation does (0.308). A single projection must be read out twice — sparse code for writing, dense evidence for retrieval — and the two readouts are consolidated by interleaved replay into a semantic store that the episodic retrieval can consult (lam-fusion; Sec. 4.10 direct comparison).

3. **Verifiability as a first-class result.** Every operator is a computable operator with stated parameters; claims are machine-checkable assertions (327/327 at submission); the four-tier scale protocol L0–L3 runs from laptop to 2×10⁷ items (sparse column storage ~210 GB vs dense ~5.2 TB). Complexity is reported as measured, not as advertised: dense writing is O(N²) (dup gate; 0.82–0.96 s/item at N=10⁵) with an inverted-index fastpath measured at 50–197× (Sec. 4.x, exp43); retrieval is O(LNk) and was dtype-dominated until the float32 chain (137×). Negative results are reported verbatim, with a reusable instrument-first protocol for judging them (Secs. 4.12, 4.13).

（收束句：The difference from prior work is not a novel learning rule, but ... ——保留现句。）

## 连带改动点（D+10 清单，与 §4 重分组、Appendix H 砍 60%、§9 内部叙事删除合并执行）

1. §4 重分组按三主张：现 §4.2-RQ1→主张 2；§4.12/新 exp45/exp19g 段→主张 1；§4.3/4.4/4.5/4.6/4.9/4.10/4.13/4.14→按归属挪序；§4.7 负结果汇总保留独立小节。
2. §1 贡献 3 的 O(k)/O(LNk) 行随三主张稿修正（与 docs/MODEL.md §14 2026-11-09 实测修订行同口径）。
3. exp45 落地后 §4 新增小节标题占位：「4.x β*(N) online learning and cross-encoder transfer (pre-registered)」。
4. Appendix H：砍 60% = 只留「外部评审响应」七条各 2-3 句；§9 内部修订叙事全删，只留一节外部响应摘要 + 修订索引表。
5. 每节改完即跑 check_claims（327/327 护航）；全部完成后 wave-8 regen（CN/EN_final/EN token/PLOS/arXiv 五链）+ 页数检查 45-50 页。


---

# D+1 执行稿（2026-11-12 05:40，EN 链先行；CN/token 镜像=D+10）

状态：执行中（EN_final / EN_token / ARXIV repo / ARXIV staging 四文件同文替换）。本节取代上文「供 D+10 直接落地」的初稿措辞——初稿有三处未核实锚点，已修正（见下）。

## 初稿三处修正（探索中发现）

1. **初稿 claim 3 引用的 exp43 数字（50–197×/137×）与 exp37 数字（0.82–0.96 s/item）均不在现稿**——§7 只有 L1 pilot（N=2×10⁴），Appendix D 只有 exp17。直接写进 §1 会造成无锚数字。修正：本波同时把 §7 补 L2 pilot 段（exp37）+ Appendix D 补 exp43 段，claim 3 锚到 (Sec. 7)/(Appendix D)。
2. **§2 L35 悬空指针**：「a verified operating point (β=100 full-pass at N=10⁵, §7)」——§7 原文并无 10⁵ 结果。E2 落地后该指针闭合。
3. **five/seven 漂移**：§1 para2 EN_final/EN_token=「five benchmark families」、ARXIV=「seven」。§4.1 明文「Five synthetic generator families」。修正：新句显式拆分为「five synthetic generator families (Sec. 4.1) + two real corpora (Secs. 4.13, 4.20)」，漂移消灭。
4. （附发现）**§8 陈旧**：「reproduces all 19 steps」实际 reproduce_all.py STEPS=48（含 build/export）；bench 枚举止于 exp25。E4 修正。

## E1. §1 替换块（四文件同文）

Para 2 整行替换（两种旧变体 five/seven 均换）：
> This paper supplies all three. We present HCM — a nine-operator, single-file, NumPy-only reference implementation — and systematically test the core claims of CLS on the five synthetic generator families (Sec. 4.1) and on two real corpora (Secs. 4.13, 4.20).

「Four main contributions:」清单（含 items 1-4）整块替换为：

> Three claims, each falsifiable and each tested:
>
> 1. **A log-N collapse law in attractor memory.** The retrieval softmax gain β is not a free parameter: as library size N grows, retrieval contrast collapses unless β scales as log N. We give the analytical basis (Appendix E), a dose–response map of β across library sizes (Sec. 4.12), a cross-encoder replication of the turning point (Sec. 4.21), and a verified operating point — β=100 full-pass at the 10⁵-item tier (Sec. 7).
>
> 2. **A dual-readout architecture for complementary memory.** Pattern separation is re-located: expansion plus k-WTA sparsification does not improve item separability (code cosine 0.872 at overlap 0.90); subtracting a slowly-varying expectation does (0.308). A single projection must be read out twice — the sparse code for writing, the dense evidence for retrieval — and the two readouts are consolidated by interleaved replay into a semantic store that episodic retrieval can consult (Secs. 4.2, 4.4, 4.10).
>
> 3. **Verifiability as a first-class result.** Every operator is a computable operator with stated parameters; claims are machine-checkable assertions (327/327 at submission); the four-tier scale protocol L0–L3 runs from laptop to 2×10⁷ items (sparse column storage ~210 GB vs dense ~5.2 TB). Complexity is reported as measured, not as advertised: writing is O(k) on the sparse path, but the duplication gate makes the measured write cost O(N²) at scale — 0.82–0.96 s/item at N=10⁵ (Sec. 7) — with an opt-in inverted-index fastpath measured at 50–197× (Appendix D); retrieval is O(LNk) and was dtype-dominated until a float32 chain (137×, Appendix D). Negative results are reported verbatim, with a reusable instrument-first protocol for judging them (Secs. 4.12, 4.13).
>
> The difference from prior work is not a novel learning rule, but the composition of a closed loop out of literature-derived mechanisms, followed by minimal decisive experiments that determine, for each mechanism, whether it is actually doing the work or merely "sounds plausible". This process produced several conclusions that contradict popular narratives; we consider those conclusions valuable in themselves.

（旧 para2 尾句「The difference from prior work…valuable in themselves.」随清单替换移到三主张之后作收束段；「This paper supplies all three.」保留在新 para2 开头。）

## E2. §7 L2-pilot 段（四文件，插在「> **L1 pilot:」blockquote 行之后、空行+「**Five questions」之前）

> **L2 pilot at N=10⁵ (exp37).** Four workers — three no-residual seeds and one with-residual seed (the memory-capped cloud protocol de-scoped the remaining with-residual seeds and the 2×10⁵ tier) — each attempted 10⁵ writes of synthetic unit patterns (d_x=2,048, d_h=16,384, k=328). Three workers completed the full battery: 60/60 exact recall at degradation 0 and 60/60 at degradation 25%, abstain 0 in every block (results/exp37_N100000_F_s1.json, exp37_N100000_F_s2.json, exp37_N100000_T_s0.json) — gate G-S1 (β=100 sufficiency: recall ≥ 0.90 at 25% degradation) PASSES at the 10⁵ ceiling, extending the dose–response of Sec. 4.12 two scale steps beyond the L1 pilot; the completed with-residual arm scoring full marks is consistent with residual neutrality at this tier. The fourth worker (no-residual, seed 0) completed writing and its deg-0 probes, then terminated out-of-memory during evaluation; its logs are retained. Write cost honestly FAILS the linearity gate (G-S3): the duplication gate scans the library per write, so measured writing is O(N²) — 0.82–0.96 s/item at N=10⁵, peak resident 28.2 GB per worker. The engineering remedies are measured in Appendix D.

（事实依据=40-EXP-LEDGER exp37 收官行 + bench/exp37_l2_scalelaw.py docstring：cells {F_b100,T_b100}×seeds{0,1,2}，resid=CELL.startswith('T')，G-S1/S2/S3 预注册，synthetic unit patterns only。G-S2 因缩编 2v1 未正式判决，故不写门判决、只写「consistent with residual neutrality」。）

## E3. Appendix D exp43 段（四文件，插在「Honest conclusions: …」行之后、空行+「## Appendix E」之前）

> **Measured at the L2 tier (exp43, N=10⁵).** The exp37 write bottleneck localises in the duplication gate's full-library scan. Two opt-in engineering paths, both exact (retrieval agreement |Δa| ≤ 4×10⁻¹⁰), were measured on identical data: (i) an inverted-index write fastpath replaces the scan with posting-list intersection — 3.45 ms/item vs 0.680 s/item on the reference machine (197×) and 8.1 ms vs 0.410 s on the cloud replica (50×) — reducing write-phase memory from 6,250 MB (dense) to 512 MB + 372 MB CSR (7.1×); (ii) a float32 retrieval chain cuts canonical query time from ~213 s to 1.16 s at N=10⁵ (137×). The canonical semantic path is unchanged; the fastpath is an engineering layer, not a model change.

## E4. §8 陈旧数字（四文件）

- 「A single driver reproduces all 19 steps end to end: bench/reproduce_all.py.」→「A single driver reproduces the full chain end to end (48 steps, including build and export): bench/reproduce_all.py.」
- 「bench/ (exp1_separation … exp25_collapse_external)」→「bench/ (exp1_separation … exp45_online_beta)」（描述目录现状）。
- 顺带统一：ARXIV §7 L1-pilot 标题「the subsequent correction」→ EN_final 版「third-round correction」。

## D+10 清单增补（在上文 5 条基础上）

6. claim 1 追加 exp45 从句（D+2 exp45 判决后）：「…and a pre-registered online-learning test in which the model itself tracks β*(N) across checkpoints on the 92,760-message corpus (Sec. 4.22).」
7. §4.22 新小节（exp45 结果）+ §4.7 负结果汇总补 exp44 行（判决如实）。
8. CN PAPER.md：§1 三主张译写 + E2/E3/E4 对应段（CN 链 build_paper.py regen）。
9. PLOS regen 时核对「327/327」表述与 check_claims 实际数一致。
## D+1 Wave 2 执行稿（§4 按主张重分组 — 阅读地图 + 标签，2026-11-12 05:5x）

**方案裁定**：不物理重排 §4（重编号会打断 25 处可复现锚点、PLOS SI 重映射、外部评审引用的 4.x 编号）；改为 §4 头部插入 Reading map 段 + 全部 21 个小节标题挂主主张标签。页数削减仍由 wave 3（App H）与 wave 4（§9）承担。

**变更一（标题去内部叙事）**：ENF/EN 标题去掉 round 标记，对齐 ARXIV 既有形态：4.12 去 "(third-round diagnosis)"、4.13 去 "(third round)"、4.14 "(round four, exp27/exp28)"→"(exp27/exp28)"、4.16 "(exp34, round four)"→"(exp34)"、4.17 同理、4.18 "(round five, exp36–36d)"→"(exp36–36d)"、4.19/4.20/4.21 "(round six, expN)"→"(expN)"。四副本标题现逐字一致（含 exp 编号保留）。

**变更二（Reading map，插 "## 4 Experiments" 标题后）**：Claim 1 ← 4.3/4.11/4.12/4.19/4.21；Claim 2 ← 4.2/4.4/4.10/4.13/4.14/4.18/4.20；Claim 3 ← 4.5–4.9/4.13/4.15–4.17；4.1=共享基础设施；scale ladder=Sec.7；复杂度补救=App D。

**变更三（21 标题挂标签）**：4.2 C2 / 4.3 C1 / 4.4 C2 / 4.5 C2→C3 / 4.6–4.9 C3 / 4.10 C2 / 4.11 C1 / 4.12 C1 / 4.13 *(Claims 2, 3)* / 4.14 C2 / 4.15–4.17 C3 / 4.18 C2 / 4.19 C1 / 4.20 C2 / 4.21 *(Claims 1, 2)*。4.1 不挂（共享 setup）。

**工程注记**：四文件为 CRLF——逐行处理须 `l.replace(/\r$/, '')` 归一后再锚定；JS `.` 不吃 `\r`，`(.*)$` 在 CRLF 行上必失败（本次 NOMATCH 根因）。

**验证**：四副本 tags=20/untagged=1(4.1)/map=1/deno-left=0 + wave1 五项检查全绿。CN 链不在此波（D+10 译写时同步 CN §4 头部）。

## wave-3 执行记录（§9 / Appendix H 砍 60%）

- **裁定**：§9（ENF 104 行 / EN 101 行 / ARXIV App H 104 行）→ 统一 4 小节结构：9.1 Independent adversarial reviews（三轮合并）、9.2 Independent clean replications（两次合并）、9.3 Consolidated revision record rounds 3–7（五行表，每行 scope+outcome+§4 指针）、9.4 Methodological summary。ENF→45 行、EN→45 行、ARXIV→33 行（ARXIV 体去行话、行号随体）。全量数字/结论保留（exp14 27.8pp、exp15 13.85±2.66 vs 14.80±3.96 p=0.32 dz=−0.24、exp13 corr −0.070、D-1..D-4、θ 0.000→1.000 全距等），删的只是轮次内部叙事。
- **指针重映射**（写规则前先对每文件 grep 现场）：ENF/EN §9.8→§9.2、§9.2 P1-1→§9.1（顺修历史悬空指针）、§9.9×2→§9.3、§9.1 不变；ARXIV 变体 Appendix H §9.8→§9.2、Appendix H §9.2 P1-1→§9.1、Appendix H §9.9×2→§9.3、(see Appendix H, §9.6)→§9.2（EMA 修正句）、Appendix H §9.1×2 不变。裸 §9 / Appendix H 指针（Limitation 7、AI 披露）仍有效不动。
- **ARXIV 双副本同步**：repo paper/submission/arxiv/ 与 staging <user-home>/arxiv_build/ 逐字节一致（162,322 chars，33 行）。
- **验证**：四文件 9.1–9.4 小节齐全；旧体痕迹（## 9.10–9.13、|#|issue| 表头、Methodological harvest）=0；stale §9.6–9.9 指针=0。
- **CN 链不动**：D+10 译写波做 CN §9 同步压缩 + CN 指针映射。
- **教训 L-067**：重映射规则必须从每个文件的实际文本 grep 现场生成，不能从 ENF 的规则泛化到 ARXIV（两体指针对照表不同：ENF 用裸 §9.x 形式，ARXIV 用 Appendix H §9.x 形式）。

---

# Wave-8 主线稿（2026-11-13 晚，S-010/S-011/S-012/S-013 落地件；等待 exp45/52 期间执行）

状态：草稿 v1（未动正文）。本节取代 Wave 2 的『不物理重排』裁定——S-011 升级为物理重排 §4 82K→40K，25 处可复现锚点与 PLOS 重映射在 D+10 regen 时统一重指向（接受成本，一次做完）。

## W1. 一句话主张（S-012 第一步；三候选，推荐 A）

**A（推荐，定律+双通道）**：Retrieval capacity in sparse attractor memory does not fall off a cliff as the store grows: it decays along a measurable log-N law that a single scalar — the retrieval temperature β — compensates across four orders of magnitude (10³→10⁶ items), and the same instrument that measures the law locates where episodic identity and semantic generalisation must be read out by separate channels.

CN：稀疏吸引子记忆的检索容量不随库增长而断崖下跌——它沿一条可测的 log-N 定律衰减，而单一标量（检索温度 β）在四个数量级（10³→10⁶）上补偿它；测量这条定律的同一套仪器，恰好定位了情景身份与语义泛化必须分道读出的位置。

**B（更紧）**：Sparse attractor memory obeys a log-N collapse of retrieval contrast under realistic interference, compensated by one inverse-temperature knob; we measure the law from 10³ to 10⁶ in-model, out-of-model, and on two real corpora, and use it to expose where fast identity and slow generalisation separate.

**C（带巩固价值钩子，连 S-013）**：We turn the complementary-learning-systems analogy into measured laws — a log-N collapse with β compensation across 10³→10⁶, and a consolidation-value theorem (replay cures retrieval only while its source is retained) — each stated as a falsifiable gate and decided by pre-registered experiment.

裁定待主人过目；§1 首段第二句将放此句（第一句保留现开场张力句）。

## W2. §4 物理重排地图 v2（21 小节 + 7 新小节 → 四发现章）

新 §4 骨架（发现级命名，小节重编号 D+10 regen 时统一）：

- **4.A The law: β compensation and scale（定律）**：现 4.3（容量瓶颈=去噪）＋4.12（β 剂量响应）＋4.11（生成器敏感性）＋4.21（残差边界与坍缩转折）＋§7 L1/L2 pilot（exp26/37 并入正文）＋新 §exp45（β*(N) 在线学习+跨编码器迁移）＋新 §exp52（1e6 收口档；明午数据）＋App F（exp25 模型外复现，升格入正文一小节）。
- **4.B The collapse: identity dies, priors survive（坍缩边界）**：新 §exp44（语义通道身份坍缩，8 seeds NEGATIVE）＋现 4.2（模式分离机制=re-location）＋4.10（双读出直接对比）＋4.19（公平基线 exp38/47：恒等任务 BM25 打平、吸引子对经典容量优势）＋4.13（真实语料法证——压缩为仪器叙事）。
- **4.C Consolidation value is retrieval-index value（巩固价值论题）**：现 4.4（交错回放必要）＋4.14（时间链 null→replay-link）＋4.20（DailyDialog 外部验证）＋新 §exp50（CURE_GENERALIZES 1%→70.5%）＋新 §exp53（CURE_DIES_WITH_SOURCE FAIL=测量能力缺口）＋新 §exp49（语义融合检索端零贡献 NEGATIVE）＋4.6/4.15/4.16/4.17/4.18（置信/RIF/上下文=门控与消融带，砍过程留结论）。
- **4.D The instrument（可验证性）**：现 4.1（setup 压缩）＋4.7（负结果汇总，更新至 53 实验）＋4.8（九件套账本，砍到一表）＋4.9（统计稳健性）＋§8（methods 并入）。
- 砍除规则（S-011）：过程叙述删除（round/三审往来/内部 D-/L- 编号）；负结果保留『前提-判决-机制解释』三句式；exp 编号只留发现级（19g/25/44/45/47/50/52/53 + 首次出现编号）；卫生性语句（预注册声明/门禁自我表扬）移 Methods 一处集中。

## W3. §2 related work 增补（十件必引 + audit 测量清单段）

插入段（EN 草稿，放 §2 末尾）：

> **The measurement agenda, executed.** Mahendrakar (2026) audits what the CLS analogy licenses in agent memory and lists five measurements that would settle the open questions: a generalisation test for textual consolidation; a repetition-schedule curve at matched compute; an interleaving ablation in the parametric branch; a shared-versus-isolated comparison; and a relinquishment test — delete the source episodes after consolidation and re-evaluate. None of the systems it audits has run these. The present paper is a mechanism instantiation of the audited commitment, and has pre-registered and executed three of the five plus the sixth (consistency-gated scheduling): the interleaving ablation (interleaved replay dominates sequential scheduling, Sec. 4.4); the repetition-schedule structure (online replay adaptation at 10⁵ items fails while replay-cure generalises — the sign structure the audit predicts, Secs. 4.x); and the relinquishment test, whose predicted failure we confirm and quantify: after deleting every consolidated source episode, 1.4% of cured retrieval survives (Sec. 4.x) — the capability gap the audit asks builders to measure. The audit also observes that absence claims from abstract-level literature search are its weakest part (its §14); our claim is not absence but presence, with artifacts.

十件清单（标题/venue/一句话站位，wave-8 全部入 §2 + refs bank）：①Mahendrakar 22699102（audit 文=测量议程）②Hippo 2610.24453（training-free 海马回放自进化；无巩固基线跌破 no-memory=exp44 镜像）③Adaptive Dual Memory 2610.33019（双库 JSD 路由 +4.67%=双通道概念撞车零测量）④Anthropic components of agent memory 2026-11-06（dreaming 离线巩固=工业采认 CLS 分工）⑤SoMem 2611.11929 ⑥ENGRAM 2611.09472 ⑦MemAsRec 2610.34673 ⑧MEMOIR 2611.07631（judge-gated merge 100 次审计合并 0 损失上界 2.95%）⑨AfterDeleteBench 2610.31366（删源泄漏审计：安全方向 first-mover 是它，utility-survival 方向 exp53 首测）⑩Distributed Engrammics Zenodo 20931468（六 prereg 假设 60 seeds）。refs.jsonl 加 10 条 T2/T3 条目 + {{token}} 接线（D+10 regen 前完成）。

## W4. §6/§7/§9/附录 动作

- §6 标题『Limitations and honesty statement』→『Limitations』；内容砍半（honesty 清单并入 Methods verifiability 节）。
- §7 超算协议 → 移附录（正文只留一行指针 + 4.A 引用 pilot 数字）。
- §9 四小节（wave-3 已压至 45 行）→ 再砍至 ~3K chars：9.1 保留、9.2 保留、9.3 五行表保留、9.4 三句。
- App B（结果文件索引）/App C（引用验证报告）→ 移出正文包，随 supplement 或仓库 README。
- 全文 D-xxx/L-xxx 内部引用清零（现 12+2 处）；『(round N)』类 round 标记清零（wave-2 已清标题，正文残留再扫一遍）。

## W5. 执行序（今晚→明晚）

1. 今晚：W1 主张句定稿（待主人裁定）→ §2 related work 增补段 + refs bank 10 条 → §6/§7/§9/AppB-C 动作（不动 §4）。
2. 明晨 exp45 JSON 落地判门后：§4 新小节叙事（exp44/45/50/53/49/47）按 W2 地图写稿。
3. 明午 exp52 1e6 落地后：4.A 收口段（10³→10⁶ 定律链全数字）。
4. D+10（11-21）：§4 物理重排执行 + 四链 regen + check_claims 327/327 + 页数检查。
5. 每完成一节跑一次 EN 引文闭合校验（verify_citations + 双向闭合 grep）。
