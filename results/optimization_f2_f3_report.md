# exp29 · F2 公平真实语料对照 + F3 语义收益独立验收 报告

执行代理：opt-verify（fresh 验证代理，Lead 指派）。F1 回归+审查由 Lead 完成（commit 70b08bd，results/optimization_f1_review.md）。
**状态：F2 诊断口径 + F3 完成（2026-11-08）**。full 四臂链按 Lead 裁决（2026-11-07）移交 Lead 在机2 执行，本代理不碰 full（§3.4 由 Lead 续填）；B1 综合判定待 full 后由 Lead 汇总。

## 0. 任务与范围

- F2：按手册 §7.3 六条，两阶段（David 诊断 → 四会话全训练 + test 留出），四臂对照（R512 / R256 / 旧 S256npz 泄漏参考 / 新 CharacterSemanticEncoder256）。
- F3：按手册 §7.4，冻结近义对与改写查询资产 → Lead 审查放行（results/optimization_f3_freeze_review.md，"proceed"）→ 评测 + 统计。
- Lead 追加（外部评审）：两个 post-hoc 廉价基线列（字符 n-gram 余弦、BM25-lite），预测登记见 §4。
- 写作用域：bench/exp29_text_frontend.py、results/exp29_text_frontend.json、bench/frozen_synpairs.json、bench/frozen_paraphrases.json、results/optimization_f2_f3_report.md、memory/40-EXP-LEDGER.md（只追加）。未修改 src/hcm/、未覆盖 exp22–28 任何结果、未动 paper/、未 ssh 二号机（exp26 占用中）。

## 1. 环境与执行方式

- 机器1（本机）：Windows-11-10.0.26200，Intel Core Ultra（Family 6 Model 186），16,105 MB 总内存；实验启动时空闲 2,820 MB（mem_load 82%）。
- Python 3.14.0 / numpy 2.4.4；BLAS = scipy-openblas 0.3.31.188.0（Haswell 内核，MAX_THREADS=24），MSVC 19.44 构建。np.__config__ 全文存于 results/exp29_text_frontend.json → meta.np_config（N-3：确定性仅本机成立）。
- exp23 npz SHA256：2f516de6cb3bb542d0ab9e717623e1bdda7d86653836d597db98e7633b280c84。
- Shell 为 PowerShell 5.1（无 &&）；长实验全部走后台 job 并记录 job id。
- exp29 脚本 SHA256（selftest 时点）：e729d6ba80c4326e71f7d5ab7ec350666dd343923f166ac6fab401c841bd85c5（后续 F3 代码追加会变，以 git 提交链为准）。
- 共同超参（meta.hyperparams）：beta=100.0、L=3、lam=1.0、use_resid=False、do_rif=False、a=0.02、capacity=N+10、d_c=d_x、d_h=d_x×8、k=round(0.02·d_h)、model seed=0。判据：action∈{recall,separate} 且 idx≥0 且 E.t[idx]==目标观察序号（L-035）；拒写计 miss（固定分母）；内容等价口径另列（payload 余弦≥0.999）。mock 自检含弃权/idx<0/未写目标/dup 拒写不误计（selftest 26 项）。
- 种子（meta.seeds）：proj=7、deg_base=41000、dprime=1234、pairs=5678、nnz=5679、intrain=2029、randtest=3049、boot=7；探针损坏位置 = fresh default_rng(41000+j) 冻结删除位表；d' 他码样本 = rng(1234) 每探针 50 随机他项，跨臂共享。

## 2. 数据（SHA256 与规模）

| 文件 | SHA256 |
|---|---|
| train__David.csv | 4940a20bbd342d91b22c606ff65cdd39b3b7ac8011c35a49e3ec6d0cf6ff731e |
| train___.csv | e33bf6c9124f212f288f22e5ca05915a730ad3c7c1ce151b73f1ed153e9fb6ab |
| train__水泊梁山大家庭🐊🏠❤️.csv | a2b3f3cfc591ac8160fe4d9475dc7360590a3ccb781cfd37c6b8ccb3d2b29ffe |
| train__滚木是我爹.csv | 0b1696d561bd72f8e23264d619eb03227017c168852be2dcb0ffa73c189135f9 |
| test__David.csv | b6d5b547967529afced5cde0e62897e94cf0e9c3ef72e7b990fefdac9ea90204 |
| test___.csv | 16635fdeeaf2148e5ef81c774999a4d6e499471f43888f257a6c753b0512ce0c |
| test__水泊梁山大家庭🐊🏠❤️.csv | e0061698b70db771f32070aad7d7f44100b8319125eca8dd76badca81aa83805 |
| test__滚木是我爹.csv | 402f537034069659c52eb0e9c80e20c115a1669e4536dae3ad3a9f0b42977cf2 |

- 四会话 train = 103,221 条消息；test = 25,806 条；合计 129,027。全 train 词表（cleaned 字符 + \ufffd）= 4,729。David-2000 诊断词表 = 1,032。fit 只用 train；test 仅作留出评测与 dup200/rand200 探针来源。
- N-1（PPMI 共现含自身对）：CharacterSemanticEncoder 的共现定义含自身对（对角=unigram），属定义项非 bug；本实验各臂共用同一实现，在此注明。
- N-2：新 256 臂只使用 src/hcm/text_encoder.py 的 CharacterSemanticEncoder，且只在阶段 train split 上 fit；exp23 npz 与之是两套独立 PPMI-SVD，旧 S256npz 臂仅作泄漏风险参考臂，不计入 B1 证据。
- 报告正文不含任何聊天语料原文（探针/查询一律以 ID、长度、字符重叠率等统计量描述）。

## 3. F2 文本前端公平对照

### 3.1 selftest（判据与统计单测）

- 命令：python bench/exp29_text_frontend.py --stage selftest → 退出码 0，26/26 PASS。
- 首跑 24/26，暴露两处失败并修复：① t_sf 对负 t 未用对称性（真 bug，修复为 t<0 返回 1−half）；② t_sf(df=30,t=2) 的已知值硬编码错误（0.0273561 → 数值积分独立复核真值 0.0273125225）。_betainc 实现本身正确。
- 覆盖：judge 七项 mock（弃权/idx<0/错槽/未写目标→miss；exact/separate→hit；dup 场景 exact-miss 与 content-hit 分离）、t 分布 6 项、paired_t 已知值、bootstrap 确定性、tiny HCM 集成（D=32, 21 条含 1 dup）。

### 3.2 smoke（David 前 1000，后台 job bash-f0bf0e0e9fdf46e0，80.7s，exit 0）

| 臂 | written | deg0.3 命中 | d'(0.3) |
|---|---|---|---|
| NEW256 | 781/1000 | 199/200 | 10.136 |
| R256 | 758/1000 | 200/200 | 11.060 |
| S256npz（泄漏参考） | 862/1000 | 200/200 | 8.487 |
| R512 | 758/1000 | 200/200 | 12.086 |

gate 共同已写子集 n=199（子集口径命中一致，详见 JSON）。

### 3.3 diag（David 前 2000，后台 job bash-390e40b556604dd2，363.0s，exit 0）

| 臂 | written | 拒写率 | deg0.0 | deg0.3 | deg0.5 | d'(0.3) | 码间余弦 | 非零码数 | 输入随机对余弦 | fit/写/查询耗时(s) |
|---|---|---|---|---|---|---|---|---|---|---|
| NEW256 | 1490/2000 | 25.5% | 200/200 | 199/200 | 194/200 | 10.112 | 0.0424 | 41.0 | 0.7219 | 2.2 / 8.6 / 55.2 |
| R256 | 1455/2000 | 27.3% | 200/200 | 200/200 | 198/200 | 11.256 | 0.0303 | 41.0 | 0.2545 | 0 / 9.3 / 53.3 |
| S256npz（泄漏参考） | 1693/2000 | 15.4% | 200/200 | 199/200 | 196/200 | 8.827 | 0.0329 | 35.2 | 0.6364 | 0 / 7.8 / 65.1 |
| R512 | 1455/2000 | 27.3% | 199/200 | 199/200 | 198/200 | 12.482 | 0.0293 | 82.0 | 0.1313 | 0 / 27.8 / 127.5 |

- 拒写分解（dup/gate）：NEW256 503/7、R256 538/7、S256npz 300/7、R512 538/7——四臂 gate 拒写同为 7 条（探针集固定），差异全部来自 dup 拒写；弃权率与 action 分布见 JSON。
- B1 诊断口径（200 探针、同口径 deg0.3≥199 且 d'≥4）：四臂全部达标（S256npz 为泄漏参考臂，不作为证据臂）。
- 说明：NEW256 输入随机对余弦 0.72 属 PPMI-SVD 语义嵌入的预期稠密相关结构（对照 exp22 口径 d' 定义仍成立，因 d' 用的是查询向量 vs 自/他写入输入向量的方差归一）。

### 3.4 full 阶段（四会话 103,221 训练 + test 留出）——待填

- 命令：python bench/exp29_text_frontend.py --stage full --job-id full1（后台 job bash-85fa928365104c1f，单臂顺序 NEW256→R256→S256npz→R512，每臂完成即落盘）。
- 探针组（写入前冻结，JSON full.probes）：intrain200（rng(2029)，deg 0/0.3/0.5）、dup200（test 与 train 完全重复前 200，精确事件目标=train 首现，另报内容等价口径，deg 0/0.3）、rand200（test 随机 200，无真值→action 分布+字符包含率，deg 0/0.3）。
- 结果：**（由 Lead 在机2 完成后填写：四臂 written/拒写率、intrain/dup/rand 三组三口径、d'、码指标、耗时、峰值内存、B1 留出判定）**
- 归属变更（2026-11-07 Lead 裁决）：full1/full2 两次尝试均死于 subagent 宿主级回收后，full 四臂链改由 **Lead 在机2**（you@100.112.23.115）执行；本代理负责提供 F3 JSON 基座（results/exp29_text_frontend.json 的 f3 key，已完成；资产 rev2 已 scp 两机一致）。

## 4. 预测登记（post-hoc 廉价基线——先登记，后运行）

Lead 追加要求（评审后）：在不改 bench/frozen_*.json 任何内容的前提下，对同一批冻结查询/目标加评两个"便宜基线"，报告中标注为**评审后追加（post-hoc）**，与预注册臂（NEW256/R256）分开列：

1. 字符 n-gram 余弦：query_clean 与候选 target 的 cleaned 文本取字符 2-gram / 3-gram 集合，Jaccard-余弦 = |A∩B|/sqrt(|A||B|)，无训练。
2. BM25-lite：字符 bigram 词袋 + BM25 权重（k1=1.5，b=0.75；IDF=ln(1+(N−df+0.5)/(df+0.5))；|d|=bigram token 数；语料=David-only 8,844 条，与预注册检索语料一致）。

**登记的预测（在运行前写下）**：零重叠层（char_overlap_ratio=0.00 的 8 条改写查询 + 14 个零共享字符近义对）上：
- 字符 n-gram/BM25 对目标的相似度恒等于 0（零共享字符 ⇒ 零共享 bigram/trigram ⇒ 无表面信号）——这是构造性事实而非经验预测；
- 因此基线检索 hit@1（严格最大且 >0）在零重叠层应≈0。**若任何基线在零重叠层命中，说明冻结资产隔离性有问题，立即停止并报告 Lead。**

**核验结果（f3run1，2026-11-08）**：预测成立——零重叠层 ngram2/ngram3/BM25 对 8 条 ov=0 改写查询的 hit@1 均为 0（target_score 全 0），24 组零共享近义对的 ngram2/ngram3/BM25 对目标相似度均值均为 0（构造性=0 得证，数据 baselines_posthoc.pair_similarity.synonym_zero_shared 全 0）。全层 hit@1（n=100）：ngram2 38、ngram3 32、BM25 40。无隔离性问题，未触发停止条件。

## 5. F3 语义收益独立验收——方法已冻结，结果待填

- 冻结资产（commit 092c0ba，Lead 审查 PASS；**rev2 修订 commit 60230e2**，Lead 批准 D-047：unrelated 对 pid 0-23→100-123 消除与 synonym 组的 pid 冲突，修订后 sha256=d72875968ff983a4db689e7062a83d920987b735792183e010a5d045ffc9aff9，scp 到机2 后 roundtrip 回读哈希两机一致；bench/frozen_paraphrases.json 未动，sha256=0a2b63c8fa651232466ccdfeafa845e0cc72dd026563bc94c1bb298788aaa0b3）：bench/frozen_synpairs.json（24 组去重近义对 = 14 零共享 + 10 共享字符；另附 24 组人工无关对作噪声基线；OOV 覆盖 0/24，全 train 词表全覆盖）；bench/frozen_paraphrases.json（100 条人工确认同义改写查询，查询文本经程序化验证不在 4 个 train 会话 103,221 条中出现；目标=train__David.csv 行号首现观察序号；overlap mean=0.563 / min=0.000（8 条）/ max=0.889）。
- 预注册评测（资产内嵌 eval_plan_frozen + Lead 放行附加要求）：臂=NEW256/R256；HCM seed∈{0,1,2}；同义对→输入余弦 + 每对独立小 HCM(capacity=10) 码余弦 + 无关对基线；改写→David-only store（8844 条，capacity=N+10；David 为拼接顺序第一会话，目标首现观察序号与全量拼接一致——此设计在见任何 F3 结果前冻结）检索，判据同 F2（L-035），固定分母 100，exact 与 content 两口径；分层 zero/low/high（zero=overlap==0）；14 零共享对 NEW vs R256 逐对胜负表（n=14 如实标注）；统计=逐查询配对 t（exact 二值差 + target_sim 连续差）+ bootstrap CI(10000, seed7) + 3 seeds 聚合；S256npz 不作为证据臂。
- 基线列：§4 的两个 post-hoc 廉价基线，同表并列，标注 post-hoc。
- 结果（f3run1，job bash-8308c54a91224d8c，2026-11-08 00:59:31–01:37:53 +08:00，stage elapsed 2305.3s，STAGE_OK，机1 独占无计时污染源；store=David-only 8844 条 cap 8854，encoder 全 train fit，hc_seeds=[0,1,2]，first_obs≠row=11 与设计一致）：
  - **改写检索（100 查询 ×3 seeds；每 seed 内 exact 与 content 两口径计数完全一致，如实并列）**：
    - NEW256：exact/content 40/42/39（均值 40.33/100）；abstain 0/0/0；written 6358；mean target_sim 0.6878
    - R256：exact/content 39/36/37（均值 37.33/100）；abstain 1/1/0；written 6272；mean target_sim 0.5071
    - 配对差（NEW−R256，逐查询 paired t）：exact seed0 +0.01（p=0.783）/ seed1 +0.06（p=0.134）/ seed2 +0.02（p=0.620）；**pooled +0.03（n=300，t=1.343，p=0.180，CI95 [−0.014, +0.074]，bootstrap CI95 [−0.013, +0.073]）→ exact 命中率差异不显著**
    - target_sim 配对差：pooled +0.1806（t=26.04，p≈7.3e-79，CI95 [0.167, 0.194]）→ **语义近邻质量差异极显著**
    - low_overlap 子集（overlap<0.33，15 查询×3 seeds=45 行）：exact 差 =0（两臂全 miss，sd=0）；target_sim 差 +0.2854（t=15.81，p≈0，CI95 [0.249, 0.322]）→ 无表面重叠时 NEW256 的最近邻仍显著更近
    - 分层（pooled 300 行；F3_STRATA：zero<1e-9 / low<0.5 / high≥0.5）：zero 24 行两臂 0 命中；low 63 行 NEW 3 vs R256 0；high 213 行 NEW 118 vs R256 112
  - **近义对（24 synonym + 24 unrelated；synpair 输入余弦 seed 无关，码余弦 3 seeds）**：
    - 输入余弦：synonym NEW 0.3714 vs R256 0.1997（配对差 +0.172，t=7.09，p≈5e-8）；unrelated NEW 0.2319 vs R256 0.0364（配对差 +0.196，t=9.07）
    - 码余弦配对差（NEW−R256）逐 seed：synonym +0.063/+0.073/+0.063（p=0.006/0.006/0.016）；unrelated +0.041/+0.034/+0.038（p≈0.006/0.006/0.006）→ NEW256 对两类对的码相似度均显著更高，synonym 幅度更大（syn>unrel 方向两臂一致，无 Lead 警告的层间方向异常）
    - 码余弦绝对值（synonym by seed）：NEW 0.141/0.166/0.167 vs R256 0.077/0.093/0.105
  - **14 零共享对 h2h（n=14 小样本，如实标注）**：input NEW 13 : R256 1；code NEW 13 : R256 1
  - **post-hoc 廉价基线（同 100 查询，David-only 8844）**：hit@1 ngram2 38 / ngram3 32 / BM25 40（hit@5：50/40/54）；**零重叠层三基线全部 0 命中（§4 预测成立）**
  - **判读（如实）**：在 100 改写查询、David-only store、3 seeds 的设置下，NEW256 相对 R256 的 **exact 命中提升小且不显著（pooled +3pp，p=0.180）**；但其**检索语义质量（target_sim）显著更优（+18pp，p≈1e-78）**，且 low_overlap 子集最近邻显著更近（+0.285）、零共享近义对码区分显著更高。语义收益主要落在连续相似度与零/低重叠泛化上，而非 exact 命中数。
- B1 综合判定：本代理只报分项（F2 诊断口径 4/4 臂过 + F3 分项如上）；**综合判定留待 full 留出完成后由 Lead 汇总**（full 已移交 Lead 机2）。

## 6. 成功与失败 / 未完成事项 / 资源

- 成功：exp29 脚本（1470+ 行）+ selftest 26/26（修复 t_sf 真 bug）；smoke PASS；diag PASS（四臂 deg0.3≥199，d'≥8.8）；F3 冻结资产构建+程序化隔离验证+Lead 审查放行；F3 评测管线小样 dry-run（抓出 3 处实现错误并在运行正式评测前修复：run_f3 结果未落盘、payload 应为 m.E.P、_paired 键名与 paired_t 不符）。
- 失败/修正记录：dry-run 首跑 ASSET SHA MISMATCH 为 dry-run 截断补丁自身造成（截断后未重算自洽 sha），非资产问题；driver 路径 raw-string 转义错误一次。
- 未完成：full 四臂（**已移交 Lead 在机2 执行**，本代理不再触碰）；B1 综合判定（等 full，Lead 汇总）。已完成：F3 正式评测 f3run1（STAGE_OK）；台账 exp29 行已追加；本报告 F3 段即终版（§3.4 full 段由 Lead 续填）。
- 资源：峰值内存按臂记录于 JSON（peak_ws_bytes，psapi，单调语义）；R512 全量臂预计 ~3.6 GB 峰值，启动时空闲 2.86 GB，可能换页（已把 R512 排最后）。

## 附：提交链（本代理）

0c53839 脚本+selftest → 645a3a8 smoke → 59917b1 diag → 092c0ba F3 冻结资产 → 2504db6 报告骨架+预测登记 → 19deef7 F3 代码 → 005f21d 事故声明 → 60230e2 资产修订 rev2（Lead-approved D-047）→ （本提交）F3 结果填报告 + 台账 exp29 行。
基线：70b08bd（F1）。Lead 的未提交改动（memory/10-STATE.md、20-DECISIONS.md、tools/collect.py、tools/survey_screen.py、refs/survey/、results/survey/）未触碰、未提交。

## 6.1 运行事故与计时污染声明（先于 full 结果写入）

**事故**：full1（job bash-85fa928365104c1f，18:35 启动）于 19:19–19:22 之间死亡。Lead 调查结论：subagent 宿主级回收（Lead 会话 job 注册表完好、但双方被管子进程连带终止；机2 exp26 ssh 长持任务同窗死亡），非全应用重启。死亡时 NEW256 已写 ~80–85k/103221，无臂完成、无部分落盘（臂级落盘设计），无科学性影响。

**重启**：full2（job bash-50175d0431024594，19:24 启动）复用 18:35 冻结的 probes（full.probes.groups：intrain200/dup200/rand200 各 200，含损坏位置表与 SHA 锚点），确定性重算。

**计时污染源（三个，均只影响耗时字段、不影响命中率等确定性指标）**：
1. NEW256 末段与 F3 dry-run 重叠：full1 的 80k 检查点（19:05）早于 dry-run 启动（19:18:44），受污染窗口为最后 ~23k 写入 + eval 的一部分；dry-run 19:19:22 被 job_kill。上界估计：受污染写入 ~23k 条 × 单条写入耗时上限（~60ms，n=100k 时单条含两次全库 matvec 的保守上限）≈ 23 分钟墙钟上限，实际重叠 ~1 分钟级别；且 NEW256 与其他臂的对比只用到"同臂内相对耗时"，受污染段占比 <25%。
2. Lead 侧 collect.py（PID 38808，19:20:19 启动）与 full2 并行：同检查点对比 full1→full2：20k 141s→165s、40k 413s→471s，拖慢 ~13–17%。lit-surveyor 已被要求错峰 CPU 重阶段（API 网络型采集不受限）。
3. full1→full2 本身：两次运行的 fit/write/query 计时字段不可跨 job 直接比较，报告以 full2 的计时为准，full1 计时仅作参考。

**预测登记（复述，原文见 §4，commit 2504db6 先于任何 F3 运行提交）**：零重叠层（8 条 ov=0 查询 + 14 零共享近义对）的 ngram2/ngram3/BM25 基线对目标相似度构造性为 0，hit@1 应≈0；若任一基线在零重叠层命中 => 语料隔离性问题 => 立即停止并报告 Lead，不自行解释后继续。

**f3run1 计时说明（2026-11-08 00:59:31–01:37:53 +08:00，job bash-8308c54a91224d8c）**：F3 正式评测在机1 独占运行（full2 已死、collect.py 已停、dry-run 已结束、本会话无其他 CPU 重活），无已知重叠污染源。臂级 elapsed：NEW256 830.6s（含 encoder fit 62.1s）/ R256 1472.7s，stage 总 2305.3s，STAGE_OK。dry-run（250s）先于 f3run1 结束，不构成重叠。
