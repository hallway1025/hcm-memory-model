# exp35 设计稿：M0 置信度校准（I 区行动项 C3′，投稿前必做）

版本 v1.0 草案 · 2026-11-07 · Owner: lead · 状态：设计待审，实现等 full2 结束（机1 计时干净）

## 1. 动机（为什么这是投稿前必做）

- 审查红旗：exp13（results/exp13_confidence.json，n=120，base_rate=0.5）raw_evidence corr=+0.9745、AUC=1.000，库内 0.809 vs 库外 0.412——**过于完美**。两类可能：机制真好，或任务过易（库内外分布差距构造得太干净）。审稿人会按后者读。外部评审 GPT-6Astra ⑧ 与 SURVEY-WEAKPOINTS I 区独立指出同一问题。
- 论文声明风险：§4.6/§7 的置信度主张目前只有 AUC 单指标，无校准（ECE）、无 reliability diagram、无分布证据——按 I 区调查（S763 OU 扩散、S776 信心≠似然比、S768 极值 SDT），规范做法是两阶段读出+校准曲线。
- 空位论证：I 区总判定=「检索式记忆系统的端到端校准基准缺失」。本实验若做成，HCM 顺带成为该基准的早期实践（论文叙事点）。

## 2. 目标与可证伪预言（预登记）

- G-1（主）：在 David 真实语料 test 留出上，raw_evidence 的 ECE（10-bin）与 Brier 报告出来；**预言：ECE > 0.05**（即存在可观失配——若 <0.05 且 AUC 仍 ≈1.0，判任务过易，触发 G-3）。
- G-2：检索集内部一致性特征（top-k 码间平均余弦，S776 教训）加入后，ECE 相对单特征下降（两特征 logistic 拟合，dev/test 分离）。预言：下降但 AUC 不降。
- G-3（难探针升级门）：若 G-1 触发「任务过易」，加难探针三件套（改写查询 frozen_paraphrases 100 条复用 / 缺失 30% / 交织干扰）重测；难探针下 AUC 应 <0.95。**不加难不得把置信度主张写进论文**。
- G-4：三分支决策（RECALL/SEPARATE/ABSTAIN）阈值从固定 0.5 改为校准曲线分位数后，覆盖率-风险曲线（exp8 范式）不劣化。

## 3. 设计

### 3.1 数据与探针
- 库：David train（exp23 npz S256 嵌入路径，M=E.T，与 exp27/30b 同加载器——行号语义必须用 exp29 同款 csv.reader utf-8-sig，防 252 条多行消息错位）。
- 探针：test 留出抽样 库内近邻（对 train 高 cos）与库外（低 cos）各 ≥200；另备改写 100 条（frozen_paraphrases.json 复用，标注 post-hoc）。
### 3.2 两阶段读出（实现为 bench/exp35_calibration.py）
- 阶段 1 证据分布：对每条查询取 (a) top-k 候选 raw_evidence 分布（k=5/10 两档）；(b) 查询扰动（字符级 5 次扰动检索）的 evidence 方差。
- 阶段 2 置信映射：特征 [evidence_top1, evidence_mean, consistency(top-k 码间平均余弦), perturb_var] → dev 集 logistic 拟合 → test 报 ECE/Brier/AUC + reliability diagram（matplotlib，进论文 fig）。
- 对照臂：raw_evidence 单特征（现状基线）；消融：去 consistency 臂。
### 3.3 判据与产出
- 主产出：results/exp35_calibration.json（每特征组 ECE/Brier/AUC/n）+ paper/figures/fig-calibration.png。
- check_claims 增块：ECE 数值、consistency 增益方向、G-3 触发与否。
- 论文落点：§4.6 替换「AUC=1.000」表述为校准叙事；§7 架构缺口 8（两阶段读出）状态更新。

## 4. 顺序与资源

exp30c（事件层对照，等 full2）→ exp35（本实验）→ exp31 CLS 预算 → exp34 干扰消融。全程机1、CPU 单核、预计 <10 min 量级（N=库 8.8k×d_h=256 S256 路径，exp27 已验证该规模检索 <1s/条）。

## 5. 与既有结论的关系

不推翻 exp13（其 AUC=1.000 在合成库/库外二分下真实）；exp35 把同一机制放到真实语料+难探针下重新定量，产出可写进论文的校准曲线。若 G-1 触发且 G-3 难探针下 AUC 仍 ≈1.0，则怀疑泄漏路径（须查 npz 探针是否在 train 内）——那是 bug 不是结论。
