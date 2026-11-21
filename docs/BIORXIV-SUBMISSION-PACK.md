# bioRxiv 投稿材料包（D+14 = 2026-11-22；D-069 定稿：bioRxiv 先发，arXiv 推迟）

状态：材料清单 v2（2026-11-18，交叉审核修正后更新；判门全 PASS，PDF/HTML 已重渲染）。

## 1. 稿件本体
- [ ] 主文件 PDF：MANUSCRIPT-ARXIV.pdf（2026-11-18 V1 终版，68 页，SHA 前缀 425EBFF8017268E6）
  - 源=paper/submission/arxiv/MANUSCRIPT-ARXIV.md → export（staging <user-home>/arxiv_build，L-064 重同步）
  - 摘要已含 P0 三修（20-seed 数字 / 129/200 条件化 / biologically inspired）commit 2b14a03
- [ ] 图件：figures/fig1..fig5（regen 时重出，caption 与正文一致）
- [ ] 补充材料：Appendix A–J（含 exp52 10⁶ 判门 + exp47b 跨架构对照）
- [x] 作者与单位：**Haowei Wang（单人署名）；表单 Institution 栏=Undergraduate Student, Shandong University of Finance and Economics, Jinan, China**（D-073；与 MANUSCRIPT-ARXIV.md 封面逐字一致）
- [x] ORCID：**0009-0004-2241-5511**（MOD 11-2 校验位核 ✓；bioRxiv 账号已注册 ✓，投稿时绑定）

## 2. 表单字段（bioRxiv 提交系统）
| 字段 | 填写值 |
|---|---|
| Category | Neuroscience |
| Title | A Log-N Collapse Regime in Attractor Memory: A Reproducible, Dual-Readout Hippocampal–Cortical Complementary Memory Model (HCM) |
| Author(s) | Haowei Wang（单人署名；Institution=Undergraduate Student, Shandong University of Finance and Economics, Jinan, China） |
| Correspondence email | 202529640232@stu.sdufe.edu.cn（ORCID 0009-0004-2241-5511 已绑定于稿件封面） |
| Abstract | **源=桌面 arXiv-v1-上传包/ABSTRACT-BIORXIV-FORM.txt（298 词，含 10⁶ 门通过 / 99.6% 迁移 / 六架构对照三新结果；2026-11-18 交叉审核后定稿）**。不用 PLOS 旧版（仍含 3 种子 16.0/15.3）；不用 ENF/PDF 全摘要（≈514 词超表单限） |
| License | CC BY 4.0（默认款；改动需主人拍板） |
| Preprint DOI option | 不用 DOI 印章（免费档） |
| Previously posted | 如实填：否（arXiv 未挂；D-069 后无其它预印本） |
| Funding | None |
| Conflicts of interest | None |
| Data availability | 仓库匿名化说明（微信语料 SHA-256 脱敏 + 合成语料 + 种子；公开部分=代码+合成语料） |
| AI disclosure | v6 披露段（稿件内已有；表单 Additional information 再粘一次） |
| Ethics | 王骏启授权句（六会话语料授权；S8/Ethics 节文本复用） |
| Referee suggestions | 可空（R1–R4 为匿名审查，不填） |

## 3. 提交前检查单
- [ ] check_claims 398/398 PASS（每次 regen 后必跑）
- [ ] verify_citations PASS（bank=447，T1+T2=79.6%=(258+98)/447，分母=全库含 curated-reject；cited-reject=0、tier_mismatch=0、failed=0、retracted=0；commit 4a838af 语义修订：stored tier 优先+curated/cited reject 分离）
- [ ] paper-cited-ids 闭合 69/69（tools/update_cited_ids.py：论文编号引用→refs.jsonl ref_id，longest-title 匹配，unmatched=0）
- [ ] PMID 交叉面 263/263 PASS（tools/verify_pmid_crossface.py）
- [ ] SHA256SUMS 重建（make_manifest.py）
- [ ] PDF 页数：68 页（bioRxiv 无硬限；arXiv 若后续提交需关注）
- [ ] 摘要词数 ≤ 300（bioRxiv 表单限；当前 298 词 ✓）
- [ ] 主人终读通过（D+10–D+11 窗口）

## 4. 已定决策回链
- D-069：bioRxiv 先发、arXiv 推迟、背书被动
- 工单：回音-工单-2周增量-20261111.md（发布顺序节）
- 摘要 P0：commit 2b14a03（2026-11-12）
- D-072（2026-11-13）：表单摘要源=PLOS 300 词版；Echo 回执①②（REJECT 口径+分母）处置=commit 4a838af
- D-073（2026-11-13）：作者三要素拍板（经回音转达主人决策面板）=Haowei Wang 单人 / 202529640232@stu.sdufe.edu.cn / ORCID 0009-0004-2241-5511 / Undergraduate Student, Shandong University of Finance and Economics, Jinan, China——A3 关闭，与 ARX 封面一致
