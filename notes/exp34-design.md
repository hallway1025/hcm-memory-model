# exp34 设计：M8 遗忘算子的干扰情境消融（RIF 检索练习范式）

版本 v0.1 · 2026-11-07 · 起草 lead · 状态：待实现（排在 exp35 收口后，D-044 优先序第 ②位）

## 1. 动机
- G4 总账（bench/g4_ledger.py）判 M8_FOR = INERT：RIF 练习后 sup 差分仅 −0.0035 且 top-1 不变。该判定的适用域=低干扰、40 条小库、5 轮练习——调查 B 区（notes/lit/SURVEY-WEAKPOINTS.md）指出根因是缺干扰/增量实验设计，不是机制无效。
- 预登记期望（D-044）：在检索练习范式下 INERT → LOAD-BEARING。对应 audit/CLAIM-STATUS.md C15。
- 生物学映射：Anderson 检索诱发遗忘（RIF）= 记忆选择时对竞争者的主动抑制；「功能遗忘」S128 / MESU S187 视角：遗忘是适应性资源分配而非缺陷。

## 2. 机制事实（实现核对 2026-11-07）
- `src/hcm/episodic.py:166-177` suppress(a, rho_rif=0.15, topk=5)：命中后对 a 的 top-5 中除 winner 外的竞争者 `sup[i] *= (1-0.15)`；winner 豁免且 use+1。
- `episodic.py:133` 检索分数 = `M.T @ h * sup[:n]`（乘法抑制点）；`disinhibit(idx)` 可逆。
- `src/hcm/model.py:57-58` cfg.rif_on=True、rho_rif=0.15；`model.py:254-256` 每次成功检索自动触发（query(do_rif=None) 继承 cfg）。
- 注意：exp35 及此前所有真实语料实验的 query 都带默认 RIF——这是部署语义的一部分（模型出厂即开），非污染；exp34 把它变成受控变量。

## 3. 假设（预登记）
- H1（RIF 特异性）：对子集 P 做检索练习后，与 P 码结构共享的 Rp− 条目召回率下降，匹配对照 Nrp 不降；仅 RIF on 臂出现。
- H2（重叠剂量）：Rp− 降幅随码重叠（合成库 share 0.3/0.6/0.9）单调增大。
- H3（剂量）：竞争者 sup 渐近 = (1−rho)^k_practice（稳定共激活者），降幅随练习次数 k 次线性加深。
- H4（功能收益，可选臂）：练习 P 后注入与 P 重叠的新干扰项，RIF-on 库中 P 自身召回保持率高于 RIF-off 库（遗忘=保护已检索轨迹）。

## 4. 设计
- 语料：主 cell = David train（S256，exp23 路径，同 exp35 加载器）；合成 cell = overlapping_set（share 0.3/0.6/0.9，注意 exp12 教训：share 是混合权重非输入余弦，需报告实测两两余弦）。
- 阶段：write N=2,000（L0）→ practice：对 P（100 行）各自用干净查询检索 k∈{1,3,10} 次（RIF on 臂自然累积 sup；RIF off 臂 query(do_rif=False)）→ test：L-035 判据（action∈{recall,separate} ∧ idx==obs_of[target]）。
- Rp−/Nrp 选取：练习后读 `m.E.M[:, obs_of[P]]` 与全库码的余弦，取 top 四分位为 Rp−、bottom 四分位且 recency/use 匹配为 Nrp（各 ≥50 条）。
- 臂：主 cell（David）2 RIF 臂 × k=3，12 种子；合成 cell 2×3 overlap × k=3，12 种子 → 8 cell × 12 = 96 runs（每 run ≈1 min，总 ≈2h，机1 L0）。
- 机制直读：除行为差分外直接记录 sup 向量差（Rp− 均值 vs Nrp 均值），行为与机制双证据。
- 特异性核查：P 自身召回率在两臂均应保持（抑制不得命中 winner 豁免语义）；disinhibit 后 Rp− 召回应恢复（可逆性）。

## 5. 门禁
- G-A（LOAD-BEARING）：RIF-on 下 hit(Rp−) 相对 hit(Nrp) 降幅 ≥10% 且 12 种子 CI 不重叠；RIF-off 下同差分 ≈0（CI 含 0）。两条同时成立 → 改写 G4 台账 M8 行为 LOAD-BEARING（标注情境限定）。
- G-B（可逆性）：disinhibit 恢复 ≥80% 降幅。
- G-C（负结果处置）：若干扰下仍无差分 → M8 维持 INERT 且范围扩大（低干扰+高干扰双重否定），规格改写为 MESU 式三档功能开关（调查 B 建议），论文 §4.8/§4.12 同步改写——诚实负结果，不许隐藏。

## 6. 与其他实验的衔接
- exp35（校准）结论决定 exp34 是否加校准特征读出；exp31（CLS 预算）共用 practice 阶段基建；CLAIM-STATUS C15 的处置由本实验终结。
