# §4.18 草稿骨架（写于 exp36c 数据之前——占位符映射到 results/exp36c_ctx_energy.json 字段）

## 标题（入 PAPER.md）
### 4.18 多时间尺度情境作为检索线索：两次否定与一次机制补全（第四轮，exp36/exp36b/exp36c）

## 4.18.1 动机与设计
- B2 主张（40-OPTIMIZATION-ROADMAP B2）：真实情境记忆需要**多时间尺度**时间上下文 τ∈{1,10,100,1000}（快 τ 绑事件边界、慢 τ 绑"时代"），且情境必须能**主动参与检索**，不是只存不用。
- 探针组（预登记于 bench/exp36_b2_multitau.py docstring，复用于 36b/36c）：
  - P1 era-anchor：内容 x_j（目标 i 的跨时代内容近邻，cos≥0.5，|pos_i−pos_j|>2000）+ 时代 i 的 ctx → 命中 slot_i 才算成功（情境把内容竞争者拉回目标时代）。
  - P2 shuffle（对照）：x_j + 洗牌 ctx → 应低。
  - P3 ceiling：退化 30% 的 x_i + 自身 ctx → 内容能力上限。
  - P4 era-swap（防平凡解）：退化 x_i + 近邻 target 的 ctx → 仍须命中 slot_i（内容不被情境劫持）。
- 门：G-T=ΔP1(multi−single)≥0.15 且 ≥2/3 seed bootstrap 95%CI 排 0；G-G=multi P3≥single−0.02；G-C=P3≥0.85；P2≤0.20。κ*_arm=P4_dev≥0.85∧P2_dev≤0.20 的最大 κ（era-swap 强制内容仍主导，防 κ→∞ 纯情境平凡解）。
- 数据：David 8,844 条，S256 嵌入（exp23 npz），200 targets（pos≥3000、间隔≥25），dev=1/3 targets（seed0）、test=其余 ×3 seeds。单臂 γ=0.95,d_c=256 vs 多臂 τ={1,10,100,1000},d_c=1024；β=100、L=3、use_resid=False、θ=0（D-038）、judge obs_of（L-035）。

## 4.18.2 第一次否定（exp36，默认 κ_ctx=0.6）与第二次否定（exp36b，κ 全网格）
- exp36：双臂 P1=0/135×3 seeds（job bash-ad707e7749d045e5，results/exp36_full.log）→ 归因=κ 标定问题假设（D-051），进 exp36b。
- exp36b（job bash-7c704af053704e6d，results/exp36b_kappa_calib.json）：dev 全 κ∈{1,2,4,8,16,32} P1=0、P3=P4=1.0 → κ* 退化为 32；test ΔP1=0 全种子，**G-T FAIL**；G-G/G-C/P2 PASS（P3=P4=0.977–0.989）。
- **机制归因（读码）**：query() 中 ev 的情境项只进 softmax(ev) 的**初始分布** a（model.py L221），但 L>0 分支用 E.retrieve 的**纯内容** Hopfield 吸引子结果 a_hop 覆盖 a（L223-226）→ 预偏置对最终读出无效。这不是 κ 太小，是**通路错误**。

## 4.18.3 机制补全（exp36c）：情境作为吸引子能量偏置
- 机制修改（bench/exp36c_ctx_energy.py，预登记 docstring）：episodic.retrieve() 加 ebias=None 参数，softmax_beta(Mn.T@z + (0 if ebias is None else ebias), beta)（初始 + 每次 L 迭代）；model.py ctx_in_energy=True 时 ev 不再加 ctx 项（通路干净分离），ebias = κ·max(z(E.Ctxᵀ·c),0)。默认 False 逐位不变（smoke 锚点验证）。
- 微测试（bench/_micro_ebias.py）：κ=16 时错误时代情境 → 检索证据分裂、元控制弃权；正确时代情境 → 恢复命中——旋钮确有作用且内容主导不被默认破坏。
- dev 校准：【占位 exp36c.dev curve：single/multi 各 κ 的 P1/P2/P3/P4】；κ*：【占位 calib.kappa_star_per_arm】。
- test 判定：【占位 gates：G_T deltas/cis/verdict、G_G、G_C、P2】。
- 生物学映射：能量偏置=情境对 CA3 吸引子能量景观的先验倾斜；与 M3_CTX 的 use_ctx_bias 证据项互补（证据项=自顶向下先验，能量项=吸引子内的竞争偏置）。

## 方法学注记
- 三连实验是"负结果驱动机制定位"的完整案例：exp36 否定（默认参数）→ exp36b 排除"κ 标定"解释（全网格无效）→ 读码找到通路错误 → exp36c 修通路。若只跑 exp36 会错误结论"多时间尺度无用"；若只跑 exp36b 会停在"参数没调对"。
- 预登记纪律：36b/36c 门与判据均写于数据之前（docstring），κ* 由 dev 选择、test 从不参与选择。

## 入文位置
插在 §4.17 末尾 "---" 之后、"## 5 讨论" 之前。§4.7 列表第 10 条后加第 11 条（视 exp36c 判定改写）。§9.10 表加第五行。
