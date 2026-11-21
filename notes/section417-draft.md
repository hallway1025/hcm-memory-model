# §4.17 草稿骨架（exp34b 存储级 RIF 三难）— 数字待 exp34b_analyze.json，不得先填

## 结构（插在 §4.16 末尾"---"前）

### 4.17 存储级抑制的三难：幅值衰减产生枢纽塌缩而非选择性遗忘（exp34b，第四轮）

1. **动机与预登记**。§4.16 的可检验推论：把抑制改为存储级写入后 below-baseline RIF 应当出现。
   本节以预登记共主判据检验（判据与分析器 bench/exp34b_analyze.py 写于全量数据之前）：
   - G-A'（below-baseline）：son015 臂 hit_nrp − hit_rp_sup ≥ 0.10 且 bootstrap 95%CI 排 0，安慰剂臂设计忠实亏损 < 0.05；
   - G-S1（练习项塌缩）：hit_p(son030) < hit_p(soff) − 0.10 且 CI 排 0（配对种子）；
   - G-S2（枢纽形成）：hub_share(son030) ≥ max(0.15, 2×hub_share(soff))；
   - G-B'（精确逆恢复）：M[:,s]/=att[s] 后 |hit_recovered − hit_p| ≤ 0.05；
   - G-B''（塌缩逆恢复）：全部衰减列恢复后 hit_p 回到同种子 soff 的 0.05 以内。
2. **设计**。peer-02 单 cell（预登记：exp34 显示 syn cell 无动态范围）× 4 臂 {soff, son015, son030, son050} × 12 seeds，
   配对 rng（crc32(b"david")（内部种子常量，与旧标签同名），同库同练习集同候选池）。存储级操作化 = 列幅值衰减 M[:,j] ← (1−ρ)·M[:,j]，
   竞争者选择镜像 E.suppress（top-5 吸引子、赢家豁免、use[winner]+=1）；att[s] 记累计衰减供逆恢复。
   测试 do_rif=False、判据 obs_of（L-035）。smoke（N=60）与全量（N=2000，深度 ~0.63 事件/槽）分开报告。
3. **结果：[按分析器数字填写三段——(i) G-A' 判决；(ii) G-S1/G-S2 判决与 hub_share 表；(iii) G-B'/G-B'' 数字]**
4. **三难（trilemma）**。在余弦几何吸引子库中，"抑制一条记忆"只有三种实现：
   (a) 读出增益（sup 乘子）：局部、可逆、不触存储——exp34：零行为亏损；
   (b) 幅值衰减：进入吸引子动力学 → 未衰减列平均方向主导 z-更新 → 枢纽槽吸走探针 → 全局塌缩（本节）；
   (c) 方向改变 = 该码不再匹配自身 = 删除而非抑制。
   解析注：检索证据 ed/‖ed‖ 在整向量层面归一化，逐列幅值是唯一进入证据排序的存储把手；
   列重归一化会把任何纯幅值操作完全抵消（可解析证明），故"无副作用的存储级抑制"在此几何下不存在。
5. **结论与生物学映射**。M8 作为检索增益闸门（exp34 重定位）不是实现妥协而是结构必然：
   迹衰减要么被归一化抵消（无效）要么破坏全局动力学（灾难）。突触稳态缩放（homeostatic
   synaptic scaling，{{Turrigiano1998}}）在生物学里扮演的正是"归一化抵消"角色——单个突触的
   强弱变化在树突尺度上被全局缩放吸收，表达层面的抑制必须走增益/情境通路。
   方法学注记：smoke 规模（N=60，衰减覆盖 40/60 槽、深度 ~4 事件）足以暴露枢纽机制，
   但规模外推不可靠（N=2000 深度仅 ~0.63）——诊断性中间量（modal-slot share）应在全量前
   仪器化，这是 L-035（机制归因先验中间量）的又一实例。

## 数字占位（分析器字段映射，写文时替换）
- G-A'：deficit_mean / ci95 / placebo_deficit_mean（results/exp34b_analyze.json → G-A）
- G-S1：collapse_mean / ci95（→ G-S1）
- G-S2：hub_share 表 soff/son015/son030/son050（→ G-S2.hub_share）
- G-B'：recovered_mean / gap_rec_hp_mean（→ G-B）
- G-B''：hp_all_rec_mean / gap_to_soff_hp_mean（→ G-B）
- 剂量：dose_response + dose_monotone_hitp_collapse
- 塌缩逆恢复后 hub：hub_share_after_restore（rows 字段，如需）
