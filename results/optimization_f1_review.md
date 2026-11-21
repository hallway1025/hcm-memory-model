# F1 快速回归与独立代码审查（Lead 执行，2026-11-07）

对象：优化者在 src/hcm/text_encoder.py 新增的 CharacterSemanticEncoder（手册 §7.2 任务 F1）。
独立性声明：审查者（Lead）未参与该实现；实现者与审查者会话隔离。

## 1. 回归命令（全部真实退出码）

| 命令 | 退出码 | 结果 |
|---|---|---|
| py -X utf8 tests/test_text_encoder.py | 0 | 4 tests PASS |
| py -X utf8 bench/text_encoder_smoke.py | 0 | PASS vocab=7 episodes=3 |
| py -X utf8 tests/smoke.py | 0 | completion@30% noise 20/20 |
| py -X utf8 tools/check_claims.py | 0 | 151/151 通过（手册旧数字 93/129 已过期，与新交接一致） |

## 2. 代码审查（按 §7.2 清单逐条）

| 审查项 | 结论 | 证据 |
|---|---|---|
| 同 seed 跨进程确定性 | 成立（本机实测） | 独立进程两次 fit+encode，embedding SHA256 逐位一致：e6c985fa...；encode 输出 SHA 4cde8789...（探针 corpus×240，dim=64）。注意：该结论仅对本机 BLAS 构建成立，跨机器复现应以存档 npz 为准（现有实践） |
| 回退向量可区分性 | 成立 | blake2b(seed:char) 派生，纯标记消息以 empty:+原文身份回退（text_encoder.py:107），emoji-only 不坍缩 |
| 回退向量被调用者修改（aliasing） | 无现行 bug，契约脆弱 | _fallback 缓存数组按引用返回（:54），当前仅被 np.sum 只读消费；encode 公开输出为新数组。建议未来加 .copy() 或注释只读 |
| 缓存无限增长 | 理论成立，实践有界 | _fallback 按 OOV 字符增长（字符级，非词级）；对抗输入可无限，语料场景 ≤数千项。记录为限制 |
| fit 失败半更新状态 | 安全 | SVD 中途异常时 vocab/_index 已换、embedding 未换，但 fitted=False 使 encode 抛 RuntimeError（:100-101），无可用但错误的状态 |
| 重复 fit / 不同 seed / 空训练集 / 全表情 | 通过 | 4 项单测覆盖；fit() 幂等重入（_fallback 不清空但内容只依赖 seed:char，无害） |
| 输出 dtype/维度/有限值 | 通过 | float32，unit 归一，PPMI 路径 nan/inf 已 nan_to_num |

## 3. 新发现（此前未记录）

- N-1（定义项，F2 必须统一）：共现计数含自身对（text_encoder.py:80 的内层循环含 pos 自身）→ PPMI 对角线=unigram 计数，高频字自 PMI 偏高。这是定义选择不是 bug，但 F2 各臂与未来复现必须保持同一实现并在报告注明。
- N-2：exp23 的 npz 嵌入（results/exp23_semantic_embed.npz）与 text_encoder.py 是两套 PPMI-SVD 实现。F2 的新-256 臂必须用 text_encoder.py 且只在 train split 上 fit（手册已要求）；旧 npz 只能作为标注了训练集泄漏风险的参考臂，不得当 B1 证据。
- N-3：确定性成立范围=本机；exp29 报告须记录 numpy/BLAS 版本（np.__config__）。

## 4. 未修限制（不算缺陷）

- 未做投影载入接口（手册已声明）；B1 达标与否待 F2/F3。
- 短文本/单字消息的语义质量依赖窗口=3 共现，无消歧能力——B2（多时间尺度）而非 B1 的职责。

## 5. 环境

numpy 2.4.4 / Python 3.14.0 / Windows PowerShell 5.1 / 机器1 i5-13500H。审查耗时约 4 分钟，未占用二号机（exp26 运行中）。