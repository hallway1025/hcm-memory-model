# exp30 设计：事件层（内容/事件身份分离）

日期：2026-11-07 19:30（状态更新 19:55） · 负责：Lead · 依据：D-042 / audit/CLAIM-STATUS.md C12 / 外部评审③
写作用域：src/hcm/episodic.py（增量，不破坏现有语义）、bench/exp30_*.py、results/exp30_*.json、notes/、ledger

## 1. 问题（已证实）
src/hcm/episodic.py:38-39：`if self.max_cos(h) >= theta_dup: return -1`——内容去重在元数据落盘前**整条丢弃**事件。
后果："昨天说的上号"与"今天说的上号"只剩第一条；时间检索（t/sender/session）在重复内容上失效。
佐证：emoji 拒写 545/2000（§4.13.1）；OK×155 塌缩单码（§4.13.4）。

## 2. 设计原则
- **码（内容）与事件（发生）分离**：E 的码槽仍按 theta_dup 去重（保留容量收益）；新增事件记录层，每条输入恒写。
- 事件记录：`{event_id, t, sender, session, slot}`，slot=内容码槽 id（新码=新 slot；重复内容=引用最近码槽）。
- 检索分层：内容检索走原路径（attractor over slots）；时间/来源检索走事件层（scan events，O(N_e)，N_e=事件数）。
- P（原始载荷）保持 slot 级（存首次出现）；事件级载荷差异在 theta_dup 下视为等价内容（L0 限制，如实报告）。

## 3. API 增量（向后兼容）
- `write_event(x, t, sender="", session="") -> {event_id, slot, is_new_code}`
- `query_events(t0=None, t1=None, sender=None, session=None) -> [event 记录 + slot 码 + recon 分数]`
- `query_at_slot(slot)` / 既有 `write/query` 语义不变；现有测试必须全绿。

## 4. 实验
- exp30a 事件可分性（核心验收）：同一内容两个时间戳写入 → 按时间窗检索两条事件均返回且指向同一 slot；按内容检索返回两条事件 id。判据：2/2 事件可分，且与不写事件层的基线对比证明修复。
- exp30b 去重率与开销实测：David 2000 条跑 write_event，报告 事件数/新码数（去重率）、事件层内存（每事件 ~100B 级）、写入延迟增量。
- exp30c 回归：tests/smoke 20/20、check_claims 151/151、exp22p 复跑无回归。

## 5. 验收门槛
①exp30a 全过；②三套回归全绿；③去重率与内存开销写入 MODEL.md §6 与 CLAIM-STATUS C12（状态改"已修复-实验验证"）；④论文 §7 缺口 6 补"修复与实测"小节。

## 6. 状态（2026-11-07 19:55）
- §3 API：已实现（episodic.py 事件层 + model.py observe(event=True) 透传）。实现细节与设计差异：事件记录恒写，但**新内容码的创建仍受门控约束**（保持 E 容量语义与默认路径逐位一致）；重复内容引用既有码不受门控影响。
- §4 exp30a：PASS 100/100（共享码/时间窗分离/内容检索）+ 基线缺陷复现 100/100 + 三次发生 20/20。
- §4 exp30b/exp30c(exp22p 复跑)：排队，full2 结束后执行（保机1 计时干净）。
- 回归：text_encoder 4/4、smoke 20/20、check_claims PASS。
