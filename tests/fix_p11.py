import os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# --- P1-1: 审计报告不再硬编码数字，改为指向实时命令 ---
p = os.path.join(ROOT, 'audit', 'AUDIT-REPORT.md')
s = open(p, encoding='utf-8').read()
s = s.replace('**结果：46/46 通过，0 孤儿引文，0 越界措辞。**',
              '**结果：全部通过，0 孤儿引文，0 越界措辞。**（具体条数随论文增长，以下为运行时刻的输出；不要相信本文中任何写死的条数。）')
s = s.replace('| G5 | claim ↔ 引文 ↔ 数字三向比对 | **PASS**（46/46） |',
              '| G5 | claim ↔ 引文 ↔ 数字三向比对 | **PASS**（实时运行 `py -3 tools\\check_claims.py` 查看条数） |')
s = s.replace('详见论文 §4.8。**9 个子系统：7 LOAD-BEARING / 1 REDUNDANT / 1 INERT。**',
              '详见论文 §4.8。**9 个子系统：6 LOAD-BEARING / 1 REDUNDANT / 1 INCONCLUSIVE / 1 INERT。**'
              + chr(10) + chr(10)
              + '> 更正记录（第二轮独立审查 P1-1）：本句早期版本写「7 LOAD-BEARING / 1 REDUNDANT / 1 INERT」，'
              + '漏了 INCONCLUSIVE，且是过期快照。现改为与 `results/g4_ledger.json` 一致的 6/1/1/1。')
s = s.replace('| G4 | 九子系统消融齐全（含专属指标） | **PASS**（7/1/1） |',
              '| G4 | 九子系统消融齐全（含专属指标） | **PASS**（6 LOAD-BEARING / 1 REDUNDANT / 1 INCONCLUSIVE / 1 INERT） |')
open(p, 'w', encoding='utf-8').write(s)
print('AUDIT-REPORT: stale numbers fixed')

# --- README 同步 ---
r = os.path.join(ROOT, 'README.md')
t = open(r, encoding='utf-8').read()
t = t.replace('| G5 | claim ↔ 引文 ↔ 数字三向比对 | ✅ 通过（条数随论文增长，以 check_claims.py 的实时输出为准） |',
              '| G5 | claim ↔ 引文 ↔ 数字三向比对 | ✅ 通过（条数随论文增长，以 check_claims.py 实时输出为准，勿引用任何写死的数字） |')
open(r, 'w', encoding='utf-8').write(t)
print('README synced')