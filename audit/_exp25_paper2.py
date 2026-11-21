# -*- coding: utf-8 -*-
import io
NL = '\r\n'
def block(s):
    return s.replace('\n', NL)

P = 'paper/PAPER.md'
t = io.open(P, encoding='utf-8').read()

# 1) fix exp20b N range in prose
o = '固定 β=100 后重跑同协议：L0 8 种子 × N∈{4000, 8000} 双臂全部 1.000/1.000（CI 完全重叠）；β=20 单种子锚点复现旧效应（N=8000：无残差 0.900 / 有残差 0.775）。'
n = '固定 β=100 后重跑同协议：L0 8 种子 × N∈{4000, 8000, 16000} 双臂全部 1.000/1.000（CI 完全重叠）；β=20 单种子锚点复现旧效应且危害随 N 增长（N=8000：无残差 0.900 / 有残差 0.775；N=16000：0.575 / 0.275）。'
assert o in t; t = t.replace(o, block(n), 1)

# 2) add two table rows
o = '| L0 N=8×10³ | 20 | 0.900（锚点） | 0.775（锚点） | exp20b |'
n = (o + NL + '| L0 N=1.6×10⁴ | 100 | 1.000±0.000 | 1.000±0.000 | exp20b（8 种子） |' + NL +
'| L0 N=1.6×10⁴ | 20 | 0.575（锚点） | 0.275（锚点） | exp20b |')
assert o in t; t = t.replace(o, block(n), 1)

# 3) appendix F pointer after scale-law sentence
o = 'β 的充分值随 log N 增长（本域内 β=100 覆盖 N≤2×10⁴）。'
n = ('β 的充分值随 log N 增长（本域内 β=100 覆盖 N≤2×10⁴）。该律已在**零 HCM 代码**的 vanilla 注意力软检索中外部复现'
'（附录 F）：聚簇共享质心键下 β*(N) 从 2 升至 ≥5、固定 β=2 的命中从 73% 塌至 0%（N=10⁵）；'
'无共享结构的 iid 键则完全鲁棒（N=10⁶ 仍满分）——坍缩的载体是候选间的共享结构，不是 softmax 本身。')
assert o in t; t = t.replace(o, block(n), 1)

# 4) abstract pointers zh/en
o = '（剂量-响应与解析理论见 4.12 节与附录 E）'
n = '（剂量-响应、解析理论与外部复现见 4.12 节与附录 E/F）'
assert o in t; t = t.replace(o, block(n), 1)
o = '(dose-response plus analytical theory; Sec. 4.12, Appendix E)'
n = '(dose-response, analytical theory, and an external replication in vanilla attention; Sec. 4.12, Appendices E/F)'
assert o in t; t = t.replace(o, block(n), 1)

io.open(P, 'w', encoding='utf-8', newline='').write(t)
print('4.12 + abstract updated, len', len(t))
