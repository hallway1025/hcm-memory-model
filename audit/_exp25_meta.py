# -*- coding: utf-8 -*-
import io
NL = '\r\n'
def block(s):
    return s.replace('\n', NL)

# ledger row
P = r'memory/40-EXP-LEDGER.md'
t = io.open(P, encoding='utf-8').read()
row = '''

| exp25 | 坍缩律外部复现（零 HCM 代码，vanilla softmax 软检索） | 本机 bash-c2a1f50b6a3c44e6 + 机2 bash-d9b66df68e09465b | 运行中 | dense iid / sparse 超几何 / dense 聚簇三臂 × N≤1e6；β* 随 N 标度拟合 | results/exp25_collapse_external.json |
'''
if 'exp25 |' not in t:
    t = t.rstrip() + block(row) + NL
io.open(P, 'w', encoding='utf-8', newline='').write(t)

# lesson
P = r'memory/30-LESSONS.md'
t = io.open(P, encoding='utf-8').read()
lesson = '''

L-039 检测 β/增益现象必须走软读出；argmax(β·s) 对 β 严格不变（2026-11-07）
- exp25 首版用 pred=argmax(β·S) 测坍缩 → 全 β 同结果，实验设计本身把现象消掉了。β 只通过 softmax 权重被**使用**的路径起作用：加权平均检索（a@K）、多通道融合（λa+(1−λ)s）、迭代吸引子。
- 同批教训：exp25 首版把 argmax **索引**当**分数**存（smax=s[o[0]] 写成 o[0]）→ β(s−537) 全下溢 → 0/0=NaN → 花 hits。排序索引与分数值是两个空间，拷贝循环变量时逐个核对。
- 边界条件（同样重要）：无结构 iid 键 + 单步软检索 = 无坍缩（N=1e6 β=2 仍 50/50，噪声均值≈0）；**坍缩需要候选共享结构**（聚簇/离散原子）或竞争通道。
'''
if 'L-039' not in t:
    t = t.rstrip() + block(lesson) + NL
io.open(P, 'w', encoding='utf-8', newline='').write(t)
print('ledger+lesson done')
