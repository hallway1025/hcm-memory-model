# -*- coding: utf-8 -*-
import io
NL = '\r\n'
def block(s):
    return s.replace('\n', NL)

# README key-results row
P = 'README.md'
t = io.open(P, encoding='utf-8').read()
o = '| B2 重复内容消歧判决 | 无 cue 0/40（拒猜）、精确 40/40、τ 38/40、部分 6/40、错误 cue 特异性 100% | exp24_b2_disambiguation.json |'
n = (o + NL + '| **坍缩律外部复现（附录 F，零 HCM 代码）** | 聚簇键：β=2 命中 73.5%→0%（N=10³→10⁵）、β*(N)=2→≥5；无结构 iid 键 N=10⁶ 仍满分 | exp25_collapse_external.json |')
assert o in t; t = t.replace(o, block(n), 1)
io.open(P, 'w', encoding='utf-8', newline='').write(t)

# ledger row -> done
P = 'memory/40-EXP-LEDGER.md'
t = io.open(P, encoding='utf-8').read()
o = '| exp25 | 坍缩律外部复现（零 HCM 代码，vanilla softmax 软检索） | 本机 bash-c2a1f50b6a3c44e6 + 机2 bash-d9b66df68e09465b | 运行中 | dense iid / sparse 超几何 / dense 聚簇三臂 × N≤1e6；β* 随 N 标度拟合 | results/exp25_collapse_external.json |'
n = '| exp25 | 坍缩律外部复现（零 HCM 代码，vanilla softmax 软检索） | 本机 bash-c2a1f50b6a3c44e6 + 机2 bash-d9b66df68e09465b | 完成 | 聚簇键 β=2 命中 73.5%→0%、β*(N)=2→≥5（坍缩律外部成立）；iid 键无坍缩（边界条件）；稀疏原子近免疫 → 附录 F + §4.12 指针 | results/exp25_collapse_external.json |'
assert o in t; t = t.replace(o, block(n), 1)
io.open(P, 'w', encoding='utf-8', newline='').write(t)
print('readme+ledger done')
