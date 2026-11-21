import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ---------- MODEL.md ----------
p = os.path.join(ROOT, 'docs', 'MODEL.md')
s = open(p, encoding='utf-8').read()
s = s.replace('μ^(j) ← (1 − γ_μ) μ^(j) + γ_μ x_t ,   γ_μ = 0.9',
              'μ^(j) ← γ_μ μ^(j) + (1 − γ_μ) x_t ,   γ_μ = 0.9   # 慢速：每步只吃进 10% 新样本')
open(p, 'w', encoding='utf-8').write(s)
print('MODEL.md EMA fixed')

# ---------- PAPER.md ----------
q = os.path.join(ROOT, 'paper', 'PAPER.md')
t = open(q, encoding='utf-8').read()

gen = ('',
 '**数据生成器的精确定义（此前缺失，是洁净复现失败的根因之一）**。所有「overlap」实验使用：',
 '',
 '~~~text',
 'shared = N(0, I_d)                      # 一个共享方向',
 'x_i    = unit( share * shared + (1-share) * priv_i )   # priv_i ~ N(0, I_d)',
 '~~~',
 '',
 '**注意：「overlap」是混合权重 share，不是输入余弦。** share = 0.90 时实测输入两两余弦为',
 '**0.986 ± 0.004**（min 0.981 / max 0.990），不是 0.90。这一点此前没有写清，导致独立复现者',
 '在「输入余弦恰为 0.90」的数据上重建时得到完全不同的结论（他们那组数据太容易，无残差臂也接近满分）。',
 '这是本项目在文档完整性上的一个实质缺陷，由冻结版洁净复现发现并推动修正。',
 '')
anchor = '**指标。** 所有检索类实验统一报告三个量'
if anchor in t and '数据生成器的精确定义' not in t:
    t = t.replace(anchor, chr(10).join(gen) + chr(10) + anchor)

t = t.replace('μ ← (1 − γ_μ) μ^(j) + γ_μ x_t', 'μ ← γ_μ μ + (1 − γ_μ) x_t（慢速 EMA，每步只吃进 10%）')
open(q, 'w', encoding='utf-8').write(t)
print('PAPER.md generator documented:', '数据生成器的精确定义' in t)