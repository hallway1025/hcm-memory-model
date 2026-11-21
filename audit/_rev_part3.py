# -*- coding: utf-8 -*-
import io
NL = '\r\n'
def block(s):
    return s.replace('\n', NL)

# ---------- README ----------
P = r'README.md'
t = io.open(P, encoding='utf-8').read()

t = t.replace('中文正文 + 英文摘要，8 表 5 图，32 条编号文献', '中文正文 + 英文摘要，39 条编号文献')

o = '| 残差代价随库规模增长 | N=4000 差 0.02 → N=8000 差 0.12（8 种子） | exp14_resid_harmful.json |'
n = ('| ~~残差代价随库规模增长~~（第三轮推翻：β=20 伪影，见下） | β=20 下 N=8000 差 0.12；β=100 下 8 种子全 1.000 | exp14_resid_harmful.json / exp20b_resid_beta_l0.json |' + NL +
'| **β 检索对比度尺度律（第三轮主发现）** | β=20@L1：39/8/0 → β=100/500/2500：全 40/40（剂量响应） | exp19g_beta_scale.json |' + NL +
'| 残差×β 跨规模定案 | β=100：L0 8 种子 + L1 4 种子两臂全满分；β=20 锚点随 N 恶化 0.900→0.575 | exp20_resid_beta_l1.json / exp20b_resid_beta_l0.json |' + NL +
'| **真实语料端到端（12.9 万条微信）** | 正确判据：200/200@30% 缺失、199/200@50%（λ=0.6 与 1.0 双臂） | exp22p_correct_judge.json |' + NL +
'| 学习型字符编码器（B1） | 129k 共现：near +0.106（t=2.451, p≈0.03）、检索 200/200、188/200 | exp23c_data_scale.json / exp23f_embed_stats.json |' + NL +
'| 未见字符随机回退协议 | asis 124/200 → randfallback **200/200**@0.3+0.5 | exp23e_emoji_verdict.json |' + NL +
'| B2 重复内容消歧判决 | 无 cue 0/40（拒猜）、精确 40/40、τ 38/40、部分 6/40、错误 cue 特异性 100% | exp24_b2_disambiguation.json |')
assert o in t; t = t.replace(o, block(n), 1)

o = '| L1 先导（跨机器） | 精确 1.000 迁移；25% 缺失崩到 0.250；残差臂 0.031 | exp16_l1_scale.json |'
n = '| L1 先导（跨机器，第三轮重评） | 精确迁移；初报 0.250/0.031 为 β=20 伪影——β=100 四种子两臂全 40/40 | exp16_l1_scale.json / exp20_resid_beta_l1.json |'
assert o in t; t = t.replace(o, block(n), 1)

t = t.replace('├── bench/      31 个基准 + reproduce_all.py（一键复现）', '├── bench/      40+ 基准（exp1–exp24 系列）+ reproduce_all.py（一键复现）')
t = t.replace('├── refs/       refs.jsonl（373 条）', '├── refs/       refs.jsonl（380 条）')
t = t.replace('└── memory/     10-STATE（断点）、20-DECISIONS（30 条）、30-LESSONS（22 条）', '└── memory/     10-STATE（断点）、20-DECISIONS（39 条）、30-LESSONS（38 条）')

o = '第二轮洁净复现已完成'
i = t.find(o)
print('G6 line ctx:', t[i-160:i+160].replace(chr(13), ''))

io.open(P, 'w', encoding='utf-8', newline='').write(t)
print('README done')
