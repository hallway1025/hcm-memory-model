# -*- coding: utf-8 -*-
import io
P = r'paper/PAPER.md'
t = io.open(P, encoding='utf-8').read()
NL = '\r\n'
def block(s):
    return s.replace('\n', NL)

# ---- 1) Chinese abstract: insert third-round results before the closing sentence ----
zh_old = '我们还报告了三个反直觉发现：'
zh_new = ('第三轮修订带来本文最大的自我修正：检索逆温度 β 的危害随库规模按 log N 增长，'
'早前「残差随规模有害」与「L1 规模去噪不迁移」两个负结果均为其伪影，β=100 下全部消失'
'（剂量-响应与解析理论见 4.12 节与附录 E）。在修正后的参数下我们新增四组结果：'
'12.9 万条真实微信对话端到端检索 200/200；从局部字符共现学习的编码器习得语义相似度结构'
'（近义对码余弦提升 +0.106，t=2.451）且不损检索；重复内容消歧判决实验证明 episode 分辨率'
'由情境与时间通道因果承载（错误线索特异性 100%，无线索时模型拒绝任意召回）；'
'以及一份「三个评测伪影」的法证案例。我们还报告了三个反直觉发现：')
assert zh_old in t, 'zh abstract anchor missing'
t = t.replace(zh_old, zh_new, 1)

en_old = 'We also report three counter-intuitive findings:'
en_new = ('The third revision brings our largest self-correction: the harm of the retrieval inverse temperature beta grows with library size as log N, and two earlier negative results (residual harm growing with scale; L1-scale denoising failure) were artifacts of a single mis-set beta, disappearing entirely at beta=100 (dose-response plus analytical theory; Sec. 4.12, Appendix E). Under the corrected setting we add four results: end-to-end retrieval of 200/200 on 129k real WeChat messages; a character co-occurrence encoder that acquires semantic similarity (synonym code-cosine +0.106, t=2.451) without sacrificing retrieval; a disambiguation verdict experiment showing episode resolution is carried causally by context and time channels (wrong-cue specificity 100%, and the model refuses arbitrary recall when uncued); and a forensic case study of three evaluation artifacts. '
'We also report three counter-intuitive findings:')
assert en_old in t, 'en abstract anchor missing'
t = t.replace(en_old, en_new, 1)

# ---- 2) contributions: 3 -> 4 ----
c_old = '三项主要贡献：'
c_new = '四项主要贡献：'
assert c_old in t
t = t.replace(c_old, c_new, 1)
contrib_anchor = '而稠密存储需要约 5.2 TB。'
c4 = (contrib_anchor + NL + NL +
'4. **增益的规模律与负结果的仪器学**。检索 softmax 的逆温度必须随库规模按 log N 增长（附录 E 给出解析依据与实测验证）。我们如实报告两个曾被当作「发现」的负结果——残差随规模有害、L1 去噪不迁移——最终证明它们是同一增益误设的伪影；由此提炼出可复用的判负流程：**负结果先查仪器（评测对齐、增益标定），再谈机制**（4.12、4.13 节）。')
assert contrib_anchor in t
t = t.replace(contrib_anchor, c4, 1)

# ---- 3) §4.3 narrative rewrite ----
o1 = '完整的规模曲线是：N=4000 两臂 CI 重叠，**效应不存在**；N=8000 分离确立（0.822 vs 0.938，CI 不重叠）；N=16000 差距扩大到 27.8 个百分点（0.291 vs 0.569），且残差臂连**精确**召回都开始受损（0.972±0.011 < 1.000）。所以正确的表述是：**残差的代价随库规模增长而显现并放大**，不是「任何情况下都有害」。机制上，残差在随机正交数据上引入的是零均值的额外抖动，条目少时 k-WTA 吸收得住；条目多时抖动叠加干扰把缺失线索检索推过悬崖。早期版本用 N=16000 的单次测量下结论，未做多种子验证，由第二轮独立审查指出后补做（审查 P0-1）。'
o2 = 'L1 先导实测（第 7 节，exp16）进一步显示：该差距随规模增长，在 N=2×10^4、d_h=4 096 处扩大到 0.22（无残差 0.250±0.040 vs 有残差 0.031±0.012）。'
n1 = ('> **第三轮定案（D-036，exp20b）**。上表与表 3 的两臂差异全部在 β=20 下测得。β=100、8 种子重跑同一协议'
'（bench/exp20b_resid_beta_l0.py）：N∈{4 000, 8 000, 16 000} 两臂的精确与 25% 缺失召回**全部 1.000±0.000**'
'——「残差有害」在 β=100 下不存在。β=20 单种子锚点复现上表模式，且危害随 N 增长'
'（N=8 000 无残差 0.900，N=16 000 无残差 0.575、有残差 0.275）：残差臂只是证据对比度更低，在 β 不足时先跌落悬崖'
'（4.12 节、附录 E）。本节 β=20 数字保留，作为「误设的增益参数会在规模放大时把伪效应归因到任意子系统」的案例研究；'
'上段「零均值抖动」机制解释作废。RQ2 结论不变——容量瓶颈是去噪——但去噪失效的主杠杆是检索对比度需随 log N 增长，不是残差。')
assert o1 in t, 'o1 missing'
t = t.replace(o1, block(n1), 1)
assert o2 in t, 'o2 missing'
t = t.replace(o2, '', 1).replace(NL + NL + NL, NL + NL)

# ---- 4) §7 pilot rewrite ----
o3s = '**L1 先导实测（exp16，二号机 RTX 5050 笔记本，4 种子）**。'
i3 = t.index(o3s)
i3e = t.index('**必须在超算上回答的五个问题**', i3)
n3 = block('> **第三轮重评**。初报三结论：(i) 精确检索完美迁移（1.000±0.000 / 0.994±0.012）；(ii) 退化线索去噪不迁移（25% 缺失召回 0.250±0.040，无残差臂）；(iii) 残差害处随规模增长（有残差臂 0.031±0.012）。诊断链（exp19–exp20）把 (ii)(iii) 全部推翻：崩塌由 β=20 造成——β=100 时 L1 四种子、两臂在缺失 0/10/25/35% 下**全部 40/40**（results/exp20_resid_beta_l1.json），β 剂量响应与附录 E 的对比度理论定量一致。修正后的结论：(i) 不变；(ii′) 精确检索与去噪在 β 按 log N 缩放时都能迁移；(iii′) **超参数的规模律是迁移的一部分**——在新规模报告负结果之前，必须先排除增益误设（与 4.13.2 的评测法证同族：负结果先查仪器）。计时数据不变：单臂写入 448–453 s、单次检索 2.5 s。' + NL + NL)
t = t[:i3] + n3 + t[i3e:]

q1o = '1. **容量—去噪曲线在大 d_h 下是否移动？** 若瓶颈是去噪，d_h 增大应把 25% 缺失线索的失效点从 N≈4000 推后；否则曲线不动。'
q1n = '1. **容量—去噪曲线在大 d_h 下是否移动？** L1 先导已给第一答：β=100 下 25% 缺失在 N=2×10⁴、d_h=4 096 处 40/40——失效点被 d_h 与 β 的联合缩放推后；L2/L3 待确认。'
assert q1o in t; t = t.replace(q1o, block(q1n), 1)
q2o = '2. **残差在高维下是否仍有增益？** L0 上残差在随机数据上反而有害；高维下随机向量近正交，残差可能变中性。这决定残差是否应作默认组件。'
q2n = '2. **残差在高维下是否仍有代价？** L1 先导已答：β=100 下残差中性（四种子两臂全 40/40）；「残差有害」为 β=20 伪影（4.3 定案）。残差是否作默认组件取决于其收益侧证据（模式分离，4.1）而非代价。'
assert q2o in t; t = t.replace(q2o, block(q2n), 1)

io.open(P, 'w', encoding='utf-8', newline='').write(t)
print('part1 done, len', len(t))
