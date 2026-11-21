# -*- coding: utf-8 -*-
import io
NL = '\r\n'
def block(s):
    return s.replace('\n', NL)

# ---------- README: G6 line add third round ----------
P = r'README.md'
t = io.open(P, encoding='utf-8').read()
o = '→ §9.8、MODEL.md §3.2a |'
n = '→ §9.8、MODEL.md §3.2a；第三轮修订（β 尺度律定案 + 真实语料正名 + 评测法证）→ §4.12、§4.13、§9.9 |'
assert o in t; t = t.replace(o, block(n), 1)
io.open(P, 'w', encoding='utf-8', newline='').write(t)

# ---------- 20-DECISIONS: append D-038 ----------
P = r'memory/20-DECISIONS.md'
t = io.open(P, encoding='utf-8').read()
d38 = '''

D-038 θ 发育协议必须写死且不得混用（2026-11-07）
- 实验：bench/diag_theta_protocol.py（N=8000，3 种子）。θ≡0 → deg25 0.950/0.842；复现者二值占空比发育 → 1.000/0.950；码值语义发育 η=0.35 → **0.000/0.000**。一个旋钮横跨 [0,1] 全域。
- **决定**：主结果实验的 θ 状态写死在 Config；跨协议只比定性方向，绝不比绝对数值；任何「复现对不上」先查发育协议差异（§9.8.3）。
'''
if 'D-038' not in t:
    t = t.rstrip() + block(d38) + NL
io.open(P, 'w', encoding='utf-8', newline='').write(t)

# ---------- 40-EXP-LEDGER: fix stale statuses ----------
P = r'memory/40-EXP-LEDGER.md'
t = io.open(P, encoding='utf-8').read()
t = t.replace('| exp20b | exp14 重跑（β=100, L0） | bash-54ef5a339d7f42c7 | 运行中 | 8 种子×N{4000,8000,16000}×residF/T+β20 锚点 | results/exp20b_resid_beta_l0.json |',
              '| exp20b | exp14 重跑（β=100, L0） | bash-54ef5a339d7f42c7 | 完成 | β=100 全 cell 1.000（8 种子）；β=20 锚点随 N 恶化 0.900→0.575（F）/0.275（T）→ D-036 跨规模定案 | results/exp20b_resid_beta_l0.json |')
t = t.replace('| exp23d | 门控取证双臂（hcos_true 对比） | bash-36092be07c1c44fa | 重跑中 | 2k 嵌入 k-WTA 翻转 vs 129k 修复 | results/exp23d_gate_forensics.json |',
              '| exp23d | 门控取证双臂（hcos_true 对比） | bash-36092be07c1c44fa | 完成 | 2k 嵌入失败探针 hcos_true=0（k-WTA 翻转）；129k 嵌入 0 失败、hcos 0.428→0.703 | results/exp23d_gate_forensics.json |')
t = t.replace('| exp23f | 嵌入层配对统计（无模型） | bash-54e3631e66724dae | 运行中 | near/dis/rand 三组 × 三嵌入 | results/exp23f_embed_stats.json |',
              '| exp23f | 嵌入层配对统计（无模型） | bash-ffb357fea8334bdd | 完成 | near 129k vs R256 Δ+0.106 t=2.451 p≈0.03；dis 0.0759=2.7×R256 | results/exp23f_embed_stats.json |')
verdict = '''

**第三轮收官（2026-11-07 06:20）：PAPER.md 大修订完成**——摘要中英第三轮修正句、贡献 3→4、§4.3 D-036 定案块、§7 五问 1/2 已答、§9.9 新节（9 项处置表）；check_claims 93→129 条全 PASS；build 链全绿（PAPER_final 63905 chars / docx 441KB / html 78KB / manifest 316 文件）。
'''
if '第三轮收官' not in t:
    t = t.rstrip() + block(verdict) + NL
io.open(P, 'w', encoding='utf-8', newline='').write(t)
print('decisions+ledger done')
