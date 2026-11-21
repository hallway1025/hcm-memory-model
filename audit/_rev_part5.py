# -*- coding: utf-8 -*-
import io
NL = '\r\n'
def block(s):
    return s.replace('\n', NL)

state = '''# 10-STATE · brain-memory-model（HCM 项目检查点）
更新：2026-11-07 06:25

## 当前阶段
**第三轮论文大修订完成**。摘要中英插修正句、贡献 3→4、§4.3 D-036 定案块、§7 L1 重评+五问 1/2 已答、§9.9 新节（第三轮修订记录 9 项处置表）。check_claims **129/129 PASS**；build 链全绿：PAPER_final.md 63905 chars / HCM_PAPER.docx 441KB / html 78KB / SHA256SUMS 316 文件。README/20-DECISIONS（D-038 补）/40-EXP-LEDGER 已同步。

## 已定案（第三轮全部判决）
- D-035 β 对比度坍缩：exp19g 剂量响应（β=20→39/8/0；β≥100 全 40/40@L1）+ exp19h（L0 同模式）。
- D-036 残差有害=β 伪影：exp20（L1 4 种子 β=100 全 40/40）+ exp20b（L0 8 种子 β=100 全 1.000；β=20 锚点 0.900→0.575）。
- D-037 develop() 阈值路线判死（exp22q）。D-039 未见字符随机回退协议（exp23e：124→200/200）。
- 真实语料 200/200@30% 缺失（exp22p 正确判据）；学习编码器 129k near +0.106 t=2.451（exp23c/f）；B2 消歧判决（exp24：无 cue 拒猜/精确 40/40/τ 38/40/错误 cue 100% 特异）。
- 论文新节：§4.12 β 尺度律、§4.13 真实语料+学习编码器+B2、§2.7 LLM 记忆系统六路线（SYS001-007）、附录 D/E、§9.9。

## 下一步（优先序）
1. 用户 B1 优化结果回传（手册在桌面 HCM-算法优化手册.md；用户 m03387）。
2. 二号机空闲：坍缩律外部复现（五招之一：Mem0/HippoRAG 等外部系统复现 β 对比度坍缩律）——可先写 exp25 协议。
3. L2 规模实验设计（d_h=16k+，需 48G/80G 机器）。
4. 语料深度利用：LoCoMo 式长程 QA 基准（bench/corpus 已备）。
5. 投稿格式：NeurIPS/PLoS 模板套壳、fig1 重绘、英文全文。

## 环境（不变）
机 1：i5-13500H/15.7GB。机 2：ssh you_windows2 you@100.112.23.115（RTX5050/31GB，Py3.13.7）。git 已配身份（8f364ca→）。PS 5.1 无 &&（L-036）；check_claims=tools/check_claims.py；build=tools/build_paper.py + export_docx/html.py + make_manifest.py。
'''
io.open(r'memory/10-STATE.md', 'w', encoding='utf-8', newline='').write(block(state))
print('state written')
