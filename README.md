# HCM — 海马—皮层互补记忆模型（可复现参考实现）

一个九算子的计算记忆模型，配可种子复现的纯 NumPy 实现与完整实验台账。用于检验互补学习系统（CLS）的核心机制在**工程上是否真的成立**——包括它们不成立的地方。

> **论文与补充材料**：见 [Zenodo](https://doi.org/10.5281/zenodo.24359580)（概念 DOI，永远指向最新版）。
> 本仓库是论文的**复现代码与实验台账**；机制细节、理论推导与结果讨论以论文为准。

## 模型是什么

把记忆分解为九个算子：输入/情境空间、编码器、情节快存、语义慢存、工作记忆、巩固、检索控制、遗忘、神经调质门控。

核心主张三条（细节见论文）：

1. **坍缩律**——检索逆温度 β 的危害随库规模按 log N 增长；β 不足时注意力被压平，与负载无关。
2. **单投影双读出**——k-稀疏码负责写入，稠密证据作为吸引子初态负责检索；60% 缺失下稀疏读出掉到 39/50，稠密仍 50/50。
3. **巩固的价值在交错而非优先级**——无回放旧任务崩到 0.0/24，交错回放保留 13.85/24，优先级回放 14.80/24（无显著差异，如实报告）。

## 快速开始

**前置**：需要 Python 3.10+（实测 3.12 与 3.14 均可）。Windows 上如果 `python` / `py` 指向应用商店占位符（文件 0 字节），先装一个真的：
`winget install Python.Python.3.12`。

```bash
git clone https://github.com/hallway1025/hcm-memory-model
cd hcm-memory-model

# 1) 建隔离环境并装钉版依赖（约 1 分钟）
python -m venv .venv-repro
.venv-repro/Scripts/pip install -r repro/requirements.txt    # Windows
# source .venv-repro/bin/activate && pip install -r repro/requirements.txt  # Linux/macOS

# 2) 一条命令跑完全部实验与四道验收（约 11 分钟，纯 CPU）
.venv-repro/Scripts/python bench/reproduce_all.py            # Windows
# .venv-repro/bin/python bench/reproduce_all.py              # Linux/macOS
```

跑完会依次执行 24 个步骤并给出验收结论；任一步失败即非零退出。最后再跑一遍四道守卫确认交付物完整：

```bash
.venv-repro/Scripts/python repro/run_guards.py
```

**只跑单个实验**：

```bash
py -3 bench/exp1_separation.py    # 模式分离主表
py -3 bench/exp6_continual.py     # 持续学习 / 回放
py -3 bench/exp11_readout.py      # 双读出互补
```

> ⚠️ 单个脚本会覆写 `results/*.json`，导致后续一致性检查失配。单跑实验后请重跑完整的 `reproduce_all.py` 恢复。

## 在自己的数据上用

```python
import sys; sys.path.insert(0, 'src')
from hcm import HCM, Config

m = HCM(Config(d_x=128, a=0.02, capacity=2000, L=3, use_resid=True))

for x in stream:                 # 在线：编码 → 门控 → 写入
    m.observe(x, r=reward)

out = m.query(cue)               # 检索：三分支决策（recall / separate / abstain）
if out['action'] == 'recall':
    memory, confidence = out['pred'], out['raw_evidence']

m.consolidate(cycles=10)         # 离线：交错回放
m.forget()                       # 遗忘：剪枝 + 抑制
```

## 仓库结构

```
src/hcm/     模型参考实现（12 个模块，纯 NumPy，无深度学习框架依赖）
bench/       实验脚本（exp1–exp59）与 reproduce_all.py 一键入口
results/     每个实验的 JSON 结果（含全部条件与随机种子）
repro/       一键环境（setup + run_repro + 四道验收守卫）
refs/        448 条引文库与机器核验元数据
docs/        MODEL.md（模型规格）· SCALE-PROTOCOL.md（超算协议）· REVISIONS.md（构建期台账）
audit/       主张台账、引文核验报告、SHA256 清单
paper/       论文源文件与图表
tests/       诊断脚本（每个对应一个已解决的失效模式）
```

## 复现口径

四道验收守卫（`repro/run_guards.py`）全绿才算通过：

| 守卫 | 判据 |
|---|---|
| G1 数字溯源 | 457/457 断言与结果文件逐位一致 |
| G2 引文核验 | 448 条逐条机检（DOI 可解析、标题相似度、年份、作者、撤稿、期刊分级） |
| G3 引文闭合 | 57 条正文引用与引文库双向闭合 |
| G4 校验和 | 681 个交付文件的 SHA256 一致 |

**环境**：Python 3.10+（开发用 3.14）· numpy · matplotlib · python-docx。不需要 torch / scipy。全程纯 CPU，单线程。

## 边界（请先读）

- **核心实验基于合成数据**；真实语料结果限于单个语料家族的描述性对照，向图像/语言/真实情景记忆的外推未经验证。
- **L0/L1/L2 已实测**（最高到 N=10⁵）；**L3（N=2×10⁷）是设计协议而非实测**。
- **写入复杂度 O(N²)**（稠密门逐条扫库）；`fastpath.py` 提供的倒排+f32 修复是 opt-in 的，未接入默认路径。
- **三个诚实的否定结果**：读出闸门干预无效、遗忘策略无已证收益、学习型编码器未转化为检索优势——都写在论文里，不藏。
- **真实语料不随本仓库分发**（含真实姓名与私人对话）。复现真实语料那部分需要自备同结构语料；其余全部可复现。

完整局限见论文 §6。

## 引文

全部引文由 API 产出，无手写 DOI：检索（Crossref/OpenAlex）→ 六通道核验（Crossref、PubMed、Europe PMC、OpenAlex、arXiv API、图书/网页交叉核验）→ 回填 → 编译期未匹配 token 硬失败。核验报告在 `audit/citations-report.json`。

## 许可与引用

- 代码见 `LICENSE`；论文与补充材料见 Zenodo 记录所附许可。
- **引用本工作**：用 Zenodo **版本 DOI**（固定你读的那一版）；指向最新版用**概念 DOI** `10.5281/zenodo.24359580`。
