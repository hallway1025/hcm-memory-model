# HCM v1.4 — Release README

**HCM（Hippocampal–Cortical Complementary Memory）**：一个九算子的计算记忆模型，带可执行、可种子复现的 NumPy 参考实现与系统化基准。

---

## 这个版本包含什么

| 文件 | 内容 |
|---|---|
| `HCM_PAPER.pdf` | 中文主稿（权威版，52 页） |
| `HCM_PAPER_EN.pdf` | 英文主稿（对应译本，66 页） |
| `SUPPLEMENTAL-S2.pdf` | 规模试验实测全叙事（L1 更正 + L2 @N=10⁵ 的 exp37）与五个待超算回答的问题 |
| `SUPPLEMENTAL-S4.pdf` | 引文核验完整台账（448 条，逐条含 DOI/PMID、核验通道与检索时间戳） |
| `SHA256SUMS.txt` | 全仓 681 个交付文件的 SHA256 校验和（git 跟踪面；本包 4 份 PDF 不在清单内，以 Zenodo 记录的文件校验和为准） |

---

## 代码与结果在哪里

论文的全部实验代码、结果 JSON、图表脚本与复现环境**随仓库分发**：

- **GitHub**：`https://github.com/hallway1025/hcm-memory-model`
- 仓库内主要目录：
  - `src/hcm/` — 模型参考实现（12 个模块，纯 NumPy）
  - `bench/` — 实验脚本（exp1–exp59）
  - `results/` — 每个实验的 JSON 结果（含全部条件与随机种子）
  - `repro/` — 一键复现环境与四道守卫（`run_guards.py`）
  - `audit/SHA256SUMS.txt` — 与本文档同名文件的仓库版
  - `docs/REVISIONS.md` — 构建期修订与 QA 台账（非论文内容，供审计追溯）

**校验方式**：本版本的 `SHA256SUMS.txt` 与该仓库的 `audit/SHA256SUMS.txt` **内容一致**，可直接用于核对 clone 下来的交付文件是否与发布版逐字节相同：

```bash
python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" <file>
```

---

## 如何复现

```powershell
git clone https://github.com/hallway1025/hcm-memory-model
cd hcm-memory-model

# L0 全套基准（单线程 CPU 笔记本约 3 分钟）
py -3 bench\exp1_separation.py
py -3 bench\exp2c_capacity_edge.py
py -3 bench\exp6_continual.py
py -3 bench\exp6b_replay_buffer.py
py -3 bench\exp7_wm_capacity.py
py -3 bench\exp8_meta.py

# 四道守卫（论文全部定量主张、引文、闭合性、校验和）
py -3 repro\run_guards.py
```

一键容器复现（Docker）：见仓库 `repro/` 目录，tag `hcm-v14`。

---

## 版本要点（v1.4 相对 v1.3）

1. **优先工作定位与边界对照**：§2 补引 Context Poisoning as Extreme-Value Attention Interference（arXiv 2609.22101，公开早于本稿三个月）——同一现象域、同一数学内核、标度律同族（Ω(√log N) 下界 vs 本研究实测 log N）。本稿贡献如实定位为**补偿侧首次实测 + 记忆库场景系统化 + 边界标注**；附录 E 增补实测上侧锚点与下界不矛盾的对照段。
2. **G-R3b′ 密扫（预登记判负，如实报告）**：o=0.2、N=2×10³、×√2 步长密扫 β，实测 knee·gap = 1.0–1.4，全部 < ln N ≈ 7.6（差 5.4–7.6 倍）——修正存活条件不成立，未升级受控验证，负结果原样入稿（附录 G/K）。
3. **引文库全量回填与台账重生成**：448 条记录 verified/tier 字段补齐（T1=258、T2=98、T3=92，T1+T2=79.5%），module evidence 与 S4 台账按新库重生成。
4. **全仓过时守卫数字修正**：三语稿件、README、审计报告、复现文档中的旧读数（444/447/四通道/69 条闭包）统一修正为实测值。
5. **EN 标题页渲染修复**：v1.3 的英文主稿因源文件标题行缺 `#` 前缀，PDF 首页标题被渲染成小号左对齐段落而非居中大字标题；v1.4 已修复为正常标题页。四份 PDF（CN 52 页 / EN 66 页 / ARX 46 页 / S4 32 页）重渲并复验 /Title；ARX HTML 导出器硬编码占位串修复为从稿首标题取值。
6. **守卫体系**：四道守卫全绿（G1 数字溯源 457/457、G2 引文核验 n=448、T1+T2=79.5%、G3 引文闭包 57/57、G4 校验和 681/0）。

---

## 重要边界（务必先读）

- **全部核心实验基于合成数据**；真实语料结果限于单个语料家族的描述性对照，向图像/语言/真实情景记忆的外推未经验证。
- **L2/L3 规模档是设计协议而非实测**。L0/L1 已实测；L2/L3 给出资源估算与验收协议。
- **没有独立第三方复现**。对抗评审与两轮洁净复现均由同一技术栈内的自动化代理执行，属内部 QA。
- **弃权门的适用范围收窄**：在公开同质语料上失效，定位为分布移位检测器而非库成员资格检测器。
- 详见论文 §6「局限与诚实声明」。

---

## 许可与引用

- 论文与补充材料：随 Zenodo 记录所附许可。
- 代码：见仓库 `LICENSE`。
- 如需引用，请引用本 Zenodo 记录的 DOI（概念 DOI 固定，正文版本引用具体版本 DOI）。
