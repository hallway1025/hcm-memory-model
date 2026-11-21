# HCM v1.3 — Release README

**HCM（Hippocampal–Cortical Complementary Memory）**：一个九算子的计算记忆模型，带可执行、可种子复现的 NumPy 参考实现与系统化基准。

---

## 这个版本包含什么

| 文件 | 内容 |
|---|---|
| `HCM_PAPER.pdf` | 中文主稿（权威版，51+ 页） |
| `HCM_PAPER_EN.pdf` | 英文主稿（对应译本，56+ 页） |
| `SUPPLEMENTAL-S2.pdf` | 规模试验实测全叙事（L1 更正 + L2 @N=10⁵ 的 exp37）与五个待超算回答的问题 |
| `SUPPLEMENTAL-S4.pdf` | 引文核验完整台账（447 条，逐条含 DOI/PMID、核验通道与检索时间戳） |
| `SHA256SUMS.txt` | 本版本全部交付文件的 SHA256 校验和（882 条） |

---

## 代码与结果在哪里

论文的全部实验代码、结果 JSON、图表脚本与复现环境**随仓库分发**：

- **GitHub**：`https://github.com/hallway1025/hcm-memory-model`
- 仓库内主要目录：
  - `src/hcm/` — 模型参考实现（12 个模块，纯 NumPy）
  - `bench/` — 实验脚本（exp1–exp52）
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

一键容器复现（Docker）：见仓库 `repro/` 目录，tag `hcm-v13`。

---

## 版本要点（v1.3 相对 v1.2）

1. **G-R3b 机制附录**：把 v1.2 如实报告的 G-R3 FAIL 从"失败"升级为"理解失败机制"——解析下界 overshoot 的三机制解读（典型 vs 最坏竞争、双读出饱和膝点、网格偏差），以及 log-N 方向在未饱和处的一致性核查。
2. **Stage 0 真实记忆交叉核对**：对一处真实智能体记忆的检索统计做描述性对照（单日快照，132 条池、中位 43 天、8–30 天档零命中），作为饱和膝点的定性信号；明确限定为量级参考，跨年外推留待 Stage 1。
3. **G-R3b 精确披露**：合成阶梯 30 格上 pooled Spearman **ρ = 0.838**（预登记门 ≥ 0.9，差 0.062），按预登记记实报告。
4. **Section 9 重组**：正文只保留 P0 级问题（直接动摇结论者）与方法学总结；构建期工程记录归入仓库 `docs/REVISIONS.md`，不进发表件。
5. **匿名化**：全部真实会话标识替换为中性代号（peer-01/02、group-01…05），贯穿仓库、校验和清单与三语稿件。
6. **守卫体系**：四道守卫全绿（G1 数字溯源 444/444、G2 引文核验 n=447、T1+T2=79.6%、G3 引文闭包 56/56、G4 校验和 677/0）。

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
