# LLM 记忆系统的工业/学术路线普查（related work 素材）

采集：2026-11-07，Lead。arXiv API 摘要核对，链接见各条。

## 六条路线
1. **MemGPT**（arXiv:2310.08560，Letta 前身）——OS 虚拟内存隐喻：主上下文/外部存储分层，LLM 自调用工具页入页出，中断控制流。路线=上下文工程。
2. **Generative Agents**（arXiv:2304.03442）——自然语言经验流全量存储；检索分=recency×importance×relevance（LLM 打分）；定期 LLM reflection 合成高层记忆。注意：其检索三因子与 M4 巩固优先级 p∝recency^a·surprise^b·reward^c 结构同型，但分值来自 LLM 直觉而非统计校准。
3. **Mem0 / Mem0g**（arXiv:2504.19413）——生产管线：LLM 抽取候选事实→向量近邻比对→LLM 决定 ADD/UPDATE/DELETE/NOOP→向量库（+图变体）。LoCoMo 评测自称超六类基线与全上下文。路线=LLM 当记忆 CRUD 管理器。
4. **Zep / Graphiti**（arXiv:2501.13956）——双时间轴（事件 t / 摄取 t_valid）动态知识图谱；边失效当"遗忘"；混合检索（语义+BM25+图）。DMR/LongMemEval。路线=时间符号图。
5. **HippoRAG**（arXiv:2405.14831，NeurIPS 2024）与 **HippoRAG 2**（arXiv:2502.14802，From RAG to Memory）——学术圈最近的生物启发对手：海马索引理论（Teyler & Discenna）→ LLM 抽 KG 当新皮层、KG 节点当索引、Personalized PageRank 当海马模式完成。HippoRAG2 走向 non-parametric continual learning 框架。**related work 必须正面处理；也是现成对照基线。**
6. **A-MEM**（arXiv:2502.12110）——Zettelkasten 卡片盒：每条记忆=结构化笔记（上下文/关键词/标签），动态链接，新记忆触发邻居笔记演化。

## 共同点（=我们的差异化坐标）
- LLM 是记忆处理器（抽取/打分/组织），存储是符号结构（串/KG/卡）
- 相似度全部来自冻结预训练 embedding；无一家从用户流学表征
- 巩固=LLM 摘要/反思（长期表征不变）；遗忘=显式删除/边失效（无校准）；无弃权概念
- 评测集中：LoCoMo、DMR、LongMemEval（可直接借用其协议与基线代码）

## HCM 定位
它们=符号管理层；HCM=动力学内核（门控写入/重放巩固/检索诱发遗忘/校准弃权全是测得的量）。
集成路径：任何一家的存储层可替换为 HCM 内核。对照臂：HippoRAG（生物启发 RAG 代表）。
评审利好：HippoRAG 以生物隐喻发 NeurIPS——说明该社区接受这类工作；我们要在"隐喻→动力学+校准"上拉开距离。
