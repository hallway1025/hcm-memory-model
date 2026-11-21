# -*- coding: utf-8 -*-
import io, json
NL = '\r\n'
def block(s):
    return s.replace('\n', NL)

j = json.load(io.open('results/exp25_collapse_external.json', encoding='utf-8'))
clus = {d['N']: d for d in j['dense_cluster']}
den = {d['N']: d for d in j['dense']}
sp = {d['N']: d for d in j['sparse']}
def hits(d, N, b):
    r = d[N]
    return r['hits'][r['betas'].index(b)]

P = 'paper/PAPER.md'
t = io.open(P, encoding='utf-8').read()

# --- appendix F after appendix E; find the appendix E end = next '## 附录' or '---' ... append before '## 参考文献'? locate appendix E heading then find its table end by next top-level heading
iE = t.index('## 附录 E')
nxt = len(t)  # 附录 E 是文末最后一节
appF = '''

## 附录 F：外部复现——坍缩律不依赖 HCM 实现（exp25）

为零代码依赖地检验「检索对比度坍缩」是否是 HCM 实现的伪影，我们在 vanilla 软检索（现代 Hopfield / Transformer cross-attention 同构：a = softmax(β·Kq)，检索模式 r = unit(a@K)，分类 = argmax(K r)）上重演同一现象。三种键几何 × N∈{10^3, 10^4, 10^5, 10^6}（N=10^6 时 50 探针，其余 200）：

| 键几何 | β*(N) 随 N | β=2 时 hit@1 随 N |
|---|---|---|
| iid 高斯键（无共享结构） | 2, 2, 2, 2（无坍缩） | 98%, 100%, 100%, 100% |
| 二值 k-稀疏码（超几何原子） | 2, 2, 2, 2（近免疫） | 89%, 100%, 88%, 88% |
| 16 簇共享质心键（真实记忆共享要旨的还原） | **2, 5, 5, 5** | **73%, 8%, 0%, 4%** |

三个结论：(i) **坍缩需要候选共享结构**——无结构键的一步软检索对 β 完全鲁棒（其他候选的平均 ≈ 0，噪声自消）；(ii) 一旦候选共享要旨（聚簇几何），固定 β 的坍缩随 N 加重（β=2：73%→0%），β*(N) 从 2 升到 ≥5 并在 10^6 处保持——**增益必须随库规模增长**这一律在外部系统中成立；(iii) 离散原子码（附录 E 的模型）在分数间隙上近免疫，说明 HCM 实测坍缩（§4.12）来自「语义先验融合 + 近键」的组合，而不是 softmax 本身——这解释了为什么 λ 通道（exp19b）是 L1 崩塌的第一现场。全部数字：results/exp25_collapse_external.json（零 HCM 代码，bench/exp25_collapse_external.py 独立可跑）。

'''
t = t[:nxt] + block(appF) + t[nxt+1:]

# --- §4.12 pointer sentence ---
anchor = 'β 需随 log N 增长'
i = t.find(anchor)
print('4.12 anchor found:', i >= 0)
if i >= 0:
    # append a pointer sentence after the paragraph containing it: insert after the end of that line
    jline = t.index(NL, i)
    add = '（该律已在一个零 HCM 代码的 vanilla 注意力软检索中外部复现：聚簇键几何下 β*(N) 随 N 从 2 升至 ≥5、固定 β=2 的命中从 73% 塌到 0%；无结构键则完全鲁棒——见附录 F）'
    t = t[:jline] + block(add) + t[jline:]

io.open(P, 'w', encoding='utf-8', newline='').write(t)
print('appendix F inserted, len', len(t))
