# Supplemental Material S3: Engineering Feasibility and Analytic Theory (Appendix D+E full)

Full text of Appendix D (engineering feasibility, exp17) and Appendix E (analytic false-alarm bound, exp18), moved from the main manuscript during compression. Referenced from §4.12 and Appendix F.

## Appendix D: engineering feasibility — sparse storage and inverted index (exp17)

The dominant cost of retrieval is Mₙᵀv (a d_h × N dense mat-vec) and the Hopfield iteration's Mₙa. Since every column stores a **unit code with exactly k=20 non-zeros**, Mₙ can be rewritten losslessly as (row, value) pairs plus a unit-level inverted table (CSR): on the query side Mₙᵀv degenerates to a scatter over inverted buckets of v's non-zeros (bincount-vectorised); on the iteration side Mₙa is a per-column gather. Equivalence acceptance (3 seeds × N∈{2000, 16000}, 300 queries): maximum absolute difference over all mat-vecs and Hopfield soft assignments ≤ 2.2×10⁻⁶; top-1 argmax **100% identical** to the dense path.

| N | mat-vec dense/sparse (µs) | Hopfield L=3 dense/sparse (ms) | memory dense/sparse (MB) | ratio |
|---|---|---|---|---|
| 2000 | 137–211 / 162–322 | 31–57 / 27–65 | 8.2 / 0.8 | 10.1× |
| 16000 | 1672–1961 / 3247–3984 | 272–315 / 225–391 | 65.5 / 6.4 | 10.2× |

Honest conclusions: (i) **equivalence is free** — sparsification changes no retrieval semantics; (ii) the **memory gain is ~10×** (the implementation keeps columnar and inverted indices, 10.2×; inverted-only caps at ~17×, theoretical bound d_h/(2k)≈25.6×), which decides whether tiers L2/L3 (N≥10⁷) fit in memory at all; (iii) **on this machine's CPU the single dense BLAS mat-vec at N=16000 is still ~2× faster** (a streaming 65 MB beats random scatter on cache) — the sparse path's speed gains are expected in three places: memory-infeasible L2/L3 scales, inverted candidate pruning on genuinely correlated corpora, and GPU cusparse CSR kernels. Data: results/exp17_sparse_index.json; script bench/exp17_sparse_index.py (standalone; not in the main reproduction chain).

**Measured at the L2 tier (exp43, N=10⁵).** The exp37 write bottleneck localises in the duplication gate's full-library scan. Two opt-in engineering paths, both exact (retrieval agreement |Δa| ≤ 4×10⁻¹⁰), were measured on identical data: (i) an inverted-index write fastpath replaces the scan with posting-list intersection — 3.45 ms/item vs 0.680 s/item on the reference machine (197×) and 8.1 ms vs 0.410 s on the cloud replica (50×) — reducing write-phase memory from 6,250 MB (dense) to 512 MB + 372 MB CSR (7.1×); (ii) a float32 retrieval chain cuts canonical query time from ~159 s to 1.16 s at N=10⁵ (137×). The canonical semantic path is unchanged; the fastpath is an engineering layer, not a model change.



## Appendix E: theory — analytic false-alarm bound for k-sparse codes (exp18)

**Model**: for an out-of-store query code h and stored codes c_j, the support overlap S_j = |supp(c_j) ∩ supp(h)| follows a hypergeometric distribution under random codes, S ~ Hypergeom(d_h, k, k) (mean k²/d_h); for uniform codes the score is S/k; real encoder codes (truncated-Gaussian order statistics) are verified pointwise. The CDF of the **maximum false-alarm score** max_j ⟨c_j, h⟩ over an out-of-store query is F_S(x)^N (N independent codes).

**Validation** (bench/exp18_false_alarm.py; uniform codes, 6 configurations × encoder codes, 2 configurations):

| d_h | N | single-pair S distribution KS | max mean (empirical / analytic median) | max p99 (empirical / analytic) |
|---|---|---|---|---|
| 1024 | 2000 | 0.0140 | 0.187 / 0.20 | 0.25 / 0.25 |
| 1024 | 16000 | 0.0072 | 0.222 / 0.20 | 0.30 / 0.30 |
| 1024 | 200,000 | 0.0023 | 0.264 / 0.25 | 0.30 / 0.30 |
| 4096 | 2000 | 0.0030 | 0.112 / 0.10 | 0.15 / 0.15 |
| 4096 | 16000 | 0.0003 | 0.145 / 0.15 | 0.20 / 0.20 |
| 4096 | 200,000 | 0.0003 | 0.170 / 0.15 | 0.20 / 0.20 |

The real encoder path (orthogonal W_e, top-k truncated Gaussian): at d_h=1024 the out-of-store maximum false alarm is 0.192 vs in-store self-match 0.998 (**margin 0.81**); at d_h=4096, 0.120 / 0.999 (**margin 0.88**). Under the retrieval softmax (β=20) these margins correspond to posterior ratios of several orders of magnitude.

**Capacity implication**: the union bound N·P(S ≥ s\*) ≪ 1 quantifies "retrieval confusion is not the bottleneck" — the library size at which a false alarm reaches half of self-match (S ≥ 10) is N\* ≈ 1.3×10¹¹ (d_h=1024) and 3.2×10¹⁶ (d_h=4096). The real capacity bottleneck is **denoising** (§4.3 RQ2): after cue degradation the self-match score falls toward the false-alarm band, rather than the band rising. Data: results/exp18_false_alarm.json.

