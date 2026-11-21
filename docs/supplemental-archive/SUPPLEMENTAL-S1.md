# Supplemental Material S1: External Review and Replication Record

> This material was moved from the main manuscript appendix sequence to reduce length.
> All experimental evidence remains in the corresponding sections of the main text and in results/.

## Appendix H: External review and replication record


### 9.1 Independent adversarial reviews

**Review 1** (independent agent; read-only access to all code and results) found the priority/uniform comparison running character-identical code — the headline "priority is no better than uniform" was one experiment run twice, with the blocked arm trained on half the samples — plus further unfair-denominator, untrained-arm, wrong-constant, and suppressed-control defects, all fixed. Its deepest finding was directional: four independent inconsistencies all made the mechanism look cleaner than the measurements, and no listed-number gate catches that pattern (the numeric gates were green throughout). Rule adopted: **no number that has not been independently recomputed is quoted.**

**Review 2** (fourth agent, against the frozen deliverable; rating B−) found "residual harm" resting on single seeds — retested as exp14 (8 seeds × N∈{4,000, 8,000, 16,000}; at N=16,000 the gap widens to 27.8pp, bit-identical across two machines; the finding was later re-attributed to the β=20 artefact, §4.12) — and the priority test underpowered at 3 seeds — retested as exp15 (20-seed paired: uniform 13.85±2.66 vs priority 14.80±3.96, sign-flip p=0.32, dz=−0.24; restated as "no detectable difference" with the ~140-seed resolution limit published). The same review confirmed one-command reproduction 18/18 PASS, matching checksums, genuine sampled citations, and tables matching result files digit-for-digit. An unsourced r=0.222 was replaced by exp13's measurement (corr(m1, correct)=−0.070, AUC=0.458).

**Review 3** (code-level probe; read-only against the frozen deliverable) closed all four issues by pre-registered retest rather than re-reading: exp41 (M7 gating redundant with θ_dup — either gate or dedup suffices; reward selectivity unmeasurable), exp42 (forgetting benefit never demonstrated — importance pruning is bit-identical to FIFO under uniform reward), the θ protocol pinned as the MODEL.md §3.2a triplet, and the ablation matrix's M0 mirror row confirmed.

### 9.2 Independent clean replications

**Replicator 1** (third agent; documents + manifest only, **no source code**): verdict **partially reproducible; the headline contrast did not reproduce** — the most valuable criticism this project received. Matched: checksums, capacity (20/20 at N=16,000), dual readout (41/50 vs 50/50), no-attractor gain, L3 storage arithmetic, residual-arm magnitude. Not matched: the no-residual arm (27.17±0.47 vs paper 1.08±0.16). Root cause: four documentation defects, all fixed — "overlap 0.90" is a mixing weight, not input cosine (actual 0.986±0.004); the EMA formula was written backwards (code is 0.9μ+0.1x); the θ development set was unspecified (this choice alone swings code cosine 0.054→0.632); residual-measure semantics were inconsistent between sections. §4.11 is the answer. Adversarial review finds paper-vs-results inconsistencies; clean replication finds the gap between documentation and rebuildability.

**Replicator 2** (frozen deliverable, 2026-11): re-implemented the three L0 groups from the documents. Groups A (capacity) and C (dual readout) match throughout (60/60 at 1.000; 38.7/50 vs 39/50); group B (residual) is systematically high, and a differential diagnosis (bench/diag_theta_protocol.py, N=8,000, 3 seeds) localised the discrepancy to the **θ development protocol**: θ≡0 gives deg25 0.950/0.842, duty-cycle development gives 1.000/0.950, the default code-value development gives **0.000/0.000** — one knob spans the whole [0.000, 1.000] range. Lessons formalised as MODEL.md §3.2a: qualitative directions are invariant across converging protocols; absolute values are not comparable across protocols.

### 9.3 Revision record

Rounds 3–7 responded to internal diagnoses, code-level reviews, the journal-readiness assessment and external literature memos. Their dispositions are not repeated row-by-row: the evidence for each lives in the experiment section it points to.

| Round | Scope | Outcome |
|---|---|---|
| 3 | β scale law, real corpus, evaluation forensics | 9 dispositions; 5 re-attributed anomalies to instruments (β calibration, evaluation alignment, development protocol), 1 genuine mechanistic refutation (threshold-homeostatic develop()); evidence in §4.12–4.13 |
| 4 | temporal chains, abstention operating point, RIF causal test | exp27 vacuum → exp28 link mechanism; exp35b calibration (ECE 0.283→0.047, cost −55%; the G-4 FAIL recorded); exp34/34b placebo-controlled negatives and the suppression trilemma; evidence in §4.14–4.17 |
| 5 | code-level review response, external baselines, v2-corpus rerun, comprehensive audit | exp38 Hopfield/SDM baselines, exp40 public corpus, exp39 cross-encoder residual boundary; the §4.15 verdict flip recorded; v1-artefact numbers purged (§4.13); evidence in §4.19–4.21 |
| 6 | neighbouring-literature citations | refs bank +6 machine-verified entries; claim-scope narrowing only — no result changes |
| 7 | external-review response | title finalised (residual claim demoted from the headline; "Reproducible" added); core citations completed — no result changes |

### 9.4 Methodological summary

Across the revision record (§9.3), most dispositions re-attributed anomalies to instruments rather than mechanisms; the negative results gained verdict power from pre-registered criteria and control arms — without the off-arm placebo, exp34's data would have been read as "RIF makes memory stronger"; without exp27's 3/200-vs-0/200 signal, the link mechanism would never have been motivated. The code-level probe method (read-only import, bit-for-bit reproduction of the deliverable's numbers, hunting mechanisms rather than numeric contradictions) yields one more decidable evidence chain than document review: "INCONCLUSIVE" does not survive a functional-coverage analysis — the honest tightening is a pre-registered retest. **Instrument-first stepwise attribution precedes mechanistic explanation.**
