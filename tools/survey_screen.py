#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Survey screening: coarse title-level pass -> fine abstract-level pass.

Usage:
  python tools/survey_screen.py --in refs/survey/refs_survey.jsonl \
      --outdir refs/survey [--threshold 2]

Outputs (into --outdir):
  screen-stats.json          per-area pipeline counts (raw/dedup/coarse/fine)
  screen_coarse.tsv          id, area, coarse-keep, matched terms (title level)
  screen_fine.jsonl          records passing the fine screen (full records + _score)
  screen_fine.tsv            id, area, score, title (for manual review)
Per-area keyword rules live in AREA_RULES below. Recall-oriented: coarse keeps
anything remotely plausible; fine uses weighted core terms and reject terms.
"""
import argparse, json, os, re, sys, io
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ---- area keyword rules -------------------------------------------------
# "core": strong area terms (weight 2, matched in title+abstract)
# "aux":  supporting terms (weight 1)
# "rej":  false-positive terms (weight -2; title rejects are absolute in fine)
AREA_RULES = {
    "A_timescale": dict(
        core=["time cell", "temporal context", "multiple timescale", "multi-timescale",
              "timescale", "scale-free", "temporal abstraction", "temporal representation",
              "complementary learning", "Laplace", "temporal difference"],
        aux=["temporal", "hierarchy", "slowly changing", "context drift", "temporal credit",
             "time constants", "successor", "history"],
        rej=["memory alloy", "memory foam", "shape-memory", "DRAM", "cache", "chemical", "genomic"]),
    "B_forget": dict(
        core=["forgetting", "unlearning", "retrieval-induced", "catastrophic forgetting",
              "interference", "memory decay", "motivated forgetting"],
        aux=["decay", "washout", "extinction", "overwriting", "clearance", "pruning",
             "homeostatic", "synaptic scaling"],
        rej=["forget-me-not", "dementia trial", "questionnaire cohort"]),
    "C_neuro": dict(
        core=["eligibility trace", "three-factor", "neuromodulat", "dopamin", "acetylcholin",
              "cholinerg", "noradrenal", "norepinephr", "adrenerg", "locus coeruleus",
              "reward prediction error", "neuromodulatory"],
        aux=["VTA", "SNc", "GABA", "serotonin", "mAChR", "D1", "D2", "gating", "surprise"],
        rej=["dopant", "chromatograph", "fuel cell"]),
    "D_consolidation": dict(
        core=["consolidation", "sharp-wave ripple", "sleep", "slow-wave", "spindle",
              "replay", "reactivation", "Benna", "cascade", "synaptic consolidation",
              "systems consolidation"],
        aux=["NREM", "REM", "hippocampo", "neocortical", "stabilization", "cueing",
             "targeted memory", "wake", "plasticity"],
        rej=["cement", "soil consolidation", "data center"]),
    "E_wm": dict(
        core=["working memory", "theta", "gamma", "phase code", "phase coding",
              "oscillat", "persistent activity", "chunking", "population clock"],
        aux=["prefrontal", "attractor", "bump", "preload", "load-dependent", "n-back",
             "delay period", "slot"],
        rej=["working memory questionnaire", "schoolchildren", "rehab"]),
    "F_capacity": dict(
        core=["capacity", "scaling", "storage", "signal-to-noise"],
        aux=["Hopfield", "associative memory", "episodic", "attractor network",
             "number of memories", "information content", "bits", "retrieval quality"],
        rej=["bearing capacity", "heat capacity", "carrying capacity", "lung capacity"]),
    "G_structure": dict(
        core=["successor representation", "cognitive map", "Tolman", "transitive inference",
              "schema", "structural learning", "task structure", "relational memory",
              "graph", "compositional"],
        aux=["temporal-difference", "factoriz", "Laplacian eigenmap", "generalization",
             "abstract", "relational", "concept"],
        rej=["graphene", "graph coloring", "graphic design"]),
    "H_scale": dict(
        core=["sparse distributed memory", "locality-sensitive", "LSH", "approximate nearest",
              "nearest neighbor", "HNSW", "vector index", "semantic hashing", "Kanerva",
              "inverted index", "sublinear"],
        aux=["sparse", "hashing", "index", "retrieval speed", "ANN search", "quantization",
             "sketch"],
        rej=["sparse matrix solver", "traffic index"]),
    "I_confidence": dict(
        core=["confidence", "metamemory", "feeling of knowing", "judgment of learning",
              "Bayesian", "uncertainty", "calibration", "signal detection",
              "familiarity", "drift-diffusion"],
        aux=["readout", "decision", "evidence accumulation", "posterior", "Viterbi",
             "belief", "subjective"],
        rej=["confidence interval", "confidence interval only", "consumer confidence"]),
    "J_llm_memory": dict(
        core=["LLM", "large language model", "language model", "agent", "GPT",
              "retrieval-augmented", "RAG", "conversational", "dialogue", "MemGPT",
              "external memory", "prompt", "transformer"],
        aux=["assistant", "chat", "instruction", "fine-tun", "long context", "context window"],
        rej=["protein language model", "cell language model", "speech corpus"]),
    # ---- survey2 (K areas, 更优方案审计) ----
    "K1_sparse_encoder": dict(
        core=["winner-take-all", "winner take all", "k-winners", "sparse coding",
              "sparse distributed memory", "locality sensitive hashing",
              "locality-sensitive hashing", "product quantization", "sparse autoencoder",
              "top-k sparse", "hyperplane hashing", "sparse hashing"],
        aux=["sparsity", "expansion", "sparse representation", "lsh", "quantization",
             "dictionary learning", "codebook", "top-k", "ann search"],
        rej=["video coding", "jpeg", "genomic", "speech codec"]),
    "K2_episodic_store": dict(
        core=["hopfield", "kanerva", "differentiable neural computer", "neural turing machine",
              "memory network", "vector symbolic", "hyperdimensional", "product key",
              "associative memory", "episodic memory module", "content addressable"],
        aux=["attractor", "one-shot", "external memory", "holographic reduced", "binding",
             "cleanup memory", "capacity", "memorization"],
        rej=["quantum memory", "ion trap", "spin glass only"]),
    "K3_temporal_links": dict(
        core=["successor representation", "successor feature", "temporal knowledge graph",
              "link prediction", "temporal association", "forward association",
              "sequence replay", "temporal link", "transition structure"],
        aux=["next item", "cognitive map", "graph", "temporal context", "chain", "sequence"],
        rej=["blockchain", "supply chain", "chain of custody"]),
    "K4_consolidation_sched": dict(
        core=["elastic weight consolidation", "experience replay", "continual learning",
              "catastrophic forgetting", "generative replay", "complementary learning",
              "prioritized replay", "replay buffer", "synaptic intelligence",
              "lifelong learning", "rehearsal"],
        aux=["gradient episodic", "stability plasticity", "memory aware", "offline replay",
             "incremental learning", "sampling strategy"],
        rej=["construction", "data center", "oil reservoir"]),
    "K5_semantic_slow": dict(
        core=["fast weights", "parameter efficient", "low-rank adaptation", "low rank adaptation",
              "adapter module", "lora", "slow weights", "prompt tuning", "bit fit", "prefix tuning"],
        aux=["fine-tun", "hypernetwork", "incremental learning", "plasticity", "distillation",
             "weight update", "forgetting"],
        rej=["ethernet", "power adapter", "network adapter hardware", "wireless adapter"]),
    "K6_working_memory": dict(
        core=["working memory", "activity silent", "activity-silent", "pointer network",
              "theta gamma", "theta-gamma", "short-term memory", "spike frequency adaptation"],
        aux=["slot", "persistent activity", "bump attractor", "prefrontal", "refresh",
             "n-back", "load", "chunk"],
        rej=["DRAM", "cache memory", "shared memory gpu"]),
    "K7_meta_control": dict(
        core=["value of information", "metacognition", "metacognitive", "abstention",
              "selective prediction", "cost-sensitive", "information gain", "retrieval policy",
              "active inference", "information seeking"],
        aux=["confidence", "calibration", "uncertainty", "decision threshold", "bandit",
             "control policy", "regret"],
        rej=[]),
    "K8_event_layer": dict(
        core=["event knowledge graph", "event ontology", "episodic buffer", "event segmentation",
              "event schema", "event extraction", "temporal event", "event reasoning",
              "event stream"],
        aux=["script", "narrative", "temporal grounding", "event", "timeline", "log"],
        rej=["adverse event", "cardiac event", "extinction event", "life events survey"]),
    "K9_text_frontend": dict(
        core=["tf-idf", "tfidf", "n-gram", "ngram", "feature hashing", "hashing trick",
              "ppmi", "pointwise mutual information", "word embedding", "sentence embedding",
              "text encoder", "character-level"],
        aux=["vector space model", "bm25", "co-occurrence", "svd", "latent semantic",
             "distillation", "bag of words", "char-level"],
        rej=["idf oil well", "surveying instrument"]),
}

TOKEN = re.compile(r"[\w\-]+", re.UNICODE)

def norm_text(s):
    return (s or "").lower()

def match_terms(text, terms):
    hits = []
    for t in terms:
        if t.lower() in text:
            hits.append(t)
    return hits

def score_record(rec, area):
    rules = AREA_RULES.get(area, dict(core=[], aux=[], rej=[]))
    title = norm_text(rec.get("title"))
    abstract = norm_text(rec.get("abstract"))
    text = title + " . " + abstract
    score = 0.0
    t_core = match_terms(title, rules["core"])
    t_aux = match_terms(title, rules["aux"])
    t_rej = match_terms(title, rules["rej"])
    a_core = match_terms(abstract, rules["core"])
    a_aux = match_terms(abstract, rules["aux"])
    score += 3.0 * len(t_core) + 1.0 * len(t_aux)
    score += 2.0 * len(a_core) + 0.5 * len(a_aux)
    score -= 3.0 * len(t_rej)
    return score, {"t_core": t_core, "t_aux": t_aux, "t_rej": t_rej, "a_core": a_core}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="refs/survey/refs_survey.jsonl")
    ap.add_argument("--outdir", default="refs/survey")
    ap.add_argument("--threshold", type=float, default=3.0)
    args = ap.parse_args()
    recs = []
    with open(args.inp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    # coarse pass (title level)
    coarse_keep = {}
    fine_rows = []
    stats = defaultdict(lambda: dict(total=0, coarse=0, fine=0))
    with open(os.path.join(args.outdir, "screen_coarse.tsv"), "w", encoding="utf-8") as fc:
        fc.write("id\tarea\tcoarse_keep\ttitle_hits\n")
        for rec in recs:
            area = rec.get("area") or rec.get("module") or "?"
            stats[area]["total"] += 1
            title = norm_text(rec.get("title"))
            rules = AREA_RULES.get(area, dict(core=[], aux=[], rej=[]))
            hits = match_terms(title, rules["core"] + rules["aux"])
            keep = bool(hits)
            if keep:
                stats[area]["coarse"] += 1
                coarse_keep[rec.get("ref_id")] = True
            fc.write("%s\t%s\t%s\t%s\n" % (
                rec.get("ref_id"), area, "Y" if keep else "N",
                ";".join(hits)))
        # fine pass over coarse-kept records
    for rec in recs:
        rid = rec.get("ref_id")
        if not coarse_keep.get(rid):
            continue
        area = rec.get("area") or rec.get("module") or "?"
        sc, detail = score_record(rec, area)
        ok = (sc >= args.threshold) and not detail["t_rej"]
        if ok:
            stats[area]["fine"] += 1
            rec["_score"] = sc
            rec["_area"] = area
            fine_rows.append((sc, rec))
    fine_rows.sort(key=lambda x: (-x[0], x[1].get("ref_id", "")))
    with open(os.path.join(args.outdir, "screen_fine.jsonl"), "w", encoding="utf-8") as fj:
        for sc, rec in fine_rows:
            fj.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(os.path.join(args.outdir, "screen_fine.tsv"), "w", encoding="utf-8") as ft:
        ft.write("id\tarea\tscore\ttitle\n")
        for sc, rec in fine_rows:
            ft.write("%s\t%s\t%.1f\t%s\n" % (rec.get("ref_id"), rec.get("_area"), sc,
                                                 (rec.get("title") or "")[:150]))
    with open(os.path.join(args.outdir, "screen-stats.json"), "w", encoding="utf-8") as fsj:
        json.dump(dict(threshold=args.threshold, areas=dict(stats)), fsj, ensure_ascii=False, indent=1)
    print("coarse kept: %d / %d" % (sum(v["coarse"] for v in stats.values()), len(recs)))
    print("fine kept:   %d" % sum(v["fine"] for v in stats.values()))
    for a in sorted(stats):
        s = stats[a]
        print("  %-14s total=%-4d coarse=%-4d fine=%d" % (a, s["total"], s["coarse"], s["fine"]))

if __name__ == "__main__":
    main()
