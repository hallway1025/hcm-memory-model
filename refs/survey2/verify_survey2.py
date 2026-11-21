#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify only the survey records cited in notes/lit/SURVEY-ALTS.md.

Reads the report, extracts every citation token — ref_id (S+digits) and bare
arXiv id (NNNN.NNNNN) — verifies exactly those records from
refs/survey2/refs_survey2.jsonl (T) + refs/survey/refs_survey.jsonl (S) via tools/verify_citations.verify_record, and
writes refs/survey/verify-report2.md + verify-report2.json.

Bare arXiv ids that do NOT exist in the survey library are listed as warnings
(they can legitimately appear inside web URLs in the 近期动态 sections); they
are never counted as MISSING. Library citations (S-ids and in-lib arxiv ids)
that fail verification are fatal for the report.

Usage: python refs/survey/verify_survey.py [--report notes/lit/SURVEY-ALTS.md]
"""
import argparse, json, os, re, sys, io
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import verify_citations as vc  # noqa: E402

REF_RE = re.compile(r"\b[ST]\d{3,}\b")
ARXIV_RE = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=os.path.join(ROOT, "notes", "lit", "SURVEY-WEAKPOINTS.md"))
    ap.add_argument("--out", default=os.path.join(ROOT, "refs", "survey2"))
    args = ap.parse_args()
    text = open(args.report, encoding="utf-8").read()
    cited = sorted(set(REF_RE.findall(text)))
    arxiv_cited = sorted(set(ARXIV_RE.findall(text)))
    refs, lib_arxiv = {}, {}
    # dual-library: T-ids from survey2, S-ids (cross-refs) from survey-1
    for sub in (("refs", "survey2", "refs_survey2.jsonl"),
                ("refs", "survey", "refs_survey.jsonl")):
        path = os.path.join(ROOT, *sub)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                refs[rec.get("ref_id")] = rec
                if rec.get("arxiv_id"):
                    lib_arxiv[rec["arxiv_id"]] = rec
    tiers = vc.load_tiers()
    results = []
    missing = [r for r in cited if r not in refs]
    for rid in cited:
        rec = refs.get(rid)
        if rec is None:
            results.append(dict(ref_id=rid, verified="MISSING",
                                fail_reason="ref_id not in refs_survey.jsonl"))
            continue
        # arXiv-only records: role=system and no doi/pmid routes to the arXiv API check
        probe = dict(rec)
        if probe.get("arxiv_id") and not probe.get("doi") and not probe.get("pmid"):
            probe["role"] = "system"
        results.append(vc.verify_record(probe, tiers))
    # bare arXiv ids that exist in the survey library get verified too
    n_arxiv_verified = 0
    for aid in arxiv_cited:
        rec = lib_arxiv.get(aid)
        if rec is None:
            continue  # web-URL-only id -> warning list, not MISSING
        probe = dict(rec)
        if probe.get("arxiv_id") and not probe.get("doi") and not probe.get("pmid"):
            probe["role"] = "system"
        r = vc.verify_record(probe, tiers)
        r["ref_id"] = r.get("ref_id") or ("arxiv:" + aid)
        r["arxiv_id"] = aid
        results.append(r)
        n_arxiv_verified += 1
    arxiv_unlib = [a for a in arxiv_cited if a not in lib_arxiv]
    dist = Counter(r.get("verified") for r in results)
    n = len(results)
    lines = []
    lines.append("# Survey citation verification report")
    lines.append("")
    lines.append("- report: %s" % args.report)
    lines.append("- cited ref_ids: %d (+%d bare arxiv_id verified)" % (len(cited), n_arxiv_verified))
    lines.append("- verified: %d | partial: %d | failed: %d | MISSING: %d" % (
        dist.get("verified", 0), dist.get("partial", 0), dist.get("failed", 0), dist.get("MISSING", 0)))
    lines.append("- generated: %s" % vc.now_iso())
    lines.append("")
    lines.append("| ref_id | verified | sim | tier | year | journal | fail_reason |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            r.get("ref_id"), r.get("verified"),
            ("%.2f" % r["title_similarity"]) if r.get("title_similarity") is not None else "",
            r.get("tier", ""), r.get("year", ""),
            str(r.get("journal", ""))[:40].replace("|", "/"),
            r.get("fail_reason", "")))
    md = "\n".join(lines) + "\n"
    out_md = os.path.join(args.out, "verify-report2.md")
    out_js = os.path.join(args.out, "verify-report2.json")
    open(out_md, "w", encoding="utf-8").write(md)
    json.dump(results, open(out_js, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("cited=%d arxiv_verified=%d total=%d verified=%d partial=%d failed=%d missing=%d" % (
        len(cited), n_arxiv_verified, n, dist.get("verified", 0), dist.get("partial", 0),
        dist.get("failed", 0), dist.get("MISSING", 0)))
    if missing:
        print("MISSING ids: " + ", ".join(missing))
    if arxiv_unlib:
        print("arxiv ids cited but not in survey lib (web links only): " + ", ".join(arxiv_unlib))
    print("wrote", out_md)


if __name__ == "__main__":
    main()
