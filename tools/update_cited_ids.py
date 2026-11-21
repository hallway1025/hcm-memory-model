# -*- coding: utf-8 -*-
"""update_cited_ids.py -- rebuild refs/paper-cited-ids.json from the paper's numbered References.

Maps each [n] entry in paper/PAPER_EN_final.md References to a refs.jsonl record by
normalized title containment (or DOI), so verify_citations.py can separate
curated-reject (uncited, informational) from cited-reject (gate FAIL).
Offline and deterministic; prints unmatched entries for manual review.
"""
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper", "PAPER_EN_final.md")
REFS = os.path.join(ROOT, "refs", "refs.jsonl")
OUT = os.path.join(ROOT, "refs", "paper-cited-ids.json")


def norm(t: str) -> str:
    s = (t or "").lower()
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main() -> int:
    text = open(PAPER, encoding="utf-8").read().replace("\r\n", "\n")
    lines = text.split("\n")
    i0 = next(i for i, l in enumerate(lines) if l.strip().lower().startswith("## references"))
    entries = {}
    cur = None
    for l in lines[i0 + 1:]:
        m = re.match(r"^\[(\d+)\]\s+(.*)", l.strip())
        if m:
            cur = int(m.group(1))
            entries[cur] = m.group(2)
        elif cur is not None and l.strip() and not l.strip().startswith("#"):
            entries[cur] += " " + l.strip()
    recs = [json.loads(x) for x in open(REFS, encoding="utf-8") if x.strip()]
    by_doi = {(r.get("doi") or "").lower(): r for r in recs if r.get("doi")}
    mapped, unmatched = {}, []
    for n in sorted(entries):
        body = entries[n]
        dm = re.search(r"doi:(10\.[^\s*]+)", body, re.I)
        rec = None
        if dm:
            rec = by_doi.get(dm.group(1).rstrip(".").lower())
        if rec is None:
            nb = norm(body)
            # longest bank-title contained in the entry wins (shorter titles like
            # "Sparse Distributed Memory" are substrings of longer ones like
            # "Attention Approximates Sparse Distributed Memory" -- first-match
            # would mis-bind the Bricken & Pehlevan entry to the Kanerva book).
            best_len = 0
            for r in recs:
                nt = norm(r.get("title") or "")
                if nt and nt in nb and len(nt) > best_len:
                    best_len = len(nt)
                    rec = r
        if rec:
            mapped[n] = rec["ref_id"]
        else:
            unmatched.append(n)
    cited = sorted(set(mapped.values()))
    out = {
        "generated_from": os.path.relpath(PAPER, ROOT),
        "n_entries": len(entries),
        "n_mapped": len(mapped),
        "n_unique_ref_ids": len(cited),
        "cited_ref_ids": cited,
        "entry_map": {str(k): v for k, v in sorted(mapped.items())},
        "unmatched_entries": unmatched,
    }
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"entries={len(entries)} mapped={len(mapped)} unique_ref_ids={len(cited)} unmatched={unmatched}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
