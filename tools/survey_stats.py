#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""survey_stats.py -- per-query raw hit counts for the weakpoint survey report.

Recomputes every query's fetch URL exactly as the collectors did, hashes it with
verify_citations.cache_key, and reads the response from refs/cache (NO network).
Counts:
  crossref  -> message.total-results (raw Crossref hits for the bibliographic query)
  arxiv     -> opensearch:totalResults in the Atom feed
Funnels (collected/dedup/coarse/fine) come from refs/survey/screen-stats.json and
the collect logs; raw per-query counts are joined with the query text here.

Output: refs/survey/query-stats.json + stdout table.

Usage: python tools/survey_stats.py [--queries refs/survey/queries.jsonl ...]
"""
import argparse, json, os, re, sys, io
from urllib.parse import quote, urlencode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import verify_citations as vc  # noqa: E402


def crossref_url(q, n):
    q = quote(q.strip())
    return ("https://api.crossref.org/works?query.bibliographic=" + q +
            "&rows=" + str(n) + "&select=DOI,title,container-title,issued,author,type")


def arxiv_url(q, n, year_from):
    terms = " AND ".join("all:" + w for w in q.split())
    if year_from:
        terms += " AND submittedDate:[" + str(year_from) + "01010000 TO 202612312359]"
    return "http://export.arxiv.org/api/query?" + urlencode(
        {"search_query": terms, "start": 0, "max_results": n, "sortBy": "relevance"})


def cached(url):
    p = os.path.join(ROOT, "refs", "cache", vc.cache_key(url) + ".txt")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return f.read()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", nargs="+", default=[
        os.path.join(ROOT, "refs", "survey", "queries.jsonl"),
        os.path.join(ROOT, "refs", "survey", "queries_supp.jsonl")])
    ap.add_argument("--out", default=os.path.join(ROOT, "refs", "survey", "query-stats.json"))
    args = ap.parse_args()
    rows = []
    for qf in args.queries:
        with open(qf, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    out = []
    for r in rows:
        area = r.get("area", r.get("module", "?"))
        q = r["query"]
        ch = r.get("channel", "crossref")
        n = int(r.get("n", 10))
        yf = r.get("year_from")
        raw = None
        if ch == "crossref":
            url = crossref_url(q, max(n * 6, 14))
            txt = cached(url)
            if txt:
                try:
                    raw = json.loads(txt).get("message", {}).get("total-results")
                except Exception:
                    raw = None
        else:
            url = arxiv_url(q, n, yf)
            txt = cached(url)
            if txt:
                m = re.search(r'opensearch:totalResults[^>]*>\s*(\d+)', txt)
                raw = int(m.group(1)) if m else None
        out.append(dict(area=area, query=q, channel=ch, n=n, year_from=yf,
                        raw_hits=raw, cached=(txt is not None)))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for r in out:
        print("%-16s %-9s raw=%-8s cached=%-5s n=%-3d %s" % (
            r["area"], r["channel"], r["raw_hits"], r["cached"], r["n"], r["query"][:70]))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
