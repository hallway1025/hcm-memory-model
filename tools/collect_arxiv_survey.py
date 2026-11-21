#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_arxiv.py — 调查库 arXiv 通道采集（export.arxiv.org Atom API，无 key）

为 HCM 薄弱环节调查服务（refs/survey/）。与 collect.py --out 调查模式配套：
  - 同一 JSONL schema；channel="arxiv"、arxiv_id、abstract、primary_category
  - 无 DOI 留空（arXiv 元数据偶带 <arxiv:doi> 则保留，便于与 Crossref 记录合流）
  - ref_id 沿用调查库 S 前缀，从 --out 现有最大号续（先跑 collect.py 再跑本脚本）
  - 去重：doi / arxiv_id / 归一化标题[:120]（跨通道同文以先入为准，优先 Crossref 版）
  - 限速：相邻请求间隔 >= --sleep 秒（默认 3.2，官方要求 >=3s）
  - 检索式：AND-of-words（召回导向，relevance 排序，再按标题相似度 --min-sim 过滤）
  - year_from：转成 submittedDate 区间过滤（J 区 2023+ 用）

用法：
  python tools/collect_arxiv.py --queries refs/survey/queries_survey.jsonl ^
      --out refs/survey/refs_survey.jsonl --min-sim 0.35 [--only A_timescale] [--dry]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from urllib.parse import urlencode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_citations as vc

ROOT = vc.ROOT
QUERIES = os.path.join(ROOT, "refs", "survey", "queries.jsonl")
OUT_DEFAULT = os.path.join(ROOT, "refs", "survey", "refs_survey.jsonl")

NS = {"a": "http://www.w3.org/2005/Atom", "ax": "http://arxiv.org/schemas/atom"}
API = "http://export.arxiv.org/api/query"


def arxiv_search(query: str, n: int, year_from: int | None, sleep: float) -> list[dict]:
    terms = " AND ".join("all:" + w for w in query.split())
    if year_from:
        terms += f" AND submittedDate:[{year_from}01010000 TO 202612312359]"
    url = API + "?" + urlencode({"search_query": terms, "start": 0,
                                 "max_results": n, "sortBy": "relevance"})
    xml = vc.http_get(url)          # 统一 HTTP 层：refs/cache 落盘 + 3 次重试
    time.sleep(sleep)               # arXiv 官方限速 >=3s/请求
    feed = ET.fromstring(xml)
    out = []
    for e in feed.findall("a:entry", NS):
        title = " ".join((e.findtext("a:title", "", NS) or "").split())
        authors = [a.findtext("a:name", "", NS) or "" for a in e.findall("a:author", NS)]
        pub = e.findtext("a:published", "", NS) or ""
        year = int(pub[:4]) if pub[:4].isdigit() else None
        summary = " ".join((e.findtext("a:summary", "", NS) or "").split())
        raw = e.findtext("a:id", "", NS) or ""
        aid = re.sub(r"v\d+$", "", raw.rsplit("/abs/", 1)[-1])
        cat_el = e.find("ax:primary_category", NS)
        out.append({
            "arxiv_id": aid, "title": title, "authors": authors, "year": year,
            "abstract": summary,
            "primary_category": cat_el.get("term") if cat_el is not None else "",
            "doi": (e.findtext("ax:doi", "", NS) or "").strip(),
            "journal_ref": (e.findtext("ax:journal_ref", "", NS) or "").strip(),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=QUERIES)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--only", default="", help="按 area 过滤（只跑一个区）")
    ap.add_argument("--min-sim", dest="min_sim", type=float, default=0.35)
    ap.add_argument("--sleep", type=float, default=3.2, help="相邻请求间隔秒（>=3）")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    qpath = args.queries if os.path.isabs(args.queries) else os.path.join(ROOT, args.queries)
    opath = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    if not os.path.exists(qpath):
        print(f"[FAIL] missing {qpath}")
        return 2
    tiers = vc.load_tiers()
    qs = [json.loads(l) for l in open(qpath, encoding="utf-8") if l.strip() and not l.startswith("#")]
    qs = [q for q in qs if q.get("channel") == "arxiv"
          and (not args.only or q.get("area") == args.only)]

    existing = vc.load_refs(opath)
    have_doi = {(r.get("doi") or "").lower() for r in existing if r.get("doi")}
    have_aid = {(r.get("arxiv_id") or "").lower() for r in existing if r.get("arxiv_id")}
    have_title = {vc.norm(r.get("title", ""))[:120] for r in existing if r.get("title")}
    nextid = 0
    for r in existing:
        rid = r.get("ref_id", "")
        if len(rid) > 1 and rid[0].isalpha() and rid[1:].isdigit():
            nextid = max(nextid, int(rid[1:]))
    nextid += 1

    added, log = [], []
    for q in qs:
        area, query, n = q["area"], q["query"], int(q.get("n", 15))
        try:
            hits = arxiv_search(query, n, q.get("year_from"), args.sleep)
        except Exception as e:
            log.append(f"- [{area}] [{query}] -> ERROR {e}")
            print(f"[ERR] {query}: {e}", file=sys.stderr)
            continue
        kept = 0
        for h in hits:
            sim = vc.title_similarity(query, h["title"])
            if sim < args.min_sim:
                continue
            doi = (h.get("doi") or "").lower()
            aid = (h.get("arxiv_id") or "").lower()
            tkey = vc.norm(h["title"])[:120]
            if (doi and doi in have_doi) or (aid and aid in have_aid) or tkey in have_title:
                continue
            rec = {
                "ref_id": f"S{nextid:03d}", "module": area, "role": "survey",
                "channel": "arxiv", "arxiv_id": h["arxiv_id"], "doi": doi,
                "title": h["title"], "journal": "arXiv preprint",
                "journal_ref": h.get("journal_ref", ""),
                "year": h.get("year"),
                "first_author": (h["authors"][0].split()[-1] if h["authors"] else ""),
                "n_authors": len(h["authors"]),
                "source_query": query, "resolved_by": "arxiv_api",
                "retrieved_at": vc.now_iso(), "sim_to_query": round(sim, 4),
                "abstract": (h.get("abstract") or "")[:1200],
                "primary_category": h.get("primary_category", ""),
            }
            rec["tier"] = vc.tier_of(rec["journal"], tiers)
            rec["evidence_level"] = "abstract" if rec.get("abstract") else "unread"
            added.append(rec)
            if doi:
                have_doi.add(doi)
            have_aid.add(aid)
            have_title.add(tkey)
            nextid += 1
            kept += 1
        log.append(f"- [{area}] [{query}] -> raw {len(hits)}, kept {kept}")
        print(f"[OK] {query}: raw {len(hits)} kept {kept}")

    if not args.dry and added:
        with open(opath, "a", encoding="utf-8") as f:
            for r in added:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[{'DRY' if args.dry else 'OK'}] +{len(added)} arxiv refs -> {opath}")
    print("\n".join(log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
