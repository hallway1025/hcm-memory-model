#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect.py — 批量文献采集（发现 + 富化 + 去重 + 分级），写入 refs/refs.jsonl

输入：refs/queries.jsonl，每行 {"query": "...", "module": "M2_EPI", "role": "mechanism", "n": 3}
输出：refs/refs.jsonl（追加，按 DOI 去重），refs/collect-log.md

硬规则：
  - 预印本 (REJECT) 与撤稿一律不写入。
  - 相似度 < min_sim 不写入。
  - ref_id 全局单调，不回收。

用法：
  python tools/collect.py                     # 处理全部未处理 query
  python tools/collect.py --only cross        # 只跑某个 module
  python tools/collect.py --dry               # 不写盘，只打印
"""
from __future__ import annotations
import json, os, sys, argparse
from html import unescape as _unescape
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_citations as vc

ROOT = vc.ROOT
QUERIES = os.path.join(ROOT, "refs", "queries.jsonl")
REFS = os.path.join(ROOT, "refs", "refs.jsonl")
LOG = os.path.join(ROOT, "refs", "collect-log.md")

def load_existing(path: str) -> tuple[list[dict], set[str], int]:
    recs = vc.load_refs(path)
    dois = {(r.get("doi") or "").lower() for r in recs if r.get("doi")}
    maxid = 0
    for r in recs:
        rid = r.get("ref_id", "")
        if len(rid) > 1 and rid[0].isalpha() and rid[1:].isdigit():
            maxid = max(maxid, int(rid[1:]))
    return recs, dois, maxid

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--queries", default=QUERIES,
                    help="检索式文件（每行一个 JSON）。默认 refs/queries.jsonl；"
                         "论文必需引文用 refs/queries_paper.jsonl。")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--min-sim", dest="min_sim", type=float, default=0.80)
    ap.add_argument("--out", default="",
                    help="调查模式：输出 JSONL 路径覆盖（如 refs/survey/refs_survey.jsonl）。"
                         "默认空 = refs/refs.jsonl（向后兼容）。设置后 ref_id 前缀改用 S、"
                         "去重与续号读该文件自身、REJECT 分级只记录不过滤、日志写 <out>.log。")
    ap.add_argument("--discover", action="store_true",
                    help="发现模式：主题检索式不是标题，故不按 sim 过滤；改为只收白名单期刊"
                         "并按 (分级, 相关度, 年份) 排序。详见 D-013。")
    args = ap.parse_args()

    OUT_PATH = args.out if args.out else REFS
    LOG_PATH = (os.path.splitext(OUT_PATH)[0] + ".log") if args.out else LOG
    ID_PREFIX = "S" if args.out else "R"
    QUERY_FILE = args.queries if os.path.isabs(args.queries) else os.path.join(ROOT, args.queries)
    if not os.path.exists(QUERY_FILE):
        print(f"[FAIL] missing {QUERY_FILE}")
        return 2
    tiers = vc.load_tiers()
    existing, dois, nextid = load_existing(OUT_PATH)
    qs = [json.loads(l) for l in open(QUERY_FILE, encoding="utf-8") if l.strip() and not l.startswith("#")]
    if args.only:
        qs = [q for q in qs if q.get("module") == args.only]

    TIER_RANK = {"T1": 0, "T2": 1, "T3": 2, "unknown": 3, "REJECT": 9}

    def rank_candidates(query: str, want: int, year_hint=None) -> list[dict]:
        """同题多版本（重印/再版/勘误）时：分级优 -> 年份早 -> 相似度高。
        这是被 Scoville & Milner 1957 误解析到 Neurocase 重印版所逼出来的规则。"""
        cands = vc.crossref_search(query, n=max(want * 6, 14))
        scored = []
        for c in cands:
            if not c.get("doi"):
                continue
            if c.get("type") and c["type"] not in ("journal-article", "proceedings-article", "book-chapter"):
                continue
            sim = vc.title_similarity(query, c.get("title", ""))
            if sim < (0.0 if args.discover else args.min_sim):
                continue
            year = c.get("year") or 9999
            if year_hint and abs(int(year) - int(year_hint)) > 2:
                continue
            tier = vc.tier_of(c.get("journal", ""), tiers)
            if tier == "REJECT":
                continue
            if args.discover and tier == "unknown":
                continue  # 发现模式只收白名单期刊，避免 unknown 污染门禁
            if args.discover:
                scored.append((TIER_RANK.get(tier, 3), -sim, int(year), c))
            else:
                scored.append((TIER_RANK.get(tier, 3), int(year), -sim, c))
        scored.sort(key=lambda t: (t[0], t[1], t[2]))
        seen, out = set(), []
        for _, _, _, c in scored:
            key = vc.norm(c.get("title", ""))[:120]
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
        return out

    added, skipped, log = [], [], []
    for q in qs:
        query, module = q["query"], q.get("module", "cross")
        role, n = q.get("role", "mechanism"), int(q.get("n", 3))
        cands = rank_candidates(query, n, q.get("year"))
        acc = 0
        for c in cands:
            doi = (c.get("doi") or "").lower()
            if not doi:
                continue
            if doi in dois:
                skipped.append((query, c.get("title", ""), "dup")); continue
            sim = vc.title_similarity(query, c.get("title", ""))
            if sim < (0.0 if args.discover else args.min_sim):
                skipped.append((query, c.get("title", ""), f"sim={sim:.2f}")); continue
            rec = {
                "ref_id": f"{ID_PREFIX}{nextid:03d}", "module": module, "role": role,
                "doi": doi, "title": _unescape(c.get("title", "")),
                "journal": _unescape(c.get("journal", "")), "year": c.get("year"),
                "first_author": _unescape(c.get("first_author", "")), "n_authors": c.get("n_authors", 0),
                "source_query": query, "resolved_by": "crossref",
                "retrieved_at": vc.now_iso(), "sim_to_query": sim,
            }
            rec["tier"] = vc.tier_of(rec["journal"], tiers)
            if rec["tier"] == "REJECT" and not args.out:
                skipped.append((query, rec["title"], "PREPRINT/REJECT")); continue
            for pid in vc.pubmed_esearch_doi(doi):
                pm = vc.pubmed_efetch(pid)
                if pm:
                    rec["pmid"] = pm.get("pmid")
                    rec["retracted"] = bool(pm.get("retracted"))
                    if not rec.get("journal"): rec["journal"] = pm.get("journal", "")
                    if not rec.get("year"): rec["year"] = pm.get("year")
                break
            ep = vc.europepmc_by_ids(doi=doi)
            if ep:
                rec["pmcid"] = ep.get("pmcid", "")
                rec["is_open_access"] = bool(ep.get("is_open_access"))
                rec["retracted"] = bool(rec.get("retracted")) or bool(ep.get("retracted"))
                rec["abstract"] = (ep.get("abstract") or "")[:1200]
                if not rec.get("journal"): rec["journal"] = ep.get("journal", "")
            if rec.get("retracted"):
                skipped.append((query, rec["title"], "RETRACTED")); continue
            rec["evidence_level"] = "abstract" if rec.get("abstract") else "unread"
            added.append(rec); dois.add(doi); nextid += 1; acc += 1
            if acc >= n:
                break
        log.append(f"- [{module}] 「{query}」 -> +{acc}")

    if not args.dry and added:
        # 写前重读：防并发/半写状态导致的 ref_id 撞号（R372/R406/R414 三次教训）
        used_ids = set()
        try:
            for _r in vc.load_refs(OUT_PATH):
                used_ids.add(_r.get("ref_id", ""))
        except Exception:
            used_ids = set()
        _max = 0
        for _u in used_ids:
            if len(_u) > 1 and _u[0].isalpha() and _u[1:].isdigit():
                _max = max(_max, int(_u[1:]))
        nextid = max(nextid, _max + 1)
        for _r in added:
            while _r["ref_id"] in used_ids:
                _r["ref_id"] = f"{ID_PREFIX}{nextid:03d}"
                nextid += 1
            used_ids.add(_r["ref_id"])
        with open(OUT_PATH, "a", encoding="utf-8") as f:
            for r in added:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n## {vc.now_iso()} batch (+{len(added)})\n" + "\n".join(log) + "\n")
    print(f"[{'DRY' if args.dry else 'OK'}] +{len(added)} refs, skipped {len(skipped)} -> {OUT_PATH}")
    for r in added:
        print(f"  {r['ref_id']} {r['tier']:<8} {r.get('year')} {str(r.get('journal'))[:34]:<34} "
              f"{str(r.get('title'))[:60]}")
    for s in skipped[:25]:
        print(f"  skip: {s[1][:62]} :: {s[2]}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
