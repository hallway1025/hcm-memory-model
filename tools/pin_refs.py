#!/usr/bin/env python3
"""pin_refs.py —— 锚点文献定向登记（Pin）。

问题：Crossref 的 query.bibliographic 排序器对少数经典论文返回不到（实测
"Overcoming catastrophic forgetting in neural networks" 只回 arXiv 与相似标题）。
解决：对这些"论文必须引用、但检索器找不到"的锚点，用 OpenAlex 标题检索取回
DOI（DOI 仍由 API 产出，绝不手写），再用 Crossref DOI 查得规范元数据。
流程全部留痕在 refs/pin-log.md，可审计。

用法：
    py -3 tools/pin_refs.py --file refs/pins.jsonl [--dry]
输入每行：{"title": "...", "year": 2017, "module": "M5_SEM", "role": "mechanism"}
可选字段：journal（要求刊名包含该子串）、note。
"""
import json, os, sys, argparse, urllib.parse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_citations as vc
import requests

ROOT = vc.ROOT
REFS = os.path.join(ROOT, "refs", "refs.jsonl")
LOG = os.path.join(ROOT, "refs", "pin-log.md")


def openalex_title(title: str, n: int = 10) -> list[dict]:
    url = ("https://api.openalex.org/works?search=" + urllib.parse.quote(title)
           + "&per-page=" + str(n)
           + "&select=id,doi,title,publication_year,type,primary_location,authorships")
    try:
        r = requests.get(url, timeout=40, headers={"User-Agent": "hcm-pin/1.0 (mailto:none@example.org)"})
        return (r.json() or {}).get("results", [])
    except Exception as e:
        print("  [warn] openalex error", e)
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=os.path.join(ROOT, "refs", "pins.jsonl"))
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    tiers = vc.load_tiers()
    recs = vc.load_refs(REFS)
    have = {(r.get("doi") or "").lower() for r in recs if r.get("doi")}
    maxid = 0
    for r in recs:
        rid = r.get("ref_id") or ""
        if rid.startswith("R") and rid[1:].isdigit():
            maxid = max(maxid, int(rid[1:]))

    pins = [json.loads(l) for l in open(args.file, encoding="utf-8") if l.strip() and not l.startswith("#")]
    log = ["", f"## pin run {vc.now_iso()}", ""]
    added, skipped, failed = [], [], []

    for p in pins:
        title, year = p["title"], int(p["year"])
        cands = openalex_title(title)
        scored = []
        for c in cands:
            doi = (c.get("doi") or "").replace("https://doi.org/", "")
            if not doi:
                continue
            if c.get("publication_year") != year:
                continue
            if (c.get("type") or "") not in ("article", "review", "book-chapter"):
                continue
            sim = vc.title_similarity(title, c.get("title") or "")
            if sim < 0.95:
                continue
            src = ((c.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
            scored.append((sim, doi, src))
        scored.sort(reverse=True)
        if not scored:
            failed.append(title)
            log.append(f"- [FAIL] {title} ({year}) <- OpenAlex 无 >=0.95 同年命中")
            print("  [FAIL]", title)
            continue
        sim, doi, src = scored[0]
        if doi in have:
            skipped.append(title)
            log.append(f"- [dup] {title} -> {doi}")
            continue
        meta = vc.crossref_by_doi(doi)
        if not meta:
            failed.append(title)
            log.append(f"- [FAIL] {title} -> {doi} Crossref 取不到元数据")
            continue
        journal = meta.get("journal") or src
        tier = vc.tier_of(journal, tiers)
        if p.get("journal") and p["journal"].lower() not in journal.lower():
            failed.append(title)
            log.append(f"- [FAIL] {title} -> 刊名不符：得到 {journal}，期望含 {p['journal']}")
            continue
        if tier in ("REJECT", "unknown"):
            failed.append(title)
            log.append(f"- [FAIL] {title} -> 分级 {tier}（刊：{journal}），需先登记期刊")
            continue
        maxid += 1
        rid = f"R{maxid:03d}"
        rec = {
            "ref_id": rid, "module": p.get("module", ""), "role": p.get("role", ""),
            "doi": meta["doi"], "pmid": None, "pmcid": None,
            "title": meta.get("title") or title,
            "authors": meta.get("first_author") or "",
            "first_author": meta.get("first_author") or "",
            "year": meta.get("year") or year,
            "journal": journal,
            "tier": tier,
            "verified": None, "title_similarity": round(sim, 3),
            "evidence_level": p.get("evidence_level", "abstract"),
            "retracted": False,
            "source_query": "PIN: " + title,
            "resolved_by": "openalex->crossref",
            "retrieved_at": vc.now_iso(),
            "claims": [], "notes": p.get("note", ""),
        }
        added.append(rec)
        have.add(doi)
        log.append(f"- [OK] {rid} {tier} {meta.get('year')} {journal} | {rec['title'][:80]}")
        print(f"  [OK] {rid} {tier} {journal[:40]} | {rec['title'][:60]}")
        if not args.dry:
            with open(REFS, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            vc.http_get  # keep import used

    log.append("")
    log.append(f"added={len(added)} dup={len(skipped)} failed={len(failed)}")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    print(f"added={len(added)} dup={len(skipped)} failed={len(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
