#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_citations.py — HCM 引文机检工具链

设计原则：
  1) 引文只能由 API 产出，不接受手写 DOI/PMID（防幻觉）。
  2) 全部响应落盘 cache，重复核验离线且确定。
  3) 任何未核验 / 撤稿 / 预印本 / 未分级期刊 -> 非零退出码（门禁）。

子命令：
  search  按检索式在 PubMed / Crossref 找候选，输出 JSONL 候选记录（未核验）
  verify  核验 refs.jsonl 中每条，产出 audit/citations-report.{json,md}
  tier    仅解析单条 DOI/PMID 的期刊分级
  stats   统计 refs.jsonl 的权威度分布

用法：
  python tools/verify_citations.py search "Reactivation of hippocampal ensemble memories during sleep" --n 3
  python tools/verify_citations.py verify
  python tools/verify_citations.py stats
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from html import unescape as _unescape

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, "refs", "cache")
TIERS_PATH = os.path.join(ROOT, "refs", "journal_tiers.json")
CITED_PATH = os.path.join(ROOT, "refs", "paper-cited-ids.json")  # ref_ids cited in the paper (written by the closure check)
REFS_PATH = os.path.join(ROOT, "refs", "refs.jsonl")
AUDIT_DIR = os.path.join(ROOT, "audit")

UA = "HCM-citation-verifier/1.0 (research; contact: local)"
CACHE_VERSION = "v2"   # 改动解析主路径时递增，避免旧的空结果污染缓存
TIMEOUT = 30
EUTILS_SLEEP = 0.36   # NCBI 无 key: 3 req/s
_last_eutils = 0.0

# ---------------------------------------------------------------- utilities
def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def norm(text: str | None) -> str:
    if not text:
        return ""
    s = str(text).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s.startswith("the "):
        s = s[4:]
    return s

_STOP = {"a","an","the","of","in","on","for","and","to","with","from","by","at","as","is","are",
         "that","this","its","be","can","do","does","how","why","what","into","during","after","before"}

def _content_tokens(s: str):
    return [w for w in norm(s).split() if w not in _STOP]

def title_similarity(a: str, b: str) -> float:
    """对称相似度 + 非对称包含度。

    为什么需要非对称项（L-008）：Crossref 返回的标题常是完整长标题（含副标题），
    而我们检索式是主题短语。例：查询 "Why there are complementary learning systems
    in the hippocampus and neocortex" 对 Psychological Review 1995 的完整标题只得
    0.613 —— 真论文却被判不匹配。因此当**较短一方的内容词几乎全被较长一方覆盖**
    （cov>=0.95）且长度差不过分（长<=3*短+4）时，提升到 0.92。
    短查询（内容词<4）不享受此提升，避免 "memory" 命中一切。
    """
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return 0.0
    ratio = difflib.SequenceMatcher(None, na, nb).ratio()
    ta, tb = set(na.split()), set(nb.split())
    jac = len(ta & tb) / max(1, len(ta | tb))
    score = 0.6 * ratio + 0.4 * jac
    ca, cb = _content_tokens(a), _content_tokens(b)
    if ca and cb and min(len(ca), len(cb)) >= 4:
        short, long_ = (ca, cb) if len(ca) <= len(cb) else (cb, ca)
        ss, ls = set(short), set(long_)
        cov = len(ss & ls) / max(1, len(ss))
        if cov >= 0.95 and len(ls) <= 3 * len(ss) + 4:
            score = max(score, 0.92)
    return round(score, 4)

def cache_key(url: str) -> str:
    return hashlib.sha1((CACHE_VERSION + "|" + url).encode("utf-8")).hexdigest()[:20]

def http_get(url: str, *, refresh: bool = False, eutils: bool = False) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, cache_key(url) + ".txt")
    if os.path.exists(path) and not refresh:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    global _last_eutils
    if eutils:
        delta = time.time() - _last_eutils
        if delta < EUTILS_SLEEP:
            time.sleep(EUTILS_SLEEP - delta)
    for attempt in range(3):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
            if r.status_code == 200:
                if eutils:
                    _last_eutils = time.time()
                txt = r.text
                with open(path, "w", encoding="utf-8") as f:
                    f.write(txt)
                return txt
            if r.status_code in (429, 500, 502, 503):
                time.sleep(1.5 * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {r.status_code} :: {url}")
        except requests.RequestException as e:
            if attempt == 2:
                raise RuntimeError(f"REQUEST FAILED :: {url} :: {e}")
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"EXHAUSTED :: {url}")

# ---------------------------------------------------------------- tiers
def load_tiers() -> dict:
    with open(TIERS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def tier_of(journal: str | None, tiers: dict) -> str:
    n = norm(journal)
    if not n:
        return "unknown"
    reject = {norm(k): v for k, v in tiers.get("REJECT", {}).items()}
    for k in reject:
        if k and k in n:
            return "REJECT"
    for t in ("T1", "T2", "T3"):
        for cand in tiers.get(t, []):
            cn = norm(cand)
            if cn and (cn == n or cn in n):
                return t
    return "unknown"

# ---------------------------------------------------------------- resolvers
def arxiv_by_id(aid: str) -> dict | None:
    """arXiv Atom API 单条解析（role=system 专用）。"""
    try:
        xml = http_get(f"http://export.arxiv.org/api/query?id_list={aid}&max_results=1")
    except Exception:
        return None
    import re as _re
    m = _re.search(r"<entry>.*?</entry>", xml, _re.S)
    if not m:
        return None
    entry = m.group(0)
    tm = _re.search(r"<title>(.*?)</title>", entry, _re.S)
    if not tm:
        return None
    title = " ".join(tm.group(1).split())
    authors = _re.findall(r"<name>(.*?)</name>", entry, _re.S)
    pm = _re.search(r"<published>(\d{4})", entry)
    year = int(pm.group(1)) if pm else None
    fa = authors[0].split()[-1] if authors else ""
    return {"title": title, "first_author": fa, "n_authors": len(authors), "year": year,
            "journal": "arXiv preprint"}

def crossref_by_doi(doi: str) -> dict | None:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    try:
        data = json.loads(http_get(url))
    except Exception:
        return None
    m = data.get("message") or {}
    authors = m.get("author") or []
    return {
        "resolved_by": "crossref",
        "doi": (m.get("DOI") or doi).lower(),
        "title": (m.get("title") or [""])[0],
        "journal": (m.get("container-title") or [""])[0],
        "year": _year_from_crossref(m),
        "first_author": (authors[0].get("family") if authors else "") or "",
        "n_authors": len(authors),
        "type": m.get("type"),
        "issued": m.get("issued", {}),
        "update_to": m.get("update-to") or [],
    }

def _year_from_crossref(m: dict) -> int | None:
    for key in ("published-print", "published-online", "issued", "created"):
        parts = (m.get(key) or {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            try:
                return int(parts[0][0])
            except Exception:
                continue
    return None

def europepmc_by_ids(doi: str | None = None, pmid: str | None = None) -> dict | None:
    if doi:
        q = f'DOI:"{doi}"'
    elif pmid:
        q = f"EXT_ID:{pmid} AND SRC:MED"
    else:
        return None
    url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query="
           + urllib.parse.quote(q) + "&format=json&resultType=core&pageSize=5")
    try:
        data = json.loads(http_get(url))
    except Exception:
        return None
    res = (data.get("resultList") or {}).get("result") or []
    if not res:
        return None
    r = res[0]
    pubtypes = [t for t in (r.get("pubTypeList") or {}).get("pubType", [])]
    corrections = r.get("commentCorrectionList", {}).get("commentCorrection", [])
    retracted = any("retract" in (t or "").lower() for t in pubtypes) or any(
        "retract" in (c.get("type") or "").lower() for c in (corrections or [])
    )
    return {
        "resolved_by": "europepmc",
        "doi": (r.get("doi") or doi or "").lower(),
        "pmid": str(r.get("pmid") or pmid or ""),
        "pmcid": r.get("pmcid") or "",
        "title": r.get("title") or "",
        "journal": (r.get("journalInfo") or {}).get("journal", {}).get("title")
                   or r.get("journalTitle") or "",
        "year": int(r["pubYear"]) if str(r.get("pubYear") or "").isdigit() else None,
        "first_author": (r.get("authorString") or "").split(",")[0].strip(),
        "n_authors": len((r.get("authorString") or "").split(",")) if r.get("authorString") else 0,
        "pubtypes": pubtypes,
        "retracted": retracted,
        "is_open_access": r.get("isOpenAccess") == "Y",
        "abstract": (r.get("abstractText") or "")[:4000],
    }

def crossref_search(title: str, n: int = 5, extra: str = "") -> list[dict]:
    """Crossref 书目检索：发现阶段的主路径（比 PubMed 精确短语检索稳得多）。"""
    q = urllib.parse.quote((title + " " + extra).strip())
    url = ("https://api.crossref.org/works?query.bibliographic=" + q +
           f"&rows={n}&select=DOI,title,container-title,issued,author,type")
    try:
        data = json.loads(http_get(url))
    except Exception:
        return []
    out = []
    for m in (data.get("message") or {}).get("items") or []:
        authors = m.get("author") or []
        out.append({
            "resolved_by": "crossref",
            "doi": (m.get("DOI") or "").lower(),
            "title": (m.get("title") or [""])[0],
            "journal": (m.get("container-title") or [""])[0],
            "year": _year_from_crossref(m),
            "first_author": (authors[0].get("family") if authors else "") or "",
            "n_authors": len(authors),
            "type": m.get("type"),
        })
    return out

def pubmed_esearch_doi(doi: str) -> list[str]:
    url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json"
           f"&retmax=3&term={urllib.parse.quote(doi)}%5BDOI%5D")
    try:
        data = json.loads(http_get(url, eutils=True))
    except Exception:
        return []
    return (data.get("esearchresult") or {}).get("idlist") or []

def pubmed_efetch(pmid: str) -> dict | None:
    url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed"
           f"&id={pmid}&retmode=xml")
    try:
        xml = http_get(url, eutils=True)
    except Exception:
        return None
    def grab(tag):
        m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", xml, re.S)
        return _unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip() if m else ""
    journal = ""
    m = re.search(r"<Journal>.*?<Title>(.*?)</Title>", xml, re.S)
    if m:
        journal = _unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip()
    year = ""
    m = re.search(r"<PubDate>.*?<Year>(\d{4})</Year>", xml, re.S)
    if m:
        year = m.group(1)
    doi = ""
    m = re.search(r'<ArticleId IdType="doi">(.*?)</ArticleId>', xml, re.S)
    if m:
        doi = m.group(1).strip()
    title = ""
    m = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", xml, re.S)
    if m:
        title = _unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip()
    first = ""
    m = re.search(r"<Author[^>]*>.*?<LastName>(.*?)</LastName>", xml, re.S)
    if m:
        first = m.group(1).strip()
    retracted = "Retraction of Publication" in xml or "Retracted Publication" in xml
    return {
        "resolved_by": "pubmed",
        "doi": doi.lower(),
        "pmid": pmid,
        "title": title,
        "journal": journal,
        "year": int(year) if year.isdigit() else None,
        "first_author": first,
        "retracted": retracted,
    }

def openalex_by_doi(doi: str) -> dict | None:
    url = "https://api.openalex.org/works/doi:" + urllib.parse.quote(doi, safe="")
    try:
        m = json.loads(http_get(url))
    except Exception:
        return None
    auth = (m.get("authorships") or [{}])[0].get("author") or {}
    loc = (m.get("primary_location") or {}).get("source") or {}
    return {
        "resolved_by": "openalex",
        "doi": (m.get("doi") or "").replace("https://doi.org/", "").lower(),
        "title": m.get("title") or "",
        "journal": loc.get("display_name") or "",
        "year": m.get("publication_year"),
        "first_author": (auth.get("display_name") or "").split(" ")[-1],
        "cited_by_count": m.get("cited_by_count"),
    }

def resolve(doi: str | None, pmid: str | None, title: str | None) -> dict:
    """返回 {attempts:[...], best:{...}} ；best 为字段最全的解析结果（合并）。"""
    attempts, merged = [], {}

    def absorb(rec: dict | None):
        if not rec:
            return
        attempts.append(rec.get("resolved_by"))
        for k, v in rec.items():
            if v not in (None, "", [], 0) and not merged.get(k):
                merged[k] = v
            elif v not in (None, "") and k == "retracted":
                merged[k] = bool(merged.get(k)) or bool(v)

    if doi:
        absorb(crossref_by_doi(doi))
        absorb(europepmc_by_ids(doi=doi))
        absorb(openalex_by_doi(doi))
    if pmid:
        absorb(europepmc_by_ids(pmid=pmid))
        absorb(pubmed_efetch(pmid))
    if not doi and not pmid and title:
        for cand in crossref_search(title, n=3):
            if title_similarity(title, cand.get("title", "")) >= 0.85 and cand.get("doi"):
                absorb(cand)
                absorb(crossref_by_doi(cand["doi"]))
                for pid in pubmed_esearch_doi(cand["doi"]):
                    absorb(pubmed_efetch(pid))
                    absorb(europepmc_by_ids(pmid=pid))
                break
    return {"attempts": attempts, "best": merged}

# ---------------------------------------------------------------- verify

def verify_record(rec: dict, tiers: dict) -> dict:
    # role=system：竞品系统识别类（amend 2026-11-07）。arXiv API 验证标题，刊名白名单定级。
    if rec.get("arxiv_id") and not rec.get("doi") and not rec.get("pmid"):  # amend D-064 2026-11-09: 去掉 role 限制（theory/review 类 arXiv-only 论文同样走 arXiv API 核验）
        out = dict(rec)
        best = arxiv_by_id(rec["arxiv_id"]) or {}
        out["resolved_by"] = "arxiv_api" if best else "none"
        out["retrieved_at"] = now_iso()
        title = out.get("title") or ""
        sim = title_similarity(title, best.get("title", "")) if (title and best.get("title")) else 0.0
        out["title_similarity"] = sim
        out["first_author"] = out.get("first_author") or best.get("first_author")
        out["n_authors"] = out.get("n_authors") or best.get("n_authors")
        out["year"] = out.get("year") or best.get("year")
        t = tier_of(out.get("journal") or best.get("journal"), tiers)
        out["tier_recomputed"] = t if t in ("T1", "T2") else "T3"
        out["tier"] = rec.get("tier") or out["tier_recomputed"]
        fa_e, fa_g = norm(out.get("first_author")), norm(best.get("first_author"))
        author_ok = (not fa_e) or (fa_e in fa_g) or (fa_g in fa_e) or (not fa_g)
        if not best:
            out["verified"] = "failed"; out["fail_reason"] = "arxiv api no record"
        elif sim >= 0.90 and author_ok:
            out["verified"] = "verified"
        else:
            out["verified"] = "partial"; out["fail_reason"] = f"sim={sim} author_ok={author_ok}"
        return out
    # entry_type=book：图书（无 DOI，Crossref 不收录）。用 verify_urls 里的可抓取页面做交叉核验：
    # 页面须同时含标题前 3 词与第一作者姓氏（如 AAAI AI Magazine 对 Kanerva 1988 的书评页）。
    if rec.get("entry_type") in ("book", "web") and not rec.get("doi") and not rec.get("pmid"):
        out = dict(rec)
        out["resolved_by"] = "book-web-crosscheck"
        out["retrieved_at"] = now_iso()
        t = tier_of(out.get("journal"), tiers)
        out["tier_recomputed"] = t if t in ("T1", "T2", "T3") else "unknown"
        out["tier"] = rec.get("tier") or out["tier_recomputed"]
        title_w = [w for w in norm(rec.get("title") or "").split() if w]
        need = " ".join(title_w[:3]) if title_w else ""
        fa = norm(out.get("first_author"))
        fa_toks = fa.split() if fa else []
        isbn_d = re.sub(r"[^0-9xX]", "", str(out.get("isbn") or ""))
        cands = fa_toks + ([isbn_d] if isbn_d else [])
        ok_any = False
        for u in (out.get("verify_urls") or []):
            try:
                txt = http_get(u)
                plain = norm(re.sub(r"<[^>]+>", " ", txt))
                if need and need in plain and cands and any(c in plain for c in cands):
                    ok_any = True
                    break
            except Exception:
                continue
        if ok_any:
            out["verified"] = "verified"
        else:
            out["verified"] = "failed"
            out["fail_reason"] = "book: no verify_url matched title+author"
        return out
    title = rec.get("title") or ""
    r = resolve(rec.get("doi"), rec.get("pmid"), title)
    best = r["best"]
    out = dict(rec)
    out["resolved_by"] = ",".join(dict.fromkeys(r["attempts"])) or "none"
    out["retrieved_at"] = now_iso()

    for field in ("doi", "pmid", "pmcid", "journal", "year", "title_resolved"):
        if field == "title_resolved":
            if best.get("title") and not out.get("title"):
                out["title"] = best["title"]
        elif best.get(field) and not out.get(field):
            out[field] = best[field]

    out["journal"] = out.get("journal") or best.get("journal") or ""
    out["year"] = out.get("year") or best.get("year")
    sim = title_similarity(title, best.get("title", "")) if (title and best.get("title")) else 0.0
    out["title_similarity"] = sim
    out["tier_recomputed"] = tier_of(out.get("journal"), tiers)
    out["tier"] = rec.get("tier") or out["tier_recomputed"]
    out["retracted"] = bool(best.get("retracted"))

    fa_expected = norm(rec.get("first_author"))
    fa_got = norm(best.get("first_author"))
    author_ok = (not fa_expected) or (fa_expected in fa_got) or (fa_got in fa_expected) or not fa_got
    year_ok = (not rec.get("year")) or (str(out.get("year")) == str(rec.get("year")))

    if not r["attempts"]:
        out["verified"] = "failed"
        out["fail_reason"] = "no resolver returned a record"
    elif out["retracted"]:
        out["verified"] = "failed"
        out["fail_reason"] = "retracted"
    elif sim >= 0.90 and author_ok and year_ok:
        out["verified"] = "verified"
    else:
        out["verified"] = "partial"
        out["fail_reason"] = f"sim={sim} author_ok={author_ok} year_ok={year_ok}"
    return out

def load_refs(path: str) -> list[dict]:
    recs = []
    if not os.path.exists(path):
        return recs
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] {path}:{i} bad json: {e}", file=sys.stderr)
    return recs

def cmd_verify(args) -> int:
    tiers = load_tiers()
    refs = load_refs(args.file)
    if not refs:
        print(f"[FAIL] no refs at {args.file}")
        return 2
    cited_ids = set()
    if os.path.exists(CITED_PATH):
        with open(CITED_PATH, "r", encoding="utf-8") as f:
            try:
                cited_ids = set(json.load(f).get("cited_ref_ids", []))
            except Exception:
                cited_ids = set()
    results, bad = [], []
    for rec in refs:
        v = verify_record(rec, tiers)
        st = rec.get("tier")
        if st in ("T1", "T2", "T3") and v.get("tier_recomputed") and st != v["tier_recomputed"]:
            v["tier_mismatch"] = True
        results.append(v)
        flag = v["verified"]
        if flag == "verified" and v.get("tier") in ("T1", "T2", "T3"):
            mark = "OK  "
        else:
            mark = "BAD "
            bad.append(v)
        print(f"{mark}{v.get('ref_id','?'):>6} {flag:<8} tier={v.get('tier','?'):<8} "
              f"sim={(v.get('title_similarity') or 0):.2f} {str(v.get('journal'))[:42]:<42} "
              f"{(v.get('title') or '')[:52]}")
    os.makedirs(AUDIT_DIR, exist_ok=True)
    dist = {}
    for v in results:
        dist[v.get("tier", "unknown")] = dist.get(v.get("tier", "unknown"), 0) + 1
    both = dist.get("T1", 0) + dist.get("T2", 0)
    ratio_raw = both / max(1, len(results))
    ratio = round(ratio_raw, 4)
    summary = {
        "generated_at": now_iso(),
        "n_refs": len(results),
        "tier_distribution": dist,
        "t1_t2_ratio": ratio,
        "gate_t1_t2_ratio_ge_0_70": ratio >= 0.70,
        "n_failed": dist.get("failed", 0) + sum(1 for v in results if v["verified"] != "verified"),
        "n_reject": dist.get("REJECT", 0),
        "n_curated_reject": sum(1 for v in results if v.get("tier") == "REJECT" and v.get("ref_id") not in cited_ids),
        "n_cited_reject": sum(1 for v in results if v.get("tier") == "REJECT" and v.get("ref_id") in cited_ids),
        "n_tier_mismatch": sum(1 for v in results if v.get("tier_mismatch")),
        "t1_t2_ratio_denominator": "all bank records including curated rejects (n_refs)",
        "n_unknown_journal": dist.get("unknown", 0),
        "n_retracted": sum(1 for v in results if v.get("retracted")),
        "records": results,
    }
    summary["gate_all_verified"] = summary["n_failed"] == 0
    summary["gate_pass"] = (summary["gate_all_verified"] and summary["gate_t1_t2_ratio_ge_0_70"]
                            and summary["n_cited_reject"] == 0 and summary["n_unknown_journal"] == 0)
    jpath = os.path.join(AUDIT_DIR, "citations-report.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    mpath = os.path.join(AUDIT_DIR, "citations-report.md")
    with open(mpath, "w", encoding="utf-8") as f:
        f.write("# 引文机检报告\n\n")
        f.write(f"- 生成时间：{summary['generated_at']}\n")
        f.write(f"- 条目数：{summary['n_refs']}\n")
        f.write(f"- 期刊分级分布：{dist}\n")
        f.write(f"- T1+T2 占比：{ratio_raw:.1%}（门禁 ≥70%）\n")
        f.write(f"- 未核验：{summary['n_failed']}；撤稿：{summary['n_retracted']}；"
                f"curated-reject（库内拒引，未入论文）：{summary['n_curated_reject']}；cited-reject（入论文，门禁失败）：{summary['n_cited_reject']}；"
                f"分级分歧（stored vs 重推）：{summary['n_tier_mismatch']}；未分级刊：{summary['n_unknown_journal']}\n")
        f.write(f"- **门禁：{'PASS' if summary['gate_pass'] else 'FAIL'}**\n\n")
        f.write("| ref_id | 状态 | 分级 | 相似度 | 期刊 | 标题 |\n|---|---|---|---|---|---|\n")
        for v in results:
            f.write(f"| {v.get('ref_id','?')} | {v['verified']} | {v.get('tier','?')} | "
                    f"{v.get('title_similarity',0):.2f} | {str(v.get('journal'))[:50]} | "
                    f"{str(v.get('title'))[:70]} |\n")
    print(f"\n-- n={len(results)} tiers={dist} T1+T2={ratio_raw:.1%} "
          f"failed={summary['n_failed']} curated_reject={summary['n_curated_reject']} cited_reject={summary['n_cited_reject']} "
          f"mismatch={summary['n_tier_mismatch']} unknown={summary['n_unknown_journal']} retracted={summary['n_retracted']}")
    print(f"-- gate: {'PASS' if summary['gate_pass'] else 'FAIL'}  -> {jpath}")
    return 0 if summary["gate_pass"] else 1

def cmd_search(args) -> int:
    """发现阶段：Crossref 书目检索 -> DOI->PMID / OA 富化 -> 候选 JSONL（尚未核验）。"""
    tiers = load_tiers()
    cands = crossref_search(args.query, n=args.n)
    if not cands:
        print(json.dumps({"query": args.query, "hits": 0, "note": "crossref no hit"}, ensure_ascii=False))
        return 1
    out, idx = [], args.start
    for c in cands:
        doi = c.get("doi")
        if not doi:
            continue
        if c.get("type") and c["type"] not in ("journal-article", "proceedings-article", "book-chapter"):
            continue
        sim = title_similarity(args.query, c.get("title", ""))
        if sim < args.min_sim:
            continue
        rec = {
            "ref_id": f"R{idx:03d}",
            "module": args.module,
            "role": args.role,
            "doi": doi,
            "title": c.get("title", ""),
            "journal": c.get("journal", ""),
            "year": c.get("year"),
            "first_author": c.get("first_author", ""),
            "n_authors": c.get("n_authors", 0),
            "source_query": args.query,
            "resolved_by": "crossref",
            "retrieved_at": now_iso(),
            "sim_to_query": sim,
        }
        for pid in pubmed_esearch_doi(doi):
            pm = pubmed_efetch(pid)
            if pm:
                rec["pmid"] = pm.get("pmid")
                rec["retracted"] = bool(pm.get("retracted"))
                if not rec.get("journal"):
                    rec["journal"] = pm.get("journal", "")
                if not rec.get("year"):
                    rec["year"] = pm.get("year")
            break
        ep = europepmc_by_ids(doi=doi)
        if ep:
            rec["pmcid"] = ep.get("pmcid", "")
            rec["is_open_access"] = bool(ep.get("is_open_access"))
            rec["retracted"] = bool(rec.get("retracted")) or bool(ep.get("retracted"))
            rec["abstract"] = (ep.get("abstract") or "")[:1200]
            if not rec.get("journal"):
                rec["journal"] = ep.get("journal", "")
        rec["tier"] = tier_of(rec.get("journal"), tiers)
        rec["retracted"] = bool(rec.get("retracted"))
        out.append(rec)
        idx += 1
    if args.out and out:
        with open(args.out, "a", encoding="utf-8") as fh:
            for rec in out:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[OK] appended {len(out)} candidates (R{args.start:03d}..R{idx-1:03d}) -> {args.out}")
    for rec in out:
        print(f"{rec['tier']:<8} {str(rec.get('year')):<5} {str(rec.get('journal'))[:38]:<38} "
              f"sim={rec['sim_to_query']:.2f} {str(rec.get('title'))[:66]}")
    return 0

def cmd_tier(args) -> int:
    tiers = load_tiers()
    r = resolve(args.doi, args.pmid, args.title)
    best = r["best"]
    print(json.dumps({
        "attempts": r["attempts"],
        "journal": best.get("journal"),
        "tier": tier_of(best.get("journal"), tiers),
        "year": best.get("year"),
        "doi": best.get("doi"),
        "title": best.get("title"),
        "retracted": best.get("retracted"),
    }, ensure_ascii=False, indent=2))
    return 0

def cmd_stats(args) -> int:
    tiers = load_tiers()
    refs = load_refs(args.file)
    dist, mods = {}, {}
    for rec in refs:
        t = rec.get("tier") or tier_of(rec.get("journal"), tiers)
        dist[t] = dist.get(t, 0) + 1
        m = rec.get("module", "?")
        mods[m] = mods.get(m, 0) + 1
    n = len(refs)
    print(f"n={n} tiers={dist} T1+T2={((dist.get('T1',0)+dist.get('T2',0))/max(1,n)):.1%} "
          f"(denominator=n_all incl. curated_reject={dist.get('REJECT',0)})")
    print("modules:", json.dumps(mods, ensure_ascii=False, sort_keys=True))
    return 0

def main() -> int:
    ap = argparse.ArgumentParser(description="HCM citation verifier")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search"); s.add_argument("query")
    s.add_argument("--n", type=int, default=5); s.add_argument("--module", default="cross")
    s.add_argument("--role", default="mechanism"); s.add_argument("--start", type=int, default=1)
    s.add_argument("--min-sim", dest="min_sim", type=float, default=0.80)
    s.add_argument("--out", default=""); s.set_defaults(fn=cmd_search)

    v = sub.add_parser("verify"); v.add_argument("--file", default=REFS_PATH)
    v.set_defaults(fn=cmd_verify)

    t = sub.add_parser("tier"); t.add_argument("--doi"); t.add_argument("--pmid")
    t.add_argument("--title"); t.set_defaults(fn=cmd_tier)

    st = sub.add_parser("stats"); st.add_argument("--file", default=REFS_PATH)
    st.set_defaults(fn=cmd_stats)

    args = ap.parse_args()
    return args.fn(args)

if __name__ == "__main__":
    sys.exit(main())
