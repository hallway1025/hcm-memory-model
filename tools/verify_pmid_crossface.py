# -*- coding: utf-8 -*-
"""verify_pmid_crossface.py -- PMID -> title cross-face check (L-065 debt).

Catches the R003-class bug (Buzsaki 2015 carrying Penfield 1958's PMID 13519951):
format-resolvers pass any valid PMID; this tool checks the PMID actually RETRIEVES
the same paper. Source: Europe PMC REST (no key needed).
  GET https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:{pmid}%20AND%20SRC:MED&resultType=core&format=json

For each refs.jsonl entry with a pmid: compare retrieved title vs bank title
(normalized SequenceMatcher ratio, FLAG < 0.75), and DOI equality when both present.
Output: audit/pmid-crossface-report.json; exit 1 if any FLAG (gate-ready).
"""
import difflib, json, os, re, sys, time
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFS = os.path.join(ROOT, "refs", "refs.jsonl")
CACHE = os.path.join(ROOT, "refs", "cache", "europepmc")
OUT = os.path.join(ROOT, "audit", "pmid-crossface-report.json")
UA = "HCM-pmid-crossface/1.0 (research; local)"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.makedirs(CACHE, exist_ok=True)

def norm_title(t):
    t = (t or "").lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def fetch(pmid):
    cp = os.path.join(CACHE, pmid + ".json")
    if os.path.exists(cp):
        with open(cp, "r", encoding="utf-8") as f:
            return json.load(f)
    url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:"
           + pmid + "%20AND%20SRC:MED&resultType=core&format=json")
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    data = r.json()
    hits = (data.get("resultList", {}) or {}).get("result", []) or []
    rec = hits[0] if hits else None
    slim = None
    if rec:
        slim = {"title": rec.get("title"), "authorString": rec.get("authorString", "")[:120],
                "doi": rec.get("doi"), "pubYear": rec.get("pubYear"),
                "journalTitle": rec.get("journalTitle"), "pmid": rec.get("pmid")}
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(slim, f, ensure_ascii=False)
    time.sleep(0.3)
    return slim

def main():
    rows = []
    with open(REFS, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    with_pmid = [r for r in rows if r.get("pmid")]
    print("refs=%d with_pmid=%d" % (len(rows), len(with_pmid)), flush=True)
    report = {"checked": 0, "ok": 0, "flag": 0, "no_hit": 0, "flags": [], "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for k, r in enumerate(with_pmid):
        pmid = str(r["pmid"])
        try:
            rec = fetch(pmid)
        except Exception as e:
            report["flags"].append({"ref": r.get("ref_id"), "pmid": pmid, "issue": "FETCH_ERROR", "detail": str(e)[:120]})
            report["flag"] += 1
            continue
        if not rec:
            report["no_hit"] += 1
            report["flags"].append({"ref": r.get("ref_id"), "pmid": pmid, "issue": "NO_HIT",
                                    "bank_title": (r.get("title") or "")[:100]})
            continue
        report["checked"] += 1
        ratio = difflib.SequenceMatcher(None, norm_title(r.get("title")), norm_title(rec.get("title"))).ratio()
        _ndoi = lambda s: (s or "").lower().replace("https://doi.org/", "").replace("//", "/")
        doi_ok = (not r.get("doi") or not rec.get("doi") or _ndoi(r["doi"]) == _ndoi(rec["doi"]))
        if ratio < 0.75 or not doi_ok:
            report["flag"] += 1
            report["flags"].append({"ref": r.get("ref_id"), "pmid": pmid, "issue": "TITLE_MISMATCH" if ratio < 0.75 else "DOI_MISMATCH",
                                    "ratio": round(ratio, 3), "doi_ok": doi_ok,
                                    "bank_title": (r.get("title") or "")[:100],
                                    "epmc_title": (rec.get("title") or "")[:100],
                                    "bank_doi": r.get("doi"), "epmc_doi": rec.get("doi")})
        else:
            report["ok"] += 1
        if (k + 1) % 50 == 0:
            print("  %d/%d ok=%d flag=%d nohit=%d" % (k + 1, len(with_pmid), report["ok"], report["flag"], report["no_hit"]), flush=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("DONE checked=%d ok=%d flag=%d no_hit=%d -> %s" % (report["checked"], report["ok"], report["flag"], report["no_hit"], OUT))
    if report["flag"]:
        for fl in report["flags"]:
            print("FLAG", json.dumps(fl, ensure_ascii=False)[:220])
        sys.exit(1)

if __name__ == "__main__":
    main()
