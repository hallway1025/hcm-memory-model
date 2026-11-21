# -*- coding: utf-8 -*-
"""refs/survey2/_fetch_ids2.py -- survey2 定点补采（web-row 转 D-003 合规库条目）。一次性工具。"""
import json, sys, time
sys.path.insert(0, "tools")
import verify_citations as vc
import xml.etree.ElementTree as ET
NS = "{http://www.w3.org/2005/Atom}"
ids = ["2510.04127", "2412.09764", "2407.04153", "2512.14709", "2607.13591", "2511.20665"]
AREAS = {"2510.04127": "K1_sparse_encoder", "2412.09764": "K2_episodic_store",
         "2407.04153": "K2_episodic_store", "2512.14709": "K2_episodic_store",
         "2607.13591": "K7_meta_control", "2511.20665": "K9_text_frontend"}
url = "http://export.arxiv.org/api/query?id_list=" + ",".join(ids) + "&max_results=10"
xmltext = vc.http_get(url)
root = ET.fromstring(xmltext)
recs = []
for e in root.findall(NS + "entry"):
    t = " ".join((e.findtext(NS + "title") or "").split())
    authors = [a.findtext(NS + "name") for a in e.findall(NS + "author")]
    pub = e.findtext(NS + "published") or ""
    aidfull = (e.findtext(NS + "id") or "").split("/abs/")[-1]
    aid = aidfull.split("v")[0]
    summ = " ".join((e.findtext(NS + "summary") or "").split())
    recs.append(dict(title=t, authors=authors, year=int(pub[:4]) if pub[:4].isdigit() else None,
                     arxiv_id=aid, abstract=summ[:1200]))
existing = [json.loads(l) for l in open("refs/survey2/refs_survey2.jsonl", encoding="utf-8") if l.strip()]
ids = [i for i in ids if i not in {r.get("arxiv_id") for r in existing}]
print("to fetch:", ids)
have = {r.get("arxiv_id") for r in existing if r.get("arxiv_id")}
have |= {r.get("doi") for r in existing if r.get("doi")}
nxt = 1 + max([int(r["ref_id"][1:]) for r in existing if r.get("ref_id", "").startswith("S")] or [0])
added = []
for r in recs:
    if r["arxiv_id"] in have:
        print("skip dup", r["arxiv_id"]); continue
    rec = dict(ref_id="S%03d" % nxt, area=AREAS.get(r["arxiv_id"], "K2_episodic_store"),
               module=AREAS.get(r["arxiv_id"], "K2_episodic_store"), role="survey",
               channel="arxiv", title=r["title"], authors=r["authors"], year=r["year"],
               journal="arXiv preprint", doi=None, pmid=None, pmcid=None, arxiv_id=r["arxiv_id"],
               abstract=r["abstract"], primary_category=None, tier="preprint", sim=1.0,
               added_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    existing.append(rec)
    added.append(rec["ref_id"] + " " + r["arxiv_id"] + " " + r["title"][:60])
    nxt += 1
with open("refs/survey2/refs_survey2.jsonl", "w", encoding="utf-8") as f:
    for r in existing:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("added:", added)
print("total:", len(existing))
