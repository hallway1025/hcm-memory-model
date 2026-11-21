# -*- coding: utf-8 -*-
"""export_refs_ris.py — 从 paper/PAPER_final.md 参考文献表导出投稿系统用 RIS。
匹配链：行内 doi:... → refs.jsonl 精确匹配（authoritative 元数据：作者全列表/期刊/年份）；
无 DOI 行退回标题匹配（norm 后包含）。输出 paper/references.ris（TY JOUR / TY CHAP）。
"""
import json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper", "PAPER_final.md")
REFS = os.path.join(ROOT, "refs", "refs.jsonl")
OUT = os.path.join(ROOT, "paper", "submission", "references.ris")

def norm(s):
    s = str(s).lower().replace("&", " and ")
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

bank = [json.loads(l) for l in open(REFS, encoding="utf-8") if l.strip()]
by_doi = {r.get("doi", "").lower(): r for r in bank if r.get("doi")}
titles = [(norm(r.get("title", "")), r) for r in bank]

text = open(PAPER, encoding="utf-8").read()
tail = text.partition("## 参考文献")[2]
entries = re.findall(r"^\[(\d+)\]\s*(.+?)(?=\n\n|\Z)", tail, re.S | re.M)
rows, missing = [], []
for num, body in entries:
    doi = None
    m = re.search(r"doi[:\s]*([0-9.]+/[^\s\u3011]+)", body, re.I)
    if not m:
        m = re.search(r"10\.\d{4,9}/[^\s]+", body)
    if m:
        doi = m.group(1).rstrip(".").lower()
    r = by_doi.get(doi) if doi else None
    if r is None:
        nb = norm(re.sub(r"doi:.*", "", body))[:90]
        for nt, cand in titles:
            if nb and (nb in nt or nt[:60] in nb):
                r = cand
                break
    if r is None:
        missing.append((num, body[:80]))
        continue
    ty = "CHAP" if (r.get("journal", "") in ("Novel Mechanisms of Memory", "Reinforcement Learning Foundations")) else "JOUR"
    lines = ["TY  - " + ty]
    for a in (r.get("authors") or []):
        lines.append("AU  - " + a)
    if not (r.get("authors") or []):
        fa = r.get("first_author")
        if fa:
            lines.append("AU  - " + fa)
    lines += ["TI  - " + (r.get("title") or ""), "JO  - " + (r.get("journal") or "")]
    if r.get("year"):
        lines.append("PY  - %d" % int(r["year"]))
    if r.get("doi"):
        lines.append("DO  - " + r["doi"])
    if r.get("pmid"):
        lines.append("AN  - " + str(r["pmid"]))
    lines.append("ER  - ")
    rows.append((int(num), "\r\n".join(lines)))
rows.sort()
with open(OUT, "w", encoding="utf-8", newline="") as f:
    f.write("\r\n".join(b for _, b in rows) + "\r\n")
print("RIS entries:", len(rows), "/ refs in list:", len(entries))
for num, b in missing:
    print("MISSING [%d] %s" % (num, b))
sys.exit(1 if missing else 0)
