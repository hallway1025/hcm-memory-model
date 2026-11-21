#!/usr/bin/env python3
"""gen_supplemental_s4.py — generate SUPPLEMENTAL-S4-citations.md from citations-report.json."""
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "submission", "arxiv", "SUPPLEMENTAL-S4-citations.md")
REPORT = os.path.join(ROOT, "audit", "citations-report.json")

def truncate(s, maxlen=60):
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s[:maxlen] + "…" if len(s) > maxlen else s

# Records whose identifiers were verified by hand against the physical artefact
# (no resolver returns them). Kept in the generator so the ledger stays reproducible.
STATIC_IDENTIFIERS = {
    "R419": "ISBN 0-262-11132-2",
    "R430": "arXiv:2111.05898",
}

def main():
    report = json.load(open(REPORT, encoding="utf-8"))
    records = report["records"]
    
    lines = [
        "# Supplemental Material S4: Citation Verification Report (Full Ledger)",
        "",
        "> Auto-generated from audit/citations-report.json by tools/gen_supplemental_s4.py.",
        f"> Report generated: {report.get('generated_at', 'N/A')}",
        f"> Total records: {report['n_refs']} · verified: {report['n_refs'] - report['n_failed']} · failed: {report['n_failed']} · retracted: {report['n_retracted']}",
        f"> T1: {report['tier_distribution'].get('T1', 'N/A')} · T2: {report['tier_distribution'].get('T2', 'N/A')} · T3: {report['tier_distribution'].get('T3', 'N/A')} · T1+T2: {report['t1_t2_ratio']:.1%}",
        "",
        "Each record shows the resolution channel(s), retrieval timestamp, and resolvable identifier (DOI/PMID/arXiv ID) for independent verification.",
        "",
        "| ref_id | title (truncated) | first_author | year | resolved_by | retrieved_at | identifier | tier |",
        "|--------|-------------------|-------------|------|-------------|--------------|-----------|------|",
    ]
    
    for r in records:
        rid = r.get("ref_id", "?")
        title = truncate(r.get("title", "?"), 55)
        author = truncate(r.get("first_author", "?"), 20)
        year = r.get("year", "?")
        resolved = r.get("resolved_by", "?")
        retrieved = r.get("retrieved_at", "?")[:10] if r.get("retrieved_at") else "N/A"
        tier = r.get("tier", "?")
        
        # Build identifier (prefer DOI > arXiv ID > PMID; sanitize pipes that break the table)
        ident = ""
        if r.get("doi"):
            ident = "doi:" + str(r["doi"]).strip()
        elif r.get("arxiv_id"):
            ident = "arXiv:" + str(r["arxiv_id"])
        elif r.get("pmid"):
            ident = "PMID:" + str(r["pmid"])

        # Zenodo / web-only records: embed resolvable URL from journal hint when known
        if not ident:
            journal = (r.get("journal") or "")
            m = re.search(r"record (\d+)", journal)
            if m:
                ident = "doi:10.5281/zenodo." + m.group(1)
            elif "anthropic" in journal.lower():
                ident = "URL:anthropic.com/research/components-of-agent-memory"
        # Static identifiers for records the resolvers cannot supply (checked by hand
        # against the physical artefact; keep here so regeneration is reproducible).
        if not ident:
            ident = STATIC_IDENTIFIERS.get(rid, "")

        def cell(x):
            x = str(x)
            return x.replace("|", "/").replace("\n", " ")

        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            cell(rid), cell(title), cell(author), cell(year), cell(resolved), cell(retrieved), cell(ident), cell(tier)))
    
    lines.append("")
    lines.append("---")
    lines.append("*Auto-generated — do not edit manually. Source: audit/citations-report.json*")
    
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    
    print(f"Wrote {OUT}: {len(records)} records, {len(lines)} lines")

if __name__ == "__main__":
    main()
