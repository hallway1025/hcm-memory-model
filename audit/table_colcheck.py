import os
# -*- coding: utf-8 -*-
"""Table-integrity audit: column-count consistency for every markdown table
in MANUSCRIPT-ARXIV.md. Pure mechanical check, no judgement."""
import io, sys, re

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "PAPER.md")

def cells(line):
    # split a markdown table row into cells; strip leading/trailing pipe
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]

def is_sep(line):
    return re.fullmatch(r"\|?[\s:|-]+\|?", line.strip()) is not None and "-" in line

with io.open(PATH, encoding="utf-8") as f:
    lines = f.read().splitlines()

tables = []   # list of dicts: start, rows[(lineno, ncells, is_sep)]
cur = None
for i, ln in enumerate(lines, 1):
    if ln.strip().startswith("|"):
        if cur is None:
            cur = {"start": i, "rows": []}
        cur["rows"].append((i, len(cells(ln)), is_sep(ln), ln.strip()[:80]))
    else:
        if cur is not None:
            tables.append(cur)
            cur = None
if cur is not None:
    tables.append(cur)

print(f"total tables: {len(tables)}")
bad = 0
for t in tables:
    counts = {}
    for (lineno, n, sep, _) in t["rows"]:
        counts.setdefault(n, []).append((lineno, sep))
    status = "OK" if len(counts) == 1 else "MISMATCH"
    if status != "OK":
        bad += 1
    hdr = t["rows"][0][3]
    print(f"--- table @L{t['start']} ({len(t['rows'])} rows) {status}")
    print(f"    header: {hdr}")
    for n, occ in sorted(counts.items()):
        locs = ",".join(f"L{l}{'(sep)' if s else ''}" for l, s in occ)
        print(f"    {n} cols x{len(occ)}: {locs}")
print(f"\nRESULT: {len(tables)} tables checked, {bad} with column-count mismatch")
