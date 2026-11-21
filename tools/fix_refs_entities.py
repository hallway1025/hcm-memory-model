# -*- coding: utf-8 -*-
"""One-off: unescape HTML entities (&amp; etc.) in refs.jsonl title/journal/first_author."""
from html import unescape
import json, shutil
P = r"refs/refs.jsonl"
shutil.copy(P, P + ".bak-entities")
out, changed = [], 0
with open(P, "r", encoding="utf-8") as f:
    for ln in f:
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        for k in ("title", "journal", "first_author"):
            v = r.get(k)
            if isinstance(v, str) and "&" in v:
                u = unescape(v)
                if u != v:
                    r[k] = u
                    changed += 1
        out.append(json.dumps(r, ensure_ascii=False))
with open(P, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(out) + "\n")
print("fields changed:", changed)
