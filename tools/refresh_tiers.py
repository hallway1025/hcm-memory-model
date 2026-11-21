# -*- coding: utf-8 -*-
"""One-off: refresh stored 'tier' field in refs.jsonl from current journal_tiers (no network)."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import verify_citations as vc
P = os.path.join(vc.ROOT, "refs", "refs.jsonl")
tiers = vc.load_tiers()
out, changed = [], 0
with open(P, encoding="utf-8") as f:
    for ln in f:
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        t = vc.tier_of(r.get("journal"), tiers)
        if t != r.get("tier"):
            r["tier"] = t
            changed += 1
        out.append(json.dumps(r, ensure_ascii=False))
with open(P, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(out) + "\n")
print("tier fields refreshed:", changed)
