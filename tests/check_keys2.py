import os

import json, os, unicodedata
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
recs = [json.loads(l) for l in open(os.path.join(ROOT,"refs","refs.jsonl"), encoding="utf-8") if l.strip()]
def fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c for c in s.lower() if c.isalnum())
want = [("winocur",2011),("benna",2016),("kirkpatrick",2017),("ramsauer",2020),("demircigil",2017),
        ("buzsaki",2015),("mcnaughton",2006),("dudai",2004),("nadel",2013),("preston",2013),
        ("frankland",2005),("hopfield",1982)]
for nm, yr in want:
    hits=[]
    for r in recs:
        if str(r.get("year") or "") != str(yr): continue
        blob = fold(r.get("first_author")) + "|" + fold(r.get("authors"))
        if nm in blob:
            hits.append((r["ref_id"], r.get("tier"), (r.get("journal") or "")[:34], (r.get("title") or "")[:58]))
    print(("OK   " if hits else "MISS "), nm, yr, hits[:2])
print("--- total", len(recs))
