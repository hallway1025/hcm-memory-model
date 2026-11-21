import os

import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
recs = [json.loads(l) for l in open(os.path.join(ROOT,"refs","refs.jsonl"), encoding="utf-8") if l.strip()]
keys = [("Squire",1992),("Kumaran",2016),("Yassa",2011),("Girardeau",2009),("Klinzing",2019),
        ("Frankland",2005),("Winocur",2011),("BennaFusi",2016),("Miller",1956),("Cowan",2001),
        ("Kirkpatrick",2017),("French",1999),("Tulving",2002),("Ranganath",2012),("Leutgeb",2007),
        ("Neunuebel",2014),("Rolls",2013),("Hopfield",1982),("Ramsauer",2020),("Wilson",1994),
        ("Buzsaki",2015),("Foster",2006),("Diba",2007),("Sekeres",2018),("Shenhav",2013),
        ("Fleming",2012),("Fusi",2005),("McClelland",1995),("Scoville",1957),("Lisman",1995),
        ("McNaughton",2006),("Eichenbaum",2017),("Dudai",2004),("Nadel",2013),("Preston",2013)]
def norm(s): return "".join(ch for ch in (s or "").lower() if ch.isalnum())
hits = {}
for r in recs:
    y = str(r.get("year") or "")
    fa = norm(r.get("first_author") or "")
    auths = norm(r.get("authors") or "")
    t = r.get("title") or ""
    for name, yr in keys:
        n = norm(name)
        if str(yr) == y and (n in fa or n in auths):
            hits.setdefault((name,yr), []).append((r["ref_id"], r.get("tier"), r.get("journal","")[:38], t[:60]))
for name, yr in keys:
    v = hits.get((name,yr), [])
    if v:
        print("OK  ", name, yr, "->", v[0][0], v[0][1], "|", v[0][2], "|", v[0][3], f"(n={len(v)})")
    else:
        print("MISS", name, yr)
