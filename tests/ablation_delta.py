import os

import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = json.load(open(os.path.join(ROOT,"results","ablation.json"), encoding="utf-8"))
base = next(r for r in rows if r["arm"] == "full")
print("baseline:", base)
deltas = []
for r in rows:
    if r["arm"] == "full": continue
    d = dict(arm=r["arm"], d_sep=r["sep_exact_090"]-base["sep_exact_090"],
             d_drop=r["drop30"]-base["drop30"], d_cls=r["cls_old"]-base["cls_old"],
             d_wm=round(r["wm_kept"]-base["wm_kept"],2))
    d["load_bearing"] = any(abs(v) >= 2 for v in (d["d_sep"], d["d_drop"], d["d_cls"])) or abs(d["d_wm"])>=0.5
    deltas.append(d)
for d in deltas:
    print("%-28s dSEP=%+3d dDROP=%+3d dCLS=%+3d dWM=%+5.2f  %s" % (
        d["arm"], d["d_sep"], d["d_drop"], d["d_cls"], d["d_wm"],
        "LOAD-BEARING" if d["load_bearing"] else "*** NON-LOAD-BEARING ***"))
print()
print("non-load-bearing count:", sum(1 for d in deltas if not d["load_bearing"]))
