import os

import json, io
b = json.load(io.open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "exp20b_resid_beta_l0.json"), encoding="utf-8"))
print("exp20b type", type(b).__name__, "keys" if isinstance(b, dict) else "len")
s = json.dumps(b)[:900]
print(s)
f = json.load(io.open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "exp23f_embed_stats.json"), encoding="utf-8"))
print("PAIRED:", json.dumps(f["PAIRED"])[:600])
d = json.load(io.open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "exp24_b2_disambiguation.json"), encoding="utf-8"))
print("exp24 arms:", list(d["arms"].keys()))
print("exp24 wrong:", json.dumps({k: d["arms"][k] for k in d["arms"] if "wrong" in k or "Wrong" in k})[:300])
