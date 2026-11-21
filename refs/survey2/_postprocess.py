import os
# -*- coding: utf-8 -*-
"""refs/survey2/_postprocess.py -- survey2 后处理：
1) 与调查一库 (refs/survey/refs_survey.jsonl) 按 doi/arxiv_id/标题规整串去重；
2) 与调查二库内部去重（collect 已做，二次保险）；
3) ref_id 前缀 S -> T（调查二专用前缀，避免与调查一混淆）；
4) 打印漏斗对照表（每区 removed_by_survey1 / final）。
用法: python refs/survey2/_postprocess.py
"""
import json, re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P1 = ROOT + "/refs/survey/refs_survey.jsonl"
P2 = ROOT + "/refs/survey2/refs_survey2.jsonl"
def norm(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())[:120]
def keys(r):
    ks = set()
    if r.get("doi"): ks.add("doi:" + r["doi"].lower())
    if r.get("arxiv_id"): ks.add("ax:" + r["arxiv_id"])
    if r.get("title"): ks.add("t:" + norm(r["title"]))
    return ks
s1 = [json.loads(l) for l in open(P1, encoding="utf-8") if l.strip()]
have = set()
for r in s1: have |= keys(r)
s2 = [json.loads(l) for l in open(P2, encoding="utf-8") if l.strip()]
seen, out, dropped = set(), [], {"survey1": 0, "internal": 0}
for r in s2:
    ks = keys(r)
    if ks & have:
        dropped["survey1"] += 1
        continue
    if ks & seen:
        dropped["internal"] += 1
        continue
    seen |= ks
    out.append(r)
# rename S -> T, renumbering after the max existing T number (idempotent across runs)
tnums = [int(r["ref_id"][1:]) for r in out if r.get("ref_id", "").startswith("T")]
nxt = 1 + max(tnums) if tnums else 0
for r in out:
    if r.get("ref_id", "").startswith("S"):
        r["ref_id"] = "T%03d" % nxt
        nxt += 1
with open(P2, "w", encoding="utf-8") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
from collections import Counter
c = Counter((r.get("area") or r.get("module") or "?") for r in out)
d1 = Counter()
print("dropped (already in survey-1):", dropped["survey1"], "| internal dup:", dropped["internal"])
print("final records:", len(out))
for a in sorted(c): print("  %-24s %d" % (a, c[a]))
