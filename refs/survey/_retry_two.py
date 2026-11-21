# -*- coding: utf-8 -*-
'''refs/survey/_retry_two.py -- 重试 S1126/S1123 核验'''
import json, sys, time
sys.path.insert(0, "tools")
import verify_citations as vc
recs = [json.loads(l) for l in open("refs/survey/refs_survey.jsonl", encoding="utf-8") if l.strip()]
tiers = vc.load_tiers()
for rid in ["S1126", "S1123"]:
    rec = next(r for r in recs if r["ref_id"] == rid)
    orig = rec.get("role")
    if not rec.get("doi") and not rec.get("pmid"):
        rec["role"] = "system"  # 走 arXiv API 分支
    res = vc.verify_record(rec, tiers)
    print(rid, {k: res.get(k) for k in ("verified","sim","fail_reason","reason","status") if k in res})
    time.sleep(3.5)
