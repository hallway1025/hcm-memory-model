#!/usr/bin/env python3
'''exp30_smoke.py -- 事件层烟测（L0，秒级）。验收 D-042：
同一内容两个时刻 -> 两条事件记录共享一个内容码；时间/来源检索可用；默认路径不变。'''
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit

rng = np.random.default_rng(11)
D = 128
m = HCM(Config(d_x=D, d_c=D, capacity=100, beta=20.0, a=0.02, L=3, use_resid=False, seed=0))

X = [unit(v) for v in rng.standard_normal((6, D))]
x = X[0]
r1 = m.observe(x, event=True, sender="A", session="s1")   # t=0
m.observe(X[1])                                            # 填充（非事件路径）
r2 = m.observe(x, event=True, sender="B", session="s1")   # 同内容，t=2
evs = m.query_events(t0=0, t1=10)
assert len(evs) == 2, ("event count", len(evs))
assert evs[0]["code_id"] == evs[1]["code_id"], ("same content must share one code", evs)
assert evs[1]["is_new_code"] is False, "second occurrence must reuse code"
assert evs[0]["slot"] == evs[1]["slot"] and evs[0]["slot"] is not None
assert len(m.query_events(sender="A")) == 1 and len(m.query_events(sender="B")) == 1
assert len(m.query_events(t0=2, t1=10)) == 1 and m.query_events(t0=2, t1=10)[0]["sender"] == "B"
assert len(m.query_events(session="s1")) == 2
o = m.query(x)
assert o.get("action") in ("recall", "separate"), o
assert o.get("idx") == evs[0]["slot"], ("content query slot", o.get("idx"), evs[0]["slot"])
assert r1["idx"] is not None and r1["idx"] >= 0, r1
assert r2["idx"] == r1["idx"] and r2["eid"] != r1["eid"], (r1, r2)
d = m.E.dump()
assert d["n_events"] == 2 and d["n_event_codes_reused"] == 1, d
print("EXP30 SMOKE PASS", json.dumps(d))
