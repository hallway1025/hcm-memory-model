#!/usr/bin/env python3
'''exp30a_event_identity.py -- 事件身份可分性正式实验（L0，exp30 核心验收）。
问题（C12/D-042）：同一内容在不同时刻发生，系统能否分别检索到两次"发生"？
基线（无事件层）：theta_dup 在元数据落盘前丢弃第二次发生（episodic.py 旧 :38-39 行为）。
判据：事件层 100/100 探针对两条事件均可按时间窗分别检索且共享同一内容码；
基线 100/100 丢失第二次事件。judge: 结构断言，无阈值参数（D-038 不适用）。'''
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit

N_PAIRS = int(os.environ.get('EXP30A_PAIRS', '100'))
FILLERS = 3
rng = np.random.default_rng(300)
D = 128
res = dict(n_pairs=N_PAIRS, both_retrieved=0, window_correct=0, shared_code=0,
           baseline_lost=0, triple_all_retrieved=0, n_triples=20)

def make_model():
    return HCM(Config(d_x=D, d_c=D, capacity=2000, beta=20.0, a=0.02, L=3,
                      use_resid=False, seed=0))

p = 0
while p < N_PAIRS:
    m = make_model()
    x = unit(rng.standard_normal(D))
    _probe = m.observe(x, event=True, sender="S%d" % (p % 3), session="sessA")
    if not _probe["wrote"]:
        continue  # 门控拒绝首次写入的条件不构成本实验对象（判定条件：首次发生已入库）
    fill = [unit(rng.standard_normal(D)) for _ in range(FILLERS)]
    for f in fill:
        m.observe(f, event=True)                                        # 间隔流
    m.observe(x, event=True, sender="S%d" % (p % 3), session="sessB")   # t=4 第二次发生
    evs = m.query_events(sender="S%d" % (p % 3))
    same_pair = [e for e in evs if e["session"] in ("sessA", "sessB")]
    ok_pair = (len(same_pair) == 2
               and same_pair[0]["code_id"] == same_pair[1]["code_id"]
               and same_pair[0]["t"] != same_pair[1]["t"])
    if ok_pair:
        res["shared_code"] += 1
        a_win = m.query_events(t0=same_pair[0]["t"] - 0.5, t1=same_pair[0]["t"] + 0.5)
        b_win = m.query_events(t0=same_pair[1]["t"] - 0.5, t1=same_pair[1]["t"] + 0.5)
        if (len(a_win) == 1 and len(b_win) == 1
                and a_win[0]["session"] == "sessA" and b_win[0]["session"] == "sessB"):
            res["window_correct"] += 1
        o = m.query(x)
        if o.get("idx") == same_pair[0]["slot"] and o.get("action") in ("recall", "separate"):
            res["both_retrieved"] += 1
    # 基线对照：同一内容流走旧路径（无事件层）
    mb = make_model()
    rb1 = mb.observe(x)            # 第一次：正常写入
    rb2 = mb.observe(x)            # 第二次：theta_dup 丢弃
    if rb1["wrote"] and not rb2["wrote"] and rb2["idx"] == -1:
        res["baseline_lost"] += 1
    if p < res["n_triples"]:
        m.observe(x, event=True, sender="S%d" % (p % 3), session="sessC")
        tri = [e for e in m.query_events(sender="S%d" % (p % 3)) if e["session"] in ("sessA", "sessB", "sessC")]
        if len(tri) == 3 and len({e["code_id"] for e in tri}) == 1:
            res["triple_all_retrieved"] += 1
    p += 1

res["verdict"] = ("PASS" if res["shared_code"] == N_PAIRS and res["window_correct"] == N_PAIRS
                  and res["both_retrieved"] == N_PAIRS and res["baseline_lost"] == N_PAIRS
                  and res["triple_all_retrieved"] == res["n_triples"] else "FAIL")
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'exp30a_event_identity.json')
json.dump(res, open(out, 'w', encoding='utf-8'), indent=1)
print('EXP30A', res['verdict'], json.dumps(res), flush=True)
