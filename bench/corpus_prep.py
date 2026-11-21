import os
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''corpus_prep.py —— 微信真实语料预处理（HCM 真实数据实验的第一步）。

输入：桌面 HCM语料/*.csv（TalkBook 导出：时间,发送者,内容；UTF-8 BOM）
输出：
  bench/corpus/{split}__{sess}.csv   清洗后数据（发送者哈希化），split ∈ train/test（80/20 按时间）
  results/corpus_quality.json        质量统计（论文可引用）
清洗规则：
  - 去空内容/去纯系统消息（[图片] 等单独成条则丢弃——它不是文本语义）
  - 完全重复行（时间+内容相同）去重
  - 发送者 sha256(会话名+发送者)[:10] 哈希，映射表只存本地 bench/corpus/_sender_map.json（不入 results）
'''
import csv, json, os, re, hashlib, sys
from datetime import datetime

SRC = os.environ.get("HCM_CORPUS_DIR", "")  # 自备语料目录；本仓库不分发真实语料
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "bench", "corpus")
os.makedirs(OUT, exist_ok=True)
SESSIONS = ["group-01", "group-04", "peer-02", "group-02", "group-03", "group-05"]
SYS_PAT = re.compile(r'^\[(图片|视频|语音|文件|链接|动画表情|合并转发|红包|转账|位置|名片|音乐|小程序|引用|拍一拍)\]$')

def clean_rows(path, sess):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        assert header[0].startswith("时间"), header
        for t, sender, content in rd:
            if not t or not sender: continue
            content = (content or "").strip()
            if not content: continue
            if SYS_PAT.match(content): continue
            try: ts = datetime.strptime(t.strip(), "%Y-%m-%d %H:%M:%S")
            except ValueError: continue
            rows.append((ts, sender, content))
    # 完全重复（时间+内容）去重，保序
    seen = set(); ded = []
    for r in rows:
        k = (r[0], r[2])
        if k in seen: continue
        seen.add(k); ded.append(r)
    ded.sort(key=lambda r: r[0])
    return ded

def sender_hash(sess, name):
    return "S" + hashlib.sha256((sess + "\x00" + name).encode("utf-8")).hexdigest()[:10]

report = {}
smap = {}
for sess in SESSIONS:
    path = os.path.join(SRC, sess + ".csv")
    rows = clean_rows(path, sess)
    n0 = len(rows)
    uniq_senders = sorted({r[1] for r in rows})
    for nm in uniq_senders:
        smap[sess + "/" + nm] = sender_hash(sess, nm)
    # 80/20 时间切分
    k = int(len(rows) * 0.8)
    splits = {"train": rows[:k], "test": rows[k:]}
    lens = [len(r[2]) for r in rows]
    days = (rows[-1][0] - rows[0][0]).days or 1
    per_day = len(rows) / days
    gaps = [(rows[i+1][0] - rows[i][0]).total_seconds() for i in range(len(rows)-1)]
    gaps.sort()
    def pct(a, p): return a[min(len(a)-1, int(len(a)*p))]
    srep = dict(messages=n0, start=str(rows[0][0]), end=str(rows[-1][0]),
                span_days=days, msgs_per_day=round(per_day, 1),
                unique_senders=len(uniq_senders),
                char_len_mean=round(sum(lens)/len(lens), 1),
                char_len_p50=pct(sorted(lens), .5), char_len_p90=pct(sorted(lens), .9),
                gap_s_p50=round(pct(gaps, .5), 1), gap_s_p90=round(pct(gaps, .9), 1),
                gap_s_p99=round(pct(gaps, .99), 1),
                emoji_line_rate=round(sum(1 for r in rows if re.fullmatch(r'(\[[^\]]{1,8}\]|\d{1,4}|[哈嘿呵哦嗯嗯]{1,6}|6+|。+|！+|？+)', r[2]))/n0, 3),
                train_n=len(splits["train"]), test_n=len(splits["test"]))
    for sp, rr in splits.items():
        fp = os.path.join(OUT, "%s__%s.csv" % (sp, sess))
        with open(fp, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time", "sender", "content"])
            for ts, nm, ct in rr:
                w.writerow([ts.strftime("%Y-%m-%d %H:%M:%S"), smap[sess + "/" + nm], ct])
    report[sess] = srep
    print(sess, json.dumps(srep, ensure_ascii=False), flush=True)

json.dump(report, open(os.path.join(ROOT, "results", "corpus_quality.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(smap, open(os.path.join(OUT, "_sender_map.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
tot = sum(v["messages"] for v in report.values())
print("TOTAL messages:", tot)
print("saved results/corpus_quality.json + bench/corpus/*.csv")
