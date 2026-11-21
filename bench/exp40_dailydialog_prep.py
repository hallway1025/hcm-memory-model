# -*- coding: utf-8 -*-
'''exp40_dailydialog_prep.py —— 公开语料 DailyDialog 预处理（exp40 第一步）。

输入：bench/corpus_public/{train,validation}/train/dialogues_train.txt（HF roskoN/dailydialog，原始 DailyDialog,
      Li et al. 2017，MIT 级开放许可；一行 = 一段对话，话轮以 __eou__ 分隔）
输出：bench/corpus_public/{train,test}__dailydialog.csv —— 三列（时间,发送者,内容），与 bench/corpus/ 同构：
  - 时间 = 合成单调时间戳：对话 i 的基准 = i*10000s（epoch 起点 2020-01-01），话轮间隔 2s
    → "相邻话轮" 自然落进 gap≤30s 判据（兼容 exp27 探针协议）
  - 发送者 = A/B 交替（DailyDialog 为双人对话），无隐私问题（公开数据，不哈希也合规，仍做 A/B 化）
  - 清洗：去空话轮、去 __eou__ 残留、去 __eot__/__eoc__ 标记
  - 切分：按对话 80/20（train = 前 80% 对话，test = 后 20% 对话，从未写入）
质量统计写 results/corpus_quality_dailydialog.json（论文可引用）。
'''
import csv, json, os, re
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "bench", "corpus_public", "train", "train", "dialogues_train.txt")
OUT = os.path.join(ROOT, "bench", "corpus_public")
os.makedirs(OUT, exist_ok=True)

EOT_PAT = re.compile(r"__eot__|__eoc__")

def parse():
    dialogues = []
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            turns = [EOT_PAT.sub("", t).strip() for t in line.split("__eou__")]
            turns = [t for t in turns if t]
            if len(turns) >= 2:
                dialogues.append(turns)
    return dialogues

def main():
    dialogues = parse()
    base = datetime(2020, 1, 1)
    k = int(len(dialogues) * 0.8)
    splits = {"train": dialogues[:k], "test": dialogues[k:]}
    report = {"source": "DailyDialog (Li et al. 2017) via HF roskoN/dailydialog",
              "license": "open (research use, CC BY 4.0 per dataset card)",
              "n_dialogues_total": len(dialogues)}
    for name, dgs in splits.items():
        rows = []
        for i, turns in enumerate(dgs):
            base_i = base + timedelta(seconds=i * 10000)
            for j, t in enumerate(turns):
                ts = base_i + timedelta(seconds=2 * j)
                rows.append((ts.strftime("%Y-%m-%d %H:%M:%S"), "A" if j % 2 == 0 else "B", t))
        path = os.path.join(OUT, "%s__dailydialog.csv" % name)
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["时间", "发送者", "内容"])
            w.writerows(rows)
        lens = [len(r[2]) for r in rows]
        nturn = [len(d) for d in dgs]
        report[name] = dict(dialogues=len(dgs), messages=len(rows),
                            turns_per_dialogue_mean=round(sum(nturn)/len(nturn), 2),
                            char_len_mean=round(sum(lens)/len(lens), 1),
                            unique_chars=len({ch for r in rows for ch in r[2]}))
        print(name, report[name])
    with open(os.path.join(ROOT, "results", "corpus_quality_dailydialog.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("OK ->", OUT)

if __name__ == "__main__":
    main()
