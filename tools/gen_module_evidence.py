# -*- coding: utf-8 -*-
"""gen_module_evidence.py —— 由 refs/refs.jsonl 生成 audit/module_evidence.md（模块证据基座）。

此前该文件是 2026-11-09 一次性写入的静态文件（仓库内无生成脚本），此后引文库从
436 条增长到 448 条、50 条记录补齐 verified、8 条补齐 tier，文件因此陈旧：
计数停在 447、缺 R445、8 处 "[undefined]" 分级占位、若干条目因旧乱码（â€“ / RacsmÃ¡ny）
与首作者缺失写成 "?"。本脚本把生成规则固化成代码，使"auto-generated"名副其实。

规则（与 v1.3 一次性写入的文件一致，可逐条对照）：
  * 分区顺序：按模块名升序（M0_META … M8_FOR, SYSTEM, related_work）。
  * 组内排序：先按证据等级 T1 < T2 < T3，同级再按 ref_id 字符串升序。
  * 条目格式：- [R254] FirstAuthor (year). Title *Journal*. [T1] ✓
      - 首作者缺失写 "?"；年份缺失写 "null"；verified == "verified" 才带 ✓。
      - 标题内的换行/多余空白折叠为单空格（保持一行一条）。
  * 分区标题：## MODULE (N records, M T1)，N 为该模块总数，M 为其中 T1 条数。
  * 首行：# Module evidence base (auto-generated from refs/refs.jsonl, N records)

用法：py -X utf8 tools/gen_module_evidence.py [--check]
  --check 只比对不写盘，文件与预期不一致时退出码非 0。
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFS = os.path.join(ROOT, "refs", "refs.jsonl")
OUT = os.path.join(ROOT, "audit", "module_evidence.md")

TIER_RANK = {"T1": 0, "T2": 1, "T3": 2}

HEADER = ("# Module evidence base (auto-generated from refs/refs.jsonl, %d records)\n"
          "\n"
          "Per-model-module evidence coverage in the citation bank, sorted T1 first, then by reference id. "
          "Fields (DOI, PMID, verification status, evidence level) live in refs/refs.jsonl; "
          "verification pipeline in audit/citations-report.md. This table does NOT claim each reference "
          "directly supports a specific claim of the paper; it documents evidence coverage per module.\n")


def load_refs():
    with open(REFS, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def one_line(text):
    """折叠换行与多余空白，保证一条记录一行。"""
    return re.sub(r"\s+", " ", (text or "")).strip()


def entry(r):
    tier = r.get("tier") or "UNRATED"
    fa = one_line(r.get("first_author")) or "?"
    year = r.get("year")
    year = "null" if year is None else str(year)
    title = one_line(r.get("title")).rstrip(".")
    journal = one_line(r.get("journal"))
    line = "- [%s] %s (%s). %s" % (r.get("ref_id"), fa, year, title)
    if journal:
        line += " *%s*." % journal
    line += " [%s]" % tier
    if (r.get("verified") or "") == "verified":
        line += " ✓"
    return line


def render(refs):
    parts = [HEADER % len(refs)]
    mods = {}
    for r in refs:
        mods.setdefault(r.get("module") or "UNSORTED", []).append(r)
    for m in sorted(mods):
        group = mods[m]
        rows = sorted(group, key=lambda r: (TIER_RANK.get(r.get("tier"), 3), r.get("ref_id") or ""))
        n_t1 = sum(1 for r in group if r.get("tier") == "T1")
        parts.append("\n## %s (%d records, %d T1)\n\n" % (m, len(group), n_t1))
        parts.append("\n".join(entry(r) for r in rows))
        parts.append("\n")
    return "".join(parts)


def main():
    check = "--check" in sys.argv
    refs = load_refs()
    text = render(refs)
    if check:
        cur = open(OUT, encoding="utf-8").read() if os.path.exists(OUT) else ""
        if cur == text:
            print("[OK] audit/module_evidence.md up to date (%d records)" % len(refs))
            return 0
        print("[FAIL] audit/module_evidence.md 与 refs/refs.jsonl 不同步"
              "（当前 %d 字符，预期 %d 字符）" % (len(cur), len(text)))
        return 1
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print("wrote %s: %d records, %d chars" % (OUT, len(refs), len(text)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
