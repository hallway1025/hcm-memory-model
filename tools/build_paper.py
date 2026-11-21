#!/usr/bin/env python3
"""build_paper.py —— 把 {{AuthorYear}} 记号编译为编号引用 + 参考文献表 + 模块证据基座。

铁律：未匹配的 token 一律硬失败（exit 1），因此论文里不可能出现 refs.jsonl 之外的引文。
"""
import json, os, re, sys, unicodedata, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFS = os.path.join(ROOT, "refs", "refs.jsonl")
SRC = os.path.join(ROOT, "paper", "PAPER.md")
OUT = os.path.join(ROOT, "paper", "PAPER_final.md")
TOKEN = re.compile(r"\{\{([A-Za-z\u00C0-\u024F]+?)(\d{4})\}\}")
RANK = {"T1": 0, "T2": 1, "T3": 2}


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c for c in s.lower() if c.isalnum())


def load_refs() -> list:
    return [json.loads(l) for l in open(REFS, encoding="utf-8") if l.strip()]


def fmt_ref(num: int, r: dict) -> str:
    fa = r.get("first_author") or ""
    n = r.get("n_authors") or 0
    who = f"{fa} 等" if n and n > 2 else fa
    title = (r.get("title") or "").strip().rstrip(".")
    line = f"[{num}] {who} ({r.get('year')}). {title}. *{r.get('journal')}*."
    if r.get("doi"):
        line += f" doi:{r['doi']}"
    if r.get("pmid"):
        line += f" PMID:{r['pmid']}"
    return line + f" 〔{r.get('tier')}〕"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--src", default=None, help="override source md (default paper/PAPER.md)")
    ap.add_argument("--out", default=None, help="override output md (default paper/PAPER_final.md)")
    args = ap.parse_args()

    global SRC, OUT
    if args.src:
        SRC = os.path.join(ROOT, args.src)
    if args.out:
        OUT = os.path.join(ROOT, args.out)

    refs = load_refs()
    by_key = {}
    for r in refs:
        fa, yr = fold(r.get("first_author") or ""), str(r.get("year") or "")
        if fa:
            by_key.setdefault((fa, yr), []).append(r)

    text = open(SRC, encoding="utf-8").read()
    tokens = TOKEN.findall(text)
    uniq = []
    for t in tokens:
        if t not in uniq:
            uniq.append(t)

    unresolved, chosen = [], {}
    for surname, year in uniq:
        k = fold(surname)
        cands = []
        for (fa, yr), lst in by_key.items():
            if yr == year and (fa in k or k in fa):
                cands.extend(lst)
        if not cands:
            unresolved.append(surname + year)
            continue
        cands.sort(key=lambda r: (0 if fold(r.get("first_author") or "") == k else 1, RANK.get(r.get("tier"), 3), len(r.get("title") or "")))
        chosen[(surname, year)] = cands[0]
        if len(cands) > 1:
            print("  [warn] %s%s matched %d refs, chose %s (%s)" %
                  (surname, year, len(cands), cands[0]["ref_id"], cands[0].get("tier")))

    if unresolved:
        print("[FAIL] %d unresolved citation token(s): %s" % (len(unresolved), ", ".join(unresolved)))
        return 1
    print("tokens ok: %d unique keys resolved" % len(uniq))
    if args.check:
        return 0

    order, numbering = [], {}
    for t in tokens:
        if t not in numbering:
            order.append(t)
            numbering[t] = len(order)

    out = TOKEN.sub(lambda m: "[%d]" % numbering[(m.group(1), m.group(2))], text)
    out = out.replace("{{NCITE}}", str(len(refs)))

    # EN 源输出英文标题（update_cited_ids.py 按 "## References" 定位文献区）
    is_en = "PAPER_EN" in os.path.basename(SRC)
    refs_heading = "## References" if is_en else "## 参考文献"
    appx_heading = ("## Appendix D: module evidence base (auto-generated)\n"
                    "Per model module, the highest-authority sources in the citation bank (up to 8 each, T1 first)."
                    ) if is_en else "## 附录 D：模块证据基座（自动生成）\n"
    appx_note = ("This table does NOT claim each reference directly supports a specific claim of this paper; "
                 "it only reports evidence-source coverage per module. Full fields (DOI, PMID, verification "
                 "status, tier) live in refs/refs.jsonl and audit/citations-report.md.\n"
                 ) if is_en else (
                 "下表按模型模块列出文献库中权威度最高的证据来源（每模块至多 8 条，T1 优先）。"
                 "此表**不声称**每篇文献直接支持本文的具体论断，只说明每个模块的证据来源覆盖情况；"
                 "完整字段（DOI、PMID、核验状态、证据等级）见 refs/refs.jsonl 与 audit/citations-report.md。\n")

    parts = [out, "\n---\n", refs_heading + "\n"]
    for t in order:
        parts.append(fmt_ref(numbering[t], chosen[t]))
        parts.append("")
    parts.append("\n---\n")
    parts.append(appx_heading + "\n")
    if not is_en:
        parts.append(appx_note)

    mods = {}
    for r in refs:
        mods.setdefault(r.get("module") or "UNSORTED", []).append(r)
    for m in sorted(mods):
        rows = sorted(mods[m], key=lambda r: (RANK.get(r.get("tier"), 3), -(r.get("year") or 0)))
        parts.append("**%s**（%d 条，此处列 %d 条）\n" % (m, len(mods[m]), min(8, len(rows))))
        for r in rows[:8]:
            parts.append("- %s (%s). %s. *%s*. 〔%s〕" % (
                r.get("first_author"), r.get("year"), (r.get("title") or "")[:110],
                r.get("journal"), r.get("tier")))
        parts.append("")

    open(OUT, "w", encoding="utf-8").write("\n".join(parts))
    print("wrote %s: %d refs, %d chars" % (OUT, len(order), sum(len(p) for p in parts)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
