#!/usr/bin/env python3
"""export_docx.py —— 把 paper/PAPER_final.md 导出为 DOCX（含表格与插图）。

支持：# 标题、**粗体**、*斜体*、- 列表、| 表格 |、![alt](path) 图片、~~~ 代码块、--- 分隔线。
"""
import os, re, sys
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "paper", "PAPER_final.md")
OUT = os.path.join(ROOT, "paper", "HCM_PAPER.docx")
IMG = re.compile(r"!\[(.*?)\]\((.*?)\)")
BOLD = re.compile(r"\*\*(.+?)\*\*")


def set_font(run, name="Microsoft YaHei", size=None, bold=None, italic=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def add_runs(par, text, base_size=10.5):
    pos = 0
    for m in BOLD.finditer(text):
        if m.start() > pos:
            r = par.add_run(text[pos:m.start()])
            set_font(r, size=base_size)
        r = par.add_run(m.group(1))
        set_font(r, size=base_size, bold=True)
        pos = m.end()
    if pos < len(text):
        r = par.add_run(text[pos:])
        set_font(r, size=base_size)


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else SRC
    outp = sys.argv[2] if len(sys.argv) > 2 else OUT
    lines = open(src, encoding="utf-8").read().split("\n")
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Microsoft YaHei"
    st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")

    i = 0
    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln.strip():
            i += 1
            continue

        m = IMG.fullmatch(ln.strip())
        if m:
            path = os.path.join(ROOT, "paper", m.group(2))
            if os.path.exists(path):
                doc.add_picture(path, width=Cm(15.5))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            i += 1
            if i < len(lines) and lines[i].strip().startswith("*图"):
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                add_runs(p, lines[i].strip().strip("*"), base_size=9)
                i += 1
            continue

        if ln.startswith("~~~"):
            i += 1
            buf = []
            while i < len(lines) and not lines[i].startswith("~~~"):
                buf.append(lines[i].rstrip())
                i += 1
            i += 1
            p = doc.add_paragraph()
            for k, seg in enumerate(buf):
                r = p.add_run(seg + ("\n" if k else ""))
                set_font(r, name="Consolas", size=8.5)
            continue

        if ln.startswith("#"):
            lvl = len(ln) - len(ln.lstrip("#"))
            text = ln[lvl:].strip()
            p = doc.add_heading(level=min(3, max(1, lvl - 1)))
            r = p.add_run(text)
            set_font(r, size=16 if lvl == 1 else 13 if lvl == 2 else 11.5, bold=True)
            if lvl == 1:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            i += 1
            continue

        if ln.strip() == "---":
            p = doc.add_paragraph()
            r = p.add_run("—" * 20)
            set_font(r, size=8)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            i += 1
            continue

        if ln.lstrip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:\-|]+\|\s*$", lines[i + 1]):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                raw = lines[i].strip().strip("|")
                cells = [c.strip().replace("**", "") for c in raw.split("|")]
                if not re.match(r"^[\s:\-]+$", "".join(cells)):
                    rows.append(cells)
                i += 1
            if rows:
                ncol = max(len(r) for r in rows)
                table = doc.add_table(rows=0, cols=ncol)
                table.style = "Table Grid"
                table.alignment = WD_TABLE_ALIGNMENT.CENTER
                for ri, row in enumerate(rows):
                    cells = table.add_row().cells
                    for ci in range(ncol):
                        txt = row[ci] if ci < len(row) else ""
                        p = cells[ci].paragraphs[0]
                        add_runs(p, txt, base_size=8.5)
                        if ri == 0:
                            for r_ in p.runs:
                                r_.bold = True
            continue

        if re.match(r"^\s*[-*]\s+", ln) or re.match(r"^\d+\.\s+", ln):
            p = doc.add_paragraph(style="List Bullet" if not re.match(r"^\d", ln.strip()) else "List Number")
            add_runs(p, re.sub(r"^\s*([-*]|\d+\.)\s+", "", ln))
            i += 1
            continue

        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.space_after = Pt(4)
        add_runs(p, ln.strip())
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        i += 1

    doc.save(outp)
    print("wrote", outp, os.path.getsize(outp) // 1024, "KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
