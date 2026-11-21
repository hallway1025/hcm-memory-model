
import os, re
from docx import Document
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = Document(os.path.join(ROOT, "paper", "HCM_PAPER.docx"))
print("paragraphs", len(d.paragraphs), "tables", len(d.tables))
print("inline shapes", len(d.inline_shapes))
txt = "\n".join(p.text for p in d.paragraphs)
print("chars", len(txt))
print("headings:", [p.text[:40] for p in d.paragraphs if p.style.name.startswith("Heading")][:14])
md = open(os.path.join(ROOT,"paper","PAPER_final.md"), encoding="utf-8").read()
print("leftover tokens:", re.findall(r"\{\{[^}]*\}\}", md))
print("ncite:", "NCITE" in md)
print("citations numbered:", len(re.findall(r"\[\d+\]", md)))
print("tables md:", md.count("|---"))
