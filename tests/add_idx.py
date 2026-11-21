import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(ROOT, "docs", "MODEL.md")
s = open(p, encoding="utf-8").read()
idx = open(os.path.join(ROOT, "docs", "_idx.md"), encoding="utf-8").read()
if "## 1.1 九个子系统与模块编号" not in s:
    k = s.index("## 2. 七条设计原则")
    s = s[:k] + idx.lstrip(chr(10)) + chr(10) + s[k:]
    open(p, "w", encoding="utf-8").write(s)
    print("index inserted; all 9 ids present:", all(x in s for x in ["M1_SEP","M2_EPI","M3_CTX","M4_CON","M5_SEM","M6_WM","M7_NEU","M8_FOR","M0_META"]))
else:
    print("already present")
