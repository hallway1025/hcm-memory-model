
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(ROOT, "paper", "PAPER.md")
c4 = open(os.path.join(ROOT, "paper", "_chunk4.md"), encoding="utf-8").read()
s = open(p, encoding="utf-8").read()
if "## 4 实验" not in s:
    s = s.rstrip() + "\n\n" + c4
open(p, "w", encoding="utf-8").write(s)
print("chars", len(s), "lines", s.count("\n"))
print([l for l in s.split("\n") if l.startswith("## ")])
