import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(ROOT, "paper", "PAPER.md")
s = open(p, encoding="utf-8").read()
sec = open(os.path.join(ROOT, "paper", "_sec96.md"), encoding="utf-8").read()
if "### 9.6" not in s:
    k = s.index("## 附录 A")
    s = s[:k] + sec + s[k:]
    open(p, "w", encoding="utf-8").write(s)
    print("9.6 inserted; chars", len(s))
else:
    print("already present")
