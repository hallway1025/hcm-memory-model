import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(ROOT, "paper", "PAPER.md")
s = open(p, encoding="utf-8").read()
sec = open(os.path.join(ROOT, "paper", "_sec411.md"), encoding="utf-8").read()
if "### 4.11" not in s:
    k = s.index("## 5 讨论")
    s = s[:k] + sec + s[k:]
    open(p, "w", encoding="utf-8").write(s)
    print("4.11 inserted; chars", len(s))
else:
    print("already present")
