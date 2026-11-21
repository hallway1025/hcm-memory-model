import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(ROOT, "memory", "30-LESSONS.md")
s = open(p, encoding="utf-8").read()
add = open(os.path.join(ROOT, "memory", "_l24.md"), encoding="utf-8").read()
if "L-024" not in s:
    open(p, "w", encoding="utf-8").write(s.rstrip() + chr(10) + add)
    print("L-024/L-025 recorded in project memory")
else:
    print("already present")
