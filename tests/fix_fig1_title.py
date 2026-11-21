
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(ROOT, "tools", "make_figures.py")
s = open(p, encoding="utf-8").read()
old = "HCM：互补双系统 + 期望库残差分离 + 优先级回放"
new = "HCM：互补双系统 + 期望库残差分离 + 交错回放"
if old in s:
    s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)
    print("fig1 title fixed ->", new)
else:
    print("title not found; current:", [l for l in s.split(chr(10)) if "双系统" in l])
