
import os, zipfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hcm_deploy.zip")
INCLUDE = ["src", "bench", "tools", "tests", "spec", "docs", "memory"]
EXCL_DIR = {"__pycache__", ".git", "refs", "results", "audit", "paper", "notes"}
EXCL_EXT = {".log", ".pyc"}
n = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for top in INCLUDE:
        base = os.path.join(ROOT, top)
        for dp, dns, fns in os.walk(base):
            dns[:] = [d for d in dns if d not in EXCL_DIR]
            for fn in fns:
                if os.path.splitext(fn)[1].lower() in EXCL_EXT:
                    continue
                p = os.path.join(dp, fn)
                if os.path.getsize(p) > 60_000_000:
                    print("skip big:", p); continue
                z.write(p, os.path.relpath(p, ROOT)); n += 1
print("files:", n, "bytes:", os.path.getsize(OUT))
