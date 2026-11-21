#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HCM v1.3 一键复现验收器：跑三个守卫 + SHA256 清单对账，打印 verdict 表。

验收口径（主人批准件 3 底线：外部实验室一键复现 v1.2 守卫全绿）：
  G1  claims      tools/check_claims.py          -> 444/444 PASS
  G2  citations   tools/verify_citations.py verify -> n=447 gate PASS
  G3  closure     tools/update_cited_ids.py      -> entries=69 mapped=69 unmatched=[]
  G4  sha256      audit/SHA256SUMS.txt           -> 全部条目哈希一致
任一 FAIL 即非零退出。
"""
import hashlib, os, re, subprocess, sys, time

# --- 控制台编码兜底：中文 Windows 默认 GBK，子进程输出含替换字符时会崩 ---
try:
    import sys as _sys
    if hasattr(_sys.stdout, "reconfigure"):
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(_sys.stderr, "reconfigure"):
        _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
# ---


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
SUMS = os.path.join(ROOT, "audit", "SHA256SUMS.txt")


def run(name, args, parse=None):
    t0 = time.time()
    r = subprocess.run([PY, "-u"] + args, cwd=ROOT, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    out = r.stdout + r.stderr
    verdict = "PASS" if r.returncode == 0 else "FAIL(rc=%d)" % r.returncode
    detail = parse(out) if parse else ""
    print("[%s] %-12s %s %s  %.1fs" % (verdict, name, "", detail, time.time() - t0))
    return r, out, verdict


def parse_claims(out):
    m = re.search(r"B\. 数字溯源：(\d+)/(\d+) 通过", out)
    if m:
        return "%s/%s" % (m.group(1), m.group(2))
    return out.strip().splitlines()[-1] if out.strip() else "?"


def parse_citations(out):
    m = re.search(r"n=(\d+) tiers=\{[^}]*\} T1\+T2=(\d+\.\d)%", out)
    g = re.search(r"gate: (PASS|FAIL)", out)
    if m and g:
        return "n=%s T1+T2=%s%% gate=%s" % (m.group(1), m.group(2), g.group(1))
    return out.strip().splitlines()[-1] if out.strip() else "?"


def parse_closure(out):
    m = re.search(r"entries=(\d+) mapped=(\d+) unique_ref_ids=(\d+) unmatched=(\[.*?\])", out)
    if m:
        return "entries=%s mapped=%s unmatched=%s" % (m.group(1), m.group(2), m.group(4))
    return out.strip().splitlines()[-1] if out.strip() else "?"


def check_sums():
    if not os.path.isfile(SUMS):
        return "FAIL(no manifest)", 0, 0
    lines = [l for l in open(SUMS, encoding="utf-8").read().splitlines()
             if l and not l.startswith("#")]
    ok, bad, missing = 0, [], []
    for l in lines:
        parts = l.split(None, 1)
        if len(parts) != 2:
            continue
        want, rel = parts
        p = os.path.join(ROOT, rel)
        if not os.path.isfile(p):
            missing.append(rel)
            continue
        got = hashlib.sha256(open(p, "rb").read()).hexdigest()
        if got == want:
            ok += 1
        else:
            bad.append(rel)
    detail = "ok=%d bad=%d missing=%d" % (ok, len(bad), len(missing))
    if bad:
        detail += " bad[:3]=%s" % bad[:3]
    if missing:
        detail += " missing[:3]=%s" % missing[:3]
    return ("PASS" if not bad and not missing else "FAIL"), ok, len(bad) + len(missing)


print("=" * 68)
print("HCM v1.2 一键复现验收（基线 9553bba）")
print("=" * 68)
rows = []

r, out, v = run("G1 claims", ["tools/check_claims.py"], parse_claims)
rows.append(("G1 claims", v, parse_claims(out)))

r, out, v = run("G2 citations", ["tools/verify_citations.py", "verify"], parse_citations)
rows.append(("G2 citations", v, parse_citations(out)))

r, out, v = run("G3 closure", ["tools/update_cited_ids.py"], parse_closure)
rows.append(("G3 closure", v, parse_closure(out)))

vsum, ok, bad = check_sums()
rows.append(("G4 sha256", vsum, "ok=%d bad=%d" % (ok, bad)))

print("-" * 68)
fails = [row for row in rows if row[1] != "PASS"]
for name, v, detail in rows:
    print("%-14s %-6s %s" % (name, v, detail))
print("=" * 68)
if fails:
    print("VERDICT: FAIL (%d/%d 未过)" % (len(fails), len(rows)))
    sys.exit(1)
print("VERDICT: ALL PASS (4/4)")
