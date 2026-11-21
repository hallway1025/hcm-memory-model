#!/usr/bin/env python3
"""check_p10_clipping.py — verify §3.10 pseudocode lines are not physically clipped.

Checks that the pseudocode block in §3.10 of the ARX manuscript contains no
lines longer than the printable width (80 chars), which would cause physical
clipping in the PDF render.
"""
import os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARX = os.path.join(ROOT, "paper", "submission", "arxiv", "MANUSCRIPT-ARXIV.md")
ENF = os.path.join(ROOT, "paper", "PAPER_EN_final.md")

MAX_LINE = 80

def check_file(path, label):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    lines = text.split("\n")
    inBlock = False
    issues = []
    n_lines = 0
    for i, line in enumerate(lines):
        if "~~~text" in line:
            inBlock = True
            continue
        if inBlock and "~~~" in line:
            inBlock = False
            continue
        if inBlock and line.strip():
            n_lines += 1
            if len(line.rstrip()) > MAX_LINE:
                issues.append(f"  L{i+1} ({len(line.rstrip())}c): {line.strip()[:100]}")
    if issues:
        print(f"[{label}] FAIL: {len(issues)} pseudocode lines exceed {MAX_LINE} chars")
        for iss in issues:
            print(iss)
        return False
    else:
        print(f"[{label}] OK: {n_lines} pseudocode lines, all <= {MAX_LINE} chars")
        return True

def main():
    ok1 = check_file(ARX, "ARX")
    ok2 = check_file(ENF, "ENF")
    if ok1 and ok2:
        print("\nCHECK PASSED: no pseudocode line exceeds printable width")
        sys.exit(0)
    else:
        print("\nCHECK FAILED: pseudocode clipping detected")
        sys.exit(1)

if __name__ == "__main__":
    main()
