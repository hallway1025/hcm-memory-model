#!/usr/bin/env bash
# HCM v1.2 一键复现入口（Linux/macOS）：环境 -> 25 步全流程 -> 三守卫 -> 哈希对账 -> verdict
# 任一步失败即非零退出。验收口径见 repro/README.md
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv-repro/bin/python"
if [ ! -x "$PY" ]; then bash "$ROOT/repro/setup.sh"; fi
echo "[repro] step 1/3: reproduce_all.py（25 步全流程）"
"$PY" -u bench/reproduce_all.py
echo "[repro] step 2/3: guards"
"$PY" -u repro/run_guards.py
echo "[repro] step 3/3: 全部完成，verdict 见上方表格"
