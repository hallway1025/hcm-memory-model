#!/usr/bin/env bash
# 创建隔离 venv 并安装钉版依赖。用法：bash repro/setup.sh（可先 export PYTHON=python3.12 指定解释器）
# 镜像：默认 pypi.org；境内网络可 export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
cd "$ROOT"
if [ ! -x ".venv-repro/bin/python" ]; then
    echo "[setup] creating .venv-repro with $PY"
    $PY -m venv .venv-repro
fi
PIP_ARGS=(install --quiet --disable-pip-version-check -r repro/requirements.txt)
if [ -n "${PIP_INDEX_URL:-}" ]; then PIP_ARGS+=(--index-url "$PIP_INDEX_URL"); fi
echo "[setup] installing pinned requirements (index: ${PIP_INDEX_URL:-pypi.org})"
.venv-repro/bin/pip "${PIP_ARGS[@]}"
echo "[setup] done. 下一步：bash repro/run_repro.sh"
