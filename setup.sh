#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "Use Python 3.10, 3.11 or 3.12"'
if [ ! -d .venv ]; then
  "$PYTHON_BIN" -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt -c constraints.txt
if [ ! -f .env ]; then
  cp .env.example .env
  chmod 600 .env
fi
echo 'Ready: .venv/bin/python -m navguide'
echo 'Camera: .venv/bin/python -m pip install -r requirements-vision.txt'
