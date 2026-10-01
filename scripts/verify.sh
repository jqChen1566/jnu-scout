#!/usr/bin/env bash
# Unified verification for JNUScout.
#   1. syntax gate: compile every source file (catches version-floor syntax)
#   2. test suite: pytest
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== syntax gate (compileall) =="
python -m compileall -q src scripts tests
echo "   OK"

echo "== test suite =="
python -m pytest -q
