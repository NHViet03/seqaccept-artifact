#!/usr/bin/env bash
# Quick check (about a minute, no Docker, no LLM): run the tests, then regenerate every number, table
# and figure of the paper from the recorded results into ./paper_out (or $SEQACCEPT_PAPER_DIR).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

echo "[1/2] unit and conformance tests"
"$PY" -m pytest -q

echo "[2/2] paper numbers, tables and figures"
"$PY" paper_numbers.py
