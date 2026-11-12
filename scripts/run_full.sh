#!/usr/bin/env bash
# Recompute every result in results/ from the recorded data (episodes/, fullsuite/, episodes_flaky/,
# runs/exec_order/). Takes roughly half an hour, needs no Docker and no LLM. It OVERWRITES results/*.json;
# the numbers are deterministic (seeded), only the "generated" timestamps change.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

step() { echo; echo "=== $*"; }

step "constructed worst-case streams and operating curve  -> results/worst_case.json, operating_curve.json"
"$PY" run_sim.py
step "evidence-reuse construction                          -> results/reuse_demo.json"
"$PY" reuse_demo.py
step "replay of every rule on the recorded episodes        -> results/results.json, charts"
"$PY" replay.py
step "extra analyses (spending baselines, full suite, real-flaky arm, ...) -> results/extra.json"
"$PY" extra_analyses.py
step "mixture wallet, gamma/rho sweep, KL floor            -> results/power_tuning.json"
"$PY" power_tuning.py
step "replay-pool and label-uncertainty checks             -> results/review2.json"
"$PY" review2.py
step "independence in execution order                      -> results/exec_order.json"
"$PY" exec_order.py analyze
step "paper numbers, tables and figures                    -> ${SEQACCEPT_PAPER_DIR:-paper_out}"
"$PY" paper_numbers.py
echo; echo "done"
