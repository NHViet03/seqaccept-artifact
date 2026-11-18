#!/usr/bin/env bash
# Recompute every result in results/ from the recorded data (episodes/, fullsuite/, episodes_flaky/,
# runs/exec_order/). Takes roughly 45 to 60 minutes, needs no Docker and no LLM. It OVERWRITES results/*.json;
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
step "replay-pool and label-uncertainty checks             -> results/label_risk.json"
"$PY" label_risk.py
step "independence in execution order                      -> results/exec_order.json"
"$PY" exec_order.py analyze
step "refund ablation, shortcut, truth vs pool, cost, operating curve -> results/ablations.json"
"$PY" ablations.py
step "prefilter, refund at cap, per bug, competitors      -> results/comparators.json"
"$PY" comparators.py
step "clustered intervals, near-eps dependence, eps from L, calibrated cost -> results/intervals.json"
"$PY" intervals.py
step "attempt caps, adversary, dependence to FAR, break-even, per task -> results/caps_adversary.json"
"$PY" caps_adversary.py
step "cost with calibration, block size, regime map, pool scoring -> results/sensitivity.json"
"$PY" sensitivity.py
step "invariant, runs-test power, paired intervals, caps, no replacement -> results/checks.json"
"$PY" checks.py
step "cap ablation: shared cap against caps scaled with log(1/alpha_j) -> results/slice_caps.json"
"$PY" slice_caps.py
step "round-7 checks: near-eps comparators and labels, cap 10,000, refund crossing, redrawn eps, start-up cost -> results/round7.json"
"$PY" round7_checks.py
step "paper numbers, tables and figures                    -> ${SEQACCEPT_PAPER_DIR:-paper_out}"
"$PY" paper_numbers.py
echo; echo "done"
