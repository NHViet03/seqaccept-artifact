# SeqAccept artifact

Code and recorded data for the paper *Knowing When to Stop: Anytime-Valid Patch Acceptance under Adaptive Retries and Stochastic Test Runs*.

**Version.** The paper describes the tag `ease2027-submission` (EASE 2027 revision of 2026-11-18). Its supplement is `supplement.pdf` in this folder; the supplement's last table maps every table, figure and table row of the paper to its script and output file.

SeqAccept decides when an LLM repair loop may stop at a passing test run, with a guarantee that holds across adaptively proposed candidate patches. StopSafe-SE is the dataset of recorded repair loops (4 LLMs, 17 planted bugs in 10 Python repositories) the rules are replayed on.

## Contents

| Path | What it holds |
|---|---|
| `rules.py`, `sim.py` | acceptance rules (B1-B6, SeqAccept, spending baselines) and simulated streams |
| offline replay and analyses | see *Analysis scripts* below |
| `paper_numbers.py` | every number, table and figure of the paper, generated from `results/` |
| `run_block.py`, `collect.py`, `agent.py`, `llm.py`, `build_episode.py`, `run_all.py`, `online.py`, `fullsuite.py`, `flaky_arm.py`, `near_eps_arm.py` | Docker and LLM harness that recorded the data |
| `tasks/`, `bugs/`, `repos/`, `screening/` | task definitions, original and planted-bug sources, subject Dockerfiles, IDoFT screening |
| `streams/`, `episodes/`, `episodes_flaky/`, `episodes_neareps/`, `fullsuite/`, `online/`, `runs/exec_order/` | recorded data |
| `results/` | recorded results |
| `supplement.pdf` | the paper's supplement (proofs, per-task tables, robustness checks, result-to-script map) |
| `tests/`, `scripts/` | tests, run scripts |

## Analysis scripts

All run offline from the recorded data; each writes one file in `results/`. The supplement's last table maps every table, figure and table row of the paper to the script behind it.

| Script (output) | What it computes |
|---|---|
| `replay.py` (`results.json`) | main replay of B1-B6 and SeqAccept on all episodes: FAR, FRR, cost, splits, late streams |
| `extra_analyses.py` (`extra.json`) | spending comparators, late and mixed streams, worst case, full-suite and real-flaky arms, first-green cut |
| `ablations.py` (`ablations.json`) | refund ablation, truth against pool rates, operating curve, cost frontier, shortcut re-split |
| `comparators.py` (`comparators.json`) | comparators on natural loops, bug-level intervals, leave one repository out, prefilter and refund-at-cap variants |
| `intervals.py` (`intervals.json`) | bug- and repository-clustered intervals, near-eps dependence, ordered replay, eps from a lower bound on p0, calibrated cost |
| `caps_adversary.py` (`caps_adversary.json`) | Bonferroni with attempt caps, white-box adversary, dependence to FAR, full-suite break-even, per-task real-flaky table |
| `sensitivity.py` (`sensitivity.json`) | cost with calibration per task and per loop, block-size sweep, regime map, FAR against pool rates, repository intervals, comparators on the real-flaky arm, margin curve |
| `checks.py` (`checks.json`) | the proof's invariant in the adversary, runs-test power, paired intervals on FRR and cost, Bonferroni caps up to 10,000, replay without replacement, abstention without the block cap, dropped tasks, run-to-run drift |
| `slice_caps.py` (`slice_caps.json`) | the shared block cap against caps scaled with log(1/alpha_j) |
| `design_checks.py` (`design_checks.json`) | mixture wallet and gamma 3/4 on every arm, harmonic spending with the refund, fixed-sample Bonferroni sized for power, Corollary 4.3 at total level 0.05, near-eps exceedances by lower bound |
| `near_eps_checks.py` (`near_eps_checks.json`) | near-eps arm with every comparator and the earlier settings, near-eps labels, a cap of 10,000, refund crossing against harmonic spending, eps redrawn from calibration blocks, start-up cost in the cost model |
| `deploy_checks.py` (`deploy_checks.json`) | refund share and gamma under dependent blocks, deployment configurations in matched conditions (eps = L + 0.05 at alpha 0.04), on-policy prefix of the recorded streams, chosen against unchosen IDoFT pairs, comparator latency, break-even dependence, yamicache and near-eps false acceptances per candidate, discarding on constructed streams and against an adversary, exposure to candidates near eps |
| `neareps_truth.py` (`neareps_truth.json`, Docker) | 2,000 extra blocks for each of the 60 near-eps candidates within 3 points of eps; `near_eps_checks.py` step 7 relabels the arm with them |
| `power_tuning.py`, `label_risk.py`, `exec_order.py`, `run_sim.py`, `reuse_demo.py` | mixture wallet and gamma/rho sweep; pool size and label risk; execution-order check; simulated worst case; Theorem 4.4 simulation |

## Setup

Python 3.12 or newer.

```sh
uv sync                                            # or: python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
```

## Quick check (about a minute, no Docker, no LLM)

```sh
uv run bash scripts/run_smoke.sh                   # with pip: activate the venv, then bash scripts/run_smoke.sh
```

Runs the tests and regenerates the paper's numbers, tables and figures into `paper_out/` (set `SEQACCEPT_PAPER_DIR` to write elsewhere).

## Reproduce all results (about 45 to 60 minutes, no Docker, no LLM)

```sh
uv run bash scripts/run_full.sh
```

Recomputes every file in `results/` from the recorded data. Runs are seeded, so the numbers are identical to the recorded ones; only the `generated` timestamps change.

## Re-collecting data (optional)

Needs Docker. Agent runs also need an LLM: put credentials in a `.env` file (it is git-ignored, never commit it) with `LLM_API_KEY`, `LLM_ENDPOINT`, `LLM_DEPLOYMENT_NAME`, `LLM_API_VERSION` for Azure, and a local Ollama for llama3.1:8b.

```sh
uv run --env-file .env python run_all.py --help    # stage 1-5 driver: tasks, agent streams, episodes, replay, IDoFT screening
uv run --env-file .env python online.py            # SeqAccept in the loop with a live agent
uv run python fullsuite.py                         # Docker only: full-suite runs of the recorded candidates
uv run python flaky_arm.py                         # Docker only: real flaky tests kept
uv run python near_eps_arm.py --plan                # Docker only (without --plan): recorded loops on smaller blocks, broken candidates near eps
uv run python neareps_truth.py --plan               # Docker only (without --plan): 2,000 extra truth blocks per near-eps candidate, for firmer labels
uv run python neareps_truth.py --plan --episodes episodes_flaky --task yamicache@realflaky --out results/flaky_truth.json   # Docker only (without --plan): the same for yamicache with flaky tests kept; deploy_checks.py step 14 rescores it
uv run python exec_order.py run                    # Docker only: execution-order independence check
```

Re-screening IDoFT needs its `py-data.csv` (`run_all.py --idoft PATH`). Run logs, block checkpoints and superseded results are not part of this repository.
