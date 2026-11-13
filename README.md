# SeqAccept artifact

Code and recorded data for the paper *Knowing When to Stop: Anytime-Valid Patch Acceptance under Adaptive Retries and Stochastic Test Runs*.

SeqAccept decides when an LLM repair loop may stop at a passing test run, with a guarantee that holds across adaptively proposed candidate patches. StopSafe-SE is the dataset of recorded repair loops (4 LLMs, 17 planted bugs in 10 Python repositories) the rules are replayed on.

## Contents

| Path | What it holds |
|---|---|
| `rules.py`, `sim.py` | acceptance rules (B1-B6, SeqAccept, spending baselines) and simulated streams |
| `replay.py`, `extra_analyses.py`, `power_tuning.py`, `review2.py`, `exec_order.py` | offline replay and analyses |
| `paper_numbers.py` | every number, table and figure of the paper, generated from `results/` |
| `run_block.py`, `collect.py`, `agent.py`, `llm.py`, `build_episode.py`, `run_all.py`, `online.py`, `fullsuite.py`, `flaky_arm.py` | Docker and LLM harness that recorded the data |
| `tasks/`, `bugs/`, `repos/`, `screening/` | task definitions, original and planted-bug sources, subject Dockerfiles, IDoFT screening |
| `streams/`, `episodes/`, `episodes_flaky/`, `fullsuite/`, `online/`, `runs/exec_order/` | recorded data |
| `results/` | recorded results |
| `tests/`, `scripts/` | 269 tests, run scripts |

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

## Reproduce all results (about 15 to 20 minutes, no Docker, no LLM)

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
uv run python exec_order.py run                    # Docker only: execution-order independence check
```

Re-screening IDoFT needs its `py-data.csv` (`run_all.py --idoft PATH`). Run logs, block checkpoints and superseded results are not part of this repository.
