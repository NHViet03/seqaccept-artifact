"""Guard: the code implements the paper (FSE-2027-03-paper.md) and the plan's logged changes, and nothing else.

Fails when:
  - a module, public function or class appears that is not traced to a paper/plan section (TRACE),
  - a constant the paper fixes changes (test_paper_constants),
  - SeqAccept stops matching a line-by-line transcription of Algorithm 1 (test_seq_accept_is_algorithm_1),
  - a deviation from the paper is not listed in DEVIATIONS with its reason.
To add or change behaviour: cite the paper section in TRACE, or log the deviation here, in the same change.
"""
import importlib
import inspect
import math
import random
from pathlib import Path

import pytest

import build_episode
import replay
import rules
import screen_idoft
import sim
from rules import Result, seq_accept

IMPL = Path(__file__).resolve().parent.parent

TRACE = {  # every public function/class in impl/*.py -> where the paper or the plan asks for it
    "rules.Result": "Alg. 1 output: Accept(c_j) or nothing",
    "rules.stop_on_first_pass": "§9.2 B1",
    "rules.rerun_k": "§9.2 B2 rerun-k all green",
    "rules.majority_of_5": "§9.2 B3",
    "rules.peeking_cp": "§9.2 B4 fixed-sample Clopper-Pearson at a data-dependent stopping time (+ D16)",
    "rules.naive_cs": "§9.2 B5 naive confidence sequence, no error spending (+ D16)",
    "rules.bets": "§7 truncated Kelly bet (+ D2 reject wallet)",
    "rules.seq_accept": "§5 Alg. 1; refund=False is §9.2 B6",
    "rules.all_rules": "plan 2.6: the baselines the pilot keeps",
    "rules.fixed_spending": "review round 1: valid no-refund spending baselines (harmonic, Bonferroni) for §10 refunds",
    "rules.harmonic": "review round 1: harmonic schedule alpha / (j (j + 1))",
    "rules.bonferroni": "review round 1: Bonferroni split over the 15-attempt budget",
    "rules.fixed_n": "review round 3: valid fixed-sample binomial test per candidate, Bonferroni slices",
    "rules.binom_cdf": "review round 3: exact binomial CDF for rules.fixed_n",
    "rules.discard_spending": "review round 5: online FWER discarding comparator for refunds",
    "rules.mix_lambdas": "external review: fixed-bet grid of the mixture accept wallet",
    "flaky_arm.idoft_tests": "review round 4: the tests IDoFT lists as flaky for a repository",
    "flaky_arm.flaky_selection": "review round 4: keep known flaky tests, other deselects stay",
    "flaky_arm.make_variant": "review round 4: <bug>@realflaky task with its own test ids and block size",
    "flaky_arm.jobs": "review round 4: every recorded natural loop of the main table",
    "power_tuning.kl": "external review: KL lower bound on blocks to accept",
    "power_tuning.single": "external review: one-candidate operating point, Kelly vs mixture",
    "power_tuning.replay_summary": "external review: replay summary for the mixture wallet and the gamma / rho sweep",
    "review2.counted_replay": "external review 2: replay with a shortened pool, counting draws per candidate",
    "review2.beta_cdf": "external review 2: Beta posterior tail for label uncertainty",
    "review2.label_risk": "external review 2: labels whose posterior puts p on the other side of eps",
    "review2.kl": "external review 2: KL divergence for the margin-planning table",
    "exec_order.done_seeds": "external review 2: resumable execution-order log",
    "exec_order.run_task": "external review 2: blocks of the original code with start/end time, worker and load",
    "exec_order.summarize": "external review 2: independence in execution order vs seed order, busy vs quiet host",
    "exec_order.two_prop_z": "external review 2: z for a difference of two failure rates",
    "flaky_seeds.loops": "external review 2: high-p0 real-flaky loops with a fixed loop index for disjoint seeds",
    "flaky_seeds.grade": "external review 2: re-grade one loop with its own seeds",
    "flaky_seeds.compare": "external review 2: SeqAccept and B4 with shared vs disjoint seeds",
    "fullsuite.jobs": "review round 2: every distinct recorded candidate of the main table (full-suite arm)",
    "fullsuite.run_candidate": "review round 2: whole test selection, random and default order, no sampling",
    "sim.label": "§8 candidate labelling (+ D4)",
    "sim.run_episodes": "§9.3 FAR, FRR, blocks per accepted patch; Table 3 late-accept",
    "sim.experiment": "plan week 1: simulated streams with known truth (§9.3 calibration subset)",
    "replay.load_episode": "§9.4 offline replay of released streams",
    "replay.labels": "§8 labels from the reference-oracle blocks",
    "replay.replay": "§9.4 identical candidates in identical order (+ D6)",
    "replay.masked": "§8 injected flakiness with known parameters, near the eps boundary (D14)",
    "replay.episode_seed": "§9.4 replications: per-episode seed from its file name (stable under D18)",
    "replay.cp_upper": "§9.4 zero-event FAR reporting, one-sided Clopper-Pearson",
    "replay.far_upper": "§9.4 zero-event FAR reporting by episode and by repository cluster",
    "online.online_episode": "§12 online subset: SeqAccept in the loop (D20)",
    "online.online_tasks": "§12 online subset: the main-table tasks, one run per model (D20)",
    "online.main": "§12 online subset runner (D20)",
    "online.OutOfAttempts": "§12 online subset: agent out of replies (D20)",
    "replay.protected_view": "§9.2 B7 protected hidden tests (D21)",
    "collect.hidden_split": "§9.2 B7 held-out suite (D21)",
    "collect.hidden_runs": "§9.2 B7 held-out suite runs (D21)",
    "run_all.make_b7": "§9.2 B7 streams recorded against the visible tests (D21)",
    "replay.excluded": "§10 one episode set per model; D18 drops freezegun, D23 kanren_b2, for every model",
    "replay.late_streams": "Table 3 late-accept and §6 refunds, on streams built from recorded candidates (D15)",
    "replay.evaluate": "§9.4 every policy on every episode, 20 replications",
    "replay.summarize": "Table 2/3 columns: FAR + CI, FRR, budget, time-to-accept, blocks/accept, late-accept",
    "replay.cluster_ci": "§9.4 cluster bootstrap by repository",
    "replay.primary_endpoint": "§9.4 preregistered paired FAR contrast with a 3-point margin",
    "replay.q_band": "§12 FAR stratified by q",
    "replay.composition": "Table 1 composition",
    "replay.split": "§12 FAR stratified by flakiness type; plan H2 per model",
    "replay.pct": "Table 2/3 printing",
    "replay.num": "Table 2/3 printing",
    "replay.print_report": "plan day 14 results table",
    "build_episode.build_episode": "§8 reference oracle, 400 blocks per candidate (+ D5)",
    "agent.extract_code": "plan 2.7 mini agent",
    "agent.run_stream": "§8 record every failed attempt; plan 2.7 record mode",
    "run_block.BlockResult": "Def. 4 block outcome Y_B",
    "run_block.BlockError": "§12: unmodelled environment failures are not evidence (+ D7)",
    "run_block.DockerDown": "§12: an unreachable Docker daemon is not evidence; the run stops (+ D7)",
    "run_block.run_block": "Def. 4 / §7 block execution; default_order is Prop. 10's default-order rerun",
    "screen_idoft.shortlist": "§8 natural flakiness from IDoFT; plan 2.2",
    "llm.azure": "plan 2.7 agents",
    "llm.ollama": "plan 2.7 agents",
    "collect.make_run": "Def. 4 one block per call",
    "collect.fail_rate": "plan days 6-8: p0, bug check; Prop. 10 demo",
    "collect.record": "§8 recorded candidate streams",
    "collect.image_id": "§8 release: container digests",
    "collect.blocks": "§8 per-block outcomes with seeds",
    "collect.now": "§14 access dates",
    "collect.log": "progress output for long runs (no effect on results)",
    "collect.block_size": "§5.5 block of 8-32 tests (+ D12)",
    "collect.subset": "Def. 4 random subset S per block, drawn from the block seed",
    "collect.collect_test_ids": "Def. 4: the test population blocks are drawn from",
    "flakeinject.pytest_runtest_makereport": "§8 injected Bernoulli / Markov-modulated flakiness, known parameters",
    "run_all.collect_subsets": "Def. 4 per task",
    "run_all.make_variants": "§8 injected-flakiness subset (+ D12)",
    "collect.cached": "block checkpoint for resumed runs (same seeds, same blocks: no effect on results)",
    "collect.pool_map": "several containers at once (+ D10)",
    "collect.write_json": "atomic writes so a stop never corrupts results",
    "collect.measure": "plan days 6-8: p0 and bug check",
    "collect.stream_name": "file naming",
    "collect.record_stream": "§8 recorded candidate streams; plan days 9-10",
    "collect.episode_file": "§8 per-block outcomes; plan days 11-13",
    "run_all.Ledger": "resumable overnight run of plan weeks 2-3 and days 5-7",
    "run_all.stream_finished": "resume check",
    "run_all.prepare": "plan days 6-8 per task",
    "run_all.experiment": "plan days 9-14",
    "run_all.screening": "plan days 5-7 for every IDoFT repo/commit; §8 flakiness sources; §12 selection filter",
    "run_all.main": "entry point",
    "screen_idoft.slug": "file naming",
    "screen_idoft.dockerfile": "same image recipe as repos/<name>/Dockerfile",
    "screen_idoft.screen": "plan days 5-7 screening rules; Prop. 10 default vs random order on real suites",
    "screen_idoft.write_summary": "plan days 6-7 screening log",
    "collect.logged": "progress output for long runs (no effect on results)",
    "collect.hardware": "§14 hardware profile for wall-clock results",
    "collect.check_task": "plan days 6-8 screening rules (installs, reproduces, <= 15 s, p0 <= 5%, bug > p0 + 5%)",
    "collect.check_llm": "plan day 9: the agent works end to end",
    "collect.preflight": "plan days 6-9 checks, run before recording",
    "reuse_demo.episode": "Thm. 8 construction (plan day 18 stretch)",
    "reuse_demo.far": "Thm. 8: FAR with reuse is a constant independent of alpha; §10.4",
}
SCRIPTS = {"run_sim": "plan week 1 Chart 1 (§10.1 Figure 3) + cost",
           "build_report": "§10 write-up: renders results/results.json and the run data as an HTML report, computes no new metric",
           "extra_analyses": "review round 1: spending baselines, cost-matched rerun, refund stress, first green, relabelled margin and boundary stress; offline, writes results/extra.json",
           "paper_numbers": "§10 write-up: every number, table and figure in latex/main.tex, from the result files; no new metric"}  # entry points with no public functions

DEVIATIONS = {  # paper says -> we do -> why. Each one is pinned by a check below.
    "D1 per-candidate cap": "§5.5 caps at 20 blocks -> 800 (max_blocks). 20 can never accept: 1/(1-eps)^20 = 2.8 < 40 "
                            "(plan 1.8, test_cap_of_20_from_paper_can_never_accept).",
    "D2 rejection rule": "Alg. 1 line 9 'p_lcb > eps' with no level -> a second betting wallet R >= 20, i.e. an "
                         "anytime-valid lcb > eps at level 1/20 (plan 2.8).",
    "D3 no refund on give-up": "Alg. 1 line 14 refunds when the cost cap is hit -> no refund. With it FAR exceeds alpha "
                               "(test_alg1_line_14_refund_breaks_validity).",
    "D4 background rate": "§5.5 eps = 0.05, §8 labels 0.01/0.05 -> both shifted by the original code's failure "
                          "rate p0 (plan 2.3: real flaky tests fail correct code too).",
    "D5 early stop of clearly broken fixes": "§8 400 oracle blocks per candidate -> stop after 40 if >= 75% fail; "
                                             "those blocks serve as truth and pool (plan week 3).",
    "D6 replay draws from a pool": "§9.4 fresh blocks -> sample with replacement from 200 recorded pool blocks (plan 2.5).",
    "D7 block plumbing": "§7 says nothing -> offline container (epoch, Def. 2, cannot drift), a hung block, a block killed for using over 2 GB (per container) or running over 180 s, or a candidate that crashes pytest counts "
                         "as failed (2026-11-04: one gpt-4o kanren candidate leaked memory until the Docker VM restarted; "
                         "the same day 24 stalled 300 s blocks of correct pdir2 candidate 29be16113294, second runs of "
                         "seeds whose first run had passed, were dropped from its checkpoint and rerun under the cap, "
                         "backup in archive/pdir2-stall-2026-11-04),an infrastructure error is retried with the same seed in a new container (3 tries) "
                         "then raised, never counted.",
    "D8 late-accept threshold": "Table 3 does not define 'late' -> first Correct fix at position >= LATE = 5.",
    "D10 parallel blocks": "§7 says nothing about concurrency -> up to --workers fresh containers at once; timing-"
                           "sensitive tests feel the contention, so p0 is measured with the same workers (recorded).",
    "D11 bug reports": "plan 2.4 human-written -> written by an independent agent from failing test output only (no "
                       "access to the original file, the diff or the library source).",
    "D12 unstated parameters": "§5.5 'block size 8-32 tests' -> min(32, max(8, n/4), n/2) for n tests; §8 injected "
                               "flakiness rates are not given -> run_all.INJECT, applied to the ordinary-bug tasks.",
    "D13 who plants bugs": "plan 2.4 by hand -> the 6 bugs added 2026-10-31 (pymeterbus, anonapi, filestack, penman, kanren, "
                           "qtaf) were planted by agents, each checked on 40 subset blocks (original 0/40, buggy 0.12-0.60); "
                           "the planting agents never wrote the reports (D11).",
    "D14 boundary stress": "§10 calibration needs broken candidates just above eps; the per-test flips of D12 leave "
                           "nearly all of them far above it (2026-11-04: 4 of 610 within 0.05 of eps, SeqAccept FAR 0) "
                           "-> replay.MASKS: on natural episodes, a recorded failed block reads as passed with known "
                           "probability m, fresh on every draw; truth rates and p0 scale by 1 - m. Reported apart from "
                           "Table 2 (test_boundary_stress_scales_rates_by_the_known_mask).",
    "D15 built late streams": "Table 3 late streams come from agents -> only one recorded stream is late (first "
                              "Correct fix at position 6), so replay.late_streams also builds them: per task, k in "
                              "LATE_K Broken fixes drawn from its recorded candidates, then one of its Correct fixes. "
                              "Reported apart from Table 2 (test_built_late_streams_put_k_broken_fixes_before_a_correct_one).",
    "D16 B4 and B5 details": "§9.2 gives no level, cap or rejection rule -> B4: one-sided Clopper-Pearson at alpha, "
                             "checked after every block, accept when the upper bound < eps, reject when the lower bound "
                             "> eps, cap 800 (D1). B5: SeqAccept's wallets with the full alpha for every candidate. "
                             "Replayed blocks are always randomized, so B5 drops error spending only, not block "
                             "randomization (2026-11-04, added after the pilot plan had left B4/B5 out).",
    "D17 injection-rate sweep": "§10.5 'flakiness prevalence' sensitivity, rates not given -> run_all.SWEEP: Bernoulli "
                                "false-pass 0.1 and 0.6 around INJECT's 0.3, on every ordinary-bug task (kanren, "
                                "8.6 s per block, was skipped at first and added 2026-11-05); replay keeps these episodes out of the main table "
                                "(replay.MAIN_FLAKINESS) and compares them with the 0.3 episodes of the same repos.",
    "D18 freezegun left out for every model": "§10 plans 2 runs x 24 episodes per model. grok-4.7's freezegun run 1 job "
                                             "died on Azure 429 (token rate limit) after attempt 10-13 of 15 on every full "
                                             "try (the agent resends its whole history, ~80k tokens by attempt 11) and was "
                                             "dropped on 2026-11-05; the partial stream is archived, not used. Run 2 fits and was "
                                             "recorded, but on 2026-11-07 the user asked for the same episodes in every model, "
                                             "so freezegun (both runs, all variants) is left out for all models: 42 episodes per "
                                             "model from the first bugs (plus the second bugs). replay.EXCLUDED_RUNS; the recorded "
                                             "streams stay on disk. The history window (an agent protocol change, would have "
                                             "been D23) was not adopted. freezegun_b2, a small file, is kept.",
    "D19 second bug per repository": "§9.1 plans more bugs per repo -> <repo>_b2 tasks: one more agent-planted bug in each of "
                                      "the 10 repos (same rules, D13; independent reports, D11), recorded like the first. They "
                                      "enter the main table; the D17 sweep and the D20 online subset stay on the first bugs.",
    "D20 online subset": "§12 '40-episode online subset where SeqAccept is in the loop' -> online.py: the 24 main-table "
                         "tasks x gpt-4.1-mini and gpt-4o, one run each (48 episodes). grok-4.7 and llama3.1:8b (D22) were added "
                         "on 2026-11-06 with 21 episodes each (freezegun is left out for both: its history exceeds grok's "
                         "80k tokens/min and llama's 96k window; online.NO_FREEZEGUN). The agent writes candidate j only "
                         "when SeqAccept moves on to it and gets the rejection evidence as feedback.",
    "D21 B7 protected hidden tests": "§9.2 B7 -> a fixed hashed third of each first-bug task's tests is held out; <task>@b7 "
                                     "streams (1 run per model) are recorded with feedback from the visible two thirds; "
                                     "every fix gets 400 full-suite blocks plus 20 hidden-suite runs, and B7 accepts the "
                                     "first fix whose drawn hidden run passes. WITHDRAWN on 2026-11-06 for lack of time (the 9 grok-4.7 "
                                     "streams alone would take about 8 hours): run_all.B7 = False starts no new streams, "
                                     "replay.excluded leaves the 20 streams and 7 episodes already recorded out, so there is "
                                     "no B7 table and B7 is not evaluated. The code stays, tested.",
    "D22 fourth model llama3.1:8b": "§10 uses the paper's agents -> llama3.1:8b (Ollama, local, Apple M5 Pro) is added on the user's "
                                    "request as a smaller open model: models arg ollama:llama3.1:8b, temperature 0.8 like the "
                                    "others, num_ctx 96k (Ollama's default 4k would silently drop the bug report). A request "
                                    "whose history fills the window fails the stream instead of truncating it. freezegun "
                                    "(history ~130k tokens) is not recorded for it (run_all.NOT_RECORDED); D18 leaves freezegun out "
                                    "for every model, so all four models score the same episodes. "
                                    "Added after the 3-model results were in, with 21 online episodes (D20).",
    "D23 kanren_b2 left out for every model": "§10 one episode set per model -> grok-4.7's kanren_b2 run 1 was still "
                                              "recording on 2026-11-08 when the Azure subscription ran out of free credit "
                                              "(every deployment then answered 401). The stream had no saved attempts, so "
                                              "kanren_b2 (both runs, all variants) is left out for all four models, as D18 "
                                              "does for freezegun: replay.EXCLUDED_RUNS. The recorded streams stay on disk. "
                                              "No further Azure calls are possible, so the run ends here.",
    "D24 refund capped at the wallet": "Alg. 1 line 11 refunds alpha_j (1 - s_j) -> alpha_j min(E_tau, 1 - s_j). Theorem 6's proof sums the slices alpha_j, which refunds can push past alpha; with the cap, W + alpha_j E_j is a nonnegative supermartingale under the null and Ville gives the bound (test_refund_never_exceeds_the_wallet). The online subset (D20) ran before the cap, with alpha_j (1 - s_j).",
    "D9 budget unit": "Table 2 budget in CPU-seconds -> block wall-clock seconds, container start included (§12 says "
                      "both are reported; wall clock is what we can measure per block). Blocks recorded before the "
                      "180 s cap (D7) are charged at most replay.CAP_SECONDS = 180 s, as the cap would have killed them.",
}
SKIPPED = {  # paper features the plan (2.6) leaves out; they must stay absent, not half-built
    "§5.1/§5.3 requirements, importance weights, cost-aware allocation": "one requirement per bug",
    "§5.4 epoch resets": "one fixed image per repo",
    "§9.2 B8": "identical to B1 here: every task already deselects IDoFT's known-flaky tests (p0 is measured without them)",
}


def public_names(module):
    return {f"{module.__name__}.{n}" for n, o in vars(module).items()
            if not n.startswith("_") and (inspect.isfunction(o) or inspect.isclass(o)) and o.__module__ == module.__name__}


def test_every_module_is_traced():
    traced = {k.split(".")[0] for k in TRACE} | set(SCRIPTS)
    assert {p.stem for p in IMPL.glob("*.py")} == traced


def test_every_public_function_and_class_is_traced_and_nothing_traced_is_missing():
    found = set()
    for mod in {k.split(".")[0] for k in TRACE}:
        found |= public_names(importlib.import_module(mod))
    assert found == set(TRACE), f"untraced: {found - set(TRACE)}; traced but gone: {set(TRACE) - found}"


def test_paper_constants():
    d = {k: v.default for k, v in inspect.signature(seq_accept).parameters.items() if v.default is not inspect._empty}
    assert d == {"alpha": 0.05, "eps": 0.05, "gamma": 0.5,  # §5.5
                 "refund": True,
                 "max_blocks": 800, "reject_at": 20, "bet": "kelly"}  # D1, D2; the paper's wallet is Kelly
    assert list(rules.all_rules()) == ["B1 stop at first pass", "B2 rerun-3 all green", "B3 majority of 5",
                                       "B4 peeking Clopper-Pearson", "B5 naive confidence sequence",
                                       "B6 SeqAccept, no refund", "SeqAccept"]  # §9.2, "we report k=3"; SKIPPED
    assert replay.REPS == 20  # §9.4
    assert replay.MARGIN == 0.05 and replay.LATE == 5  # §5.5 eps; D8
    b = inspect.signature(build_episode.build_episode).parameters
    assert b["n_truth"].default + b["n_pool"].default == 400  # §8 reference oracle
    assert (b["probe"].default, b["probe_frac"].default) == (40, 0.75)  # D5
    assert "NIO" not in screen_idoft.KEEP  # plan 2.2: fresh containers make NIO irrelevant


@pytest.mark.parametrize("p, want", [(0.0, "correct"), (0.0099, "correct"), (0.01, "marginal"),
                                     (0.05, "marginal"), (0.0501, "broken")])
def test_labels_are_section_8(p, want):  # Correct p < 0.01, Broken p > 0.05
    assert sim.label(p) == want


def algorithm_1(n, block, alpha=0.05, eps=0.05, gamma=0.5, refund=True, cap=800, reject_at=20, line_14=False,
                on_refund=None):
    """Algorithm 1 transcribed from the paper; D1-D3 are marked. line_14=True is the paper's give-up refund."""
    W, blocks = alpha, 0  # Require: budget pool W <- alpha
    for j in range(n):  # line 1
        a_j = gamma * W  # line 2
        W -= a_j  # line 3
        E, G, R, fails = 1.0, 1.0, 1.0, 0  # line 3 E_0 = 1; G_j = max_t E_t (§6.1 step 3); R: D2
        for t in range(cap):  # line 4; cap: line 13 with D1
            ybar = (fails + 0.5) / (t + 1)  # §7 1/2-smoothed running mean
            lam = min(0.9 / (1 - eps), max(0.0, (eps - ybar) / (eps * (1 - eps))))  # §7 truncated Kelly
            mu = min(0.9 / eps, max(0.0, (ybar - eps) / (eps * (1 - eps))))  # D2
            y = block(j)  # line 6, one requirement: Z_t = Y_t (SKIPPED)
            blocks, fails = blocks + 1, fails + y
            E *= 1 + lam * (eps - y)  # §5.1 betting supermartingale
            R *= 1 + mu * (y - eps)  # D2
            G = max(G, E)
            if E >= 1 / a_j:  # line 7
                return Result(j, blocks)  # line 8
            if R >= reject_at:  # line 9 (D2)
                if refund:
                    r = a_j * min(E, 1 - a_j * G)  # line 11, s_j = alpha_j G_j, capped at the wallet (D24)
                    if on_refund:
                        on_refund(a_j, E, r)
                    W += r
                break  # line 12
        else:
            if refund and line_14:
                W += a_j * (1 - a_j * G)  # line 14 (D3: off)
    return Result(None, blocks)


def scripted(outcomes):
    pos = [0] * len(outcomes)

    def block(j):
        pos[j] += 1
        return outcomes[j][pos[j] - 1]

    return block


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("refund", [True, False])
def test_seq_accept_is_algorithm_1(seed, refund):
    rng = random.Random(seed)
    p0 = rng.choice([0.0, 0.02, 0.05])
    ps = [rng.choice([p0, p0 + 0.03, p0 + 0.06, p0 + 0.2, 0.5, 0.96]) for _ in range(rng.randint(1, 12))]
    outcomes = [[int(rng.random() < p) for _ in range(800)] for p in ps]
    kw = {"eps": p0 + 0.05, "refund": refund}
    assert seq_accept(len(ps), scripted(outcomes), **kw) == algorithm_1(len(ps), scripted(outcomes), **kw)


def test_refund_never_exceeds_the_wallet():  # D24: what Theorem 6's supermartingale W + alpha_j E_j needs
    seen = []
    for seed in range(40):
        rng = random.Random(seed)
        ps = [rng.choice([0.06, 0.08, 0.12, 0.3]) for _ in range(10)]
        outcomes = [[int(rng.random() < p) for _ in range(800)] for p in ps]
        algorithm_1(len(ps), scripted(outcomes), on_refund=lambda a, E, r: seen.append((a, E, r)))
    assert seen and all(0 <= r <= a * min(E, 1) + 1e-15 for a, E, r in seen)
    assert any(E < 1 for _, E, _ in seen)  # the cap is exercised, not vacuous


def test_real_flaky_variant_keeps_only_the_known_flaky_tests():  # review round 4
    import flaky_arm
    tests = ["tests", "--deselect=tests/a.py::flaky", "--deselect=tests/b.py::needs_network"]
    assert flaky_arm.flaky_selection(tests, {"tests/a.py::flaky"}) == ["tests", "--deselect=tests/b.py::needs_network"]
    assert flaky_arm.flaky_selection(tests, set()) == tests
    assert flaky_arm.idoft_tests("filestack/filestack-python") >= {"tests/intelligent_ingestion_test.py::test_upload_part_success"}
    assert len(flaky_arm.jobs()) == 136


def test_fixed_n_sizes_its_sample_and_accepts_only_on_a_rejected_binomial_null():  # review round 3
    assert rules.binom_cdf(0, 50, 0.05) == pytest.approx(0.95 ** 50)
    assert rules.binom_cdf(1, 50, 0.05) == pytest.approx(0.95 ** 50 + 50 * 0.05 * 0.95 ** 49)
    assert rules.binom_cdf(50, 50, 0.05) == pytest.approx(1.0)
    slices = rules.bonferroni(15)
    m0 = math.ceil(math.log(0.05 / 15) / math.log(0.95))  # fewest blocks at which no failure can be accepted
    assert 0.95 ** m0 <= 0.05 / 15 < 0.95 ** (m0 - 1)
    assert rules.fixed_n(3, lambda j: 0, slices) == rules.Result(0, 2 * m0)  # perfect candidate: accepted at 2 m0
    crit = rules.critical(2 * m0, 0.05, 0.05 / 15)  # stops a candidate at crit + 1 failures: it can no longer pass
    assert rules.binom_cdf(crit, 2 * m0, 0.05) <= 0.05 / 15 < rules.binom_cdf(crit + 1, 2 * m0, 0.05)
    assert rules.fixed_n(3, lambda j: 1 if j < 2 else 0, slices) == rules.Result(2, 2 * (crit + 1) + 2 * m0)
    seq = iter([0] * (2 * m0 - crit - 1) + [1] * (crit + 1) + [0] * (2 * m0))  # crit + 1 failures, the last at m: rejected
    assert rules.fixed_n(2, lambda j: next(seq), slices) == rules.Result(1, 4 * m0)
    assert rules.fixed_n(20, lambda j: 0 if j == 16 else 1, slices).accepted is None  # no slice after 15


def test_fixed_spending_schedules_sum_to_alpha_and_match_seq_accept_without_refund():  # review round 1
    assert sum(rules.harmonic()(j) for j in range(10**5)) == pytest.approx(0.05, rel=1e-4)
    assert sum(rules.bonferroni(15)(j) for j in range(40)) == pytest.approx(0.05)
    geometric = lambda j: 0.05 * 0.5 ** (j + 1)  # B6's slices when nothing is refunded
    rng = random.Random(3)
    outcomes = [[int(rng.random() < p) for _ in range(800)] for p in (0.5, 0.06, 0.0)]
    assert (rules.fixed_spending(3, scripted(outcomes), geometric)
            == seq_accept(3, scripted(outcomes), refund=False))


def test_full_suite_arm_runs_the_whole_selection_and_default_order_only_on_natural_tasks():  # review round 2
    import fullsuite
    from run_block import BlockResult

    calls = []

    def factory(task, default_order=False):
        assert "test_ids" not in task  # the whole selection, never a sampled subset
        return lambda code, seed: calls.append((default_order, seed)) or BlockResult(int(default_order), "", 1.0)

    out = fullsuite.run_candidate({"test_ids": ["a"], "tests": ["t"]}, "code", runs=3, factory=factory)
    assert out["random"] == [0, 0, 0] and out["default"] == 1 and len(calls) == 4
    assert fullsuite.run_candidate({"tests": ["t"], "inject": {"mode": "bernoulli"}}, "c", 2, factory)["default"] is None


def test_cap_of_20_from_paper_can_never_accept():  # D1
    assert (1 / 0.95) ** 20 < 1 / (0.5 * 0.05)
    assert seq_accept(1, lambda j: 0, max_blocks=20).accepted is None
    assert math.log(40) / math.log(1 / 0.95) > 41.2  # Table 3's 41.2 blocks/accept is below the floor (plan 1.8)


def test_boundary_stress_scales_rates_by_the_known_mask():  # D14
    assert replay.MASKS == (0.80, 0.90, 0.94)
    always = {"p0": 0.02, "stream": ["x"], "fixes": {"x": {"truth": [1] * 200, "pool": [1] * 200}}}
    ep = replay.masked(always, 0.9)
    assert ep["p0"] == pytest.approx(0.002) and replay.labels(ep) == ["broken"]  # 0.1 > 0.002 + 0.05
    fails = []

    def count(n, block):
        fails.append(sum(block(0) for _ in range(1000)))
        return Result(None, 0)

    replay.replay(count, ep, reps=1)
    assert 70 <= fails[0] <= 130  # ~10% of 1000 drawn failures survive the mask
    assert replay.masked(always, 0.95)["fixes"] is always["fixes"]  # recorded blocks are never rewritten


def test_built_late_streams_put_k_broken_fixes_before_a_correct_one():  # D15
    assert replay.LATE_K == (5, 10, 20, 40) and replay.NEAR_REPS == 200
    bad, good = {"truth": [1] * 40, "pool": [1] * 40}, {"truth": [0] * 200, "pool": [0] * 200}
    eps = [{"p0": 0.0, "repo": "r", "stream": ["b"], "fixes": {"b": bad}},
           {"p0": 0.0, "repo": "r", "stream": ["g"], "fixes": {"g": good}},  # same task, another episode
           {"p0": 0.0, "repo": "s", "stream": ["b"], "fixes": {"b": bad}}]  # no Correct fix: no stream
    built = replay.late_streams(eps, 10, n=3)
    assert len(built) == 3 and all(replay.labels(ep) == ["broken"] * 10 + ["correct"] for ep in built)
    assert replay.evaluate(built, reps=5)["SeqAccept"][0]["correct_accept"] == 1.0


def test_the_same_task_runs_are_left_out_for_every_model():  # D18
    assert replay.EXCLUDED_RUNS == {("freezegun", 1), ("freezegun", 2), ("kanren_b2", 1), ("kanren_b2", 2)}
    for name in ("kanren_b2-gpt-4o-1.json", "kanren_b2@markov-llama3.1_8b-2.json"):  # D23
        assert replay.excluded(name)
    assert not replay.excluded("kanren-grok-4.7-1.json")  # the first kanren bug is kept
    for name in ("freezegun-grok-4.7-1.json", "freezegun@bernoulli-gpt-4o-1.json", "freezegun@bernoulli60-gpt-4.1-mini-1.json"):
        assert replay.excluded(name)
    for name in ("freezegun-gpt-4o-2.json", "freezegun@markov-grok-4.7-2.json"):  # both runs: grok cannot record either
        assert replay.excluded(name)
    for name in ("pdir2-gpt-4.1-mini-1.json", "freezegun_b2-grok-4.7-2.json"):  # the second freezegun bug is a small file, kept
        assert not replay.excluded(name)


def test_sweep_variants_are_named_apart_from_the_main_injection(tmp_path):  # D17
    import run_all

    assert {k: v["false_pass"] for k, v in run_all.SWEEP.items()} == {"bernoulli10": 0.1, "bernoulli60": 0.6}
    assert not {f"injected-{k}" for k in run_all.SWEEP} & set(replay.MAIN_FLAKINESS)
    base = {"bug_kind": "ordinary", "repo": "r"}
    for stem, want in (("pdir2", 4), ("kanren", 4)):
        (tmp_path / f"{stem}.json").write_text(__import__("json").dumps(base))
        assert len(run_all.make_variants(str(tmp_path / f"{stem}.json"))) == want


def test_alg1_line_14_refund_breaks_validity():  # D3
    def far(rule):
        rng = random.Random(0)
        return sum(rule(20, lambda j: int(rng.random() < 0.11), eps=0.10).accepted is not None for _ in range(300)) / 300

    assert far(seq_accept) <= 0.05
    assert far(lambda n, b, **kw: algorithm_1(n, b, line_14=True, **kw)) > 0.05


def test_blocks_are_random_subsets_of_8_to_32_tests_drawn_from_the_seed():  # Def. 4, §5.5, D12
    from collect import block_size, subset

    assert [block_size(n) for n in (10, 40, 95, 400)] == [5, 10, 23, 32]
    task = {"test_ids": [f"t{i}" for i in range(95)], "block_size": block_size(95)}
    assert subset(task, 7) == subset(task, 7) != subset(task, 8)  # reproducible per seed, different across seeds
    assert len(set(subset(task, 7))) == 23


def test_b7_accepts_the_first_fix_whose_hidden_suite_run_passes(tmp_path):  # D21
    import collect
    import run_all

    ids = [f"tests/t.py::test_{i}" for i in range(30)]
    visible, hidden = collect.hidden_split(ids)
    assert set(visible) | set(hidden) == set(ids) and not set(visible) & set(hidden)
    assert 5 <= len(hidden) <= 15 and collect.hidden_split(ids) == (visible, hidden)  # fixed, not drawn
    base = {"bug_kind": "ordinary", "repo": "r", "test_ids": ids, "block_size": 8, "p0": 0.01}
    (tmp_path / "x.json").write_text(__import__("json").dumps(base))
    b7 = __import__("json").loads(Path(run_all.make_b7(str(tmp_path / "x.json"))).read_text())
    assert b7["test_ids"] == visible and b7["hidden_ids"] == hidden and b7["runs"] == 1 and b7["flakiness"] == "b7"
    ep = {"name": "x@b7-m-1", "p0": 0.0, "stream": ["bad", "good"], "repo": "r",
          "fixes": {"bad": {"truth": [1] * 200, "pool": [0] * 200, "hidden": [1] * 20},  # visible tests miss the bug
                    "good": {"truth": [0] * 200, "pool": [0] * 200, "hidden": [0] * 20}}}
    b1 = replay.replay(rules.stop_on_first_pass, ep, 5)
    b7r = replay.replay(rules.stop_on_first_pass, replay.protected_view(ep), 5)
    assert b1["far"] == 1.0 and b7r["far"] == 0.0 and b7r["correct_accept"] == 1.0


def test_discard_spending_returns_a_discarded_slice_and_spends_the_rest():  # review round 5
    perfect = rules.discard_spending(1, lambda j: 0).blocks
    # a candidate that fails at once never lifts its wallet: discarded, so the next one faces the same threshold
    r = rules.discard_spending(2, lambda j: 1 if j == 0 else 0)
    assert r.accepted == 1 and r.blocks - perfect == rules.discard_spending(1, lambda j: 1).blocks
    # tested at tau * s_j: a perfect candidate needs more blocks than SeqAccept at gamma * alpha
    assert perfect > rules.seq_accept(1, lambda j: 0).blocks


def test_mixture_wallet_is_the_mean_of_fixed_bet_wallets_and_keeps_kelly_default():  # external review
    import random
    rng = random.Random(3)
    ys = [int(rng.random() < 0.02) for _ in range(800)]
    assert rules.seq_accept(1, lambda j, it=iter(ys): next(it)) == rules.seq_accept(1, lambda j, it=iter(ys): next(it), bet="kelly")
    r = rules.seq_accept(1, lambda j: 0, bet="mixture")
    lams = rules.mix_lambdas(0.05)
    # a perfect candidate is accepted at the first t where the mean of prod(1 + l * eps) reaches 1 / alpha_1
    t = next(t for t in range(1, 800) if sum((1 + l * 0.05) ** t for l in lams) / len(lams) >= 1 / 0.025)
    assert r == rules.Result(0, t)
    assert rules.seq_accept(5, lambda j: 1, bet="mixture").accepted is None


def test_review2_beta_cdf_and_counted_replay_match_closed_forms():  # external review 2
    import review2
    for b, x in ((5, 0.1), (200, 0.02)):
        assert abs(review2.beta_cdf(x, 1, b) - (1 - (1 - x) ** b)) < 1e-4  # Beta(1, b) has a closed-form CDF
    ep = {"p0": 0.0, "stream": ["a"], "fixes": {"a": {"truth": [0] * 200, "pool": [0] * 200}}}
    res, drawn = review2.counted_replay(rules.seq_accept, ep, reps=3, seed=1)
    perfect = rules.seq_accept(1, lambda j: 0).blocks
    assert res["blocks"] == 3 * perfect and drawn == [(200, perfect)] * 3


def test_exec_order_summary_and_disjoint_seed_loops():  # external review 2
    import exec_order, flaky_seeds, build_episode
    recs = [{"seed": 10 - i, "failed": i % 2, "start": float(i), "end": i + 0.5, "load": 1.0, "worker": "w"} for i in range(10)]
    s = exec_order.summarize(recs)
    assert s["blocks"] == 10 and s["fail_rate"] == 0.5 and s["concurrency_median"] == 0
    assert s["runs_reject_exec"] and s["lag1_exec"] < -0.8  # strict alternation in execution order is caught
    assert s["order_inversions"] == 1.0 and s["lag1_seed"] < -0.8  # seeds were assigned in reverse start order
    assert abs(exec_order.two_prop_z([1, 1, 0, 0], [1, 0, 0, 0]) - 0.730) < 0.01
    ks = [k for _, _, _, k in flaky_seeds.loops(flaky_seeds.HIGH)]
    assert ks == list(range(len(ks))) and [k for *_, k in flaky_seeds.loops(["yamicache"])] == \
        [k for b, *_, k in flaky_seeds.loops(flaky_seeds.HIGH) if b == "yamicache"]  # same seeds whatever --tasks
    seen = []
    ep = build_episode.build_episode([{"fix_id": "a", "code": "x"}], 0.0,
                                     lambda code, seed: (seen.append(seed), type("R", (), {"failed": 0, "seconds": 0.0})())[1],
                                     n_truth=2, n_pool=2, probe=2, first_seed=5_000_001)
    assert seen == [5_000_001, 5_000_002, 5_000_003, 5_000_004] and ep["fixes"]["a"]["seeds"] == seen
