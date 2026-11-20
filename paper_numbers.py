"""Every measured value in the paper, generated from the result and data files (§10 write-up; computes no new metric).

Reads results/results.json, worst_case.json, operating_curve.json, reuse_demo.json, online/*.json, tasks/*.json,
episodes/*.json and screening/summary.csv. Writes <out>/generated/numbers.tex (one macro per inline number),
<out>/generated/tab_*.tex (tables) and <out>/figs/*.pdf (figures), where <out> is $SEQACCEPT_PAPER_DIR or ./paper_out.

Usage: uv run python paper_numbers.py
"""
import csv
import json
import os
from collections import Counter
from pathlib import Path
from statistics import median

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import replay

OUT = Path(os.environ.get("SEQACCEPT_PAPER_DIR") or Path(__file__).resolve().parent / "paper_out")
GEN, FIGS = OUT / "generated", OUT / "figs"
RULES = ["B1 stop at first pass", "B2 rerun-3 all green", "B3 majority of 5", "B4 peeking Clopper-Pearson",
         "B5 naive confidence sequence", "B6 SeqAccept, no refund", "SeqAccept"]
KEY = dict(zip(RULES, ["Bone", "Btwo", "Bthree", "Bfour", "Bfive", "Bsix", "Sa"]))  # macro suffixes
SHOW = dict(zip(RULES, ["B1 Stop-on-first-pass", "B2 Rerun-3 (all green)", "B3 Majority-of-5", "B4 Fixed-sample CI",
                        "B5 Naive conf.\\ sequence", "B6 \\sa{}-NoRefund", "\\sa{}"]))
MODELS = {"llama3.1:8b": "Llama", "gpt-4o": "Gptfouro", "gpt-4.1-mini": "Gptmini", "grok-4.7": "Grok"}
MODEL_SHOW = {"llama3.1:8b": "llama3.1:8b", "gpt-4o": "gpt-4o", "gpt-4.1-mini": "gpt-4.1-mini", "grok-4.7": "grok-4.7"}
ONLINE_MODEL = {"ollama:llama3.1:8b": "llama3.1:8b"}
# dataviz reference palette, light mode, slots 1-4 (validated: scripts/validate_palette.js); SeqAccept takes slot 1
COLOR = {"SeqAccept": "#2a78d6", "B1 stop at first pass": "#eb6834", "B2 rerun-3 all green": "#1baf7a",
         "B3 majority of 5": "#eda100", "B4 peeking Clopper-Pearson": "#e34948", "B5 naive confidence sequence": "#008300",
         "B6 SeqAccept, no refund": "#6250d6"}  # one hue per rule across all figures; red/green pair carries marker + dash
MARK = {"SeqAccept": "o", "B1 stop at first pass": "s", "B2 rerun-3 all green": "^", "B3 majority of 5": "D",
        "B4 peeking Clopper-Pearson": "s", "B5 naive confidence sequence": "^", "B6 SeqAccept, no refund": "D"}
LINE = {"SeqAccept": "-", "B1 stop at first pass": "--", "B2 rerun-3 all green": "-.", "B3 majority of 5": ":",
        "B4 peeking Clopper-Pearson": "--", "B5 naive confidence sequence": "-.", "B6 SeqAccept, no refund": ":"}
INK, MUTED = "#0b0b0b", "#52514e"

macros = {}


def m(name, value):
    assert name.isalpha(), name
    assert name not in macros, f"duplicate macro {name}"
    macros[name] = str(value)


def pct(v, d=1):
    return f"{100 * v:.{d}f}"


def num(v, d=1):
    return f"{v:,.{d}f}".replace(",", "{,}")


def word_int(v):
    return f"{v:,}".replace(",", "{,}")


def load(path):
    return json.loads(Path(path).read_text())


def write(name, text):
    (GEN / name).write_text(text)


def main_episodes():
    """The episode files the main table uses (same rule as replay.py)."""
    out = []
    for p in sorted(Path("episodes").glob("*.json")):
        if replay.excluded(p):
            continue
        ep = replay.load_episode(p)
        if ep.get("flakiness", "natural") in replay.MAIN_FLAKINESS:
            out.append(ep)
    return out


def task_of(name):
    return name.split("-")[0]  # task names have no dash; model names do (as replay.excluded)


def base_task(name):
    return name.split("@")[0]


def benchmark(res, episodes):
    comp = res["composition"]
    for k in ("episodes", "repositories", "candidates", "correct", "broken", "marginal"):
        m(f"n{k.capitalize()}", word_int(comp[k]))
    m("medianCandidates", f"{comp['candidates_median_per_episode']:.0f}")
    m("meanQ", f"{comp['mean_false_pass_rate_q']:.2f}")
    m("pctBroken", pct(comp["broken"] / comp["candidates"]))
    m("pctCorrect", pct(comp["correct"] / comp["candidates"]))
    m("pctMarginal", pct(comp["marginal"] / comp["candidates"]))

    # tasks and bugs actually in the main table
    stems = sorted({task_of(ep["name"]) for ep in episodes})
    bugs = sorted({base_task(s) for s in stems})
    tasks = {b: load(f"tasks/{b}.json") for b in bugs}
    first, second = [b for b in bugs if "_b2" not in b], [b for b in bugs if "_b2" in b]
    order = [b for b in bugs if tasks[b]["bug_kind"] == "order"]
    m("nBugs", len(bugs))
    m("nFirstBugs", len(first))
    m("nSecondBugs", len(second))
    m("nOrderBugs", len(order))
    m("nOrdinaryBugs", len(bugs) - len(order))
    m("nOrderRepos", len({tasks[b]["repo"] for b in order}))
    ntests = [len(tasks[b]["test_ids"]) for b in bugs]
    m("minTests", min(ntests))
    m("maxTests", max(ntests))
    sizes = [tasks[b]["block_size"] for b in bugs]
    m("minBlockSize", min(sizes))
    m("maxBlockSize", max(sizes))
    nat_p0 = [tasks[b]["p0"] for b in bugs]
    m("maxNaturalPzero", f"{max(nat_p0):.2f}")
    m("nNaturalPzeroZero", sum(p == 0 for p in nat_p0))
    inj = [load(f"tasks/{b}@{mode}.json")["p0"] for b in bugs if tasks[b]["bug_kind"] == "ordinary"
           for mode in ("bernoulli", "markov")]
    m("minInjectedPzero", f"{min(inj):.3f}")
    m("maxInjectedPzero", f"{max(inj):.3f}")
    m("nInjectedTasks", len(inj) // 2)
    p0_blocks = {tasks[b]["p0_blocks"] for b in bugs}
    assert len(p0_blocks) == 1
    m("pzeroBlocks", p0_blocks.pop())
    # the bug check that excluded a planted bug: buggy code must fail more than p0 + 0.05 of blocks
    all_bugs = sorted(p.stem for p in Path("tasks").glob("*.json") if "@" not in p.stem)
    m("nPlantedBugs", len(all_bugs))
    not_recorded = {t for t, _ in replay.EXCLUDED_RUNS}  # planted bugs grok-4.7 could not record (D18, D23)
    m("nNotRecorded", len(not_recorded))
    m("nBugCheckFailed", len(all_bugs) - len(bugs) - len(not_recorded))

    # agents
    per_model = Counter(ep["model"] for ep in episodes)
    assert len(set(per_model.values())) == 1, per_model
    m("nModels", len(per_model))
    m("episodesPerModel", per_model.most_common(1)[0][1])
    runs = {int(ep["name"].rsplit("-", 1)[1]) for ep in episodes}
    m("nRuns", len(runs))
    agents = {ep["agent"]["deployment"]: ep["agent"] for ep in episodes}
    assert {a["temperature"] for a in agents.values()} == {0.8} and {a["max_attempts"] for a in agents.values()} == {15}
    m("temperature", "0.8")
    m("maxAttempts", 15)
    for model, key in MODELS.items():
        dates = sorted(ep["agent"]["started"][:10] for ep in episodes if ep["model"] == model)
        m(f"firstDate{key}", dates[0])
        m(f"lastDate{key}", dates[-1])
    m("apiVersion", agents["gpt-4o"]["api_version"])
    flak = Counter(ep.get("flakiness", "natural") for ep in episodes)
    m("nNaturalEpisodes", flak["natural"])
    m("nBernoulliEpisodes", flak["injected-bernoulli"])
    m("nMarkovEpisodes", flak["injected-markov"])

    # blocks executed for the main table, and their wall clock
    blocks = secs = clearly = fixes = 0
    seconds = []
    for ep in episodes:
        for f in ep["fixes"].values():
            n = len(f["seeds"])
            blocks += n
            fixes += 1
            clearly += len(f["truth"]) < 200
            seconds += f["seconds"]
    over = sum(sec > replay.CAP_SECONDS for ep in episodes for f in ep["fixes"].values() for sec in f["seconds"])
    m("nBlocksOverCap", word_int(over))
    m("pctBlocksOverCap", pct(over / blocks))
    cand = {r["slug"].rsplit("_", 1)[0] for r in csv.DictReader(open("screening/summary.csv")) if r["verdict"] == "candidate"}
    repos = {tasks[b]["repo"] for b in bugs}
    from_screen = {r for r in repos if r.replace("/", "_").replace("-", "_").lower() in cand}
    m("nReposFromScreen", len(from_screen))
    m("nReposByHand", len(repos) - len(from_screen))
    chosen = {r.replace("/", "_").replace("-", "_").lower() for r in repos}
    unchosen = sorted(c for c in cand if c not in chosen)
    m("nUnchosen", len(unchosen))
    m("unchosenList", ", ".join("\\texttt{" + c.replace("_", "\\_") + "}" for c in unchosen))
    m("nBlocksRun", word_int(blocks))
    m("nFixesGraded", word_int(fixes))
    m("nClearlyBroken", word_int(clearly))
    m("medianBlockSeconds", f"{median(seconds):.1f}")
    m("blockHours", word_int(round(sum(min(s, replay.CAP_SECONDS) for s in seconds) / 3600)))
    m("nWithBroken", res["table"]["SeqAccept"]["far_upper"]["episodes"][1])
    m("nWithCorrect", sum("correct" in replay.labels(ep) for ep in episodes))
    m("nNoBroken", res["splits"]["q_band"]["no broken"]["streams"])

    # false-pass rate q of Broken candidates by source of flakiness
    def q(eps_):
        qs = [1 - sum(e["fixes"][f]["truth"]) / len(e["fixes"][f]["truth"]) for e in eps_
              for f, lab in zip(e["stream"], replay.labels(e)) if lab == "broken"]
        return sum(qs) / len(qs)

    nat = [e for e in episodes if e.get("flakiness", "natural") == "natural"]
    m("qNatural", f"{q(nat):.2f}")
    m("qNaturalOrder", f"{q([e for e in nat if e['bug_kind'] == 'order']):.2f}")
    m("qNaturalOrdinary", f"{q([e for e in nat if e['bug_kind'] == 'ordinary']):.2f}")
    m("qBernoulli", f"{q([e for e in episodes if e.get('flakiness') == 'injected-bernoulli']):.2f}")
    m("qMarkov", f"{q([e for e in episodes if e.get('flakiness') == 'injected-markov']):.2f}")

    # composition table
    rows = [("Repositories (IDoFT, Python)", macros["nRepositories"]),
            ("Planted bugs (first / second per repository)", f"{len(bugs)} ({len(first)} / {len(second)})"),
            ("\\quad order bugs / ordinary bugs", f"{len(order)} / {len(bugs) - len(order)}"),
            ("Tests per task", f"{min(ntests)}--{max(ntests)}"),
            ("Repair agents (LLMs) $\\times$ runs", f"{len(per_model)} $\\times$ {len(runs)}"),
            ("Recorded repair loops (natural episodes)", f"{flak['natural']}"),
            ("Episodes: natural / Bernoulli / Markov replays",
             f"{comp['episodes']}: {flak['natural']} / {flak['injected-bernoulli']} / {flak['injected-markov']}"),
            ("Stream positions (median per episode)",
             f"{word_int(comp['candidates'])} ({comp['candidates_median_per_episode']:.0f})"),
            ("\\quad labelled \\textsc{Correct} / \\textsc{Broken} / \\textsc{Marginal}",
             f"{word_int(comp['correct'])} / {word_int(comp['broken'])} / {comp['marginal']}"),
            ("Distinct candidates graded (counted per episode)", word_int(fixes)),
            ("Episodes with at least one \\textsc{Broken} candidate", macros["nWithBroken"]),
            ("Mean false-pass rate $q$ of \\textsc{Broken} candidates", macros["meanQ"]),
            ("Blocks executed (container hours)", f"{macros['nBlocksRun']} ({macros['blockHours']})")]
    write("tab_composition.tex",
          "\\begin{tabular}{@{}lr@{}}\n\\toprule\n\\textbf{Property} & \\textbf{Value}\\\\\n\\midrule\n"
          + "".join(f"{a} & {b}\\\\\n" for a, b in rows) + "\\bottomrule\n\\end{tabular}\n")
    return bugs, tasks


def screening():
    rows = list(csv.DictReader(open("screening/summary.csv")))
    v = Counter(r["verdict"] for r in rows)
    m("nScreened", len(rows))
    m("nScreenPassed", v["candidate"])
    m("nScreenPzeroHigh", v["p0 > 0.05 without the flaky tests"])
    m("nScreenNoRun", v["tests do not run"] + v["tests need uninstalled modules"] + v["no tests left without the flaky ones"])
    m("nScreenNoFlake", v["flake not reproduced"])
    m("nScreenBuild", v["build failed"])
    assert sum(v.values()) == len(rows) and set(v) <= {"candidate", "p0 > 0.05 without the flaky tests",
                                                         "tests do not run", "tests need uninstalled modules",
                                                         "no tests left without the flaky ones",
                                                         "flake not reproduced", "build failed"}, v
    cand = [r for r in rows if r["verdict"] == "candidate"]
    rates = [float(r["flaky_rate"]) for r in cand]
    m("nDefaultOrderPassed", sum(r["default_order_failed"] == "0" for r in cand))
    m("minRandomFlakeRate", pct(min(rates), 0))
    m("maxRandomFlakeRate", pct(max(rates), 0))
    from screen_idoft import SCREEN_BLOCKS
    m("screenBlocks", SCREEN_BLOCKS)


def main_table(res):
    t = res["table"]
    for name in RULES:
        s, k = t[name], KEY[name]
        m(f"far{k}", pct(s["far"]))
        m(f"farLo{k}", pct(s["far_ci"][0]))
        m(f"farHi{k}", pct(s["far_ci"][1]))
        m(f"frr{k}", pct(s["frr"]))
        m(f"marg{k}", pct(s["marginal_accept"]))
        m(f"bpa{k}", num(s["blocks_per_accept"]))
        m(f"budget{k}", num(s["budget_x_b1"]))
        m(f"tta{k}", f"{s['time_to_accept_min']:.2f}")
        m(f"calib{k}", f"{s['calibration']:.2f}")
        k_ep, n_ep, up = s["far_upper"]["episodes"]
        m(f"farEpHits{k}", k_ep)
        m(f"farUpper{k}", pct(up))
        kr, nr, upr = s["far_upper"]["repos"]
        m(f"farRepoHits{k}", kr)
        m(f"farRepoUpper{k}", pct(upr))
    m("nRepos", t["SeqAccept"]["far_upper"]["repos"][1])
    sa, b1, b3 = t["SeqAccept"], t["B1 stop at first pass"], t["B3 majority of 5"]
    m("saOverBthreeBlocks", f"{sa['blocks_per_accept'] / b3['blocks_per_accept']:.0f}")
    m("saOverBthreeBudget", f"{sa['budget_x_b1'] / b3['budget_x_b1']:.0f}")
    m("saSecondsToAccept", f"{60 * sa['time_to_accept_min']:.0f}")
    m("boneSecondsToAccept", f"{60 * b1['time_to_accept_min']:.0f}")
    m("saBudgetSeconds", f"{sa['budget_seconds']:.0f}")
    m("boneBudgetSeconds", f"{b1['budget_seconds']:.1f}")
    m("refundBlockSaving", pct(1 - sa["blocks_per_accept"] / t["B6 SeqAccept, no refund"]["blocks_per_accept"]))
    p = res["primary"]
    m("primaryDiff", pct(p["diff"]))
    m("primaryLo", pct(p["ci"][0]))
    m("primaryHi", pct(p["ci"][1]))
    m("primaryMargin", pct(p["margin"], 0))
    assert p["exceeds"]
    m("nReps", res["settings"]["reps"])

    bug_hits = {}
    for name in RULES:
        hit = {}
        for e in res["per_episode"]:
            if "broken" in e["labels"]:
                b = e["file"].split("-")[0].split("@")[0]
                hit[b] = hit.get(b, False) or e["rules"][name]["far"] > 0
        bug_hits[name] = (sum(hit.values()), len(hit), replay.cp_upper(sum(hit.values()), len(hit)))
        m(f"bugHits{KEY[name]}", bug_hits[name][0])
        m(f"bugUpper{KEY[name]}", pct(bug_hits[name][2]))
    body = ""
    for name in RULES:
        s = t[name]
        if name == "SeqAccept":
            body += "\\midrule\n"
        ci = "" if s["far_ci"] == [0.0, 0.0] else f" {{\\scriptsize[{pct(s['far_ci'][0])}, {pct(s['far_ci'][1])}]}}"
        far = f"{pct(s['far'])}{ci}"
        kb, nb, ub = bug_hits[name]
        body += (f"{SHOW[name]} & {far} & {kb} & {pct(ub)} & {s['far_upper']['episodes'][0]} & "
                 f"{pct(s['frr'])} & {pct(s['marginal_accept'])} & {num(s['blocks_per_accept'])} & "
                 f"{num(s['budget_x_b1'])} & {s['time_to_accept_min']:.2f} & {s['calibration']:.2f}\\\\\n")
    n_ep = t["SeqAccept"]["far_upper"]["episodes"][1]
    nb = bug_hits["SeqAccept"][1]
    write("tab_main.tex",
          "\\begin{tabular}{@{}lrrrrrrrrrr@{}}\n\\toprule\n"
          "& & \\multicolumn{2}{c}{\\textbf{Bugs hit}} & \\textbf{Ep.} & & & & & & \\\\\n\\cmidrule(lr){3-4}\n"
          "\\textbf{Rule} & \\textbf{FAR \\%} {\\scriptsize[95\\% CI]} & \\textbf{of " + str(nb) + "} & "
          "\\textbf{UB \\%} & \\textbf{of " + str(n_ep) + "} & \\textbf{FRR \\%} & \\textbf{Marg.\\ \\%} & \\textbf{Blocks/acc.} & "
          "\\textbf{Budget} & \\textbf{Min.} & \\textbf{Inj./$\\alpha$}\\\\\n\\midrule\n" + body
          + "\\bottomrule\n\\end{tabular}\n")


def splits(res):
    sp = res["splits"]
    for model, key in MODELS.items():
        r = sp["model"][model]
        m(f"attempts{key}", f"{r['attempts']:.1f}")
        m(f"brokenShare{key}", pct(r["broken_share"]))
        for name in RULES[:3]:
            m(f"far{KEY[name]}{key}", pct(r["far"][name]))
        m(f"farBfour{key}", pct(r["far"]["B4 peeking Clopper-Pearson"]))
    valid = RULES[3:]
    for fl, key in (("natural", "Natural"), ("injected-bernoulli", "Bernoulli"), ("injected-markov", "Markov")):
        for name in RULES[:3]:
            m(f"far{KEY[name]}{key}", pct(sp["flakiness"][fl]["far"][name]))
    for bk, key in (("order", "Order"), ("ordinary", "Ordinary")):
        m(f"farBone{key}", pct(sp["bug_kind"][bk]["far"]["B1 stop at first pass"]))
    for band, key in (("q >= 0.1", "Qhigh"), ("0.01 <= q < 0.1", "Qmid"), ("q < 0.01", "Qlow")):
        m(f"nEp{key}", sp["q_band"][band]["streams"])
        for name in RULES[:3]:
            m(f"far{KEY[name]}{key}", pct(sp["q_band"][band]["far"][name]))
    # every split, every valid-looking rule: the largest FAR anywhere
    worst = max(r["far"][n] for groups in sp.values() for r in groups.values() for n in valid)
    m("maxSplitFarValid", pct(worst))
    worst_sa = max(r["far"][n] for groups in sp.values() for r in groups.values() for n in ("SeqAccept", "B6 SeqAccept, no refund", "B5 naive confidence sequence"))
    assert worst_sa == 0.0
    # persistence: FAR after the first k candidates
    pre = res["chart2_far_by_prefix"]
    m("prefixFirst", pre["prefixes"][0])
    m("prefixLast", pre["prefixes"][-1])
    for model, key in MODELS.items():
        ys = pre["far"][model]["B1 stop at first pass"]
        m(f"farBoneFirst{key}", pct(ys[0]))
        m(f"farBoneLast{key}", pct(ys[-1]))
    per = res["per_episode"]
    for model, key in MODELS.items():
        mine = [e for e in per if e["model"] == model]
        for name in RULES[:3]:
            lo, hi = replay.cluster_ci([e["rules"][name]["far"] for e in mine], [e["repo"] for e in mine])
            m(f"farLo{KEY[name]}{key}", pct(lo))
            m(f"farHi{KEY[name]}{key}", pct(hi))
        m(f"nWithCorrect{key}", sum("correct" in e["labels"] for e in mine))
    bugs = {}
    for e in per:
        if "broken" in e["labels"]:
            b = e["file"].split("-")[0].split("@")[0]
            bugs[b] = bugs.get(b, False) or e["rules"]["SeqAccept"]["far"] > 0
    m("nBugsWithBroken", len(bugs))
    nat = [e for e in per if e["flakiness"] == "natural"]
    m("nNaturalWithBroken", sum("broken" in e["labels"] for e in nat))
    for name in RULES:
        m(f"farNat{KEY[name]}", pct(sum(e["rules"][name]["far"] for e in nat) / len(nat)))
    gap = max(pre["far"][mo][n][i] - pre["far"][mo]["SeqAccept"][i] for mo in pre["far"] for n in RULES[3:6]
              for i in range(len(pre["prefixes"])))
    m("maxPrefixGapValid", pct(gap))
    shares = [sp["model"][mo]["broken_share"] for mo in MODELS]
    m("brokenShareRatio", f"{max(shares) / min(shares):.1f}")
    return pre


def figure_persistence(pre):
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False})
    order = ["llama3.1:8b", "gpt-4o", "gpt-4.1-mini", "grok-4.7"]
    shown = ["B1 stop at first pass", "B3 majority of 5", "B2 rerun-3 all green", "SeqAccept"]
    fig, axes = plt.subplots(1, 4, figsize=(5.4, 1.65), sharey=True)
    for ax, model in zip(axes, order):
        for name in shown:
            ys = [100 * v for v in pre["far"][model][name]]
            ax.plot(pre["prefixes"], ys, LINE[name], color=COLOR[name], marker=MARK[name], ms=3.2, lw=1.2,
                    label=SHOW[name].replace("\\sa{}", "SeqAccept").replace("\\ ", " "))
        valid = [100 * max(pre["far"][model][n][i] for n in RULES[3:6]) for i in range(len(pre["prefixes"]))]
        ax.plot(pre["prefixes"], valid, "-", color="#8a8984", lw=2.2, alpha=0.6, zorder=1, label="B4-B6 (highest)")
        ax.axhline(5, color=MUTED, lw=0.6, ls=(0, (2, 2)))
        ax.set_title(MODEL_SHOW[model], fontsize=7, color=INK)
        ax.set_xticks(pre["prefixes"])
        ax.set_ylim(-2, 50)
        ax.grid(axis="y", color="#e6e5e1", lw=0.5)
        ax.set_xlabel("candidates tried")
    axes[0].set_ylabel("FAR (%)")
    axes[0].text(15, 6, r"$\alpha$", color=MUTED, fontsize=7, ha="right", va="bottom")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=6.2, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(top=0.78)
    fig.savefig(FIGS / "persistence.pdf", bbox_inches="tight")
    plt.close(fig)


def sensitivity(res):
    mg = res["chart3_far_by_margin"]
    for i, k in enumerate(("Three", "Five", "Ten")):
        for name in ("B4 peeking Clopper-Pearson", "B5 naive confidence sequence", "B6 SeqAccept, no refund", "SeqAccept"):
            m(f"far{KEY[name]}Margin{k}", pct(mg["far"][name][i], 2))
    for name in RULES[:3]:
        assert len(set(mg["far"][name])) == 1  # B1-B3 do not use eps

    # boundary stress
    nm = res["nearmiss"]
    rows = ""
    for mask, key in (("0.8", "Eighty"), ("0.9", "Ninety"), ("0.94", "NinetyFour")):
        d = nm[mask]
        c = d["composition"]
        m(f"stressBroken{key}", c["broken"])
        m(f"stressNear{key}", c["broken_within_0.05_of_eps"])
        m(f"stressRuns{key}", word_int(d["runs"]))
        for name in RULES:
            s = d["table"][name]
            m(f"stress{KEY[name]}{key}", pct(s["far"], 2))
            m(f"stressHits{KEY[name]}{key}", word_int(s["false_accepts"]))
        b4 = d["table"]["B4 peeking Clopper-Pearson"]["far"]
        m(f"stressBfourOverAlpha{key}", f"{b4 / replay.ALPHA:.1f}")
        rows += (f"{mask} & {1 - float(mask):.2f} & {c['broken']} ({c['broken_within_0.05_of_eps']}) & "
                 + " & ".join(pct(d["table"][n]["far"], 2) for n in RULES) + "\\\\\n")
    m("stressReps", replay.NEAR_REPS)
    m("stressEpisodes", nm["0.8"]["composition"]["episodes"])
    write("tab_boundary.tex",
          "\\begin{tabular}{@{}lll" + "r" * len(RULES) + "@{}}\n\\toprule\n"
          "$m$ & \\textbf{Max $p$} & \\textbf{Broken (near $\\eps$)} & "
          + " & ".join(f"\\textbf{{{s.split(' ')[0] if s.startswith('B') else 'SA'}}}" for s in SHOW.values())
          + "\\\\\n\\midrule\n" + rows + "\\bottomrule\n\\end{tabular}\n")

    # late streams
    late = res["late_streams"]
    rows = ""
    for k in ("5", "10", "20", "40"):
        d = late[k]
        for name in RULES:
            s = d["table"][name]
            m(f"late{KEY[name]}K{['Five', 'Ten', 'Twenty', 'Forty'][['5', '10', '20', '40'].index(k)]}", pct(s["late_accept"]))
        rows += f"{k} & " + " & ".join(pct(d["table"][n]["late_accept"]) for n in RULES) + "\\\\\n"
    m("lateStreams", late["5"]["streams"])
    assert len({d["streams"] for d in late.values()}) == 1
    for name in RULES:
        m(f"lateFar{KEY[name]}Forty", pct(late["40"]["table"][name]["far"]))
        m(f"lateBpa{KEY[name]}Forty", num(late["40"]["table"][name]["blocks_per_accept"]))
    m("lateFrrBsixForty", pct(late["40"]["table"]["B6 SeqAccept, no refund"]["frr"]))
    m("lateFrrSaForty", pct(late["40"]["table"]["SeqAccept"]["frr"]))
    write("tab_late.tex",
          "\\begin{tabular}{@{}r" + "r" * len(RULES) + "@{}}\n\\toprule\n$k$ & "
          + " & ".join(f"\\textbf{{{s.split(' ')[0] if s.startswith('B') else 'SA'}}}" for s in SHOW.values())
          + "\\\\\n\\midrule\n" + rows + "\\bottomrule\n\\end{tabular}\n")

    # injection-rate sweep
    sw = res["injection_sweep"]
    for fl, key in (("injected-bernoulli10", "Ten"), ("injected-bernoulli", "Thirty"), ("injected-bernoulli60", "Sixty")):
        d = sw[fl]
        m(f"sweepRate{key}", f"{d['false_pass']:.1f}")
        m(f"sweepQ{key}", f"{d['composition']['mean_false_pass_rate_q']:.2f}")
        m(f"sweepEpisodes{key}", d["episodes"])
        for name in RULES:
            m(f"sweep{KEY[name]}{key}", pct(d["table"][name]["far"]))
        m(f"sweepBpaSa{key}", num(d["table"]["SeqAccept"]["blocks_per_accept"], 0))
    m("sweepMaxValid", pct(max(d["table"][n]["far"] for d in sw.values() for n in RULES[3:])))


def worst_case():
    w = load("results/worst_case.json")
    shown = RULES[3:]
    rows = ""
    for r in w["rows"]:
        rows += f"{r['p']:.3f} & {r['n']} & " + " & ".join(pct(r["far"][n]) for n in shown) + "\\\\\n"
    lo_n1 = min(r["far"][n] for r in w["rows"] if r["n"] == 1 for n in RULES[:3])
    m("worstFixedMinNone", pct(lo_n1, 0))
    assert all(r["far"][n] == 1.0 for r in w["rows"] if r["n"] >= 5 for n in RULES[:3])
    reps = sorted({r["reps"] for r in w["rows"]})
    m("worstRepsSmall", word_int(reps[-1]))
    m("worstRepsLarge", word_int(reps[0]))
    half = max((wilson(round(r["far"][n] * r["reps"]), r["reps"])[1] - wilson(round(r["far"][n] * r["reps"]), r["reps"])[0]) / 2
               for r in w["rows"] for n in shown)
    m("worstHalfWidth", f"{100 * half:.1f}")
    for r in w["rows"]:
        if r["p"] == 0.051 or (r["p"] == 0.06 and r["n"] in (15, 50)):
            pk = {0.051: "Edge", 0.06: "Six"}[r["p"]]
            nk = {1: "One", 5: "Five", 15: "Fifteen", 50: "Fifty"}[r["n"]]
            for n in shown:
                m(f"worst{KEY[n]}{pk}{nk}", pct(r["far"][n]))
    m("worstMaxSa", pct(max(r["far"]["SeqAccept"] for r in w["rows"])))
    for r in w["rows"]:
        if r["n"] == 15 and r["p"] in (0.051, 0.06, 0.08):
            m(f"worstBlocksPerCand{ {0.051: 'Edge', 0.06: 'Six', 0.08: 'Eight'}[r['p']] }",
              f"{r['blocks']['SeqAccept'] / r['n']:.0f}")
    m("worstGapBsix", pct(max(abs(r["far"]["B6 SeqAccept, no refund"] - r["far"]["SeqAccept"]) for r in w["rows"])))
    m("worstMaxBsix", pct(max(r["far"]["B6 SeqAccept, no refund"] for r in w["rows"])))
    write("tab_worst.tex",
          "\\begin{tabular}{@{}lr" + "r" * len(shown) + "@{}}\n\\toprule\n$p$ & $n$ & "
          + " & ".join(f"\\textbf{{{SHOW[n].split(' ')[0] if SHOW[n].startswith('B') else 'SA'}}}" for n in shown) + "\\\\\n\\midrule\n" + rows
          + "\\bottomrule\n\\end{tabular}\n")

    fig, axes = plt.subplots(1, 3, figsize=(5.4, 1.65), sharey=True)
    for ax, p in zip(axes, (0.051, 0.06, 0.08)):
        rows = [r for r in w["rows"] if r["p"] == p]
        ns = [r["n"] for r in rows]
        for name in ("B4 peeking Clopper-Pearson", "B5 naive confidence sequence", "SeqAccept"):
            ax.plot(ns, [100 * r["far"][name] for r in rows], LINE[name], color=COLOR[name], marker=MARK[name],
                    ms=3.2, lw=1.2, label=SHOW[name].replace("\\sa{}", "SeqAccept").replace("\\ ", " "))
        ax.axhline(5, color=MUTED, lw=0.6, ls=(0, (2, 2)))
        ax.set_xscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.minorticks_off()
        ax.set_title(f"$p = {p}$", fontsize=7, color=INK)
        ax.set_xlabel("broken candidates in stream")
        ax.grid(axis="y", color="#e6e5e1", lw=0.5)
        ax.set_ylim(-3, 103)
    axes[0].set_ylabel("FAR (%)")
    axes[0].text(50, 7, r"$\alpha$", color=MUTED, fontsize=7, ha="right", va="bottom")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(top=0.78)
    fig.savefig(FIGS / "worstcase.pdf", bbox_inches="tight")
    plt.close(fig)

    oc = load("results/operating_curve.json")
    for r in oc["single"]:
        key = {0.0: "Zero", 0.01: "One", 0.02: "Two", 0.03: "Three", 0.04: "Four", 0.05: "Five", 0.06: "Six",
               0.1: "Ten"}.get(r["p"])
        if key:
            m(f"opAccept{key}", pct(r["accept"]))
            m(f"opBlocks{key}", f"{r['blocks']:.0f}")


def reuse():
    d = load("results/reuse_demo.json")
    for r in d["rows"]:
        key = {0.05: "Five", 0.01: "One", 0.001: "Tenth"}[r["alpha"]]
        m(f"reuseFresh{key}", pct(r["fresh"]))
        m(f"reuseReused{key}", pct(r["reused"]))
    p = d["params"]
    m("reuseBranches", p["K"])
    m("reuseBad", p["n_bad"])
    m("reuseStop", f"{p['stop_at']}")
    m("reuseReps", word_int(d["reps"]))
    m("reusePtwo", f"{p['n_bad'] / p['K']:.2f}")
    import math  # Theorem 2's constant at the defaults (eps = 0.05, gamma = 1/2, largest s)
    eps_, gamma_ = 0.05, 0.5
    s_ = (1 - eps_) / (1 - 0.1 * eps_)
    m_ = math.ceil(math.log(1 / (s_ * (1 - gamma_))) / math.log(1 + eps_ / 2))
    m("reuseM", m_)
    m("reuseC", f"{(1 - 2 * eps_) ** m_:.3f}")
    eta_ = 0.01  # sharper constant: p_2 <= eps + 1/K <= eps + eta once K >= 1/eta
    m("reuseEta", f"{eta_:.2f}")
    m("reuseCeta", f"{(1 - eps_ - eta_) ** m_:.2f}")


def online():
    eps = []
    for p in sorted(Path("online").glob("*.json")):
        if "freezegun" in p.stem or "@b7" in p.stem:
            continue
        d = load(p)
        assert d.get("finished"), p
        eps.append(d)
    lab = Counter(d["label"] for d in eps)
    m("nOnline", len(eps))
    per = Counter(ONLINE_MODEL.get(d["model"], d["model"]) for d in eps)
    assert len(set(per.values())) == 1
    m("onlinePerModel", per.most_common(1)[0][1])
    accepted = len(eps) - lab[None]
    m("onlineAccepted", accepted)
    m("onlineCorrect", lab["correct"])
    m("onlineMarginal", lab["marginal"])
    inj = [d for d in eps if d.get("flakiness", "natural").startswith("injected")]
    marg_nat = sum(d["label"] == "marginal" for d in eps if d not in inj)
    assert marg_nat == 0, "text: every Marginal acceptance online is on an injected task"
    assert all(d["flakiness"] == "injected-bernoulli" for d in inj if d["label"] == "marginal"), "text: all Bernoulli"
    m("onlineMarginalInjected", sum(d["label"] == "marginal" for d in inj))
    m("onlineAcceptedInjected", sum(d["label"] is not None for d in inj))
    m("onlineBroken", lab["broken"])
    m("onlineNone", lab[None])
    m("onlineBrokenUpper", pct(replay.cp_upper(lab["broken"], accepted)))
    m("onlineBrokenUpperAll", pct(replay.cp_upper(lab["broken"], len(eps))))
    blocks = [sum(c["blocks"] for c in d["candidates"]) for d in eps]
    m("onlineMedianBlocks", f"{median(blocks):.0f}")
    rejected = [c for d in eps for i, c in enumerate(d["candidates"]) if i != d["accepted"] and c["blocks"]]
    m("onlineCandidates", sum(len(d["candidates"]) for d in eps))
    m("onlineRejected", len(rejected))
    m("onlineRejectedHigh", sum(c["fails"] / c["blocks"] > d_eps for c, d_eps in
                                ((c, d["eps"]) for d in eps for i, c in enumerate(d["candidates"])
                                 if i != d["accepted"] and c["blocks"])))
    m("onlineMultiCandidate", sum(len(d["candidates"]) > 1 for d in eps))
    m("onlineMedianCandidates", f"{median(len(d['candidates']) for d in eps):.0f}")
    flak = Counter(d["flakiness"] for d in eps)
    m("onlineNatural", flak["natural"] // len(per))
    m("onlineInjected", (len(eps) - flak["natural"]) // len(per))
    rows = ""
    for model in ("llama3.1:8b", "gpt-4o", "gpt-4.1-mini", "grok-4.7"):
        mine = [d for d in eps if ONLINE_MODEL.get(d["model"], d["model"]) == model]
        c = Counter(d["label"] for d in mine)
        rows += (f"{model} & {len(mine)} & {c['correct']} & {c['marginal']} & {c['broken']} & {c[None]} & "
                 f"{median(sum(x['blocks'] for x in d['candidates']) for d in mine):.0f}\\\\\n")
        key = MODELS[model]
        m(f"onlineCorrect{key}", c["correct"])
        m(f"onlineNone{key}", c[None])
    rows += (f"\\midrule\nAll & {len(eps)} & {lab['correct']} & {lab['marginal']} & {lab['broken']} & {lab[None]} & "
             f"{median(blocks):.0f}\\\\\n")
    write("tab_online.tex",
          "\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n\\textbf{Model} & \\textbf{Ep.} & \\textbf{Corr.} & "
          "\\textbf{Marg.} & \\textbf{Brok.} & \\textbf{None} & \\textbf{Med.\\ blocks}\\\\\n\\midrule\n"
          + rows + "\\bottomrule\n\\end{tabular}\n")


def constants():
    import inspect

    from rules import seq_accept
    d = {k: v.default for k, v in inspect.signature(seq_accept).parameters.items() if v.default is not inspect._empty}
    m("alphaVal", f"{d['alpha']}")
    m("rejectAt", d["reject_at"])
    from rules import derived_cap
    m("capBlocks", derived_cap(replay.MARGIN))  # the derived cap at p0 = 0
    m("capBlocksHigh", f"{derived_cap(0.2 + replay.MARGIN):,}".replace(",", "{,}"))  # the derived cap at p0 = 0.2
    m("marginVal", f"{replay.MARGIN}")
    m("lateK", replay.LATE)
    m("capSeconds", replay.CAP_SECONDS)
    import build_episode
    b = inspect.signature(build_episode.build_episode).parameters
    m("nTruth", b["n_truth"].default)
    m("nPool", b["n_pool"].default)
    m("probeBlocks", b["probe"].default)
    m("probeFrac", pct(b["probe_frac"].default, 0))
    import subprocess
    import sys
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], capture_output=True, text=True).stdout
    m("nUnitTests", out.strip().splitlines()[-1].split()[0])  # "258 tests collected"
    bern, mark = load("tasks/filestack@bernoulli.json")["inject"], load("tasks/filestack@markov.json")["inject"]
    assert all(load(f"tasks/{t.stem}.json")["inject"] == (bern if "bernoulli" in t.stem else mark)
               for t in Path("tasks").glob("*@*.json") if t.stem.endswith(("@bernoulli", "@markov")))
    m("bernFalsePass", f"{bern['false_pass']}")
    m("bernFalseFail", f"{bern['false_fail']}")
    m("markEnter", f"{mark['enter']}")
    m("markStay", f"{mark['stay']}")
    m("markFalsePass", f"{mark['false_pass']}")
    m("markFalseFail", f"{mark['false_fail']}")
    import collect
    import llm
    import run_block
    m("numCtx", word_int(llm.NUM_CTX))
    m("numCtxK", llm.NUM_CTX // 1024)
    m("memoryCap", run_block.MEMORY.upper().replace("G", "~GB"))
    m("maxBlockSeconds", collect.MAX_BLOCK_SECONDS)
    m("nBoot", word_int(inspect.signature(replay.cluster_ci).parameters["n_boot"].default))
    for mk, key in zip(replay.MASKS, ("Eighty", "Ninety", "NinetyFour")):
        m(f"mask{key}", f"{mk}")
    for k, key in zip(replay.LATE_K, ("Five", "Ten", "Twenty", "Forty")):
        m(f"lateK{key}", k)
    for mg, key in zip(replay.MARGINS, ("Three", "Five", "Ten")):
        m(f"margin{key}", f"{mg:.2f}")


def extra(res):
    """Review-round analyses from results/extra.json (extra_analyses.py)."""
    d = load("results/extra.json")
    m("kMatch", d["k_match"])
    fg = d["first_green"]["natural"]
    tot = {k: sum(v[k] for v in fg.values()) for k in ("episodes", "broken", "correct", "marginal", "none")}
    m("firstGreenEpisodes", tot["episodes"])
    m("firstGreenBroken", tot["broken"])
    m("firstGreenNone", tot["none"])
    m("firstGreenBrokenPct", pct(tot["broken"] / (tot["episodes"] - tot["none"])))
    m("firstGreenWith", tot["episodes"] - tot["none"])
    for model, key in MODELS.items():
        m(f"firstGreenWith{key}", fg[model]["episodes"] - fg[model]["none"])
    for model, key in MODELS.items():
        m(f"firstGreenBroken{key}", fg[model]["broken"])
        m(f"firstGreenEpisodes{key}", fg[model]["episodes"])
    tr = d["truncated"]
    m("truncEpisodes", tr["episodes"])
    m("truncDropped", tr["dropped_no_green"])
    for name in RULES:
        m(f"trunc{KEY[name]}", pct(tr["table"][name]["far"]))
    m("truncMaxValid", pct(max(s["far"] for n, s in tr["table"].items() if n not in RULES[:3])))
    for name in RULES[3:]:
        m(f"relabel{KEY[name]}", pct(d["margin_relabel"]["table"][name]["far"], 2))
    bp = d["boundary_pool"]
    for mk, key in (("0.8", "Eighty"), ("0.9", "Ninety"), ("0.94", "NinetyFour")):
        for name in ("B4 peeking Clopper-Pearson", "SeqAccept"):
            m(f"pool{KEY[name]}{key}", pct(bp[mk][name]["far"], 2))
    st = d["stress"]
    for design, key in (("alt-0.10", "Ten"), ("alt-0.08", "Eight")):
        f = st[design]["far"]
        m(f"stressMixSa{key}", pct(f["SeqAccept"]))
        m(f"stressMixUncapped{key}", pct(f["SeqAccept, uncapped refund"]))
        m(f"stressMixBsix{key}", pct(f["B6 SeqAccept, no refund"]))
        m(f"stressMixHarm{key}", pct(f["Harmonic spending"]))
        m(f"stressMixDisc{key}", pct(f["Discard spending"]))
        m(f"stressMixBfive{key}", pct(f["B5 naive confidence sequence"]))
        m(f"stressMixBfour{key}", pct(f["B4 peeking Clopper-Pearson"]))
    m("stressMixN", st["alt-0.10"]["n"])
    m("stressMixReps", word_int(st["alt-0.10"]["reps"]))
    corr = {(r["rho"], r["n"]): r["far"] for r in d["correlated"]["rows"]}
    m("corrP", f"{d['correlated']['p']}")
    for (rho, n), far in corr.items():
        rk = {0.0: "Zero", 0.2: "Low", 0.5: "Mid", 0.8: "High"}[rho]
        nk = {1: "One", 15: "Fifteen"}[n]
        for name, tag in (("SeqAccept", "Sa"), ("B6 SeqAccept, no refund", "Bsix"), ("Harmonic spending", "Harm"),
                          ("B5 naive confidence sequence", "Bfive"), ("B4 peeking Clopper-Pearson", "Bfour")):
            m(f"corr{tag}{rk}{nk}", pct(far[name]))
    sr = d["serial"]
    m("serialTested", word_int(sr["tested"]))
    m("serialRejected", sr["rejected"])
    m("serialPct", pct(sr["rejected"] / sr["tested"]))
    m("lagMedian", f"${sr['lag1_median']:.2f}$")
    m("lagOver", sr["lag1_abs_over_02"])
    ne = d["near_eps"]
    m("nearBroken", word_int(ne["broken"]))
    m("nearThree", ne["within_003"])
    m("nearFive", ne["within_005"])
    m("nearThreePct", pct(ne["within_003"] / ne["broken"]))
    ml = d["mixed_late"]
    for name, tag in (("SeqAccept", "Sa"), ("B6 SeqAccept, no refund", "Bsix"), ("Harmonic spending", "Harm")):
        m(f"mixedLateMin{tag}", pct(min(v[name]["correct_accept"] for v in ml.values())))
    m("mixedLateBonfFifteen", pct(ml["15"]["Bonferroni-15 spending"]["correct_accept"]))
    m("mixedLateFixnFifteen", pct(ml["15"]["Fixed-n Bonferroni-15 (1n0)"]["correct_accept"]))
    m("mixedLateMaxFar", pct(max(v[n]["far"] for v in ml.values() for n in v)))
    fs = d["fullsuite"]
    for name in RULES[:3]:
        k = KEY[name]
        m(f"fullFar{k}", pct(fs["table"][name]["far"]))
        m(f"fullFarLo{k}", pct(fs["table"][name]["far_ci"][0]))
        m(f"fullFarHi{k}", pct(fs["table"][name]["far_ci"][1]))
        for fl, tag in (("natural", "Nat"), ("injected-bernoulli", "Bern"), ("injected-markov", "Markov")):
            m(f"full{tag}{k}", pct(fs[fl][name]))
    fp = fs["natural_false_pass_bugs"]
    assert set(fp.values()) == {"order"}, fp  # the text says only order bugs pass a full-suite run
    m("fullFalsePassBugs", word_int(len(fp)))
    dp = fs["default_first_pass"]
    m("defaultEpisodes", dp["episodes"])
    m("defaultWith", dp["episodes"] - dp["none"])
    m("defaultBroken", word_int(dp["broken"]))
    m("defaultBrokenPct", pct(dp["broken"] / (dp["episodes"] - dp["none"])))
    runs = {len(load(p)["random"]) for p in Path("fullsuite").glob("*.json")}
    assert len(runs) == 1, runs
    m("fullRuns", runs.pop())
    bp = fs["broken_passes"]
    m("fullBrokenCands", sum(bp.values()))
    m("fullBrokenAllFail", bp["0"])
    m("fullBrokenSomePass", sum(bp.values()) - bp["0"])
    m("fullBrokenMaxPass", max(int(k) for k, v in bp.items() if v))
    m("fullMedianSeconds", f"{fs['cost']['full_median']:.2f}")
    m("blockMedianSeconds", f"{fs['cost']['block_median']:.2f}")
    m("testsPerBlock", f"{fs['cost']['tests_per_block']:.1f}")
    rf = d["realflaky"]  # real-flaky arm: natural loops re-graded with the repositories' IDoFT flaky tests kept
    m("rfEpisodes", rf["episodes"])
    m("rfBugs", rf["bugs"])
    m("rfMinPzero", f"{min(rf['p0'].values()):.2f}")
    m("rfMaxPzero", f"{max(rf['p0'].values()):.2f}")
    m("rfHighTasks", word_int(sum(p >= 0.2 for p in rf["p0"].values())))
    for name in RULES:
        k = KEY[name]
        small = lambda v: "0" if v == 0 else pct(v, 2 if v < 0.01 else 1)  # noqa: E731
        m(f"rfFar{k}", small(rf["table"][name]["far"]))
        m(f"rfFarWithout{k}", small(rf["same_loops_without"][name]["far"]))
        m(f"rfFrr{k}", pct(rf["table"][name]["frr"]))
        m(f"rfBpa{k}", num(rf["table"][name]["blocks_per_accept"], 0))
        m(f"rfBpaWithout{k}", num(rf["same_loops_without"][name]["blocks_per_accept"], 0))
        m(f"rfFrrHigh{k}", pct(rf["p0_high"]["table"][name]["frr"]))
        m(f"rfFrrLow{k}", pct(rf["p0_low"]["table"][name]["frr"]))
    assert all(rf["same_loops_without"][n]["frr"] == 0 for n in RULES[3:]), "text: no eps-based rule loses a correct patch without them"
    hit = rf["table"]["SeqAccept"]["far_upper"]
    m("rfSaEpHits", word_int(hit["episodes"][0]))
    m("rfWithBroken", hit["episodes"][1])
    m("rfSaRepoHits", word_int(hit["repos"][0]))
    hit_repos = {}  # repo -> highest p0 among its tasks with a SeqAccept false accept; the text names them
    for b in rf["seqaccept_false_accept_bugs"]:
        hit_repos[b.split("_b")[0]] = max(hit_repos.get(b.split("_b")[0], 0), rf["p0"][b])
    assert hit_repos, "text: SeqAccept has false accepts on this arm"
    parts = [f"{r} ($p_0={p:.2f}$)" for r, p in sorted(hit_repos.items())]
    m("rfSaHitRepos", parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1])
    m("rfRepos", rf["repos"])
    m("rfRepoHitsBound", rf["table"]["SeqAccept"]["far_upper"]["repos"][1])
    m("rfFarLoSa", pct(rf["table"]["SeqAccept"]["far_ci"][0], 2))
    m("rfFarHiSa", pct(rf["table"]["SeqAccept"]["far_ci"][1], 2))
    m("rfFrrLoSa", pct(rf["frr_ci_seqaccept"][0]))
    m("rfFrrHiSa", pct(rf["frr_ci_seqaccept"][1]))
    m("rfFarLoBfour", pct(rf["table"]["B4 peeking Clopper-Pearson"]["far_ci"][0], 2))
    m("rfFarHiBfour", pct(rf["table"]["B4 peeking Clopper-Pearson"]["far_ci"][1]))
    m("rfPzeroSeMax", f"{max(rf['p0_se'].values()):.3f}")
    m("deltaEffRf", f"{0.05 + 1.645 * max(rf['p0_se'].values()):.3f}")  # eps = p0hat + 0.05 = L + delta_eff, one-sided 95% normal bound
    mt = rf["margin_010"]
    m("rfMarginFrrSa", pct(mt["SeqAccept"]["frr"]))
    m("rfMarginBpaSa", num(mt["SeqAccept"]["blocks_per_accept"], 0))
    for name in ("B4 peeking Clopper-Pearson", "B5 naive confidence sequence", "SeqAccept"):
        v = mt[name]["far"]
        m(f"rfMarginFar{KEY[name]}", "0" if v == 0 else pct(v, 2 if v < 0.01 else 1))
    good = {}  # distinct Correct candidates of the p0 >= 0.2 tasks: where their pool rate sits against eps
    for p in Path("episodes_flaky").glob("*.json"):
        ep = replay.load_episode(p)
        if ep["p0"] >= 0.2:
            eps = ep["p0"] + replay.MARGIN
            for f, lab in zip(ep["stream"], replay.labels(ep)):
                if lab == "correct":
                    good[(ep["name"].split("-")[0], f)] = sum(ep["fixes"][f]["pool"]) / len(ep["fixes"][f]["pool"]) - eps
    m("rfHighCorrect", len(good))
    vecs, seed_ranges, by_seed = set(), set(), {}  # outcomes are nearly fixed by the block seed
    for p in Path("episodes_flaky").glob("*.json"):
        ep = replay.load_episode(p)
        if ep["p0"] >= 0.2:
            for f, lab in zip(ep["stream"], replay.labels(ep)):
                if lab == "correct":
                    fx, task = ep["fixes"][f], ep["name"].split("-")[0]
                    vecs.add((task, tuple(fx["truth"]), tuple(fx["pool"])))
                    seed_ranges.add((task, fx["seeds"][0]))
                    by_seed.setdefault((task, fx["seeds"][0]), set()).add((tuple(fx["truth"]), tuple(fx["pool"])))
    m("rfHighCorrectVectors", len(vecs))
    m("rfHighCorrectSeeds", len(seed_ranges))
    m("rfHighCorrectSeedsSame", sum(len(v) == 1 for v in by_seed.values()))
    m("rfHighCorrectAbove", word_int(sum(v >= 0 for v in good.values())))
    m("rfHighCorrectNear", word_int(sum(-0.03 < v < 0 for v in good.values())))
    for tag, ne_ in (("", rf["near_eps"]), ("Without", rf["near_eps_without"])):
        m(f"rfNearBroken{tag}", ne_["broken"])
        m(f"rfNearThree{tag}", word_int(ne_["within_003"]))
        m(f"rfPoolBelow{tag}", word_int(ne_["pool_below_eps"]))
    rr = f"Rerun-{d['k_match']} (all green)"
    main, late, worst = d["main"], d["late"], {(r["p"], r["n"]): r["far"] for r in d["worst"]}
    w = load("results/worst_case.json")
    wsa = {(r["p"], r["n"]): r["far"] for r in w["rows"]}
    t = res["table"]
    rows = [("B6 (geometric)", t["B6 SeqAccept, no refund"], res["late_streams"], "B6 SeqAccept, no refund", wsa, st["alt-0.10"]["far"].get("B6 SeqAccept, no refund")),
            ("Harmonic", main["Harmonic spending"], {k: {"table": v} for k, v in late.items()}, "Harmonic spending", worst, st["alt-0.10"]["far"].get("Harmonic spending")),
            ("Discard ($\\tau=\\tfrac12$)", main["Discard spending"], {k: {"table": v} for k, v in late.items()}, "Discard spending", worst, st["alt-0.10"]["far"]["Discard spending"]),
            ("Bonferroni-15", main["Bonferroni-15 spending"], {k: {"table": v} for k, v in late.items()}, "Bonferroni-15 spending", worst, None),
            *[(f"Fixed-$n$ Bonf.-15 (${k}n_0$)".replace("$1n_0$", "$n_0$"), main[f"Fixed-n Bonferroni-15 ({k}n0)"], {kk: {"table": v} for kk, v in late.items()},
               f"Fixed-n Bonferroni-15 ({k}n0)", worst, None) for k in (1, 2, 4)],
            (f"Rerun-{d['k_match']} (consecutive)", main[rr], {k: {"table": v} for k, v in late.items()}, rr, worst, None),
            ("\\sa{}", t["SeqAccept"], res["late_streams"], "SeqAccept", wsa, st["alt-0.10"]["far"]["SeqAccept"])]
    body = ""
    for label_, s_, late_, key_, worst_, stress_ in rows:
        la = " / ".join(pct(late_[str(k)]["table"][key_]["late_accept"], 0) for k in (10, 20, 40))
        wf = " / ".join(pct(worst_[(0.051, n)][key_]) for n in (15, 50))
        valid_ = {"Discard spending": "unproved", rr: "no"}.get(key_, "yes")
        body += (f"{label_} & {valid_} & {num(s_['blocks_per_accept'])} & {pct(s_['frr'])} & {la} & "
                 f"{num(late_[str(40)]['table'][key_]['blocks_per_accept'] or 0, 0) if late_[str(40)]['table'][key_]['blocks_per_accept'] else '--'} & {wf} & "
                 f"{pct(stress_) if stress_ is not None else '--'}\\\\\n")
        tag = {"B6 SeqAccept, no refund": "Geo", "Harmonic spending": "Harm", "Bonferroni-15 spending": "Bonf", "Fixed-n Bonferroni-15 (1n0)": "FixnOne", "Fixed-n Bonferroni-15 (2n0)": "FixnTwo", "Fixed-n Bonferroni-15 (4n0)": "FixnFour",
               "Discard spending": "Disc",
               rr: "Rerun", "SeqAccept": "Sa"}[key_]
        m(f"spendBpa{tag}", num(s_["blocks_per_accept"]))
        m(f"spendFrr{tag}", pct(s_["frr"]))
        for k, kk in ((10, "Ten"), (20, "Twenty"), (40, "Forty")):
            m(f"spendLate{tag}{kk}", pct(late_[str(k)]["table"][key_]["late_accept"]))
            m(f"spendLateFrr{tag}{kk}", pct(late_[str(k)]["table"][key_]["frr"]))
        b40 = late_[str(40)]["table"][key_]["blocks_per_accept"]
        m(f"spendLateBpa{tag}Forty", num(b40, 0) if b40 else "--")
        for n, nn in ((15, "Fifteen"), (50, "Fifty")):
            m(f"spendWorst{tag}{nn}", pct(worst_[(0.051, n)][key_]))
    write("tab_spending.tex",
          "\\begin{tabular}{@{}llrrcrcr@{}}\n\\toprule\n"
          "& & \\multicolumn{2}{c}{\\textbf{Recorded loops}} & \\multicolumn{2}{c}{\\textbf{Late streams}} & "
          "\\textbf{Worst case} & \\textbf{Mixed}\\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
          "\\textbf{Rule} & \\textbf{Valid} & \\textbf{Blk/acc.} & \\textbf{FRR \\%} & \\textbf{Late acc.\\ \\%} & "
          "\\textbf{Blk/acc.\\ ($k$=40)} & \\textbf{FAR \\% ($n$=15/50)} & \\textbf{FAR \\%}\\\\\n\\midrule\n"
          + body + "\\bottomrule\n\\end{tabular}\n")


def power_tuning():
    """External-review analyses from results/power_tuning.json (power_tuning.py)."""
    d = load("results/power_tuning.json")
    m("mixK", d["mix_k"])
    for r in d["operating"]:
        key = {0.0: "Zero", 0.02: "Two", 0.03: "Three", 0.04: "Four", 0.05: "Five"}[r["p"]]
        for b, tag in (("kelly", "Kelly"), ("mixture", "Mix")):
            m(f"op{tag}Accept{key}", pct(r[b]["accept"]))
            m(f"op{tag}Blocks{key}", num(r[b]["blocks_per_accept"], 0))
        if r["kl_bound"] is not None:
            m(f"klBound{key}", num(r["kl_bound"], 0))
    for n, nk in (("15", "Fifteen"), ("50", "Fifty")):
        for b, tag in (("kelly", "Kelly"), ("mixture", "Mix")):
            m(f"worst{tag}Pow{nk}", pct(d["worst"][n][b]))
    rm = d["recorded_mixture"]
    m("mixRecBpa", num(rm["blocks_per_accept"]))
    assert rm["far"] == 0, "text: the mixture wallet has no false acceptance"
    m("mixRecFrr", pct(rm["frr"], 2))
    for r in d["sweep"]:
        tag = {(0.25, 5): "GammaLow", (0.75, 5): "GammaHigh", (0.5, 3): "RhoLow", (0.5, 20): "RhoHigh", (0.5, 5): "Default"}[(r["gamma"], r["rho"])]
        m(f"sweepBpa{tag}", num(r["recorded"]["blocks_per_accept"]))
        m(f"sweepFar{tag}", pct(r["recorded"]["far"], 2))
        m(f"sweepLateForty{tag}", pct(r["late"]["40"]["late_accept"]))
    m("sweepFrrMax", pct(max(r["recorded"]["frr"] for r in d["sweep"]), 2))
    m("sweepLateMin", pct(min(min(r["late"]["10"]["late_accept"], r["late"]["40"]["late_accept"]) for r in d["sweep"])))


def label_risk():
    """Second external review: results/label_risk.json (label_risk.py)."""
    d = load("results/label_risk.json")
    m("shortPool", d["short_pool"])
    sa = d["pool"]["SeqAccept"]
    assert sa["full"]["far"] == sa["short"]["far"] == 0 and sa["full"]["frr"] == sa["short"]["frr"], "text: unchanged"
    m("poolBpaFull", num(sa["full"]["blocks_per_accept"]))
    m("poolBpaShort", num(sa["short"]["blocks_per_accept"]))
    m("poolOverFull", pct(sa["full"]["over_pool"] / sa["full"]["evaluations"]))
    m("poolOverShort", pct(sa["short"]["over_pool"] / sa["short"]["evaluations"]))
    b4 = d["pool"]["B4 peeking Clopper-Pearson"]
    m("poolFarBfourFull", pct(b4["full"]["far"], 2))
    m("poolFarBfourShort", pct(b4["short"]["far"], 2))
    runs = len(main_episodes()) * replay.REPS  # replications behind each FAR
    m("poolRuns", word_int(runs))
    m("poolHitsBfourFull", round(b4["full"]["far"] * runs))
    m("poolHitsBfourShort", round(b4["short"]["far"] * runs))
    for arm, tag in (("natural", "Nat"), ("realflaky", "Rf")):
        lab = d["labels"][arm]
        assert lab["broken_risk50"] == 0 and lab["correct_risk50"] == 0, "text: no label is more likely wrong than right"
        m(f"labBroken{tag}", word_int(lab["broken"]))
        m(f"labBrokenRisk{tag}", word_int(lab["broken_risk05"]))
        m(f"labCorrect{tag}", word_int(lab["correct"]))
        m(f"labCorrectRisk{tag}", word_int(lab["correct_risk05"]))
    for r in d["planning"]:
        key = {0.0: "Zero", 0.05: "Five", 0.2: "Twenty"}[r["p0"]] + {0.05: "Small", 0.1: "Wide"}[r["margin"]]
        m(f"plan{key}", num(r["blocks"], 0))


def exec_order():
    """Execution-order check (exec_order.py, results/exec_order.json)."""
    d = load("results/exec_order.json")
    seqs = {k: v for k, v in d.items() if isinstance(v, dict)}
    assert all(v["order_inversions"] == 0 for v in seqs.values()), "text: blocks start in seed order"
    m("execBlocks", word_int(sum(v["blocks"] for v in seqs.values())))
    m("execSeqs", len(seqs))
    m("execTasks", len({k.split("-")[0] for k in seqs}))
    m("execRejects", sum(bool(v["runs_reject_exec"]) for v in seqs.values()))
    m("execLagMax", f"{max(abs(v['lag1_exec']) for v in seqs.values()):.2f}")
    m("execLoadZMax", f"{max(abs(v['busy_quiet_z']) for v in seqs.values()):.2f}")
    ya_s, ya_p = seqs["yamicache@realflaky-serial"], seqs["yamicache@realflaky-parallel"]
    m("execYamiSerial", pct(ya_s["fail_rate"]))
    m("execYamiParallel", pct(ya_p["fail_rate"]))
    m("execYamiZ", f"{d['yamicache@realflaky-serial_vs_parallel_z']:.1f}")
    m("execPenmanZ", f"{d['penman@realflaky-serial_vs_parallel_z']:.1f}")


def wilson(k, n, z=1.96):
    """95% Wilson interval for k events in n runs (Monte Carlo uncertainty of a simulated rate)."""
    if n == 0:
        return 0.0, 1.0
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * (ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5 / d
    return max(0.0, c - h), min(1.0, c + h)


def ease():
    """EASE 2027 mentor review: results/ablations.json (ablations.py)."""
    d = load("results/ablations.json")
    # 1. refund ablation
    tag = {"none": "None", "max": "Max", "capped": "Capped", "uncapped": "Uncapped"}
    for k, t in tag.items():
        r = d["refund"][k]
        m(f"refBpa{t}", num(r["recorded"]["blocks_per_accept"]))
        m(f"refFrr{t}", pct(r["recorded"]["frr"]))
        m(f"refFar{t}", pct(r["recorded"]["far"], 2))
        for kk, kt in (("10", "Ten"), ("40", "Forty")):
            m(f"refLate{t}{kt}", pct(r["late"][kk]["late_accept"]))
        m(f"refLateBpa{t}Forty", num(r["late"]["40"]["blocks_per_accept"] or 0, 0))
        m(f"refStress{t}Ten", pct(r["stress"]["alt-0.10"]))
        m(f"refStress{t}Eight", pct(r["stress"]["alt-0.08"]))
        m(f"refWorst{t}Fifteen", pct(r["worst"]["15"]))
    # 2. shortcut
    sc = d["shortcut"]
    m("shortSlots", word_int(sc["candidate_slots"]))
    m("shortDistinct", word_int(sc["distinct"]))
    m("shortBrokenDistinct", word_int(sc["broken_distinct"]))
    for name in RULES:
        m(f"shortFar{KEY[name]}", pct(sc["table"][name], 1 if sc["table"][name] >= 0.01 else 2))
        m(f"shortFarNat{KEY[name]}", pct(sc["natural"][name], 1 if sc["natural"][name] >= 0.01 else 2))
    # 3. truth vs pool
    for arm, t in (("natural", "Nat"), ("realflaky", "Rf")):
        rs = d["rates"][arm]
        m(f"ratesN{t}", word_int(len(rs)))
        m(f"ratesSplit{t}", word_int(sum((r["pool"] > r["eps"]) != (r["truth"] > r["eps"]) for r in rs)))
    # 4. units and bug-level intervals
    u = d["units"]
    for k, t in (("repositories", "Repos"), ("bugs", "Bugs"), ("loops", "Loops"), ("episodes", "Episodes"),
                 ("distinct_candidates", "Cands"), ("candidate_positions", "Positions"), ("blocks", "Blocks")):
        m(f"unit{t}", word_int(u[k]))
    m("unitBern", u["episodes_by_flakiness"]["injected-bernoulli"])
    m("unitMarkov", u["episodes_by_flakiness"]["injected-markov"])
    for name in RULES:
        nf = d["natural_far"][name]
        m(f"natBugLo{KEY[name]}", pct(nf["ci_bug"][0]))
        m(f"natBugHi{KEY[name]}", pct(nf["ci_bug"][1]))
    # 5. cost
    c = d["cost"]
    m("calibBlocks", c["p0_blocks_per_task"])
    m("calibPerLoop", num(c["p0_blocks_per_task"] / c["loops_per_task"], 0))
    for name in RULES:
        r = c["rules"][name]
        m(f"costBlocks{KEY[name]}", num(r["blocks_per_episode"]))
        m(f"costMin{KEY[name]}", num(r["seconds_per_episode"] / 60))
    for name in RULES[:3]:
        r = c["rules"][f"{name} (full suite)"]
        m(f"costFullRuns{KEY[name]}", num(r["runs_per_episode"]))
        m(f"costFullFar{KEY[name]}", pct(r["far"]))
        m(f"costFullFrr{KEY[name]}", pct(r["frr"]))
    for r in d["operating"]:
        if r["p0"] == 0.0 and round(r["p"], 2) == 0.03:
            m("opLowKellyAcceptThree", pct(r["kelly"]["accept"]))
    # 6. operating curve at high p0
    for r in d["operating"]:
        if r["p0"] == 0.2:
            key = {0.0: "Zero", 0.01: "One", 0.02: "Two", 0.03: "Three", 0.04: "Four", 0.05: "Five"}[round(r["p"] - r["p0"], 2)]
            m(f"opHighKellyAccept{key}", pct(r["kelly"]["accept"]))
            m(f"opHighMixAccept{key}", pct(r["mixture"]["accept"]))
            if r["kelly"]["blocks_per_accept"]:
                m(f"opHighKellyBlocks{key}", num(r["kelly"]["blocks_per_accept"], 0))

    # Table 2 (EASE): outcome and coverage panel, natural loops first, then all episodes; cost panel
    res = load("results/results.json")
    per = res["per_episode"]
    nat = [e for e in per if e["flakiness"] == "natural"]
    t = res["table"]
    rows = ""
    for name in RULES:
        if name == "SeqAccept":
            rows += "\\midrule\n"
        nf = d["natural_far"][name]
        hit = {}
        for e in nat:
            if "broken" in e["labels"]:
                b = e["file"].split("-")[0]
                hit[b] = hit.get(b, False) or e["rules"][name]["far"] > 0
        kb, nb = sum(hit.values()), len(hit)
        with_c = [e["rules"][name]["frr"] for e in nat if "correct" in e["labels"]]
        marg = sum(e["rules"][name]["marginal_accept"] for e in nat) / len(nat)
        ci = "" if nf["far"] == 0 else f" {{\\scriptsize[{pct(nf['ci_bug'][0])}, {pct(nf['ci_bug'][1])}]}}"
        sa = t[name]
        ci_all = "" if sa["far"] == 0 else f" {{\\scriptsize[{pct(sa['far_ci'][0])}, {pct(sa['far_ci'][1])}]}}"
        c = d["cost"]["rules"][name]
        rows += (f"{SHOW[name]} & {pct(nf['far'])}{ci} & {kb}/{nb} & {pct(replay.cp_upper(kb, nb))} & "
                 f"{pct(sum(with_c) / len(with_c))} & {pct(marg)} & {pct(sa['far'])}{ci_all} & {sa['calibration']:.2f} & "
                 f"{num(sa['blocks_per_accept'])} & {num(c['blocks_per_episode'])} & {num(sa['budget_x_b1'])}\\\\\n")
        m(f"natFrr{KEY[name]}", pct(sum(with_c) / len(with_c)))
        m(f"natBugHits{KEY[name]}", f"{kb}")
    m("natBugsWithBroken", nb)
    write("tab_outcome.tex",
          "\\begin{tabular}{@{}lrrrrrrr|rrr@{}}\n\\toprule\n"
          "& \\multicolumn{5}{c}{\\textbf{Natural loops (" + str(len(nat)) + ")}} & \\multicolumn{2}{c|}{\\textbf{All " + str(len(per)) + " episodes}} & \\multicolumn{3}{c}{\\textbf{Cost}}\\\\\n"
          "\\cmidrule(lr){2-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-11}\n"
          "\\textbf{Rule} & \\textbf{FAR \\%} {\\scriptsize[bug CI]} & \\textbf{Bugs hit} & \\textbf{UB \\%} & \\textbf{FRR \\%} & \\textbf{Marg.\\ \\%} & "
          "\\textbf{FAR \\%} {\\scriptsize[repo CI]} & \\textbf{Inj./$\\alpha$} & \\textbf{Blk/acc.} & \\textbf{Blk/loop} & \\textbf{$\\times$B1}\\\\\n\\midrule\n"
          + rows + "\\bottomrule\n\\end{tabular}\n")
    # refund ablation table
    proof = {"none": "yes", "max": "yes", "capped": "yes", "uncapped": "no"}
    show = {"none": "None (B6)", "max": "Maximal (\\sa{})", "capped": "Discounted", "uncapped": "Uncapped"}
    body = ""
    for k in ("none", "max", "capped", "uncapped"):
        r = d["refund"][k]
        body += (f"{show[k]} & {proof[k]} & {num(r['recorded']['blocks_per_accept'])} & {pct(r['late']['10']['late_accept'])} & "
                 f"{pct(r['late']['40']['late_accept'])} & {num(r['late']['40']['blocks_per_accept'], 0)} & "
                 f"{pct(r['stress']['alt-0.10'])} / {pct(r['stress']['alt-0.08'])} & {pct(r['worst']['15'])}\\\\\n")
    write("tab_refund.tex",
          "\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule\n"
          "& & \\textbf{Rec.} & \\multicolumn{3}{c}{\\textbf{Late streams}} & \\multicolumn{2}{c}{\\textbf{FAR \\%}}\\\\\n"
          "\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}\n"
          "\\textbf{Refund} & \\textbf{Valid} & \\textbf{Blk/acc.} & $k{=}10$ \\% & $k{=}40$ \\% & \\textbf{Blk/acc.} & \\textbf{Stress} & \\textbf{Worst}\\\\\n\\midrule\n"
          + body + "\\bottomrule\n\\end{tabular}\n\n\\smallskip\n" + refund_caps_panel())
    figure_ease(d)


def refund_caps_panel():
    """Second panel of the refund table: shared cap against caps scaled with the slice (slice_caps.py)."""
    sc = load("results/slice_caps.json")
    body = ""
    for n, show in (("B6", "None (B6)"), ("Harmonic", "Harmonic, none"), ("SeqAccept", "Maximal (\\sa{})")):
        cells = [f'{pct(sc[c]["late"]["40"][n])}' for c in ("shared", "slice")]
        cells += [num(sc[c]["neareps"][n]["blocks_per_loop"], 0) for c in ("shared", "slice")]
        cells += [num(sc[c]["natural"][n]["blocks_per_accept"], 0) for c in ("shared", "slice")]
        body += f"{show} & " + " & ".join(cells) + "\\\\\n"
    return ("\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n"
            "& \\multicolumn{2}{c}{\\textbf{Late $k{=}40$ \\%}} & \\multicolumn{2}{c}{\\textbf{Near-$\\eps$ blk/loop}} & "
            "\\multicolumn{2}{c}{\\textbf{Natural blk/acc.}}\\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(l){6-7}\n"
            "\\textbf{Refund, schedule} & shared & scaled & shared & scaled & shared & scaled\\\\\n\\midrule\n"
            + body + "\\bottomrule\n\\end{tabular}\n")


def figure_ease(d):
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    # truth vs pool (both arms)
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.75), sharey=True)
    colors = {"correct": "#2a78d6", "marginal": "#8a8984", "broken": "#e34948"}
    for ax, (arm, title) in zip(axes, (("natural", "natural loops"), ("realflaky", "real flaky tests kept"))):
        for lab in ("correct", "marginal", "broken"):
            rs = [r for r in d["rates"][arm] if r["label"] == lab]
            ax.scatter([r["truth"] - r["eps"] for r in rs], [r["pool"] - r["eps"] for r in rs], s=4, alpha=0.55,
                       color=colors[lab], label=lab.capitalize(), linewidths=0)
        ax.axhline(0, color=MUTED, lw=0.6, ls=(0, (2, 2)))
        ax.axvline(0, color=MUTED, lw=0.6, ls=(0, (2, 2)))
        ax.set_xlim(-0.31, 0.35)
        ax.set_ylim(-0.31, 0.35)
        ax.set_title(title, fontsize=7, color=INK)
        ax.set_xlabel(r"truth rate $-\ \varepsilon$")
        # two binomial standard errors of a 200-block rate at p = eps, around the diagonal
        xs = [i / 100 for i in range(-31, 36)]
        ax.plot(xs, xs, color="#c9c8c4", lw=0.5)
    axes[0].set_ylabel(r"pool rate $-\ \varepsilon$")
    axes[1].legend(frameon=False, fontsize=5.5, loc="lower right", handletextpad=0.2, markerscale=2)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGS / "truthpool.pdf", bbox_inches="tight")
    plt.close(fig)
    # operating curve near eps at p0 = 0 and 0.2
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.8))
    for p0, mk in ((0.0, "o"), (0.2, "s")):
        rows = [r for r in d["operating"] if r["p0"] == p0]
        gaps = [r["eps"] - r["p"] for r in rows]
        for b, ls in (("kelly", "-"), ("mixture", "--")):
            col = COLOR["SeqAccept"] if b == "kelly" else "#6250d6"
            mfc = col if p0 == 0 else "white"  # hollow markers for p0 = 0.2: separable in greyscale
            axes[0].plot(gaps, [100 * r[b]["accept"] for r in rows], ls, marker=mk, ms=3, lw=1, mfc=mfc, mew=0.7,
                         color=col, label=f"{b}, $p_0={p0:g}$")
            axes[1].plot([g for g, r in zip(gaps, rows) if r[b]["blocks_per_accept"] and g > 0],
                         [r[b]["blocks_per_accept"] for g, r in zip(gaps, rows) if r[b]["blocks_per_accept"] and g > 0],
                         ls, marker=mk, ms=3, lw=1, mfc=mfc, mew=0.7, color=col)
        axes[1].plot([r["eps"] - r["p"] for r in rows if r["kl_floor"]], [r["kl_floor"] for r in rows if r["kl_floor"]],
                     ":", marker=mk, ms=2, lw=0.8, color=MUTED)
    axes[0].set_xlabel(r"$\varepsilon - p$")
    axes[0].set_ylabel("accepted (%)")
    axes[1].set_xlabel(r"$\varepsilon - p$")
    axes[1].set_ylabel("blocks per accept")
    axes[1].set_yscale("log")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=6, bbox_to_anchor=(0.5, 1.08),
               handlelength=1.6, columnspacing=0.8)
    axes[1].text(0.97, 0.92, "dotted: KL floor", transform=axes[1].transAxes, fontsize=6, color=MUTED, ha="right")
    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(top=0.84)
    fig.savefig(FIGS / "operating.pdf", bbox_inches="tight")
    plt.close(fig)
    # risk-coverage-cost frontier on natural loops (blocks or full-suite runs per episode)
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    full_s = load("results/caps_adversary.json")["breakeven"]["full_s"]  # mean recorded seconds of one full-suite run
    for name in RULES:
        r = d["cost"]["rules"][name]
        x = r["seconds_per_episode"] / 60
        ax.scatter(x, 100 * r["far"], s=18, color=COLOR[name], marker=MARK[name], edgecolors="none", zorder=3)
        if name in RULES[:3]:
            ax.annotate(SHOW[name].split(" ")[0], (x, 100 * r["far"]), textcoords="offset points", xytext=(3, 3),
                        fontsize=5.5, color=INK)
    b4 = d["cost"]["rules"]["B4 peeking Clopper-Pearson"]
    ax.annotate("B4", (b4["seconds_per_episode"] / 60, 0), textcoords="offset points", xytext=(-14, 9), fontsize=5.5,
                color=INK, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.5})
    sa = d["cost"]["rules"]["SeqAccept"]
    ax.annotate("B5, B6, SeqAccept", (sa["seconds_per_episode"] / 60, 0), textcoords="offset points", xytext=(-6, 17),
                fontsize=5.5, color=INK, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.5})
    for name in RULES[:3]:
        r = d["cost"]["rules"][f"{name} (full suite)"]
        x = r["runs_per_episode"] * full_s / 60
        ax.scatter(x, 100 * r["far"], s=18, facecolors="none", edgecolors=COLOR[name], marker=MARK[name], zorder=3)
        ax.annotate(SHOW[name].split(" ")[0] + " full", (x, 100 * r["far"]), textcoords="offset points",
                    xytext=(3, -7), fontsize=5.5, color=MUTED)
    ax.axhline(5, color=MUTED, lw=0.6, ls=(0, (2, 2)))
    ax.set_xscale("log")
    ax.set_xticks([0.1, 1, 10], ["0.1", "1", "10"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("minutes of recorded run time per loop")
    ax.set_ylabel("FAR, natural loops (%)")
    ax.grid(axis="y", color="#e6e5e1", lw=0.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGS / "frontier.pdf", bbox_inches="tight")
    plt.close(fig)
    # persistence with bug-level bands
    pers = d["persistence"]
    order = ["llama3.1:8b", "gpt-4o", "gpt-4.1-mini", "grok-4.7"]
    shown = ["B1 stop at first pass", "B2 rerun-3 all green", "B3 majority of 5", "SeqAccept"]
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 1.6), sharey=True)
    for ax, model in zip(axes, order):
        for name in shown:
            rows = pers[model][name]
            ks = [r["k"] for r in rows]
            ax.plot(ks, [100 * r["far"] for r in rows], LINE[name], color=COLOR[name], marker=MARK[name], ms=3, lw=1.1,
                    label=SHOW[name].replace("\\sa{}", "SeqAccept").replace("\\ ", " "))
            if name == "B1 stop at first pass":  # bug-level 95% band for B1; the others are in the supplement
                ax.fill_between(ks, [100 * r["ci"][0] for r in rows], [100 * r["ci"][1] for r in rows],
                                color=COLOR[name], alpha=0.15, lw=0)
        ax.axhline(5, color=MUTED, lw=0.6, ls=(0, (2, 2)))
        ax.set_title(MODEL_SHOW[model], fontsize=7, color=INK)
        ax.set_xticks(replay.PREFIXES)
        ax.set_ylim(-2, 62)
        ax.grid(axis="y", color="#e6e5e1", lw=0.5)
        ax.set_xlabel("candidates tried")
    axes[0].set_ylabel("FAR (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=6.2, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(top=0.78)
    fig.savefig(FIGS / "persistence.pdf", bbox_inches="tight")
    plt.close(fig)
    # worst case heatmap: FAR by stream length and p, per eps-based rule, with Monte Carlo intervals in the table
    w = load("results/worst_case.json")
    shown = ["B4 peeking Clopper-Pearson", "B5 naive confidence sequence", "B6 SeqAccept, no refund", "SeqAccept"]
    ps, ns = (0.051, 0.06, 0.08, 0.10), (1, 5, 15, 50)
    w10 = load("results/caps_adversary.json")["worst010"]  # p = 0.10 row (caps_adversary.py)

    def cell(p, n, name):  # (FAR, runs) of one heatmap cell
        if p == 0.10:
            return w10[str(n)][name], (1000 if n < 50 else 400)
        r = next(r for r in w["rows"] if r["p"] == p and r["n"] == n)
        return r["far"][name], r["reps"]
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 1.8))
    for ax, name in zip(axes, shown):
        grid = [[cell(p, n, name)[0] for n in ns] for p in ps]
        ax.imshow(grid, cmap="Reds", vmin=0, vmax=1, aspect="auto")
        for i, p in enumerate(ps):
            for j, n in enumerate(ns):
                v = 100 * grid[i][j]
                txt = f"{v:.0f}" if v >= 10 else f"{v:.1f}"
                far, reps = cell(p, n, name)  # every cell: the 95% Wilson upper bound below the rate
                up = wilson(round(far * reps), reps)[1]
                txt += f"\n\u2264{100 * up:.0f}" if up >= 0.1 else f"\n\u2264{100 * up:.1f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6, linespacing=0.9,
                        color="white" if v > 50 else (INK if v <= 5 else "#7a1f1f"), fontweight="bold" if v > 5 else "normal")
                if far > 0.05:  # outline cells whose rate exceeds alpha
                    ax.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, ec=INK, lw=0.8))
        ax.set_xticks(range(len(ns)), [str(n) for n in ns])
        ax.set_yticks(range(len(ps)), [f"{p:.3f}" for p in ps])
        ax.set_title(SHOW[name].replace("\\sa{}", "SeqAccept").replace("\\ ", " "), fontsize=6.5, color=INK)
        ax.set_xlabel("broken candidates $n$")
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel("failure rate $p$")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGS / "worstheat.pdf", bbox_inches="tight")
    plt.close(fig)
    worst_hi = max(wilson(round(r["far"]["SeqAccept"] * r["reps"]), r["reps"])[1] for r in w["rows"])
    m("worstSaUpperMC", pct(worst_hi))


def ease3():
    """Round-2 review follow-ups: clustered intervals, near-eps dependence, ordered replay, eps from L, calibrated cost."""
    d = load("results/intervals.json")
    K = {"B1 stop at first pass": "Bone", "B2 rerun-3 all green": "Btwo", "B3 majority of 5": "Bthree",
         "B4 peeking Clopper-Pearson": "Bfour", "B5 naive confidence sequence": "Bfive", "SeqAccept": "Sa"}
    fixed = ("B1 stop at first pass", "B2 rerun-3 all green", "B3 majority of 5")
    for n in fixed:
        lo, hi = d["fullsuite"][n]["far_ci_bug"]
        m(f"fullCiLo{K[n]}", pct(lo))
        m(f"fullCiHi{K[n]}", pct(hi))
    for n in ("B4 peeking Clopper-Pearson", "B5 naive confidence sequence", "SeqAccept"):
        lo, hi = d["stress094"][n]["far_ci_bug"]
        m(f"stressCiLo{K[n]}", pct(lo, 2))
        m(f"stressCiHi{K[n]}", pct(hi, 2))
        o = d["ordered_realflaky"][n]
        m(f"ordFar{K[n]}", pct(o["ordered"]["far"], 2))
        m(f"ordFrr{K[n]}", pct(o["ordered"]["frr"]))
        m(f"iidFar{K[n]}", pct(o["iid"]["far"], 2))
        lc = d["realflaky_lcb"][n]
        m(f"lcbFar{K[n]}", pct(lc["far"], 2))
        m(f"lcbFrr{K[n]}", pct(lc["frr"]))
        m(f"lcbFrrHigh{K[n]}", pct(lc["frr_high"]))
    m("lcbGap", f"{d['realflaky_lcb']['max_gap']:.3f}")
    from intervals import cp_lower  # largest gap between the estimate and the bound over every outcome of the p0 blocks
    n0 = 400
    m("lcbSupGap", f"{max(k / n0 - cp_lower(k, n0) for k in range(n0 + 1)):.3f}")
    for arm, key in (("natural", "Nat"), ("realflaky", "Rf")):
        x = d["near_eps_dependence"][arm]
        m(f"dep{key}Tested", word_int(x["tested"]))
        m(f"dep{key}Rejected", x["rejected"])
        m(f"dep{key}RejPct", pct(x["rejected"] / x["tested"]))
        m(f"dep{key}RejUpper", pct(x["rejected_upper"]))
        m(f"dep{key}LagLo", f"${x['lag1_ci_task'][0]:.3f}$")
        m(f"dep{key}LagHi", f"${x['lag1_ci_task'][1]:.3f}$")
        m(f"dep{key}LagOver", x["lag1_abs_over_02"])
    c = d["cost_calibrated"]
    for n, k in K.items():
        m(f"calBpa{k}", num(c[n]["blocks_per_accept"], 0))
        m(f"calMin{k}", num(c[n]["minutes_per_accept"]))
    for n in fixed:
        m(f"fullMin{K[n]}", f"{c[n + ' (full suite)']['minutes_per_accept']:.2f}")
        m(f"fullRpa{K[n]}", num(c[n + " (full suite)"]["runs_per_accept"]))
    m("calMinBoneExact", f"{c['B1 stop at first pass']['minutes_per_accept']:.2f}")


def ease4():
    """Round-3 review follow-ups: results/caps_adversary.json (caps_adversary.py)."""
    d = load("results/caps_adversary.json")
    W = {1: "One", 5: "Five", 10: "Ten", 15: "Fifteen", 20: "Twenty", 40: "Forty", 50: "Fifty", 100: "Hundred"}
    nat, late = d["caps"]["natural"], d["caps"]["late"]
    m("capBpaSa", num(nat["SeqAccept"]["blocks_per_accept"]))
    for k in (10, 20, 40):
        m(f"capLateSa{W[k]}", pct(late[str(k)]["SeqAccept"]["accept"]))
    for n in (15, 40, 100):
        for rule, tag in ((f"Bonferroni-{n} spending", "Spend"), (f"Fixed-n Bonferroni-{n}", "Fixn")):
            m(f"capBpa{tag}{W[n]}", num(nat[rule]["blocks_per_accept"]))
            m(f"capFrr{tag}{W[n]}", pct(nat[rule]["frr"]))
            for k in (10, 20, 40):
                m(f"capLate{tag}{W[n]}{W[k]}", pct(late[str(k)][rule]["accept"]))
    rows = ""
    for n in (15, 40, 100):
        for rule, label in ((f"Bonferroni-{n} spending", f"Bonferroni-{n} spending"), (f"Fixed-n Bonferroni-{n}", f"Fixed-$n$ Bonferroni-{n}")):
            rows += (f"{label} & {num(nat[rule]['blocks_per_accept'])} & {pct(nat[rule]['frr'])} & "
                     + " & ".join(pct(late[str(k)][rule]["accept"]) for k in (10, 20, 40)) + "\\\\\n")
    ck = load("results/checks.json")
    for rule, label in (("Bonferroni-1000 spending", "Bonferroni-1000 spending"), ("Fixed-n Bonferroni-1000", "Fixed-$n$ Bonferroni-1000")):
        rows += (f"{label} & {num(ck['caps'][rule]['blocks_per_accept'])} & {pct(ck['caps'][rule]['frr'])} & "
                 f"{pct(ck['caps_late']['10'][rule])} & -- & {pct(ck['caps_late']['40'][rule])}\\\\\n")
    rows += ("\\midrule\n\\sa{} & " + num(nat["SeqAccept"]["blocks_per_accept"]) + " & " + pct(nat["SeqAccept"]["frr"]) + " & "
             + " & ".join(pct(late[str(k)]["SeqAccept"]["accept"]) for k in (10, 20, 40)) + "\\\\\n")
    write("tab_caps.tex", "\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n"
          "& \\multicolumn{2}{c}{\\textbf{Natural loops}} & \\multicolumn{3}{c}{\\textbf{Late: accepted \\% after $k$}}\\\\\n"
          "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\n"
          "\\textbf{Rule} & \\textbf{Blk/acc.} & \\textbf{FRR \\%} & $k=10$ & $k=20$ & $k=40$\\\\\n\\midrule\n" + rows
          + "\\bottomrule\n\\end{tabular}\n")
    adv = d["adversary"]
    for key, v in adv.items():
        policy, n, refund = key.split("-")
        m(f"adv{policy.capitalize()}{W[int(n)]}{refund.capitalize()}", pct(v["far"], 1))
    worst = max(v["far"] for k, v in adv.items() if k.endswith("-max") or k.endswith("-capped"))
    m("advProvedMax", pct(worst, 1))
    big = [v for k, v in adv.items() if (k.endswith("-max") or k.endswith("-capped")) and v["reps"] == 1000]
    top = max(big, key=lambda v: v["far"])  # widest 95% Wilson upper bound among the 1,000-run cells
    m("advProvedUpper", pct(wilson(round(top["far"] * top["reps"]), top["reps"])[1], 1))
    m("advFiftyReps", min(v["reps"] for v in adv.values()))
    dep = d["dependence"]
    m("depPhiQninety", f"{dep['q90']:.2f}")
    m("depPhiCands", word_int(dep["candidates"]))
    for tag, key in (("zero", "Zero"), ("q90", "Qninety")):
        for n in (1, 15):
            m(f"depFar{key}{W[n]}", pct(dep["far"][tag][str(n)], 1))
    be = d["breakeven"]
    m("beBlockSec", f"{be['block_s']:.1f}")
    m("beFullSec", f"{be['full_s']:.1f}")
    for k, t in (("B1", "Bone"), ("B2", "Btwo")):
        m(f"be{t}Ratio", f"{be[k]['ratio']:.0f}")
        m(f"be{t}Tests", word_int(int(round(be[k]["suite_tests"], -2))))
    am = d["amortised"]
    for name, t in (("SeqAccept", "Sa"), ("B4 peeking Clopper-Pearson", "Bfour")):
        m(f"am{t}None", num(am[name]["none"], 0))
        m(f"am{t}Task", num(am[name]["per_task"], 0))
        m(f"am{t}Loop", num(am[name]["per_loop"], 0))
    u = d["realflaky_unamb"]
    for name, t in (("SeqAccept", "Sa"), ("B4 peeking Clopper-Pearson", "Bfour")):
        m(f"unambFar{t}", pct(u[name]["far"], 2))
        m(f"plainFar{t}", pct(u[name]["far_plain"], 2))
    rows = ""
    for task, r in d["realflaky_tasks"].items():
        rows += (f"{task.replace('@realflaky', '').replace('_', '\\_')} & {r['p0']:.3f} & {r['loops']} & {r['near_broken']} & "
                 f"{pct(r['far_b4'], 2)} & {pct(r['far_b4_unamb'], 2)} & {pct(r['far_sa'], 2)} & {pct(r['far_sa_unamb'], 2)} & {pct(r['frr_sa'])}\\\\\n")
    write("tab_rf_tasks.tex", "\\begin{tabular}{@{}lrrrrrrrr@{}}\n\\toprule\n"
          "& & & & \\multicolumn{2}{c}{\\textbf{B4 FAR \\%}} & \\multicolumn{2}{c}{\\textbf{\\sa{} FAR \\%}} & \\\\\n"
          "\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n"
          "\\textbf{Task} & $p_0$ & \\textbf{Loops} & \\textbf{Near} & all & unamb. & all & unamb. & \\textbf{FRR \\%}\\\\\n\\midrule\n"
          + rows + "\\bottomrule\n\\end{tabular}\n")
    pr = d["prefilter_rates"]
    rates = sorted(b["rate"] for b in pr["bugs"].values())
    m("preBugs", len(rates))
    m("preMinRate", pct(rates[0]))
    m("preMedianRate", pct(__import__("statistics").median(rates)))
    m("preBelowSix", sum(1 for r in rates if r <= 0.10))
    bm = pr["broken_minus_eps"]
    m("preBrokenN", word_int(bm["n"]))
    m("preBrokenLeFive", bm["le_005"])
    m("preBrokenLeTwentyFive", word_int(bm["le_025"]))
    m("preBrokenMedian", f"{bm['median']:.2f}")
    cf = d["conditional_far"]
    m("condLoops", cf["loops"])
    for name in RULES:
        m(f"condFar{KEY[name]}", pct(cf[name]))


def sens():
    """Sensitivity and cost: results/sensitivity.json (sensitivity.py)."""
    d = load("results/sensitivity.json")
    W = {1: "One", 2: "Two", 3: "Three", 5: "Five", 10: "Ten"}
    c = d["cost"]
    m("costTaskSa", num(c["SeqAccept"]["bpa_task"], 0))
    m("costLoopSa", num(c["SeqAccept"]["bpa_loop"], 0))
    m("costNoneSa", num(c["SeqAccept"]["blocks_per_accept"]))
    m("costFixnTask", num(c["Fixed-n Bonferroni-15 (1n0)"]["bpa_task"], 0))
    m("costFixnMin", num(c["Fixed-n Bonferroni-15 (1n0)"]["minutes_per_accept"]))
    m("costBonfHundredTask", num(c["Bonferroni-100 spending"]["bpa_task"], 0))
    m("costSaOverBonfHundred", f"{c['SeqAccept']['bpa_task'] / c['Bonferroni-100 spending']['bpa_task']:.2f}")
    m("costMinTaskSa", num(c["SeqAccept"]["minutes_per_accept"]))
    bf = d["block_fraction"]
    m("bfCandidates", word_int(bf["candidates"]))
    m("bfUnexplained", bf["candidates"] - bf["explained"])
    m("bfGap", f"{100 * bf['model_gap']['mean']:.1f}")
    for f, tag in (("0.5", "Half"), ("1", "Rec"), ("2", "Double")):
        for name in RULES[:3]:
            m(f"bf{tag}{KEY[name]}", pct(bf["rules"][f][name]["far"]))
        m(f"bf{tag}MaxValid", pct(max(bf["rules"][f][n]["far"] for n in ("B4 peeking Clopper-Pearson", "SeqAccept"))))
    rg = d["regime"]
    for r in (1, 2, 3):
        vals = [rg[f"{N}-{r}"]["cost_ratio_B1"] for N in (1000, 3000, 10000)]
        m(f"regRatio{W[r]}", f"{min(vals):.2f}" if abs(max(vals) - min(vals)) < 0.005 else f"{min(vals):.2f}--{max(vals):.2f}")
        vals2 = [rg[f"{N}-{r}"]["cost_ratio_B2"] for N in (1000, 3000, 10000)]
        m(f"regRatioBtwo{W[r]}", f"{min(vals2):.2f}" if abs(max(vals2) - min(vals2)) < 0.005 else f"{min(vals2):.2f}--{max(vals2):.2f}")
    rows = ""
    for N in (1000, 3000, 10000):
        rows += f"{word_int(N)}" + "".join(f" & {rg[f'{N}-{r}']['block']} & {rg[f'{N}-{r}']['cost_ratio_B1']:.2f}" for r in (1, 2, 3, 5, 10)) + "\\\\\n"
    write("tab_regime.tex", "\\begin{tabular}{@{}r" + "rr" * 5 + "@{}}\n\\toprule\n& "
          + " & ".join(f"\\multicolumn{{2}}{{c}}{{$\\ell={r}$}}" for r in (1, 2, 3, 5, 10)) + "\\\\\n"
          + "".join(f"\\cmidrule(lr){{{2 * i + 2}-{2 * i + 3}}}" for i in range(5)) + "\n"
          + "\\textbf{Suite} $N$" + " & block & ratio" * 5 + "\\\\\n\\midrule\n" + rows + "\\bottomrule\n\\end{tabular}\n")
    ps = d["pool_scored"]
    m("poolFarRfSa", pct(ps["realflaky"]["SeqAccept"], 2))
    m("poolFarRfBfour", pct(ps["realflaky"]["B4 peeking Clopper-Pearson"], 2))
    m("poolFarRfBfive", pct(ps["realflaky"]["B5 naive confidence sequence"], 2))
    assert all(abs(ps["natural"][n] - load("results/ablations.json")["natural_far"][n]["far"]) < 1e-9 for n in RULES)
    nc = d["natural_repo_ci"]
    m("natRepos", nc["SeqAccept"]["repos"])
    for name in RULES[:3]:
        m(f"natRepoLo{KEY[name]}", pct(nc[name]["ci"][0]))
        m(f"natRepoHi{KEY[name]}", pct(nc[name]["ci"][1]))
    m("natRepoMinLo", pct(min(nc[n]["ci"][0] for n in RULES[:3])))
    rm = d["realflaky_matched"]
    show = {"SeqAccept": "\\sa{}", "Harmonic spending": "Harmonic spending", "Discard spending": "Discarding (unproved)",
            "Bonferroni-15 spending": "Bonferroni-15 spending", "Bonferroni-100 spending": "Bonferroni-100 spending",
            "Fixed-n Bonferroni-15 (1n0)": "Fixed-$n$ Bonf.-15 ($n_0$)", "Fixed-n Bonferroni-15 (2n0)": "Fixed-$n$ Bonf.-15 ($2n_0$)",
            "Fixed-n Bonferroni-15 (4n0)": "Fixed-$n$ Bonf.-15 ($4n_0$)", "B4 peeking Clopper-Pearson": "B4 (invalid)"}
    rr = next(k for k in rm if k.startswith("Rerun-"))
    show[rr] = rr.replace(" (all green)", " (no guarantee)")
    rows = ""
    for k, lab in show.items():
        r = rm[k]
        rows += f"{lab} & {pct(r['far'], 2)} & {pct(r['frr'])} & {pct(r['frr_high'])} & {num(r['blocks_per_accept'], 0)}\\\\\n"
    write("tab_rf_matched.tex", "\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n\\textbf{Rule} & \\textbf{FAR \\%} & \\textbf{FRR \\%} & "
          "\\textbf{FRR, $p_0\\ge0.2$} & \\textbf{Blk/acc.}\\\\\n\\midrule\n" + rows + "\\bottomrule\n\\end{tabular}\n")
    valid = [k for k in rm if k in ("Harmonic spending", "Bonferroni-15 spending", "Bonferroni-100 spending")
             or k.startswith("Fixed-n")]
    m("rfMatchedMinFrr", pct(min(rm[k]["frr"] for k in valid)))
    m("rfMatchedMaxFrr", pct(max(rm[k]["frr"] for k in valid)))
    m("rfMatchedHarmFrr", pct(rm["Harmonic spending"]["frr"]))
    m("rfMatchedFixnFrr", pct(rm["Fixed-n Bonferroni-15 (1n0)"]["frr"]))
    m("rfMatchedRerunFrr", pct(rm[rr]["frr"]))
    m("rfMatchedSaBpa", num(rm["SeqAccept"]["blocks_per_accept"], 0))
    assert rm["SeqAccept"]["frr"] <= min(rm[k]["frr"] for k in valid)
    mc = d["margin_curve"]
    rows = ""
    for p0 in ("0.0", "0.1", "0.2", "0.3"):
        rows += f"{float(p0):.1f}" + "".join(f" & {pct(mc[f'{p0}-{mm}']['accept'])}" for mm in ("0.03", "0.05", "0.1", "0.15")) + "\\\\\n"
    write("tab_margin.tex", "\\begin{tabular}{@{}rrrrr@{}}\n\\toprule\n$p_0$ & $m=0.03$ & 0.05 & 0.10 & 0.15\\\\\n\\midrule\n"
          + rows + "\\bottomrule\n\\end{tabular}\n")
    m("mcAcceptTwoFive", pct(mc["0.2-0.05"]["accept"]))
    m("mcAcceptThreeFive", pct(mc["0.3-0.05"]["accept"]))
    m("mcAcceptMinTen", pct(min(mc[f"{p0}-0.1"]["accept"] for p0 in ("0.0", "0.1", "0.2", "0.3"))))
    rmg = d["realflaky_margin"]
    m("rfMarginTenFrrHigh", pct(rmg["0.1"]["frr_high"]))
    m("rfMarginTenFar", pct(rmg["0.1"]["far"], 2))
    m("rfMarginFifteenFrr", pct(rmg["0.15"]["frr"]))


def checks():
    """Validity, interval and selection checks: results/checks.json (checks.py)."""
    d = load("results/checks.json")
    inv = d["invariant"]
    sa = [v for k, v in inv.items() if k.endswith("-max")]
    un = [v for k, v in inv.items() if k.endswith("-uncapped")]
    m("invSaMin", f"{min(v['mean_stop'] for v in sa):.3f}")
    m("invSaMax", f"{max(v['mean_stop'] for v in sa):.3f}")
    m("invUncMin", f"{min(v['mean_stop'] for v in un):.3f}")
    m("invUncMax", f"{max(v['mean_stop'] for v in un):.3f}")
    m("invUncLowerMin", f"{min(v['mean_stop'] - 1.96 * v['se'] for v in un):.3f}")
    rp = d["runs_power"]
    m("runsPowerTwo", pct(rp["0.2"], 0))
    m("runsSize", pct(rp["0.0"]))
    m("runsCands", rp["candidates"])
    fd = d["frr_diff"]
    assert all(v["ci_task"][1] < 0 for v in fd.values())  # SeqAccept loses fewer correct patches than each valid rule
    m("frrDiffHarm", pct(-fd["Harmonic spending"]["diff"]))
    m("frrDiffHarmLo", pct(-fd["Harmonic spending"]["ci_task"][1]))
    m("frrDiffHarmHi", pct(-fd["Harmonic spending"]["ci_task"][0]))
    m("frrDiffBsix", pct(-fd["B6 SeqAccept, no refund"]["diff"]))
    m("frrDiffBsixLo", pct(-fd["B6 SeqAccept, no refund"]["ci_task"][1]))
    m("frrDiffBsixHi", pct(-fd["B6 SeqAccept, no refund"]["ci_task"][0]))
    cr = d["cost_ratio"]
    for k, t in (("Bonferroni-100 spending", "BonfHundred"), ("Fixed-n Bonferroni-15 (1n0)", "Fixn"), ("B4 peeking Clopper-Pearson", "Bfour")):
        m(f"cr{t}", f"{cr[k]['ratio']:.2f}")
        m(f"cr{t}Lo", f"{cr[k]['ci_task'][0]:.2f}")
        m(f"cr{t}Hi", f"{cr[k]['ci_task'][1]:.2f}")
    c = d["caps"]
    sa_t = c["SeqAccept"]["bpa_task"]
    cheaper = [n for n in (15, 40, 100, 300, 1000, 3000, 10000) if c[f"Fixed-n Bonferroni-{n}"]["bpa_task"] < sa_t]
    dearer = [n for n in (15, 40, 100, 300, 1000, 3000, 10000) if c[f"Fixed-n Bonferroni-{n}"]["bpa_task"] >= sa_t]
    assert max(cheaper) < min(dearer)
    m("capFixnCheapUpTo", word_int(max(cheaper)))
    m("capFixnDearFrom", word_int(min(dearer)))
    m("capFixnThousandTask", num(c["Fixed-n Bonferroni-1000"]["bpa_task"], 0))
    m("capFixnThreeThousandTask", num(c["Fixed-n Bonferroni-3000"]["bpa_task"], 0))
    m("capSpendThousandTask", num(c["Bonferroni-1000 spending"]["bpa_task"], 0))
    cl = d["caps_late"]
    m("capLateFixnThousandTen", pct(cl["10"]["Fixed-n Bonferroni-1000"]))
    m("capLateSpendThousandForty", pct(cl["40"]["Bonferroni-1000 spending"]))
    nr = d["no_replacement"]
    m("nrNatMaxFar", pct(max(v["far"] for v in nr["natural"].values()), 2))
    m("nrRfFarSa", pct(nr["realflaky"]["SeqAccept"]["far"], 2))
    m("nrRfFrrSa", pct(nr["realflaky"]["SeqAccept"]["frr"]))
    m("nrRfFarBfour", pct(nr["realflaky"]["B4 peeking Clopper-Pearson"]["far"], 2))
    fu = d["frr_unamb"]
    m("unambFrrSa", pct(fu["SeqAccept"]["frr"]))
    m("unambFrrLoops", fu["SeqAccept"]["loops"])
    ub = d["uncapped_B"]
    m("ubAcceptTwo", pct(ub["0.2-10000"]["accept"]))
    m("ubBlocksTwo", num(ub["0.2-10000"]["blocks"], 0))
    m("ubBlocksTwoCap", num(ub["0.2-cap"]["blocks"], 0))
    m("ubBpaTwo", num(ub["0.2-10000"]["blocks"] / ub["0.2-10000"]["accept"], 0))
    m("ubBpaTwoCap", num(ub["0.2-cap"]["blocks"] / ub["0.2-cap"]["accept"], 0))
    m("ubAcceptMin", pct(min(ub[f"{p}-10000"]["accept"] for p in (0.1, 0.2, 0.3))))
    rb = d["realflaky_uncapped_B"]
    m("ubRfFrr", pct(rb["frr"], 2))
    m("ubRfFrrHigh", pct(rb["frr_high"], 2))
    m("ubRfFar", pct(rb["far"], 2))
    m("ubRfBpa", num(rb["blocks_per_accept"], 0))
    co = d["corollary"]
    m("corDelta", f"{co['delta']:.3f}")
    m("corFarNat", pct(co["natural"]["far"]))
    m("corBpaNat", num(co["natural"]["blocks_per_accept"]))
    m("corBpaTaskNat", num(co["natural"]["bpa_task"], 0))
    m("corRfFar", pct(co["realflaky"]["far"], 2))
    m("corRfFrr", pct(co["realflaky"]["frr"]))
    m("corRfFrrHigh", pct(co["realflaky"]["frr_high"]))
    m("corRfBpa", num(co["realflaky"]["blocks_per_accept"], 0))
    bl = d["realflaky_broken_loops"]
    m("rfBrokenLoops", bl["loops"])
    for name in ("B4 peeking Clopper-Pearson", "SeqAccept", "B6 SeqAccept, no refund", "B5 naive confidence sequence"):
        k = KEY[name]
        m(f"rfCondFar{k}", pct(bl[name]["far"], 2))
        m(f"worstTaskFar{k}", pct(bl[name]["worst_far"]))
        m(f"worstTaskUp{k}", pct(bl[name]["worst_upper"]))
    assert bl["SeqAccept"]["worst_task"] == bl["B4 peeking Clopper-Pearson"]["worst_task"] == "yamicache@realflaky"  # text names yamicache
    rbk = d["refund_breakeven"]
    m("rbRefund", rbk["refund"])
    m("rbWorstFrom", ["", "first", "second", "third", "fourth", "fifth", "sixth", "seventh"][rbk["worst_from"]])
    for j, t in ((5, "Five"), (15, "Fifteen"), (40, "Forty")):
        m(f"rbHarm{t}", rbk[f"harmonic_{j}"])
    for j, t in ((5, "Five"), (15, "Fifteen")):
        m(f"rbNone{t}", rbk[f"norefund_{j}"])
    m("capBpaSpendThousand", num(c["Bonferroni-1000 spending"]["blocks_per_accept"]))
    dt = d["dropped_tasks"]
    m("dropLoops", dt["loops_added"])
    m("dropFarBone", pct(dt["with"]["B1 stop at first pass"]))
    m("dropMaxValid", pct(max(dt["with"][n] for n in RULES[3:]), 2))
    dr = d["drift"]
    gaps = {k: abs(v["run1"] - v["run2"]) for k, v in dr.items()}
    worst = max(gaps, key=gaps.get)
    m("driftModel", worst)
    m("driftOne", pct(dr[worst]["run1"]))
    m("driftTwo", pct(dr[worst]["run2"]))
    # main-text real-flaky table
    rf = load("results/extra.json")["realflaky"]
    t, th = rf["table"], rf["p0_high"]["table"]
    rm = load("results/sensitivity.json")["realflaky_matched"]
    rows = ""
    for name, valid in (("B4 peeking Clopper-Pearson", "no"), ("B5 naive confidence sequence", "per cand."), ("B6 SeqAccept, no refund", "yes"), ("SeqAccept", "yes")):
        r = t[name]
        ci = f" {{\\scriptsize[{pct(r['far_ci'][0], 2)}, {pct(r['far_ci'][1], 2)}]}}"
        if name == "SeqAccept":
            rows += "\\midrule\n"
        rows += f"{SHOW[name]} & {valid} & {pct(r['far'], 2)}{ci} & {pct(r['frr'])} & {pct(th[name]['frr'])} & {num(r['blocks_per_accept'], 0)}\\\\\n"
    rows += "\\midrule\n"
    rr = next(k for k in rm if k.startswith("Rerun-"))
    for key, lab, valid in (("Harmonic spending", "Harmonic spending", "yes"), ("Bonferroni-15 spending", "Bonferroni-15 spending", "yes"),
                            ("Bonferroni-100 spending", "Bonferroni-100 spending", "yes"), ("Fixed-n Bonferroni-15 (1n0)", "Fixed-$n$ Bonf.-15 ($n_0$)", "yes"),
                            (rr, rr.replace(" (all green)", " (consec.)"), "no")):
        r = rm[key]
        rows += f"{lab} & {valid} & {pct(r['far'], 2)} & {pct(r['frr'])} & {pct(r['frr_high'])} & {num(r['blocks_per_accept'], 0)}\\\\\n"
    rows += "\\midrule\n"
    rows += f"\\sa{{}}, $T=10{{,}}000$ & yes & {pct(rb['far'], 2)} & {pct(rb['frr'], 2)} & {pct(rb['frr_high'], 2)} & {num(rb['blocks_per_accept'], 0)}\\\\\n"
    cr_ = co["realflaky"]
    rows += f"\\sa{{}}, $\\eps=L+\\delta$ & yes$^\\dagger$ & {pct(cr_['far'], 2)} & {pct(cr_['frr'])} & {pct(cr_['frr_high'])} & {num(cr_['blocks_per_accept'], 0)}\\\\\n"
    mg = load("results/sensitivity.json")["realflaky_margin"]["0.1"]
    rows += f"\\sa{{}}, margin 0.10 & yes$^*$ & {pct(mg['far'], 2)} & {pct(mg['frr'])} & {pct(mg['frr_high'])} & {macros['rfMarginBpaSa']}\\\\\n"
    write("tab_rf_main.tex", "\\begin{tabular}{@{}llrrrr@{}}\n\\toprule\n\\textbf{Rule} & \\textbf{Valid} & \\textbf{FAR \\%} {\\scriptsize[repo.\\ CI]} & "
          "\\textbf{FRR \\%} & \\textbf{FRR$_{p_0\\ge.2}$} & \\textbf{Blk/acc.}\\\\\n\\midrule\n" + rows + "\\bottomrule\n\\end{tabular}\n")
    # near-eps arm (checks.py 13): recorded natural loops re-graded on smaller blocks
    ne = d["neareps"]
    m("neLoops", ne["loops"])
    m("neTasks", len(ne["tasks"]))
    m("neCandidates", ne["candidates"])
    m("neBroken", ne["broken"])
    m("neBrokenNear", ne["broken_near"])
    m("neReps", ne["reps"])
    m("neBlockList", ", ".join(f"{t.split('@')[0]} {bs[1]} to {bs[0]}" for t, bs in sorted(ne["block_sizes"].items())))
    nr = ne["rules"]
    for name in RULES:
        k = KEY[name]
        m(f"neFar{k}", pct(nr[name]["far"], 2))
        m(f"neWorst{k}", pct(nr[name]["worst_loop"]))
        m(f"neWorstUp{k}", pct(nr[name]["worst_loop_upper"]))
        m(f"neBpl{k}", num(nr[name]["blocks_per_loop"], 0))
        m(f"neBpa{k}", num(nr[name]["blocks_per_accept"], 0) if nr[name]["blocks_per_accept"] else "--")
        m(f"neFrr{k}", pct(nr[name]["frr"]))
    old = nr["SeqAccept, rho=20, T=800"]
    m("neFarOld", pct(old["far"], 2))
    m("neBplOld", num(old["blocks_per_loop"], 0))
    m("neWorstOld", pct(old["worst_loop"]))
    sa_, b5_, b4_ = nr["SeqAccept"], nr["B5 naive confidence sequence"], nr["B4 peeking Clopper-Pearson"]
    m("neCostRatioBfive", num(sa_["blocks_per_loop"] / b5_["blocks_per_loop"], 2))
    m("neCostRatioBfour", num(sa_["blocks_per_loop"] / b4_["blocks_per_loop"], 1))
    m("neCostRatioOld", num(old["blocks_per_loop"] / sa_["blocks_per_loop"], 1))
    rows = ""
    for name in RULES:
        r = nr[name]
        if name == "SeqAccept":
            rows += "\\midrule\n"
        ci = f" {{\\scriptsize[{pct(r['far_ci_repo'][0], 2)}, {pct(r['far_ci_repo'][1], 2)}]}}"
        bpa = num(r["blocks_per_accept"], 0) if r["blocks_per_accept"] else "--"
        rows += (f"{SHOW[name]} & {pct(r['far'], 2)}{ci} & {pct(r['worst_loop'])} ({pct(r['worst_loop_upper'])}) & "
                 f"{pct(r['frr'])} & {num(r['blocks_per_loop'], 0)} & {bpa}\\\\\n")
    rows += (f"\\sa{{}}, $\\rho=20$, $T=800$ & {pct(old['far'], 2)} & {pct(old['worst_loop'])} ({pct(old['worst_loop_upper'])}) & "
             f"{pct(old['frr'])} & {num(old['blocks_per_loop'], 0)} & {num(old['blocks_per_accept'], 0)}\\\\\n")
    write("tab_neareps.tex", "\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n\\textbf{Rule} & \\textbf{FAR \\%} {\\scriptsize[CI]} & "
          "\\textbf{Worst} & \\textbf{FRR} & \\textbf{Blk/loop} & \\textbf{Blk/acc.}\\\\\n\\midrule\n" + rows + "\\bottomrule\n\\end{tabular}\n")
    # shared cost knobs (checks.py 14): rho and T for B4, B5 and SeqAccept on every arm
    kn = d["knobs"]
    ARM = {"natural": "Natural", "realflaky": "Real flaky", "neareps": "Near-$\\eps$"}
    SET = {"rho=20, T=800": "$\\rho=20$, $T=800$", "rho=5, T=800": "$\\rho=5$, $T=800$",
           "rho=20, derived T": "$\\rho=20$, $T(p_0)$", "rho=5, derived T": "$\\rho=5$, $T(p_0)$ (default)"}
    rows = ""
    for arm, sets in kn.items():
        rows += f"\\multicolumn{{11}}{{@{{}}l}}{{\\textit{{{ARM[arm]}}}}}\\\\\n"
        for kname, rs in sets.items():
            cells = [pct(rs[n]["far"], 2) for n in ("B4", "B5", "SeqAccept")] + \
                    [pct(rs[n]["frr"] or 0) for n in ("B4", "B5", "SeqAccept")] + \
                    [num(rs[n]["blocks_per_loop"], 0) for n in ("B4", "B5", "SeqAccept")]
            rows += f"\\quad {SET[kname]} & " + " & ".join(cells) + "\\\\\n"
        big = load("results/near_eps_checks.json")["big_cap"][arm]  # near_eps_checks.py 3: the same rules with T = 10,000
        cells = [pct(big[n]["far"], 2) for n in ("B4", "B5", "SeqAccept")] + \
                [pct(big[n]["frr"] or 0) for n in ("B4", "B5", "SeqAccept")] + \
                [num(big[n]["blocks_per_loop"], 0) for n in ("B4", "B5", "SeqAccept")]
        rows += "\\quad $\\rho=5$, $T=10{,}000$ & " + " & ".join(cells) + "\\\\\n"
    write("tab_knobs.tex", "\\begin{tabular}{@{}lrrrrrrrrr@{}}\n\\toprule\n & \\multicolumn{3}{c}{\\textbf{FAR \\%}} & "
          "\\multicolumn{3}{c}{\\textbf{FRR \\%}} & \\multicolumn{3}{c}{\\textbf{Blocks per loop}}\\\\\n"
          "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(l){8-10}\n\\textbf{Setting} & B4 & B5 & \\sa{} & B4 & B5 & \\sa{} & B4 & B5 & \\sa{}\\\\\n"
          "\\midrule\n" + rows.replace("multicolumn{11}", "multicolumn{10}") + "\\bottomrule\n\\end{tabular}\n")
    for arm, key in (("natural", "Nat"), ("realflaky", "Rf"), ("neareps", "Ne")):
        dflt = kn[arm]["rho=5, derived T"]
        m(f"kbCostRatioBfive{key}", num(dflt["SeqAccept"]["blocks_per_loop"] / dflt["B5"]["blocks_per_loop"], 2))
        m(f"kbBplSa{key}", num(dflt["SeqAccept"]["blocks_per_loop"], 0))
        m(f"kbBplOldSa{key}", num(kn[arm]["rho=20, T=800"]["SeqAccept"]["blocks_per_loop"], 0))


def ease2():
    """EASE round-1 follow-ups: results/comparators.json (comparators.py)."""
    d = load("results/comparators.json")
    v = d["variants"]
    for name, t in (("SeqAccept", "Base"), ("no refund at cap", "Nocap"), ("prefilter 1", "Prefilter"),
                    ("earlier default", "Old")):
        for arm, at in (("natural", "Nat"), ("realflaky", "Rf"), ("realflaky_high", "RfHigh")):
            r = v[name][arm]
            m(f"var{t}{at}Frr", pct(r["frr"]))
            m(f"var{t}{at}Far", pct(r["far"], 2))
            m(f"var{t}{at}Bpl", num(r["blocks_per_loop"]))
            m(f"var{t}{at}Bpa", num(r["blocks_per_accept"], 0))
    m("prefilterSaving", pct(1 - v["prefilter 1"]["natural"]["blocks_per_loop"] / v["SeqAccept"]["natural"]["blocks_per_loop"], 0))
    w = d["worst010"]
    for n, nk in (("15", "Fifteen"), ("50", "Fifty")):
        for name in ("B4 peeking Clopper-Pearson", "B5 naive confidence sequence", "SeqAccept"):
            m(f"worstTen{KEY[name]}{nk}", pct(w[n][name]))
    lo = d["leave_one_repo_out"]
    for name in RULES[:3]:
        m(f"loroLo{KEY[name]}", pct(lo[name][0]))
        m(f"loroHi{KEY[name]}", pct(lo[name][1]))
    pb = d["per_bug"]
    m("bugsWithBoneError", sum(1 for b in pb.values() if b["B1 stop at first pass"] > 0))
    m("bugsBoneMax", pct(max(b["B1 stop at first pass"] for b in pb.values())))
    fk = d["fullsuite_kind"]
    for kind, t in (("ordinary", "Ordinary"), ("order", "Order")):
        for name in RULES[:3]:
            m(f"fullKind{t}{KEY[name]}", pct(fk[kind][name]))
        m(f"fullKind{t}Loops", fk[kind]["loops"])
        m(f"sampledKind{t}Bone", pct(fk[kind]["sampled_B1"]))
    m("pzeroSeNat", f"{d['p0_se']['natural_max']:.3f}")
    m("pzeroSeInj", f"{d['p0_se']['injected_max']:.3f}")
    m("deltaEffNat", f"{0.05 + 1.645 * d['p0_se']['natural_max']:.3f}")
    nr = d["natural_rules"]
    m("secPerAcceptSa", num(nr["SeqAccept"]["seconds_per_accept"] / 60))
    m("secPerAcceptBone", num(nr["B1 stop at first pass"]["seconds_per_accept"]))
    # deployment envelope: a block of b tests from N; a regression breaking k always-failing tests fails it w.p. ~ 1-(1-k/N)^b
    frac = 1 - (1 - 0.05) ** (1 / 32)
    m("envFrac", f"{100 * frac:.2f}")
    # blocks sample 32 tests without replacement: P(block hits one of k broken tests of N) = 1 - C(N-k,32)/C(N,32)
    def hit(n_, k_, b_=32):
        q = 1.0
        for i in range(b_):
            q *= (n_ - k_ - i) / (n_ - i)
        return 1 - q
    m("envOneTestSuite", f"{max(n_ for n_ in range(32, 5000) if hit(n_, 1) > 0.05 + 1e-12)}")
    m("envThreeK", f"{min(k_ for k_ in range(1, 3000) if hit(3000, k_) > 0.05 + 1e-12)}")
    # operating curve: 2,000 runs per point (ablations.OP_REPS); widest 95% Wilson half-width is at 50%
    lo, hi = wilson(1000, 2000)
    m("opHalfWidth", f"{100 * (hi - lo) / 2:.1f}")
    # main table: natural loops (all rules), injected modes, natural cost
    res = load("results/results.json")
    per = res["per_episode"]
    nat = [e for e in per if e["flakiness"] == "natural"]
    e1 = load("results/ablations.json")
    rows = ""
    bm = d["by_mode"]
    sens = load("results/sensitivity.json")

    def cost_cells(name):  # blocks per loop, per accept, with p0 per task and per loop, minutes per accept (p0 per task)
        c = sens["cost"][name]
        same = abs(c["bpa_task"] - c["blocks_per_accept"]) < 1e-9
        return (f"{num(c['blocks_per_loop'])} & {num(c['blocks_per_accept'])} & {'--' if same else num(c['bpa_task'])} & "
                f"{'--' if same else num(c['bpa_loop'])} & {num(c['minutes_per_accept'], 2)}\\\\\n")
    for name in RULES:
        if name == "SeqAccept":
            rows += "\\midrule\n"
        nf = e1["natural_far"][name]
        hit = {}
        for e in nat:
            if "broken" in e["labels"]:
                b = e["file"].split("-")[0]
                hit[b] = hit.get(b, False) or e["rules"][name]["far"] > 0
        kb, nb = sum(hit.values()), len(hit)
        rci = sens["natural_repo_ci"][name]["ci"]
        ci = "" if nf["far"] == 0 else f" {{\\scriptsize[{pct(rci[0])}, {pct(rci[1])}]}}"
        ub = f" {{\\scriptsize({pct(replay.cp_upper(kb, nb))})}}" if kb == 0 else ""
        r = nr[name]
        valid = {"B5 naive confidence sequence": "per cand.", "B6 SeqAccept, no refund": "yes", "SeqAccept": "yes"}.get(name, "no")
        rows += (f"{SHOW[name]} & {valid} & {pct(nf['far'])}{ci} & {kb}/{nb}{ub} & {pct(r['frr'])} & "
                 f"{pct(bm['injected-bernoulli'][name])} & {pct(bm['injected-markov'][name])} & " + cost_cells(name))
    co = load("results/checks.json")["corollary"]  # Corollary 4.3 as run: eps = L + delta, level 0.10 against p0 + delta
    cn = co["natural"]
    ub = f" {{\\scriptsize({pct(cn['bugs_upper'])})}}" if cn["bugs_hit"] == 0 else ""
    rows += (f"\\sa{{}}, $\\eps=L+\\delta$ & yes$^\\dagger$ & {pct(cn['far'])} & {cn['bugs_hit']}/{cn['bugs_with_broken']}{ub} & {pct(cn['frr'])} & "
             f"{pct(co['bernoulli'])} & {pct(co['markov'])} & {num(cn['blocks_per_loop'])} & {num(cn['blocks_per_accept'])} & "
             f"{num(cn['bpa_task'])} & {num(cn['bpa_loop'])} & {num(cn['minutes_per_accept'], 2)}\\\\\n")
    rows += "\\midrule\n"
    comp = {"Harmonic spending": "Harmonic spending", "Discard spending": "Discarding",
            "Fixed-n Bonferroni-15 (1n0)": "Fixed-$n$ Bonf.-15 ($n_0$)", "Fixed-n Bonferroni-15 (4n0)": "Fixed-$n$ Bonf.-15 ($4n_0$)"}
    rr = next(k for k in nr if k.startswith("Rerun-"))
    comp[rr] = rr.replace(" (all green)", " (consec.)")
    cvalid = {"Discard spending": "unproved", rr: "no"}
    for key, label in comp.items():
        r = nr[key]
        rows += f"{label} & {cvalid.get(key, 'yes')} & {pct(r['far'])} & -- & {pct(r['frr'])} & -- & -- & " + cost_cells(key)
    write("tab_outcome.tex",
          "\\begin{tabular}{@{}llrrrrr|rrrrr@{}}\n\\toprule\n"
          "& & \\multicolumn{3}{c}{\\textbf{Natural loops (" + str(len(nat)) + ")}} & \\multicolumn{2}{c|}{\\textbf{Injected FAR \\%}} & \\multicolumn{5}{c}{\\textbf{Cost, natural loops}}\\\\\n"
          "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-12}\n"
          "& & & & & & & & \\multicolumn{3}{c}{\\textbf{Blk/acc., $p_0$ calibration}} & \\\\\n"
          "\\textbf{Rule} & \\textbf{Valid} & \\textbf{FAR \\%} {\\scriptsize[repo.\\ CI]} & \\textbf{Bugs hit} {\\scriptsize(UB \\%)} & \\textbf{FRR \\%} & "
          "\\textbf{Bern.} & \\textbf{Markov} & \\textbf{Blk/loop} & \\textbf{none} & \\textbf{per task} & \\textbf{per loop} & \\textbf{Min/acc.}\\\\\n\\midrule\n"
          + rows + "\\bottomrule\n\\end{tabular}\n")



def slice_caps():
    """Cap ablation: results/slice_caps.json (slice_caps.py)."""
    d = load("results/slice_caps.json")
    tag = {"SeqAccept": "Sa", "B5": "Bfive", "B6": "Bsix", "Harmonic": "Harm", "Bonferroni-15": "BonfFifteen",
           "Bonferroni-1000": "BonfThousand"}
    for cap, ct in (("shared", "Sh"), ("slice", "Sl")):
        for n, t in tag.items():
            m(f"sc{ct}LateTen{t}", pct(d[cap]["late"]["10"][n]))
            m(f"sc{ct}LateForty{t}", pct(d[cap]["late"]["40"][n]))
            m(f"sc{ct}NeBpl{t}", num(d[cap]["neareps"][n]["blocks_per_loop"], 0))
            m(f"sc{ct}NatBpa{t}", num(d[cap]["natural"][n]["blocks_per_accept"] or 0, 0))
            m(f"sc{ct}NatFrr{t}", pct(d[cap]["natural"][n]["frr"]))
    sl = d["slice"]["neareps"]
    for n in ("B6", "Harmonic", "Bonferroni-1000"):
        m(f"scNeRatio{tag[n]}", f'{sl[n]["blocks_per_loop"] / sl["SeqAccept"]["blocks_per_loop"]:.1f}')
    m("scNeRatioSaBfive", f'{sl["SeqAccept"]["blocks_per_loop"] / sl["B5"]["blocks_per_loop"]:.2f}')
    assert d["slice"]["natural"]["SeqAccept"]["far"] == 0, "text: no false acceptance on natural loops with either cap"

def near_eps_checks():
    """Near-eps arm and cost checks: results/near_eps_checks.json (near_eps_checks.py)."""
    d = load("results/near_eps_checks.json")
    ne = d["neareps"]
    per_loop = lambda r: r["blocks_per_accept"] * (1 + 400 / r["blocks_per_loop"])  # calibration repeated per loop
    K = {"B1 stop at first pass": "Bone", "B4 peeking Clopper-Pearson": "Bfour", "B5 naive confidence sequence": "Bfive",
         "B6 SeqAccept, no refund": "Bsix", "Harmonic spending": "Harm", "Bonferroni-100 spending": "BonfHundred",
         "Fixed-n Bonferroni-15 (1n0)": "Fixn", "SeqAccept": "Sa"}
    LAB = {"B1 stop at first pass": "B1", "B4 peeking Clopper-Pearson": "B4", "B5 naive confidence sequence": "B5",
           "B6 SeqAccept, no refund": "B6", "Harmonic spending": "Harmonic", "Bonferroni-100 spending": "Bonf.-100",
           "Fixed-n Bonferroni-15 (1n0)": "Fixed-$n$ Bonf.-15", "SeqAccept": "\\sa{}"}
    VALID = {"B1 stop at first pass": "no", "B4 peeking Clopper-Pearson": "no", "B5 naive confidence sequence": "cand.",
             "B6 SeqAccept, no refund": "yes", "Harmonic spending": "yes", "Bonferroni-100 spending": "yes",
             "Fixed-n Bonferroni-15 (1n0)": "yes", "SeqAccept": "yes"}
    m("neLoopsAll", ne["SeqAccept"]["loops"])
    for n, k in K.items():
        r = ne[n]
        m(f"neAbove{k}", r["loops_above_alpha"])
        m(f"neFarAll{k}", pct(r["far"], 2))
        m(f"neWorstAll{k}", pct(r["worst_loop"]))
        m(f"neBplAll{k}", num(r["blocks_per_loop"], 0))
        if r["bpa_calib"]:
            m(f"neBpaCal{k}", num(r["bpa_calib"], 0))
            m(f"neMin{k}", num(r["min_per_accept"]))
    for n, k in (("old B4", "Bfour"), ("old B5", "Bfive"), ("old B6", "Bsix"), ("old SeqAccept", "Sa")):
        r = ne[n]
        m(f"neOldFar{k}", pct(r["far"], 2))
        m(f"neOldWorst{k}", pct(r["worst_loop"]))
        m(f"neOldAbove{k}", r["loops_above_alpha"])
        m(f"neOldBpl{k}", num(r["blocks_per_loop"], 0))
    sa = ne["SeqAccept"]
    m("neRatioBsixSa", num(ne["B6 SeqAccept, no refund"]["blocks_per_loop"] / sa["blocks_per_loop"], 2))
    m("neRatioHarmSa", num(ne["Harmonic spending"]["blocks_per_loop"] / sa["blocks_per_loop"], 2))
    fr = d["firmer"]  # near_eps_checks.py 7: labels re-measured on 2,400 blocks, exact for the replay's draws
    po = fr["pooled"]
    m("frRemeasured", fr["remeasured"])
    m("frExtra", f"{fr['extra_blocks'][0]:,}".replace(",", "{,}"))
    m("frBlocks", f"{fr['extra_blocks'][0] + 400:,}".replace(",", "{,}"))
    m("frMargBroken", fr["changes"]["marginal->broken"])
    m("frMargMarg", fr["changes"]["marginal->marginal"])
    m("frBrokenKept", fr["changes"]["broken->broken"])
    m("frOtherChanges", sum(v for k, v in fr["changes"].items() if k not in ("marginal->broken", "marginal->marginal", "broken->broken")))
    for n, k in {**K, "old B5": "OldBfive", "old SeqAccept": "OldSa"}.items():
        r = po[n]
        m(f"frFar{k}", pct(r["far"], 2))
        m(f"frWorst{k}", pct(r["worst_loop"]))
        m(f"frWorstUp{k}", pct(r["worst_loop_upper"]))
        m(f"frAbove{k}", r["loops_above_alpha"])
        m(f"frLow{k}", r["loops_lower_above"])
        m(f"frTasks{k}", r["tasks_lower_above"])
    for n, k in (("B5 naive confidence sequence", "Bfive"), ("SeqAccept", "Sa"), ("B6 SeqAccept, no refund", "Bsix")):
        m(f"frToFar{k}", pct(fr["truth_only"][n]["far"], 2))
        m(f"frToLow{k}", fr["truth_only"][n]["loops_lower_above"])
    rows = ""
    for n, k in K.items():
        r = ne[n]
        if n == "SeqAccept":
            rows += "\\midrule\n"
        bpa = num(per_loop(r), 0) if r["blocks_per_accept"] else "--"
        rows += (f"{LAB[n]} & {VALID[n]} & {pct(r['far'], 2)} & {r['loops_above_alpha']} & {pct(r['worst_loop'])} & "
                 f"{num(r['blocks_per_loop'], 0)} & {bpa} & {pct(po[n]['far'], 2)} & {po[n]['loops_lower_above']}\\\\\n")
    cor = load("results/design_checks.json")["split"]["total10"]["neareps"]  # design_checks.py 4: Corollary 4.3 row
    rows += (f"\\sa{{}}, $\\eps=L+\\delta$ & yes$^\\dagger$ & {pct(cor['far'], 2)} & -- & {pct(cor['worst_loop'])} & "
             f"{num(cor['blocks_per_loop'], 0)} & {num(per_loop(cor), 0)} & -- & --\\\\\n")
    rows += "\\midrule\n\\multicolumn{9}{@{}l}{\\textit{Earlier settings $\\rho=20$, $T=800$}}\\\\\n"
    for n in ("old B4", "old B5", "old B6", "old SeqAccept"):
        r = ne[n]
        lab = LAB[{"old B4": "B4 peeking Clopper-Pearson", "old B5": "B5 naive confidence sequence",
                   "old B6": "B6 SeqAccept, no refund", "old SeqAccept": "SeqAccept"}[n]]
        bpa = num(per_loop(r), 0) if r["blocks_per_accept"] else "--"
        rows += (f"\\quad {lab} & {VALID[{'old B4': 'B4 peeking Clopper-Pearson', 'old B5': 'B5 naive confidence sequence', 'old B6': 'B6 SeqAccept, no refund', 'old SeqAccept': 'SeqAccept'}[n]]} & {pct(r['far'], 2)} & {r['loops_above_alpha']} & {pct(r['worst_loop'])} & "
                 f"{num(r['blocks_per_loop'], 0)} & {bpa} & {pct(po[n]['far'], 2)} & {po[n]['loops_lower_above']}\\\\\n")
    write("tab_neareps.tex", "\\begin{tabular}{@{}llrrrrrrr@{}}\n\\toprule\n& & \\multicolumn{5}{c}{\\textbf{Labels on \\nTruth{} blocks}} & "
          "\\multicolumn{2}{c}{\\textbf{Re-measured}}\\\\\n\\cmidrule(lr){3-7}\\cmidrule(l){8-9}\n\\textbf{Rule} & \\textbf{Valid} & \\textbf{FAR \\%} & $>\\alpha$ & "
          "\\textbf{Worst} & \\textbf{Blk/loop} & \\textbf{Blk/acc.} & \\textbf{FAR \\%} & \\textbf{LB}\\\\\n\\midrule\n"
          + rows + "\\bottomrule\n\\end{tabular}\n")
    # per task (supplement)
    tasks = sorted(sa["per_task"])
    four = ("B4 peeking Clopper-Pearson", "B5 naive confidence sequence", "B6 SeqAccept, no refund", "SeqAccept")
    rows = "".join(f"{t.split('@')[0]} & " + " & ".join(pct(ne[n]["per_task"][t], 2) for n in four) + "\\\\\n" for t in tasks)
    write("tab_neareps_tasks.tex", "\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n\\textbf{Task} & B4 & B5 & B6 & \\sa{}\\\\\n\\midrule\n"
          + rows + "\\bottomrule\n\\end{tabular}\n")
    # labels near eps
    lb = d["neareps_labels"]
    m("neNearDistinct", lb["near_distinct"])
    m("neNearRisky", lb["near_risky"])
    for tag, T in (("unamb", "Unamb"), ("pool", "Pool"), ("swap", "Swap")):
        for n, k in zip(four, ("Bfour", "Bfive", "Bsix", "Sa")):
            v = lb["far"][tag][n]
            m(f"ne{T}Far{k}", pct(v["far"], 2))
            m(f"ne{T}Worst{k}", pct(v["worst_loop"]))
            m(f"ne{T}Above{k}", v["loops_above_alpha"])
    # cap of 10,000
    bc = d["big_cap"]
    for arm, A in (("natural", "Nat"), ("realflaky", "Rf"), ("neareps", "Ne")):
        for n, k in (("B4", "Bfour"), ("B5", "Bfive"), ("SeqAccept", "Sa")):
            v = bc[arm][n]
            m(f"bc{A}Far{k}", pct(v["far"], 2))
            m(f"bc{A}Frr{k}", pct(v["frr"] or 0, 2))
            m(f"bc{A}Bpl{k}", num(v["blocks_per_loop"], 0))
    m("bcNeRatioSa", num(bc["neareps"]["SeqAccept"]["blocks_per_loop"] / sa["blocks_per_loop"], 1))
    m("bcNeRatioBfiveSa", num(bc["neareps"]["SeqAccept"]["blocks_per_loop"] / ne["B5 naive confidence sequence"]["blocks_per_loop"], 1))
    # refund crossing
    rc = d["refund_crossing"]
    for kept, T in (("0.0", "Zero"), ("0.5", "Half"), ("0.9", "Nine"), ("0.99", "NineNine")):
        m(f"rcCross{T}", f"{rc['by_kept'][kept]:,}".replace(",", "{,}"))
    for arm, A in (("natural", "Nat"), ("neareps", "Ne"), ("realflaky", "Rf")):
        m(f"rcKept{A}", f"{rc[arm]['kept']:.2f}")
        m(f"rcCross{A}", f"{rc[arm]['crossing']:,}".replace(",", "{,}"))
    # eps redrawn per replay
    rd = d["redrawn_eps"]
    m("rdFarSa", pct(rd["far"]["SeqAccept"], 2))
    m("rdFarBfour", pct(rd["far"]["B4 peeking Clopper-Pearson"], 2))
    m("rdFarBfive", pct(rd["far"]["B5 naive confidence sequence"], 2))
    m("rdWorstSa", pct(rd["worst_task_sa"][0], 2))
    m("rdWorstTask", rd["worst_task_sa"][1].split("@")[0].replace("_", "\\_"))
    m("rdDraws", rd["draws"])
    # cost model with a start-up cost
    oh = d["overhead"]
    m("ohExtraMs", f"{1000 * oh['seconds_per_extra_test']:.1f}")
    for k, T in (("100", "Hundred"), ("1000", "Thousand")):
        for ell, L in ((1, "One"), (2, "Two"), (3, "Three"), (5, "Five"), (10, "Ten")):
            vals = [oh["ratio_B1"][k][f"{N}-{ell}"] for N in (1000, 3000, 10000)]
            m(f"oh{T}{L}", f"{min(vals):.2f}--{max(vals):.2f}")
            m(f"oh{T}{L}Big", f"{oh['ratio_B1'][k][f'10000-{ell}']:.2f}")


def deploy_main(d, dl):
    """Main-text deployment table: the default, the default at alpha = 0.04, eps = L_0.99 + 0.05 and + delta, and B4,
    with calibration repeated per loop (deploy_checks.json steps 2 and 12; the L + delta row from design_checks.json)."""
    sp = load("results/design_checks.json")["split"]["total05"]
    per_loop = lambda v: v["blocks_per_accept"] * (1 + 400 / v["blocks_per_loop"])
    rows = [("Default", "$\\eps$, 0.05", {a: dl[a]["default"] for a in dl}, lambda v: v["bpa_loop"]),
            ("Default, $\\alpha{=}.04$", "$p_0{+}\\delta$, 0.05", d["default04"], lambda v: v["bpa_loop"]),
            ("$L_{.99}{+}.05$, $\\alpha{=}.04$", "$p_0{+}.05$, 0.05", {a: dl[a]["total05"] for a in dl}, lambda v: v["bpa_loop"]),
            ("$L_{.99}{+}\\delta$, $\\alpha{=}.04^*$", "$p_0{+}\\delta$, 0.05", sp, per_loop),
            ("B4", "none", {a: dl[a]["B4"] for a in dl}, lambda v: v["bpa_loop"])]
    body = ""
    for name, g, v, bpl in rows:
        n, e, r = v["natural"], v["neareps"], v["realflaky"]
        body += (f"{name} & {g} & {num(bpl(n), 0)} & {pct(e['far'], 2)} & {num(bpl(e), 0)} & {pct(r['far'], 2)} & "
                 f"{pct(r['frr'] or 0, 1)} & {num(bpl(r), 0)}\\\\\n")
    write("tab_deploy_main.tex", "\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule\n& & \\textbf{Nat.} & "
          "\\multicolumn{2}{c}{\\textbf{Near $\\eps$}} & \\multicolumn{3}{c}{\\textbf{Real flaky (exploratory)}}\\\\\n"
          "\\cmidrule(lr){3-3}\\cmidrule(lr){4-5}\\cmidrule(l){6-8}\n\\textbf{Rule} & \\textbf{Protects} & Blk/acc. & FAR & Blk/acc. & "
          "FAR & FRR & Blk/acc.\\\\\n\\midrule\n" + body + "\\bottomrule\n\\end{tabular}\n")


def dependence_table(d):
    """Dependence variants: FAR on 15 broken candidates at p = 0.06 under a lag-1 chain, and cost and FRR on recorded
    loops (deploy_checks.json steps 1 and 13)."""
    dep, cost, frr, df = d["dependence"], d["dependence_cost"], d["dependence_frr"], d["deflated"]
    body = ""
    for k, name in (("e1", "\\sa{} (maximal)"), ("e05", "Half refund"), ("e0", "No refund (B6)"), ("g025", "$\\gamma=1/4$")):
        far = [pct(dep[f"{phi}-15"][k]) for phi in ("0.0", "0.2", "0.5")]
        body += (f"{name} & independent & " + " & ".join(far) + f" & {num(cost['natural'][k], 0)} & "
                 f"{num(cost['realflaky'][k], 0)} & {pct(frr['realflaky'][k] or 0)}\\\\\n")
    for pm in ("0.25", "0.5"):
        v = df[pm]
        far = [pct(v["far"][phi]) for phi in ("0.0", "0.2", "0.5")]
        body += (f"$\\eps(1-{pm.lstrip('0')})$ wallet & $\\phi\\le{pm.lstrip('0')}$ & " + " & ".join(far) + f" & "
                 f"{num(v['natural']['blocks_per_accept'], 0)} & {num(v['realflaky']['blocks_per_accept'], 0)} & "
                 f"{pct(v['realflaky']['frr'] or 0)}\\\\\n")
    write("tab_dependence.tex", "\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule\n& & \\multicolumn{3}{c}{\\textbf{FAR \\% at $\\phi$}} & "
          "\\multicolumn{2}{c}{\\textbf{Blk/acc.}} & \\textbf{FRR}\\\\\n\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\\cmidrule(l){8-8}\n"
          "\\textbf{Variant} & \\textbf{Valid for} & 0 & 0.2 & 0.5 & Nat. & Flaky & Flaky \\%\\\\\n\\midrule\n"
          + body + "\\bottomrule\n\\end{tabular}\n")


def deploy_checks():
    """Deployment checks: results/deploy_checks.json (deploy_checks.py) and the near-eps re-measurement per task."""
    d = load("results/deploy_checks.json")
    # 1. refund share and gamma under dependent blocks
    V = {"e1": "Max", "e05": "Half", "e0": "None", "g025": "Quarter"}
    for phi, P in (("0.2", "Low"), ("0.5", "Mid")):
        for k, K in V.items():
            m(f"rsh{K}{P}Fifteen", pct(d["dependence"][f"{phi}-15"][k]))
    for arm, A in (("natural", "Nat"), ("neareps", "Ne"), ("realflaky", "Rf")):
        c = d["dependence_cost"][arm]
        m(f"rshExtraHalf{A}", f"{100 * (c['e05'] / c['e1'] - 1):.0f}")
    # 2. deployment rows; per-loop calibration-inclusive ratios
    dl = d["deployment"]
    for arm, A in (("natural", "Nat"), ("neareps", "Ne"), ("realflaky", "Rf")):
        for k, K in (("default", "Sa"), ("total05", "Tot"), ("B4", "Bfour"), ("fixedn15", "Fixn")):
            v = dl[arm][k]
            m(f"dlFar{K}{A}", pct(v["far"], 2))
            m(f"dlFrr{K}{A}", pct(v["frr"] or 0, 1))
            m(f"dlBpl{K}{A}", num(v["bpa_loop"], 0))
        m(f"plRatioBfour{A}", num(dl[arm]["default"]["bpa_loop"] / dl[arm]["B4"]["bpa_loop"], 2))
    m("plRatioFixnNat", num(dl["natural"]["default"]["bpa_loop"] / dl["natural"]["fixedn15"]["bpa_loop"], 2))
    # cost-ratio table (round 15, E15-F): SeqAccept's blocks per accept over B4's and Bonferroni-15's, by accounting
    rows = ""
    for arm, A, show in (("natural", "Nat", "Natural"), ("neareps", "Ne", "Near $\\eps$ (expl.)"), ("realflaky", "Rf", "Real flaky (expl.)")):
        sa, cells = dl[arm]["default"], []
        for k in ("B4", "fixedn15"):
            r0, r1 = sa["blocks_per_accept"] / dl[arm][k]["blocks_per_accept"], sa["bpa_loop"] / dl[arm][k]["bpa_loop"]
            m(f"ctNone{'Bfour' if k == 'B4' else 'Fixn'}{A}", num(r0, 2))
            cells += [num(r0, 2), num(r1, 2)]
        rows += (f"{show} & {cells[0]} & {cells[1]} & {pct(dl[arm]['B4']['far'])} & {cells[2]} & {cells[3]} & "
                 f"{pct(dl[arm]['fixedn15']['frr'] or 0)} & {pct(sa['frr'] or 0)}\\\\\n")
    write("tab_cost.tex",
          "\\begin{tabular}{@{}lrrrrrrr@{}}\n\\toprule\n"
          "& \\multicolumn{3}{c}{\\textbf{vs.\\ B4}} & \\multicolumn{3}{c}{\\textbf{vs.\\ Bonf.-15}} & \\textbf{\\sa{}}\\\\\n"
          "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(l){8-8}\n"
          "\\textbf{Arm} & none & loop & FAR & none & loop & FRR & FRR\\\\\n\\midrule\n"
          + rows + "\\bottomrule\n\\end{tabular}\n")
    m("plCalibShareNat", f"{100 * 400 / dl['natural']['default']['bpa_loop']:.0f}")
    sp = load("results/design_checks.json")["split"]["total05"]
    rows = []
    lab = {"default": ("\\sa{}, $\\eps=\\hat p_0+0.05$", "declared $\\eps$, 0.05"),
           "total05": ("\\sa{}, $\\eps=L_{0.99}+0.05$, $\\alpha=0.04$", "$p_0+0.05$, 0.05"),
           "B4": ("B4 Fixed-sample CI", "none"), "fixedn15": ("Fixed-$n$ Bonf.-15 ($n_0$)", "declared $\\eps$, cap 15")}
    for k, (name, g) in lab.items():
        cells = []
        for arm in ("natural", "neareps", "realflaky"):
            v = dl[arm][k]
            cells += [pct(v["far"], 2), pct(v["frr"] or 0, 1), num(v["bpa_loop"], 0)]
        rows.append(f"{name} & {g} & " + " & ".join(cells) + "\\\\\n")
    cells = []
    for arm in ("natural", "neareps", "realflaky"):
        v = d["default04"][arm]
        cells += [pct(v["far"], 2), pct(v["frr"] or 0, 1), num(v["bpa_loop"], 0)]
    rows.insert(1, f"\\sa{{}}, $\\eps=\\hat p_0+0.05$, $\\alpha=0.04$ & $p_0+\\delta$, 0.05$^\\ddagger$ & " + " & ".join(cells) + "\\\\\n")
    cells = []
    for arm in ("natural", "neareps", "realflaky"):
        v = sp[arm]
        per_loop = v["blocks_per_accept"] * (1 + 400 / v["blocks_per_loop"])
        cells += [pct(v["far"], 2), pct(v["frr"] or 0, 1), num(per_loop, 0)]
    rows.append(f"\\sa{{}}, $\\eps=L_{{0.99}}+\\delta$, $\\alpha=0.04$$^*$ & $p_0+\\delta$, 0.05 & " + " & ".join(cells) + "\\\\\n")
    write("tab_deploy.tex", "\\begin{tabular}{@{}llrrrrrrrrr@{}}\n\\toprule\n& & \\multicolumn{3}{c}{\\textbf{Natural}} & "
          "\\multicolumn{3}{c}{\\textbf{Near $\\eps$}} & \\multicolumn{3}{c}{\\textbf{Real flaky}}\\\\\n"
          "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(l){9-11}\n\\textbf{Configuration} & \\textbf{Guarantee, level} & FAR & FRR & Blk/acc. & "
          "FAR & FRR & Blk/acc. & FAR & FRR & Blk/acc.\\\\\n\\midrule\n" + "".join(rows) + "\\bottomrule\n\\end{tabular}\n")
    # 3. on-policy prefix
    pf = d["prefix"]
    m("pfAfter", word_int(pf["natural"]["after_green"]))
    m("pfCands", word_int(pf["natural"]["candidates"]))
    m("pfAfterPct", f"{100 * pf['natural']['after_green'] / pf['natural']['candidates']:.0f}")
    for arm, A in (("natural", "Nat"), ("realflaky", "Rf")):
        for n, K in (("B1 stop at first pass", "Bone"), ("B4 peeking Clopper-Pearson", "Bfour"), ("SeqAccept", "Sa")):
            m(f"pf{K}{A}", pct(pf[arm]["far_prefix"][n], 2 if K != "Bone" else 1))
    # 4. screening: chosen against unchosen usable pairs
    sc = d["screening"]
    for t, T in (("chosen", "Chosen"), ("unchosen", "Unchosen")):
        m(f"selc{T}N", sc[t]["n"])
        m(f"selcFlaky{T}", f"{sc[t]['flaky_rate']:.2f}")
        m(f"selcPzero{T}", f"{sc[t]['p0']:.2f}")
        m(f"selcSec{T}", f"{sc[t]['seconds_per_block']:.2f}")
    # 6-9. break-even dependence, yamicache count, near-eps per candidate, discarding
    m("bePhiFifteen", f"{d['breakeven_phi']['15']:.2f}")
    m("bePhiFifty", f"{d['breakeven_phi']['50']:.2f}")
    y = d["yamicache"]
    assert y["risky"] == y["false"], "the paper says all of SeqAccept's yamicache false acceptances have uncertain labels"
    m("yamFalse", y["false"]); m("yamRisky", y["risky"]); m("yamReplays", y["replays"]); m("yamDistinct", y["distinct"])
    # 14. the yamicache candidates behind those false acceptances, re-measured on fresh blocks (neareps_truth.py)
    yf, ft = d["yamicache_firm"], json.loads(Path("results/flaky_truth.json").read_text())
    runs = [v for t in ft.values() for f, v in t.items() if f in y["ids"]]
    kn = [(round(v["truth200"] * 200) + sum(v["extra"]), 200 + len(v["extra"])) for v in runs]
    assert len(runs) == y["distinct"] and all(k / n < yf["eps"] for k, n in kn) and yf["far"] == 0
    m("yamFirmExtra", f"{len(runs[0]['extra']):,}"); m("yamFirmBlocks", f"{kn[0][1]:,}")
    m("yamFirmLo", pct(min(k / n for k, n in kn))); m("yamFirmHi", pct(max(k / n for k, n in kn)))
    m("yamFirmUp", pct(max(replay.cp_upper(k, n) for k, n in kn)))
    m("yamFirmEps", pct(yf["eps"], 0)); m("yamFirmFar", pct(yf["far"]))
    for n, K in (("B4 peeking Clopper-Pearson", "Bfour"), ("B5 naive confidence sequence", "Bfive"), ("SeqAccept", "Sa")):
        m(f"nc{K}Distinct", d["near_candidates"][n]["distinct"])
        m(f"nc{K}False", d["near_candidates"][n]["false"])
    dmax = max(d["discarding"].values())
    m("discMax", pct(dmax))
    z = 1.96; nr = 2000
    m("discMaxUp", pct((dmax + z * z / (2 * nr) + z * ((dmax * (1 - dmax) / nr + z * z / (4 * nr * nr)) ** 0.5)) / (1 + z * z / nr)))
    da = max(d["discard_adversary"].values())
    m("discAdvMax", pct(da))
    m("discAdvMaxUp", pct((da + z * z / (2 * nr) + z * ((da * (1 - da) / nr + z * z / (4 * nr * nr)) ** 0.5)) / (1 + z * z / nr)))
    ex = d["exposure"]
    m("expOneNat", pct(ex["natural"]["at_least_1"]))
    m("expOneRf", pct(ex["realflaky"]["at_least_1"]))
    m("expFiveRf", pct(ex["realflaky"]["at_least_5"]))
    for arm, A in (("natural", "Nat"), ("realflaky", "Rf")):
        m(f"expThree{A}", pct(ex[arm]["at_least_1_w3"])); m(f"expPost{A}", pct(ex[arm]["posterior_1"]))
    m("expFiveThreeRf", pct(ex["realflaky"]["at_least_5_w3"]))
    deploy_main(d, dl)
    dependence_table(d)
    # step 15: attempts per loop; retained suite sizes; power of the serial-against-parallel gate
    at = d["attempts"]
    for arm, A in (("natural", "Nat"), ("realflaky", "Rf")):
        for n, K in (("SeqAccept", "Sa"), ("B4 peeking Clopper-Pearson", "Bfour")):
            m(f"att{K}{A}Med", f"{at[arm][n]['median']:.0f}"); m(f"att{K}{A}Ninety", f"{at[arm][n]['p90']:.0f}")
    import statistics
    tasks = sorted({p.stem.split("-")[0] for p in Path("episodes").glob("*.json") if not replay.excluded(p)
                    and "@" not in p.stem.split("-")[0]})
    sizes = [len(json.loads(Path(f"tasks/{t}.json").read_text())["test_ids"]) for t in tasks]
    assert len(sizes) == 17, len(sizes)
    m("suiteMin", min(sizes)); m("suiteMed", f"{statistics.median(sizes):.0f}"); m("suiteMax", max(sizes))
    from statistics import NormalDist
    nb, p0g = 400, 0.2  # calibration blocks per mode; p0 of the flakiest tasks
    def gate_power(gap):
        p1, p2 = p0g, p0g + gap
        pb = (p1 + p2) / 2
        se0, se1 = (2 * pb * (1 - pb) / nb) ** 0.5, (p1 * (1 - p1) / nb + p2 * (1 - p2) / nb) ** 0.5
        return 1 - NormalDist().cdf((1.96 * se0 - gap) / se1)
    m("gatePowFive", pct(gate_power(0.05), 0)); m("gatePowTen", pct(gate_power(0.10), 0))
    for arm, A in (("natural", "Nat"), ("realflaky", "Rf")):
        v = dl[arm]["default"]
        m(f"lat{A}Cal", num(d["latency"][arm]["default"] * v["bpa_loop"] / v["blocks_per_accept"]))
    spr = load("results/design_checks.json")["split"]["total05"]["realflaky"]
    m("dlBplDeltaRf", num(spr["blocks_per_accept"] * (1 + 400 / spr["blocks_per_loop"]), 0))
    ftb = load("results/near_eps_checks.json")["firmer"]["pooled"]["B5 naive confidence sequence"]["tasks_lower_above"]
    m("frTasksBfiveWord", "none" if ftb == 0 else str(ftb))
    m("fullMinRatio", f"{float(macros['costMinTaskSa']) / float(macros['fullMinBone']):.0f}")
    # per-candidate FAR near eps (step 8), default at alpha 0.04 (step 12), deflated wallet (step 13)
    nc = d["near_candidates"]
    m("ncCands", nc["SeqAccept"]["candidates"])
    for n, K in (("B4 peeking Clopper-Pearson", "Bfour"), ("B5 naive confidence sequence", "Bfive"), ("SeqAccept", "Sa")):
        m(f"nc{K}AboveLow", nc[n]["above_alpha_lower"])
        w = nc[n]["worst"]
        m(f"nc{K}Worst", pct(w["far"])); m(f"nc{K}WorstLo", pct(w["lower"])); m(f"nc{K}WorstUp", pct(w["upper"]))
    dz = d["default04"]
    for arm, A in (("natural", "Nat"), ("neareps", "Ne"), ("realflaky", "Rf")):
        m(f"dzFrr{A}", pct(dz[arm]["frr"] or 0)); m(f"dzFar{A}", pct(dz[arm]["far"], 2))
        m(f"dzBpl{A}", num(dz[arm]["bpa_loop"], 0)); m(f"dzFarWide{A}", pct(dz[arm]["far_wide"]["default04"], 2))
    m("dzFrrHighRf", pct(dz["realflaky"]["frr_high"] or 0))
    spn = load("results/design_checks.json")["split"]["total05"]["natural"]
    m("dlBplDeltaNat", num(spn["blocks_per_accept"] * (1 + 400 / spn["blocks_per_loop"]), 0))
    df = d["deflated"]
    for pm, P in (("0.25", "Quarter"), ("0.5", "Half")):
        m(f"dfFarLow{P}", pct(df[pm]["far"]["0.2"])); m(f"dfFarMid{P}", pct(df[pm]["far"]["0.5"]))
        m(f"dfFrrRf{P}", pct(df[pm]["realflaky"]["frr"] or 0)); m(f"dfBpaNat{P}", num(df[pm]["natural"]["blocks_per_accept"], 0))
        m(f"dfBpaRf{P}", num(df[pm]["realflaky"]["blocks_per_accept"], 0))
    from near_eps_checks import BIG_CAP
    m("bigCap", f"{BIG_CAP:,}".replace(",", "{,}"))
    # inverse of the supplement's result-to-script map: each paper table and figure with every script it draws on
    import re as _re
    sup = (GEN.parent / "supplement.tex").read_text()
    body = sup[sup.index("\\label{stab:map}"):]
    body = body[body.index("\\midrule"):body.index("\\end{tabular}")]
    inv = {}
    for rowtxt in body.split("\\\\"):
        refs = [("Table" if k == "tab" else "Figure", f"M-{k}:{n}") for k, n in _re.findall(r"\\ref\{M-(tab|fig):([A-Za-z]+)\}", rowtxt)]
        scripts = _re.findall(r"\\texttt\{([A-Za-z0-9_\\]+?)\.py\}", rowtxt)
        for kind, ref in refs:
            inv.setdefault((kind, ref), set()).update(scripts)
    rows = "".join(f"{kind}~\\ref{{{ref}}} & " + (", ".join(f"\\texttt{{{sc}.py}}" for sc in sorted(v)) or "drawn by hand") + "\\\\\n"
                   for (kind, ref), v in sorted(inv.items(), key=lambda kv: (kv[0][0] != "Table", kv[0][1])))
    write("tab_map_inverse.tex", "\\begin{tabular}{@{}lp{0.72\\linewidth}@{}}\n\\toprule\n\\textbf{In the paper} & \\textbf{Scripts}\\\\\n\\midrule\n"
          + rows + "\\bottomrule\n\\end{tabular}\n")
    # 5. latency
    m("latBfourNat", num(d["latency"]["natural"]["B4"]))
    m("latBfourRf", num(d["latency"]["realflaky"]["B4"]))
    # near-eps re-measurement per task (near_eps_checks.py 7, pooled) and the re-measured rates
    ne = load("results/near_eps_checks.json")["firmer"]["pooled"]
    T = {"filestack@neareps": "Filestack", "kanren@neareps": "Kanren", "pymeterbus@neareps": "Pymeterbus"}
    for n, K in (("B4 peeking Clopper-Pearson", "Bfour"), ("B5 naive confidence sequence", "Bfive"),
                 ("B6 SeqAccept, no refund", "Bsix"), ("SeqAccept", "Sa"), ("B1 stop at first pass", "Bone")):
        for t, TT in T.items():
            m(f"frTk{TT}{K}", pct(ne[n]["per_task"][t], 2))
            m(f"frTk{TT}{K}Lb", "$^\\ddagger$" if ne[n]["tasks_lower"][t] > 0.05 else "")
    loops = {t: sum(1 for p in Path("episodes_neareps").glob(f"{t}-*.json")) for t in T}
    for t, TT in T.items():
        m(f"frTk{TT}Loops", loops[t])
    truth = load("results/neareps_truth.json")
    rates = []
    for t, fs in truth.items():
        for f, v in fs.items():
            ep = next(load(p) for p in sorted(Path("episodes_neareps").glob(f"{t}-*.json")) if f in load(p)["fixes"])
            fx = ep["fixes"][f]
            allb = (fx["truth"] if len(fx["seeds"]) == len(fx["truth"]) else fx["truth"] + fx["pool"]) + v["extra"]
            r = sum(allb) / len(allb)
            if r > ep["p0"] + 0.05:
                rates.append(r - ep["p0"] - 0.05)
    rates.sort()
    m("frRateN", len(rates))
    m("frRateMed", pct(median(rates), 1))
    m("frRateMax", pct(rates[-1], 1))
    m("frRateWithinOne", sum(r <= 0.01 for r in rates))


def design_checks():
    """Design checks: results/design_checks.json (design_checks.py)."""
    d = load("results/design_checks.json")
    A = {"natural": "Nat", "neareps": "Ne", "realflaky": "Rf"}
    W = {"SeqAccept": "Sa", "SeqAccept, mixture wallet": "Mix", "SeqAccept, gamma 3/4": "Gam"}
    for arm, a in A.items():
        for n, w in W.items():
            v = d["wallets"][arm][n]
            m(f"dc{w}Far{a}", pct(v["far"], 2))
            m(f"dc{w}Frr{a}", pct(v["frr"] or 0, 2))
            m(f"dc{w}Bpl{a}", num(v["blocks_per_loop"], 0))
    S = {"SeqAccept": "Sa", "B6": "Bsix", "Harmonic": "Harm", "Harmonic + refund": "HarmRef"}
    for cap, c in (("shared", "Sh"), ("slice", "Sl")):
        for n, k in S.items():
            m(f"hr{c}LateForty{k}", pct(d["schedules"][cap]["late40"][n]))
            m(f"hr{c}LateTen{k}", pct(d["schedules"][cap]["late10"][n]))
            m(f"hr{c}NeBpl{k}", num(d["schedules"][cap]["neareps"][n]["blocks_per_loop"], 0))
    fp = d["fixed_power"]
    sa_bpa = d["wallets"]["natural"]["SeqAccept"]["blocks_per_accept"]
    for cap, c in (("15", "Fifteen"), ("100", "Hundred"), ("1000", "Thousand")):
        m(f"fpBpa{c}", num(fp[cap]["fixed"]["blocks_per_accept"], 0))
        m(f"fpRatio{c}", num(fp[cap]["fixed"]["blocks_per_accept"] / sa_bpa, 1))
        m(f"fpLateTen{c}", pct(fp[cap]["fixed"]["late10"]))
        m(f"fpLateForty{c}", pct(fp[cap]["fixed"]["late40"]))
        m(f"fpM{c}", fp[cap]["m"])
        m(f"fpNzeroRatio{c}", num(fp[cap]["fixed n0"]["blocks_per_accept"] / sa_bpa, 1))
    m("fpSaBpa", num(sa_bpa, 0))
    sp = d["split"]
    m("spDelta", f"{sp['total05']['delta']:.3f}")
    for arm, a in A.items():
        v = sp["total05"][arm]
        m(f"sp{a}Far", pct(v["far"], 2))
        m(f"sp{a}Frr", pct(v["frr"] or 0, 2))
        m(f"sp{a}Bpa", num(v["blocks_per_accept"], 0))
        m(f"sp{a}Worst", pct(v["worst_loop"]))
    m("spRfFrrHigh", pct(sp["total05"]["realflaky"]["frr_high"] or 0, 2))
    m("spNatRatio", num(sp["total05"]["natural"]["blocks_per_accept"] / sa_bpa, 1))
    t10 = sp["total10"]["neareps"]
    m("spTenNeFar", pct(t10["far"], 2))
    m("spTenNeWorst", pct(t10["worst_loop"]))
    m("spTenNeBpl", num(t10["blocks_per_loop"], 0))
    m("spTenNeBpa", num(t10["bpa_calib"], 0))
    nl = d["near_lower"]
    for n, k in (("B4 peeking Clopper-Pearson", "Bfour"), ("B5 naive confidence sequence", "Bfive"), ("SeqAccept", "Sa")):
        m(f"nlLoops{k}", nl[n]["loops_lower_above"])
        m(f"nlTasks{k}", nl[n]["tasks_lower_above"])
        m(f"nlTaskMax{k}", pct(max(v["far"] for v in nl[n]["tasks"].values()), 2))
        m(f"nlTaskMin{k}", pct(min(v["far"] for v in nl[n]["tasks"].values()), 2))
    lat = d["latency"]
    for arm, a in A.items():
        m(f"lat{a}", num(lat[arm]["minutes_at_8"]))
    ne = load("results/near_eps_checks.json")["neareps"]
    m("crBfourNe", num(ne["SeqAccept"]["bpa_calib"] / ne["B4 peeking Clopper-Pearson"]["bpa_calib"], 1))
    rf = load("results/extra.json")["realflaky"]["table"]
    m("crBfourRf", num(rf["SeqAccept"]["blocks_per_accept"] / rf["B4 peeking Clopper-Pearson"]["blocks_per_accept"], 1))
    ct = load("results/caps_adversary.json")["realflaky_tasks"]
    m("rfYamiNear", ct["yamicache@realflaky"]["near_broken"])


if __name__ == "__main__":
    GEN.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    res = load("results/results.json")
    episodes = main_episodes()
    assert len(episodes) == res["episodes"]
    benchmark(res, episodes)
    screening()
    main_table(res)
    figure_persistence(splits(res))
    sensitivity(res)
    worst_case()
    reuse()
    online()
    constants()
    extra(res)
    power_tuning()
    slice_caps()
    label_risk()
    ease()
    ease2()
    ease3()
    ease4()
    sens()
    checks()
    near_eps_checks()
    design_checks()
    deploy_checks()
    exec_order()
    (GEN / "numbers.tex").write_text(
        "% generated by intern1/impl/paper_numbers.py; do not edit\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
    print(f"{len(macros)} macros, tables and figures written to {GEN} and {FIGS}")
