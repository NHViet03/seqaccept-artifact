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
    m("capBlocks", d["max_blocks"])
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
    assert rf["table"]["SeqAccept"]["far"] == rf["table"]["B6 SeqAccept, no refund"]["far"], "text: SeqAccept and B6 share one FAR"
    hit = rf["table"]["SeqAccept"]["far_upper"]
    m("rfSaEpHits", word_int(hit["episodes"][0]))
    m("rfWithBroken", hit["episodes"][1])
    m("rfSaRepoHits", word_int(hit["repos"][0]))
    assert rf["seqaccept_false_accept_bugs"] and {b.split("_b")[0] for b in rf["seqaccept_false_accept_bugs"]} == {"yamicache"}, \
        rf["seqaccept_false_accept_bugs"]  # the text names yamicache
    m("rfYamiPzero", f"{rf['p0']['yamicache']:.2f}")
    m("rfRepos", rf["repos"])
    m("rfRepoHitsBound", rf["table"]["SeqAccept"]["far_upper"]["repos"][1])
    m("rfFarLoSa", pct(rf["table"]["SeqAccept"]["far_ci"][0], 2))
    m("rfFarHiSa", pct(rf["table"]["SeqAccept"]["far_ci"][1], 2))
    m("rfFrrLoSa", pct(rf["frr_ci_seqaccept"][0]))
    m("rfFrrHiSa", pct(rf["frr_ci_seqaccept"][1]))
    m("rfFarLoBfour", pct(rf["table"]["B4 peeking Clopper-Pearson"]["far_ci"][0], 2))
    m("rfFarHiBfour", pct(rf["table"]["B4 peeking Clopper-Pearson"]["far_ci"][1]))
    m("rfPzeroSeMax", f"{max(rf['p0_se'].values()):.3f}")
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
            (f"Rerun-{d['k_match']} (all green)", main[rr], {k: {"table": v} for k, v in late.items()}, rr, worst, None),
            ("\\sa{}", t["SeqAccept"], res["late_streams"], "SeqAccept", wsa, st["alt-0.10"]["far"]["SeqAccept"])]
    body = ""
    for label_, s_, late_, key_, worst_, stress_ in rows:
        la = " / ".join(pct(late_[str(k)]["table"][key_]["late_accept"], 0) for k in (10, 20, 40))
        wf = " / ".join(pct(worst_[(0.051, n)][key_]) for n in (15, 50))
        body += (f"{label_} & {num(s_['blocks_per_accept'])} & {pct(s_['frr'])} & {la} & "
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
          "\\begin{tabular}{@{}lrrcrcr@{}}\n\\toprule\n"
          "& \\multicolumn{2}{c}{\\textbf{Recorded loops}} & \\multicolumn{2}{c}{\\textbf{Late streams}} & "
          "\\textbf{Worst case} & \\textbf{Mixed}\\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
          "\\textbf{Rule} & \\textbf{Blk/acc.} & \\textbf{FRR \\%} & \\textbf{Late acc.\\ \\%} & "
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
    assert rm["far"] == 0 and rm["frr"] == 0, "text: the mixture wallet has no false acceptance and no lost patch"
    for r in d["sweep"]:
        tag = {(0.25, 20): "GammaLow", (0.75, 20): "GammaHigh", (0.5, 10): "RhoLow", (0.5, 50): "RhoHigh", (0.5, 20): "Default"}[(r["gamma"], r["rho"])]
        m(f"sweepBpa{tag}", num(r["recorded"]["blocks_per_accept"]))
        m(f"sweepFar{tag}", pct(r["recorded"]["far"], 2))
        m(f"sweepLateForty{tag}", pct(r["late"]["40"]["late_accept"]))
        assert r["recorded"]["frr"] == 0, "text: no setting loses a correct patch on the recorded loops"
    m("sweepLateMin", pct(min(min(r["late"]["10"]["late_accept"], r["late"]["40"]["late_accept"]) for r in d["sweep"])))


def review2():
    """Second external review: results/review2.json (review2.py)."""
    d = load("results/review2.json")
    m("shortPool", d["short_pool"])
    sa = d["pool"]["SeqAccept"]
    assert sa["full"]["far"] == sa["short"]["far"] == 0 and sa["full"]["frr"] == sa["short"]["frr"] == 0, "text: unchanged"
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
    review2()
    exec_order()
    (GEN / "numbers.tex").write_text(
        "% generated by intern1/impl/paper_numbers.py; do not edit\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
    print(f"{len(macros)} macros, tables and figures written to {GEN} and {FIGS}")
