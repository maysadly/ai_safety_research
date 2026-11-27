"""Figures for the A1 landscape poster (Phase 5 confirmatory data), sized for its columns.

Usage: python scripts/13_a1_figures.py
Outputs: poster/a1/figures/*.svg (+ .png copies in results/figures/)
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.plots import INK, INK2, setup  # noqa: E402
from src.utils import ROOT  # noqa: E402
from src.vectors import wilson_ci  # noqa: E402

OUT = ROOT / "poster/a1/figures"
MON = {"emotion": ("emotion vectors", "#f47230"), "refusal": ("refusal direction", "#00b894"),
       "refusal+emotion": ("refusal + emotion", "#6a4ce2"), "text_tfidf": ("text of first k tokens", "#4d5868")}
RANDOM = "#b5b5b5"
COMPLIED, REFUSED = "#ec4647", "#257de0"
MODEL_COL = {"qwen": "#2b3a55", "mistral": "#9aa7b8"}
NAMES = {"qwen": "Qwen2.5-Omni-7B", "mistral": "Mistral-7B-Instruct"}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(ROOT / "results/figures" / f"{name}.png")
    plt.close(fig)
    print(OUT / f"{name}.svg")


def fig_auroc(M):
    fig, axes = plt.subplots(2, 1, figsize=(5.4, 3.95), sharex=True)
    for ax, m in zip(axes, ("qwen", "mistral")):
        mon = M[m]["monitors"]
        ks = [int(k) for k in mon]
        xs = np.arange(len(ks))
        ax.fill_between(xs, [mon[str(k)]["random"]["p2.5_p97.5"][0] for k in ks],
                        [mon[str(k)]["random"]["p2.5_p97.5"][1] for k in ks], color=RANDOM, alpha=0.4, linewidth=0,
                        label="200 random 5-direction monitors (95 % range)")
        for j, (name, (lab, col)) in enumerate(MON.items()):
            pts = [(i, mon[str(k)][name]) for i, k in enumerate(ks) if name in mon[str(k)]]
            x = np.array([p[0] for p in pts]) + (j - 1.5) * 0.06
            a = np.array([p[1]["auc"] for p in pts])
            lo = np.array([p[1]["ci95"][0] for p in pts])
            hi = np.array([p[1]["ci95"][1] for p in pts])
            ax.errorbar(x, a, yerr=[a - lo, hi - a], color=col, linewidth=0, elinewidth=1.0, alpha=0.8)
            ax.plot(x, a, color=col, marker="s" if name == "text_tfidf" else "o", markersize=5.5,
                    markeredgecolor="white", markeredgewidth=1.2, label=lab)
        ax.axhline(0.5, color=INK2, linewidth=0.8)
        n = M[m]["n"]
        ax.set_title(f"{NAMES[m]}: {n['compliant']:,} of {n['harmful']:,} harmful answers complied", loc="left", fontsize=11)
        ax.set_ylabel("AUROC (5-fold CV)")
        ax.set_ylim(0.45, 1.0)
        ax.set_xticks(xs, [str(k) for k in ks])
    axes[-1].set_xlabel("generated tokens seen by the monitor (0 = prompt only)")
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8.6, handlelength=1.4, columnspacing=1.0, frameon=False)
    save(fig, "a1_fig2_auroc")


def fig_traj(T):
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.3))
    for ax, m in zip(axes, ("qwen", "mistral")):
        for lab, col in (("complied", COMPLIED), ("refused", REFUSED)):
            t = T[m]["desperate"][lab]
            x = np.arange(len(t["mean"]))
            ax.fill_between(x, t["lo"], t["hi"], color=col, alpha=0.15, linewidth=0)
            ax.plot(x, t["mean"], color=col, label=f"{lab} (n = {t['n']:,})")
        ax.axvline(0, color=INK2, linewidth=0.8)
        n = T[m]["desperate"]
        ax.set_title(f"{NAMES[m]} (L{T[m]['layer']}; {n['complied']['n']:,} vs {n['refused']['n']:,})", loc="left", fontsize=9.6)
        ax.set_xlabel("token (0 = end of prompt)")
    axes[0].set_ylabel("'desperate' projection (a.u.)")
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.legend(h, ["complied", "refused"], loc="upper center", ncol=2, fontsize=8.8, frameon=False)
    save(fig, "a1_fig3_trajectories")


def rates(tag):
    lab = list(csv.DictReader(open(ROOT / f"data/labels_{tag}.csv")))
    fam = {"neutral": "plain request", "emotional_pressure": "emotional pressure", "roleplay": "role-play",
           "hb_direct": "HarmBench direct", "hb_human": "human jailbreak", "pair": "PAIR", "prs": "random-search"}
    harm, ben = defaultdict(list), defaultdict(list)
    for l in lab:
        w = l["wrapper"]
        key = "pair" if w.startswith("pair") else "prs" if w.startswith("prs") else w
        (harm if l["kind"] == "harmful" else ben)[fam[key]].append(int(l["compliance"]))
    return harm, ben


def fig_safety():
    order = ["plain request", "emotional pressure", "role-play", "HarmBench direct", "human jailbreak", "PAIR", "random-search"]
    fig, ax = plt.subplots(figsize=(4.3, 3.38))
    y = np.arange(len(order))
    for j, m in enumerate(("qwen", "mistral")):
        harm, _ = rates(f"p5_{m}")
        v = np.array([np.mean(harm[f]) for f in order])
        ci = np.array([wilson_ci(int(sum(harm[f])), len(harm[f])) for f in order])
        yy = y + (j - 0.5) * 0.38
        ax.barh(yy, v, 0.34, color=MODEL_COL[m], label=NAMES[m], zorder=2)
        ax.errorbar(v, yy, xerr=[v - ci[:, 0], ci[:, 1] - v], fmt="none", color=INK, linewidth=0.9, capsize=2, zorder=3)
        for vi, yi, hi in zip(v, yy, ci[:, 1]):
            ax.text(hi + 0.015, yi, f"{vi * 100:.0f} %", va="center", fontsize=8.3, color=INK)
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("harmful compliance rate (attack success), 95 % CI")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=8.6, frameon=False)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "a1_fig4_safety")

def fig_steer():
    conds = [("base", "no steering"), ("desp_minus_4", "\u2212desperate (4 SD)"), ("desp_minus_12", "\u2212desperate (12 SD)"),
             ("calm_plus_12", "+calm (12 SD)"), ("desp_plus_12", "+desperate (12 SD)"), ("rand_minus_12", "random dir. (12 SD)")]
    cols = {"base": "#8a8a8a", "desp_minus_4": "#f7b3cf", "desp_minus_12": "#ee7fac", "calm_plus_12": "#f2a600",
            "desp_plus_12": "#b4235a", "rand_minus_12": "#cfcfcf"}
    fig, axes = plt.subplots(2, 2, figsize=(4.3, 5.0), gridspec_kw={"width_ratios": [1.3, 1]})
    for r, m in enumerate(("qwen", "mistral")):
        d = json.load(open(ROOT / f"results/phase6_steer_{m}.json"))
        for c_i, (key, title, xmax) in enumerate((("over_refusal:emotional_pressure", "over-refusal", 0.62),
                                                   ("harmful_compliance", "harmful compliance", 0.45))):
            ax = axes[r, c_i]
            y = np.arange(len(conds))
            for i, (c, lab) in enumerate(conds):
                if key == "harmful_compliance":
                    a, b = d["rates"][c]["harmful_compliance:neutral"], d["rates"][c]["harmful_compliance:emotional_pressure"]
                    v = (a["rate"] * a["n"] + b["rate"] * b["n"]) / (a["n"] + b["n"])
                    k, n = round(v * (a["n"] + b["n"])), a["n"] + b["n"]
                    lo, hi = wilson_ci(k, n)
                else:
                    v = d["rates"][c][key]["rate"]
                    lo, hi = d["rates"][c][key]["ci95"]
                ax.barh(i, v, 0.7, color=cols[c], zorder=2)
                ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt="none", color=INK, linewidth=0.9, capsize=2, zorder=3)
                ax.text(hi + 0.012, i, f"{v * 100:.0f} %", va="center", fontsize=8, color=INK)
            ax.set_yticks(y, [lab for _, lab in conds] if c_i == 0 else [])
            ax.invert_yaxis()
            ax.set_xlim(0, xmax)
            ax.grid(axis="y", visible=False)
            short = "Qwen" if m == "qwen" else "Mistral"
            ax.set_title(f"{short}: {title}" if c_i == 0 else title, fontsize=9.4, loc="left")
            if r == 1:
                ax.set_xlabel("rate (95 % CI)")
    fig.tight_layout(h_pad=1.4)
    save(fig, "a1_fig5_steering")


def main():
    setup()
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
                         "legend.fontsize": 9, "lines.linewidth": 1.8})
    M = {m: json.load(open(ROOT / f"results/phase5_metrics_{m}.json")) for m in ("qwen", "mistral")}
    fig_auroc(M)
    fig_traj(json.load(open(ROOT / "results/phase5_trajectories.json")))
    fig_safety()
    fig_steer()


if __name__ == "__main__":
    main()
