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
    fig, axes = plt.subplots(2, 1, figsize=(5.4, 4.4), sharex=True)
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
    fig, axes = plt.subplots(2, 1, figsize=(4.3, 5.75), gridspec_kw={"height_ratios": [7, 3.3]})
    ax = axes[0]
    y = np.arange(len(order))
    out = {}
    for j, m in enumerate(("qwen", "mistral")):
        harm, ben = rates(f"p5_{m}")
        out[m] = (harm, ben)
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
    ax = axes[1]
    w = ["plain request", "emotional pressure", "role-play"]
    y = np.arange(len(w))
    for j, m in enumerate(("qwen", "mistral")):
        ben = out[m][1]
        v = np.array([1 - np.mean(ben[f]) for f in w])
        ci = np.array([wilson_ci(int(len(ben[f]) - sum(ben[f])), len(ben[f])) for f in w])
        yy = y + (j - 0.5) * 0.38
        ax.barh(yy, v, 0.34, color=MODEL_COL[m], zorder=2)
        ax.errorbar(v, yy, xerr=[v - ci[:, 0], ci[:, 1] - v], fmt="none", color=INK, linewidth=0.9, capsize=2, zorder=3)
        for vi, yi, hi in zip(v, yy, ci[:, 1]):
            ax.text(hi + 0.01, yi, f"{vi * 100:.0f} %", va="center", fontsize=8.3, color=INK)
    ax.set_yticks(y, w)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.66)
    ax.set_xlabel("over-refusal of benign requests, 95 % CI")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "a1_fig4_safety")


def main():
    setup()
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
                         "legend.fontsize": 9, "lines.linewidth": 1.8})
    M = {m: json.load(open(ROOT / f"results/phase5_metrics_{m}.json")) for m in ("qwen", "mistral")}
    fig_auroc(M)
    fig_traj(json.load(open(ROOT / "results/phase5_trajectories.json")))
    fig_safety()


if __name__ == "__main__":
    main()
