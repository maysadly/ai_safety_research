"""Phase 5 figures from results/phase5_metrics_{qwen,mistral}.json.

Usage: python scripts/11_phase5_figures.py [--poster]
Outputs: results/figures/fig6_p5_auroc_vs_tokens, fig7_p5_delta_forest, fig8_p5_within_prompt (PNG + SVG);
         with --poster, SVGs in poster/figures/.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.plots import INK, INK2, save, setup  # noqa: E402
from src.utils import ROOT  # noqa: E402

MON = {"emotion": ("emotion vectors", "#f47230"), "refusal": ("refusal direction", "#00b894"),
       "refusal+emotion": ("refusal + emotion", "#6a4ce2"), "text_tfidf": ("text of first k tokens", "#4d5868")}
RANDOM = "#b5b5b5"
MODELS = [("qwen", "Qwen2.5-Omni-7B"), ("mistral", "Mistral-7B-Instruct-v0.3")]


def out(fig, name, poster):
    if poster:
        d = ROOT / "poster/figures"
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.svg")
        plt.close(fig)
        return d / f"{name}.svg"
    return save(fig, name)[0]


def fig6(M, poster):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4) if poster else (9.0, 3.6), sharey=True)
    for ax, (key, title) in zip(axes, MODELS):
        m = M[key]["monitors"]
        ks = [int(k) for k in m]
        xs = np.arange(len(ks))
        ax.fill_between(xs, [m[str(k)]["random"]["p2.5_p97.5"][0] for k in ks],
                        [m[str(k)]["random"]["p2.5_p97.5"][1] for k in ks], color=RANDOM, alpha=0.35, linewidth=0,
                        label="random directions (95 % range)")
        for j, (name, (lab, col)) in enumerate(MON.items()):
            pts = [(i, m[str(k)][name]) for i, k in enumerate(ks) if name in m[str(k)]]
            x = np.array([p[0] for p in pts]) + (j - 1.5) * 0.07
            a = np.array([p[1]["auc"] for p in pts])
            lo = np.array([p[1]["ci95"][0] for p in pts])
            hi = np.array([p[1]["ci95"][1] for p in pts])
            ax.errorbar(x, a, yerr=[a - lo, hi - a], color=col, linewidth=0, elinewidth=1.0, alpha=0.7)
            ax.plot(x, a, color=col, marker="s" if name == "text_tfidf" else "o", markersize=5.5,
                    markeredgecolor="white", markeredgewidth=1.2, label=lab)
        ax.axhline(0.5, color=INK2, linewidth=0.8)
        ax.set_xticks(xs, [str(k) for k in ks])
        ax.set_xlabel("generated tokens seen (0 = prompt only)")
        n = M[key]["n"]
        ax.set_title(f"{title}  ({n['compliant']} / {n['harmful']} complied, {n['behaviours']} behaviours)",
                     loc="left", fontsize=10 if not poster else 11.5)
    axes[0].set_ylabel("cross-validated AUROC")
    axes[0].set_ylim(0.3, 1.0)
    axes[1].legend(loc="lower right", fontsize=8.5 if not poster else 10)
    fig.tight_layout()
    return out(fig, "fig6_p5_auroc_vs_tokens", poster)


def fig7(M, margin, poster):
    fig, ax = plt.subplots(figsize=(4.6, 3.6) if poster else (6.0, 3.4))
    ks = [int(k) for k in M["qwen"]["monitors"]]
    cols = {"qwen": "#257de0", "mistral": "#ec4647"}
    ax.axvspan(-margin, margin, color=RANDOM, alpha=0.3, linewidth=0, label=f"equivalence margin ±{margin}")
    ax.axvline(0, color=INK2, linewidth=0.8)
    for j, (key, title) in enumerate(MODELS):
        for i, k in enumerate(ks):
            d = M[key]["monitors"][str(k)]["delta_refusal+emotion_minus_refusal"]
            yv = i + (j - 0.5) * 0.28
            ax.plot(d["ci90"], [yv, yv], color=cols[key], linewidth=2.2, solid_capstyle="butt")
            ax.plot(d["point"], yv, "o", color=cols[key], markersize=6, markeredgecolor="white", markeredgewidth=1.2,
                    label=title if i == 0 else None)
    ax.set_yticks(range(len(ks)), [f"k = {k}" for k in ks])
    ax.invert_yaxis()
    ax.set_xlabel("Δ AUROC: (refusal + emotion) − refusal, 90 % CI")
    ax.legend(loc="lower right", fontsize=8.5 if not poster else 10)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return out(fig, "fig7_p5_delta_forest", poster)


def fig8(M, poster):
    fig, ax = plt.subplots(figsize=(4.6, 3.2) if poster else (5.6, 3.0))
    labels, vals, los, his, cols, rnd = [], [], [], [], [], []
    for key, title in MODELS:
        c4 = M[key]["C4_within_prompt"]
        for e, col in (("desperate", "#ee7fac"), ("calm", "#f2a600")):
            labels.append(f"{title.split('-')[0]}\n{e}")
            vals.append(c4[e]["d_z"])
            los.append(c4[e]["d_z_ci95"][0])
            his.append(c4[e]["d_z_ci95"][1])
            cols.append(col)
            rnd.append(c4["random_abs_dz_p95"])
    x = np.arange(len(vals))
    for xi, r in zip(x, rnd):
        ax.fill_between([xi - 0.35, xi + 0.35], -r, r, color=RANDOM, alpha=0.35, linewidth=0)
    ax.bar(x, vals, 0.55, color=cols, zorder=2)
    v = np.array(vals)
    ax.errorbar(x, v, yerr=[v - np.array(los), np.array(his) - v], fmt="none", color=INK, capsize=3, linewidth=1.1, zorder=3)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("within-prompt d_z\n(complied − refused), tokens 1–5")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return out(fig, "fig8_p5_within_prompt", poster)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poster", action="store_true")
    poster = ap.parse_args().poster
    setup()
    if poster:
        plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
                             "legend.fontsize": 10, "axes.titlesize": 12})
    M = {k: json.load(open(ROOT / f"results/phase5_metrics_{k}.json")) for k, _ in MODELS}
    margin = 0.03
    for f in (fig6(M, poster), fig7(M, margin, poster), fig8(M, poster)):
        print(f)


if __name__ == "__main__":
    main()
