"""Figures for the A1 portrait poster (v2): sized for 672 px columns / a 1394 px two-column band so that
text renders at about 18-21 pt on paper. Same data and style as the report figures (Phase 1-3 results).

Usage: python poster/make_a1_figures.py   -> poster/figures_a1/*.svg
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.plots import INK, INK2, setup  # noqa: E402

MON = {"emotion": ("emotion vectors", "#f47230"), "refusal": ("refusal (baseline)", "#00b894"),
       "refusal+emotion": ("refusal + emotion (improved)", "#6a4ce2")}
RANDOM = "#b5b5b5"
COMPLIANT, REFUSED = "#ec4647", "#257de0"
OUT = HERE / "figures_a1"


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print(OUT / f"{name}.svg")


def fig3_wide(m):
    """AUROC vs generated tokens seen; random-direction band; 95 % CI bars."""
    mon = m["monitors"]
    ks = [int(k) for k in mon]
    xs = np.arange(len(ks))
    fig, ax = plt.subplots(figsize=(7.6, 2.3))
    lo = [mon[str(k)]["random"]["p2.5_p97.5"][0] for k in ks]
    hi = [mon[str(k)]["random"]["p2.5_p97.5"][1] for k in ks]
    ax.fill_between(xs, lo, hi, color=RANDOM, alpha=0.35, linewidth=0, label="random directions (95 % range)")
    ax.plot(xs, [mon[str(k)]["random"]["median"] for k in ks], color="#8a8a8a", linewidth=1.3)
    for j, (key, (lab, col)) in enumerate(MON.items()):
        a = np.array([mon[str(k)][key]["test_auc"] for k in ks])
        l_ = np.array([mon[str(k)][key]["ci95"][0] for k in ks])
        h_ = np.array([mon[str(k)][key]["ci95"][1] for k in ks])
        x = xs + (j - 1) * 0.08
        ax.errorbar(x, a, yerr=[a - l_, h_ - a], color=col, linewidth=0, elinewidth=1.1, alpha=0.75)
        ax.plot(x, a, color=col, marker="o", markersize=6.5, markeredgecolor="white", markeredgewidth=1.4, label=lab)
    ax.axhline(0.5, color=INK2, linewidth=0.9)
    ax.text(1.5, 0.51, "chance (0.5)", color=INK2, fontsize=10, ha="center", va="bottom")
    ax.set_xticks(xs, [str(k) for k in ks])
    ax.set_xlim(-0.3, len(ks) - 0.65)
    ax.set_ylim(0.2, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("generated tokens seen by the monitor, k (0 = prompt only)")
    ax.set_ylabel("test AUROC (unitless)")
    h, l = ax.get_legend_handles_labels()
    l = [x.replace("emotion vectors", "emotion").replace("random directions (95 % range)", "random (95 % range)") for x in l]
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=10, handlelength=1.3,
               columnspacing=1.0, borderaxespad=0.1)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, "fig3_auroc_vs_tokens")


def fig4_stacked(m):
    tr = m["h1"]["trajectories"]
    fig, axes = plt.subplots(2, 1, figsize=(4.2, 3.7), sharex=True)
    for ax, name in zip(axes, ("desperate", "calm")):
        for lab, col in (("complied", COMPLIANT), ("refused", REFUSED)):
            key = "compliant" if lab == "complied" else "refused"
            t = tr[name][key]
            x = np.arange(len(t["mean"]))
            ax.fill_between(x, t["lo"], t["hi"], color=col, alpha=0.13, linewidth=0)
            ax.plot(x, t["mean"], color=col, label=f"{lab} (n = {t['n']})")
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_title(f"'{name}' vector, layer {m['h1']['layer_emotion']}", loc="left", fontsize=11.5)
        ax.set_ylabel("projection (a.u.)")
    axes[1].set_xlabel("token position (0 = end of prompt; 1–20 = answer)")
    axes[1].legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    save(fig, "fig4_emotion_trajectories")


def fig5_combined(m_main, m_pair):
    """Harmful compliance and benign over-refusal per prompt condition on one percent axis."""
    wr = m_main["wrappers"]
    order = [("neutral", "neutral"), ("emotional_pressure", "emotional\npressure"), ("roleplay", "role-play")]
    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    w = 0.36
    for i, (key, _) in enumerate(order):
        for dx, val, ci, col in ((-w / 2, wr[key]["harmful_compliance"], wr[key]["ci95"], COMPLIANT),
                                 (w / 2, wr[key]["benign_over_refusal"], wr[key]["benign_ci95"], REFUSED)):
            ax.bar(i + dx, val * 100, w * 0.92, color=col, zorder=2)
            ax.errorbar(i + dx, val * 100, yerr=[[100 * (val - ci[0])], [100 * (ci[1] - val)]], color=INK, capsize=2.5,
                        linewidth=1.0, zorder=3)
            ax.text(i + dx, ci[1] * 100 + 1.5, f"{val * 100:.0f}" if val >= 0.05 else f"{val * 100:.1f}",
                    ha="center", va="bottom", fontsize=9.5, color=INK)
    pw = m_pair["wrappers"]["pair"]
    i = len(order)
    ax.bar(i - w / 2, pw["harmful_compliance"] * 100, w * 0.92, color="#f4a3a3", zorder=2)
    ax.errorbar(i - w / 2, pw["harmful_compliance"] * 100,
                yerr=[[100 * (pw["harmful_compliance"] - pw["ci95"][0])], [100 * (pw["ci95"][1] - pw["harmful_compliance"])]],
                color=INK, capsize=2.5, linewidth=1.0, zorder=3)
    ax.text(i - w / 2, pw["ci95"][1] * 100 + 1.5, f"{pw['harmful_compliance'] * 100:.0f}", ha="center", va="bottom",
            fontsize=9.5, color=INK)
    ax.set_xticks(range(len(order) + 1), [lab for _, lab in order] + ["PAIR\njailbreak"])
    ax.set_ylabel("rate (%, 95 % CI)")
    ax.set_xlabel("prompt condition")
    ax.set_ylim(0, 62)
    ax.grid(axis="x", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COMPLIANT), plt.Rectangle((0, 0), 1, 1, color=REFUSED)]
    ax.legend(handles, ["harmful request: complied", "benign request: refused"], loc="lower left",
              bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=9.5, handlelength=1.0, columnspacing=1.0, borderaxespad=0.2)
    fig.tight_layout()
    save(fig, "fig5_safety_rates")


def main():
    setup()
    plt.rcParams.update({"font.size": 11.5, "axes.labelsize": 11.5, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
                         "legend.fontsize": 10, "axes.titlesize": 11.5, "lines.linewidth": 2.0})
    m_main = json.load(open(ROOT / "results/phase3_metrics.json"))
    m_pair = json.load(open(ROOT / "results/phase3_metrics_pair.json"))
    fig3_wide(m_pair)
    fig4_stacked(m_pair)
    fig5_combined(m_main, m_pair)


if __name__ == "__main__":
    main()
