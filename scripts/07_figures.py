"""Phase 3 figures (Fig. 2-5) from results/phase3_metrics*.json.

Usage: python scripts/07_figures.py            -> results/figures/fig*.{png,svg} (report style, with titles)
       python scripts/07_figures.py --poster   -> poster/figures/fig*.svg (large type, no titles, single-column sizes)
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
       "refusal+emotion": ("refusal + emotion", "#6a4ce2")}
RANDOM = "#b5b5b5"
COMPLIANT, REFUSED = "#ec4647", "#257de0"

REPORT = {"poster": False, "small": 9, "base": 11, "title": 11,
          "size": {"fig2": (7.2, 4.6), "fig3": (7.2, 4.0), "fig4": (7.6, 3.6), "fig5": (7.6, 3.8)}}
POSTER = {"poster": True, "small": 12.5, "base": 13.5, "title": 13.5,
          "size": {"fig2": (7.2, 4.3), "fig3": (4.6, 4.1), "fig4": (4.6, 5.6), "fig5": (4.6, 5.9)}}


def style_rc(S):
    plt.rcParams.update({"font.size": S["base"], "axes.titlesize": S["title"], "axes.labelsize": S["base"],
                         "xtick.labelsize": S["small"], "ytick.labelsize": S["small"], "legend.fontsize": S["small"],
                         "lines.linewidth": 2.0 if S["poster"] else 1.6})


def out(fig, name, S):
    if S["poster"]:
        d = ROOT / "poster/figures"
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.svg")
        fig.savefig(d / f"{name}.png")
        plt.close(fig)
        return [d / f"{name}.svg"]
    return save(fig, name)


def fig2(m, ks, src, S):
    fig, ax = plt.subplots(figsize=S["size"]["fig2"])
    w = 0.19
    fs = S["small"]
    for gi, k in enumerate(ks):
        e = m["monitors"][str(k)]
        r = e["random"]
        ax.bar(gi - 1.5 * w, r["median"], w * 0.9, color=RANDOM, zorder=2)
        ax.errorbar(gi - 1.5 * w, r["median"], yerr=[[r["median"] - r["p2.5_p97.5"][0]], [r["p2.5_p97.5"][1] - r["median"]]],
                    color=INK2, capsize=3, linewidth=1.2, zorder=3)
        ax.text(gi - 1.5 * w, r["p2.5_p97.5"][1] + 0.012, f"{r['median']:.2f}", ha="center", va="bottom", fontsize=fs, color=INK)
        for j, key in enumerate(("emotion", "refusal", "refusal+emotion")):
            v = e[key]
            xx = gi + (j - 0.5) * w
            ax.bar(xx, v["test_auc"], w * 0.9, color=MON[key][1], zorder=2)
            ax.errorbar(xx, v["test_auc"], yerr=[[v["test_auc"] - v["ci95"][0]], [v["ci95"][1] - v["test_auc"]]],
                        color=INK, capsize=3, linewidth=1.2, zorder=3)
            ax.text(xx, v["ci95"][1] + 0.012, f"{v['test_auc']:.2f}", ha="center", va="bottom", fontsize=fs, color=INK)
    ax.axhline(0.5, color=INK2, linewidth=0.8, zorder=1)
    ax.text(0.5, 0.505, "chance", color=INK2, fontsize=fs, ha="center", va="bottom")
    ax.set_xticks(range(len(ks)), [("before generation (k = 0)" if k == 0 else f"after {k} generated tokens") for k in ks])
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("test AUROC (95 % CI)")
    if not S["poster"]:
        ax.set_title(f"Predicting compliance on held-out behaviours ({src})")
    handles = [plt.Rectangle((0, 0), 1, 1, color=RANDOM)] + [plt.Rectangle((0, 0), 1, 1, color=MON[k][1]) for k in MON]
    ax.legend(handles, ["200 random 5-direction monitors (median, 95 % range)"] + [MON[k][0] for k in MON],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.grid(axis="x", visible=False)
    return out(fig, "fig2_auroc_monitors", S)


def fig3(m, ks, src, S):
    fig, ax = plt.subplots(figsize=S["size"]["fig3"])
    xs = np.arange(len(ks))
    rlo = [m["monitors"][str(k)]["random"]["p2.5_p97.5"][0] for k in ks]
    rhi = [m["monitors"][str(k)]["random"]["p2.5_p97.5"][1] for k in ks]
    rmed = [m["monitors"][str(k)]["random"]["median"] for k in ks]
    ax.fill_between(xs, rlo, rhi, color=RANDOM, alpha=0.35, linewidth=0, label="random (95 % range)")
    ax.plot(xs, rmed, color="#8a8a8a", linewidth=1.2)
    for j, (key, (lab, col)) in enumerate(MON.items()):
        a = np.array([m["monitors"][str(k)][key]["test_auc"] for k in ks])
        lo = np.array([m["monitors"][str(k)][key]["ci95"][0] for k in ks])
        hi = np.array([m["monitors"][str(k)][key]["ci95"][1] for k in ks])
        xx = xs + (j - 1) * 0.09
        ax.errorbar(xx, a, yerr=[a - lo, hi - a], color=col, linewidth=0, elinewidth=1.0, capsize=0, alpha=0.7)
        ax.plot(xx, a, color=col, marker="o", markersize=7 if S["poster"] else 6, markeredgecolor="white",
                markeredgewidth=1.5, label=lab)
    ax.axhline(0.5, color=INK2, linewidth=0.8)
    ax.set_xticks(xs, [str(k) for k in ks])
    ax.set_xlim(-0.3, len(ks) - 0.6)
    ax.set_xlabel("generated tokens seen (0 = prompt only)")
    ax.set_ylabel("test AUROC")
    ax.set_ylim(0.2, 1.0)
    if not S["poster"]:
        ax.set_title(f"Early warning: AUROC vs tokens seen ({src})")
    ax.legend(loc="lower right")
    return out(fig, "fig3_auroc_vs_tokens", S)


def fig4(m, src, S):
    tr = m["h1"]["trajectories"]
    if S["poster"]:
        fig, axes = plt.subplots(2, 1, figsize=S["size"]["fig4"], sharex=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=S["size"]["fig4"], sharex=True)
    for ax, name in zip(axes, ("desperate", "calm")):
        for lab, col in (("compliant", COMPLIANT), ("refused", REFUSED)):
            t = tr[name][lab]
            x = np.arange(len(t["mean"]))
            ax.fill_between(x, t["lo"], t["hi"], color=col, alpha=0.12, linewidth=0)
            ax.plot(x, t["mean"], color=col, label=f"{lab} (n = {t['n']})")
        ax.set_title(f"'{name}' vector, layer {m['h1']['layer_emotion']}", loc="left")
        ax.set_ylabel("projection (a.u.)")
        ax.axvline(0, color=INK2, linewidth=0.8)
    axes[-1].set_xlabel("token position (0 = end of prompt)")
    axes[0].legend(loc="best")
    if not S["poster"]:
        fig.suptitle(f"Emotion projections over the first 20 generated tokens ({src}; val + test behaviours)",
                     fontsize=11, color=INK)
    fig.tight_layout()
    return out(fig, "fig4_emotion_trajectories", S)


def fig5(m, m_pair, S):
    wr = m["wrappers"]
    order = ["neutral", "emotional_pressure", "roleplay"]
    labels = ["neutral", "emotional\npressure", "role-play"]
    fs = S["small"]
    if S["poster"]:
        fig, axes = plt.subplots(2, 1, figsize=S["size"]["fig5"])
    else:
        fig, axes = plt.subplots(1, 2, figsize=S["size"]["fig5"], gridspec_kw={"width_ratios": [4, 3]})
    ax = axes[0]
    v = [wr[w]["harmful_compliance"] for w in order]
    c = [wr[w]["ci95"] for w in order]
    labels_h = list(labels)
    if m_pair is not None:
        pw = m_pair["wrappers"]["pair"]
        v.append(pw["harmful_compliance"])
        c.append(pw["ci95"])
        labels_h.append("PAIR\n(JBB artifact)")
    v, c = np.array(v), np.array(c)
    x = np.arange(len(v))
    cols = [COMPLIANT] * 3 + ["#f4a3a3"] * (len(v) - 3)
    ax.bar(x, v, 0.55, color=cols, zorder=2)
    ax.errorbar(x, v, yerr=[v - c[:, 0], c[:, 1] - v], color=INK, capsize=3, linewidth=1.2, fmt="none", zorder=3)
    for xi, vi, hi in zip(x, v, c[:, 1]):
        ax.text(xi, hi + 0.01, f"{vi * 100:.1f} %", ha="center", va="bottom", fontsize=fs, color=INK)
    ax.set_xticks(x, labels_h)
    ax.set_ylim(0, min(1.0, float(c[:, 1].max()) + 0.1))
    ax.set_title("harmful requests: compliance", loc="left")
    ax.set_ylabel("rate (95 % CI)")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    v = np.array([wr[w]["benign_over_refusal"] for w in order])
    c = np.array([wr[w]["benign_ci95"] for w in order])
    x = np.arange(3)
    ax.bar(x, v, 0.55, color=REFUSED, zorder=2)
    ax.errorbar(x, v, yerr=[v - c[:, 0], c[:, 1] - v], color=INK, capsize=3, linewidth=1.2, fmt="none", zorder=3)
    for xi, vi, hi in zip(x, v, c[:, 1]):
        ax.text(xi, hi + 0.01, f"{vi * 100:.0f} %", ha="center", va="bottom", fontsize=fs, color=INK)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, min(1.0, float(c[:, 1].max()) + 0.12))
    ax.set_title("benign requests: over-refusal", loc="left")
    if S["poster"]:
        ax.set_ylabel("rate (95 % CI)")
    ax.grid(axis="x", visible=False)
    if not S["poster"]:
        fig.text(0.01, -0.02, f"n = {wr['neutral']['n']} responses per wrapper (100 behaviours x 5 samples)"
                 + (f"; PAIR: n = {m_pair['wrappers']['pair']['n']}" if m_pair else ""), fontsize=8, color=INK2)
    fig.tight_layout()
    return out(fig, "fig5_wrappers", S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poster", action="store_true")
    S = POSTER if ap.parse_args().poster else REPORT
    setup()
    style_rc(S)
    m_main = json.load(open(ROOT / "results/phase3_metrics.json"))
    pair_path = ROOT / "results/phase3_metrics_pair.json"
    m_pair = json.load(open(pair_path)) if pair_path.exists() else None
    m, src = (m_pair, "PAIR prompts, exploratory") if m_pair else (m_main, "fixed wrappers")
    ks = [int(k) for k in m["monitors"]]
    for f in (fig2(m, [0, 5], src, S), fig3(m, ks, src, S), fig4(m, src, S), fig5(m_main, m_pair, S)):
        print(f[0])


if __name__ == "__main__":
    main()
