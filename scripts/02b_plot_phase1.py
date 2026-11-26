"""Phase 1 diagnostic figures from results/phase1_vectors.json (runs locally, no GPU).

Usage: python scripts/02b_plot_phase1.py --config configs/phase1.yaml
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.plots import EMOTION_COLORS, INK2, SERIES, save, setup  # noqa: E402
from src.utils import ROOT, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    cfg = load_config(ap.parse_args().config)
    r = json.load(open(ROOT / cfg["out"]["results_json"]))
    setup()
    emotions = r["emotions"]
    h = r["heldout"]
    L = np.arange(len(h["acc_raw"]))

    # 1. held-out emotion classification accuracy per layer (chosen variant)
    v = h["variant"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ci = np.array(h["loto_ci95_denoised"]) if v == "denoised" else None
    if ci is not None:
        ax.fill_between(L, ci[:, 0], ci[:, 1], color=SERIES[0], alpha=0.10, linewidth=0)
    ax.plot(L, h[f"loto_acc_{v}"], color=SERIES[0], label=f"leave-one-topic-out (n = {h['loto_n']})")
    ax.plot(L, h[f"acc_{v}"], color=SERIES[1], label=f"fixed 3-topic hold-out (n = {h['n_heldout_stories']})")
    ax.axhline(h["chance"], color=INK2, linewidth=0.8)
    ax.text(0.5, h["chance"] + 0.02, "chance (5 emotions)", color=INK2, ha="left", va="bottom", fontsize=9)
    ax.set(xlabel="layer (0 = embeddings)", ylabel="accuracy", ylim=(0, 1.02),
           title=f"Emotion vectors classify stories on unseen topics ({v}; band = 95 % CI)")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "p1_heldout_accuracy")

    # 2. refusal direction: val AUROC harmful vs benign
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(L, r["refusal"]["val_auroc_per_layer"], color=SERIES[0])
    ax.axhline(0.5, color=INK2, linewidth=0.8)
    ax.set(xlabel="layer", ylabel="AUROC", ylim=(0.4, 1.02),
           title="Refusal direction separates val harmful vs benign prompts "
                 f"(n = {r['refusal']['n_check']['harmful']} + {r['refusal']['n_check']['benign']})")
    save(fig, "p1_refusal_val_auroc")

    # 3. cosine between emotion vectors and refusal direction
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for e in emotions:
        ax.plot(L, r["refusal"]["cos_with_emotions"][e], color=EMOTION_COLORS[e], label=e)
    rnd = np.array(r["random"]["max_abs_cos_with_refusal_per_layer"])
    ax.fill_between(L, -rnd, rnd, color=INK2, alpha=0.12, linewidth=0, label="random (max |cos|, n = 200)")
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set(xlabel="layer", ylabel="cosine", title="Cosine between emotion vectors and the refusal direction")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    save(fig, "p1_cos_emotion_refusal")

    # 4. topic sensitivity: harmful vs benign val prompts (the "dangerous topic" trap)
    ts = r["topic_sensitivity_val"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    p95 = np.array(ts["random_abs_d_p95_per_layer"])
    ax.fill_between(L, -p95, p95, color=INK2, alpha=0.12, linewidth=0, label="random 95 % band")
    for e in emotions:
        ax.plot(L, ts["emotions"][e], color=EMOTION_COLORS[e], label=e)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set(xlabel="layer", ylabel="Cohen's d (harmful − benign)",
           title="Emotion projections at the prompt: harmful vs benign topics (val)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    save(fig, "p1_topic_sensitivity")
    print("figures written to results/figures/")


if __name__ == "__main__":
    main()
