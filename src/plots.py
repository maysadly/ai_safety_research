"""One figure style for the whole project (poster + reports). Palette: validated categorical
order (colour-blind safe on adjacent pairs); identity is never colour-only (legend / labels)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.utils import ROOT  # noqa: E402

SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
INK, INK2, GRID, AXIS = "#191919", "#666666", "#e7e7e7", "#c2c2c2"
EMOTION_COLORS = {"fear": SERIES[0], "joy": SERIES[1], "sadness": SERIES[2], "calm": SERIES[3],
                  "desperate": SERIES[4]}


def setup():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Seravek", "Helvetica", "Arial", "DejaVu Sans"],  # Seravek = poster face
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.labelcolor": INK,
        "svg.fonttype": "none", "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def save(fig, name: str, outdir: str = "results/figures") -> list[Path]:
    d = ROOT / outdir
    d.mkdir(parents=True, exist_ok=True)
    paths = [d / f"{name}.png", d / f"{name}.svg"]
    for p in paths:
        fig.savefig(p)
    plt.close(fig)
    return paths
