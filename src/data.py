"""JailbreakBench behaviours, fixed splits and prompt wrappers."""
from __future__ import annotations

import csv
import random
import urllib.request
from collections import defaultdict

import yaml

from src.utils import ROOT

FILES = {"harmful": "harmful-behaviors.csv", "benign": "benign-behaviors.csv"}


def ensure_jbb(cfg: dict) -> None:
    """Download the two JBB-Behaviors CSVs into the git-ignored raw dir if missing."""
    d = ROOT / cfg["data"]["jbb_dir"]
    d.mkdir(parents=True, exist_ok=True)
    for fname in FILES.values():
        if not (d / fname).exists():
            urllib.request.urlretrieve(f"{cfg['data']['jbb_url']}/{fname}", d / fname)


def load_behaviors(cfg: dict, kind: str) -> list[dict]:
    """kind in {'harmful', 'benign'}. Returns rows with int Index."""
    ensure_jbb(cfg)
    with open(ROOT / cfg["data"]["jbb_dir"] / FILES[kind]) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["Index"] = int(r["Index"])
    return rows


def make_splits(cfg: dict, seed: int) -> list[dict]:
    """Split behaviour pairs (same Index in harmful and benign files) by Index,
    stratified by the harmful Category: 5 train / 2 val / 3 test per category."""
    harmful = load_behaviors(cfg, "harmful")
    by_cat = defaultdict(list)
    for r in harmful:
        by_cat[r["Category"]].append(r["Index"])
    counts = cfg["data"]["split_counts_per_category"]
    rng = random.Random(seed)
    out = []
    for cat in sorted(by_cat):
        idx = sorted(by_cat[cat])
        rng.shuffle(idx)
        names = ["train"] * counts["train"] + ["val"] * counts["val"] + ["test"] * counts["test"]
        assert len(names) == len(idx), (cat, len(idx))
        out += [{"Index": i, "Category": cat, "split": s} for i, s in zip(idx, names)]
    return sorted(out, key=lambda r: r["Index"])


def load_splits(cfg: dict) -> dict[int, str]:
    with open(ROOT / cfg["data"]["splits_file"]) as f:
        return {int(r["Index"]): r["split"] for r in csv.DictReader(f)}


def load_wrappers(cfg: dict) -> dict[str, str]:
    with open(ROOT / cfg["data"]["wrappers_file"]) as f:
        return yaml.safe_load(f)


def wrap(template: str, request: str) -> str:
    return template.replace("{request}", request)
