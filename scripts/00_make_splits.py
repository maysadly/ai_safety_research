"""Create the fixed behaviour split (train/val/test by JBB Index, stratified by category).

Usage: python scripts/00_make_splits.py --config configs/base.yaml
Writes data/splits.csv (IDs and categories only; committed).
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import make_splits
from src.utils import ROOT, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    cfg = load_config(ap.parse_args().config)
    rows = make_splits(cfg, seed=cfg["seed"])
    out = ROOT / cfg["data"]["splits_file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Index", "Category", "split"])
        w.writeheader()
        w.writerows(rows)
    print(out, Counter(r["split"] for r in rows))


if __name__ == "__main__":
    main()
