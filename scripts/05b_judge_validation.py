"""Agreement of the HarmBench judge with an independent review of the stratified validation samples.

Usage: python scripts/05b_judge_validation.py
Inputs:  data/labels_{tag}.csv, data/review_{source}_{tag}.csv (gid,label,confidence,reason; no text)
Output:  results/judge_validation.json
The review files come from an independent LLM annotator (Claude, instructed with the HarmBench rule, blind to
judge labels). Human labels (data/raw/{tag}/manual_validation_{tag}.csv) are still pending; when filled in,
save them as data/review_human_{tag}.csv with the same columns and re-run this script.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.utils import ROOT, save_json  # noqa: E402


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe))


out = {}
for source in ("llm", "human"):
    for tag in ("main", "pair"):
        f = ROOT / f"data/review_{source}_{tag}.csv"
        if not f.exists():
            continue
        lab = {l["gid"]: l for l in csv.DictReader(open(ROOT / f"data/labels_{tag}.csv"))}
        H = [l for l in lab.values() if l["kind"] == "harmful"]
        p_hb = np.mean([int(l["compliance"]) for l in H])
        rev = list(csv.DictReader(open(f)))
        for conf in ("all", "high"):
            rr = [r for r in rev if conf == "all" or r["confidence"] == "high"]
            hb = np.array([int(lab[r["gid"]]["compliance"]) for r in rr])
            rv = np.array([int(r["label"] == "compliance") for r in rr])
            agree_pos = float((rv[hb == 1] == 1).mean()) if (hb == 1).any() else float("nan")
            agree_neg = float((rv[hb == 0] == 0).mean()) if (hb == 0).any() else float("nan")
            out[f"{source}_{tag}_{conf}"] = {
                "n": int(len(rr)), "n_hb_compliant": int(hb.sum()), "n_review_compliant": int(rv.sum()),
                "raw_agreement": round(float((hb == rv).mean()), 3), "kappa_in_sample": round(kappa(hb, rv), 3),
                "P(review compliant | HB compliant)": round(agree_pos, 3),
                "P(review refused | HB refused)": round(agree_neg, 3),
                "population_HB_rate": round(float(p_hb), 4),
                "reweighted_accuracy": round(float(p_hb * agree_pos + (1 - p_hb) * agree_neg), 3),
                "implied_review_compliance_rate": round(float(p_hb * agree_pos + (1 - p_hb) * (1 - agree_neg)), 4),
            }
save_json(out, "results/judge_validation.json")
print(json.dumps(out, indent=1))
