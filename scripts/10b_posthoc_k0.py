"""POST-HOC (not pre-registered): where does the pre-generation (k = 0) signal come from in Phase 5?
AUROC of the same out-of-fold k = 0 monitors, computed within each prompt family. If the pooled AUROC is high but
within-family AUROC is low, the monitor mostly recognises the type of prompt, not the decision.

Usage: python scripts/10b_posthoc_k0.py --model qwen|mistral
"""
import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from src.layers import fixed_layers  # noqa: E402
from src.utils import ROOT, load_config, save_json  # noqa: E402

spec = importlib.util.spec_from_file_location("conf", ROOT / "scripts/10_confirmatory.py")
conf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conf)


def family(w):
    return "D1 plain + wrappers" if w in ("neutral", "emotional_pressure", "roleplay") else \
        "PAIR" if w.startswith("pair") else "random-search (PRS)" if w.startswith("prs") else \
        "HarmBench direct" if w == "hb_direct" else "HarmBench human jailbreak"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    cfg = load_config("configs/phase5.yaml")
    A = cfg["analysis"]
    tag = f"p5_{args.model}"
    L_emo, L_ref = fixed_layers(ROOT / cfg["models"][args.model]["phase1_json"])
    act = ROOT / cfg["out"]["act_dir"].format(tag=tag)
    meta = json.load(open(act / "meta.json"))
    names = meta["directions"]
    gid2row = {r["gid"]: r["row"] for r in meta["rows"]}
    H = [l for l in csv.DictReader(open(ROOT / cfg["out"]["labels_csv"].format(tag=tag))) if l["kind"] == "harmful"]
    rows = np.array([gid2row[l["gid"]] for l in H])
    y = np.array([int(l["compliance"]) for l in H])
    groups = np.array([l["group"] for l in H])
    fam = np.array([family(l["wrapper"]) for l in H])
    pm = np.load(act / "proj_main.npy", mmap_mode="r")
    pr = np.load(act / "proj_rand.npy", mmap_mode="r")
    X0 = np.asarray(pm[rows][:, :, 0, :])
    R0 = np.asarray(pr[rows][:, 0, :])
    E = [names.index(f"emo_den:{e}") for e in conf.EMOS]
    R = names.index("refusal")
    folds = list(StratifiedGroupKFold(n_splits=A["n_folds"], shuffle=True, random_state=A["fold_seed"]).split(np.zeros(len(y)), y, groups))
    C = A["logistic_C"]
    sc = {"emotion": conf.oof_scores(X0[:, L_emo, E], y, folds, C), "refusal": conf.oof_scores(X0[:, L_ref, [R]], y, folds, C),
          "family_only": None}
    rng = np.random.default_rng(A["random_subset_seed"])
    rsets = [rng.choice(200, size=5, replace=False) for _ in range(50)]
    rnd = [conf.oof_scores(R0[:, s], y, folds, C) for s in rsets]
    # a monitor that only knows the prompt family (one-hot), as a reference for "recognising the prompt type"
    fams = sorted(set(fam))
    F = np.stack([(fam == f).astype(float) for f in fams], 1)
    sc["family_only"] = conf.oof_scores(F, y, folds, C)
    out = {"pooled": {n: round(conf.auc(y, s), 4) for n, s in sc.items()},
           "pooled_random_median": round(float(np.median([conf.auc(y, s) for s in rnd])), 4), "within_family": {}}
    for f in fams:
        m = fam == f
        out["within_family"][f] = {"n": int(m.sum()), "compliant": int(y[m].sum()),
                                   **{n: round(conf.auc(y[m], s[m]), 4) for n, s in sc.items() if n != "family_only"},
                                   "random_median": round(float(np.nanmedian([conf.auc(y[m], s[m]) for s in rnd])), 4)}
    save_json(out, f"results/phase5_posthoc_k0_{args.model}.json")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
