"""Phase 5 (descriptive): mean emotion projection per token position, complied vs refused harmful answers,
at each model's fixed emotion layer. 95 % CI from a bootstrap over behaviours.

Usage: python scripts/12_phase5_trajectories.py
Output: results/phase5_trajectories.json
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from src.layers import fixed_layers  # noqa: E402
from src.utils import ROOT, load_config, save_json  # noqa: E402


def main():
    cfg = load_config("configs/phase5.yaml")
    out = {}
    for m in ("qwen", "mistral"):
        tag = f"p5_{m}"
        L_emo, _ = fixed_layers(ROOT / cfg["models"][m]["phase1_json"])
        act = ROOT / cfg["out"]["act_dir"].format(tag=tag)
        meta = json.load(open(act / "meta.json"))
        names = meta["directions"]
        g2r = {r["gid"]: r["row"] for r in meta["rows"]}
        H = [l for l in csv.DictReader(open(ROOT / cfg["out"]["labels_csv"].format(tag=tag))) if l["kind"] == "harmful"]
        rows = np.array([g2r[l["gid"]] for l in H])
        y = np.array([int(l["compliance"]) for l in H])
        grp = np.array([l["group"] for l in H])
        ngen = np.array([int(l["n_gen_tokens"]) for l in H])
        pm = np.load(act / "proj_main.npy", mmap_mode="r")
        out[m] = {"layer": L_emo}
        for e in ("desperate", "calm"):
            v = np.asarray(pm[rows][:, L_emo, :, names.index(f"emo_den:{e}")])  # [N, P]
            P = v.shape[1]
            out[m][e] = {}
            for lab, cls in (("complied", 1), ("refused", 0)):
                idx = np.flatnonzero(y == cls)
                mean, lo, hi = [], [], []
                rng = np.random.default_rng(0)
                uniq = np.unique(grp[idx])
                members = {g: idx[grp[idx] == g] for g in uniq}
                for p in range(P):
                    ok = idx[ngen[idx] >= p]
                    mean.append(float(v[ok, p].mean()))
                boots = []
                for _ in range(300):
                    pick = np.concatenate([members[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
                    boots.append([float(v[pick[ngen[pick] >= p], p].mean()) for p in range(P)])
                boots = np.array(boots)
                out[m][e][lab] = {"mean": mean, "lo": np.percentile(boots, 2.5, 0).tolist(),
                                  "hi": np.percentile(boots, 97.5, 0).tolist(), "n": int(len(idx))}
    save_json(out, "results/phase5_trajectories.json")
    print({m: {e: {l: (round(d["mean"][0], 2), round(d["mean"][3], 2), d["n"]) for l, d in out[m][e].items()}
               for e in ("desperate", "calm")} for m in out})


if __name__ == "__main__":
    main()
