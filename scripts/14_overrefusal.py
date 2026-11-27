"""Phase 6A (EXPLORATORY, CPU, existing Phase 5 data): over-refusal of benign requests.
1. Can a monitor predict, before / early in generation, that the model will refuse a *benign* request?
2. Is the emotional-pressure effect on over-refusal statistically mediated by the 'desperate' projection at the prompt?

Usage: python scripts/14_overrefusal.py
Output: results/phase6a_overrefusal.json
"""
import csv
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from src.layers import fixed_layers  # noqa: E402
from src.utils import ROOT, load_config, save_json  # noqa: E402

spec = importlib.util.spec_from_file_location("conf", ROOT / "scripts/10_confirmatory.py")
conf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conf)


def main():
    cfg = load_config("configs/phase5.yaml")
    A = cfg["analysis"]
    out = {}
    for m in ("qwen", "mistral"):
        tag = f"p5_{m}"
        L_emo, L_ref = fixed_layers(ROOT / cfg["models"][m]["phase1_json"])
        act = ROOT / cfg["out"]["act_dir"].format(tag=tag)
        meta = json.load(open(act / "meta.json"))
        names = meta["directions"]
        g2r = {r["gid"]: r["row"] for r in meta["rows"]}
        B = [l for l in csv.DictReader(open(ROOT / cfg["out"]["labels_csv"].format(tag=tag))) if l["kind"] == "benign"]
        gens = {}
        for line in open(ROOT / cfg["out"]["raw_dir"].format(tag=tag) / "generations.jsonl"):
            g = json.loads(line)
            if g["kind"] == "benign":
                gens[g["gid"]] = g["first_text"]
        rows = np.array([g2r[l["gid"]] for l in B])
        y = np.array([1 - int(l["compliance"]) for l in B])  # 1 = over-refused
        groups = np.array([l["index"] for l in B])
        wr = np.array([l["wrapper"] for l in B])
        ngen = np.array([int(l["n_gen_tokens"]) for l in B])
        pm = np.load(act / "proj_main.npy", mmap_mode="r")
        pr = np.load(act / "proj_rand.npy", mmap_mode="r")
        main, rand = np.asarray(pm[rows]), np.asarray(pr[rows])
        P = main.shape[2]
        valid = (np.arange(P)[None] <= ngen[:, None]).astype(np.float32)
        cnt = np.cumsum(valid, 1)
        cm = np.cumsum(main * valid[:, None, :, None], 2)
        cr = np.cumsum(rand * valid[:, :, None], 1)
        E = [names.index(f"emo_den:{e}") for e in conf.EMOS]
        R = names.index("refusal")
        D = names.index("emo_den:desperate")
        folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, groups))
        rng = np.random.default_rng(11)
        rsets = [rng.choice(200, size=5, replace=False) for _ in range(100)]
        res = {"n": int(len(y)), "over_refused": int(y.sum()), "monitors": {}}
        for k in (0, 1, 5):
            Xm = cm[:, :, k, :] / cnt[:, k][:, None, None]
            Xr = cr[:, k, :] / cnt[:, k][:, None]
            sc = {"emotion": conf.oof_scores(Xm[:, L_emo, E], y, folds, 1.0),
                  "refusal": conf.oof_scores(Xm[:, L_ref, [R]], y, folds, 1.0),
                  "desperate_only": Xm[:, L_emo, D]}
            if k > 0:
                sc["text_tfidf"] = conf.oof_text([gens[l["gid"]][str(k)] for l in B], y, folds, 1.0)
            rnd = np.array([conf.auc(y, conf.oof_scores(Xr[:, s], y, folds, 1.0)) for s in rsets])
            e = {n: round(conf.auc(y, s), 4) for n, s in sc.items()}
            boots = {n: [] for n in sc}
            for idx in conf.boot_indices(groups, 1000, 7 + k):
                if 0 < y[idx].sum() < len(idx):
                    for n, s in sc.items():
                        boots[n].append(conf.auc(y[idx], s[idx]))
            e = {n: {"auc": v, "ci95": conf.ci(boots[n])} for n, v in e.items()}
            e["random_median"] = round(float(np.median(rnd)), 4)
            e["random_p95"] = round(float(np.percentile(rnd, 95)), 4)
            e["emotion_percentile"] = round(float((rnd < e["emotion"]["auc"]).mean() * 100), 1)
            res["monitors"][str(k)] = e
        # mediation (prompt position): over-refusal ~ emotional-pressure wrapper (+ desperate projection)
        mask = np.isin(wr, ["neutral", "emotional_pressure"])
        x_w = (wr[mask] == "emotional_pressure").astype(float)
        desp = cm[mask, L_emo, 0, D] / cnt[mask, 0]
        dz = (desp - desp.mean()) / desp.std()
        def coef(X):
            return LogisticRegression(C=1e6, max_iter=5000).fit(X, y[mask]).coef_[0]
        b_total = coef(x_w[:, None])[0]
        b_direct, b_desp = coef(np.column_stack([x_w, dz]))
        res["mediation"] = {"logit_wrapper_total": round(float(b_total), 3),
                            "logit_wrapper_given_desperate": round(float(b_direct), 3),
                            "logit_desperate_per_sd": round(float(b_desp), 3),
                            "share_of_wrapper_effect_explained": round(float(1 - b_direct / b_total), 3),
                            "desperate_shift_sd": round(float(dz[x_w == 1].mean() - dz[x_w == 0].mean()), 3)}
        res["over_refusal_by_wrapper"] = {w: round(float(y[wr == w].mean()), 4) for w in np.unique(wr)}
        out[m] = res
        print(m, json.dumps({k: {n: (v["auc"] if isinstance(v, dict) else v) for n, v in e.items()} for k, e in res["monitors"].items()}),
              "| mediation", res["mediation"], "| rates", res["over_refusal_by_wrapper"])
    save_json(out, "results/phase6a_overrefusal.json")


if __name__ == "__main__":
    main()
