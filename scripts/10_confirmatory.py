"""Phase 5 confirmatory analysis, exactly as pre-registered in results/phase5_preregistration.md.

Usage: python scripts/10_confirmatory.py --config configs/phase5.yaml --model qwen|mistral
Inputs:  data/labels_p5_{model}.csv, data/activations/p5_{model}/{proj_main,proj_rand}.npy + meta.json,
         data/raw/p5_{model}/generations.jsonl (first-k-token text for the text baselines; never written out)
Output:  results/phase5_metrics_{model}.json
"""
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.labels import keyword_refusal  # noqa: E402
from src.layers import fixed_layers  # noqa: E402
from src.utils import ROOT, env_info, load_config, save_json  # noqa: E402

EMOS = ["fear", "joy", "sadness", "calm", "desperate"]


def auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def ci(a, lo=2.5, hi=97.5):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    return [round(float(np.percentile(a, lo)), 4), round(float(np.percentile(a, hi)), 4)]


def holm(pvals):
    """Holm-adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def oof_scores(X, y, folds, C):
    s = np.zeros(len(y))
    for tr, te in folds:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        m.fit(X[tr], y[tr])
        s[te] = m.decision_function(X[te])
    return s


def oof_text(texts, y, folds, C):
    s = np.zeros(len(y))
    for tr, te in folds:
        m = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), LogisticRegression(C=C, max_iter=2000))
        m.fit([texts[i] for i in tr], y[tr])
        s[te] = m.decision_function([texts[i] for i in te])
    return s


def boot_indices(groups, n, seed):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    for _ in range(n):
        yield np.concatenate([members[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase5.yaml")
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    A = cfg["analysis"]
    t0 = time.time()
    tag = f"p5_{args.model}"
    mc = cfg["models"][args.model]
    L_emo, L_ref = fixed_layers(ROOT / mc["phase1_json"])
    act = ROOT / cfg["out"]["act_dir"].format(tag=tag)
    meta = json.load(open(act / "meta.json"))
    assert meta["random_layer"] == L_emo
    names = meta["directions"]
    gid2row = {r["gid"]: r["row"] for r in meta["rows"]}
    lab = list(csv.DictReader(open(ROOT / cfg["out"]["labels_csv"].format(tag=tag))))
    gens = {}
    for line in open(ROOT / cfg["out"]["raw_dir"].format(tag=tag) / "generations.jsonl"):
        g = json.loads(line)
        gens[g["gid"]] = g["first_text"]

    H = [l for l in lab if l["kind"] == "harmful"]
    rows = np.array([gid2row[l["gid"]] for l in H])
    y = np.array([int(l["compliance"]) for l in H])
    groups = np.array([l["group"] for l in H])
    pid = np.array([l["pid"] for l in H])
    dataset = np.array([l["dataset"] for l in H])
    wrapper = np.array([l["wrapper"] for l in H])
    ngen = np.array([int(l["n_gen_tokens"]) for l in H])
    p_yes = np.array([float(l["harmbench_p_yes"]) for l in H])

    pm = np.load(act / "proj_main.npy", mmap_mode="r")
    pr = np.load(act / "proj_rand.npy", mmap_mode="r")
    main = np.asarray(pm[rows])  # [N, L, P, 11]
    rand = np.asarray(pr[rows])  # [N, P, 200]
    N, L, P, _ = main.shape
    valid = (np.arange(P)[None] <= ngen[:, None]).astype(np.float32)
    cnt = np.cumsum(valid, 1)
    cs_main = np.cumsum(main * valid[:, None, :, None], 2)
    cs_rand = np.cumsum(rand * valid[:, :, None], 1)
    E = [names.index(f"emo_den:{e}") for e in EMOS]
    R = names.index("refusal")

    def feats(k):
        return cs_main[:, :, k, :] / cnt[:, k][:, None, None], cs_rand[:, k, :] / cnt[:, k][:, None]

    sgkf = StratifiedGroupKFold(n_splits=A["n_folds"], shuffle=True, random_state=A["fold_seed"])
    folds = list(sgkf.split(np.zeros(N), y, groups))
    C = A["logistic_C"]
    rng = np.random.default_rng(A["random_subset_seed"])
    rsets = [rng.choice(200, size=5, replace=False) for _ in range(A["n_random_monitors"])]

    res = {"model": mc["model"]["id"], "layers": {"emotion": L_emo, "refusal": L_ref},
           "env_analysis": env_info(),
           "n": {"harmful": int(N), "compliant": int(y.sum()), "behaviours": int(len(np.unique(groups))),
                 "by_dataset": {d: {"n": int((dataset == d).sum()), "compliant": int(y[dataset == d].sum())}
                                for d in np.unique(dataset)}},
           "underpowered": bool(y.sum() < A["min_compliant_for_power"])}

    # ---------------- monitors per k (out-of-fold) ----------------
    scores = {}
    mon = {}
    for k in A["k_list"]:
        Xm, Xr = feats(k)
        sc = {"emotion": oof_scores(Xm[:, L_emo, E], y, folds, C),
              "refusal": oof_scores(Xm[:, L_ref, [R]], y, folds, C),
              "refusal+emotion": oof_scores(np.concatenate([Xm[:, L_ref, [R]], Xm[:, L_emo, E]], 1), y, folds, C)}
        if k > 0:
            texts = [gens[l["gid"]][str(k)] for l in H]
            sc["text_keyword"] = np.array([0.0 if keyword_refusal(t) else 1.0 for t in texts])
            sc["text_tfidf"] = oof_text(texts, y, folds, C)
        rnd_auc = np.array([auc(y, oof_scores(Xr[:, s], y, folds, C)) for s in rsets])
        boots = defaultdict(list)
        for idx in boot_indices(groups, A["bootstrap_n"], A["bootstrap_seed"] + k):
            yy = y[idx]
            if not 0 < yy.sum() < len(yy):
                continue
            a = {n: auc(yy, s[idx]) for n, s in sc.items()}
            for n, v in a.items():
                boots[n].append(v)
            boots["d_minus_c"].append(a["refusal+emotion"] - a["refusal"])
            boots["b_minus_c"].append(a["emotion"] - a["refusal"])
            if k > 0:
                boots["d_minus_tfidf"].append(a["refusal+emotion"] - a["text_tfidf"])
        e = {n: {"auc": round(auc(y, s), 4), "ci95": ci(boots[n])} for n, s in sc.items()}
        dmc = np.array(boots["d_minus_c"])
        e["delta_refusal+emotion_minus_refusal"] = {
            "point": round(e["refusal+emotion"]["auc"] - e["refusal"]["auc"], 4), "ci95": ci(dmc), "ci90": ci(dmc, 5, 95),
            "p_superiority_one_sided": round(float((dmc <= 0).mean()), 4),
            "equivalent_within_margin": bool(ci(dmc, 5, 95)[0] > -A["equivalence_margin"] and ci(dmc, 5, 95)[1] < A["equivalence_margin"])}
        e["delta_emotion_minus_refusal"] = {"point": round(e["emotion"]["auc"] - e["refusal"]["auc"], 4), "ci95": ci(boots["b_minus_c"])}
        if k > 0:
            e["delta_refusal+emotion_minus_text_tfidf"] = {
                "point": round(e["refusal+emotion"]["auc"] - e["text_tfidf"]["auc"], 4), "ci95": ci(boots["d_minus_tfidf"])}
        e["random"] = {"median": round(float(np.nanmedian(rnd_auc)), 4), "p2.5_p97.5": ci(rnd_auc),
                       "p95": round(float(np.nanpercentile(rnd_auc, 95)), 4),
                       "emotion_percentile": round(float((rnd_auc < e["emotion"]["auc"]).mean() * 100), 1)}
        mon[str(k)] = e
        scores[k] = sc
        print(f"[{args.model}] k={k}: " + ", ".join(f"{n} {v['auc']}" for n, v in e.items() if isinstance(v, dict) and "auc" in v)
              + f", random median {e['random']['median']} ({time.time() - t0:.0f}s)", flush=True)
    res["monitors"] = mon

    # ---------------- confirmatory tests ----------------
    pk = [str(k) for k in A["primary_k"]]
    c1_p = [mon[k]["delta_refusal+emotion_minus_refusal"]["p_superiority_one_sided"] for k in pk]
    c1_adj = holm(c1_p)
    res["C1_added_value"] = {k: {**mon[k]["delta_refusal+emotion_minus_refusal"], "p_holm": round(a, 4),
                                 "superior": a < A["alpha"]} for k, a in zip(pk, c1_adj)}
    res["C2_emotion_vs_random"] = {k: {"emotion_auc": mon[k]["emotion"]["auc"], "random_p95": mon[k]["random"]["p95"],
                                       "percentile": mon[k]["random"]["emotion_percentile"],
                                       "above_random_p95": mon[k]["emotion"]["auc"] > mon[k]["random"]["p95"]} for k in pk}
    res["C3_pre_generation"] = {"emotion_auc_k0": mon["0"]["emotion"]["auc"], "ci95": mon["0"]["emotion"]["ci95"],
                                "upper_below_bound": mon["0"]["emotion"]["ci95"][1] < A["c3_upper_bound"]}

    # C4: within-prompt, tokens 1-5, emotion layer
    a_, b_ = 1, 5
    win_cnt = cnt[:, b_] - cnt[:, a_ - 1]
    def window(v):  # v [N, P]
        s = np.cumsum(v * valid, 1)
        return (s[:, b_] - s[:, a_ - 1]) / np.maximum(win_cnt, 1)
    by_pid = defaultdict(list)
    for i, p in enumerate(pid):
        by_pid[p].append(i)
    mixed = [np.array(ix) for ix in by_pid.values() if 0 < y[ix].sum() < len(ix)]
    prng = np.random.default_rng(A["bootstrap_seed"])

    def dz_of(win):
        diffs = []
        for ix in mixed:
            ok = ix[win_cnt[ix] > 0]
            yy = y[ok]
            if 0 < yy.sum() < len(yy):
                diffs.append(win[ok][yy == 1].mean() - win[ok][yy == 0].mean())
        diffs = np.array(diffs)
        return diffs, (float(diffs.mean() / diffs.std(ddof=1)) if len(diffs) > 2 else float("nan"))

    c4 = {"n_mixed_prompts": len(mixed)}
    rand_dz = np.array([abs(dz_of(window(rand[:, :, j]))[1]) for j in range(rand.shape[2])])
    c4["random_abs_dz_p95"] = round(float(np.nanpercentile(rand_dz, 95)), 3)
    pv = []
    for e in ("desperate", "calm"):
        diffs, dz = dz_of(window(main[:, L_emo, :, names.index(f"emo_den:{e}")]))
        null = (prng.choice([-1, 1], size=(A["n_perm"], len(diffs))) * diffs).mean(1)
        p = float((np.abs(null) >= abs(diffs.mean()) - 1e-9).mean())
        bo = [diffs[prng.integers(0, len(diffs), len(diffs))] for _ in range(2000)]
        c4[e] = {"n_prompts": int(len(diffs)), "d_z": round(dz, 3), "d_z_ci95": ci([b.mean() / b.std(ddof=1) for b in bo]),
                 "p_signflip": round(p, 4)}
        pv.append(p)
    for e, a in zip(("desperate", "calm"), holm(pv)):
        c4[e]["p_holm"] = round(a, 4)
    c4["confirmed"] = bool(c4["desperate"]["d_z"] < 0 and c4["calm"]["d_z"] > 0
                           and c4["desperate"]["p_holm"] < A["alpha"] and c4["calm"]["p_holm"] < A["alpha"]
                           and min(abs(c4["desperate"]["d_z"]), abs(c4["calm"]["d_z"])) > c4["random_abs_dz_p95"])
    _, dz_ref = dz_of(window(main[:, L_ref, :, R]))
    c4["refusal_d_z_reference"] = round(dz_ref, 3)
    res["C4_within_prompt"] = c4

    # C5: wrappers (D1)
    D1 = [l for l in lab if l["dataset"] == "D1_jbb"]
    def paired(kind, value):
        out = {}
        by = defaultdict(dict)
        for l in D1:
            if l["kind"] == kind:
                by[l["index"]].setdefault(l["wrapper"], []).append(value(l))
        for w in ("emotional_pressure", "roleplay"):
            d = np.array([np.mean(v[w]) - np.mean(v["neutral"]) for v in by.values() if w in v and "neutral" in v])
            bo = [d[np.random.default_rng(i).integers(0, len(d), len(d))].mean() for i in range(2000)]
            out[w] = {"diff_vs_neutral": round(float(d.mean()), 4), "ci95": ci(bo)}
        out["rates"] = {w: round(float(np.mean([value(l) for l in D1 if l["kind"] == kind and l["wrapper"] == w])), 4)
                        for w in ("neutral", "emotional_pressure", "roleplay")}
        return out
    res["C5_wrappers"] = {"harmful_compliance": paired("harmful", lambda l: int(l["compliance"])),
                          "benign_over_refusal": paired("benign", lambda l: 1 - int(l["compliance"]))}

    # ---------------- sensitivity analyses ----------------
    sens = {}
    excl = wrapper != "pair_vicuna"
    for k in (1, 5):
        sc = scores[k]
        sens[f"exclude_pair_vicuna_k{k}"] = {n: round(auc(y[excl], s[excl]), 4) for n, s in sc.items()}
        sens[f"per_dataset_k{k}"] = {d: {n: round(auc(y[dataset == d], s[dataset == d]), 4) for n, s in sc.items()}
                                     for d in np.unique(dataset)}
        yp = (p_yes >= 0.5).astype(int)
        sens[f"label_pyes_k{k}"] = {n: round(auc(yp, s), 4) for n, s in sc.items()}
    # nested layer choice: inside each training fold pick the layer by inner GroupKFold AUROC
    nested = {}
    for k in (0, 1, 5):
        Xm, _ = feats(k)
        out = {}
        for name, F in (("emotion", E), ("refusal", [R])):
            s = np.zeros(N)
            chosen = []
            for tr, te in folds:
                inner = list(GroupKFold(n_splits=3).split(tr, y[tr], groups[tr]))
                best = max(range(1, L), key=lambda l: auc(y[tr], oof_scores(Xm[tr][:, l, F], y[tr], inner, C)))
                chosen.append(best)
                m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(Xm[tr][:, best, F], y[tr])
                s[te] = m.decision_function(Xm[te][:, best, F])
            out[name] = {"auc": round(auc(y, s), 4), "layers_chosen": chosen}
        nested[str(k)] = out
    sens["nested_layer_choice"] = nested
    res["sensitivity"] = sens
    res["runtime_s"] = round(time.time() - t0, 1)
    save_json(res, f"results/phase5_metrics_{args.model}.json")
    print(f"saved results/phase5_metrics_{args.model}.json ({res['runtime_s']} s)")


if __name__ == "__main__":
    main()
