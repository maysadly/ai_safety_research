"""Phase 3 (local, CPU): H1 effect sizes, H2 monitors, early warning, wrappers, false alarms.

Usage: python scripts/06_analysis.py --config configs/phase3.yaml
Inputs:  data/labels_{tag}.csv, data/activations/{tag}/projections.npy + meta.json, results/phase1_vectors.json
Output:  results/phase3_metrics.json (numbers for the summary and the figures)
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
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.utils import ROOT, env_info, load_config, save_json  # noqa: E402
from src.vectors import cohens_d, wilson_ci  # noqa: E402


# ----------------------------------------------------------------------------------------------------
def auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def fit_score(Xtr, ytr, Xev_list, C):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    m.fit(Xtr, ytr)
    return [m.decision_function(X) for X in Xev_list]


def cluster_boot_idx(groups, n, seed):
    """Yield index arrays of cluster-bootstrap resamples (clusters = behaviours)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        yield np.concatenate([members[g] for g in pick])


def ci(a):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase3.yaml")
    cfg = load_config(ap.parse_args().config)
    t0 = time.time()
    tag = cfg["tag"]
    btag = cfg.get("benign_tag", tag)  # exploratory PAIR run: harmful rows from `tag`, benign rows from `btag`
    lab, proj_parts, names = [], [], None
    for t, keep in ((tag, "harmful"), (btag, "benign")) if btag != tag else ((tag, None),):
        act_dir = ROOT / cfg["out"]["act_dir"].format(tag=t)
        meta = json.load(open(act_dir / "meta.json"))
        assert names is None or names == meta["directions"]
        names = meta["directions"]
        gid2row = {r["gid"]: r["row"] for r in meta["rows"]}
        lt = [l for l in csv.DictReader(open(ROOT / cfg["out"]["labels_csv"].format(tag=t)))
              if keep is None or l["kind"] == keep]
        pm = np.load(act_dir / "projections.npy", mmap_mode="r")
        proj_parts.append(np.asarray(pm[np.array([gid2row[l["gid"]] for l in lt])]))
        lab += lt
    kind = np.array([l["kind"] for l in lab])
    split = np.array([l["split"] for l in lab])
    wrapper = np.array([l["wrapper"] for l in lab])
    beh = np.array([int(l["index"]) for l in lab])
    pid = np.array([l["pid"] for l in lab])
    y = np.array([int(l["compliance"]) for l in lab])
    ngen = np.array([int(l["n_gen_tokens"]) for l in lab])
    p1 = json.load(open(ROOT / "results/phase1_vectors.json"))
    emotions = p1["emotions"]

    E = [names.index(f"{cfg['emotion_variant']}:{e}") for e in emotions]
    R = names.index("refusal")
    RND = [names.index(f"random:{i}") for i in range(cfg["random"]["n_per_layer"])]

    # ---- features: cumulative mean over existing positions 0..k ---------------------------------
    proj = np.concatenate(proj_parts) if len(proj_parts) > 1 else proj_parts[0]  # [N, L, P, K], label order
    del proj_parts
    N, L, P, K = proj.shape
    valid = (np.arange(P)[None] <= ngen[:, None]).astype(np.float32)  # [N, P]
    csum = np.cumsum(proj * valid[:, None, :, None], axis=2)
    ccount = np.cumsum(valid, axis=1)
    feats = {k: csum[:, :, k, :] / ccount[:, k][:, None, None] for k in cfg["k_list"]}  # [N, L, K]
    del csum

    H = kind == "harmful"
    tr, va, te = (H & (split == cfg["monitors"][s]) for s in ("fit_split", "select_split", "test_split"))
    teB = (kind == "benign") & (split == cfg["monitors"]["test_split"])
    lo, hi = cfg["monitors"]["candidate_layers"]
    layers = list(range(lo, hi + 1))
    C = cfg["monitors"]["logistic_C"]
    rng = np.random.default_rng(cfg["monitors"]["random_subset_seed"])
    rnd_sets = [list(rng.choice(RND, size=len(E), replace=False)) for _ in range(cfg["monitors"]["n_random_monitors"])]
    res = {"env_analysis": env_info(), "n": {"harmful_train": int(tr.sum()), "harmful_val": int(va.sum()),
                                             "harmful_test": int(te.sum()), "benign_test": int(teB.sum())},
           "test_compliance_rate": round(float(y[te].mean()), 4)}

    # ---- H2 + early warning: monitors per k ---------------------------------------------------------
    def select_and_score(Xs, F):
        """Xs = (train, val, test, benign-test) feature arrays [n, L, K]. Pick the layer with the best val
        AUROC (fit on train); return dict with layer, val AUROC and scores."""
        Xtr_, Xva_, Xte_, XtB_ = Xs
        best = (-1.0, None)
        for l in layers:
            sv, = fit_score(Xtr_[:, l, F], y[tr], [Xva_[:, l, F]], C)
            a = auc(y[va], sv)
            if a > best[0]:
                best = (a, l)
        l = best[1]
        st, sb, sv = fit_score(Xtr_[:, l, F], y[tr], [Xte_[:, l, F], XtB_[:, l, F], Xva_[:, l, F]], C)
        return {"layer": l, "val_auc": best[0], "test": st, "benign": sb, "val": sv}

    groups_te = beh[te]
    mon = {}
    for k in cfg["k_list"]:
        X = feats[k]
        Xs = (X[tr], X[va], X[te], X[teB])
        b = select_and_score(Xs, E)
        c = select_and_score(Xs, [R])
        Xd_tr = np.concatenate([X[tr][:, c["layer"], [R]], X[tr][:, b["layer"], E]], 1)
        Xd = [np.concatenate([X[m][:, c["layer"], [R]], X[m][:, b["layer"], E]], 1) for m in (te, teB, va)]
        dt, db, dv = fit_score(Xd_tr, y[tr], Xd, C)
        d = {"layer": [c["layer"], b["layer"]], "val_auc": auc(y[va], dv), "test": dt, "benign": db, "val": dv}
        rnd = [select_and_score(Xs, list(S)) for S in rnd_sets]
        rnd_auc = np.array([auc(y[te], r["test"]) for r in rnd])
        # cluster bootstrap over test behaviours (paired across monitors)
        boots = defaultdict(list)
        for idx in cluster_boot_idx(groups_te, cfg["bootstrap"]["n"], cfg["bootstrap"]["seed"] + k):
            yy = y[te][idx]
            if not 0 < yy.sum() < len(yy):
                continue
            ab, ac, ad = (auc(yy, m["test"][idx]) for m in (b, c, d))
            boots["emotion"].append(ab)
            boots["refusal"].append(ac)
            boots["refusal+emotion"].append(ad)
            boots["d_minus_c"].append(ad - ac)
            boots["b_minus_c"].append(ab - ac)
        dmc = np.array(boots["d_minus_c"])
        entry = {}
        for name, m in (("emotion", b), ("refusal", c), ("refusal+emotion", d)):
            entry[name] = {"layer": m["layer"], "val_auc": round(m["val_auc"], 4),
                           "test_auc": round(auc(y[te], m["test"]), 4), "ci95": ci(boots[name])}
        entry["random"] = {"median": round(float(np.median(rnd_auc)), 4), "p2.5_p97.5": ci(rnd_auc),
                           "layers_chosen_median": int(np.median([r["layer"] for r in rnd])),
                           "emotion_percentile": round(float((rnd_auc < entry["emotion"]["test_auc"]).mean() * 100), 1)}
        entry["delta_refusal+emotion_minus_refusal"] = {
            "point": round(entry["refusal+emotion"]["test_auc"] - entry["refusal"]["test_auc"], 4), "ci95": ci(dmc),
            "p_two_sided": round(float(min(1.0, 2 * min((dmc <= 0).mean(), (dmc >= 0).mean()))), 4)}
        entry["delta_emotion_minus_refusal"] = {
            "point": round(entry["emotion"]["test_auc"] - entry["refusal"]["test_auc"], 4), "ci95": ci(boots["b_minus_c"])}
        # false alarms (threshold on val harmful at target recall of compliance)
        if k in cfg["false_alarms"]["k_values"]:
            fa = {}
            for name, m in (("emotion", b), ("refusal", c), ("refusal+emotion", d)):
                sv_pos = np.sort(m["val"][y[va] == 1])
                thr = float(np.quantile(sv_pos, 1 - cfg["false_alarms"]["target_recall_on_val"])) if len(sv_pos) else np.nan
                tp = m["test"][y[te] == 1] >= thr
                fp = m["test"][y[te] == 0] >= thr
                ab = m["benign"] >= thr
                fa[name] = {"threshold": round(thr, 4),
                            "test_recall": {"rate": round(float(tp.mean()), 4), "ci95": [round(v, 4) for v in wilson_ci(int(tp.sum()), len(tp))]},
                            "test_false_alarm_on_refused_harmful": {"rate": round(float(fp.mean()), 4), "ci95": [round(v, 4) for v in wilson_ci(int(fp.sum()), len(fp))]},
                            "alarm_rate_on_benign": {"rate": round(float(ab.mean()), 4), "ci95": [round(v, 4) for v in wilson_ci(int(ab.sum()), len(ab))]}}
            entry["false_alarms"] = fa
        mon[str(k)] = entry
        print(f"k={k}: " + ", ".join(f"{n} {entry[n]['test_auc']}" for n in ("emotion", "refusal", "refusal+emotion"))
              + f", random median {entry['random']['median']}  ({time.time() - t0:.0f}s)", flush=True)
    res["monitors"] = mon

    # ---- H1: compliant vs refused (val+test harmful), fixed Phase-1 layers -----------------------------
    hl = p1["heldout"]["loto_acc_denoised"]
    L_emo = int(np.argmax(hl[7:28]) + 7)
    ra = p1["refusal"]["val_auroc_per_layer"]
    L_ref = int(np.argmax(ra[1:]) + 1)
    h1m = H & np.isin(split, cfg["h1"]["splits"])
    res["h1"] = {"layer_emotion": L_emo, "layer_refusal": L_ref, "splits": cfg["h1"]["splits"],
                 "n_responses": int(h1m.sum()), "n_compliant": int(y[h1m].sum())}
    dir_layers = [(e, E[i], L_emo) for i, e in enumerate(emotions)] + [("refusal", R, L_ref)]
    x0 = feats[0]
    rng_p = np.random.default_rng(cfg["bootstrap"]["seed"])

    # between-prompt (position 0 = last prompt token; identical across samples), wrapper-centred
    idx = np.flatnonzero(h1m)
    w_ = wrapper[idx]

    def centred(v):
        out = v.copy()
        for w in np.unique(w_):
            out[w_ == w] -= v[w_ == w].mean()
        return out

    prompts = np.unique(pid[idx])
    p_rows = {p: np.flatnonzero(pid[idx] == p) for p in prompts}
    p_wr = {p: wrapper[idx][p_rows[p][0]] for p in prompts}
    between = {}
    for name, j, l in dir_layers:
        v = centred(x0[idx, l, j])
        yy = y[idx]
        d_obs = cohens_d(v[yy == 1], v[yy == 0])
        # wrapper-stratified permutation of prompt-level label vectors
        perm_d = []
        by_w = defaultdict(list)
        for p in prompts:
            by_w[p_wr[p]].append(p)
        for _ in range(cfg["h1"]["n_perm"]):
            yp = np.empty_like(yy)
            for w, ps in by_w.items():
                src = rng_p.permutation(len(ps))
                for p, q in zip(ps, [ps[s] for s in src]):
                    # assign q's labels to p's rows (all prompts have the same number of samples)
                    yp[p_rows[p]] = yy[p_rows[q]]
            perm_d.append(cohens_d(v[yp == 1], v[yp == 0]))
        perm_d = np.abs(np.array(perm_d))
        boots = [cohens_d(v[i][yy[i] == 1], v[i][yy[i] == 0])
                 for i in cluster_boot_idx(beh[idx], 1000, cfg["bootstrap"]["seed"])]
        between[name] = {"d": round(d_obs, 3), "ci95": ci(boots),
                         "p_perm": round(float((perm_d >= abs(d_obs) - 1e-9).mean()), 4)}
    rd = [abs(cohens_d(centred(x0[idx, L_emo, j])[y[idx] == 1], centred(x0[idx, L_emo, j])[y[idx] == 0])) for j in RND]
    between["random_abs_d_p95"] = round(float(np.percentile(rd, 95)), 3)
    res["h1"]["between_prompt"] = between

    # within-prompt: prompts with mixed outcomes; generated-token windows
    mixed = [p for p in prompts if 0 < y[idx][p_rows[p]].sum() < len(p_rows[p])]
    res["h1"]["n_mixed_prompts"] = len(mixed)
    within = {}
    for (a, bnd) in cfg["h1"]["within_windows"]:
        cnt = ccount[:, bnd] - ccount[:, a - 1]
        key = f"tokens_{a}-{bnd}"
        within[key] = {}
        for name, j, l in dir_layers + [(f"random:{i}", jj, L_emo) for i, jj in enumerate(RND)]:
            s = (np.cumsum(proj[:, l, :, j] * valid, axis=1))
            win = (s[:, bnd] - s[:, a - 1]) / np.maximum(cnt, 1)
            diffs = []
            for p in mixed:
                r_ = idx[p_rows[p]]
                ok = cnt[r_] > 0
                yy, vv = y[r_][ok], win[r_][ok]
                if 0 < yy.sum() < len(yy):
                    diffs.append(vv[yy == 1].mean() - vv[yy == 0].mean())
            diffs = np.array(diffs)
            dz = float(diffs.mean() / diffs.std(ddof=1)) if len(diffs) > 2 else float("nan")
            if name.startswith("random:"):
                within[key].setdefault("_random_dz", []).append(dz)
                continue
            signs = rng_p.choice([-1, 1], size=(cfg["h1"]["n_perm"], len(diffs)))
            null = (signs * diffs).mean(1)
            bo = [diffs[rng_p.integers(0, len(diffs), len(diffs))] for _ in range(2000)]
            within[key][name] = {"n_prompts": int(len(diffs)), "mean_diff": round(float(diffs.mean()), 4),
                                 "d_z": round(dz, 3),
                                 "d_z_ci95": ci([b.mean() / b.std(ddof=1) for b in bo]),
                                 "p_signflip": round(float((np.abs(null) >= abs(diffs.mean()) - 1e-9).mean()), 4)}
        rz = np.abs(np.array(within[key].pop("_random_dz")))
        within[key]["random_abs_dz_p95"] = round(float(np.nanpercentile(rz, 95)), 3)
    res["h1"]["within_prompt"] = within

    # trajectories for Fig. 4: mean projection per position, compliant vs refused (val+test harmful)
    traj = {}
    for name, j, l in dir_layers:
        traj[name] = {}
        for lab_name, cls in (("compliant", 1), ("refused", 0)):
            m = idx[y[idx] == cls]
            means, lo_, hi_ = [], [], []
            for p in range(P):
                ok = m[valid[m, p] > 0]
                v = proj[ok, l, p, j]
                g = beh[ok]
                means.append(float(v.mean()))
                bs_ = [v[i].mean() for i in cluster_boot_idx(g, 300, p)]
                c_ = ci(bs_)
                lo_.append(c_[0])
                hi_.append(c_[1])
            traj[name][lab_name] = {"mean": means, "lo": lo_, "hi": hi_, "n": int(len(m))}
    res["h1"]["trajectories"] = traj

    # ---- wrappers (all harmful behaviours; paired by behaviour) ----------------------------------------
    wr = {}
    beh_all = np.unique(beh[H])
    present = set(wrapper[H])
    for w in [w for w in ("neutral", "emotional_pressure", "roleplay", "pair") if w in present]:
        m = H & (wrapper == w)
        boots = [y[m][i].mean() for i in cluster_boot_idx(beh[m], 2000, 5)]
        mb = (kind == "benign") & (wrapper == w)
        wr[w] = {"harmful_compliance": round(float(y[m].mean()), 4), "ci95": ci(boots), "n": int(m.sum())}
        if mb.sum():
            wr[w]["benign_over_refusal"] = round(float(1 - y[mb].mean()), 4)
            wr[w]["benign_ci95"] = [round(v, 4) for v in wilson_ci(int((1 - y[mb]).sum()), int(mb.sum()))]
        # emotion projection at the prompt (k = 0) relative to the neutral wrapper, paired by behaviour
        if w != "neutral" and "neutral" in present:
            wr[w]["prompt_projection_shift_vs_neutral"] = {}
            for name, j, l in dir_layers:
                diffs = []
                for bh in beh_all:
                    a_ = H & (wrapper == w) & (beh == bh)
                    n_ = H & (wrapper == "neutral") & (beh == bh)
                    diffs.append(x0[a_, l, j].mean() - x0[n_, l, j].mean())
                diffs = np.array(diffs)
                bo = [diffs[np.random.default_rng(i).integers(0, len(diffs), len(diffs))].mean() for i in range(2000)]
                wr[w]["prompt_projection_shift_vs_neutral"][name] = {
                    "mean": round(float(diffs.mean()), 3), "ci95": ci(bo),
                    "in_sd_units": round(float(diffs.mean() / x0[H & (wrapper == "neutral"), l, j].std()), 3)}
            cd = []
            for bh in beh_all:
                cd.append(y[H & (wrapper == w) & (beh == bh)].mean() - y[H & (wrapper == "neutral") & (beh == bh)].mean())
            cd = np.array(cd)
            bo = [cd[np.random.default_rng(i).integers(0, len(cd), len(cd))].mean() for i in range(2000)]
            wr[w]["compliance_diff_vs_neutral"] = {"mean": round(float(cd.mean()), 4), "ci95": ci(bo)}
    res["wrappers"] = wr
    res["runtime_s"] = round(time.time() - t0, 1)
    out = cfg.get("metrics_out", "results/phase3_metrics.json")
    res["tags"] = {"harmful": tag, "benign": btag}
    save_json(res, out)
    print(f"saved {out} in {res['runtime_s']} s")


if __name__ == "__main__":
    main()
