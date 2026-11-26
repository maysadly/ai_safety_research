"""Phase 1b: emotion vectors (+ held-out validation), refusal direction, random baselines.

Usage: python scripts/02_build_vectors.py --config configs/phase1.yaml
Inputs:  data/stories/stories.jsonl
Outputs: data/vectors/*.npz (git-ignored), results/phase1_vectors.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.data import load_behaviors, load_splits, load_wrappers, wrap  # noqa: E402
from src.utils import ROOT, Timer, env_info, load_config, save_json, set_seed  # noqa: E402
from src.vectors import (auroc, classify, cohens_d, emotion_vectors, neutral_pc_bases,  # noqa: E402
                         project_out, random_unit_vectors, unit, wilson_ci)


def story_acts(model, tok, processor, cfg, texts, bs=32):
    """Mean residual over the text tokens (skipping the first k) when the text is the assistant turn
    after a fixed neutral user prompt. Returns [N, L+1, d] float32."""
    import torch
    from src.model import chat_text, left_pad, residual_mean
    prefix = tok(chat_text(processor, cfg["reading_user_prompt"], cfg["model"]["system_prompt"]),
                 add_special_tokens=False)["input_ids"]
    suffix = [tok.convert_tokens_to_ids("<|im_end|>")]
    contents = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    order = np.argsort([len(c) for c in contents])
    out = [None] * len(texts)
    k = cfg["skip_first_tokens"]
    for b in range(0, len(order), bs):
        idx = order[b: b + bs]
        seqs = [prefix + contents[i] + suffix for i in idx]
        ids, am, offs = left_pad(seqs, tok.pad_token_id)
        starts = torch.tensor([o + len(prefix) + k for o in offs])
        ends = torch.tensor([o + len(prefix) + len(contents[i]) for o, i in zip(offs, idx)])
        acts = residual_mean(model, ids.to(model.device), am.to(model.device), starts, ends)
        for j, i in enumerate(idx):
            out[i] = acts[j].numpy()
    return np.stack(out)


def prompt_acts(model, tok, processor, cfg, texts, bs=32):
    """Residual at the last token of the chat-formatted prompt. Returns [N, L+1, d] float32."""
    import torch
    from src.model import chat_text, left_pad, residual_at
    seqs = [tok(chat_text(processor, t, cfg["model"]["system_prompt"]), add_special_tokens=False)["input_ids"]
            for t in texts]
    out = []
    for b in range(0, len(seqs), bs):
        ids, am, _ = left_pad(seqs[b: b + bs], tok.pad_token_id)
        pos = torch.full((ids.shape[0], 1), ids.shape[1] - 1)
        out.append(residual_at(model, ids.to(model.device), am.to(model.device), pos)[:, :, 0].numpy())
    return np.concatenate(out)


def heldout_accuracy(acts, labels, topics, held, emotions, neutral_acts=None, neutral_topics=None, var=None):
    tr = ~np.isin(topics, held)
    V, mu = emotion_vectors(acts[tr], labels[tr], emotions)
    if neutral_acts is not None:
        bases = neutral_pc_bases(neutral_acts[~np.isin(neutral_topics, held)], var)
        V = project_out(V, bases)
    pred = classify(acts[~tr], mu, unit(V))
    y = np.array([emotions.index(e) for e in labels[~tr]])
    correct = (pred == y[:, None])
    return correct.mean(0), int((~tr).sum()), correct.sum(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--from-cache", action="store_true",
                    help="skip the model: reuse data/vectors/story_acts.npz and prompt_acts_trainval.npz")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    T = Timer()
    emotions = cfg["emotions"]
    vdir = ROOT / cfg["out"]["vectors_dir"]
    vdir.mkdir(parents=True, exist_ok=True)
    res = {"env": env_info(cfg["model"]["id"], cfg["model"]["revision"]), "emotions": emotions}
    env_file = vdir / "activations_env.json"
    if not args.from_cache:
        save_json(res["env"], env_file)
    res["env_activations"] = json.load(open(env_file))  # GPU run that produced the cached activations

    if not args.from_cache:
        from src.model import load_thinker
        with T("load_s"):
            model, processor, tok = load_thinker(cfg)

    # ---- stories -> activations ----------------------------------------------------------------
    if not args.from_cache:
        recs = [json.loads(l) for l in open(ROOT / cfg["out"]["stories"])]
        with T("story_acts_s"):
            A = story_acts(model, tok, processor, cfg, [r["text"] for r in recs])
        np.savez_compressed(vdir / "story_acts.npz", acts=A.astype(np.float16), ids=np.array([r["id"] for r in recs]),
                            kind=np.array([r["kind"] for r in recs]), emotion=np.array([r["emotion"] for r in recs]),
                            topic_id=np.array([r["topic_id"] for r in recs]))
    # all analysis below runs on the saved (float16) activations, so --from-cache reproduces it exactly
    z = np.load(vdir / "story_acts.npz")
    A, kind, emo, top = z["acts"].astype(np.float32), z["kind"], z["emotion"], z["topic_id"]
    S, N = kind == "story", kind == "neutral"
    n_layers = A.shape[1]

    # ---- held-out validation (3 topics held out for all emotions) ------------------------------
    rng = np.random.default_rng(cfg["seed"])
    n_topics = int(top.max()) + 1
    held = np.sort(rng.choice(n_topics, size=cfg["heldout_topics"], replace=False))
    acc_raw, n_held, k_raw = heldout_accuracy(A[S], emo[S], top[S], held, emotions)
    acc_den, _, k_den = heldout_accuracy(A[S], emo[S], top[S], held, emotions, A[N], top[N], cfg["pc_variance"])
    mid = slice(7, n_layers - 1)  # middle-late layers used for the variant rule
    variant = "denoised" if acc_den[mid].mean() >= acc_raw[mid].mean() - 0.02 else "raw"
    res["heldout"] = {
        "heldout_topic_ids": held.tolist(), "n_heldout_stories": n_held, "chance": 1 / len(emotions),
        "acc_raw": acc_raw.round(4).tolist(), "acc_denoised": acc_den.round(4).tolist(),
        "ci95_raw": [wilson_ci(int(k), n_held) for k in k_raw],
        "ci95_denoised": [wilson_ci(int(k), n_held) for k in k_den],
        "variant_rule": "denoised unless its mean held-out accuracy over layers 7-27 is >2 pp below raw",
        "variant": variant,
    }

    # supplementary, more stable estimate: leave-one-topic-out over all 10 topics (150 test stories)
    loto = {"raw": np.zeros(n_layers), "denoised": np.zeros(n_layers)}
    for t in range(n_topics):
        _, n_t, k_r = heldout_accuracy(A[S], emo[S], top[S], np.array([t]), emotions)
        _, _, k_d = heldout_accuracy(A[S], emo[S], top[S], np.array([t]), emotions, A[N], top[N], cfg["pc_variance"])
        loto["raw"] += k_r
        loto["denoised"] += k_d
    n_s = int(S.sum())
    res["heldout"]["loto_acc_raw"] = (loto["raw"] / n_s).round(4).tolist()
    res["heldout"]["loto_acc_denoised"] = (loto["denoised"] / n_s).round(4).tolist()
    res["heldout"]["loto_ci95_denoised"] = [wilson_ci(int(k), n_s) for k in loto["denoised"]]
    res["heldout"]["loto_n"] = n_s
    # confusion matrix (single held-out split) at the best middle-late layer of the chosen variant
    acc_v = acc_den if variant == "denoised" else acc_raw
    best = int(np.arange(n_layers)[mid][acc_v[mid].argmax()])
    tr = ~np.isin(top[S], held)
    Vb, mub = emotion_vectors(A[S][tr], emo[S][tr], emotions)
    if variant == "denoised":
        Vb = project_out(Vb, neutral_pc_bases(A[N][~np.isin(top[N], held)], cfg["pc_variance"]))
    pred = classify(A[S][~tr], mub, unit(Vb))[:, best]
    y = np.array([emotions.index(e) for e in emo[S][~tr]])
    res["heldout"]["confusion_layer"] = best
    res["heldout"]["confusion_rows_true_cols_pred"] = [[int(((y == i) & (pred == j)).sum()) for j in range(len(emotions))]
                                                       for i in range(len(emotions))]

    # ---- final vectors: all 30 stories per emotion, PCs from all neutral texts -------------------
    V, mu = emotion_vectors(A[S], emo[S], emotions)
    bases = neutral_pc_bases(A[N], cfg["pc_variance"])
    Vd = project_out(V, bases)
    V_final = unit(Vd if variant == "denoised" else V)
    np.savez_compressed(vdir / "emotion_vectors.npz", vectors=V_final.astype(np.float32), mu=mu.astype(np.float32),
                        raw=unit(V).astype(np.float32), denoised=unit(Vd).astype(np.float32),
                        emotions=np.array(emotions), variant=variant)
    res["pc_k_per_layer"] = [int(b.shape[1]) for b in bases]
    res["emotion_cos"] = {str(l): np.round(V_final[l] @ V_final[l].T, 3).tolist() for l in range(n_layers)}

    # ---- refusal direction (Arditi et al.): train harmful vs benign, neutral wrapper ------------
    if not args.from_cache:
        splits = load_splits(cfg)
        wr = load_wrappers(cfg)[cfg["refusal"]["wrapper"]]
        rows = []
        for k in ("harmful", "benign"):
            for r in load_behaviors(cfg, k):
                if splits[r["Index"]] in (cfg["refusal"]["fit_split"], cfg["refusal"]["check_split"]):
                    rows.append((k, r["Index"], splits[r["Index"]], wrap(wr, r["Goal"])))
        with T("prompt_acts_s"):
            P = prompt_acts(model, tok, processor, cfg, [x[3] for x in rows])
        np.savez_compressed(vdir / "prompt_acts_trainval.npz", acts=P.astype(np.float16),
                            kind=np.array([x[0] for x in rows]), split=np.array([x[2] for x in rows]),
                            index=np.array([x[1] for x in rows]))
    z = np.load(vdir / "prompt_acts_trainval.npz")
    P, pk, ps = z["acts"].astype(np.float32), z["kind"], z["split"]
    fit, chk = ps == cfg["refusal"]["fit_split"], ps == cfg["refusal"]["check_split"]
    m_h, m_b = P[fit & (pk == "harmful")].mean(0), P[fit & (pk == "benign")].mean(0)
    R = m_h - m_b
    R_unit = unit(R)
    np.savez_compressed(vdir / "refusal_direction.npz", direction=R_unit.astype(np.float32),
                        norm=np.linalg.norm(R, axis=-1).astype(np.float32),
                        mean_harmful=m_h.astype(np.float32), mean_benign=m_b.astype(np.float32))
    proj = np.einsum("nld,ld->nl", P, R_unit)
    res["refusal"] = {
        "n_fit": {"harmful": int((fit & (pk == "harmful")).sum()), "benign": int((fit & (pk == "benign")).sum())},
        "n_check": {"harmful": int((chk & (pk == "harmful")).sum()), "benign": int((chk & (pk == "benign")).sum())},
        "norm_per_layer": np.linalg.norm(R, axis=-1).round(2).tolist(),
        "val_auroc_per_layer": [round(auroc(proj[chk & (pk == "harmful"), l], proj[chk & (pk == "benign"), l]), 4)
                                for l in range(n_layers)],
        "cos_with_emotions": {e: np.round(np.einsum("ld,ld->l", V_final[:, i], R_unit), 3).tolist()
                              for i, e in enumerate(emotions)},
    }

    # ---- random baselines + topic-sensitivity preview on val prompts ----------------------------
    Rnd = random_unit_vectors(n_layers, cfg["random"]["n_per_layer"], A.shape[-1], cfg["random"]["seed"])
    res["random"] = {"seed": cfg["random"]["seed"], "n_per_layer": cfg["random"]["n_per_layer"],
                     "max_abs_cos_with_refusal_per_layer": np.abs(np.einsum("lnd,ld->ln", Rnd, R_unit)).max(1).round(3).tolist()}
    Xh, Xb = P[chk & (pk == "harmful")] - mu, P[chk & (pk == "benign")] - mu
    d_emo = {e: [round(cohens_d(Xh[:, l] @ V_final[l, i], Xb[:, l] @ V_final[l, i]), 3) for l in range(n_layers)]
             for i, e in enumerate(emotions)}
    d_rnd = np.array([[cohens_d(Xh[:, l] @ Rnd[l, j], Xb[:, l] @ Rnd[l, j]) for j in range(Rnd.shape[1])]
                      for l in range(n_layers)])
    res["topic_sensitivity_val"] = {
        "what": "Cohen's d of projection, harmful vs benign val prompts (neutral wrapper), last prompt token",
        "emotions": d_emo,
        "random_abs_d_p95_per_layer": np.percentile(np.abs(d_rnd), 95, axis=1).round(3).tolist(),
        "refusal_d_per_layer": [round(cohens_d(Xh[:, l] @ R_unit[l], Xb[:, l] @ R_unit[l]), 3) for l in range(n_layers)],
    }
    res["timings_s"] = T.t
    save_json(res, cfg["out"]["results_json"])
    print(json.dumps({"variant": variant, "acc_raw": res["heldout"]["acc_raw"], "acc_den": res["heldout"]["acc_denoised"],
                      "refusal_val_auroc": res["refusal"]["val_auroc_per_layer"], "timings": T.t}))


if __name__ == "__main__":
    main()
