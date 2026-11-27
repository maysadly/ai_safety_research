"""Phase 6B: generate answers while steering the residual stream along emotion directions (pre-registered in
results/phase6_preregistration.md).

Usage: python scripts/15_steer.py --config configs/phase6.yaml --model qwen|mistral [--limit N]
Output: data/raw/p6_{model}/generations.jsonl (git-ignored) with one row per (prompt, condition, sample)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data import load_behaviors, load_splits, load_wrappers, wrap  # noqa: E402
from src.labels import keyword_refusal  # noqa: E402
from src.layers import fixed_layers  # noqa: E402
from src.model import chat_text, encode, generate, load_thinker, stop_ids, text_config, trim_generation  # noqa: E402
from src.utils import ROOT, Timer, env_info, load_config, save_json, set_seed  # noqa: E402


def decoder_layers(model):
    for path in ("model.layers", "model.language_model.layers", "language_model.model.layers"):
        obj = model
        try:
            for a in path.split("."):
                obj = getattr(obj, a)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("decoder layers not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase6.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N behaviours per kind")
    args = ap.parse_args()
    cfg = load_config(args.config)
    mc = cfg["models"][args.model]
    cfg["model"] = mc["model"]
    set_seed(cfg["seed"])
    T = Timer()
    L_emo, _ = fixed_layers(ROOT / mc["phase1_json"])
    z = np.load(ROOT / mc["vectors_dir"] / "emotion_vectors.npz")
    emos = [str(e) for e in z["emotions"]]
    V = z["denoised"][L_emo]  # [E, d] unit vectors at the steering layer
    sd = cfg["steering"]["sd_desperate"][args.model]
    d = V.shape[1]
    rng = np.random.default_rng(cfg["steering"]["random_seed"])
    u_rand = rng.standard_normal(d).astype(np.float32)
    u_rand /= np.linalg.norm(u_rand)
    vec = {"desperate": V[emos.index("desperate")], "calm": V[emos.index("calm")], "random": u_rand}
    conditions = cfg["steering"]["conditions"]  # name -> [direction or null, sign, strength in SD]

    splits, wrappers = load_splits(cfg), load_wrappers(cfg)
    prompts = []
    for kind in ("harmful", "benign"):
        for r in load_behaviors(cfg, kind)[: args.limit]:
            for w in cfg["steering"]["wrappers"]:
                prompts.append({"pid": f"{kind}-{r['Index']}-{w}", "kind": kind, "index": r["Index"], "category": r["Category"],
                                "split": splits[r["Index"]], "wrapper": w, "goal": r["Goal"], "prompt": wrap(wrappers[w], r["Goal"])})
    S = cfg["steering"]["samples_per_prompt"]
    with T("load_s"):
        model, processor, tok = load_thinker(cfg)
    layer = decoder_layers(model)[L_emo - 1]  # hidden_states[L] is the output of block L
    dev, dtype = model.device, next(model.parameters()).dtype
    chat = [chat_text(processor, p["prompt"], cfg["model"]["system_prompt"]) for p in prompts]
    jobs = [(pi, s) for pi in range(len(prompts)) for s in range(S)]
    stops = set(stop_ids(tok))
    out_dir = ROOT / cfg["out"]["raw_dir"].format(tag=f"p6_{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    fout = open(out_dir / "generations.jsonl", "w")
    state = {"delta": None}

    def hook(_m, _inp, output):
        if state["delta"] is None:
            return output
        if isinstance(output, tuple):
            return (output[0] + state["delta"],) + tuple(output[1:])
        return output + state["delta"]

    handle = layer.register_forward_hook(hook)
    bs = cfg["generation"]["batch_size"]
    meta = {}
    with T("generate_s"):
        for ci, (cname, (direction, sign, strength)) in enumerate(conditions.items()):
            if direction is None:
                state["delta"] = None
                alpha = 0.0
            else:
                alpha = sign * strength * sd
                state["delta"] = torch.tensor(alpha * vec[direction], device=dev, dtype=dtype)
            meta[cname] = {"direction": direction, "alpha": alpha}
            for bi, b in enumerate(range(0, len(jobs), bs)):
                batch = jobs[b: b + bs]
                enc = encode(tok, [chat[pi] for pi, _ in batch], dev)
                gen = generate(model, tok, enc, cfg["generation"], seed=cfg["seed"] + 1000 * ci + bi)
                for (pi, s), row in zip(batch, gen):
                    g = trim_generation(tok, row)
                    text = tok.decode(g, skip_special_tokens=True)
                    p = prompts[pi]
                    toks = text.split()
                    bigrams = list(zip(toks, toks[1:]))
                    fout.write(json.dumps({"gid": f"{p['pid']}-{cname}-s{s}", "pid": p["pid"], "condition": cname, "sample": s,
                                           **{k: p[k] for k in ("kind", "index", "category", "split", "wrapper", "goal", "prompt")},
                                           "response": text, "n_gen_tokens": len(g), "hit_stop": bool(g and g[-1] in stops),
                                           "keyword_refusal": keyword_refusal(text),
                                           "distinct2": round(len(set(bigrams)) / max(1, len(bigrams)), 4)}, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{args.model}] condition {cname} (alpha {alpha:+.3f}) done; {T.t}", flush=True)
    handle.remove()
    fout.close()
    save_json({"env": env_info(cfg["model"]["id"], cfg["model"]["revision"]), "layer": L_emo, "sd_desperate": sd,
               "conditions": meta, "n_prompts": len(prompts), "samples": S, "timings_s": T.t,
               "hidden_size": text_config(model).hidden_size},
              ROOT / f"results/phase6_steer_meta_{args.model}.json")
    print("done", T.t)


if __name__ == "__main__":
    main()
