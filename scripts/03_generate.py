"""Phase 2a: sample responses for every (behaviour x wrapper), then a teacher-forced pass that projects
the residual stream (last prompt token + generated tokens 1..20, all layers) onto the emotion vectors,
the refusal direction and the random directions.

Usage: python scripts/03_generate.py --config configs/phase2.yaml [--tag smoke --limit 2]
Outputs (git-ignored):
  data/raw/{tag}/generations.jsonl          prompts + responses (text never committed)
  data/activations/{tag}/projections.npy    float32 [N, L+1, 21, K]
  data/activations/{tag}/prompt_resid.npy   float16 [n_prompts, L+1, d] (last prompt token)
  data/activations/{tag}/meta.json          row order, direction names, prompt ids
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
from src.model import (chat_text, encode, generate, left_pad, load_thinker, projections_at,  # noqa: E402
                       stop_ids, text_config, trim_generation)
from src.utils import ROOT, Timer, env_info, load_config, save_json, set_seed  # noqa: E402
from src.vectors import random_unit_vectors  # noqa: E402


def directions(cfg, n_layers, d):
    z = np.load(ROOT / cfg["vectors_dir"] / "emotion_vectors.npz")
    emos = [str(e) for e in z["emotions"]]
    r = np.load(ROOT / cfg["vectors_dir"] / "refusal_direction.npz")["direction"]
    rnd = random_unit_vectors(n_layers, cfg["random"]["n_per_layer"], d, cfg["random"]["seed"])
    D = np.concatenate([z["raw"], z["denoised"], r[:, None], rnd], axis=1).astype(np.float32)
    names = ([f"emo_raw:{e}" for e in emos] + [f"emo_den:{e}" for e in emos] + ["refusal"] +
             [f"random:{i}" for i in range(rnd.shape[1])])
    assert D.shape == (n_layers, len(names), d), D.shape
    return D, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase2.yaml")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N behaviours per kind")
    ap.add_argument("--source", choices=["jbb", "pair", "phase5"], default="jbb",
                    help="jbb = JBB goals x fixed wrappers (pre-registered); pair = JailbreakBench's published "
                         "PAIR artifacts (vicuna-13b-v1.5), harmful only (exploratory extension)")
    ap.add_argument("--samples", type=int, default=None, help="override samples per prompt")
    ap.add_argument("--model", default=None, help="phase5: key in cfg['models'] (qwen | mistral)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    l_emo = None
    if args.model:  # Phase 5: select the model block; random directions are stored only at its fixed emotion layer
        from src.layers import fixed_layers
        mc = cfg["models"][args.model]
        cfg["model"], cfg["vectors_dir"] = mc["model"], mc["vectors_dir"]
        l_emo, l_ref = fixed_layers(ROOT / mc["phase1_json"])
        print(f"model {args.model}: fixed layers emotion {l_emo}, refusal {l_ref}")
    set_seed(cfg["seed"])
    T = Timer()
    raw_dir = ROOT / cfg["out"]["raw_dir"].format(tag=args.tag)
    act_dir = ROOT / cfg["out"]["act_dir"].format(tag=args.tag)
    raw_dir.mkdir(parents=True, exist_ok=True)
    act_dir.mkdir(parents=True, exist_ok=True)

    # ---- prompts ---------------------------------------------------------------------------------
    splits, wrappers = load_splits(cfg), load_wrappers(cfg)
    prompts = []
    if args.source == "phase5":
        prompts = [json.loads(l) for l in open(ROOT / cfg["prompts"]["out_jsonl"])][: args.limit]
    elif args.source == "jbb":
        for kind in ("harmful", "benign"):
            for r in load_behaviors(cfg, kind)[: args.limit]:
                for w, tpl in wrappers.items():
                    prompts.append({"pid": f"{kind}-{r['Index']}-{w}", "kind": kind, "index": r["Index"],
                                    "category": r["Category"], "split": splits[r["Index"]], "wrapper": w,
                                    "goal": r["Goal"], "prompt": wrap(tpl, r["Goal"])})
    else:
        harm = {r["Index"]: r for r in load_behaviors(cfg, "harmful")}
        art = json.load(open(ROOT / cfg["pair"]["artifact_file"]))
        for j in art["jailbreaks"][: args.limit]:
            if not j.get("prompt"):
                continue
            r = harm[int(j["index"])]
            if r["Goal"] != j["goal"]:  # artifact made with an older goal wording -> skip (3 of 82)
                print("skip PAIR index", j["index"], "(goal text differs from current JBB)")
                continue
            prompts.append({"pid": f"harmful-{r['Index']}-pair", "kind": "harmful", "index": r["Index"],
                            "category": r["Category"], "split": splits[r["Index"]], "wrapper": "pair",
                            "goal": r["Goal"], "prompt": j["prompt"]})
    S = args.samples or cfg["generation"]["samples_per_prompt"]
    jobs = [(pi, s) for pi in range(len(prompts)) for s in range(S)]
    print(f"{len(prompts)} prompts x {S} samples = {len(jobs)} generations")

    with T("load_s"):
        model, processor, tok = load_thinker(cfg)
    sysmsg = cfg["model"]["system_prompt"]
    chat = [chat_text(processor, p["prompt"], sysmsg) for p in prompts]
    prompt_ids = [tok(c, add_special_tokens=False)["input_ids"] for c in chat]

    n_layers = text_config(model).num_hidden_layers + 1
    d = text_config(model).hidden_size
    D_np, names = directions(cfg, n_layers, d)
    D = torch.tensor(D_np, device=model.device)
    P = cfg["activations"]["n_gen_positions"] + 1
    compact = args.source == "phase5"
    n_main = len(names) - cfg["random"]["n_per_layer"]  # emotion raw + denoised + refusal
    if compact:  # main directions at every layer; random directions only at the fixed emotion layer
        proj = np.lib.format.open_memmap(act_dir / "proj_main.npy", mode="w+", dtype=np.float32,
                                         shape=(len(jobs), n_layers, P, n_main))
        proj_rand = np.lib.format.open_memmap(act_dir / "proj_rand.npy", mode="w+", dtype=np.float32,
                                              shape=(len(jobs), P, len(names) - n_main))
    else:
        proj = np.lib.format.open_memmap(act_dir / "projections.npy", mode="w+", dtype=np.float32,
                                         shape=(len(jobs), n_layers, P, len(names)))
    presid = np.zeros((len(prompts), n_layers, d), dtype=np.float16)
    stops = set(stop_ids(tok))

    bs = cfg["generation"]["batch_size"]
    rows = []
    with T("generate_and_project_s"):
        for bi, b in enumerate(range(0, len(jobs), bs)):
            batch = jobs[b: b + bs]
            enc = encode(tok, [chat[pi] for pi, _ in batch], model.device)
            gen = generate(model, tok, enc, cfg["generation"], seed=cfg["seed"] + bi)
            gids = [trim_generation(tok, row) for row in gen]
            # teacher-forced pass on prompt + generated tokens (stop token kept)
            seqs = [prompt_ids[pi] + g for (pi, _), g in zip(batch, gids)]
            ids, am, offs = left_pad(seqs, tok.pad_token_id)
            pos = torch.zeros((len(batch), P), dtype=torch.long)
            for i, ((pi, _), g) in enumerate(zip(batch, gids)):
                last_prompt = offs[i] + len(prompt_ids[pi]) - 1
                last = offs[i] + len(seqs[i]) - 1
                pos[i] = torch.clamp(last_prompt + torch.arange(P), max=last)
            pr, first = projections_at(model, ids.to(model.device), am.to(model.device), pos, D)
            if compact:
                proj[b: b + len(batch)] = pr[..., :n_main].numpy()
                proj_rand[b: b + len(batch)] = pr[:, l_emo, :, n_main:].numpy()
            else:
                proj[b: b + len(batch)] = pr.numpy()
            for i, ((pi, s), g) in enumerate(zip(batch, gids)):
                if s == 0:
                    presid[pi] = first[i].numpy()
                text = tok.decode(g, skip_special_tokens=True)
                p = prompts[pi]
                rows.append({"gid": f"{p['pid']}-s{s}", "row": b + i, "pid": p["pid"], "sample": s,
                             **{k: p[k] for k in ("kind", "index", "category", "split", "wrapper", "goal", "prompt")},
                             **{k: p[k] for k in ("dataset", "group") if k in p},
                             "response": text, "n_gen_tokens": len(g), "hit_stop": bool(g and g[-1] in stops),
                             "keyword_refusal": keyword_refusal(text),
                             "first_text": {k: tok.decode(g[:k], skip_special_tokens=True) for k in (1, 2, 5, 10, 20)}})
            if bi % 5 == 0:
                print(f"batch {bi + 1}/{-(-len(jobs) // bs)}  {T.t}", flush=True)
    proj.flush()
    if compact:
        proj_rand.flush()
    else:
        np.save(act_dir / "prompt_resid.npy", presid)
    with open(raw_dir / "generations.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    save_json({"env": env_info(cfg["model"]["id"], cfg["model"]["revision"]), "model_key": args.model,
               "directions": names[:n_main] if compact else names, "random_layer": l_emo,
               "random_directions": names[n_main:] if compact else None,
               "positions": "0 = last prompt token, k = generated token k (clamped to the last token)",
               "layers": "0 = embeddings, i = block i output, last = after final norm",
               "rows": [{"gid": r["gid"], "row": r["row"], "n_gen_tokens": r["n_gen_tokens"]} for r in rows],
               "prompt_ids": [p["pid"] for p in prompts], "timings_s": T.t,
               "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)},
              act_dir / "meta.json")
    n_stop = sum(r["hit_stop"] for r in rows)
    print(f"done: {len(rows)} generations, {n_stop} hit stop token; timings {T.t}")


if __name__ == "__main__":
    main()
