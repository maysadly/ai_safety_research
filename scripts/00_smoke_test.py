"""Phase 0 smoke test: model loading (thinker only), chat template, 3-prompt generation,
padding / teacher-forcing sanity checks, hidden-state extraction and throughput timing.

Usage: python scripts/00_smoke_test.py --config configs/base.yaml
Outputs:
  data/raw/smoke/generations.jsonl   raw text (git-ignored, never committed)
  results/phase0_smoke.json          IDs, lengths, refusal flags, timings, env (no completions)
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.data import load_behaviors, load_splits, load_wrappers, wrap  # noqa: E402
from src.labels import keyword_refusal  # noqa: E402
from src.model import (chat_text, encode, generate, load_thinker, residual_at,  # noqa: E402
                       text_config, trim_generation)
from src.utils import Timer, append_jsonl, env_info, load_config, save_json, set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    cfg = load_config(ap.parse_args().config)
    set_seed(cfg["seed"])
    T = Timer()
    res = {"env": env_info(cfg["model"]["id"], cfg["model"]["revision"])}
    raw_path = Path(cfg["paths"]["raw_dir"]) / "smoke" / "generations.jsonl"
    (Path(__file__).resolve().parents[1] / raw_path).unlink(missing_ok=True)

    # ---- 1. load ---------------------------------------------------------------------------
    with T("load_model_s"):
        model, processor, tok = load_thinker(cfg)
    tc = text_config(model)
    res["model"] = {
        "class": type(model).__name__,
        "n_layers": tc.num_hidden_layers, "hidden_size": tc.hidden_size,
        "n_params_b": round(sum(p.numel() for p in model.parameters()) / 1e9, 2),
        "has_talker_attr": hasattr(model, "talker"),
        "gpu_mem_after_load_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "generation_config": {k: v for k, v in model.generation_config.to_dict().items()
                              if k in ("do_sample", "temperature", "top_p", "top_k",
                                       "eos_token_id", "pad_token_id", "repetition_penalty")},
        "pad_token": tok.pad_token, "eos_token": tok.eos_token, "padding_side": tok.padding_side,
    }
    print(res["model"])

    # ---- 2. chat template --------------------------------------------------------------------
    sysmsg = cfg["model"]["system_prompt"]
    demo_user = "Hello! What is the capital of France?"
    res["template"] = {
        "with_system": chat_text(processor, demo_user, sysmsg),
        "without_system": chat_text(processor, demo_user, None),
    }
    res["template"]["auto_adds_system_when_missing"] = "<|im_start|>system" in res["template"]["without_system"]
    print(res["template"]["with_system"])

    # ---- 3. three smoke prompts --------------------------------------------------------------
    splits = load_splits(cfg)
    idx = min(i for i, s in splits.items() if s == "train")
    harm = {r["Index"]: r for r in load_behaviors(cfg, "harmful")}[idx]
    ben = {r["Index"]: r for r in load_behaviors(cfg, "benign")}[idx]
    wr = load_wrappers(cfg)
    prompts = [
        {"id": f"benign-{idx}-neutral", "kind": "benign", "wrapper": "neutral", "text": wrap(wr["neutral"], ben["Goal"])},
        {"id": f"harmful-{idx}-neutral", "kind": "harmful", "wrapper": "neutral", "text": wrap(wr["neutral"], harm["Goal"])},
        {"id": f"harmful-{idx}-emotional_pressure", "kind": "harmful", "wrapper": "emotional_pressure",
         "text": wrap(wr["emotional_pressure"], harm["Goal"])},
    ]
    texts = [chat_text(processor, p["text"], sysmsg) for p in prompts]
    enc = encode(tok, texts, model.device)
    with T("smoke_generate_3_s"):
        gen = generate(model, tok, enc, cfg["generation"], seed=cfg["seed"])
    smoke = []
    raw = []
    for p, row in zip(prompts, gen):
        ids = trim_generation(tok, row)
        out = tok.decode(ids, skip_special_tokens=True)
        raw.append({**p, "response": out})
        smoke.append({"id": p["id"], "kind": p["kind"], "wrapper": p["wrapper"], "n_tokens": len(ids),
                      "keyword_refusal": keyword_refusal(out)})
        # Only benign output is printed; harmful completions stay in data/raw.
        print(p["id"], smoke[-1], (out[:200] + "...") if p["kind"] == "benign" else "")
    append_jsonl(raw, raw_path)
    res["smoke_prompts"] = {"behaviour_index": idx, "category": harm["Category"], "results": smoke}

    # ---- 4. padding sanity: single vs left-padded batch (greedy) -----------------------------
    gcfg = dict(cfg["generation"], max_new_tokens=32)
    single_tokens, single_h = [], []
    for t in texts:
        e1 = encode(tok, [t], model.device)
        g1 = generate(model, tok, e1, gcfg, seed=0, do_sample=False)[0]
        single_tokens.append(trim_generation(tok, g1))
        pos = torch.tensor([[e1["input_ids"].shape[1] - 1]])
        single_h.append(residual_at(model, e1["input_ids"], e1["attention_mask"], pos)[0, :, 0])
    gb = generate(model, tok, enc, gcfg, seed=0, do_sample=False)
    posb = torch.full((len(texts), 1), enc["input_ids"].shape[1] - 1)
    hb = residual_at(model, enc["input_ids"], enc["attention_mask"], posb)[:, :, 0]  # [B, L+1, d]
    pad_check = []
    for i in range(len(texts)):
        bt = trim_generation(tok, gb[i])
        n = min(len(bt), len(single_tokens[i]))
        first_div = next((k for k in range(n) if bt[k] != single_tokens[i][k]), n)
        cos = torch.nn.functional.cosine_similarity(hb[i], single_h[i], dim=-1)  # per layer
        pad_check.append({"id": prompts[i]["id"], "greedy_tokens_identical_prefix": first_div,
                          "of": n, "min_cos_last_prompt_tok_over_layers": round(cos.min().item(), 5)})
    res["padding_check"] = pad_check
    print("padding check", pad_check)

    # ---- 5. teacher forcing: prompt + generated tokens -> residuals at 21 positions ----------
    P = cfg["activations"]["n_gen_positions"]
    seqs, prompt_lens = [], []
    for t, row in zip(texts, gen):
        p_ids = tok(t, add_special_tokens=False)["input_ids"]
        g_ids = trim_generation(tok, row)
        seqs.append(p_ids + g_ids)
        prompt_lens.append((len(p_ids), len(g_ids)))
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), tok.pad_token_id)
    am = torch.zeros((len(seqs), L), dtype=torch.long)
    positions = torch.zeros((len(seqs), P + 1), dtype=torch.long)
    for i, s in enumerate(seqs):
        off = L - len(s)  # left padding
        ids[i, off:] = torch.tensor(s)
        am[i, off:] = 1
        pl, gl = prompt_lens[i]
        last_prompt = off + pl - 1
        positions[i] = torch.tensor([min(last_prompt + k, off + pl + gl - 1) for k in range(P + 1)])
    with T("teacher_forcing_3_s"):
        H = residual_at(model, ids.to(model.device), am.to(model.device), positions)
    # consistency: residual at the last prompt token must equal the prompt-only pass (causality)
    tf_cos = torch.nn.functional.cosine_similarity(H[:, :, 0], hb, dim=-1)
    res["teacher_forcing"] = {
        "residual_shape_[B,L+1,P,d]": list(H.shape),
        "min_cos_vs_prompt_only_pass": round(tf_cos.min().item(), 5),
        "residual_norm_by_layer_mean": [round(x, 1) for x in H[:, :, 0].norm(dim=-1).mean(0).tolist()],
    }
    print(res["teacher_forcing"])

    # ---- 6. throughput: batched sampling + teacher forcing on benign prompts -----------------
    bs = cfg["generation"]["batch_size"]
    benign = load_behaviors(cfg, "benign")
    tp_texts = []
    for name, tpl in wr.items():
        for r in benign:
            tp_texts.append(chat_text(processor, wrap(tpl, r["Goal"]), sysmsg))
    tp_texts = tp_texts[: cfg["smoke"]["throughput_n_prompts"]]
    torch.cuda.reset_peak_memory_stats()
    e = encode(tok, tp_texts[:bs], model.device)
    torch.cuda.synchronize()
    with T("throughput_generate_batch_s"):
        g = generate(model, tok, e, cfg["generation"], seed=cfg["seed"])
        torch.cuda.synchronize()
    n_gen_tokens = sum(len(trim_generation(tok, r)) for r in g)
    append_jsonl([{"id": f"throughput-{i}", "kind": "benign", "response": tok.decode(trim_generation(tok, r), skip_special_tokens=True)}
                  for i, r in enumerate(g)], raw_path)
    full = torch.cat([e["input_ids"], g], dim=1)
    full_am = torch.cat([e["attention_mask"], (g != tok.pad_token_id).long()], dim=1)
    pos = (e["input_ids"].shape[1] - 1 + torch.arange(P + 1))[None].expand(full.shape[0], -1)
    with T("throughput_teacher_forcing_batch_s"):
        residual_at(model, full, full_am, pos)
        torch.cuda.synchronize()
    n_main = 200 * len(wr) * cfg["generation"]["samples_per_prompt"]
    n_batches = math.ceil(n_main / bs)
    per_batch = T.t["throughput_generate_batch_s"] + T.t["throughput_teacher_forcing_batch_s"]
    res["throughput"] = {
        "batch_size": len(tp_texts[:bs]), "generated_tokens": n_gen_tokens,
        "tokens_per_s": round(n_gen_tokens / T.t["throughput_generate_batch_s"], 1),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "phase2_generations": n_main,
        "phase2_est_generation_plus_tf_min": round(n_batches * per_batch / 60, 1),
    }
    res["timings_s"] = T.t
    print(res["throughput"], T.t)
    out = save_json(res, Path(cfg["paths"]["results_dir"]) / "phase0_smoke.json")
    print("saved", out)


if __name__ == "__main__":
    main()
