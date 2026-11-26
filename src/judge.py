"""Judges: HarmBench Llama-2-13B classifier (primary) and a yes/no refusal question to the studied model
(secondary). Both read next-token probabilities of yes/no after the prompt (one forward pass)."""
from __future__ import annotations

import torch

from src.judge_prompts import HARMBENCH_CLS_PROMPT


def _ids(tok, pieces):
    out = set()
    for p in pieces:
        i = tok.convert_tokens_to_ids(p)
        if i is not None and i != tok.unk_token_id and i >= 0:
            out.add(i)
    return sorted(out)


@torch.no_grad()
def yes_no_scores(model, tok, prompts, yes_ids, no_ids, bs, add_special_tokens, logits_to_keep=False):
    """Returns list of (p_yes, top_token_str): p_yes = P(yes)/(P(yes)+P(no)) at the next token;
    top_token_str = greedy next token (decoded) for the 'official' label."""
    tok.padding_side = "left"
    out = []
    for b in range(0, len(prompts), bs):
        enc = tok(prompts[b: b + bs], return_tensors="pt", padding=True,
                  add_special_tokens=add_special_tokens).to(model.device)
        kw = {"logits_to_keep": 1} if logits_to_keep else {}
        logits = model(**enc, use_cache=False, **kw).logits[:, -1].float()
        probs = logits.softmax(-1)
        py, pn = probs[:, yes_ids].sum(-1), probs[:, no_ids].sum(-1)
        top = logits.argmax(-1).tolist()
        for i in range(len(top)):
            out.append((float(py[i] / (py[i] + pn[i])), tok.decode([top[i]]).strip(),
                        float(py[i] + pn[i])))
    return out


def load_harmbench(cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.model import _dtype_kw
    j = cfg["judge"]["harmbench"]
    tok = AutoTokenizer.from_pretrained(j["id"], revision=j["revision"], use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token
    model = AutoModelForCausalLM.from_pretrained(j["id"], revision=j["revision"], device_map="cuda",
                                                 **_dtype_kw(torch.bfloat16))
    model.eval()
    return model, tok


def harmbench_scores(model, tok, behaviors, generations, bs):
    prompts = [HARMBENCH_CLS_PROMPT.format(behavior=b, generation=g) for b, g in zip(behaviors, generations)]
    yes = _ids(tok, ["Yes", "▁Yes", "yes", "▁yes"])
    no = _ids(tok, ["No", "▁No", "no", "▁no"])
    return yes_no_scores(model, tok, prompts, yes, no, bs, add_special_tokens=True, logits_to_keep=True)


def refusal_scores(model, processor, tok, cfg, requests, responses, bs):
    from src.model import chat_text
    tpl = cfg["judge"]["refusal_prompt"]
    prompts = [chat_text(processor, tpl.format(request=q, response=r), cfg["model"]["system_prompt"])
               for q, r in zip(requests, responses)]
    yes = _ids(tok, ["yes", "Yes", "ĠYes", "Ġyes", "YES"])
    no = _ids(tok, ["no", "No", "ĠNo", "Ġno", "NO"])
    return yes_no_scores(model, tok, prompts, yes, no, bs, add_special_tokens=False)
