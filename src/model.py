"""Qwen2.5-Omni thinker (text path only, talker never loaded): loading, chat formatting,
sampling and residual-stream extraction."""
from __future__ import annotations

import torch


def _dtype_kw(dtype):
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    # `torch_dtype` was renamed to `dtype` in transformers 4.56.
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}


def load_thinker(cfg: dict):
    """Load only the thinker (Qwen2_5OmniThinkerForConditionalGeneration) as in the HF docs;
    this skips the talker / token2wav weights entirely."""
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

    m = cfg["model"]
    processor = Qwen2_5OmniProcessor.from_pretrained(m["id"], revision=m["revision"])
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        m["id"], revision=m["revision"], device_map="cuda",
        attn_implementation=m["attn_implementation"], **_dtype_kw(getattr(torch, m["dtype"])))
    model.eval()
    tok = processor.tokenizer
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, processor, tok


def text_config(model):
    c = model.config
    return getattr(c, "text_config", c)


def chat_text(processor, user_text: str, system_prompt: str | None) -> str:
    conv = []
    if system_prompt:
        conv.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    conv.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
    return processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)


def stop_ids(tok) -> list[int]:
    ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    return sorted(i for i in ids if i is not None and i >= 0)


def encode(tok, texts: list[str], device):
    return tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)


@torch.no_grad()
def generate(model, tok, enc, gen_cfg: dict, seed: int, do_sample: bool = True):
    """Returns generated token ids [B, T_gen] (right-padded with pad_token_id after stop)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    kw = dict(max_new_tokens=gen_cfg["max_new_tokens"], pad_token_id=tok.pad_token_id,
              eos_token_id=stop_ids(tok))
    if do_sample:
        kw.update(do_sample=True, temperature=gen_cfg["temperature"], top_p=gen_cfg["top_p"],
                  top_k=gen_cfg.get("top_k", 0))
    else:
        kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    out = model.generate(**enc, **kw)
    return out[:, enc["input_ids"].shape[1]:]


def trim_generation(tok, gen_row: torch.Tensor) -> list[int]:
    """Drop padding after the first stop token (stop token kept). A row without a stop token
    ran to max_new_tokens and therefore contains no padding."""
    ids = gen_row.tolist()
    stops = set(stop_ids(tok))
    for i, t in enumerate(ids):
        if t in stops:
            return ids[: i + 1]
    return ids


def left_pad(seqs: list[list[int]], pad_id: int):
    """Left-pad token lists. Returns (ids [B, T], attention_mask [B, T], offsets [B])."""
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    am = torch.zeros((len(seqs), T), dtype=torch.long)
    offs = []
    for i, s in enumerate(seqs):
        off = T - len(s)
        ids[i, off:] = torch.tensor(s, dtype=torch.long)
        am[i, off:] = 1
        offs.append(off)
    return ids, am, offs


@torch.no_grad()
def residual_mean(model, input_ids, attention_mask, starts, ends):
    """Mean residual stream over token range [start, end) per row, at every layer.
    Returns float32 CPU tensor [B, L+1, d]."""
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True, use_cache=False)
    dev = input_ids.device
    ar = torch.arange(input_ids.shape[1], device=dev)[None]
    w = ((ar >= starts.to(dev)[:, None]) & (ar < ends.to(dev)[:, None])).float()
    w = w / w.sum(1, keepdim=True)
    return torch.stack([torch.einsum("bt,btd->bd", w, h.float()) for h in out.hidden_states], 1).cpu()


@torch.no_grad()
def residual_at(model, input_ids, attention_mask, positions):
    """Residual stream at every layer (index 0 = embeddings, i = output of block i; the last
    entry is after the final RMSNorm, as returned by HF) at the given token positions.

    positions: LongTensor [B, P] of indices into the (padded) sequence.
    Returns float32 CPU tensor [B, L+1, P, d]."""
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    pos = positions.to(hs[0].device)
    picked = [h.gather(1, pos[:, :, None].expand(-1, -1, h.shape[-1])) for h in hs]  # [B, P, d] each
    return torch.stack(picked, dim=1).float().cpu()
