"""Phase 1a: the model writes emotion stories (emotion felt, never named) and neutral texts.

Usage: python scripts/01_generate_stories.py --config configs/phase1.yaml [--limit-topics N]
Output: data/stories/stories.jsonl (benign texts) + results/phase1_stories_stats.json
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import chat_text, encode, generate, load_thinker, trim_generation  # noqa: E402
from src.utils import ROOT, Timer, env_info, load_config, save_json, set_seed  # noqa: E402


def clean(text: str) -> str:
    lines = [ln.strip() for ln in text.strip().splitlines()]
    lines = [ln for ln in lines if ln]
    if lines and (lines[0].lower().startswith("title") or (lines[0].startswith("**") and lines[0].endswith("**"))):
        lines = lines[1:]
    text = "\n".join(lines).strip()
    if len(text) > 1 and text[0] == text[-1] == '"' and text.count('"') == 2:
        text = text[1:-1].strip()
    return text


def banned_regex(words):
    return re.compile(r"\b(" + "|".join(map(re.escape, sorted(set(words), key=len, reverse=True))) + r")\b", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--limit-topics", type=int, default=None, help="smoke test: use only the first N topics")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    T = Timer()
    topics = cfg["topics"][: args.limit_topics]
    emotions = cfg["emotions"]
    all_banned = banned_regex([w for ws in cfg["banned_words"].values() for w in ws])
    banned = {e: banned_regex(cfg["banned_words"][e]) for e in emotions}
    gcfg = cfg["story_generation"]
    bs = cfg["generation"]["batch_size"]

    with T("load_s"):
        model, processor, tok = load_thinker(cfg)
    sysmsg = cfg["model"]["system_prompt"]

    # jobs: (kind, emotion, topic_id) -> how many texts still needed
    need = {("story", e, t): cfg["stories_per_topic"] for e in emotions for t in range(len(topics))}
    need.update({("neutral", "neutral", t): cfg["neutral_per_topic"] for t in range(len(topics))})
    kept, rejects = [], Counter()
    with T("generate_s"):
        for rnd in range(cfg["max_rounds"]):
            todo = [k for k, n in need.items() if n > 0]
            if not todo:
                break
            reqs = []
            for key in todo:
                kind, emo, t = key
                n_cand = cfg["candidates_per_topic"] if kind == "story" else cfg["neutral_per_topic"] + 2
                prompt = (cfg["story_prompt"].format(topic=topics[t], emotion=emo) if kind == "story"
                          else cfg["neutral_prompt"].format(topic=topics[t]))
                reqs += [(key, prompt)] * n_cand
            for b in range(0, len(reqs), bs):
                batch = reqs[b: b + bs]
                enc = encode(tok, [chat_text(processor, p, sysmsg) for _, p in batch], model.device)
                gen = generate(model, tok, enc, gcfg, seed=cfg["seed"] + 1000 * rnd + b)
                for (key, _), row in zip(batch, gen):
                    kind, emo, t = key
                    if need[key] <= 0:
                        continue
                    ids = trim_generation(tok, row)
                    text = clean(tok.decode(ids, skip_special_tokens=True))
                    n_words = len(text.split())
                    hit_stop = ids and ids[-1] in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id)
                    rx = banned[emo] if kind == "story" else all_banned
                    if not hit_stop:
                        rejects["truncated"] += 1
                    elif not (cfg["min_words"] <= n_words <= cfg["max_words"]):
                        rejects["length"] += 1
                    elif rx.search(text):
                        rejects[f"names_emotion:{emo}"] += 1
                    else:
                        need[key] -= 1
                        kept.append({"kind": kind, "emotion": emo, "topic_id": t, "topic": topics[t],
                                     "round": rnd, "n_words": n_words, "text": text})
            print(f"round {rnd}: kept {len(kept)}, still needed {sum(max(n, 0) for n in need.values())}, rejects {dict(rejects)}")

    for i, r in enumerate(kept):
        r["id"] = f"{r['kind']}-{r['emotion']}-t{r['topic_id']}-{i}"
    out = ROOT / cfg["out"]["stories"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in sorted(kept, key=lambda r: (r["kind"], r["emotion"], r["topic_id"])):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {
        "env": env_info(cfg["model"]["id"], cfg["model"]["revision"]),
        "n_topics": len(topics),
        "kept_per_emotion": dict(Counter(r["emotion"] for r in kept)),
        "missing": {f"{k[0]}|{k[1]}|{k[2]}": n for k, n in need.items() if n > 0},
        "rejects": dict(rejects),
        "mean_words": {e: round(sum(r["n_words"] for r in kept if r["emotion"] == e) /
                                max(1, sum(r["emotion"] == e for r in kept)), 1)
                       for e in emotions + ["neutral"]},
        "timings_s": T.t,
    }
    save_json(stats, "results/phase1_stories_stats.json")
    print(json.dumps({k: v for k, v in stats.items() if k != "env"}, indent=1))


if __name__ == "__main__":
    main()
