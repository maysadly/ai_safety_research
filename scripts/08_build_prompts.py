"""Phase 5: assemble the pre-registered prompt sets D1-D3 from public benchmarks (no new prompts).

D1  JBB behaviours (100 harmful + 100 benign) x 3 fixed wrappers
D2  JailbreakBench black-box attack artifacts (PAIR, prompt_with_random_search; 4 targets each)
D3  HarmBench standard behaviours: direct + 2 human-jailbreak templates chosen exactly as HarmBench's
    HumanJailbreaks(random_subset=2, seed=1) does; behaviours duplicating a JBB goal are dropped.

Usage: python scripts/08_build_prompts.py --config configs/phase5.yaml
Outputs: data/raw/phase5/prompts.jsonl (text, git-ignored), data/phase5_prompts_meta.csv (IDs only)
"""
import argparse
import ast
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_behaviors, load_splits, load_wrappers, wrap  # noqa: E402
from src.utils import ROOT, load_config  # noqa: E402

SHORT = {"PAIR": "pair", "prompt_with_random_search": "prs", "gpt-3.5-turbo-1106": "gpt35",
         "gpt-4-0125-preview": "gpt4", "llama-2-7b-chat-hf": "llama2", "vicuna-13b-v1.5": "vicuna"}


def fix_unicode(t: str) -> str:
    """Some random-search artifacts contain UTF-16 surrogates; join valid pairs, replace lone ones with U+FFFD."""
    return t.encode("utf-16", "surrogatepass").decode("utf-16", "replace")


def words(t):
    return set(re.findall(r"[a-z0-9]+", t.lower()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase5.yaml")
    cfg = load_config(ap.parse_args().config)
    P = cfg["prompts"]
    splits, wrappers = load_splits(cfg), load_wrappers(cfg)
    harm = {r["Index"]: r for r in load_behaviors(cfg, "harmful")}
    rows = []

    # D1
    for kind in ("harmful", "benign"):
        for r in load_behaviors(cfg, kind):
            for w, tpl in wrappers.items():
                rows.append({"pid": f"{kind}-{r['Index']}-{w}", "dataset": "D1_jbb", "kind": kind,
                             "group": f"jbb:{r['Index']}", "index": r["Index"], "category": r["Category"],
                             "split": splits[r["Index"]], "wrapper": w, "goal": r["Goal"], "prompt": wrap(tpl, r["Goal"])})
    # D2
    skipped = Counter()
    for m in P["jbb_artifacts"]["methods"]:
        for t in P["jbb_artifacts"]["targets"]:
            art = json.load(open(ROOT / P["jbb_artifacts"]["dir"] / f"{m}_{t}.json"))
            for j in art["jailbreaks"]:
                if not j.get("prompt"):
                    skipped[f"{m}:{t}:no_prompt"] += 1
                    continue
                r = harm[int(j["index"])]
                if r["Goal"] != j["goal"]:
                    skipped[f"{m}:{t}:goal_mismatch"] += 1
                    continue
                w = f"{SHORT[m]}_{SHORT[t]}"
                rows.append({"pid": f"harmful-{r['Index']}-{w}", "dataset": "D2_jbb_artifact", "kind": "harmful",
                             "group": f"jbb:{r['Index']}", "index": r["Index"], "category": r["Category"],
                             "split": splits[r["Index"]], "wrapper": w, "goal": r["Goal"],
                             "prompt": fix_unicode(j["prompt"]), "unicode_fixed": fix_unicode(j["prompt"]) != j["prompt"]})
    # D3 (HarmBench procedure reproduced: random.seed(seed); per behaviour shuffle the template list in place)
    tree = ast.parse(open(ROOT / P["harmbench"]["templates_py"]).read())
    templates = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)][0]
    tpl_index = {t: i for i, t in enumerate(templates)}
    hb = [r for r in csv.DictReader(open(ROOT / P["harmbench"]["csv"]))
          if r["FunctionalCategory"] == P["harmbench"]["functional_category"]]
    random.seed(P["harmbench"]["human_jailbreaks_seed"])
    jbb_words = [words(r["Goal"]) for r in harm.values()]
    n_dup = 0
    for r in hb:
        random.shuffle(templates)
        chosen = templates[: P["harmbench"]["human_jailbreaks_random_subset"]]
        bw = words(r["Behavior"])
        if max(len(bw & jw) / len(bw | jw) for jw in jbb_words) >= P["harmbench"]["dedupe_jaccard_vs_jbb"]:
            n_dup += 1
            continue
        base = {"dataset": "D3_harmbench", "kind": "harmful", "group": f"hb:{r['BehaviorID']}", "index": r["BehaviorID"],
                "category": r["SemanticCategory"], "split": "cv", "goal": r["Behavior"]}
        rows.append({**base, "pid": f"hb-{r['BehaviorID']}-direct", "wrapper": "hb_direct", "prompt": r["Behavior"]})
        for jb in chosen:
            rows.append({**base, "pid": f"hb-{r['BehaviorID']}-human{tpl_index[jb]}", "wrapper": "hb_human",
                         "prompt": f"{jb}\n\n{r['Behavior']}"})

    for r in rows:  # repair any surrogates in any text field (artifacts and HarmBench templates)
        for k in ("prompt", "goal"):
            fixed = fix_unicode(r[k])
            if fixed != r[k]:
                r[k] = fixed
                r["unicode_fixed"] = True
    out = ROOT / P["out_jsonl"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(ROOT / P["out_meta_csv"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pid", "dataset", "kind", "group", "index", "category", "split", "wrapper"])
        w.writeheader()
        w.writerows({k: r[k] for k in w.fieldnames} for r in rows)
    c = Counter((r["dataset"], r["kind"]) for r in rows)
    print("prompts:", dict(c), "total", len(rows))
    print("D2 by wrapper:", dict(Counter(r["wrapper"] for r in rows if r["dataset"] == "D2_jbb_artifact")))
    print("D2 skipped:", dict(skipped))
    print("prompts with repaired unicode:", dict(Counter(r["wrapper"] for r in rows if r.get("unicode_fixed"))))
    print("D3 behaviours kept:", len({r["group"] for r in rows if r["dataset"] == "D3_harmbench"}), "dropped as JBB duplicates:", n_dup)
    print("harmful behaviour groups:", len({r["group"] for r in rows if r["kind"] == "harmful"}))


if __name__ == "__main__":
    main()
