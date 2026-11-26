"""Phase 2b: judge every response.
  - secondary: the studied model answers a fixed yes/no refusal question (all responses);
  - primary:   HarmBench Llama-2-13B classifier, "is this an instance of the behaviour?" (harmful only).

Usage: python scripts/04_judge.py --config configs/phase2.yaml [--tag smoke]
Output: data/judge_{tag}.csv (gid + scores, no text; committed)
"""
import argparse
import csv
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.judge import harmbench_scores, load_harmbench, refusal_scores  # noqa: E402
from src.model import load_thinker  # noqa: E402
from src.utils import ROOT, Timer, env_info, load_config, save_json, set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase2.yaml")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--refusal-scope", choices=["all", "benign"], default="all",
                    help="which responses get the secondary refusal question (Phase 5: benign only)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    T = Timer()
    rows = [json.loads(l) for l in open(ROOT / cfg["out"]["raw_dir"].format(tag=args.tag) / "generations.jsonl")]

    # 1. refusal question, always asked of Qwen2.5-Omni (cfg["model"] from base.yaml) as a fixed instrument
    ref_rows = rows if args.refusal_scope == "all" else [r for r in rows if r["kind"] == "benign"]
    with T("refusal_judge_s"):
        model, processor, tok = load_thinker(cfg)
        ref_sc = refusal_scores(model, processor, tok, cfg, [r["prompt"] for r in ref_rows],
                                [r["response"] for r in ref_rows], cfg["judge"]["refusal_batch_size"])
        ref_by_gid = {r["gid"]: x for r, x in zip(ref_rows, ref_sc)}
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # 2. HarmBench classifier (harmful behaviours only; behaviour = the JBB goal without wrapper)
    harm = [r for r in rows if r["kind"] == "harmful"]
    with T("harmbench_s"):
        hb_model, hb_tok = load_harmbench(cfg)
        hb = harmbench_scores(hb_model, hb_tok, [r["goal"] for r in harm], [r["response"] for r in harm],
                              cfg["judge"]["harmbench"]["batch_size"])
    hb_by_gid = {r["gid"]: s for r, s in zip(harm, hb)}

    out = ROOT / cfg["out"]["judge_csv"].format(tag=args.tag)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gid", "refusal_p_yes", "refusal_top_token", "refusal_yes_no_mass",
                    "harmbench_p_yes", "harmbench_top_token", "harmbench_yes_no_mass"])
        for r in rows:
            q = ref_by_gid.get(r["gid"])
            h = hb_by_gid.get(r["gid"])
            w.writerow([r["gid"]] + ([round(q[0], 5), q[1], round(q[2], 4)] if q else ["", "", ""]) +
                       ([round(h[0], 5), h[1], round(h[2], 4)] if h else ["", "", ""]))
    save_json({"env": env_info(cfg["model"]["id"], cfg["model"]["revision"]),
               "harmbench": cfg["judge"]["harmbench"], "timings_s": T.t,
               "n_rows": len(rows), "n_harmbench": len(harm)},
              ROOT / cfg["out"]["act_dir"].format(tag=args.tag) / "judge_meta.json")
    print(f"wrote {out}; timings {T.t}")


if __name__ == "__main__":
    main()
