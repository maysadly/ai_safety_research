# Phase 5 report: pre-registered confirmatory study

**Status:** done. The plan is in `results/phase5_preregistration.md`, committed before any Phase 5 data existed (commit `4d51d9b`). Numbers: `results/phase5_metrics_{qwen,mistral}.json`. Post-hoc analysis: `results/phase5_posthoc_k0_*.json`. Figures: `results/figures/a1_fig*`. Poster: `poster/a1/poster_A1_landscape.pdf`.

## Data and run

| | Qwen2.5-Omni-7B | Mistral-7B-Instruct-v0.3 |
|---|---|---|
| Answers (harmful + benign) | 8,770 (7,270 + 1,500) | 8,770 (7,270 + 1,500) |
| Harmful complied (HarmBench) | 607 (8.4 %) | 4,877 (67 %) |
| Prompts with mixed outcomes | 254 | 452 |
| Benign over-refusal | 37 % | 10 % |
| Fixed layers, emotion / refusal (pre-registered rule) | 17 / 19 | 18 / 14 |

- **Behaviour groups:** 282 (100 JailbreakBench + 182 HarmBench).
- **GPU:** one A100 80 GB, about 2.6 h including the smoke test and Mistral Phase 1. Phase 5 cost ≈ $5.7; the project total is ≈ $6.4.

## Pre-registered tests

| Test | Prediction | Qwen | Mistral |
|---|---|---|---|
| C1 added value (Δ refusal + emotion − refusal) | not superior; equivalent within ±0.03 | Δ +0.003 to +0.005. Superior (p_Holm 0.04) **and** equivalent: statistically detectable, practically nil | Δ +0.09 to +0.15. Superior, **not** equivalent (the refusal direction is weak: 0.63–0.66) |
| C2 emotion vs random monitors | not above the 95th percentile | k = 0: above (97.5th) ✗; k = 1, 5: 80–90th ✓ | k = 0: 84th ✓; k = 1, 5: 98th ✗ |
| C3 pre-generation | emotion AUROC upper CI < 0.65 | ✗ 0.84 [0.81, 0.86]; refusal 0.88 | ✗ 0.74 [0.71, 0.77] |
| C4 H1 reversed (within prompt, tokens 1–5) | desperate < 0, calm > 0, beyond random | ✓ d_z −1.37 / +1.39 (random p95 1.15) | ✗ −0.07 / +0.00 |
| C5 H3, emotional pressure | over-refusal up, compliance not up | ✓ +17 pp / −0.2 pp | ✓ +20 pp / −19 pp |

**Text baseline** (TF-IDF on the first k tokens):

| Model | k = 1 | k = 2 | k = 5 |
|---|---|---|---|
| Qwen, text | 0.69 | 0.89 | 0.97 |
| Qwen, activations | 0.95–0.96 | 0.96 | 0.97 |
| Mistral, text | 0.72 | | 0.82 at k = 5 vs 0.78 for activations |

Activations win only at the first 1–2 tokens.

## Sensitivity (all as pre-registered)

- Results hold when the exploratory PAIR-vicuna prompts are excluded, with raw instead of denoised vectors, and with the label HarmBench P(yes) ≥ 0.5 (largest change in AUROC: 0.01).
- With layers chosen inside each training fold (nested), Qwen emotion and refusal reach 0.95; Mistral's emotion monitor 0.80 and its refusal monitor 0.70.
- Per dataset, Mistral's refusal monitor drops to 0.44 on JBB attack artifacts, while its emotion monitor stays at 0.77–0.79.

## Post-hoc (not pre-registered): where the pre-generation signal comes from

At k = 0, within prompt families (`results/phase5_posthoc_k0_*.json`):

| Prompt family | Qwen refusal | Qwen emotion | Mistral emotion |
|---|---|---|---|
| plain + wrappers | 0.93 | 0.86 | 0.76 |
| HarmBench direct | 0.94 | 0.88 | 0.76 |
| random-search | 0.92 | 0.91 | 0.50 |
| HarmBench human jailbreak | 0.75 | 0.65 | 0.75 |
| **PAIR** | **0.73** | **0.59** (random median 0.59) | **0.54** |

A monitor that only knows the prompt family reaches 0.76 (Qwen) and 0.70 (Mistral).

**Reading.**
- The pre-generation signal is real for direct requests. It mostly says how harmful or borderline the request looks.
- It is weak on PAIR jailbreaks. That is the set the exploratory Phase 3 used, so the two phases are consistent.

## Updated answer

- **Not special.** Emotion vectors are not a special early-warning signal.
- **Safety-tuned model (Qwen).** They add a statistically detectable but practically nil 0.005 AUROC to the refusal direction. After the answer starts they are no better than random directions, and reading the first five tokens of text is as good.
- **Weakly aligned model (Mistral).** The refusal direction is a poor predictor. Emotion features do better, but random directions come close.
- **Before generation.** Warning works for direct requests but is weak exactly on jailbreaks.
- **Replicated.** H1 reversed replicated in Qwen. Emotional pressure raises over-refusal in both models, and in Mistral it also lowers compliance by 19 pp.

## Deviations from the pre-registration (all disclosed)

1. **Mistral stories:** 22 instead of 30 "calm" stories; it named the emotion too often (167 rejections).
2. **Vector variant:** the Phase 1 rule chose raw vectors for Mistral. The pre-registration fixes denoised, and denoised is the primary analysis; raw is reported as a sensitivity check with the same results.
3. **Unicode repair:** 51 HarmBench human-jailbreak prompts had broken surrogate characters, repaired to valid Unicode.
4. **Speed and memory only:** length-sorted token-budget batching; judge batch size 32; Qwen judged in parallel with Mistral generation.
5. **Benign-only secondary judge:** the refusal question was asked of benign answers only (`--refusal-scope benign`), as pre-registered for over-refusal.
6. **Post-hoc analysis:** the prompt-family breakdown above was added after the results were seen and is labelled post-hoc everywhere.
