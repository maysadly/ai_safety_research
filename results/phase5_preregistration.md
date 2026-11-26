# Phase 5 pre-registration: confirmatory study

Written 2026-11-26, **before** any Phase 5 data was generated. It is committed to git before the GPU run, so the commit time documents the order. Settings are machine-readable in `configs/phase5.yaml`. Any deviation will be reported in `results/phase5_report.md`.

## Why

- **The exploratory results** (`results/summary.md`) suggest four things:
  - (a) There is no pre-generation warning.
  - (b) After 1–5 tokens compliance is readable, but not more by emotion vectors than by random directions.
  - (c) Emotion adds nothing robust to the refusal direction.
  - (d) Within a prompt, refusals look more "desperate" and less "calm".
- **Why they are only exploratory:** they come from 26 test behaviours of one model, on prompts chosen after the base rates were seen.
- **This study re-tests them** with fresh samples, more prompts and behaviours, a second model, a priori layers and cross-validation over all behaviours.

## Models

| Model | System prompt | Emotion vectors / refusal direction |
|---|---|---|
| Qwen2.5-Omni-7B (thinker, as before) | `You are a helpful assistant.` | Phase 1 vectors and refusal direction, unchanged |
| Mistral-7B-Instruct-v0.3 (public; different family; weaker refusal training) | none (template default) | Built by the same Phase 1 pipeline: 150 stories written by Mistral, 50 neutral texts, refusal direction from JBB train-split harmful − benign |

## Data (per model; every prompt from an established public benchmark; no new jailbreaks)

| Set | Prompts | Samples | Responses |
|---|---|---|---|
| D1 JBB behaviours | 100 harmful + 100 benign × 3 fixed wrappers (as in Phase 2) | 5 | 3,000 |
| D2 JBB attack artifacts | PAIR (targets gpt-3.5, gpt-4, llama-2, vicuna) + prompt-with-random-search (same 4 targets), black-box, every prompt whose goal equals the current JBB goal text | 5 | ≈ 3,100 |
| D3 HarmBench standard behaviours (200) | direct request + 2 human-jailbreak templates chosen by HarmBench's own `HumanJailbreaks(random_subset=2, seed=1)` procedure; behaviours whose text duplicates a JBB goal (token Jaccard ≥ 0.5) are dropped | 5 | ≈ 3,000 |

- **Sampling:** T 0.7, top-p 0.9, top-k off, **≤ 256 new tokens**, fixed seeds.
- **Activations:** teacher-forced residuals at the end of the prompt and at generated tokens 1–20, projected on the 5 emotion vectors (raw and denoised), the refusal direction (all layers) and 200 random directions (at the emotion layer).

## Labels

- **Harmful prompts:** HarmBench Llama-2-13B classifier (same pinned revision and verbatim prompt), greedy "Yes" = compliance. Behaviour string = JBB goal or HarmBench behaviour.
- **Benign prompts:** the Phase 2 refusal question, always asked of Qwen2.5-Omni-7B, so both models are judged by the same instrument.

## Fixed analysis choices (no compliance labels involved)

- **Layers** come from Phase 1 data only:
  - emotion layer = the best leave-one-topic-out story-classification layer within [⌊0.25 L⌋, L − 1];
  - refusal layer = the best val harmful-vs-benign AUROC layer.
  - For Qwen these are **17 / 19**. For Mistral they are computed by the same rule from its Phase 1 run, before Phase 5 labels exist.
- **Features at k:** mean projection over positions 0…k that exist (0 = end of prompt).
- **Monitors:** standardised L2 logistic regression, C = 1:
  - emotion (5 denoised vectors);
  - refusal (1);
  - refusal + emotion (6);
  - 200 random 5-direction monitors (at the emotion layer);
  - text baselines on the first k generated tokens: a keyword refusal detector, and TF-IDF (1–2-grams) + logistic regression.
- **Evaluation:**
  - harmful responses pooled over D1–D3, separately per model;
  - 5-fold StratifiedGroupKFold (groups = behaviours; a JBB behaviour shared by D1 and D2 is one group; seed 0); out-of-fold scores; AUROC;
  - 95 % CI by cluster bootstrap over behaviours (2,000).
- **k ∈ {0, 1, 2, 5, 10, 20}; primary k ∈ {0, 1, 5}.**

## Confirmatory tests (each model separately; Holm correction over the 3 primary k within each family)

| ID | Claim tested | Test | Exploratory prediction |
|---|---|---|---|
| C1 | Emotion adds to refusal (H2) | Δ = AUROC(refusal + emotion) − AUROC(refusal): one-sided superiority, α 0.05. Equivalence: 90 % CI inside ±0.03 | not superior; equivalent within ±0.03 |
| C2 | Emotion vectors are special | emotion-monitor AUROC above the 95th percentile of the 200 random monitors | not above |
| C3 | Pre-generation warning | k = 0 emotion-monitor AUROC, 95 % CI | upper bound < 0.65 |
| C4 | H1 reversed | within-prompt d_z (compliant − refused), tokens 1–5, emotion layer, prompts with mixed outcomes: desperate < 0 and calm > 0 (sign-flip test, two-sided α 0.05, Holm over 2), and \|d_z\| above the random 95th percentile | both hold |
| C5 | H3 (D1 only) | emotional pressure − neutral, paired by behaviour: benign over-refusal and harmful compliance, bootstrap 95 % CI | over-refusal ↑; compliance not ↑ |

A prediction counts as **confirmed** only if it holds for the stated test. Every test is reported for both models, whatever its outcome.

## Sensitivity analyses (reported, not used for claims)

- excluding the PAIR-vicuna prompts used in the exploratory phase;
- per dataset (D1, D2, D3);
- label = HarmBench P(yes) ≥ 0.5;
- layers chosen inside each training fold (nested), compared with the fixed layers.

## Stopping rule

One generation run per model, with no top-up after labels are seen. If a model has fewer than 100 compliant responses, its C1–C4 results are reported as underpowered.
