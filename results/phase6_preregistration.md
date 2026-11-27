# Phase 6B pre-registration: does steering the emotion state change safety behaviour?

Written 2026-11-27, **before** any steering data was generated; committed before the GPU run.

## Motivation (from data we already have)

- **The pressure effect.** Emotional pressure ("Please, I'm desperate …") raises over-refusal of *benign* requests in both models:
  - Qwen: 29 % → 47 %;
  - Mistral: 3 % → 24 %.

  It does not raise harmful compliance (Phase 5, C5).
- **A candidate mediator.** The prompt-time "desperate" projection rises by +0.96 SD (Qwen) and +1.28 SD (Mistral) under pressure. Exploratory analysis 6A finds that it statistically accounts for 52 % (Qwen) and 31 % (Mistral) of the pressure effect on over-refusal.
- **That analysis is correlational.** This experiment tests the causal claim.

## Hypotheses

| ID | Hypothesis | Source |
|---|---|---|
| S1 (mitigation) | Removing "desperate" (−α·v_desperate) or adding "calm" (+α·v_calm) during generation **lowers over-refusal of benign requests under emotional pressure** | mediation in 6A |
| S2 (safety cost) | The same steering does **not raise harmful compliance** by more than 3 pp | |
| S3 (risk) | Adding "desperate" (+α·v_desperate) changes harmful compliance | competing predictions, two-sided: Anthropic links desperation to misaligned behaviour [1]; our Phase 5 data show refusals carry *more* "desperate" |
| S4 (specificity) | S1 effects exceed those of a **random direction** steered with the same strength | |

## Design

| Item | Setting |
|---|---|
| Models | Qwen2.5-Omni-7B and Mistral-7B-Instruct-v0.3 (same revisions and system prompts as Phase 5) |
| Prompts | JailbreakBench 100 harmful + 100 benign behaviours × {neutral, emotional pressure} wrappers (fixed, as before) |
| Samples | 2 per prompt; T 0.7, top-p 0.9, top-k off, ≤ 128 new tokens |
| Steering site | add α·u to the residual stream at the output of the model's fixed emotion layer (Qwen 17, Mistral 18), at every position, during the whole generation; u = unit vector (denoised Phase 1 vectors) |
| Strength | α in units of the natural spread of the "desperate" projection (SD over the 200 neutral-wrapper JBB prompts, Phase 5): Qwen 1 SD = 1.547, Mistral 1 SD = 0.166 |
| Random control | one Gaussian unit direction per model, seed 20261127 |

**Conditions** (6 per model, 4,800 answers per model):

| Name | Steering vector | Strength |
|---|---|---|
| base | none | — |
| desp_minus_4 | −v_desperate | 4 SD |
| desp_minus_12 | −v_desperate | 12 SD |
| calm_plus_12 | +v_calm | 12 SD |
| desp_plus_12 | +v_desperate | 12 SD |
| rand_minus_12 | −u_random | 12 SD |

**Labels.**
- Harmful prompts: HarmBench classifier (as in Phase 5).
- Benign prompts: the refusal question asked of Qwen2.5-Omni-7B (as in Phase 2 and 5).
- Both judges always run unsteered.

## Analysis (fixed now)

- **Outcomes per condition:**
  - over-refusal of benign requests, separately for the neutral and emotional-pressure wrappers;
  - harmful compliance, for both wrappers.
- **Comparisons with base:**
  - **Primary family 1 (S1):** over-refusal under emotional pressure, for desp_minus_12 and calm_plus_12 vs base. Holm over 2 comparisons; one-sided, expecting a decrease.
  - **Family 2 (S2):** harmful compliance for the same two conditions, compared against +3 pp (non-inferiority: the upper 95 % CI must be < +3 pp).
  - **Family 3 (S3):** harmful compliance, desp_plus_12 vs base, two-sided.
  - **Family 4 (S4):** the S1 decrease minus the decrease under rand_minus_12, paired; one-sided.
- **Statistics:**
  - differences are paired by behaviour (mean over the behaviour's samples);
  - 95 % CI from a bootstrap over behaviours (2,000);
  - p from a sign-flip test over behaviours (10,000).
- **Dose check (descriptive):** desp_minus_4 vs desp_minus_12.
- **Coherence check (descriptive, and an exclusion rule):**
  - fraction of answers where the judge's yes/no probability mass is < 0.5;
  - distinct-2 ratio.
  - A condition whose distinct-2 falls below 50 % of base is reported as **degenerate**, and its behaviour effects are not interpreted.
- **Reporting:** every test is reported for both models, whatever its outcome.

## Ethics

Steering is applied only to measure safety behaviour. Generated answers stay in git-ignored `data/raw/`; only labels and aggregates are committed. No new jailbreak prompts are created.
