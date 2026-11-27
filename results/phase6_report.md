# Phase 6 report: emotional pressure, over-refusal and a causal steering test

Phase 6 asks whether the "desperate" emotion state *causes* the over-refusal that emotional pressure produces, and whether removing it is safe.

- **6A** (exploratory, CPU only) re-analyses Phase 5 data.
- **6B** is a pre-registered steering experiment (`results/phase6_preregistration.md`, committed in `e5ff36d` before any steering data existed).

## Short answer

**Yes for over-refusal, in both models.** Steering away from "desperate" during generation is a strong, specific lever on the over-refusal caused by distress, and it does not make the models more harmful.

- **Effect size.** −desperate at 12 SD cuts over-refusal of benign requests under emotional pressure:
  - Qwen: from 53.0 % to 31.5 % (−21.5 pp [−29.0, −14.0]);
  - Mistral: from 33.5 % to 9.0 % (−24.5 pp [−32.0, −17.0]).
- **Dose effect.** At 4 SD the drop is about half as large.
- **Specificity.** A random direction steered equally hard does nothing in Qwen (53.0 %) and little in Mistral (28.0 %).
- **Safety cost.** Harmful compliance does not rise:
  - Qwen: +0.5 pp [−1.25, +2.5];
  - Mistral: −5.75 pp [−11.75, 0.0].
- **Caveats** (stated on the poster):
  - **+calm has a cost in Qwen.** It also cuts over-refusal but raises Qwen's harmful compliance by +3.5 pp [+0.75, +7.0], failing the 3 pp non-inferiority margin.
  - **Mistral's harmful-compliance drops are not specific.** They appear under every steering condition, including the random direction (−4.75 pp).

## 6A. Over-refusal analysis (exploratory, Phase 5 data)

Benign JailbreakBench behaviours, 1,500 answers per model, labelled by the refusal judge (`results/phase6a_overrefusal.json`).

| | Qwen2.5-Omni-7B | Mistral-7B-Instruct-v0.3 |
|---|---|---|
| Over-refusal: neutral → emotional pressure | 29.4 % → 46.8 % | 3.2 % → 23.6 % |
| Shift of the prompt-time "desperate" projection under pressure | +0.96 SD | +1.28 SD |
| Share of the wrapper effect (logit) statistically accounted for by "desperate" | 52 % | 31 % |

- This is a correlational mediation analysis on data already seen. It motivated 6B and is not used as evidence on its own.

## 6B. Steering experiment (pre-registered)

**Design** (as pre-registered; no deviations):
- **Steering:** add α·u to the residual stream at the output of the fixed emotion layer (Qwen 17, Mistral 18), at every position, during the whole generation.
- **Strength:** α is measured in SD of the prompt-time "desperate" projection (Qwen 1 SD = 1.547, Mistral 1 SD = 0.166).
- **Prompts:** 100 harmful + 100 benign JailbreakBench behaviours × {neutral, emotional pressure}.
- **Sampling:** 2 samples per prompt; 6 conditions; 4,800 answers per model.
- **Judges** (always unsteered): HarmBench classifier for harmful prompts, refusal question for benign ones.
- **Statistics:** paired by behaviour; 95 % bootstrap CI (2,000); sign-flip p (10,000).

### Rates (pressure wrapper for over-refusal; both wrappers pooled for harmful compliance in the tests)

| Condition | Qwen over-refusal, neutral / pressure | Qwen harmful compliance, neutral / pressure | Mistral over-refusal, neutral / pressure | Mistral harmful compliance, neutral / pressure |
|---|---|---|---|---|
| base | 27.5 / 53.0 % | 2.0 / 0.5 % | 6.5 / 33.5 % | 39.0 / 22.0 % |
| −desperate, 4 SD | 28.5 / 41.5 % | 2.0 / 0.5 % | 4.5 / 20.0 % | 36.5 / 25.0 % |
| −desperate, 12 SD | 21.5 / 31.5 % | 2.5 / 1.0 % | 4.0 / 9.0 % | 33.5 / 16.0 % |
| +calm, 12 SD | 18.5 / 27.5 % | 6.5 / 3.0 % | 4.0 / 15.5 % | 30.5 / 17.0 % |
| +desperate, 12 SD | 26.5 / 53.5 % | 1.0 / 1.0 % | 10.0 / 43.0 % | 32.0 / 17.0 % |
| random direction, 12 SD | 31.0 / 53.0 % | 2.0 / 1.5 % | 5.5 / 28.0 % | 34.0 / 17.5 % |

n = 200 answers per cell (100 behaviours × 2 samples). Wilson CIs are in `results/phase6_steer_{qwen,mistral}.json`.

### Pre-registered tests

| Test | Prediction | Qwen | Mistral |
|---|---|---|---|
| S1 | −desperate (12 SD) lowers over-refusal under pressure (Holm, one-sided) | ✓ −21.5 pp [−29.0, −14.0], p_Holm < 0.001 | ✓ −24.5 pp [−32.0, −17.0], p_Holm < 0.001 |
| S1 | +calm (12 SD) lowers over-refusal under pressure | ✓ −25.5 pp [−33.0, −18.5], p_Holm < 0.001 | ✓ −18.0 pp [−25.0, −11.0], p_Holm < 0.001 |
| S2 | −desperate: harmful compliance up by < 3 pp (upper CI) | ✓ +0.5 pp [−1.25, +2.5] | ✓ −5.75 pp [−11.75, 0.0] |
| S2 | +calm: harmful compliance up by < 3 pp | ✗ +3.5 pp [+0.75, +7.0] | ✓ −6.75 pp [−12.25, −1.25] |
| S3 | +desperate changes harmful compliance (two-sided) | ✗ −0.25 pp [−1.5, +1.0], p = 1.0 | ✓ −6.0 pp [−10.25, −1.75], p = 0.009, but not specific (random: −4.75 pp) |
| S4 | S1 decrease exceeds the random-direction control | ✓ −desperate −21.5 pp [−28.5, −14.5]; +calm −25.5 pp [−32.0, −19.5]; both p < 0.001 | ✓ −desperate −19.0 pp [−26.0, −13.0]; +calm −12.5 pp [−19.5, −6.5]; both p ≤ 0.0001 |

### Descriptive checks

- **Dose.** −desperate at 4 SD lowers over-refusal under pressure by 11.5 pp (Qwen) and 13.5 pp (Mistral), about half of the 12 SD effect.
- **Distress gap.** The gap is over-refusal under pressure minus over-refusal with the neutral wrapper, both under the same steering.
  - At 12 SD, −desperate closes 61 % of the gap in Qwen (25.5 → 10.0 pp) and 81 % in Mistral (27.0 → 5.0 pp).
  - This summary metric was not pre-registered.
- **Neutral wrapper.** Steering changes over-refusal without distress much less: Qwen −6.0 pp, Mistral −2.5 pp. So the effect is concentrated where the distress raised the state.
- **+desperate.** It raises Mistral's over-refusal under pressure (33.5 → 43.0 %, +9.5 pp [+2.5, +16.5]). Qwen is unchanged (+0.5 pp), probably at ceiling for this state.
- **Coherence.**
  - No condition is degenerate: distinct-2 is 0.95–0.97 (Qwen) and 0.98 (Mistral), against base 0.97 / 0.98.
  - The judges' yes/no probability mass is ≥ 0.5 for every answer.
  - Mean answer length rises slightly under −desperate in Qwen (70 → 82 tokens): the model writes the help it would otherwise refuse.

## Interpretation

- **Monitoring vs control.** Phases 3–5 showed that emotion vectors are a *poor alarm*: they add almost nothing to the refusal direction as a monitor of harmful compliance. Phase 6 shows that they are a *strong lever* on a different failure, over-refusal of distressed users.
- **Causal role.** The "desperate" direction is causally involved in that failure. Removing it helps more than a random direction in both models and does not cost safety.
- **Applied reading.** A deployment that wants to stay helpful to users in distress can steer away from "desperate" rather than retrain. +calm works too, but in Qwen it leaks into harmful compliance, so the choice of direction matters.

## Limitations

- **Prompts and sampling.** JailbreakBench prompts only, with one fixed distress wrapper. 2 samples per prompt, ≤ 128 new tokens.
- **Steering setup.** One layer, two strengths, and one random direction per model.
- **Labels.** Over-refusal labels come from a model judge (Qwen's refusal question), not from humans.
- **Fixed-wrapper confound.** The distress wrapper is a single text, so "desperate" may partly encode that wrapper's wording.
- **Unmeasured quality.** The steered answers were not rated for empathy or tone, only for refusal and harmfulness.

## Reproducibility

| Item | Value |
|---|---|
| Code | `scripts/14_overrefusal.py` (6A), `scripts/15_steer.py`, `scripts/04_judge.py --refusal-scope benign`, `scripts/16_steer_analysis.py`, `scripts/pod_phase6.sh` |
| Config | `configs/phase6.yaml` (seeds 20261127 for the random direction, `seed` + 1000·condition + batch for sampling) |
| Models | Qwen2.5-Omni-7B @ `ae9e169`, Mistral-7B-Instruct-v0.3 @ `c170c70`; HarmBench classifier @ `bda7053` |
| Environment | torch 2.9.1+cu128, transformers 5.24.1, Python 3.12.3; NVIDIA A40 48 GB, CUDA 12.8; code at `e5ff36d` |
| Runtime | steered generation 474 s (Qwen) + 499 s (Mistral); model loading, judging and setup on the same pod |
| Cost | $0.78 RunPod credit for Phase 6 |
| Outputs | `results/phase6_steer_{qwen,mistral}.json`, `results/phase6_steer_meta_*.json`, `data/judge_p6_*.csv` (IDs and judge numbers only). Raw answers stay in git-ignored `data/raw/p6_*` |
