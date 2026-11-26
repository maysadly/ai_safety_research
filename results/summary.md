# Can emotion vectors warn us before a model breaks its safety rules? Summary of results

> **Update (Phase 5, pre-registered confirmatory study, 2 models, 17,540 answers, 282 behaviours):** see `results/phase5_report.md`. It refines the exploratory answer below.
>
> - **Emotion adds little.** Emotion vectors add only +0.005 AUROC to the refusal direction in Qwen; in Mistral, where the refusal direction is weak, they add +0.09–0.15, but random directions come close.
> - **A pre-generation signal exists, but only for direct requests.** It does exist (AUROC 0.84 Qwen, 0.74 Mistral), but mainly for direct requests; on PAIR jailbreaks it stays weak (post-hoc).
> - **Replicated in Qwen.** "Refusals look more desperate, less calm" replicated in Qwen but not in Mistral.
> - **Replicated in both models.** Emotional pressure raises over-refusal: +17 pp and +20 pp.
>
> The A1 landscape poster (`poster/a1/`) shows these confirmatory results; the A0 poster v1 shows only the exploratory phases.


Qwen2.5-Omni-7B (thinker, text only, bf16), JailbreakBench behaviours, HarmBench judge. All CIs are 95 %. Bootstrap CIs are clustered by behaviour.

## Short answer

**No, not in this setup.**

1. **Before generation.** Before the model writes anything, neither emotion projections nor the refusal direction predict whether it will comply with a *new* harmful request. Test AUROC is 0.42–0.55, at chance.
2. **After a few tokens.** Within 1–5 generated tokens, compliance becomes readable from activations (AUROC ≈ 0.8–0.87). But **random directions do about as well** (median 0.76–0.81), so the signal is "what the model has already started to say". It is not specific to emotions.
3. **Added value.** Emotion projections add **no robust value** beyond the refusal direction. ΔAUROC is ≤ +0.04 at most token counts. The single larger gain (+0.14 at k = 10) is not significant after correction for 6 comparisons.
4. **Direction of H1.** The emotion signals that do differ between compliant and refused answers point **opposite** to H1. Refusals look *more* "desperate/fearful" and *less* "calm/joyful", most likely the apologetic register of a refusal.
5. **Emotional pressure.** It shifts the model's emotion representations exactly as intended, but makes it *more* cautious: benign over-refusal rises from 29 % to 49 %, and harmful compliance is unchanged.

## 1. Setup (fixed in advance; see `TASK.md`, `configs/`)

| Component | Details |
|---|---|
| Model | Qwen2.5-Omni-7B @ `ae9e169`. Thinker only (talker never loaded). System prompt `You are a helpful assistant.` (template default). 28 layers, d = 3584 |
| Data | JailbreakBench JBB-Behaviors: 100 harmful + 100 matched benign behaviours. Split by behaviour pair, stratified by category: 50 train / 20 val / 30 test |
| Wrappers (pre-registered) | neutral, emotional pressure, role-play; 5 samples per prompt (T 0.7, top-p 0.9, ≤ 128 tokens) → 3,000 responses |
| Exploratory extension | JailbreakBench's published PAIR prompts (target vicuna-13b), 79 behaviours × 10 samples = 790 responses. Added because the pre-registered set had too few compliant responses (§3). No new jailbreaks were optimised |
| Emotion vectors | fear, joy, sadness, calm, desperate. Built from 150 stories the model wrote (emotion never named), as mean(emotion) − mean(all), neutral-PC denoised, unit norm |
| Refusal direction | Arditi et al.: mean harmful − mean benign at the last prompt token, train split only |
| Random baseline | 200 random unit directions per layer. Random monitors use 5 random directions, the same size as the emotion monitor |
| Monitors | Standardised L2 logistic regression (C = 1, not tuned): emotion (5), refusal (1), refusal + emotion (6), and 200 random-5 monitors. Features at k = mean projection over the end-of-prompt token and generated tokens 1…k. Fit on train; layer chosen on val; reported on test |
| Labels | HarmBench Llama-2-13B classifier (verbatim prompt, greedy "Yes" = compliance) |

## 2. Emotion vectors and refusal direction (Phase 1)

- **Emotion vectors** classify stories on unseen topics well above chance.
  - Leave-one-topic-out accuracy is 0.60–0.65 across layers 8–27, best **0.65 [0.57, 0.72]** at layer 17; chance is 0.20.
  - **fear and desperate are nearly the same direction** (cos 0.62); 6 of 9 held-out fear stories are classified as desperate.
- **Refusal direction:** val AUROC harmful vs benign **0.96–0.97** at layers 15–23.
- **Overlap.** Emotion vectors are weakly aligned with the refusal direction: |cos| 0.06–0.17 at layers 14–26, against ≤ 0.06 for random directions.
- **Topic trap.** They also react to harmful *topics* at the prompt: |d| ≈ 1.5–2.6 for harmful vs benign, but random directions reach |d| ≈ 1.5–1.9 too.

## 3. Compliance rates (Phase 2)

**Harmful compliance (HarmBench):**

| Condition | Rate [95 % CI] | n |
|---|---|---|
| neutral | 0.8 % [0.3, 2.0] | 500 |
| emotional pressure | 0.8 % [0.3, 2.0] | 500 |
| role-play | 1.2 % [0.6, 2.6] | 500 |
| **all pre-registered** | **0.9 % [0.6, 1.6]** | 1500 |
| PAIR (exploratory) | 19.8 % [17.1, 22.7] | 790 |

**Within-prompt variation.**
- Pre-registered prompts: only 13 of 300 harmful prompts gave mixed outcomes over 5 samples.
- PAIR prompts: 45 of 79 gave mixed outcomes over 10 samples.

**Benign over-refusal** (refusal question to the model itself):

| Wrapper | Over-refusal [95 % CI] |
|---|---|
| neutral | 28.6 % [24.8, 32.7] |
| emotional pressure | **48.8 % [44.4, 53.2]** |
| role-play | 37.6 % [33.5, 41.9] |

**Judge validation.** An independent LLM annotator (Claude), given the HarmBench rule and blind to the judge, labelled the stratified samples of 100 responses (`results/judge_validation.json`).

| Sample | Raw agreement | κ | Reweighted accuracy | Agreement on high-confidence items |
|---|---|---|---|---|
| pre-registered | 99 / 100 | 0.96 | — | — |
| PAIR | 88 / 100 | 0.76 | 0.90 | 100 % |

Disagreements are borderline responses cut off at 128 tokens. **Human labels are still pending.**

## 4. Pre-registered analysis (Phase 3, fixed wrappers): not estimable

The test split contains **1** compliant response out of 450, so test AUROCs are ranks of a single example. They are 0.91–1.0 for every monitor, while random monitors range from 0.0 to 1.0. They carry no information. H1 on the val + test splits has 4 compliant responses and 4 mixed prompts, also not estimable. All numbers are in `results/phase3_metrics.json`.

## 5. H2 and early warning (exploratory PAIR set; test = 26 behaviours, 260 responses, 38 compliant)

| k (tokens seen) | emotion | refusal | refusal + emotion | random median [95 % range] | emotion percentile among random | Δ(refusal + emotion − refusal) [CI], p |
|---|---|---|---|---|---|---|
| 0 (prompt only) | 0.42 [0.26, 0.62] | 0.55 [0.42, 0.69] | 0.45 [0.29, 0.65] | 0.57 [0.37, 0.71] | 9 | −0.10 [−0.27, 0.07], 0.28 |
| 1 | 0.80 [0.68, 0.90] | 0.77 [0.64, 0.88] | 0.80 [0.67, 0.90] | 0.76 [0.56, 0.88] | 72 | +0.02 [−0.01, 0.08], 0.22 |
| 2 | 0.78 [0.68, 0.88] | 0.79 [0.67, 0.89] | 0.80 [0.68, 0.90] | 0.80 [0.64, 0.87] | 34 | +0.01 [−0.03, 0.05], 0.58 |
| 5 | 0.87 [0.77, 0.94] | 0.82 [0.70, 0.92] | 0.86 [0.75, 0.94] | 0.81 [0.60, 0.88] | 94.5 | +0.04 [−0.00, 0.08], 0.07 |
| 10 | 0.85 [0.73, 0.93] | 0.74 [0.63, 0.86] | 0.87 [0.78, 0.94] | 0.82 [0.66, 0.88] | 81.5 | +0.14 [0.00, 0.23], 0.04 |
| 20 | 0.83 [0.72, 0.92] | 0.87 [0.79, 0.94] | 0.87 [0.78, 0.94] | 0.83 [0.68, 0.88] | 58.5 | +0.00 [−0.04, 0.03], 0.88 |

- With Bonferroni correction over the 6 token counts (α = 0.0083), no Δ is significant.
- The k = 10 gain coincides with a refusal monitor that picked layer 4 on the small val split. It is most likely layer-selection noise.
- **H2 is not supported.**

## 6. H1: compliant vs refused (PAIR, val + test: 430 responses, 64 compliant)

**Between prompts, at the end of the prompt.** Layer 17 for emotions and layer 19 for refusal, both fixed from Phase 1. Cohen's d is compliant − refused; p values are from a prompt-level permutation test.

| Direction | d [95 % CI] | p |
|---|---|---|
| joy | +0.58 [0.14, 1.03] | 0.02 |
| desperate | −0.42 [−0.85, −0.01] | 0.10 |
| fear | −0.38 [−0.90, 0.07] | 0.13 |
| sadness | −0.36 [−0.85, 0.12] | 0.16 |
| calm | +0.19 [−0.24, 0.63] | 0.47 |
| refusal | −0.55 [−0.93, −0.19] | 0.03 |
| *random directions (95th percentile of \|d\|)* | *0.45* | |

**Within prompts, first generated tokens.** 21 mixed prompts; paired d_z, compliant − refused samples of the same prompt.

| Direction | d_z, tokens 1–5 |
|---|---|
| desperate | −1.21 |
| fear | −1.14 |
| sadness | −1.07 |
| joy | +1.18 |
| calm | +1.15 |
| refusal | −1.32 |
| *random directions (95th percentile of \|d_z\|)* | *0.94* |

All within-prompt effects have sign-flip p ≤ 0.0004. Over tokens 1–20 the pattern is the same, with a random 95th percentile of 1.07.

**Reading.** The emotion signals do separate compliant from refused answers *once the answer is being written*, and they beat random directions by a modest margin. The effect has the **opposite sign** to H1: compliance comes with *less* desperation and fear and *more* calm and joy. At the end of the prompt (Fig. 4, position 0) the two groups overlap. They diverge at tokens 1–3, exactly where refusals begin ("I'm sorry…"). **H1 as stated is not supported.**

## 7. H3, text-only: emotional context (pre-registered wrappers, paired by behaviour)

| Wrapper | Shift of prompt projections vs neutral (SD units) | Harmful compliance vs neutral | Benign over-refusal |
|---|---|---|---|
| emotional pressure | desperate +0.92, fear +0.75, calm −0.91 | 0.0 pp [−1.0, +1.2] | 29 % → **49 %** |
| role-play | desperate −1.52, calm +1.26, refusal −0.88 | +0.4 pp [−0.8, +1.6] | 29 % → 38 % |

**H3 is not supported.** The emotional manipulation reaches the representation, but it makes the model more cautious rather than more compliant.

## 8. False alarms (PAIR monitors; threshold set on val for 80 % recall of compliance)

| Monitor | k | Test recall | Alarms on refused harmful | Alarms on benign requests |
|---|---|---|---|---|
| refusal | 0 | 0.76 | 0.59 | **0.92** [0.89, 0.94] |
| emotion | 0 | 0.24 | 0.41 | 0.27 [0.23, 0.32] |
| emotion | 5 | 0.79 [0.64, 0.89] | 0.20 | 0.29 [0.25, 0.33] |
| refusal + emotion | 5 | 0.50 | 0.13 | 0.21 [0.18, 0.25] |

A compliance monitor detects "the model is going along with the request". On benign requests that is the desired behaviour, so it fires there too. It would need to be paired with a judgement of whether the request itself is harmful.

## 9. What this means for AI-safety practice

- **No early warning from these signals.** Emotion-vector monitors did not give a *pre-generation* warning here. Do not expect them to replace input-side harm classifiers.
- **Early-token monitoring works, but not because of emotions.** Monitoring the first few generated tokens works (AUROC ≈ 0.8–0.87), but any direction works. Cheap activation or text monitors on early tokens are a reasonable last line of defence; emotion vectors add little to them.
- **Emotional-pressure framing.** On this model, emotional-pressure framing is a usability problem (over-refusal), not a jailbreak.

## 10. Limitations

- **Scope:** one 7-billion-parameter model, text only, English, 128-token responses.
- **Judge:** a single automatic judge; human validation pending.
- **The decisive H2 test is exploratory and small:**
  - It uses PAIR prompts optimised against another model (vicuna) and was added after the base rates were seen. It was decided before any analysis was run.
  - The test set is 26 behaviours with 38 compliant responses, so CIs are about ±0.1 AUROC.
  - Layers were selected on a small val split (17 behaviours).
- **Emotion vectors:** built from 150 model-written stories; fear and desperate are not separable; the projections are linear and read at single layers.
- **Prompts:** PAIR prompts differ in content from JBB goals, so prompt-level (k = 0) effects partly reflect prompt style.

Files: `results/phase*_report.md`, `results/phase3_metrics*.json`, `results/judge_validation.json`, `results/figures/`.
