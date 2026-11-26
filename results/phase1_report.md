# Phase 1 report: emotion vectors, refusal direction, random baselines

**Status:** done.
- **Emotion vectors:** they generalise to unseen story topics well above chance, but only moderately (≈ 0.63 accuracy vs 0.20 chance), and **fear and desperate are hard to tell apart**.
- **Refusal direction:** it works as expected (val AUROC ≈ 0.96).
- **Two findings that shape Phase 3:** (1) the emotion vectors are slightly aligned with the refusal direction; (2) they do react to a harmful *topic* at the prompt. Both make the planned controls (within-harmful comparison, refusal baseline) necessary.

Date: 2026-11-26. Numbers: `results/phase1_vectors.json`, `results/phase1_stories_stats.json`. Figures: `results/figures/p1_*.png|svg`.

## 1. Environment and cost

| Item | Value |
|---|---|
| GPU run | 1 × A40 48 GB (RunPod Secure Cloud, CA-MTL-1), ≈ 6 min of pod time |
| Software | torch 2.9.1+cu128, transformers 5.24.1, Python 3.12.3 (same image as Phase 0) |
| Model | `Qwen/Qwen2.5-Omni-7B` @ `ae9e1690543ffd5c0221dc27f79834d0294cba00`, thinker only, bf16 |
| System prompt | `You are a helpful assistant.` (decision D1) |
| Analysis | CPU, local. `python scripts/02_build_vectors.py --from-cache` reproduces every number from the saved activations |
| Seeds | 1234 (generation, hold-out topics), 20261126 (random vectors) |
| Cost | Phase 1 ≈ $0.05; project total so far ≈ $0.25 (balance $8.58). No pods running |

Runtimes: story generation 91 s (3 rounds), story activations 6 s, prompt activations 1 s.

## 2. Stories

- **What:** the model wrote 30 stories per emotion (10 everyday topics × 3), about 115–128 words each, plus 50 neutral factual texts (10 topics × 5).
- **Sampling:** T = 0.9, top-p 0.95.
- **Filters:** a candidate was rejected if it named the emotion (whole-word list of the emotion and its synonyms, in `configs/phase1.yaml`), was cut off at 220 tokens, or fell outside 60–180 words. The model often names the emotion even when told not to. Rejections:

  | Reason | Rejected |
  |---|---|
  | named calm | 70 |
  | named fear | 42 |
  | named joy | 20 |
  | named sadness | 9 |
  | named desperate | 8 |
  | named an emotion (neutral texts) | 2 |
  | cut off at 220 tokens | 32 |
  | length outside 60–180 words | 10 |

- **Storage:** stories are benign and committed in `data/stories/stories.jsonl`.
- **Spot check:** 4 random texts were read. They were on topic and the emotion was conveyed indirectly. One "calm" story also carries sadness cues, so calm stories are the least pure.

## 3. Emotion vectors

- **Reading context.** Each text is fed as the *assistant* turn after the fixed user prompt `Write a short text.` This matches the position where the monitor will read the model's own answers. Residuals are averaged over the text tokens, skipping the first 10.
- **Vector.** v_e = mean(stories of e) − mean of the five emotion means, computed at every layer (0 = embeddings, 28 = after the final norm).
- **Denoised variant.** Project out the top PCs of the neutral texts that explain 50 % of their variance (3–6 PCs per layer).
- **Variant rule, fixed before looking:** use denoised unless it is more than 2 pp worse than raw over layers 7–27. **Result: denoised.** The two variants differ by ≤ 0.04 at every layer.
- **Final vectors.** Built from all 30 stories per emotion and normalised to unit length. The held-out numbers below estimate the quality of the procedure.

**Held-out validation.** Nearest-direction classification, argmax_e (x − μ)·v_e; chance = 0.20. Figure: `p1_heldout_accuracy`.

| Evaluation | Best layer(s) | Accuracy [95 % Wilson CI] | Layers 8–27 |
|---|---|---|---|
| TASK.md split: 3 held-out topics (7, 8, 9), n = 45 | 14–16, 27 | 0.62 (28/45) [0.48, 0.75] | 0.56–0.62 |
| Supplementary: leave-one-topic-out, n = 150 | 17 | 0.65 [0.57, 0.72] | 0.60–0.65 |

**Confusion at layer 14** (rows = true, cols = predicted; 9 stories per row):

|  | fear | joy | sadness | calm | desperate |
|---|---|---|---|---|---|
| fear | 2 | 0 | 0 | 1 | **6** |
| joy | 0 | 7 | 0 | 1 | 1 |
| sadness | 1 | 0 | 6 | 0 | 2 |
| calm | 0 | 1 | 0 | 8 | 0 |
| desperate | 2 | 0 | 2 | 0 | 5 |

**Geometry (layer 20 cosines).**
- fear · desperate = +0.62
- calm · fear = −0.76, calm · desperate = −0.78
- joy · desperate = −0.69
- sadness is the most independent (|cos| ≤ 0.62 with joy, ≤ 0.35 otherwise)

The vectors form a clear valence axis. **Fear and desperate are nearly the same direction**, so H1 results for them will not be independent. They should be read as one "threat / negative arousal" signal.

## 4. Refusal direction (Arditi et al. baseline)

- **Fit.** Difference of means at the last prompt token, 50 train harmful vs 50 train benign prompts, neutral wrapper. No val or test prompts were used for fitting.
- **Separation on held-out prompts.** Val harmful vs benign (20 + 20): **AUROC 0.96–0.97 at layers 15–23**, and 0.71–0.86 at layers 1–12 (`p1_refusal_val_auroc`).
- **Layer 0.** The last prompt token is the same for every prompt, so the direction is undefined there. It is stored as a zero vector.

## 5. Random baselines

200 Gaussian directions per layer, unit norm, seed 20261126. They are regenerated deterministically from the seed (`src/vectors.random_unit_vectors`) and not stored.

## 6. Two findings that matter for Phase 3

1. **Emotion vectors partly overlap the refusal direction** (`p1_cos_emotion_refusal`). At layers 14–26:

   | Emotion | Cosine with refusal direction |
   |---|---|
   | fear, desperate | +0.07 to +0.16 |
   | sadness | up to +0.14 (layer 19) |
   | joy, calm | −0.06 to −0.17 |
   | random directions (max \|cos\|, n = 200) | ≤ 0.06 |

   Part of any emotion signal can therefore be refusal-direction signal. That is why H2 (refusal + emotion vs refusal alone) is the key test.
2. **The "dangerous topic" trap is real** (`p1_topic_sensitivity`). At the last prompt token, val harmful vs benign prompts, layers 16–26:

   | Direction | Cohen's d (harmful − benign) |
   |---|---|
   | fear | +1.6 to +2.3 |
   | desperate | +1.7 to +2.6 |
   | calm | −1.5 to −2.6 |
   | joy | −1.4 to −2.1 |
   | refusal | +2.4 to +2.9 |
   | random directions (95th percentile of \|d\|) | 1.5–1.9 |

   Harmful and benign prompts differ along almost every direction. The emotion vectors exceed the random band only modestly. A large emotion shift on harmful prompts is therefore **not** evidence of a warning signal. This confirms the design: H1/H2 compare compliant vs refused responses *within* harmful prompts.

## 7. Deviations and notes

- **Added (supplementary, not used for any selection):** leave-one-topic-out accuracy and the confusion matrix, because 45 held-out stories give wide CIs.
- **Fixed:** the layer-0 refusal direction was NaN (zero difference vector); `unit()` now returns zeros. The analysis was re-run from cached activations, and all other numbers are identical to the pod run.
- **Precision:** activations are stored as float16, about 60 MB, in git-ignored `data/vectors/`.
- **Not done:** comparison with the thesis-pilot vectors (D3). The `emotional-ai` volume is in CA-MTL-1, which had no free GPU or CPU instance on three attempts. It can be done later if you want; it is optional in TASK.md.

## 8. Plan for Phase 2 (needs go-ahead)

**What runs:**
- 600 prompts (200 behaviours × 3 wrappers), 5 samples each, giving 3,000 generations (T 0.7, top-p 0.9, top-k off, 128 tokens).
- A teacher-forced pass reads the residuals at the last prompt token and at generated tokens 1–20, for all 29 layers.

**What is stored:**
- Projections onto the 5 emotion vectors (raw and denoised), the refusal direction and the 200 random directions: about 0.8 GB as float16, git-ignored.
- Full residuals at the last prompt token (600 × 29 × 3584, about 125 MB).

**Labelling:**
- `cais/HarmBench-Llama-2-13b-cls` labels every harmful response (yes = compliance). It runs after Qwen is unloaded.
- Benign responses are labelled answered / over-refused by the same classifier prompt, with a keyword cross-check. This is to be confirmed in the Phase 2 report.

**Manual validation:** export of 100 random harmful responses to `data/raw/manual_validation.csv` (git-ignored) for your manual labels. Note that labelling these means reading model responses to harmful requests.

**Estimate:** ≈ 35 min of pod time (including the 26 GB judge download), ≈ $0.35.
