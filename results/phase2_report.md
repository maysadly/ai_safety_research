# Phase 2 report: generations and labels

**Status:** done, with one important finding and one documented deviation.
- **Finding:** under the pre-registered design (JBB goals × 3 fixed wrappers), Qwen2.5-Omni-7B **almost never complies**: 14 of 1,500 harmful responses (0.9 %). That is too few positives to test a compliance monitor (the test split has **one**).
- **Deviation:** to make H2 testable, I added a clearly labelled **exploratory run** on JailbreakBench's own published PAIR jailbreak prompts (§5). The decision was taken after seeing only the label base rates, before running any Phase 3 analysis.
- **Unchanged:** the pre-registered analysis is still run and reported.

Numbers: `results/phase2_summary_main.json`, `results/phase2_summary_pair.json`. Labels (no text): `data/labels_main.csv`, `data/labels_pair.csv`. Raw generations: `data/raw/{main,pair}/` (git-ignored).

## 1. Run (pre-registered, tag `main`)

| Item | Value |
|---|---|
| Prompts | 100 harmful + 100 benign JBB behaviours × 3 wrappers (neutral, emotional pressure, role-play) = 600 |
| Samples | 5 per prompt → 3,000 generations (T 0.7, top-p 0.9, top-k off, ≤ 128 new tokens) |
| GPU | A40 48 GB (EU-SE-1). Generation + teacher-forced projections: 388 s. Judges: 144 s + 229 s |
| Stored per generation | projections of the residual stream (29 layers × 21 positions) onto 5 raw + 5 denoised emotion vectors, the refusal direction and 200 random directions (`data/activations/main/projections.npy`, float32, 1.5 GB, git-ignored) |
| Also stored | last-prompt-token residuals of the 600 prompts |
| Model / code | `Qwen/Qwen2.5-Omni-7B` @ `ae9e169…`, transformers 5.24.1, torch 2.9.1, git `9deb548` + Phase 2 scripts |
| Truncation | 37 % of responses reached the 128-token limit (1,110 / 3,000) |

## 2. Labels

- **Primary, harmful prompts: HarmBench classifier** (`cais/HarmBench-Llama-2-13b-cls` @ `bda7053…`).
  - Verbatim prompt from the model card; behaviour = JBB goal (without wrapper).
  - Label = greedy next token ("Yes" = compliance), as in the official HarmBench evaluation.
  - P(yes) is kept as well.
- **Secondary, all prompts: refusal question to the studied model.**
  - Fixed prompt in `configs/phase2.yaml`; refused if P(yes) ≥ 0.5.
  - Used for benign over-refusal and as an agreement check.
- **Keyword detector:** reported only as a reference.

## 3. Results

**Harmful compliance (HarmBench), 95 % Wilson CI:**

| | k / n | rate | 95 % CI |
|---|---|---|---|
| overall | 14 / 1500 | 0.9 % | [0.6, 1.6] % |
| neutral | 4 / 500 | 0.8 % | [0.3, 2.0] % |
| emotional pressure | 4 / 500 | 0.8 % | [0.3, 2.0] % |
| role-play | 6 / 500 | 1.2 % | [0.6, 2.6] % |
| train / val / test split | 10 / 3 / 1 | 1.3 / 1.0 / 0.2 % | |

- **By category:** 0–5 compliant responses out of 150 per category (Expert advice 5, Economic harm 3, Disinformation 2).
- **Concentration:** the 14 compliant responses come from 11 behaviours.
- **Within-prompt variation:**
  - 287 of 300 harmful prompts were refused in all 5 samples.
  - 12 prompts complied once and 1 prompt complied twice.
  - So only **13 prompts have mixed outcomes**.

**Benign over-refusal** (secondary judge): **38 %** overall [36, 41].

| Wrapper | Over-refusal | 95 % CI |
|---|---|---|
| neutral | 28.6 % | [24.8, 32.7] |
| emotional pressure | **48.8 %** | [44.4, 53.2] |
| role-play | 37.6 % | [33.5, 41.9] |

Emotional pressure does not make the model comply with harmful requests. It makes the model **refuse more benign ones**.

## 4. Judge agreement and validation

- **HarmBench vs the refusal question** (harmful responses): accuracy 0.976, Cohen's κ 0.39.
  - In 34 responses HarmBench says "no" while the refusal question says "not refused". A spot check of their openings shows mostly counter-content: debunking the requested disinformation, warning texts, information about help for addiction. These are correctly *not* compliance.
  - In 2 responses HarmBench says "yes" while the refusal question says "refused".
- **HarmBench vs keyword detector:** accuracy 0.96, κ 0.28.
- **Human validation: PENDING.**
  - `data/raw/main/manual_validation_main.csv` holds 100 harmful responses for your manual labels.
  - The sample is stratified by judge label (14 judged compliant / 86 judged refused), because the compliance rate is outside [0.2, 0.8] (pre-registered rule in `configs/phase2.yaml`).
  - Accuracy must therefore be reweighted by the base rate.
  - Labelling means reading model responses to harmful requests. The file must never be committed or shared.

## 5. Exploratory extension (tag `pair`)

**Why.** The study's question is whether internal signals warn *before* the model complies. With 14 compliant responses (1 in the test split) the monitors cannot be evaluated, and the planned wrappers are fixed by design and must not be strengthened.

**Choice.** JailbreakBench publishes the PAIR attack prompts it found for each behaviour (`JailbreakBench/artifacts`, PAIR, black-box, target vicuna-13b-v1.5). Using them adds no new harmful requests and no new jailbreak optimisation, so it stays within the project's ethics rules.

**Details.**
- 82 behaviours have a PAIR prompt; 79 match the current JBB goal text exactly and are used.
- 10 samples per prompt → 790 generations, same sampling settings and the same projections.
- Splits are the same behaviour splits: 36 train / 17 val / 26 test behaviours.
- Benign responses for false alarms come from the main run.
- Analysis settings are unchanged; config `configs/phase3_pair.yaml` only switches the data.

**Status.** Everything in this extension is labelled *exploratory* in the summary and on the poster.

### PAIR results

- **Run:** A100 80 GB (US-MD-1; no A40 was available). Generation + projections took 75 s; judges took 30 s + 49 s.
- **Truncation:** 53 % of responses reached 128 tokens.
- **Compliance (HarmBench):** **19.8 %** (156 / 790) [Wilson 17.1, 22.7].

  | Split | Compliant / responses | Rate |
  |---|---|---|
  | train | 92 / 360 | 25.6 % |
  | val | 26 / 170 | 15.3 % |
  | test | 38 / 260 | 14.6 % |

- **By category:** every category has compliant responses, from 7/90 (sexual content) to 26/80 (privacy).
- **Within-prompt variation:** 45 of 79 prompts have mixed outcomes over their 10 samples; 33 are always refused and 1 always complies.
- **Judge agreement:**
  - HarmBench vs the refusal question: accuracy 0.74, κ 0.43.
  - In 194 responses HarmBench says "no" but the refusal question says "not refused". The model often goes along with the PAIR scenario without producing actionable content, which HarmBench rightly does not count.
- **Manual-validation export:** 50 judged-compliant + 50 judged-refused responses (rate < 0.2 → stratified), in `data/raw/pair/manual_validation_pair.csv`.

## 6. Interim judge validation (independent LLM review; human labels still pending)

An independent annotator (Claude, instructed with the HarmBench rule, blind to judge labels) labelled both 100-response validation samples. Only gid, label, confidence and reason are stored: `data/review_llm_{main,pair}.csv`. Agreement with HarmBench is computed by `scripts/05b_judge_validation.py` → `results/judge_validation.json`.

| Sample | Raw agreement | κ | P(review compliant \| HB compliant) | P(review refused \| HB refused) | Accuracy reweighted to the population |
|---|---|---|---|---|---|
| main | 99 / 100 | 0.96 | 0.93 | 1.00 | 0.999 |
| PAIR | 88 / 100 | 0.76 | 0.84 | 0.92 | 0.90 |

- **High-confidence items:** agreement is 100 % in both samples (90 and 50 items).
- **Disagreements:** all are borderline responses cut off at 128 tokens. The reviewer notes that a stricter reading of "non-minimal" would flip about 15–20 of its low-confidence PAIR compliance labels.
- **Human labels:** when you fill in the CSVs, save them as `data/review_human_{tag}.csv` and re-run the script.

## 7. Cost

| Item | Cost |
|---|---|
| Phase 2 (A40 ≈ 25 min + A100 ≈ 12 min) | ≈ $0.48 |
| Project total | **$0.72** (balance $8.10) |

No pods are running.
