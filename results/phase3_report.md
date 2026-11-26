# Phase 3 report: analysis

**Status:** done. Full results and interpretation are in `results/summary.md`. Figures are in `results/figures/fig2…fig5` (report style) and `poster/figures/` (poster style).

## What was run

| Analysis | Data | Config | Output |
|---|---|---|---|
| Pre-registered H1, H2, early warning, wrappers, false alarms | fixed wrappers (`main`) | `configs/phase3.yaml` | `results/phase3_metrics.json` |
| Same analysis, settings unchanged | exploratory PAIR set (harmful) + `main` benign | `configs/phase3_pair.yaml` | `results/phase3_metrics_pair.json` |
| Judge agreement | stratified validation samples | `scripts/05b_judge_validation.py` | `results/judge_validation.json` |

- **Settings fixed in `configs/phase3.yaml` before looking at labels or projections:**
  - k list and features (mean projection over end-of-prompt + tokens 1…k);
  - logistic C = 1;
  - layer selection on val per monitor and k;
  - (d) = refusal at (c)'s layer + emotions at (b)'s layer;
  - 200 random 5-direction monitors;
  - cluster bootstrap over behaviours (2,000);
  - H1 layers fixed from Phase 1 (17 for emotions, 19 for refusal), H1 on val + test only;
  - false-alarm threshold at 80 % val recall.
- **Runtime:** ≈ 2.3 min per analysis on an M1 Pro CPU (Python 3.12, numpy 2.x, scikit-learn; see the `env_analysis` block in the metrics JSON).

## Results in one paragraph

- **Pre-registered analysis:** not estimable. There is 1 compliant response in the test split.
- **Exploratory PAIR set:**
  - At the prompt (k = 0) no monitor beats chance on held-out behaviours (AUROC 0.42–0.55).
  - From 1 generated token on, all monitors reach 0.77–0.87, but 200 random 5-direction monitors reach a median of 0.76–0.83. The emotion monitor never exceeds the random 95th percentile (best: 94.5th at k = 5).
  - Refusal + emotion vs refusal: Δ ≤ +0.04, except +0.14 at k = 10 (p = 0.04 uncorrected; not significant with Bonferroni over 6 k).
  - H1 effects exist within prompts once the answer is being written (|d_z| ≈ 1.1–1.2 vs a random 95th percentile of 0.94), but with the **opposite sign**: refusals look more desperate and less calm.
- **H3, text only:** emotional pressure moves the emotion projections as intended (desperate +0.92 SD, calm −0.91 SD), leaves compliance unchanged (Δ 0.0 pp) and raises benign over-refusal from 29 % to 49 %.

## Deviations and fixes (all documented, none tuned on results)

1. **Exploratory PAIR extension** (see `phase2_report.md` §5). It was added because of the 0.9 % base rate, before any Phase 3 analysis was run.
2. **Bug fix: permutation p-values.** They had no floating-point tolerance, which reported p = 0 where the exact minimum was 2/16. Fixed and re-run; the effect sizes were unchanged.
3. **Bug fix: hard-coded 5 samples.** The mixed-prompt count and the within-prompt filter assumed 5 samples per prompt. Generalised for the 10-sample PAIR run before that analysis was run.
4. **Added (not in TASK.md):** an independent LLM review of the judge, as an interim check while human labels are pending.
