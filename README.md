# Can Emotion Vectors Warn Us Before a Model Breaks Its Safety Rules?

AI safety course project (QAIRU, AI+X master's programme) by Danat Bakzhanov, Yerkin Maratuly, Kasymzhomart Beketov, Tleukhan Makhmutov.

We test whether internal emotion representations ("emotion vectors") in **Qwen2.5-Omni-7B** predict that the model is about to comply with a harmful request, and whether they add information beyond the known refusal direction.

**Answer (pre-registered confirmatory study, 2 models, 17,540 answers):** emotion vectors are not a special early-warning signal.
- **Little added value.** They add only +0.005 AUROC to the refusal direction in Qwen2.5-Omni-7B. In Mistral-7B the refusal direction is weak and emotion features help, but random directions come close.
- **Pre-generation warning works only for direct requests.** On jailbreaks it is weak.
- **Within a prompt,** Qwen's refusals look *more* desperate and *less* calm.
- **Emotional pressure** raises refusals of harmless requests in both models.

**But emotion vectors are a strong causal lever (pre-registered steering test, 9,600 answers):**
- **Over-refusal falls.** Steering away from "desperate" during generation cuts over-refusal of distressed users from 53 % to 32 % (Qwen) and from 34 % to 9 % (Mistral).
- **No rise in harmful compliance.**
- **Specific.** The effect is well beyond that of a random direction.

In short, a poor alarm but a strong lever.

Full results: [`results/summary.md`](results/summary.md), [`results/phase5_report.md`](results/phase5_report.md), [`results/phase6_report.md`](results/phase6_report.md).
Posters: **A1 landscape (final)** [`poster/a1/poster_A1_landscape.pdf`](poster/a1/poster_A1_landscape.pdf); v1 A0 portrait (exploratory phases) [`poster/poster.pdf`](poster/poster.pdf).
Presentation scripts: [`poster/presentation_script.md`](poster/presentation_script.md) (A1), [`poster/presentation_script_v1.md`](poster/presentation_script_v1.md) (v1).

- Plan and phases: [`TASK.md`](TASK.md); working rules: [`CLAUDE.md`](CLAUDE.md)
- Phase reports: [`results/phase0_report.md`](results/phase0_report.md) … [`results/phase4_report.md`](results/phase4_report.md)

## Reproduce

| Step | Where | Command |
|---|---|---|
| 0 | local | `python scripts/00_make_splits.py` (fixed behaviour split) |
| 1 | GPU | `bash scripts/pod_setup.sh` |
| 1a | GPU | `python scripts/01_generate_stories.py --config configs/phase1.yaml` |
| 1b | GPU | `python scripts/02_build_vectors.py --config configs/phase1.yaml` (re-analyse locally with `--from-cache`) |
| 2 | GPU | `python scripts/03_generate.py --config configs/phase2.yaml --tag main`, then `python scripts/04_judge.py --config configs/phase2.yaml --tag main` |
| 2, exploratory | GPU | `python scripts/03_generate.py --config configs/phase2.yaml --tag pair --source pair --samples 10`, then `04_judge.py --tag pair` |
| 2c | local | `python scripts/05_labels.py --tag {main,pair}`; `python scripts/05b_judge_validation.py` |
| 3 | local | `python scripts/06_analysis.py --config configs/phase3.yaml` and `--config configs/phase3_pair.yaml`; `python scripts/07_figures.py [--poster]` |
| 4 | local | `python poster/build_poster.py && python poster/export.py` |
| 5 | GPU | `bash scripts/pod_phase5.sh smoke` then `full`; locally `python scripts/10_confirmatory.py --model {qwen,mistral}`, `scripts/13_a1_figures.py`, `poster/a1/build_a1.py`, `poster/a1/export_a1.py` |
| 6A | local | `python scripts/14_overrefusal.py` |
| 6B | GPU | `bash scripts/pod_phase6.sh` (steered generation + judges); locally `python scripts/16_steer_analysis.py --model {qwen,mistral}` |

GPU work cost about $7.2 in total on RunPod (A40 and A100), of which Phase 6 was $0.78.

> Ethics: harmful prompts come only from JailbreakBench, including its published PAIR artifacts; no new jailbreaks were created. Model completions to harmful prompts are never committed or published; only IDs, labels and aggregate metrics are. Raw generations stay in the git-ignored `data/raw/`.
