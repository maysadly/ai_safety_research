# Can Emotion Vectors Warn Us Before a Model Breaks Its Safety Rules?

AI safety course project (QAIRU, AI+X master's programme) by Danat Bakzhanov, Yerkin Maratuly, Kasymzhomart Beketov, Tleukhan Makhmutov.

We test whether internal emotion representations ("emotion vectors") in **Qwen2.5-Omni-7B** predict that the model is about to comply with a harmful request, and whether they add information beyond the known refusal direction.

**Short answer: no early warning.**
- Before the model writes anything, neither emotion vectors nor the refusal direction predict compliance on unseen JailbreakBench behaviours (AUROC 0.42–0.55).
- After 1–5 generated tokens compliance is easy to read (AUROC ≈ 0.8–0.87), but random directions read it about as well.
- Emotion projections add no robust value beyond the refusal direction.
- Emotional-pressure prompts raise benign over-refusal (29 % → 49 %) without raising harmful compliance.

Full results: [`results/summary.md`](results/summary.md). Poster: [`poster/poster.pdf`](poster/poster.pdf).

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

GPU work took about 1 GPU-hour on RunPod (A40, plus one A100 run) and cost $0.72 in total.

> Ethics: harmful prompts come only from JailbreakBench, including its published PAIR artifacts; no new jailbreaks were created. Model completions to harmful prompts are never committed or published; only IDs, labels and aggregate metrics are. Raw generations stay in the git-ignored `data/raw/`.
