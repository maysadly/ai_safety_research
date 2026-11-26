# CLAUDE.md — AI Safety research project

## What this repo is

A small, self-contained research project for the QAIRU master's course in AI safety, ending in an **academic poster**.

**Working title:** *Can Emotion Vectors Warn Us Before a Model Breaks Its Safety Rules?*
**Model:** Qwen2.5-Omni-7B (open weights; text input only for the main study).
**Author:** Tileukhan Makhmutov, QAIRU (AI+X master's programme).

The project is a spin-off of the author's master's thesis, which studies whether emotional videos induce functional emotional states in Qwen2.5-Omni. This project reuses the same emotion-vector method but asks a safety question: **do internal emotion signals predict, before it happens, that the model will comply with a harmful request — and do they add anything beyond the known refusal direction?**

The full plan, phases and deliverables are in **`TASK.md`**. Read it before doing anything.

## How to work

- Work in phases (see `TASK.md`). **Stop at the end of every phase**, write a short report to `results/phaseN_report.md`, and wait for the user's go-ahead before starting the next phase.
- Prefer small, readable Python scripts over notebooks. Every script takes a config (`configs/*.yaml`) and writes outputs to `results/` or `data/`.
- Fix random seeds; log package versions, GPU, model revision and runtime in every report.
- Before writing model-loading code, check the current Hugging Face docs for Qwen2.5-Omni (class names and the thinker-only / talker-disabled setup change between `transformers` versions). Do not guess APIs.
- Run a tiny smoke test (2–3 prompts) before any long run. Estimate runtime and GPU cost and tell the user before launching anything longer than ~15 minutes.
- Report results honestly. A negative or weak result is acceptable and must be stated plainly; never tune the analysis until it "works".

## Compute

- The user rents GPUs on RunPod (an NVIDIA A40, 48 GB, was used for the pilot). Keep total GPU time small (target: under ~4 hours for the whole project).
- Never hard-code tokens. Read `HF_TOKEN` and any other secrets from environment variables. Never commit `.env`.
- Cache models under `$HF_HOME` (on the pod: `/workspace/hf`).

## Safety and ethics rules (strict)

- This project generates responses to harmful requests in order to measure safety behaviour. **Generated completions must never be committed, published or shown on the poster.**
  - Raw generations go only to `data/raw/` (git-ignored).
  - Committed artefacts contain only IDs, labels (refusal / compliance), activations-derived numbers and aggregate metrics.
- Use only established public benchmarks for harmful prompts. Do not invent new harmful requests and do not optimise new jailbreaks; the prompt wrappers are fixed and simple (see `TASK.md`).
  - Allowed: JailbreakBench (behaviours and its published attack artifacts) and HarmBench (text behaviours and its published human-jailbreak templates). The second set was approved by the user on 2026-11-26 for the Phase 5 confirmatory study.
  - Models: Qwen2.5-Omni-7B (main) and, for Phase 5 only, Mistral-7B-Instruct-v0.3 as a second model.
- Do not use any model output to actually carry out harmful actions.

## Repo layout (create as needed)

```
configs/        YAML configs for every step
src/            reusable code (model loading, activations, vectors, probes, metrics)
scripts/        entry points, one per step (01_..., 02_..., ...)
data/           git-ignored except small metadata files
results/        metrics, figures (PNG + SVG), phase reports
poster/         poster source (HTML/CSS) and exported PDF
```

## Writing style

- Code comments and reports: concise English.
- Poster: English, academic but readable; numbers with units and confidence intervals; no harmful text.
