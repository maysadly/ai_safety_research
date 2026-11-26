#!/bin/bash
# Phase 5 GPU run (one pod). Usage on the pod: cd /workspace/proj && bash scripts/pod_phase5.sh [smoke|full]
set -euo pipefail
export HF_HOME=${HF_HOME:-/workspace/hf}
MODE=${1:-full}
if [ "$MODE" = "smoke" ]; then
  bash scripts/pod_setup.sh
  python -c "from huggingface_hub import snapshot_download; snapshot_download('mistralai/Mistral-7B-Instruct-v0.3', revision='c170c708c41dac9275d15a8fff4eca08d52bab71', allow_patterns=['*.json','*.safetensors','tokenizer*'])"
  # second model: Phase-1 pipeline (stories, emotion vectors, refusal direction) -> fixed layers by rule
  python scripts/01_generate_stories.py --config configs/phase5_mistral_p1.yaml
  python scripts/02_build_vectors.py --config configs/phase5_mistral_p1.yaml
  for M in qwen mistral; do
    python scripts/03_generate.py --config configs/phase5.yaml --tag p5s_$M --source phase5 --model $M --limit 12
    python scripts/04_judge.py --config configs/phase5.yaml --tag p5s_$M --refusal-scope benign
  done
else
  for M in qwen mistral; do
    python scripts/03_generate.py --config configs/phase5.yaml --tag p5_$M --source phase5 --model $M
  done
  for M in qwen mistral; do
    python scripts/04_judge.py --config configs/phase5.yaml --tag p5_$M --refusal-scope benign
  done
fi
echo "PHASE5_${MODE}_DONE"
