#!/bin/bash
# Phase 6B GPU run. Usage on the pod: cd /workspace/proj && bash scripts/pod_phase6.sh
set -euo pipefail
export HF_HOME=${HF_HOME:-/workspace/hf}
bash scripts/pod_setup.sh
python -c "from huggingface_hub import snapshot_download; snapshot_download('mistralai/Mistral-7B-Instruct-v0.3', revision='c170c708c41dac9275d15a8fff4eca08d52bab71', allow_patterns=['*.json','model-*.safetensors','tokenizer*'])"
# smoke: 2 behaviours per kind, both models
for M in qwen mistral; do python scripts/15_steer.py --config configs/phase6.yaml --model $M --limit 2; done
echo SMOKE_OK
for M in qwen mistral; do python scripts/15_steer.py --config configs/phase6.yaml --model $M; done
for M in qwen mistral; do python scripts/04_judge.py --config configs/phase6.yaml --tag p6_$M --refusal-scope benign; done
echo "PHASE6_DONE"
