#!/bin/bash
# One-time setup on a RunPod pod (image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404).
# Usage (on the pod): cd /workspace/proj && bash scripts/pod_setup.sh
set -euo pipefail
export HF_HOME=${HF_HOME:-/workspace/hf}
pip install -q --break-system-packages -r requirements.txt
pip install -q --break-system-packages torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch, torchvision, transformers; print(torch.__version__, torchvision.__version__, transformers.__version__)"
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-Omni-7B', revision='ae9e1690543ffd5c0221dc27f79834d0294cba00')"
# Phase 2 judge (public, MIT), pinned revision
python -c "from huggingface_hub import snapshot_download; snapshot_download('cais/HarmBench-Llama-2-13b-cls', revision='bda705349d1144fa618770bea64d99ce54e3835b', allow_patterns=['*.json','*.model','*.safetensors'])"
