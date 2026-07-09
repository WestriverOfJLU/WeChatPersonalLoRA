#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python scripts/chat_batch.py \
  --model Qwen/Qwen3-8B \
  --adapter outputs/qwen3_8b_wechat_lora \
  --prompts scripts/sample_prompts.txt \
  --max-new-tokens 96
