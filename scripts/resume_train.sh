#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CHECKPOINT="${1:-outputs/qwen3_8b_wechat_lora/checkpoint-500}"
mkdir -p logs outputs
source .venv/bin/activate

nohup env \
  HF_HUB_DISABLE_XET=1 \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  python scripts/train_lora.py \
  --model Qwen/Qwen3-8B \
  --train data/train_context.jsonl \
  --val data/val_context.jsonl \
  --output outputs/qwen3_8b_wechat_lora \
  --max-seq-length 1024 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 16 \
  --eval-max-samples 256 \
  --eval-steps 500 \
  --save-steps 500 \
  --logging-steps 10 \
  --resume-from-checkpoint "$CHECKPOINT" \
  >> logs/qwen3_8b_train.log 2>&1 &

echo $! > logs/qwen3_8b_train.pid
echo "Resumed training from $CHECKPOINT with PID $(cat logs/qwen3_8b_train.pid)"
