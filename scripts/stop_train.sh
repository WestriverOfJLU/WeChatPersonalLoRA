#!/usr/bin/env bash
set -euo pipefail

PIDS="$(pgrep -f "scripts/train_lora.py.*outputs/qwen3_8b_wechat_lora" || true)"
if [[ -z "$PIDS" ]]; then
  echo "No matching training process found."
  exit 0
fi

echo "$PIDS" | xargs kill
echo "Sent SIGTERM to: $PIDS"
