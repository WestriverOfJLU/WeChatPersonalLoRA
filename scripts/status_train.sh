#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT/logs/qwen3_8b_train.log"

echo "== Process =="
pgrep -af "outputs/qwen3_8b_wechat_lora" || true

echo
echo "== GPU =="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader

echo
echo "== Recent Progress =="
if [[ -f "$LOG" ]]; then
  python3 - "$LOG" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(errors="ignore")
progress = re.findall(r"\s*(\d+)%\|[^|\n]*\|\s*(\d+)/4481\s*\[[^\n\r]*", text)
metrics = re.findall(r"\{[^{}]*(?:loss|eval_loss|train_runtime)[^{}]*\}", text)

if progress:
    pct, step = progress[-1]
    print(f"latest_step={step}/4481 ({pct}%)")
else:
    print("No progress line found yet.")

for item in metrics[-8:]:
    print(item)
PY
else
  echo "No log file yet: $LOG"
fi
