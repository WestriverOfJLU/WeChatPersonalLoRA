$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$WslRoot = (wsl -- wslpath -a -u "$ProjectRoot").Trim()
$WslRootEscaped = $WslRoot.Replace("'", "'\''")

wsl -- bash -lc "cd '$WslRootEscaped' && source .venv/bin/activate && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/chat_loop.py --temperature 0.3 --top-p 0.7 --history-turns 10 --max-new-tokens 64"
