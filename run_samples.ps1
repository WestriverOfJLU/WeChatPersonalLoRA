$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$WslRoot = (wsl -- wslpath -a -u "$ProjectRoot").Trim()
$WslRootEscaped = $WslRoot.Replace("'", "'\''")

wsl -- bash -lc "cd '$WslRootEscaped' && ./scripts/run_samples.sh"
