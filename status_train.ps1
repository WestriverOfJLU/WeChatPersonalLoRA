$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$WslRoot = (wsl -- wslpath -a -u "$ProjectRoot").Trim()
$WslRootEscaped = $WslRoot.Replace("'", "'\''")

wsl -- bash -lc "'$WslRootEscaped/scripts/status_train.sh'"
