$ErrorActionPreference = "Stop"
$projectDir = $PSScriptRoot
$parentVenvPython = Join-Path (Split-Path -Parent $projectDir) ".venv\Scripts\python.exe"

if (Test-Path -LiteralPath $parentVenvPython) {
    $pythonExe = $parentVenvPython
} else {
    $pythonExe = (Get-Command python -ErrorAction Stop).Source
}

& $pythonExe (Join-Path $projectDir "app.py")
exit $LASTEXITCODE

