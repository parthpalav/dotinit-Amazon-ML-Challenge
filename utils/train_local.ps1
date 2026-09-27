param(
    [string]$Python = '',
    [ValidateSet('GPU', 'CPU')][string]$Device = 'GPU',
    [string]$Output = 'artifacts/local_training'
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    if (-not $Python) {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($pythonCommand) { $Python = $pythonCommand.Source }
        else {
            $Python = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
        }
    }
    if (-not (Test-Path -LiteralPath $Python)) {
        throw 'Python was not found. Pass -Python with the full path to a Python 3.12 executable.'
    }
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
$venvPython = Join-Path (Get-Location) '.venv/Scripts/python.exe'
& $venvPython -m pip install -r requirements-local-training.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed; training was not started.' }
& $venvPython -u -m src.train_local --device $Device --output $Output
if ($LASTEXITCODE -ne 0) { throw 'Local training failed. See the error above.' }
