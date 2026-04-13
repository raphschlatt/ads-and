param(
    [string]$EnvFile = ".env",
    [string]$WorkDir = "tmp/modal_mwe_smoke",
    [string]$Runtime = "gpu",
    [string]$InferStage = "smoke"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

function Import-DotEnv([string]$Path) {
    if (-not (Test-Path $Path)) { return }
    foreach ($line in Get-Content $Path) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#") -or -not $trimmed.Contains("=")) { continue }
        $name, $value = $trimmed -split "=", 2
        $name = $name.Trim()
        $value = $value.Trim().Trim("'").Trim('"')
        if ($name -in @("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")) {
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

Import-DotEnv $EnvFile

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv not found. Open a new PowerShell terminal so the updated PATH is picked up."
}
$PythonExe = (Get-Command python -ErrorAction Stop).Source

$hasEnvToken = [string]::IsNullOrWhiteSpace($env:MODAL_TOKEN_ID) -eq $false -and [string]::IsNullOrWhiteSpace($env:MODAL_TOKEN_SECRET) -eq $false
$hasModalSetup = Test-Path (Join-Path $HOME ".modal.toml")
if (-not $hasEnvToken -and -not $hasModalSetup) {
    throw "No Modal auth found. Either run 'modal setup' in the active env or put MODAL_TOKEN_ID / MODAL_TOKEN_SECRET into $EnvFile."
}

$InputDir = Join-Path $WorkDir "input"
$OutputDir = Join-Path $WorkDir "out"

uv run --python $PythonExe python .\scripts\modal_mwe\build_testset.py --output-dir $InputDir
uv run --python $PythonExe --with modal python -m modal deploy .\scripts\modal_mwe\modal_app.py
uv run --python $PythonExe --with modal python .\scripts\modal_mwe\client_mwe.py `
    --publications (Join-Path $InputDir "publications.parquet") `
    --references (Join-Path $InputDir "references.parquet") `
    --output-dir $OutputDir `
    --runtime $Runtime `
    --infer-stage $InferStage

Write-Host ""
Write-Host "Smoke test finished:"
Write-Host "  Input:  $InputDir"
Write-Host "  Output: $OutputDir"
