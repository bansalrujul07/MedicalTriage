param(
    [Parameter(Mandatory = $false, Position = 0)]
    [string]$PingUrl,

    [Parameter(Mandatory = $false, Position = 1)]
    [string]$RepoDir = "."
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if ([string]::IsNullOrWhiteSpace($PingUrl)) {
    Write-Host "Usage: .\validate-submission.ps1 <ping_url> [repo_dir]"
    Write-Host ""
    Write-Host "  ping_url   Your HuggingFace Space URL (for example https://your-space.hf.space)"
    Write-Host "  repo_dir   Path to your repo (default: current directory)"
    exit 1
}

$PythonCandidates = @(
    Join-Path $ScriptDir ".venv\Scripts\python.exe",
    (Get-Command python3 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
    (Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue)
) | Where-Object { $_ -and (Test-Path $_) }

if (-not $PythonCandidates -or $PythonCandidates.Count -eq 0) {
    Write-Error "No Python interpreter found. Create .venv or make python/python3 available on PATH."
    exit 127
}

$PythonBin = $PythonCandidates[0]
$ValidationScript = Join-Path $ScriptDir "validation.py"

& $PythonBin $ValidationScript $PingUrl $RepoDir
exit $LASTEXITCODE