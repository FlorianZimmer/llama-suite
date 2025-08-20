<# 
  tools/windows/eval.ps1
  Wrapper for: python -m llama_suite.eval.eval
#>

[CmdletBinding()]
param(
  # Required-ish high-level args
  [string]$Data,
  [string]$OutDir,
  [Parameter(Mandatory=$true)][string]$Endpoint,
  [Parameter(Mandatory=$true)][string]$JudgeModel,

  # Model selection
  [switch]$ModelsFromSwap,            # defaulted to $true after param parsing
  [string]$SwapConfig,                # defaults to configs\config.base.yaml

  # Runtime behavior
  [int]$Timeout = 7200,
  [int]$HealthWaitHeavy = 180,
  [int]$HealthWait = 10,
  [switch]$LlamaGuard,                # default OFF; we add --no-llama-guard unless set

  # Optional passthroughs (e.g., --concurrency, --temp, --max-tokens, --models, etc.)
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Extra
)

# Default ModelsFromSwap to $true unless explicitly provided (satisfies PSScriptAnalyzer)
if (-not $PSBoundParameters.ContainsKey('ModelsFromSwap')) {
  $ModelsFromSwap = $true
}

# --- Repo root detection (this file is at tools/windows/eval.ps1) ---
$ScriptDir = $PSScriptRoot
$RepoRoot  = Split-Path (Split-Path $ScriptDir -Parent) -Parent

# --- Choose Python (prefer venv) ---
$PythonCandidates = @(
  (Join-Path $RepoRoot "llama-suite-venv\Scripts\python.exe"),
  (Join-Path $RepoRoot ".venv\Scripts\python.exe")
)
$Python = $null
foreach ($p in $PythonCandidates) {
  if (Test-Path $p) { $Python = $p; break }
}
if (-not $Python) { $Python = "python" }

# --- Resolve data path ---
function Resolve-DataPath([string]$p) {
  if (-not $p) { return $null }
  if (Test-Path $p) { return (Resolve-Path $p).Path }
  $candidate = Join-Path $RepoRoot ("datasets\custom\" + $p)
  if (Test-Path $candidate) { return $candidate }
  throw "Data file not found: '$p' (also tried datasets\custom\$p)."
}

# --- Resolve OutDir (default to runs\eval\runs\<YYYY-MM-DD>) ---
function Resolve-OutDir([string]$p) {
  if ([string]::IsNullOrWhiteSpace($p)) {
    $stamp = Get-Date -Format "yyyy-MM-dd"
    $p = Join-Path $RepoRoot ("runs\eval\runs\" + $stamp)
  } elseif (-not (Split-Path $p -IsAbsolute)) {
    $p = Join-Path $RepoRoot $p
  }
  New-Item -ItemType Directory -Force -Path $p | Out-Null
  return (Resolve-Path $p).Path
}

# --- Resolve SwapConfig (default to configs\config.base.yaml) ---
function Resolve-SwapConfig([string]$p) {
  if ([string]::IsNullOrWhiteSpace($p)) {
    $p = Join-Path $RepoRoot "configs\config.base.yaml"
  } elseif (-not (Split-Path $p -IsAbsolute)) {
    if (-not (Test-Path $p)) { $p = Join-Path $RepoRoot $p }
  }
  if (-not (Test-Path $p)) { throw "Swap config not found: '$p'" }
  return (Resolve-Path $p).Path
}

# --- Build arguments ---
$dataPath = Resolve-DataPath $Data
$outPath  = Resolve-OutDir  $OutDir
$swapPath = Resolve-SwapConfig $SwapConfig

$ArgsList = @(
  '-m', 'llama_suite.eval.eval',
  '--endpoint', $Endpoint,
  '--judge-model', $JudgeModel,
  '--timeout', $Timeout,
  '--health-wait-heavy', $HealthWaitHeavy,
  '--health-wait', $HealthWait,
  '--swap-config', $swapPath
)

if ($dataPath) { $ArgsList += @('--data', $dataPath) }
if ($outPath)  { $ArgsList += @('--out-dir', $outPath) }

# models-from-swap vs. explicit models
if ($ModelsFromSwap) {
  $ArgsList += '--models-from-swap'
}

# llama guard toggle (default OFF)
if ($LlamaGuard) { 
  $ArgsList += '--llama-guard'
} else {
  $ArgsList += '--no-llama-guard'
}

# Forward anything else verbatim
if ($Extra) { $ArgsList += $Extra }

# --- Run from repo root so relative imports work ---
Push-Location $RepoRoot
try {
  Write-Host "`nRunning eval via:" -ForegroundColor Cyan
  Write-Host ("  {0} {1}" -f $Python, ($ArgsList -join ' ')) -ForegroundColor DarkGray
  Write-Host ""

  & $Python @ArgsList
  $code = $LASTEXITCODE
} finally {
  Pop-Location
}

exit ($code -as [int])