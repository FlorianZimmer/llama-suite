#Requires -Version 5.1
<#
.SYNOPSIS
    Starts llama-swap in auto-restart mode.  Whenever config.yaml
    is saved, the current process is stopped and a fresh one starts
    with the *same* argument list.
#>
param()

# ── Configuration ────────────────────────────────────────────────
$ScriptDir   = $PSScriptRoot
$InstallDir  = "C:\Users\Florian\OneDrive\Dokumente\Privat\Programmieren\llama_suite"

# llama-swap paths
$LlamaSwapDir     = Join-Path $InstallDir "llama-swap"
$LlamaSwapExePath = Join-Path $LlamaSwapDir "llama-swap.exe"
$ConfigYamlPath   = Join-Path $ScriptDir   "model-configs\config-desktop.yaml"

# Listen address for clients (Open WebUI, etc.)
$LlamaSwapListenAddress = ":8080"

# ── Validate paths ───────────────────────────────────────────────
if (-not (Test-Path -Path $LlamaSwapExePath -PathType Leaf)) {
    Write-Error "llama-swap executable not found at: $LlamaSwapExePath"
    Read-Host "Press Enter to exit"
    exit 1
}
if (-not (Test-Path -Path $ConfigYamlPath -PathType Leaf)) {
    Write-Error "Configuration file not found at: $ConfigYamlPath"
    Read-Host "Press Enter to exit"
    exit 1
}

# ── Build *exact* argument list you used before ──────────────────
$LlamaSwapArgs = @(
    '--config', $ConfigYamlPath,
    '-listen',  $LlamaSwapListenAddress,
    '-vv'
)

Write-Host "DEBUG ARGS → $($LlamaSwapArgs -join ' | ')"

# ── Pretty banner ────────────────────────────────────────────────
Write-Host "-----------------------------------------------------" -ForegroundColor Cyan
Write-Host "Starting llama-swap  (auto-restart enabled)" -ForegroundColor Cyan
Write-Host "Executable: $LlamaSwapExePath"
Write-Host "Config    : $ConfigYamlPath"
Write-Host "Listening : $LlamaSwapListenAddress"
$DisplayPort = ($LlamaSwapListenAddress -split ':')[-1]
Write-Host "Open WebUI → http://host.docker.internal:$DisplayPort/v1"
Write-Host "-----------------------------------------------------" -ForegroundColor Cyan
Write-Host "--- llama-swap OUTPUT (live & restarts on save) ---"  -ForegroundColor Yellow

# -- Build argument list for llama-swap ---------------------------
$LlamaSwapArgs = @(
    '--config', $ConfigYamlPath,
    '-listen',  $LlamaSwapListenAddress
)

# -- Delegate to helper ------------------------------------------
$RestartScript = Join-Path $InstallDir 'tools\restart-on-config-change.ps1'

Push-Location $LlamaSwapDir
try {
    & "$RestartScript" `
        -ExePath  $LlamaSwapExePath `
        -CmdArgs  $LlamaSwapArgs `
        -ConfigPath $ConfigYamlPath
}
finally {
    Pop-Location
    Write-Host "-----------------------------------------------------" -ForegroundColor Cyan
    Write-Host "llama-swap stopped." -ForegroundColor Cyan
    Read-Host "Press Enter to exit"
}