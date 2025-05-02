#Requires -Version 5.1
#Requires -Modules @{ModuleName = 'Microsoft.PowerShell.Utility'; ModuleVersion = '1.0.0.0'}, @{ModuleName = 'Microsoft.PowerShell.Management'; ModuleVersion = '1.0.0.0'}

<#
.SYNOPSIS
    Benchmarks LLMs defined in a llama-swap configuration file, including memory usage (GB).
.DESCRIPTION
    Performs a two-pass benchmark:
    1. Memory Scan: Briefly starts each model server to parse VRAM and CPU RAM usage from startup logs.
    2. Benchmark: Runs a predefined query against selected models, measuring performance (TPS)
       and combining it with the pre-scanned memory data.

    Outputs results to an XLSX file by default, with a CSV option available.
    Compatible with PowerShell 5.1.
.PARAMETER ConfigPath
    Path to the llama-swap configuration YAML file.
    Default: '..\config.yaml' (relative to script directory).
.PARAMETER OutputFile
    Base path for the output results file. The correct extension (.xlsx or .csv) will be appended.
    Default: 'benchmark_results_memory.xlsx'.
.PARAMETER Question
    The question/prompt used for benchmarking model response speed.
    Default: 'Why is the sky blue?'.
.PARAMETER HealthCheckTimeoutSeconds
    Maximum seconds to wait for a llama-server to become responsive after starting.
    Default: Value from config file or 60 seconds.
.PARAMETER HealthCheckPollIntervalSeconds
    Seconds between health check attempts during server startup.
    Default: 2.
.PARAMETER ModelToTest
    Optional. If specified, only benchmarks the model with this exact name (must match a key
    under 'models:' in the config). Skips Pass 1 if specified.
    If omitted, all models in the config file are scanned (Pass 1) and benchmarked (Pass 2).
.PARAMETER OutputFormat
    Specifies the output file format. Valid options are 'xlsx' (default) or 'csv'.
.EXAMPLE
    .\benchmark_models.ps1
    # Scans memory for all models, benchmarks all models, outputs to benchmark_results_memory.xlsx.
.EXAMPLE
    .\benchmark_models.ps1 -ModelToTest "Qwen3-0.6B"
    # Skips memory scan, benchmarks only 'Qwen3-0.6B', outputs to benchmark_results_memory.xlsx.
.EXAMPLE
    .\benchmark_models.ps1 -OutputFormat csv
    # Scans memory for all models, benchmarks all models, outputs to benchmark_results_memory.csv.
.EXAMPLE
    .\benchmark_models.ps1 -OutputFile "MyRun" -OutputFormat csv
    # Scans memory, benchmarks all, outputs to MyRun.csv.
.NOTES
    - Requires PowerShell 5.1 or later.
    - Requires 'powershell-yaml' module for config parsing (Install-Module -Name powershell-yaml).
    - Requires 'ImportExcel' module for default XLSX output (Install-Module -Name ImportExcel).
    - Assumes llama-server.exe provides an OpenAI-compatible API at /v1/chat/completions and /health.
    - Needs permissions to execute processes, write output file, create/delete temp files.
    - A console window for llama-server will briefly appear during startup due to output redirection requirements.
    - Memory parsing relies on specific patterns in llama-server's stderr - may need adjustment if output format changes.
#>

#==============================================================================
# Script Parameters
#==============================================================================
param(
    [Parameter(Mandatory=$false)]
    [string]$ConfigPath = "..\model-configs\config-desktop.yaml",

    [Parameter(Mandatory=$false)]
    [string]$OutputFile = "benchmark_results_memory.xlsx", # Base name, extension added later

    [Parameter(Mandatory=$false)]
    [string]$Question = "Why is the sky blue?",

    [Parameter(Mandatory=$false)]
    [int]$HealthCheckPollIntervalSeconds = 2,

    [Parameter(Mandatory=$false)]
    [int]$OverrideHealthCheckTimeoutSeconds,

    [Parameter(Mandatory=$false)]
    [string]$ModelToTest, # If specified, skips Pass 1

    [Parameter(Mandatory=$false)]
    [ValidateSet('xlsx', 'csv')]
    [string]$OutputFormat = 'xlsx'
)

#==============================================================================
# Helper Functions
#==============================================================================

# --- Parse Memory String (Returns GB) ---
function Parse-MemoryString {
    param(
        [string]$MemoryLine
    )
    if ($MemoryLine -match '(\d+(?:[.,]\d+)?)\s*([KMGT]i?B)\b') {
        $Value = [double]($Matches[1].Replace(',', '.'))
        $Unit = $Matches[2].ToUpperInvariant()
        $ValueInMB = 0
        switch ($Unit) {
            'KB'  { $ValueInMB = $Value / 1024 }
            'KIB' { $ValueInMB = $Value / 1024 }
            'MB'  { $ValueInMB = $Value }
            'MIB' { $ValueInMB = $Value }
            'GB'  { $ValueInMB = $Value * 1000 }
            'GIB' { $ValueInMB = $Value * 1024 }
            'TB'  { $ValueInMB = $Value * 1000 * 1000 }
            'TIB' { $ValueInMB = $Value * 1024 * 1024 }
            default { return $null }
        }
        if ($ValueInMB -ne 0) { return [math]::Round($ValueInMB / 1024, 2) } else { return 0 }
    } else { return $null }
}

# --- Parse Llama Server Command String ---
function Parse-LlamaServerCommand {
    param(
        [Parameter(Mandatory=$true)][string]$FullCommand,
        [Parameter(Mandatory=$true)][string]$ModelName # For logging
    )
    $ExePathPattern = '^((?:\\.|[^"\s])+?\.(?:exe|cmd|bat))(?:\s|$)'
    $ServerExecutable = $null
    $ServerArguments = ""
    $Result = @{ Executable = $null; Arguments = "" }

    if ($FullCommand -match $ExePathPattern) {
        $Result.Executable = $Matches[1].Replace('\\', '\') # Capture and fix slashes
        $ArgStartIndex = $Matches[1].Length
        if ($FullCommand.Length -gt $ArgStartIndex -and $FullCommand[$ArgStartIndex] -match '\s') {
            $ArgStartIndex++
        }
        if ($FullCommand.Length -gt $ArgStartIndex) {
             $Result.Arguments = $FullCommand.Substring($ArgStartIndex).TrimStart()
        }
    } else {
        Write-Warning "Parse-LlamaServerCommand: Could not reliably find executable for '$ModelName' using regex. Attempting split by first space."
        $CommandWords = $FullCommand -split '\s+', 2
        $Result.Executable = $CommandWords[0].Replace('\\', '\') # Fix slashes
        if ($CommandWords.Length -gt 1) {
            $Result.Arguments = $CommandWords[1]
        }
    }

    if (-not $Result.Executable) {
         Write-Warning "Parse-LlamaServerCommand: Failed to extract server executable path for '$ModelName' from '$FullCommand'."
         return $null # Indicate failure
    }
    return $Result
}

# --- Start Llama Server Process ---
function Start-LlamaServer {
    param(
        [Parameter(Mandatory=$true)][string]$Executable,
        [Parameter(Mandatory=$true)][string]$Arguments,
        [Parameter(Mandatory=$true)][string]$ModelName # For logging
    )
    $StdOutLogPath = [System.IO.Path]::GetTempFileName()
    $StdErrLogPath = [System.IO.Path]::GetTempFileName()
    $ProcessInfo = $null

    Write-Verbose "Start-LlamaServer: Starting '$ModelName'"
    Write-Verbose "  Executable: '$Executable'"
    Write-Verbose "  Arguments: '$Arguments'"
    Write-Verbose "  Stdout Log: $StdOutLogPath"
    Write-Verbose "  Stderr Log: $StdErrLogPath"

    $ProcessArgs = @{
        FilePath            = $Executable
        ArgumentList        = $Arguments
        PassThru            = $true
        RedirectStandardOutput = $StdOutLogPath
        RedirectStandardError  = $StdErrLogPath
        NoNewWindow         = $true
    }
    try {
        $ProcessInfo = Start-Process @ProcessArgs -ErrorAction Stop
        if (-not $ProcessInfo) { throw "Start-Process returned null." }
        return @{ Process = $ProcessInfo; StdOutLog = $StdOutLogPath; StdErrLog = $StdErrLogPath }
    } catch {
        Write-Error "Start-LlamaServer: Failed to start server process for '$ModelName': $($_.Exception.Message)"
        # Clean up logs if process failed to start
        if ($StdOutLogPath -and (Test-Path $StdOutLogPath)) { Remove-Item $StdOutLogPath -Force -EA SilentlyContinue }
        if ($StdErrLogPath -and (Test-Path $StdErrLogPath)) { Remove-Item $StdErrLogPath -Force -EA SilentlyContinue }
        return $null # Indicate failure
    }
}

# --- Wait for Server Health ---
function Wait-LlamaServerHealthy {
    param(
        [Parameter(Mandatory=$true)]$Process, # Process object from Start-Process
        [Parameter(Mandatory=$true)][string]$HealthCheckUrl,
        [Parameter(Mandatory=$true)][int]$TimeoutSeconds,
        [Parameter(Mandatory=$true)][int]$PollIntervalSeconds,
        [Parameter(Mandatory=$true)][string]$ModelName # For logging
    )
    Write-Host "  Waiting for server '$ModelName' (PID: $($Process.Id)) to become healthy at $HealthCheckUrl ..."
    $HealthCheckStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    while ($HealthCheckStopwatch.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        if ($Process.HasExited) {
            Write-Warning "Wait-LlamaServerHealthy: Server process '$ModelName' (PID: $($Process.Id)) exited prematurely (Exit Code: $($Process.ExitCode))."
            return $false # Indicate failure
        }
        try {
            # Use a short timeout for the web request itself
            $HealthResponse = Invoke-WebRequest -Uri $HealthCheckUrl -Method Get -TimeoutSec ($PollIntervalSeconds * 2) -UseBasicParsing -ErrorAction Stop
            if ($HealthResponse.StatusCode -eq 200) {
                Write-Host "  Server '$ModelName' is healthy." -ForegroundColor Green
                return $true # Indicate success
            }
        } catch {
            Write-Verbose "Wait-LlamaServerHealthy: Health check failed for '$ModelName': $($_.Exception.Message). Retrying..."
        }
        Start-Sleep -Seconds $PollIntervalSeconds
    }
    $HealthCheckStopwatch.Stop()
    Write-Warning "Wait-LlamaServerHealthy: Server '$ModelName' did not become healthy within $TimeoutSeconds seconds."
    return $false # Indicate timeout
}

# --- Stop Llama Server Process ---
function Stop-LlamaServer {
    param(
        [Parameter(Mandatory=$true)]$Process, # Process object
        [Parameter(Mandatory=$true)][string]$ModelName # For logging
    )
    if ($Process -and -not $Process.HasExited) {
        if (Get-Process -Id $Process.Id -ErrorAction SilentlyContinue) {
            Write-Host "  Stopping server process '$ModelName' (PID: $($Process.Id))..."
            Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 2 # Allow time for termination
        } else {
             Write-Verbose "Stop-LlamaServer: Server process '$ModelName' (PID: $($Process.Id)) already stopped (Get-Process failed)."
        }
    } elseif ($Process) {
         Write-Verbose "Stop-LlamaServer: Server process '$ModelName' (PID: $($Process.Id)) already exited."
    }
}

# --- Parse Memory From Log File ---
function Parse-MemoryFromLog {
    param(
        [Parameter(Mandatory=$true)][string]$StdErrLogPath, # Expecting info in stderr
        [Parameter(Mandatory=$true)][string]$ModelName # For logging
    )
    $GpuMemFoundGB = $null
    $CpuMemFoundGB = $null
    $ScanStatus = "Failed"

    if (-not (Test-Path $StdErrLogPath)) {
        Write-Warning "Parse-MemoryFromLog: Stderr log file not found for '$ModelName': '$StdErrLogPath'"
        return @{ GpuGB = $null; CpuGB = $null; Status = "Failed" }
    }

    Write-Verbose "Parse-MemoryFromLog: Reading stderr log for '$ModelName': $StdErrLogPath"
    Start-Sleep -Seconds 1 # Give logs time to flush
    $ServerOutput = Get-Content -Path $StdErrLogPath -Raw -ErrorAction SilentlyContinue

    if (-not $ServerOutput) {
        Write-Warning "Parse-MemoryFromLog: Stderr log file is empty for '$ModelName'."
         return @{ GpuGB = $null; CpuGB = $null; Status = "Failed" }
    }

    # Define patterns within the function
    $SpecificVramPattern = '(?i)\s*load_tensors:\s+CUDA\d+\s+model\s+buffer\s+size\s*=\s*(\d+(?:[.,]\d+)?\s*[KMGT]i?B)'
    $SpecificRamPattern = '(?i)\s*load_tensors:\s+CPU_Mapped\s+model\s+buffer\s+size\s*=\s*(\d+(?:[.,]\d+)?\s*[KMGT]i?B)'

    # Extract VRAM
    $GpuMemMatch = $ServerOutput | Select-String -Pattern $SpecificVramPattern -ErrorAction SilentlyContinue
    if ($GpuMemMatch) {
            $GpuMatchLine = $GpuMemMatch | Select-Object -First 1
            $GpuCapture = $GpuMatchLine.Matches[0].Groups[1].Value
            $GpuMemFoundGB = Parse-MemoryString -MemoryLine $GpuCapture
            if ($GpuMemFoundGB -ne $null) { $ScanStatus = "Success" }
            else { Write-Warning "Parse-MemoryFromLog: Failed to parse GPU value for '$ModelName' from line: $($GpuMatchLine.Line)" }
    } else { Write-Verbose "Parse-MemoryFromLog: Did not find GPU memory pattern for '$ModelName'." }

    # Extract RAM
    $CpuMemMatch = $ServerOutput | Select-String -Pattern $SpecificRamPattern -ErrorAction SilentlyContinue
    if ($CpuMemMatch) {
        $CpuMatchLine = $CpuMemMatch | Select-Object -First 1
            $CpuCapture = $CpuMatchLine.Matches[0].Groups[1].Value
            $CpuMemFoundGB = Parse-MemoryString -MemoryLine $CpuCapture
            if ($CpuMemFoundGB -ne $null) { $ScanStatus = "Success" }
            else { Write-Warning "Parse-MemoryFromLog: Failed to parse CPU value for '$ModelName' from line: $($CpuMatchLine.Line)" }
    } else { Write-Verbose "Parse-MemoryFromLog: Did not find CPU memory pattern for '$ModelName'." }

    # Report combined result
    if ($ScanStatus -eq "Success") {
        $GpuDisplay = if ($GpuMemFoundGB -ne $null) { $GpuMemFoundGB } else { '?' }
        $CpuDisplay = if ($CpuMemFoundGB -ne $null) { $CpuMemFoundGB } else { '?' }
        Write-Host ("  Parsed Memory for '{0}': GPU={1} GB, CPU={2} GB" -f $ModelName, $GpuDisplay, $CpuDisplay) -ForegroundColor Cyan
    } else {
        Write-Warning "Parse-MemoryFromLog: Failed to parse valid memory values for '$ModelName'."
    }

    return @{ GpuGB = $GpuMemFoundGB; CpuGB = $CpuMemFoundGB; Status = $ScanStatus }
}

# --- Clean Up Log Files ---
function CleanUp-LogFiles {
     param(
        [string]$StdOutLog,
        [string]$StdErrLog
     )
     Write-Verbose "Cleaning up log files..."
     if ($StdOutLog -and (Test-Path $StdOutLog)) { Remove-Item -Path $StdOutLog -Force -ErrorAction SilentlyContinue; Write-Verbose "  Removed: $StdOutLog" }
     if ($StdErrLog -and (Test-Path $StdErrLog)) { Remove-Item -Path $StdErrLog -Force -ErrorAction SilentlyContinue; Write-Verbose "  Removed: $StdErrLog" }
}


#==============================================================================
# Main Script Logic
#==============================================================================

# --- Initial Setup ---
$ScriptStartTime = Get-Date
Write-Host "Starting LLM Benchmark Script at $($ScriptStartTime.ToString('yyyy-MM-dd HH:mm:ss'))..."

# --- Calculate Full Paths --- ENSURE THESE ARE PRESENT AND CORRECT ---
$ScriptDir = $PSScriptRoot
$ConfigFullPath = Join-Path -Path $ScriptDir -ChildPath $ConfigPath
# Calculate the base output path using the script dir and the OutputFile parameter
$OutputFullPathBase = Join-Path -Path $ScriptDir -ChildPath $OutputFile
# --- End Path Calculation ---

# Use the variables in the Write-Host messages
Write-Host "Config File: $ConfigFullPath"
Write-Host "Base Output File: $OutputFullPathBase" # Corrected variable used here
Write-Host "Output Format: $($OutputFormat.ToUpper())"
Write-Host "Test Question: '$Question'"
Write-Warning "A console window for llama-server may briefly appear during startup."


# --- Load Config ---
Write-Host "Loading configuration..."
$Config = $null
$Config = $null
try {
    if (-not (Test-Path -Path $ConfigFullPath -PathType Leaf)) { throw "Configuration file not found at '$ConfigFullPath'." }
    # Ensure ConvertFrom-Yaml is available (requires powershell-yaml module in PS 5.1)
    if (-not (Get-Command ConvertFrom-Yaml -ErrorAction SilentlyContinue)) {
         Import-Module powershell-yaml -ErrorAction Stop
         if (-not (Get-Command ConvertFrom-Yaml -ErrorAction SilentlyContinue)) { throw "ConvertFrom-Yaml command not found after importing module."}
    }
    $ConfigContent = Get-Content -Path $ConfigFullPath -Raw -ErrorAction Stop
    $Config = ConvertFrom-Yaml -Yaml $ConfigContent -ErrorAction Stop
    Write-Host "Configuration loaded successfully."
} catch {
    Write-Error "Failed to load or parse configuration file '$ConfigFullPath': $($_.Exception.Message)"
    Exit 1
}

# --- Determine Health Check Timeout ---
$DefaultHealthCheckTimeout = 60
$HealthCheckTimeoutSeconds = $null
if ($PSBoundParameters.ContainsKey('OverrideHealthCheckTimeoutSeconds')) { $HealthCheckTimeoutSeconds = $OverrideHealthCheckTimeoutSeconds }
elseif ($Config -ne $null -and $Config.PSObject.Properties.Name -contains 'healthCheckTimeout' -and $Config.healthCheckTimeout -ne $null) {
    try { $HealthCheckTimeoutSeconds = [int]$Config.healthCheckTimeout } catch { $HealthCheckTimeoutSeconds = $DefaultHealthCheckTimeout }
} else { $HealthCheckTimeoutSeconds = $DefaultHealthCheckTimeout }
if ($HealthCheckTimeoutSeconds -isnot [int] -or $HealthCheckTimeoutSeconds -le 0) { $HealthCheckTimeoutSeconds = $DefaultHealthCheckTimeout }
Write-Host "Health Check Timeout: $HealthCheckTimeoutSeconds seconds"
Write-Host "Health Check Poll Interval: $HealthCheckPollIntervalSeconds seconds"

# --- Validate Config Structure ---
if (-not $Config.models) { Write-Error "No 'models' section found in the configuration file."; Exit 1 }
$AllModelKeys = $Config.models.Keys
if ($AllModelKeys -eq $null -or $AllModelKeys.Count -eq 0) { Write-Error "No models defined under 'models:' section."; Exit 1 }


# --- Initialize Result Storage ---
$MemoryScanResults = @{} # Stores { ModelName -> { GpuGB, CpuGB, Status } }
$FinalResults = @()     # Stores final benchmark result objects


# --- Determine Models to Benchmark ---
$ModelsToTestList = $null
if ($PSBoundParameters.ContainsKey('ModelToTest')) {
    $SingleModelName = $ModelToTest
    if ($AllModelKeys -contains $SingleModelName) {
        $ModelsToTestList = @($SingleModelName)
    } else {
        Write-Error "The specified model '$SingleModelName' was not found in the configuration file."
        Write-Host "Available models: $($AllModelKeys -join ', ')"
        Exit 1
    }
} else {
    $ModelsToTestList = $AllModelKeys # Benchmark all if no specific model requested
}
Write-Host "Models selected for benchmarking: $($ModelsToTestList -join ', ')"

#==============================================================================
# PASS 1: Memory Scan (Skip if -ModelToTest is used)
#==============================================================================

Write-Host ("="*60)
Write-Host "Starting PASS 1: Memory Scan for $($ModelsToTestList.Count) models..."
Write-Host ("="*60)

foreach ($ModelName in $ModelsToTestList) {
    Write-Host ("-"*40)
    Write-Host "Memory Scan: Processing '$ModelName'"

    $ModelConfig = $Config.models.$ModelName
    if ($null -eq $ModelConfig -or -not $ModelConfig.proxy -or -not $ModelConfig.cmd) {
        Write-Warning "Memory Scan: Skipping '$ModelName' - Invalid or incomplete config (missing proxy or cmd)."
        $MemoryScanResults[$ModelName] = @{ GpuGB = $null; CpuGB = $null; Status = "Skipped" }
        continue
    }

    $ParsedCommand = Parse-LlamaServerCommand -FullCommand $ModelConfig.cmd.Trim() -ModelName $ModelName
    if ($null -eq $ParsedCommand -or -not (Test-Path -Path $ParsedCommand.Executable -PathType Leaf)) {
        Write-Warning "Memory Scan: Skipping '$ModelName' - Cannot parse command or executable not found at '$($ParsedCommand.Executable)'."
        $MemoryScanResults[$ModelName] = @{ GpuGB = $null; CpuGB = $null; Status = "Cmd Error" }
        continue
    }

    $ServerInfo = $null
    $MemoryResult = $null
    $CurrentScanStatus = "Failed"

    try {
        $ServerInfo = Start-LlamaServer -Executable $ParsedCommand.Executable -Arguments $ParsedCommand.Arguments -ModelName $ModelName
        if ($null -eq $ServerInfo) { throw "Failed to start server." }

        $ProxyUrl = $ModelConfig.proxy; if ($ProxyUrl -notmatch '^https?://') { $ProxyUrl = "http://$ProxyUrl" }
        $Healthy = Wait-LlamaServerHealthy -Process $ServerInfo.Process -HealthCheckUrl "$ProxyUrl/health" `
                    -TimeoutSeconds $HealthCheckTimeoutSeconds -PollIntervalSeconds $HealthCheckPollIntervalSeconds -ModelName $ModelName

        if ($Healthy) {
            $MemoryResult = Parse-MemoryFromLog -StdErrLogPath $ServerInfo.StdErrLog -ModelName $ModelName
            $CurrentScanStatus = if ($MemoryResult) { $MemoryResult.Status } else { "Parse Error" }
        } else {
            $CurrentScanStatus = "Health Timeout"
                # Try to get stderr content on health timeout/failure
                if ($ServerInfo.Process.HasExited) {
                    $errLogContent = Get-Content $ServerInfo.StdErrLog -Raw -EA SilentlyContinue
                    if ($errLogContent) { Write-Warning "  Server exited. Stderr: $($errLogContent.Substring(0, [System.Math]::Min($errLogContent.Length, 300)))"}
                }
        }
    } catch {
        Write-Warning "Memory Scan: Error during processing '$ModelName': $($_.Exception.Message)"
        $CurrentScanStatus = "Error"
    } finally {
        if ($ServerInfo) {
            Stop-LlamaServer -Process $ServerInfo.Process -ModelName $ModelName
            # Store result (even if failed)
            $MemoryScanResults[$ModelName] = @{
                GpuGB = if ($MemoryResult) { $MemoryResult.GpuGB } else { $null }
                CpuGB = if ($MemoryResult) { $MemoryResult.CpuGB } else { $null }
                Status = $CurrentScanStatus
            }
            CleanUp-LogFiles -StdOutLog $ServerInfo.StdOutLog -StdErrLog $ServerInfo.StdErrLog
        } else {
                # Ensure an entry exists if Start-LlamaServer failed
                if (-not $MemoryScanResults.ContainsKey($ModelName)) {
                    $MemoryScanResults[$ModelName] = @{ GpuGB = $null; CpuGB = $null; Status = "Start Error" }
                }
        }
    }
} # End foreach model in Pass 1

Write-Host ("="*60)
Write-Host "PASS 1: Memory Scan Complete."
Write-Host ("="*60)
Start-Sleep -Seconds 1

#==============================================================================
# PASS 2: Benchmark
#==============================================================================
Write-Host ("="*60)
Write-Host "Starting PASS 2: Benchmarking Selected Models..."
Write-Host ("="*60)

# --- Benchmark Loop ---
foreach ($ModelName in $ModelsToTestList) {
    Write-Host ("-"*40)
    Write-Host "Benchmark: Processing '$ModelName'"

    $ModelConfig = $Config.models.$ModelName
     # Basic validation (already checked keys, but check content again)
    if ($null -eq $ModelConfig -or -not $ModelConfig.proxy -or -not $ModelConfig.cmd) {
        Write-Warning "Benchmark: Skipping '$ModelName' - Invalid or incomplete config."
        # Create a failed result object
         $FinalResults += [PSCustomObject]@{ ModelName=$ModelName; Timestamp=Get-Date; Status="Config Error"; Error="Invalid/incomplete config"; GpuMemoryGB=$null; CpuMemoryGB=$null }
        continue
    }

    # Prepare command again for this pass
    $ParsedCommand = Parse-LlamaServerCommand -FullCommand $ModelConfig.cmd.Trim() -ModelName $ModelName
    if ($null -eq $ParsedCommand -or -not (Test-Path -Path $ParsedCommand.Executable -PathType Leaf)) {
        Write-Warning "Benchmark: Skipping '$ModelName' - Cannot parse command or executable not found at '$($ParsedCommand.Executable)'."
        $FinalResults += [PSCustomObject]@{ ModelName=$ModelName; Timestamp=Get-Date; Status="Cmd Error"; Error="Cannot parse cmd or exe not found"; GpuMemoryGB=$null; CpuMemoryGB=$null }
        continue
    }

    # --- Retrieve Pre-Scanned Memory ---
    $GpuGB_FromScan = $null
    $CpuGB_FromScan = $null
    if ($MemoryScanResults.ContainsKey($ModelName)) {
        $ScanData = $MemoryScanResults[$ModelName]
        if ($ScanData.Status -eq 'Success') {
            $GpuGB_FromScan = $ScanData.GpuGB
            $CpuGB_FromScan = $ScanData.CpuGB
        } else {
             Write-Warning "Benchmark: Memory scan data for '$ModelName' indicates failure (Status: $($ScanData.Status)). Memory fields will be empty."
        }
    } else {
        # This happens if -ModelToTest was used, skipping Pass 1
        Write-Host "Benchmark: No pre-scanned memory data available for '$ModelName' (expected if -ModelToTest was used)."
    }

    # --- Initialize Result Object ---
    $CurrentResult = [PSCustomObject]@{
        ModelName        = $ModelName
        Timestamp        = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
        ProxyUrl         = $ModelConfig.proxy # Use the direct proxy URL
        Status           = "Not Run"
        GpuMemoryGB      = $GpuGB_FromScan
        CpuMemoryGB      = $CpuGB_FromScan
        DurationSeconds  = $null
        PromptTokens     = $null
        CompletionTokens = $null
        TotalTokens      = $null
        TokensPerSecond  = $null
        Error            = $null
    }

    # --- Start Server and Benchmark ---
    $ServerInfo = $null
    try {
        $ServerInfo = Start-LlamaServer -Executable $ParsedCommand.Executable -Arguments $ParsedCommand.Arguments -ModelName $ModelName
        if ($null -eq $ServerInfo) { throw "Failed to start server for benchmark." }

        $ProxyUrl = $ModelConfig.proxy; if ($ProxyUrl -notmatch '^https?://') { $ProxyUrl = "http://$ProxyUrl" }
        $ApiEndpoint = "$ProxyUrl/v1/chat/completions"
        $Healthy = Wait-LlamaServerHealthy -Process $ServerInfo.Process -HealthCheckUrl "$ProxyUrl/health" `
                    -TimeoutSeconds $HealthCheckTimeoutSeconds -PollIntervalSeconds $HealthCheckPollIntervalSeconds -ModelName $ModelName

        if ($Healthy) {
            # --- Perform Benchmark Request ---
            Write-Host "  Server healthy. Sending benchmark request to $ApiEndpoint..."
            $ApiPayload = @{
                model = $ModelName # Include model name in payload
                messages = @( @{ role = "user"; content = $Question } )
                temperature = 0.7 # Example temperature
                stream = $false
            } | ConvertTo-Json -Depth 5

            $Response = $null
            $BenchmarkTime = $null
            try {
                 $BenchmarkTime = Measure-Command {
                    $Response = Invoke-RestMethod -Uri $ApiEndpoint -Method Post -ContentType 'application/json' -Body $ApiPayload -TimeoutSec 600 # 10 min timeout for potentially long responses
                 }
            } catch {
                 throw "API request failed: $($_.Exception.Message) - $($_.ErrorDetails.Message)"
            }

            # --- Process Results ---
            if ($Response -and $Response.usage -and $Response.choices) {
                Write-Host "  Received response from $ModelName."
                $CurrentResult.Status = "Success"
                $CurrentResult.DurationSeconds = [math]::Round($BenchmarkTime.TotalSeconds, 3)
                $CurrentResult.PromptTokens = $Response.usage.prompt_tokens
                $CurrentResult.CompletionTokens = $Response.usage.completion_tokens
                $CurrentResult.TotalTokens = $Response.usage.total_tokens

                if ($CurrentResult.DurationSeconds -gt 0 -and $CurrentResult.CompletionTokens -gt 0) {
                    $CurrentResult.TokensPerSecond = [math]::Round($CurrentResult.CompletionTokens / $CurrentResult.DurationSeconds, 2)
                    Write-Host ("  Result: {0:N3}s, {1} completion tokens, {2:N2} TPS" -f $CurrentResult.DurationSeconds, $CurrentResult.CompletionTokens, $CurrentResult.TokensPerSecond) -ForegroundColor Green
                } else {
                    Write-Warning "  Could not calculate TokensPerSecond (Duration or Completion Tokens were zero)."
                    $CurrentResult.TokensPerSecond = 0
                }
            } else {
                throw "Invalid or incomplete API response received."
            }

        } else {
            $CurrentResult.Status = "Health Timeout"
            $CurrentResult.Error = "Server did not become healthy for benchmark."
             # Try to get stderr content on health timeout/failure
             if ($ServerInfo.Process.HasExited) {
                  $errLogContent = Get-Content $ServerInfo.StdErrLog -Raw -EA SilentlyContinue
                  if ($errLogContent) { $CurrentResult.Error += " | Stderr: $($errLogContent.Substring(0, [System.Math]::Min($errLogContent.Length, 300)))"}
             }
        }

    } catch {
        Write-Warning "Benchmark: Error during processing '$ModelName': $($_.Exception.Message)"
        $CurrentResult.Status = "Error"
        $CurrentResult.Error = $_.Exception.Message
        # Include stderr if server exited during benchmark attempt
        if ($ServerInfo -and $ServerInfo.Process -and $ServerInfo.Process.HasExited) {
            $errLogContent = Get-Content $ServerInfo.StdErrLog -Raw -ErrorAction SilentlyContinue
            if ($errLogContent) { $CurrentResult.Error += " | Stderr: $($errLogContent.Substring(0, [System.Math]::Min($errLogContent.Length, 500)))" }
        }
    } finally {
         if ($ServerInfo) {
            Stop-LlamaServer -Process $ServerInfo.Process -ModelName $ModelName
            CleanUp-LogFiles -StdOutLog $ServerInfo.StdOutLog -StdErrLog $ServerInfo.StdErrLog
         }
         # Add the result (success or failure object) to the final list
         $FinalResults += $CurrentResult
    }

} # End foreach model for Benchmark

Write-Host ("="*60)
Write-Host "PASS 2: Benchmark Complete."
Write-Host ("="*60)


#==============================================================================
# Save Final Results
#==============================================================================
Write-Host ("-"*40)
if ($FinalResults.Count -gt 0) {
    # Construct final output path
    $FinalOutputFullPath = $null
    try {
        # Use $OutputFullPathBase here (which should now be correctly populated)
        $Directory = Split-Path -Path $OutputFullPathBase -Parent -ErrorAction Stop
        $BaseName = [System.IO.Path]::GetFileNameWithoutExtension($OutputFullPathBase)
        # Use $OutputFormat parameter to set extension
        $FileNameWithCorrectExt = [System.IO.Path]::ChangeExtension($BaseName, $OutputFormat)
        $FinalOutputFullPath = Join-Path -Path $Directory -ChildPath $FileNameWithCorrectExt -ErrorAction Stop
    } catch {
        # The Warning message now uses the correct variable name in its text
        Write-Warning "Could not reliably construct final output path using base '$OutputFullPathBase' and format '$OutputFormat'. Using original OutputFile parameter value."
        # Fallback uses the variable which *should* have the initial value
        $FinalOutputFullPath = $OutputFullPathBase
    }

    if (-not $FinalOutputFullPath) {
         Write-Error "Failed to determine a final output path. Cannot save results."
    } else {
        Write-Host "Saving final benchmark results to '$FinalOutputFullPath' (Format: $($OutputFormat.ToUpper()))..."
        try {
            # Select properties for output
            $OutputData = $FinalResults | Select-Object ModelName, Timestamp, Status, GpuMemoryGB, CpuMemoryGB, DurationSeconds, TokensPerSecond, PromptTokens, CompletionTokens, TotalTokens, ProxyUrl, Error

            if ($OutputFormat -eq 'xlsx') {
                $OutputData | Export-Excel -Path $FinalOutputFullPath -AutoSize -TableName "BenchmarkResults" -TableStyle Medium6 -FreezeTopRow -WorksheetName "Benchmarks" -ErrorAction Stop
            } elseif ($OutputFormat -eq 'csv') {
                $OutputData | Export-Csv -Path $FinalOutputFullPath -NoTypeInformation -Encoding UTF8 -ErrorAction Stop
            }
            Write-Host "Results saved successfully." -ForegroundColor Green
        } catch {
            Write-Error "Failed to save results to '$FinalOutputFullPath': $($_.Exception.Message)"
            if ($OutputFormat -eq 'xlsx' -and ($_.Exception.ToString() -match 'Export-Excel' -or $_.Exception.InnerException.ToString() -match 'EPPlus')) {
                Write-Warning "XLSX export failed. Ensure the 'ImportExcel' module is correctly installed."
            }
        }
    }
} else {
    Write-Warning "No final benchmark results were generated to save."
}

$ScriptEndTime = Get-Date
$ScriptDuration = New-TimeSpan -Start $ScriptStartTime -End $ScriptEndTime
Write-Host "Benchmark script finished at $($ScriptEndTime.ToString('yyyy-MM-dd HH:mm:ss')). Total duration: $($ScriptDuration.ToString('g'))"