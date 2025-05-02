<#
    Helper: watch one config file; restart llama-swap with the same
    arguments whenever the file is modified.
#>

param(
    [Parameter(Mandatory)][string]   $ExePath,   # full path to llama-swap.exe
    [Parameter(Mandatory)][string[]] $CmdArgs,   # *exact* Start-Process arg list
    [Parameter(Mandatory)][string]   $ConfigPath # file to watch
)

function Start-LlamaSwap {
    param([string]$Exe, [string[]]$CmdArgs)

    Write-Host "`n$Exe $($CmdArgs -join ' ')" -Foreground Cyan
    Start-Process -FilePath $Exe `
                  -ArgumentList $CmdArgs `
                  -PassThru -NoNewWindow
}

function Stop-LlamaSwap ([System.Diagnostics.Process]$p) {
    if ($p -and -not $p.HasExited) {
        Write-Host "stopping PID $($p.Id)" -Foreground DarkCyan
        try { $null = $p.CloseMainWindow() } catch {}
        Start-Sleep 2
        if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
        $p.WaitForExit()
    }
}

while ($true) {
    $proc = Start-LlamaSwap -Exe $ExePath -CmdArgs $CmdArgs

    # ---- watch config.yaml (one-shot) --------------------------------------
    $watcher                = New-Object IO.FileSystemWatcher
    $watcher.Path           = (Split-Path $ConfigPath)
    $watcher.Filter         = (Split-Path $ConfigPath -Leaf)
    $watcher.NotifyFilter   = [IO.NotifyFilters]'LastWrite'
    $watcher.EnableRaisingEvents = $true

    $srcId = "ConfigChanged_" + ([guid]::NewGuid().Guid)
    Register-ObjectEvent $watcher Changed -SourceIdentifier $srcId | Out-Null
    Wait-Event -SourceIdentifier $srcId | Out-Null

    Unregister-Event -SourceIdentifier $srcId
    Remove-Event     -SourceIdentifier $srcId
    $watcher.Dispose()

    Stop-LlamaSwap $proc
    Start-Sleep 1
}
