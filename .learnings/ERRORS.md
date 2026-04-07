# Errors (raw logs / fixes)

## 2026-03-05 - `cmd /c` quoting for `VsDevCmd.bat` from PowerShell

- Symptom: Launching the VS 2022 developer environment via a heavily escaped `cmd /c "\"C:\...\VsDevCmd.bat\" ... && ..."` string failed immediately with `Der Befehl "\" ist entweder falsch geschrieben oder konnte nicht gefunden werden.`
- Context: This happened while trying to build `llama.cpp` from PowerShell with a chained `VsDevCmd.bat` + Python updater command.
- Fix: From PowerShell, call `cmd.exe /c` with the batch path directly quoted once inside the command string, e.g. `cmd.exe /c "\"C:\...\VsDevCmd.bat\" -arch=x64 -host_arch=x64 && set CMAKE_GENERATOR=Ninja && .\\.venv\\Scripts\\python.exe ..."` or build the command in a PowerShell variable first to avoid over-escaping.

## 2026-02-03 - PowerShell vs cmd/unix syntax pitfalls in commands

- Symptom: Commands like `cd /d f:\LLMs\llama-suite && ...` failed (`Set-Location: A positional parameter cannot be found that accepts argument ...`) because `/d` and `&&` are `cmd.exe` syntax.
- Symptom: Filters like `Where-Object { .Name -match ... }` failed because property access needs `$_` (e.g. `$_.Name`) inside script blocks.
- Fix: In PowerShell, use `cd f:\LLMs\llama-suite; ...` and `Where-Object { $_.Name -match ... }`. For `head`, use `Select-Object -First N` or `Get-Content -TotalCount N`.

## 2026-02-02 — PowerShell `-Command` string interpolation broke scripts

- Symptom: Running `powershell -Command "..."` from an outer PowerShell context caused `$var` / `$_` to be expanded before the inner PowerShell executed, leading to parse errors like `.StatusCode` / `.Exception.Message` / `EmptyPipeElement`.
- Fix: Wrap the inner `-Command` argument in single quotes (`'...'`) (or escape `$`) so variables are interpreted by the intended PowerShell instance.

## 2026-02-02 — self-improving-agent scripts path mismatch (Windows vs WSL)

- Symptom: `bash ~/.codex/.../activator.sh` failed because `~` resolved inside WSL and the path didn’t exist.
- Fix: Use the WSL-mounted Windows path: `/mnt/c/Users/Florian/.codex/skills/self-improving-agent-1.0.1/scripts/{activator,error-detector}.sh`.

## 2026-04-06 - `rg.exe` from Codex WindowsApps package failed to start

- Symptom: `rg -n ...` failed immediately with `Program 'rg.exe' failed to run ... Zugriff verweigert` when launched from PowerShell in this Codex desktop environment.
- Context: This happened while inspecting `llama-suite` config and runtime state on Windows.
- Fix: Fall back to PowerShell-native search (`Get-ChildItem ... | Select-String ...`) instead of assuming the bundled `rg.exe` is executable in this environment.
