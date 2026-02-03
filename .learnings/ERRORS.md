# Errors (raw logs / fixes)

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
