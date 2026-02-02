# Errors (raw logs / fixes)

## 2026-02-02 — PowerShell `-Command` string interpolation broke scripts

- Symptom: Running `powershell -Command "..."` from an outer PowerShell context caused `$var` / `$_` to be expanded before the inner PowerShell executed, leading to parse errors like `.StatusCode` / `.Exception.Message` / `EmptyPipeElement`.
- Fix: Wrap the inner `-Command` argument in single quotes (`'...'`) (or escape `$`) so variables are interpreted by the intended PowerShell instance.

## 2026-02-02 — self-improving-agent scripts path mismatch (Windows vs WSL)

- Symptom: `bash ~/.codex/.../activator.sh` failed because `~` resolved inside WSL and the path didn’t exist.
- Fix: Use the WSL-mounted Windows path: `/mnt/c/Users/Florian/.codex/skills/self-improving-agent-1.0.1/scripts/{activator,error-detector}.sh`.

