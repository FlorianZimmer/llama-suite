# Learnings (distilled / evergreen)

## 2026-02-02 — Web UI button “does nothing” can be a JS crash from missing DOM ids

- Symptom: Clicking “⬆ Run Update” reloads the page (form submit) but no `/api/system/update` request is sent; console shows `Cannot read properties of null (reading 'addEventListener')`.
- Root cause: `src/llama_suite/webui/static/js/app.js` bound event listeners to non-existent ids (e.g. `btn-bench-stop`, `btn-memory-stop`, `btn-eval-*-stop`), throwing early and preventing the rest of the bindings (including `update-form`) from being registered.
- Fix: Make the Event Bindings section null-safe (guard `getElementById(...)` before `.addEventListener`), and avoid binding to ids that don’t exist in `src/llama_suite/webui/static/index.html`.

## 2026-02-02 — Repo “.sh” rule scripts should be run via WSL bash on Windows

- Symptom: Running `cd /d f:\LLMs\llama-suite && ~/.codex/.../activator.sh` from PowerShell fails (`cd /d` is not valid in PowerShell, and `~/.codex` refers to WSL home).
- Fix: Invoke the scripts with `bash -lc` (WSL) and use `/mnt/c/...` paths, e.g. `bash -lc "/mnt/c/Users/Florian/.codex/skills/self-improving-agent-1.0.1/scripts/activator.sh"`.
