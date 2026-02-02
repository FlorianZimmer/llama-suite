# Learnings (distilled / evergreen)

## 2026-02-02 — Web UI button “does nothing” can be a JS crash from missing DOM ids

- Symptom: Clicking “⬆ Run Update” reloads the page (form submit) but no `/api/system/update` request is sent; console shows `Cannot read properties of null (reading 'addEventListener')`.
- Root cause: `src/llama_suite/webui/static/js/app.js` bound event listeners to non-existent ids (e.g. `btn-bench-stop`, `btn-memory-stop`, `btn-eval-*-stop`), throwing early and preventing the rest of the bindings (including `update-form`) from being registered.
- Fix: Make the Event Bindings section null-safe (guard `getElementById(...)` before `.addEventListener`), and avoid binding to ids that don’t exist in `src/llama_suite/webui/static/index.html`.

