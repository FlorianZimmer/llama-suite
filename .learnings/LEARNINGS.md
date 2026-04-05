# Learnings (distilled / evergreen)

## 2026-03-05 - Current llama.cpp source builds need GGML backend flags; Windows builds may need OpenSSL disabled

- Symptom: The repo helpers still used `LLAMA_CUBLAS` / `LLAMA_VULKAN`, which are deprecated in current upstream `llama.cpp` and block current source builds. A direct Windows CUDA configure also failed to complete cleanly until `-DLLAMA_OPENSSL=OFF` was added on a machine without OpenSSL dev libraries.
- Fix: Use `-DGGML_CUDA=ON` / `-DGGML_VULKAN=ON` (and the `OFF` variants for CPU-only builds) in `tools/scripts/install.py` and `tools/scripts/update.py`. For local Windows source builds without OpenSSL dev packages, add `-DLLAMA_OPENSSL=OFF` during CMake configure.

## 2026-02-02 — Web UI button “does nothing” can be a JS crash from missing DOM ids

- Symptom: Clicking “⬆ Run Update” reloads the page (form submit) but no `/api/system/update` request is sent; console shows `Cannot read properties of null (reading 'addEventListener')`.
- Root cause: `src/llama_suite/webui/static/js/app.js` bound event listeners to non-existent ids (e.g. `btn-bench-stop`, `btn-memory-stop`, `btn-eval-*-stop`), throwing early and preventing the rest of the bindings (including `update-form`) from being registered.
- Fix: Make the Event Bindings section null-safe (guard `getElementById(...)` before `.addEventListener`), and avoid binding to ids that don’t exist in `src/llama_suite/webui/static/index.html`.

## 2026-02-02 — Repo “.sh” rule scripts should be run via WSL bash on Windows

- Symptom: Running `cd /d f:\LLMs\llama-suite && ~/.codex/.../activator.sh` from PowerShell fails (`cd /d` is not valid in PowerShell, and `~/.codex` refers to WSL home).
- Fix: Invoke the scripts with `bash -lc` (WSL) and use `/mnt/c/...` paths, e.g. `bash -lc "/mnt/c/Users/Florian/.codex/skills/self-improving-agent-1.0.1/scripts/activator.sh"`.

## 2026-02-02 â€” Web UI can look â€œstuckâ€ due to cached `index.html` even when JS/CSS are updated

- Symptom: UI shows an older navigation label/content even though `src/llama_suite/webui/static/index.html` was changed; a hard refresh fixes it.
- Root cause: The dev server disables caching for `/static/*` but not for `GET /` / SPA fallback responses serving `index.html`.
- Fix: Serve `index.html` with `Cache-Control: no-store` (and optionally other non-API file responses) in `src/llama_suite/webui/server.py`.

## 2026-04-05 - `update.py` should reuse the OrbStack Open WebUI named volume when available

- Symptom: Running `tools/scripts/update.py` recreated the `open-webui` container with a bind mount at `var/open-webui/data`, even when OrbStack already had the desired named volume `open-webui_open-webui`.
- Root cause: The updater only used a named volume when `--webui-data-volume` was passed explicitly; otherwise it always fell back to the repo bind mount before recreating the container.
- Fix: In `tools/scripts/update.py`, resolve the data mount before recreation by preferring an explicit `--webui-data-volume`, then an existing container volume, then the existing OrbStack volume `open-webui_open-webui`, and only fall back to the bind mount if no named volume is available.

## [LRN-20260405-001] best_practice

**Logged**: 2026-04-05T21:38:43Z
**Priority**: high
**Status**: pending
**Area**: infra

### Summary
`tools/scripts/update.py` can report an Open WebUI refresh while still reusing a stale cached image.

### Details
The updater does not manage Compose or Helm image references. It only refreshes one local runtime-managed container by pulling `ghcr.io/open-webui/open-webui:main`, stopping/removing `open-webui`, and recreating it via `python -m llama_suite.utils.openwebui`.

The critical gap is that the image pull is best-effort: `refresh_openwebui()` calls `run([rt_path, "pull", image], check=False)`. If the pull fails due to network, registry, or daemon issues, the script continues and recreates the container from whatever cached image is already present locally. That makes the update look successful while the actual image may be unchanged.

### Suggested Action
Make the Open WebUI pull step fail-fast, and if Compose-managed Open WebUI is a supported workflow, add a dedicated compose-aware update path instead of assuming a standalone container.

### Metadata
- Source: conversation
- Related Files: tools/scripts/update.py, deploy/compose/docker-compose.yml, src/llama_suite/webui/api/system.py
- Tags: openwebui, updater, docker, compose

---
