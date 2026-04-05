# llama-suite

Local LLM tooling for `llama.cpp`: model config management, swap/watch automation, a FastAPI Web UI, and eval/benchmark helpers.

## What is here

- `src/llama_suite/watchers/`: launch and restart `llama-swap`
- `src/llama_suite/webui/`: FastAPI Web UI and static assets
- `src/llama_suite/eval/`: eval harness integrations
- `src/llama_suite/bench/`: benchmarking and memory-scan utilities
- `configs/`: base config plus machine-specific overrides
- `deploy/`: container, Compose, Helm, and marketplace packaging

## Quick start

Use Python 3.10+.

macOS/Linux:

```bash
python tools/scripts/install.py --dev-extras
./.venv/bin/python -m llama_suite.webui.server
```

Windows PowerShell:

```powershell
python tools\scripts\install.py --dev-extras
.\.venv\Scripts\python.exe -m llama_suite.webui.server
```

The Web UI serves on `http://localhost:8088`.

## Common commands

Install/update vendor deps and the repo venv:

```bash
python tools/scripts/install.py --dev-extras
./.venv/bin/python tools/scripts/update.py --dev-extras
```

Run the watcher with an override:

```bash
./.venv/bin/python -m llama_suite.watchers.llama_swap_watch -o configs/overrides/mac-m3-max-36G.yaml
```

Run tests:

```bash
./.venv/bin/python -m pytest -q
```

## Notes

- `models/`, `vendor/`, `runs/`, `var/`, and generated configs are intentionally local and ignored.
- Containerized deployments are under `deploy/`.
- The Web UI package includes its static assets and schema when built from this repo.
