# llama-suite

`llama-suite` is a config-driven local LLM operations toolkit built around `llama.cpp` and `llama-swap`.
It combines runtime/watch automation, a small FastAPI Web UI, and eval/benchmark helpers in one repo.

This is an active personal infrastructure repo, not a general-purpose Python library or a polished hosted product. The value is in keeping local model configs, runtime control, evaluation, and deployment packaging in one coherent place.

## What it does

- Launches and restarts `llama-swap` from machine-specific config overrides.
- Serves a local Web UI for config inspection, model/runtime tasks, and run results.
- Runs evaluation and benchmarking helpers against local model setups.
- Packages the Web UI for local containers, OrbStack, Compose, and Helm-based deployment.

## Repo map

- `src/llama_suite/watchers/`: launch and restart `llama-swap`
- `src/llama_suite/webui/`: FastAPI Web UI, API routes, and static assets
- `src/llama_suite/eval/`: eval harness integrations
- `src/llama_suite/bench/`: benchmarking and memory-scan utilities
- `configs/`: shared baseline config plus machine-specific overrides
- `deploy/`: container, Compose, OrbStack, Helm, and marketplace packaging

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

Install or refresh the repo environment:

```bash
python tools/scripts/install.py --dev-extras
./.venv/bin/python tools/scripts/update.py --dev-extras
```

Run the watcher with a machine override:

```bash
./.venv/bin/python -m llama_suite.watchers.llama_swap_watch -o configs/overrides/mac-m3-max-36G.yaml
```

Run tests:

```bash
./.venv/bin/python -m pytest -q
```

## Deployment notes

- `deploy/orbstack/README.md`: easiest local container run on macOS.
- `deploy/compose/docker-compose.yml`: local container deployment.
- `deploy/charts/llama-suite-webui/README.md`: Helm chart for the Web UI.

## Notes

- `models/`, `runs/`, `var/`, and generated configs are intentionally local and ignored.
- The repo vendors only the pieces needed to support local runtime workflows.
- The Web UI package includes its static assets and schema when built from this repo.
