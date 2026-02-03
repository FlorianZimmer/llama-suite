# Repository Guidelines

## Project Structure & Module Organization
- `src/llama_suite/`: core Python code (namespace package).
  - `watchers/`: utilities to run/restart `llama-swap` (e.g. `llama_swap_watch.py`).
  - `eval/`, `bench/`: evaluation + benchmarking scripts.
  - `webui/`: FastAPI Web UI + static assets (`webui/static/`).
  - `utils/`: config helpers, graphs, Open WebUI container helper.
- `configs/`: YAML config source of truth (`config.base.yaml`, `overrides/*.yaml`); generated configs live in `configs/generated/` (ignored).
- `datasets/`, `runs/`, `var/`: local data, results, and runtime state (ignored).
- `vendor/`: downloaded/built external binaries (llama.cpp, llama-swap) (ignored).
- `models/`: local GGUFs (often a symlink; ignored).

## Build, Test, and Development Commands
Bootstrap with any Python 3.10+ to create the repo venv, then prefer the venv Python (`.\.venv\Scripts\python.exe` on Windows, `./.venv/bin/python` on macOS/Linux).

- Install/setup (creates venv, installs editable, fetches/builds vendor deps):
  - `python tools\scripts\install.py --dev-extras` (first run; creates `.venv/`)
  - Tip: add `--no-webui` to skip Open WebUI container setup.
- Update external deps (venv, vendor tools, optional Open WebUI container):
  - `.\.venv\Scripts\python.exe tools\scripts\update.py --dev-extras`
- Uninstall deps/build artifacts but keep data (`runs/`, `var/`, `datasets/`):
  - `.\.venv\Scripts\python.exe tools\scripts\uninstall.py -y`
- Run Web UI:
  - `.\.venv\Scripts\python.exe -m llama_suite.webui.server` (serves on `http://localhost:8088`)
- Run watcher (example override):
  - `.\.venv\Scripts\python.exe -m llama_suite.watchers.llama_swap_watch -o configs\overrides\win-3080-10G.yaml`

## Coding Style & Naming Conventions
- Python: PEP 8, `snake_case` for functions/vars, `PascalCase` for classes.
- Prefer type hints; keep scripts cross-platform (avoid hard-coded paths).
- Lint/format/typecheck (install with `--dev-extras`):
  - `.\.venv\Scripts\python.exe -m ruff check src tools`
  - `.\.venv\Scripts\python.exe -m mypy src\llama_suite`

## Testing Guidelines
There is no dedicated test suite directory today, but `pytest` is available via `[dev]`.
- Add new tests under `tests/` as `test_*.py`.
- Run: `.\.venv\Scripts\python.exe -m pytest -q`

## Commit & Pull Request Guidelines
- Commit subjects in history are short and action-focused (often lowercase/past tense); follow that style (no trailing period).
- Do not commit large local artifacts: `models/`, `vendor/`, `runs/`, `var/`, `configs/generated/` are intentionally ignored.
- PRs: include a clear description, repro/validation steps, and screenshots for Web UI changes; link related issues/configs.

## Security & Configuration Tips
- Keep API keys and tokens out of YAML and git; pass them via env vars or `--api-key` flags.
- Treat `configs/config.base.yaml` as the baseline; put machine-specific changes in `configs/overrides/`.

## MCP

Always use Context7 MCP when I need library/API documentation, code generation, setup or configuration steps without me having to explicitly ask.
Call Chrome DevTools MCP whenever you need a real browser to inspect/interact with live webpages (DOM/CSS/JS), debug UI behavior, or capture network/console output to verify what the page actually does.

## Learning systems (avoid duplication)

DeployKube uses two complementary systems to reduce repeat work and prevent repeated mistakes: a **feedback loop** (self-improving-agent) and a **semantic long-term memory** (Forgetful MCP).

### Self-improving-agent = capture → distill → promote
- Use this when something goes wrong (failed command, unexpected tool behavior) or when the user corrects you.
- Store raw events and fixes in `.learnings/` (e.g. `ERRORS.md`, `LEARNINGS.md`) with enough context to reproduce.
- Periodically distill: “what should we do differently next time?”

### Forgetful (MCP) = durable, searchable memory
- Use this for DeployKube-specific facts/decisions/gotchas you want to recall later via semantic search.
- Write **atomic** notes (one idea per memory) and tag them (e.g. `deploykube`, plus `gitops` / `argocd` / `forgejo` / `kyverno` / `istio`).
- Prefer linking related memories (and optionally a Forgetful project/entity) over duplicating text.

### Where does the knowledge go?
- **“How the agent/contributor should behave every time”** → `agents.md`
- **“A fact/decision/gotcha you want to recall when relevant”** → Forgetful MCP memory
- **“Raw failure logs / scratch learnings / command output”** → `.learnings/`

### When to invoke which (quick triggers)
- Shell command fails → run the error detector (see Repo Rules) and log to `.learnings/ERRORS.md`
- User corrects you / you discover you were wrong/outdated → log to `.learnings/LEARNINGS.md` (category: `correction` / `knowledge_gap`)
- You make a durable decision or uncover a repo-specific gotcha → save an atomic Forgetful memory (tag + link)
- You discover an evergreen “do this every time” rule → update `agents.md` (keep it short)

Example:
- `kubectl apply` fails due to admission policy → log the failure + fix to `.learnings/ERRORS.md`
- The repeatable rule (“tenant namespaces must follow label contract A1”) → save as a Forgetful memory (tags: `kyverno`, `admission`, `rbac-profile`)
- The evergreen behavior (“check the policy docs before creating new namespaces”) → add a short rule to `agents.md`

## Repo Rules (ignore on windows)

- At the start of every session, run: ~/.codex/skills/self-improving-agent-1.0.1/scripts/activator.sh
- After any failed shell command, run: ~/.codex/skills/self-improving-agent-1.0.1/scripts/error-detector.sh
