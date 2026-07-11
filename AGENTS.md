# Repository Guidelines

## Project Structure
- `src/llama_suite/`: core Python code.
- `src/llama_suite/watchers/`: `llama-swap` launch/restart helpers.
- `src/llama_suite/eval/`, `src/llama_suite/bench/`: evaluation and benchmarking tools.
- `src/llama_suite/webui/`: FastAPI Web UI, schema, and static assets.
- `configs/`: base config plus machine-specific overrides.
- `deploy/`: container, Compose, Helm, and marketplace packaging.

## Build And Test
- Install: `python tools/scripts/install.py --dev-extras`
- Update deps/tools: `./.venv/bin/python tools/scripts/update.py --dev-extras`
- Run Web UI: `./.venv/bin/python -m llama_suite.webui.server`
- Run tests: `./.venv/bin/python -m pytest -q`
- Lint: `./.venv/bin/python -m ruff check src tools`
- Type check: `./.venv/bin/python -m mypy src/llama_suite`

## Coding Conventions
- Follow PEP 8.
- Prefer type hints.
- Keep scripts cross-platform.
- Use `snake_case` for functions/variables and `PascalCase` for classes.

## Repo Hygiene
- Do not commit local runtime data or generated artifacts.
- Keep secrets out of YAML and git; use environment variables or CLI flags.
- Treat `configs/config.base.yaml` as the shared baseline and keep machine-specific changes in `configs/overrides/`.

## Repo Rules (ignore on Windows)
- At the start of every session, run: `~/.codex/skills/self-improving-agent-1.0.5/scripts/activator.sh`
- After any failed shell command, run: `~/.codex/skills/self-improving-agent-1.0.5/scripts/error-detector.sh`

## Task Sizing and Scope Discipline

- When the user identifies the files, desired change, and acceptance criteria, make the direct edit and run focused verification.
- Do not create design documents, implementation plans, tracking artifacts, or commits beyond the requested changes unless the user asks or a genuine unresolved design, cross-cutting dependency, or material scope decision requires it.
- Use brainstorming or detailed planning only for new behavior, architecture decisions, or genuinely ambiguous requirements.
- Skills guide execution; they do not authorize expanding the accepted scope.
