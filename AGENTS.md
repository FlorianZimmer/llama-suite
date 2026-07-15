# llama-suite contributor guidance

`llama-suite` is an ops-first local LLM control plane. Its primary outcome is a
reproducible effective configuration and safe runtime workflow across machines,
not a general Python SDK.

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

## Engineering judgment

- Make the smallest maintainable change that delivers the requested config,
  runtime, evaluation, Web UI, or packaging outcome using existing boundaries.
- Preserve config merge precedence, explicit machine overrides, process
  lifecycle safety, authentication, local filesystem isolation, and clear
  failure messages. Add abstraction or infrastructure only for demonstrated
  reuse, risk, or an accepted deployment requirement.
- Do not download models, launch model servers, run benchmarks/evaluations,
  start containers, or deploy to a cluster merely to validate local code.
  External or resource-intensive actions require explicit user intent.
- Ask only when a machine-wide effect, external action, or material product
  choice is required. Otherwise implement, run focused checks, and stop when
  the requested outcome is complete.

## Repo Hygiene
- Do not commit local runtime data or generated artifacts.
- Keep secrets out of YAML and git; use environment variables or CLI flags.
- Treat `configs/config.base.yaml` as the shared baseline and keep machine-specific changes in `configs/overrides/`.

## Proportionate validation

- Run the nearest relevant pytest tests and Ruff for changed Python behavior;
  use mypy when a shared typed interface changes.
- Run the full pytest suite for shared config, runtime registry/process, Web UI
  infrastructure, or release-facing changes—not for isolated docs or metadata.
- Render and exercise the affected Web UI workflow for user-facing UI changes.
  Check only the machines/deployment targets implicated by the change unless a
  shared contract or release requires broader coverage.
- Non-executable metadata needs focused diff/link checks only. Report validation
  that needs unavailable hardware or models instead of substituting unrelated CI.
