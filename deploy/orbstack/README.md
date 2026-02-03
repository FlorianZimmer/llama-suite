# OrbStack “single-click” local deployment (macOS)

This starts the **llama-suite Web UI** in a container using OrbStack’s Docker-compatible engine.
It does **not** run `llama-swap` / `llama.cpp` inside the container; you can run those on your Mac or on another machine (GPU desktop).

## Prereqs
- OrbStack installed and running
- `docker` CLI available (OrbStack provides this)

## One-click start (Finder)
1. Open this repo in Finder.
2. Double-click `deploy/orbstack/start.command`.
3. Visit `http://localhost:8088`

## Configure runtime URLs (optional)
Copy `deploy/compose/.env.example` to `deploy/compose/.env` and edit as needed:
- `LLAMA_SUITE_SWAP_API_URL=http://localhost:8080/v1`
- `LLAMA_SUITE_SWAP_UI_URL=http://localhost:8080/ui`
- `LLAMA_SUITE_OPEN_WEBUI_URL=http://localhost:3000`

## Stop
- Double-click `deploy/orbstack/stop.command`, or run `docker compose -f deploy/compose/docker-compose.yml down` from the repo root
