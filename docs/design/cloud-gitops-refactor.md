# Cloud-ready + GitOps-first refactor plan for `llama-suite`

Last updated: 2026-02-03

## Summary
This document tracks the work to refactor `llama-suite` so it can be deployed:
- Locally with a container runtime (Docker/Podman), including “single-click” start on macOS (OrbStack)
- On Kubernetes in a cloud-agnostic way
- In a GitOps-first model (read-only app in-cluster; config changes via Git)
- With independently deployable components (Web UI separate from the LLM runtime)

Key decision: the Web UI will be **endpoint-only** in Kubernetes (it does not start/stop `llama-swap`/`llama.cpp`).

## Goals
- **Cloud-agnostic**: no dependency on cloud-specific managed services by default.
- **GitOps-first**: in-cluster deployment is declarative; no mutable config writes from the UI.
- **Composable**: deploy each component separately (e.g., GPU runtime on desktop; UI in cluster).
- **Secure-by-default in cluster**: Ingress OIDC + optional app API key (defense-in-depth).
- **Easy local deploy**: OrbStack “double click” start; Docker/Podman compose support.

## Non-goals (initial phase)
- Web UI directly manipulating Kubernetes resources (avoid breaking GitOps purity).
- Remote-start/stop of runtimes from the UI (no agent in the first iteration).
- Full multi-tenant SaaS features.

## Current repo observations (as of 2026-02-03)
- The Web UI currently:
  - Serves a FastAPI SPA (`src/llama_suite/webui/server.py`)
  - Exposes APIs that read/write configs and models (`src/llama_suite/webui/api/config.py`, `src/llama_suite/webui/api/models.py`)
  - Can start long-running subprocess tasks (watcher/bench/eval/download/update/install) via `ProcessManager`
- There were no container/K8s deployment artifacts originally.

## Decisions (locked)
- **Runtime management model**: Endpoint-only (UI points at an existing OpenAI-compatible endpoint).
- **Kubernetes packaging**: Helm chart (GitOps-friendly with Argo CD).
- **Artifact storage (K8s)**: PVC only (default; can evolve to S3 later).
- **Config mutation in cluster**: Read-only (UI does not write `configs/`/`models/` in GitOps mode).
- **Auth**: Both (Ingress OIDC + optional app-level API key).
- **Image registry**: Both (publish to GHCR; mirror/replicate to DeployKube Harbor).

## Target architecture

### Components
1. **`llama-suite-webui`** (Kubernetes or local container)
   - FastAPI + static assets
   - In Kubernetes: **read-only**, **no subprocess spawning**
   - Reads config/results (if mounted) and renders UI
   - Points at external runtime endpoints via env-configured URLs

2. **LLM runtime** (outside the Web UI)
   - `llama-swap` + `llama.cpp` (desktop GPU host or K8s GPU nodes later)
   - Treated as an external OpenAI-compatible endpoint by the UI

### Data roots and mount layout
To run in containers/K8s, the app must not assume “repo root” paths.

Canonical container layout:
- `/data/configs` (ConfigMap or bind mount; read-only in GitOps mode)
- `/data/runs` (PVC or bind mount)
- `/data/var` (optional; logs/cache)

Environment variables:
- `LLAMA_SUITE_ROOT=/data`
- `LLAMA_SUITE_MODE=local|gitops` (default: `local` until enforced everywhere)

## Configuration and “modes”

### `LLAMA_SUITE_MODE=local`
- Allowed:
  - Editing config files via API
  - Upload/delete model files (if desired)
  - Starting subprocess tasks (watcher/bench/eval/download/update/install)
- Intended for local workstation use.

### `LLAMA_SUITE_MODE=gitops`
- Disallowed:
  - Any endpoint that writes under `configs/` or `models/`
  - Any endpoint that spawns subprocesses (watcher, update/install, etc.)
- Allowed:
  - Read-only browsing of effective config / results (where mounted)
  - Health endpoints

Implementation plan:
- Add centralized “capability gates” (FastAPI dependencies) and apply them to routes:
  - `require_local_mode()` for subprocess endpoints
  - `require_not_read_only()` for config/model mutation endpoints

## External endpoints (UI links)
The UI must not hardcode `localhost` URLs.

Proposed env vars:
- `LLAMA_SUITE_SWAP_API_URL` (default `http://localhost:8080/v1`)
- `LLAMA_SUITE_SWAP_UI_URL` (default `http://localhost:8080/ui`)
- `LLAMA_SUITE_OPEN_WEBUI_URL` (default `http://localhost:3000`)

Proposed endpoint:
- `GET /api/system/links` returns these URLs so the SPA can render correct links.

## Security model

### Layer 1: Ingress OIDC (primary for Kubernetes)
- Protect the Web UI via DeployKube ingress/gateway using Keycloak/OIDC.
- Keeps the app simple and cloud-agnostic (OIDC is externalized).

### Layer 2: App-level API key (optional, defense-in-depth)
- If `LLAMA_SUITE_API_KEY` is set:
  - Require auth for `/api/**` and `/ws/**` (except `/api/health` and auth endpoints)
  - Implement `POST /api/auth/login` which sets an httpOnly cookie for browser UX
  - Validate cookie on each request and on WebSocket handshake

## Packaging and deployment

### Local containers (Docker/Podman/OrbStack)
- Provide a Compose stack that:
  - Runs `llama-suite-webui`
  - Optionally runs Open WebUI (`open-webui` profile)
  - Mounts `configs/` read-only and `runs/` persistent

### Kubernetes (Helm chart)
Helm chart deliverables:
- Deployment + Service
- Optional Ingress (disabled by default; DeployKube uses Gateway API)
- Config mount options:
  - `existingConfigMap` (recommended for GitOps)
  - optional inline config for dev
- PVC for `runs/` with configurable access mode
- Secure pod settings (non-root, readOnlyRootFilesystem, seccomp, drop caps)
- Probes: `/api/health`

## DeployKube integration (Argo CD)
Add a DeployKube GitOps component for `llama-suite-webui`:
- Namespace + Service + HTTPRoute
- ExternalSecret for API key (optional)
- Argo CD Application that deploys the Helm chart (OCI in Harbor or GHCR)

## Work breakdown (tracking)

### Phase 0 (done)
- [x] Add `LLAMA_SUITE_ROOT` override for root discovery.
- [x] Add container artifacts (Web UI image + compose).
- [x] Add macOS OrbStack “double click” start/stop scripts.

### Phase 1 (GitOps safety + URLs)
- [ ] Implement `LLAMA_SUITE_MODE=gitops` enforcement (disable writes + subprocess endpoints).
- [ ] Add `/api/system/links` and update SPA to use it (remove hardcoded `localhost` links).

### Phase 2 (app-level auth)
- [ ] Add optional API key auth + cookie-based browser login flow.
- [ ] Add frontend login prompt when auth is required.

### Phase 3 (Helm chart)
- [x] Create `deploy/charts/llama-suite-webui` Helm chart.
- [ ] Document values for DeployKube (URLs, mode, config mount, PVC).

### Phase 4 (DeployKube GitOps component)
- [ ] Add DeployKube manifests + Argo CD Application for `llama-suite-webui`.
- [ ] Add smoke test instructions (curl `/api/health`, browser access).

## Test plan / acceptance criteria

### Automated tests (Python)
- Root discovery honors `LLAMA_SUITE_ROOT`.
- `LLAMA_SUITE_MODE=gitops` blocks mutation endpoints.
- Auth cookie works when `LLAMA_SUITE_API_KEY` is set.
- `/api/system/links` returns env-configured URLs.

### Local acceptance (macOS OrbStack)
- Double-click `start.command` brings up UI at `http://localhost:8088`
- Config mount read-only behaves as expected
- UI links reflect configured runtime URLs from `.env`

### Kubernetes acceptance
- Pod is Ready via `/api/health`
- Argo CD syncs cleanly and reconciles drift
- Gateway route exposes UI with OIDC protection
- No mutation endpoints available in GitOps mode

## Risks / mitigations
- **CRLF + executable scripts** on Windows checkouts:
  - Mitigation: `.gitattributes` enforces LF for `.command`/Dockerfiles.
  - Also document `chmod +x` after zip downloads.
- **UI expectations vs GitOps**:
  - Mitigation: make “local mode” explicit; in-cluster mode clearly disables controls.
- **PVC access mode**:
  - Default RWO works for single webui instance; RWX might be needed if runners are separate later.

## Repository pointers
- Web UI server: `src/llama_suite/webui/server.py`
- Config utils/root detection: `src/llama_suite/utils/config_utils.py`
- Compose: `deploy/compose/docker-compose.yml`
- OrbStack launchers: `deploy/orbstack/start.command`, `deploy/orbstack/stop.command`
- Web UI image: `deploy/containers/webui/Dockerfile`
- Marketplace prep templates: `deploy/marketplace/llama-suite-webui/`
