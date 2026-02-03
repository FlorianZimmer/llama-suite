# DeployKube Marketplace Prep: `llama-suite-webui`

This folder exists to make `llama-suite-webui` easy to onboard into the DeployKube Marketplace once the marketplace APIs/CRDs exist.

## What class of “marketplace product” is this?

This is a **Marketplace Offering (Curated Deployment)** (not a Fully Managed Service):
- The platform provides a curated, hardened, GitOps-friendly deployment package.
- The tenant owns day-2 operations (upgrades, backups for persisted data, monitoring/alerting consumption).

## Artifacts to publish

- Container image: `llama-suite-webui`
- Helm chart: `deploy/charts/llama-suite-webui` (prefer publishing as an OCI chart)

## Installation model (GitOps-first)

- A tenant GitOps repo should contain:
  - An Argo CD Application (or ApplicationSet) that installs the chart
  - A `values.yaml` (tenant-specific)
  - Optional: ConfigMaps for `config.base.yaml` + overrides

See:
- `values.tenant.example.yaml`
- `argocd-application.example.yaml`

## Notes on endpoints / “hybrid runtime”

The Web UI is **endpoint-only** in `LLAMA_SUITE_MODE=gitops` and will not start/stop runtimes.
For a “desktop GPU when available, otherwise in-cluster CPU fallback” setup, the recommended approach is to provide a stable OpenAI-compatible endpoint URL via `LLAMA_SUITE_SWAP_API_URL` (e.g., a small failover proxy).
