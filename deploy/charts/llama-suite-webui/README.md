# `llama-suite-webui` Helm chart

This chart deploys the `llama-suite` Web UI in a GitOps-friendly, endpoint-only setup (no subprocess spawning when `mode=gitops`).

## Marketplace readiness (DeployKube)

This chart is intended to be a **Marketplace Offering (Curated Deployment)** as described in DeployKube’s marketplace draft (not a “Fully Managed Service”):
- Platform guarantees: packaging quality, default hardening, GitOps reconciliation compatibility.
- Tenant guarantees: day-2 ops (upgrades, backups for any state you persist, monitoring/alerting consumption).

## Quick start

1. Build/publish the image from `deploy/containers/webui/Dockerfile`.
2. Create a ConfigMap containing `config.base.yaml`:
   - Key: `config.base.yaml`
3. Install:

```bash
helm install llama-suite-webui deploy/charts/llama-suite-webui \
  --set image.repository=your-registry/llama-suite-webui \
  --set image.tag=YOUR_TAG \
  --set configs.existingBaseConfigMap=llama-suite-configs
```

## Key values

- `mode`: `gitops` (recommended) or `local` (enables mutation/subprocess APIs)
- `configs.existingBaseConfigMap`: required unless using `configs.inline.enabled=true`
- `configs.existingOverridesConfigMap`: optional ConfigMap mounted at `/data/configs/overrides`
- `persistence.runs.*`: PVC for `/data/runs`
- `auth.*`: optional app-level API key (defense-in-depth; typically behind Ingress OIDC)

## Air-gapped / pinned artifacts

- Prefer pinning by digest using `image.digest` (and omit `image.tag` changes) for reproducible upgrades.
- Mirror the image/chart into your internal registry (e.g., Harbor) and reference that registry from values.
