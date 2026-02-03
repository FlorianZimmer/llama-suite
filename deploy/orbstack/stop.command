#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker CLI not found. Install/start OrbStack first."
  exit 1
fi

echo "Stopping llama-suite Web UI..."
docker compose -f deploy/compose/docker-compose.yml down

