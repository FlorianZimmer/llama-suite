#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker CLI not found. Install/start OrbStack first."
  exit 1
fi

echo "Starting llama-suite Web UI..."
docker compose -f deploy/compose/docker-compose.yml up -d --build

echo
echo "llama-suite Web UI is starting. Open:"
echo "  http://localhost:8088"
echo
echo "Logs:"
echo "  docker compose -f deploy/compose/docker-compose.yml logs -f llama-suite-webui"

