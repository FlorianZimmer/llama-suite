#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mlx_repo="${repo_root}/vendor/mlx-lm-mtp"
patch_file="${repo_root}/patches/mlx-lm-mtp/qwen-mtp-single-request-prefix-cache.patch"

if [[ ! -d "${mlx_repo}/.git" ]]; then
  echo "Missing MLX MTP checkout: ${mlx_repo}" >&2
  exit 1
fi

if git -C "${mlx_repo}" apply --check "${patch_file}" 2>/dev/null; then
  git -C "${mlx_repo}" apply "${patch_file}"
  echo "Applied ${patch_file}"
elif git -C "${mlx_repo}" apply --reverse --check "${patch_file}" 2>/dev/null; then
  echo "Patch already applied: ${patch_file}"
else
  echo "Patch does not apply cleanly: ${patch_file}" >&2
  exit 1
fi
