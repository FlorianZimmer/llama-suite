#!/usr/bin/env python3
"""
Uninstaller for llama-suite (new layout, pyproject-first).

Removes ONLY dependencies and build artifacts:
  - .venv/ (virtual environment)
  - vendor/ (llama-swap + llama.cpp binaries and sources)
  - src/llama_suite.egg-info (editable install metadata)
  - __pycache__ / *.pyc / .ruff_cache / .mypy_cache / .pytest_cache

Keeps ALL runtime/user data:
  - runs/ (bench + eval results, logs)
  - var/ (Open WebUI data, other runtime state)
  - configs/ (incl. configs/generated/)
  - models/
  - datasets/
  - src/, tools/, vendor/*NOT kept* (binaries only), pyproject.toml, etc.

Also:
  - Stops & removes the Open WebUI container (docker/podman/nerdctl), BUT NOT its data dir.

Usage:
  python tools/scripts/uninstall.py [-y] [--runtime docker|podman|nerdctl]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from typing import Iterable, Optional, Tuple

# ------------------------------ repo discovery ------------------------------

def find_repo_root(start: Path | None = None) -> Path:
    cur = (start or Path.cwd()).resolve()
    for p in [cur, *cur.parents]:
        if (p / "pyproject.toml").exists():
            return p
    sys.exit("Could not find repo root (no pyproject.toml found). Run from inside the repo.")

# ------------------------------ prompts / io ------------------------------

def yn(prompt: str, default: bool | None = None) -> bool:
    suffix = " [Y/n] " if default else " [y/N] " if default is False else " [y/n] "
    while True:
        try:
            resp = input(prompt + suffix).strip().lower()
        except EOFError:
            return bool(default)
        if not resp and default is not None:
            return default
        if resp in ("y", "yes"):
            return True
        if resp in ("n", "no"):
            return False
        print("Please answer y or n.")

# ------------------------------ container runtime ------------------------------

RUNTIMES_ORDER = ("docker", "podman", "nerdctl")

def detect_runtime(explicit: Optional[str] = None) -> Tuple[str, str] | None:
    import shutil as _sh
    if explicit:
        exe = _sh.which(explicit)
        return (explicit, exe) if exe else None
    for name in RUNTIMES_ORDER:
        exe = _sh.which(name)
        if exe:
            return name, exe
    return None

def run_quiet(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except Exception:
        pass

# ------------------------------ deletion helpers ------------------------------

def rm_dir_if_exists(path: Path, label: str) -> None:
    if path.exists():
        if path.is_dir():
            try:
                shutil.rmtree(path)
                print(f"  Removed {label}: {path}")
            except Exception as e:
                print(f"  ERROR removing {path}: {e}")
        else:
            try:
                path.unlink()
                print(f"  Removed file {label}: {path}")
            except Exception as e:
                print(f"  ERROR removing file {path}: {e}")
    else:
        print(f"  {label} not found, skipping: {path}")

def rm_globs(root: Path, globs: Iterable[str], label: str) -> None:
    removed = 0
    for pattern in globs:
        for p in root.rglob(pattern):
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
                removed += 1
            except Exception:
                pass
    print(f"  Removed {removed} {label} under {root}")

# ------------------------------ main ------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Uninstall llama-suite dependencies while preserving data.")
    ap.add_argument("-y", "--yes", action="store_true", help="Assume 'yes' to prompts.")
    ap.add_argument("--runtime", choices=list(RUNTIMES_ORDER), help="Force a container runtime to stop/remove the container.")
    ap.add_argument("--container-name", default="open-webui", help="Open WebUI container name (default: open-webui)")
    ap.add_argument("--remove-image", action="store_true", help="Also remove the Open WebUI image (optional).")
    args = ap.parse_args()

    repo = find_repo_root()
    vendor_dir   = repo / "vendor"
    venv_dir     = repo / ".venv"
    egg_info_dir = repo / "src" / "llama_suite.egg-info"  # created by editable install

    # Data we KEEP
    keep_items = [
        repo / "runs",
        repo / "var",                       # includes var/open-webui/data
        repo / "configs",                   # includes configs/generated
        repo / "models",
        repo / "datasets",
        repo / "src",
        repo / "tools",
        repo / "pyproject.toml",
        repo / "README.md",
        repo / "LICENSE",
    ]

    print(dedent(f"""
    ============================================================================
    llama-suite UNINSTALLER
    Repo root: {repo}
    ----------------------------------------------------------------------------
    This will DELETE dependency artifacts:
      - Virtual environment: {venv_dir}
      - Vendor binaries/sources: {vendor_dir}
      - Editable metadata: {egg_info_dir}
      - Python caches: __pycache__, *.pyc, .ruff_cache, .mypy_cache, .pytest_cache

    It will STOP & REMOVE the container '{args.container_name}' (if present),
    but it WILL NOT delete its persisted data under var/open-webui/data.

    It will KEEP your runtime data and project sources, including:
    {chr(10).join("      - " + str(p) for p in keep_items)}
    ============================================================================
    """).strip())

    if not args.yes and not yn("Proceed with uninstall?", default=False):
        print("Uninstallation aborted.")
        return

    # ------------------ Stop/remove Open WebUI container (no data deletion) ------------------
    rt = detect_runtime(args.runtime)
    print("\n--- Container cleanup ---")
    if rt is None:
        print("  No container runtime found in PATH (docker/podman/nerdctl). Skipping container removal.")
    else:
        rt_name, rt_path = rt
        print(f"  Using container runtime: {rt_name} ({rt_path})")
        print(f"  Stopping '{args.container_name}' (if running)...")
        run_quiet([rt_path, "stop", args.container_name])
        print(f"  Removing '{args.container_name}' (if exists)...")
        run_quiet([rt_path, "rm", args.container_name])

        if args.remove_image:
            # Best-effort: try to figure latest pulled image from logs/installer; default to the official main tag
            image = "ghcr.io/open-webui/open-webui:main"
            print(f"  Removing image '{image}' (best effort)...")
            run_quiet([rt_path, "rmi", image])

    # ------------------ Delete dependency artifacts ------------------
    print(f"\n--- Filesystem cleanup under {repo} ---")

    rm_dir_if_exists(venv_dir, "virtual environment")
    rm_dir_if_exists(vendor_dir, "vendor binaries & sources")
    rm_dir_if_exists(egg_info_dir, "editable install metadata")

    # Python caches and common tool caches
    rm_globs(repo, ["__pycache__", "*.pyc", "*.pyo", ".ruff_cache", ".mypy_cache", ".pytest_cache"], "cache items")

    print("\n----------------------------------------------------------------------------")
    print("Uninstallation complete. Your data and results were preserved.")
    print("Kept (if present):")
    for p in keep_items:
        if p.exists():
            print(f"  - {p}")
    print("----------------------------------------------------------------------------")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
