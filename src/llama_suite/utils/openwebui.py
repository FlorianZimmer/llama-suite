#!/usr/bin/env python3
"""
Ensure an Open WebUI container exists and is running.

Usage (called by installer):
  python -m llama_suite.utils.openwebui --name open-webui \
      --port 3000 --data-dir F:\\LLMs\\llama-suite\\var\\open-webui\\data \
      --image ghcr.io/open-webui/open-webui:main

Or use an existing Docker/Podman named volume (instead of a host directory bind-mount):
  python -m llama_suite.utils.openwebui --name open-webui \
      --port 3000 --data-volume open-webui \
      --image ghcr.io/open-webui/open-webui:main

It will:
  - Detect a container runtime: docker, podman, or nerdctl (in that order),
    or use --runtime to force one.
  - Create the container if it doesn't exist (with data mount + port mapping).
  - Start the container if it exists but isn't running.
  - No-op if it's already running.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def _print(level: str, msg: str) -> None:
    print(f"{level}: {msg}", flush=True)


def info(msg: str) -> None:
    _print("INFO", msg)


def warn(msg: str) -> None:
    _print("WARN", msg)


def err(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)


# ------------------------------ runtime detection ------------------------------

RUNTIMES_ORDER = ("docker", "podman", "nerdctl")


def detect_runtime(explicit: Optional[str] = None) -> Tuple[str, str]:
    """
    Returns (name, path). Raises SystemExit if not found.
    """
    if explicit:
        exe = shutil.which(explicit)
        if not exe:
            raise SystemExit(f"Requested container runtime '{explicit}' not found in PATH.")
        return explicit, exe

    for name in RUNTIMES_ORDER:
        exe = shutil.which(name)
        if exe:
            return name, exe
    raise SystemExit("No container runtime found. Install Docker, Podman, or nerdctl and try again.")


# ------------------------------ subprocess helper ------------------------------

def run(rt_path: str, args: List[str], check: bool = False) -> subprocess.CompletedProcess:
    """
    Run runtime command and return CompletedProcess.
    """
    cmd = [rt_path, *args]
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=check)


# ------------------------------ container helpers ------------------------------

def container_exists(rt_path: str, name: str) -> bool:
    # IMPORTANT: plain `docker inspect <name>` can succeed for non-container objects
    # (e.g. volumes/networks/images with the same name). Use container-scoped inspect.
    cp = run(rt_path, ["container", "inspect", name])
    return cp.returncode == 0


def container_running(rt_path: str, name: str) -> Optional[bool]:
    """
    Returns True/False if we could determine running state, or None if unknown.
    Tries common templates across docker/podman/nerdctl.
    """
    # Try boolean Running flag
    cp = run(rt_path, ["container", "inspect", "-f", "{{.State.Running}}", name])
    if cp.returncode == 0:
        out = (cp.stdout or "").strip().lower()
        if out in {"true", "false"}:
            return out == "true"

    # Fallback: status string
    cp = run(rt_path, ["container", "inspect", "-f", "{{.State.Status}}", name])
    if cp.returncode == 0:
        out = (cp.stdout or "").strip().lower()
        if out:
            return out == "running"

    # Podman older templates sometimes use `.State` directly
    cp = run(rt_path, ["container", "inspect", "-f", "{{.State}}", name])
    if cp.returncode == 0:
        out = (cp.stdout or "").strip().lower()
        if out:
            return out == "running"

    return None


def create_container(
    rt: str,
    rt_path: str,
    name: str,
    host_port: int,
    data_dir: Optional[Path],
    data_volume: Optional[str],
    image: str,
    container_port: int = 8080,
) -> None:
    """
    Creates the container with data mount + port mapping. Uses flags compatible with docker/podman/nerdctl.
    """
    if (data_dir is None) == (data_volume is None):
        raise SystemExit("Exactly one of --data-dir or --data-volume must be provided.")

    if data_volume is not None:
        mount_spec = f"{data_volume}:/app/backend/data"
        data_desc = f"named volume '{data_volume}'"
    else:
        assert data_dir is not None
        data_dir.mkdir(parents=True, exist_ok=True)
        mount_spec = f"{str(data_dir)}:/app/backend/data"
        data_desc = f"host dir {data_dir}"

    args = [
        "run", "-d",
        "--name", name,
        "-p", f"{host_port}:{container_port}",
        "-v", mount_spec,
        "--restart", "unless-stopped",
        image,
    ]

    info(f"Creating container '{name}' ({rt}) on port {host_port} with data at {data_desc}")
    cp = run(rt_path, args)
    if cp.returncode != 0:
        err(f"Failed to create container:\n{cp.stderr.strip()}")
        raise SystemExit(1)

    container_id = (cp.stdout or "").strip()
    info(f"Created container {container_id[:12]} ({name}).")


def start_container(rt_path: str, name: str) -> None:
    cp = run(rt_path, ["start", name])
    if cp.returncode != 0:
        err(f"Failed to start container '{name}':\n{cp.stderr.strip()}")
        raise SystemExit(1)
    info(f"Started container '{name}'.")

def stop_container(rt_path: str, name: str) -> None:
    cp = run(rt_path, ["stop", name])
    if cp.returncode != 0:
        err(f"Failed to stop container '{name}':\n{cp.stderr.strip()}")
        raise SystemExit(1)
    info(f"Stopped container '{name}'.")


# ------------------------------ main orchestration ------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Ensure an Open WebUI container exists and is running.")
    p.add_argument("--name", default="open-webui", help="Container name (default: open-webui)")
    p.add_argument("--port", type=int, default=3000, help="Host port to expose (default: 3000)")
    p.add_argument("--stop", action="store_true", help="Stop the container (no-op if missing) and exit")
    p.add_argument("--status", action="store_true", help="Print container status and exit")

    data_group = p.add_mutually_exclusive_group(required=False)
    data_group.add_argument("--data-dir", type=Path, help="Host directory to persist /app/backend/data (bind mount)")
    data_group.add_argument("--data-volume", type=str, help="Named volume to mount at /app/backend/data")
    p.add_argument("--image", default="ghcr.io/open-webui/open-webui:main", help="Container image")
    p.add_argument("--runtime", choices=list(RUNTIMES_ORDER), help="Force a specific runtime (docker|podman|nerdctl)")
    p.add_argument("--container-port", type=int, default=8080, help="Container internal port (default: 8080)")
    args = p.parse_args(argv)

    rt_name, rt_path = detect_runtime(args.runtime)
    info(f"Using container runtime: {rt_name} ({rt_path})")

    exists = container_exists(rt_path, args.name)
    if args.status:
        running = container_running(rt_path, args.name) if exists else False
        info(f"Container exists: {exists}")
        info(f"Container running: {running}")
        return

    if args.stop:
        if not exists:
            warn(f"Container '{args.name}' does not exist. Nothing to stop.")
            return
        stop_container(rt_path, args.name)
        return

    if (args.data_dir is None) == (args.data_volume is None):
        raise SystemExit("Exactly one of --data-dir or --data-volume must be provided.")

    if not exists:
        create_container(
            rt=rt_name,
            rt_path=rt_path,
            name=args.name,
            host_port=args.port,
            data_dir=args.data_dir.resolve() if args.data_dir is not None else None,
            data_volume=args.data_volume,
            image=args.image,
            container_port=args.container_port,
        )
        # After creation, it’s already started due to -d
        info(f"Open WebUI is now available at http://localhost:{args.port}")
        return

    # Exists: check running state
    running = container_running(rt_path, args.name)
    if running is True:
        info(f"Container '{args.name}' already running. Nothing to do.")
        info(f"Open WebUI at http://localhost:{args.port}")
        return
    elif running is False:
        info(f"Container '{args.name}' exists but is not running. Starting...")
        start_container(rt_path, args.name)
        info(f"Open WebUI at http://localhost:{args.port}")
        return
    else:
        # Unknown state - try start anyway
        warn(f"Could not determine running state for '{args.name}'. Attempting to start...")
        start_container(rt_path, args.name)
        info(f"Open WebUI at http://localhost:{args.port}")


if __name__ == "__main__":
    main()
