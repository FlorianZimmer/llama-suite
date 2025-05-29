#!/usr/bin/env python3
"""
Uninstaller for the llama-suite.

Deletes:
  - llama.cpp build/source
  - llama-swap binaries/source
  - The Python virtual environment (llama-suite-venv)
  - Stops and removes the Open WebUI Docker container.

Keeps:
  - Model files (models/)
  - Open WebUI persistent data (open-webui-data/)
  - Log files (logs/)
  - Benchmark scripts AND THEIR RESULTS (bench/)
  - Evaluation scripts AND THEIR RESULTS (eval/ and eval/results/)
  - Configuration files (config.*.yaml)
  - Core scripts (installer.py, uninstall.py, evaluate-models.py, etc.)
  - Other user-created files.
"""

from __future__ import annotations

import argparse
# import glob # No longer needed
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

# ---------------------------------------------------------------------------
# Defaults (same locations the installer used)
# ---------------------------------------------------------------------------

DEFAULT_INSTALL_DIR = {
    "Windows": Path.home() / "llama_suite",
    "Darwin": Path.home() / "Documents" / "code" / "llama-suite-py",
    "Linux": Path.home() / "llama_suite",
}.get(platform.system(), Path.home() / "llama_suite")

VENV_NAME = "llama-suite-venv" # Must match installer
OPEN_WEBUI_CONTAINER = "open-webui"
OPEN_WEBUI_IMAGE = "ghcr.io/open-webui/open-webui:main" # For optional image deletion

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def yn(prompt: str, default: bool | None = None) -> bool:
    """Yes/No question that works in cmd, PowerShell, bash, etc."""
    suffix = " [Y/n] " if default else " [y/N] " if default is False else " [y/n] "
    while True:
        resp = input(prompt + suffix).strip().lower()
        if not resp and default is not None:
            return default
        if resp in ("y", "yes"):
            return True
        if resp in ("n", "no"):
            return False
        print("Please answer y or n.")

def run_quiet(*cmd: str) -> None:
    """Run a command; ignore errors and output (for Docker stop/rm)."""
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except FileNotFoundError:
        print(f"  Command '{cmd[0]}' not found, skipping.")
    except Exception as e:
        print(f"  Error running {' '.join(cmd)}: {e}")


def rm_dir_if_exists(path: Path, operation_desc: str) -> None:
    """Removes a directory if it exists."""
    if path.exists():
        if path.is_dir():
            try:
                shutil.rmtree(path)
                print(f"  Removed {operation_desc}: {path}")
            except Exception as e:
                print(f"  ERROR removing {path}: {e}. You may need to remove it manually.")
        else:
            print(f"  Skipping {path}: Not a directory (unexpected).")
    else:
        print(f"  {operation_desc} not found, skipping: {path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Uninstalls components of the llama-suite setup by the installer."
    )
    ap.add_argument(
        "--install-dir",
        type=Path,
        default=DEFAULT_INSTALL_DIR,
        help=f"Root directory of the llama-suite installation (default: {DEFAULT_INSTALL_DIR})",
    )
    ap.add_argument(
        "--yes", "-y", action="store_true",
        help="Automatically answer yes to prompts (use with caution)."
    )
    args = ap.parse_args()

    root: Path = args.install_dir.expanduser().resolve()

    if not root.is_dir():
        print(f"Error: Installation directory '{root}' not found or is not a directory.")
        sys.exit(1)

    # Directories created/managed by the installer that should be deleted
    dirs_to_delete_map = {
        "llama.cpp source/build": root / "llama.cpp",
        "llama-swap binaries/source": root / "llama-swap",
        "Python Virtual Environment": root / VENV_NAME,
        # "Evaluation results": root / "eval" / "results", # <-- REMOVED FROM DELETION
    }

    # Items explicitly kept
    items_to_keep_display = [
        str(root / "models"),
        str(root / "open-webui-data"),
        str(root / "logs"),
        str(root / "bench"),          # Entire bench directory, including results
        str(root / "eval"),           # Entire eval directory, including scripts and results
        str(root / "config.base.yaml"),
        str(root / "config.effective.yaml"),
        str(root / "overrides"),
        str(root / "installer.py"),
        str(root / "uninstall.py"),
        str(root / "utils"),
        # Add other key scripts or directories here
    ]
    # Filter out non-existent items for cleaner display
    items_to_keep_display_filtered = sorted([p_str for p_str in items_to_keep_display if Path(p_str).exists() or "*" in p_str])


    print(
        dedent(
            f"""
        ============================================================================
        llama-suite UNINSTALLER
        Target Installation Directory: {root}
        ----------------------------------------------------------------------------
        This script will attempt to DELETE the following items if they exist:
        {chr(10).join(f'  • {desc} (at {path})' for desc, path in dirs_to_delete_map.items())}

        It will also stop and remove the Docker container "{OPEN_WEBUI_CONTAINER}".

        The following essential items will NOT be touched:
        {chr(10).join('  • ' + p_str for p_str in items_to_keep_display_filtered)}
        (This includes the 'bench/' and 'eval/' directories and all their contents
         like benchmark and evaluation results, and other user-created files/directories)
        ============================================================================
        """
        )
    )

    if not args.yes and not yn("Are you sure you want to continue?", default=False):
        print("Uninstallation aborted.")
        return

    # -----------------------------------------------------------------------
    # Stop / remove Docker container
    # -----------------------------------------------------------------------
    print(f"\n--- Docker Operations ---")
    print(f"Stopping Docker container '{OPEN_WEBUI_CONTAINER}' (if running) …")
    run_quiet("docker", "stop", OPEN_WEBUI_CONTAINER)
    print(f"Removing Docker container '{OPEN_WEBUI_CONTAINER}' (if it exists) …")
    run_quiet("docker", "rm", OPEN_WEBUI_CONTAINER)

    if args.yes or yn(f"Delete the Docker image '{OPEN_WEBUI_IMAGE}' as well? This frees up disk space.", default=False):
        print(f"Removing Docker image '{OPEN_WEBUI_IMAGE}' …")
        run_quiet("docker", "rmi", OPEN_WEBUI_IMAGE)
    else:
        print(f"Keeping Docker image '{OPEN_WEBUI_IMAGE}'.")


    # -----------------------------------------------------------------------
    # Delete installer-managed directories
    # -----------------------------------------------------------------------
    print(f"\n--- Filesystem Cleanup in {root} ---")
    for desc, path in dirs_to_delete_map.items():
        rm_dir_if_exists(path, desc)

    print("\n----------------------------------------------------------------------------")
    print("Uninstallation of installer-managed components complete.")
    print("The following items were intentionally left in place (if they existed):")
    for p_str in items_to_keep_display_filtered:
        if Path(p_str).exists(): # Re-check existence
            print(f"  • {p_str}")

    print(
        "\nNOTE: System-wide packages (Git, CMake, Go, Docker, system Python, etc.) "
        "and Python packages installed outside the removed venv (if any) are untouched."
    )
    print(f"If you wish to remove the entire llama-suite, delete the directory: {root}")
    print("----------------------------------------------------------------------------")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)