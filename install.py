#!/usr/bin/env python3
"""
Cross-platform installer for the llama-swap + llama.cpp stack.

Features
========
* Creates a Python virtual environment "llama-suite-venv".
* Installs dependencies (lm-eval, etc.) into the venv.
* Works on Windows, macOS, Linux with a single file.
* By default **downloads the latest GitHub release assets** for the host
  platform so users do NOT need compilers, Go, CMake…
* `--build-from-source swap|cpp|all` lets power-users compile anyway.
* Creates the same directory structure the watcher expects.
* Prints a next-steps summary identical to your Bash version.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import venv # Added for venv creation
import zipfile
from pathlib import Path
from textwrap import dedent
from typing import Dict, List, Optional, Tuple
from urllib.request import Request, urlopen

# -----------------------------------------------------------------------------
# Configuration you may want to tweak
# -----------------------------------------------------------------------------

DEFAULT_INSTALL_BASE = {
    "Windows": Path.home() / "llama_suite",
    "Darwin": Path.home() / "Documents" / "code" / "llama-suite-py",
    "Linux": Path.home() / "llama_suite",
}.get(platform.system(), Path.home() / "llama_suite")

LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp.git"
LLAMA_SWAP_REPO = "https://github.com/mostlygeek/llama-swap.git"
LLAMA_CPP_RELEASE_REPO = "ggml-org/llama.cpp"
LLAMA_SWAP_RELEASE_REPO = "mostlygeek/llama-swap"
OPEN_WEBUI_IMAGE = "ghcr.io/open-webui/open-webui:main"

VENV_NAME = "llama-suite-venv"
REQUIRED_PACKAGES = [
    "lm-eval[llama_cpp,api]",
    "PyYAML",
    "requests",
    "psutil",
    "colorama",
    # Add any other direct Python dependencies of your scripts here
    # if they aren't pulled in by lm-eval.
]

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

IS_WINDOWS = platform.system() == "Windows"

def run(cmd: List[str] | str, cwd: Optional[Path] = None, check_return_code: bool = True) -> None:
    """Run a command and optionally raise on non-zero exit."""
    if isinstance(cmd, list):
        printable = " ".join(str(c) for c in cmd)
    else:
        printable = cmd
    print(f"→ {printable}")
    # Ensure all parts of cmd are strings for Popen
    cmd_str_list = [str(c) for c in cmd] if isinstance(cmd, list) else cmd
    result = subprocess.run(cmd_str_list, cwd=cwd, shell=isinstance(cmd, str))
    if check_return_code and result.returncode != 0:
        sys.exit(f"ERROR: command failed (code {result.returncode}): {printable}")


def download(url: str, dest: Path) -> None:
    # ... (same as before)
    print(f"Downloading {url} …")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as resp, open(dest, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    size = dest.stat().st_size / 1024**2
    print(f"  saved {dest} ({size:.1f} MB)")


def get_latest_release(repo: str) -> Dict:
    # ... (same as before)
    api = f"https://api.github.com/repos/{repo}/releases/latest"
    req = Request(api, headers={"User-Agent": "llama-stack-installer"})
    with urlopen(req) as resp:
        return json.load(resp)


def asset_for_platform(release: Dict, stem_match: str) -> Optional[Dict]:
    # ... (same as before)
    system = platform.system().lower()
    machine = platform.machine().lower().replace("x86_64", "amd64")

    for asset in release["assets"]:
        name = asset["name"].lower()
        if stem_match in name and system in name and machine in name:
            return asset
    return None


def extract_archive(archive: Path, dest_dir: Path) -> None:
    # ... (same as before)
    print(f"Extracting {archive} …")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest_dir)
    elif archive.suffixes[-2:] == [".tar", ".gz"] or archive.suffix == ".tgz":
        with tarfile.open(archive) as tf:
            tf.extractall(dest_dir)
    else:
        dest = dest_dir / archive.name
        shutil.move(str(archive), dest)
        archive = dest
    for p in dest_dir.glob("**/*"):
        if p.is_file() and os.access(p, os.X_OK) and not IS_WINDOWS: # chmod mostly for non-Windows
            try:
                p.chmod(p.stat().st_mode | 0o111) # add +x
            except Exception as e:
                print(f"  Warning: could not chmod {p}: {e}")


def ensure_executable(path: Path) -> None:
    # ... (same as before, with a check for Windows)
    if not IS_WINDOWS:
        try:
            path.chmod(path.stat().st_mode | 0o111)  # add +x
        except Exception as e:
            print(f"  Warning: could not chmod {path}: {e}")


# -----------------------------------------------------------------------------
# Venv and Pip Helper
# -----------------------------------------------------------------------------

def get_venv_paths(venv_dir: Path) -> Tuple[Path, Path]:
    """Gets paths to python and pip executables within a venv."""
    if IS_WINDOWS:
        python_exe = venv_dir / "Scripts" / "python.exe"
        pip_exe = venv_dir / "Scripts" / "pip.exe"
    else:
        python_exe = venv_dir / "bin" / "python"
        pip_exe = venv_dir / "bin" / "pip"
    return python_exe, pip_exe

# -----------------------------------------------------------------------------
# Build / download functions
# -----------------------------------------------------------------------------
# obtain_llama_swap, build_llama_swap, download_llama_swap
# obtain_llama_cpp, build_llama_cpp, download_llama_cpp, asset_for_platform_cpp
# (These functions remain largely the same as in your provided script,
#  I'm omitting them here for brevity but they should be included)

def download_llama_swap(binary_dir: Path) -> Path:
    release = get_latest_release(LLAMA_SWAP_RELEASE_REPO)
    asset = asset_for_platform(release, "llama-swap")
    if not asset:
        sys.exit(
            f"No release asset for platform in {LLAMA_SWAP_RELEASE_REPO}. "
            "Try --build-from-source swap."
        )
    url = asset["browser_download_url"]
    tmp = binary_dir / asset["name"]
    download(url, tmp)
    extract_archive(tmp, binary_dir)
    # Find executable
    for cand in binary_dir.iterdir():
        if cand.name.lower().startswith("llama-swap") and (cand.is_file() or cand.is_symlink()):
            # On non-Windows, os.access check can be part of this condition
            # if IS_WINDOWS or os.access(cand, os.X_OK):
            ensure_executable(cand)
            return cand
    sys.exit("Downloaded release but could not locate llama-swap binary.")


def build_llama_swap(src_dir: Path, binary_dir: Path) -> Path:
    if src_dir.exists():
        run(["git", "pull"], cwd=src_dir)
    else:
        run(["git", "clone", LLAMA_SWAP_REPO, str(src_dir)])
    exe_name = "llama-swap.exe" if IS_WINDOWS else "llama-swap"
    exe_path = binary_dir / exe_name
    run(["go", "build", "-o", str(exe_path), "."], cwd=src_dir)
    ensure_executable(exe_path)
    return exe_path

def obtain_llama_swap(base: Path, build: bool) -> Path:
    src_dir = base / "llama-swap-source"
    bin_dir = base / "llama-swap"
    if bin_dir.exists() and bin_dir.is_file():
        legacy_file = bin_dir
        backup = bin_dir.with_suffix(".old")
        print(f"Found old binary at {legacy_file}. Moving to {backup}")
        legacy_file.rename(backup)
    bin_dir.mkdir(parents=True, exist_ok=True)
    return (
        build_llama_swap(src_dir, bin_dir)
        if build
        else download_llama_swap(bin_dir)
    )


def build_llama_cpp(src_dir: Path) -> Path:
    if src_dir.exists():
        run(["git", "pull"], cwd=src_dir)
    else:
        run(["git", "clone", "--depth", "1", LLAMA_CPP_REPO, str(src_dir)])

    build_dir = src_dir / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir()

    cmake_flags = [
        "-DGGML_METAL=ON" if platform.system() == "Darwin" else "", # Keep this for macOS Metal
        "-DLLAMA_CURL=OFF", # As per your original
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    # Add CUDA flags if on Windows and potentially wanting CUDA
    if IS_WINDOWS:
        # This is a simplistic check; a real build system might detect CUDA toolkit
        # cmake_flags.append("-DGGML_CUDA=ON")
        pass


    cmake_args = ["cmake", ".."] + [flag for flag in cmake_flags if flag] # Filter empty strings
    run(cmake_args, cwd=build_dir)

    jobs = os.cpu_count() or 4
    run(["cmake", "--build", ".", "--config", "Release", "-j", str(jobs)], cwd=build_dir)

    # server binary can be in different spots
    # Preferring the direct bin output first
    server_exe_name = "server.exe" if IS_WINDOWS else "server" # llama.cpp names it 'server'
    
    # Check common locations
    # 1. build/bin/ (most common for CMake projects)
    # 2. build/ (sometimes for simpler CMake or if no install step)
    # 3. build/Release/ (for MSVC multi-config generators if "Release" is specified in build command) - covered by --config Release

    # llama.cpp specific server name is 'server' or 'server.exe'
    # but sometimes people rename it to 'llama-server'
    possible_names = { "server", "llama-server"}
    if IS_WINDOWS:
        possible_names = {name + ".exe" for name in possible_names}

    for loc in [build_dir / "bin", build_dir]:
        for name_stem in ["server", "llama-server"]:
            exe_name = f"{name_stem}.exe" if IS_WINDOWS else name_stem
            cand = loc / exe_name
            if cand.is_file():
                ensure_executable(cand)
                return cand
    
    raise SystemExit("Build finished but could not find llama.cpp server executable (server/server.exe or llama-server/llama-server.exe).")


def asset_for_platform_cpp(release_json: dict) -> Optional[dict]:
    system_token = {
        "Darwin": "macos",
        "Windows": "win",
        "Linux": "ubuntu", # or just "linux" sometimes
    }.get(platform.system(), platform.system().lower())

    arch_token = {
        "amd64": "x64",
        "x86_64": "x64",
        "arm64": "arm64",
        "aarch64": "arm64",
    }.get(platform.machine().lower(), platform.machine().lower())

    candidates: list[dict] = []
    for asset in release_json["assets"]:
        name = asset["name"].lower()
        # Make matching more robust
        is_zip = name.endswith(".zip")
        # Some releases might use "linux" instead of "ubuntu"
        system_match = system_token in name or (system_token == "ubuntu" and "linux" in name)

        if system_match and arch_token in name and is_zip and "bin" in name : # "bin" to ensure it's a binary release
            candidates.append(asset)

    if not candidates:
        return None

    def priority(a: dict) -> int:
        n = a["name"].lower()
        if "cpu" in n: return 0
        if "avx2" in n and "cpu" in n: return 0 # Prefer AVX2 CPU if available
        if "vulkan" in n: return 1
        if "cuda" in n or "cu1" in n or "cudart" in n: return 2 # cu1 for cu11, cu12 etc.
        if "metal" in n and system_token == "macos": return 0 # Metal is preferred on macOS
        return 3 # generic or unknown

    candidates.sort(key=priority)
    if candidates:
      print(f"  Found C++ release candidates: {[c['name'] for c in candidates]}. Selected: {candidates[0]['name']}")
      return candidates[0]
    return None


def download_llama_cpp(binary_dir: Path) -> Path:
    """Downloads pre-built llama.cpp server and extracts it."""
    if not LLAMA_CPP_RELEASE_REPO:
        raise SystemExit(
            "LLAMA_CPP_RELEASE_REPO is not set; cannot download pre-built "
            "binaries. Use --build-from-source cpp."
        )

    print(f"Fetching latest release from GitHub API: {LLAMA_CPP_RELEASE_REPO}")
    release = get_latest_release(LLAMA_CPP_RELEASE_REPO)
    print(f"  Latest release tag: {release.get('tag_name', 'N/A')}")
    asset = asset_for_platform_cpp(release)
    if not asset:
        raise SystemExit(
            f"No suitable asset for this platform ({platform.system()}/{platform.machine()}) "
            f"in {LLAMA_CPP_RELEASE_REPO} release {release.get('tag_name', 'N/A')}. "
            "Try --build-from-source cpp."
        )

    url = asset["browser_download_url"]
    # binary_dir for llama.cpp downloads is the 'llama.cpp' folder itself usually.
    # The zip often contains a folder structure like 'llama-bXYZ-bin-os-arch/...'
    # So we extract into a temporary place or directly into binary_dir and then find the exe.
    tmp_archive_path = binary_dir / asset["name"] # Save archive in llama.cpp dir
    download(url, tmp_archive_path)
    extract_archive(tmp_archive_path, binary_dir) # Extract into llama.cpp dir

    # Find the server executable (server.exe or server)
    # It might be in a subdirectory if the zip was structured that way,
    # or directly in binary_dir if the zip was flat.
    possible_server_names = {"server.exe", "server"} if IS_WINDOWS else {"server"}
    
    # Search recursively
    for p in binary_dir.rglob("*"):
        if p.is_file() and p.name.lower() in possible_server_names:
            # Check if it's executable (especially after extraction from zip)
            # The file might be in e.g. llama.cpp/bin/server or llama.cpp/server
            ensure_executable(p)
            return p # Return the first one found

    raise SystemExit(
        f"Downloaded and extracted {asset['name']}, but could not locate "
        f"the llama.cpp server executable ({'/'.join(possible_server_names)}) within {binary_dir}."
    )


def obtain_llama_cpp(base: Path, build: bool) -> Path:
    src_dir = base / "llama.cpp" # This is where source OR downloaded binaries will live
    if src_dir.exists() and src_dir.is_file():
        backup = src_dir.with_suffix(".old")
        print(f"Found old file at {src_dir}. Moving it to {backup}")
        src_dir.rename(backup)
    src_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists for both build and download

    if build:
        return build_llama_cpp(src_dir)
    else:
        # download_llama_cpp expects the directory where binaries (or the archive) land.
        # For llama.cpp, this is typically the 'llama.cpp' directory itself.
        return download_llama_cpp(src_dir)


# -----------------------------------------------------------------------------
# Installer main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install llama.cpp + llama-swap + Open WebUI helper files."
    )
    parser.add_argument(
        "--install-dir",
        type=Path,
        default=DEFAULT_INSTALL_BASE,
        help=f"Where to place everything (default: {DEFAULT_INSTALL_BASE})",
    )
    parser.add_argument(
        "--build-from-source",
        choices=["swap", "cpp", "all"],
        help="Compile instead of downloading releases for the given project(s).",
    )
    args = parser.parse_args()

    install_dir: Path = args.install_dir.expanduser().resolve()
    model_dir = install_dir / "models"
    webui_dir = install_dir / "open-webui-data"
    eval_dir = install_dir / "eval"
    venv_dir = install_dir / VENV_NAME

    build_swap = args.build_from_source in {"swap", "all"}
    build_cpp = args.build_from_source in {"cpp", "all"}

    print(
        dedent(
            f"""
        =============================================================================
        llama-suite installer (with venv support)
        Target directory  : {install_dir}
        Virtual Env       : {venv_dir}
        Build from source : swap={build_swap}  cpp={build_cpp}
        =============================================================================
        """
        )
    )

    # -----------------------------------------------------------------------------
    # Directory layout (Create all necessary directories first)
    # -----------------------------------------------------------------------------
    for d in (install_dir, model_dir, webui_dir, eval_dir, eval_dir / "results"):
        d.mkdir(parents=True, exist_ok=True)
    print(f"Ensured base directories exist under: {install_dir}")

    # -----------------------------------------------------------------------------
    # Create/Update Virtual Environment
    # -----------------------------------------------------------------------------
    print(f"\n=== Setting up Python virtual environment: {venv_dir} ===")
    if not venv_dir.exists():
        print(f"Creating venv using Python: {sys.executable}")
        builder = venv.EnvBuilder(with_pip=True, upgrade_deps=True)
        builder.create(str(venv_dir))
        print("Venv created.")
    else:
        print("Venv already exists. Skipping creation, will install/update packages.")

    venv_python, venv_pip = get_venv_paths(venv_dir)
    if not venv_python.exists() or not venv_pip.exists():
        sys.exit(f"ERROR: Venv created but Python/Pip not found at expected paths: {venv_python}, {venv_pip}")

    print(f"Using pip from venv: {venv_pip}")
    for package_spec in REQUIRED_PACKAGES:
        print(f"Installing/Updating: {package_spec} into venv...")
        # Using --upgrade to ensure latest compatible versions
        # Using --no-cache-dir can sometimes help with stale cache issues
        run([str(venv_pip), "install", "--upgrade", package_spec], check_return_code=False) # Don't exit if one fails, print warning
    print("Python package installation attempt complete.")


    # -----------------------------------------------------------------------------
    # llama-swap
    # -----------------------------------------------------------------------------
    print("\n=== Installing llama-swap ===")
    swap_path = obtain_llama_swap(install_dir, build_swap)
    print(f"llama-swap binary → {swap_path}")

    # -----------------------------------------------------------------------------
    # llama.cpp server
    # -----------------------------------------------------------------------------
    print("\n=== Installing llama.cpp server ===")
    cpp_server_path = obtain_llama_cpp(install_dir, build_cpp)
    print(f"llama-server binary → {cpp_server_path}")

    # -----------------------------------------------------------------------------
    # Pull Open WebUI image (optional)
    # -----------------------------------------------------------------------------
    try:
        print("\n=== Pulling Open WebUI image ===")
        run(["docker", "pull", OPEN_WEBUI_IMAGE], check_return_code=False) # Don't fail install if docker not present/fails
    except Exception as e:
        print(f"WARNING: Docker pull failed ({e}). You can pull manually later.")

    # -----------------------------------------------------------------------------
    # Summary & next steps
    # -----------------------------------------------------------------------------
    port_hint = "3000"
    eval_script_path = eval_dir / "evaluate-models.py"
    eval_results_path = eval_dir / "results"

    activation_command_unix = f"source \"{venv_dir / 'bin' / 'activate'}\""
    activation_command_windows = f"CALL \"{venv_dir / 'Scripts' / 'activate.bat'}\"" # CALL for batch files
    # For PowerShell, it would be: `& "{venv_dir / 'Scripts' / 'Activate.ps1'}"`

    summary = dedent(
        f"""
        -----------------------------------------------------------------------------
        Installation complete 🎉
        -----------------------------------------------------------------------------
        llama.cpp server : {cpp_server_path}
        llama-swap       : {swap_path}
        Models           : {model_dir}
        Open WebUI mount : {webui_dir}
        Docker image     : {OPEN_WEBUI_IMAGE} (if pull succeeded)
        Python Venv      : {venv_dir}
                           (Python packages like lm-eval are installed here)

        NEXT STEPS
        ==========
        0. Place your `evaluate-models.py` and `benchmark-models.py` (if used)
           into the appropriate subdirectories (`{eval_dir}` or `{install_dir/'bench'}`).
           This installer does not copy them.

        1. **ACTIVATE THE VIRTUAL ENVIRONMENT** in your terminal:
           - On Linux/macOS (bash/zsh):
             {activation_command_unix}
           - On Windows (Command Prompt):
             {activation_command_windows}
           - On Windows (PowerShell):
             & "{venv_dir / 'Scripts' / 'Activate.ps1'}"
           (You'll need to do this in every new terminal session where you run the scripts)

        2. Drop your GGUF models into `{model_dir}`.
        3. Create/update `config.base.yaml` in `{install_dir}`.
        4. Start llama-swap (ensure venv is active):

               "{swap_path}" -config config.base.yaml

        5. Start Open WebUI (example):

               docker run -d -p {port_hint}:8080 ^
                   -v "{webui_dir}:/app/backend/data" ^
                   --add-host=host.docker.internal:host-gateway ^
                   --name open-webui --restart unless-stopped ^
                   {OPEN_WEBUI_IMAGE}

           (Replace `^` with `\` on Unix shells)

        6. Open http://localhost:{port_hint} (Open WebUI) and configure connections.

        7. To run evaluations (ensure venv is active and scripts/configs are set up):

               python "{eval_script_path}" --tasks hellaswag --limit 10

           Results will be in `{eval_results_path}`.
           Run benchmarks similarly using python "{install_dir/'bench'/'benchmark-models.py'}"
        -----------------------------------------------------------------------------
        """
    )
    print(summary)

if __name__ == "__main__":
    main()