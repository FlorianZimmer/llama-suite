#!/usr/bin/env python3
"""
Cross-platform installer for the llama-swap + llama.cpp stack.

Highlights
----------
- Creates a virtual environment (default: "llama-suite-venv") and installs Python deps.
- Downloads prebuilt releases or builds from source (llama.cpp, llama-swap).
- Sets up a persistent Open WebUI container (only if it doesn't exist).
- Works on Windows, macOS, and Linux.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from textwrap import dedent
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import venv
import re

# -----------------------------------------------------------------------------
# Constants / Configuration
# -----------------------------------------------------------------------------

DEFAULT_DIRNAME = "llama-suite"
CURRENT_DIR = Path.cwd()
DEFAULT_INSTALL_BASE = (
    CURRENT_DIR if CURRENT_DIR.name.lower() == DEFAULT_DIRNAME.lower()
    else CURRENT_DIR / DEFAULT_DIRNAME
)

LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp.git"
LLAMA_SWAP_REPO = "https://github.com/mostlygeek/llama-swap.git"
LLAMA_CPP_RELEASE_REPO = "ggml-org/llama.cpp"
LLAMA_SWAP_RELEASE_REPO = "mostlygeek/llama-swap"
OPEN_WEBUI_IMAGE = "ghcr.io/open-webui/open-webui:main"

VENV_NAME = "llama-suite-venv"
REQUIRED_PACKAGES: list[str] = [
    "lm-eval[api]",
    "PyYAML",
    "requests",
    "psutil",
    "colorama",
    "watchdog~=4.0",
]

IS_WINDOWS = platform.system() == "Windows"

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

LOG = logging.getLogger("installer")


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )
    # Quieter third-party noise if any gets imported later
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# -----------------------------------------------------------------------------
# Subprocess utility
# -----------------------------------------------------------------------------

def run(cmd: Sequence[str] | str, cwd: Optional[Path] = None) -> None:
    """
    Run a command and raise on non-zero exit. Logs the command executed.
    """
    printable = " ".join(map(str, cmd)) if isinstance(cmd, (list, tuple)) else cmd
    LOG.debug("exec: %s", printable)
    try:
        subprocess.run(cmd, cwd=cwd, check=True, shell=isinstance(cmd, str))
    except subprocess.CalledProcessError as exc:
        LOG.error("Command failed (%s): %s", exc.returncode, printable)
        sys.exit(1)


# -----------------------------------------------------------------------------
# Networking / Downloads
# -----------------------------------------------------------------------------

def download(url: str, dest: Path) -> None:
    """
    Download a URL to a local file (streamed). Creates parent dirs.
    """
    LOG.info("Downloading: %s", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url) as resp, open(dest, "wb") as fh:
            shutil.copyfileobj(resp, fh)
    except (HTTPError, URLError) as e:
        LOG.error("Failed to download %s: %s", url, e)
        sys.exit(1)
    size_mb = dest.stat().st_size / 1024**2
    LOG.debug("Saved %s (%.1f MB)", dest, size_mb)


def get_latest_release(repo: str) -> Dict:
    """
    Query GitHub API for latest release of a repo (org/name).
    Supports optional GITHUB_TOKEN to raise rate limits.
    """
    api = f"https://api.github.com/repos/{repo}/releases/latest"
    headers = {
        "User-Agent": "llama-suite-installer",
        "Accept": "application/vnd.github+json",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = Request(api, headers=headers)
    try:
        with urlopen(req) as resp:
            return json.load(resp)
    except HTTPError as e:
        if e.code == 403:
            LOG.error("GitHub API rate limited. Set GITHUB_TOKEN env var and retry.")
        else:
            LOG.error("GitHub API error (%s): %s", e.code, api)
        sys.exit(1)
    except URLError as e:
        LOG.error("GitHub API unreachable: %s", e)
        sys.exit(1)


# -----------------------------------------------------------------------------
# Archive extraction (safe)
# -----------------------------------------------------------------------------

def _is_within_dir(base: Path, target: Path) -> bool:
    try:
        base_resolved = base.resolve(strict=False)
        target_resolved = target.resolve(strict=False)
        return str(target_resolved).startswith(str(base_resolved))
    except Exception:
        return False


def extract_archive(archive: Path, dest_dir: Path) -> None:
    """
    Extract .zip / .tar.gz safely into dest_dir.
    Adds +x to extracted files on non-Windows when appropriate.
    """
    LOG.info("Extracting: %s", archive.name)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                target = dest_dir / member.filename
                if not _is_within_dir(dest_dir, target):
                    raise SystemExit(f"Unsafe path in zip: {member.filename}")
            zf.extractall(dest_dir)

    elif archive.suffixes[-2:] == [".tar", ".gz"] or archive.suffix == ".tgz":
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                target = dest_dir / member.name
                if not _is_within_dir(dest_dir, target):
                    raise SystemExit(f"Unsafe path in tar: {member.name}")
            tf.extractall(dest_dir)

    else:
        # Unknown: just move the file into dest
        shutil.move(str(archive), str(dest_dir / archive.name))

    # Make probable binaries executable on POSIX
    if not IS_WINDOWS:
        for p in dest_dir.glob("**/*"):
            if p.is_file() and os.access(p, os.R_OK):
                try:
                    st = p.stat().st_mode
                    # Heuristic: give +x to files already having some execute bit or in /bin-like names
                    if "bin" in str(p.parent).lower() or (st & 0o111):
                        p.chmod(st | 0o111)
                except Exception as e:
                    LOG.debug("chmod warning for %s: %s", p, e)


def ensure_executable(path: Path) -> None:
    if not IS_WINDOWS:
        try:
            path.chmod(path.stat().st_mode | 0o111)
        except Exception as e:
            LOG.debug("chmod warning for %s: %s", path, e)


# -----------------------------------------------------------------------------
# Venv helpers
# -----------------------------------------------------------------------------

def get_venv_paths(venv_dir: Path) -> Tuple[Path, Path]:
    if IS_WINDOWS:
        python_exe = venv_dir / "Scripts" / "python.exe"
        pip_exe = venv_dir / "Scripts" / "pip.exe"
    else:
        python_exe = venv_dir / "bin" / "python"
        pip_exe = venv_dir / "bin" / "pip"
    return python_exe, pip_exe


# -----------------------------------------------------------------------------
# Release asset selection
# -----------------------------------------------------------------------------

def _has_token(name: str, token: str) -> bool:
    # true if token appears as a standalone chunk (separated by non-alnum)
    return re.search(rf'(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])', name) is not None

def asset_for_platform(release: dict, stem_match: str) -> Optional[dict]:
    """
    Find a release asset for current OS/arch containing stem_match in its name.
    Uses token/word-boundary matches so 'win' doesn't match 'darwin'.
    """
    system = platform.system().lower()        # 'windows' | 'linux' | 'darwin'
    machine = platform.machine().lower()      # 'amd64'/'x86_64' or 'arm64'/...

    arch_ok_tokens = {
        "x86_64", "amd64", "x64",
    } if machine in {"x86_64", "amd64"} else {
        "arm64", "aarch64"
    }

    def os_match(n: str) -> bool:
        if system == "windows":
            return _has_token(n, "windows") or _has_token(n, "win")
        if system == "darwin":
            return _has_token(n, "darwin") or _has_token(n, "macos") or _has_token(n, "mac")
        if system == "linux":
            return _has_token(n, "linux") or _has_token(n, "ubuntu")
        return False

    best: Optional[dict] = None
    best_score = 1_000

    for asset in release.get("assets", []):
        name = asset["name"].lower()
        if stem_match not in name:
            continue
        if not os_match(name):
            continue
        if not any(_has_token(name, tok) for tok in arch_ok_tokens):
            continue

        # prefer native archive types per-OS; lower score is better
        score = 10
        if system == "windows" and name.endswith(".zip"):
            score = 0
        elif system in {"linux", "darwin"} and (name.endswith(".tar.gz") or name.endswith(".tgz")):
            score = 0

        if score < best_score:
            best = asset
            best_score = score

    return best


def _os_token_match(n: str, sys_token: str) -> bool:
    # sys_token is 'win'|'macos'|'ubuntu'
    if sys_token == "win":
        return _has_token(n, "windows") or _has_token(n, "win")
    if sys_token == "macos":
        return _has_token(n, "macos") or _has_token(n, "darwin") or _has_token(n, "mac")
    if sys_token == "ubuntu":
        return _has_token(n, "ubuntu") or _has_token(n, "linux")
    return False

def asset_for_platform_cpp(release: Dict, gpu_backend_pref: str) -> Optional[Dict]:
    system_token = {"Darwin": "macos", "Windows": "win", "Linux": "ubuntu"}.get(
        platform.system(), platform.system().lower()
    )
    arch_token = {
        "amd64": "x64", "x86_64": "x64",
        "arm64": "arm64", "aarch64": "arm64",
    }.get(platform.machine().lower(), platform.machine().lower())

    candidates: list[tuple[int, Dict]] = []
    for asset in release.get("assets", []):
        name = asset["name"].lower()
        if not name.endswith(".zip"):
            continue

        # skip runtime/deps-only bundles (these have no server binary)
        if any(t in name for t in ("cudart", "runtime", "deps", "cudnn", "cutensor")):
            continue

        system_match = _os_token_match(name, system_token)
        if not (system_match and arch_token in name and "bin" in name):
            continue

        # prefer archives that clearly look like the main llama bin bundles
        looks_like_main = (name.startswith("llama-") or name.startswith("llama-b")) and "-bin-" in name

        score = 1000
        has = lambda kw: kw in name

        if gpu_backend_pref == "cuda":
            score = 0 if (has("cuda") and looks_like_main) else (1 if has("cuda") else 1000)
        elif gpu_backend_pref == "vulkan":
            score = 0 if (has("vulkan") and looks_like_main) else (1 if has("vulkan") else 1000)
        elif gpu_backend_pref == "cpu":
            if has("cpu") and looks_like_main: score = 0
            elif not any(has(x) for x in ("cuda", "vulkan", "metal")): score = 1
        elif gpu_backend_pref == "auto":
            if system_token == "macos" and has("metal") and looks_like_main: score = 0
            elif has("cuda") and looks_like_main: score = 10
            elif has("vulkan") and looks_like_main: score = 20
            elif has("cpu") and looks_like_main: score = 30
            elif not any(has(x) for x in ("metal", "cuda", "vulkan")): score = 40

        if score < 1000:
            candidates.append((score, asset))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]["name"]))
    return candidates[0][1]



# -----------------------------------------------------------------------------
# llama-swap obtain
# -----------------------------------------------------------------------------

def download_llama_swap(binary_dir: Path) -> Path:
    release = get_latest_release(LLAMA_SWAP_RELEASE_REPO)
    asset = asset_for_platform(release, "llama-swap")
    if not asset:
        sys.exit(
            f"No release asset for this platform in {LLAMA_SWAP_RELEASE_REPO}. "
            "Try --build-from-source swap."
        )

    # 1) Download to a TEMP folder (not inside binary_dir!)
    with tempfile.TemporaryDirectory() as td:
        tmp_archive = Path(td) / asset["name"]
        download(asset["browser_download_url"], tmp_archive)

        # 2) Clean target bin dir so we don't mix platforms/artifacts
        binary_dir.mkdir(parents=True, exist_ok=True)
        for p in list(binary_dir.iterdir()):
            try:
                if p.is_file() or p.is_symlink():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)
            except Exception as e:
                print(f"Warning: could not remove {p}: {e}")

        # 3) Extract into the clean bin dir
        extract_archive(tmp_archive, binary_dir)

    # 4) Locate and normalize the executable name on Windows
    exe_name = "llama-swap.exe" if IS_WINDOWS else "llama-swap"
    for cand in binary_dir.iterdir():
        if cand.is_file() and cand.name.lower().startswith("llama-swap"):
            target = binary_dir / exe_name
            if cand != target:
                try:
                    if target.exists():
                        target.unlink()
                    shutil.copy2(cand, target)
                    cand = target
                except Exception:
                    # best-effort fallback
                    cand.rename(target)
                    cand = target
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
        backup = bin_dir.with_suffix(".old_binary_backup")
        LOG.info("Found legacy llama-swap file, moving to %s", backup)
        try:
            bin_dir.rename(backup)
        except OSError as e:
            raise SystemExit(f"Failed to move old llama-swap binary: {e}")
    bin_dir.mkdir(parents=True, exist_ok=True)
    return build_llama_swap(src_dir, bin_dir) if build else download_llama_swap(bin_dir)


# -----------------------------------------------------------------------------
# llama.cpp obtain
# -----------------------------------------------------------------------------

def _copy_server_payload(src_dir: Path, target_bin_dir: Path, final_name: str) -> Path:
    """
    Copy server executable and its side files from src_dir into target_bin_dir.
    """
    target_bin_dir.mkdir(parents=True, exist_ok=True)
    possible = {"server", "llama-server"}
    if IS_WINDOWS:
        possible = {f"{n}.exe" for n in possible}

    server_src: Optional[Path] = None
    for p in src_dir.glob("*"):
        if p.is_file() and p.name.lower() in possible:
            server_src = p
            break
    if not server_src:
        raise SystemExit("Server executable not found in extracted content.")

    final_exe = target_bin_dir / final_name
    copied = 0
    for item in src_dir.iterdir():
        dest_name = final_name if item == server_src else item.name
        dest = target_bin_dir / dest_name
        if item.is_file():
            if dest.exists() and dest.resolve() != item.resolve():
                dest.unlink(missing_ok=True)
            shutil.copy2(item, dest)
            if dest == final_exe:
                ensure_executable(dest)
            copied += 1
    LOG.debug("Copied %d files into %s", copied, target_bin_dir)
    return final_exe


def download_llama_cpp(base_install_dir: Path, gpu_backend: str) -> Path:
    LOG.info("Preparing llama.cpp (download, backend=%s)", gpu_backend)
    llama_cpp_dir = base_install_dir / "llama.cpp"
    target_bin_dir = llama_cpp_dir / "build" / "bin"
    final_name = "llama-server.exe" if IS_WINDOWS else "llama-server"

    release = get_latest_release(LLAMA_CPP_RELEASE_REPO)
    asset = asset_for_platform_cpp(release, gpu_backend)
    if not asset:
        tag = release.get("tag_name", "latest")
        err = (
            f"No suitable llama.cpp asset for {platform.system()}/{platform.machine()} "
            f"(backend={gpu_backend}) in {LLAMA_CPP_RELEASE_REPO} release {tag}. "
        )
        err += "Try --gpu-backend auto or --build-from-source cpp."
        raise SystemExit(err)
    LOG.info("Selected llama.cpp asset: %s", asset["name"])

    tmp_archive = llama_cpp_dir / asset["name"]
    download(asset["browser_download_url"], tmp_archive)

    with tempfile.TemporaryDirectory() as td:
        temp_extract_dir = Path(td)
        extract_archive(tmp_archive, temp_extract_dir)
        tmp_archive.unlink(missing_ok=True)

        # find a directory containing the server exe
        exe_dir: Optional[Path] = None
        possible = {"server", "llama-server"}
        if IS_WINDOWS:
            possible = {f"{n}.exe" for n in possible}
        for p in temp_extract_dir.rglob("*"):
            if p.is_file() and p.name.lower() in possible:
                exe_dir = p.parent
                break
        if not exe_dir:
            raise SystemExit("Downloaded llama.cpp, but server executable not found.")

        final_exe = _copy_server_payload(exe_dir, target_bin_dir, final_name)
        return final_exe


def build_llama_cpp(src_dir: Path, gpu_backend: str) -> Path:
    LOG.info("Building llama.cpp from source (backend=%s)", gpu_backend)
    if src_dir.exists():
        run(["git", "pull"], cwd=src_dir)
    else:
        run(["git", "clone", "--depth", "1", LLAMA_CPP_REPO, str(src_dir)])

    build_dir = src_dir / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir()

    target_bin_dir = build_dir / "bin"
    target_bin_dir.mkdir(parents=True, exist_ok=True)

    final_name = "llama-server.exe" if IS_WINDOWS else "llama-server"

    cmake_flags = [
        f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={str(target_bin_dir)}",
        f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE={str(target_bin_dir)}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if gpu_backend == "cuda":
        cmake_flags.append("-DLLAMA_CUBLAS=ON")
    elif gpu_backend == "vulkan":
        cmake_flags.append("-DLLAMA_VULKAN=ON")
    elif gpu_backend == "cpu":
        cmake_flags += ["-DLLAMA_CUBLAS=OFF", "-DLLAMA_VULKAN=OFF", "-DGGML_METAL=OFF"]
    elif gpu_backend == "auto" and platform.system() == "Darwin":
        cmake_flags.append("-DGGML_METAL=ON")

    run(["cmake", "..", *cmake_flags], cwd=build_dir)
    jobs = str(os.cpu_count() or 4)
    run(["cmake", "--build", ".", "--config", "Release", "-j", jobs], cwd=build_dir)

    # locate the produced server
    for p in (target_bin_dir, target_bin_dir / "Release", build_dir, build_dir / "Release"):
        if IS_WINDOWS:
            cand = p / "llama-server.exe"
            cand_alt = p / "server.exe"
        else:
            cand = p / "llama-server"
            cand_alt = p / "server"
        for c in (cand, cand_alt):
            if c.is_file():
                ensure_executable(c)
                # normalize name/location
                final_exe = target_bin_dir / final_name
                if final_exe.exists() and final_exe != c:
                    final_exe.unlink(missing_ok=True)
                if c != final_exe:
                    shutil.copy2(c, final_exe)
                    ensure_executable(final_exe)
                return final_exe

    raise SystemExit("Build complete, but server executable was not found.")


def obtain_llama_cpp(base_install_dir: Path, build: bool, gpu_backend: str) -> Path:
    llama_cpp_root_dir = base_install_dir / "llama.cpp"
    if llama_cpp_root_dir.exists() and llama_cpp_root_dir.is_file():
        backup = llama_cpp_root_dir.with_suffix(".old_file_backup")
        try:
            llama_cpp_root_dir.rename(backup)
        except OSError as e:
            raise SystemExit(f"Cannot move conflicting file {llama_cpp_root_dir}: {e}")
    llama_cpp_root_dir.mkdir(parents=True, exist_ok=True)
    return build_llama_cpp(llama_cpp_root_dir, gpu_backend) if build else download_llama_cpp(base_install_dir, gpu_backend)


# -----------------------------------------------------------------------------
# Open WebUI container
# -----------------------------------------------------------------------------

def ensure_openwebui_container(webui_dir: Path, name: str, port: int, image: str) -> None:
    """
    Call the helper script utils/setup_openwebui.py which:
      - creates a container only if it doesn't exist,
      - mounts the provided data dir,
      - maps the chosen port,
      - and starts the container.
    """
    setup_script = Path(__file__).parent / "utils" / "setup_openwebui.py"
    if not setup_script.exists():
        LOG.warning("%s not found. Skipping Open WebUI container setup.", setup_script)
        return

    LOG.info("Ensuring Open WebUI container is present and running")
    LOG.info("  data dir   : %s", webui_dir)
    LOG.info("  name       : %s", name)
    LOG.info("  host port  : %d", port)
    LOG.info("  image      : %s", image)

    webui_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(setup_script),
        "--name", name,
        "--port", str(port),
        "--data-dir", str(webui_dir),
        "--image", image,
    ]
    try:
        run(cmd)
    except SystemExit:
        raise SystemExit("Failed to set up Open WebUI container. See logs above.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Install llama.cpp + llama-swap + Open WebUI helper files.")
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
    parser.add_argument(
        "--gpu-backend",
        choices=["auto", "cpu", "cuda", "vulkan"],
        default="auto",
        help="Preferred GPU backend for llama.cpp (download/build).",
    )
    parser.add_argument(
        "--webui-name",
        default="open-webui",
        help="Container name for Open WebUI (default: open-webui)",
    )
    parser.add_argument(
        "--webui-port",
        type=int,
        default=3000,
        help="Host port to expose Open WebUI on (default: 3000)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args()
    setup_logging(args.verbose)

    install_dir: Path = args.install_dir.expanduser().resolve()
    model_dir = install_dir / "models"
    webui_dir = install_dir / "open-webui-data"
    eval_dir = install_dir / "eval"
    venv_dir = install_dir / VENV_NAME

    build_swap = args.build_from_source in {"swap", "all"}
    build_cpp = args.build_from_source in {"cpp", "all"}
    gpu_backend_preference = args.gpu_backend

    LOG.info(
        dedent(
            f"""
            =============================================================================
            llama-suite installer
            Target directory  : {install_dir}
            Virtual Env       : {venv_dir}
            Build from source : swap={build_swap}  cpp={build_cpp}
            GPU Backend Pref  : {gpu_backend_preference}
            =============================================================================
            """
        ).strip()
    )

    # Ensure directories
    for d in (install_dir, model_dir, webui_dir, eval_dir, eval_dir / "results"):
        d.mkdir(parents=True, exist_ok=True)
    LOG.debug("Ensured base directories in %s", install_dir)

    # Virtual environment
    LOG.info("Setting up Python virtual environment: %s", venv_dir)
    if not venv_dir.exists():
        LOG.debug("Creating venv with Python: %s", sys.executable)
        venv.EnvBuilder(with_pip=True, upgrade_deps=True).create(str(venv_dir))
        LOG.info("Virtual environment created.")
    else:
        LOG.info("Virtual environment already exists; will ensure packages are present.")

    venv_python, venv_pip = get_venv_paths(venv_dir)
    if not venv_python.exists() or not venv_pip.exists():
        sys.exit(f"Venv created but Python/Pip not found: {venv_python}, {venv_pip}")

    LOG.info("Installing/updating Python packages into venv")
    for spec in REQUIRED_PACKAGES:
        run([str(venv_pip), "install", "--upgrade", spec])

    # llama-swap
    LOG.info("Installing llama-swap")
    swap_path = obtain_llama_swap(install_dir, build_swap)
    LOG.info("llama-swap binary: %s", swap_path)

    # llama.cpp server
    LOG.info("Installing llama.cpp server")
    cpp_server_path = obtain_llama_cpp(install_dir, build_cpp, gpu_backend_preference)
    LOG.info("llama-server binary: %s", cpp_server_path)

    # Open WebUI image (best-effort)
    LOG.info("Pulling Open WebUI image (best-effort)")
    try:
        run(["docker", "pull", OPEN_WEBUI_IMAGE])
    except SystemExit:
        LOG.warning("Docker pull failed. You can pull manually later.")

    # Ensure Open WebUI container exists and is running
    ensure_openwebui_container(
        webui_dir=webui_dir,
        name=args.webui_name,
        port=args.webui_port,
        image=OPEN_WEBUI_IMAGE,
    )

    # Summary
    summary = dedent(
        f"""
        -----------------------------------------------------------------------------
        Installation complete 🎉
        -----------------------------------------------------------------------------
        llama.cpp server : {cpp_server_path} (GPU backend: {gpu_backend_preference})
        llama-swap       : {swap_path}
        Models           : {model_dir}
        Open WebUI data  : {webui_dir}
        Open WebUI URL   : http://localhost:{args.webui_port}
        Container name   : {args.webui_name}
        Python Venv      : {venv_dir}

        NEXT STEPS
        ==========
        1) Activate the virtual environment:
           - Linux/macOS:  source "{venv_dir / 'bin' / 'activate'}"
           - Windows CMD:  CALL "{venv_dir / 'Scripts' / 'activate.bat'}"
           - Windows PS:   & "{venv_dir / 'Scripts' / 'Activate.ps1'}"

        2) Add GGUF models into:
              {model_dir}

        3) Configure your config.base.yaml in:
              {install_dir}

        4) Start llama-swap (with venv active):
              "{swap_path}" -config config.base.yaml

        5) Open Open WebUI in your browser:
              http://localhost:{args.webui_port}

           (Container '{args.webui_name}' was created if missing and started automatically.
            Use 'docker stop {args.webui_name}' / 'docker start {args.webui_name}' to manage it.)

        6) Run quick evals (with venv active):
              python "{eval_dir / 'evaluate-models.py'}" --tasks hellaswag --limit 10
           Results: {eval_dir / 'results'}
        -----------------------------------------------------------------------------
        """
    )
    print(summary)


if __name__ == "__main__":
    main()
