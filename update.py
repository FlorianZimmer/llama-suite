#!/usr/bin/env python3
"""
Update all components installed by the llama-suite installer to their newest versions.

What it updates
---------------
1) Python virtualenv packages (pip --upgrade for the known list)
2) llama-swap (rebuild from source if sources exist, else re-download latest release)
3) llama.cpp server (rebuild from source if repo exists, else re-download latest release)
4) Open WebUI container (pull latest image; recreate container using your helper)

Notes
-----
- Uses the same default install layout as the installer.
- Respects GITHUB_TOKEN env var for higher GitHub API limits.
- You can force methods (build/download) and set gpu backend via CLI.
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
from typing import Dict, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# -----------------------------------------------------------------------------
# Constants (keep in sync with installer)
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
OPEN_WEBUI_IMAGE_DEFAULT = "ghcr.io/open-webui/open-webui:main"

VENV_NAME = "llama-suite-venv"
REQUIRED_PACKAGES = [
    "lm-eval[llama_cpp,api]",
    "PyYAML",
    "requests",
    "psutil",
    "colorama",
    "watchdog~=4.0",
]

IS_WINDOWS = platform.system() == "Windows"

LOG = logging.getLogger("llama-suite-update")


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# -----------------------------------------------------------------------------
# Shell helpers
# -----------------------------------------------------------------------------

def run(cmd: Sequence[str] | str, cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    printable = " ".join(map(str, cmd)) if isinstance(cmd, (list, tuple)) else str(cmd)
    LOG.debug("exec: %s", printable)
    try:
        return subprocess.run(cmd, cwd=cwd, check=check, shell=isinstance(cmd, str))
    except subprocess.CalledProcessError as e:
        LOG.error("Command failed (%s): %s", e.returncode, printable)
        if check:
            sys.exit(1)
        return e


# -----------------------------------------------------------------------------
# GitHub API / downloads
# -----------------------------------------------------------------------------

def _gh_headers() -> dict:
    headers = {
        "User-Agent": "llama-suite-updater",
        "Accept": "application/vnd.github+json",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def get_latest_release(repo: str) -> Dict:
    api = f"https://api.github.com/repos/{repo}/releases/latest"
    try:
        with urlopen(Request(api, headers=_gh_headers())) as resp:
            return json.load(resp)
    except HTTPError as e:
        if e.code == 403:
            LOG.error("GitHub rate limit reached. Set GITHUB_TOKEN and retry.")
        else:
            LOG.error("GitHub API error (%s): %s", e.code, api)
        sys.exit(1)
    except URLError as e:
        LOG.error("GitHub API unreachable: %s", e)
        sys.exit(1)


def download(url: str, dest: Path) -> None:
    LOG.info("Downloading: %s", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url) as resp, open(dest, "wb") as fh:
            shutil.copyfileobj(resp, fh)
    except (HTTPError, URLError) as e:
        LOG.error("Download failed: %s", e)
        sys.exit(1)


# -----------------------------------------------------------------------------
# Safe extraction
# -----------------------------------------------------------------------------

def _safe_extract_zip(zf: zipfile.ZipFile, dest_dir: Path) -> None:
    for info in zf.infolist():
        target = dest_dir / info.filename
        if not str(target.resolve(strict=False)).startswith(str(dest_dir.resolve(strict=False))):
            raise SystemExit(f"Unsafe path in zip: {info.filename}")
    zf.extractall(dest_dir)


def _safe_extract_tar(tf: tarfile.TarFile, dest_dir: Path) -> None:
    for m in tf.getmembers():
        target = dest_dir / m.name
        if not str(target.resolve(strict=False)).startswith(str(dest_dir.resolve(strict=False))):
            raise SystemExit(f"Unsafe path in tar: {m.name}")
    tf.extractall(dest_dir)


def extract_archive(archive: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            _safe_extract_zip(zf, dest_dir)
    elif archive.suffixes[-2:] == [".tar", ".gz"] or archive.suffix == ".tgz":
        with tarfile.open(archive) as tf:
            _safe_extract_tar(tf, dest_dir)
    else:
        # Unknown: just move it in
        shutil.move(str(archive), str(dest_dir / archive.name))


def ensure_executable(path: Path) -> None:
    if not IS_WINDOWS and path.exists():
        try:
            path.chmod(path.stat().st_mode | 0o111)
        except Exception as e:
            LOG.debug("chmod warning: %s", e)


# -----------------------------------------------------------------------------
# Venv
# -----------------------------------------------------------------------------

def get_venv_paths(venv_dir: Path) -> Tuple[Path, Path]:
    if IS_WINDOWS:
        return venv_dir / "Scripts" / "python.exe", venv_dir / "Scripts" / "pip.exe"
    return venv_dir / "bin" / "python", venv_dir / "bin" / "pip"


def ensure_venv(venv_dir: Path) -> Tuple[Path, Path]:
    py, pip = get_venv_paths(venv_dir)
    if not venv_dir.exists():
        LOG.info("Virtualenv missing; creating: %s", venv_dir)
        import venv as _venv
        _venv.EnvBuilder(with_pip=True, upgrade_deps=True).create(str(venv_dir))
    if not py.exists() or not pip.exists():
        sys.exit(f"Venv broken: expected {py} and {pip}")
    return py, pip


def upgrade_packages(pip: Path) -> None:
    LOG.info("Upgrading Python packages in venv")
    for spec in REQUIRED_PACKAGES:
        run([str(pip), "install", "--upgrade", spec])


# -----------------------------------------------------------------------------
# Asset selection (same logic as installer, condensed)
# -----------------------------------------------------------------------------

def asset_for_platform_swap(release: Dict) -> Optional[Dict]:
    system = platform.system().lower()
    machine = platform.machine().lower().replace("x86_64", "amd64").replace("aarch64", "arm64")
    for asset in release.get("assets", []):
        name = asset["name"].lower()
        if "llama-swap" not in name:
            continue
        os_ok = (
            (system == "darwin" and ("darwin" in name or "macos" in name)) or
            (system == "windows" and ("windows" in name or "win" in name)) or
            (system == "linux" and ("linux" in name or "ubuntu" in name))
        )
        if os_ok and machine in name:
            return asset
    return None


def asset_for_platform_cpp(release: Dict, gpu_backend_pref: str) -> Optional[Dict]:
    sys_token = {"Darwin": "macos", "Windows": "win", "Linux": "ubuntu"}.get(
        platform.system(), platform.system().lower()
    )
    arch_token = {
        "amd64": "x64", "x86_64": "x64",
        "arm64": "arm64", "aarch64": "arm64",
    }.get(platform.machine().lower(), platform.machine().lower())

    cands: list[tuple[int, Dict]] = []
    for asset in release.get("assets", []):
        name = asset["name"].lower()
        if not (name.endswith(".zip") and "bin" in name):
            continue
        if not (sys_token in name or (sys_token == "ubuntu" and "linux" in name)):
            continue
        if arch_token not in name:
            continue

        has = lambda kw: kw in name
        full = ("llama-" in name and "-bin-" in name) or ("server" in name and "-bin-" in name and "llama" in name)
        score = 10_000
        if gpu_backend_pref == "cuda":
            score = 0 if (has("cuda") and full) else (1 if has("cuda") else 10_000)
        elif gpu_backend_pref == "vulkan":
            score = 0 if (has("vulkan") and full) else (1 if has("vulkan") else 10_000)
        elif gpu_backend_pref == "cpu":
            if has("cpu") and full:
                score = 0
            elif not any(has(x) for x in ("cuda", "vulkan", "metal")):
                score = 1
        elif gpu_backend_pref == "auto":
            if sys_token == "macos" and has("metal") and full:
                score = 0
            elif has("cuda") and full:
                score = 10
            elif has("vulkan") and full:
                score = 20
            elif has("cpu") and full:
                score = 30
            elif not any(has(x) for x in ("metal", "cuda", "vulkan")):
                score = 40

        if score < 10_000:
            cands.append((score, asset))

    if not cands:
        return None
    cands.sort(key=lambda t: (t[0], t[1]["name"]))
    return cands[0][1]


# -----------------------------------------------------------------------------
# Update steps
# -----------------------------------------------------------------------------

def update_llama_swap(base: Path, method: str) -> Path:
    """
    method: 'auto' | 'build' | 'download'
    """
    src_dir = base / "llama-swap-source"
    bin_dir = base / "llama-swap"
    exe_name = "llama-swap.exe" if IS_WINDOWS else "llama-swap"
    bin_dir.mkdir(parents=True, exist_ok=True)

    effective = method
    if method == "auto":
        effective = "build" if src_dir.exists() else "download"
    LOG.info("Updating llama-swap via: %s", effective)

    if effective == "build":
        if src_dir.exists():
            run(["git", "fetch", "--tags", "--force"], cwd=src_dir)
            run(["git", "pull"], cwd=src_dir)
        else:
            run(["git", "clone", LLAMA_SWAP_REPO, str(src_dir)])
        out = bin_dir / exe_name
        run(["go", "build", "-o", str(out), "."], cwd=src_dir)
        ensure_executable(out)
        return out

    # download path
    rel = get_latest_release(LLAMA_SWAP_RELEASE_REPO)
    asset = asset_for_platform_swap(rel)
    if not asset:
        sys.exit("No suitable llama-swap release asset for this platform.")
    tmp = bin_dir / asset["name"]
    download(asset["browser_download_url"], tmp)
    # clean out old files to avoid confusion
    for p in bin_dir.iterdir():
        if p.is_file():
            p.unlink()
    extract_archive(tmp, bin_dir)
    tmp.unlink(missing_ok=True)

    # find the binary
    for cand in bin_dir.iterdir():
        if cand.name.lower().startswith("llama-swap") and cand.is_file():
            ensure_executable(cand)
            return cand
    sys.exit("Updated llama-swap but could not locate the binary.")


def _copy_server_payload(src_dir: Path, target_bin_dir: Path, final_name: str) -> Path:
    target_bin_dir.mkdir(parents=True, exist_ok=True)
    possibles = {"server", "llama-server"}
    if IS_WINDOWS:
        possibles = {f"{n}.exe" for n in possibles}

    server_src: Optional[Path] = None
    for p in src_dir.iterdir():
        if p.is_file() and p.name.lower() in possibles:
            server_src = p
            break
    if not server_src:
        raise SystemExit("Server executable not found in extracted content.")

    final_exe = target_bin_dir / final_name
    # clear target dir to avoid stale DLLs mismatch
    for item in target_bin_dir.iterdir():
        if item.is_file():
            item.unlink(missing_ok=True)

    for item in src_dir.iterdir():
        dest_name = final_name if item == server_src else item.name
        dest = target_bin_dir / dest_name
        if item.is_file():
            shutil.copy2(item, dest)
            if dest_name == final_name:
                ensure_executable(dest)
    return final_exe


def update_llama_cpp(base: Path, method: str, gpu_backend: str) -> Path:
    """
    method: 'auto' | 'build' | 'download'
    """
    root = base / "llama.cpp"
    build_dir = root / "build"
    bin_dir = build_dir / "bin"
    final_name = "llama-server.exe" if IS_WINDOWS else "llama-server"

    # Decide method
    effective = method
    if method == "auto":
        # If root is a git repo → build, else download
        effective = "build" if (root / ".git").exists() else "download"
    LOG.info("Updating llama.cpp via: %s (backend=%s)", effective, gpu_backend)

    if effective == "build":
        if (root / ".git").exists():
            run(["git", "fetch", "--tags", "--force"], cwd=root)
            run(["git", "pull"], cwd=root)
        else:
            run(["git", "clone", "--depth", "1", LLAMA_CPP_REPO, str(root)])
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir()
        bin_dir.mkdir(parents=True, exist_ok=True)

        cmake_flags = [
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={str(bin_dir)}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE={str(bin_dir)}",
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

        # normalize name
        candidates = [
            bin_dir / final_name,
            bin_dir / ("server.exe" if IS_WINDOWS else "server"),
        ]
        for c in candidates:
            if c.is_file():
                if c.name != final_name:
                    final_exe = bin_dir / final_name
                    if final_exe.exists():
                        final_exe.unlink(missing_ok=True)
                    shutil.copy2(c, final_exe)
                    ensure_executable(final_exe)
                    return final_exe
                ensure_executable(c)
                return c

        raise SystemExit("Build finished, but server executable not found.")

    # download path
    release = get_latest_release(LLAMA_CPP_RELEASE_REPO)
    asset = asset_for_platform_cpp(release, gpu_backend)
    if not asset:
        tag = release.get("tag_name", "latest")
        sys.exit(f"No suitable llama.cpp asset for this platform (backend={gpu_backend}) in release {tag}.")

    tmp_archive = root / asset["name"]
    download(asset["browser_download_url"], tmp_archive)
    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        extract_archive(tmp_archive, temp_dir)
        tmp_archive.unlink(missing_ok=True)

        exe_dir: Optional[Path] = None
        possibles = {"server", "llama-server"}
        if IS_WINDOWS:
            possibles = {f"{n}.exe" for n in possibles}
        for p in temp_dir.rglob("*"):
            if p.is_file() and p.name.lower() in possibles:
                exe_dir = p.parent
                break
        if not exe_dir:
            raise SystemExit("Downloaded llama.cpp payload, but server not found.")

        final_exe = _copy_server_payload(exe_dir, bin_dir, final_name)
        return final_exe


def update_openwebui(install_dir: Path, name: str, port: int, image: str) -> None:
    """
    Pull latest image, stop & remove existing container, then recreate via helper.
    """
    webui_dir = install_dir / "open-webui-data"
    webui_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Pulling latest Open WebUI image: %s", image)
    run(["docker", "pull", image], check=False)

    LOG.info("Recreating container: %s", name)
    run(["docker", "stop", name], check=False)
    run(["docker", "rm", name], check=False)

    setup_script = Path(__file__).parent / "utils" / "setup_openwebui.py"
    if not setup_script.exists():
        LOG.warning("Helper %s not found; skipping container recreate.", setup_script)
        return

    cmd = [
        sys.executable, str(setup_script),
        "--name", name,
        "--port", str(port),
        "--data-dir", str(webui_dir),
        "--image", image,
    ]
    run(cmd)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Update llama-suite components to newest versions.")
    p.add_argument("--install-dir", type=Path, default=DEFAULT_INSTALL_BASE,
                   help=f"Install base directory (default: {DEFAULT_INSTALL_BASE})")
    p.add_argument("--gpu-backend", choices=["auto", "cpu", "cuda", "vulkan"], default="auto",
                   help="Preferred GPU backend for llama.cpp downloads/builds.")
    p.add_argument("--swap-method", choices=["auto", "build", "download"], default="auto",
                   help="How to update llama-swap.")
    p.add_argument("--cpp-method", choices=["auto", "build", "download"], default="auto",
                   help="How to update llama.cpp server.")
    p.add_argument("--webui-name", default="open-webui",
                   help="Open WebUI container name.")
    p.add_argument("--webui-port", type=int, default=3000,
                   help="Open WebUI host port.")
    p.add_argument("--webui-image", default=OPEN_WEBUI_IMAGE_DEFAULT,
                   help=f"Open WebUI image (default: {OPEN_WEBUI_IMAGE_DEFAULT})")
    p.add_argument("--skip", choices=["none", "venv", "swap", "cpp", "webui"],
                   default="none", help="Skip updating a component.")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = p.parse_args()
    setup_logging(args.verbose)

    install_dir = args.install_dir.expanduser().resolve()
    venv_dir = install_dir / VENV_NAME

    LOG.info("=== llama-suite updater ===")
    LOG.info("Install dir : %s", install_dir)

    # Venv
    if args.skip != "venv":
        py, pip = ensure_venv(venv_dir)
        upgrade_packages(pip)
    else:
        LOG.info("Skipping venv update")

    # llama-swap
    if args.skip != "swap":
        swap_path = update_llama_swap(install_dir, args.swap_method)
        LOG.info("llama-swap updated: %s", swap_path)
    else:
        LOG.info("Skipping llama-swap update")

    # llama.cpp
    if args.skip != "cpp":
        cpp_server = update_llama_cpp(install_dir, args.cpp_method, args.gpu_backend)
        LOG.info("llama.cpp server updated: %s", cpp_server)
    else:
        LOG.info("Skipping llama.cpp update")

    # Open WebUI
    if args.skip != "webui":
        update_openwebui(install_dir, args.webui_name, args.webui_port, args.webui_image)
        LOG.info("Open WebUI container refreshed: %s (port %d)", args.webui_name, args.webui_port)
    else:
        LOG.info("Skipping Open WebUI update")

    LOG.info("All done ✅")


if __name__ == "__main__":
    main()
