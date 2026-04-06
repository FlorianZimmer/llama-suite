#!/usr/bin/env python3
"""
Update all external components used by llama-suite (pyproject-first layout).

Updates:
  1) Python virtualenv deps (pip install -U -e . with eager strategy)
  2) llama-swap (build if vendor/llama-swap-source exists, else download latest)
  3) llama.cpp server (build if vendor/llama.cpp/source exists, else download latest)
  4) Open WebUI container (pull image, recreate via module helper; keeps data)

Requires:
  - Git (for builds)
  - Go (for llama-swap builds)
  - CMake + toolchain (for llama.cpp builds)
  - Docker/Podman/nerdctl for container ops
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LOG = logging.getLogger("llama-suite-update")
IS_WINDOWS = platform.system() == "Windows"

# Upstream repos
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp.git"
IK_LLAMA_CPP_REPO = "https://github.com/ikawrakow/ik_llama.cpp.git"
LLAMA_SWAP_REPO = "https://github.com/mostlygeek/llama-swap.git"
LLAMA_CPP_RELEASE_REPO = "ggml-org/llama.cpp"
LLAMA_SWAP_RELEASE_REPO = "mostlygeek/llama-swap"
OPEN_WEBUI_RELEASE_REPO = "open-webui/open-webui"

OPEN_WEBUI_IMAGE_DEFAULT = "ghcr.io/open-webui/open-webui:main"
OPEN_WEBUI_IMAGE_DOCKER_HUB_DEFAULT = "openwebui/open-webui:latest"
OPEN_WEBUI_ORBSTACK_DATA_VOLUME_DEFAULT = "open-webui_open-webui"

RUNTIMES_ORDER = ("docker", "podman", "nerdctl")


# ────────────────────────── logging / shell helpers ──────────────────────────

def setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(levelname)s: %(message)s",
                        stream=sys.stdout)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def emit_step(i: int, n: int, message: str) -> None:
    # ASCII-only, stable format (parsed by Web UI for progress).
    print(f"STEP {i}/{n}: {message}", flush=True)


def run(cmd: Sequence[str] | str, cwd: Optional[Path] = None, check: bool = True) -> None:
    """Run a command; raise on non-zero exit iff check=True. No return value."""
    printable = " ".join(map(str, cmd)) if isinstance(cmd, (list, tuple)) else str(cmd)
    LOG.debug("exec: %s", printable)
    try:
        subprocess.run(cmd, cwd=cwd, check=check, shell=isinstance(cmd, str))
    except subprocess.CalledProcessError as e:
        LOG.error("Command failed (%s): %s", e.returncode, printable)
        if check:
            # propagate so the caller (or top-level) fails with non-zero exit
            raise
        # If check=False, subprocess.run wouldn't raise; this is just defensive.


def run_capture(
    cmd: Sequence[str] | str,
    cwd: Optional[Path] = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a command and capture stdout/stderr."""
    printable = " ".join(map(str, cmd)) if isinstance(cmd, (list, tuple)) else str(cmd)
    LOG.debug("exec(capture): %s", printable)
    try:
        return subprocess.run(
            cmd,
            cwd=cwd,
            check=check,
            shell=isinstance(cmd, str),
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        LOG.error("Command failed (%s): %s", e.returncode, printable)
        if check:
            raise
        return e



# ───────────────────────────── repo discovery ────────────────────────────────

def find_repo_root(start: Path | None = None) -> Path:
    cur = (start or Path.cwd()).resolve()
    for p in [cur, *cur.parents]:
        if (p / "pyproject.toml").exists():
            return p
    sys.exit("Could not find repo root (no pyproject.toml). Run from inside the repo.")


# ───────────────────────────── venv utilities ────────────────────────────────

def venv_paths(venv_dir: Path) -> Tuple[Path, Path]:
    if IS_WINDOWS:
        return venv_dir / "Scripts" / "python.exe", venv_dir / "Scripts" / "pip.exe"
    return venv_dir / "bin" / "python", venv_dir / "bin" / "pip"


def _venv_python_looks_usable(venv_dir: Path, venv_python: Path) -> bool:
    """
    Return True if `venv_python` is runnable and reports a sys.prefix equal to `venv_dir`.

    This catches common macOS/Homebrew breakage where a venv python points at a removed
    Python.framework path after a brew upgrade, and also catches moved venv directories.
    """
    if not venv_python.exists():
        return False

    try:
        proc = subprocess.run(
            [str(venv_python), "-c", "import json, sys; print(json.dumps(sys.prefix))"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as e:
        LOG.warning("Venv python is not runnable (%s): %s", venv_python, e)
        return False

    # Be tolerant of extra output; prefer the last non-empty line.
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        return False

    try:
        prefix = Path(json.loads(lines[-1])).resolve()
    except Exception:
        return False

    return prefix == venv_dir.resolve()


def _move_aside(dir_path: Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = dir_path.with_name(f"{dir_path.name}.broken-{stamp}")
    candidate = base
    i = 1
    while candidate.exists():
        candidate = dir_path.with_name(f"{dir_path.name}.broken-{stamp}-{i}")
        i += 1
    shutil.move(str(dir_path), str(candidate))
    return candidate


def ensure_venv(repo: Path, venv: Optional[Path]) -> Tuple[Path, Path, Path]:
    venv_dir = (venv or (repo / ".venv")).resolve()
    py, pip = venv_paths(venv_dir)
    import venv as _venv

    if not venv_dir.exists():
        LOG.info("Virtualenv missing; creating at %s", venv_dir)
        _venv.EnvBuilder(with_pip=True, upgrade_deps=True).create(str(venv_dir))
    elif not py.exists() or not pip.exists() or not _venv_python_looks_usable(venv_dir, py):
        LOG.warning("Virtualenv seems broken/stale at %s", venv_dir)
        backup = _move_aside(venv_dir)
        LOG.info("Moved broken venv aside to %s", backup)
        LOG.info("Recreating virtualenv at %s", venv_dir)
        _venv.EnvBuilder(with_pip=True, upgrade_deps=True).create(str(venv_dir))

    if not py.exists() or not pip.exists():
        sys.exit(f"Venv seems broken: expected {py} and {pip}")
    return venv_dir, py, pip


def upgrade_python_deps(repo: Path, venv_python: Path, dev_extras: bool) -> None:
    LOG.info("Upgrading pip/setuptools/wheel/build")
    run([
        str(venv_python),
        "-m",
        "pip",
        "--disable-pip-version-check",
        "--no-input",
        "-q",
        "install",
        "-U",
        "pip",
        "setuptools",
        "wheel",
        "build",
    ])

    target = ".[dev]" if dev_extras else "."
    LOG.info("Upgrading project deps from pyproject.toml with eager strategy: %s", target)
    run([
        str(venv_python),
        "-m",
        "pip",
        "--disable-pip-version-check",
        "--no-input",
        "-q",
        "install",
        "-U",
        "--upgrade-strategy",
        "eager",
        "-e",
        target,
    ], cwd=repo)


# ───────────────────────── GitHub API / downloads ────────────────────────────

def _gh_headers() -> dict:
    headers = {
        "User-Agent": "llama-suite-updater",
        "Accept": "application/vnd.github+json",
    }
    tok = os.getenv("GITHUB_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    return headers


def get_latest_release(repo: str) -> Dict:
    api = f"https://api.github.com/repos/{repo}/releases/latest"
    try:
        with urlopen(Request(api, headers=_gh_headers())) as resp:
            return json.load(resp)
    except HTTPError as e:
        if e.code == 403:
            LOG.error("GitHub rate limit. Set GITHUB_TOKEN and retry.")
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


# ───────────────────────────── safe extraction ───────────────────────────────

def _safe_within(base: Path, target: Path) -> bool:
    return str(target.resolve(strict=False)).startswith(str(base.resolve(strict=False)))


def extract_archive(archive: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if not _safe_within(dest_dir, dest_dir / info.filename):
                    raise SystemExit(f"Unsafe zip path: {info.filename}")
            zf.extractall(dest_dir)
    elif archive.suffixes[-2:] == [".tar", ".gz"] or archive.suffix == ".tgz":
        with tarfile.open(archive) as tf:
            for m in tf.getmembers():
                if not _safe_within(dest_dir, dest_dir / m.name):
                    raise SystemExit(f"Unsafe tar path: {m.name}")
            tf.extractall(dest_dir)
    else:
        shutil.move(str(archive), str(dest_dir / archive.name))


def ensure_executable(path: Path) -> None:
    if not IS_WINDOWS and path.exists():
        try:
            path.chmod(path.stat().st_mode | 0o111)
        except Exception:
            pass


# ───────────────────────── asset selection helpers ───────────────────────────

def _has_token(name: str, token: str) -> bool:
    return re.search(rf'(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])', name.lower()) is not None


def _os_token_match(asset_name: str, system: str) -> bool:
    """Accept common OS synonyms in the selector."""
    n = asset_name.lower()
    s = system.lower()
    if s in ("windows", "win"):
        return _has_token(n, "windows") or _has_token(n, "win")
    if s in ("darwin", "macos", "mac"):
        return _has_token(n, "darwin") or _has_token(n, "macos") or _has_token(n, "mac")
    if s in ("linux", "ubuntu"):
        return _has_token(n, "linux") or _has_token(n, "ubuntu")
    return False


def asset_for_platform_swap(release: Dict) -> Optional[Dict]:
    system = platform.system().lower()
    machine = platform.machine().lower()
    arch_tokens = {"x86_64", "amd64", "x64"} if machine in {"x86_64", "amd64"} else {"arm64", "aarch64"}

    best, best_score = None, 10_000
    for asset in release.get("assets", []):
        n = asset["name"].lower()
        if "llama-swap" not in n: continue
        if not _os_token_match(n, system): continue
        if not any(_has_token(n, t) for t in arch_tokens): continue

        score = 10
        if system == "windows":
            score = 0 if n.endswith(".zip") else 2
        else:
            score = 0 if (n.endswith(".tar.gz") or n.endswith(".tgz")) else (2 if n.endswith(".zip") else 5)

        if score < best_score or (score == best_score and (not best or n < best["name"].lower())):
            best, best_score = asset, score
    return best


def asset_for_platform_cpp(release: Dict, gpu_backend_pref: str) -> Optional[Dict]:
    system_token = {"Darwin": "macos", "Windows": "win", "Linux": "ubuntu"}.get(
        platform.system(), platform.system().lower()
    )
    arch_token = {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(
        platform.machine().lower(), platform.machine().lower()
    )

    candidates: list[tuple[int, Dict]] = []
    for asset in release.get("assets", []):
        n = asset["name"].lower()
        if not n.endswith(".zip"): continue
        if any(t in n for t in ("cudart", "runtime", "deps", "cudnn", "cutensor")): continue
        if not (_os_token_match(n, system_token) and arch_token in n and "bin" in n): continue

        looks_main = (n.startswith("llama-") or n.startswith("llama-b")) and "-bin-" in n
        score = 1000
        has = lambda s: s in n

        if gpu_backend_pref == "cuda":
            score = 0 if (has("cuda") and looks_main) else (1 if has("cuda") else 1000)
        elif gpu_backend_pref == "vulkan":
            score = 0 if (has("vulkan") and looks_main) else (1 if has("vulkan") else 1000)
        elif gpu_backend_pref == "cpu":
            if has("cpu") and looks_main: score = 0
            elif not any(has(x) for x in ("cuda", "vulkan", "metal")): score = 1
        elif gpu_backend_pref == "auto":
            if system_token == "macos" and has("metal") and looks_main: score = 0
            elif has("cuda") and looks_main: score = 10
            elif has("vulkan") and looks_main: score = 20
            elif has("cpu") and looks_main: score = 30
            elif not any(has(x) for x in ("metal", "cuda", "vulkan")): score = 40

        if score < 1000:
            candidates.append((score, asset))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]["name"]))
    return candidates[0][1]

def asset_for_cpp_cudart(release: Dict) -> Optional[Dict]:
    """
    Pick the Windows CUDA runtime bundle shipped alongside llama.cpp releases,
    e.g. 'cudart-llama-bin-win-cuda-12.4-x64.zip'.
    Prefer highest CUDA 12.* version if multiple are present.
    """
    if platform.system() != "Windows":
        return None

    def is_cudart_asset(name: str) -> bool:
        n = name.lower()
        return (
            n.startswith("cudart-llama-bin-win-cuda-")
            and n.endswith("-x64.zip")
            and "12." in n  # constrain to CUDA 12.* which ggml-cuda expects today
        )

    picks = []
    for asset in release.get("assets", []):
        n = asset["name"]
        if is_cudart_asset(n):
            # Extract the '12.X' part for sorting (fallback to 12.0 if parse fails)
            m = re.search(r"cuda-(12\.\d+)", n.lower())
            ver = tuple(map(int, (m.group(1).split("."))) ) if m else (12, 0)
            picks.append((ver, asset))
    if not picks:
        return None
    # Prefer the highest 12.X
    picks.sort(key=lambda x: x[0], reverse=True)
    return picks[0][1]


# ─────────────────────────── llama-swap updater ──────────────────────────────

def update_llama_swap(vendor_dir: Path, method: str) -> Path:
    """
    method: 'auto' | 'build' | 'download'
    """
    src_dir = vendor_dir / "llama-swap-source"
    bin_dir = vendor_dir / "llama-swap"
    exe_name = "llama-swap.exe" if IS_WINDOWS else "llama-swap"
    bin_dir.mkdir(parents=True, exist_ok=True)

    effective = "build" if (method == "auto" and src_dir.exists()) else ("download" if method == "auto" else method)
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

    with tempfile.TemporaryDirectory() as td:
        tmp_archive = Path(td) / asset["name"]
        download(asset["browser_download_url"], tmp_archive)

        # Clean only after successful download
        for p in list(bin_dir.iterdir()):
            try:
                p.unlink() if p.is_file() or p.is_symlink() else shutil.rmtree(p)
            except Exception:
                pass
        extract_archive(tmp_archive, bin_dir)

    # normalize exe name
    for cand in bin_dir.iterdir():
        if cand.is_file() and cand.name.lower().startswith("llama-swap"):
            target = bin_dir / exe_name
            if cand != target:
                if target.exists():
                    target.unlink(missing_ok=True)
                shutil.copy2(cand, target)
                cand = target
            ensure_executable(cand)
            return cand
    sys.exit("Updated llama-swap but could not locate the binary.")


# ─────────────────────────── llama.cpp updater ───────────────────────────────

def _copy_server_payload(src_dir: Path, target_bin_dir: Path, final_name: str) -> Path:
    target_bin_dir.mkdir(parents=True, exist_ok=True)
    # Clear out old DLLs/binaries to avoid mismatches
    for item in list(target_bin_dir.iterdir()):
        try:
            item.unlink() if item.is_file() else shutil.rmtree(item)
        except Exception:
            pass

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
    for item in src_dir.iterdir():
        dest = target_bin_dir / (final_name if item == server_src else item.name)
        if item.is_file():
            shutil.copy2(item, dest)
            if dest.name == final_name:
                ensure_executable(dest)
    return final_exe


def _macos_fix_rpath_to_executable_dir(exe: Path) -> None:
    if platform.system() != "Darwin":
        return

    install_name_tool = shutil.which("install_name_tool")
    if not install_name_tool:
        LOG.warning("install_name_tool not found; cannot adjust rpath for %s", exe)
        return

    try:
        proc = subprocess.run(["otool", "-l", str(exe)], check=True, capture_output=True, text=True)
        lines = proc.stdout.splitlines()
    except Exception as e:
        LOG.warning("Failed to inspect rpaths for %s: %s", exe, e)
        return

    rpaths: list[str] = []
    for i, line in enumerate(lines):
        if "cmd LC_RPATH" not in line:
            continue
        for j in range(i + 1, min(i + 8, len(lines))):
            m = re.match(r"\s*path\s+(.+?)\s+\(offset\s+\d+\)\s*$", lines[j])
            if m:
                rpaths.append(m.group(1))
                break

    desired = "@executable_path"
    for rp in rpaths:
        # Delete absolute build paths (common after moving the repo or rebuilding).
        if rp.startswith("/") and "llama-suite" in rp:
            run([install_name_tool, "-delete_rpath", rp, str(exe)], check=False)

    if desired not in rpaths:
        run([install_name_tool, "-add_rpath", desired, str(exe)], check=False)


def _copy_macos_dylibs_next_to_exe(build_dir: Path, bin_dir: Path) -> None:
    if platform.system() != "Darwin":
        return

    dylibs = list(bin_dir.glob("*.dylib"))
    if dylibs:
        return

    # When building llama.cpp from source, the dylibs often land in build/bin while
    # executables go elsewhere unless CMAKE_LIBRARY_OUTPUT_DIRECTORY is set.
    candidates = list((build_dir / "bin").glob("*.dylib"))
    if not candidates:
        candidates = list(build_dir.rglob("*.dylib"))

    if not candidates:
        LOG.warning("No .dylib files found under %s; llama-server may not be runnable", build_dir)
        return

    for lib in candidates:
        dest = bin_dir / lib.name
        if not dest.exists():
            shutil.copy2(lib, dest)
            ensure_executable(dest)


def update_llama_cpp(vendor_dir: Path, method: str, gpu_backend: str) -> Path:
    """
    method: 'auto' | 'build' | 'download'
    """
    src_root = vendor_dir / "llama.cpp" / "source"
    bin_dir  = vendor_dir / "llama.cpp" / "bin"
    final_name = "llama-server.exe" if IS_WINDOWS else "llama-server"

    effective = method
    if method == "auto":
        effective = "build" if (src_root / ".git").exists() else "download"
    LOG.info("Updating llama.cpp via: %s (backend=%s)", effective, gpu_backend)

    if effective == "build":
        if (src_root / ".git").exists():
            run(["git", "fetch", "--tags", "--force"], cwd=src_root)
            run(["git", "pull"], cwd=src_root)
        else:
            src_root.parent.mkdir(parents=True, exist_ok=True)
            run(["git", "clone", "--depth", "1", LLAMA_CPP_REPO, str(src_root)])

        build_dir = src_root / "build"
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir()
        bin_dir.mkdir(parents=True, exist_ok=True)

        cmake_flags = [
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={str(bin_dir)}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE={str(bin_dir)}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={str(bin_dir)}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={str(bin_dir)}",
            f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={str(bin_dir)}",
            f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE={str(bin_dir)}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        if IS_WINDOWS:
            # Avoid requiring a separate OpenSSL dev install for local Windows builds.
            cmake_flags.append("-DLLAMA_OPENSSL=OFF")
        if gpu_backend == "cuda":
            cmake_flags.append("-DGGML_CUDA=ON")
        elif gpu_backend == "vulkan":
            cmake_flags.append("-DGGML_VULKAN=ON")
        elif gpu_backend == "cpu":
            cmake_flags += ["-DGGML_CUDA=OFF", "-DGGML_VULKAN=OFF", "-DGGML_METAL=OFF"]
        elif gpu_backend == "auto" and platform.system() == "Darwin":
            cmake_flags.append("-DGGML_METAL=ON")

        run(["cmake", "..", *cmake_flags], cwd=build_dir)
        jobs = str(os.cpu_count() or 4)
        run(["cmake", "--build", ".", "--config", "Release", "--target", "llama-server", "-j", jobs], cwd=build_dir)

        _copy_macos_dylibs_next_to_exe(build_dir, bin_dir)

        # normalize name
        candidates = [bin_dir / final_name, bin_dir / ("server.exe" if IS_WINDOWS else "server")]
        for c in candidates:
            if c.is_file():
                if c.name != final_name:
                    final_exe = bin_dir / final_name
                    if final_exe.exists():
                        final_exe.unlink(missing_ok=True)
                    shutil.copy2(c, final_exe)
                    ensure_executable(final_exe)
                    _macos_fix_rpath_to_executable_dir(final_exe)
                    return final_exe
                ensure_executable(c)
                _macos_fix_rpath_to_executable_dir(c)
                return c
        raise SystemExit("Build complete, but server executable not found.")

    # download route
    release = get_latest_release(LLAMA_CPP_RELEASE_REPO)
    asset = asset_for_platform_cpp(release, gpu_backend)
    if not asset:
        tag = release.get("tag_name", "latest")
        sys.exit(f"No suitable llama.cpp asset for {platform.system()}/{platform.machine()} (backend={gpu_backend}) in release {tag}.")

    tmp_dir = vendor_dir / "llama.cpp" / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_archive = tmp_dir / asset["name"]
    download(asset["browser_download_url"], tmp_archive)

    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        extract_archive(tmp_archive, temp_dir)
        tmp_archive.unlink(missing_ok=True)

        exe_dir: Optional[Path] = None
        possibles = {"server", "llama-server"}
        if IS_WINDOWS: possibles = {f"{n}.exe" for n in possibles}
        for p in temp_dir.rglob("*"):
            if p.is_file() and p.name.lower() in possibles:
                exe_dir = p.parent
                break
        if not exe_dir:
            raise SystemExit("Downloaded llama.cpp payload, but server not found.")

        final_exe = _copy_server_payload(exe_dir, bin_dir, final_name)
        # On Windows + CUDA preference: also fetch cudart runtime bundle and drop DLLs next to the server.
        if IS_WINDOWS and gpu_backend in {"cuda", "auto"}:
            cudart_asset = asset_for_cpp_cudart(release)
            if cudart_asset:
                LOG.info("Fetching CUDA runtime bundle: %s", cudart_asset["name"])
                cudart_archive = tmp_dir / cudart_asset["name"]
                download(cudart_asset["browser_download_url"], cudart_archive)
                # Extract directly into the bin dir so DLLs sit next to llama-server.exe
                extract_archive(cudart_archive, bin_dir)
                cudart_archive.unlink(missing_ok=True)
            else:
                LOG.warning("CUDA runtime bundle not found in this release; assuming system CUDA 12.x is present.")
        return final_exe


def update_ik_llama_cpp(vendor_dir: Path, method: str, gpu_backend: str) -> Path:
    """
    method: 'auto' | 'build'
    """
    src_root = vendor_dir / "ik_llama.cpp" / "source"
    bin_dir = vendor_dir / "ik_llama.cpp" / "bin"
    final_name = "llama-server.exe" if IS_WINDOWS else "llama-server"

    effective = "build" if method == "auto" else method
    LOG.info("Updating ik_llama.cpp via: %s (backend=%s)", effective, gpu_backend)

    if effective != "build":
        raise SystemExit("ik_llama.cpp currently supports source builds only.")

    if (src_root / ".git").exists():
        run(["git", "fetch", "--tags", "--force"], cwd=src_root)
        run(["git", "pull"], cwd=src_root)
    else:
        src_root.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", IK_LLAMA_CPP_REPO, str(src_root)])

    build_dir = src_root / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir()
    bin_dir.mkdir(parents=True, exist_ok=True)

    cmake_flags = [
        f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={str(bin_dir)}",
        f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE={str(bin_dir)}",
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={str(bin_dir)}",
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={str(bin_dir)}",
        f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={str(bin_dir)}",
        f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE={str(bin_dir)}",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGGML_NATIVE=ON",
    ]
    if IS_WINDOWS:
        cmake_flags.append("-DLLAMA_OPENSSL=OFF")
    if gpu_backend == "cuda":
        cmake_flags.append("-DGGML_CUDA=ON")
    elif gpu_backend == "vulkan":
        cmake_flags.append("-DGGML_VULKAN=ON")
    elif gpu_backend == "cpu":
        cmake_flags += ["-DGGML_CUDA=OFF", "-DGGML_VULKAN=OFF", "-DGGML_METAL=OFF"]
    elif gpu_backend == "auto" and platform.system() == "Darwin":
        cmake_flags.append("-DGGML_METAL=ON")

    run(["cmake", "..", *cmake_flags], cwd=build_dir)
    jobs = str(os.cpu_count() or 4)
    run(["cmake", "--build", ".", "--config", "Release", "--target", "llama-server", "-j", jobs], cwd=build_dir)

    candidates = [bin_dir / final_name, bin_dir / ("server.exe" if IS_WINDOWS else "server")]
    for c in candidates:
        if c.is_file():
            if c.name != final_name:
                final_exe = bin_dir / final_name
                if final_exe.exists():
                    final_exe.unlink(missing_ok=True)
                shutil.copy2(c, final_exe)
                ensure_executable(final_exe)
                _macos_fix_rpath_to_executable_dir(final_exe)
                return final_exe
            ensure_executable(c)
            _macos_fix_rpath_to_executable_dir(c)
            return c
    raise SystemExit("ik_llama.cpp build complete, but server executable was not found.")


# ───────────────────────── Open WebUI refresh (container) ────────────────────

def detect_runtime(explicit: Optional[str] = None) -> Tuple[str, str] | None:
    if explicit:
        exe = shutil.which(explicit)
        return (explicit, exe) if exe else None
    for name in RUNTIMES_ORDER:
        exe = shutil.which(name)
        if exe:
            return name, exe
    return None


def runtime_available(rt_name: str, rt_path: str) -> Tuple[bool, str]:
    try:
        cp = subprocess.run(
            [rt_path, "info"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as e:
        return False, str(e)

    if cp.returncode == 0:
        return True, ""

    detail = (cp.stderr or cp.stdout or "").strip()
    if not detail:
        detail = f"'{rt_name} info' exited with status {cp.returncode}"
    return False, detail


def container_inspect(rt_path: str, name: str) -> Optional[Dict[str, object]]:
    cp = subprocess.run(
        [rt_path, "container", "inspect", name],
        check=False,
        capture_output=True,
        text=True,
    )
    if cp.returncode != 0:
        return None

    try:
        payload = json.loads(cp.stdout)
    except json.JSONDecodeError:
        return None

    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0]
    return None


def volume_exists(rt_path: str, name: str) -> bool:
    cp = subprocess.run(
        [rt_path, "volume", "inspect", name],
        check=False,
        capture_output=True,
        text=True,
    )
    return cp.returncode == 0


def split_image_reference(image: str) -> Tuple[str, Optional[str], Optional[str]]:
    if "@" in image:
        repo, digest = image.split("@", 1)
        return repo, None, digest

    last_slash = image.rfind("/")
    last_colon = image.rfind(":")
    if last_colon > last_slash:
        return image[:last_colon], image[last_colon + 1 :], None
    return image, None, None


def inspect_image_repo_digests(rt_path: str, image: str) -> list[str]:
    cp = run_capture([rt_path, "image", "inspect", image], check=True)
    try:
        payload = json.loads(cp.stdout or "[]")
    except json.JSONDecodeError as e:
        raise SystemExit(f"Could not parse image inspect output for {image}: {e}") from e

    if isinstance(payload, list) and payload:
        payload0 = payload[0]
    elif isinstance(payload, dict):
        payload0 = payload
    else:
        payload0 = {}

    digests = payload0.get("RepoDigests", [])
    if not isinstance(digests, list):
        return []
    return [str(item) for item in digests if isinstance(item, str)]


def resolve_pulled_image_reference(rt_path: str, image: str) -> str:
    repo, _tag, digest = split_image_reference(image)
    if digest:
        return image

    repo_digests = inspect_image_repo_digests(rt_path, image)
    for candidate in repo_digests:
        cand_repo, _cand_tag, cand_digest = split_image_reference(candidate)
        if cand_digest and cand_repo == repo:
            return candidate

    if repo_digests:
        LOG.warning("Could not find a repo-matching digest for %s; using first available digest %s", image, repo_digests[0])
        return repo_digests[0]

    raise SystemExit(f"Could not resolve a pulled image digest for {image}.")


def update_compose_openwebui_image(compose_path: Path, image: str) -> bool:
    if not compose_path.exists():
        LOG.warning("Compose file not found; skipping Open WebUI image patch: %s", compose_path)
        return False

    text = compose_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    in_openwebui = False
    service_indent: Optional[int] = None

    for idx, line in enumerate(lines):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))

        if stripped == "open-webui:":
            in_openwebui = True
            service_indent = indent
            continue

        if not in_openwebui:
            continue

        if stripped and indent <= (service_indent or 0) and not line.lstrip(" ").startswith("#"):
            break

        if stripped.startswith("image:"):
            prefix = line[: len(line) - len(line.lstrip(" "))]
            current = stripped[len("image:") :].strip()
            new_line = f"{prefix}image: {image}"
            if line == new_line:
                return False
            LOG.info("Patching Compose Open WebUI image: %s -> %s", current, image)
            lines[idx] = new_line
            compose_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return True

    raise SystemExit(f"Could not locate open-webui image entry in {compose_path}")


def latest_openwebui_release_version() -> str:
    release = get_latest_release(OPEN_WEBUI_RELEASE_REPO)
    tag = str(release.get("tag_name") or "").strip()
    if not tag:
        raise SystemExit(f"Could not determine latest release tag for {OPEN_WEBUI_RELEASE_REPO}.")
    return tag[1:] if tag.startswith("v") else tag


def openwebui_pull_candidates(image: str) -> list[str]:
    if image != OPEN_WEBUI_IMAGE_DEFAULT:
        return [image]

    version = latest_openwebui_release_version()
    candidates = [
        OPEN_WEBUI_IMAGE_DEFAULT,
        f"openwebui/open-webui:{version}",
        OPEN_WEBUI_IMAGE_DOCKER_HUB_DEFAULT,
    ]

    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def pull_openwebui_image(rt_path: str, image: str) -> str:
    candidates = openwebui_pull_candidates(image)

    last_error: Optional[subprocess.CalledProcessError] = None
    for idx, candidate in enumerate(candidates):
        LOG.info("Pulling latest image: %s", candidate)
        try:
            run([rt_path, "pull", candidate], check=True)
            return candidate
        except subprocess.CalledProcessError as e:
            last_error = e
            if idx + 1 < len(candidates):
                LOG.warning(
                    "Failed to pull %s. Trying fallback image: %s",
                    candidate,
                    candidates[idx + 1],
                )
                continue
            raise

    if last_error is not None:
        raise last_error
    raise SystemExit(f"Could not pull Open WebUI image: {image}")


def extract_data_mount_kind_and_name(info: Dict[str, object]) -> Tuple[Optional[str], Optional[str]]:
    mounts = info.get("Mounts")
    if not isinstance(mounts, list):
        return None, None

    for mount in mounts:
        if not isinstance(mount, dict):
            continue
        if mount.get("Destination") != "/app/backend/data":
            continue
        kind = mount.get("Type")
        if kind == "volume":
            name = mount.get("Name")
            return "volume", str(name) if name else None
        if kind == "bind":
            return "bind", None
        return str(kind) if kind else None, None

    return None, None


def resolve_openwebui_data_volume(rt_path: str, container_name: str, requested_data_volume: Optional[str]) -> Optional[str]:
    if requested_data_volume:
        return requested_data_volume

    inspected = container_inspect(rt_path, container_name)
    if inspected is not None:
        existing_kind, existing_volume_name = extract_data_mount_kind_and_name(inspected)
        if existing_kind == "volume" and existing_volume_name:
            return existing_volume_name

    if volume_exists(rt_path, OPEN_WEBUI_ORBSTACK_DATA_VOLUME_DEFAULT):
        return OPEN_WEBUI_ORBSTACK_DATA_VOLUME_DEFAULT

    return None


def refresh_openwebui(
    repo: Path,
    venv_python: Path,
    container_name: str,
    port: int,
    image: str,
    runtime: Optional[str],
    data_volume: Optional[str],
) -> None:
    data_dir = repo / "var" / "open-webui" / "data"

    rt = detect_runtime(runtime)
    if rt is None:
        LOG.warning("No container runtime found (docker/podman/nerdctl). Skipping Open WebUI refresh.")
        return
    rt_name, rt_path = rt
    LOG.info("Using container runtime: %s (%s)", rt_name, rt_path)
    available, detail = runtime_available(rt_name, rt_path)
    if not available:
        LOG.warning(
            "Container runtime '%s' is installed but its daemon/service is unavailable. Skipping Open WebUI refresh.",
            rt_name,
        )
        LOG.warning("%s", detail)
        return

    requested_data_volume = data_volume
    existing_container = container_inspect(rt_path, container_name)
    existing_kind, _existing_volume_name = extract_data_mount_kind_and_name(existing_container) if existing_container else (None, None)
    data_volume = resolve_openwebui_data_volume(rt_path, container_name, requested_data_volume)
    if data_volume is None:
        data_dir.mkdir(parents=True, exist_ok=True)
    elif requested_data_volume is None:
        LOG.info("Using named volume for Open WebUI data: %s", data_volume)
        if existing_kind == "bind":
            LOG.warning(
                "Switching Open WebUI data from the repo bind mount to named volume '%s'. Existing bind-mounted data is not migrated automatically.",
                data_volume,
            )

    pulled_image = pull_openwebui_image(rt_path, image)
    resolved_image = resolve_pulled_image_reference(rt_path, pulled_image)
    if resolved_image != pulled_image:
        LOG.info("Resolved Open WebUI image to immutable digest: %s", resolved_image)
    update_compose_openwebui_image(repo / "deploy" / "compose" / "docker-compose.yml", resolved_image)

    LOG.info("Stopping/removing container: %s", container_name)
    run([rt_path, "stop", container_name], check=False)
    run([rt_path, "rm",   container_name], check=False)

    # Recreate via our helper (keeps data dir; helper uses -v + -p)
    LOG.info("Recreating Open WebUI container via module helper")
    cmd = [
        str(venv_python), "-m", "llama_suite.utils.openwebui",
        "--name", container_name,
        "--port", str(port),
        "--image", resolved_image,
    ]
    if data_volume is not None:
        cmd.extend(["--data-volume", data_volume])
    else:
        cmd.extend(["--data-dir", str(data_dir)])
    run(cmd)


# ─────────────────────────────────── main ────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Update llama-suite external dependencies.")
    p.add_argument("--venv", type=Path, default=None, help="Path to venv (default: <repo>/.venv)")
    p.add_argument("--dev-extras", action="store_true", help="Upgrade with [dev] extras.")
    p.add_argument("--swap-method", choices=["auto", "build", "download"], default="auto",
                   help="How to update llama-swap.")
    p.add_argument("--cpp-method", choices=["auto", "build", "download"], default="auto",
                   help="How to update llama.cpp server.")
    p.add_argument("--ik-cpp-method", choices=["auto", "build"], default="auto",
                   help="How to update ik_llama.cpp server.")
    p.add_argument("--gpu-backend", choices=["auto", "cpu", "cuda", "vulkan"], default="cuda",
                   help="Backend preference when downloading/building llama.cpp.")
    p.add_argument("--runtime", choices=list(RUNTIMES_ORDER), help="Force container runtime for Open WebUI.")
    p.add_argument("--webui-name", default="open-webui", help="Open WebUI container name.")
    p.add_argument("--webui-port", type=int, default=3000, help="Open WebUI host port.")
    p.add_argument("--webui-image", default=OPEN_WEBUI_IMAGE_DEFAULT, help=f"Open WebUI image (default: {OPEN_WEBUI_IMAGE_DEFAULT})")
    p.add_argument("--webui-data-volume", default=None, help="Named volume for Open WebUI data (mounted to /app/backend/data). If set, overrides var/open-webui/data.")
    p.add_argument(
        "--skip",
        action="append",
        choices=["none", "venv", "swap", "cpp", "ik_cpp", "webui"],
        default=[],
        help="Skip updating component(s). Can be passed multiple times (e.g. --skip venv --skip webui).",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    args = p.parse_args()
    setup_logging(args.verbose)

    repo = find_repo_root()
    vendor_dir = repo / "vendor"

    LOG.info("=== llama-suite updater ===")
    LOG.info("Repo root : %s", repo)
    LOG.info("Vendor    : %s", vendor_dir)

    skip = set(args.skip)
    if "none" in skip:
        skip.clear()

    step_plan: list[str] = []
    if "venv" not in skip:
        step_plan.append("Update Python dependencies")
    if "swap" not in skip:
        step_plan.append("Update llama-swap")
    if "cpp" not in skip:
        step_plan.append("Update llama.cpp")
    if "ik_cpp" not in skip:
        step_plan.append("Update ik_llama.cpp")
    if "webui" not in skip:
        step_plan.append("Refresh Open WebUI container")

    step_i = 0

    # venv & project deps
    venv_dir, venv_python, venv_pip = ensure_venv(repo, args.venv)
    if "venv" not in skip:
        step_i += 1
        emit_step(step_i, len(step_plan), "Update Python dependencies")
        upgrade_python_deps(repo, venv_python, args.dev_extras)
    else:
        LOG.info("Skipping venv deps update")

    # llama-swap
    if "swap" not in skip:
        step_i += 1
        emit_step(step_i, len(step_plan), "Update llama-swap")
        swap_path = update_llama_swap(vendor_dir, args.swap_method)
        LOG.info("llama-swap updated: %s", swap_path)
    else:
        LOG.info("Skipping llama-swap update")

    # llama.cpp
    if "cpp" not in skip:
        step_i += 1
        emit_step(step_i, len(step_plan), "Update llama.cpp")
        cpp_server = update_llama_cpp(vendor_dir, args.cpp_method, args.gpu_backend)
        LOG.info("llama.cpp server updated: %s", cpp_server)
    else:
        LOG.info("Skipping llama.cpp update")

    if "ik_cpp" not in skip:
        step_i += 1
        emit_step(step_i, len(step_plan), "Update ik_llama.cpp")
        ik_cpp_server = update_ik_llama_cpp(vendor_dir, args.ik_cpp_method, args.gpu_backend)
        LOG.info("ik_llama.cpp server updated: %s", ik_cpp_server)
    else:
        LOG.info("Skipping ik_llama.cpp update")

    # Open WebUI
    if "webui" not in skip:
        step_i += 1
        emit_step(step_i, len(step_plan), "Refresh Open WebUI container")
        refresh_openwebui(repo, venv_python, args.webui_name, args.webui_port, args.webui_image, args.runtime, args.webui_data_volume)
        LOG.info("Open WebUI refreshed: %s (port %d)", args.webui_name, args.webui_port)
    else:
        LOG.info("Skipping Open WebUI refresh")

    LOG.info("All done")


if __name__ == "__main__":
    main()
