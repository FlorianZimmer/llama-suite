#!/usr/bin/env python3
"""
llama-suite installer (new-structure, pyproject-first)

- Creates a venv at repo root (default: .venv/) and installs this project via `pip install -e .`
- Obtains llama-swap and llama.cpp into vendor/ (download or build)
- Ensures an Open WebUI container with data at var/open-webui/data/
- Works on Windows/macOS/Linux and is VS Code + PowerShell friendly
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
import zipfile
from pathlib import Path
from textwrap import dedent
from typing import Dict, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import venv

LOG = logging.getLogger("installer")

# Repos
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp.git"
LLAMA_SWAP_REPO = "https://github.com/mostlygeek/llama-swap.git"
LLAMA_CPP_RELEASE_REPO = "ggml-org/llama.cpp"
LLAMA_SWAP_RELEASE_REPO = "mostlygeek/llama-swap"

OPEN_WEBUI_IMAGE = "ghcr.io/open-webui/open-webui:main"

IS_WINDOWS = platform.system() == "Windows"


# ──────────────────────────────────────────────────────────────────────────────
# logging / running
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def run(cmd: Sequence[str] | str, cwd: Optional[Path] = None) -> None:
    printable = " ".join(map(str, cmd)) if isinstance(cmd, (list, tuple)) else cmd
    LOG.debug("exec: %s", printable)
    try:
        subprocess.run(cmd, cwd=cwd, check=True, shell=isinstance(cmd, str))
    except subprocess.CalledProcessError as exc:
        LOG.error("Command failed (%s): %s", exc.returncode, printable)
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# repo discovery
# ──────────────────────────────────────────────────────────────────────────────

def find_repo_root(start: Path | None = None) -> Path:
    """
    Walk upward to find a directory containing pyproject.toml (repo root).
    """
    cur = (start or Path.cwd()).resolve()
    for p in [cur, *cur.parents]:
        if (p / "pyproject.toml").exists():
            return p
    sys.exit("Could not find repo root (no pyproject.toml found). Run from inside the repo.")


# ──────────────────────────────────────────────────────────────────────────────
# venv helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_venv_paths(venv_dir: Path) -> Tuple[Path, Path, Path]:
    if IS_WINDOWS:
        python_exe = venv_dir / "Scripts" / "python.exe"
        pip_exe = venv_dir / "Scripts" / "pip.exe"
        activate_hint = venv_dir / "Scripts" / "Activate.ps1"
    else:
        python_exe = venv_dir / "bin" / "python"
        pip_exe = venv_dir / "bin" / "pip"
        activate_hint = venv_dir / "bin" / "activate"
    return python_exe, pip_exe, activate_hint


# ──────────────────────────────────────────────────────────────────────────────
# downloads / releases
# ──────────────────────────────────────────────────────────────────────────────

def download(url: str, dest: Path) -> None:
    LOG.info("Downloading: %s", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url) as resp, open(dest, "wb") as fh:
            shutil.copyfileobj(resp, fh)
    except (HTTPError, URLError) as e:
        LOG.error("Failed to download %s: %s", url, e)
        sys.exit(1)
    LOG.debug("Saved %s (%.1f MB)", dest, dest.stat().st_size / 1024**2)


def get_latest_release(repo: str) -> Dict:
    api = f"https://api.github.com/repos/{repo}/releases/latest"
    headers = {"User-Agent": "llama-suite-installer", "Accept": "application/vnd.github+json"}
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


# ──────────────────────────────────────────────────────────────────────────────
# safe extract
# ──────────────────────────────────────────────────────────────────────────────

def _is_within_dir(base: Path, target: Path) -> bool:
    try:
        return str(target.resolve(strict=False)).startswith(str(base.resolve(strict=False)))
    except Exception:
        return False


def extract_archive(archive: Path, dest_dir: Path) -> None:
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
        shutil.move(str(archive), str(dest_dir / archive.name))

    if not IS_WINDOWS:
        for p in dest_dir.glob("**/*"):
            if p.is_file() and os.access(p, os.R_OK):
                try:
                    p.chmod(p.stat().st_mode | 0o111)
                except Exception:
                    pass


def ensure_executable(path: Path) -> None:
    if not IS_WINDOWS:
        try:
            path.chmod(path.stat().st_mode | 0o111)
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# asset selection
# ──────────────────────────────────────────────────────────────────────────────

def _has_token(name: str, token: str) -> bool:
    return re.search(rf'(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])', name) is not None


def asset_for_platform(release: dict, stem_match: str) -> Optional[dict]:
    system = platform.system().lower()
    machine = platform.machine().lower()
    arch_ok_tokens = {"x86_64", "amd64", "x64"} if machine in {"x86_64", "amd64"} else {"arm64", "aarch64"}

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
        score = 10
        if system == "windows" and name.endswith(".zip"):
            score = 0
        elif system in {"linux", "darwin"} and (name.endswith(".tar.gz") or name.endswith(".tgz")):
            score = 0
        if score < best_score:
            best, best_score = asset, score
    return best


def _os_token_match(n: str, sys_token: str) -> bool:
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
    arch_token = {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(
        platform.machine().lower(), platform.machine().lower()
    )

    candidates: list[tuple[int, Dict]] = []
    for asset in release.get("assets", []):
        name = asset["name"].lower()
        if not name.endswith(".zip"):
            continue
        if any(t in name for t in ("cudart", "runtime", "deps", "cudnn", "cutensor")):
            continue
        if not (_os_token_match(name, system_token) and arch_token in name and "bin" in name):
            continue

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


# ──────────────────────────────────────────────────────────────────────────────
# obtain llama-swap
# ──────────────────────────────────────────────────────────────────────────────

def download_llama_swap(binary_dir: Path) -> Path:
    release = get_latest_release(LLAMA_SWAP_RELEASE_REPO)
    asset = asset_for_platform(release, "llama-swap")
    if not asset:
        sys.exit(f"No release asset for this platform in {LLAMA_SWAP_RELEASE_REPO}. Try --build-from-source swap.")

    with tempfile.TemporaryDirectory() as td:
        tmp_archive = Path(td) / asset["name"]
        download(asset["browser_download_url"], tmp_archive)

        binary_dir.mkdir(parents=True, exist_ok=True)
        for p in list(binary_dir.iterdir()):
            try:
                p.unlink() if p.is_file() or p.is_symlink() else shutil.rmtree(p)
            except Exception:
                pass
        extract_archive(tmp_archive, binary_dir)

    exe_name = "llama-swap.exe" if IS_WINDOWS else "llama-swap"
    for cand in binary_dir.iterdir():
        if cand.is_file() and cand.name.lower().startswith("llama-swap"):
            target = binary_dir / exe_name
            if cand != target:
                if target.exists():
                    target.unlink(missing_ok=True)
                try:
                    shutil.copy2(cand, target)
                except Exception:
                    cand.rename(target)
            ensure_executable(target)
            return target
    sys.exit("Downloaded llama-swap but could not locate the binary.")


def build_llama_swap(src_dir: Path, binary_dir: Path) -> Path:
    if src_dir.exists():
        run(["git", "pull"], cwd=src_dir)
    else:
        run(["git", "clone", LLAMA_SWAP_REPO, str(src_dir)])
    exe_name = "llama-swap.exe" if IS_WINDOWS else "llama-swap"
    exe_path = binary_dir / exe_name
    binary_dir.mkdir(parents=True, exist_ok=True)
    run(["go", "build", "-o", str(exe_path), "."], cwd=src_dir)
    ensure_executable(exe_path)
    return exe_path


def obtain_llama_swap(vendor_dir: Path, build: bool) -> Path:
    src_dir = vendor_dir / "llama-swap-source"
    bin_dir = vendor_dir / "llama-swap"
    if bin_dir.exists() and bin_dir.is_file():
        backup = bin_dir.with_suffix(".old_binary_backup")
        LOG.info("Found legacy llama-swap file, moving to %s", backup)
        try:
            bin_dir.rename(backup)
        except OSError as e:
            raise SystemExit(f"Failed to move old llama-swap binary: {e}")
    bin_dir.mkdir(parents=True, exist_ok=True)
    return build_llama_swap(src_dir, bin_dir) if build else download_llama_swap(bin_dir)


# ──────────────────────────────────────────────────────────────────────────────
# obtain llama.cpp
# ──────────────────────────────────────────────────────────────────────────────

def _copy_server_payload(src_dir: Path, target_bin_dir: Path, final_name: str) -> Path:
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
    for item in src_dir.iterdir():
        dest_name = final_name if item == server_src else item.name
        dest = target_bin_dir / dest_name
        if item.is_file():
            if dest.exists() and dest.resolve() != item.resolve():
                dest.unlink(missing_ok=True)
            shutil.copy2(item, dest)
            if dest == final_exe:
                ensure_executable(dest)
    return final_exe


def download_llama_cpp(vendor_dir: Path, gpu_backend: str) -> Path:
    LOG.info("Preparing llama.cpp (download, backend=%s)", gpu_backend)
    target_bin_dir = vendor_dir / "llama.cpp" / "bin"
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

    tmp_dir = vendor_dir / "llama.cpp" / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_archive = tmp_dir / asset["name"]
    download(asset["browser_download_url"], tmp_archive)

    with tempfile.TemporaryDirectory() as td:
        temp_extract_dir = Path(td)
        extract_archive(tmp_archive, temp_extract_dir)
        tmp_archive.unlink(missing_ok=True)

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


def build_llama_cpp(src_dir: Path, gpu_backend: str, target_bin_dir: Path) -> Path:
    LOG.info("Building llama.cpp from source (backend=%s)", gpu_backend)
    if src_dir.exists():
        run(["git", "pull"], cwd=src_dir)
    else:
        run(["git", "clone", "--depth", "1", LLAMA_CPP_REPO, str(src_dir)])

    build_dir = src_dir / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir()

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
                final_exe = target_bin_dir / final_name
                if final_exe.exists() and final_exe != c:
                    final_exe.unlink(missing_ok=True)
                if c != final_exe:
                    shutil.copy2(c, final_exe)
                    ensure_executable(final_exe)
                return final_exe

    raise SystemExit("Build complete, but server executable was not found.")


def obtain_llama_cpp(vendor_dir: Path, build: bool, gpu_backend: str) -> Path:
    src_dir = vendor_dir / "llama.cpp" / "source"
    bin_dir = vendor_dir / "llama.cpp" / "bin"
    if build:
        return build_llama_cpp(src_dir, gpu_backend, bin_dir)
    return download_llama_cpp(vendor_dir, gpu_backend)


# ──────────────────────────────────────────────────────────────────────────────
# Open WebUI container via module
# ──────────────────────────────────────────────────────────────────────────────

def ensure_openwebui_container(venv_python: Path, data_dir: Path, name: str, port: int, image: str) -> None:
    """
    Calls the installed module's CLI: python -m llama_suite.utils.openwebui --args ...
    The module should:
      - create container only if missing
      - mount data_dir
      - map host port
      - start the container
    """
    LOG.info("Ensuring Open WebUI container (name=%s, port=%d, image=%s)", name, port, image)
    data_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(venv_python), "-m", "llama_suite.utils.openwebui",
        "--name", name,
        "--port", str(port),
        "--data-dir", str(data_dir),
        "--image", image,
    ]
    try:
        run(cmd)
    except SystemExit:
        raise SystemExit("Failed to set up Open WebUI container. See logs above.")


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Install llama-suite vendor binaries and Python deps (pyproject-first).")
    parser.add_argument("--venv", type=Path, default=None, help="Path to create/use venv (default: <repo>/.venv)")
    parser.add_argument("--build-from-source", choices=["swap", "cpp", "all"], help="Build these from source instead of downloading releases.")
    parser.add_argument("--gpu-backend", choices=["auto", "cpu", "cuda", "vulkan"], default="auto", help="Preferred GPU backend for llama.cpp.")
    parser.add_argument("--webui-name", default="open-webui", help="Container name for Open WebUI.")
    parser.add_argument("--webui-port", type=int, default=3000, help="Host port for Open WebUI (default: 3000).")
    parser.add_argument("--no-webui", action="store_true", help="Skip Open WebUI container setup.")
    parser.add_argument("--dev-extras", action="store_true", help="Install project with [dev] extras if defined.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args()
    setup_logging(args.verbose)

    repo_root = find_repo_root()
    vendor_dir = repo_root / "vendor"
    models_dir = repo_root / "models"
    configs_dir = repo_root / "configs"
    configs_gen_dir = configs_dir / "generated"
    runs_dir = repo_root / "runs"
    var_dir = repo_root / "var"
    webui_data_dir = var_dir / "open-webui" / "data"

    venv_dir = (args.venv or (repo_root / ".venv")).resolve()
    venv_python, venv_pip, venv_activate = get_venv_paths(venv_dir)

    build_swap = args.build_from_source in {"swap", "all"}
    build_cpp = args.build_from_source in {"cpp", "all"}

    LOG.info(dedent(f"""
    =============================================================================
    llama-suite installer
    Repo root         : {repo_root}
    Virtual Env       : {venv_dir}
    Vendor dir        : {vendor_dir}
    GPU Backend Pref  : {args.gpu_backend}
    Build from source : swap={build_swap}  cpp={build_cpp}
    =============================================================================
    """).strip())

    # ensure directories
    for d in (vendor_dir, models_dir, configs_dir, configs_gen_dir, runs_dir, webui_data_dir):
        d.mkdir(parents=True, exist_ok=True)

    # venv
    LOG.info("Setting up Python virtual environment: %s", venv_dir)
    if not venv_dir.exists():
        venv.EnvBuilder(with_pip=True, upgrade_deps=True).create(str(venv_dir))
        LOG.info("Virtual environment created.")
    else:
        LOG.info("Virtual environment already exists; will ensure packages are present.")

    if not venv_python.exists():
        sys.exit(f"Venv created but python not found: {venv_python}")

    # upgrade base tooling and install this project via pyproject
    LOG.info("Upgrading pip/setuptools/wheel")
    run([str(venv_python), "-m", "pip", "install", "-U", "pip", "setuptools", "wheel", "build"])

    # Install project (editable) from repo root
    LOG.info("Installing this project from pyproject.toml (%s)", repo_root / "pyproject.toml")
    install_target = "."
    if args.dev_extras:
        install_target = ".[dev]"
    run([str(venv_python), "-m", "pip", "install", "-e", install_target], cwd=repo_root)

    # llama-swap
    LOG.info("Obtaining llama-swap")
    swap_path = obtain_llama_swap(vendor_dir, build_swap)
    LOG.info("llama-swap binary: %s", swap_path)

    # llama.cpp
    LOG.info("Obtaining llama.cpp server")
    cpp_server_path = obtain_llama_cpp(vendor_dir, build_cpp, args.gpu_backend)
    LOG.info("llama-server binary: %s", cpp_server_path)

    # Open WebUI (best-effort)
    if not args.no_webui:
        LOG.info("Pulling Open WebUI image (best-effort)")
        try:
            run(["docker", "pull", OPEN_WEBUI_IMAGE])
        except SystemExit:
            LOG.warning("Docker pull failed. You can pull manually later.")
        ensure_openwebui_container(
            venv_python=venv_python,
            data_dir=webui_data_dir,
            name=args.webui_name,
            port=args.webui_port,
            image=OPEN_WEBUI_IMAGE,
        )

    # summary
    summary = dedent(f"""
    -----------------------------------------------------------------------------
    Installation complete 🎉
    -----------------------------------------------------------------------------
    Repo root        : {repo_root}
    Python Venv      : {venv_dir}
    Activate (PS)    : & "{venv_activate}"
    Vendor           : {vendor_dir}
      • llama-swap   : {swap_path}
      • llama.cpp    : {cpp_server_path}
    Models           : {models_dir}
    Configs          : {configs_dir}
      • generated    : {configs_gen_dir}
    Runs             : {runs_dir}
    WebUI data       : {webui_data_dir}
    WebUI URL        : http://localhost:{args.webui_port}  (container: {args.webui_name})

    NEXT STEPS
    ==========
    1) Activate venv in PowerShell:
         & "{venv_activate}"
       Or CMD:
         CALL "{venv_dir / ('Scripts' if IS_WINDOWS else 'bin') / ('activate.bat' if IS_WINDOWS else 'activate')}"

    2) Put GGUF models into:
         {models_dir}

    3) Edit your configs in:
         {configs_dir}
       Generated files (e.g., config.effective.yaml) should go to:
         {configs_gen_dir}

    4) Start llama-swap with your config (example):
         "{swap_path}" --config "{configs_dir / 'config.base.yaml'}"

    5) Open WebUI:
         http://localhost:{args.webui_port}
       (Manage container with: docker stop/start {args.webui_name})

    6) Use your package modules as CLI (after we wire entry points in pyproject),
       e.g.: python -m llama_suite.bench.benchmark --help
    -----------------------------------------------------------------------------
    """).strip("\n")
    print(summary)


if __name__ == "__main__":
    main()
