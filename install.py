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

# Determine the default target directory name based on OS.
# This is the name of the directory that will be created if not already the CWD's name.
_default_target_dirname_str = "llama-suite" # Consistent hyphenated name

# Set the default installation base path relative to the current working directory.
_current_dir = Path.cwd()
if _current_dir.name.lower() == _default_target_dirname_str.lower():
    # If the current directory is already named appropriately (e.g., "llama-suite"),
    # then the default installation base is the current directory itself.
    DEFAULT_INSTALL_BASE = _current_dir
else:
    # Otherwise, the default is to create/use a subdirectory with the target name
    # inside the current directory.
    DEFAULT_INSTALL_BASE = _current_dir / _default_target_dirname_str

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
    "watchdog~=4.0",
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

def obtain_llama_swap(base: Path, build: bool) -> Path: # Does NOT take gpu_backend
    src_dir = base / "llama-swap-source"
    bin_dir = base / "llama-swap" # Changed from "llama-swap" to ensure it's a directory
    if bin_dir.exists() and bin_dir.is_file(): # Check if it's a file (old single binary)
        legacy_file = bin_dir
        backup = bin_dir.with_suffix(bin_dir.suffix + ".old_binary_backup") # More descriptive backup name
        print(f"Found old llama-swap binary at {legacy_file}. Moving to {backup}")
        try:
            legacy_file.rename(backup)
        except OSError as e:
            # If rename fails (e.g. target exists or permissions), it's safer to halt.
            raise SystemExit(f"Failed to move old llama-swap binary {legacy_file} to {backup}: {e}. Please handle this manually.")
    
    # Ensure bin_dir is a directory for placing the new binary (and potentially DLLs if any in future)
    bin_dir.mkdir(parents=True, exist_ok=True)

    return (
        build_llama_swap(src_dir, bin_dir) # build_llama_swap places output in bin_dir
        if build
        else download_llama_swap(bin_dir) # download_llama_swap also targets bin_dir
    )

def asset_for_platform(release: Dict, stem_match: str) -> Optional[Dict]:
    """
    Finds a release asset that broadly matches the current platform (OS and architecture)
    and contains a specific stem in its name. Used for llama-swap.
    """
    system = platform.system().lower()
    machine = platform.machine().lower().replace("x86_64", "amd64").replace("aarch64", "arm64")
    
    # Use a slightly more verbose print to distinguish from asset_for_platform_cpp
    print(f"  (llama-swap) Searching for '{stem_match}' asset for system='{system}', machine='{machine}'")

    for asset in release["assets"]:
        name_lower = asset["name"].lower()
        if stem_match not in name_lower:
            continue

        os_match = False
        if system == "darwin" and ("darwin" in name_lower or "macos" in name_lower):
            os_match = True
        elif system == "windows" and ("windows" in name_lower or "win" in name_lower):
            os_match = True
        elif system == "linux" and ("linux" in name_lower or "ubuntu" in name_lower):
            os_match = True
        
        if not os_match:
            continue

        if machine not in name_lower:
            continue
            
        print(f"    (llama-swap) Found matching asset: {asset['name']}")
        return asset
    
    print(f"    (llama-swap) No asset found matching all criteria for '{stem_match}'.")
    return None

def asset_for_platform_cpp(release_json: dict, gpu_backend_pref: str) -> Optional[dict]:
    system_token = {
        "Darwin": "macos", "Windows": "win", "Linux": "ubuntu",
    }.get(platform.system(), platform.system().lower())
    arch_token = {
        "amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64",
    }.get(platform.machine().lower(), platform.machine().lower())

    print(f"  Searching for assets for platform: {system_token}-{arch_token}, GPU preference: {gpu_backend_pref}")

    candidates_data: list[dict] = [] 
    known_runtime_prefixes = ("cudart-", "vulkan-sdk-", "opencl-")

    for asset in release_json["assets"]:
        name = asset["name"].lower()
        is_zip = name.endswith(".zip")
        
        system_match = system_token in name or (system_token == "ubuntu" and "linux" in name)
        if not (system_match and arch_token in name and is_zip and "bin" in name):
            continue

        is_runtime_package = any(name.startswith(prefix) for prefix in known_runtime_prefixes)
        
        is_full_binary_package = False
        if not is_runtime_package:
            if ("llama-" in name and "-bin-" in name) or \
               ("server" in name and "-bin-" in name and "llama" in name and not name.startswith("llama-swap")):
                is_full_binary_package = True
        
        current_asset_score = 100 

        if gpu_backend_pref == "cuda":
            if "cuda" in name: 
                current_asset_score = 0 
                if "cuda-12" in name or "cu12" in name: current_asset_score = -2 
                elif "cuda-11" in name or "cu11" in name: current_asset_score = -1
            else: continue 
        elif gpu_backend_pref == "vulkan":
            if "vulkan" in name and not is_runtime_package: current_asset_score = 0 
            elif "vulkan" in name: current_asset_score = 1 
            else: continue
        elif gpu_backend_pref == "cpu":
            if "cpu" in name and not is_runtime_package : current_asset_score = 0 
            elif not any(gpu_kw in name for gpu_kw in ["cuda", "vulkan", "metal"]):
                current_asset_score = 1 
            else: continue
        elif gpu_backend_pref == "auto":
            if system_token == "macos" and "metal" in name and is_full_binary_package: current_asset_score = 0
            elif "cuda" in name and is_full_binary_package: current_asset_score = 10 
            elif "vulkan" in name and is_full_binary_package: current_asset_score = 20 
            elif "cpu" in name and is_full_binary_package: current_asset_score = 30 
            elif "cuda" in name: current_asset_score = 11 
            elif "vulkan" in name: current_asset_score = 21 
            elif not any(gpu_kw in name for gpu_kw in ["metal", "cuda", "vulkan"]):
                 current_asset_score = 40 
            else: continue
        else: 
            print(f"  Warning: Unknown gpu_backend_pref '{gpu_backend_pref}'")
            continue 
            
        if current_asset_score != 100:
            # print(f"    Debug: Considering asset: {name}, RawScore: {current_asset_score}, IsFullBinary: {is_full_binary_package}, IsRuntime: {is_runtime_package}")
            candidates_data.append({
                "asset_dict": asset, 
                "name": name,
                "raw_score": current_asset_score, 
                "is_full_binary": is_full_binary_package,
                "is_runtime": is_runtime_package
            })

    if not candidates_data:
        print(f"  No candidates found matching current platform and GPU preference '{gpu_backend_pref}'.")
        return None
    
    print(f"  Found {len(candidates_data)} C++ release candidate(s) BEFORE sorting:") # DEBUG
    for i, c_debug in enumerate(candidates_data):
        print(f"    Candidate Pre-Sort [{i}]: {c_debug['name']} (RawScore: {c_debug['raw_score']}, FullBinary: {c_debug['is_full_binary']}, IsRuntime: {c_debug['is_runtime']})")

    def sort_key_final(c_dict_item):
        full_binary_penalty = 0 if c_dict_item["is_full_binary"] else 1
        runtime_package_penalty = 1 if c_dict_item["is_runtime"] else 0
        key_tuple = (c_dict_item["raw_score"], full_binary_penalty, runtime_package_penalty, c_dict_item["name"])
        # print(f"      Debug Sort Key for {c_dict_item['name']}: {key_tuple}") # Optional: very verbose
        return key_tuple

    candidates_data.sort(key=sort_key_final) 

    print(f"  Found {len(candidates_data)} C++ release candidate(s) AFTER sorting:") # DEBUG
    if len(candidates_data) > 0:
        # This loop will print the *actually sorted* list and their keys
        for i, c_debug in enumerate(candidates_data):
             print(f"    Candidate Post-Sort [{i}]: {c_debug['name']} (RawScore: {c_debug['raw_score']}, FullBinary: {c_debug['is_full_binary']}, IsRuntime: {c_debug['is_runtime']}) -> SortKey: {sort_key_final(c_debug)}")
    
    selected_candidate_dict_after_sort = candidates_data[0]
    
    # The print for the selected one should use the values from selected_candidate_dict_after_sort
    print(f"  Selected based on preference '{gpu_backend_pref}': {selected_candidate_dict_after_sort['name']} "
          f"(RawScore: {selected_candidate_dict_after_sort['raw_score']}, FullBinary: {selected_candidate_dict_after_sort['is_full_binary']})")
    
    return selected_candidate_dict_after_sort["asset_dict"]


def download_llama_cpp(base_install_dir: Path, gpu_backend: str) -> Path:
    print(f"--- Debug: Entering download_llama_cpp for gpu_backend='{gpu_backend}' ---")
    llama_cpp_dir = base_install_dir / "llama.cpp"
    llama_cpp_dir.mkdir(parents=True, exist_ok=True)
    
    FINAL_TARGET_EXE_NAME = "llama-server.exe" if IS_WINDOWS else "llama-server"
    target_bin_dir = llama_cpp_dir / "build" / "bin"
    print(f"  Debug: Final target directory for binaries: {target_bin_dir}")
    target_bin_dir.mkdir(parents=True, exist_ok=True)

    if not LLAMA_CPP_RELEASE_REPO:
        raise SystemExit("LLAMA_CPP_RELEASE_REPO not set...")

    print(f"Fetching latest release from GitHub API: {LLAMA_CPP_RELEASE_REPO}")
    release = get_latest_release(LLAMA_CPP_RELEASE_REPO)
    print(f"  Latest release tag: {release.get('tag_name', 'N/A')}")
    
    asset = asset_for_platform_cpp(release, gpu_backend) 
    
    if not asset:
        # (Error message remains the same)
        err_msg = (
            f"No suitable asset for this platform ({platform.system()}/{platform.machine()}) "
            f"with GPU preference '{gpu_backend}' in {LLAMA_CPP_RELEASE_REPO} release {release.get('tag_name', 'N/A')}. "
        )
        if gpu_backend != "auto":
            err_msg += "Try --gpu-backend auto, or an alternative backend, or --build-from-source cpp."
        else: 
            err_msg += "Try specifying a GPU backend (e.g., --gpu-backend cuda if you have CUDA) or --build-from-source cpp."
        raise SystemExit(err_msg)
    
    print(f"  Debug: Selected asset for download: {asset.get('name')}")
    url = asset["browser_download_url"]
    tmp_archive_path = llama_cpp_dir / asset["name"]
    print(f"  Debug: Temporary archive path: {tmp_archive_path}")
    download(url, tmp_archive_path) # download() already prints messages

    temp_extract_dir = llama_cpp_dir / "_temp_extract"
    print(f"  Debug: Temporary extraction directory: {temp_extract_dir}")
    if temp_extract_dir.exists():
        print(f"    Debug: Removing existing temp_extract_dir: {temp_extract_dir}")
        shutil.rmtree(temp_extract_dir)
    temp_extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {tmp_archive_path.name} to {temp_extract_dir} ...")
    extract_archive(tmp_archive_path, temp_extract_dir) # extract_archive() prints messages
    
    if tmp_archive_path.exists():
        print(f"  Debug: Deleting temporary archive: {tmp_archive_path}")
        tmp_archive_path.unlink() 

    possible_original_exe_names = {"server", "llama-server"}
    if IS_WINDOWS:
        possible_original_exe_names = {name + ".exe" for name in possible_original_exe_names}
    print(f"  Debug: Searching for executables {possible_original_exe_names} in {temp_extract_dir} and subdirectories...")
    
    found_original_exe_path_in_extract: Optional[Path] = None
    for p in temp_extract_dir.rglob("*"): 
        if p.is_file() and p.name.lower() in possible_original_exe_names:
            found_original_exe_path_in_extract = p
            print(f"    Debug: Found potential executable: {p}")
            break 

    if found_original_exe_path_in_extract:
        source_bin_content_dir = found_original_exe_path_in_extract.parent 
        final_target_server_exe_path = target_bin_dir / FINAL_TARGET_EXE_NAME

        print(f"  Found an executable at: {found_original_exe_path_in_extract}")
        print(f"  Its containing directory (for DLLs etc.): {source_bin_content_dir}")
        print(f"  Copying relevant files from {source_bin_content_dir} to target {target_bin_dir}...")
        print(f"    Debug: Contents of source_bin_content_dir ({source_bin_content_dir}):")
        for item_debug in source_bin_content_dir.iterdir():
            print(f"      - {item_debug.name} {'(DIR)' if item_debug.is_dir() else '(FILE)'}")


        copied_files_count = 0
        for item in source_bin_content_dir.iterdir():
            source_item_path = source_bin_content_dir / item.name
            target_item_name_in_final_dir = item.name
            is_main_executable_being_copied = (source_item_path.name.lower() in possible_original_exe_names)

            if is_main_executable_being_copied:
                target_item_name_in_final_dir = FINAL_TARGET_EXE_NAME

            target_item_path = target_bin_dir / target_item_name_in_final_dir
            
            if source_item_path.is_file():
                print(f"    Attempting to copy {source_item_path.name} to {target_item_path}")
                if target_item_path.exists() and source_item_path.resolve() != target_item_path.resolve():
                    print(f"      Debug: Target {target_item_path} exists, unlinking.")
                    target_item_path.unlink(missing_ok=True)
                shutil.copy2(str(source_item_path), str(target_item_path))
                print(f"      Debug: Copied {source_item_path.name} successfully.")
                copied_files_count += 1
                if is_main_executable_being_copied:
                    print(f"      Debug: Ensuring executable permissions for {target_item_path}")
                    ensure_executable(target_item_path)
            else:
                print(f"    Debug: Skipping directory during copy: {source_item_path.name}")


        if not final_target_server_exe_path.exists():
             print(f"  Debug: Main server executable {final_target_server_exe_path} NOT FOUND after copy loop.")
             if temp_extract_dir.exists(): # Ensure cleanup before raising
                 print(f"    Debug: Cleaning up temp_extract_dir: {temp_extract_dir}")
                 shutil.rmtree(temp_extract_dir)
             raise SystemExit(f"Failed to copy or rename the server executable to {final_target_server_exe_path} after extraction.")

        print(f"  Successfully copied {copied_files_count} file(s) to {target_bin_dir}.")
        if temp_extract_dir.exists():
            print(f"  Debug: Cleaning up temp_extract_dir: {temp_extract_dir}")
            shutil.rmtree(temp_extract_dir)
        print(f"--- Debug: Exiting download_llama_cpp, returning: {final_target_server_exe_path} ---")
        return final_target_server_exe_path
    else:
        print(f"  Debug: No suitable server executable found in {temp_extract_dir} after extraction.")
        if temp_extract_dir.exists():
            print(f"    Debug: Cleaning up temp_extract_dir: {temp_extract_dir}")
            shutil.rmtree(temp_extract_dir)
        raise SystemExit(
            f"Downloaded and extracted {asset['name']}, but could not locate "
            f"a suitable server executable (checked for {', '.join(possible_original_exe_names)}) within the extracted content."
        )

def build_llama_cpp(src_dir: Path, gpu_backend: str) -> Path:
    print(f"--- Debug: Entering build_llama_cpp for gpu_backend='{gpu_backend}' ---")
    if src_dir.exists():
        print(f"  Debug: Pulling existing Llama.cpp source: {src_dir}")
        run(["git", "pull"], cwd=src_dir)
    else:
        print(f"  Debug: Cloning Llama.cpp source to: {src_dir}")
        run(["git", "clone", "--depth", "1", LLAMA_CPP_REPO, str(src_dir)])

    cmake_build_dir = src_dir / "build"
    print(f"  Debug: CMake build directory: {cmake_build_dir}")
    if cmake_build_dir.exists():
        print(f"    Debug: Removing existing CMake build directory: {cmake_build_dir}")
        shutil.rmtree(cmake_build_dir)
    cmake_build_dir.mkdir()

    final_target_bin_dir = src_dir / "build" / "bin" 
    print(f"  Debug: Final target directory for binaries: {final_target_bin_dir}")
    final_target_bin_dir.mkdir(parents=True, exist_ok=True)
    
    FINAL_TARGET_EXE_NAME = "llama-server.exe" if IS_WINDOWS else "llama-server"
    cmake_intended_runtime_output_dir = final_target_bin_dir

    cmake_flags = [
        f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={str(cmake_intended_runtime_output_dir)}",
        f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE={str(cmake_intended_runtime_output_dir)}", 
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    # (GPU backend flag logic remains the same)
    if gpu_backend == "cuda":
        cmake_flags.append("-DLLAMA_CUBLAS=ON") 
        print("  Configuring llama.cpp build with CUDA backend.")
    elif gpu_backend == "vulkan":
        cmake_flags.append("-DLLAMA_VULKAN=ON")
        print("  Configuring llama.cpp build with Vulkan backend.")
    elif gpu_backend == "cpu":
        cmake_flags.extend(["-DLLAMA_CUBLAS=OFF", "-DLLAMA_VULKAN=OFF", "-DGGML_METAL=OFF"])
        print("  Configuring llama.cpp build for CPU only.")
    elif gpu_backend == "auto":
        if platform.system() == "Darwin":
            if not any("GGML_METAL=ON" in f for f in cmake_flags): cmake_flags.append("-DGGML_METAL=ON")
            print("  Configuring llama.cpp build with Metal backend (auto for macOS).")
        elif IS_WINDOWS:
            print("  Configuring llama.cpp build (auto for Windows - relies on CMake defaults).")
        else: 
            print("  Configuring llama.cpp build (auto for Linux - relies on CMake defaults).")
    if platform.system() == "Darwin" and gpu_backend != "cpu":
        if not any("GGML_METAL=ON" in f for f in cmake_flags):
             cmake_flags.append("-DGGML_METAL=ON")

    cmake_args = ["cmake", ".."] + [flag for flag in cmake_flags if flag]
    print(f"  Debug: Running CMake configure with args: {cmake_args} in {cmake_build_dir}")
    run(cmake_args, cwd=cmake_build_dir)

    jobs = os.cpu_count() or 4
    print(f"  Debug: Running CMake build with jobs={jobs} in {cmake_build_dir}")
    run(["cmake", "--build", ".", "--config", "Release", "-j", str(jobs)], cwd=cmake_build_dir)
    
    print(f"  Debug: CMake build command finished. Searching for artifacts...")
    search_dirs_for_artifacts = [
        cmake_intended_runtime_output_dir, cmake_intended_runtime_output_dir / "Release",
        cmake_build_dir / "bin", cmake_build_dir / "bin" / "Release",
        cmake_build_dir, cmake_build_dir / "Release" 
    ]
    print(f"    Debug: Search directories for artifacts: {search_dirs_for_artifacts}")
    
    actual_artifact_source_dir: Optional[Path] = None
    original_exe_name_style1 = "server.exe" if IS_WINDOWS else "server"
    original_exe_name_style2 = "llama-server.exe" if IS_WINDOWS else "llama-server"

    for d_candidate in search_dirs_for_artifacts:
        d = d_candidate.resolve()
        if d.exists() and d.is_dir():
            print(f"      Debug: Checking directory {d} for executables...")
            if (d / original_exe_name_style1).is_file():
                actual_artifact_source_dir = d
                print(f"        Debug: Found '{original_exe_name_style1}' in {d}")
                break
            if (d / original_exe_name_style2).is_file():
                actual_artifact_source_dir = d
                print(f"        Debug: Found '{original_exe_name_style2}' in {d}")
                break
        else:
            print(f"      Debug: Search directory {d} does not exist or is not a directory.")


    if not actual_artifact_source_dir:
        raise SystemExit(f"Build finished, but could not locate the directory containing an executable ('{original_exe_name_style1}' or '{original_exe_name_style2}').")

    print(f"  Found build artifacts source directory: {actual_artifact_source_dir}")
    final_exe_at_target_path = final_target_bin_dir / FINAL_TARGET_EXE_NAME

    if actual_artifact_source_dir.resolve() != final_target_bin_dir.resolve():
        print(f"  Artifacts source ({actual_artifact_source_dir}) is different from final target ({final_target_bin_dir}). Relocating/Standardizing...")
        print(f"    Debug: Contents of actual_artifact_source_dir ({actual_artifact_source_dir}):")
        for item_debug in actual_artifact_source_dir.iterdir():
            print(f"      - {item_debug.name} {'(DIR)' if item_debug.is_dir() else '(FILE)'}")
        
        copied_files_count = 0
        for item in actual_artifact_source_dir.iterdir():
            source_item_path = actual_artifact_source_dir / item.name
            target_item_name_in_final_dir = item.name
            is_main_exe_from_build = (item.name.lower() == original_exe_name_style1.lower() or 
                                      item.name.lower() == original_exe_name_style2.lower())

            if is_main_exe_from_build:
                target_item_name_in_final_dir = FINAL_TARGET_EXE_NAME
            
            target_item_path = final_target_bin_dir / target_item_name_in_final_dir
            
            if source_item_path.is_file():
                if IS_WINDOWS and not source_item_path.name.lower().endswith((".exe", ".dll", ".pdb", ".so", ".dylib")):
                    print(f"    Skipping non-essential file (Win build): {source_item_path.name}")
                    continue
                
                print(f"    Attempting to copy {source_item_path.name} to {target_item_path}")
                if target_item_path.exists() and target_item_path.resolve() != source_item_path.resolve():
                    print(f"      Debug: Target {target_item_path} exists, unlinking.")
                    target_item_path.unlink(missing_ok=True)
                shutil.copy2(str(source_item_path), str(target_item_path))
                print(f"      Debug: Copied {source_item_path.name} successfully.")
                copied_files_count += 1
                if is_main_exe_from_build:
                    print(f"      Debug: Ensuring executable permissions for {target_item_path}")
                    ensure_executable(target_item_path)
            else:
                print(f"    Debug: Skipping directory during copy: {source_item_path.name}")

        print(f"  Successfully copied/relocated {copied_files_count} file(s) to {final_target_bin_dir}.")
    else:
        print(f"  Build artifacts already in the final target directory: {final_target_bin_dir}")
        path_orig_style1_in_target = final_target_bin_dir / original_exe_name_style1
        path_orig_style2_in_target = final_target_bin_dir / original_exe_name_style2
        renamed_in_place = False

        if path_orig_style1_in_target.is_file() and path_orig_style1_in_target.name != FINAL_TARGET_EXE_NAME:
            print(f"    Renaming in-place {path_orig_style1_in_target.name} to {FINAL_TARGET_EXE_NAME}")
            if final_exe_at_target_path.exists() and final_exe_at_target_path.resolve() != path_orig_style1_in_target.resolve() : 
                final_exe_at_target_path.unlink(missing_ok=True)
            path_orig_style1_in_target.rename(final_exe_at_target_path); renamed_in_place = True
        elif path_orig_style2_in_target.is_file() and path_orig_style2_in_target.name != FINAL_TARGET_EXE_NAME:
            print(f"    Renaming in-place {path_orig_style2_in_target.name} to {FINAL_TARGET_EXE_NAME}")
            if final_exe_at_target_path.exists() and final_exe_at_target_path.resolve() != path_orig_style2_in_target.resolve(): 
                final_exe_at_target_path.unlink(missing_ok=True)
            path_orig_style2_in_target.rename(final_exe_at_target_path); renamed_in_place = True
        
        if renamed_in_place or final_exe_at_target_path.is_file():
            print(f"    Debug: Ensuring executable permissions for {final_exe_at_target_path}")
            ensure_executable(final_exe_at_target_path)

    if not final_exe_at_target_path.is_file():
        print(f"  Debug: Final executable {final_exe_at_target_path} NOT FOUND after all processing.")
        raise SystemExit(f"Build process complete, but the server executable '{FINAL_TARGET_EXE_NAME}' was not found at the final location: {final_exe_at_target_path}")

    print(f"--- Debug: Exiting build_llama_cpp, returning: {final_exe_at_target_path} ---")
    return final_exe_at_target_path

def obtain_llama_cpp(base_install_dir: Path, build: bool, gpu_backend: str) -> Path: # ADDED gpu_backend
    llama_cpp_root_dir = base_install_dir / "llama.cpp"
    
    if llama_cpp_root_dir.exists() and llama_cpp_root_dir.is_file():
        backup_path = llama_cpp_root_dir.with_suffix(llama_cpp_root_dir.suffix + ".old_file_backup")
        print(f"Found an existing file at target directory location {llama_cpp_root_dir}. Moving it to {backup_path}")
        try:
            llama_cpp_root_dir.rename(backup_path)
        except OSError as e:
            # It's critical to exit here if we can't move a conflicting file.
            raise SystemExit(f"Failed to move conflicting file {llama_cpp_root_dir}: {e}. Please remove it manually before proceeding.")
            
    llama_cpp_root_dir.mkdir(parents=True, exist_ok=True)

    if build:
        return build_llama_cpp(llama_cpp_root_dir, gpu_backend) # PASS gpu_backend
    else:
        return download_llama_cpp(base_install_dir, gpu_backend) # PASS gpu_backend


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
    parser.add_argument(
        "--gpu-backend",
        choices=["auto", "cpu", "cuda", "vulkan"], 
        default="auto", 
        help="Preferred GPU backend for llama.cpp (affects download and build-from-source). 'auto' tries to pick best available for platform."
    )
    args = parser.parse_args()

    install_dir: Path = args.install_dir.expanduser().resolve()
    model_dir = install_dir / "models"
    webui_dir = install_dir / "open-webui-data"
    eval_dir = install_dir / "eval"
    venv_dir = install_dir / VENV_NAME

    build_swap = args.build_from_source in {"swap", "all"}
    build_cpp = args.build_from_source in {"cpp", "all"}
    gpu_backend_preference = args.gpu_backend 

    print(
        dedent(
            f"""
        =============================================================================
        llama-suite installer (with venv support)
        Target directory  : {install_dir}
        Virtual Env       : {venv_dir}
        Build from source : swap={build_swap}  cpp={build_cpp}
        GPU Backend Pref  : {gpu_backend_preference} (for llama.cpp)
        =============================================================================
        """
        )
    )

    for d in (install_dir, model_dir, webui_dir, eval_dir, eval_dir / "results"):
        d.mkdir(parents=True, exist_ok=True)
    print(f"Ensured base directories exist under: {install_dir}")

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
        run([str(venv_pip), "install", "--upgrade", package_spec], check_return_code=False)
    print("Python package installation attempt complete.")

    print("\n=== Installing llama-swap ===")
    # llama-swap installation does NOT use gpu_backend_preference
    swap_path = obtain_llama_swap(install_dir, build_swap) # CORRECTED: No gpu_backend_preference here
    print(f"llama-swap binary → {swap_path}")

    print("\n=== Installing llama.cpp server ===")
    # llama.cpp installation DOES use gpu_backend_preference
    cpp_server_path = obtain_llama_cpp(install_dir, build_cpp, gpu_backend_preference)
    print(f"llama-server binary → {cpp_server_path}")

    try:
        print("\n=== Pulling Open WebUI image ===")
        run(["docker", "pull", OPEN_WEBUI_IMAGE], check_return_code=False)
    except Exception as e:
        print(f"WARNING: Docker pull failed ({e}). You can pull manually later.")

    port_hint = "3000"
    eval_script_path = eval_dir / "evaluate-models.py"
    eval_results_path = eval_dir / "results"

    activation_command_unix = f"source \"{venv_dir / 'bin' / 'activate'}\""
    activation_command_windows = f"CALL \"{venv_dir / 'Scripts' / 'activate.bat'}\"" 

    summary = dedent(
        f"""
        -----------------------------------------------------------------------------
        Installation complete 🎉
        -----------------------------------------------------------------------------
        llama.cpp server : {cpp_server_path} (GPU backend: {gpu_backend_preference})
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
           For llama.cpp models, ensure the `bin:` path is (the script adds `.exe` on Windows):
           `  bin: llama.cpp/build/bin/llama-server`
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