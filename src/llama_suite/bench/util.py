#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import csv
import shutil
import signal
import platform
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable

import psutil

# Package utils (installed via editable install)
from llama_suite.utils.config_utils import (  # type: ignore
    Logger,
    generate_processed_config,
    build_llama_server_command_util,  # imported but unused; handy for debug if needed
    start_llama_server,
    stop_llama_server,
    wait_for_server_health,
    _dump_stderr_on_failure,
    color_status,
    PROCESS_TERMINATE_TIMEOUT_S,
    DEFAULT_HEALTH_POLL_INTERVAL_S,
)
from llama_suite.utils.runtime_registry import all_runtime_server_candidates

# ---------- Project / paths ----------
def find_project_root() -> Path:
    """
    Minimal, predictable:
      1) Env LLS_PROJECT_ROOT (must contain configs/)
      2) Walk up from this file until configs/ exists
      3) CWD if it contains configs/
      4) Walk up from CWD until configs/ exists
      5) Fallback to CWD
    """
    env = os.environ.get("LLS_PROJECT_ROOT")
    if env:
        p = Path(env).resolve()
        if (p / "configs").is_dir():
            return p

    here = Path(__file__).resolve()
    for ancestor in here.parents:
        if (ancestor / "configs").is_dir():
            return ancestor

    cwd = Path.cwd().resolve()
    if (cwd / "configs").is_dir():
        return cwd
    for ancestor in cwd.parents:
        if (ancestor / "configs").is_dir():
            return ancestor

    return cwd


PROJECT_ROOT: Path = find_project_root()
CONFIGS_DIR = PROJECT_ROOT / "configs"
OVERRIDES_DIR = CONFIGS_DIR / "overrides"
RUNS_BENCH_DIR = PROJECT_ROOT / "runs" / "bench"
LOGS_DIR = RUNS_BENCH_DIR / "logs"
RESULTS_DIR = RUNS_BENCH_DIR / "results"

RUN_DIR_PREFIX = "run_"
RETENTION_KEEP_DEFAULT = 10

# keys ignored when building flags from the model-level section
MODEL_CONFIG_META_KEYS_COMMON = {
    "aliases", "sampling", "_name_for_log", "generated_cmd_str",
    "hf_tokenizer_for_model", "supports_no_think_toggle",
}

# ---------- Small utilities ----------
def timestamp_str() -> str:
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def enforce_retention(
    dir_path: Path,
    pattern: str,
    keep: int,
    *,
    delete_dirs: bool = False,
    logger: Optional[Logger] = None
) -> None:
    try:
        items = sorted(dir_path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in items[keep:]:
            if delete_dirs and old.is_dir():
                shutil.rmtree(old, ignore_errors=True)
                if logger: logger.debug(f"Retention: removed dir {old}")
            elif old.is_file():
                old.unlink(missing_ok=True)
                if logger: logger.debug(f"Retention: removed file {old}")
    except Exception as e:
        if logger:
            logger.warn(f"Retention error in {dir_path} ({pattern}): {e}")


def write_csv(headers: Iterable[str], rows: Iterable[Iterable[Any]], out_path: Path, logger: Optional[Logger] = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(headers))
        for r in rows:
            w.writerow(list(r))
    if logger:
        logger.success(f"Saved CSV: {out_path}")


# ---------- Path normalization & executable discovery ----------
def _remap_from_configs_abs(p: Path, logger: Logger) -> Optional[Path]:
    """
    If an absolute was formed under CONFIGS_DIR, remap to PROJECT_ROOT.
    """
    try:
        if p.is_absolute() and str(p).lower().startswith(str(CONFIGS_DIR).lower()):
            rel = p.relative_to(CONFIGS_DIR)
            candidate = (PROJECT_ROOT / rel).resolve()
            if candidate.exists():
                logger.debug(f"    Remapped from CONFIGS_DIR -> PROJECT_ROOT: {candidate}")
                return candidate
    except Exception:
        pass
    return None


def _fallback_llama_server(logger: Logger) -> Optional[str]:
    names = ["llama-server"]
    if platform.system().lower().startswith("win"):
        names.append("llama-server.exe")
    for n in names:
        for cand in all_runtime_server_candidates(PROJECT_ROOT, base_name=n):
            cand = cand.resolve()
            if cand.is_file():
                logger.debug(f"    Fallback llama-server at: {cand}")
                return str(cand)
    return None


def normalize_path_str(
    raw: str,
    *,
    must_exist: bool,
    logger: Logger,
    is_executable: bool = False,
) -> str:
    """
    Simple and predictable:
      - expand ~ and env vars
      - relative -> PROJECT_ROOT relative
      - if absolute-but-under CONFIGS_DIR and missing -> remap to PROJECT_ROOT
      - for executables, try simple fallback locations
    """
    if not raw:
        return raw
    p = Path(os.path.expandvars(os.path.expanduser(raw)))

    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()

    if must_exist and not p.exists():
        remapped = _remap_from_configs_abs(p, logger)
        if remapped and remapped.exists():
            p = remapped

    if is_executable and must_exist and not p.exists():
        fb = _fallback_llama_server(logger)
        if fb:
            return fb

    return str(p)


def resolve_llama_server_executable(path_str: str, logger: Logger) -> str:
    """
    Convenience: if a folder or loose path given, look for common build subfolders.
    """
    p = Path(path_str).expanduser()
    if p.is_file():
        return str(p)

    search_roots = [p] if p.is_dir() else [p.parent]
    exe_name = p.name if p.name else "llama-server"
    is_windows = platform.system().lower().startswith("win")

    names = [exe_name]
    if is_windows and not exe_name.lower().endswith(".exe"):
        names.append(exe_name + ".exe")

    subdirs = ["", "Release", "RelWithDebInfo", "MinSizeRel", "Debug"]
    candidates: List[Path] = []
    for root in search_roots:
        for sub in subdirs:
            for name in names:
                candidate = (root / sub / name) if sub else (root / name)
                candidates.append(candidate)

    for root in search_roots:
        for name in set(names):
            try:
                for hit in root.rglob(name):
                    candidates.append(hit)
            except Exception:
                pass

    for c in candidates:
        if c.is_file():
            logger.debug(f"    Resolved llama-server candidate: {c}")
            return str(c)

    return path_str


# ---------- Parsing helpers ----------
def parse_param_size_from_alias(model_alias: str, logger: Optional[Logger] = None) -> str:
    moe = re.search(r"([0-9]+(?:\.[0-9]+)?x[0-9]+(?:\.[0-9]+)?)[Bb]", model_alias, re.IGNORECASE)
    if moe:
        size = f"{moe.group(1)}B"
        if logger: logger.debug(f"Parsed param size '{size}' from MoE alias '{model_alias}'")
        return size
    simple = re.search(r"([0-9]+(?:\.[0-9]+)?)[Bb]", model_alias, re.IGNORECASE)
    if simple:
        size = f"{simple.group(1)}B"
        if logger: logger.debug(f"Parsed param size '{size}' from alias '{model_alias}'")
        return size
    if logger: logger.debug(f"Could not parse param size from alias '{model_alias}'")
    return "-"


def parse_quant_from_string(s: str, logger: Optional[Logger] = None) -> str:
    patterns = [
        (r"[Qq]([2-8AXL]_[A-Z0-9_]+(?:_L)?)", lambda m: m.group(0).upper()),
        (r"IQ[1-4]_[A-Z0-9_]+", lambda m: m.group(0).upper()),
        (r"\b(F32|F16|BF16)\b", lambda m: m.group(1).upper()),
        (r"[Qq]([2-8](?:_0)?)\b", lambda m: m.group(0).upper()),
        (r"[Qq](FP[48])\b", lambda m: m.group(0).upper()),
    ]
    for pat, norm in patterns:
        m = re.search(pat, s, re.IGNORECASE)
        if m:
            q = norm(m)
            if logger: logger.debug(f"Parsed quant '{q}' from '{s}'")
            return q
    if logger: logger.debug(f"No quant parsed from '{s}'")
    return "-"


def parse_memory_string_to_gb(mem_str: str) -> Optional[float]:
    mem_str = mem_str.strip()
    m = re.match(r"([0-9]+(?:[.,][0-9]+)?)\s*([KMGT])(?:i?B)?", mem_str, re.IGNORECASE)
    if not m:
        return None
    value = float(m.group(1).replace(",", "."))
    unit = m.group(2).upper()
    mult = {"K": 1/1024/1024, "M": 1/1024, "G": 1, "T": 1024}[unit]
    return value * mult


def parse_memory_from_log(stderr_log_path: Path, model_name: str, logger: Logger) -> Tuple[str, str, str]:
    total_gpu_gb, total_cpu_gb = 0.0, 0.0
    lines_found, parse_errors = 0, 0

    if not stderr_log_path.exists():
        logger.warn(f"    Memory log not found for '{model_name}': {stderr_log_path}")
        return "0.00", "0.00", "Failed (No Log)"

    import time
    buffer_regex = re.compile(r"(ggml_vk_)?(Metal|CUDA|CPU).*?buffer size\s*=\s*([0-9.,]+\s*[KMGT]i?B)", re.IGNORECASE)
    time.sleep(0.2)
    try:
        with stderr_log_path.open("r", encoding="utf-8", errors="replace") as f:
            for _, line in enumerate(f, 1):
                m = buffer_regex.search(line)
                if not m:
                    continue
                lines_found += 1
                device_type, mem_str = m.group(2), m.group(3)
                mem_gb = parse_memory_string_to_gb(mem_str)
                if mem_gb is None:
                    logger.warn(f"      Could not parse memory value: '{mem_str}' in line: {line.strip()}")
                    parse_errors += 1
                    continue
                if device_type.upper() in ["METAL", "CUDA"]:
                    total_gpu_gb += mem_gb
                elif device_type.upper() == "CPU":
                    total_cpu_gb += mem_gb
    except Exception as e:
        logger.warn(f"    Error reading/parsing memory log {stderr_log_path}: {e}")
        return "0.00", "0.00", "Failed (Log Read Error)"

    final_status = "Parse Error" if (lines_found > 0 and parse_errors > 0) else ("Success" if lines_found > 0 else "Failed (No Buffers)")
    return f"{total_gpu_gb:.2f}", f"{total_cpu_gb:.2f}", final_status


# ---------- Process & config helpers ----------
def kill_lingering_servers_on_port(port: str, logger: Logger) -> int:
    logger.step(f"Cleaning possible lingering servers on port {port}...")
    killed = 0
    target_port_arg = f"--port {port}"
    current_user = psutil.Process().username()

    for proc in psutil.process_iter(["pid", "name", "cmdline", "exe", "username"]):
        pid: Optional[int] = None
        try:
            info = proc.info
            pid = info.get("pid")
            name = (info.get("name") or "").lower()
            exe = (info.get("exe") or "").lower()
            cmdline = " ".join(info.get("cmdline") or []).lower()
            username = info.get("username")

            is_llama = (target_port_arg in cmdline) or ("llama-server" in name) or ("llama-server" in exe)
            if not is_llama:
                continue

            if username and username != current_user and "root" in (username or "").lower() and target_port_arg not in cmdline:
                logger.debug(f"  PID {pid} ('{name}') owned by '{username}' (different user). Skipping.")
                continue

            logger.warn(f"  Terminating PID {pid} ({name})")
            p = psutil.Process(pid)
            p.terminate()
            try:
                p.wait(timeout=PROCESS_TERMINATE_TIMEOUT_S / 2.0)
            except psutil.TimeoutExpired:
                logger.warn(f"    PID {pid} did not terminate, killing...")
                p.kill()
                p.wait(timeout=PROCESS_TERMINATE_TIMEOUT_S / 2.0)
            killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pid_repr = str(pid) if pid is not None else "unknown"
            logger.debug(f"  Skipping inaccessible/gone process PID {pid_repr}")
        except Exception as e:
            pid_repr = str(pid) if pid is not None else "unknown"
            logger.warn(f"  psutil error on PID {pid_repr}: {e}")

    if killed:
        logger.success(f"Killed {killed} process(es).")
    else:
        logger.success("No lingering servers found.")
    return killed


def default_override_for_hostname() -> Optional[Path]:
    default_host = platform.node().split(".")[0].lower()
    maybe_override = OVERRIDES_DIR / f"{default_host}.yaml"
    return maybe_override if maybe_override.exists() else None


def load_and_process_config(
    base_config: Path,
    override: Optional[Path],
    *,
    verbose: bool,
    logger: Logger,
) -> Dict[str, Any]:
    return generate_processed_config(
        base_config_path_arg=base_config.resolve(),
        override_config_path_arg=override.resolve() if override else None,
        script_dir_for_overrides=PROJECT_ROOT,
        verbose_logging=verbose,
    )


def select_models(models_cfg: Dict[str, Any], only_alias: Optional[str], logger: Logger) -> Dict[str, Dict[str, Any]]:
    if not isinstance(models_cfg, dict) or not models_cfg:
        raise ValueError("'models' section not found/empty in config")
    if only_alias:
        if only_alias not in models_cfg or not isinstance(models_cfg[only_alias], dict):
            logger.error(f"Model '{only_alias}' not found or malformed in config.")
            logger.info("Available: " + ", ".join(k for k, v in models_cfg.items() if isinstance(v, dict)))
            return {}
        return {only_alias: models_cfg[only_alias]}
    return {k: v for k, v in models_cfg.items() if isinstance(v, dict)}


# ------- Simple draft + flag handling -------
DRAFT_PATH_KEYS = {"model-draft", "draft-model", "model_draft", "draft_model"}

def _is_path_like(s: str) -> bool:
    s = s.strip().strip('"').strip("'")
    return ("/" in s) or ("\\" in s) or s.lower().endswith(".gguf")


def _to_flag_name(k: str) -> str:
    k = k.replace("_", "-").lower()
    # one small compatibility alias for llama.cpp
    if k == "gpu-layers":
        return "n-gpu-layers"
    return k


def _normalize_model_path(raw: str, *, logger: Logger) -> str:
    p = normalize_path_str(raw, must_exist=True, logger=logger, is_executable=False)
    pp = Path(p)
    if not pp.is_file():
        # if it was wrongly rooted under configs/, try simple remap (already done in normalize_path_str)
        raise FileNotFoundError(f"Model file not found: {p}")
    return str(pp)


def build_server_command(
    model_alias: str,
    model_cfg: Dict[str, Any],
    *,
    static_port: str,
    logger: Logger,
) -> Tuple[str, List[str], str]:
    """
    Returns: (executable_path, args_list, ctx_size_str)
    Rule of thumb:
      - special-case bin/model/model-draft (normalize paths)
      - if key == draft and looks like a path -> treat as model-draft path
      - else pass everything through as --flag [value]
    """
    original_cmd = model_cfg.get("cmd")
    if not isinstance(original_cmd, dict):
        raise ValueError("Config Error: missing 'cmd' dict")

    cmd = dict(original_cmd)
    cmd["port"] = static_port

    # --- bin / model ---
    raw_bin = cmd.get("bin")
    if not raw_bin or not isinstance(raw_bin, str):
        raise ValueError("Resolved 'cmd.bin' is missing/invalid.")
    server_exe = normalize_path_str(raw_bin, must_exist=True, logger=logger, is_executable=True)
    server_exe = resolve_llama_server_executable(server_exe, logger)
    if not Path(server_exe).is_file():
        raise FileNotFoundError(f"Executable not found at '{server_exe}'")

    raw_model = cmd.get("model")
    if not isinstance(raw_model, str) or not raw_model:
        raise ValueError("Model path 'cmd.model' is required.")
    cmd["model"] = _normalize_model_path(raw_model, logger=logger)

    # --- Build args (pass-through) ---
    handled = {"bin", "port", "model"}
    server_args: List[str] = [
        "--port", str(cmd.get("port")),
        "--model", str(cmd.get("model")),
    ]

    # First collect special draft args (cmd-level)
    extra_args: List[str] = []
    for k, v in list(cmd.items()):
        if k in handled:
            continue
        k_norm = k.replace("_", "-").lower()

        # draft model path synonyms
        if k_norm in DRAFT_PATH_KEYS:
            if isinstance(v, str) and v:
                path = _normalize_model_path(v, logger=logger)
                extra_args += ["--model-draft", path]
            continue

        # draft tokens or path shoved into 'draft'
        if k_norm == "draft":
            if isinstance(v, (int, float)) or (isinstance(v, str) and v.strip().replace(".", "", 1).isdigit()):
                n = int(float(v))
                if n > 0:
                    extra_args += ["--draft", str(n)]
            elif isinstance(v, str) and _is_path_like(v):
                path = _normalize_model_path(v, logger=logger)
                extra_args += ["--model-draft", path]
            # else: ignore falsey
            continue

        # generic other keys -> pass through
        if isinstance(v, bool):
            if v:
                extra_args += [f"--{_to_flag_name(k)}"]
        else:
            if v not in (None, "", False, "auto", "Auto"):
                extra_args += [f"--{_to_flag_name(k)}", str(v)]

    # top-level model_cfg keys (outside cmd)
    for k, v in model_cfg.items():
        if k in {"cmd"} | MODEL_CONFIG_META_KEYS_COMMON:
            continue
        k_norm = k.replace("_", "-").lower()

        if k_norm in DRAFT_PATH_KEYS:
            if isinstance(v, str) and v:
                path = _normalize_model_path(v, logger=logger)
                extra_args += ["--model-draft", path]
            continue
        if k_norm == "draft":
            if isinstance(v, (int, float)) or (isinstance(v, str) and v.strip().replace(".", "", 1).isdigit()):
                n = int(float(v))
                if n > 0:
                    extra_args += ["--draft", str(n)]
            elif isinstance(v, str) and _is_path_like(v):
                path = _normalize_model_path(v, logger=logger)
                extra_args += ["--model-draft", path]
            continue

        # sampling stays pass-through like anything else (if present at top-level)
        if isinstance(v, bool):
            if v:
                extra_args += [f"--{_to_flag_name(k)}"]
        else:
            if v not in (None, "", False, "auto", "Auto"):
                extra_args += [f"--{_to_flag_name(k)}", str(v)]

    # include ctx-size string for reporting (if present)
    ctx_size = "-"
    # prefer cmd.ctx-size, fall back to top-level ctx-size
    if isinstance(cmd.get("ctx-size"), (int, str)) and str(cmd.get("ctx-size")):
        ctx_size = str(cmd.get("ctx-size"))
    elif isinstance(model_cfg.get("ctx-size"), (int, str)) and str(model_cfg.get("ctx-size")):
        ctx_size = str(model_cfg.get("ctx-size"))

    server_args += extra_args
    return server_exe, server_args, ctx_size


# ---------- Signals ----------
def register_signal_handlers(app_label: str, get_logger: Callable[[], Optional[Logger]], get_tempdir: Callable[[], Optional[Path]]) -> None:
    def _handler(signum, _frame):
        lg = get_logger()
        log = print if not lg else lg.warn
        log(f"\nSignal {signal.Signals(signum).name} received. Cleaning up {app_label}...")
        td = get_tempdir()
        if td:
            log(f"  Temporary dir/logs: {td}")
        if lg and hasattr(lg, "close") and callable(lg.close):
            lg.close()
        raise SystemExit(1)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
