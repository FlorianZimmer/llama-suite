#!/usr/bin/env python3
"""
LLM Evaluation Script using lm-evaluation-harness.

- Works with the new repo layout (package under src/llama_suite).
- Resolves llama-server.exe and model paths robustly (remaps configs/... to project root, fallbacks to vendor/ or build/bin).
- Outputs logs to runs/eval/logs/task-<tasks>_<timestamp>/...
- Outputs results (json + csv) to runs/eval/results/task-<tasks>_<timestamp>/...

Run as either:
  python -m llama_suite.eval.evaluate_models_evalharness [args...]
or
  python src/llama_suite/eval/evaluate_models_evalharness.py [args...]
"""

from __future__ import annotations

# --- Capture originals before other libs might patch them ---
import os
_original_os_unlink = os.unlink
import tempfile
_original_tempfile_TemporaryDirectory = tempfile.TemporaryDirectory

# --- Standard Library Imports ---
import argparse
import atexit
import csv
import datetime
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence, Mapping, Union, Callable, cast
import multiprocessing
import logging as std_logging
import json as json_serializer
import traceback
import re
import shutil

# --- Third-Party Imports ---
import psutil
from lm_eval import simple_evaluate
import lm_eval.tasks

def preprocess_for_json(data: Any) -> Any:
    """Make arbitrary nested data JSON-serializable, logging interesting fallbacks."""
    if isinstance(data, (str, int, float, bool, type(None))):
        return data
    if isinstance(data, dict):
        return {preprocess_for_json(k): preprocess_for_json(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [preprocess_for_json(elem) for elem in data]
    if callable(data):
        return f"<function {getattr(data, '__name__', 'unknown_callable')}>"

    # numpy-like scalars
    if hasattr(data, 'item') and callable(getattr(data, 'item')):
        try:
            item_val = data.item()
            if isinstance(item_val, (str, int, float, bool, type(None))):
                return item_val
            return str(item_val)
        except Exception:
            return str(data)

    # numpy/pandas arrays/series
    if hasattr(data, 'tolist') and callable(getattr(data, 'tolist')):
        try:
            return preprocess_for_json(data.tolist())
        except Exception:
            return str(data)

    # Last resort: check if json can already handle it; otherwise stringify
    try:
        json_serializer.dumps(data)
        return data
    except TypeError:
        _preprocess_logger.debug(
            f"Defaulting to str() for type {type(data)}, value (first 100 chars): {str(data)[:100]}..."
        )
        return str(data)
    except Exception as e_json_test:
        _preprocess_logger.warning(
            f"Error during serialization test for type {type(data)}: {e_json_test}. "
            f"Defaulting to str(). Value (first 100 chars): {str(data)[:100]}"
        )
        return str(data)


# --- Logging setup for JSON preprocess ---
_preprocess_logger = std_logging.getLogger("lm_eval.preprocess_for_json")

# =============================================================================
# Repo root / import shim so the file works as module or script
# =============================================================================

_THIS_FILE = Path(__file__).resolve()

def _find_repo_root() -> Path:
    # 0) Allow a more specific env var if provided
    env_lls = os.getenv("LLS_PROJECT_ROOT")
    if env_lls:
        p = Path(env_lls).resolve()
        if (p / "configs").is_dir():
            return p

    # 1) Also accept LLAMA_SUITE_ROOT / LLS_ROOT used elsewhere
    for env in ("LLAMA_SUITE_ROOT", "LLS_ROOT"):
        val = os.getenv(env)
        if val:
            p = Path(val).resolve()
            if (p / "configs").is_dir():
                return p

    # 2) Walk up until we see expected layout
    cur = _THIS_FILE
    for parent in [cur] + list(cur.parents):
        if (parent / "configs").is_dir() and (parent / "src").is_dir():
            return parent

    # 3) Fallback: assume ../../../ from file
    try:
        return _THIS_FILE.parents[3]
    except Exception:
        return Path.cwd()

REPO_ROOT = _find_repo_root()
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# Project path handle (compat with helpers)
project_root_dir = REPO_ROOT

# --- Project-Specific Imports (package style) ---
from llama_suite.utils.config_utils import (  # type: ignore
    Logger, generate_processed_config, build_llama_server_command_util,
    start_llama_server, stop_llama_server, wait_for_server_health,
    _dump_stderr_on_failure, color_status,
    PROCESS_TERMINATE_TIMEOUT_S, DEFAULT_HEALTH_POLL_INTERVAL_S
)

# Importing the module is enough to register both adapters with lm-eval.
import llama_suite.utils.custom_lm_eval_models as _custom_models  # noqa: F401

# =============================================================================
# Environment & Constants
# =============================================================================

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_TASKS = "humaneval"
DEFAULT_LIMIT_SAMPLES: Optional[float] = None
DEFAULT_BATCH_SIZE_EVAL = "1"
DEFAULT_NUM_FEWSHOT: Optional[int] = None
STATIC_EVAL_PORT = "9997"
DEFAULT_HEALTH_TIMEOUT_S_EVAL = 3600
DEFAULT_QWEN3_NO_PREFIX_TASKS_STR_CLI = "humaneval;mbpp;gsm8k;gpqa;mmlu;mmlu_pro_biology;mmlu_pro_business;mmlu_pro_chemistry;mmlu_pro_computer_science;mmlu_pro_economics;mmlu_pro_engineering;mmlu_pro_health;mmlu_pro_history;mmlu_pro_law;mmlu_pro_math;mmlu_pro_other;mmlu_pro_philosophy;mmlu_pro_physics;mmlu_pro_psychology"

# Keys in model config treated as meta-only (not emitted as CLI flags here)
MODEL_CONFIG_META_KEYS_EVAL = {
    "aliases", "sampling", "_name_for_log", "generated_cmd_str",
    "hf_tokenizer_for_model", "supports_no_think_toggle",
    "num_concurrent_eval",
    "enabled", "skip", "_skip", "disabled",
}
MODEL_CONFIG_META_KEYS_COMMON = {
    "aliases", "sampling", "_name_for_log", "generated_cmd_str",
    "hf_tokenizer_for_model", "supports_no_think_toggle",
    "enabled", "skip", "_skip", "disabled",
}
COMBINED_META_KEYS = MODEL_CONFIG_META_KEYS_EVAL | MODEL_CONFIG_META_KEYS_COMMON

# Globals set during runtime
logger: Optional['Logger'] = None
TEMP_DIR_MANAGER_PATH_EVAL: Optional[Path] = None

# =============================================================================
# Helpers (path normalization, retention, CSV, signals)
# =============================================================================

CONFIGS_DIR = project_root_dir / "configs"

# --- Disabled model helpers ---------------------------------------------------
from typing import Mapping

def _truthy_str(s: str) -> bool:
    return s.strip().lower() in {"1", "true", "yes", "y", "on"}

def _is_model_disabled(mcfg: Mapping[str, Any]) -> bool:
    """Return True if a model config indicates it should be skipped/disabled."""
    try:
        # Explicit enabled=false wins
        en = mcfg.get("enabled", None)
        if isinstance(en, str):
            if not _truthy_str(en):  # "false", "0", "no", etc.
                return True
        elif en is not None and not bool(en):
            return True
    except Exception:
        pass

    for k in ("disabled", "skip", "_skip"):
        v = mcfg.get(k, False)
        # strings like "true"/"1"/"yes"
        if isinstance(v, str):
            if _truthy_str(v):
                return True
        elif bool(v):
            return True
    return False

def _filter_disabled_models(models: Dict[str, Any], logger_instance: "Logger") -> Dict[str, Any]:
    """Drop disabled models; log which ones were skipped."""
    kept: Dict[str, Any] = {}
    skipped: List[str] = []
    for alias, data in models.items():
        if not isinstance(data, dict):
            continue
        if _is_model_disabled(data):
            skipped.append(alias)
            continue
        kept[alias] = data
    if skipped:
        logger_instance.info(
            "Skipping %d model(s) disabled by config: %s" %
            (len(skipped), ", ".join(sorted(skipped)))
        )
    return kept

def _remap_from_configs_abs(p: Path, logger: Logger) -> Optional[Path]:
    """If path lives under CONFIGS_DIR (absolute), remap to project root preserving the tail."""
    try:
        if p.is_absolute() and str(p).lower().startswith(str(CONFIGS_DIR).lower()):
            rel = p.relative_to(CONFIGS_DIR)
            candidate = (project_root_dir / rel).resolve()
            if candidate.exists():
                logger.debug(f"    Remapped path from CONFIGS_DIR -> PROJECT_ROOT: {candidate}")
                return candidate
    except Exception:
        pass
    return None

def _fallback_llama_server(logger: Logger) -> Optional[str]:
    names = ["llama-server"]
    if platform.system().lower().startswith("win"):
        names.append("llama-server.exe")
    for base in [project_root_dir / "vendor" / "llama.cpp" / "bin",
                 project_root_dir / "llama.cpp" / "build" / "bin"]:
        for n in names:
            cand = (base / n).resolve()
            if cand.is_file():
                logger.debug(f"    Fallback llama-server found at: {cand}")
                return str(cand)
    return None

def normalize_path_str(
    raw: str,
    *,
    must_exist: bool,
    logger: Logger,
    is_executable: bool = False,
) -> str:
    if not raw:
        return raw
    p = Path(os.path.expandvars(os.path.expanduser(raw)))
    if not p.is_absolute():
        p = (project_root_dir / p).resolve()
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
    """Given a path or folder hint, try to find the actual llama-server binary (incl. Release/RelWithDebInfo variants)."""
    p = Path(path_str).expanduser()
    if p.is_file():
        return str(p)

    # Determine where to search and candidate names
    search_roots = [p] if p.is_dir() else [p.parent]
    exe_name = p.name if p.name else "llama-server"
    is_windows = platform.system().lower().startswith("win")

    names = [exe_name]
    if is_windows and not exe_name.lower().endswith(".exe"):
        names.append(exe_name + ".exe")

    # Common build subdirs
    subdirs = ["", "Release", "RelWithDebInfo", "MinSizeRel", "Debug"]
    candidates: List[Path] = []
    for root in search_roots:
        for sub in subdirs:
            for name in names:
                candidate = (root / sub / name) if sub else (root / name)
                candidates.append(candidate)

    # Also rglob in the root(s)
    for root in search_roots:
        for name in set(names):
            try:
                for hit in root.rglob(name):
                    candidates.append(hit)
            except Exception:
                pass

    for c in candidates:
        if c.is_file():
            logger.debug(f"    Resolved llama-server executable candidate: {c}")
            return str(c)

    return path_str

def timestamp_str() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def write_csv(headers: Sequence[str], rows: Sequence[Sequence[Any]], out_path: Path, logger: Optional[Logger] = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(headers))
        for r in rows:
            w.writerow(list(r))
    if logger:
        logger.success(f"Saved CSV: {out_path}")

def final_os_unlink_restorer_atexit():
    """Restore os.unlink at interpreter shutdown if needed."""
    global _original_os_unlink
    if _original_os_unlink is None:
        print("ATEIXT_CRITICAL: _original_os_unlink is None.", file=sys.stderr)
        return
    current_unlink = getattr(os, 'unlink', None)
    if current_unlink is None or current_unlink != _original_os_unlink:
        print("ATEIXT_INFO: os.unlink was incorrect at final Python exit. Restoring globally.", file=sys.stderr)
        os.unlink = _original_os_unlink

atexit.register(final_os_unlink_restorer_atexit)

def _ensure_os_unlink_restored(context_message: str, logger_instance: Optional[Logger]):
    """Ensure os.unlink is the original one."""
    log_func = print if logger_instance is None else logger_instance.debug
    current_unlink = getattr(os, 'unlink', None)
    if current_unlink is None or current_unlink != _original_os_unlink:
        log_func(f"DEBUG_OS_UNLINK_RESTORE: {context_message} - os.unlink was '{str(current_unlink)}'. Restoring.")
        if _original_os_unlink is not None:
            os.unlink = _original_os_unlink
        elif logger_instance:
            logger_instance.error("CRITICAL: Cannot restore os.unlink: _original_os_unlink is None.")
        else:
            print("CRITICAL: Cannot restore os.unlink: _original_os_unlink is None.", file=sys.stderr)

def signal_cleanup_handler_eval(signum, frame):
    """Terminate lingering server on our static port and exit."""
    global logger, TEMP_DIR_MANAGER_PATH_EVAL

    if logger:
        log_func = logger.warn
    else:
        def print_to_stderr(msg): print(msg, file=sys.stderr)
        log_func = print_to_stderr

    log_func(f"\nSignal {signal.Signals(signum).name} received. Cleaning up evaluate-models...")

    if TEMP_DIR_MANAGER_PATH_EVAL:
        log_func(f"  Using logging dir: {TEMP_DIR_MANAGER_PATH_EVAL}")

    log_func(f"Attempting to kill any lingering server processes on port {STATIC_EVAL_PORT} due to signal.")
    killed_during_signal = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline_list = proc.info.get('cmdline')
            if cmdline_list and f"--port {STATIC_EVAL_PORT}" in " ".join(cmdline_list):
                log_func(f"  Signal Cleanup: Killing PID={proc.pid} (Name='{proc.info.get('name', '')}') on port {STATIC_EVAL_PORT}")
                p_obj = psutil.Process(proc.pid)
                p_obj.terminate()
                try:
                    p_obj.wait(timeout=1)
                except psutil.TimeoutExpired:
                    log_func(f"    PID={proc.pid} did not terminate gracefully, forcing kill.")
                    p_obj.kill()
                    p_obj.wait(timeout=1)
                killed_during_signal += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except Exception as e_ps_signal:
            log_func(f"  Error during signal cleanup process kill: {e_ps_signal}")

    if killed_during_signal > 0:
        log_func(f"Signal cleanup terminated {killed_during_signal} server process(es).")
    else:
        log_func(f"No server processes found to kill on port {STATIC_EVAL_PORT} during signal cleanup.")

    if logger:
        logger.info("Evaluation script terminated by signal.")
    else:
        print("Evaluation script terminated by signal.", file=sys.stderr)

    _ensure_os_unlink_restored("During signal cleanup (eval)", logger)
    sys.exit(1)

def _call_simple_evaluate(**kwargs: Any) -> Optional[Dict[str, Any]]:
    """
    Typed-loose shim to satisfy Pylance regardless of lm-eval stubs version.
    Returns a dict result (or None), matching our downstream Optional[Dict[str, Any]] usage.
    """
    return cast(Optional[Dict[str, Any]], simple_evaluate(**kwargs))

# =============================================================================
# Server command building (uses robust path normalization)
# =============================================================================

def build_server_command(
    model_alias: str,
    model_cfg: Dict[str, Any],
    *,
    static_port: str,
    logger: Logger,
) -> Tuple[str, List[str], str]:
    """
    Returns: (executable_path, args_list, ctx_size_str)
    - Remaps any absolute paths under CONFIGS_DIR to project root.
    - Normalizes model/bin/other path-like flags.
    - Resolves correct llama-server executable (adds .exe, Release/, etc).
    """
    original_cmd = model_cfg.get("cmd")
    if not isinstance(original_cmd, dict):
        raise ValueError("Config Error: missing 'cmd' dict")

    cmd = dict(original_cmd)
    cmd["port"] = static_port

    # Normalize critical paths
    raw_bin = cmd.get("bin")
    if not raw_bin or not isinstance(raw_bin, str):
        raise ValueError("Resolved 'cmd.bin' is missing/invalid.")
    server_exe = normalize_path_str(raw_bin, must_exist=True, logger=logger, is_executable=True)

    raw_model = cmd.get("model")
    if isinstance(raw_model, str) and raw_model:
        cmd["model"] = normalize_path_str(raw_model, must_exist=True, logger=logger)

    # Normalize other likely path flags
    pathy_keys = {"log-file", "log_path", "chat-template", "rpc-server", "imatrix", "mmproj", "mmproj-file"}
    handled = {"bin", "port", "model", "ctx-size", "gpu-layers", "threads"}
    for k, v in list(cmd.items()):
        if k in handled:
            continue
        if isinstance(v, str) and k in pathy_keys:
            cmd[k] = normalize_path_str(v, must_exist=False, logger=logger)

    # Resolve executable variations (Release/RelWithDebInfo/…)
    server_exe_resolved = resolve_llama_server_executable(server_exe, logger)
    if server_exe_resolved != server_exe:
        logger.info(f"  Adjusted server executable path -> {server_exe_resolved}")
        server_exe = server_exe_resolved

    if not Path(server_exe).is_file():
        raise FileNotFoundError(f"Executable not found at '{server_exe}'. Tried remap + fallbacks.")

    # Build CLI args
    server_args: List[str] = [
        "--port", str(cmd.get("port")),
        "--model", str(cmd.get("model")),
    ]
    if cmd.get("ctx-size") is not None:
        server_args += ["--ctx-size", str(cmd.get("ctx-size"))]

    if str(cmd.get("gpu-layers", "auto")).lower() != "auto":
        server_args += ["--n-gpu-layers", str(cmd.get("gpu-layers"))]
    if str(cmd.get("threads", "auto")).lower() != "auto":
        server_args += ["--threads", str(cmd.get("threads"))]

    for k, v in cmd.items():
        if k in handled:
            continue
        flag = f"--{k.replace('_','-')}"
        if isinstance(v, bool):
            if v:
                server_args.append(flag)
        elif v not in (None, False, "auto", "Auto", ""):
            server_args += [flag, str(v)]

    for k, v in model_cfg.items():
        if k in {"cmd", "sampling"} | COMBINED_META_KEYS:
            continue
        flag = f"--{k.replace('_','-')}"
        if isinstance(v, bool):
            if v:
                server_args.append(flag)
        elif v not in (None, False, "auto", "Auto", ""):
            server_args += [flag, str(v)]

    sampling = model_cfg.get("sampling")
    if isinstance(sampling, dict):
        for k, v in sampling.items():
            server_args += [f"--{k.replace('_','-')}", str(v)]

    ctx_size = str(cmd.get("ctx-size", "-"))
    return server_exe, server_args, ctx_size

# =============================================================================
# Core Evaluation
# =============================================================================

class ServerOperationError(Exception):
    pass

def run_evaluation(
    processed_models_config: Dict[str, Any],
    results_output_dir: Path,
    logs_output_dir: Path,
    tasks_str: str,
    limit_samples: Optional[float],
    num_fewshot: Optional[int],
    batch_size_str: str,
    health_timeout_s: int,
    health_poll_s: float,
    model_to_eval_alias: Optional[str],
    logger_instance: Logger,
    qwen3_no_prefix_tasks_override: str,
    num_concurrent_eval_override: Optional[int],
):
    is_verbose = logger_instance.verbose_flag

    if not processed_models_config:
        logger_instance.error("No processed model configurations for evaluation.")
        return

    try:
        results_output_dir.mkdir(parents=True, exist_ok=True)
        logs_output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger_instance.error(f"CRITICAL: Cannot create output dirs: results={results_output_dir}, logs={logs_output_dir} - {e}")
        return

    # Honor disable flags before any selection
    processed_models_config = _filter_disabled_models(processed_models_config, logger_instance)

    # Select models
    if model_to_eval_alias:
        if model_to_eval_alias in processed_models_config and isinstance(processed_models_config[model_to_eval_alias], dict):
            models_to_iterate: Dict[str, Any] = {model_to_eval_alias: processed_models_config[model_to_eval_alias]}
        else:
            logger_instance.error(f"Specified model '{model_to_eval_alias}' not found, invalid, or disabled by config.")
            return
    else:
        models_to_iterate = {alias: data for alias, data in processed_models_config.items() if isinstance(data, dict)}

    if not models_to_iterate:
        logger_instance.error("No valid model configurations found.")
        return

    logger_instance.info(f"Evaluating {len(models_to_iterate)} model(s).")
    logger_instance.info(f"  Results dir: {results_output_dir}")
    logger_instance.info(f"  Logs dir   : {logs_output_dir}")

    summary_csv_path = results_output_dir / "eval_summary.csv"
    csv_header = ["ModelAlias", "Task", "Metric", "Value", "Version", "NumFewshot", "Limit", "Timestamp", "Error"]
    all_summary_rows: List[List[str]] = []
    base_server_url = f"http://127.0.0.1:{STATIC_EVAL_PORT}"
    health_url = f"{base_server_url}/health"
    eval_api_endpoint = f"{base_server_url}/v1/completions"

    tasks_list = [t.strip() for t in tasks_str.split(',') if t.strip()]
    is_humaneval_like_task_run = any(t in {"humaneval", "mbpp", "multipleefix", "multipl<x_bin_482>"} for t in tasks_list)

    for model_idx, (model_alias, model_data) in enumerate(models_to_iterate.items()):
        logger_instance.subheader(f"Processing Model ({model_idx + 1}/{len(models_to_iterate)}): {model_alias}")
        current_model_output_json_path = results_output_dir / f"{model_alias}_results.json"
        timestamp_str_local = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        eval_status = "Not Run"
        current_run_error_message = ""

        # --- Build server command from config using robust normalizers ---
        try:
            server_executable, server_args_list, _ctx_size_str = build_server_command(
                model_alias=model_alias,
                model_cfg=model_data,
                static_port=STATIC_EVAL_PORT,
                logger=logger_instance,
            )
        except Exception as e_cmd:
            current_run_error_message = f"Cmd Prep Error: {e_cmd}"
            logger_instance.error(f"  Error preparing command for '{model_alias}': {e_cmd}")
            if is_verbose:
                logger_instance.error(f"Traceback:\n{traceback.format_exc()}")
            all_summary_rows.append([model_alias, "-", "-", "-", "-", str(num_fewshot or "Def"),
                                     str(limit_samples or "None"), timestamp_str_local, current_run_error_message])
            continue

        # Optional pretty string for debug
        try:
            config_for_build_util_log = {k: v for k, v in model_data.items() if k not in ["generated_cmd_str"]}
            if "cmd" in config_for_build_util_log:
                cfg_cmd = dict(config_for_build_util_log["cmd"])
                cfg_cmd["port"] = STATIC_EVAL_PORT
                config_for_build_util_log["cmd"] = cfg_cmd
            if "sampling" in model_data:
                config_for_build_util_log["sampling"] = model_data["sampling"]
            config_for_build_util_log["_name_for_log"] = model_alias
            if is_verbose:
                full_cmd_str_for_log = build_llama_server_command_util(config_for_build_util_log)
                logger_instance.debug(f"    Full command string (ref): {full_cmd_str_for_log}")
        except Exception as e:
            logger_instance.debug(f"    Could not render full command string: {e}")

        logger_instance.info("  Server command for evaluation prepared.")
        logger_instance.debug(f"    Popen exec: '{server_executable}'")
        logger_instance.debug(f"    Popen args: {server_args_list}")

        # --- Launch server & wait for health ---
        server_process: Optional[subprocess.Popen] = None
        eval_results_dict: Optional[Dict[str, Any]] = None
        stderr_log_path_for_model: Optional[Path] = None

        try:
            if is_humaneval_like_task_run:
                _ensure_os_unlink_restored(f"Before server start for '{model_alias}'", logger_instance)

            # Use the run's logs directory as the working log dir for server
            server_process_info = start_llama_server(
                server_executable, server_args_list, model_alias,
                logs_output_dir, logger_instance, project_root_dir
            )
            if not server_process_info:
                raise ServerOperationError("Server Start Failed")
            server_process, _, stderr_log_path_for_model = server_process_info

            healthy = wait_for_server_health(
                server_process, health_url, health_timeout_s, health_poll_s,
                model_alias, logger_instance
            )
            if not healthy:
                exit_code_msg = ""
                if server_process is not None:
                    code = server_process.poll()
                    if code is not None:
                        exit_code_msg = f"(Code: {code})"
                _dump_stderr_on_failure(stderr_log_path_for_model, model_alias, logger_instance)
                raise ServerOperationError(f"Server Not Healthy {exit_code_msg}")

            logger_instance.info(f"  Server for '{model_alias}' healthy. Starting lm-eval...")

            # lm-eval model args
            hf_tokenizer_for_model = model_data.get("hf_tokenizer_for_model", model_alias)
            if isinstance(num_concurrent_eval_override, int) and num_concurrent_eval_override > 0:
                num_concurrent_requests = num_concurrent_eval_override
            else:
                per_model_nc = model_data.get("num_concurrent_eval", 1)
                try:
                    num_concurrent_requests = int(per_model_nc)
                    if num_concurrent_requests <= 0:
                        num_concurrent_requests = 1
                except Exception:
                    num_concurrent_requests = 1
            DEFAULT_LM_EVAL_MAX_LENGTH = 4096
            configured_ctx_size = None
            try:
                configured_ctx_size = model_data.get("cmd", {}).get("ctx-size")
            except Exception:
                configured_ctx_size = None
            try:
                lm_eval_max_length = int(configured_ctx_size) if configured_ctx_size else DEFAULT_LM_EVAL_MAX_LENGTH
            except (ValueError, TypeError):
                lm_eval_max_length = DEFAULT_LM_EVAL_MAX_LENGTH

            lm_eval_model_args_list = [
                f"base_url={eval_api_endpoint}",
                f"engine={model_alias}",
                f"model={model_alias}",
                "api_key=EMPTY",
                f"timeout={health_timeout_s}",
                "tokenizer_backend=huggingface",
                f"tokenizer={hf_tokenizer_for_model}",
                "truncate=True",
                "prefix_completions=True",
                f"num_concurrent={num_concurrent_requests}",
                f"max_length={lm_eval_max_length}",
            ]

            # Choose registered model type by alias
            if "qwen3" in model_alias.lower():
                logger_instance.info("  Using Qwen3 custom API model (with optional '/no_think').")
                model_type_for_lm_eval = "custom_qwen3_local_api"
                lm_eval_model_args_list.append(f"model_alias_for_qwen_check={model_alias}")
                lm_eval_model_args_list.append(f"no_prefix_tasks_str={qwen3_no_prefix_tasks_override}")
            else:
                logger_instance.info("  Using llama.cpp compatible local API model.")
                model_type_for_lm_eval = "llama_cpp_compatible_api"

            lm_eval_model_args_str = ",".join(lm_eval_model_args_list)
            logger_instance.debug(f"  lm-eval model_args: {lm_eval_model_args_str}")

            # Batch size
            actual_batch_size_val: Union[int, str] = 1
            if batch_size_str.lower() == "auto":
                actual_batch_size_val = "auto"
            elif batch_size_str.isdigit():
                actual_batch_size_val = int(batch_size_str)

            # Unsafe code tasks (e.g., humaneval)
            confirm_unsafe = bool(is_humaneval_like_task_run)
            if confirm_unsafe:
                logger_instance.warn("  Unsafe tasks detected. confirm_run_unsafe_code=True.")

            if is_humaneval_like_task_run:
                _ensure_os_unlink_restored(f"Before simple_evaluate for '{model_alias}'", logger_instance)

            eval_results_dict = _call_simple_evaluate(
                model=model_type_for_lm_eval,
                model_args=lm_eval_model_args_str,
                tasks=tasks_list,
                num_fewshot=num_fewshot,
                limit=limit_samples,
                batch_size=actual_batch_size_val,
                confirm_run_unsafe_code=confirm_unsafe,
            )

            if is_humaneval_like_task_run:
                _ensure_os_unlink_restored(f"IMMEDIATELY AFTER simple_evaluate for '{model_alias}'", logger_instance)

            # Save JSON
            if eval_results_dict is not None:
                logger_instance.debug(f"  Raw eval_results_dict (first 1k chars): {str(eval_results_dict)[:1000]}...")
                try:
                    current_model_output_json_path.parent.mkdir(parents=True, exist_ok=True)
                    eval_results_processed = preprocess_for_json(eval_results_dict.copy())
                    with open(current_model_output_json_path, 'w', encoding='utf-8') as f_json:
                        json_serializer.dump(eval_results_processed, f_json, indent=2, ensure_ascii=False)
                    logger_instance.info(f"  Detailed results saved to: {current_model_output_json_path}")
                    eval_status = "Success"
                except Exception as json_e:
                    eval_status = "Script Error (JSON Save)"
                    current_run_error_message = f"JSON Save Error: {json_e}"
                    logger_instance.error(f"  Error during JSON save for '{model_alias}': {json_e}")
                    logger_instance.error(f"  Traceback:\n{traceback.format_exc()}")
            else:
                eval_status = "Script Error (No Results)"
                current_run_error_message = "simple_evaluate returned None"
                logger_instance.warn(f"  simple_evaluate returned None for model '{model_alias}'.")

            # Produce CSV rows on success
            if eval_status == "Success":
                logger_instance.success(f"  Evaluation and result saving for '{model_alias}' completed successfully.")
                limit_str_for_csv = str(limit_samples if limit_samples is not None else "None")
                # Defensive extraction of metrics
                results_data = {}
                try:
                    if isinstance(eval_results_dict, dict):
                        results_data = eval_results_dict.get("results", {}) or {}
                except Exception:
                    results_data = {}

                versions = {}
                try:
                    if isinstance(eval_results_dict, dict):
                        versions = eval_results_dict.get("versions", {}) or {}
                except Exception:
                    versions = {}

                for task_name_res, metrics_dict in results_data.items():
                    version = versions.get(task_name_res, "N/A")
                    if isinstance(metrics_dict, dict):
                        for metric_name, metric_value in metrics_dict.items():
                            mname = str(metric_name)
                            if "_stderr" in mname.lower() or "_samples" in mname.lower() or mname == "alias":
                                continue
                            metric_value_str = f"{metric_value:.4f}" if isinstance(metric_value, float) else str(metric_value)
                            all_summary_rows.append([
                                model_alias, task_name_res, mname, metric_value_str, str(version),
                                str(num_fewshot if num_fewshot is not None else "TaskDef"),
                                limit_str_for_csv, timestamp_str_local, ""
                            ])

        except ServerOperationError as e_serv:
            eval_status = f"Failed (Server: {e_serv})"
            current_run_error_message = str(e_serv)
            if is_humaneval_like_task_run:
                _ensure_os_unlink_restored(f"After EXCEPTION (ServerOp) for '{model_alias}'", logger_instance)

        except Exception as e_eval_main:
            eval_status = "Script Error (Eval)"
            current_run_error_message = f"Eval Error: {str(e_eval_main)[:200]}"
            if is_verbose:
                logger_instance.error(f"Full traceback for Eval Error in model '{model_alias}':\n{traceback.format_exc()}")
            if is_humaneval_like_task_run:
                _ensure_os_unlink_restored(f"After EXCEPTION (in simple_evaluate) for '{model_alias}'", logger_instance)

        finally:
            if server_process:
                stop_llama_server(server_process, model_alias, logger_instance)
            if is_humaneval_like_task_run:
                _ensure_os_unlink_restored(f"In finally block after server stop for '{model_alias}'", logger_instance)

        # Log result and append error row if needed
        logger_instance.info(f"  Result for '{model_alias}': {color_status(eval_status)}")
        if current_run_error_message and not eval_status.startswith("Success"):
            all_summary_rows.append([
                model_alias,
                tasks_list[0] if tasks_list else "-",
                "-", "-", "-", "-",
                str(limit_samples if limit_samples is not None else "None"),
                timestamp_str_local,
                current_run_error_message
            ])
        logger_instance.notice("-" * 30)

    # --- Write summary CSV into results dir ---
    logger_instance.header("Writing Evaluation Summary to CSV")
    if not all_summary_rows and models_to_iterate:
        logger_instance.warn(f"No summary data was generated for {len(models_to_iterate)} model(s). CSV might be empty or only headers.")
    try:
        summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_header)
            writer.writerows(all_summary_rows)
        logger_instance.success(f"All evaluation summary results saved to: {summary_csv_path.resolve()}")
    except IOError as e_csv:
        logger_instance.error(f"Failed to write summary CSV to {summary_csv_path}: {e_csv}")
        if is_verbose:
            logger_instance.error(f"Traceback for CSV Write Error:\n{traceback.format_exc()}")

# =============================================================================
# Main
# =============================================================================

def main_eval():
    global logger, TEMP_DIR_MANAGER_PATH_EVAL

    # Multiprocessing start method (prefer 'spawn' for C-ext safety)
    try:
        target_method = 'spawn'
        cur_method = multiprocessing.get_start_method(allow_none=True)
        if cur_method != target_method and target_method in multiprocessing.get_all_start_methods():
            multiprocessing.set_start_method(target_method, force=True)
    except RuntimeError:
        pass
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="LLM Evaluation Script using lm-evaluation-harness.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Defaults reflect new layout
    default_base_config_path = (project_root_dir / "configs" / "config.base.yaml").resolve()
    hostname = platform.node().split('.')[0].lower()
    host_override_candidate = (project_root_dir / "configs" / "overrides" / f"{hostname}.yaml")
    calculated_override: Optional[str] = str(host_override_candidate) if host_override_candidate.exists() else None

    # Base dirs for runs
    base_runs_eval_dir = (project_root_dir / "runs" / "eval").resolve()

    # Use str type in argparse to avoid Path(None) typing issue; convert later.
    parser.add_argument("-c", "--config", type=str, default=str(default_base_config_path),
                        help="Path to base YAML config.")
    parser.add_argument("--override", type=str, default=calculated_override,
                        help="Path to override YAML (host-specific).")

    parser.add_argument("--tasks", type=str, default=DEFAULT_TASKS,
                        help="Comma-separated lm-eval tasks.")
    parser.add_argument("--limit", type=float, default=DEFAULT_LIMIT_SAMPLES,
                        help="Max samples per task (None for all).")
    parser.add_argument("--num-fewshot", type=int, default=DEFAULT_NUM_FEWSHOT,
                        help="Num few-shot examples (task default if None).")
    parser.add_argument("--batch-size", type=str, default=DEFAULT_BATCH_SIZE_EVAL,
                        help="Batch size for lm-eval (e.g., 1, 'auto').")

    parser.add_argument("-o", "--output-dir", type=str, default=str(base_runs_eval_dir),
                        help="Base eval output directory (parent of logs/ and results/).")
    parser.add_argument("-m", "--model", type=str, help="Evaluate only this specific model alias.")

    parser.add_argument("--poll-interval", type=float, default=DEFAULT_HEALTH_POLL_INTERVAL_S,
                        help="Health check poll interval (s).")
    parser.add_argument("--health-timeout", type=int, default=DEFAULT_HEALTH_TIMEOUT_S_EVAL,
                        help="Server health check timeout (s).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose/debug output.")

    parser.add_argument(
        "--qwen3-no-prefix-tasks", type=str, default=DEFAULT_QWEN3_NO_PREFIX_TASKS_STR_CLI,
        help="Semicolon-separated list of task names for Qwen3 models to EXCLUDE from '/no_think' prepending."
    )
    parser.add_argument(
        "--num-concurrent-eval", type=int, default=None,
        help="Global override for per-model 'num_concurrent_eval' (lm-eval 'num_concurrent')."
    )

    args = parser.parse_args()

    # Normalize to Paths safely
    base_config_path = Path(args.config).resolve()
    override_path: Optional[Path] = Path(args.override).resolve() if args.override else None
    runs_eval_parent = Path(args.output_dir).resolve()  # parent of logs/ and results/

    # Initialize logger (non-optional alias for Pylance)
    logger = Logger(verbose=args.verbose)
    log = logger  # local alias with non-optional type

    # Configure lm-eval logging
    log.header("LLM EVALUATOR INITIALIZATION (llama_suite/eval/evaluate_models_evalharness.py)")
    lm_eval_root_logger = std_logging.getLogger("lm_eval")
    lm_eval_root_logger.setLevel(std_logging.DEBUG if args.verbose else std_logging.INFO)
    if not lm_eval_root_logger.hasHandlers():
        stream_handler_eval = std_logging.StreamHandler(sys.stderr)
        formatter_eval = std_logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        stream_handler_eval.setFormatter(formatter_eval)
        lm_eval_root_logger.addHandler(stream_handler_eval)
    lm_eval_root_logger.propagate = False
    log.info(f"Base 'lm_eval' logger level set to: {std_logging.getLevelName(lm_eval_root_logger.level)}")

    # Signal handlers
    signal.signal(signal.SIGINT, signal_cleanup_handler_eval)
    signal.signal(signal.SIGTERM, signal_cleanup_handler_eval)
    log.info("Signal handlers registered.")

    # Config checks
    if not base_config_path.is_file():
        log.error(f"Base config not found: {base_config_path}")
        sys.exit(1)
    log.info(f"Using base configuration: {base_config_path}")
    if override_path:
        if override_path.is_file():
            log.info(f"Using override configuration: {override_path}")
        else:
            log.warn(f"Override config not found: {override_path}. Proceeding without.")
            override_path = None
    else:
        log.info("No override configuration file specified or found by default logic.")

    # Compute run name and concrete dirs:
    run_ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"task-{args.tasks.replace(',', '_')}_{run_ts}"
    logs_dir = runs_eval_parent / "logs" / run_name
    results_dir = runs_eval_parent / "results" / run_name

    # Store globally for signal handler info:
    TEMP_DIR_MANAGER_PATH_EVAL = logs_dir

    # Kill any lingering server on our static port
    log.step(f"Initial cleanup of lingering llama-server on port {STATIC_EVAL_PORT}...")
    killed_procs = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline_list = proc.info.get('cmdline')
            if cmdline_list and f"--port {STATIC_EVAL_PORT}" in " ".join(cmdline_list):
                log.warn(f"  Terminating lingering PID={proc.pid}, Name='{proc.info.get('name', '')}'")
                p_obj = psutil.Process(proc.pid)
                p_obj.terminate()
                try:
                    p_obj.wait(timeout=2)
                except psutil.TimeoutExpired:
                    p_obj.kill()
                    p_obj.wait(timeout=1)
                killed_procs += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except Exception as e_psutil:
            log.warn(f"  Error during psutil check: {e_psutil}")
    if killed_procs > 0:
        log.success(f"Initial cleanup terminated {killed_procs} process(es).")
    else:
        log.info(f"No lingering eval server processes found on port {STATIC_EVAL_PORT}.")

    # Load and process configurations
    log.step("Loading configurations via config_utils...")
    try:
        effective_conf_dict = generate_processed_config(
            base_config_path_arg=base_config_path,
            override_config_path_arg=override_path,
            script_dir_for_overrides=project_root_dir,
            verbose_logging=args.verbose
        )
        log.success("Configurations processed.")
    except Exception as e_conf:
        log.error(f"Failed to load/process configurations: {e_conf}")
        if args.verbose:
            log.error(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(1)

    # Honor environment variables if CLI flag not provided
    effective_num_conc: Optional[int] = args.num_concurrent_eval
    if effective_num_conc is None:
        env_nc = os.getenv("LLS_NUM_CONCURRENT_EVAL") or os.getenv("NUM_CONCURRENT_EVAL")
        if env_nc:
            try:
                parsed = int(env_nc)
                if parsed > 0:
                    effective_num_conc = parsed
            except Exception:
                pass
    if effective_num_conc is not None:
        log.info(f"Global override: num_concurrent_eval = {effective_num_conc}")

    script_start_time = time.monotonic()
    try:
        # Use our run's logs dir (not a tempdir) so logs persist where we expect
        logs_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Logs dir   : {logs_dir}")
        log.info(f"Results dir: {results_dir}")
        log.header("STARTING EVALUATION RUN")

        processed_models_data = effective_conf_dict.get("models", {})
        if not isinstance(processed_models_data, dict) or not processed_models_data:
            log.error("'models' section not found or empty in config.")
            sys.exit(1)

        # Pre-register tasks (ensures lm_eval loads the task modules)
        lm_eval.tasks.get_task_dict(args.tasks.split(','))
        _ensure_os_unlink_restored("After lm_eval.tasks.get_task_dict() in main", log)

        run_evaluation(
            processed_models_config=processed_models_data,
            results_output_dir=results_dir,
            logs_output_dir=logs_dir,
            tasks_str=args.tasks,
            limit_samples=args.limit,
            num_fewshot=args.num_fewshot,
            batch_size_str=args.batch_size,
            health_timeout_s=args.health_timeout,
            health_poll_s=args.poll_interval,
            model_to_eval_alias=args.model,
            logger_instance=log,
            qwen3_no_prefix_tasks_override=args.qwen3_no_prefix_tasks,
            num_concurrent_eval_override=effective_num_conc,
        )
    finally:
        _ensure_os_unlink_restored("At the very end of main execution (eval)", log)
        script_duration_s = time.monotonic() - script_start_time
        log.header("EVALUATION SCRIPT COMPLETE")
        log.success(f"Total script execution time: {script_duration_s:.2f} seconds.")
        log.close()

if __name__ == "__main__":
    # Prefer 'spawn' on non-macOS for C-extension safety
    if platform.system() != "Darwin":
        try:
            if multiprocessing.get_start_method(allow_none=True) != 'spawn' and \
               'spawn' in multiprocessing.get_all_start_methods():
                multiprocessing.set_start_method('spawn', force=True)
                print("INFO: Set multiprocessing start method to 'spawn'.", flush=True)
        except RuntimeError as e:
            print(f"WARNING: Could not set multiprocessing start method to 'spawn': {e}. Using default.", flush=True)
    else:
        print(f"INFO: Using default multiprocessing start method for macOS ('{multiprocessing.get_start_method(allow_none=True)}').", flush=True)

    main_eval()
