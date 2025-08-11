#!/usr/bin/env python3
"""
LLM Model Memory Scanner Script.
Uses config_utils module to load configurations and manage server processes.
Starts each specified model, waits for it to become healthy,
parses memory usage from its logs, and then stops it.
Outputs results to CSV and console.
Uses a static port for running llama-server instances.
"""

import argparse
import csv
import datetime # Needed for main script's timestamping, even if Logger has its own
import platform
# shlex is not directly used here anymore for splitting command, but good to have for Popen context
# import shlex 
import signal
import subprocess # Still needed for Popen type hint if not fully abstracted
import sys
import tempfile
import time
import re # For parsing memory string
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TextIO, List # Added List
import shutil

# --- Third-Party Library Imports ---
import psutil
# requests is used by wait_for_server_health in config_utils
# from colorama import Fore, Style, init as colorama_init # Handled by Logger in config_utils

# --- Global Variables ---
# Logger will be initialized in main
logger: Optional['Logger'] = None # Logger type from config_utils
TEMP_DIR_MANAGER_PATH: Optional[Path] = None
project_root_dir: Path # Will be defined in main

# --- Import from shared config utility ---
try:
    # Determine project root dynamically for sys.path modification
    _current_script_file_path_for_root_scan = Path(__file__).resolve()
    project_root_dir = _current_script_file_path_for_root_scan.parent.parent # Assuming script is in bench/
    if str(project_root_dir) not in sys.path:
        sys.path.insert(0, str(project_root_dir))

    from utils.config_utils import (
        Logger, generate_processed_config, build_llama_server_command_util,
        start_llama_server, stop_llama_server, wait_for_server_health,
        _dump_stderr_on_failure, color_status, # _resolve_executable_path_robustly is used internally by start_server
        PROCESS_TERMINATE_TIMEOUT_S, DEFAULT_HEALTH_POLL_INTERVAL_S
    )
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import from config_utils.py. Ensure it's in project_root/utils.")
    print(f"  Attempted to add '{project_root_dir if 'project_root_dir' in locals() else 'UNKNOWN'}' to PYTHONPATH.")
    print(f"  Current PYTHONPATH: {sys.path}")
    print(f"  Details: {e}")
    sys.exit(1)

# --- Constants specific to this script ---
DEFAULT_OUTPUT_FILENAME = "memory_scan_results.csv"
DEFAULT_HEALTH_TIMEOUT_S_SCAN = 60 # scan_model_memory specific default
STATIC_SERVER_PORT_SCAN = "9998"

# Keys in the model's root config (not in 'cmd' or 'sampling') that are meta-data
# and should NOT be passed as CLI arguments to llama-server by this script's logic.
# build_llama_server_command_util in config_utils has its own filtering for its purpose.
# This list is for run_memory_scan's argument reconstruction.
MODEL_CONFIG_META_KEYS_SCAN = {
    "aliases", "sampling", "_name_for_log", "generated_cmd_str",
    # Add any custom keys from your YAML specific to scan_model_memory context here
    # e.g., if scan_model_memory uses some model-level flags that aren't for llama-server
    "hf_tokenizer_for_model", # Example, if it was at root for some reason
    "supports_no_think_toggle" # Example
}

SCRIPT_DIR = Path(__file__).resolve().parent
LOGS_DIR = SCRIPT_DIR / "logs"
CSV_BASENAME = "scan\scan_results"
RUN_DIR_PREFIX = "run_"
RETENTION_KEEP = 10  # keep last 10 CSVs and log runs

def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def _enforce_retention(dir_path: Path, pattern: str, keep: int, delete_dirs: bool = False, logger_instance: Optional['Logger'] = None):
    try:
        items = sorted(dir_path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in items[keep:]:
            if delete_dirs and old.is_dir():
                shutil.rmtree(old, ignore_errors=True)
                if logger_instance: logger_instance.debug(f"Retention: removed directory {old}")
            elif old.is_file():
                old.unlink(missing_ok=True)
                if logger_instance: logger_instance.debug(f"Retention: removed file {old}")
    except Exception as e:
        if logger_instance:
            logger_instance.warn(f"Retention error in {dir_path} ({pattern}): {e}")

def _resolve_llama_server_executable(path_str: str, logger_instance: 'Logger') -> str:
    """
    Tries to resolve the actual llama-server executable if the configured path doesn't exist.
    On Windows, it also checks common build subfolders like 'Release', 'RelWithDebInfo', etc.
    Returns the best candidate path (string). If nothing is found, returns the original path_str.
    """
    p = Path(path_str).expanduser()
    if p.is_file():
        return str(p)

    # If the provided path is a directory, we expect the executable inside it
    # otherwise, treat parent as the search root.
    search_roots = [p] if p.is_dir() else [p.parent]

    exe_name = p.name if p.name else "llama-server"
    is_windows = platform.system().lower().startswith("win")

    # Make sure we consider .exe variant on Windows
    names = [exe_name]
    if is_windows and not exe_name.lower().endswith(".exe"):
        names.append(exe_name + ".exe")

    # Common build subfolders to try
    subdirs = ["", "Release", "RelWithDebInfo", "MinSizeRel", "Debug"]

    # Direct candidates first (fast path)
    candidates = []
    for root in search_roots:
        for sub in subdirs:
            for name in names:
                candidate = (root / sub / name) if sub else (root / name)
                candidates.append(candidate)

    # Fallback: rglob under roots (still scoped to bin/build trees; cheap enough)
    for root in search_roots:
        for name in set(names):
            try:
                for hit in root.rglob(name):
                    candidates.append(hit)
            except Exception:
                # Ignore permission errors etc.
                pass

    for c in candidates:
        if c.is_file():
            logger_instance.debug(f"    Resolved llama-server executable candidate: {c}")
            return str(c)

    # Nothing found; return original
    return path_str

# --- Data Extraction (Specific to Memory Scan) ---
def parse_memory_string_to_gb(mem_str: str) -> Optional[float]:
    mem_str = mem_str.strip()
    match = re.match(r"([0-9]+(?:[.,][0-9]+)?)\s*([KMGT])(?:i?B)?", mem_str, re.IGNORECASE)
    if not match: return None
    value_str, unit_prefix = match.groups()
    try: value = float(value_str.replace(',', '.'))
    except ValueError: return None
    unit_prefix = unit_prefix.upper()
    multipliers = {'K': 1/1024/1024, 'M': 1/1024, 'G': 1, 'T': 1024}
    multiplier = multipliers.get(unit_prefix)
    return value * multiplier if multiplier is not None else None

def parse_memory_from_log(stderr_log_path: Path, model_name: str, logger_instance: Logger) -> Tuple[str, str, str]:
    total_gpu_gb, total_cpu_gb = 0.0, 0.0
    lines_found, parse_errors = 0, 0
    final_status = "Failed (Unknown)"

    if not stderr_log_path.exists():
        logger_instance.warn(f"    Memory log not found for '{model_name}': {stderr_log_path}")
        return "0.00", "0.00", "Failed (No Log)"
    logger_instance.debug(f"    Parsing memory from log: {stderr_log_path}")
    # Regex to find lines like "ggml_vk_CUDA buffer size = ..." or "Metal buffer size = ..." or "CPU buffer size = ..."
    buffer_regex = re.compile(r"(ggml_vk_)?(Metal|CUDA|CPU).*?buffer size\s*=\s*([0-9.,]+\s*[KMGT]i?B)", re.IGNORECASE)
    time.sleep(0.2) # Give a very short moment for logs to flush, might not be needed.
    try:
        with stderr_log_path.open('r', encoding='utf-8', errors='replace') as f:
            for line_num, line in enumerate(f, 1):
                match = buffer_regex.search(line)
                if match:
                    lines_found += 1
                    # group(1) is optional prefix like "ggml_vk_"
                    # group(2) is device_type ("Metal", "CUDA", "CPU")
                    # group(3) is mem_str ("123.45 MiB")
                    device_type, mem_str = match.group(2), match.group(3)
                    logger_instance.debug(f"      Found buffer on line {line_num}: Device='{device_type}', MemStr='{mem_str}'")
                    mem_gb = parse_memory_string_to_gb(mem_str)
                    if mem_gb is not None:
                        if device_type.upper() in ["METAL", "CUDA"]: total_gpu_gb += mem_gb
                        elif device_type.upper() == "CPU": total_cpu_gb += mem_gb
                    else:
                        logger_instance.warn(f"      Could not parse memory value: '{mem_str}' in line: {line.strip()}")
                        parse_errors += 1
    except Exception as e:
        logger_instance.warn(f"    Error reading or parsing memory log file {stderr_log_path}: {e}")
        return "0.00", "0.00", "Failed (Log Read Error)"

    if lines_found > 0: final_status = "Parse Error" if parse_errors > 0 else "Success"
    else:
        logger_instance.warn(f"    No 'buffer size' lines found in log for '{model_name}'.")
        final_status = "Failed (No Buffers)" # More specific status
    gpu_str, cpu_str = f"{total_gpu_gb:.2f}", f"{total_cpu_gb:.2f}"
    return gpu_str, cpu_str, final_status

# --- Main Memory Scan Logic ---
def run_memory_scan(
    processed_models_config: Dict[str, Any],
    output_csv_path: Path,
    health_timeout_s: int,
    health_poll_s: float,
    model_to_scan_alias: Optional[str],
    temp_dir_path: Path,
    logger_instance: Logger # Expecting Logger from config_utils
):
    # is_verbose for this function can be taken from logger_instance.verbose_flag
    is_verbose = logger_instance.verbose_flag

    if not processed_models_config:
        logger_instance.error("No processed model configurations provided for memory scan.")
        return

    models_to_iterate = {}
    if model_to_scan_alias:
        if model_to_scan_alias in processed_models_config:
            model_entry = processed_models_config[model_to_scan_alias]
            if isinstance(model_entry, dict): models_to_iterate = {model_to_scan_alias: model_entry}
            else:
                logger_instance.error(f"Config for specified model '{model_to_scan_alias}' is malformed. Skipping.")
                return
        else:
            logger_instance.error(f"Specified model '{model_to_scan_alias}' not found.")
            logger_instance.info("Available models: " + ", ".join(processed_models_config.keys()))
            return
    else:
        models_to_iterate = {
            alias: data for alias, data in processed_models_config.items() if isinstance(data, dict)
        }
        if not models_to_iterate:
            logger_instance.error("No valid model configurations found to scan.")
            return
        logger_instance.info(f"Scanning memory for all {len(models_to_iterate)} processed models.")

    csv_header = ["ModelName", "Timestamp", "ScanStatus", "GpuMemoryGB", "CpuMemoryGB", "Error"]
    all_results_data = []
    
    base_proxy_url = f"http://127.0.0.1:{STATIC_SERVER_PORT_SCAN}"
    health_url_template = f"{base_proxy_url}/health"

    for model_idx, (model_alias, model_data) in enumerate(models_to_iterate.items()):
        logger_instance.subheader(f"Scanning Model ({model_idx + 1}/{len(models_to_iterate)}): {model_alias}")

        gpu_gb, cpu_gb, scan_status = "-", "-", "Not Scanned"
        error_message = ""
        timestamp = logger_instance._get_timestamp()

        logger_instance.step(f"Preparing server command for '{model_alias}'")
        
        original_cmd_dict_from_model_data = model_data.get("cmd")
        if not isinstance(original_cmd_dict_from_model_data, dict):
            logger_instance.warn(f"  '{model_alias}' missing 'cmd' dictionary. Skipping.")
            scan_status, error_message = "Config Error", "Missing 'cmd' dict"
            all_results_data.append([model_alias, timestamp, scan_status, gpu_gb, cpu_gb, error_message])
            continue

        # This is the 'cmd' dict that will be used to reconstruct arguments,
        # with the port overridden for this scan.
        cmd_options_for_scan = original_cmd_dict_from_model_data.copy()
        cmd_options_for_scan["port"] = STATIC_SERVER_PORT_SCAN
        
        # For logging the full command string (optional, for reference)
        config_for_build_util_log = {
            k: v for k, v in model_data.items() if k not in ["cmd", "generated_cmd_str"]
        }
        config_for_build_util_log["cmd"] = cmd_options_for_scan
        if "sampling" in model_data:
             config_for_build_util_log["sampling"] = model_data["sampling"]
        config_for_build_util_log["_name_for_log"] = model_alias


        server_executable = ""
        server_args_list: List[str] = [] 

        try:
            resolved_bin_path_from_config = cmd_options_for_scan.get("bin")
            if not resolved_bin_path_from_config or not isinstance(resolved_bin_path_from_config, str):
                logger_instance.error(f"  Could not retrieve 'bin' path for '{model_alias}'. Skipping.")
                # ... (handle error and continue)
                all_results_data.append([model_alias, timestamp, "Config Error", "-", "-", "Missing 'bin' in cmd dict"])
                continue
            
            server_executable = resolved_bin_path_from_config
            logger_instance.debug(f"  Using server_executable (from resolved config): '{server_executable}'")

            # NEW: resolve alternate locations (e.g., build/bin/Release on Windows)
            resolved_exec = _resolve_llama_server_executable(server_executable, logger_instance)
            if resolved_exec != server_executable:
                logger_instance.info(f"  Adjusted server executable path -> {resolved_exec}")
            server_executable = resolved_exec

            # Guard: ensure it exists before trying to start
            if not Path(server_executable).is_file():
                logger_instance.error(f"  Executable not found at '{server_executable}'. Tried common build subfolders too.")
                all_results_data.append([model_alias, timestamp, "Server Start Failed", "-", "-", "Executable not found"])
                logger_instance.notice("-" * 30)
                continue

            # For reference logging only:
            if is_verbose:
                full_cmd_str_for_log = build_llama_server_command_util(config_for_build_util_log)
                logger_instance.debug(f"    Full command string (for reference) from build_util: {full_cmd_str_for_log}")

            # Reconstruct server_args_list from cmd_options_for_scan and other model_data parts
            # Mandatory args from cmd_options_for_scan
            server_args_list.extend(["--port", str(cmd_options_for_scan.get("port"))])
            server_args_list.extend(["--model", str(cmd_options_for_scan.get("model"))])
            server_args_list.extend(["--ctx-size", str(cmd_options_for_scan.get("ctx-size"))])

            if str(cmd_options_for_scan.get("gpu-layers", "auto")).lower() != "auto":
                server_args_list.extend(["--n-gpu-layers", str(cmd_options_for_scan.get("gpu-layers"))])
            if str(cmd_options_for_scan.get("threads", "auto")).lower() != "auto":
                server_args_list.extend(["--threads", str(cmd_options_for_scan.get("threads"))])
            
            handled_cmd_keys = {"bin", "port", "model", "ctx-size", "gpu-layers", "threads"}
            for key, value in cmd_options_for_scan.items():
                if key in handled_cmd_keys: continue
                cli_flag = f"--{key.replace('_', '-')}"
                if isinstance(value, bool) and value: server_args_list.append(cli_flag)
                elif value not in (None, False, "auto", "Auto", ""): server_args_list.extend([cli_flag, str(value)])
            
            # Args from model_data root (excluding 'cmd' and meta keys)
            for key, value in model_data.items(): # model_data is config_for_command_build without 'cmd' and 'sampling' handled separately
                if key == "cmd" or key in MODEL_CONFIG_META_KEYS_SCAN: continue
                cli_flag = f"--{key.replace('_', '-')}"
                if isinstance(value, bool) and value: server_args_list.append(cli_flag)
                elif value not in (None, False, "auto", "Auto", ""): server_args_list.extend([cli_flag, str(value)])

            sampling_conf = model_data.get("sampling", {}) # Get from original model_data
            if isinstance(sampling_conf, dict):
                for key, s_value in sampling_conf.items():
                    cli_flag = f"--{key.replace('_', '-')}"
                    server_args_list.extend([cli_flag, str(s_value)])
            
            logger_instance.info(f"  Server command for memory scan prepared.")
            logger_instance.debug(f"    Final server_executable for Popen: '{server_executable}'")
            logger_instance.debug(f"    Final server_args_list for Popen: {server_args_list}")

        except Exception as e:
            logger_instance.error(f"  Could not build/parse server command for '{model_alias}': {e}.")
            if is_verbose: 
                import traceback
                logger_instance.error(f"Traceback:\n{traceback.format_exc()}")
            scan_status, error_message = "Cmd Build Error", f"Cmd Build/Parse Error: {e}"
            all_results_data.append([model_alias, timestamp, scan_status, gpu_gb, cpu_gb, error_message])
            continue
        
        logger_instance.step(f"Initiating memory scan for '{model_alias}'")
        server_p: Optional[subprocess.Popen] = None
        stderr_log_path_for_model: Optional[Path] = None # Renamed to avoid conflict with _dump_stderr_on_failure param name
        
        # project_root_dir is global, defined in main via _current_script_file_path_for_root_scan
        server_process_info = start_llama_server(
            server_executable, server_args_list, model_alias, temp_dir_path, 
            logger_instance, project_root_dir # Pass project_root_dir
        )

        if server_process_info:
            server_p, _, stderr_log_path_for_model = server_process_info # Assign to renamed var
            healthy = wait_for_server_health(
                server_p, health_url_template, health_timeout_s, health_poll_s, 
                model_alias, logger_instance
            )
            
            if healthy:
                logger_instance.info(f"  Server healthy, parsing memory usage from logs for '{model_alias}'...")
                # Ensure stderr_log_path_for_model is not None before passing
                if stderr_log_path_for_model:
                    gpu_gb, cpu_gb, scan_status = parse_memory_from_log(stderr_log_path_for_model, model_alias, logger_instance)
                else:
                    scan_status, error_message = "Internal Error", "stderr_log_path_for_model was None after server start"
                    logger_instance.error(f"  {error_message} for model '{model_alias}'")

            else: # Not healthy
                if server_p.poll() is not None: 
                    exit_code = server_p.returncode
                    scan_status = f"Server Exited (Code: {exit_code})"
                    error_message = f"Server (PID {server_p.pid}) exited (code {exit_code}) during/after health check."
                    logger_instance.warn(f"  {error_message}")
                    _dump_stderr_on_failure(stderr_log_path_for_model, model_alias, logger_instance)
                else: 
                    scan_status = "Health Timeout"
                    error_message = f"Server (PID {server_p.pid}) health check timed out."
                    logger_instance.warn(f"  {error_message} Process still running, will be stopped.")
            stop_llama_server(server_p, model_alias, logger_instance)
        else: # start_llama_server failed
            scan_status = "Server Start Failed"
            error_message = "Server process failed to start (start_llama_server returned None)."
            # start_llama_server already logged details

        logger_instance.info(f"  Memory Scan Result for '{model_alias}': GPU {color_status(gpu_gb + ' GB')}, CPU {color_status(cpu_gb + ' GB')} - Status: {color_status(scan_status)}")
        if error_message and scan_status not in ["Success", "Parse Error", "Failed (No Buffers)"]:
             logger_instance.warn(f"    Details: {error_message}")

        all_results_data.append([model_alias, timestamp, scan_status, gpu_gb, cpu_gb, error_message])
        logger_instance.notice("-" * 30)

    logger_instance.header("Writing Memory Scan Results to CSV")
    try:
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_header)
            writer.writerows(all_results_data)
        logger_instance.success(f"All memory scan results saved to: {output_csv_path.resolve()}")
    except IOError as e:
        logger_instance.error(f"Failed to write CSV data to {output_csv_path}: {e}")
        if is_verbose: import traceback; logger_instance.error(f"Traceback for CSV write error:\n{traceback.format_exc()}")


# --- Cleanup and Main Execution ---
def signal_cleanup_handler_scan(signum, frame): # Renamed for clarity
    global logger, TEMP_DIR_MANAGER_PATH # Use script-specific globals
    log_func = print if not logger else logger.warn
    log_func(f"\nSignal {signal.Signals(signum).name} received. Initiating cleanup for scan_model_memory...")
    if TEMP_DIR_MANAGER_PATH: log_func(f"  Note: Temporary directory {TEMP_DIR_MANAGER_PATH} should be auto-cleaned if using 'with'.")
    # Add specific cleanup for scan script if any lingering servers need to be killed by port
    # For now, relying on individual server stops and TemporaryDirectory cleanup.
    if logger: logger.info("Memory scan script terminated by signal.")
    else: print("Memory scan script terminated by signal.")
    sys.exit(1)

def main_scan():
    global logger, TEMP_DIR_MANAGER_PATH, project_root_dir

    parser = argparse.ArgumentParser(
        description="LLM Model Memory Scanner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    default_base_config_path = (project_root_dir / "config.base.yaml").resolve()
    parser.add_argument("-c", "--config", type=Path, default=default_base_config_path,
                        help="Path to the base YAML configuration file.")
    default_override_path_hostname = (project_root_dir / "overrides" / f"{platform.node().split('.')[0].lower()}.yaml").resolve()
    calculated_default_override = default_override_path_hostname if default_override_path_hostname.exists() else None
    parser.add_argument("--override", type=Path, default=calculated_default_override,
                        help="Path to override YAML. (Default logic checks hostname based override)")
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_HEALTH_POLL_INTERVAL_S,
                        help="Health check poll interval (seconds).")
    parser.add_argument("--health-timeout", type=int, default=DEFAULT_HEALTH_TIMEOUT_S_SCAN,
                        help=f"Override health check timeout (seconds). Default: {DEFAULT_HEALTH_TIMEOUT_S_SCAN}s.")
    parser.add_argument("-m", "--model", type=str, help="Scan only this specific model alias (from config).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose/debug output.")

    args = parser.parse_args()

    # --- Forced static locations with timestamp ---
    ts = _timestamp()

    # Ensure logs dir exists
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # CSV beside script, timestamped; keep ALL CSVs
    output_file_abs_csv = (SCRIPT_DIR / f"{CSV_BASENAME}_{ts}.csv").resolve()
    try:
        output_file_abs_csv.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"CRITICAL ERROR: Could not create directory for CSV file: {output_file_abs_csv.parent} - {e}", file=sys.stderr)
        sys.exit(1)

    # Main script log inside logs/, timestamped; logs will be rotated
    main_log_file_path = (LOGS_DIR / f"scan_{ts}.log").resolve()
    try:
        main_log_file_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"CRITICAL ERROR: Could not create directory for main log file: {main_log_file_path.parent} - {e}", file=sys.stderr)
        sys.exit(1)

    # Initialize logger
    logger = Logger(verbose=args.verbose, log_file_path=main_log_file_path)

    try:
        logger.header("LLM MODEL MEMORY SCANNER INITIALIZATION")
        logger.info(f"Main script log file: {main_log_file_path}")

        signal.signal(signal.SIGINT, signal_cleanup_handler_scan)
        signal.signal(signal.SIGTERM, signal_cleanup_handler_scan)
        logger.info("Signal handlers registered.")

        resolved_base_config = args.config.resolve()
        if not resolved_base_config.is_file():
            logger.error(f"Base configuration file not found: {resolved_base_config}")
            sys.exit(1)
        logger.info(f"Using base configuration: {resolved_base_config}")

        resolved_override = args.override.resolve() if args.override else None
        if resolved_override:
            if resolved_override.is_file():
                logger.info(f"Using override configuration: {resolved_override}")
            else:
                logger.warn(f"Specified override configuration not found: {resolved_override}. Proceeding without.")
                resolved_override = None
        else:
            logger.info("No override configuration file specified or found.")

        # --- Initial cleanup of potentially lingering llama-server processes on our static port ---
        logger.step(f"Performing initial cleanup of potentially lingering llama-server processes on port {STATIC_SERVER_PORT_SCAN}...")
        killed_procs = 0
        target_port_arg_scan = f"--port {STATIC_SERVER_PORT_SCAN}"

        llama_exec_patterns = ["llama-server", "server.exe", "server"]  # common names
        llama_path_patterns = ["llama.cpp/server", "llama.cpp/build/bin/server", "build/bin/server"]

        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'exe', 'username']):
            try:
                proc_info = proc.info
                pid = proc_info.get('pid')
                proc_name_raw = proc_info.get('name')
                proc_name = proc_name_raw.lower() if proc_name_raw else ""
                exe_path_raw = proc_info.get('exe')
                exe_path = exe_path_raw.lower() if exe_path_raw else ""
                cmdline_list = proc_info.get('cmdline')
                cmdline_str = " ".join(cmdline_list).lower() if cmdline_list else ""
                username = proc_info.get('username')
                is_potential_llama_server = False

                if target_port_arg_scan in cmdline_str:
                    is_potential_llama_server = True
                elif proc_name in llama_exec_patterns:
                    is_potential_llama_server = True
                elif any(patt in exe_path for patt in llama_path_patterns):
                    is_potential_llama_server = True
                elif any(patt in cmdline_str for patt in llama_path_patterns):
                    is_potential_llama_server = True

                if is_potential_llama_server:
                    current_user = psutil.Process().username()
                    if username and username != current_user and "root" in (username.lower() if username else "") and target_port_arg_scan not in cmdline_str:
                        logger.debug(f"  PID {pid} ('{proc_name_raw}') matched pattern but owned by '{username}' (not current user '{current_user}' or root on target port). Skipping kill.")
                        continue

                    system_daemon_names = ["windowserver", "systemuiserver", "cvmserver", "nfstorageserver", "powerd", "logd"]
                    if proc_name in system_daemon_names and target_port_arg_scan not in cmdline_str:
                        logger.debug(f"  PID {pid} ('{proc_name_raw}') is a known system daemon not on target port. Skipping kill.")
                        continue

                    cmd_display_str = (" ".join(cmdline_list))[:100] if cmdline_list else ""
                    logger.warn(f"  Terminating potential lingering server: PID={pid}, Name='{proc_name_raw}', User='{username}', Cmd='{cmd_display_str}'")
                    p_obj = psutil.Process(pid)
                    p_obj.terminate()
                    try:
                        p_obj.wait(timeout=PROCESS_TERMINATE_TIMEOUT_S / 2.0)
                    except psutil.TimeoutExpired:
                        logger.warn(f"    PID {pid} did not terminate, killing...")
                        p_obj.kill()
                        p_obj.wait(timeout=PROCESS_TERMINATE_TIMEOUT_S / 2.0)
                    killed_procs += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                logger.debug(f"  Skipping inaccessible/gone process PID {pid if 'pid' in locals() and pid else 'unknown'}")
                continue
            except Exception as e_psutil:
                logger.warn(f"  Error during psutil check for process {pid if 'pid' in locals() and pid else 'unknown'}: {e_psutil}")

        if killed_procs > 0:
            logger.success(f"Initial cleanup terminated {killed_procs} process(es).")
        else:
            logger.info(f"No lingering server processes found (port {STATIC_SERVER_PORT_SCAN} or patterns).")

        # --- Load and process configurations ---
        logger.step("Loading and processing configurations...")
        try:
            effective_conf_dict = generate_processed_config(
                base_config_path_arg=resolved_base_config,
                override_config_path_arg=resolved_override,
                script_dir_for_overrides=project_root_dir,
                verbose_logging=args.verbose
            )
            logger.success("Configurations processed successfully.")
        except Exception as e:
            logger.error(f"Failed to load or process configurations: {e}")
            if args.verbose:
                import traceback
                logger.error("Traceback:\n" + traceback.format_exc())
            sys.exit(1)

        # Health timeout selection
        health_timeout_final = (
            args.health_timeout
            if args.health_timeout is not None and args.health_timeout != DEFAULT_HEALTH_TIMEOUT_S_SCAN
            else effective_conf_dict.get("healthCheckTimeout", DEFAULT_HEALTH_TIMEOUT_S_SCAN)
        )
        if not isinstance(health_timeout_final, int) or health_timeout_final <= 0:
            logger.warn(f"Invalid healthCheckTimeout '{health_timeout_final}'. Using default: {DEFAULT_HEALTH_TIMEOUT_S_SCAN}s.")
            health_timeout_final = DEFAULT_HEALTH_TIMEOUT_S_SCAN
        logger.info(f"Effective health check timeout: {health_timeout_final}s")
        logger.info(f"Health check poll interval: {args.poll_interval}s")

        # --- Persistent per-run server logs under logs/run_<ts> ---
        run_logs_dir = LOGS_DIR / f"{RUN_DIR_PREFIX}{ts}"
        run_logs_dir.mkdir(parents=True, exist_ok=True)
        TEMP_DIR_MANAGER_PATH = run_logs_dir  # for signal handler info

        logger.info(f"Server logs will be kept in: {run_logs_dir.resolve()}")
        logger.header("STARTING MEMORY SCAN RUN")

        processed_models_data = effective_conf_dict.get("models", {})
        if not isinstance(processed_models_data, dict) or not processed_models_data:
            logger.error("'models' section not found/empty in config. Cannot run scan.")
            sys.exit(1)

        script_start_time = time.monotonic()

        run_memory_scan(
            processed_models_config=processed_models_data,
            output_csv_path=output_file_abs_csv,
            health_timeout_s=health_timeout_final,
            health_poll_s=args.poll_interval,
            model_to_scan_alias=args.model,
            temp_dir_path=run_logs_dir,        # ALWAYS under logs/
            logger_instance=logger
        )

        # Keep a convenience "latest" CSV copy without timestamp
        try:
            latest_csv = SCRIPT_DIR / f"{CSV_BASENAME}.csv"
            shutil.copy2(output_file_abs_csv, latest_csv)
            logger.info(f"Also wrote latest CSV copy: {latest_csv}")
        except Exception as e:
            logger.warn(f"Could not write latest CSV copy: {e}")

        # --- Retention: rotate ONLY logs (keep last RETENTION_KEEP) ---
        _enforce_retention(LOGS_DIR, "scan_*.log", keep=RETENTION_KEEP, delete_dirs=False, logger_instance=logger)
        _enforce_retention(LOGS_DIR, f"{RUN_DIR_PREFIX}*", keep=RETENTION_KEEP, delete_dirs=True, logger_instance=logger)
        # Note: CSVs are NOT rotated and are kept indefinitely by design.

        script_duration_s = time.monotonic() - script_start_time
        logger.header("MEMORY SCAN SCRIPT COMPLETE")
        logger.success(f"Total script execution time: {script_duration_s:.2f} seconds.")

    except Exception as e_main_global:
        if logger:
            logger.error(f"A critical unexpected error occurred in main execution: {e_main_global}")
            if args.verbose or (hasattr(logger, 'verbose_flag') and logger.verbose_flag):
                import traceback
                logger.error("Traceback:\n" + traceback.format_exc())
        else:
            print(f"CRITICAL UNCAUGHT EXCEPTION (Logger not available or failed): {e_main_global}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
        sys.exit(2)
    finally:
        if logger and hasattr(logger, 'close') and callable(logger.close):
            logger.info("Closing main log file at script completion.")
            logger.close()
        elif logger:
            print("Warning: Logger was initialized but does not have a close method or it's not callable.", file=sys.stderr)



if __name__ == "__main__":
    main_scan()