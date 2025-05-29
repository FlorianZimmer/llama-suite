#!/usr/bin/env python3
"""
LLM Model Memory Scanner Script.

Uses config_utils module to load configurations.
Starts each specified model, waits for it to become healthy,
parses memory usage from its logs, and then stops it.
Outputs results to CSV and console.
Uses a static port for running llama-server instances.
"""

import argparse
import csv
import datetime
import platform
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TextIO

# --- Dependencies ---
import psutil
import requests
from colorama import Fore, Style, init as colorama_init

# --- Global Logger ---
logger: Optional['Logger'] = None

# --- Import from shared config utility ---
try:
    current_script_dir = Path(__file__).resolve().parent
    project_root_dir = current_script_dir.parent
    if str(project_root_dir) not in sys.path:
        sys.path.insert(0, str(project_root_dir))

    from utils.config_utils import generate_processed_config, build_llama_server_command_util
except ImportError as e:
    print(f"Error: Could not import from config_utils. Make sure config_utils.py is accessible in project_root/utils.")
    print(f"Attempted to add '{project_root_dir}' to PYTHONPATH.")
    print(f"Current PYTHONPATH: {sys.path}")
    print(f"Details: {e}")
    sys.exit(1)

# --- Constants ---
DEFAULT_OUTPUT_FILENAME = "memory_scan_results.csv"
DEFAULT_HEALTH_POLL_INTERVAL_S = 2.0
DEFAULT_HEALTH_TIMEOUT_S = 60
PROCESS_TERMINATE_TIMEOUT_S = 10
STATIC_SERVER_PORT = "9998" # Using a different static port than benchmark script to avoid conflict if run concurrently for some reason

# --- Logger Class (Copied from benchmark script for consistency) ---
class Logger:
    def __init__(self, verbose: bool = False):
        self.verbose_flag = verbose
        colorama_init(autoreset=True)

    def _get_timestamp(self) -> str:
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def _log(self, level: str, message: str, color: str = "", bright: bool = False, file: TextIO = sys.stdout):
        style_prefix = Style.BRIGHT if bright else ""
        timestamp = self._get_timestamp()
        print(f"{timestamp} [{style_prefix}{color}{level}{Style.RESET_ALL}] {style_prefix}{color}{message}{Style.RESET_ALL}", file=file)

    def info(self, message: str): self._log("INFO", message, Fore.BLUE)
    def warn(self, message: str): self._log("WARN", message, Fore.YELLOW, bright=True)
    def error(self, message: str): self._log("ERROR", message, Fore.RED, bright=True, file=sys.stderr)
    def success(self, message: str): self._log("SUCCESS", message, Fore.GREEN, bright=True)
    def debug(self, message: str):
        if self.verbose_flag:
            timestamp = self._get_timestamp()
            print(f"{timestamp} [{Fore.MAGENTA}VERBOSE{Style.RESET_ALL}] {message}", file=sys.stderr)
    def header(self, title: str): print(f"\n{Fore.CYAN}{Style.BRIGHT}{'=' * 70}\n{title.center(70)}\n{'=' * 70}{Style.RESET_ALL}")
    def subheader(self, title: str): print(f"\n{Fore.CYAN}{'-' * 70}\n{title.center(70)}\n{'-' * 70}{Style.RESET_ALL}")
    def step(self, message: str): print(f"{Fore.CYAN}>> {message}{Style.RESET_ALL}")
    def notice(self, message: str): print(f"{Fore.WHITE}{message}{Style.RESET_ALL}")

# --- Utility Functions (Copied/Adapted) ---
def color_status(status: str) -> str:
    status_lower = status.lower()
    if "success" in status_lower: return f"{Fore.GREEN}{Style.BRIGHT}{status}{Style.RESET_ALL}"
    elif any(term in status_lower for term in ["fail", "error", "timeout", "invalid", "missing", "exited"]): return f"{Fore.RED}{Style.BRIGHT}{status}{Style.RESET_ALL}"
    elif any(term in status_lower for term in ["warn", "not run", "not scanned", "parse error", "no buffers"]): return f"{Fore.YELLOW}{Style.BRIGHT}{status}{Style.RESET_ALL}"
    return status

def _dump_stderr_on_failure(stderr_log_path: Optional[Path], model_name: str, logger_instance: Logger):
    global logger
    if not logger: logger = logger_instance
    if stderr_log_path and stderr_log_path.exists():
        logger.warn(f"  Attempting to read last lines from stderr log: {stderr_log_path}")
        try:
            with open(stderr_log_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
                last_lines = lines[-10:]
                if last_lines:
                    logger.warn("  Last lines of server stderr:")
                    for line in last_lines: logger.warn(f"    {line.strip()}")
                else: logger.warn(f"    Log file '{stderr_log_path}' was empty.")
        except Exception as e: logger.warn(f"    Could not read stderr log ({stderr_log_path}) for details: {e}")
    else: logger.warn(f"  Stderr log for '{model_name}' not available (Path: {stderr_log_path}).")


# --- Process Management (Copied/Adapted) ---
def start_llama_server(
    executable_path_str: str, arguments_str: str, model_name: str,
    temp_dir: Path, logger_instance: Logger
) -> Optional[Tuple[subprocess.Popen, Path, Path]]:
    global logger
    if not logger: logger = logger_instance
    exec_path = Path(executable_path_str)
    try:
        resolved_exec = str(exec_path.resolve(strict=True)) if exec_path.is_file() else executable_path_str
        args_list = [resolved_exec] + shlex.split(arguments_str)
        logger.debug(f"Final Popen args for '{model_name}': {args_list}")
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Error preparing command for '{model_name}': {e}")
        return None

    unique_suffix = datetime.datetime.now().strftime('%H%M%S%f')
    stdout_log = temp_dir / f"{model_name}_{unique_suffix}_stdout.log"
    stderr_log = temp_dir / f"{model_name}_{unique_suffix}_stderr.log"

    logger.info(f"  Attempting to start server for '{model_name}'...")
    logger.debug(f"    Command: {' '.join(args_list)}")
    logger.debug(f"    Stdout Log: {stdout_log}, Stderr Log: {stderr_log}")
    try:
        with open(stdout_log, 'wb') as f_out, open(stderr_log, 'wb') as f_err:
            process = subprocess.Popen(args_list, stdout=f_out, stderr=f_err)
        time.sleep(0.7)
        if process.poll() is not None:
            logger.error(f"Server for '{model_name}' failed to start or exited immediately. Exit code: {process.returncode}")
            _dump_stderr_on_failure(stderr_log, model_name, logger)
            return None
        logger.success(f"  Server for '{model_name}' started (PID: {process.pid}).")
        return process, stdout_log, stderr_log
    except FileNotFoundError:
        logger.error(f"Executable not found for '{model_name}': {args_list[0]}")
        return None
    except Exception as e:
        logger.error(f"Exception starting server for '{model_name}': {e}")
        return None

def stop_llama_server(process: Optional[subprocess.Popen], model_name: str, logger_instance: Logger):
    global logger
    if not logger: logger = logger_instance
    if not process:
        logger.debug(f"No process object to stop for '{model_name}'.")
        return
    pid = process.pid
    if process.poll() is not None:
        logger.info(f"  Server '{model_name}' (PID {pid}) was already stopped (exit code {process.returncode}).")
        return
    logger.info(f"  Stopping server '{model_name}' (PID {pid})...")
    try:
        process.terminate()
        process.wait(timeout=PROCESS_TERMINATE_TIMEOUT_S)
        logger.success(f"  Server '{model_name}' (PID {pid}) terminated gracefully.")
    except subprocess.TimeoutExpired:
        logger.warn(f"  Server '{model_name}' (PID {pid}) did not terminate gracefully. Forcing kill...")
        try:
            process.kill()
            process.wait(timeout=5)
            logger.success(f"  Server '{model_name}' (PID {pid}) killed.")
        except Exception as e: logger.error(f"  Error during force kill of '{model_name}' (PID {pid}): {e}")
    except Exception as e: logger.warn(f"  Error during termination of '{model_name}' (PID {pid}): {e}.")

def wait_for_server_health(
    process: subprocess.Popen, health_check_url: str, timeout_s: int,
    poll_interval_s: float, model_name: str, logger_instance: Logger
) -> bool:
    global logger
    if not logger: logger = logger_instance
    logger.info(f"  Waiting for '{model_name}' (PID: {process.pid}) health at {health_check_url} (timeout: {timeout_s}s)...")
    start_time = time.monotonic()
    attempt = 0
    while time.monotonic() - start_time < timeout_s:
        attempt += 1
        if process.poll() is not None:
            logger.warn(f"  Server '{model_name}' (PID {process.pid}) exited prematurely (code {process.returncode}) while waiting for health.")
            return False
        try:
            req_timeout = max(1.0, poll_interval_s - 0.5)
            response = requests.get(health_check_url, timeout=req_timeout)
            if response.status_code == 200:
                logger.success(f"  Server '{model_name}' is healthy (attempt {attempt}).")
                return True
            else: logger.debug(f"  Health check '{model_name}' (attempt {attempt}, PID {process.pid}): HTTP {response.status_code}. Retrying...")
        except requests.exceptions.ConnectionError: logger.debug(f"  Health check '{model_name}' (attempt {attempt}, PID {process.pid}): Connection refused. Retrying...")
        except requests.exceptions.Timeout: logger.debug(f"  Health check '{model_name}' (attempt {attempt}, PID {process.pid}): Request timed out. Retrying...")
        except requests.RequestException as e: logger.debug(f"  Health check '{model_name}' (attempt {attempt}, PID {process.pid}): Request failed: {e}. Retrying...")
        
        sleep_end_time = time.monotonic() + poll_interval_s
        while time.monotonic() < sleep_end_time:
            if process.poll() is not None:
                 logger.warn(f"  Server '{model_name}' (PID {process.pid}) exited (code {process.returncode}) during health check polling interval.")
                 return False
            time.sleep(0.1)
    logger.warn(f"  Server '{model_name}' (PID {process.pid}) did not become healthy at {health_check_url} within {timeout_s} seconds.")
    return False

# --- Data Extraction (Copied/Adapted) ---
# Assuming parse_param_size_from_alias and parse_quant_from_string are not strictly needed for memory scan output,
# but can be added if desired in the CSV. For now, focusing on memory.
# If needed, they are identical to the benchmark script.

def parse_memory_string_to_gb(mem_str: str) -> Optional[float]: # Identical
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

def parse_memory_from_log(stderr_log_path: Path, model_name: str, logger_instance: Logger) -> Tuple[str, str, str]: # Identical logic
    global logger
    if not logger: logger = logger_instance
    total_gpu_gb, total_cpu_gb = 0.0, 0.0
    lines_found, parse_errors = 0, 0
    final_status = "Failed (Unknown)"

    if not stderr_log_path.exists():
        logger.warn(f"    Memory log not found for '{model_name}': {stderr_log_path}")
        return "0.00", "0.00", "Failed (No Log)"
    logger.debug(f"    Parsing memory from log: {stderr_log_path}")
    buffer_regex = re.compile(r"(ggml_vk_)?(Metal|CUDA|CPU).*?buffer size\s*=\s*([0-9.,]+\s*[KMGT]i?B)", re.IGNORECASE)
    time.sleep(0.5)
    try:
        with stderr_log_path.open('r', encoding='utf-8', errors='replace') as f:
            for line_num, line in enumerate(f, 1):
                match = buffer_regex.search(line)
                if match:
                    lines_found += 1
                    device_type, mem_str = match.group(2), match.group(3)
                    logger.debug(f"      Found buffer on line {line_num}: Device='{device_type}', MemStr='{mem_str}'")
                    mem_gb = parse_memory_string_to_gb(mem_str)
                    if mem_gb is not None:
                        if device_type.upper() in ["METAL", "CUDA"]: total_gpu_gb += mem_gb
                        elif device_type.upper() == "CPU": total_cpu_gb += mem_gb
                    else:
                        logger.warn(f"      Could not parse memory value: '{mem_str}' in line: {line.strip()}")
                        parse_errors += 1
    except Exception as e:
        logger.warn(f"    Error reading or parsing memory log file {stderr_log_path}: {e}")
        return "0.00", "0.00", "Failed (Log Read Error)"

    if lines_found > 0: final_status = "Parse Error" if parse_errors > 0 else "Success"
    else:
        logger.warn(f"    No 'buffer size' lines found in log for '{model_name}'.")
        final_status = "Failed (No Buffers)"
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
    logger_instance: Logger
):
    global logger
    if not logger: logger = logger_instance

    if not processed_models_config:
        logger.error("No processed model configurations provided for memory scan.")
        return

    models_to_iterate = {}
    if model_to_scan_alias:
        if model_to_scan_alias in processed_models_config:
            model_entry = processed_models_config[model_to_scan_alias]
            if isinstance(model_entry, dict): models_to_iterate = {model_to_scan_alias: model_entry}
            else:
                logger.error(f"Config for specified model '{model_to_scan_alias}' is malformed. Skipping.")
                return
        else:
            logger.error(f"Specified model '{model_to_scan_alias}' not found.")
            logger.info("Available models: " + ", ".join(processed_models_config.keys()))
            return
    else:
        models_to_iterate = {
            alias: data for alias, data in processed_models_config.items() if isinstance(data, dict)
        }
        if not models_to_iterate:
            logger.error("No valid model configurations found to scan.")
            return
        logger.info(f"Scanning memory for all {len(models_to_iterate)} processed models.")

    csv_header = ["ModelName", "Timestamp", "ScanStatus", "GpuMemoryGB", "CpuMemoryGB", "Error"]
    all_results_data = []
    
    base_proxy_url = f"http://127.0.0.1:{STATIC_SERVER_PORT}"
    health_url_template = f"{base_proxy_url}/health"

    for model_idx, (model_alias, model_data) in enumerate(models_to_iterate.items()):
        logger.subheader(f"Scanning Model ({model_idx + 1}/{len(models_to_iterate)}): {model_alias}")

        gpu_gb, cpu_gb, scan_status = "-", "-", "Not Scanned"
        error_message = ""
        timestamp = logger._get_timestamp()

        logger.step(f"Preparing server command for '{model_alias}'")
        original_cmd_dict = model_data.get("cmd")
        if not isinstance(original_cmd_dict, dict):
            logger.warn(f"  '{model_alias}' missing 'cmd' dictionary. Skipping.")
            scan_status, error_message = "Config Error", "Missing 'cmd' dict"
            all_results_data.append([model_alias, timestamp, scan_status, gpu_gb, cpu_gb, error_message])
            continue

        scan_cmd_dict = original_cmd_dict.copy()
        scan_cmd_dict["port"] = STATIC_SERVER_PORT

        temp_model_config_for_cmd_build = {
            k: v for k, v in model_data.items() if k not in ["cmd", "generated_cmd_str"]
        }
        temp_model_config_for_cmd_build["cmd"] = scan_cmd_dict
        if "sampling" in model_data: temp_model_config_for_cmd_build["sampling"] = model_data["sampling"]
        
        final_scan_cmd_str, server_executable, server_args_str = "", "", ""
        try:
            final_scan_cmd_str = build_llama_server_command_util(temp_model_config_for_cmd_build)
            cmd_parts = shlex.split(final_scan_cmd_str)
            server_executable, server_args_str = cmd_parts[0], " ".join(cmd_parts[1:])
            logger.info(f"  Server command for memory scan generated.")
            logger.debug(f"    Full command: {final_scan_cmd_str}")
        except Exception as e:
            logger.error(f"  Could not build/parse server command for '{model_alias}': {e}. Skipping.")
            scan_status, error_message = "Cmd Build Error", f"Cmd Build/Parse Error: {e}"
            all_results_data.append([model_alias, timestamp, scan_status, gpu_gb, cpu_gb, error_message])
            continue
        
        logger.step(f"Initiating memory scan for '{model_alias}'")
        server_p: Optional[subprocess.Popen] = None
        stderr_log: Optional[Path] = None
        server_process_info = start_llama_server(
            server_executable, server_args_str, model_alias, temp_dir_path, logger
        )

        if server_process_info:
            server_p, _, stderr_log = server_process_info
            healthy = wait_for_server_health(server_p, health_url_template, health_timeout_s, health_poll_s, model_alias, logger)
            
            if healthy:
                logger.info(f"  Server healthy, parsing memory usage from logs for '{model_alias}'...")
                gpu_gb, cpu_gb, scan_status = parse_memory_from_log(stderr_log, model_alias, logger)
            else: # Not healthy
                if server_p.poll() is not None: # Server died
                    exit_code = server_p.returncode
                    scan_status = f"Server Exited (Code: {exit_code})"
                    error_message = f"Server (PID {server_p.pid}) exited (code {exit_code}) during/after health check."
                    logger.warn(f"  {error_message}")
                    _dump_stderr_on_failure(stderr_log, model_alias, logger)
                else: # Still running, but health check timed out
                    scan_status = "Health Timeout"
                    error_message = f"Server (PID {server_p.pid}) health check timed out."
                    logger.warn(f"  {error_message} Process still running, will be stopped.")
            stop_llama_server(server_p, model_alias, logger)
        else: # start_llama_server failed
            scan_status = "Server Start Failed"
            error_message = "Server process failed to start."
            # start_llama_server already logged details

        logger.info(f"  Memory Scan Result for '{model_alias}': GPU {Fore.CYAN}{gpu_gb} GB{Style.RESET_ALL}, CPU {Fore.CYAN}{cpu_gb} GB{Style.RESET_ALL} - Status: {color_status(scan_status)}")
        if error_message and scan_status not in ["Success", "Parse Error", "Failed (No Buffers)"]: # Avoid logging redundant "error" if it's a parse status
             logger.warn(f"    Details: {error_message}")


        all_results_data.append([model_alias, timestamp, scan_status, gpu_gb, cpu_gb, error_message])
        logger.notice("-" * 30)

    logger.header("Writing Memory Scan Results to CSV")
    try:
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_header)
            writer.writerows(all_results_data)
        logger.success(f"All memory scan results saved to: {output_csv_path.resolve()}")
    except IOError as e:
        logger.error(f"Failed to write CSV data to {output_csv_path}: {e}")


# --- Cleanup and Main Execution (Copied/Adapted) ---
TEMP_DIR_MANAGER_PATH: Optional[Path] = None

def signal_cleanup_handler(signum, frame):
    global logger, TEMP_DIR_MANAGER_PATH
    log_func = print if not logger else logger.warn
    log_func(f"\nSignal {signal.Signals(signum).name} received. Initiating cleanup...")
    if TEMP_DIR_MANAGER_PATH: log_func(f"  Note: Temporary directory {TEMP_DIR_MANAGER_PATH} should be auto-cleaned.")
    if logger: logger.info("Memory scan script terminated by signal.")
    else: print("Memory scan script terminated by signal.")
    sys.exit(1)

def main():
    global logger, TEMP_DIR_MANAGER_PATH

    parser = argparse.ArgumentParser(
        description="LLM Model Memory Scanner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    script_dir_path = Path(__file__).resolve().parent
    project_root = script_dir_path.parent
    default_base_config_path = (project_root / "config.base.yaml").resolve()

    parser.add_argument(
        "-c", "--config", type=Path, default=default_base_config_path,
        help="Path to the base YAML configuration file."
    )
    # Override logic copied from benchmark script
    default_override_path_hostname = (project_root / "overrides" / f"{platform.node().split('.')[0].lower()}.yaml").resolve()
    specific_override_filename = "mac-m3-max-36G.yaml" # Example
    default_override_path_specific = (project_root / "overrides" / specific_override_filename).resolve()
    calculated_default_override = default_override_path_hostname if default_override_path_hostname.exists() else \
                                  (default_override_path_specific if default_override_path_specific.exists() else None)
    parser.add_argument(
        "--override", type=Path, default=calculated_default_override,
        help=f"Path to override YAML. (Default logic checks hostname, then {specific_override_filename})"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=Path.cwd() / DEFAULT_OUTPUT_FILENAME,
        help="Path for the output CSV file."
    )
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_HEALTH_POLL_INTERVAL_S, help="Health check poll interval (seconds).")
    parser.add_argument("--health-timeout", type=int, help=f"Override health check timeout (seconds). Default: {DEFAULT_HEALTH_TIMEOUT_S}s.")
    parser.add_argument("-m", "--model", type=str, help="Scan only this specific model alias (from config).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose/debug output.")
    
    args = parser.parse_args()

    logger = Logger(verbose=args.verbose)
    logger.header("LLM MODEL MEMORY SCANNER INITIALIZATION")

    signal.signal(signal.SIGINT, signal_cleanup_handler)
    signal.signal(signal.SIGTERM, signal_cleanup_handler)
    logger.info("Signal handlers registered.")

    resolved_base_config = args.config.resolve()
    if not resolved_base_config.is_file():
        logger.error(f"Base configuration file not found: {resolved_base_config}")
        sys.exit(1)
    logger.info(f"Using base configuration: {resolved_base_config}")

    resolved_override = args.override.resolve() if args.override else None
    if resolved_override:
        if resolved_override.is_file(): logger.info(f"Using override configuration: {resolved_override}")
        else:
            logger.warn(f"Specified override configuration not found: {resolved_override}. Proceeding without.")
            resolved_override = None
    else: logger.info("No override configuration file specified or found.")

    logger.step("Performing initial cleanup of potentially lingering llama-server processes...")
    killed_procs = 0
    target_port_arg = f"--port {STATIC_SERVER_PORT}"
    
    # Common executable names/paths for llama.cpp server
    # Make these more specific to avoid matching general system "server" processes
    llama_exec_patterns = ["llama-server"] # Exact name
    llama_path_patterns = [
        "llama.cpp/server", # Common if run from within llama.cpp dir
        "llama.cpp/build/bin/server", # Common build path
        "build/bin/server" # If in a project with build/bin structure
    ]

    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'exe', 'username']):
        try:
            proc_info = proc.info
            pid = proc_info.get('pid') # Get pid for logging
            
            # Defensive checks for None before .lower() or other operations
            proc_name_raw = proc_info.get('name')
            proc_name = proc_name_raw.lower() if proc_name_raw else ""

            exe_path_raw = proc_info.get('exe')
            exe_path = exe_path_raw.lower() if exe_path_raw else ""
            
            cmdline_list = proc_info.get('cmdline')
            cmdline_str = " ".join(cmdline_list).lower() if cmdline_list else ""

            username = proc_info.get('username')

            is_potential_llama_server = False

            # 1. Strongest indicator: our specific port in command line
            if target_port_arg in cmdline_str:
                is_potential_llama_server = True
                logger.debug(f"  PID {pid} ('{proc_name_raw}') matched by port {STATIC_SERVER_PORT}.")

            # 2. Check for specific executable names or paths
            if not is_potential_llama_server:
                if proc_name in llama_exec_patterns:
                    is_potential_llama_server = True
                    logger.debug(f"  PID {pid} ('{proc_name_raw}') matched by exec name pattern.")
                elif any(patt in exe_path for patt in llama_path_patterns):
                    is_potential_llama_server = True
                    logger.debug(f"  PID {pid} ('{proc_name_raw}') matched by exec path pattern in '{exe_path_raw}'.")
                elif any(patt in cmdline_str for patt in llama_path_patterns): # check cmdline for paths too
                    is_potential_llama_server = True
                    logger.debug(f"  PID {pid} ('{proc_name_raw}') matched by path pattern in cmdline.")


            # Avoid terminating critical system processes or processes owned by other users easily
            # This is a simple check; more robust would involve checking for root/system UIDs
            if is_potential_llama_server:
                # Heuristic: if it's owned by root and not matched by our port, be very cautious
                # or if it's a very common system server name not matching our port.
                # For now, the port match is the strongest signal.
                # If not matched by port, but by a generic name like "server", and owned by root, skip.
                if username and "root" in username.lower() and target_port_arg not in cmdline_str:
                    logger.debug(f"  PID {pid} ('{proc_name_raw}') matched a generic pattern but is owned by '{username}'. Skipping kill to be safe.")
                    continue # Skip killing this likely system process

                # Additional check: ensure it's not something *very obviously* a system daemon
                # This list could be expanded for common OS daemons mistaken for "server"
                system_daemon_names = ["windowserver", "systemuiserver", "cvmserver", "nfstorageserver", "systemstatusd"]
                if proc_name in system_daemon_names and target_port_arg not in cmdline_str:
                    logger.debug(f"  PID {pid} ('{proc_name_raw}') appears to be a known system daemon and not using target port. Skipping kill.")
                    continue


                logger.warn(f"  Terminating potential lingering llama-server: PID={pid}, Name='{proc_name_raw}', User='{username}', Cmd='{(" ".join(cmdline_list))[:100] if cmdline_list else ""}'")
                p_obj = psutil.Process(pid)
                p_obj.terminate()
                try:
                    p_obj.wait(timeout=1) # Shorter timeout for cleanup
                except psutil.TimeoutExpired:
                    logger.warn(f"    PID {pid} did not terminate, killing...")
                    p_obj.kill()
                    p_obj.wait(timeout=1)
                killed_procs += 1
            else:
                logger.debug(f"  Skipping process check for PID={pid}, Name='{proc_name_raw}' (no strong match).")

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # These are expected for some processes, especially if they disappear during iteration
            logger.debug(f"  Skipping inaccessible/gone process PID {pid if 'pid' in locals() else 'unknown'}")
            continue
        except Exception as e_psutil:
            # Catch any other psutil error for a specific process and log it
            logger.warn(f"  Error during psutil check for process PID {pid if 'pid' in locals() else 'unknown'}: {e_psutil}")

    if killed_procs > 0:
        logger.success(f"Initial cleanup terminated {killed_procs} potential lingering llama-server process(es).")
    else:
        logger.info(f"No suspected lingering llama-server processes found during initial cleanup (checked for port {STATIC_SERVER_PORT} or specific patterns).")

    logger.step("Loading and processing configurations...")
    try:
        effective_conf_dict = generate_processed_config(
            base_config_path_arg=resolved_base_config,
            override_config_path_arg=resolved_override, # Corrected kwarg
            script_dir_for_overrides=project_root,
            verbose_logging=args.verbose
        )
        logger.success("Configurations processed successfully.")
    except Exception as e:
        logger.error(f"Failed to load or process configurations: {e}")
        if args.verbose: import traceback; logger.error("Traceback:\n" + traceback.format_exc())
        sys.exit(1)

    health_timeout_final = args.health_timeout if args.health_timeout is not None else \
                           effective_conf_dict.get("healthCheckTimeout", DEFAULT_HEALTH_TIMEOUT_S)
    if not isinstance(health_timeout_final, int) or health_timeout_final <= 0:
        logger.warn(f"Invalid healthCheckTimeout. Using default: {DEFAULT_HEALTH_TIMEOUT_S}s.")
        health_timeout_final = DEFAULT_HEALTH_TIMEOUT_S
    logger.info(f"Effective health check timeout: {health_timeout_final}s")
    logger.info(f"Health check poll interval: {args.poll_interval}s")
    
    output_file_abs = args.output.resolve()
    try:
        output_file_abs.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output CSV will be: {output_file_abs}")
    except OSError as e:
        logger.error(f"Could not create output directory {output_file_abs.parent}: {e}")
        sys.exit(1)

    with tempfile.TemporaryDirectory(prefix="llm_memscan_") as temp_dir_name:
        TEMP_DIR_MANAGER_PATH = Path(temp_dir_name)
        logger.info(f"Using temporary directory for logs: {TEMP_DIR_MANAGER_PATH}")
        logger.header("STARTING MEMORY SCAN RUN")
        script_start_time = time.monotonic()
        
        processed_models_data = effective_conf_dict.get("models", {})
        if not isinstance(processed_models_data, dict) or not processed_models_data:
            logger.error("'models' section not found/empty in config. Cannot run scan.")
            sys.exit(1)

        run_memory_scan(
            processed_models_config=processed_models_data,
            output_csv_path=output_file_abs,
            health_timeout_s=health_timeout_final,
            health_poll_s=args.poll_interval,
            model_to_scan_alias=args.model,
            temp_dir_path=TEMP_DIR_MANAGER_PATH,
            logger_instance=logger
        )
        script_duration_s = time.monotonic() - script_start_time
        logger.header("MEMORY SCAN SCRIPT COMPLETE")
        logger.success(f"Total script execution time: {script_duration_s:.2f} seconds.")

if __name__ == "__main__":
    main()