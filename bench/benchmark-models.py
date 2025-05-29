#!/usr/bin/env python3
"""
Cross-platform LLM Benchmarking Script.

Uses config_utils module to load configurations.
Performs memory scan (Pass 1) and benchmark run (Pass 2).
Uses a static port for running llama-server instances.
Outputs results to CSV.
"""

import argparse
import csv
import datetime
import json
import os
import platform
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TextIO

# --- Dependencies ---
import psutil
import requests  # For HTTP health checks and API calls
from colorama import Fore, Style, init as colorama_init # Renamed to avoid conflict

# --- Global Logger ---
# Will be initialized in main()
logger: Optional['Logger'] = None

# --- Import from shared config utility ---
try:
    current_script_dir = Path(__file__).resolve().parent
    project_root_dir = current_script_dir.parent 
    if str(project_root_dir) not in sys.path:
        sys.path.insert(0, str(project_root_dir))
    
    from utils.config_utils import generate_processed_config, build_llama_server_command_util
except ImportError as e:
    # Use basic print as logger is not yet initialized
    print(f"Error: Could not import from config_utils. Make sure config_utils.py is accessible in project_root/utils.")
    print(f"Attempted to add '{project_root_dir}' to PYTHONPATH.")
    print(f"Current PYTHONPATH: {sys.path}")
    print(f"Details: {e}")
    sys.exit(1)


# --- Constants ---
DEFAULT_OUTPUT_FILENAME = "benchmark_results.csv"
DEFAULT_QUESTION = "Why is the sky blue?"
DEFAULT_HEALTH_POLL_INTERVAL_S = 2.0 # Allow float for finer control
DEFAULT_HEALTH_TIMEOUT_S = 60
PROCESS_TERMINATE_TIMEOUT_S = 10
API_REQUEST_TIMEOUT_S = 600  # 10 minutes
STATIC_BENCHMARK_PORT = "9999"

# --- Logger Class ---
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

    def info(self, message: str):
        self._log("INFO", message, Fore.BLUE)

    def warn(self, message: str):
        self._log("WARN", message, Fore.YELLOW, bright=True)

    def error(self, message: str):
        self._log("ERROR", message, Fore.RED, bright=True, file=sys.stderr)

    def success(self, message: str):
        self._log("SUCCESS", message, Fore.GREEN, bright=True)

    def debug(self, message: str): # For verbose messages
        if self.verbose_flag:
            # Using a distinct color for debug messages
            timestamp = self._get_timestamp()
            print(f"{timestamp} [{Fore.MAGENTA}VERBOSE{Style.RESET_ALL}] {message}", file=sys.stderr)
            
    def header(self, title: str):
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{'=' * 70}\n{title.center(70)}\n{'=' * 70}{Style.RESET_ALL}")

    def subheader(self, title: str):
        print(f"\n{Fore.CYAN}{'-' * 70}\n{title.center(70)}\n{'-' * 70}{Style.RESET_ALL}")

    def step(self, message: str):
        print(f"{Fore.CYAN}>> {message}{Style.RESET_ALL}")

    def notice(self, message: str): # For general, non-level specific notices
        print(f"{Fore.WHITE}{message}{Style.RESET_ALL}")

class ServerOperationError(Exception):
    """Custom exception for flow control during server operations within a model's benchmark."""
    pass

# --- Utility Functions ---
def color_status(status: str) -> str:
    """Colorizes a status string for display."""
    status_lower = status.lower()
    if "success" in status_lower:
        return f"{Fore.GREEN}{Style.BRIGHT}{status}{Style.RESET_ALL}"
    elif any(err_term in status_lower for err_term in ["fail", "error", "timeout", "invalid", "missing", "exited"]):
        return f"{Fore.RED}{Style.BRIGHT}{status}{Style.RESET_ALL}"
    elif any(warn_term in status_lower for warn_term in ["warn", "not run", "not scanned", "parse error", "no buffers"]):
        return f"{Fore.YELLOW}{Style.BRIGHT}{status}{Style.RESET_ALL}"
    return status # Default no color

def _dump_stderr_on_failure(stderr_log_path: Optional[Path], model_name: str, logger_instance: Logger):
    """Helper to dump last lines of stderr log upon failure."""
    global logger
    if not logger: logger = logger_instance

    if stderr_log_path and stderr_log_path.exists():
        logger.warn(f"  Attempting to read last lines from stderr log: {stderr_log_path}")
        try:
            with open(stderr_log_path, 'r', encoding='utf-8', errors='replace') as err_f_read:
                # Read all lines and take the last 10. Handles potentially large files better than reading all into memory if only a few lines needed.
                # For very large files, more advanced techniques might be needed, but this is usually fine.
                lines = err_f_read.readlines()
                last_lines = lines[-10:]
                if last_lines:
                    logger.warn("  Last lines of server stderr:")
                    for line in last_lines:
                        logger.warn(f"    {line.strip()}")
                else:
                    logger.warn(f"    Log file '{stderr_log_path}' was empty.")
        except Exception as log_read_e:
            logger.warn(f"    Could not read stderr log ({stderr_log_path}) for details: {log_read_e}")
    else:
        logger.warn(f"  Stderr log for '{model_name}' not available (Path: {stderr_log_path}). Cannot dump details.")


# --- Process Management ---
def start_llama_server(
    executable_path_str: str,
    arguments_str: str,
    model_name: str,
    temp_dir: Path,
    logger_instance: Logger
) -> Optional[Tuple[subprocess.Popen, Path, Path]]:
    """Starts llama-server, returns (Popen, stdout_log, stderr_log) or None."""
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

    # Use a unique suffix for logs in case of multiple starts (e.g. Pass1 then Pass2)
    # This is handled by run_benchmark by using fresh temp_dir or distinct log names if needed.
    # For now, simple naming is fine as temp_dir helps isolate.
    stdout_log = temp_dir / f"{model_name}_{datetime.datetime.now().strftime('%H%M%S%f')}_stdout.log"
    stderr_log = temp_dir / f"{model_name}_{datetime.datetime.now().strftime('%H%M%S%f')}_stderr.log"


    logger.info(f"  Attempting to start server for '{model_name}'...")
    logger.debug(f"    Command: {' '.join(args_list)}")
    logger.debug(f"    Stdout Log: {stdout_log}")
    logger.debug(f"    Stderr Log: {stderr_log}")

    try:
        with open(stdout_log, 'wb') as f_out, open(stderr_log, 'wb') as f_err:
            process = subprocess.Popen(args_list, stdout=f_out, stderr=f_err)
        
        time.sleep(0.7) 
        if process.poll() is not None: 
            logger.error(f"Server for '{model_name}' failed to start or exited immediately. Exit code: {process.returncode}")
            _dump_stderr_on_failure(stderr_log, model_name, logger) # Call helper here
            return None
        logger.success(f"  Server for '{model_name}' started (PID: {process.pid}).")
        return process, stdout_log, stderr_log
    except FileNotFoundError:
        logger.error(f"Executable not found for '{model_name}': {args_list[0]}")
        logger.error("  Ensure the path in your configuration is correct and the file is executable.")
        return None
    except Exception as e:
        logger.error(f"Exception starting server for '{model_name}': {e}")
        return None

def stop_llama_server(process: Optional[subprocess.Popen], model_name: str, logger_instance: Logger):
    """Stops the llama-server process gracefully, then forcefully."""
    global logger
    if not logger: logger = logger_instance

    if not process:
        logger.debug(f"No process object to stop for '{model_name}'.")
        return
    
    pid = process.pid # Store pid before poll, as process object might become invalid
    if process.poll() is not None:
        logger.info(f"  Server '{model_name}' (PID {pid}) was already stopped (exit code {process.returncode}).")
        return

    logger.info(f"  Stopping server '{model_name}' (PID {pid})...")
    try:
        process.terminate() 
        process.wait(timeout=PROCESS_TERMINATE_TIMEOUT_S)
        logger.success(f"  Server '{model_name}' (PID {pid}) terminated gracefully.")
    except subprocess.TimeoutExpired:
        logger.warn(f"  Server '{model_name}' (PID {pid}) did not terminate gracefully. Forcing kill (SIGKILL)...")
        try:
            process.kill()
            process.wait(timeout=5) 
            logger.success(f"  Server '{model_name}' (PID {pid}) killed.")
        except Exception as e_kill:
             logger.error(f"  Error during force kill of server '{model_name}' (PID {pid}): {e_kill}")
    except Exception as e_term: # Catch other errors during terminate/wait e.g. process already died
        logger.warn(f"  Error during graceful termination of server '{model_name}' (PID {pid}): {e_term}. It might have already exited.")


def wait_for_server_health(
    process: subprocess.Popen, health_check_url: str, timeout_s: int,
    poll_interval_s: float, model_name: str, logger_instance: Logger
) -> bool:
    """Checks the server's /health endpoint until it returns 200 or timeout."""
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
            else:
                logger.debug(f"  Health check '{model_name}' (attempt {attempt}, PID {process.pid}): HTTP {response.status_code}. Retrying...")
        except requests.exceptions.ConnectionError:
             logger.debug(f"  Health check '{model_name}' (attempt {attempt}, PID {process.pid}): Connection refused. Retrying...")
        except requests.exceptions.Timeout:
            logger.debug(f"  Health check '{model_name}' (attempt {attempt}, PID {process.pid}): Request timed out. Retrying...")
        except requests.RequestException as e:
            logger.debug(f"  Health check '{model_name}' (attempt {attempt}, PID {process.pid}): Request failed: {e}. Retrying...")

        sleep_end_time = time.monotonic() + poll_interval_s
        while time.monotonic() < sleep_end_time:
            if process.poll() is not None:
                 logger.warn(f"  Server '{model_name}' (PID {process.pid}) exited (code {process.returncode}) during health check polling interval.")
                 return False
            time.sleep(0.1) 

    logger.warn(f"  Server '{model_name}' (PID {process.pid}) did not become healthy at {health_check_url} within {timeout_s} seconds.")
    return False

# --- Data Extraction Functions ---
def parse_param_size_from_alias(model_alias: str, logger_instance: Logger) -> str:
    global logger
    if not logger: logger = logger_instance
    moe_match = re.search(r"([0-9]+(?:\.[0-9]+)?x[0-9]+(?:\.[0-9]+)?)[Bb]", model_alias, re.IGNORECASE)
    if moe_match:
        size = f"{moe_match.group(1)}B"
        logger.debug(f"Parsed param size '{size}' from MoE alias '{model_alias}'")
        return size
    simple_match = re.search(r"([0-9]+(?:\.[0-9]+)?)[Bb]", model_alias, re.IGNORECASE)
    if simple_match:
        size = f"{simple_match.group(1)}B"
        logger.debug(f"Parsed param size '{size}' from alias '{model_alias}'")
        return size
    logger.debug(f"Could not parse param size from alias '{model_alias}'")
    return "-"

def parse_memory_string_to_gb(mem_str: str) -> Optional[float]:
    mem_str = mem_str.strip()
    match = re.match(r"([0-9]+(?:[.,][0-9]+)?)\s*([KMGT])(?:i?B)?", mem_str, re.IGNORECASE)
    if not match: return None
    value_str, unit_prefix = match.groups()
    try: value = float(value_str.replace(',', '.'))
    except ValueError: return None
    
    unit_prefix = unit_prefix.upper()
    multipliers = {'K': 1/1024/1024, 'M': 1/1024, 'G': 1, 'T': 1024} # To GiB
    multiplier = multipliers.get(unit_prefix)
    return value * multiplier if multiplier is not None else None

def parse_memory_from_log(stderr_log_path: Path, model_name: str, logger_instance: Logger) -> Tuple[str, str, str]:
    global logger
    if not logger: logger = logger_instance

    total_gpu_gb, total_cpu_gb = 0.0, 0.0
    lines_found, parse_errors = 0, 0
    final_status = "Failed (Unknown)"

    if not stderr_log_path.exists():
        logger.warn(f"    Memory log not found for '{model_name}': {stderr_log_path}")
        return "0.00", "0.00", "Failed (No Log)"

    logger.debug(f"    Parsing memory from log: {stderr_log_path}")
    buffer_regex = re.compile(r"(ggml_vk_)?(Metal|CUDA|CPU).*?buffer size\s*=\s*([0-9.,]+\s*[KMGT]i?B)", re.IGNORECASE) # Added ggml_vk_
    
    time.sleep(0.5) 

    try:
        with stderr_log_path.open('r', encoding='utf-8', errors='replace') as f:
            for line_num, line in enumerate(f, 1):
                match = buffer_regex.search(line)
                if match:
                    lines_found += 1
                    # group(1) is optional prefix, group(2) is device_type, group(3) is mem_str
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

    if lines_found > 0:
        final_status = "Parse Error" if parse_errors > 0 else "Success"
    else:
        logger.warn(f"    No 'buffer size' lines found in log for '{model_name}'. Memory usage might not be reported by this server version/config.")
        final_status = "Failed (No Buffers)"
    
    gpu_str, cpu_str = f"{total_gpu_gb:.2f}", f"{total_cpu_gb:.2f}"
    return gpu_str, cpu_str, final_status

def parse_quant_from_string(input_str: str, logger_instance: Logger) -> str:
    global logger
    if not logger: logger = logger_instance

    patterns = [
        (r"[Qq]([2-8AXL]_[A-Z0-9_]+(?:_L)?)" , lambda m: m.group(0).upper()), # Q4_K_M, Q6_K, Q5_K_S, Q8_0_L etc.
        (r"IQ[1-4]_[A-Z0-9_]+", lambda m: m.group(0).upper()),       # IQ2_XS, IQ4_NL etc.
        (r"\b(F32|F16|BF16)\b", lambda m: m.group(1).upper()),     # F16, F32 etc.
        (r"[Qq]([2-8](?:_0)?)\b", lambda m: m.group(0).upper()),     # Q4, Q8_0 etc. (simple)
        (r"[Qq](FP[48])\b", lambda m: m.group(0).upper()),         # QFP8, QFP4
    ]
    for pattern, normalizer in patterns:
        match = re.search(pattern, input_str, re.IGNORECASE)
        if match:
            quant = normalizer(match)
            logger.debug(f"Parsed quantization '{quant}' from string '{input_str}'")
            return quant
    logger.debug(f"Could not parse quantization from string '{input_str}'")
    return "-"

# --- Main Benchmark Logic ---
# --- Main Benchmark Logic ---
def run_benchmark(
    processed_models_config: Dict[str, Any],
    output_csv_path: Path,
    question: str,
    health_timeout_s: int,
    health_poll_s: float,
    model_to_test_alias: Optional[str],
    temp_dir_path: Path,
    logger_instance: Logger
):
    global logger
    if not logger: logger = logger_instance

    if not processed_models_config:
        logger.error("No processed model configurations provided to benchmark.")
        return

    models_to_iterate = {}
    if model_to_test_alias:
        if model_to_test_alias in processed_models_config:
            model_entry = processed_models_config[model_to_test_alias]
            if isinstance(model_entry, dict):
                models_to_iterate = {model_to_test_alias: model_entry}
                logger.info(f"Targeting specific model for benchmark: {model_to_test_alias}")
            else:
                logger.error(f"Config for specified model '{model_to_test_alias}' is malformed. Skipping.")
                return
        else:
            logger.error(f"Specified model '{model_to_test_alias}' not found in processed configurations.")
            logger.info("Available models: " + ", ".join(processed_models_config.keys()))
            return
    else:
        models_to_iterate = {
            alias: data for alias, data in processed_models_config.items() if isinstance(data, dict)
        }
        if not models_to_iterate:
            logger.error("No valid model configurations found to benchmark.")
            return
        logger.info(f"Benchmarking all {len(models_to_iterate)} processed models.")

    csv_header = [
        "ModelName", "ParameterSize", "Quantization", "ContextSize", "Timestamp",
        "MemoryScanStatus", "GpuMemoryGB", "CpuMemoryGB",
        "BenchStatus", "DurationSeconds", "TokensPerSecond",
        "PromptTokens", "CompletionTokens", "TotalTokens",
        "ProxyUrl", "Error"
    ]
    all_results_data = []
    
    base_proxy_url = f"http://127.0.0.1:{STATIC_BENCHMARK_PORT}"
    health_url_template = f"{base_proxy_url}/health"
    api_endpoint_template = f"{base_proxy_url}/v1/chat/completions"

    for model_idx, (model_alias, model_data) in enumerate(models_to_iterate.items()):
        logger.subheader(f"Processing Model ({model_idx + 1}/{len(models_to_iterate)}): {model_alias}")

        # Initialize all result fields for this model
        param_size, quantization, ctx_size_str = "-", "-", "-"
        mem_gpu, mem_cpu, mem_status = "-", "-", "Not Scanned" # Default for memory
        bench_status, duration_s, tps = "Not Run", "", ""       # Default for benchmark
        prompt_t, completion_t, total_t = "", "", ""
        timestamp = logger._get_timestamp()
        
        # This error message is for pre-server-start issues (config, command build)
        setup_error_message = ""
        # This error message is for issues during server run (start, health, API call)
        current_run_error_message = ""

        logger.step(f"Extracting static info for '{model_alias}'")
        param_size = parse_param_size_from_alias(model_alias, logger)
        original_cmd_dict = model_data.get("cmd") # Used for model path and ctx-size
        if not isinstance(original_cmd_dict, dict): # Should have been caught by config processing, but good check
            logger.warn(f"  '{model_alias}' missing 'cmd' dictionary for static info. This is unexpected.")
            # Not critical for static info if other sources exist
        
        model_file_path_str = original_cmd_dict.get("model") if original_cmd_dict else None
        ctx_size_str = str(original_cmd_dict.get("ctx-size", "-")) if original_cmd_dict else "-"
        
        if model_file_path_str:
            model_filename = Path(model_file_path_str).name
            quantization = parse_quant_from_string(model_filename, logger)
            if quantization == "-": quantization = parse_quant_from_string(model_alias, logger)
        else:
            quantization = parse_quant_from_string(model_alias, logger)
        logger.info(f"  Model Details - Params: {Fore.CYAN}{param_size}{Style.RESET_ALL}, Quant: {Fore.CYAN}{quantization}{Style.RESET_ALL}, Context: {Fore.CYAN}{ctx_size_str} tokens{Style.RESET_ALL}")

        logger.step(f"Preparing server command for '{model_alias}'")
        # Use original_cmd_dict for building the command, as it's what config_utils expects
        if not isinstance(original_cmd_dict, dict): # Critical for command build
            setup_error_message = "Internal Config Error: Missing 'cmd' dict for command building"
            logger.warn(f"  '{model_alias}' missing 'cmd' dictionary in config. Skipping this model.")
            all_results_data.append([
                model_alias, param_size, quantization, ctx_size_str, timestamp,
                mem_status, mem_gpu, mem_cpu,
                "Config Error", "", "", "", "", "", base_proxy_url, setup_error_message
            ])
            continue # Skip to next model if command can't be built

        bench_cmd_dict = original_cmd_dict.copy()
        bench_cmd_dict["port"] = STATIC_BENCHMARK_PORT 

        temp_model_config_for_cmd_build = {
            key: val for key, val in model_data.items() if key not in ["cmd", "generated_cmd_str"]
        }
        temp_model_config_for_cmd_build["cmd"] = bench_cmd_dict
        if "sampling" in model_data:
             temp_model_config_for_cmd_build["sampling"] = model_data["sampling"]
        
        final_bench_cmd_str, server_executable, server_args_str = "", "", ""
        try:
            final_bench_cmd_str = build_llama_server_command_util(temp_model_config_for_cmd_build)
            cmd_parts = shlex.split(final_bench_cmd_str)
            server_executable = cmd_parts[0]
            server_args_str = " ".join(cmd_parts[1:])
            logger.info(f"  Server command generated successfully.")
            logger.debug(f"    Full command: {final_bench_cmd_str}")
        except Exception as e:
            setup_error_message = f"Cmd Build/Parse Error: {e}"
            logger.error(f"  Could not build/parse server command for '{model_alias}': {e}. Skipping model.")
            all_results_data.append([
                model_alias, param_size, quantization, ctx_size_str, timestamp,
                mem_status, mem_gpu, mem_cpu,
                "Cmd Build Error", "", "", "", "", "", base_proxy_url, setup_error_message
            ])
            continue # Skip to next model

        # --- Combined Server Run, Memory Scan, and Benchmark ---
        logger.step(f"Starting server, scanning memory, and benchmarking for '{model_alias}'...")

        server_process: Optional[subprocess.Popen] = None
        stderr_log_path: Optional[Path] = None
        
        try:
            # Start server
            server_process_info = start_llama_server(
                server_executable, server_args_str, model_alias, temp_dir_path, logger
            )
            if not server_process_info:
                mem_status = "Server Start Failed"
                bench_status = "Start Failed"
                current_run_error_message = "Server process failed to start (see logs for details from start_llama_server)."
                # start_llama_server already logged details and potentially stderr dump
                raise ServerOperationError("Server start failed") # Jumps to finally block for this model
            
            server_process, _, stderr_log_path = server_process_info

            # Health check
            healthy = wait_for_server_health(
                server_process, health_url_template, health_timeout_s, health_poll_s, model_alias, logger
            )
            if not healthy:
                if server_process.poll() is not None: # Server died
                    exit_code = server_process.returncode
                    status_msg = f"Server Exited (Code: {exit_code})"
                    mem_status = bench_status = status_msg
                    current_run_error_message = f"Server (PID {server_process.pid}) exited (code {exit_code}) before/during health check."
                    logger.warn(f"  {current_run_error_message}")
                    if stderr_log_path: _dump_stderr_on_failure(stderr_log_path, model_alias, logger)
                else: # Health timeout, server might still be running
                    status_msg = "Health Timeout"
                    mem_status = bench_status = status_msg
                    current_run_error_message = f"Server (PID {server_process.pid}) did not become healthy (timed out)."
                    logger.warn(f"  {current_run_error_message} Server will be stopped.")
                raise ServerOperationError("Server not healthy") # Jumps to finally

            # --- Server is UP and HEALTHY ---
            logger.info(f"  Server for '{model_alias}' is healthy. Proceeding with memory scan and benchmark.")

            # 1. Memory Scan (while server is running)
            logger.info(f"  Parsing memory usage from logs for '{model_alias}'...")
            if server_process.poll() is None and stderr_log_path: # Check if server still running
                gpu_gb_val, cpu_gb_val, mem_status_val = parse_memory_from_log(stderr_log_path, model_alias, logger)
                mem_gpu, mem_cpu, mem_status = gpu_gb_val, cpu_gb_val, mem_status_val # Update from defaults
                logger.info(f"  Memory Scan Result for '{model_alias}': GPU {Fore.CYAN}{mem_gpu} GB{Style.RESET_ALL}, CPU {Fore.CYAN}{mem_cpu} GB{Style.RESET_ALL} - Status: {color_status(mem_status)}")
            elif stderr_log_path: # Server died between health check and memory parsing, or stderr_log_path is None (latter unlikely)
                exit_code = server_process.returncode if server_process.returncode is not None else "Unknown"
                mem_status = f"Server Exited Prematurely (Code: {exit_code})"
                if bench_status == "Not Run": bench_status = mem_status # Update bench_status if not already set
                current_run_error_message = f"Server for '{model_alias}' (PID {server_process.pid}) died (code {exit_code}) after health check, before memory parsing."
                logger.warn(f"  {current_run_error_message}")
                _dump_stderr_on_failure(stderr_log_path, model_alias, logger)
                raise ServerOperationError("Server died before memory parsing") # Jumps to finally
            else: # Should not happen if server_process_info was valid
                 mem_status = "Error (No Log Path)"
                 current_run_error_message = "Internal error: stderr_log_path not available for memory parsing."
                 logger.warn(f"  {current_run_error_message}")
                 # Bench status remains "Not Run" or as set by health check failure if that path was taken.

            # 2. Performance Benchmark (if server is still running)
            api_error_message = "" # Specific error from API call
            if server_process.poll() is None:
                logger.info(f"  Sending benchmark request to {api_endpoint_template} for '{model_alias}'...")
                payload = {"model": model_alias, "messages": [{"role": "user", "content": question}], "temperature": 0.7, "stream": False}
                req_start_time = time.monotonic()
                try:
                    response = requests.post(api_endpoint_template, json=payload, timeout=API_REQUEST_TIMEOUT_S)
                    req_duration_val = time.monotonic() - req_start_time
                    duration_s = f"{req_duration_val:.3f}"

                    if response.status_code == 200:
                        try:
                            api_data = response.json()
                            usage = api_data.get("usage", {})
                            prompt_t_val = usage.get("prompt_tokens", 0)
                            completion_t_val = usage.get("completion_tokens", 0)
                            prompt_t = str(prompt_t_val)
                            completion_t = str(completion_t_val)
                            total_t = str(usage.get("total_tokens", prompt_t_val + completion_t_val))
                            if req_duration_val > 1e-6 and completion_t_val > 0:
                                tps = f"{completion_t_val / req_duration_val:.2f}"
                            else:
                                tps = "0.00"
                                if completion_t_val == 0: logger.warn("    Benchmark warning: Completion tokens is 0, TPS will be 0.00.")
                                if req_duration_val <= 1e-6 : logger.warn("    Benchmark warning: Request duration too short, TPS may be inaccurate.")
                            bench_status = "Success"
                            logger.success(f"  Benchmark for '{model_alias}' completed.")
                            logger.info(f"    Duration: {Fore.CYAN}{duration_s}s{Style.RESET_ALL}, Tokens (P/C/T): {prompt_t}/{completion_t}/{total_t}, TPS: {Fore.CYAN}{tps}{Style.RESET_ALL}")
                        except json.JSONDecodeError as e_json_dec:
                            bench_status, api_error_message = "API Response Error", f"JSON decode error: {e_json_dec} - Response text: {response.text[:100]}"
                            logger.error(f"  Error decoding JSON response for '{model_alias}': {api_error_message}")
                        except Exception as e_json_proc:
                            bench_status, api_error_message = "API Response Error", f"Response processing error: {e_json_proc}"
                            logger.error(f"  Error processing API response for '{model_alias}': {api_error_message}")
                    else:
                        bench_status = "API Request Failed"
                        api_error_message = f"HTTP {response.status_code}: {response.text[:200]}"
                        logger.error(f"  API request for '{model_alias}' failed: {api_error_message}")
                except requests.Timeout:
                    duration_s = f"{time.monotonic() - req_start_time:.3f}" # Ensure duration is recorded
                    bench_status, api_error_message = "API Request Failed", "Request timed out"
                    logger.error(f"  API request for '{model_alias}' timed out after {duration_s}s.")
                except requests.RequestException as e_req:
                    duration_s = f"{time.monotonic() - req_start_time:.3f}" # Ensure duration is recorded
                    bench_status, api_error_message = "API Request Failed", f"Request Exception: {e_req}"
                    logger.error(f"  API request exception for '{model_alias}': {api_error_message}")
                
                if api_error_message: # If an API error occurred, it's the most relevant run error
                    current_run_error_message = api_error_message
            
            elif stderr_log_path: # Server died between memory scan and benchmark attempt
                exit_code = server_process.returncode if server_process.returncode is not None else "Unknown"
                if bench_status == "Not Run": bench_status = f"Server Exited Prematurely (Code: {exit_code})"
                err_msg = f"Server for '{model_alias}' (PID {server_process.pid}) died (code {exit_code}) before benchmark API call."
                current_run_error_message = f"{current_run_error_message}; {err_msg}" if current_run_error_message else err_msg
                logger.warn(f"  {err_msg}")
                _dump_stderr_on_failure(stderr_log_path, model_alias, logger)
                # No raise, will fall through to finally and append results
            
            logger.info(f"  Benchmark Result for '{model_alias}': Status: {color_status(bench_status)}")
            if api_error_message and bench_status != "Success": # Log API error details if any
                 logger.debug(f"    Error details for benchmark API of '{model_alias}': {api_error_message}")


        except ServerOperationError as e: # Catch our controlled jumps
            logger.debug(f"Server operation for '{model_alias}' ended due to: {e}. Statuses captured.")
            # current_run_error_message and relevant statuses (mem_status, bench_status)
            # should have been set by the block that raised the error.
        except Exception as e_outer_scope: # Catch any other unexpected errors during server ops
            logger.error(f"Unexpected error during combined processing of '{model_alias}': {e_outer_scope}")
            current_run_error_message = f"Unexpected script error: {str(e_outer_scope)[:200]}"
            if bench_status == "Not Run": bench_status = "Script Error" # Fallback status
            if mem_status == "Not Scanned": mem_status = "Script Error" # Fallback status
            if logger.verbose_flag:
                import traceback
                logger.error("Traceback:\n" + traceback.format_exc())
        finally:
            if server_process: # Always try to stop the server if it was started
                stop_llama_server(server_process, model_alias, logger)

        # Final error message for CSV (setup_error_message is handled by `continue` earlier)
        # So, error_message_for_csv will be current_run_error_message from this combined pass.
        error_message_for_csv = current_run_error_message

        all_results_data.append([
            model_alias, param_size, quantization, ctx_size_str, timestamp,
            mem_status, mem_gpu, mem_cpu,
            bench_status, duration_s, tps, prompt_t, completion_t, total_t,
            base_proxy_url, error_message_for_csv
        ])
        logger.notice("-" * 30) # Separator for model processing in console

    # --- After iterating all models ---
    logger.header("Writing Results to CSV")
    try:
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_header)
            writer.writerows(all_results_data)
        logger.success(f"All benchmark results saved to: {output_csv_path.resolve()}")
    except IOError as e:
        logger.error(f"Failed to write CSV data to {output_csv_path}: {e}")

# --- Cleanup and Main Execution ---
TEMP_DIR_MANAGER_PATH: Optional[Path] = None 

def signal_cleanup_handler(signum, frame):
    global logger, TEMP_DIR_MANAGER_PATH
    log_func = print 
    if logger: log_func = logger.warn
    
    log_func(f"\nSignal {signal.Signals(signum).name} received. Initiating cleanup...")
    if TEMP_DIR_MANAGER_PATH:
        log_func(f"  Note: Temporary directory {TEMP_DIR_MANAGER_PATH} should be auto-cleaned on script exit.")
    if logger: logger.info("Benchmark script terminated by signal.")
    else: print("Benchmark script terminated by signal.")
    sys.exit(1)

def main():
    global logger, TEMP_DIR_MANAGER_PATH 

    parser = argparse.ArgumentParser(
        description="Cross-platform LLM Benchmarking Script with enhanced output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    script_dir_path = Path(__file__).resolve().parent
    project_root = script_dir_path.parent 
    default_base_config_path = (project_root / "config.base.yaml").resolve()

    parser.add_argument(
        "-c", "--config", type=Path, default=default_base_config_path,
        help="Path to the base YAML configuration file."
    )

    default_override_path_hostname = (project_root / "overrides" / f"{platform.node().split('.')[0].lower()}.yaml").resolve()
    specific_override_filename = "mac-m3-max-36G.yaml" 
    default_override_path_specific = (project_root / "overrides" / specific_override_filename).resolve()

    calculated_default_override = None
    if default_override_path_hostname.exists():
        calculated_default_override = default_override_path_hostname
    elif default_override_path_specific.exists():
        calculated_default_override = default_override_path_specific
    
    parser.add_argument(
        "--override", type=Path, default=calculated_default_override,
        help=f"Path to override YAML. (Default logic: checks for ../overrides/<hostname>.yaml, then ../overrides/{specific_override_filename})"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=Path.cwd() / DEFAULT_OUTPUT_FILENAME,
        help="Path for the output CSV file."
    )
    parser.add_argument("-q", "--question", type=str, default=DEFAULT_QUESTION, help="Question/prompt for benchmarking.")
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_HEALTH_POLL_INTERVAL_S, help="Health check poll interval (seconds).")
    parser.add_argument("--health-timeout", type=int, help=f"Override health check timeout (seconds). Default: {DEFAULT_HEALTH_TIMEOUT_S}s.")
    parser.add_argument("-m", "--model", type=str, help="Benchmark only this specific model alias (from config).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose/debug output.")
    
    args = parser.parse_args()

    logger = Logger(verbose=args.verbose)
    logger.header("LLM BENCHMARKER INITIALIZATION")

    signal.signal(signal.SIGINT, signal_cleanup_handler)
    signal.signal(signal.SIGTERM, signal_cleanup_handler)
    logger.info("Signal handlers for SIGINT and SIGTERM registered.")

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
            logger.warn(f"Specified override configuration not found: {resolved_override}. Proceeding without this override.")
            resolved_override = None
    else:
        logger.info("No override configuration file specified or found by default logic.")

    logger.step("Performing initial cleanup of potentially lingering llama-server processes...")
    killed_procs = 0
    server_exec_patterns = ["llama-server", "server"] 
    server_cmdline_patterns = ["llama.cpp/server", "build/bin/server", "--port", f"{STATIC_BENCHMARK_PORT}"] 

    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'exe']):
        try:
            proc_info = proc.info
            proc_name = proc_info.get('name', '').lower() if proc_info.get('name') else ''
            cmdline_list = proc_info.get('cmdline')
            cmdline_str = " ".join(cmdline_list) if cmdline_list else ""
            exe_path = proc_info.get('exe', '').lower() if proc_info.get('exe') else ''

            is_server_process = False
            if any(patt in proc_name for patt in server_exec_patterns if proc_name): is_server_process = True
            if not is_server_process and any(patt in exe_path for patt in server_exec_patterns if exe_path): is_server_process = True
            if not is_server_process and any(patt in cmdline_str for patt in server_cmdline_patterns if cmdline_str): is_server_process = True
            # Specifically check for our benchmark port if other checks are inconclusive
            if not is_server_process and f"--port {STATIC_BENCHMARK_PORT}" in cmdline_str: is_server_process = True
            
            if is_server_process:
                 logger.warn(f"  Terminating lingering process: PID={proc.pid}, Name='{proc_name}', Exe='{exe_path}', Cmd='{cmdline_str[:100]}...'")
                 p_obj = psutil.Process(proc.pid)
                 p_obj.terminate()
                 try: p_obj.wait(timeout=2)
                 except psutil.TimeoutExpired:
                     logger.warn(f"    PID {proc.pid} did not terminate, killing...")
                     p_obj.kill()
                     p_obj.wait(timeout=1) 
                 killed_procs +=1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            logger.debug(f"  Skipping inaccessible/gone process PID {proc.pid if 'proc' in locals() and hasattr(proc, 'pid') else 'unknown'}")
            continue
        except Exception as e_psutil:
            logger.warn(f"  Error during psutil check for process {proc.pid if 'proc' in locals() and hasattr(proc, 'pid') else 'unknown'}: {e_psutil}")

    if killed_procs > 0: logger.success(f"Initial cleanup terminated {killed_procs} lingering process(es).")
    else: logger.info("No lingering llama-server processes found during initial cleanup.")

    logger.step("Loading and processing configurations...")
    try:
        effective_conf_dict = generate_processed_config(
            base_config_path_arg=resolved_base_config,
            override_config_path_arg=resolved_override, # <<< CORRECTED
            script_dir_for_overrides=project_root, 
            verbose_logging=args.verbose 
        )
        logger.success("Configurations processed successfully.")
    except Exception as e: 
        logger.error(f"Failed to load or process configurations: {e}")
        if args.verbose:
            import traceback
            logger.error("Traceback:\n" + traceback.format_exc())
        sys.exit(1)

    health_timeout_final = args.health_timeout 
    if health_timeout_final is None: 
        health_timeout_final = effective_conf_dict.get("healthCheckTimeout", DEFAULT_HEALTH_TIMEOUT_S)
    if not isinstance(health_timeout_final, int) or health_timeout_final <= 0:
        logger.warn(f"Invalid healthCheckTimeout value '{health_timeout_final}'. Using default: {DEFAULT_HEALTH_TIMEOUT_S}s.")
        health_timeout_final = DEFAULT_HEALTH_TIMEOUT_S
    logger.info(f"Effective health check timeout: {health_timeout_final}s")
    logger.info(f"Health check poll interval: {args.poll_interval}s")
    
    # --- Modify output filename to include a date tag ---
    original_output_path = args.output 
    date_tag = datetime.datetime.now().strftime('%Y%m%d')
    
    output_dir = original_output_path.parent
    output_stem = original_output_path.stem
    output_suffix = original_output_path.suffix # e.g., ".csv"

    # Construct the new filename with the date tag
    # Example: if original is "results.csv", new will be "results_YYYYMMDD.csv"
    # If original is "results" (no suffix), new will be "results_YYYYMMDD"
    dated_filename = f"{output_stem}_{date_tag}{output_suffix}"
    output_file_abs = (output_dir / dated_filename).resolve()
    # --- End of filename modification ---

    try:
        output_file_abs.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output CSV will be written to: {output_file_abs}") # Log the new, dated path
    except OSError as e:
        logger.error(f"Could not create output directory: {output_file_abs.parent} - {e}")
        sys.exit(1)

    with tempfile.TemporaryDirectory(prefix="llm_bench_") as temp_dir_name:
        TEMP_DIR_MANAGER_PATH = Path(temp_dir_name) 
        logger.info(f"Using temporary directory for logs: {TEMP_DIR_MANAGER_PATH}")
        logger.header("STARTING BENCHMARK RUN")
        script_start_time = time.monotonic()
        
        processed_models_data = effective_conf_dict.get("models", {})
        if not isinstance(processed_models_data, dict) or not processed_models_data:
            logger.error("'models' section not found, empty, or not a dictionary in processed configuration.")
            sys.exit(1)

        run_benchmark(
            processed_models_config=processed_models_data,
            output_csv_path=output_file_abs, # Pass the modified, dated path
            question=args.question,
            health_timeout_s=health_timeout_final,
            health_poll_s=args.poll_interval,
            model_to_test_alias=args.model,
            temp_dir_path=TEMP_DIR_MANAGER_PATH,
            logger_instance=logger 
        )
        script_duration_s = time.monotonic() - script_start_time
        logger.header("BENCHMARK SCRIPT COMPLETE")
        logger.success(f"Total script execution time: {script_duration_s:.2f} seconds.")

if __name__ == "__main__":
    main()