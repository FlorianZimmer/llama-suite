#!/usr/bin/env python3
"""
Cross-platform LLM Benchmarking Script.

Uses config_utils module to load configurations.
Performs memory scan and benchmark run in a single server instance per model.
Uses a static port for running llama-server instances.
Outputs results to CSV.
Supports keeping temporary logs.
"""

import argparse
import csv
import datetime
import json
import os # Keep os for psutil.Process().username() compatibility on all platforms if needed
import platform
import re
# import shlex # Not directly used
import signal
import subprocess
import sys
import tempfile
import time
import copy # For deepcopy
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# --- Third-Party Library Imports ---
import psutil
import requests  # For HTTP health checks and API calls
# colorama Fore, Style are used directly for some inline coloring, init handled by Logger
from colorama import Fore, Style 

# --- Global Variables ---
logger: Optional['Logger'] = None 
TEMP_DIR_MANAGER_PATH: Optional[Path] = None
project_root_dir: Path # Will be defined at script start

# --- Import from shared config utility ---
try:
    # Determine project root dynamically for sys.path modification
    _current_script_file_path_for_root_scan = Path(__file__).resolve()
    project_root_dir = _current_script_file_path_for_root_scan.parent.parent # Assuming script is in bench/
    if str(project_root_dir) not in sys.path:
        sys.path.insert(0, str(project_root_dir))
    
    from utils.config_utils import (
        generate_processed_config, 
        build_llama_server_command_util,
        Logger,
        start_llama_server, 
        stop_llama_server,  
        wait_for_server_health, 
        _dump_stderr_on_failure, 
        color_status, 
        PROCESS_TERMINATE_TIMEOUT_S, 
        DEFAULT_HEALTH_POLL_INTERVAL_S
    )
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import from 'utils.config_utils'.\n"
          f"Ensure 'config_utils.py' is in the '{project_root_dir if 'project_root_dir' in locals() else 'UNKNOWN' / 'utils'}' directory "
          f"and the project root '{project_root_dir if 'project_root_dir' in locals() else 'UNKNOWN'}' is in PYTHONPATH if necessary.\n"
          f"PYTHONPATH: {sys.path}\n"
          f"Details: {e}", file=sys.stderr)
    sys.exit(1)


# --- Constants Specific to This Script ---
DEFAULT_OUTPUT_FILENAME = "benchmark_results.csv" # Output filename without date tag by default
DEFAULT_QUESTION = "Why is the sky blue?"
DEFAULT_HEALTH_TIMEOUT_S_BENCH = 60 # benchmark-models specific default
API_REQUEST_TIMEOUT_S = 600  # 10 minutes
STATIC_BENCHMARK_PORT = "9999"

MODEL_CONFIG_META_KEYS_BENCHMARK = {
    "aliases", "sampling", "_name_for_log", "generated_cmd_str",
    "hf_tokenizer_for_model", 
    "supports_no_think_toggle" 
}

# --- Custom Exception ---
class ServerOperationError(Exception):
    """Custom exception for flow control during server operations within a model's benchmark."""
    pass

# --- Utility Functions (Adopted from scan_model_memory.py where identical) ---
def parse_param_size_from_alias(model_alias: str) -> str:
    # This function is specific to benchmark-models.py
    global logger
    assert logger is not None, "Logger not initialized for parse_param_size_from_alias"
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

SCRIPT_DIR = Path(__file__).resolve().parent
LOGS_DIR = SCRIPT_DIR / "logs"
CSV_BASENAME = "bench/benchmark_results"
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

def parse_memory_string_to_gb(mem_str: str) -> Optional[float]:
    # Identical to scan_model_memory.py
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
    # Adapted from scan_model_memory.py (mainly logger instance and sleep timer)
    total_gpu_gb, total_cpu_gb = 0.0, 0.0
    lines_found, parse_errors = 0, 0
    final_status = "Failed (Unknown)"

    if not stderr_log_path.exists():
        logger_instance.warn(f"    Memory log not found for '{model_name}': {stderr_log_path}")
        return "0.00", "0.00", "Failed (No Log)"
    logger_instance.debug(f"    Parsing memory from log: {stderr_log_path}")
    buffer_regex = re.compile(r"(ggml_vk_)?(Metal|CUDA|CPU).*?buffer size\s*=\s*([0-9.,]+\s*[KMGT]i?B)", re.IGNORECASE)
    
    time.sleep(0.2) # Aligning sleep with scan_model_memory.py; 0.5 was here before

    try:
        with stderr_log_path.open('r', encoding='utf-8', errors='replace') as f:
            for line_num, line in enumerate(f, 1):
                match = buffer_regex.search(line)
                if match:
                    lines_found += 1
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
        logger_instance.warn(f"    No 'buffer size' lines found in log for '{model_name}'. Memory usage might not be reported by this server version/config.")
        final_status = "Failed (No Buffers)" # More specific status
    
    gpu_str, cpu_str = f"{total_gpu_gb:.2f}", f"{total_cpu_gb:.2f}"
    return gpu_str, cpu_str, final_status

def parse_quant_from_string(input_str: str) -> str:
    # This function is specific to benchmark-models.py
    global logger
    assert logger is not None, "Logger not initialized for parse_quant_from_string"

    patterns = [
        (r"[Qq]([2-8AXL]_[A-Z0-9_]+(?:_L)?)" , lambda m: m.group(0).upper()),
        (r"IQ[1-4]_[A-Z0-9_]+", lambda m: m.group(0).upper()),      
        (r"\b(F32|F16|BF16)\b", lambda m: m.group(1).upper()),    
        (r"[Qq]([2-8](?:_0)?)\b", lambda m: m.group(0).upper()),    
        (r"[Qq](FP[48])\b", lambda m: m.group(0).upper()),        
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
def run_benchmark(
    processed_models_config: Dict[str, Any],
    output_csv_path: Path,
    question: str,
    health_timeout_s: int,
    health_poll_s: float,
    model_to_test_alias: Optional[str],
    temp_dir_path: Path, # This is the key path for logs
    logger_instance: Logger,
    project_root_path: Path # Used by start_llama_server
):
    # is_verbose for this function can be taken from logger_instance.verbose_flag
    is_verbose = logger_instance.verbose_flag

    if not processed_models_config:
        logger_instance.error("No processed model configurations provided to benchmark.")
        return

    models_to_iterate = {}
    if model_to_test_alias:
        if model_to_test_alias in processed_models_config:
            model_entry = processed_models_config[model_to_test_alias]
            if isinstance(model_entry, dict):
                models_to_iterate = {model_to_test_alias: model_entry}
                logger_instance.info(f"Targeting specific model for benchmark: {model_to_test_alias}")
            else:
                logger_instance.error(f"Config for specified model '{model_to_test_alias}' is malformed. Skipping.")
                return
        else:
            logger_instance.error(f"Specified model '{model_to_test_alias}' not found in processed configurations.")
            logger_instance.info("Available models: " + ", ".join(processed_models_config.keys()))
            return
    else:
        models_to_iterate = {
            alias: data for alias, data in processed_models_config.items() if isinstance(data, dict)
        }
        if not models_to_iterate:
            logger_instance.error("No valid model configurations found to benchmark.")
            return
        logger_instance.info(f"Benchmarking all {len(models_to_iterate)} processed models.")

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

    for model_idx, (model_alias, model_data_original) in enumerate(models_to_iterate.items()):
        logger_instance.subheader(f"Processing Model ({model_idx + 1}/{len(models_to_iterate)}): {model_alias}")

        # Initialize result fields
        param_size, quantization, ctx_size_str = "-", "-", "-"
        mem_gpu, mem_cpu, mem_status = "-", "-", "Not Scanned"
        bench_status, duration_s, tps = "Not Run", "", ""
        prompt_t, completion_t, total_t = "", "", ""
        timestamp = logger_instance._get_timestamp()
        current_run_error_message = "" # Consolidated error message for the current run

        logger_instance.step(f"Extracting static info for '{model_alias}'")
        param_size = parse_param_size_from_alias(model_alias) 

        original_cmd_dict_for_static_info = model_data_original.get("cmd")
        model_file_path_str = original_cmd_dict_for_static_info.get("model") if isinstance(original_cmd_dict_for_static_info, dict) else None
        ctx_size_str = str(original_cmd_dict_for_static_info.get("ctx-size", "-")) if isinstance(original_cmd_dict_for_static_info, dict) else "-"

        if model_file_path_str:
            model_filename = Path(model_file_path_str).name
            quantization = parse_quant_from_string(model_filename)
            if quantization == "-": quantization = parse_quant_from_string(model_alias)
        else:
            quantization = parse_quant_from_string(model_alias)
        logger_instance.info(f"  Model Details - Params: {Fore.CYAN}{param_size}{Style.RESET_ALL}, Quant: {Fore.CYAN}{quantization}{Style.RESET_ALL}, Context: {Fore.CYAN}{ctx_size_str} tokens{Style.RESET_ALL}")

        logger_instance.step(f"Preparing server command for '{model_alias}'")
        original_cmd_dict_from_config = model_data_original.get("cmd")
        if not isinstance(original_cmd_dict_from_config, dict):
            current_run_error_message = "Config Error: Missing 'cmd' dict in model_data_original"
            logger_instance.error(f"  {current_run_error_message} for '{model_alias}'. Skipping.")
            all_results_data.append([
                model_alias, param_size, quantization, ctx_size_str, timestamp,
                mem_status, mem_gpu, mem_cpu,
                "Config Error", "", "", "", "", "", base_proxy_url, current_run_error_message
            ])
            continue

        cmd_options_for_benchmark = original_cmd_dict_from_config.copy()
        cmd_options_for_benchmark["port"] = STATIC_BENCHMARK_PORT

        server_executable: str = ""
        server_args_list: List[str] = []

        try:
            # This relies on generate_processed_config resolving 'bin' to an absolute path
            resolved_bin_path_from_config = cmd_options_for_benchmark.get("bin")
            if not resolved_bin_path_from_config or not isinstance(resolved_bin_path_from_config, str) :
                # The generate_processed_config should have made this absolute. If not, error.
                 raise ValueError(f"Resolved 'bin' path ('{resolved_bin_path_from_config}') for model '{model_alias}' "
                                 f"is not an absolute path or is invalid. Check config processing output for this model's 'cmd.bin'.")
            if not Path(resolved_bin_path_from_config).is_absolute(): # Final check
                 logger_instance.warn(f"  Warning: 'bin' path '{resolved_bin_path_from_config}' for '{model_alias}' resolved by config_utils but not absolute. Proceeding, but this is unusual.")


            server_executable = resolved_bin_path_from_config
            logger_instance.debug(f"  Using server_executable (from resolved config): '{server_executable}'")

            # Construct server_args_list (similar to scan_model_memory.py's approach)
            server_args_list.extend(["--port", str(cmd_options_for_benchmark.get("port"))])
            server_args_list.extend(["--model", str(cmd_options_for_benchmark.get("model"))])
            server_args_list.extend(["--ctx-size", str(cmd_options_for_benchmark.get("ctx-size"))])

            if str(cmd_options_for_benchmark.get("gpu-layers", "auto")).lower() != "auto":
                server_args_list.extend(["--n-gpu-layers", str(cmd_options_for_benchmark.get("gpu-layers"))])
            if str(cmd_options_for_benchmark.get("threads", "auto")).lower() != "auto":
                server_args_list.extend(["--threads", str(cmd_options_for_benchmark.get("threads"))])
            
            handled_cmd_keys = {"bin", "port", "model", "ctx-size", "gpu-layers", "threads"}
            for key, value in cmd_options_for_benchmark.items():
                if key in handled_cmd_keys: continue
                cli_flag = f"--{key.replace('_', '-')}"
                if isinstance(value, bool) and value: server_args_list.append(cli_flag)
                elif value not in (None, False, "auto", "Auto", ""): server_args_list.extend([cli_flag, str(value)])
            
            for key, value in model_data_original.items():
                if key == "cmd" or key == "sampling" or key in MODEL_CONFIG_META_KEYS_BENCHMARK: continue
                cli_flag = f"--{key.replace('_', '-')}"
                if isinstance(value, bool) and value: server_args_list.append(cli_flag)
                elif value not in (None, False, "auto", "Auto", ""): server_args_list.extend([cli_flag, str(value)])

            sampling_conf = model_data_original.get("sampling")
            if isinstance(sampling_conf, dict):
                for key, s_value in sampling_conf.items():
                    cli_flag = f"--{key.replace('_', '-')}"
                    server_args_list.extend([cli_flag, str(s_value)])
            elif sampling_conf is not None:
                logger_instance.warn(f"  'sampling' block for model '{model_alias}' is not a dictionary, ignoring.")

            logger_instance.info(f"  Server command arguments prepared for benchmark.")
            logger_instance.debug(f"    Final server_executable for Popen: '{server_executable}'")
            logger_instance.debug(f"    Final server_args_list for Popen: {server_args_list}")

            if is_verbose: # For reference logging only
                # Construct a temporary config dict for build_llama_server_command_util
                config_for_build_util_log = {k: v for k, v in model_data_original.items() if k not in ["cmd", "generated_cmd_str"]}
                config_for_build_util_log["cmd"] = cmd_options_for_benchmark # Use the one with the overridden port
                if "sampling" in model_data_original: config_for_build_util_log["sampling"] = model_data_original["sampling"]
                config_for_build_util_log["_name_for_log"] = model_alias

                full_cmd_str_for_log = build_llama_server_command_util(config_for_build_util_log)
                logger_instance.debug(f"    Full command string (for reference via build_util): {full_cmd_str_for_log}")

        except Exception as e:
            current_run_error_message = f"Cmd Prepare Error: {e}"
            logger_instance.error(f"  Could not prepare server command for '{model_alias}': {e}. Skipping model.")
            if is_verbose: import traceback; logger_instance.debug(traceback.format_exc())
            all_results_data.append([
                model_alias, param_size, quantization, ctx_size_str, timestamp,
                mem_status, mem_gpu, mem_cpu,
                "Cmd Prepare Error", "", "", "", "", "", base_proxy_url, current_run_error_message
            ])
            continue
        
        logger_instance.step(f"Starting server, scanning memory, and benchmarking for '{model_alias}'...")
        server_process: Optional[subprocess.Popen] = None
        stderr_log_path: Optional[Path] = None
        
        try:
            server_process_info = start_llama_server(
                executable_path_str=server_executable,
                arguments_list=server_args_list,
                model_name=model_alias,
                temp_dir=temp_dir_path, # Use the passed temp_dir_path
                logger_instance=logger_instance,
                project_root_for_resolution=project_root_path
            )
            if not server_process_info:
                mem_status = "Server Start Failed"
                bench_status = "Start Failed"
                current_run_error_message = "Server process failed to start (see logs for details from start_llama_server)."
                raise ServerOperationError("Server start failed")
            
            server_process, _, stderr_log_path = server_process_info

            healthy = wait_for_server_health(
                process=server_process,
                health_check_url=health_url_template,
                timeout_s=health_timeout_s,
                poll_interval_s=health_poll_s,
                model_name=model_alias,
                logger_instance=logger_instance
            )
            if not healthy:
                if server_process.poll() is not None:
                    exit_code = server_process.returncode
                    status_msg = f"Server Exited (Code: {exit_code})"
                    mem_status = bench_status = status_msg
                    current_run_error_message = f"Server (PID {server_process.pid}) exited (code {exit_code}) before/during health check."
                    logger_instance.warn(f"  {current_run_error_message}")
                    if stderr_log_path: _dump_stderr_on_failure(stderr_log_path, model_alias, logger_instance)
                else:
                    status_msg = "Health Timeout"
                    mem_status = bench_status = status_msg
                    current_run_error_message = f"Server (PID {server_process.pid}) did not become healthy (timed out)."
                    logger_instance.warn(f"  {current_run_error_message} Server will be stopped.")
                raise ServerOperationError("Server not healthy")

            logger_instance.info(f"  Server for '{model_alias}' is healthy. Proceeding with memory scan and benchmark.")

            logger_instance.info(f"  Parsing memory usage from logs for '{model_alias}'...")
            if server_process.poll() is None and stderr_log_path:
                # Pass logger_instance to parse_memory_from_log
                gpu_gb_val, cpu_gb_val, mem_status_val = parse_memory_from_log(stderr_log_path, model_alias, logger_instance)
                mem_gpu, mem_cpu, mem_status = gpu_gb_val, cpu_gb_val, mem_status_val
                logger_instance.info(f"  Memory Scan Result for '{model_alias}': GPU {Fore.CYAN}{mem_gpu} GB{Style.RESET_ALL}, CPU {Fore.CYAN}{mem_cpu} GB{Style.RESET_ALL} - Status: {color_status(mem_status)}")
            elif stderr_log_path: # Server died after health but before parsing
                exit_code = server_process.returncode if server_process.returncode is not None else "Unknown"
                mem_status = f"Server Exited Prematurely (Code: {exit_code})"
                if bench_status == "Not Run": bench_status = mem_status # Propagate status
                err_msg = f"Server for '{model_alias}' (PID {server_process.pid}) died (code {exit_code}) after health check, before memory parsing."
                current_run_error_message = f"{current_run_error_message}; {err_msg}" if current_run_error_message else err_msg
                logger_instance.warn(f"  {err_msg}")
                _dump_stderr_on_failure(stderr_log_path, model_alias, logger_instance)
                raise ServerOperationError("Server died before memory parsing")
            else: # No stderr_log_path, should not happen if start_server succeeded
                 mem_status = "Error (No Log Path)"
                 current_run_error_message = "Internal error: stderr_log_path not available for memory parsing."
                 logger_instance.warn(f"  {current_run_error_message}")
            
            api_error_message_specific = "" # Specific error from API block
            if server_process.poll() is None: # Check again if server is running before API call
                logger_instance.info(f"  Sending benchmark request to {api_endpoint_template} for '{model_alias}'...")
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
                                if completion_t_val == 0: logger_instance.warn("    Benchmark warning: Completion tokens is 0, TPS will be 0.00.")
                                if req_duration_val <= 1e-6 : logger_instance.warn("    Benchmark warning: Request duration too short, TPS may be inaccurate.")
                            bench_status = "Success"
                            logger_instance.success(f"  Benchmark for '{model_alias}' completed.")
                            logger_instance.info(f"    Duration: {Fore.CYAN}{duration_s}s{Style.RESET_ALL}, Tokens (P/C/T): {prompt_t}/{completion_t}/{total_t}, TPS: {Fore.CYAN}{tps}{Style.RESET_ALL}")
                        except json.JSONDecodeError as e_json_dec:
                            bench_status = "API Response Error"
                            api_error_message_specific = f"JSON decode error: {e_json_dec} - Response text: {response.text[:100]}"
                            logger_instance.error(f"  Error decoding JSON response for '{model_alias}': {api_error_message_specific}")
                        except Exception as e_json_proc:
                            bench_status = "API Response Error"
                            api_error_message_specific = f"Response processing error: {e_json_proc}"
                            logger_instance.error(f"  Error processing API response for '{model_alias}': {api_error_message_specific}")
                    else:
                        bench_status = "API Request Failed"
                        api_error_message_specific = f"HTTP {response.status_code}: {response.text[:200]}"
                        logger_instance.error(f"  API request for '{model_alias}' failed: {api_error_message_specific}")
                except requests.Timeout:
                    duration_s = f"{time.monotonic() - req_start_time:.3f}"
                    bench_status = "API Request Failed"
                    api_error_message_specific = "Request timed out"
                    logger_instance.error(f"  API request for '{model_alias}' timed out after {duration_s}s.")
                except requests.RequestException as e_req:
                    duration_s = f"{time.monotonic() - req_start_time:.3f}"
                    bench_status = "API Request Failed"
                    api_error_message_specific = f"Request Exception: {e_req}"
                    logger_instance.error(f"  API request exception for '{model_alias}': {api_error_message_specific}")
                
                if api_error_message_specific: # If there was an API error, append it
                    current_run_error_message = f"{current_run_error_message}; {api_error_message_specific}" if current_run_error_message else api_error_message_specific
            
            elif stderr_log_path: # Server died before API call (but after memory parse check)
                exit_code = server_process.returncode if server_process.returncode is not None else "Unknown"
                if bench_status == "Not Run": bench_status = f"Server Exited Prematurely (Code: {exit_code})"
                err_msg = f"Server for '{model_alias}' (PID {server_process.pid}) died (code {exit_code}) before benchmark API call."
                current_run_error_message = f"{current_run_error_message}; {err_msg}" if current_run_error_message else err_msg
                logger_instance.warn(f"  {err_msg}")
                _dump_stderr_on_failure(stderr_log_path, model_alias, logger_instance)
            
            logger_instance.info(f"  Benchmark Result for '{model_alias}': Status: {color_status(bench_status)}")
            if api_error_message_specific and bench_status != "Success": # Log details if specific API error occurred
                 logger_instance.debug(f"    Error details for benchmark API of '{model_alias}': {api_error_message_specific}")

        except ServerOperationError as e_soe: # Custom exception for controlled exit from this model's processing
            logger_instance.debug(f"Server operation for '{model_alias}' ended due to: {e_soe}. Statuses captured should reflect this.")
            # current_run_error_message should already be set by the block raising ServerOperationError
        except Exception as e_outer_scope: # Unexpected errors during this model's processing
            logger_instance.error(f"Unexpected error during combined processing of '{model_alias}': {e_outer_scope}")
            err_msg_unexpected = f"Unexpected script error: {str(e_outer_scope)[:200]}"
            current_run_error_message = f"{current_run_error_message}; {err_msg_unexpected}" if current_run_error_message else err_msg_unexpected
            if bench_status == "Not Run": bench_status = "Script Error"
            if mem_status == "Not Scanned": mem_status = "Script Error"
            if is_verbose:
                import traceback
                logger_instance.error("Traceback:\n" + traceback.format_exc())
        finally:
            if server_process:
                stop_llama_server(server_process, model_alias, logger_instance)
        
        all_results_data.append([
            model_alias, param_size, quantization, ctx_size_str, timestamp,
            mem_status, mem_gpu, mem_cpu,
            bench_status, duration_s, tps, prompt_t, completion_t, total_t,
            base_proxy_url, current_run_error_message.strip("; ")
        ])
        logger_instance.notice("-" * 30)

    logger_instance.header("Writing Benchmark Results to CSV")
    try:
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_header)
            writer.writerows(all_results_data)
        logger_instance.success(f"All benchmark results saved to: {output_csv_path.resolve()}")
    except IOError as e:
        logger_instance.error(f"Failed to write CSV data to {output_csv_path}: {e}")
        if is_verbose: import traceback; logger_instance.error(f"Traceback for CSV write error:\n{traceback.format_exc()}")


# --- Cleanup and Main Execution ---
def signal_cleanup_handler_benchmark(signum, frame): # Renamed for clarity
    global logger, TEMP_DIR_MANAGER_PATH 
    log_func = print if not logger else logger.warn
    
    log_func(f"\nSignal {signal.Signals(signum).name} received. Initiating cleanup for benchmark-models...")
    if TEMP_DIR_MANAGER_PATH:
        log_func(f"  Note: Temporary directory {TEMP_DIR_MANAGER_PATH} should be auto-cleaned if using 'with' context, "
                 "or may persist if --keep-temp-logs was used.")
    
    # Add any specific cleanup for benchmark script if needed (e.g. kill by port)
    # For now, relying on individual server stops and TemporaryDirectory cleanup for non-kept logs.
    
    if logger: logger.info("Benchmark script terminated by signal.")
    else: print("Benchmark script terminated by signal.")
    if logger and hasattr(logger, 'close') and callable(logger.close): logger.close()
    sys.exit(1)

def main():
    global logger, TEMP_DIR_MANAGER_PATH, project_root_dir

    parser = argparse.ArgumentParser(
        description="Cross-platform LLM Benchmarking Script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # kept args (no -o/--output, --keep-temp-logs, or --log-file)
    default_base_config_path = (project_root_dir / "config.base.yaml").resolve()
    parser.add_argument(
        "-c", "--config", type=Path, default=default_base_config_path,
        help="Path to the base YAML configuration file."
    )
    default_override_path_hostname = (project_root_dir / "overrides" / f"{platform.node().split('.')[0].lower()}.yaml").resolve()
    calculated_default_override = default_override_path_hostname if default_override_path_hostname.exists() else None
    parser.add_argument(
        "--override", type=Path, default=calculated_default_override,
        help="Path to override YAML. (Default logic checks hostname based override)"
    )
    parser.add_argument("-q", "--question", type=str, default=DEFAULT_QUESTION, help="Question/prompt for benchmarking.")
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_HEALTH_POLL_INTERVAL_S, help="Health check poll interval (seconds).")
    parser.add_argument("--health-timeout", type=int, default=DEFAULT_HEALTH_TIMEOUT_S_BENCH, help=f"Override health check timeout (seconds).")
    parser.add_argument("-m", "--model", type=str, help="Benchmark only this specific model alias (from config).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose/debug output.")

    args = parser.parse_args()

    # --- Forced static locations with timestamp ---
    ts = _timestamp()

    # Ensure logs dir exists
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # CSV goes alongside this script, timestamped; CSVs are kept forever
    output_file_abs_csv = (SCRIPT_DIR / f"{CSV_BASENAME}_{ts}.csv").resolve()
    try:
        output_file_abs_csv.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"CRITICAL ERROR: Could not create directory for CSV file: {output_file_abs_csv.parent} - {e}", file=sys.stderr)
        sys.exit(1)

    # Main script log inside logs/, timestamped (logs are rotated)
    main_log_file_path = (LOGS_DIR / f"benchmark_{ts}.log").resolve()
    try:
        main_log_file_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"CRITICAL ERROR: Could not create directory for main log file: {main_log_file_path.parent} - {e}", file=sys.stderr)
        sys.exit(1)

    # Initialize logger
    logger = Logger(verbose=args.verbose, log_file_path=main_log_file_path)

    try:
        logger.header("LLM BENCHMARKER INITIALIZATION")
        logger.info(f"Main script log file: {main_log_file_path}")

        signal.signal(signal.SIGINT, signal_cleanup_handler_benchmark)
        signal.signal(signal.SIGTERM, signal_cleanup_handler_benchmark)
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
                logger.warn(f"Specified override configuration not found: {resolved_override}. Proceeding without.")
                resolved_override = None
        else:
            logger.info("No override configuration file specified or found by default logic.")

        # --- Initial cleanup of potential lingering llama-server processes on our static port ---
        logger.step(f"Performing initial cleanup of potentially lingering llama-server processes on port {STATIC_BENCHMARK_PORT}...")
        killed_procs = 0
        target_port_arg_bench = f"--port {STATIC_BENCHMARK_PORT}"

        llama_exec_patterns = ["llama-server", "server.exe", "server"]
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

                if target_port_arg_bench in cmdline_str:
                    is_potential_llama_server = True
                elif proc_name in llama_exec_patterns:
                    is_potential_llama_server = True
                elif any(patt in exe_path for patt in llama_path_patterns):
                    is_potential_llama_server = True
                elif any(patt in cmdline_str for patt in llama_path_patterns):
                    is_potential_llama_server = True

                if is_potential_llama_server:
                    current_user = psutil.Process().username()
                    if username and username != current_user and "root" in (username.lower() if username else "") and target_port_arg_bench not in cmdline_str:
                        logger.debug(f"  PID {pid} ('{proc_name_raw}') matched pattern but owned by '{username}' (not current user '{current_user}' or root on target port). Skipping kill.")
                        continue

                    system_daemon_names = ["windowserver", "systemuiserver", "cvmserver", "nfstorageserver", "powerd", "logd"]
                    if proc_name in system_daemon_names and target_port_arg_bench not in cmdline_str:
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
            logger.info(f"No lingering server processes found (port {STATIC_BENCHMARK_PORT} or patterns).")

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
            if args.health_timeout is not None and args.health_timeout != DEFAULT_HEALTH_TIMEOUT_S_BENCH
            else effective_conf_dict.get("healthCheckTimeout", DEFAULT_HEALTH_TIMEOUT_S_BENCH)
        )
        if not isinstance(health_timeout_final, int) or health_timeout_final <= 0:
            logger.warn(f"Invalid healthCheckTimeout '{health_timeout_final}'. Using default: {DEFAULT_HEALTH_TIMEOUT_S_BENCH}s.")
            health_timeout_final = DEFAULT_HEALTH_TIMEOUT_S_BENCH
        logger.info(f"Effective health check timeout: {health_timeout_final}s")
        logger.info(f"Health check poll interval: {args.poll_interval}s")

        # --- Persistent per-run server logs under logs/run_<ts> ---
        run_logs_dir = LOGS_DIR / f"{RUN_DIR_PREFIX}{ts}"
        run_logs_dir.mkdir(parents=True, exist_ok=True)
        TEMP_DIR_MANAGER_PATH = run_logs_dir  # for signal handler info

        logger.info(f"Server logs will be kept in: {run_logs_dir.resolve()}")
        logger.header("STARTING BENCHMARK RUN")

        processed_models_data = effective_conf_dict.get("models", {})
        if not isinstance(processed_models_data, dict) or not processed_models_data:
            logger.error("'models' section not found/empty in config. Cannot run benchmark.")
            sys.exit(1)

        script_start_time = time.monotonic()

        run_benchmark(
            processed_models_config=processed_models_data,
            output_csv_path=output_file_abs_csv,
            question=args.question,
            health_timeout_s=health_timeout_final,
            health_poll_s=args.poll_interval,
            model_to_test_alias=args.model,
            temp_dir_path=run_logs_dir,        # ALWAYS under logs/
            logger_instance=logger,
            project_root_path=project_root_dir
        )

        # Keep a convenience "latest" CSV copy without timestamp
        try:
            latest_csv = SCRIPT_DIR / f"{CSV_BASENAME}.csv"
            shutil.copy2(output_file_abs_csv, latest_csv)
            logger.info(f"Also wrote latest CSV copy: {latest_csv}")
        except Exception as e:
            logger.warn(f"Could not write latest CSV copy: {e}")

        # --- Retention: rotate ONLY logs (keep last RETENTION_KEEP) ---
        _enforce_retention(LOGS_DIR, "benchmark_*.log", keep=RETENTION_KEEP, delete_dirs=False, logger_instance=logger)
        _enforce_retention(LOGS_DIR, f"{RUN_DIR_PREFIX}*", keep=RETENTION_KEEP, delete_dirs=True, logger_instance=logger)
        # Note: CSVs are NOT rotated and are kept indefinitely by design.

        script_duration_s = time.monotonic() - script_start_time
        logger.header("BENCHMARK SCRIPT COMPLETE")
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
    main()