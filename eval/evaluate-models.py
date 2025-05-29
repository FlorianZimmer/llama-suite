#!/usr/bin/env python3
"""
LLM Evaluation Script using lm-evaluation-harness.
Focuses on robustly restoring os.unlink after calls to lm-eval components
that are known to modify it (like humaneval's code_eval).
Outputs results to a run-specific timestamped directory.
Handles Qwen3 models by conditionally prepending /no_think to prompts based on the task.
"""

# --- Core Python Imports (Capture originals before other libraries can modify them) ---
import os
_original_os_unlink = os.unlink
import shutil
# _original_shutil_rmtree = shutil.rmtree
import tempfile
# import weakref

# --- Store original TemporaryDirectory for script's own use ---
_original_tempfile_TemporaryDirectory = tempfile.TemporaryDirectory

# --- Standard Library Imports ---
import argparse
import atexit
import csv
import datetime
import platform
import re
import shlex
import signal
import subprocess
import sys
# tempfile already imported
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TextIO
import multiprocessing
import logging as std_logging
import json as json_serializer
import dataclasses
import traceback # ADDED for detailed tracebacks

# --- Third-Party Library Imports ---
import psutil
import requests
from colorama import Fore, Style, init as colorama_init_eval

from lm_eval import simple_evaluate
import lm_eval.tasks
from lm_eval.utils import make_table
from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import LocalCompletionsAPI
from lm_eval.api.instance import Instance

# --- Setup a logger for preprocess_for_json ---
# This ensures it uses the same configuration as other lm_eval logs.
_preprocess_logger = std_logging.getLogger("lm_eval.preprocess_for_json")


# --- Custom lm-eval Model for Qwen3 ---
_qwen_custom_api_logger = std_logging.getLogger("lm_eval.QwenLocalCompletionsAPI") # More specific logger name

@register_model("custom-local-completions-qwen")
class QwenLocalCompletionsAPIWithNoThink(LocalCompletionsAPI):
    def __init__(self, 
                 model_alias_for_qwen_check: Optional[str] = None, 
                 no_prefix_tasks_str: str = "humaneval;mbpp", 
                 **kwargs):
        super().__init__(**kwargs)
        
        self.effective_model_name_for_log = model_alias_for_qwen_check if model_alias_for_qwen_check else self.model
        
        self.is_qwen3_model_for_no_think = False
        if model_alias_for_qwen_check and "qwen3" in model_alias_for_qwen_check.lower():
            self.is_qwen3_model_for_no_think = True
            _qwen_custom_api_logger.info(
                f"({self.effective_model_name_for_log}): Initialized. Detected as Qwen3. Conditional '/no_think' prefixing enabled."
            )
        else:
            _qwen_custom_api_logger.debug(
                f"({self.effective_model_name_for_log}): Initialized. NOT detected as Qwen3. '/no_think' prefixing will be disabled."
            )

        self.no_prefix_tasks_set = set()
        if self.is_qwen3_model_for_no_think and no_prefix_tasks_str:
            self.no_prefix_tasks_set = {task.strip() for task in no_prefix_tasks_str.split(';') if task.strip()} 
            if self.no_prefix_tasks_set:
                 _qwen_custom_api_logger.info(
                    f"({self.effective_model_name_for_log}): Tasks excluded from '/no_think' prefix: {self.no_prefix_tasks_set}"
                )
            else:
                _qwen_custom_api_logger.info(
                    f"({self.effective_model_name_for_log}): No tasks specified for exclusion via 'no_prefix_tasks_str' (value was '{no_prefix_tasks_str}'). '/no_think' may be prefixed for all applicable tasks."
                )
        elif self.is_qwen3_model_for_no_think:
            _qwen_custom_api_logger.info(
                f"({self.effective_model_name_for_log}): 'no_prefix_tasks_str' was empty or not provided. '/no_think' will be prefixed for all applicable tasks."
            )

    def _should_prefix_prompt_for_task(self, task_name: str) -> bool:
        if not self.is_qwen3_model_for_no_think:
            return False 
        
        if task_name in self.no_prefix_tasks_set:
            _qwen_custom_api_logger.debug(
                f"({self.effective_model_name_for_log}): Task '{task_name}' IS IN exclusion list {self.no_prefix_tasks_set}. SKIPPING '/no_think' prefix."
            )
            return False
        else:
            _qwen_custom_api_logger.debug(
                f"({self.effective_model_name_for_log}): Task '{task_name}' IS NOT in exclusion list {self.no_prefix_tasks_set}. WILL ATTEMPT '/no_think' prefix."
            )
            return True

    def _conditionally_prefix_requests(self, requests: List[Instance], is_generate_until: bool) -> List[Instance]:
        if not self.is_qwen3_model_for_no_think:
            return requests

        processed_requests: List[Instance] = []
        prefix_applied_count = 0
        prefix_skipped_count = 0

        for i, instance in enumerate(requests):
            task_name = instance.task_name
            # Access the first argument (the context string) from the arguments tuple
            current_context = instance.arguments[0] # <<< CHANGE HERE
            
            new_context = current_context
            should_prefix = self._should_prefix_prompt_for_task(task_name)

            if should_prefix:
                if isinstance(current_context, str):
                    if not current_context.startswith("/no_think"):
                        new_context = "/no_think " + current_context
                        prefix_applied_count += 1
                    else: 
                        prefix_skipped_count +=1 
                elif isinstance(current_context, list) and getattr(self, 'tokenized_requests', False):
                     _qwen_custom_api_logger.warning(
                        f"({self.effective_model_name_for_log}): Task '{task_name}' (Request {i+1}) is Qwen3 and requires prefix, "
                        "but is using 'tokenized_requests=True'. Conditional prepending of '/no_think' to tokenized "
                        "prompts is not automatically handled."
                    )
                     prefix_skipped_count +=1
                else: 
                    prefix_skipped_count +=1
            else: 
                prefix_skipped_count +=1
            
            if new_context is not current_context:
                # Construct the new arguments tuple
                if is_generate_until:
                    # For generate_until, arguments is (context, gen_kwargs)
                    # instance.arguments[1] should be the gen_kwargs dictionary
                    new_arguments_tuple = (new_context, instance.arguments[1]) # <<< CHANGE HERE
                else: # For loglikelihood, arguments is (context, continuation)
                    # instance.arguments[1] should be the continuation string
                    new_arguments_tuple = (new_context, instance.arguments[1]) # <<< CHANGE HERE
                
                # Replace the 'arguments' field
                processed_requests.append(dataclasses.replace(instance, arguments=new_arguments_tuple)) # <<< CHANGE HERE
            else:
                processed_requests.append(instance)
        
        if self.is_qwen3_model_for_no_think and (prefix_applied_count > 0 or prefix_skipped_count > 0) :
            _qwen_custom_api_logger.info(
                f"({self.effective_model_name_for_log}): Processed {len(requests)} requests for {'generate_until' if is_generate_until else 'loglikelihood'}. "
                f"'/no_think' prefix applied to {prefix_applied_count} requests, skipped for {prefix_skipped_count} requests."
            )
        return processed_requests


# --- Project-Specific Imports (utils.config_utils) ---
_current_script_file_path_for_root = Path(__file__).resolve()
project_root_dir = _current_script_file_path_for_root.parent.parent
if str(project_root_dir) not in sys.path:
    sys.path.insert(0, str(project_root_dir))
from utils.config_utils import generate_processed_config, build_llama_server_command_util


# --- Environment Variable Configuration ---
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Constants ---
DEFAULT_BASE_OUTPUT_DIR_NAME = "results"
DEFAULT_TASKS = "humaneval"
DEFAULT_LIMIT_SAMPLES: Optional[float] = None
DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_FEWSHOT: Optional[int] = None
STATIC_EVAL_PORT = "9998"
DEFAULT_HEALTH_POLL_INTERVAL_S = 2.0
DEFAULT_HEALTH_TIMEOUT_S = 3600
PROCESS_TERMINATE_TIMEOUT_S = 10
DEFAULT_QWEN3_NO_PREFIX_TASKS_STR_CLI = "humaneval;mbpp"

# --- Global Variables ---
logger: Optional['Logger'] = None
TEMP_DIR_MANAGER_PATH_EVAL: Optional[Path] = None

# --- ATEIXT Restorer for os.unlink ---
def final_os_unlink_restorer_atexit():
    global _original_os_unlink
    if _original_os_unlink is None:
        print("ATEIXT_CRITICAL: _original_os_unlink is None.", file=sys.stderr); return
    current_unlink = getattr(os, 'unlink', None)
    if current_unlink is None or current_unlink != _original_os_unlink:
        print("ATEIXT_INFO: os.unlink was incorrect at final Python exit. Restoring globally.", file=sys.stderr)
        os.unlink = _original_os_unlink
atexit.register(final_os_unlink_restorer_atexit)

# --- Logger Class ---
class Logger:
    def __init__(self, verbose: bool = False):
        self.verbose_flag = verbose
        colorama_init_eval(autoreset=True)
    def _get_timestamp(self) -> str: return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    def _log(self, level: str, message: str, color: str = "", bright: bool = False, file: TextIO = sys.stdout):
        style_prefix = Style.BRIGHT if bright else ""
        timestamp = self._get_timestamp()
        print(f"{timestamp} [{style_prefix}{color}{level}{Style.RESET_ALL}] {style_prefix}{color}{message}{Style.RESET_ALL}", file=file)
    def info(self, message: str): self._log("INFO", message, Fore.BLUE)
    def warn(self, message: str): self._log("WARN", message, Fore.YELLOW, bright=True)
    def error(self, message: str): self._log("ERROR", message, Fore.RED, bright=True, file=sys.stderr)
    def success(self, message: str): self._log("SUCCESS", message, Fore.GREEN, bright=True)
    def debug(self, message: str):
        if self.verbose_flag: self._log("DEBUG", message, Fore.MAGENTA, file=sys.stderr) # Changed to stderr for debug
    def header(self, title: str): print(f"\n{Fore.CYAN}{Style.BRIGHT}{'=' * 70}\n{title.center(70)}\n{'=' * 70}{Style.RESET_ALL}")
    def subheader(self, title: str): print(f"\n{Fore.CYAN}{'-' * 70}\n{title.center(70)}\n{'-' * 70}{Style.RESET_ALL}")
    def step(self, message: str): print(f"{Fore.CYAN}>> {message}{Style.RESET_ALL}")
    def notice(self, message: str): print(f"{Fore.WHITE}{message}{Style.RESET_ALL}")

class ServerOperationError(Exception): pass

# --- Utility Functions ---
def _ensure_os_unlink_restored(context_message: str, logger_instance: Optional[Logger]):
    log_func = print if logger_instance is None else logger_instance.debug
    current_unlink = getattr(os, 'unlink', None)
    if current_unlink is None or current_unlink != _original_os_unlink:
        log_func(f"DEBUG_OS_UNLINK_RESTORE: {context_message} - os.unlink was '{str(current_unlink)}'. Attempting restore.")
        if _original_os_unlink is not None:
            os.unlink = _original_os_unlink
        elif logger_instance: logger_instance.error("CRITICAL: Cannot restore os.unlink: _original_os_unlink is None.")
        else: print("CRITICAL: Cannot restore os.unlink: _original_os_unlink is None.", file=sys.stderr)

def color_status(status: str) -> str:
    status_lower = status.lower()
    if "success" in status_lower: return f"{Fore.GREEN}{Style.BRIGHT}{status}{Style.RESET_ALL}"
    if any(term in status_lower for term in ["fail", "error", "timeout", "invalid", "missing", "exited"]): return f"{Fore.RED}{Style.BRIGHT}{status}{Style.RESET_ALL}"
    if any(term in status_lower for term in ["warn", "not run", "parse error"]): return f"{Fore.YELLOW}{Style.BRIGHT}{status}{Style.RESET_ALL}"
    return status

def _dump_stderr_on_failure(stderr_log_path: Optional[Path], model_name: str, logger_instance: Logger):
    if not stderr_log_path or not stderr_log_path.exists():
        logger_instance.warn(f"  Stderr log for '{model_name}' not available (Path: {stderr_log_path}).")
        return
    logger_instance.warn(f"  Attempting to read last lines from stderr log: {stderr_log_path}")
    try:
        with open(stderr_log_path, 'r', encoding='utf-8', errors='replace') as err_f_read:
            lines = err_f_read.readlines(); last_lines = lines[-20:]
            if last_lines:
                logger_instance.warn("  Last lines of server stderr:")
                for line in last_lines: logger_instance.warn(f"    {line.strip()}")
            else: logger_instance.warn(f"    Log file '{stderr_log_path}' was empty.")
    except Exception as log_read_e: logger_instance.warn(f"    Could not read stderr log ({stderr_log_path}): {log_read_e}")

def start_llama_server( executable_path_str: str, arguments_str: str, model_name: str, temp_dir: Path, logger_instance: Logger ) -> Optional[Tuple[subprocess.Popen, Path, Path]]:
    exec_path = Path(executable_path_str)
    try:
        resolved_exec = str(exec_path.resolve(strict=True)) if exec_path.is_file() else executable_path_str
        args_list = [resolved_exec] + shlex.split(arguments_str)
    except (ValueError, FileNotFoundError) as e: logger_instance.error(f"Error preparing command for '{model_name}': {e}"); return None
    stdout_log = temp_dir / f"{model_name}_{datetime.datetime.now().strftime('%H%M%S%f')}_stdout.log"
    stderr_log = temp_dir / f"{model_name}_{datetime.datetime.now().strftime('%H%M%S%f')}_stderr.log"
    logger_instance.info(f"  Attempting to start server for '{model_name}'...")
    logger_instance.debug(f"    Command: {' '.join(args_list)}")
    try:
        with open(stdout_log, 'wb') as f_out, open(stderr_log, 'wb') as f_err:
            process = subprocess.Popen(args_list, stdout=f_out, stderr=f_err)
        time.sleep(0.7)
        if process.poll() is not None:
            logger_instance.error(f"Server for '{model_name}' failed to start or exited immediately (Code: {process.returncode}).")
            _dump_stderr_on_failure(stderr_log, model_name, logger_instance); return None
        logger_instance.success(f"  Server for '{model_name}' started (PID: {process.pid}).")
        return process, stdout_log, stderr_log
    except FileNotFoundError: logger_instance.error(f"Executable not found for '{model_name}': {args_list[0]}"); return None
    except Exception as e: logger_instance.error(f"Exception starting server for '{model_name}': {e}"); return None

def stop_llama_server(process: Optional[subprocess.Popen], model_name: str, logger_instance: Logger):
    if not process: return
    pid = process.pid
    if process.poll() is not None: logger_instance.info(f"  Server '{model_name}' (PID {pid}) was already stopped (Code: {process.returncode})."); return
    logger_instance.info(f"  Stopping server '{model_name}' (PID {pid})...")
    try:
        process.terminate(); process.wait(timeout=PROCESS_TERMINATE_TIMEOUT_S)
        logger_instance.success(f"  Server '{model_name}' (PID {pid}) terminated gracefully.")
    except subprocess.TimeoutExpired:
        logger_instance.warn(f"  Server '{model_name}' (PID {pid}) did not terminate gracefully. Forcing kill...")
        try: process.kill(); process.wait(timeout=5); logger_instance.success(f"  Server '{model_name}' (PID {pid}) killed.")
        except Exception as e_kill: logger_instance.error(f"  Error during force kill of server '{model_name}' (PID {pid}): {e_kill}")
    except Exception as e_term: logger_instance.warn(f"  Error during graceful termination of '{model_name}' (PID {pid}): {e_term}.")

def wait_for_server_health( process: subprocess.Popen, health_check_url: str, timeout_s: int, poll_interval_s: float, model_name: str, logger_instance: Logger ) -> bool:
    logger_instance.info(f"  Waiting for '{model_name}' (PID: {process.pid}) health at {health_check_url} (timeout: {timeout_s}s)...")
    start_time = time.monotonic(); attempt = 0
    while time.monotonic() - start_time < timeout_s:
        attempt += 1; req_timeout = max(1.0, poll_interval_s - 0.5)
        if process.poll() is not None: logger_instance.warn(f"  Server '{model_name}' exited prematurely (Code: {process.returncode})"); return False
        try:
            response = requests.get(health_check_url, timeout=req_timeout)
            if response.status_code == 200:
                try:
                    health_data = response.json()
                    if health_data.get("status") in ["ok", "healthy"]: logger_instance.success(f"  Server '{model_name}' healthy (Attempt {attempt}, Status: {health_data.get('status')})."); return True
                except requests.exceptions.JSONDecodeError: logger_instance.success(f"  Server '{model_name}' healthy (Attempt {attempt}, HTTP 200 but not JSON)."); return True
                logger_instance.debug(f"  Health check '{model_name}' (Attempt {attempt}): HTTP 200, but status is '{health_data.get('status', 'N/A')}'.")
            else: logger_instance.debug(f"  Health check '{model_name}' (Attempt {attempt}): HTTP {response.status_code}.")
        except requests.exceptions.ConnectionError: logger_instance.debug(f"  Health check '{model_name}' (Attempt {attempt}): Connection refused.")
        except requests.exceptions.Timeout: logger_instance.debug(f"  Health check '{model_name}' (Attempt {attempt}): Request timed out.")
        except requests.RequestException as e: logger_instance.debug(f"  Health check '{model_name}' (Attempt {attempt}): Request failed: {e}.")
        sleep_until = time.monotonic() + poll_interval_s
        while time.monotonic() < sleep_until:
            if process.poll() is not None: logger_instance.warn(f"  Server '{model_name}' exited (Code: {process.returncode}) during health poll interval."); return False
            time.sleep(0.1)
    logger_instance.warn(f"  Server '{model_name}' did not become healthy within {timeout_s}s."); return False

def preprocess_for_json(data: Any) -> Any: # MODIFIED
    if isinstance(data, (str, int, float, bool, type(None))):
        return data
    if isinstance(data, dict):
        return {preprocess_for_json(k): preprocess_for_json(v) for k, v in data.items()}
    if isinstance(data, list) or isinstance(data, tuple):
        return [preprocess_for_json(elem) for elem in data]
    if callable(data):
        return f"<function {getattr(data, '__name__', 'unknown_callable')}>"
    
    # Handle numpy types or types with item()
    if hasattr(data, 'item') and callable(data.item): 
        try:
            item_val = data.item()
            # Further check if item_val itself is simple
            if isinstance(item_val, (str, int, float, bool, type(None))):
                return item_val
            return str(item_val) # If item() returns complex obj, stringify
        except Exception:
            return str(data) 
    
    # Handle types with tolist()
    if hasattr(data, 'tolist') and callable(data.tolist): 
        try:
            list_val = data.tolist()
            # Recursively process the list, as it might contain non-serializable items
            return preprocess_for_json(list_val)
        except Exception:
            return str(data)
    
    # Final fallback for other types (e.g. sets, custom objects)
    try:
        # This is a test, not the primary conversion method.
        # If json.dumps can handle it, it's likely a simple type missed by explicit checks.
        json_serializer.dumps(data) 
        return data # Should be rare to reach here if not caught by initial checks
    except TypeError:
        _preprocess_logger.debug(f"Defaulting to str() for type {type(data)}, value (first 100 chars): {str(data)[:100]}...")
        return str(data)
    except Exception as e_json_test:
        _preprocess_logger.warning(f"Error during serialization test for type {type(data)}: {e_json_test}. Defaulting to str(). Value (first 100 chars): {str(data)[:100]}")
        return str(data)


# --- Core Evaluation Logic ---
def run_evaluation(
    processed_models_config: Dict[str, Any],
    run_output_dir: Path,
    tasks_str: str,
    limit_samples: Optional[float], num_fewshot: Optional[int], batch_size_str: str,
    health_timeout_s: int, health_poll_s: float, model_to_eval_alias: Optional[str],
    temp_dir_path: Path, logger_instance: Logger,
    qwen3_no_prefix_tasks_override: str
):
    if not processed_models_config:
        logger_instance.error("No processed model configurations for evaluation."); return
    # ... (rest of the function header is the same)

    # Ensure the main run directory exists before iterating models
    # This is where the summary CSV will go.
    try:
        run_output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger_instance.error(f"CRITICAL: Could not create base run output directory: {run_output_dir} - {e}")
        return # Cannot proceed if this fails

    models_to_iterate: Dict[str, Any] = {}
    if model_to_eval_alias:
        if model_to_eval_alias in processed_models_config:
            model_entry = processed_models_config[model_to_eval_alias]
            if isinstance(model_entry, dict): models_to_iterate = {model_to_eval_alias: model_entry}
            else: logger_instance.error(f"Config for '{model_to_eval_alias}' malformed."); return
        else: logger_instance.error(f"Specified model '{model_to_eval_alias}' not found."); return
    else:
        models_to_iterate = {alias: data for alias, data in processed_models_config.items() if isinstance(data, dict)}
        if not models_to_iterate: logger_instance.error("No valid model configurations found."); return
    logger_instance.info(f"Evaluating {len(models_to_iterate)} model(s). Output to: {run_output_dir}")

    summary_csv_path = run_output_dir / "eval_summary.csv"
    csv_header = ["ModelAlias", "Task", "Metric", "Value", "Version", "NumFewshot", "Limit", "Timestamp", "Error"]
    all_summary_rows: List[List[str]] = []
    base_server_url = f"http://127.0.0.1:{STATIC_EVAL_PORT}"; health_url = f"{base_server_url}/health"
    eval_api_endpoint = f"{base_server_url}/v1/completions"

    for model_idx, (model_alias, model_data) in enumerate(models_to_iterate.items()):
        logger_instance.subheader(f"Processing Model ({model_idx + 1}/{len(models_to_iterate)}): {model_alias}")
        # Per-model JSON results go into the run-specific directory
        current_model_output_json_path = run_output_dir / f"{model_alias}_results.json"
        
        timestamp_str = logger_instance._get_timestamp()
        eval_status = "Not Run"; current_run_error_message = ""
        tasks_list = [t.strip() for t in tasks_str.split(',')]
        is_humaneval_like_task_run = any(task in ["humaneval", "mbpp"] for task in tasks_list)

        original_cmd_dict = model_data.get("cmd")
        # ... (server command prep as before) ...
        if not isinstance(original_cmd_dict, dict):
            current_run_error_message = "Config Error: Missing 'cmd' dict"
            all_summary_rows.append([model_alias, "-", "-", "-", "-", str(num_fewshot or "Def"), str(limit_samples or "None"), timestamp_str, current_run_error_message])
            continue
        eval_cmd_dict = original_cmd_dict.copy(); eval_cmd_dict["port"] = STATIC_EVAL_PORT
        if "alias" not in eval_cmd_dict: eval_cmd_dict["alias"] = model_alias
        temp_model_config_for_cmd_build = {"cmd": eval_cmd_dict}
        if "sampling" in model_data and model_data["sampling"] is not None:
            temp_model_config_for_cmd_build["sampling"] = model_data["sampling"]
        try:
            final_eval_cmd_str = build_llama_server_command_util(temp_model_config_for_cmd_build)
            cmd_parts = shlex.split(final_eval_cmd_str)
            server_executable, server_args_str = cmd_parts[0], " ".join(cmd_parts[1:])
        except Exception as e:
            current_run_error_message = f"Cmd Build Error: {e}"
            all_summary_rows.append([model_alias, "-", "-", "-", "-", str(num_fewshot or "Def"), str(limit_samples or "None"), timestamp_str, current_run_error_message])
            continue
        
        server_process: Optional[subprocess.Popen] = None
        eval_results_dict: Optional[Dict[str, Any]] = None
        
        try:
            if is_humaneval_like_task_run:
                 _ensure_os_unlink_restored(f"Before server start for '{model_alias}'", logger_instance)

            server_process_info = start_llama_server(server_executable, server_args_str, model_alias, temp_dir_path, logger_instance)
            if not server_process_info: raise ServerOperationError("Server Start Failed")
            server_process, _, stderr_log_path = server_process_info
            healthy = wait_for_server_health(server_process, health_url, health_timeout_s, health_poll_s, model_alias, logger_instance)
            if not healthy:
                exit_code_msg = f"(Code: {server_process.poll()})" if server_process.poll() is not None else ""
                _dump_stderr_on_failure(stderr_log_path, model_alias, logger_instance)
                raise ServerOperationError(f"Server Not Healthy {exit_code_msg}")

            logger_instance.info(f"  Server for '{model_alias}' healthy. Starting lm-eval...")
            hf_tokenizer_for_model = model_data.get("hf_tokenizer_for_model", model_alias)
            num_concurrent_requests = model_data.get("num_concurrent_eval", 1)
            DEFAULT_LM_EVAL_MAX_LENGTH = 2048; configured_ctx_size_str = original_cmd_dict.get("ctx-size")
            try: lm_eval_max_length = int(configured_ctx_size_str) if configured_ctx_size_str else DEFAULT_LM_EVAL_MAX_LENGTH
            except ValueError: lm_eval_max_length = DEFAULT_LM_EVAL_MAX_LENGTH
            
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
                f"max_length={lm_eval_max_length}"
            ]
            
            model_type_for_lm_eval: str
            if "qwen3" in model_alias.lower():
                logger_instance.info(f"  Model '{model_alias}' identified as Qwen3. Using custom API for conditional '/no_think' prefixing.")
                model_type_for_lm_eval = "custom-local-completions-qwen"
                lm_eval_model_args_list.append(f"model_alias_for_qwen_check={model_alias}")
                lm_eval_model_args_list.append(f"no_prefix_tasks_str={qwen3_no_prefix_tasks_override}")
            else:
                model_type_for_lm_eval = "local-completions"

            lm_eval_model_args = ",".join(lm_eval_model_args_list)
            logger_instance.debug(f"  lm-eval model_args: {lm_eval_model_args}")

            actual_batch_size_val: Union[int, str] = 1
            if batch_size_str.lower() == "auto": actual_batch_size_val = "auto"
            elif batch_size_str.isdigit(): actual_batch_size_val = int(batch_size_str)
            
            confirm_unsafe = True if is_humaneval_like_task_run else False
            if confirm_unsafe: logger_instance.warn(f"  Unsafe tasks detected. Setting confirm_run_unsafe_code=True.")

            if is_humaneval_like_task_run:
                 _ensure_os_unlink_restored(f"Before simple_evaluate for '{model_alias}'", logger_instance)
            
            eval_results_dict = simple_evaluate(
                model=model_type_for_lm_eval,
                model_args=lm_eval_model_args,
                tasks=tasks_list,
                num_fewshot=num_fewshot,
                limit=limit_samples,
                batch_size=actual_batch_size_val,
                confirm_run_unsafe_code=confirm_unsafe
            )
            
            if is_humaneval_like_task_run:
                _ensure_os_unlink_restored(f"IMMEDIATELY AFTER simple_evaluate for '{model_alias}'", logger_instance)

            # --- MODIFIED JSON Saving Block ---
            if eval_results_dict is not None:
                logger_instance.debug(f"  Raw eval_results_dict for '{model_alias}' before preprocessing: {str(eval_results_dict)[:1000]}...") # Log snippet
                try:
                    # Ensure parent directory for JSON exists
                    current_model_output_json_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    eval_results_dict_processed = preprocess_for_json(eval_results_dict.copy())
                    if logger_instance.verbose_flag: # Only log potentially large dict if verbose
                        logger_instance.debug(f"  eval_results_dict for '{model_alias}' AFTER preprocessing: {str(eval_results_dict_processed)[:1000]}...")

                    with open(current_model_output_json_path, 'w', encoding='utf-8') as f_json:
                        json_serializer.dump(eval_results_dict_processed, f_json, indent=2, ensure_ascii=False)
                    logger_instance.info(f"  Detailed results saved to: {current_model_output_json_path}")
                    eval_status = "Success" # Set status to success ONLY if JSON saving is successful
                except Exception as json_e:
                    eval_status = "Script Error (JSON Save)"; current_run_error_message = f"JSON Save Error: {json_e}"
                    logger_instance.error(f"  Error during JSON serialization or saving for '{model_alias}': {json_e}")
                    logger_instance.error(f"  Traceback for JSON Save Error:\n{traceback.format_exc()}")
            else: # eval_results_dict is None
                eval_status = "Script Error (No Results)"; current_run_error_message = "simple_evaluate returned None"
                logger_instance.warn(f"  simple_evaluate returned None for model '{model_alias}'. No detailed results to save.")
            # --- End of MODIFIED JSON Saving Block ---
            
            # This was originally before the JSON save success, moved it up.
            # But eval_status should reflect JSON save success too.
            # logger_instance.success(f"  Evaluation for '{model_alias}' completed.") # Original placement

            if eval_status == "Success": # Only populate summary rows if everything including JSON save was successful
                logger_instance.success(f"  Evaluation and result saving for '{model_alias}' completed successfully.")
                limit_str_for_csv = str(limit_samples or "None")
                # Guard against results not being a dict or missing 'results' key
                results_data = eval_results_dict.get("results", {}) if isinstance(eval_results_dict, dict) else {}
                for task_name_res, metrics_dict in results_data.items():
                    version = eval_results_dict.get("versions", {}).get(task_name_res, "N/A") if isinstance(eval_results_dict, dict) else "N/A"
                    if isinstance(metrics_dict, dict):
                        for metric_name, metric_value in metrics_dict.items():
                            if "_stderr" in metric_name.lower() or "_samples" in metric_name.lower() or metric_name == "alias": continue
                            metric_value_str = f"{metric_value:.4f}" if isinstance(metric_value, float) else str(metric_value)
                            all_summary_rows.append([ model_alias, task_name_res, metric_name, metric_value_str, str(version), str(num_fewshot or "Task Def"), limit_str_for_csv, timestamp_str, "" ])
                    else:
                        logger_instance.warn(f"  Metrics for task '{task_name_res}' in model '{model_alias}' is not a dictionary: {metrics_dict}")

        except ServerOperationError as e:
            eval_status = f"Failed (Server: {e})"; current_run_error_message = str(e)
            if is_humaneval_like_task_run: _ensure_os_unlink_restored(f"After EXCEPTION (ServerOp) for '{model_alias}'", logger_instance)
        except Exception as e_eval:
            eval_status = "Script Error (Eval)"; current_run_error_message = f"Eval Error: {str(e_eval)[:200]}"
            if logger_instance.verbose_flag:
                logger_instance.error(f"Full traceback for Eval Error in model '{model_alias}':\n{traceback.format_exc()}")
            if is_humaneval_like_task_run: _ensure_os_unlink_restored(f"After EXCEPTION (in simple_evaluate) for '{model_alias}'", logger_instance)
            if logger_instance.verbose_flag and eval_results_dict: # Log raw dict if eval failed but we got something
                try: import pprint; logger_instance.error(f"  eval_results_dict content on Eval Error for '{model_alias}': {pprint.pformat(eval_results_dict)}")
                except: pass
        finally:
            if server_process: stop_llama_server(server_process, model_alias, logger_instance)
            if is_humaneval_like_task_run:
                 _ensure_os_unlink_restored(f"In finally block after server stop for '{model_alias}'", logger_instance)
        
        logger_instance.info(f"  Result for '{model_alias}': {color_status(eval_status)}")
        if current_run_error_message and not eval_status.startswith("Success"): # Add to summary if error occurred
             all_summary_rows.append([ model_alias, tasks_list[0] if tasks_list else "-", "-", "-", "-", str(num_fewshot or "Task Def"), str(limit_samples or "None"), timestamp_str, current_run_error_message ])
        logger_instance.notice("-" * 30)
    
    # Write summary CSV
    logger_instance.header("Writing Evaluation Summary to CSV")
    if not all_summary_rows and models_to_iterate: # If no rows were added but we tried to eval models
        logger_instance.warn(f"No summary data was generated for {len(models_to_iterate)} model(s). CSV might be empty or only headers.")
    
    try:
        # Ensure the directory for summary_csv_path exists (it's run_output_dir)
        summary_csv_path.parent.mkdir(parents=True, exist_ok=True) # Should be same as run_output_dir
        with open(summary_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile); writer.writerow(csv_header); writer.writerows(all_summary_rows)
        logger_instance.success(f"All evaluation summary results saved to: {summary_csv_path.resolve()}")
    except IOError as e: 
        logger_instance.error(f"Failed to write summary CSV to {summary_csv_path}: {e}")
        logger_instance.error(f"Traceback for CSV Write Error:\n{traceback.format_exc()}")


# --- Main Execution ---
def signal_cleanup_handler_eval(signum, frame):
    global logger, TEMP_DIR_MANAGER_PATH_EVAL
    log_func = print if logger is None else logger.warn
    log_func(f"\nSignal {signal.Signals(signum).name} received. Cleaning up...")
    if TEMP_DIR_MANAGER_PATH_EVAL: log_func(f"  Main temp dir {TEMP_DIR_MANAGER_PATH_EVAL} cleanup via atexit.")
    log_func(f"Killing processes on port {STATIC_EVAL_PORT} due to signal.")
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline_list = proc.info.get('cmdline')
            if cmdline_list and f"--port {STATIC_EVAL_PORT}" in " ".join(cmdline_list):
                 log_func(f"  Killing PID={proc.pid} ({proc.info.get('name', '')})")
                 psutil.Process(proc.pid).kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess): continue
        except Exception as e: log_func(f"  Error during signal process kill: {e}")
    log_func("Evaluation script terminated by signal.")
    _ensure_os_unlink_restored("During signal cleanup", logger)
    sys.exit(1)

def main():
    global logger, TEMP_DIR_MANAGER_PATH_EVAL, project_root_dir

    try:
        current_method = multiprocessing.get_start_method(allow_none=True)
        target_method = 'spawn'
        can_set_method = target_method in multiprocessing.get_all_start_methods()
        if current_method != target_method and can_set_method:
            multiprocessing.set_start_method(target_method, force=True)
    except RuntimeError: pass
    except Exception: pass

    parser = argparse.ArgumentParser(
        description="LLM Evaluation Script using lm-evaluation-harness.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    default_base_config_path = (project_root_dir / "config.base.yaml").resolve()
    parser.add_argument("-c", "--config", type=Path, default=default_base_config_path, help="Path to base YAML config.")
    hostname = platform.node().split('.')[0].lower()
    default_override_path = (project_root_dir / "overrides" / f"{hostname}.yaml").resolve()
    calculated_override = default_override_path if default_override_path.exists() else None
    parser.add_argument("--override", type=Path, default=calculated_override, help="Path to override YAML.")
    parser.add_argument("--tasks", type=str, default=DEFAULT_TASKS, help="Comma-separated lm-eval tasks.")
    parser.add_argument("--limit", type=float, default=DEFAULT_LIMIT_SAMPLES, help="Samples per task (None for all).")
    parser.add_argument("--num-fewshot", type=int, default=DEFAULT_NUM_FEWSHOT, help="Num few-shot examples (task default if None).")
    parser.add_argument("--batch-size", type=str, default=str(DEFAULT_BATCH_SIZE), help="Batch size for lm-eval (e.g., 1, 'auto').")
    
    default_base_output_dir = project_root_dir / "eval" / DEFAULT_BASE_OUTPUT_DIR_NAME
    parser.add_argument("-o", "--output-dir", type=Path, default=default_base_output_dir,
                        help="Base directory for output. A run-specific timestamped subdirectory will be created here.")
    
    parser.add_argument("-m", "--model", type=str, help="Evaluate only this specific model alias.")
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_HEALTH_POLL_INTERVAL_S, help="Health check poll interval (s).")
    parser.add_argument("--health-timeout", type=int, default=DEFAULT_HEALTH_TIMEOUT_S, help="Server health check timeout (s).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose/debug output.")
    parser.add_argument(
        "--qwen3-no-prefix-tasks", type=str, default=DEFAULT_QWEN3_NO_PREFIX_TASKS_STR_CLI,
        help="Semicolon-separated list of task names for which Qwen3 models should NOT have '/no_think' prepended (e.g., 'humaneval;mbpp')."
    )
    args = parser.parse_args()
    
    logger = Logger(verbose=args.verbose)
    
    # --- Logger Setup ---
    logger.header("LLM EVALUATOR INITIALIZATION (eval/evaluate-models.py)")
    # Configure the root logger for lm_eval namespace
    # This will affect _qwen_custom_api_logger, _preprocess_logger, and lm_eval's internal logs
    lm_eval_root_logger = std_logging.getLogger("lm_eval")
    lm_eval_root_logger.setLevel(std_logging.DEBUG if args.verbose else std_logging.INFO)
    if not lm_eval_root_logger.hasHandlers():
        stream_handler_eval = std_logging.StreamHandler(sys.stderr)
        # You can customize formatter for all lm_eval logs here if desired
        # formatter_eval = std_logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # stream_handler_eval.setFormatter(formatter_eval)
        lm_eval_root_logger.addHandler(stream_handler_eval)
        lm_eval_root_logger.propagate = False 
    logger.info(f"Base 'lm_eval' logger level set to: {std_logging.getLevelName(lm_eval_root_logger.level)}")
    if args.verbose:
        logger.info(f"  (This means lm_eval.QwenLocalCompletionsAPI and lm_eval.preprocess_for_json will also log at DEBUG level)")


    # --- Signal Handling and Config Loading ---
    signal.signal(signal.SIGINT, signal_cleanup_handler_eval); signal.signal(signal.SIGTERM, signal_cleanup_handler_eval)
    logger.info("Signal handlers registered.")
    if not args.config.is_file(): logger.error(f"Base config not found: {args.config}"); sys.exit(1)
    logger.info(f"Using base configuration: {args.config}")
    if args.override:
        if args.override.is_file(): logger.info(f"Using override configuration: {args.override}")
        else: logger.warn(f"Override config not found: {args.override}. Proceeding without."); args.override = None
    else: logger.info("No override configuration file specified or found by default logic.")
    
    # --- Initial Server Cleanup ---
    logger.step(f"Initial cleanup of lingering llama-server on port {STATIC_EVAL_PORT}...")
    # ... (cleanup code as before) ...
    killed_procs = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline_list = proc.info.get('cmdline')
            if cmdline_list and f"--port {STATIC_EVAL_PORT}" in " ".join(cmdline_list):
                 logger.warn(f"  Terminating lingering PID={proc.pid}, Name='{proc.info.get('name', '')}'")
                 p_obj = psutil.Process(proc.pid); p_obj.terminate()
                 try: p_obj.wait(timeout=2)
                 except psutil.TimeoutExpired: p_obj.kill(); p_obj.wait(timeout=1)
                 killed_procs +=1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess): continue
        except Exception as e_psutil: logger.warn(f"  Error during psutil check: {e_psutil}")
    if killed_procs > 0: logger.success(f"Initial cleanup terminated {killed_procs} process(es).")
    else: logger.info(f"No lingering eval server processes found on port {STATIC_EVAL_PORT}.")

    logger.step("Loading configurations...")
    try:
        effective_conf_dict = generate_processed_config( base_config_path_arg=args.config, override_config_path_arg=args.override, script_dir_for_overrides=project_root_dir, verbose_logging=args.verbose )
        logger.success("Configurations processed.")
    except Exception as e: logger.error(f"Failed to load/process configurations: {e}"); sys.exit(1)

    # --- Create Run-Specific Output Directory ---
    run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir_name = f"RUN_{run_timestamp}"
    run_specific_output_dir = args.output_dir.resolve() / run_dir_name
    try:
        run_specific_output_dir.mkdir(parents=True, exist_ok=True) # This is crucial
        logger.info(f"Evaluation outputs for this run will be saved to: {run_specific_output_dir}")
    except OSError as e:
        logger.error(f"Could not create run-specific output directory: {run_specific_output_dir} - {e}")
        sys.exit(1) # Exit if this fails, as nothing can be saved.

    script_start_time = time.monotonic()
    try:
        with _original_tempfile_TemporaryDirectory(prefix="llm_eval_main_temp_") as temp_dir_name:
            TEMP_DIR_MANAGER_PATH_EVAL = Path(temp_dir_name)
            logger.info(f"Using main temp dir for script (e.g. server logs): {TEMP_DIR_MANAGER_PATH_EVAL}")
            logger.header("STARTING EVALUATION RUN")
            processed_models_data = effective_conf_dict.get("models", {})
            if not isinstance(processed_models_data, dict) or not processed_models_data:
                logger.error("'models' section not found or empty in config."); sys.exit(1)

            logger.debug("Registering lm-eval tasks...")
            lm_eval.tasks.get_task_dict(args.tasks.split(','))
            _ensure_os_unlink_restored("After lm_eval.tasks.get_task_dict() in main", logger)
            logger.debug("lm-eval tasks registered.")

            run_evaluation(
                processed_models_config=processed_models_data,
                run_output_dir=run_specific_output_dir, # This dir must exist
                tasks_str=args.tasks,
                limit_samples=args.limit,
                num_fewshot=args.num_fewshot,
                batch_size_str=args.batch_size,
                health_timeout_s=args.health_timeout,
                health_poll_s=args.poll_interval,
                model_to_eval_alias=args.model,
                temp_dir_path=TEMP_DIR_MANAGER_PATH_EVAL,
                logger_instance=logger,
                qwen3_no_prefix_tasks_override=args.qwen3_no_prefix_tasks
            )
    finally:
        _ensure_os_unlink_restored("At the very end of main execution", logger)
        script_duration_s = time.monotonic() - script_start_time
        logger.header("EVALUATION SCRIPT COMPLETE")
        logger.success(f"Total script execution time: {script_duration_s:.2f} seconds.")

if __name__ == "__main__":
    current_method = multiprocessing.get_start_method(allow_none=True)
    target_method = 'spawn'
    can_set_method = target_method in multiprocessing.get_all_start_methods()

    if current_method != target_method and can_set_method:
        try:
            multiprocessing.set_start_method(target_method, force=True)
            print(f"INFO: Multiprocessing start method set to '{target_method}'.", flush=True)
        except RuntimeError as e:
            print(f"WARNING: Could not set multiprocessing start method to '{target_method}': {e}. Using '{current_method}'.", flush=True)
        except Exception as e:
            print(f"WARNING: Unknown error setting multiprocessing start method to '{target_method}': {e}. Using '{current_method}'.", flush=True)
    else:
        if not can_set_method and current_method != target_method:
            print(f"INFO: Multiprocessing start method is '{current_method}'. Target '{target_method}' is not available.", flush=True)
        else:
             print(f"INFO: Multiprocessing start method is '{current_method}'.", flush=True)
    main()