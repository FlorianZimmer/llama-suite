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
# import shutil # Not directly used now in shared funcs, but keep if script uses it elsewhere
import tempfile
_original_tempfile_TemporaryDirectory = tempfile.TemporaryDirectory # Keep for script's own temp dirs

# --- Standard Library Imports ---
import argparse
import atexit
import csv
import datetime
import platform
# import re # Not directly used at top level, but used by imported functions
# import shlex # Not directly used for splitting command here, but Popen context
import signal
import subprocess # Still needed for Popen type hint if not fully abstracted
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TextIO # List, Tuple already here
import multiprocessing
import logging as std_logging # For lm-eval's logger config
import json as json_serializer
import dataclasses
import traceback

# --- Third-Party Library Imports ---
import psutil
# import requests # Used by wait_for_server_health in config_utils
# from colorama import Fore, Style, init as colorama_init_eval # Handled by Logger in config_utils

from lm_eval import simple_evaluate
import lm_eval.tasks
from lm_eval.utils import make_table
from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import LocalCompletionsAPI
from lm_eval.api.instance import Instance

# --- Setup a logger for preprocess_for_json ---
_preprocess_logger = std_logging.getLogger("lm_eval.preprocess_for_json")
_qwen_custom_api_logger = std_logging.getLogger("lm_eval.QwenLocalCompletionsAPI")

# --- Custom lm-eval Model for Qwen3 (remains the same) ---
@register_model("custom-local-completions-qwen")
class QwenLocalCompletionsAPIWithNoThink(LocalCompletionsAPI):
    # ... (Your QwenLocalCompletionsAPIWithNoThink class definition - unchanged) ...
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
            current_context = instance.arguments[0]
            
            new_context = current_context
            should_prefix = self._should_prefix_prompt_for_task(task_name)

            if should_prefix:
                if isinstance(current_context, str):
                    if not current_context.startswith("/no_think"):
                        new_context = "/no_think " + current_context
                        prefix_applied_count += 1
                    else: prefix_skipped_count +=1 
                elif isinstance(current_context, list) and getattr(self, 'tokenized_requests', False):
                     _qwen_custom_api_logger.warning(
                        f"({self.effective_model_name_for_log}): Task '{task_name}' (Request {i+1}) is Qwen3 and requires prefix, "
                        "but is using 'tokenized_requests=True'. Conditional prepending of '/no_think' to tokenized "
                        "prompts is not automatically handled."
                    )
                     prefix_skipped_count +=1
                else: prefix_skipped_count +=1
            else: prefix_skipped_count +=1
            
            if new_context is not current_context:
                new_arguments_tuple = (new_context, instance.arguments[1]) 
                processed_requests.append(dataclasses.replace(instance, arguments=new_arguments_tuple))
            else:
                processed_requests.append(instance)
        
        if self.is_qwen3_model_for_no_think and (prefix_applied_count > 0 or prefix_skipped_count > 0) :
            _qwen_custom_api_logger.info(
                f"({self.effective_model_name_for_log}): Processed {len(requests)} requests for {'generate_until' if is_generate_until else 'loglikelihood'}. "
                f"'/no_think' prefix applied to {prefix_applied_count} requests, skipped for {prefix_skipped_count} requests."
            )
        return processed_requests
    
    # Re-add loglikelihood and generate_until from LocalCompletionsAPI and call _conditionally_prefix_requests
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        prefixed_requests = self._conditionally_prefix_requests(requests, is_generate_until=False)
        return super().loglikelihood(prefixed_requests)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        prefixed_requests = self._conditionally_prefix_requests(requests, is_generate_until=True)
        return super().generate_until(prefixed_requests)


# --- Project-Specific Imports (utils.config_utils) ---
_current_script_file_path_for_root_eval = Path(__file__).resolve()
project_root_dir = _current_script_file_path_for_root_eval.parent.parent # Assuming script is in eval/
if str(project_root_dir) not in sys.path:
    sys.path.insert(0, str(project_root_dir))

# Import shared functions from config_utils
try:
    from utils.config_utils import (
        Logger, generate_processed_config, build_llama_server_command_util,
        start_llama_server, stop_llama_server, wait_for_server_health,
        _dump_stderr_on_failure, color_status, # _resolve_executable_path_robustly is used by start_server
        PROCESS_TERMINATE_TIMEOUT_S, DEFAULT_HEALTH_POLL_INTERVAL_S
    )
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import from config_utils.py. Ensure it's in project_root/utils.")
    print(f"  Details: {e}")
    sys.exit(1)

try:
    # Import the specific classes you will use by their registered names
    from utils.custom_lm_eval_models import LlamaCppCompatibleCompletionsAPI, Qwen3LocalCompletionsAPI
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import custom API models from utils.custom_lm_eval_models.py."
          f" Ensure the file exists and is correct: {e}", file=sys.stderr)
    sys.exit(1)

# --- Environment Variable Configuration ---
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Constants specific to this script ---
DEFAULT_BASE_OUTPUT_DIR_NAME = "results"
DEFAULT_TASKS = "humaneval" # Example, can be changed
DEFAULT_LIMIT_SAMPLES: Optional[float] = None # Use float for lm-eval limit
DEFAULT_BATCH_SIZE_EVAL = "1" # lm-eval can take "auto" or int
DEFAULT_NUM_FEWSHOT: Optional[int] = None # lm-eval task default if None
STATIC_EVAL_PORT = "9997" # Different from scan_model_memory port
DEFAULT_HEALTH_TIMEOUT_S_EVAL = 3600 # Longer for potentially slow model loads
DEFAULT_QWEN3_NO_PREFIX_TASKS_STR_CLI = "humaneval;mbpp;gsm8k;gpqa" # Default for CLI arg

# Keys in the model's root config (not in 'cmd' or 'sampling') that are meta-data
# and should NOT be passed as CLI arguments to llama-server by this script's logic.
MODEL_CONFIG_META_KEYS_EVAL = {
    "aliases", "sampling", "_name_for_log", "generated_cmd_str",
    "hf_tokenizer_for_model", "supports_no_think_toggle",
    "num_concurrent_eval" # Example if this was a meta key for eval script
}


# --- Global Variables ---
logger: Optional['Logger'] = None # Will use Logger from config_utils
TEMP_DIR_MANAGER_PATH_EVAL: Optional[Path] = None # Specific to this script

# --- ATEIXT Restorer for os.unlink (remains the same) ---
def final_os_unlink_restorer_atexit():
    # ... (same as before) ...
    global _original_os_unlink
    if _original_os_unlink is None:
        print("ATEIXT_CRITICAL: _original_os_unlink is None.", file=sys.stderr); return
    current_unlink = getattr(os, 'unlink', None)
    if current_unlink is None or current_unlink != _original_os_unlink:
        print("ATEIXT_INFO: os.unlink was incorrect at final Python exit. Restoring globally.", file=sys.stderr)
        os.unlink = _original_os_unlink
atexit.register(final_os_unlink_restorer_atexit)


# --- ServerOperationError Class (remains the same) ---
class ServerOperationError(Exception): pass

# --- Utility Functions specific to eval or simple enough to keep local ---
def _ensure_os_unlink_restored(context_message: str, logger_instance: Optional[Logger]):
    # ... (same as before, but ensure Logger type hint matches the one from config_utils) ...
    log_func = print if logger_instance is None else logger_instance.debug
    current_unlink = getattr(os, 'unlink', None)
    if current_unlink is None or current_unlink != _original_os_unlink:
        log_func(f"DEBUG_OS_UNLINK_RESTORE: {context_message} - os.unlink was '{str(current_unlink)}'. Attempting restore.")
        if _original_os_unlink is not None:
            os.unlink = _original_os_unlink
        elif logger_instance: logger_instance.error("CRITICAL: Cannot restore os.unlink: _original_os_unlink is None.")
        else: print("CRITICAL: Cannot restore os.unlink: _original_os_unlink is None.", file=sys.stderr)


def preprocess_for_json(data: Any) -> Any: # MODIFIED to use _preprocess_logger
    # ... (same as before, ensures _preprocess_logger is used) ...
    if isinstance(data, (str, int, float, bool, type(None))):
        return data
    if isinstance(data, dict):
        return {preprocess_for_json(k): preprocess_for_json(v) for k, v in data.items()}
    if isinstance(data, list) or isinstance(data, tuple):
        return [preprocess_for_json(elem) for elem in data]
    if callable(data):
        return f"<function {getattr(data, '__name__', 'unknown_callable')}>"
    
    if hasattr(data, 'item') and callable(data.item): 
        try:
            item_val = data.item()
            if isinstance(item_val, (str, int, float, bool, type(None))): return item_val
            return str(item_val) 
        except Exception: return str(data) 
    
    if hasattr(data, 'tolist') and callable(data.tolist): 
        try: return preprocess_for_json(data.tolist())
        except Exception: return str(data)
    
    try:
        json_serializer.dumps(data) 
        return data 
    except TypeError:
        _preprocess_logger.debug(f"Defaulting to str() for type {type(data)}, value (first 100 chars): {str(data)[:100]}...")
        return str(data)
    except Exception as e_json_test:
        _preprocess_logger.warning(f"Error during serialization test for type {type(data)}: {e_json_test}. Defaulting to str(). Value (first 100 chars): {str(data)[:100]}")
        return str(data)

def signal_cleanup_handler_eval(signum, frame):
    global logger, TEMP_DIR_MANAGER_PATH_EVAL # Ensure it uses the eval script's globals
    
    # Use a more direct way to get the logger or print
    if logger:
        log_func = logger.warn 
    else:
        # If logger is None at this point, something is very wrong with initialization order,
        # but fall back to print for critical messages.
        def print_to_stderr(msg): print(msg, file=sys.stderr)
        log_func = print_to_stderr

    log_func(f"\nSignal {signal.Signals(signum).name} received. Cleaning up for evaluate-models...")
    
    if TEMP_DIR_MANAGER_PATH_EVAL: # Check if this global is set and used
        log_func(f"  Note: Main temp dir {TEMP_DIR_MANAGER_PATH_EVAL} cleanup is typically handled by 'with TemporaryDirectory()'.")
    
    log_func(f"Attempting to kill any lingering server processes on port {STATIC_EVAL_PORT} due to signal.")
    killed_during_signal = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']): # Ensure psutil is imported
        try:
            cmdline_list = proc.info.get('cmdline')
            if cmdline_list and f"--port {STATIC_EVAL_PORT}" in " ".join(cmdline_list):
                 log_func(f"  Signal Cleanup: Killing PID={proc.pid} (Name='{proc.info.get('name', '')}') on port {STATIC_EVAL_PORT}")
                 p_obj = psutil.Process(proc.pid)
                 p_obj.terminate() # Try to terminate first
                 try:
                     p_obj.wait(timeout=1) # Short wait
                 except psutil.TimeoutExpired:
                     log_func(f"    PID={proc.pid} did not terminate gracefully, forcing kill.")
                     p_obj.kill()
                     p_obj.wait(timeout=1) # Wait for kill
                 killed_during_signal +=1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue # Process might have already exited
        except Exception as e_ps_signal: # Catch any other error during this sensitive cleanup
            log_func(f"  Error during signal cleanup process kill (PID {proc.pid if 'proc' in locals() and proc else 'unknown'}): {e_ps_signal}")

    if killed_during_signal > 0:
        log_func(f"Signal cleanup terminated {killed_during_signal} server process(es).")
    else:
        log_func(f"No server processes found to kill on port {STATIC_EVAL_PORT} during signal cleanup.")

    if logger:
        logger.info("Evaluation script terminated by signal.")
    else:
        print("Evaluation script terminated by signal.", file=sys.stderr)
    
    _ensure_os_unlink_restored("During signal cleanup (eval)", logger) # Ensure this is defined and accessible
    sys.exit(1) # Exit with an error code to indicate abnormal termination

# --- Core Evaluation Logic ---
def run_evaluation(
    processed_models_config: Dict[str, Any],
    run_output_dir: Path,
    tasks_str: str,
    limit_samples: Optional[float], num_fewshot: Optional[int], batch_size_str: str,
    health_timeout_s: int, health_poll_s: float, model_to_eval_alias: Optional[str],
    temp_dir_path: Path, logger_instance: Logger, # Expecting Logger from config_utils
    qwen3_no_prefix_tasks_override: str
):
    is_verbose = logger_instance.verbose_flag # For local conditional logging

    if not processed_models_config:
        logger_instance.error("No processed model configurations for evaluation."); return

    try: run_output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e: logger_instance.error(f"CRITICAL: Cannot create run output dir: {run_output_dir} - {e}"); return

    models_to_iterate: Dict[str, Any] = {}
    if model_to_eval_alias:
        # ... (model selection logic - same as before) ...
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
        current_model_output_json_path = run_output_dir / f"{model_alias}_results.json"
        timestamp_str = logger_instance._get_timestamp()
        eval_status = "Not Run"; current_run_error_message = ""
        tasks_list = [t.strip() for t in tasks_str.split(',')]
        is_humaneval_like_task_run = any(task in ["humaneval", "mbpp", "multipleefix", "multipl<x_bin_482>"] for task in tasks_list) # Added humaneval variants

        # --- Command Preparation using new robust method ---
        original_cmd_dict_from_model_data = model_data.get("cmd")
        if not isinstance(original_cmd_dict_from_model_data, dict):
            current_run_error_message = "Config Error: Missing 'cmd' dict"
            all_summary_rows.append([model_alias, "-", "-", "-", "-", str(num_fewshot or "Def"), str(limit_samples or "None"), timestamp_str, current_run_error_message])
            continue
        
        cmd_options_for_eval = original_cmd_dict_from_model_data.copy()
        cmd_options_for_eval["port"] = STATIC_EVAL_PORT
        
        config_for_build_util_log = { # For logging full command string
            k: v for k, v in model_data.items() if k not in ["cmd", "generated_cmd_str"]
        }
        config_for_build_util_log["cmd"] = cmd_options_for_eval
        if "sampling" in model_data: config_for_build_util_log["sampling"] = model_data["sampling"]
        config_for_build_util_log["_name_for_log"] = model_alias

        server_executable = ""
        server_args_list: List[str] = []

        try:
            resolved_bin_path_from_config = cmd_options_for_eval.get("bin")
            if not resolved_bin_path_from_config or not isinstance(resolved_bin_path_from_config, str):
                # ... (handle error and continue)
                all_summary_rows.append([model_alias, "-", "-", "-", "-", str(num_fewshot or "Def"), str(limit_samples or "None"), timestamp_str, "Config Error: Missing 'bin' in cmd dict"])
                continue
            
            server_executable = resolved_bin_path_from_config
            logger_instance.debug(f"  Using server_executable (from resolved config): '{server_executable}'")

            if is_verbose:
                full_cmd_str_for_log = build_llama_server_command_util(config_for_build_util_log)
                logger_instance.debug(f"    Full command string (for reference) from build_util: {full_cmd_str_for_log}")
            
            # Reconstruct server_args_list from cmd_options_for_eval and other model_data parts
            server_args_list.extend(["--port", str(cmd_options_for_eval.get("port"))])
            server_args_list.extend(["--model", str(cmd_options_for_eval.get("model"))])
            server_args_list.extend(["--ctx-size", str(cmd_options_for_eval.get("ctx-size"))])
            if str(cmd_options_for_eval.get("gpu-layers", "auto")).lower() != "auto":
                server_args_list.extend(["--n-gpu-layers", str(cmd_options_for_eval.get("gpu-layers"))])
            if str(cmd_options_for_eval.get("threads", "auto")).lower() != "auto":
                server_args_list.extend(["--threads", str(cmd_options_for_eval.get("threads"))])
            
            handled_cmd_keys = {"bin", "port", "model", "ctx-size", "gpu-layers", "threads"}
            for key, value in cmd_options_for_eval.items():
                if key in handled_cmd_keys: continue
                cli_flag = f"--{key.replace('_', '-')}"
                if isinstance(value, bool) and value: server_args_list.append(cli_flag)
                elif value not in (None, False, "auto", "Auto", ""): server_args_list.extend([cli_flag, str(value)])
            
            for key, value in model_data.items():
                if key == "cmd" or key in MODEL_CONFIG_META_KEYS_EVAL: continue # Use EVAL specific meta keys
                cli_flag = f"--{key.replace('_', '-')}"
                if isinstance(value, bool) and value: server_args_list.append(cli_flag)
                elif value not in (None, False, "auto", "Auto", ""): server_args_list.extend([cli_flag, str(value)])

            sampling_conf = model_data.get("sampling", {})
            if isinstance(sampling_conf, dict):
                for key, s_value in sampling_conf.items():
                    cli_flag = f"--{key.replace('_', '-')}"
                    server_args_list.extend([cli_flag, str(s_value)])
            
            logger_instance.info(f"  Server command for evaluation prepared.")
            logger_instance.debug(f"    Final server_executable for Popen: '{server_executable}'")
            logger_instance.debug(f"    Final server_args_list for Popen: {server_args_list}")

        except Exception as e_cmd_prep:
            current_run_error_message = f"Cmd Prep Error: {e_cmd_prep}"
            logger_instance.error(f"  Error preparing command for '{model_alias}': {e_cmd_prep}")
            if is_verbose: logger_instance.error(f"Traceback:\n{traceback.format_exc()}")
            all_summary_rows.append([model_alias, "-", "-", "-", "-", str(num_fewshot or "Def"), str(limit_samples or "None"), timestamp_str, current_run_error_message])
            continue
        # --- End Command Preparation ---
        
        server_process: Optional[subprocess.Popen] = None
        eval_results_dict: Optional[Dict[str, Any]] = None
        stderr_log_path_for_model: Optional[Path] = None # For _dump_stderr_on_failure
        
        try:
            if is_humaneval_like_task_run:
                 _ensure_os_unlink_restored(f"Before server start for '{model_alias}'", logger_instance)

            # project_root_dir is global
            server_process_info = start_llama_server(
                server_executable, server_args_list, model_alias, 
                temp_dir_path, logger_instance, project_root_dir
            )
            if not server_process_info: raise ServerOperationError("Server Start Failed")
            server_process, _, stderr_log_path_for_model = server_process_info

            healthy = wait_for_server_health(
                server_process, health_url, health_timeout_s, health_poll_s, 
                model_alias, logger_instance
            )
            if not healthy:
                exit_code_msg = f"(Code: {server_process.poll()})" if server_process.poll() is not None else ""
                _dump_stderr_on_failure(stderr_log_path_for_model, model_alias, logger_instance)
                raise ServerOperationError(f"Server Not Healthy {exit_code_msg}")

            logger_instance.info(f"  Server for '{model_alias}' healthy. Starting lm-eval...")
            # ... (lm-eval args setup - same as before) ...
            hf_tokenizer_for_model = model_data.get("hf_tokenizer_for_model", model_alias)
            num_concurrent_requests = model_data.get("num_concurrent_eval", 1) # Use a specific key if needed
            DEFAULT_LM_EVAL_MAX_LENGTH = 2048
            configured_ctx_size_str = original_cmd_dict_from_model_data.get("ctx-size") # Use original cmd dict
            try: lm_eval_max_length = int(configured_ctx_size_str) if configured_ctx_size_str else DEFAULT_LM_EVAL_MAX_LENGTH
            except ValueError: lm_eval_max_length = DEFAULT_LM_EVAL_MAX_LENGTH
            
            lm_eval_model_args_list = [
                f"base_url={eval_api_endpoint}", f"engine={model_alias}", f"model={model_alias}", "api_key=EMPTY",
                f"timeout={health_timeout_s}", "tokenizer_backend=huggingface", f"tokenizer={hf_tokenizer_for_model}",
                "truncate=True", "prefix_completions=True", f"num_concurrent={num_concurrent_requests}",
                f"max_length={lm_eval_max_length}"
            ]
            
            model_type_for_lm_eval: str
            if "qwen3" in model_alias.lower():
                logger_instance.info(f"  Model '{model_alias}' identified as Qwen3. Using Qwen3LocalCompletionsAPI for custom prefixing and llama.cpp compatibility.")
                model_type_for_lm_eval = "custom_qwen3_local_api" # Use the new registered name for your Qwen3 class
                # These args are passed to the __init__ of your Qwen3LocalCompletionsAPI
                lm_eval_model_args_list.append(f"model_alias_for_qwen_check={model_alias}")
                lm_eval_model_args_list.append(f"no_prefix_tasks_str={qwen3_no_prefix_tasks_override}")
            else:
                logger_instance.info(f"  Model '{model_alias}' (not Qwen3). Using LlamaCppCompatibleCompletionsAPI for llama.cpp compatibility.")
                model_type_for_lm_eval = "llama_cpp_compatible_api" # Use the registered name for general llama.cpp

            lm_eval_model_args_str = ",".join(lm_eval_model_args_list) # Renamed for clarity
            logger_instance.debug(f"  lm-eval model_args: {lm_eval_model_args_str}")

            actual_batch_size_val: Union[int, str] = 1 # Default
            if batch_size_str.lower() == "auto": actual_batch_size_val = "auto"
            elif batch_size_str.isdigit(): actual_batch_size_val = int(batch_size_str)
            
            confirm_unsafe = True if is_humaneval_like_task_run else False
            if confirm_unsafe: logger_instance.warn(f"  Unsafe tasks detected. Setting confirm_run_unsafe_code=True for lm-eval.")

            if is_humaneval_like_task_run:
                 _ensure_os_unlink_restored(f"Before simple_evaluate for '{model_alias}'", logger_instance)
            
            eval_results_dict = simple_evaluate(
                model=model_type_for_lm_eval, model_args=lm_eval_model_args_str, tasks=tasks_list,
                num_fewshot=num_fewshot, limit=limit_samples, batch_size=actual_batch_size_val,
                confirm_run_unsafe_code=confirm_unsafe
            )
            if is_humaneval_like_task_run:
                _ensure_os_unlink_restored(f"IMMEDIATELY AFTER simple_evaluate for '{model_alias}'", logger_instance)

            if eval_results_dict is not None:
                # ... (JSON saving block - same as before) ...
                logger_instance.debug(f"  Raw eval_results_dict for '{model_alias}' before preprocessing: {str(eval_results_dict)[:1000]}...")
                try:
                    current_model_output_json_path.parent.mkdir(parents=True, exist_ok=True)
                    eval_results_dict_processed = preprocess_for_json(eval_results_dict.copy())
                    if logger_instance.verbose_flag:
                        logger_instance.debug(f"  eval_results_dict for '{model_alias}' AFTER preprocessing: {str(eval_results_dict_processed)[:1000]}...")
                    with open(current_model_output_json_path, 'w', encoding='utf-8') as f_json:
                        json_serializer.dump(eval_results_dict_processed, f_json, indent=2, ensure_ascii=False)
                    logger_instance.info(f"  Detailed results saved to: {current_model_output_json_path}")
                    eval_status = "Success"
                except Exception as json_e:
                    eval_status = "Script Error (JSON Save)"; current_run_error_message = f"JSON Save Error: {json_e}"
                    logger_instance.error(f"  Error during JSON serialization or saving for '{model_alias}': {json_e}")
                    logger_instance.error(f"  Traceback for JSON Save Error:\n{traceback.format_exc()}")
            else: 
                eval_status = "Script Error (No Results)"; current_run_error_message = "simple_evaluate returned None"
                logger_instance.warn(f"  simple_evaluate returned None for model '{model_alias}'. No detailed results to save.")

            if eval_status == "Success":
                logger_instance.success(f"  Evaluation and result saving for '{model_alias}' completed successfully.")
                # ... (Populate all_summary_rows - same as before) ...
                limit_str_for_csv = str(limit_samples if limit_samples is not None else "None") # Handle float limit
                results_data = eval_results_dict.get("results", {}) if isinstance(eval_results_dict, dict) else {}
                for task_name_res, metrics_dict in results_data.items():
                    version = eval_results_dict.get("versions", {}).get(task_name_res, "N/A") if isinstance(eval_results_dict, dict) else "N/A"
                    if isinstance(metrics_dict, dict):
                        for metric_name, metric_value in metrics_dict.items():
                            if "_stderr" in metric_name.lower() or "_samples" in metric_name.lower() or metric_name == "alias": continue
                            metric_value_str = f"{metric_value:.4f}" if isinstance(metric_value, float) else str(metric_value)
                            all_summary_rows.append([ model_alias, task_name_res, metric_name, metric_value_str, str(version), str(num_fewshot if num_fewshot is not None else "TaskDef"), limit_str_for_csv, timestamp_str, "" ])
                    else:
                        logger_instance.warn(f"  Metrics for task '{task_name_res}' in model '{model_alias}' is not a dictionary: {metrics_dict}")


        except ServerOperationError as e_serv: # Renamed to avoid conflict
            eval_status = f"Failed (Server: {e_serv})"; current_run_error_message = str(e_serv)
            if is_humaneval_like_task_run: _ensure_os_unlink_restored(f"After EXCEPTION (ServerOp) for '{model_alias}'", logger_instance)
        except Exception as e_eval_main: # Renamed to avoid conflict
            eval_status = "Script Error (Eval)"; current_run_error_message = f"Eval Error: {str(e_eval_main)[:200]}"
            if is_verbose: logger_instance.error(f"Full traceback for Eval Error in model '{model_alias}':\n{traceback.format_exc()}")
            if is_humaneval_like_task_run: _ensure_os_unlink_restored(f"After EXCEPTION (in simple_evaluate) for '{model_alias}'", logger_instance)
            # (log eval_results_dict on error - same as before)
            if is_verbose and eval_results_dict: 
                try: import pprint; logger_instance.error(f"  eval_results_dict content on Eval Error for '{model_alias}': {pprint.pformat(eval_results_dict)}")
                except: pass
        finally:
            if server_process: stop_llama_server(server_process, model_alias, logger_instance)
            if is_humaneval_like_task_run:
                 _ensure_os_unlink_restored(f"In finally block after server stop for '{model_alias}'", logger_instance)
        
        logger_instance.info(f"  Result for '{model_alias}': {color_status(eval_status)}")
        if current_run_error_message and not eval_status.startswith("Success"):
             all_summary_rows.append([ model_alias, tasks_list[0] if tasks_list else "-", "-", "-", "-", str(num_fewshot if num_fewshot is not None else "TaskDef"), str(limit_samples if limit_samples is not None else "None"), timestamp_str, current_run_error_message ])
        logger_instance.notice("-" * 30)
    
    # ... (Write summary CSV - same as before, ensure logger_instance used for errors) ...
    logger_instance.header("Writing Evaluation Summary to CSV")
    if not all_summary_rows and models_to_iterate:
        logger_instance.warn(f"No summary data was generated for {len(models_to_iterate)} model(s). CSV might be empty or only headers.")
    try:
        summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile); writer.writerow(csv_header); writer.writerows(all_summary_rows)
        logger_instance.success(f"All evaluation summary results saved to: {summary_csv_path.resolve()}")
    except IOError as e_csv: 
        logger_instance.error(f"Failed to write summary CSV to {summary_csv_path}: {e_csv}")
        if is_verbose: logger_instance.error(f"Traceback for CSV Write Error:\n{traceback.format_exc()}")


# --- Main Execution Function (`main_eval`) ---
def main_eval(): # Renamed for clarity
    global logger, TEMP_DIR_MANAGER_PATH_EVAL, project_root_dir # project_root_dir defined by import block

    # --- Multiprocessing Start Method (same as before) ---
    # ... (multiprocessing setup - same as before) ...
    try:
        current_method = multiprocessing.get_start_method(allow_none=True)
        target_method = 'spawn' if platform.system() != "Darwin" else "fork" # fork can be problematic, but spawn is often default
        # On some systems, 'spawn' might be the only reliable one if 'fork' causes issues with libraries like tokenizers.
        # However, 'fork' is often faster if it works. This is a tricky area.
        # Let's stick to spawn unless it's macOS and fork is usually more common.
        # A more robust check would be: if 'spawn' in multiprocessing.get_all_start_methods(): target_method = 'spawn'
        
        # Forcing spawn generally, as it's safer with complex C extensions / global state
        target_method = 'spawn'
        can_set_method = target_method in multiprocessing.get_all_start_methods()

        if current_method != target_method and can_set_method:
            multiprocessing.set_start_method(target_method, force=True)
            # No print here, logger not yet initialized. Initial prints can go to stderr.
    except RuntimeError: pass # Might already be set, or not allowed to change
    except Exception: pass


    # --- Argument Parsing (same as before, ensure defaults are script-specific) ---
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
    parser.add_argument("--limit", type=float, default=DEFAULT_LIMIT_SAMPLES, help="Max samples per task (None for all). Use float for lm-eval.")
    parser.add_argument("--num-fewshot", type=int, default=DEFAULT_NUM_FEWSHOT, help="Num few-shot examples (task default if None).")
    parser.add_argument("--batch-size", type=str, default=DEFAULT_BATCH_SIZE_EVAL, help="Batch size for lm-eval (e.g., 1, 'auto').")
    
    default_base_output_dir = project_root_dir / "eval" / DEFAULT_BASE_OUTPUT_DIR_NAME
    parser.add_argument("-o", "--output-dir", type=Path, default=default_base_output_dir, help="Base output directory. A run-specific timestamped subdir will be created.")
    
    parser.add_argument("-m", "--model", type=str, help="Evaluate only this specific model alias.")
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_HEALTH_POLL_INTERVAL_S, help="Health check poll interval (s).") # Shared default
    parser.add_argument("--health-timeout", type=int, default=DEFAULT_HEALTH_TIMEOUT_S_EVAL, help="Server health check timeout (s).") # Eval specific default
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose/debug output.")
    parser.add_argument(
        "--qwen3-no-prefix-tasks", type=str, default=DEFAULT_QWEN3_NO_PREFIX_TASKS_STR_CLI,
        help="Semicolon-separated list of task names for Qwen3 models to EXCLUDE from '/no_think' prepending."
    )
    args = parser.parse_args()
    
    # Initialize Logger (from config_utils)
    logger = Logger(verbose=args.verbose) 
    
    # --- Logger Setup for lm-eval (same as before) ---
    logger.header("LLM EVALUATOR INITIALIZATION (eval/evaluate-models.py)")
    lm_eval_root_logger = std_logging.getLogger("lm_eval") # Configure the 'lm_eval' namespace root
    lm_eval_root_logger.setLevel(std_logging.DEBUG if args.verbose else std_logging.INFO)
    if not lm_eval_root_logger.hasHandlers(): # Add handler only if none exist, to avoid duplicates on re-runs in same session
        stream_handler_eval = std_logging.StreamHandler(sys.stderr) # lm-eval logs often go to stderr
        # Basic formatter, can be customized
        formatter_eval = std_logging.Formatter('%(name)s - %(levelname)s - %(message)s') 
        stream_handler_eval.setFormatter(formatter_eval)
        lm_eval_root_logger.addHandler(stream_handler_eval)
    lm_eval_root_logger.propagate = False # Prevent lm-eval logs from going to the root Python logger if it has handlers
    logger.info(f"Base 'lm_eval' logger level set to: {std_logging.getLevelName(lm_eval_root_logger.level)}")


    # --- Signal Handling and Config Loading (same as before) ---
    signal.signal(signal.SIGINT, signal_cleanup_handler_eval); signal.signal(signal.SIGTERM, signal_cleanup_handler_eval)
    logger.info("Signal handlers registered.")
    # ... (config path checks and loading - same as before, ensure logger used for messages) ...
    if not args.config.is_file(): logger.error(f"Base config not found: {args.config}"); sys.exit(1)
    logger.info(f"Using base configuration: {args.config}")
    if args.override:
        if args.override.is_file(): logger.info(f"Using override configuration: {args.override}")
        else: logger.warn(f"Override config not found: {args.override}. Proceeding without."); args.override = None
    else: logger.info("No override configuration file specified or found by default logic.")
    
    # --- Initial Server Cleanup (same as before) ---
    logger.step(f"Initial cleanup of lingering llama-server on port {STATIC_EVAL_PORT}...")
    # ... (psutil cleanup code - same as before, ensure logger used for messages) ...
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


    logger.step("Loading configurations via config_utils...")
    try:
        effective_conf_dict = generate_processed_config( 
            base_config_path_arg=args.config, 
            override_config_path_arg=args.override, 
            script_dir_for_overrides=project_root_dir, 
            verbose_logging=args.verbose 
        )
        logger.success("Configurations processed.")
    except Exception as e_conf: # Renamed to avoid conflict
        logger.error(f"Failed to load/process configurations: {e_conf}")
        if args.verbose: logger.error(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(1)

    # --- Create Run-Specific Output Directory (same as before) ---
    run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir_name = f"RUN_{args.tasks.replace(',', '_')}_{run_timestamp}" # More descriptive run dir
    run_specific_output_dir = args.output_dir.resolve() / run_dir_name
    try:
        run_specific_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Evaluation outputs for this run will be saved to: {run_specific_output_dir}")
    except OSError as e_dir: # Renamed
        logger.error(f"Could not create run-specific output dir: {run_specific_output_dir} - {e_dir}"); sys.exit(1)

    script_start_time = time.monotonic()
    try:
        # Use the original TemporaryDirectory for the script's main temp needs
        with _original_tempfile_TemporaryDirectory(prefix="llm_eval_main_temp_") as temp_dir_name:
            TEMP_DIR_MANAGER_PATH_EVAL = Path(temp_dir_name)
            logger.info(f"Using main temp dir for script (e.g. server logs): {TEMP_DIR_MANAGER_PATH_EVAL}")
            logger.header("STARTING EVALUATION RUN")
            
            processed_models_data = effective_conf_dict.get("models", {})
            if not isinstance(processed_models_data, dict) or not processed_models_data:
                logger.error("'models' section not found or empty in config."); sys.exit(1)

            logger.debug("Registering lm-eval tasks...")
            lm_eval.tasks.get_task_dict(args.tasks.split(',')) # Pre-register tasks
            _ensure_os_unlink_restored("After lm_eval.tasks.get_task_dict() in main", logger)
            logger.debug("lm-eval tasks registered.")

            run_evaluation(
                processed_models_config=processed_models_data,
                run_output_dir=run_specific_output_dir,
                tasks_str=args.tasks,
                limit_samples=args.limit,
                num_fewshot=args.num_fewshot,
                batch_size_str=args.batch_size,
                health_timeout_s=args.health_timeout, # Use specific eval timeout
                health_poll_s=args.poll_interval,     # Use shared default poll interval
                model_to_eval_alias=args.model,
                temp_dir_path=TEMP_DIR_MANAGER_PATH_EVAL,
                logger_instance=logger, # Pass the script's logger
                qwen3_no_prefix_tasks_override=args.qwen3_no_prefix_tasks
            )
    finally:
        _ensure_os_unlink_restored("At the very end of main execution (eval)", logger) # Contextual message
        script_duration_s = time.monotonic() - script_start_time
        logger.header("EVALUATION SCRIPT COMPLETE")
        logger.success(f"Total script execution time: {script_duration_s:.2f} seconds.")
        if logger: logger.close() # Close file log if open


if __name__ == "__main__":
    # Multiprocessing start method logic (same as before)
    # ...
    if platform.system() != "Darwin": # 'spawn' is generally safer cross-platform than 'fork'
        # Check if 'spawn' is available and not already the method
        if multiprocessing.get_start_method(allow_none=True) != 'spawn' and \
           'spawn' in multiprocessing.get_all_start_methods():
            try:
                multiprocessing.set_start_method('spawn', force=True)
                # Initial print before logger is set up
                print(f"INFO: Set multiprocessing start method to 'spawn'.", flush=True)
            except RuntimeError as e:
                print(f"WARNING: Could not set multiprocessing start method to 'spawn': {e}. Using default.", flush=True)
    else: # On macOS, 'fork' is often the default and can be faster if it works.
        print(f"INFO: Using default multiprocessing start method for macOS ('{multiprocessing.get_start_method(allow_none=True)}').", flush=True)

    main_eval()