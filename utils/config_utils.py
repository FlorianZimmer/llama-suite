# config_utils.py
import os
import platform
from pathlib import Path
import re # Added
import shlex # Added
import subprocess # Added
import sys # Added
import time # Added
import datetime # Added for Logger and unique suffixes
from typing import Any, Dict, Optional, Union, List, Tuple, TextIO # Added List, Tuple, TextIO

import yaml 
import requests # Added
import psutil # Added
from colorama import Fore, Style, init as colorama_init

# Initialize Colorama
try:
    colorama_init(autoreset=True)
except Exception:
    print("Warning: colorama init failed or not installed. Proceeding without colors.")
    class DummyStyle: RESET_ALL = "" # type: ignore
    class DummyFore: RED, YELLOW, GREEN, CYAN, BLUE, MAGENTA, WHITE = ("",)*7 # type: ignore # Added WHITE
    Style = DummyStyle() # type: ignore
    Fore = DummyFore() # type: ignore

# --- Shared Constants ---
PROCESS_TERMINATE_TIMEOUT_S = 10
DEFAULT_HEALTH_POLL_INTERVAL_S = 2.0
# DEFAULT_HEALTH_TIMEOUT_S can be defined here if it's truly shared, or each script has its own.
# For now, let each script manage its own default health timeout.

# --- Shared Logger Class ---
class Logger:
    def __init__(self, verbose: bool = False, log_file_path: Optional[Path] = None):
        self.verbose_flag = verbose
        self.log_file_path = log_file_path
        self.log_file: Optional[TextIO] = None
        if self.log_file_path:
            try:
                self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
                self.log_file = open(self.log_file_path, 'a', encoding='utf-8')
            except Exception as e:
                print(f"Error opening log file {self.log_file_path}: {e}", file=sys.stderr)
                self.log_file = None
        # colorama_init(autoreset=True) # Already called globally

    def _get_timestamp(self) -> str:
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def _log(self, level: str, message: str, color: str = "", bright: bool = False, console_file: TextIO = sys.stdout):
        style_prefix = Style.BRIGHT if bright else ""
        timestamp = self._get_timestamp()
        
        console_message = f"{timestamp} [{style_prefix}{color}{level}{Style.RESET_ALL}] {style_prefix}{color}{message}{Style.RESET_ALL}"
        print(console_message, file=console_file)
        
        if self.log_file:
            # Log plain message to file, without ANSI codes for now
            file_message = f"{timestamp} [{level}] {message}"
            try:
                self.log_file.write(file_message + "\n")
                self.log_file.flush() # Ensure it's written immediately
            except Exception as e:
                # Avoid crashing logger if file write fails
                print(f"Error writing to log file: {e}", file=sys.stderr)


    def info(self, message: str): self._log("INFO", message, Fore.BLUE)
    def warn(self, message: str): self._log("WARN", message, Fore.YELLOW, bright=True)
    def error(self, message: str): self._log("ERROR", message, Fore.RED, bright=True, console_file=sys.stderr)
    def success(self, message: str): self._log("SUCCESS", message, Fore.GREEN, bright=True)
    def debug(self, message: str):
        if self.verbose_flag:
            # Keep debug to stderr for console, but also log to file if configured
            self._log("DEBUG", message, Fore.MAGENTA, console_file=sys.stderr)
    def header(self, title: str):
        header_str = f"\n{'=' * 70}\n{title.center(70)}\n{'=' * 70}"
        print(f"{Fore.CYAN}{Style.BRIGHT}{header_str}{Style.RESET_ALL}")
        if self.log_file: self.log_file.write(f"\n{title.center(70)}\n" + "="*70 + "\n")

    def subheader(self, title: str):
        subheader_str = f"\n{'-' * 70}\n{title.center(70)}\n{'-' * 70}"
        print(f"{Fore.CYAN}{subheader_str}{Style.RESET_ALL}")
        if self.log_file: self.log_file.write(f"\n{title.center(70)}\n" + "-"*70 + "\n")

    def step(self, message: str):
        console_msg = f"{Fore.CYAN}>> {message}{Style.RESET_ALL}"
        print(console_msg)
        if self.log_file: self.log_file.write(f">> {message}\n")

    def notice(self, message: str):
        console_msg = f"{Fore.WHITE}{message}{Style.RESET_ALL}" # Assuming WHITE is defined in DummyFore
        print(console_msg)
        if self.log_file: self.log_file.write(f"NOTICE: {message}\n")

    def close(self):
        if self.log_file:
            self.log_file.close()
            self.log_file = None

# --- Shared Utility Functions (from config_utils.py itself) ---
def colour_util(text: str, ansi_color: str) -> str:
    return f"{ansi_color}{text}{Style.RESET_ALL}"

def deep_merge_dicts_util(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_merge_dicts_util(target[key], value)
        else:
            target[key] = value

class ConfigEnvLoader(yaml.SafeLoader):
    pass

def _env_constructor_util(loader: yaml.SafeLoader, node: yaml.Node) -> str:
    value = loader.construct_scalar(node)
    if not isinstance(value, str) or not value.startswith("${") or not value.endswith("}"):
        print(colour_util(f"Util Warning: Expected ${{VAR}} format for !ENV tag, got: {value}", Fore.YELLOW))
        return str(value)
    raw_content = value.strip("${}")
    env_var_name, _, default_value = raw_content.partition(":-")
    return os.getenv(env_var_name, default_value or "")

ConfigEnvLoader.add_constructor("!ENV", _env_constructor_util)

def resolve_path_relative_to_config(
    path_str: Union[str, Path], config_dir: Path, key_name_for_log: str = "path", logger_instance: Optional[Logger] = None
) -> str:
    log_debug = logger_instance.debug if logger_instance and logger_instance.verbose_flag else lambda _: None # Use passed logger

    if not path_str:
        # Use a basic print if logger not available or error is critical before logger setup
        print(colour_util(f"Util Warning: Empty path for '{key_name_for_log}'.", Fore.YELLOW))
        return ""
    
    log_debug(f"    resolve_path_relative_to_config: path_str='{path_str}', config_dir='{config_dir}', key_name='{key_name_for_log}'")
    
    try:
        path_obj = Path(path_str)
        log_debug(f"    resolve_path_relative_to_config: path_obj='{str(path_obj)}', path_obj.is_absolute()='{path_obj.is_absolute()}'")
        
        resolved_p: Path
        if not path_obj.is_absolute():
            resolved_p = (config_dir / path_obj).resolve()
            log_debug(f"    resolve_path_relative_to_config: Relative path. Joined with config_dir. Resolved to: '{str(resolved_p)}'")
        else:
            resolved_p = path_obj.resolve() # Resolve even absolute paths to normalize them (e.g. symlinks, ..)
            log_debug(f"    resolve_path_relative_to_config: Absolute path. Resolved to: '{str(resolved_p)}'")
        return str(resolved_p)
    except Exception as e:
        print(colour_util(f"Util Error: Processing path '{path_str}' for '{key_name_for_log}': {e}", Fore.RED))
        return str(path_str) # Return original on error to avoid crashing, though it might be wrong

# --- Core Command Building Logic (from config_utils.py itself) ---
MANDATORY_CMD_KEYS = {"bin", "port", "model", "ctx-size"}
PATH_KEYS_IN_CMD = {"bin", "model", "model-draft"} # Add 'lora', 'mmproj' etc. if they are file paths in cmd
MODEL_META_KEYS_FOR_CMD_BUILD = {"aliases", "sampling"} # Keys at model's top-level NOT to be passed as CLI flags by this specific function

def build_llama_server_command_util(model_config: Dict[str, Any]) -> str:
    cmd_map = model_config.get("cmd")
    model_name_for_log = model_config.get("_name_for_log", "Unknown Model")
    if not isinstance(cmd_map, dict):
        raise ValueError(f"Model '{model_name_for_log}': Config missing 'cmd' dictionary.")
    missing_keys = MANDATORY_CMD_KEYS - cmd_map.keys()
    if missing_keys:
        raise ValueError(
            f"Model '{model_name_for_log}': 'cmd' block missing: {', '.join(sorted(missing_keys))}"
        )
    
    args = [str(cmd_map["bin"]), "--port", str(cmd_map["port"]), "--model", str(cmd_map["model"]), "--ctx-size", str(cmd_map["ctx-size"])]
    
    if str(cmd_map.get("gpu-layers", "auto")).lower() != "auto":
        args.extend(["--n-gpu-layers", str(cmd_map["gpu-layers"])])
    if str(cmd_map.get("threads", "auto")).lower() != "auto":
        args.extend(["--threads", str(cmd_map["threads"])])

    # Process other keys from 'cmd' dictionary
    for key, value in cmd_map.items():
        if key in MANDATORY_CMD_KEYS | {"gpu-layers", "threads"}: # Already handled or part of mandatory set
            continue
        cli_flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool) and value: args.append(cli_flag)
        elif value not in (None, False, "auto", "Auto", ""): args.extend([cli_flag, str(value)])

    # Process keys from the model's root level (excluding 'cmd', meta keys, and _name_for_log)
    for key, value in model_config.items():
        if key == "cmd" or key in MODEL_META_KEYS_FOR_CMD_BUILD or key == "_name_for_log":
            continue
        cli_flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool) and value: args.append(cli_flag)
        elif value not in (None, False, "auto", "Auto", ""): args.extend([cli_flag, str(value)])
    
    sampling_config = model_config.get("sampling")
    if isinstance(sampling_config, dict):
        for key, s_value in sampling_config.items():
            cli_flag = f"--{key.replace('_', '-')}"
            args.extend([cli_flag, str(s_value)])
    elif sampling_config is not None:
         print(colour_util(f"Util Warning: 'sampling' for model '{model_name_for_log}' not a dict, ignoring.", Fore.YELLOW))
    return " ".join(args)

# --- Main Configuration Processing Function (from config_utils.py itself) ---
REDUNDANT_BLOCK_PATTERNS = {"suffixes": ["_SAMPLING", "_FLAGS", "_COMMON"], "prefixes": ["COMMON_"]}
PRESERVED_EFFECTIVE_CONFIG_KEYS = {"models"}

def generate_processed_config(
    base_config_path_arg: Path, override_config_path_arg: Optional[Path] = None,
    script_dir_for_overrides: Optional[Path] = None, verbose_logging: bool = True
) -> Dict[str, Any]:
    base_config_path = Path(base_config_path_arg).resolve()
    if not base_config_path.is_file():
        raise FileNotFoundError(f"Base config file not found: {base_config_path}")
    config_root_dir = base_config_path.parent
    if verbose_logging: print(colour_util(f"Util: Config root for paths: {config_root_dir}", Fore.BLUE))
    try:
        effective_config = yaml.load(base_config_path.read_text(encoding="utf-8"), Loader=ConfigEnvLoader) or {}
        if not isinstance(effective_config, dict): raise ValueError("Base config not a dict.")
    except Exception as e: raise ValueError(f"Error loading base config {base_config_path}: {e}") from e
    
    override_to_load: Optional[Path] = None
    if override_config_path_arg:
        resolved_override = Path(override_config_path_arg).expanduser().resolve()
        if not resolved_override.is_file(): raise FileNotFoundError(f"Override file not found: {resolved_override}")
        override_to_load = resolved_override
    elif script_dir_for_overrides:
        hostname = platform.node().split(".")[0].lower()
        default_override = Path(script_dir_for_overrides) / "overrides" / f"{hostname}.yaml"
        if default_override.is_file(): override_to_load = default_override
        elif verbose_logging: print(colour_util(f"Util: No hostname override for '{hostname}'.", Fore.BLUE))

    if override_to_load:
        if verbose_logging: print(colour_util(f"Util: Loading override: {override_to_load}", Fore.BLUE))
        try:
            override_text = override_to_load.read_text(encoding="utf-8")
            if override_text.strip():
                override_data = yaml.load(override_text, Loader=ConfigEnvLoader)
                if isinstance(override_data, dict): deep_merge_dicts_util(effective_config, override_data)
                elif verbose_logging: print(colour_util(f"Util Warning: Override {override_to_load} not a dict.", Fore.YELLOW))
            elif verbose_logging and override_config_path_arg and Path(override_config_path_arg).resolve() == override_to_load:
                 print(colour_util(f"Util Warning: Specified override {override_to_load} is empty.", Fore.YELLOW))
        except Exception as e: raise ValueError(f"Error loading override {override_to_load}: {e}") from e
    elif verbose_logging and not override_config_path_arg :
        print(colour_util("Util: No override specified or default found.", Fore.BLUE))

    models_section = effective_config.get("models", {})
    if isinstance(models_section, dict):
        for model_name, model_data in models_section.items():
            if not isinstance(model_data, dict):
                if verbose_logging: print(colour_util(f"Util Warning: Config for model '{model_name}' not a dict.", Fore.YELLOW))
                continue
            original_cmd_dict = model_data.get("cmd")
            if not isinstance(original_cmd_dict, dict):
                raise ValueError(f"Model '{model_name}' must have a 'cmd' dict.")
            resolved_cmd_dict = original_cmd_dict.copy()
            for key_to_resolve in PATH_KEYS_IN_CMD:
                if key_to_resolve in resolved_cmd_dict and isinstance(resolved_cmd_dict[key_to_resolve], str):
                    original_path = resolved_cmd_dict[key_to_resolve]
                    if original_path:
                        resolved_path_str = resolve_path_relative_to_config(original_path, config_root_dir, key_name_for_log=f"{model_name}.cmd.{key_to_resolve}")
                        if verbose_logging and original_path != resolved_path_str:
                            print(colour_util(f"Util: Model '{model_name}', resolved 'cmd.{key_to_resolve}': '{original_path}' -> '{resolved_path_str}'", Fore.MAGENTA))
                        resolved_cmd_dict[key_to_resolve] = resolved_path_str
            model_data["cmd"] = resolved_cmd_dict
            try:
                build_input = model_data.copy(); build_input["_name_for_log"] = model_name
                model_data["generated_cmd_str"] = build_llama_server_command_util(build_input)
            except ValueError as e: raise ValueError(f"Util Error: Building command for '{model_name}': {e}") from e
    elif verbose_logging: print(colour_util("Util Warning: 'models' section not a dict.", Fore.YELLOW))

    keys_to_cull = [ k for k in list(effective_config.keys()) if k not in PRESERVED_EFFECTIVE_CONFIG_KEYS and \
                    (any(k.endswith(s) for s in REDUNDANT_BLOCK_PATTERNS["suffixes"]) or \
                     any(k.startswith(p) for p in REDUNDANT_BLOCK_PATTERNS["prefixes"])) and \
                    isinstance(effective_config.get(k), dict) ]
    for key in keys_to_cull:
        if verbose_logging: print(colour_util(f"Util: Removing top-level redundant block: {key}", Fore.MAGENTA))
        del effective_config[key]
    return effective_config

# --- Shared Server Management and Utility Functions (Moved from scan_model_memory.py) ---

def _resolve_executable_path_robustly(
    exec_str: str, project_root_path: Path, logger_instance: Optional[Logger] = None
) -> str:
    log_debug = logger_instance.debug if logger_instance and logger_instance.verbose_flag else lambda _: None
    log_error = logger_instance.error if logger_instance else lambda msg: print(f"ERROR: {msg}", file=sys.stderr)

    if not exec_str:
        log_error("  _resolve_executable_path_robustly: Received empty executable string.")
        raise ValueError("Executable path string cannot be empty.")
    
    log_debug(f"    _resolve_executable_path_robustly: Initial exec_str='{exec_str}', project_root_path='{project_root_path}'")

    resolved_exec_str = exec_str
    if platform.system() == "Windows":
        # Check if it ALREADY has common executable extensions or is a directory (unlikely for bin)
        # This aims to prevent double ".exe.exe" if "server.exe" is already provided.
        if not Path(resolved_exec_str).is_dir() and not any(resolved_exec_str.lower().endswith(ext) for ext in [".exe", ".bat", ".com", ".cmd"]):
            resolved_exec_str += ".exe"
            log_debug(f"    Windows OS: Appended '.exe' to '{exec_str}', result: '{resolved_exec_str}'")
        else:
            log_debug(f"    Windows OS: Original exec_str '{exec_str}' already has an extension or is a directory. No '.exe' appended.")

    expanded_str = os.path.expanduser(resolved_exec_str)
    log_debug(f"    _resolve_executable_path_robustly: After os.path.expanduser: '{expanded_str}' (from '{resolved_exec_str}')")

    if platform.system() == "Windows":
        # This regex is for paths like "C:path" (missing separator) -> "C:\path"
        m = re.match(r"([a-zA-Z]):(?![\\/])(.+)", expanded_str)
        if m:
            drive, rest_of_path = m.groups()
            corrected_str = f"{drive}:\\{rest_of_path}"
            log_debug(f"    Path Correction (Missing Separator): Windows path '{expanded_str}' interpreted as '{corrected_str}'")
            expanded_str = corrected_str
        else:
            log_debug(f"    Path Correction (Missing Separator): No match for '{expanded_str}', no correction applied.")
    
    path_obj = Path(expanded_str)
    log_debug(f"    _resolve_executable_path_robustly: path_obj='{str(path_obj)}', path_obj.is_absolute()='{path_obj.is_absolute()}'")
    
    if path_obj.is_absolute():
        final_path = path_obj.resolve(strict=False)
        log_debug(f"    _resolve_executable_path_robustly: Path was absolute. final_path (after resolve): '{str(final_path)}'")
    else:
        final_path = (project_root_path / path_obj).resolve(strict=False)
        log_debug(f"    _resolve_executable_path_robustly: Path was relative. Joined with project_root_path. final_path (after resolve): '{str(final_path)}'")
    
    # This check for .exe was problematic before, let's re-evaluate.
    # The primary goal is: if "server" is given, try "server.exe" on Windows.
    # If "server.exe" is given, use it.
    # If "server" is given, and "server.exe" doesn't exist but "server" (no ext) does, use "server".
    if platform.system() == "Windows" and exec_str != resolved_exec_str: # This means ".exe" was added.
        # `final_path` currently points to the ".exe" version.
        # `original_path_obj_no_ext` would be Path(exec_str) before ".exe" was added.
        original_path_obj_no_ext = Path(os.path.expanduser(exec_str)) 
        
        # Resolve original path (without .exe) relative to project_root if it was relative, or as is if absolute
        original_final_path_no_ext: Path
        if original_path_obj_no_ext.is_absolute():
            original_final_path_no_ext = original_path_obj_no_ext.resolve(strict=False)
        else:
            original_final_path_no_ext = (project_root_path / original_path_obj_no_ext).resolve(strict=False)
        
        log_debug(f"    Windows .exe check: final_path (with .exe)='{final_path}', original_final_path_no_ext='{original_final_path_no_ext}'")
        
        if not final_path.exists() and original_final_path_no_ext.exists():
            log_debug(f"    Path with .exe ('{final_path}') not found, but original ('{original_final_path_no_ext}') exists. Using original.")
            return str(original_final_path_no_ext)
            
    log_debug(f"    _resolve_executable_path_robustly: Returning final_path='{str(final_path)}'")
    return str(final_path)

def _dump_stderr_on_failure(stderr_log_path: Optional[Path], model_name: str, logger_instance: Logger):
    # Ensure logger_instance is used directly as it's passed
    if not stderr_log_path or not stderr_log_path.exists():
        logger_instance.warn(f"  Stderr log for '{model_name}' not available (Path: {stderr_log_path}).")
        return
    logger_instance.warn(f"  Attempting to read last lines from stderr log: {stderr_log_path}")
    try:
        with open(stderr_log_path, 'r', encoding='utf-8', errors='replace') as err_f_read:
            lines = err_f_read.readlines(); last_lines = lines[-20:] # Increased to 20 lines
            if last_lines:
                logger_instance.warn("  Last lines of server stderr:")
                for line in last_lines: logger_instance.warn(f"    {line.strip()}")
            else: logger_instance.warn(f"    Log file '{stderr_log_path}' was empty.")
    except Exception as log_read_e: logger_instance.warn(f"    Could not read stderr log ({stderr_log_path}): {log_read_e}")

def start_llama_server(
    executable_path_str: str, 
    arguments_list: List[str], 
    model_name: str,
    temp_dir: Path, 
    logger_instance: Logger, 
    project_root_for_resolution: Path # Added project_root_for_resolution
) -> Optional[Tuple[subprocess.Popen, Path, Path]]:
    try:
        resolved_executable_for_popen = _resolve_executable_path_robustly(
            executable_path_str, project_root_for_resolution, logger_instance
        )
    except Exception as e:
        logger_instance.error(f"Error resolving exec path '{executable_path_str}' for '{model_name}': {e}")
        _dump_stderr_on_failure(None, model_name, logger_instance)
        return None

    try:
        args_list_for_popen = [resolved_executable_for_popen] + [str(arg) for arg in arguments_list]
        logger_instance.debug(f"Final Popen args for '{model_name}': {args_list_for_popen}")
    except Exception as e: 
        logger_instance.error(f"Error preparing Popen args for '{model_name}': {e}")
        _dump_stderr_on_failure(None, model_name, logger_instance)
        return None
    
    unique_suffix = datetime.datetime.now().strftime('%H%M%S%f')
    stdout_log = temp_dir / f"{model_name}_{unique_suffix}_stdout.log"
    stderr_log = temp_dir / f"{model_name}_{unique_suffix}_stderr.log"

    logger_instance.info(f"  Attempting to start server for '{model_name}'...")
    logger_instance.debug(f"    Command (list for Popen): {args_list_for_popen}") 
    logger_instance.debug(f"    Stdout Log: {stdout_log}, Stderr Log: {stderr_log}")
    try:
        with open(stdout_log, 'wb') as f_out, open(stderr_log, 'wb') as f_err:
            process = subprocess.Popen(args_list_for_popen, stdout=f_out, stderr=f_err) 
        time.sleep(0.7) 
        if process.poll() is not None: 
            logger_instance.error(f"Server '{model_name}' failed to start/exited (Code: {process.returncode}).")
            _dump_stderr_on_failure(stderr_log, model_name, logger_instance)
            return None
        logger_instance.success(f"  Server '{model_name}' started (PID: {process.pid}).")
        return process, stdout_log, stderr_log
    except FileNotFoundError: 
        logger_instance.error(f"Executable not found for '{model_name}': {args_list_for_popen[0]}")
        _dump_stderr_on_failure(None, model_name, logger_instance) # Path might not exist yet
        return None
    except Exception as e:
        logger_instance.error(f"Exception starting server '{model_name}': {e}")
        _dump_stderr_on_failure(stderr_log if stderr_log.exists() else None, model_name, logger_instance)
        return None

def stop_llama_server(process: Optional[subprocess.Popen], model_name: str, logger_instance: Logger):
    if not process:
        logger_instance.debug(f"No process object to stop for '{model_name}'.")
        return
    pid = process.pid
    if process.poll() is not None:
        logger_instance.info(f"  Server '{model_name}' (PID {pid}) already stopped (Code: {process.returncode}).")
        return
    logger_instance.info(f"  Stopping server '{model_name}' (PID {pid})...")
    try:
        process.terminate()
        process.wait(timeout=PROCESS_TERMINATE_TIMEOUT_S)
        logger_instance.success(f"  Server '{model_name}' (PID {pid}) terminated gracefully.")
    except subprocess.TimeoutExpired:
        logger_instance.warn(f"  Server '{model_name}' (PID {pid}) did not terminate. Forcing kill...")
        try:
            process.kill()
            process.wait(timeout=5) # Shorter timeout for kill
            logger_instance.success(f"  Server '{model_name}' (PID {pid}) killed.")
        except Exception as e_kill:
            logger_instance.error(f"  Error during force kill of '{model_name}' (PID {pid}): {e_kill}")
    except Exception as e_term:
        logger_instance.warn(f"  Error during termination of '{model_name}' (PID {pid}): {e_term}.")

def wait_for_server_health(
    process: subprocess.Popen, health_check_url: str, timeout_s: int, 
    poll_interval_s: float, model_name: str, logger_instance: Logger
) -> bool:
    logger_instance.info(f"  Waiting for '{model_name}' (PID: {process.pid}) health at {health_check_url} (timeout: {timeout_s}s)...")
    start_time = time.monotonic()
    attempt = 0
    while time.monotonic() - start_time < timeout_s:
        attempt += 1
        if process.poll() is not None:
            logger_instance.warn(f"  Server '{model_name}' (PID {process.pid}) exited (Code {process.returncode}) while waiting for health.")
            return False
        try:
            req_timeout = max(1.0, poll_interval_s - 0.5) # Ensure timeout is reasonable
            response = requests.get(health_check_url, timeout=req_timeout)
            if response.status_code == 200:
                try: # Try to parse JSON, but don't fail if not JSON (some servers just return 200 OK)
                    health_data = response.json()
                    status_from_json = health_data.get("status", "ok_no_status_field")
                    if status_from_json in ["ok", "healthy", "ok_no_status_field"]:
                        logger_instance.success(f"  Server '{model_name}' healthy (Attempt {attempt}, Status: {status_from_json}).")
                        return True
                    else:
                        logger_instance.debug(f"  Health check '{model_name}' (Attempt {attempt}): HTTP 200, but status is '{status_from_json}'.")
                except requests.exceptions.JSONDecodeError:
                    logger_instance.success(f"  Server '{model_name}' healthy (Attempt {attempt}, HTTP 200 but response not JSON).")
                    return True
            else: 
                logger_instance.debug(f"  Health check '{model_name}' (Attempt {attempt}, PID {process.pid}): HTTP {response.status_code}.")
        except requests.exceptions.ConnectionError: 
            logger_instance.debug(f"  Health check '{model_name}' (Attempt {attempt}, PID {process.pid}): Connection refused.")
        except requests.exceptions.Timeout: 
            logger_instance.debug(f"  Health check '{model_name}' (Attempt {attempt}, PID {process.pid}): Request timed out.")
        except requests.RequestException as e_req: 
            logger_instance.debug(f"  Health check '{model_name}' (Attempt {attempt}, PID {process.pid}): Request Exception: {e_req}.")
        
        # Sleep for poll_interval, but check process status frequently during sleep
        sleep_end_time = time.monotonic() + poll_interval_s
        while time.monotonic() < sleep_end_time:
            if process.poll() is not None:
                 logger_instance.warn(f"  Server '{model_name}' (PID {process.pid}) exited (Code {process.returncode}) during health check polling interval.")
                 return False
            time.sleep(0.1) # Short sleep, then re-check loop condition or process status

    logger_instance.warn(f"  Server '{model_name}' (PID {process.pid}) did not become healthy at {health_check_url} within {timeout_s}s.")
    return False

def color_status(status: str) -> str: # Simple utility, can be shared
    status_lower = status.lower()
    if "success" in status_lower: return f"{Fore.GREEN}{Style.BRIGHT}{status}{Style.RESET_ALL}"
    if any(term in status_lower for term in ["fail", "error", "timeout", "invalid", "missing", "exited"]): 
        return f"{Fore.RED}{Style.BRIGHT}{status}{Style.RESET_ALL}"
    if any(term in status_lower for term in ["warn", "not run", "parse error", "no buffers"]): # Added "no buffers"
        return f"{Fore.YELLOW}{Style.BRIGHT}{status}{Style.RESET_ALL}"
    return status


# --- Self-test execution block (remains the same, for testing config_utils.py itself) ---
if __name__ == "__main__":
    # ... (self-test code from your config_utils.py) ...
    print(f"{Style.BRIGHT}--- Running config_utils.py self-test ---{Style.RESET_ALL}")
    
    current_file_path = Path(__file__).resolve()
    project_root_for_test = current_file_path.parent.parent 

    test_base_path = project_root_for_test / "config.base.yaml"
    # Example: Try with a specific override if one exists for testing
    test_override_path_arg = project_root_for_test / "overrides" / f"{platform.node().split('.')[0].lower()}.yaml"
    if not test_override_path_arg.exists():
        test_override_path_arg = None # No default override for this test run

    print(f"Base config path for test: {test_base_path}")
    print(f"Explicit override path for test: {test_override_path_arg}")
    print(f"Script dir for hostname override lookup (project root): {project_root_for_test}")

    if not test_base_path.exists():
        print(colour_util(f"Test SKIPPED: Base config '{test_base_path}' not found.", Fore.RED))
    else:
        try:
            print(colour_util("\n--- Testing config processing (verbose_logging=True) ---", Fore.CYAN))
            processed_config_result = generate_processed_config(
                base_config_path_arg=test_base_path,
                override_config_path_arg=test_override_path_arg,
                script_dir_for_overrides=project_root_for_test,
                verbose_logging=True
            )
            print(f"\n{Style.BRIGHT}--- Processed Configuration Output (YAML) ---{Style.RESET_ALL}")
            print(yaml.dump(processed_config_result, indent=2, sort_keys=False, allow_unicode=True, Dumper=yaml.SafeDumper)) # Use SafeDumper
            
            print(f"\n{Style.BRIGHT}--- Basic Validation of Processed Config ---{Style.RESET_ALL}")
            if "models" in processed_config_result and isinstance(processed_config_result["models"], dict):
                num_models = len(processed_config_result['models'])
                print(colour_util(f"Found 'models' section with {num_models} entries.", Fore.GREEN))
                
                for model_name, model_data in processed_config_result['models'].items():
                    print(f"  Validating model: '{model_name}'")
                    if "cmd" in model_data and isinstance(model_data["cmd"], dict):
                        cmd_dict = model_data["cmd"]
                        print(colour_util(f"    'cmd' dictionary found.", Fore.GREEN))
                        for path_key_check in PATH_KEYS_IN_CMD:
                            if path_key_check in cmd_dict:
                                path_val = cmd_dict[path_key_check]
                                if not path_val or Path(path_val).is_absolute(): 
                                    print(colour_util(f"      'cmd.{path_key_check}': '{path_val}' (absolute or empty, OK).", Fore.GREEN))
                                else:
                                    print(colour_util(f"      'cmd.{path_key_check}': '{path_val}' (WARNING: not absolute).", Fore.YELLOW))
                    else:
                         print(colour_util(f"    Model '{model_name}': 'cmd' key is NOT a dictionary or missing.", Fore.RED))
                         
                    if "generated_cmd_str" in model_data and isinstance(model_data["generated_cmd_str"], str):
                         print(colour_util(f"    'generated_cmd_str' found and is a string.", Fore.GREEN))
                    else:
                         print(colour_util(f"    Model '{model_name}': 'generated_cmd_str' missing or not a string.", Fore.RED))
                         
                    # This check might be too strict if some models don't have sampling (though build_util expects it)
                    # if "sampling" not in model_data: 
                    #     print(colour_util(f"    'sampling' key correctly removed from top level.", Fore.GREEN))
                    # else:
                    #      print(colour_util(f"    Model '{model_name}': 'sampling' key was NOT removed.", Fore.YELLOW))
                    break # Only validate the first model for brevity in self-test
            else:
                print(colour_util("No 'models' section found or not a dict in processed config.", Fore.RED))

            removed_test_key = "COMMON_FLAGS" # Example, adjust if your base has other common blocks
            if removed_test_key not in processed_config_result:
                 print(colour_util(f"Top-level block '{removed_test_key}' (if existed) correctly removed.", Fore.GREEN))
            else:
                 print(colour_util(f"Top-level block '{removed_test_key}' might still be present.", Fore.YELLOW))

        except Exception as e_test:
            print(colour_util(f"\n--- ERROR during config_utils self-test ---", Fore.RED))
            print(f"{type(e_test).__name__}: {e_test}")
            import traceback
            traceback.print_exc()

    print(f"\n{Style.BRIGHT}--- config_utils.py self-test finished ---{Style.RESET_ALL}")