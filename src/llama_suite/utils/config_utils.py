# config_utils.py (new-layout ready)
from copy import deepcopy
import json
import os
import platform
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import datetime
from typing import Any, Dict, Optional, Union, List, Tuple, TextIO

import yaml
import requests
import psutil
# Initialize Colorama with type-checker-safe fallbacks
from typing import Any, cast

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
except Exception:
    # Lightweight stand-ins so attributes exist for type checkers
    class _DummyStyle:
        RESET_ALL: str = ""
        BRIGHT: str = ""     # added
        DIM: str = ""        # optional but handy

    class _DummyFore:
        RED: str = ""
        YELLOW: str = ""
        GREEN: str = ""
        CYAN: str = ""
        BLUE: str = ""
        MAGENTA: str = ""
        WHITE: str = ""

    # cast(...) silences static analyzers that know the real Style/Fore types
    Style = cast(Any, _DummyStyle())
    Fore = cast(Any, _DummyFore())


# --- Shared Constants ---
PROCESS_TERMINATE_TIMEOUT_S = 10
DEFAULT_HEALTH_POLL_INTERVAL_S = 2.0

# ========== NEW: project root detection & default dirs ==========
def find_project_root(start: Optional[Path] = None) -> Path:
    """
    Walk up from 'start' (or cwd) to find the repo root by looking for common anchors:
    - 'configs' dir (new layout)
    - 'src' dir (PEP 621 layout)
    - '.git' or 'pyproject.toml'
    Fallback: the provided 'start' or cwd.
    """
    # Explicit override (for containers/Kubernetes): point to a data root that contains configs/runs/var/models.
    # Example: LLAMA_SUITE_ROOT=/data
    env_root = (os.getenv("LLAMA_SUITE_ROOT") or os.getenv("LLS_ROOT") or "").strip()
    if env_root:
        try:
            return Path(env_root).expanduser().resolve()
        except Exception:
            return Path(env_root).expanduser()

    here = (start or Path.cwd()).resolve()
    candidates = [here] + list(here.parents)
    for p in candidates:
        if (p / "configs").is_dir() or (p / "src").is_dir() or (p / ".git").exists() or (p / "pyproject.toml").exists():
            return p
    return here

def default_paths_from_base_config(base_config_path: Path) -> Dict[str, Path]:
    """
    Compute canonical directories given a base config path under the new layout.
    """
    project_root = find_project_root(base_config_path)
    return {
        "project_root": project_root,
        "configs_dir": project_root / "configs",
        "overrides_dir": project_root / "configs" / "overrides",
        "generated_dir": project_root / "configs" / "generated",
        "vendor_dir": project_root / "vendor",
        "models_dir": project_root / "models",
        "runs_dir": project_root / "runs",
        "var_dir": project_root / "var",
    }
# ===============================================================

# --- Shared Logger Class ---
class Logger:
    def __init__(self, verbose: bool = False, log_file_path: Optional[Path] = None, plain: Optional[bool] = None):
        self.verbose_flag = verbose
        self.log_file_path = log_file_path
        self.plain = _is_plain_mode() if plain is None else bool(plain)
        self.log_file: Optional[TextIO] = None
        if self.log_file_path:
            try:
                self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
                self.log_file = open(self.log_file_path, 'a', encoding='utf-8')
            except Exception as e:
                print(f"Error opening log file {self.log_file_path}: {e}", file=sys.stderr)
                self.log_file = None

    def _get_timestamp(self) -> str:
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def _log(self, level: str, message: str, color: str = "", bright: bool = False, console_file: TextIO = sys.stdout):
        timestamp = self._get_timestamp()
        if self.plain:
            console_message = f"{level}: {message}"
            print(console_message, file=console_file)
        else:
            style_prefix = Style.BRIGHT if bright else ""
            console_message = f"{timestamp} [{style_prefix}{color}{level}{Style.RESET_ALL}] {style_prefix}{color}{message}{Style.RESET_ALL}"
            print(console_message, file=console_file)
        if self.log_file:
            file_message = f"{timestamp} [{level}] {message}"
            try:
                self.log_file.write(file_message + "\n")
                self.log_file.flush()
            except Exception as e:
                print(f"Error writing to log file: {e}", file=sys.stderr)

    def info(self, message: str): self._log("INFO", message, Fore.BLUE)
    def warn(self, message: str): self._log("WARN", message, Fore.YELLOW, bright=True)
    def error(self, message: str): self._log("ERROR", message, Fore.RED, bright=True, console_file=sys.stderr)
    def success(self, message: str): self._log("SUCCESS", message, Fore.GREEN, bright=True)
    def debug(self, message: str):
        if self.verbose_flag:
            self._log("DEBUG", message, Fore.MAGENTA, console_file=sys.stderr)
    def header(self, title: str):
        if self.plain:
            print(f"== {title} ==")
        else:
            header_str = f"\n{'=' * 70}\n{title.center(70)}\n{'=' * 70}"
            print(f"{Fore.CYAN}{Style.BRIGHT}{header_str}{Style.RESET_ALL}")
        if self.log_file: self.log_file.write(f"\n{title.center(70)}\n" + "="*70 + "\n")
    def subheader(self, title: str):
        if self.plain:
            print(f"-- {title} --")
        else:
            subheader_str = f"\n{'-' * 70}\n{title.center(70)}\n{'-' * 70}"
            print(f"{Fore.CYAN}{subheader_str}{Style.RESET_ALL}")
        if self.log_file: self.log_file.write(f"\n{title.center(70)}\n" + "-"*70 + "\n")
    def step(self, message: str):
        if self.plain:
            print(f">> {message}")
        else:
            console_msg = f"{Fore.CYAN}>> {message}{Style.RESET_ALL}"
            print(console_msg)
        if self.log_file: self.log_file.write(f">> {message}\n")
    def notice(self, message: str):
        if self.plain:
            print(message)
        else:
            console_msg = f"{Fore.WHITE}{message}{Style.RESET_ALL}"
            print(console_msg)
        if self.log_file: self.log_file.write(f"NOTICE: {message}\n")
    def close(self):
        if self.log_file:
            self.log_file.close()
            self.log_file = None

# --- Shared Utility Functions ---
def _is_plain_mode() -> bool:
    v = (os.getenv("LLAMA_SUITE_PLAIN") or os.getenv("LLS_PLAIN") or "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}

def colour_util(text: str, ansi_color: str) -> str:
    if _is_plain_mode():
        return text
    return f"{ansi_color}{text}{Style.RESET_ALL}"

def deep_merge_dicts_util(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_merge_dicts_util(target[key], value)
        else:
            target[key] = value

class ConfigEnvLoader(yaml.SafeLoader):
    pass

def _env_constructor_util(loader: yaml.SafeLoader, node: yaml.ScalarNode) -> str:
    # We expect a scalar like: !ENV ${VAR:-default}
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
    log_debug = logger_instance.debug if logger_instance and logger_instance.verbose_flag else lambda _: None
    if not path_str:
        print(colour_util(f"Util Warning: Empty path for '{key_name_for_log}'.", Fore.YELLOW))
        return ""

    # NEW: expand environment variables like %PROJECT_ROOT% or $PROJECT_ROOT even if not using !ENV
    expanded = os.path.expandvars(str(path_str)).strip()
    log_debug(f"    resolve_path_relative_to_config: raw='{path_str}', expanded='{expanded}', config_dir='{config_dir}'")

    try:
        path_obj = Path(expanded)
        log_debug(f"    resolve_path_relative_to_config: path_obj='{str(path_obj)}', is_absolute()='{path_obj.is_absolute()}'")
        if not path_obj.is_absolute():
            resolved_p = (config_dir / path_obj).resolve()
            log_debug(f"    Relative path -> '{resolved_p}'")
        else:
            resolved_p = path_obj.resolve()
            log_debug(f"    Absolute path -> '{resolved_p}'")
        return str(resolved_p)
    except Exception as e:
        print(colour_util(f"Util Error: Processing path '{path_str}' for '{key_name_for_log}': {e}", Fore.RED))
        return str(path_str)

# --- Core Command Building Logic ---
MANDATORY_CMD_KEYS = {"bin", "port", "model", "ctx-size"}
PATH_KEYS_IN_CMD = {"bin", "model", "model-draft"}  # extend as needed
MODEL_META_KEYS_FOR_CMD_BUILD = {"aliases", "sampling"}
COMMON_FLAGS_KEY = "COMMON_FLAGS"


def _append_cli_arg(args: List[str], key: str, value: Any) -> None:
    key_norm = key.replace("_", "-")
    cli_flag = f"--{key_norm}"

    if key_norm.endswith("-kwargs"):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:
                pass
        if isinstance(value, (dict, list, int, float, bool)) or value is None:
            value = json.dumps(value, separators=(",", ":"))
        args.extend([cli_flag, shlex.quote(str(value))])
        return

    if key_norm == "flash-attn" and isinstance(value, bool):
        args.extend([cli_flag, "on" if value else "off"])
        return

    if isinstance(value, bool):
        if value:
            args.append(cli_flag)
        return

    if value not in (None, False, "auto", "Auto", ""):
        args.extend([cli_flag, str(value)])


def apply_common_cmd_defaults_util(config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge top-level COMMON_FLAGS into each model's cmd with model keys taking priority."""
    common_cmd = config.get(COMMON_FLAGS_KEY)
    if common_cmd is None:
        return config
    if not isinstance(common_cmd, dict):
        raise ValueError(f"Top-level '{COMMON_FLAGS_KEY}' must be a mapping.")

    models_section = config.get("models", {})
    if not isinstance(models_section, dict):
        return config

    for _, model_data in models_section.items():
        if not isinstance(model_data, dict):
            continue
        cmd = model_data.get("cmd")
        if cmd is None:
            continue
        if not isinstance(cmd, dict):
            continue
        merged_cmd = deepcopy(common_cmd)
        deep_merge_dicts_util(merged_cmd, cmd)
        model_data["cmd"] = merged_cmd

    return config

def build_llama_server_command_util(model_config: Dict[str, Any]) -> str:
    cmd_map = model_config.get("cmd")
    model_name_for_log = model_config.get("_name_for_log", "Unknown Model")
    if not isinstance(cmd_map, dict):
        raise ValueError(f"Model '{model_name_for_log}': Config missing 'cmd' dictionary.")
    missing_keys = MANDATORY_CMD_KEYS - cmd_map.keys()
    if missing_keys:
        raise ValueError(f"Model '{model_name_for_log}': 'cmd' block missing: {', '.join(sorted(missing_keys))}")

    args = [str(cmd_map["bin"]), "--port", str(cmd_map["port"]), "--model", str(cmd_map["model"]), "--ctx-size", str(cmd_map["ctx-size"])]

    if str(cmd_map.get("gpu-layers", "auto")).lower() != "auto":
        args.extend(["--n-gpu-layers", str(cmd_map["gpu-layers"])])
    if str(cmd_map.get("threads", "auto")).lower() != "auto":
        args.extend(["--threads", str(cmd_map["threads"])])

    for key, value in cmd_map.items():
        if key in MANDATORY_CMD_KEYS | {"gpu-layers", "threads"}:
            continue
        _append_cli_arg(args, key, value)

    for key, value in model_config.items():
        if key in {"cmd", "_name_for_log"} | MODEL_META_KEYS_FOR_CMD_BUILD:
            continue
        _append_cli_arg(args, key, value)

    sampling_config = model_config.get("sampling")
    if isinstance(sampling_config, dict):
        for key, s_value in sampling_config.items():
            cli_flag = f"--{key.replace('_', '-')}"
            args.extend([cli_flag, str(s_value)])
    elif sampling_config is not None:
        print(colour_util(f"Util Warning: 'sampling' for model '{model_name_for_log}' not a dict, ignoring.", Fore.YELLOW))
    return " ".join(args)

# --- Main Configuration Processing Function ---
REDUNDANT_BLOCK_PATTERNS = {"suffixes": ["_SAMPLING", "_FLAGS", "_COMMON"], "prefixes": ["COMMON_"]}
PRESERVED_EFFECTIVE_CONFIG_KEYS = {"models"}

def generate_processed_config(
    base_config_path_arg: Path,
    override_config_path_arg: Optional[Path] = None,
    script_dir_for_overrides: Optional[Path] = None,
    verbose_logging: bool = True
) -> Dict[str, Any]:
    base_config_path = Path(base_config_path_arg).resolve()
    if not base_config_path.is_file():
        raise FileNotFoundError(f"Base config file not found: {base_config_path}")

    # NEW: compute canonical dirs + export PROJECT_ROOT for !ENV or $PROJECT_ROOT
    paths = default_paths_from_base_config(base_config_path)
    project_root = paths["project_root"]
    os.environ.setdefault("PROJECT_ROOT", str(project_root))

    config_root_dir = base_config_path.parent
    if verbose_logging: print(colour_util(f"Util: Config root for paths: {config_root_dir}", Fore.BLUE))
    if verbose_logging: print(colour_util(f"Util: Detected project root: {project_root}", Fore.BLUE))

    try:
        effective_config = yaml.load(base_config_path.read_text(encoding="utf-8"), Loader=ConfigEnvLoader) or {}
        if not isinstance(effective_config, dict):
            raise ValueError("Base config not a dict.")
    except Exception as e:
        raise ValueError(f"Error loading base config {base_config_path}: {e}") from e

    # NEW: find override file (new layout first, then backward-compat)
    override_to_load: Optional[Path] = None
    hostname = platform.node().split(".")[0].lower()

    # precedence: explicit > provided-dir > new-layout default > old-layout default
    if override_config_path_arg:
        resolved_override = Path(override_config_path_arg).expanduser().resolve()
        if not resolved_override.is_file():
            raise FileNotFoundError(f"Override file not found: {resolved_override}")
        override_to_load = resolved_override
    else:
        candidates: List[Path] = []
        if script_dir_for_overrides:
            candidates.append(Path(script_dir_for_overrides) / "overrides" / f"{hostname}.yaml")
        candidates.append(paths["overrides_dir"] / f"{hostname}.yaml")              # NEW layout
        candidates.append(config_root_dir / "overrides" / f"{hostname}.yaml")       # old layout fallback
        for cand in candidates:
            if cand.is_file():
                override_to_load = cand
                break
        if verbose_logging and not override_to_load:
            print(colour_util(f"Util: No hostname override found for '{hostname}'.", Fore.BLUE))

    if override_to_load:
        if verbose_logging: print(colour_util(f"Util: Loading override: {override_to_load}", Fore.BLUE))
        try:
            override_text = override_to_load.read_text(encoding="utf-8")
            if override_text.strip():
                override_data = yaml.load(override_text, Loader=ConfigEnvLoader)
                if isinstance(override_data, dict):
                    deep_merge_dicts_util(effective_config, override_data)
                elif verbose_logging:
                    print(colour_util(f"Util Warning: Override {override_to_load} not a dict.", Fore.YELLOW))
            elif verbose_logging and override_config_path_arg and Path(override_config_path_arg).resolve() == override_to_load:
                 print(colour_util(f"Util Warning: Specified override {override_to_load} is empty.", Fore.YELLOW))
        except Exception as e:
            raise ValueError(f"Error loading override {override_to_load}: {e}") from e
    elif verbose_logging and not override_config_path_arg:
        print(colour_util("Util: No override specified or default found.", Fore.BLUE))

    apply_common_cmd_defaults_util(effective_config)

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
                        resolved_path_str = resolve_path_relative_to_config(
                            original_path, config_root_dir, key_name_for_log=f"{model_name}.cmd.{key_to_resolve}"
                        )
                        if verbose_logging and original_path != resolved_path_str:
                            print(colour_util(
                                f"Util: Model '{model_name}', resolved 'cmd.{key_to_resolve}': '{original_path}' -> '{resolved_path_str}'",
                                Fore.MAGENTA
                            ))
                        resolved_cmd_dict[key_to_resolve] = resolved_path_str
            model_data["cmd"] = resolved_cmd_dict
            try:
                build_input = model_data.copy(); build_input["_name_for_log"] = model_name
                model_data["generated_cmd_str"] = build_llama_server_command_util(build_input)
            except ValueError as e:
                raise ValueError(f"Util Error: Building command for '{model_name}': {e}") from e
    elif verbose_logging:
        print(colour_util("Util Warning: 'models' section not a dict.", Fore.YELLOW))

    keys_to_cull = [
        k for k in list(effective_config.keys())
        if k not in PRESERVED_EFFECTIVE_CONFIG_KEYS
        and (any(k.endswith(s) for s in REDUNDANT_BLOCK_PATTERNS["suffixes"])
             or any(k.startswith(p) for p in REDUNDANT_BLOCK_PATTERNS["prefixes"]))
        and isinstance(effective_config.get(k), dict)
    ]
    for key in keys_to_cull:
        if verbose_logging: print(colour_util(f"Util: Removing top-level redundant block: {key}", Fore.MAGENTA))
        del effective_config[key]
    return effective_config

# --- Server Management and Utility Functions ---
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
        if not Path(resolved_exec_str).is_dir() and not any(resolved_exec_str.lower().endswith(ext) for ext in [".exe", ".bat", ".com", ".cmd"]):
            resolved_exec_str += ".exe"
            log_debug(f"    Windows OS: Appended '.exe' to '{exec_str}', result: '{resolved_exec_str}'")
        else:
            log_debug(f"    Windows OS: '{exec_str}' already has an extension or is a directory.")

    expanded_str = os.path.expanduser(os.path.expandvars(resolved_exec_str))
    log_debug(f"    After expanduser/expandvars: '{expanded_str}'")

    if platform.system() == "Windows":
        m = re.match(r"([a-zA-Z]):(?![\\/])(.+)", expanded_str)
        if m:
            drive, rest_of_path = m.groups()
            corrected_str = f"{drive}:\\{rest_of_path}"
            log_debug(f"    Path Correction: '{expanded_str}' -> '{corrected_str}'")
            expanded_str = corrected_str

    path_obj = Path(expanded_str)
    log_debug(f"    path_obj='{str(path_obj)}', is_absolute()='{path_obj.is_absolute()}'")

    if path_obj.is_absolute():
        final_path = path_obj.resolve(strict=False)
    else:
        final_path = (project_root_path / path_obj).resolve(strict=False)

    if platform.system() == "Windows" and exec_str != resolved_exec_str:
        original_path_obj_no_ext = Path(os.path.expanduser(os.path.expandvars(exec_str)))
        if original_path_obj_no_ext.is_absolute():
            original_final_path_no_ext = original_path_obj_no_ext.resolve(strict=False)
        else:
            original_final_path_no_ext = (project_root_path / original_path_obj_no_ext).resolve(strict=False)
        log_debug(f"    with .exe='{final_path}', without .exe='{original_final_path_no_ext}'")
        if not final_path.exists() and original_final_path_no_ext.exists():
            return str(original_final_path_no_ext)

    log_debug(f"    Returning final_path='{str(final_path)}'")
    return str(final_path)

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
            else:
                logger_instance.warn(f"    Log file '{stderr_log_path}' was empty.")
    except Exception as log_read_e:
        logger_instance.warn(f"    Could not read stderr log ({stderr_log_path}): {log_read_e}")

def start_llama_server(
    executable_path_str: str,
    arguments_list: List[str],
    model_name: str,
    temp_dir: Path,
    logger_instance: Logger,
    project_root_for_resolution: Path
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
        _dump_stderr_on_failure(None, model_name, logger_instance)
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
            process.wait(timeout=5)
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
            req_timeout = max(1.0, poll_interval_s - 0.5)
            response = requests.get(health_check_url, timeout=req_timeout)
            if response.status_code == 200:
                try:
                    health_data = response.json()
                    status_from_json = health_data.get("status", "ok_no_status_field")
                    if status_from_json in ["ok", "healthy", "ok_no_status_field"]:
                        logger_instance.success(f"  Server '{model_name}' healthy (Attempt {attempt}, Status: {status_from_json}).")
                        return True
                    else:
                        logger_instance.debug(f"  Health check '{model_name}' (Attempt {attempt}): HTTP 200, but status is '{status_from_json}'.")
                except ValueError:  # JSON decode error
                    logger_instance.success(f"  Server '{model_name}' healthy (Attempt {attempt}, HTTP 200 non-JSON).")
                    return True
            else:
                logger_instance.debug(f"  Health check '{model_name}' (Attempt {attempt}, PID {process.pid}): HTTP {response.status_code}.")
        except requests.exceptions.ConnectionError:
            logger_instance.debug(f"  Health check '{model_name}' (Attempt {attempt}, PID {process.pid}): Connection refused.")
        except requests.exceptions.Timeout:
            logger_instance.debug(f"  Health check '{model_name}' (Attempt {attempt}, PID {process.pid}): Request timed out.")
        except requests.RequestException as e_req:
            logger_instance.debug(f"  Health check '{model_name}' (Attempt {attempt}, PID {process.pid}): Request Exception: {e_req}.")

        sleep_end_time = time.monotonic() + poll_interval_s
        while time.monotonic() < sleep_end_time:
            if process.poll() is not None:
                logger_instance.warn(f"  Server '{model_name}' (PID {process.pid}) exited (Code {process.returncode}) during health check polling interval.")
                return False
            time.sleep(0.1)

    logger_instance.warn(f"  Server '{model_name}' (PID {process.pid}) did not become healthy at {health_check_url} within {timeout_s}s.")
    return False

def color_status(status: str) -> str:
    status_lower = status.lower()
    if "success" in status_lower: return f"{Fore.GREEN}{Style.BRIGHT}{status}{Style.RESET_ALL}"
    if any(term in status_lower for term in ["fail", "error", "timeout", "invalid", "missing", "exited"]):
        return f"{Fore.RED}{Style.BRIGHT}{status}{Style.RESET_ALL}"
    if any(term in status_lower for term in ["warn", "not run", "parse error", "no buffers"]):
        return f"{Fore.YELLOW}{Style.BRIGHT}{status}{Style.RESET_ALL}"
    return status

# --- Self-test (updated for new layout) ---
if __name__ == "__main__":
    print(f"{Style.BRIGHT}--- Running config_utils.py self-test ---{Style.RESET_ALL}")

    current_file_path = Path(__file__).resolve()
    project_root_for_test = find_project_root(current_file_path)

    test_base_path = project_root_for_test / "configs" / "config.base.yaml"
    test_override_path_arg = project_root_for_test / "configs" / "overrides" / f"{platform.node().split('.')[0].lower()}.yaml"
    if not test_override_path_arg.exists():
        test_override_path_arg = None

    print(f"Project root (detected): {project_root_for_test}")
    print(f"Base config path for test: {test_base_path}")
    print(f"Explicit override path for test: {test_override_path_arg}")
    print(f"Overrides default dir: {project_root_for_test / 'configs' / 'overrides'}")

    if not test_base_path.exists():
        print(colour_util(f"Test SKIPPED: Base config '{test_base_path}' not found.", Fore.RED))
    else:
        try:
            print(colour_util("\n--- Testing config processing (verbose_logging=True) ---", Fore.CYAN))
            processed_config_result = generate_processed_config(
                base_config_path_arg=test_base_path,
                override_config_path_arg=test_override_path_arg,
                script_dir_for_overrides=project_root_for_test,  # accepts both old/new
                verbose_logging=True
            )
            print(f"\n{Style.BRIGHT}--- Processed Configuration Output (YAML) ---{Style.RESET_ALL}")
            print(yaml.dump(processed_config_result, indent=2, sort_keys=False, allow_unicode=True, Dumper=yaml.SafeDumper))

            print(f"\n{Style.BRIGHT}--- Basic Validation of Processed Config ---{Style.RESET_ALL}")
            if "models" in processed_config_result and isinstance(processed_config_result["models"], dict):
                num_models = len(processed_config_result['models'])
                print(colour_util(f"Found 'models' section with {num_models} entries.", Fore.GREEN))

                # only peek at the first model for brevity
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
                    break
            else:
                print(colour_util("No 'models' section found or not a dict in processed config.", Fore.RED))

            removed_test_key = "COMMON_FLAGS"
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
