#!/usr/bin/env python3
"""
Cross-platform watcher / restarter for *llama-swap*.

    • Reads   ./config.base.yaml (or specified path)
    • Merges  an override file:
        1. The file passed via --override <path>, if given, **or**
        2. ./overrides/<hostname>.yaml (relative to script), if present
    • Writes  ./config.effective.yaml (relative to script)
    • Starts  llama-swap --config config.effective.yaml
    • Restarts automatically when the base file changes
    • Logs stdout/stderr into ./logs/<tag>_YYYY-MM-DD_HH-MM-SS.log

Uses shared config_utils module for configuration processing including
resolving relative paths in config against the base config's directory.

Requirements
------------
`pip install watchdog~=4.0 colorama~=0.4 pyyaml`
"""

from __future__ import annotations

import argparse
import os
import platform
import signal
import subprocess
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO
# import shlex # Not strictly needed for this change, but good for robust cmd joining if args have spaces

# --- Dependencies ---
import yaml # Used only for dumping the final config
from colorama import Fore, Style, init # Used directly for logging
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# --- Import from shared config utility ---
try:
    current_script_dir_setup = Path(__file__).resolve().parent
    parent_dir_setup = current_script_dir_setup.parent
    if str(parent_dir_setup) not in sys.path:
        sys.path.insert(0, str(parent_dir_setup))
    from utils.config_utils import generate_processed_config, colour_util
    colour = colour_util
except ImportError as e:
    print(f"Error: Could not import from config_utils. Make sure config_utils.py is accessible.")
    print(f"PYTHONPATH: {sys.path}")
    print(f"Details: {e}")
    sys.exit(1)

# Initialize Colorama
try:
    init(autoreset=True)
except Exception:
    print("Warning: colorama init failed or not installed. Proceeding without colors.")

# --- Constants ---
DEFAULT_LOGS_TO_KEEP = 10
PROCESS_TERMINATE_TIMEOUT_SECONDS = 10

MODEL_CONFIG_META_KEYS_SWAP_WATCH = {
    "aliases",
    "sampling", # If 'sampling' is a dict, its contents are expanded into CLI args. The key 'sampling' itself is meta.
    "_name_for_log",
    "generated_cmd_str", # We build our own 'cmd' string, so this is meta if present from config_utils.
    "cmd", # This refers to the original 'cmd' dictionary from config, which we process.
    "hf_tokenizer_for_model",
    "supports_no_think_toggle"
}

# Keys for which the value "auto" (case-insensitive) means the flag should be omitted
# from the llama-server command line, allowing llama-server to use its internal defaults.
AUTO_OMIT_KEYS = {
    "gpu-layers", "n-gpu-layers", # llama-server uses --n-gpu-layers or --gpu-layers
    "threads", "threads-batch"    # llama-server uses --threads
}

# -----------------------------------------------------------------------------
# Utility Functions (Specific to this script)
# -----------------------------------------------------------------------------

import json, sys, yaml  # ensure these are imported

def _append_flag_args(
    server_args_list, key, value, model_name, verbose_script_logging
) -> None:
    normalized_key = key.replace('_', '-')
    cli_flag = f"--{normalized_key}"

    # Omit when marked as 'auto'
    if normalized_key in AUTO_OMIT_KEYS and isinstance(value, str) and value.lower() == "auto":
        if verbose_script_logging:
            print(colour(f"Verbose: Model '{model_name}' - Omitted '{cli_flag}' because value is 'auto'.", Fore.MAGENTA))
        return

    # SPECIAL CASE: any flag ending with '-kwargs' → normalize to compact JSON and quote for the OS
    if normalized_key.endswith("-kwargs"):
        def _to_compact_json(val):
            if isinstance(val, (dict, list, int, float, bool)) or val is None:
                return json.dumps(val, separators=(',', ':'))
            if isinstance(val, str):
                s = val.strip()
                # try JSON
                try:
                    return json.dumps(json.loads(s), separators=(',', ':'))
                except Exception:
                    pass
                # try YAML -> JSON
                try:
                    y = yaml.safe_load(s)
                    if isinstance(y, (dict, list, int, float, bool)) or y is None:
                        return json.dumps(y, separators=(',', ':'))
                except Exception:
                    pass
                # last resort: pass through unchanged
                return s
            return json.dumps(val, separators=(',', ':'))

        compact = _to_compact_json(value)

        # Quote for shell so inner " survive:
        if sys.platform.startswith("win"):
            # windows needs double quotes around the whole thing and \" inside
            compact_quoted = '"' + compact.replace('"', r'\"') + '"'
        else:
            # posix: single quotes preserve inner double quotes
            compact_quoted = "'" + compact + "'"

        server_args_list.extend([cli_flag, compact_quoted])
        if verbose_script_logging:
            print(colour(f"Verbose: Model '{model_name}' - Added JSON for '{cli_flag}': {compact_quoted}", Fore.MAGENTA))
        return

    # Booleans: include flag only if True
    if isinstance(value, bool):
        if value:
            server_args_list.append(cli_flag)
        return

    # Lists: repeat the flag for each item
    if isinstance(value, list):
        if not value:
            return
        for item in value:
            if item is None:
                continue
            if isinstance(item, bool):
                if item:
                    server_args_list.append(cli_flag)
            else:
                server_args_list.extend([cli_flag, str(item)])
        if verbose_script_logging:
            print(colour(f"Verbose: Model '{model_name}' - Added {len(value)} occurrences of '{cli_flag}' from list.", Fore.MAGENTA))
        return

    # Dicts: flatten to '--key subkey=value' per entry
    if isinstance(value, dict):
        if not value:
            return
        for sub_k, sub_v in value.items():
            if sub_v is None:
                continue
            server_args_list.extend([cli_flag, f"{sub_k}={sub_v}"])
        if verbose_script_logging:
            print(colour(f"Verbose: Model '{model_name}' - Added {len(value)} entries for '{cli_flag}' from dict.", Fore.MAGENTA))
        return

    # Scalars
    if value is not None:
        server_args_list.extend([cli_flag, str(value)])

def prune_logs(log_dir: Path, keep: int = DEFAULT_LOGS_TO_KEEP) -> None:
    """Deletes older log files, keeping the newest `keep` files."""
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        logs: List[Path] = sorted(
            (p for p in log_dir.glob("*.log") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        for old_log in logs[keep:]:
            try:
                old_log.unlink()
            except OSError as exc:
                print(colour(f"[WARN] Could not delete log {old_log.name}: {exc}", Fore.YELLOW))
    except OSError as exc:
        print(colour(f"[WARN] Could not access or create log directory {log_dir}: {exc}", Fore.YELLOW))

# -----------------------------------------------------------------------------
# LlamaSwap Process Runner and File Watcher
# -----------------------------------------------------------------------------

class LlamaRunner:
    """
    Manages the llama-swap process, including starting, stopping, restarting.
    """

    def __init__(self, exe_path: Path, config_path: Path, listen_address: str, verbose: bool):
        self.exe_path = exe_path
        self.config_path = config_path
        self.listen_address = listen_address
        self.verbose_logging = verbose 
        self.child_process: Optional[subprocess.Popen] = None
        self.log_file_handle: Optional[TextIO] = None

    def start(self) -> None:
        """Starts the llama-swap subprocess using the effective config."""
        if self.child_process and self.child_process.poll() is None:
            print(colour("llama-swap process already running.", Fore.YELLOW))
            return

        args = [
            str(self.exe_path),
            "--config", str(self.config_path),
            "--listen", self.listen_address,
        ]
        if self.verbose_logging: 
            args.append("--verbose")

        model_tag = os.getenv("MODEL_NAME", "llama_swap")
        log_dir = Path(__file__).resolve().parent / "logs"
        prune_logs(log_dir) 
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file_path = log_dir / f"{model_tag}_{timestamp}.log"

        try:
            self.log_file_handle = log_file_path.open("w", encoding="utf-8")
            print(colour(f"Starting llama-swap. Logging to: {log_file_path}", Fore.GREEN))
            self.child_process = subprocess.Popen(
                args, stdout=self.log_file_handle, stderr=subprocess.STDOUT,
                text=True, bufsize=1
            )
        except OSError as e:
            print(colour(f"Error starting llama-swap process: {e}", Fore.RED))
            if self.log_file_handle:
                self.log_file_handle.close(); self.log_file_handle = None
        except Exception as e:
             print(colour(f"Unexpected error starting llama-swap: {e}", Fore.RED))
             if self.log_file_handle:
                self.log_file_handle.close(); self.log_file_handle = None


    def stop(self) -> None:
        """Stops the llama-swap subprocess gracefully, with a force-kill fallback."""
        if self.child_process and self.child_process.poll() is None:
            pid = self.child_process.pid
            if self.verbose_logging: 
                print(colour(f"Stopping llama-swap process (PID {pid})...", Fore.CYAN))
            try:
                self.child_process.terminate()
                self.child_process.wait(timeout=PROCESS_TERMINATE_TIMEOUT_SECONDS)
                if self.verbose_logging: 
                    print(colour(f"llama-swap process (PID {pid}) terminated.", Fore.GREEN))
            except subprocess.TimeoutExpired:
                print(colour(f"llama-swap (PID {pid}) did not terminate gracefully. Force-killing...", Fore.RED))
                try:
                    self.child_process.kill()
                    self.child_process.wait(timeout=5)
                except Exception as e_kill:
                    print(colour(f"Error during force kill of PID {pid}: {e_kill}", Fore.RED))
            except Exception as e_term:
                print(colour(f"Error during termination of PID {pid} (process might have already exited): {e_term}", Fore.YELLOW))
            finally:
                 self.child_process = None

        if self.log_file_handle:
            try: self.log_file_handle.close()
            except Exception: pass
            finally: self.log_file_handle = None

    def restart(self) -> None:
        """Stops and then starts the llama-swap subprocess."""
        if self.verbose_logging: 
            print(colour("Restarting llama-swap process...", Fore.YELLOW))
        self.stop()
        time.sleep(0.5)
        self.start() 

    def shutdown_handler(self, signum, frame) -> None:
        """Handles SIGINT/SIGTERM signals for graceful shutdown."""
        print(colour(f"\nSignal {signal.Signals(signum).name} received. Shutting down...", Fore.YELLOW))
        self.stop()
        sys.exit(0)

# -----------------------------------------------------------------------------
# Configuration Management (using config_utils)
# -----------------------------------------------------------------------------

def find_llama_swap_executable(provided_exe_path: Optional[Path], script_dir: Path) -> Path:
    """
    Finds the llama-swap executable. Handles platform differences.
    """
    if provided_exe_path:
        resolved_provided = provided_exe_path.expanduser().resolve()
        if resolved_provided.is_file():
            return resolved_provided
        else:
            raise FileNotFoundError(f"Provided llama-swap executable not found: {resolved_provided}")

    platform_executables = {
        "win32": Path("llama-swap") / "llama-swap.exe",
        "cygwin": Path("llama-swap") / "llama-swap.exe",
        "linux": Path("llama-swap") / "llama-swap-linux-amd64",
        "darwin": Path("llama-swap") / "llama-swap",
    }
    current_platform = sys.platform
    for platform_key, relative_exe_path in platform_executables.items():
        if current_platform.startswith(platform_key):
            candidate_path = (script_dir / relative_exe_path).resolve()
            if candidate_path.is_file():
                return candidate_path

    raise FileNotFoundError(
        "llama-swap executable not found. Check 'llama-swap' subdirectory or use --exe."
    )


def process_and_write_effective_config(
    base_config_path: Path,
    override_config_path_arg: Optional[Path],
    script_dir: Path,
    verbose_script_logging: bool 
) -> Path:
    """
    Uses config_utils to generate processed config, formats it for llama-swap by
    rebuilding the command string carefully, and writes it to config.effective.yaml.
    """
    try:
        processed_dict = generate_processed_config(
            base_config_path,
            override_config_path_arg,
            script_dir_for_overrides=script_dir,
            verbose_logging=True # Keep true for generate_processed_config's own logs
        )
    except (FileNotFoundError, ValueError, OSError, Exception) as e:
        print(colour(f"Error processing configuration: {e}", Fore.RED))
        raise RuntimeError(f"Configuration processing failed: {e}") from e

    output_for_llama_swap = {"models": {}}
    models_data_from_util = processed_dict.get("models", {})

    if isinstance(models_data_from_util, dict):
        for model_name, model_config_from_utils in models_data_from_util.items():
            if not isinstance(model_config_from_utils, dict):
                print(colour(f"Skipping invalid model data for '{model_name}' (not a dictionary) in effective config.", Fore.YELLOW))
                continue

            final_model_entry_for_llama_swap = {}
            server_args_list = [] 

            original_cmd_dict = model_config_from_utils.get("cmd", {})
            if not isinstance(original_cmd_dict, dict):
                original_cmd_dict = {}
                if verbose_script_logging:
                    print(colour(f"Verbose: Model '{model_name}' has 'cmd' field that is not a dictionary or is missing. "
                                 f"Value: {model_config_from_utils.get('cmd')}", Fore.MAGENTA))
            
            server_executable_path = original_cmd_dict.get("bin")
            if not server_executable_path or not isinstance(server_executable_path, str):
                print(colour(f"Error: Model '{model_name}' - 'bin' (executable path) is missing or invalid in its 'cmd' dictionary. "
                             "Cannot construct command string for llama-swap.", Fore.RED))
            else:
                server_args_list.append(str(server_executable_path)) # Part 1: Executable

                # --- Part 2: Add arguments from the 'cmd' dictionary (excluding 'bin') ---
                for key, value in original_cmd_dict.items():
                    if key == "bin":
                        continue
                    _append_flag_args(server_args_list, key, value, model_name, verbose_script_logging)

                # --- Part 3: Add arguments from the root of the model's configuration ---
                for key, value in model_config_from_utils.items():
                    if key == "cmd" or key in MODEL_CONFIG_META_KEYS_SWAP_WATCH:
                        continue
                    _append_flag_args(server_args_list, key, value, model_name, verbose_script_logging)
                                
                # --- Part 4: Special handling for 'sampling' dictionary from root ---
                sampling_config_at_root = model_config_from_utils.get("sampling")
                if isinstance(sampling_config_at_root, dict): # 'sampling' is in META_KEYS, so this is fine
                    for skey, sval in sampling_config_at_root.items():
                        s_cli_flag = f"--{skey.replace('_', '-')}"
                        if sval is not None: # Assuming sampling values shouldn't be "auto" omitted
                            server_args_list.extend([s_cli_flag, str(sval)])
                
                # --- Part 5: Construct final command string ---
                final_cmd_str = " ".join(map(str, server_args_list))
                final_model_entry_for_llama_swap["cmd"] = final_cmd_str
                if verbose_script_logging:
                     print(colour(f"Verbose: Model '{model_name}' effective cmd: {final_cmd_str}", Fore.MAGENTA))

            # --- Part 6: Copy other top-level model parameters to the llama-swap config ---
            for key, value in model_config_from_utils.items():
                if key != "cmd" and key not in MODEL_CONFIG_META_KEYS_SWAP_WATCH:
                    final_model_entry_for_llama_swap[key] = value
            
            output_for_llama_swap["models"][model_name] = final_model_entry_for_llama_swap
    
    for key, value in processed_dict.items():
        if key != "models": 
            output_for_llama_swap[key] = value

    effective_config_file_path = script_dir / "config.effective.yaml"
    try:
        yaml_output = yaml.dump(
            output_for_llama_swap,
            Dumper=yaml.Dumper, 
            sort_keys=False,
            allow_unicode=True,
            width=120 
        )
        effective_config_file_path.write_text(yaml_output, encoding="utf-8")
        print(colour(f"Wrote effective configuration for llama-swap: {effective_config_file_path}", Fore.GREEN))
    except Exception as e:
        print(colour(f"Error writing effective configuration file {effective_config_file_path}: {e}", Fore.RED))
        raise RuntimeError(f"Failed to write effective configuration: {e}") from e

    return effective_config_file_path

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def main() -> None:
    """Main function: Parses args, sets up watcher, manages llama-swap process."""
    script_dir = Path(__file__).resolve().parent
    default_base_config_path = (script_dir / "config.base.yaml").resolve()

    parser = argparse.ArgumentParser(
        description="Watches base config and restarts llama-swap with merged configuration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "config", nargs="?", type=Path, default=default_base_config_path,
        help="Base configuration YAML file path."
    )
    parser.add_argument("--exe", type=Path, help="Path to the llama-swap executable.")
    parser.add_argument("--listen", default=":8080", help="Listen address for llama-swap.")
    parser.add_argument("-o", "--override", type=Path, help="Override YAML file path (disables hostname lookup).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging for this script and llama-swap.")

    args = parser.parse_args()

    resolved_base_config = args.config.resolve()
    if not resolved_base_config.is_file():
        print(colour(f"Base configuration file not found: {resolved_base_config}", Fore.RED))
        sys.exit(1)

    try:
        llama_swap_exe_path = find_llama_swap_executable(args.exe, script_dir)
        print(colour(f"Using llama-swap executable: {llama_swap_exe_path}", Fore.BLUE))
    except FileNotFoundError as e:
        print(colour(str(e), Fore.RED))
        sys.exit(1)

    try:
        effective_config_path = process_and_write_effective_config(
            resolved_base_config,
            args.override,
            script_dir,
            args.verbose 
        )
    except (FileNotFoundError, ValueError, RuntimeError, OSError, Exception) as e:
        # Error already printed in func, this is for exit / verbose traceback
        if args.verbose: 
            import traceback
            traceback.print_exc()
        sys.exit(1)

    runner = LlamaRunner(
        llama_swap_exe_path,
        effective_config_path,
        args.listen,
        args.verbose 
    )

    signal.signal(signal.SIGINT, runner.shutdown_handler)
    signal.signal(signal.SIGTERM, runner.shutdown_handler)

    runner.start() 

    config_observer = Observer()

    class BaseConfigChangeHandler(FileSystemEventHandler):
        def __init__(self, runner_instance: LlamaRunner, base_path: Path, override_path: Optional[Path], script_directory: Path, verbose_logging: bool): 
            self.runner = runner_instance
            self.base_config_path = base_path
            self.override_config_path = override_path
            self.script_dir = script_directory
            self.verbose_logging = verbose_logging 
            super().__init__()

        def on_modified(self, event):
            if event.src_path == str(self.base_config_path):
                if self.verbose_logging: 
                    print(colour(f"Base configuration file {self.base_config_path.name} modified.", Fore.YELLOW))
                    print(colour("Re-processing configurations and restarting llama-swap...", Fore.YELLOW))
                try:
                    new_effective_config_path = process_and_write_effective_config(
                        self.base_config_path,
                        self.override_config_path,
                        self.script_dir,
                        self.verbose_logging 
                    )
                    self.runner.config_path = new_effective_config_path
                    self.runner.restart() 
                except (FileNotFoundError, ValueError, RuntimeError, OSError, Exception) as e:
                    print(colour(f"Error during automatic re-configuration: {e}", Fore.RED))
                    print(colour("llama-swap may continue with the old configuration if running.", Fore.YELLOW))
                    if self.verbose_logging: 
                        import traceback
                        traceback.print_exc()

    event_handler = BaseConfigChangeHandler(runner, resolved_base_config, args.override, script_dir, args.verbose) 
    try:
        watch_path = str(resolved_base_config.parent)
        config_observer.schedule(event_handler, watch_path, recursive=False)
        config_observer.start()
        print(colour(f"Watching base configuration {resolved_base_config.name} in {watch_path} for changes.", Fore.BLUE))
    except Exception as e:
         print(colour(f"Failed to start configuration file watcher: {e}", Fore.RED))
         print(colour("Proceeding without automatic config reload on change.", Fore.YELLOW))

    try:
        port_str = args.listen.split(':')[-1]
        if not port_str or not port_str.isdigit(): port_str = "8080" 
    except Exception: port_str = "8080" 
    print(colour(f"llama-swap setup complete. If using Open WebUI, try: http://localhost:{port_str}/v1", Fore.GREEN))

    try:
        while True:
            time.sleep(1)
            if runner.child_process and runner.child_process.poll() is not None:
                exit_code = runner.child_process.returncode
                print(colour(f"llama-swap process exited unexpectedly (code {exit_code}). Restarting...", Fore.YELLOW))
                runner.restart() 
    except KeyboardInterrupt:
         if args.verbose: 
            print(colour("\nKeyboardInterrupt received in main loop.", Fore.YELLOW))
    finally:
        print(colour("\nShutting down watcher and llama-swap...", Fore.BLUE))
        try:
            if config_observer.is_alive():
                config_observer.stop()
                config_observer.join(timeout=2)
        except Exception as e_obs:
             print(colour(f"Error stopping file observer: {e_obs}", Fore.YELLOW))
        runner.stop() 
        print(colour("Shutdown complete.", Fore.BLUE))


if __name__ == "__main__":
    main()