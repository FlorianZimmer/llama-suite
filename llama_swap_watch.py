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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

# --- Dependencies ---
import yaml # Used only for dumping the final config
from colorama import Fore, Style, init # Used directly for logging
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# --- Import from shared config utility ---
# Assume config_utils.py is in parent directory or accessible via PYTHONPATH
try:
    current_script_dir_setup = Path(__file__).resolve().parent
    parent_dir_setup = current_script_dir_setup.parent
    if str(parent_dir_setup) not in sys.path:
        sys.path.insert(0, str(parent_dir_setup))
    # Import the necessary function from the utility module
    from utils.config_utils import generate_processed_config, colour_util
    # We can use colour_util as our local 'colour' if desired
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
    # Define dummy Fore/Style if needed, or rely on colour_util's handling


# --- Constants ---
DEFAULT_LOGS_TO_KEEP = 10
PROCESS_TERMINATE_TIMEOUT_SECONDS = 10


# -----------------------------------------------------------------------------
# Utility Functions (Specific to this script)
# -----------------------------------------------------------------------------

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
                # This is a warning, should always be shown
                print(colour(f"[WARN] Could not delete log {old_log.name}: {exc}", Fore.YELLOW))
    except OSError as exc:
        # This is a warning, should always be shown
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
        self.verbose_logging = verbose # This controls verbosity of this class's messages AND llama-swap
        self.child_process: Optional[subprocess.Popen] = None
        self.log_file_handle: Optional[TextIO] = None

    def start(self) -> None:
        """Starts the llama-swap subprocess using the effective config."""
        if self.child_process and self.child_process.poll() is None:
            # This is a status warning, should always be shown
            print(colour("llama-swap process already running.", Fore.YELLOW))
            return

        args = [
            str(self.exe_path),
            "--config", str(self.config_path),
            "--listen", self.listen_address,
        ]
        if self.verbose_logging: # Pass verbose to llama-swap executable
            args.append("--verbose")

        model_tag = os.getenv("MODEL_NAME", "llama_swap")
        log_dir = Path(__file__).resolve().parent / "logs"
        prune_logs(log_dir) # prune_logs handles its own warnings
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file_path = log_dir / f"{model_tag}_{timestamp}.log"

        try:
            self.log_file_handle = log_file_path.open("w", encoding="utf-8")
            # Part of the desired standard output
            print(colour(f"Starting llama-swap. Logging to: {log_file_path}", Fore.GREEN))
            self.child_process = subprocess.Popen(
                args, stdout=self.log_file_handle, stderr=subprocess.STDOUT,
                text=True, bufsize=1
            )
        except OSError as e:
            # Error, always show
            print(colour(f"Error starting llama-swap process: {e}", Fore.RED))
            if self.log_file_handle:
                self.log_file_handle.close(); self.log_file_handle = None
        except Exception as e:
             # Error, always show
             print(colour(f"Unexpected error starting llama-swap: {e}", Fore.RED))
             if self.log_file_handle:
                self.log_file_handle.close(); self.log_file_handle = None


    def stop(self) -> None:
        """Stops the llama-swap subprocess gracefully, with a force-kill fallback."""
        if self.child_process and self.child_process.poll() is None:
            pid = self.child_process.pid
            if self.verbose_logging: # Conditional output
                print(colour(f"Stopping llama-swap process (PID {pid})...", Fore.CYAN))
            try:
                self.child_process.terminate()
                self.child_process.wait(timeout=PROCESS_TERMINATE_TIMEOUT_SECONDS)
                if self.verbose_logging: # Conditional output (successful stop)
                    print(colour(f"llama-swap process (PID {pid}) terminated.", Fore.GREEN))
            except subprocess.TimeoutExpired:
                # Warning/Error, always show
                print(colour(f"llama-swap (PID {pid}) did not terminate gracefully. Force-killing...", Fore.RED))
                try:
                    self.child_process.kill()
                    self.child_process.wait(timeout=5)
                except Exception as e_kill:
                    # Error, always show
                    print(colour(f"Error during force kill of PID {pid}: {e_kill}", Fore.RED))
            except Exception as e_term:
                # Warning, always show
                print(colour(f"Error during termination of PID {pid} (process might have already exited): {e_term}", Fore.YELLOW))
            finally:
                 self.child_process = None

        if self.log_file_handle:
            try: self.log_file_handle.close()
            except Exception: pass
            finally: self.log_file_handle = None

    def restart(self) -> None:
        """Stops and then starts the llama-swap subprocess."""
        if self.verbose_logging: # Conditional output
            print(colour("Restarting llama-swap process...", Fore.YELLOW))
        self.stop()
        time.sleep(0.5)
        self.start() # start() will print its "Starting llama-swap. Logging to..." message

    def shutdown_handler(self, signum, frame) -> None:
        """Handles SIGINT/SIGTERM signals for graceful shutdown."""
        # Important lifecycle message, always show
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
    verbose_script_logging: bool # For config_utils verbosity
) -> Path:
    """
    Uses config_utils to generate processed config, formats it for llama-swap,
    and writes it to config.effective.yaml.
    """
    try:
        # generate_processed_config is expected to print "Merged override configuration: ..."
        # if an override is used. This line is part of the desired standard output.
        # If verbose_script_logging=False suppresses this, config_utils needs adjustment
        # or this script would need to print it manually.
        # Assuming verbose_logging=True in config_utils prints the "Merged..." line and
        # any OTHER detailed logs from config_utils are what verbose_script_logging controls.
        # For now, we pass True to ensure "Merged..." line appears as in example.
        # If this makes config_utils too noisy without -v for other messages,
        # config_utils verbosity needs refinement.
        processed_dict = generate_processed_config(
            base_config_path,
            override_config_path_arg,
            script_dir_for_overrides=script_dir,
            verbose_logging=True # Assuming this is needed for "Merged override..." line
                                 # and doesn't add other non-verbose undesired output.
                                 # If generate_processed_config has more fine-grained control
                                 # or prints "Merged..." line unconditionally, then
                                 # verbose_script_logging could be passed here instead.
        )
    except (FileNotFoundError, ValueError, OSError, Exception) as e:
        # Error, always show
        print(colour(f"Error processing configuration: {e}", Fore.RED))
        raise RuntimeError(f"Configuration processing failed: {e}") from e

    output_for_llama_swap = {"models": {}}
    models_data_from_util = processed_dict.get("models", {})

    if isinstance(models_data_from_util, dict):
        for model_name, model_data in models_data_from_util.items():
            if isinstance(model_data, dict) and "generated_cmd_str" in model_data:
                llama_swap_model_entry = {
                    key: val for key, val in model_data.items()
                    if key not in ["cmd", "generated_cmd_str", "sampling", "_name_for_log"]
                }
                llama_swap_model_entry["cmd"] = model_data["generated_cmd_str"]
                output_for_llama_swap["models"][model_name] = llama_swap_model_entry
            else:
                 if isinstance(model_data, dict):
                      output_for_llama_swap["models"][model_name] = model_data
                 else:
                      # Warning, always show
                      print(colour(f"Skipping invalid model data for '{model_name}' in effective config.", Fore.YELLOW))

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
            width=100
        )
        effective_config_file_path.write_text(yaml_output, encoding="utf-8")
        # Part of the desired standard output
        print(colour(f"Wrote effective configuration for llama-swap: {effective_config_file_path}", Fore.GREEN))
    except Exception as e:
        # Error, always show
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
        # Error, always show
        print(colour(f"Base configuration file not found: {resolved_base_config}", Fore.RED))
        sys.exit(1)

    try:
        llama_swap_exe_path = find_llama_swap_executable(args.exe, script_dir)
        # Part of the desired standard output
        print(colour(f"Using llama-swap executable: {llama_swap_exe_path}", Fore.BLUE))
    except FileNotFoundError as e:
        # Error, always show
        print(colour(str(e), Fore.RED))
        sys.exit(1)

    try:
        effective_config_path = process_and_write_effective_config(
            resolved_base_config,
            args.override,
            script_dir,
            args.verbose # Pass main verbose flag for config_utils verbosity
        )
    except (FileNotFoundError, ValueError, RuntimeError, OSError, Exception) as e:
        # Error, always show (already printed in func, this is for exit)
        # print(colour(f"Initial configuration processing failed: {e}", Fore.RED)) # Redundant if func prints
        if args.verbose: # Conditional traceback
            import traceback
            traceback.print_exc()
        sys.exit(1)

    runner = LlamaRunner(
        llama_swap_exe_path,
        effective_config_path,
        args.listen,
        args.verbose # Pass main verbose flag to runner
    )

    signal.signal(signal.SIGINT, runner.shutdown_handler)
    signal.signal(signal.SIGTERM, runner.shutdown_handler)

    runner.start() # Prints "Starting llama-swap..."

    config_observer = Observer()

    class BaseConfigChangeHandler(FileSystemEventHandler):
        def __init__(self, runner_instance: LlamaRunner, base_path: Path, override_path: Optional[Path], script_directory: Path, verbose_logging: bool): # Added verbose_logging
            self.runner = runner_instance
            self.base_config_path = base_path
            self.override_config_path = override_path
            self.script_dir = script_directory
            self.verbose_logging = verbose_logging # Store it
            super().__init__()

        def on_modified(self, event):
            if event.src_path == str(self.base_config_path):
                if self.verbose_logging: # Conditional output
                    print(colour(f"Base configuration file {self.base_config_path.name} modified.", Fore.YELLOW))
                    print(colour("Re-processing configurations and restarting llama-swap...", Fore.YELLOW))
                try:
                    new_effective_config_path = process_and_write_effective_config(
                        self.base_config_path,
                        self.override_config_path,
                        self.script_dir,
                        self.verbose_logging # Pass verbose flag down
                    )
                    self.runner.config_path = new_effective_config_path
                    self.runner.restart() # restart() handles its own verbose messages
                except (FileNotFoundError, ValueError, RuntimeError, OSError, Exception) as e:
                    # Error, always show
                    print(colour(f"Error during automatic re-configuration: {e}", Fore.RED))
                    # Warning, always show
                    print(colour("llama-swap may continue with the old configuration if running.", Fore.YELLOW))
                    if self.verbose_logging: # Conditional traceback
                        import traceback
                        traceback.print_exc()

    event_handler = BaseConfigChangeHandler(runner, resolved_base_config, args.override, script_dir, args.verbose) # Pass args.verbose
    try:
        watch_path = str(resolved_base_config.parent)
        config_observer.schedule(event_handler, watch_path, recursive=False)
        config_observer.start()
        # Part of the desired standard output
        print(colour(f"Watching base configuration {resolved_base_config.name} in {watch_path} for changes.", Fore.BLUE))
    except Exception as e:
         # Error, always show
         print(colour(f"Failed to start configuration file watcher: {e}", Fore.RED))
         # Warning, always show
         print(colour("Proceeding without automatic config reload on change.", Fore.YELLOW))

    try:
        port_str = args.listen.split(':')[-1]
        if not port_str or not port_str.isdigit(): port_str = "8080" # Default if parse fails
    except Exception: port_str = "8080" # Default on any error
    # Part of the desired standard output
    print(colour(f"llama-swap setup complete. If using Open WebUI, try: http://localhost:{port_str}/v1", Fore.GREEN))

    try:
        while True:
            time.sleep(1)
            if runner.child_process and runner.child_process.poll() is not None:
                exit_code = runner.child_process.returncode
                # Important status warning, always show
                print(colour(f"llama-swap process exited unexpectedly (code {exit_code}). Restarting...", Fore.YELLOW))
                runner.restart() # restart() handles its own verbose messages
    except KeyboardInterrupt:
         # Message already printed by shutdown_handler, this is a fallback if signal somehow missed.
         if args.verbose: # Conditional, as shutdown_handler already prints a message
            print(colour("\nKeyboardInterrupt received in main loop.", Fore.YELLOW))
    finally:
        # Important lifecycle message, always show
        print(colour("\nShutting down watcher and llama-swap...", Fore.BLUE))
        try:
            if config_observer.is_alive():
                config_observer.stop()
                config_observer.join(timeout=2)
        except Exception as e_obs:
             # Warning, always show
             print(colour(f"Error stopping file observer: {e_obs}", Fore.YELLOW))
        runner.stop() # stop() handles its own verbose messages
        # Important lifecycle message, always show
        print(colour("Shutdown complete.", Fore.BLUE))


if __name__ == "__main__":
    main()