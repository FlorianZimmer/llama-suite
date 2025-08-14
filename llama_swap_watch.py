#!/usr/bin/env python3
"""
Cross-platform watcher / restarter for *llama-swap*.

- Reads   ./config.base.yaml (or path from CLI)
- Merges  an override file:
    1) --override <path>, if provided, else
    2) ./overrides/<hostname>.yaml (handled in config_utils)
- Writes  ./config.effective.yaml
- Starts  llama-swap --config config.effective.yaml
- Restarts automatically on config change (base and override if provided)
- Logs stdout/stderr into ./logs/<tag>_YYYY-MM-DD_HH-MM-SS.log

Requires:
    pip install watchdog~=4.0 colorama~=0.4 pyyaml
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml
from colorama import Fore, Style, init
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

# --- Import from shared config utility ---
try:
    CURRENT_DIR = Path(__file__).resolve().parent
    PARENT_DIR = CURRENT_DIR.parent
    if str(PARENT_DIR) not in sys.path:
        sys.path.insert(0, str(PARENT_DIR))
    from utils.config_utils import generate_processed_config, colour_util
    colour = colour_util
except ImportError as e:
    print(f"Error: Could not import from config_utils. Ensure utils/config_utils.py is on PYTHONPATH.")
    print(f"PYTHONPATH: {sys.path}")
    print(f"Details: {e}")
    sys.exit(1)

# Initialize Colorama (non-fatal if unavailable)
try:
    init(autoreset=True)
except Exception:
    print("Warning: colorama init failed or not installed. Proceeding without colors.")

# --- Constants ---
DEFAULT_LOGS_TO_KEEP = 10
PROCESS_TERMINATE_TIMEOUT_SECONDS = 10

# Keys present in model configs that are meta-only for the watcher (not CLI flags)
MODEL_CONFIG_META_KEYS = {
    "aliases",
    "sampling",
    "_name_for_log",
    "generated_cmd_str",
    "cmd",
    "hf_tokenizer_for_model",
    "supports_no_think_toggle",
    "enabled",
    "skip",
    "_skip",
    "disabled",
}

# Keys for which value "auto" (case-insensitive) means omit the flag
AUTO_OMIT_KEYS = {
    "gpu-layers", "n-gpu-layers",
    "threads", "threads-batch",
}

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _parse_filter_list(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    if isinstance(val, Iterable):
        return [x for x in val if isinstance(x, str)]
    return []


def _collect_model_filters(cfg: Dict[str, Any]) -> tuple[List[str], List[str]]:
    """
    Collect include-only and exclude patterns from top-level config.
    Supported keys:
      only_models | models_only | include_models_only
      exclude_models | models_exclude | disabled_models | skip_models
    """
    only: List[str] = []
    exclude: List[str] = []
    for k in ("only_models", "models_only", "include_models_only"):
        only += _parse_filter_list(cfg.get(k))
    for k in ("exclude_models", "models_exclude", "disabled_models", "skip_models"):
        exclude += _parse_filter_list(cfg.get(k))
    return only, exclude


def _name_matches(name: str, pattern: str) -> bool:
    """Glob or regex (prefix 're:' for regex)."""
    if pattern.startswith("re:"):
        try:
            return re.search(pattern[3:], name) is not None
        except re.error:
            return False
    return fnmatch.fnmatch(name, pattern)


def _filter_model_names(all_names: Iterable[str], only: List[str], exclude: List[str]) -> List[str]:
    keep = set(all_names)
    if only:
        keep = {n for n in keep if any(_name_matches(n, p) for p in only)}
    if exclude:
        keep = {n for n in keep if not any(_name_matches(n, p) for p in exclude)}
    return sorted(keep)


def _compact_json(val: Any) -> str:
    """Return compact JSON for dict/list/bool/number/None or best-effort parse of string."""
    if isinstance(val, (dict, list, int, float, bool)) or val is None:
        return json.dumps(val, separators=(",", ":"))
    if isinstance(val, str):
        s = val.strip()
        # Try JSON
        try:
            return json.dumps(json.loads(s), separators=(",", ":"))
        except Exception:
            pass
        # Try YAML -> JSON
        try:
            y = yaml.safe_load(s)
            if isinstance(y, (dict, list, int, float, bool)) or y is None:
                return json.dumps(y, separators=(",", ":"))
        except Exception:
            pass
        return s  # passthrough
    return json.dumps(val, separators=(",", ":"))


def _needs_quote(token: str) -> bool:
    return any(ch.isspace() for ch in token) or any(ch in token for ch in r"&|^<>();!\"'`$*[]{}?=")


def _quote_posix(token: str) -> str:
    if not _needs_quote(token):
        return token
    # wrap in single quotes; escape internal single quotes via: ' '"'"' '
    return "'" + token.replace("'", "'\"'\"'") + "'"


def _quote_windows(token: str) -> str:
    if not _needs_quote(token):
        return token
    # Basic quoting: wrap in double quotes and escape inner double quotes
    # (Windows CreateProcess/CommandLineToArgvW style)
    token = token.replace('"', r'\"')
    return f'"{token}"'


def shell_quote(token: str) -> str:
    if sys.platform.startswith("win"):
        return _quote_windows(token)
    return _quote_posix(token)


def append_flag_args(
    args_list: List[str],
    key: str,
    value: Any,
) -> None:
    """
    Append CLI flags for key/value into args_list, normalizing keys and handling type-specific rules.
    Quoting is applied later when we build the final string.
    """
    normalized_key = key.replace("_", "-")
    cli_flag = f"--{normalized_key}"

    # Omit flags explicitly set to "auto" for specific keys
    if normalized_key in AUTO_OMIT_KEYS and isinstance(value, str) and value.lower() == "auto":
        return

    # --*-kwargs: normalize to compact JSON
    if normalized_key.endswith("-kwargs"):
        compact = _compact_json(value)
        args_list.extend([cli_flag, compact])
        return

    # Booleans: include flag only if True
    if isinstance(value, bool):
        if value:
            args_list.append(cli_flag)
        return

    # Lists: repeat the flag for each item
    if isinstance(value, list):
        for item in value:
            if item is None:
                continue
            if isinstance(item, bool):
                if item:
                    args_list.append(cli_flag)
            else:
                args_list.extend([cli_flag, str(item)])
        return

    # Dicts: flatten to '--key subkey=value'
    if isinstance(value, dict):
        for sub_k, sub_v in value.items():
            if sub_v is None:
                continue
            args_list.extend([cli_flag, f"{sub_k}={sub_v}"])
        return

    # Scalars
    if value is not None:
        args_list.extend([cli_flag, str(value)])


def prune_logs(log_dir: Path, keep: int = DEFAULT_LOGS_TO_KEEP) -> None:
    """Delete older .log files, keeping newest `keep` files."""
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        logs = sorted((p for p in log_dir.glob("*.log") if p.is_file()),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        for old in logs[keep:]:
            try:
                old.unlink()
            except OSError as exc:
                print(colour(f"[WARN] Could not delete log {old.name}: {exc}", Fore.YELLOW))
    except OSError as exc:
        print(colour(f"[WARN] Could not access or create log directory {log_dir}: {exc}", Fore.YELLOW))


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------

@dataclass
class LlamaRunner:
    exe_path: Path
    config_path: Path
    listen_address: str
    verbose: bool

    child_process: Optional[subprocess.Popen] = None
    log_file_handle: Optional[Any] = None

    _lock: threading.Lock = threading.Lock()
    _last_restart_ts: float = 0.0
    _restart_min_interval_s: float = 0.6

    def start(self) -> None:
        """Start llama-swap subprocess."""
        if self.child_process and self.child_process.poll() is None:
            if self.verbose:
                print(colour("llama-swap process already running.", Fore.YELLOW))
            return

        args = [
            str(self.exe_path),
            "--config", str(self.config_path),
            "--listen", self.listen_address,
        ]

        model_tag = os.getenv("MODEL_NAME", "llama_swap")
        log_dir = CURRENT_DIR / "logs"
        prune_logs(log_dir)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file_path = log_dir / f"{model_tag}_{timestamp}.log"

        try:
            self.log_file_handle = log_file_path.open("w", encoding="utf-8")
            if self.verbose:
                print(colour(f"Starting llama-swap. Logging to: {log_file_path}", Fore.GREEN))

            popen_kwargs = dict(
                args=args,
                stdout=self.log_file_handle,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            if sys.platform.startswith("win"):
                popen_kwargs["creationflags"] = 0x00000200  # CREATE_NEW_PROCESS_GROUP
            else:
                popen_kwargs["start_new_session"] = True

            self.child_process = subprocess.Popen(**popen_kwargs)

        except OSError as e:
            print(colour(f"Error starting llama-swap process: {e}", Fore.RED))
            if self.log_file_handle:
                self.log_file_handle.close()
                self.log_file_handle = None
        except Exception as e:
            print(colour(f"Unexpected error starting llama-swap: {e}", Fore.RED))
            if self.log_file_handle:
                self.log_file_handle.close()
                self.log_file_handle = None

    def stop(self) -> None:
        """Stop llama-swap subprocess and its child tree."""
        if self.child_process and self.child_process.poll() is None:
            pid = self.child_process.pid
            if self.verbose:
                print(colour(f"Stopping llama-swap process tree (root PID {pid})...", Fore.CYAN))
            try:
                if sys.platform.startswith("win"):
                    subprocess.run(
                        ["taskkill", "/PID", str(pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
                    )
                    self.child_process.wait(timeout=5)
                else:
                    try:
                        pgid = os.getpgid(pid)
                    except Exception:
                        pgid = None
                    if pgid is not None:
                        os.killpg(pgid, signal.SIGTERM)
                        try:
                            self.child_process.wait(timeout=PROCESS_TERMINATE_TIMEOUT_SECONDS)
                        except subprocess.TimeoutExpired:
                            os.killpg(pgid, signal.SIGKILL)
                    else:
                        self.child_process.terminate()
                        try:
                            self.child_process.wait(timeout=PROCESS_TERMINATE_TIMEOUT_SECONDS)
                        except subprocess.TimeoutExpired:
                            self.child_process.kill()
                if self.verbose:
                    print(colour("llama-swap process tree stopped.", Fore.GREEN))
            except Exception as e_term:
                print(colour(f"Error stopping llama-swap: {e_term}", Fore.YELLOW))
            finally:
                self.child_process = None

        if self.log_file_handle:
            try:
                self.log_file_handle.close()
            except Exception:
                pass
            finally:
                self.log_file_handle = None

    def restart(self) -> None:
        """Serialized + debounced restart."""
        with self._lock:
            now = time.time()
            if now - self._last_restart_ts < self._restart_min_interval_s:
                if self.verbose:
                    print(colour("Restart debounced (too soon after previous).", Fore.YELLOW))
                return
            self._last_restart_ts = now

            if self.verbose:
                print(colour("Restarting llama-swap process...", Fore.YELLOW))
            self.stop()
            time.sleep(0.5)
            self.start()

    def shutdown_handler(self, signum, _frame) -> None:
        print(colour(f"\nSignal {signal.Signals(signum).name} received. Shutting down...", Fore.YELLOW))
        self.stop()
        sys.exit(0)


# -----------------------------------------------------------------------------
# Config processing
# -----------------------------------------------------------------------------

def find_llama_swap_executable(provided_exe: Optional[Path], script_dir: Path) -> Path:
    if provided_exe:
        p = provided_exe.expanduser().resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"Provided llama-swap executable not found: {p}")

    mapping = {
        "win32": Path("llama-swap") / "llama-swap.exe",
        "cygwin": Path("llama-swap") / "llama-swap.exe",
        "linux": Path("llama-swap") / "llama-swap-linux-amd64",
        "darwin": Path("llama-swap") / "llama-swap",
    }
    plat = sys.platform
    for key, rel in mapping.items():
        if plat.startswith(key):
            candidate = (script_dir / rel).resolve()
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        "llama-swap executable not found. Check the 'llama-swap' subdirectory or use --exe."
    )


def process_and_write_effective_config(
    base_config_path: Path,
    override_config_path: Optional[Path],
    script_dir: Path,
    verbose: bool,
) -> Path:
    """
    Use config_utils to generate processed config, turn per-model config into a
    llama-server command string, and write config.effective.yaml for llama-swap.
    """
    try:
        processed = generate_processed_config(
            base_config_path,
            override_config_path,
            script_dir_for_overrides=script_dir,
            verbose_logging=verbose,
        )
    except Exception as e:
        print(colour(f"Error processing configuration: {e}", Fore.RED))
        raise RuntimeError(f"Configuration processing failed: {e}") from e

    output: Dict[str, Any] = {"models": {}}
    models = processed.get("models", {}) or {}
    only_specs, exclude_specs = _collect_model_filters(processed)

    all_model_names = [n for n, v in models.items() if isinstance(v, dict)]
    selected_names = _filter_model_names(all_model_names, only_specs, exclude_specs)

    if verbose:
        if only_specs:
            print(colour(f"only_models filter applied: {only_specs}", Fore.MAGENTA))
        if exclude_specs:
            print(colour(f"exclude_models filter applied: {exclude_specs}", Fore.MAGENTA))
        dropped = sorted(set(all_model_names) - set(selected_names))
        if dropped:
            print(colour(f"Excluded by filters: {dropped}", Fore.MAGENTA))

    for name in selected_names:
        mcfg: Dict[str, Any] = models.get(name, {}) or {}
        # Per-model disable toggles
        if str(mcfg.get("enabled", "true")).lower() in {"false", "0", "no"} \
           or bool(mcfg.get("skip")) or bool(mcfg.get("_skip")) or bool(mcfg.get("disabled")):
            if verbose:
                print(colour(f"Model '{name}' skipped due to disable flag.", Fore.MAGENTA))
            continue

        entry: Dict[str, Any] = {}
        server_args: List[str] = []

        cmd_dict = mcfg.get("cmd", {})
        if not isinstance(cmd_dict, dict):
            cmd_dict = {}
            if verbose:
                print(colour(f"Model '{name}' has non-dict or missing 'cmd'. Value: {mcfg.get('cmd')}", Fore.MAGENTA))

        # Part 1: required executable
        server_bin = cmd_dict.get("bin")
        if not server_bin or not isinstance(server_bin, str):
            print(colour(
                f"Error: Model '{name}' - missing/invalid 'cmd.bin' (llama-server path). Skipping command build.",
                Fore.RED
            ))
        else:
            server_args.append(server_bin)

            # Part 2: cmd.* (except bin)
            for k, v in cmd_dict.items():
                if k == "bin":
                    continue
                append_flag_args(server_args, k, v)

            # Part 3: top-level model keys (except meta/cmd)
            for k, v in mcfg.items():
                if k == "cmd" or k in MODEL_CONFIG_META_KEYS:
                    continue
                append_flag_args(server_args, k, v)

            # Part 4: sampling dict (special)
            sampling = mcfg.get("sampling")
            if isinstance(sampling, dict):
                for sk, sv in sampling.items():
                    append_flag_args(server_args, sk, sv)

            # Part 5: construct final command string with robust quoting
            final_tokens = [shell_quote(str(t)) for t in server_args]
            final_cmd_str = " ".join(final_tokens)
            entry["cmd"] = final_cmd_str
            if verbose:
                print(colour(f"Model '{name}' effective cmd: {final_cmd_str}", Fore.MAGENTA))

        # Part 6: Copy non-meta properties (except 'cmd' already handled)
        for k, v in mcfg.items():
            if k != "cmd" and k not in MODEL_CONFIG_META_KEYS:
                entry[k] = v

        output["models"][name] = entry

    if not output["models"]:
        print(colour("[WARN] No models remain after include/exclude/disable rules.", Fore.YELLOW))

    # Copy other top-level keys
    for k, v in processed.items():
        if k != "models":
            output[k] = v

    effective_path = script_dir / "config.effective.yaml"
    try:
        yaml_text = yaml.dump(output, sort_keys=False, allow_unicode=True, width=120)
        effective_path.write_text(yaml_text, encoding="utf-8")
        print(colour(f"Wrote effective configuration for llama-swap: {effective_path}", Fore.GREEN))
    except Exception as e:
        print(colour(f"Error writing effective configuration file {effective_path}: {e}", Fore.RED))
        raise RuntimeError(f"Failed to write effective configuration: {e}") from e

    return effective_path


# -----------------------------------------------------------------------------
# Watcher
# -----------------------------------------------------------------------------

class BaseConfigChangeHandler(FileSystemEventHandler):
    def __init__(
        self,
        runner: LlamaRunner,
        base_path: Path,
        override_path: Optional[Path],
        script_dir: Path,
        verbose: bool,
    ):
        super().__init__()
        self.runner = runner
        self.base_path = base_path.resolve()
        self.override_path = override_path.resolve() if override_path else None
        self.script_dir = script_dir
        self.verbose = verbose
        self._last_change_ts = 0.0
        self._debounce_s = 0.4

    @staticmethod
    def _samefile(a: Path, b: Path) -> bool:
        try:
            return a.resolve().samefile(b.resolve())
        except Exception:
            # Fallback cross-platform-ish compare
            return str(a.resolve()).casefold() == str(b.resolve()).casefold()

    def _is_target(self, path_str: str) -> bool:
        p = Path(path_str)
        return self._samefile(p, self.base_path) or (self.override_path and self._samefile(p, self.override_path))

    def _maybe_reload(self, event: FileSystemEvent) -> None:
        if not self._is_target(event.src_path):
            # on move events, also check destination
            dst = getattr(event, "dest_path", None)
            if not (dst and self._is_target(dst)):
                return

        now = time.time()
        if now - self._last_change_ts < self._debounce_s:
            return
        self._last_change_ts = now

        if self.verbose:
            print(colour(f"Configuration changed ({Path(event.src_path).name}). Rebuilding & restarting...", Fore.YELLOW))
        try:
            new_effective = process_and_write_effective_config(
                self.base_path,
                self.override_path,
                self.script_dir,
                self.verbose,
            )
            self.runner.config_path = new_effective
            self.runner.restart()
        except Exception as e:
            print(colour(f"Error during automatic re-configuration: {e}", Fore.RED))
            print(colour("llama-swap may continue with the old configuration if running.", Fore.YELLOW))
            if self.verbose:
                import traceback
                traceback.print_exc()

    # Trigger on common change types (created/modified/moved)
    def on_modified(self, event: FileSystemEvent) -> None: self._maybe_reload(event)
    def on_created(self, event: FileSystemEvent)  -> None: self._maybe_reload(event)
    def on_moved(self, event: FileSystemEvent)    -> None: self._maybe_reload(event)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_base_config_path = (script_dir / "config.base.yaml").resolve()

    parser = argparse.ArgumentParser(
        description="Watches base config and restarts llama-swap with merged configuration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("config", nargs="?", type=Path, default=default_base_config_path,
                        help="Base configuration YAML file path.")
    parser.add_argument("--exe", type=Path, help="Path to the llama-swap executable.")
    parser.add_argument("--listen", default=":8080", help="Listen address for llama-swap.")
    parser.add_argument("-o", "--override", type=Path, help="Override YAML file path.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")

    args = parser.parse_args()

    resolved_base = args.config.resolve()
    if not resolved_base.is_file():
        print(colour(f"Base configuration file not found: {resolved_base}", Fore.RED))
        sys.exit(1)

    try:
        exe_path = find_llama_swap_executable(args.exe, script_dir)
        print(colour(f"Using llama-swap executable: {exe_path}", Fore.BLUE))
    except FileNotFoundError as e:
        print(colour(str(e), Fore.RED))
        sys.exit(1)

    try:
        effective_config = process_and_write_effective_config(
            resolved_base,
            args.override,
            script_dir,
            args.verbose,
        )
    except Exception:
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    runner = LlamaRunner(
        exe_path=exe_path,
        config_path=effective_config,
        listen_address=args.listen,
        verbose=args.verbose,
    )

    signal.signal(signal.SIGINT, runner.shutdown_handler)
    # SIGTERM is not available on some platforms like Windows, but set if possible
    try:
        signal.signal(signal.SIGTERM, runner.shutdown_handler)
    except Exception:
        pass

    runner.start()

    observer = Observer()
    handler = BaseConfigChangeHandler(runner, resolved_base, args.override, script_dir, args.verbose)

    try:
        watch_dir = str(resolved_base.parent)
        observer.schedule(handler, watch_dir, recursive=False)
        if args.override:
            # Watch override's directory too if it differs
            override_dir = str(Path(args.override).resolve().parent)
            if override_dir != watch_dir:
                observer.schedule(handler, override_dir, recursive=False)
        observer.start()
        print(colour(f"Watching for changes in: {watch_dir}" + (f" and {Path(args.override).parent}" if args.override else ""),
                     Fore.BLUE))
    except Exception as e:
        print(colour(f"Failed to start file watcher: {e}", Fore.RED))
        print(colour("Proceeding without automatic config reload on change.", Fore.YELLOW))

    try:
        # Render a friendly hint for OpenWebUI users
        try:
            port_str = args.listen.split(":")[-1] or "8080"
            if not port_str.isdigit():
                port_str = "8080"
        except Exception:
            port_str = "8080"
        print(colour(f"llama-swap setup complete. If using Open WebUI, try: http://localhost:{port_str}/v1", Fore.GREEN))

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
            if observer.is_alive():
                observer.stop()
                observer.join(timeout=2)
        except Exception as e_obs:
            print(colour(f"Error stopping file observer: {e_obs}", Fore.YELLOW))
        runner.stop()
        print(colour("Shutdown complete.", Fore.BLUE))


if __name__ == "__main__":
    main()
