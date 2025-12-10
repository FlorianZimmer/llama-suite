#!/usr/bin/env python3
r"""
Cross-platform watcher / restarter for *llama-swap*.

New default layout (repo root assumed; override with --root or $LLAMA_SUITE_ROOT):

  configs/
    config.base.yaml
    overrides/<host>.yaml
    generated/config.effective.yaml        (written by this tool)

  vendor/llama-swap/
    llama-swap.exe (Windows)
    llama-swap     (macOS/Linux)

  var/logs/                                (stdout/stderr of llama-swap)

Run:
  python -m llama_suite.watchers.llama_swap_watch ^
      --root F:\LLMs\llama-suite ^
      --override configs/overrides/win-3080-10G.yaml ^
      --listen :8080 -v

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
from typing import Any, Callable, Dict, Iterable, List, Optional

import yaml
from colorama import Fore, Style, init as colorama_init
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

# --- Imports from package (new structure) + safe colour wrapper ----------------

from typing import TYPE_CHECKING

try:
    from llama_suite.utils.config_utils import generate_processed_config, colour_util as _colour_util  # type: ignore[attr-defined]
except Exception as _e:
    # If import fails entirely, re-raise with a clearer message
    raise ImportError(
        "Could not import generate_processed_config/colour_util from llama_suite.utils.config_utils. "
        "Make sure you run this as a module (python -m ...) and that your project is installed/loaded properly."
    ) from _e

# colour() wrapper with a stable signature: (msg: str, _c: Any | None = None) -> str
def colour(msg: str, _c: Any = None) -> str:
    try:
        # Support both colour_util(msg) and colour_util(msg, ansi_color) styles
        if _c is None:
            try:
                return _colour_util(msg)  # type: ignore[misc]
            except TypeError:
                return _colour_util(msg, "")  # type: ignore[misc]
        return _colour_util(msg, _c)  # type: ignore[misc]
    except Exception:
        return msg

# Initialize Colorama (non-fatal if unavailable)
try:
    colorama_init(autoreset=True)
except Exception:
    print("Warning: colorama init failed or not installed. Proceeding without colors.")

# --- Constants ----------------------------------------------------------------

DEFAULT_LOGS_TO_KEEP = 10
PROCESS_TERMINATE_TIMEOUT_SECONDS = 10

MODEL_CONFIG_META_KEYS = {
    "aliases",
    "sampling",
    "_name_for_log", "_name-for-log",
    "generated_cmd_str", "generated-cmd-str",
    "cmd",
    "hf_tokenizer_for_model", "hf-tokenizer-for-model",
    "supports_no_think_toggle", "supports-no-think-toggle",
    "enabled", "skip", "_skip", "disabled",
}

AUTO_OMIT_KEYS = {
    "gpu-layers", "n-gpu-layers",
    "threads", "threads-batch",
}

# --- Repo root + paths ---------------------------------------------------------

def find_repo_root(cli_root: Optional[Path]) -> Path:
    # 1) CLI flag wins
    if cli_root:
        return cli_root.resolve()

    # 2) Environment variable
    env = os.environ.get("LLAMA_SUITE_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    # 3) Walk upward until we find a folder that looks like the repo root
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        root = p if p.is_dir() else p.parent
        if (root / "configs").is_dir():
            return root

    # 4) Fallback: assume file is at <root>/src/llama_suite/watchers/...
    # parents[3] == <root> for .../src/llama_suite/watchers/<file>.py
    return here.parents[3]


def repo_path(root: Path, *parts: str) -> Path:
    return (root.joinpath(*parts)).resolve()

def default_base_config(root: Path) -> Path:
    return repo_path(root, "configs", "config.base.yaml")

def effective_config_path(root: Path) -> Path:
    gen_dir = repo_path(root, "configs", "generated")
    gen_dir.mkdir(parents=True, exist_ok=True)
    return gen_dir / "config.effective.yaml"

def logs_dir(root: Path) -> Path:
    d = repo_path(root, "var", "logs")
    d.mkdir(parents=True, exist_ok=True)
    return d

# --- Helpers ------------------------------------------------------------------

import platform
import shlex

def _candidate_model_dirs(repo_root: Path, processed_top: Dict[str, Any]) -> List[Path]:
    cands: List[Path] = []

    # from config top-level hints (optional)
    for k in ("models_root", "models_dir", "models-path"):
        v = processed_top.get(k)
        if isinstance(v, str) and v.strip():
            cands.append(Path(os.path.expandvars(v)).expanduser())

    # env override
    env = os.environ.get("LLAMA_MODELS_DIR")
    if env:
        cands.append(Path(env).expanduser())

    # repo default
    cands.append(repo_root / "models")

    # common siblings (user often stores GGUF outside repo)
    for sibling in ("GGUF", "Models", "models"):
        maybe = repo_root.parent / sibling
        if maybe.exists():
            cands.append(maybe)

    # unique, absolute, existing-or-not (we’ll check existence later)
    uniq: List[Path] = []
    seen = set()
    for p in (x.resolve() for x in cands):
        if str(p) not in seen:
            uniq.append(p)
            seen.add(str(p))
    return uniq


def _resolve_model_path(val: Any, repo_root: Path, processed_top: Dict[str, Any], verbose: bool) -> str:
    """
    Resolve the 'model' file path robustly:
      - absolute   -> re-anchor if it's under <root>/configs/
      - relative   -> try <root>/<val>, then each candidate model dir (including basename search)
    """
    s = os.path.expandvars(str(val))
    p = Path(s).expanduser()

    def _exists(pp: Path) -> bool:
        try:
            return pp.is_file()
        except Exception:
            return False

    # 1) Absolute path
    if p.is_absolute():
        p2 = _reanchor_from_configs(p, repo_root)
        if _exists(p2):
            return str(p2)
        # fall through to search

    # 2) Relative to repo root
    cand = _reanchor_from_configs((repo_root / p).resolve(), repo_root)
    if _exists(cand):
        return str(cand)

    # 3) Try candidate model directories
    name = p.name
    tried: List[str] = []
    for base in _candidate_model_dirs(repo_root, processed_top):
        # (a) preserve subfolders if provided (e.g., "qwen/Qwen3-8B.gguf")
        c1 = base / p
        tried.append(str(c1))
        if _exists(c1):
            return str(c1.resolve())

        # (b) basename search within base (first match wins)
        try:
            for found in base.rglob(name):
                if _exists(found):
                    return str(found.resolve())
        except Exception:
            # ignore permission issues etc.
            pass

    if verbose:
        print(colour(
            "[WARN] Model file not found; leaving best-effort path. Tried:\n  - " + "\n  - ".join(tried),
            Fore.YELLOW
        ))

    # 4) Best effort fallback
    return str(cand)


def _reanchor_configs_path(s: str, repo_root: Path) -> str:
    """Rewrite .../configs/{models|llama.cpp}/... -> .../{models|llama.cpp}/... in any string."""
    root = str(repo_root.resolve())
    # Normalize slashes for search
    s_norm = s.replace("\\", "/")
    root_norm = root.replace("\\", "/")
    replacements = [
        (f"{root_norm}/configs/models/", f"{root_norm}/models/"),
        (f"{root_norm}/configs/llama.cpp/", f"{root_norm}/llama.cpp/"),
    ]
    for old, new in replacements:
        s_norm = s_norm.replace(old, new)
    # Return with original style (Windows keeps backslashes)
    return s_norm.replace("/", "\\") if os.name == "nt" else s_norm

def _fix_llama_server_token(cmd_str: str, repo_root: Path) -> str:
    """Ensure the first token (llama-server) points to a real file; fallback to vendor if needed."""
    # Split respecting OS quoting
    try:
        tokens = shlex.split(cmd_str, posix=(os.name != "nt"))
    except Exception:
        return cmd_str  # best effort

    if not tokens:
        return cmd_str

    first = Path(tokens[0])
    # If first is not absolute, anchor to repo_root for existence check
    first_abs = first if first.is_absolute() else (repo_root / first)
    if first_abs.is_file():
        return cmd_str  # good

    # Try vendor + legacy fallbacks
    base = first_abs.name or "llama-server.exe" if os.name == "nt" else "llama-server"
    candidates = [
        repo_root / "vendor" / "llama.cpp" / "bin" / base,
        repo_root / "vendor" / "llama.cpp" / "bin" / ("llama-server.exe" if os.name == "nt" else "llama-server"),
        repo_root / "llama.cpp" / "build" / "bin" / base,
        repo_root / "llama.cpp" / "build" / "bin" / ("llama-server.exe" if os.name == "nt" else "llama-server"),
    ]
    for c in candidates:
        if c.is_file():
            tokens[0] = str(c.resolve())
            return " ".join(shell_quote(t) for t in tokens)

    # If nothing exists, still anchor to repo_root and return
    tokens[0] = str(first_abs)
    return " ".join(shell_quote(t) for t in tokens)

def _sanitize_cmd_string(s: str, repo_root: Path) -> str:
    # remove meta fragments
    s = re.sub(r'\s--cmd\s+"[^"]*"', "", s)
    s = re.sub(r'\s--cmd\s+\S+', "", s)
    # remove unsupported llama-server flags that may leak in
    s = re.sub(r'\s--hf-tokenizer-for-model\s+"[^"]*"', "", s, flags=re.IGNORECASE)
    s = re.sub(r'\s--hf-tokenizer-for-model\s+\S+', "", s, flags=re.IGNORECASE)

    s = _reanchor_configs_path(s, repo_root)
    s = _fix_llama_server_token(s, repo_root)
    return s.strip()


# --- path normalization helpers ---
PATHLIKE_KEYS = {
    "model",
    "mmproj",
    "log-file",
    "chat-template",
    "chat-template-file",
    "system-prompt-file",
    "lora-base",
    "lora-path",
}
CMD_PATHLIKE_KEYS = {"log-file"}  # path-like keys in cmd.{...}

def _reanchor_from_configs(p: Path, repo_root: Path) -> Path:
    """
    If a path ended up under <root>/configs/<X>/..., re-anchor it to <root>/<X>/...
    e.g. <root>/configs/models/foo.gguf -> <root>/models/foo.gguf
    Only kicks in for top-level folders we actually expect.
    """
    try:
        p_abs = p.resolve()
    except Exception:
        p_abs = Path(str(p))
    cfg_dir = (repo_root / "configs").resolve()
    try:
        rel = p_abs.relative_to(cfg_dir)
    except Exception:
        return p_abs
    # Only reanchor for folders we expect to live at repo root
    first = rel.parts[0] if rel.parts else ""
    if first.lower() in {"models", "llama.cpp", "vendor"}:
        return (repo_root / rel).resolve()
    return p_abs

def _resolve_repo_relpath(val: Any, repo_root: Path) -> str:
    """
    Convert a possibly-relative path to an absolute path at repo_root,
    then re-anchor if it accidentally points under <root>/configs/.
    """
    s = os.path.expandvars(str(val))
    p = Path(s).expanduser()
    if p.is_absolute():
        return str(_reanchor_from_configs(p, repo_root))
    # handle values like "configs/models/foo.gguf" explicitly
    q = (repo_root / p).resolve()
    return str(_reanchor_from_configs(q, repo_root))

def _resolve_llama_server_bin(bin_value: str, repo_root: Path) -> str:
    """
    Resolve cmd.bin. Prefer repo-root anchoring; if missing, try vendor/ and legacy llama.cpp tree.
    """
    candidate = _reanchor_from_configs(Path(_resolve_repo_relpath(bin_value, repo_root)), repo_root)
    if candidate.is_file():
        return str(candidate)

    base = Path(bin_value).name
    fallbacks = [
        repo_root / "vendor" / "llama.cpp" / "bin" / base,
        repo_root / "vendor" / "llama.cpp" / "bin" / (base + ".exe"),
        repo_root / "llama.cpp" / "build" / "bin" / base,             # legacy source-tree layout
        repo_root / "llama.cpp" / "build" / "bin" / (base + ".exe"),
    ]
    for fb in fallbacks:
        if fb.is_file():
            return str(fb.resolve())
    return str(candidate)

def _parse_filter_list(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    if isinstance(val, Iterable):
        return [x for x in val if isinstance(x, str)]
    return []

def _collect_model_filters(cfg: Dict[str, Any]) -> tuple[List[str], List[str]]:
    only: List[str] = []
    exclude: List[str] = []
    for k in ("only_models", "models_only", "include_models_only"):
        only += _parse_filter_list(cfg.get(k))
    for k in ("exclude_models", "models_exclude", "disabled_models", "skip_models"):
        exclude += _parse_filter_list(cfg.get(k))
    return only, exclude

def _name_matches(name: str, pattern: str) -> bool:
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
    if isinstance(val, (dict, list, int, float, bool)) or val is None:
        return json.dumps(val, separators=(",", ":"))
    if isinstance(val, str):
        s = val.strip()
        try:
            return json.dumps(json.loads(s), separators=(",", ":"))
        except Exception:
            pass
        try:
            y = yaml.safe_load(s)
            if isinstance(y, (dict, list, int, float, bool)) or y is None:
                return json.dumps(y, separators=(",", ":"))
        except Exception:
            pass
        return s
    return json.dumps(val, separators=(",", ":"))

def _needs_quote(token: str) -> bool:
    return any(ch.isspace() for ch in token) or any(ch in token for ch in r"&|^<>();!\"'`$*[]{}?=")

def _quote_posix(token: str) -> str:
    if not _needs_quote(token):
        return token
    return "'" + token.replace("'", "'\"'\"'") + "'"

def _quote_windows(token: str) -> str:
    if not _needs_quote(token):
        return token
    token = token.replace('"', r'\"')
    return f'"{token}"'

def shell_quote(token: str) -> str:
    if sys.platform.startswith("win"):
        return _quote_windows(token)
    return _quote_posix(token)

def append_flag_args(args_list: List[str], key: str, value: Any) -> None:
    normalized_key = key.replace("_", "-")
    cli_flag = f"--{normalized_key}"

    if normalized_key in AUTO_OMIT_KEYS and isinstance(value, str) and value.lower() == "auto":
        return

    if normalized_key.endswith("-kwargs"):
        args_list.extend([cli_flag, _compact_json(value)])
        return

    if isinstance(value, bool):
        if value:
            args_list.append(cli_flag)
        return

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

    if isinstance(value, dict):
        for sub_k, sub_v in value.items():
            if sub_v is None:
                continue
            args_list.extend([cli_flag, f"{sub_k}={sub_v}"])
        return

    if value is not None:
        args_list.extend([cli_flag, str(value)])

def prune_logs(log_dir: Path, keep: int = DEFAULT_LOGS_TO_KEEP) -> None:
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

# --- Runner -------------------------------------------------------------------

@dataclass
class LlamaRunner:
    exe_path: Path
    config_path: Path
    listen_address: str
    verbose: bool
    log_root: Path

    child_process: Optional[subprocess.Popen[str]] = None
    log_file_handle: Optional[Any] = None

    _lock: threading.Lock = threading.Lock()
    _last_restart_ts: float = 0.0
    _restart_min_interval_s: float = 0.6

    def start(self) -> None:
        if self.child_process and self.child_process.poll() is None:
            if self.verbose:
                print(colour("llama-swap process already running.", Fore.YELLOW))
            return

        cmd = [
            str(self.exe_path),
            "--config", str(self.config_path),
            "--listen", self.listen_address,
        ]

        model_tag = os.getenv("MODEL_NAME", "llama_swap")
        log_dir = self.log_root
        prune_logs(log_dir)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file_path = log_dir / f"{model_tag}_{timestamp}.log"

        try:
            self.log_file_handle = log_file_path.open("w", encoding="utf-8")
            if self.verbose:
                print(colour(f"Starting llama-swap. Logging to: {log_file_path}", Fore.GREEN))

            # Use PIPE for stdout to capture and forward to both file and stdout
            kwargs = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "bufsize": 1,
            }

            if sys.platform.startswith("win"):
                kwargs["creationflags"] = 0x00000200  # CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True

            self.child_process = subprocess.Popen(cmd, **kwargs)

            # Start a thread to forward logs to both file and stdout
            def _log_forwarder():
                proc = self.child_process
                if not proc or not proc.stdout:
                    return
                try:
                    for line in proc.stdout:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        if self.log_file_handle and not self.log_file_handle.closed:
                            try:
                                self.log_file_handle.write(line)
                                self.log_file_handle.flush()
                            except Exception:
                                pass
                except Exception:
                    pass

            t = threading.Thread(target=_log_forwarder, daemon=True)
            t.start()

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
                    getpgid = getattr(os, "getpgid", None)
                    killpg = getattr(os, "killpg", None)
                    if callable(getpgid) and callable(killpg):
                        try:
                            pgid = getpgid(pid)  # type: ignore[misc]
                            killpg(pgid, signal.SIGTERM)  # type: ignore[misc]
                            try:
                                self.child_process.wait(timeout=PROCESS_TERMINATE_TIMEOUT_SECONDS)
                            except subprocess.TimeoutExpired:
                                sig_kill = getattr(signal, "SIGKILL", signal.SIGTERM)
                                killpg(pgid, sig_kill)  # type: ignore[misc]
                        except Exception:
                            self.child_process.terminate()
                            try:
                                self.child_process.wait(timeout=PROCESS_TERMINATE_TIMEOUT_SECONDS)
                            except subprocess.TimeoutExpired:
                                self.child_process.kill()
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

# --- Config processing + binary discovery -------------------------------------

def find_llama_swap_executable(provided_exe: Optional[Path], repo_root: Path) -> Path:
    if provided_exe:
        p = provided_exe.expanduser().resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"Provided llama-swap executable not found: {p}")

    mapping = {
        "win32": Path("vendor") / "llama-swap" / "llama-swap.exe",
        "cygwin": Path("vendor") / "llama-swap" / "llama-swap.exe",
        "linux": Path("vendor") / "llama-swap" / "llama-swap-linux-amd64",
        "darwin": Path("vendor") / "llama-swap" / "llama-swap",
    }
    plat = sys.platform
    for key, rel in mapping.items():
        if plat.startswith(key):
            candidate = (repo_root / rel).resolve()
            if candidate.is_file():
                return candidate

    raise FileNotFoundError(
        "llama-swap executable not found. Expected under vendor/llama-swap/. "
        "Use --exe to provide a custom path, or set LLAMA_SUITE_ROOT/--root."
    )

# --- replace your existing process_and_write_effective_config with this ---
def process_and_write_effective_config(
    base_config_path: Path,
    override_config_path: Optional[Path],
    repo_root: Path,
    verbose: bool,
) -> Path:
    """
    Use config_utils to generate processed config and write configs/generated/config.effective.yaml.
    All path-like fields are resolved relative to the REPO ROOT (not the config file directory).
    """
    try:
        processed = generate_processed_config(
            base_config_path,
            override_config_path,
            script_dir_for_overrides=repo_root,   # ← repo root, not configs/
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
        if str(mcfg.get("enabled", "true")).lower() in {"false", "0", "no"} \
           or bool(mcfg.get("skip")) or bool(mcfg.get("_skip")) or bool(mcfg.get("disabled")):
            if verbose:
                print(colour(f"Model '{name}' skipped due to disable flag.", Fore.MAGENTA))
            continue

        entry: Dict[str, Any] = {}

        # --- Prefer a prebuilt command if provided (no double-expansion) ---
        prebuilt = mcfg.get("generated_cmd_str") or mcfg.get("generated-cmd-str")
        if isinstance(prebuilt, str) and prebuilt.strip():
            final_cmd_str = _sanitize_cmd_string(prebuilt, repo_root)
            entry["cmd"] = final_cmd_str
            if verbose:
                print(colour(f"Model '{name}' (prebuilt) cmd: {final_cmd_str}", Fore.MAGENTA))
        else:
            # --- Build strictly from cmd.{...} + selected top-level flags ---
            server_args: List[str] = []

            cmd_dict = mcfg.get("cmd", {})
            if not isinstance(cmd_dict, dict):
                cmd_dict = {}
                if verbose:
                    print(colour(f"Model '{name}' has non-dict or missing 'cmd'. Value: {mcfg.get('cmd')}", Fore.MAGENTA))

            server_bin = cmd_dict.get("bin")
            if not server_bin or not isinstance(server_bin, str):
                print(colour(
                    f"Error: Model '{name}' - missing/invalid 'cmd.bin' (llama-server path). Skipping command build.",
                    Fore.RED
                ))
            else:
                # 1) binary
                server_args.append(_fix_llama_server_token(_reanchor_configs_path(str(server_bin), repo_root), repo_root).split()[0])

                # 2) cmd.* keys (NO nested dicts; only flags)
                for k, v in cmd_dict.items():
                    if k == "bin":
                        continue
                    if isinstance(v, dict):
                        # refuse to emit nested dicts as '--cmd k=v'
                        continue
                    if k in CMD_PATHLIKE_KEYS and isinstance(v, str):
                        v = _reanchor_configs_path(_resolve_repo_relpath(v, repo_root), repo_root)
                    append_flag_args(server_args, k, v)

                # 3) top-level model keys (resolve known path-like)
                # ... after server_args has the binary and you’re iterating top-level model keys:
                for k, v in mcfg.items():
                    if k in ("cmd", "generated_cmd_str", "generated-cmd-str") or k in MODEL_CONFIG_META_KEYS:
                        continue

                    if k == "model" and isinstance(v, (str, os.PathLike)):
                        v = _resolve_model_path(v, repo_root, processed, verbose)
                        append_flag_args(server_args, k, v)
                        continue

                    if k in PATHLIKE_KEYS and isinstance(v, str):
                        v = _resolve_repo_relpath(v, repo_root)
                    append_flag_args(server_args, k, v)


                # 4) sampling dict (as before)
                sampling = mcfg.get("sampling")
                if isinstance(sampling, dict):
                    for sk, sv in sampling.items():
                        append_flag_args(server_args, sk, sv)

                final_tokens = [shell_quote(str(t)) for t in server_args]
                final_cmd_str = " ".join(final_tokens)
                final_cmd_str = _sanitize_cmd_string(final_cmd_str, repo_root)
                entry["cmd"] = final_cmd_str
                if verbose:
                    print(colour(f"Model '{name}' effective cmd: {final_cmd_str}", Fore.MAGENTA))

        # Copy non-meta props (don’t re-add cmd/generated strings)
        for k, v in mcfg.items():
            if k not in {"cmd", "generated_cmd_str", "generated-cmd-str"} and k not in MODEL_CONFIG_META_KEYS:
                entry[k] = v

        output["models"][name] = entry

    if not output["models"]:
        print(colour("[WARN] No models remain after include/exclude/disable rules.", Fore.YELLOW))

    # Copy other top-level keys
    for k, v in processed.items():
        if k != "models":
            output[k] = v

    effective_path = effective_config_path(repo_root)
    try:
        yaml_text = yaml.dump(output, sort_keys=False, allow_unicode=True, width=120)
        effective_path.write_text(yaml_text, encoding="utf-8")
        print(colour(f"Wrote effective configuration for llama-swap: {effective_path}", Fore.GREEN))
    except Exception as e:
        print(colour(f"Error writing effective configuration file {effective_path}: {e}", Fore.RED))
        raise RuntimeError(f"Failed to write effective configuration: {e}") from e

    return effective_path


# --- Watcher ------------------------------------------------------------------

class BaseConfigChangeHandler(FileSystemEventHandler):
    def __init__(
        self,
        runner: LlamaRunner,
        base_path: Path,
        override_path: Optional[Path],
        repo_root: Path,
        verbose: bool,
    ):
        super().__init__()
        self.runner = runner
        self.base_path = base_path.resolve()
        self.override_path = override_path.resolve() if override_path else None
        self.repo_root = repo_root
        self.verbose = verbose
        self._last_change_ts = 0.0
        self._debounce_s = 0.4

    @staticmethod
    def _samefile(a: Path, b: Path) -> bool:
        try:
            return a.resolve().samefile(b.resolve())
        except Exception:
            return str(a.resolve()).casefold() == str(b.resolve()).casefold()

    def _is_target(self, path_str: str) -> bool:
        p = Path(path_str)
        return self._samefile(p, self.base_path) or (
            self.override_path is not None and self._samefile(p, self.override_path)
        )

    def _maybe_reload(self, event: FileSystemEvent) -> None:
        if not self._is_target(event.src_path):
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
                self.repo_root,
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

    def on_modified(self, event: FileSystemEvent) -> None: self._maybe_reload(event)
    def on_created(self, event: FileSystemEvent)  -> None: self._maybe_reload(event)
    def on_moved(self, event: FileSystemEvent)    -> None: self._maybe_reload(event)

# --- Main ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Watches base config and restarts llama-swap with merged configuration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("config", nargs="?", type=Path,
                        help="Base configuration YAML file path (defaults to <root>/configs/config.base.yaml).")
    parser.add_argument("--root", type=Path, help="Repository root (defaults to --root, $LLAMA_SUITE_ROOT, or derived).")
    parser.add_argument("--exe", type=Path, help="Path to the llama-swap executable.")
    parser.add_argument("--listen", default=":8080", help="Listen address for llama-swap.")
    parser.add_argument("-o", "--override", type=Path, help="Override YAML file path.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")

    cli = parser.parse_args()

    repo_root = find_repo_root(cli.root)
    base_cfg = cli.config.resolve() if cli.config else default_base_config(repo_root)
    if not base_cfg.is_file():
        print(colour(f"Base configuration file not found: {base_cfg}", Fore.RED))
        sys.exit(1)

    try:
        exe_path = find_llama_swap_executable(cli.exe, repo_root)
        print(colour(f"Using llama-swap executable: {exe_path}", Fore.BLUE))
    except FileNotFoundError as e:
        print(colour(str(e), Fore.RED))
        sys.exit(1)

    try:
        effective_config = process_and_write_effective_config(
            base_cfg,
            cli.override,
            repo_root,
            cli.verbose,
        )
    except Exception:
        if cli.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    runner = LlamaRunner(
        exe_path=exe_path,
        config_path=effective_config,
        listen_address=cli.listen,
        verbose=cli.verbose,
        log_root=logs_dir(repo_root),
    )

    signal.signal(signal.SIGINT, runner.shutdown_handler)
    try:
        signal.signal(signal.SIGTERM, runner.shutdown_handler)
    except Exception:
        pass

    runner.start()

    observer = Observer()
    handler = BaseConfigChangeHandler(runner, base_cfg, cli.override, repo_root, cli.verbose)

    try:
        watch_dir = str(base_cfg.parent)
        observer.schedule(handler, watch_dir, recursive=False)
        if cli.override:
            override_dir = str(Path(cli.override).resolve().parent)
            if override_dir != watch_dir:
                observer.schedule(handler, override_dir, recursive=False)
        observer.start()
        print(colour(
            f"Watching for changes in: {watch_dir}"
            + (f" and {Path(cli.override).parent}" if cli.override else ""),
            Fore.BLUE
        ))
    except Exception as e:
        print(colour(f"Failed to start file watcher: {e}", Fore.RED))
        print(colour("Proceeding without automatic config reload on change.", Fore.YELLOW))

    try:
        try:
            port_str = cli.listen.split(":")[-1] or "8080"
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
        if cli.verbose:
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
