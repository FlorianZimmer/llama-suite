#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path if needed when run directly
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if __package__ in (None, "") and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llama_suite.utils.config_utils import (  # type: ignore
    Logger, DEFAULT_HEALTH_POLL_INTERVAL_S, _dump_stderr_on_failure,
    start_llama_server, stop_llama_server, wait_for_server_health, color_status
)

from llama_suite.bench import util  # type: ignore

# --------- Script constants
DEFAULT_OUTPUT_BASENAME = "memory_scan_results"
DEFAULT_HEALTH_TIMEOUT_S_SCAN = 120
STATIC_SERVER_PORT_SCAN = "9998"
RETENTION_KEEP = util.RETENTION_KEEP_DEFAULT

logger: Optional['Logger'] = None
TEMP_DIR_MANAGER_PATH: Optional[Path] = None


def emit_step(i: int, n: int, message: str) -> None:
    # ASCII-only, stable format (parsed by Web UI for progress).
    print(f"STEP {i}/{n}: {message}", flush=True)


def run_memory_scan(
    processed_models_config: Dict[str, Any],
    output_csv_path: Path,
    health_timeout_s: int,
    health_poll_s: float,
    model_to_scan_alias: Optional[str],
    run_logs_dir: Path,
    logger_instance: Logger
):
    models_to_iterate = util.select_models(processed_models_config, model_to_scan_alias, logger_instance)
    if not models_to_iterate:
        return

    base_proxy_url = f"http://127.0.0.1:{STATIC_SERVER_PORT_SCAN}"
    health_url = f"{base_proxy_url}/health"

    headers = ["ModelName", "Timestamp", "ScanStatus", "GpuMemoryGB", "CpuMemoryGB", "Error"]
    rows: List[List[str]] = []

    for idx, (alias, model_cfg) in enumerate(models_to_iterate.items(), 1):
        emit_step(idx, len(models_to_iterate), f"Memory scan: {alias}")
        logger_instance.subheader(f"Scanning Model ({idx}/{len(models_to_iterate)}): {alias}")
        ts = logger_instance._get_timestamp()

        gpu_gb, cpu_gb, scan_status = "-", "-", "Not Scanned"
        err_msg = ""

        # Build server command (simple, robust; draft handled inside util)
        logger_instance.step(f"Preparing server command for '{alias}'")
        try:
            server_exe, server_args, _ctx = util.build_server_command(
                alias, model_cfg, static_port=STATIC_SERVER_PORT_SCAN, logger=logger_instance
            )
        except Exception as e:
            err_msg = f"Cmd Build Error: {e}"
            logger_instance.error(f"  {err_msg}")
            rows.append([alias, ts, "Cmd Build Error", gpu_gb, cpu_gb, err_msg])
            logger_instance.notice("-" * 30)
            continue

        # Start -> health -> parse memory
        logger_instance.step(f"Initiating memory scan for '{alias}'")
        server_p = None
        stderr_log_path: Optional[Path] = None

        info = start_llama_server(
            executable_path_str=server_exe,
            arguments_list=server_args,
            model_name=alias,
            temp_dir=run_logs_dir,
            logger_instance=logger_instance,
            project_root_for_resolution=util.PROJECT_ROOT
        )

        if info:
            server_p, _, stderr_log_path = info
            healthy = wait_for_server_health(
                process=server_p,
                health_check_url=health_url,
                timeout_s=health_timeout_s,
                poll_interval_s=health_poll_s,
                model_name=alias,
                logger_instance=logger_instance
            )

            if healthy:
                logger_instance.info(f"  Server healthy, parsing memory usage from logs for '{alias}'...")
                if stderr_log_path:
                    gpu_gb, cpu_gb, scan_status = util.parse_memory_from_log(stderr_log_path, alias, logger_instance)
                else:
                    scan_status, err_msg = "Internal Error", "stderr_log_path was None after server start"
                    logger_instance.error(f"  {err_msg}")
            else:
                if server_p.poll() is not None:
                    exit_code = server_p.returncode
                    scan_status = f"Server Exited (Code: {exit_code})"
                    err_msg = f"Server (PID {server_p.pid}) exited (code {exit_code}) during/after health check."
                    logger_instance.warn(f"  {err_msg}")
                    _dump_stderr_on_failure(stderr_log_path, alias, logger_instance)
                else:
                    scan_status = "Health Timeout"
                    err_msg = f"Server (PID {server_p.pid}) health check timed out."
                    logger_instance.warn(f"  {err_msg} Process still running, will be stopped.")
            stop_llama_server(server_p, alias, logger_instance)
        else:
            scan_status = "Server Start Failed"
            err_msg = "Server process failed to start (start_llama_server returned None)."

        logger_instance.info(
            f"  Memory Scan Result for '{alias}': "
            f"GPU {gpu_gb} GB, CPU {cpu_gb} GB - Status: {color_status(scan_status)}"
        )
        if err_msg and scan_status not in ["Success", "Parse Error", "Failed (No Buffers)"]:
            logger_instance.warn(f"    Details: {err_msg}")

        rows.append([alias, ts, scan_status, gpu_gb, cpu_gb, err_msg])
        logger_instance.notice("-" * 30)

    util.write_csv(headers, rows, output_csv_path, logger_instance)


def main():
    global logger, TEMP_DIR_MANAGER_PATH

    p = argparse.ArgumentParser(
        description="LLM Model Memory Scanner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("-c", "--config", type=Path, default=util.CONFIGS_DIR / "config.base.yaml")
    p.add_argument("--override", type=Path, default=util.default_override_for_hostname())
    p.add_argument("--poll-interval", type=float, default=DEFAULT_HEALTH_POLL_INTERVAL_S)
    # If provided explicitly, always overrides config's healthCheckTimeout.
    p.add_argument("--health-timeout", type=int, default=None)
    p.add_argument("-m", "--model", type=str)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    util.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    util.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ts = util.timestamp_str()
    output_csv = (util.RESULTS_DIR / f"{DEFAULT_OUTPUT_BASENAME}_{ts}.csv").resolve()
    main_log = (util.LOGS_DIR / f"scan_{ts}.log").resolve()
    logger = Logger(verbose=args.verbose, log_file_path=main_log)

    logger.header("LLM MODEL MEMORY SCANNER INITIALIZATION")
    logger.info(f"Main script log file: {main_log}")

    util.register_signal_handlers("scan_model_memory", lambda: logger, lambda: TEMP_DIR_MANAGER_PATH)

    base_config = args.config.resolve()
    if not base_config.is_file():
        logger.error(f"Base configuration file not found: {base_config}")
        raise SystemExit(1)
    logger.info(f"Using base configuration: {base_config}")

    override = args.override.resolve() if args.override else None
    if override and not override.is_file():
        logger.warn(f"Specified override configuration not found: {override}. Proceeding without.")
        override = None
    logger.info("Using override configuration: " + (str(override) if override else "None"))

    util.kill_lingering_servers_on_port(STATIC_SERVER_PORT_SCAN, logger)

    logger.step("Loading and processing configurations...")
    try:
        effective_conf = util.load_and_process_config(base_config, override, verbose=args.verbose, logger=logger)
        logger.success("Configurations processed successfully.")
    except Exception as e:
        logger.error(f"Failed to load or process configurations: {e}")
        raise SystemExit(1)

    health_timeout_final = (
        args.health_timeout
        if args.health_timeout is not None
        else effective_conf.get("healthCheckTimeout", DEFAULT_HEALTH_TIMEOUT_S_SCAN)
    )
    if not isinstance(health_timeout_final, int) or health_timeout_final <= 0:
        logger.warn(f"Invalid healthCheckTimeout '{health_timeout_final}'. Using default: {DEFAULT_HEALTH_TIMEOUT_S_SCAN}s.")
        health_timeout_final = DEFAULT_HEALTH_TIMEOUT_S_SCAN
    logger.info(f"Effective health check timeout: {health_timeout_final}s")
    logger.info(f"Health check poll interval: {args.poll_interval}s")

    run_logs_dir = util.LOGS_DIR / f"{util.RUN_DIR_PREFIX}{ts}"
    run_logs_dir.mkdir(parents=True, exist_ok=True)
    TEMP_DIR_MANAGER_PATH = run_logs_dir
    logger.info(f"Server logs will be kept in: {run_logs_dir.resolve()}")
    logger.header("STARTING MEMORY SCAN RUN")

    processed_models = effective_conf.get("models", {})
    if not isinstance(processed_models, dict) or not processed_models:
        logger.error("'models' section not found/empty in config. Cannot run scan.")
        raise SystemExit(1)

    script_start_time = time.monotonic()
    run_memory_scan(
        processed_models_config=processed_models,
        output_csv_path=output_csv,
        health_timeout_s=health_timeout_final,
        health_poll_s=args.poll_interval,
        model_to_scan_alias=args.model,
        run_logs_dir=run_logs_dir,
        logger_instance=logger
    )

    # convenience latest copy
    try:
        latest_csv = util.RESULTS_DIR / f"{DEFAULT_OUTPUT_BASENAME}.csv"
        shutil.copy2(output_csv, latest_csv)
        logger.info(f"Also wrote latest CSV copy: {latest_csv}")
    except Exception as e:
        logger.warn(f"Could not write latest CSV copy: {e}")

    # rotate logs (keep last N)
    util.enforce_retention(util.LOGS_DIR, "scan_*.log",
                           keep=RETENTION_KEEP, delete_dirs=False, logger=logger)
    util.enforce_retention(util.LOGS_DIR, f"{util.RUN_DIR_PREFIX}*",
                           keep=RETENTION_KEEP, delete_dirs=True, logger=logger)

    # rotate results (keep last N)
    util.enforce_retention(util.RESULTS_DIR, f"{DEFAULT_OUTPUT_BASENAME}_*.csv",
                           keep=RETENTION_KEEP, delete_dirs=False, logger=logger)

    duration_s = time.monotonic() - script_start_time
    logger.header("MEMORY SCAN SCRIPT COMPLETE")
    logger.success(f"Total script execution time: {duration_s:.2f} seconds.")

    if hasattr(logger, "close") and callable(logger.close):
        logger.close()


if __name__ == "__main__":
    main()
