#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from colorama import Fore, Style

# Add src to path if needed when run directly
import sys, os
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if __package__ in (None, "") and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llama_suite.utils.config_utils import (  # type: ignore
    Logger, DEFAULT_HEALTH_POLL_INTERVAL_S, _dump_stderr_on_failure,
    start_llama_server, stop_llama_server, wait_for_server_health, color_status
)

from llama_suite.bench import util  # type: ignore

# --------- Constants ----------
DEFAULT_QUESTION = "Why is the sky blue?"
DEFAULT_HEALTH_TIMEOUT_S_BENCH = 120
API_REQUEST_TIMEOUT_S = 1200
STATIC_BENCHMARK_PORT = "9999"
RETENTION_KEEP = util.RETENTION_KEEP_DEFAULT

logger: Optional["Logger"] = None
TEMP_DIR_MANAGER_PATH: Optional[Path] = None


def _truthy_env(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def run_benchmark(
    processed_models_config: Dict[str, Any],
    output_csv_path: Path,
    question: str,
    health_timeout_s: int,
    health_poll_s: float,
    model_to_test_alias: Optional[str],
    run_logs_dir: Path,
    logger_instance: "Logger",
):
    def _effective_cfg_value(model_cfg: Dict[str, Any], key: str) -> str:
        cmd = model_cfg.get("cmd")
        if isinstance(cmd, dict) and key in cmd:
            v = cmd.get(key)
        else:
            v = model_cfg.get(key)
        if v is None or v == "" or v is False:
            return "-"
        return str(v)

    base_proxy_url = f"http://127.0.0.1:{STATIC_BENCHMARK_PORT}"
    health_url = f"{base_proxy_url}/health"
    api_url = f"{base_proxy_url}/v1/chat/completions"

    models_to_iterate = util.select_models(processed_models_config, model_to_test_alias, logger_instance)
    if not models_to_iterate:
        return

    headers = [
        "ModelName", "ParameterSize", "Quantization", "ContextSize",
        "GpuLayers", "CacheTypeK", "CacheTypeV", "NCpuMoe",
        "Timestamp",
        "MemoryScanStatus", "GpuMemoryGB", "CpuMemoryGB",
        "BenchStatus", "DurationSeconds", "TokensPerSecond",
        "PromptTokens", "CompletionTokens", "TotalTokens",
        "ProxyUrl", "Error",
    ]
    rows: List[List[str]] = []

    for i, (alias, model_cfg) in enumerate(models_to_iterate.items(), 1):
        # Web UI progress hook (handled by llama_suite.webui.utils.task_output)
        print(f"STEP {i}/{len(models_to_iterate)}: Benchmarking '{alias}'", flush=True)
        logger_instance.subheader(f"Model ({i}/{len(models_to_iterate)}): {alias}")
        ts = logger_instance._get_timestamp()

        # defaults
        param_size, quantization, ctx_size = "-", "-", "-"
        gpu_layers = _effective_cfg_value(model_cfg, "gpu-layers")
        cache_k = _effective_cfg_value(model_cfg, "cache-type-k")
        cache_v = _effective_cfg_value(model_cfg, "cache-type-v")
        n_cpu_moe = _effective_cfg_value(model_cfg, "n-cpu-moe")
        mem_gpu, mem_cpu, mem_status = "-", "-", "Not Scanned"
        bench_status, duration_s, tps = "Not Run", "", ""
        prompt_t, completion_t, total_t = "", "", ""
        err_msg = ""

        # static info
        logger_instance.step(f"Extracting static info for '{alias}'")
        param_size = util.parse_param_size_from_alias(alias, logger_instance)
        model_file = (model_cfg.get("cmd") or {}).get("model")
        if isinstance(model_file, str) and model_file:
            quantization = util.parse_quant_from_string(Path(model_file).name, logger_instance)
            if quantization == "-":
                quantization = util.parse_quant_from_string(alias, logger_instance)
        else:
            quantization = util.parse_quant_from_string(alias, logger_instance)

        # build command
        logger_instance.step(f"Preparing server command for '{alias}'")
        try:
            server_exe, server_args, ctx_size = util.build_server_command(
                alias, model_cfg, static_port=STATIC_BENCHMARK_PORT, logger=logger_instance
            )
            if getattr(logger_instance, "plain", False):
                logger_instance.info(
                    f"  Model Details - Params: {param_size}, Quant: {quantization}, Context: {ctx_size}"
                )
            else:
                logger_instance.info(
                    f"  Model Details - Params: {Fore.CYAN}{param_size}{Style.RESET_ALL}, "
                    f"Quant: {Fore.CYAN}{quantization}{Style.RESET_ALL}, "
                    f"Context: {Fore.CYAN}{ctx_size}{Style.RESET_ALL}"
                )
        except Exception as e:
            err_msg = f"Cmd Prepare Error: {e}"
            logger_instance.error(f"  {err_msg}")
            rows.append([
                alias,
                param_size,
                quantization,
                ctx_size,
                gpu_layers,
                cache_k,
                cache_v,
                n_cpu_moe,
                ts,
                "Config/Error",
                "-",
                "-",
                "Cmd Prepare Error",
                "",
                "",
                "",
                "",
                "",
                base_proxy_url,
                err_msg,
            ])
            logger_instance.notice("-" * 30)
            continue

        # start -> health -> memory -> bench
        logger_instance.step(f"Starting server, scanning memory, benchmarking: '{alias}'")
        server_proc = None
        stderr_log_path: Optional[Path] = None

        try:
            info = start_llama_server(
                executable_path_str=server_exe,
                arguments_list=server_args,
                model_name=alias,
                temp_dir=run_logs_dir,
                logger_instance=logger_instance,
                project_root_for_resolution=util.PROJECT_ROOT,
            )
            if not info:
                mem_status = bench_status = "Server Start Failed"
                err_msg = "Server failed to start"
                raise RuntimeError(err_msg)

            server_proc, _, stderr_log_path = info

            healthy = wait_for_server_health(
                process=server_proc,
                health_check_url=health_url,
                timeout_s=health_timeout_s,
                poll_interval_s=health_poll_s,
                model_name=alias,
                logger_instance=logger_instance,
            )
            if not healthy:
                if server_proc.poll() is not None:
                    code = server_proc.returncode
                    mem_status = bench_status = f"Server Exited (Code: {code})"
                    err_msg = f"Server exited (code {code}) before/during health check."
                    if stderr_log_path:
                        _dump_stderr_on_failure(stderr_log_path, alias, logger_instance)
                else:
                    mem_status = bench_status = "Health Timeout"
                    err_msg = "Server did not become healthy (timed out)."
                raise RuntimeError(err_msg)

            logger_instance.info("  Server healthy. Parsing memory usage...")
            if server_proc.poll() is None and stderr_log_path:
                mem_gpu, mem_cpu, mem_status = util.parse_memory_from_log(stderr_log_path, alias, logger_instance)
                logger_instance.info(
                    f"  Memory: GPU {Fore.CYAN}{mem_gpu} GB{Style.RESET_ALL}, "
                    f"CPU {Fore.CYAN}{mem_cpu} GB{Style.RESET_ALL} - {color_status(mem_status)}"
                )
            elif stderr_log_path:
                code = server_proc.returncode if server_proc.returncode is not None else "Unknown"
                mem_status = bench_status = f"Server Exited Prematurely (Code: {code})"
                _dump_stderr_on_failure(stderr_log_path, alias, logger_instance)
                raise RuntimeError("Server died before memory parsing")
            else:
                mem_status = "Error (No Log Path)"

            # benchmark call
            if server_proc.poll() is None:
                payload = {
                    "model": alias,
                    "messages": [{"role": "user", "content": question}],
                    "temperature": 0.7,
                    "stream": False,
                }
                t0 = time.monotonic()
                try:
                    resp = requests.post(api_url, json=payload, timeout=API_REQUEST_TIMEOUT_S)
                    duration = time.monotonic() - t0
                    duration_s = f"{duration:.3f}"

                    if resp.status_code == 200:
                        try:
                            data = resp.json()
                            usage = data.get("usage", {})
                            p = int(usage.get("prompt_tokens", 0) or 0)
                            c = int(usage.get("completion_tokens", 0) or 0)
                            prompt_t, completion_t = str(p), str(c)
                            total_t = str(usage.get("total_tokens", p + c))
                            tps = f"{(c / duration):.2f}" if duration > 1e-6 and c > 0 else "0.00"
                            bench_status = "Success"
                            logger_instance.success("  Benchmark completed.")
                            if getattr(logger_instance, "plain", False):
                                logger_instance.info(
                                    f"    Duration {duration_s}s, Tokens P/C/T: {p}/{c}/{total_t}, TPS {tps}"
                                )
                            else:
                                logger_instance.info(
                                    f"    Duration {Fore.CYAN}{duration_s}s{Style.RESET_ALL}, "
                                    f"Tokens P/C/T: {p}/{c}/{total_t}, TPS {Fore.CYAN}{tps}{Style.RESET_ALL}"
                                )
                        except Exception as e:
                            bench_status = "API Response Error"
                            em = f"JSON parse error: {e}"
                            err_msg = f"{err_msg}; {em}" if err_msg else em
                            logger_instance.error(f"  {em}")
                    else:
                        bench_status = "API Request Failed"
                        part = resp.text[:200].replace("\n", " ")
                        em = f"HTTP {resp.status_code}: {part}"
                        err_msg = f"{err_msg}; {em}" if err_msg else em
                        logger_instance.error(f"  {em}")
                except requests.Timeout:
                    duration_s = f"{time.monotonic() - t0:.3f}"
                    bench_status = "API Request Failed"
                    em = "Request timed out"
                    err_msg = f"{err_msg}; {em}" if err_msg else em
                    logger_instance.error(f"  {em}")
                except requests.RequestException as e:
                    duration_s = f"{time.monotonic() - t0:.3f}"
                    bench_status = "API Request Failed"
                    em = f"Request Exception: {e}"
                    err_msg = f"{err_msg}; {em}" if err_msg else em
                    logger_instance.error(f"  {em}")

        except Exception as e:
            logger_instance.error(f"Unexpected error while processing '{alias}': {e}")
            if mem_status == "Not Scanned":
                mem_status = "Script Error"
            if bench_status == "Not Run":
                bench_status = "Script Error"
        finally:
            if server_proc:
                stop_llama_server(server_proc, alias, logger_instance)

        rows.append([
            alias,
            param_size,
            quantization,
            ctx_size,
            gpu_layers,
            cache_k,
            cache_v,
            n_cpu_moe,
            ts,
            mem_status, mem_gpu, mem_cpu,
            bench_status, duration_s, tps, prompt_t, completion_t, total_t,
            base_proxy_url, (err_msg.strip("; ") if err_msg else ""),
        ])
        logger_instance.notice("-" * 30)

    util.write_csv(headers, rows, output_csv_path, logger_instance)


def main():
    global logger, TEMP_DIR_MANAGER_PATH

    p = argparse.ArgumentParser(
        description="Cross-platform LLM Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-c", "--config", type=Path, default=util.CONFIGS_DIR / "config.base.yaml")
    p.add_argument("--override", type=Path, default=util.default_override_for_hostname())
    p.add_argument("-q", "--question", type=str, default=DEFAULT_QUESTION)
    p.add_argument("--poll-interval", type=float, default=DEFAULT_HEALTH_POLL_INTERVAL_S)
    # If provided explicitly, always overrides config's healthCheckTimeout.
    p.add_argument("--health-timeout", type=int, default=None)
    p.add_argument("-m", "--model", type=str)
    p.add_argument("-v", "--verbose", action="store_true")
    plain_default = _truthy_env(os.getenv("LLAMA_SUITE_PLAIN", ""))
    p.add_argument("--plain", dest="plain", action="store_true", default=plain_default, help="Plain output (no colors/timestamps).")
    p.add_argument("--no-plain", dest="plain", action="store_false", help="Disable plain output.")
    args = p.parse_args()
    if args.plain:
        os.environ["LLAMA_SUITE_PLAIN"] = "1"

    util.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    util.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ts = util.timestamp_str()
    main_log = util.LOGS_DIR / f"benchmark_{ts}.log"
    logger = Logger(verbose=args.verbose, log_file_path=main_log, plain=args.plain)
    logger.header("LLM BENCHMARKER INITIALIZATION")
    logger.info(f"Log file: {main_log}")

    util.register_signal_handlers("benchmark", lambda: logger, lambda: TEMP_DIR_MANAGER_PATH)

    base_config = args.config
    if not base_config.is_file():
        logger.error(f"Base config not found: {base_config}")
        raise SystemExit(1)
    logger.info(f"Using base config: {base_config}")

    override = args.override
    if override and not override.is_file():
        logger.warn(f"Override not found: {override}. Ignoring.")
        override = None
    logger.info("Override: " + (str(override) if override else "None"))

    util.kill_lingering_servers_on_port(STATIC_BENCHMARK_PORT, logger)

    logger.step("Loading and processing configurations...")
    try:
        effective = util.load_and_process_config(base_config, override, verbose=args.verbose, logger=logger)
        logger.success("Configurations processed successfully.")
    except Exception as e:
        logger.error(f"Config processing failed: {e}")
        raise SystemExit(1)

    health_timeout_final = (
        args.health_timeout
        if args.health_timeout is not None
        else effective.get("healthCheckTimeout", DEFAULT_HEALTH_TIMEOUT_S_BENCH)
    )
    if not isinstance(health_timeout_final, int) or health_timeout_final <= 0:
        logger.warn(f"Invalid healthCheckTimeout '{health_timeout_final}', reverting to default {DEFAULT_HEALTH_TIMEOUT_S_BENCH}s.")
        health_timeout_final = DEFAULT_HEALTH_TIMEOUT_S_BENCH
    logger.info(f"Effective health timeout: {health_timeout_final}s")
    logger.info(f"Health poll interval: {args.poll_interval}s")

    run_logs_dir = util.LOGS_DIR / f"{util.RUN_DIR_PREFIX}{ts}"
    run_logs_dir.mkdir(parents=True, exist_ok=True)
    TEMP_DIR_MANAGER_PATH = run_logs_dir
    logger.info(f"Server logs will be kept in: {run_logs_dir}")

    models = effective.get("models", {})
    if not isinstance(models, dict) or not models:
        logger.error("'models' section not found/empty in config.")
        raise SystemExit(1)

    results_csv = util.RESULTS_DIR / f"benchmark_results_{ts}.csv"
    start = time.monotonic()
    run_benchmark(
        processed_models_config=models,
        output_csv_path=results_csv,
        question=args.question,
        health_timeout_s=health_timeout_final,
        health_poll_s=args.poll_interval,
        model_to_test_alias=args.model,
        run_logs_dir=run_logs_dir,
        logger_instance=logger,
    )
    latest = util.RESULTS_DIR / "benchmark_results.csv"
    try:
        shutil.copy2(results_csv, latest)
        logger.info(f"Also wrote latest CSV: {latest}")
    except Exception as e:
        logger.warn(f"Could not write latest CSV: {e}")

    # rotate logs (keep last N)
    util.enforce_retention(util.LOGS_DIR, "benchmark_*.log",
                           keep=RETENTION_KEEP, delete_dirs=False, logger=logger)
    util.enforce_retention(util.LOGS_DIR, f"{util.RUN_DIR_PREFIX}*",
                           keep=RETENTION_KEEP, delete_dirs=True, logger=logger)

    # rotate results (keep last N)
    util.enforce_retention(util.RESULTS_DIR, "benchmark_results_*.csv",
                           keep=RETENTION_KEEP, delete_dirs=False, logger=logger)

    elapsed = time.monotonic() - start
    logger.header("BENCHMARK COMPLETE")
    logger.success(f"Total time: {elapsed:.2f}s")
    if hasattr(logger, "close") and callable(logger.close):
        logger.close()


if __name__ == "__main__":
    main()
