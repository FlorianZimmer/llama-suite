"""Parameter sweeps for benchmark and memory scan tasks."""

from __future__ import annotations

from copy import deepcopy
import csv
import itertools
import json
from pathlib import Path
import sys
from typing import Any, Literal, Optional, Union

import yaml
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from llama_suite.utils.config_utils import ConfigEnvLoader, find_project_root
from llama_suite.webui.utils.mode import require_local_mode
from llama_suite.webui.utils.process_manager import process_manager
from llama_suite.webui.utils.task_output import handle_task_output
from llama_suite.webui.utils.ws_manager import manager as ws_manager


router = APIRouter(prefix="/api/sweeps", tags=["sweeps"])


def get_project_root() -> Path:
    return find_project_root()


def get_generated_sweeps_dir() -> Path:
    return get_project_root() / "configs" / "generated" / "sweeps"


def get_runs_sweeps_dir() -> Path:
    return get_project_root() / "runs" / "sweeps"


def get_overrides_dir() -> Path:
    return get_project_root() / "configs" / "overrides"


def _safe_name(name: str) -> str:
    safe = "".join(c for c in name if c.isalnum() or c in "-_")
    return safe or "item"


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    data = yaml.load(text, Loader=ConfigEnvLoader) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return {str(k): v for k, v in data.items()}


class SweepRange(BaseModel):
    start: float
    end: float
    step: float = Field(..., gt=0)


class SweepDimension(BaseModel):
    """
    A single sweep dimension. `path` is relative to a model config, e.g.:
      ["cmd", "ctx-size"]
      ["cmd", "cache-type-k"]
      ["sampling", "temp"]
    """

    path: list[str] = Field(..., min_length=1)
    value_type: Literal["int", "float", "bool", "str"] = "str"
    values: Optional[list[Any]] = None
    range: Optional[SweepRange] = None


class SweepRunRequest(BaseModel):
    task_type: Literal["bench", "memory"]
    baseline_override: Optional[str] = None
    models: Union[list[str], Literal["ALL"]] = "ALL"
    filter_string: Optional[str] = None
    dimensions: list[SweepDimension] = Field(default_factory=list)
    health_timeout: int = 120
    question: Optional[str] = None  # bench only


def _expand_dimension_values(dim: SweepDimension) -> list[Any]:
    if dim.values is not None:
        return list(dim.values)
    if dim.range is None:
        return []

    start = dim.range.start
    end = dim.range.end
    step = dim.range.step
    out: list[Any] = []

    # Inclusive end (best-effort)
    v = start
    guard = 0
    while v <= end + 1e-12:
        guard += 1
        if guard > 100000:
            raise ValueError("Range expansion exceeded 100000 steps")
        if dim.value_type == "int":
            out.append(int(round(v)))
        elif dim.value_type == "bool":
            out.append(bool(v))
        elif dim.value_type == "float":
            out.append(float(v))
        else:
            out.append(str(v))
        v += step

    # De-dup while keeping order
    seen = set()
    dedup: list[Any] = []
    for item in out:
        key = json.dumps(item, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(item)
    return dedup


def generate_variants(dimensions: list[SweepDimension]) -> list[dict[str, Any]]:
    """
    Return a list of variants. Each variant is a dict mapping "path_str" -> value.
    path_str is dot-joined, e.g. "cmd.ctx-size" or "sampling.temp".
    """
    if not dimensions:
        return [{}]

    expanded: list[tuple[str, list[Any]]] = []
    for dim in dimensions:
        vals = _expand_dimension_values(dim)
        if not vals:
            raise ValueError(f"Dimension {'.'.join(dim.path)} has no values")
        path_str = ".".join(dim.path)
        expanded.append((path_str, vals))

    variants: list[dict[str, Any]] = []
    keys = [k for k, _ in expanded]
    value_lists = [v for _, v in expanded]
    for combo in itertools.product(*value_lists):
        variants.append({k: v for k, v in zip(keys, combo)})
    return variants


def _apply_variant_to_override(
    override_data: dict[str, Any],
    model_name: str,
    variant: dict[str, Any],
) -> dict[str, Any]:
    data = deepcopy(override_data)
    models = data.setdefault("models", {})
    if not isinstance(models, dict):
        models = {}
        data["models"] = models
    model_cfg = models.setdefault(model_name, {})
    if not isinstance(model_cfg, dict):
        model_cfg = {}
        models[model_name] = model_cfg

    for path_str, value in variant.items():
        parts = [p for p in str(path_str).split(".") if p]
        if not parts:
            continue
        cur: Any = model_cfg
        for key in parts[:-1]:
            nxt = cur.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[key] = nxt
            cur = nxt
        cur[parts[-1]] = value

    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(
        data,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    path.write_text(text, encoding="utf-8")


def _newest_csv(dir_path: Path, pattern: str, *, exclude: Optional[str] = None) -> Optional[Path]:
    if not dir_path.exists():
        return None
    files = [p for p in dir_path.glob(pattern) if p.is_file() and (exclude is None or p.name != exclude)]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


@router.post("/run")
async def start_sweep(request: SweepRunRequest, _=Depends(require_local_mode)):
    async def run_sweep_task(task_id: str, **kwargs):
        root = get_project_root()
        venv_python = root / ".venv" / ("Scripts" if sys.platform == "win32" else "bin") / "python"
        if not venv_python.exists():
            venv_python = Path(sys.executable)

        base_config = root / "configs" / "config.base.yaml"
        if not base_config.exists():
            raise RuntimeError(f"Base config not found: {base_config}")

        task_type = str(kwargs.get("task_type") or "bench")
        health_timeout = int(kwargs.get("health_timeout") or 120)
        question = str(kwargs.get("question") or "What is the capital of France?")

        baseline_override_name = (kwargs.get("baseline_override") or None)
        baseline_override_path: Optional[Path] = None
        baseline_override_data: dict[str, Any] = {}
        if baseline_override_name:
            baseline_override_name = str(baseline_override_name).strip()
            safe_override = _safe_name(baseline_override_name)
            if not safe_override or safe_override != baseline_override_name:
                raise RuntimeError("Invalid baseline override name")
            baseline_override_path = get_overrides_dir() / f"{safe_override}.yaml"
            if not baseline_override_path.exists():
                raise RuntimeError(f"Baseline override not found: {safe_override}")
            baseline_override_data = _load_yaml_dict(baseline_override_path)

        base = _load_yaml_dict(base_config)
        base_models = base.get("models", {})
        if not isinstance(base_models, dict):
            raise RuntimeError("Base config missing 'models' mapping")

        # Resolve models list
        models_raw = kwargs.get("models", "ALL")
        if models_raw == "ALL":
            model_names = [str(k) for k in base_models.keys()]
        else:
            model_names = [str(m) for m in (models_raw or [])]

        filter_string = str(kwargs.get("filter_string") or "").strip()
        if filter_string:
            needle = filter_string.lower()
            model_names = [m for m in model_names if needle in m.lower()]

        model_names = sorted(set(model_names), key=lambda s: s.lower())
        if not model_names:
            raise RuntimeError("No models selected for sweep")

        dims = [SweepDimension(**d) if isinstance(d, dict) else d for d in (kwargs.get("dimensions") or [])]
        variants = generate_variants(dims)

        total_runs = len(model_names) * len(variants)
        await ws_manager.send_progress(task_id, 0, f"Starting sweep: {len(model_names)} model(s) × {len(variants)} variant(s) = {total_runs} runs")

        generated_dir = get_generated_sweeps_dir() / task_id
        runs_dir = get_runs_sweeps_dir() / task_id
        generated_dir.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = runs_dir / "manifest.json"
        results_path = runs_dir / "results.csv"

        # Determine where result CSVs land (from existing scripts)
        bench_results_dir = root / "runs" / "bench" / "results"
        mem_results_dir = root / "runs" / "bench" / "results"
        results_dir = bench_results_dir if task_type == "bench" else mem_results_dir

        results_rows: list[dict[str, Any]] = []
        completed = 0
        failed = 0

        for model_idx, model_name in enumerate(model_names, 1):
            for var_idx, variant in enumerate(variants, 1):
                task = process_manager.get_task(task_id)
                if task and task.status == "cancelled":
                    await ws_manager.send_progress(task_id, task.progress, "Sweep cancelled", status="cancelled")
                    break

                run_number = completed + failed + 1
                pct = (run_number - 1) / max(1, total_runs) * 100.0
                variant_label = ", ".join([f"{k}={variant[k]}" for k in sorted(variant.keys())]) or "(baseline)"
                await ws_manager.send_progress(
                    task_id,
                    pct,
                    f"Sweep run {run_number}/{total_runs}: {model_name} • {variant_label}",
                )

                # Create per-run override
                merged_override = _apply_variant_to_override(baseline_override_data, model_name, variant)
                out_override_path = (
                    generated_dir
                    / "overrides"
                    / _safe_name(model_name)
                    / f"v{var_idx:04d}.yaml"
                )
                _write_yaml(out_override_path, merged_override)

                # Observe new CSV outputs
                before_files = set(results_dir.glob("*.csv")) if results_dir.exists() else set()

                if task_type == "bench":
                    script = root / "src" / "llama_suite" / "bench" / "benchmark-models.py"
                    cmd = [
                        str(venv_python),
                        "-u",
                        str(script),
                        "--config",
                        str(base_config),
                        "--override",
                        str(out_override_path),
                        "--model",
                        model_name,
                        "--question",
                        question,
                        "--health-timeout",
                        str(health_timeout),
                        "--plain",
                    ]
                else:
                    script = root / "src" / "llama_suite" / "bench" / "scan_model_memory.py"
                    cmd = [
                        str(venv_python),
                        "-u",
                        str(script),
                        "--config",
                        str(base_config),
                        "--override",
                        str(out_override_path),
                        "--model",
                        model_name,
                        "--health-timeout",
                        str(health_timeout),
                    ]

                async def on_output(line: str):
                    await handle_task_output(ws_manager, task_id, line, is_stderr=False, progress_style="none")

                async def on_error(line: str):
                    await handle_task_output(ws_manager, task_id, line, is_stderr=True, progress_style="none")

                returncode = await process_manager.run_subprocess(
                    task_id,
                    cmd,
                    cwd=root,
                    on_stdout=on_output,
                    on_stderr=on_error,
                )

                # Pick up the newly created CSV (best-effort)
                after_files = set(results_dir.glob("*.csv")) if results_dir.exists() else set()
                new_files = [p for p in after_files - before_files if p.is_file()]
                newest = None
                if new_files:
                    newest = max(new_files, key=lambda p: p.stat().st_mtime)
                else:
                    # Fallback to latest timestamped file (exclude latest copy)
                    if task_type == "bench":
                        newest = _newest_csv(results_dir, "benchmark_results_*.csv", exclude="benchmark_results.csv")
                    else:
                        newest = _newest_csv(results_dir, "memory_scan_results_*.csv", exclude="memory_scan_results.csv")

                row: dict[str, Any] = {
                    "task_id": task_id,
                    "task_type": task_type,
                    "baseline_override": baseline_override_name or "",
                    "model": model_name,
                    "variant_index": var_idx,
                    "variant_label": variant_label,
                    "override_path": str(out_override_path),
                    "returncode": returncode,
                }
                for k, v in sorted(variant.items()):
                    row[f"param.{k}"] = v

                if returncode == 0:
                    completed += 1
                else:
                    failed += 1

                if newest and newest.exists():
                    try:
                        rows = _read_csv_rows(newest)
                        pick = next((r for r in rows if r.get("ModelName") == model_name), rows[0] if rows else {})
                        for k, v in pick.items():
                            row[f"metric.{k}"] = v
                        row["results_csv"] = str(newest)
                    except Exception as e:
                        row["parse_error"] = str(e)
                else:
                    row["results_csv"] = ""

                results_rows.append(row)

            task = process_manager.get_task(task_id)
            if task and task.status == "cancelled":
                break

        # Persist sweep artifacts
        manifest = {
            "task_id": task_id,
            "task_type": task_type,
            "baseline_override": baseline_override_name,
            "models": model_names,
            "dimensions": [d.model_dump() for d in dims],
            "variants": variants,
            "total_runs": total_runs,
            "completed": completed,
            "failed": failed,
            "generated_dir": str(generated_dir),
            "results_path": str(results_path),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        # Write CSV (flat)
        fieldnames: list[str] = []
        for r in results_rows:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(results_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in results_rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})

        success = failed == 0 and completed > 0
        await ws_manager.send_complete(task_id, success, {"results_path": str(results_path), "manifest_path": str(manifest_path)})
        return {
            "success": success,
            "completed": completed,
            "failed": failed,
            "total_runs": total_runs,
            "results_path": str(results_path),
            "manifest_path": str(manifest_path),
        }

    task_id = await process_manager.start_task(
        task_type="sweep",
        description=f"Sweep ({request.task_type})",
        coro=run_sweep_task,
        **request.model_dump(),
    )

    return {"task_id": task_id, "status": "started"}


@router.post("/cancel/{task_id}")
async def cancel_sweep(task_id: str):
    cancelled = await process_manager.cancel_task(task_id)
    if cancelled:
        return {"status": "cancelled", "task_id": task_id}
    raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found or already completed")


@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    task = process_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "description": task.description,
        "status": task.status,
        "progress": task.progress,
        "started_at": task.started_at.isoformat(),
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "result": task.result,
        "error": task.error,
        "log_count": len(task.logs),
    }


@router.get("/task/{task_id}/logs")
async def get_task_logs(task_id: str, offset: int = 0, limit: int = 200):
    task = process_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    logs = task.logs[offset : offset + limit]
    return {"task_id": task_id, "total": len(task.logs), "offset": offset, "logs": logs}


@router.get("/task/{task_id}/results")
async def get_sweep_results(task_id: str, offset: int = 0, limit: int = 200):
    task = process_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    results_path = None
    if isinstance(task.result, dict):
        results_path = task.result.get("results_path")
    if not results_path:
        raise HTTPException(status_code=404, detail="Sweep results not available yet")
    path = Path(str(results_path))
    if not path.exists():
        raise HTTPException(status_code=404, detail="Sweep results file missing")
    rows = _read_csv_rows(path)
    sliced = rows[offset : offset + limit]
    return {"task_id": task_id, "total": len(rows), "offset": offset, "rows": sliced}
