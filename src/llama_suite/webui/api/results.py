"""Results API routes for viewing benchmark and evaluation results."""

from pathlib import Path
from typing import Optional
import json
import csv

from fastapi import APIRouter, Depends, HTTPException

from llama_suite.utils.config_utils import find_project_root
from llama_suite.webui.utils.mode import require_not_read_only


router = APIRouter(prefix="/api/results", tags=["results"])


def get_project_root() -> Path:
    """Get the project root directory."""
    return find_project_root()


def get_runs_dir() -> Path:
    """Get the runs directory."""
    return get_project_root() / "runs"


@router.get("")
async def list_result_types():
    """List available result types (bench, eval)."""
    runs_dir = get_runs_dir()
    types = []
    
    for subdir in ["bench", "eval"]:
        path = runs_dir / subdir
        if path.exists() and path.is_dir():
            count = len(list(path.iterdir()))
            types.append({
                "type": subdir,
                "path": str(path),
                "run_count": count
            })
    
    return {"types": types}


@router.get("/bench")
async def list_bench_results():
    """List benchmark run results."""
    bench_dir = get_runs_dir() / "bench"
    if not bench_dir.exists():
        return {"runs": []}
    
    runs = []
    for run_dir in sorted(bench_dir.iterdir(), reverse=True):
        if run_dir.is_dir():
            # Look for CSV results
            csv_files = list(run_dir.glob("*.csv"))
            runs.append({
                "name": run_dir.name,
                "path": str(run_dir),
                "csv_files": [f.name for f in csv_files]
            })
    
    return {"runs": runs}


@router.get("/bench/merged")
async def get_merged_bench_results():
    """Get all benchmark results merged from all CSV files in the results folder."""
    results_dir = get_runs_dir() / "bench" / "results"
    if not results_dir.exists():
        return {"results": [], "files": []}
    
    all_results = []
    files_processed = []
    
    # Find all timestamped benchmark CSV files (exclude benchmark_results.csv "latest" copy)
    for csv_file in sorted(results_dir.glob("benchmark_results_*.csv"), reverse=True):
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Skip empty rows
                    if not row.get("ModelName"):
                        continue
                    # Add source file info
                    row["_source_file"] = csv_file.name
                    # Parse numeric fields
                    row["_tokens_per_second"] = _parse_float(row.get("TokensPerSecond"))
                    row["_gpu_memory_gb"] = _parse_float(row.get("GpuMemoryGB"))
                    row["_cpu_memory_gb"] = _parse_float(row.get("CpuMemoryGB"))
                    row["_duration_seconds"] = _parse_float(row.get("DurationSeconds"))
                    row["_context_size"] = _parse_int(row.get("ContextSize"))
                    row["_completion_tokens"] = _parse_int(row.get("CompletionTokens"))
                    row["_gpu_layers"] = _parse_int(row.get("GpuLayers"))
                    row["_n_cpu_moe"] = _parse_int(row.get("NCpuMoe"))
                    ct_k = row.get("CacheTypeK") or "-"
                    ct_v = row.get("CacheTypeV") or "-"
                    row["_kv_cache"] = f"{ct_k}/{ct_v}"
                    all_results.append(row)
            files_processed.append(csv_file.name)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    return {
        "results": all_results,
        "files": files_processed,
        "total_count": len(all_results)
    }


@router.get("/memory/merged")
async def get_merged_memory_results():
    """Get all memory scan results merged from all CSV files."""
    results_dir = get_runs_dir() / "bench" / "results"
    if not results_dir.exists():
        return {"results": [], "files": []}
    
    all_results = []
    files_processed = []
    
    # Find all timestamped memory scan CSV files (exclude memory_scan_results.csv "latest" copy)
    for csv_file in sorted(results_dir.glob("memory_scan_results_*.csv"), reverse=True):
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row.get("ModelName"):
                        continue
                    row["_source_file"] = csv_file.name
                    row["_gpu_memory_gb"] = _parse_float(row.get("GpuMemoryGB"))
                    row["_cpu_memory_gb"] = _parse_float(row.get("CpuMemoryGB"))
                    row["_context_size"] = _parse_int(row.get("ContextSize"))
                    row["_gpu_layers"] = _parse_int(row.get("GpuLayers"))
                    row["_n_cpu_moe"] = _parse_int(row.get("NCpuMoe"))
                    ct_k = row.get("CacheTypeK") or "-"
                    ct_v = row.get("CacheTypeV") or "-"
                    row["_kv_cache"] = f"{ct_k}/{ct_v}"
                    all_results.append(row)
            files_processed.append(csv_file.name)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    return {
        "results": all_results,
        "files": files_processed,
        "total_count": len(all_results)
    }


@router.get("/eval/merged")
async def get_merged_eval_results():
    """Get all evaluation results merged from all run directories."""
    eval_dir = get_runs_dir() / "eval"
    if not eval_dir.exists():
        return {"results": [], "runs": []}
    
    all_results = []
    runs_processed = []
    
    # Check each eval run subdirectory
    for run_dir in sorted(eval_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        
        # Check results subfolder or root for summary files
        results_subdir = run_dir / "results"
        search_dirs = [results_subdir, run_dir] if results_subdir.exists() else [run_dir]
        
        for search_dir in search_dirs:
            # Look for summary JSON files
            for json_file in search_dir.glob("summary*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    # Handle different summary formats
                    if isinstance(data, dict):
                        for model_name, metrics in data.items():
                            if isinstance(metrics, dict):
                                result = {
                                    "ModelName": model_name,
                                    "RunName": run_dir.name,
                                    "_source_file": f"{run_dir.name}/{json_file.name}",
                                    **metrics
                                }
                                all_results.append(result)
                except Exception as e:
                    print(f"Error reading {json_file}: {e}")
            
            # Also look for scores CSV
            for csv_file in search_dir.glob("scores*.csv"):
                try:
                    with open(csv_file, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            row["RunName"] = run_dir.name
                            row["_source_file"] = f"{run_dir.name}/{csv_file.name}"
                            all_results.append(row)
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")
        
        runs_processed.append(run_dir.name)
    
    return {
        "results": all_results,
        "runs": runs_processed,
        "total_count": len(all_results)
    }


def _parse_float(value) -> Optional[float]:
    """Safely parse a float value."""
    if value is None or value == "" or value == "-":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _parse_int(value) -> Optional[int]:
    """Safely parse an int value."""
    if value is None or value == "" or value == "-":
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


@router.get("/bench/{run_name}")
async def get_bench_result(run_name: str):
    """Get details of a specific benchmark run."""
    run_dir = get_runs_dir() / "bench" / run_name
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Benchmark run '{run_name}' not found")
    
    result = {
        "name": run_name,
        "path": str(run_dir),
        "files": [],
        "data": {}
    }
    
    # List all files
    for f in run_dir.iterdir():
        result["files"].append({
            "name": f.name,
            "size_bytes": f.stat().st_size if f.is_file() else None,
            "is_dir": f.is_dir()
        })
    
    # Parse CSV files
    for csv_file in run_dir.glob("*.csv"):
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                result["data"][csv_file.name] = list(reader)
        except Exception as e:
            result["data"][csv_file.name] = {"error": str(e)}
    
    return result


@router.get("/eval")
async def list_eval_results():
    """List evaluation run results."""
    eval_dir = get_runs_dir() / "eval"
    if not eval_dir.exists():
        return {"runs": []}
    
    runs = []
    for run_dir in sorted(eval_dir.iterdir(), reverse=True):
        if run_dir.is_dir():
            # Look for JSON results
            json_files = list(run_dir.glob("*.json"))
            csv_files = list(run_dir.glob("*.csv"))
            runs.append({
                "name": run_dir.name,
                "path": str(run_dir),
                "json_files": [f.name for f in json_files],
                "csv_files": [f.name for f in csv_files]
            })
    
    return {"runs": runs}


@router.get("/eval/{run_name}")
async def get_eval_result(run_name: str):
    """Get details of a specific evaluation run."""
    run_dir = get_runs_dir() / "eval" / run_name
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Evaluation run '{run_name}' not found")
    
    result = {
        "name": run_name,
        "path": str(run_dir),
        "files": [],
        "data": {}
    }
    
    # List all files
    for f in run_dir.iterdir():
        result["files"].append({
            "name": f.name,
            "size_bytes": f.stat().st_size if f.is_file() else None,
            "is_dir": f.is_dir()
        })
    
    # Parse JSON files
    for json_file in run_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                result["data"][json_file.name] = json.load(f)
        except Exception as e:
            result["data"][json_file.name] = {"error": str(e)}
    
    # Parse CSV files
    for csv_file in run_dir.glob("*.csv"):
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                result["data"][csv_file.name] = list(reader)
        except Exception as e:
            result["data"][csv_file.name] = {"error": str(e)}
    
    return result


@router.delete("/bench/{run_name}")
async def delete_bench_result(run_name: str, _=Depends(require_not_read_only)):
    """Delete a benchmark run result."""
    import shutil
    run_dir = get_runs_dir() / "bench" / run_name
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Benchmark run '{run_name}' not found")
    
    shutil.rmtree(run_dir)
    return {"status": "ok", "message": f"Deleted benchmark run '{run_name}'"}


@router.delete("/eval/{run_name}")
async def delete_eval_result(run_name: str, _=Depends(require_not_read_only)):
    """Delete an evaluation run result."""
    import shutil
    run_dir = get_runs_dir() / "eval" / run_name
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Evaluation run '{run_name}' not found")
    
    shutil.rmtree(run_dir)
    return {"status": "ok", "message": f"Deleted evaluation run '{run_name}'"}
