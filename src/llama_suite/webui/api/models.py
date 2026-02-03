"""Models API routes for managing configured models."""

from pathlib import Path
from typing import Optional, List
import yaml

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from llama_suite.utils.config_utils import find_project_root, generate_processed_config
from llama_suite.webui.utils.paths import get_base_config_path
from llama_suite.webui.utils.mode import require_not_read_only


router = APIRouter(prefix="/api/models", tags=["models"])


MANDATORY_CMD_KEYS = {"bin", "port", "model", "ctx-size"}


def get_project_root() -> Path:
    """Get the project root directory."""
    return find_project_root()


def get_configs_dir() -> Path:
    """Get the configs directory."""
    return get_project_root() / "configs"


def get_models_dir() -> Path:
    """Get the models directory."""
    return get_project_root() / "models"


def _coerce_gpu_layers(value) -> object:
    """
    The UI historically sends -1 for "auto". Keep the stored config compatible with command building.
    """
    if value is None:
        return "auto"
    try:
        if isinstance(value, str) and value.strip().lower() == "auto":
            return "auto"
        if int(value) <= 0:
            return "auto"
    except Exception:
        pass
    return value


def _default_cmd_template(config: dict) -> dict:
    """
    Pick sane defaults for new model entries.

    Prefer an existing model's cmd keys for `bin` and `port` to match the user's setup;
    otherwise fall back to the repo defaults used in `configs/config.base.yaml`.
    """
    models = config.get("models", {})
    if isinstance(models, dict):
        for _, model_cfg in models.items():
            if not isinstance(model_cfg, dict):
                continue
            cmd = model_cfg.get("cmd", {})
            if not isinstance(cmd, dict):
                continue
            if cmd.get("bin") and cmd.get("port"):
                return {"bin": cmd.get("bin"), "port": cmd.get("port")}

    return {"bin": "llama.cpp/build/bin/llama-server", "port": "${PORT}"}


def _validate_cmd_has_required_keys(model_name: str, cmd: dict) -> None:
    missing = sorted(k for k in MANDATORY_CMD_KEYS if not cmd.get(k))
    if missing:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}': cmd is missing required keys: {', '.join(missing)}")


def reanchor_from_configs(p: Path, project_root: Path) -> Path:
    """
    If a path ended up under <root>/configs/<X>/..., re-anchor it to <root>/<X>/...
    e.g. <root>/configs/models/foo.gguf -> <root>/models/foo.gguf
    This fixes paths that were incorrectly resolved relative to config file location.
    """
    try:
        p_abs = p.resolve()
    except Exception:
        p_abs = Path(str(p))
    
    cfg_dir = (project_root / "configs").resolve()
    try:
        rel = p_abs.relative_to(cfg_dir)
    except ValueError:
        # Path is not under configs/
        return p_abs
    
    # Only reanchor for folders we expect to live at project root
    first = rel.parts[0] if rel.parts else ""
    if first.lower() in {"models", "llama.cpp", "vendor"}:
        return (project_root / rel).resolve()
    return p_abs


@router.get("")
async def list_models(override: Optional[str] = None):
    """List all configured models with their properties."""
    base_path = get_base_config_path()
    override_path = None
    
    if override:
        override_path = get_configs_dir() / "overrides" / f"{override}.yaml"
        if not override_path.exists():
            raise HTTPException(status_code=404, detail=f"Override '{override}' not found")
    
    try:
        # generate_processed_config returns the effective config dict directly
        config = generate_processed_config(
            base_config_path_arg=base_path,
            override_config_path_arg=override_path,
            verbose_logging=False
        )
        models_config = config.get("models", {})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading config: {e}")
    
    models = []
    models_dir = get_models_dir()
    project_root = get_project_root()
    
    # Resolve models_dir symlink once
    try:
        resolved_models_dir = models_dir.resolve()
    except (OSError, ValueError):
        resolved_models_dir = models_dir
    
    for name, model_cfg in models_config.items():
        if not isinstance(model_cfg, dict):
            continue
            
        cmd = model_cfg.get("cmd", {})
        if not isinstance(cmd, dict):
            cmd = {}
        model_path_str = cmd.get("model", "")
        
        # Check if model file exists with multiple resolution strategies
        model_exists = False
        model_size = None
        resolved_path_str = ""
        
        if model_path_str:
            # Try multiple path resolution strategies
            candidates = []
            
            orig_path = Path(model_path_str)
            
            # 1. If absolute, first try reanchoring (fixes paths under configs/)
            if orig_path.is_absolute():
                reanchored = reanchor_from_configs(orig_path, project_root)
                candidates.append(reanchored)
                if reanchored != orig_path.resolve():
                    candidates.append(orig_path)
            else:
                # 2. Relative to project root
                candidates.append(project_root / model_path_str)
                # 3. Relative to resolved models dir (handles symlinks)
                candidates.append(resolved_models_dir / orig_path.name)
                # 4. Just the filename in models dir
                if orig_path.name != model_path_str:
                    candidates.append(resolved_models_dir / model_path_str)
            
            # Try each candidate
            for candidate in candidates:
                try:
                    resolved = candidate.resolve()
                    if resolved.exists() and resolved.is_file():
                        model_exists = True
                        model_size = resolved.stat().st_size
                        resolved_path_str = str(resolved)
                        break
                except (OSError, ValueError):
                    continue
        
        # Check if model is disabled
        is_disabled = model_cfg.get("disabled", False) or model_cfg.get("skip", False)
        
        models.append({
            "name": name,
            "hf_tokenizer": model_cfg.get("hf_tokenizer_for_model", ""),
            "model_path": model_path_str,
            "resolved_path": resolved_path_str,
            "model_exists": model_exists,
            "model_size_bytes": model_size,
            "ctx_size": cmd.get("ctx-size"),
            "gpu_layers": cmd.get("gpu-layers"),
            "threads": cmd.get("threads"),
            "sampling": model_cfg.get("sampling", {}),
            "aliases": model_cfg.get("aliases", []),
            "supports_no_think": model_cfg.get("supports_no_think_toggle", False),
            "has_draft_model": "model-draft" in cmd,
            "disabled": is_disabled
        })
    
    # Sort by name
    models.sort(key=lambda x: x["name"].lower())
    
    return {
        "count": len(models),
        "models": models
    }


@router.get("/{name}")
async def get_model(name: str, override: Optional[str] = None):
    """Get detailed configuration for a specific model."""
    base_path = get_base_config_path()
    override_path = None
    
    if override:
        override_path = get_configs_dir() / "overrides" / f"{override}.yaml"
        if not override_path.exists():
            raise HTTPException(status_code=404, detail=f"Override '{override}' not found")
    
    try:
        # generate_processed_config returns the effective config dict directly
        config = generate_processed_config(
            base_config_path_arg=base_path,
            override_config_path_arg=override_path,
            verbose_logging=False
        )
        models_config = config.get("models", {})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading config: {e}")
    
    if name not in models_config:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    
    model_cfg = models_config[name]
    cmd = model_cfg.get("cmd", {})
    
    return {
        "name": name,
        "config": model_cfg,
        "cmd": cmd
    }


@router.get("/files/available")
async def list_available_model_files():
    """List GGUF files in the models directory (handles symlinks)."""
    models_dir = get_models_dir()
    
    # Resolve symlink if models dir itself is a symlink
    try:
        resolved_models_dir = models_dir.resolve()
    except (OSError, ValueError):
        resolved_models_dir = models_dir
    
    if not resolved_models_dir.exists():
        return {"files": [], "models_dir": str(models_dir), "is_symlink": models_dir.is_symlink()}
    
    files = []
    seen_paths = set()
    
    # Search for GGUF files recursively (to handle subdirs)
    for path in sorted(resolved_models_dir.rglob("*.gguf")):
        try:
            resolved = path.resolve()
            if str(resolved) in seen_paths:
                continue
            seen_paths.add(str(resolved))
            
            # Get relative path from models dir
            try:
                rel_path = path.relative_to(resolved_models_dir)
            except ValueError:
                rel_path = path.name

            rel_path_str = str(rel_path)
            rel_path_posix = rel_path_str.replace("\\", "/")
            files.append({
                "name": path.name,
                "relative_path": rel_path_str,
                "config_path": f"./models/{rel_path_posix}",
                "path": str(path),
                "size_bytes": resolved.stat().st_size
            })
        except (OSError, ValueError):
            continue
    
    return {
        "files": files, 
        "models_dir": str(models_dir),
        "resolved_dir": str(resolved_models_dir),
        "is_symlink": models_dir.is_symlink()
    }


@router.post("/files/upload")
async def upload_model_file(
    file: UploadFile = File(...),
    subfolder: str = Form(default=""),
    _=Depends(require_not_read_only),
):
    """Upload a GGUF model file to the models directory."""
    if not file.filename.endswith(".gguf"):
        raise HTTPException(status_code=400, detail="Only .gguf files are allowed")
    
    models_dir = get_models_dir()
    
    # Resolve symlink if needed
    try:
        resolved_models_dir = models_dir.resolve()
    except (OSError, ValueError):
        resolved_models_dir = models_dir
    
    # Create target directory
    if subfolder:
        target_dir = resolved_models_dir / subfolder
    else:
        target_dir = resolved_models_dir
    
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / file.filename
    
    # Check if file already exists
    if target_path.exists():
        raise HTTPException(status_code=409, detail=f"File '{file.filename}' already exists")
    
    # Save file
    try:
        with open(target_path, "wb") as f:
            # Read in chunks to handle large files
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                f.write(chunk)
    except Exception as e:
        # Clean up partial file
        if target_path.exists():
            target_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    
    relative_path = str(target_path.relative_to(resolved_models_dir)).replace("\\", "/")
    return {
        "status": "ok",
        "filename": file.filename,
        "path": str(target_path),
        "relative_path": relative_path,
        "config_path": f"./models/{relative_path}",
        "size_bytes": target_path.stat().st_size
    }


class ModelDisableRequest(BaseModel):
    """Request to enable/disable a model."""
    disabled: bool


@router.post("/{name}/toggle")
async def toggle_model(name: str, request: ModelDisableRequest, _=Depends(require_not_read_only)):
    """Enable or disable a model by updating the base config."""
    base_path = get_base_config_path()
    
    if not base_path.exists():
        raise HTTPException(status_code=404, detail="Base config not found")
    
    content = base_path.read_text(encoding="utf-8")
    config = yaml.safe_load(content)
    
    models = config.get("models", {})
    if name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    
    # Update the disabled flag
    if request.disabled:
        models[name]["disabled"] = True
    else:
        models[name].pop("disabled", None)
        models[name].pop("skip", None)
    
    # Write back
    backup_path = base_path.with_suffix(".yaml.bak")
    backup_path.write_text(content, encoding="utf-8")
    
    with open(base_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    return {"status": "ok", "disabled": request.disabled}


class ModelUpdateRequest(BaseModel):
    """Request to update a model's configuration."""
    cmd: Optional[dict] = None
    sampling: Optional[dict] = None
    hf_tokenizer_for_model: Optional[str] = None
    aliases: Optional[List[str]] = None
    disabled: Optional[bool] = None


@router.put("/{name}")
async def update_model(name: str, request: ModelUpdateRequest, _=Depends(require_not_read_only)):
    """Update a model's configuration in the base config."""
    base_path = get_base_config_path()
    
    if not base_path.exists():
        raise HTTPException(status_code=404, detail="Base config not found")
    
    content = base_path.read_text(encoding="utf-8")
    config = yaml.safe_load(content)
    
    models = config.get("models", {})
    if name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    
    model_cfg = models[name]
    
    # Update cmd parameters
    if request.cmd is not None:
        if "cmd" not in model_cfg:
            model_cfg["cmd"] = {}
        for key, value in request.cmd.items():
            if value is None:
                if key in MANDATORY_CMD_KEYS:
                    raise HTTPException(status_code=400, detail=f"Model '{name}': cannot remove required cmd key '{key}'")
                model_cfg["cmd"].pop(key, None)
            else:
                if key == "gpu-layers":
                    model_cfg["cmd"][key] = _coerce_gpu_layers(value)
                else:
                    model_cfg["cmd"][key] = value

        _validate_cmd_has_required_keys(name, model_cfg["cmd"])
    
    # Update sampling parameters
    if request.sampling is not None:
        if "sampling" not in model_cfg:
            model_cfg["sampling"] = {}
        for key, value in request.sampling.items():
            if value is None:
                model_cfg["sampling"].pop(key, None)
            else:
                model_cfg["sampling"][key] = value
    
    # Update tokenizer
    if request.hf_tokenizer_for_model is not None:
        model_cfg["hf_tokenizer_for_model"] = request.hf_tokenizer_for_model
    
    # Update aliases
    if request.aliases is not None:
        model_cfg["aliases"] = request.aliases
    
    # Update disabled
    if request.disabled is not None:
        if request.disabled:
            model_cfg["disabled"] = True
        else:
            model_cfg.pop("disabled", None)
            model_cfg.pop("skip", None)
    
    # Write back
    backup_path = base_path.with_suffix(".yaml.bak")
    backup_path.write_text(content, encoding="utf-8")
    
    with open(base_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    return {"status": "ok", "model": name}


class ModelCopyRequest(BaseModel):
    """Request to copy parameters from one model to others."""
    target_models: List[str]
    copy_cmd: bool = True
    copy_sampling: bool = True
    keys_to_copy: Optional[List[str]] = None  # If None, copy all from section


@router.post("/{name}/copy-to")
async def copy_model_params(name: str, request: ModelCopyRequest, _=Depends(require_not_read_only)):
    """Copy parameters from one model to others."""
    base_path = get_base_config_path()
    
    if not base_path.exists():
        raise HTTPException(status_code=404, detail="Base config not found")
    
    content = base_path.read_text(encoding="utf-8")
    config = yaml.safe_load(content)
    
    models = config.get("models", {})
    if name not in models:
        raise HTTPException(status_code=404, detail=f"Source model '{name}' not found")
    
    source_cfg = models[name]
    updated = []
    
    for target_name in request.target_models:
        if target_name not in models:
            continue
        
        target_cfg = models[target_name]
        
        # Copy cmd parameters
        if request.copy_cmd and "cmd" in source_cfg:
            if "cmd" not in target_cfg:
                target_cfg["cmd"] = {}
            source_cmd = source_cfg["cmd"]
            if request.keys_to_copy:
                for key in request.keys_to_copy:
                    if key in source_cmd:
                        target_cfg["cmd"][key] = source_cmd[key]
            else:
                # Copy common parameters (not model-specific ones)
                for key in ["ctx-size", "gpu-layers", "threads", "n-gpu-layers", "batch-size", "ubatch-size"]:
                    if key in source_cmd:
                        target_cfg["cmd"][key] = source_cmd[key]
        
        # Copy sampling parameters
        if request.copy_sampling and "sampling" in source_cfg:
            if "sampling" not in target_cfg:
                target_cfg["sampling"] = {}
            source_sampling = source_cfg["sampling"]
            if request.keys_to_copy:
                for key in request.keys_to_copy:
                    if key in source_sampling:
                        target_cfg["sampling"][key] = source_sampling[key]
            else:
                for key, value in source_sampling.items():
                    target_cfg["sampling"][key] = value
        
        updated.append(target_name)
    
    # Write back
    backup_path = base_path.with_suffix(".yaml.bak")
    backup_path.write_text(content, encoding="utf-8")
    
    with open(base_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    return {"status": "ok", "source": name, "updated": updated}


class ModelCreateRequest(BaseModel):
    """Request to create a new model entry."""
    model_path: str
    ctx_size: int = 8192
    gpu_layers: int = -1
    threads: Optional[int] = None
    hf_tokenizer: Optional[str] = None
    copy_from: Optional[str] = None  # Copy base params from existing model


@router.post("")
async def create_model(name: str, request: ModelCreateRequest, _=Depends(require_not_read_only)):
    """Create a new model entry in the base config."""
    base_path = get_base_config_path()
    
    if not base_path.exists():
        raise HTTPException(status_code=404, detail="Base config not found")
    
    content = base_path.read_text(encoding="utf-8")
    config = yaml.safe_load(content)
    
    models = config.get("models", {})
    if name in models:
        raise HTTPException(status_code=409, detail=f"Model '{name}' already exists")
    
    # Create new model config
    if request.copy_from and request.copy_from in models:
        # Clone from existing model
        import copy
        new_cfg = copy.deepcopy(models[request.copy_from])
        # Update model path
        if "cmd" not in new_cfg:
            new_cfg["cmd"] = {}
        new_cfg["cmd"]["model"] = request.model_path
    else:
        template = _default_cmd_template(config)
        # Create from scratch
        new_cfg = {
            "cmd": {
                **template,
                "model": request.model_path,
                "ctx-size": request.ctx_size,
                "gpu-layers": _coerce_gpu_layers(request.gpu_layers),
            }
        }
        if request.threads:
            new_cfg["cmd"]["threads"] = request.threads

    _validate_cmd_has_required_keys(name, new_cfg.get("cmd", {}))
    
    if request.hf_tokenizer:
        new_cfg["hf_tokenizer_for_model"] = request.hf_tokenizer
    
    models[name] = new_cfg
    
    # Write back
    backup_path = base_path.with_suffix(".yaml.bak")
    backup_path.write_text(content, encoding="utf-8")
    
    with open(base_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    return {"status": "ok", "model": name}


def get_models_using_gguf(gguf_path: str) -> List[str]:
    """Find all model configs that reference a specific GGUF file."""
    base_path = get_base_config_path()
    if not base_path.exists():
        return []
    
    content = base_path.read_text(encoding="utf-8")
    config = yaml.safe_load(content)
    models = config.get("models", {})
    
    # Normalize the target path
    target_path = Path(gguf_path)
    try:
        target_resolved = target_path.resolve()
    except (OSError, ValueError):
        target_resolved = target_path
    
    matching_models = []
    project_root = get_project_root()
    
    for name, model_cfg in models.items():
        if not isinstance(model_cfg, dict):
            continue
        cmd = model_cfg.get("cmd", {})
        if not isinstance(cmd, dict):
            continue
        model_path_str = cmd.get("model", "")
        if not model_path_str:
            continue
        
        # Resolve model path
        model_path = Path(model_path_str)
        if not model_path.is_absolute():
            model_path = project_root / model_path_str
        
        try:
            model_resolved = model_path.resolve()
            # Check if they point to the same file
            if model_resolved == target_resolved or str(model_resolved) == str(target_resolved):
                matching_models.append(name)
            # Also check by filename if paths don't match exactly
            elif model_resolved.name == target_resolved.name:
                # Could be the same file via different paths
                if model_resolved.exists() and target_resolved.exists():
                    if model_resolved.samefile(target_resolved):
                        matching_models.append(name)
        except (OSError, ValueError):
            continue
    
    return matching_models


@router.get("/files/{filename}/dependencies")
async def get_file_dependencies(filename: str):
    """Get all model configs that use a specific GGUF file."""
    models_dir = get_models_dir()
    try:
        resolved_models_dir = models_dir.resolve()
    except (OSError, ValueError):
        resolved_models_dir = models_dir
    
    # Find the file
    file_path = None
    for path in resolved_models_dir.rglob("*.gguf"):
        if path.name == filename:
            file_path = path
            break
    
    if not file_path:
        raise HTTPException(status_code=404, detail=f"GGUF file '{filename}' not found")
    
    dependent_models = get_models_using_gguf(str(file_path))
    
    return {
        "filename": filename,
        "path": str(file_path),
        "dependent_models": dependent_models,
        "count": len(dependent_models)
    }


@router.delete("/{name}")
async def delete_model(name: str, delete_file: bool = False, _=Depends(require_not_read_only)):
    """Delete a model entry from the base config, optionally deleting the GGUF file."""
    base_path = get_base_config_path()
    
    if not base_path.exists():
        raise HTTPException(status_code=404, detail="Base config not found")
    
    content = base_path.read_text(encoding="utf-8")
    config = yaml.safe_load(content)
    
    models = config.get("models", {})
    if name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    
    model_cfg = models[name]
    deleted_file = None
    other_models_using_file = []
    
    # Get model path for dependency check
    cmd = model_cfg.get("cmd", {})
    model_path_str = cmd.get("model", "")
    
    # Check for other models using the same GGUF
    if model_path_str:
        all_dependent = get_models_using_gguf(model_path_str)
        other_models_using_file = [m for m in all_dependent if m != name]
    
    # Delete the model config
    del models[name]
    
    # Optionally delete the GGUF file
    if delete_file and model_path_str:
        model_path = Path(model_path_str)
        if not model_path.is_absolute():
            model_path = get_project_root() / model_path_str
        
        try:
            resolved_path = model_path.resolve()
            if resolved_path.exists() and resolved_path.is_file():
                # Safety check: only delete .gguf files
                if resolved_path.suffix.lower() == ".gguf":
                    resolved_path.unlink()
                    deleted_file = str(resolved_path)
                else:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Cannot delete non-GGUF file: {resolved_path.name}"
                    )
        except PermissionError:
            raise HTTPException(status_code=500, detail="Permission denied when deleting file")
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")
    
    # Write back
    backup_path = base_path.with_suffix(".yaml.bak")
    backup_path.write_text(content, encoding="utf-8")
    
    with open(base_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    return {
        "status": "ok", 
        "deleted": name, 
        "deleted_file": deleted_file,
        "other_models_using_file": other_models_using_file
    }


@router.delete("/files/{filename}")
async def delete_gguf_file(filename: str, delete_configs: bool = False, _=Depends(require_not_read_only)):
    """Delete a GGUF file and optionally remove all configs that reference it."""
    models_dir = get_models_dir()
    try:
        resolved_models_dir = models_dir.resolve()
    except (OSError, ValueError):
        resolved_models_dir = models_dir
    
    # Find the file
    file_path = None
    for path in resolved_models_dir.rglob("*.gguf"):
        if path.name == filename:
            file_path = path
            break
    
    if not file_path:
        raise HTTPException(status_code=404, detail=f"GGUF file '{filename}' not found")
    
    # Find dependent models
    dependent_models = get_models_using_gguf(str(file_path))
    deleted_configs = []
    
    # Optionally delete dependent configs
    if delete_configs and dependent_models:
        base_path = get_base_config_path()
        if base_path.exists():
            content = base_path.read_text(encoding="utf-8")
            config = yaml.safe_load(content)
            models = config.get("models", {})
            
            for model_name in dependent_models:
                if model_name in models:
                    del models[model_name]
                    deleted_configs.append(model_name)
            
            # Write back
            backup_path = base_path.with_suffix(".yaml.bak")
            backup_path.write_text(content, encoding="utf-8")
            
            with open(base_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    # Delete the file
    try:
        file_path.unlink()
    except PermissionError:
        raise HTTPException(status_code=500, detail="Permission denied when deleting file")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")
    
    return {
        "status": "ok",
        "deleted_file": str(file_path),
        "deleted_configs": deleted_configs,
        "orphaned_configs": [m for m in dependent_models if m not in deleted_configs]
    }
