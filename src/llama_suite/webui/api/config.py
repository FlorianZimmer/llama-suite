"""Configuration API routes for managing config.base.yaml and overrides."""

from pathlib import Path
from typing import Optional
import yaml

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from llama_suite.utils.config_utils import (
    find_project_root,
    generate_processed_config,
    deep_merge_dicts_util
)


router = APIRouter(prefix="/api/config", tags=["configuration"])


def get_project_root() -> Path:
    """Get the project root directory."""
    return find_project_root()


def get_configs_dir() -> Path:
    """Get the configs directory."""
    return get_project_root() / "configs"


def get_base_config_path() -> Path:
    """Get path to config.base.yaml."""
    return get_configs_dir() / "config.base.yaml"


def get_overrides_dir() -> Path:
    """Get the overrides directory."""
    return get_configs_dir() / "overrides"


class ConfigUpdateRequest(BaseModel):
    """Request body for updating configuration."""
    content: str


class OverrideCreateRequest(BaseModel):
    """Request body for creating a new override file."""
    name: str
    content: str


@router.get("")
async def get_base_config():
    """Get the base configuration file content."""
    config_path = get_base_config_path()
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="Base config not found")
    
    content = config_path.read_text(encoding="utf-8")
    return {
        "path": str(config_path),
        "content": content
    }


@router.put("")
async def update_base_config(request: ConfigUpdateRequest):
    """Update the base configuration file."""
    config_path = get_base_config_path()
    
    # Validate YAML syntax
    try:
        yaml.safe_load(request.content)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")
    
    # Create backup
    backup_path = config_path.with_suffix(".yaml.bak")
    if config_path.exists():
        backup_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    
    # Write new content
    config_path.write_text(request.content, encoding="utf-8")
    
    return {"status": "ok", "message": "Configuration updated"}


@router.get("/overrides")
async def list_overrides():
    """List all override configuration files."""
    overrides_dir = get_overrides_dir()
    if not overrides_dir.exists():
        return {"overrides": []}
    
    overrides = []
    for path in sorted(overrides_dir.glob("*.yaml")):
        overrides.append({
            "name": path.stem,
            "filename": path.name,
            "path": str(path)
        })
    
    return {"overrides": overrides}


@router.get("/overrides/{name}")
async def get_override(name: str):
    """Get a specific override file content."""
    override_path = get_overrides_dir() / f"{name}.yaml"
    if not override_path.exists():
        raise HTTPException(status_code=404, detail=f"Override '{name}' not found")
    
    content = override_path.read_text(encoding="utf-8")
    return {
        "name": name,
        "path": str(override_path),
        "content": content
    }


@router.put("/overrides/{name}")
async def update_override(name: str, request: ConfigUpdateRequest):
    """Update an override configuration file."""
    override_path = get_overrides_dir() / f"{name}.yaml"
    if not override_path.exists():
        raise HTTPException(status_code=404, detail=f"Override '{name}' not found")
    
    # Validate YAML syntax
    try:
        yaml.safe_load(request.content)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")
    
    # Create backup
    backup_path = override_path.with_suffix(".yaml.bak")
    backup_path.write_text(override_path.read_text(encoding="utf-8"), encoding="utf-8")
    
    # Write new content
    override_path.write_text(request.content, encoding="utf-8")
    
    return {"status": "ok", "message": f"Override '{name}' updated"}


@router.post("/overrides")
async def create_override(request: OverrideCreateRequest):
    """Create a new override configuration file."""
    overrides_dir = get_overrides_dir()
    overrides_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize filename
    safe_name = "".join(c for c in request.name if c.isalnum() or c in "-_")
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid override name")
    
    override_path = overrides_dir / f"{safe_name}.yaml"
    if override_path.exists():
        raise HTTPException(status_code=409, detail=f"Override '{safe_name}' already exists")
    
    # Validate YAML syntax
    try:
        yaml.safe_load(request.content)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")
    
    override_path.write_text(request.content, encoding="utf-8")
    
    return {"status": "ok", "name": safe_name, "path": str(override_path)}


@router.delete("/overrides/{name}")
async def delete_override(name: str):
    """Delete an override configuration file."""
    override_path = get_overrides_dir() / f"{name}.yaml"
    if not override_path.exists():
        raise HTTPException(status_code=404, detail=f"Override '{name}' not found")
    
    override_path.unlink()
    
    return {"status": "ok", "message": f"Override '{name}' deleted"}


@router.get("/effective")
async def get_effective_config(override: Optional[str] = None):
    """Get the effective merged configuration."""
    base_path = get_base_config_path()
    override_path = None
    
    if override:
        override_path = get_overrides_dir() / f"{override}.yaml"
        if not override_path.exists():
            raise HTTPException(status_code=404, detail=f"Override '{override}' not found")
    
    try:
        # generate_processed_config returns the effective config dict directly
        config = generate_processed_config(
            base_config_path_arg=base_path,
            override_config_path_arg=override_path,
            verbose_logging=False
        )
        config_yaml = yaml.safe_dump(
            config,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
        return {
            "config": config,
            "yaml": config_yaml,
            "models": config.get("models", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating config: {e}")


@router.get("/sampling-presets")
async def get_sampling_presets():
    """Extract sampling presets from the base config."""
    config_path = get_base_config_path()
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="Base config not found")
    
    content = config_path.read_text(encoding="utf-8")
    
    # Parse and find sampling presets (keys ending with _SAMPLING)
    try:
        config = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=500, detail=f"Error parsing config: {e}")
    
    presets = {}
    for key, value in config.items():
        if key.endswith("_SAMPLING") and isinstance(value, dict):
            presets[key] = value
    
    return {"presets": presets}
