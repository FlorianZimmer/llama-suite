"""Config Studio API routes (schema-driven GUI editing of configs)."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Optional, Union, cast

import yaml
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from llama_suite.utils.config_utils import ConfigEnvLoader, deep_merge_dicts_util, generate_processed_config
from llama_suite.webui.utils.mode import require_not_read_only
from llama_suite.webui.utils.paths import get_base_config_path as _get_base_config_path
from llama_suite.webui.utils.paths import get_project_root as _get_project_root
from llama_suite.webui.utils.yaml_store import apply_ops, load_yaml_rt, save_yaml_rt, to_plain
from ruamel.yaml.comments import CommentedMap


router = APIRouter(prefix="/api/config", tags=["configuration"])


def get_project_root() -> Path:
    return _get_project_root()


def get_configs_dir() -> Path:
    return get_project_root() / "configs"


def get_overrides_dir() -> Path:
    return get_configs_dir() / "overrides"


def get_base_config_path() -> Path:
    return _get_base_config_path()


def get_schema_path() -> Path:
    # <repo>/src/llama_suite/webui/config_schema.yaml
    return Path(__file__).resolve().parents[1] / "config_schema.yaml"


def _safe_override_name(name: str) -> str:
    return "".join(c for c in name.strip() if c.isalnum() or c in "-_")


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    data = yaml.load(text, Loader=ConfigEnvLoader) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    # Normalize keys to strings for JSON friendliness
    return {str(k): v for k, v in data.items()}


MANDATORY_CMD_KEYS = {"bin", "port", "model", "ctx-size"}


def _validate_model_required_cmd(merged: dict[str, Any], model_name: str) -> None:
    models = merged.get("models")
    if not isinstance(models, dict):
        raise HTTPException(status_code=400, detail="Config must contain a 'models' mapping")
    cfg = models.get(model_name)
    if not isinstance(cfg, dict):
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' config must be a mapping")
    cmd = cfg.get("cmd")
    if not isinstance(cmd, dict):
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' must have a 'cmd' mapping")
    missing = sorted(k for k in MANDATORY_CMD_KEYS if not cmd.get(k))
    if missing:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}': cmd is missing required keys: {', '.join(missing)}")


def _merge_base_and_override(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    if override:
        deep_merge_dicts_util(merged, override)
    return merged


def _collect_touched_models(ops: list[dict[str, Any]]) -> set[str]:
    touched: set[str] = set()
    for op in ops:
        path = op.get("path")
        if not isinstance(path, list) or len(path) < 2:
            continue
        if path[0] == "models" and isinstance(path[1], str) and path[1]:
            touched.add(path[1])
    return touched


def _get_config_doc_path(target_kind: str, target_name: Optional[str]) -> Path:
    if target_kind == "base":
        return get_base_config_path()
    if target_kind == "override":
        if not target_name:
            raise HTTPException(status_code=400, detail="Override target requires 'name'")
        safe_name = "".join(c for c in target_name if c.isalnum() or c in "-_")
        if not safe_name:
            raise HTTPException(status_code=400, detail="Invalid override name")
        return get_overrides_dir() / f"{safe_name}.yaml"
    raise HTTPException(status_code=400, detail=f"Invalid target kind: {target_kind}")


class StudioTarget(BaseModel):
    kind: Literal["base", "override"]
    name: Optional[str] = None


class PatchOp(BaseModel):
    op: Literal["set", "delete"]
    path: list[Union[str, int]] = Field(..., min_length=1)
    value: Optional[Any] = None


class PatchRequest(BaseModel):
    target: StudioTarget
    ops: list[PatchOp]
    # Used only to compute the returned effective snapshot after saving.
    context_override: Optional[str] = None


class BulkApplyRequest(BaseModel):
    target: StudioTarget
    models: Union[list[str], Literal["ALL"]] = "ALL"
    filter_string: Optional[str] = None
    section: Literal["cmd", "sampling", "model"] = "cmd"
    changes: dict[str, Any]
    context_override: Optional[str] = None


class PresetUpdateRequest(BaseModel):
    preset_name: str
    values: dict[str, Any]
    context_override: Optional[str] = None


class PresetApplyRequest(BaseModel):
    preset_name: str
    target: StudioTarget
    models: Union[list[str], Literal["ALL"]] = "ALL"
    filter_string: Optional[str] = None
    context_override: Optional[str] = None


@router.get("/studio")
async def get_config_studio(override: Optional[str] = Query(default=None)):
    """
    Return Config Studio data:
    - schema (UI registry)
    - base/override snapshots
    - effective (simple deep-merge without culling)
    - effective_processed (generate_processed_config output; includes generated_cmd_str)
    - meta (models, presets, union keys)
    """
    schema_path = get_schema_path()
    if not schema_path.exists():
        raise HTTPException(status_code=500, detail=f"Config Studio schema not found: {schema_path}")
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8")) or {}

    base_path = get_base_config_path()
    if not base_path.exists():
        raise HTTPException(status_code=404, detail="Base config not found")
    base = _load_yaml_dict(base_path)

    override_path: Optional[Path] = None
    override_data: dict[str, Any] = {}
    if override:
        safe_override = _safe_override_name(override)
        if not safe_override or safe_override != override:
            raise HTTPException(status_code=400, detail="Invalid override name")
        override_path = get_overrides_dir() / f"{safe_override}.yaml"
        if not override_path.exists():
            raise HTTPException(status_code=404, detail=f"Override '{safe_override}' not found")
        override_data = _load_yaml_dict(override_path)

    effective = deepcopy(base)
    if override_data:
        deep_merge_dicts_util(effective, override_data)

    try:
        effective_processed = generate_processed_config(
            base_config_path_arg=base_path,
            override_config_path_arg=override_path,
            verbose_logging=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating processed config: {e}")

    presets = sorted(
        [
            k
            for k, v in base.items()
            if isinstance(k, str) and k.endswith("_SAMPLING") and isinstance(v, dict)
        ],
        key=lambda s: s.lower(),
    )

    models = effective.get("models", {})
    model_names: list[str] = []
    model_keys: set[str] = set()
    cmd_keys: set[str] = set()
    sampling_keys: set[str] = set()
    if isinstance(models, dict):
        model_names = sorted([str(k) for k in models.keys()], key=lambda s: s.lower())
        for _, cfg in models.items():
            if not isinstance(cfg, dict):
                continue
            model_keys |= {str(k) for k in cfg.keys()}
            cmd = cfg.get("cmd")
            if isinstance(cmd, dict):
                cmd_keys |= {str(k) for k in cmd.keys()}
            sampling = cfg.get("sampling")
            if isinstance(sampling, dict):
                sampling_keys |= {str(k) for k in sampling.keys()}

    meta = {
        "base_path": str(base_path),
        "override_path": str(override_path) if override_path else None,
        "models": model_names,
        "sampling_presets": presets,
        "model_keys": sorted(model_keys),
        "cmd_keys": sorted(cmd_keys),
        "sampling_keys": sorted(sampling_keys),
    }

    return {
        "schema": schema,
        "base": base,
        "override": override_data,
        "effective": effective,
        "effective_processed": effective_processed,
        "meta": meta,
    }


# NOTE: Write endpoints (patch/bulk-apply/presets) are implemented below. They are
# guarded by require_not_read_only in gitops mode.


@router.post("/studio/validate")
async def validate_effective_config(override: Optional[str] = None):
    """
    Validate config processing (command generation, required keys) without mutating files.
    """
    base_path = get_base_config_path()
    override_path = None
    if override:
        safe_override = _safe_override_name(override)
        if not safe_override or safe_override != override:
            raise HTTPException(status_code=400, detail="Invalid override name")
        override_path = get_overrides_dir() / f"{safe_override}.yaml"
        if not override_path.exists():
            raise HTTPException(status_code=404, detail=f"Override '{safe_override}' not found")
    try:
        config = generate_processed_config(
            base_config_path_arg=base_path,
            override_config_path_arg=override_path,
            verbose_logging=False,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Config validation failed: {e}")
    return {"status": "ok", "models": sorted(list((config.get('models') or {}).keys()))}


@router.post("/studio/patch")
async def patch_config(request: PatchRequest, _=Depends(require_not_read_only)):
    """
    Apply explicit set/delete ops to base or override YAML while preserving comments/formatting.
    """
    ops = [op.model_dump() for op in request.ops]
    target_path = _get_config_doc_path(request.target.kind, request.target.name)

    doc = load_yaml_rt(target_path)
    if not isinstance(doc, CommentedMap):
        raise HTTPException(status_code=400, detail="YAML root must be a mapping")

    try:
        apply_ops(doc, ops)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid patch ops: {e}")

    # Validate required keys for touched models against effective config.
    base_data = _load_yaml_dict(get_base_config_path())
    override_data: dict[str, Any] = {}
    if request.target.kind == "base":
        base_data = cast(dict[str, Any], to_plain(doc))
    else:
        override_data = cast(dict[str, Any], to_plain(doc))
    merged = _merge_base_and_override(base_data, override_data)

    for model_name in sorted(_collect_touched_models(ops)):
        _validate_model_required_cmd(merged, model_name)

    await save_yaml_rt(target_path, doc, create_backup=True)

    context_override = request.context_override
    if request.target.kind == "override" and request.target.name:
        context_override = request.target.name

    return {"status": "ok", "target_path": str(target_path), "context_override": context_override}


@router.post("/studio/bulk-apply")
async def bulk_apply(request: BulkApplyRequest, _=Depends(require_not_read_only)):
    """
    Apply the same changes to all or a subset of models.

    For changes: use value=null to delete/reset the key (particularly useful for overrides).
    """
    if not request.changes:
        raise HTTPException(status_code=400, detail="No changes provided")

    base = _load_yaml_dict(get_base_config_path())
    base_models = base.get("models", {})
    if not isinstance(base_models, dict):
        raise HTTPException(status_code=400, detail="Base config must contain a 'models' mapping")

    # Resolve model list
    if request.models == "ALL":
        model_names = [str(k) for k in base_models.keys()]
    else:
        model_names = [str(m) for m in request.models]

    if request.filter_string:
        needle = request.filter_string.lower()
        model_names = [m for m in model_names if needle in m.lower()]

    model_names = sorted(set(model_names), key=lambda s: s.lower())
    if not model_names:
        raise HTTPException(status_code=400, detail="No models selected")

    target_path = _get_config_doc_path(request.target.kind, request.target.name)
    doc = load_yaml_rt(target_path)

    # Create intermediate structure if missing
    if "models" not in doc or not isinstance(doc.get("models"), CommentedMap):
        doc["models"] = CommentedMap()

    ops: list[dict[str, Any]] = []
    for model_name in model_names:
        for key, value in request.changes.items():
            if not isinstance(key, str) or not key:
                continue
            if request.section == "cmd":
                path: list[Union[str, int]] = ["models", model_name, "cmd", key]
            elif request.section == "sampling":
                path = ["models", model_name, "sampling", key]
            else:
                path = ["models", model_name, key]

            if value is None:
                ops.append({"op": "delete", "path": path})
            else:
                ops.append({"op": "set", "path": path, "value": value})

    try:
        apply_ops(doc, ops)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bulk apply failed: {e}")

    # Validate required keys for touched models against effective config.
    base_data = base
    override_data: dict[str, Any] = {}
    if request.target.kind == "base":
        base_data = cast(dict[str, Any], to_plain(doc))
    else:
        override_data = cast(dict[str, Any], to_plain(doc))
    merged = _merge_base_and_override(base_data, override_data)
    for model_name in model_names:
        _validate_model_required_cmd(merged, model_name)

    await save_yaml_rt(target_path, doc, create_backup=True)

    context_override = request.context_override
    if request.target.kind == "override" and request.target.name:
        context_override = request.target.name

    return {"status": "ok", "updated_models": model_names, "target_path": str(target_path), "context_override": context_override}


@router.post("/studio/presets/update")
async def update_sampling_preset(request: PresetUpdateRequest, _=Depends(require_not_read_only)):
    """
    Update a sampling preset in the base config (top-level keys ending with _SAMPLING).
    """
    preset_name = request.preset_name.strip()
    if not preset_name or not preset_name.endswith("_SAMPLING"):
        raise HTTPException(status_code=400, detail="preset_name must end with _SAMPLING")

    base_path = get_base_config_path()
    doc = load_yaml_rt(base_path)

    if preset_name not in doc or not isinstance(doc.get(preset_name), CommentedMap):
        # Create it if missing
        doc[preset_name] = CommentedMap()

    ops: list[dict[str, Any]] = []
    for key, value in request.values.items():
        if value is None:
            ops.append({"op": "delete", "path": [preset_name, key]})
        else:
            ops.append({"op": "set", "path": [preset_name, key], "value": value})
    apply_ops(doc, ops)

    await save_yaml_rt(base_path, doc, create_backup=True)
    return {"status": "ok", "preset_name": preset_name, "context_override": request.context_override}


@router.post("/studio/presets/apply")
async def apply_sampling_preset(request: PresetApplyRequest, _=Depends(require_not_read_only)):
    """
    Apply a base sampling preset to selected models.

    - If target.kind == base: set models.<name>.sampling to an alias/reference to the preset object.
    - If target.kind == override: copy the preset mapping into the override file.
    """
    preset_name = request.preset_name.strip()
    if not preset_name or not preset_name.endswith("_SAMPLING"):
        raise HTTPException(status_code=400, detail="preset_name must end with _SAMPLING")

    base = _load_yaml_dict(get_base_config_path())
    base_models = base.get("models", {})
    if not isinstance(base_models, dict):
        raise HTTPException(status_code=400, detail="Base config must contain a 'models' mapping")

    if request.models == "ALL":
        model_names = [str(k) for k in base_models.keys()]
    else:
        model_names = [str(m) for m in request.models]
    if request.filter_string:
        needle = request.filter_string.lower()
        model_names = [m for m in model_names if needle in m.lower()]
    model_names = sorted(set(model_names), key=lambda s: s.lower())
    if not model_names:
        raise HTTPException(status_code=400, detail="No models selected")

    if request.target.kind == "base":
        base_path = get_base_config_path()
        doc = load_yaml_rt(base_path)
        if preset_name not in doc or not isinstance(doc.get(preset_name), CommentedMap):
            raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' not found in base config")
        preset_obj = doc[preset_name]

        if "models" not in doc or not isinstance(doc.get("models"), CommentedMap):
            raise HTTPException(status_code=400, detail="Base config missing 'models' mapping")

        for model_name in model_names:
            if model_name not in doc["models"] or not isinstance(doc["models"].get(model_name), CommentedMap):
                continue
            doc["models"][model_name]["sampling"] = preset_obj

        await save_yaml_rt(base_path, doc, create_backup=True)
        return {"status": "ok", "updated_models": model_names, "target_path": str(base_path), "context_override": request.context_override}

    # override target: copy preset values
    preset_plain = base.get(preset_name)
    if not isinstance(preset_plain, dict):
        raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' not found in base config")

    target_path = _get_config_doc_path("override", request.target.name)
    doc = load_yaml_rt(target_path)
    if "models" not in doc or not isinstance(doc.get("models"), CommentedMap):
        doc["models"] = CommentedMap()

    ops: list[dict[str, Any]] = []
    for model_name in model_names:
        # Replace the entire sampling block for that model
        ops.append({"op": "set", "path": ["models", model_name, "sampling"], "value": deepcopy(preset_plain)})
    apply_ops(doc, ops)

    merged = _merge_base_and_override(base, cast(dict[str, Any], to_plain(doc)))
    for model_name in model_names:
        _validate_model_required_cmd(merged, model_name)

    await save_yaml_rt(target_path, doc, create_backup=True)
    context_override = request.context_override or request.target.name
    return {"status": "ok", "updated_models": model_names, "target_path": str(target_path), "context_override": context_override}
