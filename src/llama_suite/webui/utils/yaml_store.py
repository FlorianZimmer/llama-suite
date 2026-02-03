from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq


_WRITE_LOCK = asyncio.Lock()


def _yaml_rt() -> YAML:
    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    yaml.width = 4096
    yaml.indent(mapping=2, sequence=4, offset=2)
    return yaml


def load_yaml_rt(path: Path) -> CommentedMap:
    """
    Load a YAML document with round-trip preservation (comments, anchors, formatting).

    Returns a mapping. Empty or missing files result in an empty mapping.
    """
    if not path.exists():
        return CommentedMap()

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return CommentedMap()

    yaml = _yaml_rt()
    doc = yaml.load(text)
    if doc is None:
        return CommentedMap()
    if not isinstance(doc, CommentedMap):
        raise ValueError(f"Expected YAML mapping at root in {path}, got {type(doc).__name__}")
    return doc


def to_plain(obj: Any) -> Any:
    """Convert ruamel Commented* objects to plain JSON-serializable types."""
    if isinstance(obj, CommentedMap):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, CommentedSeq):
        return [to_plain(v) for v in list(obj)]
    if isinstance(obj, list):
        return [to_plain(v) for v in obj]
    return obj


def _ensure_container(parent: Any, key: Any, next_key: Any) -> Any:
    """
    Ensure parent[key] exists and is a container appropriate for next_key.
    Returns the container.
    """
    want_seq = isinstance(next_key, int)

    if isinstance(parent, (CommentedMap, dict)):
        cur = parent.get(key)
        if want_seq:
            if not isinstance(cur, (CommentedSeq, list)):
                cur = CommentedSeq()
                parent[key] = cur
        else:
            if not isinstance(cur, (CommentedMap, dict)):
                cur = CommentedMap()
                parent[key] = cur
        return cur

    if isinstance(parent, (CommentedSeq, list)):
        if not isinstance(key, int):
            raise TypeError(f"Sequence index must be int, got {type(key).__name__}")
        while len(parent) <= key:
            parent.append(None)
        cur = parent[key]
        if want_seq:
            if not isinstance(cur, (CommentedSeq, list)):
                cur = CommentedSeq()
                parent[key] = cur
        else:
            if not isinstance(cur, (CommentedMap, dict)):
                cur = CommentedMap()
                parent[key] = cur
        return cur

    raise TypeError(f"Unsupported container type: {type(parent).__name__}")


def set_path(doc: Any, path: Sequence[Any], value: Any) -> None:
    if not path:
        raise ValueError("Path cannot be empty")
    cur: Any = doc
    for idx, key in enumerate(path[:-1]):
        next_key = path[idx + 1]
        cur = _ensure_container(cur, key, next_key)

    last = path[-1]
    if isinstance(cur, (CommentedMap, dict)):
        cur[last] = value
        return
    if isinstance(cur, (CommentedSeq, list)):
        if not isinstance(last, int):
            raise TypeError("List index must be int")
        while len(cur) <= last:
            cur.append(None)
        cur[last] = value
        return
    raise TypeError(f"Unsupported container type: {type(cur).__name__}")


def delete_path(doc: Any, path: Sequence[Any]) -> None:
    if not path:
        return
    cur: Any = doc
    for key in path[:-1]:
        if isinstance(cur, (CommentedMap, dict)):
            if key not in cur:
                return
            cur = cur[key]
        elif isinstance(cur, (CommentedSeq, list)):
            if not isinstance(key, int) or key < 0 or key >= len(cur):
                return
            cur = cur[key]
        else:
            return

    last = path[-1]
    if isinstance(cur, (CommentedMap, dict)):
        cur.pop(last, None)
        return
    if isinstance(cur, (CommentedSeq, list)):
        if not isinstance(last, int) or last < 0 or last >= len(cur):
            return
        cur.pop(last)


def apply_ops(doc: CommentedMap, ops: Iterable[dict]) -> None:
    for op in ops:
        kind = op.get("op")
        path = op.get("path")
        if not isinstance(path, list) or not path:
            raise ValueError("Each op must include a non-empty 'path' list")
        if kind == "set":
            if "value" not in op:
                raise ValueError("set op must include 'value'")
            set_path(doc, path, op["value"])
        elif kind == "delete":
            delete_path(doc, path)
        else:
            raise ValueError(f"Unsupported op: {kind}")


async def save_yaml_rt(path: Path, doc: CommentedMap, *, create_backup: bool = True) -> None:
    """
    Save YAML with round-trip preservation. Creates a .yaml.bak sibling by default.

    Writes atomically using os.replace.
    """
    async with _WRITE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)

        if create_backup and path.exists():
            backup_path = path.with_suffix(".yaml.bak")
            backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

        tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        yaml = _yaml_rt()
        try:
            with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
                yaml.dump(doc, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                # Best-effort cleanup
                pass
