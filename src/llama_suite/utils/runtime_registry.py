from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class RuntimeSpec:
    name: str
    vendor_dir: str
    repo_url: str
    aliases: tuple[str, ...] = ()


RUNTIME_SPECS: tuple[RuntimeSpec, ...] = (
    RuntimeSpec(
        name="llama.cpp",
        vendor_dir="llama.cpp",
        repo_url="https://github.com/ggml-org/llama.cpp.git",
        aliases=("llama_cpp", "llama-cpp"),
    ),
    RuntimeSpec(
        name="ik_llama.cpp",
        vendor_dir="ik_llama.cpp",
        repo_url="https://github.com/ikawrakow/ik_llama.cpp.git",
        aliases=("ik_llama_cpp", "ik-llama-cpp", "ik"),
    ),
)

RUNTIME_SPEC_BY_NAME = {spec.name: spec for spec in RUNTIME_SPECS}

_RUNTIME_ALIASES: dict[str, str] = {}
for _spec in RUNTIME_SPECS:
    _RUNTIME_ALIASES[_spec.name.lower()] = _spec.name
    for _alias in _spec.aliases:
        _RUNTIME_ALIASES[_alias.lower()] = _spec.name


def normalize_runtime_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return _RUNTIME_ALIASES.get(str(value).strip().lower())


def default_server_basename() -> str:
    return "llama-server.exe" if os.name == "nt" else "llama-server"


def runtime_default_bin_hint(runtime_name: str) -> str:
    runtime = normalize_runtime_name(runtime_name)
    if runtime is None:
        raise ValueError(f"Unknown runtime: {runtime_name}")
    return f"vendor/{RUNTIME_SPEC_BY_NAME[runtime].vendor_dir}/bin/llama-server"


def runtime_server_candidates(repo_root: Path, runtime_name: str, base_name: Optional[str] = None) -> list[Path]:
    runtime = normalize_runtime_name(runtime_name)
    if runtime is None:
        return []
    spec = RUNTIME_SPEC_BY_NAME[runtime]
    base = base_name or default_server_basename()
    candidates = [
        repo_root / "vendor" / spec.vendor_dir / "bin" / base,
        repo_root / spec.vendor_dir / "build" / "bin" / base,
    ]
    if os.name == "nt" and not base.lower().endswith(".exe"):
        candidates.extend(
            [
                repo_root / "vendor" / spec.vendor_dir / "bin" / f"{base}.exe",
                repo_root / spec.vendor_dir / "build" / "bin" / f"{base}.exe",
            ]
        )
    return candidates


def all_runtime_server_candidates(repo_root: Path, base_name: Optional[str] = None) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()
    for spec in RUNTIME_SPECS:
        for candidate in runtime_server_candidates(repo_root, spec.name, base_name=base_name):
            key = str(candidate)
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)
    return candidates


def infer_runtime_from_path(path: Path) -> Optional[str]:
    norm = str(path).replace("\\", "/").lower()
    for spec in RUNTIME_SPECS:
        token = f"/{spec.vendor_dir.lower()}/"
        if token in norm:
            return spec.name
    return None


def known_runtime_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in RUNTIME_SPECS)


def known_runtime_vendor_dirs() -> Iterable[str]:
    for spec in RUNTIME_SPECS:
        yield spec.vendor_dir
