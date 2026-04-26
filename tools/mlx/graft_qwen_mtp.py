#!/usr/bin/env python3
"""Graft official Qwen MTP tensors into an existing MLX model directory.

The MLX community Qwen3.6 quantized checkpoints can keep
``mtp_num_hidden_layers`` in config but omit the actual ``mtp.*`` weights.  This
script builds a patched model directory by hard-linking the existing MLX files
and adding a ``model-mtp.safetensors`` shard extracted from the official HF
checkpoint shards.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import tempfile
from pathlib import Path

import httpx
import mlx.core as mx


MTP_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
    ".pre_fc_norm_hidden.weight",
    ".pre_fc_norm_embedding.weight",
    "mtp.norm.weight",
)


def hardlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def link_model_dir(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in src_dir.iterdir():
        if src.name == ".cache":
            continue
        dst = dst_dir / src.name
        if src.is_dir():
            if dst.exists():
                continue
            shutil.copytree(src, dst, copy_function=hardlink_or_copy)
        elif src.is_file():
            hardlink_or_copy(src, dst)


def load_mtp_tensors(index_path: Path, shard_dir: Path) -> dict[str, mx.array]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    mtp_by_file: dict[str, list[str]] = {}
    for key, filename in weight_map.items():
        if key.startswith("mtp."):
            mtp_by_file.setdefault(filename, []).append(key)

    if not mtp_by_file:
        raise RuntimeError(f"No mtp.* tensors found in {index_path}")

    tensors: dict[str, mx.array] = {}
    for filename, keys in sorted(mtp_by_file.items()):
        shard_path = shard_dir / filename
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing MTP source shard: {shard_path}")
        shard = mx.load(str(shard_path))
        for key in sorted(keys):
            if key not in shard:
                raise KeyError(f"{key} listed in index but missing from {shard_path}")
            value = shard[key]
            if any(key.endswith(suffix) for suffix in MTP_NORM_SUFFIXES) and value.ndim == 1:
                value = value + 1.0
            tensors[key] = value
        del shard

    return tensors


def hf_range_url(repo_id: str, filename: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"


def fetch_range(client: httpx.Client, url: str, start: int, end: int) -> bytes:
    response = client.get(url, headers={"Range": f"bytes={start}-{end}"})
    response.raise_for_status()
    if response.status_code != 206:
        raise RuntimeError(f"Expected HTTP 206 for ranged request to {url}, got {response.status_code}")
    return response.content


def fetch_safetensors_header(client: httpx.Client, url: str) -> tuple[int, dict]:
    header_len = struct.unpack("<Q", fetch_range(client, url, 0, 7))[0]
    header = fetch_range(client, url, 8, 8 + header_len - 1)
    return header_len, json.loads(header.rstrip(b" ").decode("utf-8"))


def write_safetensors(path: Path, tensors: dict[str, tuple[dict, bytes]]) -> None:
    header: dict[str, dict] = {}
    offset = 0
    payload_parts: list[bytes] = []
    for key in sorted(tensors):
        meta, payload = tensors[key]
        header[key] = {
            "dtype": meta["dtype"],
            "shape": meta["shape"],
            "data_offsets": [offset, offset + len(payload)],
        }
        payload_parts.append(payload)
        offset += len(payload)

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # safetensors allows trailing spaces in the header.
    header_bytes += b" " * ((8 - ((8 + len(header_bytes)) % 8)) % 8)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for payload in payload_parts:
            f.write(payload)


def load_mtp_tensors_from_hf(index_path: Path, repo_id: str) -> dict[str, mx.array]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    mtp_by_file: dict[str, list[str]] = {}
    for key, filename in index["weight_map"].items():
        if key.startswith("mtp."):
            mtp_by_file.setdefault(filename, []).append(key)

    if not mtp_by_file:
        raise RuntimeError(f"No mtp.* tensors found in {index_path}")

    token = os.environ.get("HF_TOKEN")
    if not token:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            token = token_path.read_text(encoding="utf-8").strip()
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    raw_tensors: dict[str, tuple[dict, bytes]] = {}
    with httpx.Client(follow_redirects=True, timeout=120, headers=headers) as client:
        for filename, keys in sorted(mtp_by_file.items()):
            url = hf_range_url(repo_id, filename)
            header_len, header = fetch_safetensors_header(client, url)
            data_start = 8 + header_len
            for key in sorted(keys):
                meta = header[key]
                begin, end = meta["data_offsets"]
                payload = fetch_range(client, url, data_start + begin, data_start + end - 1)
                raw_tensors[key] = (meta, payload)

    with tempfile.TemporaryDirectory(prefix="qwen-mtp-raw-") as tmp:
        raw_path = Path(tmp) / "raw-mtp.safetensors"
        write_safetensors(raw_path, raw_tensors)
        tensors = mx.load(str(raw_path))

    for key, value in list(tensors.items()):
        if any(key.endswith(suffix) for suffix in MTP_NORM_SUFFIXES) and value.ndim == 1:
            tensors[key] = value + 1.0
    return tensors


def update_index(model_dir: Path, mtp_keys: list[str], mtp_filename: str) -> None:
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.setdefault("weight_map", {})
    for key in mtp_keys:
        weight_map[key] = mtp_filename
    index["weight_map"] = {key: weight_map[key] for key in sorted(weight_map)}
    index_path.write_text(json.dumps(index, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlx-model-dir", type=Path, required=True)
    parser.add_argument("--hf-shard-dir", type=Path, required=True)
    parser.add_argument("--hf-repo")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mtp-file", default="model-mtp.safetensors")
    args = parser.parse_args()

    link_model_dir(args.mlx_model_dir, args.output_dir)

    index_path = args.hf_shard_dir / "model.safetensors.index.json"
    if args.hf_repo:
        tensors = load_mtp_tensors_from_hf(index_path, args.hf_repo)
    else:
        tensors = load_mtp_tensors(index_path, args.hf_shard_dir)
    mtp_path = args.output_dir / args.mtp_file
    mx.save_safetensors(str(mtp_path), tensors, metadata={"format": "mlx"})
    update_index(args.output_dir, sorted(tensors), args.mtp_file)

    print(f"Wrote {len(tensors)} MTP tensors to {mtp_path}")
    print(f"Patched MLX model: {args.output_dir}")


if __name__ == "__main__":
    main()
