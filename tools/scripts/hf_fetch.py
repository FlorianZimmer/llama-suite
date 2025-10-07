#!/usr/bin/env python3
"""
Download GGUF models (and optional tokenizers) defined in a base YAML plus an override YAML.

Features
- Deep-merge base+override (override wins).
- Skip models with `disabled: true` (typically set in override).
- Collect GGUFs from `cmd.model` and (optionally) `cmd.model-draft`.
- Handle sharded files like *-00001-of-00002.gguf (downloads all shards).
- Resolve actual Hugging Face repo by enumerating org repos and listing files.
- Works even when list_models returns a generator (no slicing errors).
- Supports nested paths inside repos (download by the correct relative path).
- Optional per-model `hf_repo_for_gguf` to skip searching and pin a repo.
- Optional `--also-tokenizers` to snapshot repos from `hf_tokenizer_for_model`.

Install deps:
    pip install pyyaml huggingface_hub rich
"""

import argparse
import os
import os.path
import re
import sys
import pathlib
from typing import Dict, Any, List, Tuple, Optional, Set

try:
    import yaml  # PyYAML
except ImportError:
    print("Please: pip install pyyaml huggingface_hub rich", file=sys.stderr)
    sys.exit(1)

from itertools import islice
from functools import lru_cache

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import HfHubHTTPError
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

console = Console()

SHARD_RE = re.compile(r"^(?P<prefix>.+)-(?P<i>\d{5})-of-(?P<n>\d{5})\.gguf$", re.IGNORECASE)

# --------------------------- YAML + merge helpers ---------------------------

def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge dictionaries. Values from b override a."""
    out = dict(a)
    for k, v in (b or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def merged_models(base_cfg: Dict[str, Any], override_cfg: Dict[str, Any]) -> Dict[str, Any]:
    base_models = (base_cfg or {}).get("models", {}) or {}
    over_models = (override_cfg or {}).get("models", {}) or {}
    merged = {}
    for name, base_m in base_models.items():
        merged[name] = deep_merge(base_m, over_models.get(name, {}))
    for name, over_m in over_models.items():
        if name not in merged:
            merged[name] = over_m
    return merged

def is_disabled(model_cfg: Dict[str, Any]) -> bool:
    return bool((model_cfg or {}).get("disabled", False))

def extract_cmd_paths(model_cfg: Dict[str, Any], include_drafts: bool) -> List[Tuple[str, str]]:
    """Return list of (key, path) where key is 'model' or 'model-draft'."""
    cmd = (model_cfg or {}).get("cmd", {}) or {}
    out = []
    if isinstance(cmd.get("model"), str):
        out.append(("model", cmd["model"]))
    if include_drafts and isinstance(cmd.get("model-draft"), str):
        out.append(("model-draft", cmd["model-draft"]))
    return out

def normalize_local_path(target_dir: str, p: str) -> pathlib.Path:
    return pathlib.Path(target_dir) / pathlib.Path(p).name

def expand_shards(filename: str) -> List[str]:
    m = SHARD_RE.match(filename)
    if not m:
        return [filename]
    n = int(m.group("n"))
    prefix = m.group("prefix")
    return [f"{prefix}-{i:05d}-of-{n:05d}.gguf" for i in range(1, n + 1)]

# ------------------------ Hugging Face list & search ------------------------

@lru_cache(maxsize=2048)
def _list_repo_files(api: HfApi, repo_id: str) -> List[str]:
    """
    Return file paths for a repo.
    Tries list_repo_files; falls back to model_info.siblings if needed.
    """
    try:
        return api.list_repo_files(repo_id=repo_id, repo_type="model")
    except AttributeError:
        info = api.model_info(repo_id)
        return [s.rfilename for s in getattr(info, "siblings", [])]

def _repo_find_relpath_for_filename(api: HfApi, repo_id: str, filename: str) -> Optional[str]:
    """
    Find the relative path in repo where the basename equals `filename`.
    If multiple candidates exist, choose the shortest path.
    """
    files = _list_repo_files(api, repo_id)
    matches = [p for p in files if os.path.basename(p) == filename]
    if not matches:
        return None
    matches.sort(key=lambda p: (p.count("/"), len(p), p))
    return matches[0]

def _repo_has_file(api: HfApi, repo_id: str, filename: str) -> bool:
    return _repo_find_relpath_for_filename(api, repo_id, filename) is not None

def find_repo_and_relpath_for_file(
    api: HfApi,
    filename: str,
    orgs: Optional[List[str]],
    org_scan_limit: int = 400,
    global_search_limit: int = 80,
) -> Optional[Tuple[str, str]]:
    """
    Find (repo_id, relpath) such that repo contains `filename` (by basename).
    Order:
      1) If orgs provided: scan up to `org_scan_limit` repos per org (by downloads).
      2) Fallback: global search (by metadata), then validate by listing files.
    """
    tried: Set[str] = set()

    # 1) Scan orgs exhaustively, then check file lists.
    if orgs:
        for author in orgs:
            try:
                infos_iter = api.list_models(author=author, sort="downloads", direction=-1)
            except Exception:
                continue
            for info in islice(infos_iter, org_scan_limit):
                rid = info.id
                if rid in tried:
                    continue
                tried.add(rid)
                try:
                    rel = _repo_find_relpath_for_filename(api, rid, filename)
                    if rel:
                        return rid, rel
                except Exception:
                    pass

    # 2) Fallback: global search by metadata (may not include filenames),
    # then validate by listing files.
    try:
        infos_iter = api.list_models(search=filename, sort="downloads", direction=-1)
    except Exception:
        infos_iter = iter([])

    for info in islice(infos_iter, global_search_limit):
        rid = info.id
        if rid in tried:
            continue
        tried.add(rid)
        try:
            rel = _repo_find_relpath_for_filename(api, rid, filename)
            if rel:
                return rid, rel
        except Exception:
            continue

    return None

# ------------------------------ Download helpers ---------------------------

def ensure_dir(p: pathlib.Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def download_file(
    api: HfApi,
    repo: str,
    remote_relpath: str,
    dest: pathlib.Path,
    force: bool,
    token: Optional[str],
) -> bool:
    if dest.exists() and not force:
        return True
    ensure_dir(dest)
    try:
        path = hf_hub_download(
            repo_id=repo,
            filename=remote_relpath,          # relative path in repo
            local_dir=str(dest.parent),
            local_dir_use_symlinks=False,
            resume_download=True,
            token=token,
        )
        downloaded = pathlib.Path(path)
        if downloaded.name != dest.name:
            downloaded.rename(dest)
            return dest.exists()
        return downloaded.exists()
    except HfHubHTTPError as e:
        console.print(f"[red]HTTP error[/red] for {repo}/{remote_relpath}: {e}")
        return False
    except Exception as e:
        console.print(f"[red]Failed[/red] {repo}/{remote_relpath}: {e}")
        return False

def download_tokenizer(repo: str, dest_dir: pathlib.Path, force: bool, token: Optional[str]) -> bool:
    try:
        snapshot_download(
            repo_id=repo,
            local_dir=str(dest_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            token=token,
            ignore_regex=None if force else r".^",
        )
        return True
    except Exception as e:
        console.print(f"[yellow]Tokenizer download warning[/yellow] for {repo}: {e}")
        return False

# ----------------------------------- Main -----------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download GGUF models from base + override YAML (skipping disabled).")
    parser.add_argument("--base", required=True, help="Path to base YAML, e.g. configs/config.base.yaml")
    parser.add_argument("--override", required=False, help="Path to override YAML")
    parser.add_argument("--target", required=True, help="Directory to store downloaded GGUF files")
    parser.add_argument("--include-drafts", action="store_true", help="Also download cmd.model-draft files")
    parser.add_argument("--also-tokenizers", action="store_true", help="Also snapshot tokenizer repos from hf_tokenizer_for_model")
    parser.add_argument("--tokenizers-dir", default=None, help="Optional separate directory for tokenizers (default: <target>/tokenizers)")
    parser.add_argument("--force", action="store_true", help="Re-download even if local file exists")
    parser.add_argument("--orgs", default=None, help="Comma-separated orgs/users to prioritize (e.g. unsloth,TheBloke,bartowski,lmstudio-community,Undi95)")
    parser.add_argument("--token", default=os.environ.get("HUGGINGFACE_TOKEN"), help="HF token (or set HUGGINGFACE_TOKEN env)")
    parser.add_argument("--org-scan-limit", type=int, default=400, help="Max repos to scan per org (by downloads)")
    parser.add_argument("--global-search-limit", type=int, default=80, help="Max repos from global search to validate")
    parser.add_argument("--plan-only", action="store_true", help="Only print planned downloads and exit")
    args = parser.parse_args()

    api = HfApi(token=args.token)

    base_cfg = load_yaml(args.base)
    override_cfg = load_yaml(args.override) if args.override else {}

    models = merged_models(base_cfg, override_cfg)
    target_dir = pathlib.Path(args.target).resolve()
    tokenizers_dir = pathlib.Path(args.tokenizers_dir).resolve() if args.tokenizers_dir else (target_dir / "tokenizers")
    orgs = [o.strip() for o in args.orgs.split(",")] if args.orgs else None

    # Collect files to fetch
    to_fetch: List[Tuple[str, str, str]] = []  # (model_name, desired_filename, repo_hint_or_empty)
    tokenizer_repos: List[Tuple[str, str]] = []

    for name, cfg in models.items():
        if is_disabled(cfg):
            console.print(f"[dim]Skipping disabled[/dim] {name}")
            continue
        paths = extract_cmd_paths(cfg, include_drafts=args.include_drafts)
        if not paths:
            continue

        repo_hint = cfg.get("hf_repo_for_gguf")
        if not isinstance(repo_hint, str):
            repo_hint = ""

        for key, path in paths:
            local = normalize_local_path(str(target_dir), path)
            filename = local.name
            for shard in expand_shards(filename):
                to_fetch.append((name, shard, repo_hint))

        tok_repo = cfg.get("hf_tokenizer_for_model")
        if args.also_tokenizers and isinstance(tok_repo, str) and tok_repo.strip():
            tokenizer_repos.append((name, tok_repo.strip()))

    # Deduplicate by filename
    seen: Set[str] = set()
    deduped: List[Tuple[str, str, str]] = []
    for name, fname, repo_hint in to_fetch:
        if fname not in seen:
            seen.add(fname)
            deduped.append((name, fname, repo_hint))

    # Print plan
    table = Table(title="Planned Downloads", show_lines=False)
    table.add_column("GGUF Filename", overflow="fold")
    table.add_column("From Repo (hint or search)", overflow="fold")
    for _, fname, repo_hint in deduped:
        table.add_row(fname, repo_hint or "(search)")
    console.print(table)

    if args.plan_only:
        return

    # Download GGUFs
    success = True
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Downloading GGUF files", total=len(deduped))
        for model_name, fname, repo_hint in deduped:
            dest = target_dir / fname
            if dest.exists() and not args.force:
                progress.console.print(f"[green]Exists[/green] {dest.name}")
                progress.advance(task)
                continue

            # Determine (repo, relpath)
            repo_rel: Optional[Tuple[str, str]] = None
            if repo_hint:
                rel = _repo_find_relpath_for_filename(api, repo_hint, fname)
                if rel:
                    repo_rel = (repo_hint, rel)
                else:
                    progress.console.print(f"[yellow]Hinted repo[/yellow] {repo_hint} does not contain {fname} (by basename). Falling back to search.")
            if not repo_rel:
                where = f"orgs={','.join(orgs)}" if orgs else "global"
                progress.console.print(f"[cyan]Searching {where} for[/cyan] {fname} …")
                rr = find_repo_and_relpath_for_file(
                    api, fname, orgs,
                    org_scan_limit=args.org_scan_limit,
                    global_search_limit=args.global_search_limit,
                )
                if rr:
                    repo_rel = rr

            if not repo_rel:
                progress.console.print(f"[red]Could not find[/red] {fname}. Add 'hf_repo_for_gguf' to the model entry or expand --orgs.")
                success = False
                progress.advance(task)
                continue

            repo_id, relpath = repo_rel
            ok = download_file(api, repo_id, relpath, dest, args.force, args.token)
            if ok:
                progress.console.print(f"[green]OK[/green] {repo_id}/{relpath} → {dest}")
            else:
                progress.console.print(f"[red]FAIL[/red] {repo_id}/{relpath}")
                success = False
            progress.advance(task)

    # Download tokenizers (optional)
    if tokenizer_repos:
        console.print("\n[bold]Tokenizer snapshots[/bold]")
        for model_name, repo in tokenizer_repos:
            out_dir = tokenizers_dir / model_name
            ok = download_tokenizer(repo, out_dir, args.force, args.token)
            status = "[green]OK[/green]" if ok else "[yellow]WARN[/yellow]"
            console.print(f"{status} tokenizer {repo} → {out_dir}")

    if not success:
        sys.exit(2)

if __name__ == "__main__":
    main()
