#!/usr/bin/env python3
"""
Custom LLM Eval for local llama.cpp + llama-swap
(compatible with the src/llama_suite/... package layout)

Run as a module (recommended):
    python -m llama_suite.eval.eval
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import csv
import dataclasses
import hashlib
import json
import os
import random
import re
import signal as sig
import sqlite3
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import aiohttp
import yaml
from aiohttp import ClientConnectionError, ServerDisconnectedError

# Optional progress bar
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


# ------------------------------- Utilities ---------------------------------- #

def _repo_root_from_this_file() -> Path:
    """Infer <repo root> assuming this file is at src/llama_suite/eval/eval.py. Fallback to CWD."""
    try:
        here = Path(__file__).resolve()
        repo = here.parents[3]
        if (repo / "src").exists():
            return repo
    except Exception:
        pass
    return Path.cwd()

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def sha1_obj(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def token_f1(pred: str, gold: str) -> float:
    p = normalize_text(pred).split()
    g = normalize_text(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    num_same = sum(min(p.count(t), g.count(t)) for t in set(p))
    if num_same == 0:
        return 0.0
    precision = num_same / len(p)
    recall = num_same / len(g)
    return 2 * precision * recall / (precision + recall)

def exact_match(pred: str, gold: str) -> bool:
    return normalize_text(pred) == normalize_text(gold)

def sanitize(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)

def is_heavy_model(name: str) -> bool:
    n = (name or "").lower()
    if "speculative" in n:
        return True
    nums = re.findall(r'(\d+(?:\.\d+)?)\s*[bB]\b', name or "")
    return any(float(x) > 10.0 for x in nums)

def _base_for_health(endpoint: str) -> str:
    """llama.cpp exposes /health at the root, not under /v1; strip trailing /v1 if present."""
    base = endpoint.rstrip("/")
    return re.sub(r"/v1/?$", "", base, flags=re.IGNORECASE) or endpoint

def resolve_data_path(arg: Path) -> Path:
    rr = _repo_root_from_this_file()
    candidates: List[Path] = []
    p0 = arg if arg.is_absolute() else (Path.cwd() / arg)
    candidates.append(p0)
    candidates.append(rr / "datasets" / arg)
    if len(arg.parts) == 1:
        candidates.append(rr / "datasets" / "custom" / arg.name)
        candidates.append(rr / "custom-eval" / arg.name)
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError("Data file not found: {arg}\nTried:\n  " + "\n  ".join(str(c) for c in candidates))

def resolve_swap_config_path(arg: Optional[Path]) -> Path:
    rr = _repo_root_from_this_file()
    tried: List[Path] = []
    if arg:
        p = Path(arg)
        if p.is_absolute() and p.exists():
            return p.resolve()
        c = Path.cwd() / p; tried.append(c)
        if c.exists():
            return c.resolve()
        c = rr / p; tried.append(c)
        if c.exists():
            return c.resolve()
    default = rr / "configs" / "config.base.yaml"; tried.append(default)
    if default.exists():
        return default.resolve()
    raise FileNotFoundError("Swap config not found. Tried:\n  " + "\n  ".join(str(x) for x in tried))


# -------- llama-server process guard (POSIX + Windows) -------- #

def _list_llama_server_pids() -> List[Tuple[int, str]]:
    try:
        if sys.platform == "win32":
            out = subprocess.check_output(["tasklist"], text=True, errors="ignore")
            pids: List[Tuple[int, str]] = []
            for line in out.splitlines():
                if "llama-server" in line.lower():
                    parts = line.split()
                    for i, tok in enumerate(parts):
                        if tok.lower().startswith("llama-server"):
                            try:
                                pid = int(parts[i + 1])
                                pids.append((pid, "llama-server.exe"))
                            except Exception:
                                pass
                            break
            return pids
        else:
            out = subprocess.check_output(["ps", "-A", "-o", "pid=,command="], text=True)
            pids: List[Tuple[int, str]] = []
            for line in out.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    pid_str, cmd = line.split(None, 1)
                except ValueError:
                    continue
                if "llama-server" in cmd and "grep" not in cmd:
                    pids.append((int(pid_str), cmd))
            return pids
    except Exception:
        return []

def _kill_pid_graceful(pid: int, timeout_s: float = 2.0) -> None:
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        os.kill(pid, sig.SIGTERM)
    except Exception:
        return
    end = time.time() + timeout_s
    while time.time() < end:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.1)
    try:
        os.kill(pid, sig.SIGKILL)
    except Exception:
        pass

def guard_llama_singleton(keep: int = 1) -> None:
    procs = _list_llama_server_pids()
    if len(procs) <= keep:
        return
    procs_sorted = sorted(procs, key=lambda x: x[0])
    keep_set = {pid for pid, _ in procs_sorted[-keep:]}
    for pid, _ in procs_sorted:
        if pid not in keep_set:
            _kill_pid_graceful(pid)

def kill_all_llama_servers() -> None:
    for pid, _ in _list_llama_server_pids():
        _kill_pid_graceful(pid)


# ------------------------------- Protocol helpers --------------------------- #

class ChatRequestError(Exception):
    def __init__(self, status: int | None = None, url: str | None = None,
                 body: str | None = None, cause: Exception | None = None):
        self.status = status
        self.url = url
        self.body = body
        self.cause = cause
        super().__init__(self.__str__())

    def __str__(self) -> str:
        parts: List[str] = []
        if self.status is not None: parts.append(f"HTTP {self.status}")
        if self.url: parts.append(self.url)
        if self.cause: parts.append(f"{type(self.cause).__name__}: {self.cause}")
        if self.body:
            snippet = " ".join(self.body.strip().split())
            if len(snippet) > 240: snippet = snippet[:240] + "…"
            parts.append(f"body: {snippet}")
        return " | ".join(parts) if parts else "Chat request failed"

CHANNEL_TOKEN = r"<\|\s*([a-z0-9_:-]+)\s*\|>"
THINK_XML_TAGS = ("think", "analysis", "reasoning", "reflection", "scratchpad")

def strip_channel_tokens(s: str) -> str:
    return re.sub(CHANNEL_TOKEN, "", s, flags=re.I)

def strip_balanced_think_blocks(t: str) -> str:
    out = t
    for tag in THINK_XML_TAGS:
        out = re.sub(fr"(?is)<\s*{tag}\b[^>]*>.*?</\s*{tag}\s*>", "", out)
    return out

def extract_final(text: str) -> str:
    """Extract a sensible 'final' answer from various chat formats."""
    if not text:
        return ""
    t = str(text).strip()

    m = re.search(r"(?is)(?:<\|\s*start\s*\|\>\s*assistant\s*)?<\|\s*channel\s*\|\>\s*final\s*<\|\s*message\s*\|\>(.*)", t)
    if m:
        tail = m.group(1)
        tail = re.split(rf"(?is){CHANNEL_TOKEN}", tail)[0]
        return strip_channel_tokens(tail).strip()

    parts = re.split(r"(?is)<\|\s*channel\s*\|\>\s*[a-z0-9_:-]+\s*<\|\s*message\s*\|\>", t)
    if len(parts) > 1:
        return strip_channel_tokens(parts[-1]).strip()

    m = re.search(r"(?is)<\s*final\b[^>]*>(.*?)</\s*final\s*>", t)
    if m:
        return m.group(1).strip()

    m_list = list(re.finditer(r"(?is)final answer\s*[:\-]\s*(.*)$", t))
    if m_list:
        return m_list[-1].group(1).strip()

    m_list = list(re.finditer(r"(?is)^answer\s*[:\-]\s*(.*)$", t))
    if m_list:
        return m_list[-1].group(1).strip()

    cleaned = strip_channel_tokens(strip_balanced_think_blocks(t)).strip()
    return cleaned


# ------------------------------- Data --------------------------------------- #

@dataclass
class Task:
    id: str
    task_type: str
    question: str
    choices: Optional[List[str]] = None
    correct_idx: Optional[int] = None
    reference: Optional[str] = None
    rubric: Optional[str] = None

def load_jsonl(path: Path) -> List[Task]:
    out: List[Task] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "id" not in obj or "task_type" not in obj or "question" not in obj:
                raise ValueError(f"{path}:{i} missing required keys (id, task_type, question)")
            out.append(Task(**obj))
    return out


# ------------------------------- Client ------------------------------------- #

@dataclass
class ClientConfig:
    endpoint: str
    api_key: Optional[str] = None
    timeout_s: int = 600
    max_concurrency: int = 4
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    stop: Optional[List[str]] = None
    retries: int = 8
    backoff_cap_s: float = 16.0
    health_wait: int = 10
    health_wait_heavy: int = 60

class SQLiteCache:
    """(endpoint, model, payload_hash) -> response_json."""
    def __init__(self, path: Path):
        self.path = path
        ensure_dir(path.parent)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as con:
            con.execute("CREATE TABLE IF NOT EXISTS cache(k TEXT PRIMARY KEY, v TEXT NOT NULL)")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.path) as con:
            cur = con.execute("SELECT v FROM cache WHERE k=?", (key,))
            row = cur.fetchone()
            return json.loads(row[0]) if row else None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        with sqlite3.connect(self.path) as con:
            con.execute("INSERT OR REPLACE INTO cache(k, v) VALUES (?, ?)",
                        (key, json.dumps(value, ensure_ascii=False)))

async def _wait_until_ready(session: aiohttp.ClientSession, endpoint: str, timeout_s: int = 30) -> None:
    base = _base_for_health(endpoint)
    url = base.rstrip("/") + "/health"
    end = asyncio.get_running_loop().time() + timeout_s
    while True:
        try:
            async with session.get(url, timeout=5) as r:
                if r.status < 500:
                    return
        except Exception:
            pass
        if asyncio.get_running_loop().time() > end:
            return
        await asyncio.sleep(0.5)

def _count_llama_servers() -> int:
    return len(_list_llama_server_pids())

async def _enforce_singleton_after_swap(timeout_s: float = 20.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _count_llama_servers() <= 1:
            return
        guard_llama_singleton(keep=1)
        await asyncio.sleep(0.25)
    raise RuntimeError(f"Could not enforce single llama-server within {timeout_s}s")

class OpenAICompatClient:
    def __init__(self, cfg: ClientConfig, cache: Optional[SQLiteCache] = None):
        self.cfg = cfg
        self.cache = cache
        self.sem = asyncio.Semaphore(cfg.max_concurrency)

    async def chat(self, session: aiohttp.ClientSession, model: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": model, "messages": messages}
        if self.cfg.temperature is not None:
            payload["temperature"] = self.cfg.temperature
        if self.cfg.max_tokens is not None:
            payload["max_tokens"] = self.cfg.max_tokens
        if self.cfg.seed is not None:
            payload["seed"] = self.cfg.seed
        if self.cfg.stop:
            payload["stop"] = self.cfg.stop

        cache_key = None
        if self.cache:
            cache_key = sha1_obj([self.cfg.endpoint, model, payload])
            cached = self.cache.get(cache_key)
            if cached:
                return cached

        headers = {"Authorization": f"Bearer {self.cfg.api_key}"} if self.cfg.api_key else {}
        url = self.cfg.endpoint.rstrip("/") + "/chat/completions"

        last_exc: Optional[Exception] = None
        for attempt in range(self.cfg.retries):
            try:
                async with self.sem:
                    timeout = aiohttp.ClientTimeout(total=self.cfg.timeout_s)
                    async with session.post(url, headers=headers, json=payload, timeout=timeout) as resp:
                        text = await resp.text()
                        if resp.status >= 400:
                            err = ChatRequestError(status=resp.status, url=str(resp.url), body=text)
                            if resp.status in (502, 503, 504):
                                raise err
                            raise err
                        data = json.loads(text)
                if self.cache and cache_key:
                    self.cache.set(cache_key, data)
                return data

            except (ServerDisconnectedError, ClientConnectionError, ChatRequestError, asyncio.TimeoutError) as e:
                if isinstance(e, ChatRequestError) and (e.status not in (502, 503, 504)):
                    raise
                last_exc = e
                await asyncio.sleep(min(2 ** attempt, self.cfg.backoff_cap_s) + random.uniform(0, 0.25))
                try:
                    await _wait_until_ready(session, self.cfg.endpoint, timeout_s=5)
                except Exception:
                    pass

        raise ChatRequestError(cause=last_exc)

    @staticmethod
    def extract_text(resp: Dict[str, Any]) -> str:
        try:
            return resp["choices"][0]["message"]["content"]
        except Exception:
            return ""


# ------------------------------- Judging ------------------------------------ #

JUDGE_SYSTEM_PROMPT = (
    "You are a strict, fair evaluator. "
    "You will produce a short, non-revealing justification and a numeric score, then JSON."
)

def judge_prompt(question: str, rubric: str, answer: str) -> List[Dict[str, str]]:
    user = f"""
You are grading a model answer to a task.

Task:
\"\"\"{question}\"\"\"

Rubric (be strict, 0–10):
{rubric}

Candidate answer:
\"\"\"{answer}\"\"\"

Instructions:
- Output **strict JSON** only: {{"score": <0-10 integer>, "verdict": "pass|fail", "justification": "one(short) sentence"}}
- No extra text, no preamble, no chain-of-thought.
- Consider correctness, clarity, completeness, and adherence to the rubric.
- Use "pass" iff score >= 7.
"""
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user.strip()},
    ]

def parse_strict_json(text: str) -> Dict[str, Any]:
    m = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Judge did not return JSON: {text[:200]}")
    return json.loads(m.group(0))


# ------------------------------- Prompts ------------------------------------ #

SYSTEM_NEUTRAL = "You are a concise, helpful assistant. Do not reveal chain-of-thought."

def mcq_user_prompt(q: str, choices: List[str]) -> str:
    opts = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
    return f"{q}\n\n{opts}\n\nAnswer with a single letter (A, B, C, …) only."

def short_user_prompt(q: str) -> str:
    return f"{q}\n\nAnswer concisely in 1–3 sentences."

def gen_user_prompt(q: str) -> str:
    return f"{q}\n\nKeep the answer focused and useful."

def extract_mcq_letter(text: str, n_choices: int) -> Optional[int]:
    m = re.search(r"\b([A-Z])\b|^\s*([A-Z])\s*[\).:]?", text.upper())
    letter = (m.group(1) or m.group(2)) if m else None
    if not letter:
        return None
    idx = ord(letter) - 65
    return idx if 0 <= idx < n_choices else None


# ------------------------------- Runner ------------------------------------- #

@dataclass
class ModelSpec:
    name: str
    display_name: Optional[str] = None
    pre_switch_cmd: Optional[str] = None  # e.g. "./llama_swap.sh {model_name}"

@dataclass
class QAResult:
    task_id: str
    task_type: str
    model: str
    answer: str
    answer_raw: Optional[str] = None
    correct: Optional[int] = None
    f1: Optional[float] = None
    judge_score: Optional[int] = None
    judge_verdict: Optional[str] = None
    judge_just: Optional[str] = None
    latency_s: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

@dataclass
class RunConfig:
    data_path: Path
    logs_dir: Path
    results_dir: Path
    models: List[ModelSpec]
    judge_model: str
    client: ClientConfig
    judge_client: Optional[ClientConfig] = None
    llama_guard: bool = True
    kill_llama_on_exit: bool = True

def load_models_from_swap_config(path: Path,
                                 include_aliases: bool,
                                 include_patterns: Optional[List[str]],
                                 exclude_patterns: Optional[List[str]]) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "models" not in cfg or not isinstance(cfg["models"], dict):
        raise ValueError(f"{path} does not look like a llama-swap config with a 'models:' mapping.")

    names: List[str] = []
    for model_name, model_cfg in cfg["models"].items():
        names.append(str(model_name))
        if include_aliases and isinstance(model_cfg, dict):
            for a in model_cfg.get("aliases") or []:
                if isinstance(a, str):
                    names.append(a)

    seen: set[str] = set()
    unique: List[str] = []
    for n in names:
        if n not in seen:
            seen.add(n); unique.append(n)

    def matches_any(patterns: Optional[List[str]], s: str) -> bool:
        return any(re.search(p, s) for p in (patterns or []))

    kept = [n for n in unique
            if (not include_patterns or matches_any(include_patterns, n))
            and (not exclude_patterns or not matches_any(exclude_patterns, n))]
    if not kept:
        raise ValueError("No models selected from swap config after filtering.")
    return kept

def _pick_score(summary: dict, mode: str) -> tuple[float | None, str | None]:
    def norm(name: str):
        v = summary.get(name)
        if v is None: return None
        return v/10.0 if name == "gen_judge_score" else float(v)

    if mode == "auto":
        for k in ("gen_judge_score", "short_f1", "mcq_accuracy", "short_em"):
            v = norm(k)
            if v is not None: return v, k
        return None, None
    if mode == "composite":
        vals = [v for k in ("mcq_accuracy","short_f1","gen_judge_score") if (v:=norm(k)) is not None]
        return (sum(vals)/len(vals), "composite") if vals else (None, None)
    v = norm(mode)
    return (v, mode) if v is not None else (None, None)

class Benchmark:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        ensure_dir(cfg.logs_dir)
        ensure_dir(cfg.results_dir)

        # Cache lives ONLY in logs/
        self.cache = SQLiteCache(cfg.logs_dir / "cache.sqlite3")

        self.client = OpenAICompatClient(cfg.client, cache=self.cache)
        self.judge_client = OpenAICompatClient(cfg.judge_client or cfg.client, cache=self.cache)

    def _maybe_swap(self, spec: ModelSpec) -> None:
        if spec.pre_switch_cmd:
            cmd = spec.pre_switch_cmd.format(model_name=spec.name, display_name=(spec.display_name or spec.name))
            print(f"[swap] Running: {cmd}")
            subprocess.run(cmd, shell=True, check=True)

    async def _ask_one(self, session: aiohttp.ClientSession, model: str, t: Task) -> Tuple[QAResult, Dict[str, Any]]:
        if t.task_type == "mcq":
            user = mcq_user_prompt(t.question, t.choices or [])
        elif t.task_type == "short":
            user = short_user_prompt(t.question)
        else:
            user = gen_user_prompt(t.question)

        messages = [{"role": "system", "content": SYSTEM_NEUTRAL},
                    {"role": "user", "content": user}]

        resp = await self.client.chat(session, model, messages)
        text = self.client.extract_text(resp) or ""
        usage = resp.get("usage", {}) or {}
        final_answer = extract_final(text).strip()
        return (
            QAResult(
                task_id=t.id,
                task_type=t.task_type,
                model=model,
                answer=final_answer,
                answer_raw=text.strip(),
                latency_s=resp.get("_latency_s"),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
            ),
            resp,
        )

    async def _judge_one(self, session: aiohttp.ClientSession, answer: str, t: Task) -> Tuple[int, str, str]:
        messages = judge_prompt(
            question=t.question,
            rubric=t.rubric or (f"Match the reference answer.\nReference: {t.reference or ''}\nScore 0–10."),
            answer=answer,
        )
        resp = await self.judge_client.chat(session, self.cfg.judge_model, messages)
        text = self.client.extract_text(resp)
        obj = parse_strict_json(text)
        score = int(obj["score"])
        verdict = str(obj.get("verdict", "pass"))
        just = str(obj.get("justification", "")).strip()
        return score, verdict, just

    async def run_for_model(self, spec: ModelSpec, tasks: List[Task]) -> List[QAResult]:
        if self.cfg.llama_guard:
            try:
                guard_llama_singleton(keep=1)
            except Exception as e:
                print(f"[warn] llama guard (pre) failed: {e}")

        self._maybe_swap(spec)

        dsname = spec.display_name or spec.name
        heavy = is_heavy_model(spec.name)

        # Snapshot mutable client state
        orig_sem = self.client.sem
        orig_retries = self.client.cfg.retries
        orig_backoff = self.client.cfg.backoff_cap_s

        try:
            self.client.sem = asyncio.Semaphore(1 if heavy else self.client.cfg.max_concurrency)
            if heavy:
                self.client.cfg.retries = max(self.client.cfg.retries, 12)
                self.client.cfg.backoff_cap_s = max(self.client.cfg.backoff_cap_s, 32.0)

            async with aiohttp.ClientSession() as session:
                wait_s = self.client.cfg.health_wait_heavy if heavy else self.client.cfg.health_wait
                try:
                    await _wait_until_ready(session, self.client.cfg.endpoint, timeout_s=wait_s)
                except Exception:
                    pass

                if self.cfg.llama_guard:
                    await _enforce_singleton_after_swap(timeout_s=15.0)

                ask_tasks = [asyncio.create_task(self._ask_one(session, spec.name, t)) for t in tasks]
                for fut, t in zip(ask_tasks, tasks):
                    try:
                        fut.set_name(t.id)
                    except Exception:
                        pass

                results: List[QAResult] = []
                task_index = {t.id: t for t in tasks}

                bar = tqdm(total=len(ask_tasks),
                           desc=f"[run] Model: {spec.name}",
                           unit="task",
                           dynamic_ncols=True,
                           leave=True,
                           mininterval=0.1) if tqdm else None
                try:
                    for fut in asyncio.as_completed(ask_tasks):
                        try:
                            r, _ = await fut
                            results.append(r)
                        except Exception as e:
                            tid = getattr(fut, "get_name", lambda: "?")()
                            print(f"[warn] {spec.name} [{tid}] failed: {e}")
                        finally:
                            if bar:
                                bar.update(1)  # type: ignore
                finally:
                    if bar:
                        bar.close()  # type: ignore

                # Deterministic scoring
                for r in results:
                    t = task_index[r.task_id]
                    if t.task_type == "mcq":
                        pred_idx = extract_mcq_letter(r.answer, len(t.choices or []))
                        r.correct = int(pred_idx is not None and t.correct_idx is not None and pred_idx == t.correct_idx)
                    elif t.task_type == "short" and t.reference:
                        r.correct = int(exact_match(r.answer, t.reference))
                        r.f1 = token_f1(r.answer, t.reference)

                # Judge (gen; or short with rubric)
                judge_targets = [r for r in results if (task_index[r.task_id].task_type == "gen"
                                                        or (task_index[r.task_id].task_type == "short"
                                                            and task_index[r.task_id].rubric))]
                judged = await asyncio.gather(
                    *[self._judge_one(session, r.answer, task_index[r.task_id]) for r in judge_targets],
                    return_exceptions=False
                )
                for (score, verdict, just), r in zip(judged, judge_targets):
                    r.judge_score = score
                    r.judge_verdict = verdict
                    r.judge_just = just

            # Persist artifacts (split: logs vs results)
            self._write_csv(self.cfg.logs_dir / f"raw_{sanitize(dsname)}.csv", results)
            self._write_json(self.cfg.logs_dir / f"raw_{sanitize(dsname)}.json",
                            [dataclasses.asdict(r) for r in results])
            model_summary = self._summarize(results, tasks)
            self._write_json(self.cfg.logs_dir / f"summary_{sanitize(dsname)}.json", model_summary)
            return results

        finally:
            self.client.sem = orig_sem
            self.client.cfg.retries = orig_retries
            self.client.cfg.backoff_cap_s = orig_backoff

    def _summarize(self, results: List[QAResult], tasks: List[Task]) -> Dict[str, Any]:
        task_index = {t.id: t for t in tasks}
        mcq = [r for r in results if task_index[r.task_id].task_type == "mcq"]
        short = [r for r in results if task_index[r.task_id].task_type == "short"]
        gen = [r for r in results if task_index[r.task_id].task_type == "gen"]

        def safe_mean(xs: Sequence[Optional[float]]) -> Optional[float]:
            vals: List[float] = [cast(float, x) for x in xs if x is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        return {
            "count_total": len(results),
            "count_mcq": len(mcq),
            "count_short": len(short),
            "count_gen": len(gen),
            "mcq_accuracy": safe_mean([r.correct for r in mcq if r.correct is not None]),
            "short_em": safe_mean([r.correct for r in short if r.correct is not None]),
            "short_f1": safe_mean([r.f1 for r in short if r.f1 is not None]),
            "gen_judge_score": safe_mean([r.judge_score for r in gen if r.judge_score is not None]),
            "latency_s_avg": safe_mean([r.latency_s for r in results if r.latency_s is not None]),
            "tokens_prompt_avg": safe_mean([r.prompt_tokens for r in results if r.prompt_tokens is not None]),
            "tokens_completion_avg": safe_mean([r.completion_tokens for r in results if r.completion_tokens is not None]),
            "tokens_total_avg": safe_mean([r.total_tokens for r in results if r.total_tokens is not None]),
        }

    @staticmethod
    def _write_csv(path: Path, rows: List[QAResult]) -> None:
        ensure_dir(path.parent)
        fields = [f.name for f in dataclasses.fields(QAResult)]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(asdict(r))

    @staticmethod
    def _write_json(path: Path, obj: Any) -> None:
        ensure_dir(path.parent)
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)


# ------------------------------- CLI ---------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Custom LLM Benchmark for local llama.cpp + llama-swap")
    p.add_argument("--data", type=Path, required=True)

    # New: explicit dirs; legacy --out-dir acts as a base
    p.add_argument("--logs-dir", type=Path, default=None,
                   help="Directory for raw_* artifacts. Default: <repo>/runs/eval/logs/<YYYY-MM-DD>.")
    p.add_argument("--results-dir", type=Path, default=None,
                   help="Directory for summaries and graphs. Default: <repo>/runs/eval/results/<YYYY-MM-DD>.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Legacy/base dir: uses <out-dir>/logs and <out-dir>/results (no date suffix).")

    p.add_argument("--endpoint", type=str, required=True)
    p.add_argument("--api-key", type=str, default=None)

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--models", nargs="+", metavar="MODEL",
                       help='One or more model names to send in the "model" field.')
    group.add_argument("--models-from-swap", action="store_true",
                       help="Load model names from a llama-swap YAML instead of --models.")

    p.add_argument("--judge-model", type=str, required=True)
    p.add_argument("--swap-cmd", type=str, default=None)

    # Model/gen options
    p.add_argument("--temp", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--stop", type=str, nargs="*", default=None)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--concurrency", type=int, default=4)

    # llama-swap config helpers
    p.add_argument("--swap-config", type=Path, default=Path("configs/config.base.yaml"))
    p.add_argument("--use-aliases", action="store_true")
    p.add_argument("--include", nargs="*", default=None)
    p.add_argument("--exclude", nargs="*", default=None)

    # summary csv options
    p.add_argument("--summary-metric",
                   choices=["auto", "composite", "mcq_accuracy", "short_em", "short_f1", "gen_judge_score"],
                   default="auto")
    p.add_argument("--summary-round", type=int, default=4)
    p.add_argument("--summary-include-metric", action="store_true")

    # HTTP/backoff & health
    p.add_argument("--http-retries", type=int, default=8)
    p.add_argument("--backoff-cap", type=float, default=16.0)
    p.add_argument("--health-wait", type=int, default=10)
    p.add_argument("--health-wait-heavy", type=int, default=60)

    # Llama guard defaults OFF
    p.add_argument("--llama-guard", dest="llama_guard", action="store_true",
                   help="(Advanced) Try to kill extra llama-server PIDs. Disable when using llama-swap.")
    p.add_argument("--no-llama-guard", dest="llama_guard", action="store_false",
                   help="Default. Do not kill llama-server; assume llama-swap manages lifecycle.")
    p.set_defaults(llama_guard=False)

    # Exit cleanup defaults ON
    p.add_argument("--kill-llama-on-exit", action="store_true",
                   help="Kill all llama-server processes on exit (Ctrl-C or error).")
    p.set_defaults(kill_llama_on_exit=True)
    return p.parse_args()

def _default_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    """Return (logs_dir, results_dir) based on args and sensible defaults."""
    rr = _repo_root_from_this_file()
    today = time.strftime("%Y-%m-%d")

    if args.logs_dir or args.results_dir:
        logs_dir = Path(args.logs_dir) if args.logs_dir else rr / "runs" / "eval" / "logs" / today
        results_dir = Path(args.results_dir) if args.results_dir else rr / "runs" / "eval" / "results" / today
        return logs_dir, results_dir

    if args.out_dir:
        base = Path(args.out_dir)
        return base / "logs", base / "results"

    # Pure default
    return (rr / "runs" / "eval" / "logs" / today,
            rr / "runs" / "eval" / "results" / today)

async def run_all(bench: Benchmark, run_cfg: RunConfig, tasks: List[Task]) -> Dict[str, Dict]:
    summaries: Dict[str, Dict] = {}
    for spec in run_cfg.models:
        try:
            await bench.run_for_model(spec, tasks)
        except Exception as e:
            print(f"[warn] Model {spec.name} failed: {e}")

        label = spec.display_name or spec.name
        summ_path = run_cfg.logs_dir / f"summary_{sanitize(label)}.json"

        if summ_path.exists():
            try:
                with summ_path.open("r", encoding="utf-8") as f:
                    summaries[label] = json.load(f)
            except Exception as e:
                print(f"[warn] Could not read summary for {label}: {e}")
        else:
            print(f"[warn] No summary file for {label}; skipping.")
    return summaries

def main() -> None:
    args = parse_args()

    data_path = resolve_data_path(args.data)

    # Resolve swap config
    swap_cfg_path = resolve_swap_config_path(args.swap_config)
    if args.swap_config and Path(args.swap_config) != swap_cfg_path:
        print(f"[info] Using swap config at: {swap_cfg_path}")

    logs_dir, results_dir = _default_dirs(args)
    ensure_dir(logs_dir); ensure_dir(results_dir)

    tasks = load_jsonl(data_path)

    # Build model list
    if args.models_from_swap:
        selected = load_models_from_swap_config(
            swap_cfg_path,
            include_aliases=args.use_aliases,
            include_patterns=args.include,
            exclude_patterns=args.exclude,
        )
    else:
        selected = list(args.models or [])

    models: List[ModelSpec] = [
        ModelSpec(name=m, display_name=None, pre_switch_cmd=(args.swap_cmd or None))
        for m in selected
    ]
    print(f"[info] Selected {len(models)} models: {', '.join([m.name for m in models])}")

    client_cfg = ClientConfig(
        endpoint=args.endpoint,
        api_key=args.api_key,
        timeout_s=args.timeout,
        max_concurrency=args.concurrency,
        temperature=args.temp,
        max_tokens=args.max_tokens,
        seed=args.seed,
        stop=args.stop,
        retries=args.http_retries,
        backoff_cap_s=args.backoff_cap,
        health_wait=args.health_wait,
        health_wait_heavy=args.health_wait_heavy,
    )
    run_cfg = RunConfig(
        data_path=data_path,
        logs_dir=logs_dir,
        results_dir=results_dir,
        models=models,
        judge_model=args.judge_model,
        client=client_cfg,
        judge_client=None,
        llama_guard=args.llama_guard,
        kill_llama_on_exit=args.kill_llama_on_exit,
    )

    bench = Benchmark(run_cfg)

    def _cleanup() -> None:
        if run_cfg.kill_llama_on_exit:
            print("[info] Cleaning up llama-server processes…")
            kill_all_llama_servers()

    atexit.register(_cleanup)

    def _signal_handler(signum, _frame) -> None:
        _cleanup()
        raise SystemExit(130 if signum == sig.SIGINT else 143)

    for s in (sig.SIGINT, sig.SIGTERM):
        try:
            sig.signal(s, _signal_handler)
        except Exception:
            pass

    all_summaries = asyncio.run(run_all(bench, run_cfg, tasks))

    # Combined summary + scores go to results/
    with (run_cfg.results_dir / "summary_all.json").open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    scores_path = run_cfg.results_dir / "scores.csv"
    with scores_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["model", "score"]
        if getattr(args, "summary_include_metric", False):
            header.append("metric")
        writer.writerow(header)
        for model_name, summary in all_summaries.items():
            score, metric_used = _pick_score(summary, args.summary_metric)
            if score is None:
                continue
            row = [model_name, round(score, args.summary_round)]
            if getattr(args, "summary_include_metric", False):
                row.append(metric_used)
            writer.writerow(row)

    # Plot to results/
    import importlib
    plot_summary = None
    for mod_name in ("llama_suite.utils.graph", (__package__ + ".graph") if __package__ else None):
        if not mod_name:
            continue
        try:
            mod = importlib.import_module(mod_name)
            plot_summary = getattr(mod, "plot_summary")
            break
        except Exception:
            continue

    if plot_summary is not None:
        try:
            plot_summary(
                summary_path=run_cfg.results_dir / "summary_all.json",
                out_path=run_cfg.results_dir / f"plot_{args.summary_metric}.png",
                metric=args.summary_metric,
            )
        except Exception as e:
            print(f"[warn] Plotting failed (non-fatal): {e}")
    else:
        print("[info] Plotting skipped (graph module not found).")

    print(f"[done] Logs:    {run_cfg.logs_dir.resolve()}")
    print(f"[done] Results: {run_cfg.results_dir.resolve()}")

if __name__ == "__main__":
    main()
