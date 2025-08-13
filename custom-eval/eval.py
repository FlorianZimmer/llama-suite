#!/usr/bin/env python3
"""
Custom LLM Eval for local llama.cpp + llama-swap
=====================================================

Features
- Load your *own* private tasks (JSONL) to avoid training contamination.
- Hit a local OpenAI-compatible endpoint (llama.cpp /v1/chat/completions).
- Swap models per run:
    • by passing different "model" names (if your server exposes several), or
    • by running an optional shell command (e.g., a llama-swap script) before a model’s batch.
- Task types: "mcq", "short", "gen".
- Metrics:
    • MCQ accuracy
    • Short-answer EM/F1 vs reference
    • LLM-as-Judge scoring (0–10) for "gen" (with brief justification, no chain-of-thought).
- Concurrency with asyncio, simple SQLite response cache, and reproducibility via seed.
- Saves raw results + per-model summary CSV/JSON.

Dataset format (JSONL)
----------------------
Each line is a JSON object with:
{
  "id": "q001",
  "task_type": "mcq" | "short" | "gen",
  "question": "…",
  // MCQ:
  "choices": ["A …", "B …", "C …", "D …"],       // required for mcq
  "correct_idx": 2,                                // required for mcq (0-based)
  // SHORT:
  "reference": "gold short answer",                // optional for mcq; required for short
  // GEN:
  "rubric": "Precise rubric the judge should use"  // required for gen; optional for short
}

Example lines:
{"id":"q1","task_type":"mcq","question":"2+2=?","choices":["3","4","5","6"],"correct_idx":1}
{"id":"q2","task_type":"short","question":"Define TCP handshake.","reference":"A three-way SYN, SYN-ACK, ACK process."}
{"id":"q3","task_type":"gen","question":"Explain zero trust in 4-6 sentences for a CTO.","rubric":"Clarity, correctness, concision, avoids buzzwords, concrete practices. Score 0-10."}

How to run (examples)
---------------------
python bench.py \
  --data tasks.jsonl \
  --out-dir runs/2025-08-10 \
  --endpoint http://127.0.0.1:8080/v1 \
  --judge-model "Qwen3-30B-A3B-Instruct-2507" \
  --models "Qwen3-30B-A3B-Instruct-2507" "gemma-3-12b-it"

# With llama-swap: run a command before each model’s batch.
python bench.py \
  --data tasks.jsonl \
  --out-dir runs/2025-08-10 \
  --endpoint http://127.0.0.1:8080/v1 \
  --judge-model "qwen2.5-7b-instruct" \
  --models "active-model-slot" \
  --swap-cmd "./llama_swap.sh {model_name}"

python bench.py \
  --data tasks.jsonl \
  --out-dir runs/2025-08-10 \
  --endpoint http://127.0.0.1:8080/v1 \
  --judge-model "Qwen3-30B-A3B-Instruct-2507" \
  --models-from-swap --swap-config config.base.yaml \
  --summary-include-metric \
  --use-aliases \
  --include "^(Qwen|gemma)" \
  --exclude "SpeculativeDecoding"

Notes
-----
- Set temperature low (e.g., 0.2) + fixed seed for reproducibility.
- Judge prompts force strict JSON output (no chain-of-thought).
- You can use the same local endpoint for the judge; just give a different model via --judge-model,
  or reuse the same one if needed.

"""

from __future__ import annotations

import argparse
import asyncio
import aiohttp
import csv
import dataclasses
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import yaml
import atexit
import signal as sig
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import random
from aiohttp import ClientConnectionError, ServerDisconnectedError, ClientResponseError
try:
    from tqdm.asyncio import tqdm as tqdm_async, as_completed as tqdm_as_completed
    HAS_TQDM = True
    HAS_TQDM_AS_COMPLETED = True
except Exception:
    try:
        # fallback to classic tqdm (no async helper)
        from tqdm import tqdm as tqdm_async
        HAS_TQDM = True
        HAS_TQDM_AS_COMPLETED = False
        tqdm_as_completed = None  # sentinel
    except Exception:
        HAS_TQDM = False
        HAS_TQDM_AS_COMPLETED = False
        tqdm_as_completed = None


# ------------------------------- Utilities ---------------------------------- #

# -------- Llama-server process guard (POSIX-first) -------- #

def _list_llama_server_pids() -> list[tuple[int, str]]:
    """Return [(pid, cmdline), ...] for running llama-server processes."""
    try:
        if sys.platform == "win32":
            # crude fallback: tasklist filter
            out = subprocess.check_output(["tasklist"], text=True, errors="ignore")
            pids = []
            for line in out.splitlines():
                if "llama-server" in line.lower():
                    parts = line.split()
                    # e.g., 'llama-server.exe            1234 Console    1     20,000 K'
                    for i, tok in enumerate(parts):
                        if tok.lower().startswith("llama-server"):
                            try:
                                pid = int(parts[i+1])
                                pids.append((pid, "llama-server.exe"))
                            except Exception:
                                pass
                            break
            return pids
        else:
            out = subprocess.check_output(["ps", "-A", "-o", "pid=,command="], text=True)
            pids = []
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
    """SIGTERM then SIGKILL if needed (POSIX); taskkill on Windows."""
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        os.kill(pid, sig.SIGTERM)
    except Exception:
        return
    # wait briefly
    end = time.time() + timeout_s
    while time.time() < end:
        try:
            os.kill(pid, 0)
        except OSError:
            return  # dead
        time.sleep(0.1)
    # force kill
    try:
        os.kill(pid, sig.SIGKILL)
    except Exception:
        pass

def guard_llama_singleton(keep: int = 1) -> None:
    """Ensure at most `keep` llama-server processes remain (kill extras)."""
    procs = _list_llama_server_pids()
    if len(procs) <= keep:
        return
    # keep newest by PID (usually correlates with start time)
    procs_sorted = sorted(procs, key=lambda x: x[0])
    keep_set = {pid for pid, _ in procs_sorted[-keep:]}
    for pid, _ in procs_sorted:
        if pid not in keep_set:
            _kill_pid_graceful(pid)

def kill_all_llama_servers() -> None:
    """Kill ALL llama-server processes (last-resort cleanup)."""
    for pid, _ in _list_llama_server_pids():
        _kill_pid_graceful(pid)


class ChatRequestError(Exception):
    """Richer error with HTTP status, URL, and a short body snippet."""
    def __init__(self, status: int | None = None, url: str | None = None,
                 body: str | None = None, cause: Exception | None = None):
        self.status = status
        self.url = url
        self.body = body
        self.cause = cause
        super().__init__(self.__str__())

    def __str__(self) -> str:
        parts = []
        if self.status is not None:
            parts.append(f"HTTP {self.status}")
        if self.url:
            parts.append(self.url)
        if self.cause:
            parts.append(f"{type(self.cause).__name__}: {self.cause}")
        if self.body:
            snippet = " ".join(self.body.strip().split())
            if len(snippet) > 240:
                snippet = snippet[:240] + "…"
            parts.append(f"body: {snippet}")
        return " | ".join(parts) if parts else "Chat request failed"


def is_heavy_model(name: str) -> bool:
    """
    Heavy if:
      - name contains 'speculative' (case-insensitive), OR
      - any '<number>B' token > 10 (e.g., 15B, 32B, 70B).
    """
    n = (name or "").lower()
    if "speculative" in n:
        return True
    # capture numbers before 'b'/'B' (supports 7B, 12.8B, 32B, etc.)
    nums = re.findall(r'(\d+(?:\.\d+)?)\s*[bB]\b', name or "")
    return any(float(x) > 10.0 for x in nums)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sha1_obj(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def token_f1(pred: str, gold: str) -> float:
    """Simple token-level F1 for short answers."""
    p = normalize_text(pred).split()
    g = normalize_text(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = {}
    for t in p:
        common[t] = min(p.count(t), g.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p)
    recall = num_same / len(g)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> bool:
    return normalize_text(pred) == normalize_text(gold)

CHANNEL_TOKEN = r"<\|\s*([a-z0-9_:-]+)\s*\|>"  # <|channel|>, <|start|>, etc.

def strip_channel_tokens(s: str) -> str:
    return re.sub(CHANNEL_TOKEN, "", s, flags=re.I)

THINK_XML_TAGS = ("think", "analysis", "reasoning", "reflection", "scratchpad")

def strip_balanced_think_blocks(t: str) -> str:
    """Remove only balanced XML-ish think blocks like <think>...</think> (case-insensitive)."""
    out = t
    for tag in THINK_XML_TAGS:
        out = re.sub(fr"(?is)<\s*{tag}\b[^>]*>.*?</\s*{tag}\s*>", "", out)
    return out

def extract_final(text: str) -> str:
    if not text:
        return ""
    t = text.strip()

    # 1) DeepSeek/OpenRouter-style channels: prefer explicit 'final' channel.
    #    Allow optional '<|start|>assistant' before it.
    m = re.search(r"(?is)(?:<\|\s*start\s*\|\>\s*assistant\s*)?<\|\s*channel\s*\|\>\s*final\s*<\|\s*message\s*\|\>(.*)", t)
    if m:
        tail = m.group(1)
        tail = re.split(rf"(?is){CHANNEL_TOKEN}", tail)[0]  # cut at next token if any
        return strip_channel_tokens(tail).strip()

    # 2) If there are channel/message blocks, take the LAST message block (common “analysis/final” pattern)
    parts = re.split(r"(?is)<\|\s*channel\s*\|\>\s*[a-z0-9_:-]+\s*<\|\s*message\s*\|\>", t)
    if len(parts) > 1:
        return strip_channel_tokens(parts[-1]).strip()

    # 3) <final>…</final>
    m = re.search(r"(?is)<\s*final\b[^>]*>(.*?)</\s*final\s*>", t)
    if m:
        return m.group(1).strip()

    # 4) “Final answer:” (take the LAST occurrence)
    m = list(re.finditer(r"(?is)final answer\s*[:\-]\s*(.*)$", t))
    if m:
        return m[-1].group(1).strip()

    # 5) “Answer:” (LAST occurrence)
    m = list(re.finditer(r"(?is)^answer\s*[:\-]\s*(.*)$", t))
    if m:
        return m[-1].group(1).strip()

    # 6) Otherwise: strip ONLY balanced think blocks + channel tokens and return the FULL remainder (no paragraph slicing).
    cleaned = strip_channel_tokens(strip_balanced_think_blocks(t)).strip()
    return cleaned


def load_models_from_swap_config(path: Path, include_aliases: bool,
                                 include_patterns: Optional[List[str]],
                                 exclude_patterns: Optional[List[str]]) -> List[str]:
    """
    Reads a llama-swap YAML and returns model names (and optionally aliases).
    Only top-level `models:` keys are used. Comments / disabled blocks are ignored by YAML.
    """
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict) or "models" not in cfg or not isinstance(cfg["models"], dict):
        raise ValueError(f"{path} does not look like a llama-swap config with a 'models:' mapping.")

    names: List[str] = []
    for model_name, model_cfg in cfg["models"].items():
        names.append(str(model_name))
        if include_aliases:
            aliases = []
            if isinstance(model_cfg, dict):
                aliases = model_cfg.get("aliases") or []
            for a in aliases:
                if isinstance(a, str):
                    names.append(a)

    # de-dup, preserve order
    seen = set()
    unique = []
    for n in names:
        if n not in seen:
            seen.add(n)
            unique.append(n)

    def matches_any(patterns: Optional[List[str]], s: str) -> bool:
        if not patterns:
            return False
        return any(re.search(p, s) for p in patterns)

    # apply filters
    kept = []
    for n in unique:
        if include_patterns and not matches_any(include_patterns, n):
            continue
        if exclude_patterns and matches_any(exclude_patterns, n):
            continue
        kept.append(n)

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


# ------------------------------- Caching ------------------------------------ #

class SQLiteCache:
    """(endpoint, model, payload_hash) -> response_json."""
    def __init__(self, path: Path):
        self.path = path
        ensure_dir(path.parent)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.path) as con:
            con.execute(
                """CREATE TABLE IF NOT EXISTS cache(
                       k TEXT PRIMARY KEY,
                       v TEXT NOT NULL
                   )"""
            )

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.path) as con:
            cur = con.execute("SELECT v FROM cache WHERE k=?", (key,))
            row = cur.fetchone()
            if not row:
                return None
            return json.loads(row[0])

    def set(self, key: str, value: Dict[str, Any]) -> None:
        with sqlite3.connect(self.path) as con:
            con.execute("INSERT OR REPLACE INTO cache(k, v) VALUES (?, ?)", (key, json.dumps(value, ensure_ascii=False)))

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

async def _wait_until_ready(session, base_url: str, timeout_s: int = 30):
    """Best-effort: wait until the backend responds (or timeout)."""
    url = base_url.rstrip("/") + "/health"
    end = asyncio.get_running_loop().time() + timeout_s
    while True:
        try:
            async with session.get(url, timeout=5) as r:
                # treat any non-5xx as "ready" (200/404/401 are all fine)
                if r.status < 500:
                    return
        except Exception:
            pass
        if asyncio.get_running_loop().time() > end:
            return
        await asyncio.sleep(0.5)

class OpenAICompatClient:
    def __init__(self, cfg: ClientConfig, cache: Optional[SQLiteCache] = None):
        self.cfg = cfg
        self.cache = cache
        self.sem = asyncio.Semaphore(cfg.max_concurrency)

    async def chat(self, session: aiohttp.ClientSession, model: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
        }
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

        headers = {}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        url = self.cfg.endpoint.rstrip("/") + "/chat/completions"

        attempts = self.cfg.retries
        cap = self.cfg.backoff_cap_s
        last_exc = None

        for attempt in range(self.cfg.retries):
            try:
                async with self.sem:
                    timeout = aiohttp.ClientTimeout(total=self.cfg.timeout_s)
                    async with session.post(url, headers=headers, json=payload, timeout=timeout) as resp:
                        text = await resp.text()
                        if resp.status >= 400:
                            err = ChatRequestError(status=resp.status, url=str(resp.url), body=text)
                            # 5xx -> retry; 4xx -> fail fast
                            if resp.status in (502, 503, 504):
                                raise err
                            raise err  # will be re-raised below (no retry)
                        data = json.loads(text)
                    if self.cache and cache_key:
                        self.cache.set(cache_key, data)
                    return data

            except (ServerDisconnectedError, ClientConnectionError, ChatRequestError, asyncio.TimeoutError) as e:
                # Only retry for network errors or 5xx ChatRequestError
                if isinstance(e, ChatRequestError) and (e.status not in (502, 503, 504)):
                    raise  # non-transient HTTP error → bubble up
                last_exc = e
                await asyncio.sleep(min(2 ** attempt, self.cfg.backoff_cap_s) + random.uniform(0, 0.25))
                try:
                    await _wait_until_ready(session, self.cfg.endpoint, timeout_s=5)
                except Exception:
                    pass
                continue

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
    "You will produce a short, *non-revealing* justification and a numeric score, then JSON."
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
    # Extract the first JSON object if extra text slipped through.
    m = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Judge did not return JSON: {text[:200]}")
    return json.loads(m.group(0))


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
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            out.append(Task(**obj))
    return out


# ------------------------------- Prompts ------------------------------------ #

SYSTEM_NEUTRAL = "You are a concise, helpful assistant. Do not reveal chain-of-thought."

def mcq_user_prompt(q: str, choices: List[str]) -> str:
    opts = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
    return (
        f"{q}\n\n"
        f"{opts}\n\n"
        "Answer with a single letter (A, B, C, …) only."
    )

def short_user_prompt(q: str) -> str:
    return f"{q}\n\nAnswer concisely in 1–3 sentences."

def gen_user_prompt(q: str) -> str:
    return f"{q}\n\nKeep the answer focused and useful."


def extract_mcq_letter(text: str, n_choices: int) -> Optional[int]:
    m = re.search(r"\b([A-Z])\b", text.strip())
    if not m:
        return None
    idx = ord(m.group(1)) - 65
    if 0 <= idx < n_choices:
        return idx
    return None


# ------------------------------- Runner ------------------------------------- #

@dataclass
class ModelSpec:
    name: str                 # value to send in "model" field for /chat/completions
    display_name: Optional[str] = None  # pretty label; defaults to name
    pre_switch_cmd: Optional[str] = None  # e.g. "./llama_swap.sh {model_name}"


@dataclass
class RunConfig:
    data_path: Path
    out_dir: Path
    models: List[ModelSpec]
    judge_model: str
    client: ClientConfig
    judge_client: Optional[ClientConfig] = None
    llama_guard: bool = True
    kill_llama_on_exit: bool = True


@dataclass
class QAResult:
    task_id: str
    task_type: str
    model: str
    answer: str
    answer_raw: Optional[str] = None
    correct: Optional[int] = None   # for mcq: 1/0; short: EM 1/0; gen: None
    f1: Optional[float] = None      # for short
    judge_score: Optional[int] = None
    judge_verdict: Optional[str] = None
    judge_just: Optional[str] = None
    latency_s: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class Benchmark:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        ensure_dir(cfg.out_dir)
        self.cache = SQLiteCache(cfg.out_dir / "cache.sqlite3")
        self.client = OpenAICompatClient(cfg.client, cache=self.cache)
        self.judge_client = OpenAICompatClient(cfg.judge_client or cfg.client, cache=self.cache)

    def _maybe_swap(self, spec: ModelSpec):
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
        text = self.client.extract_text(resp)
        usage = resp.get("usage", {}) or {}
        final_answer = extract_final(text)
        return (
            QAResult(
                task_id=t.id,
                task_type=t.task_type,
                model=model,
                answer=final_answer.strip(),
                answer_raw=(text or "").strip(),
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
        if getattr(self.cfg, "llama_guard", True):
            try:
                guard_llama_singleton(keep=1)
            except Exception as e:
                print(f"[warn] llama guard failed: {e}")

        self._maybe_swap(spec)
        dsname = spec.display_name or spec.name

        heavy = is_heavy_model(spec.name)
        self.client.sem = asyncio.Semaphore(1 if heavy else self.client.cfg.max_concurrency)

        async with aiohttp.ClientSession() as session:
            # longer readiness wait for heavy/speculative models
            wait_s = self.client.cfg.health_wait_heavy if heavy else self.client.cfg.health_wait
            try:
                await _wait_until_ready(session, self.client.cfg.endpoint, timeout_s=wait_s)
            except Exception:
                pass

            # bump retries/backoff for heavy models (no warmup)
            if heavy:
                self.client.cfg.retries = max(self.client.cfg.retries, 12)
                self.client.cfg.backoff_cap_s = max(self.client.cfg.backoff_cap_s, 32.0)

            # Build tasks and name them with the dataset id
            ask_tasks = []
            for t in tasks:
                fut = asyncio.create_task(self._ask_one(session, spec.name, t))
                try:
                    fut.set_name(t.id)  # Python 3.8+
                except Exception:
                    pass
                ask_tasks.append(fut)

            results: List[QAResult] = []

            if HAS_TQDM:
                if HAS_TQDM_AS_COMPLETED and tqdm_as_completed is not None:
                    # Preferred: async progress helper
                    async for fut in tqdm_as_completed(
                        ask_tasks,
                        total=len(ask_tasks),
                        desc=f"[run] Model: {spec.name}",
                        unit="task",
                        dynamic_ncols=True,
                        leave=True,
                        file=sys.stdout,          # <-- ensure it prints to stdout
                        mininterval=0.1,
                        smoothing=0,
                    ):
                        try:
                            r, _raw = await fut
                            results.append(r)
                        except Exception as e:
                            task_id = getattr(fut, "get_name", lambda: "?")()
                            print(f"[warn] {spec.name} [{task_id}] failed: {e}")
                else:
                    # Fallback: manual tqdm + asyncio.as_completed
                    bar = tqdm_async(
                        total=len(ask_tasks),
                        desc=f"[run] Model: {spec.name}",
                        unit="task",
                        dynamic_ncols=True,
                        leave=True,
                        file=sys.stdout,          # <-- ensure it prints to stdout
                        mininterval=0.1,
                        smoothing=0,
                    )
                    bar.update(0)   # draw immediately
                    bar.refresh()
                    try:
                        for fut in asyncio.as_completed(ask_tasks):
                            try:
                                r, _raw = await fut
                                results.append(r)
                            except Exception as e:
                                task_id = getattr(fut, "get_name", lambda: "?")()
                                print(f"[warn] {spec.name} [{task_id}] failed: {e}")
                            finally:
                                bar.update(1)
                                bar.refresh()
                    finally:
                        bar.close()
            else:
                # No tqdm installed: simple fallback
                print(f"[run] Model: {spec.name}")
                for fut in asyncio.as_completed(ask_tasks):
                    try:
                        r, _raw = await fut
                        results.append(r)
                    except Exception as e:
                        task_id = getattr(fut, "get_name", lambda: "?")()
                        print(f"[warn] {spec.name} [{task_id}] failed: {e}")


            # Score deterministic modes
            for r in results:
                t = next(t for t in tasks if t.id == r.task_id)
                if t.task_type == "mcq":
                    pred_idx = extract_mcq_letter(r.answer, len(t.choices or []))
                    r.correct = 1 if (pred_idx is not None and t.correct_idx is not None and pred_idx == t.correct_idx) else 0
                elif t.task_type == "short" and t.reference:
                    r.correct = 1 if exact_match(r.answer, t.reference) else 0
                    r.f1 = token_f1(r.answer, t.reference)

            # Judge phase (for gen; optional for short if rubric present)
            judge_targets = [r for r in results if (next(t for t in tasks if t.id == r.task_id).task_type == "gen"
                                                    or (next(t for t in tasks if t.id == r.task_id).task_type == "short"
                                                        and next(t for t in tasks if t.id == r.task_id).rubric))]
            judge_jobs = []
            for r in judge_targets:
                t = next(t for t in tasks if t.id == r.task_id)
                judge_jobs.append(self._judge_one(session, r.answer, t))

            judged = await asyncio.gather(*judge_jobs, return_exceptions=False)
            for (score, verdict, just), r in zip(judged, judge_targets):
                r.judge_score = score
                r.judge_verdict = verdict
                r.judge_just = just

        # Save per-model raw
        self._write_csv(self.cfg.out_dir / f"raw_{sanitize(dsname)}.csv", results)
        self._write_json(self.cfg.out_dir / f"raw_{sanitize(dsname)}.json", [dataclasses.asdict(r) for r in results])
        # Summary
        self._write_json(self.cfg.out_dir / f"summary_{sanitize(dsname)}.json", self._summarize(results, tasks))
        return results

    def _summarize(self, results: List[QAResult], tasks: List[Task]) -> Dict[str, Any]:
        mcq = [r for r in results if next(t for t in tasks if t.id == r.task_id).task_type == "mcq"]
        short = [r for r in results if next(t for t in tasks if t.id == r.task_id).task_type == "short"]
        gen = [r for r in results if next(t for t in tasks if t.id == r.task_id).task_type == "gen"]

        def safe_mean(xs: List[float]) -> Optional[float]:
            xs = [x for x in xs if x is not None]
            return round(sum(xs) / len(xs), 4) if xs else None

        summary = {
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
        return summary

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


def sanitize(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)


# ------------------------------- CLI ---------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Custom LLM Benchmark for local llama.cpp + llama-swap")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--endpoint", type=str, required=True)
    p.add_argument("--api-key", type=str, default=None)

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--models", nargs="+", metavar="MODEL",
                       help='One or more model names to send in the "model" field.')
    group.add_argument("--models-from-swap", action="store_true",
                       help="Load model names from a llama-swap YAML instead of --models.")

    p.add_argument("--judge-model", type=str, required=True)

    p.add_argument("--swap-cmd", type=str, default=None)

    # leave these optional (can be None)
    p.add_argument("--temp", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--stop", type=str, nargs="*", default=None)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--concurrency", type=int, default=4)

    # llama-swap config helpers
    p.add_argument("--swap-config", type=Path, default=Path("config.base.yaml"))
    p.add_argument("--use-aliases", action="store_true")
    p.add_argument("--include", nargs="*", default=None)
    p.add_argument("--exclude", nargs="*", default=None)

    # summary csv options
    p.add_argument("--summary-metric",
                   choices=["auto", "composite", "mcq_accuracy", "short_em", "short_f1", "gen_judge_score"],
                   default="auto")
    p.add_argument("--summary-round", type=int, default=4)
    p.add_argument("--summary-include-metric", action="store_true")

    p.add_argument("--http-retries", type=int, default=8)
    p.add_argument("--backoff-cap", type=float, default=16.0)
    p.add_argument("--health-wait", type=int, default=10)
    p.add_argument("--health-wait-heavy", type=int, default=60)
    p.add_argument("--debug", action="store_true", help="Print extra info on failures.")
    p.add_argument("--llama-guard", dest="llama_guard", action="store_true",
               help="Ensure at most one llama-server runs during the benchmark.")
    p.add_argument("--no-llama-guard", dest="llama_guard", action="store_false")
    p.set_defaults(llama_guard=True)  # default ON

    p.add_argument("--kill-llama-on-exit", action="store_true",
                help="Kill all llama-server processes on exit (Ctrl-C or error).")
    p.set_defaults(kill_llama_on_exit=True)  # default ON
    return p.parse_args()

async def run_all(bench: Benchmark, run_cfg: RunConfig, tasks: list[Task]) -> dict[str, dict]:
    summaries: dict[str, dict] = {}
    for spec in run_cfg.models:
        try:
            await bench.run_for_model(spec, tasks)
        except Exception as e:
            print(f"[warn] Model {spec.name} failed: {e}")

        # Try to read back the per-model summary file if it exists
        label = spec.display_name or spec.name
        summ_path = run_cfg.out_dir / f"summary_{sanitize(label)}.json"
        if summ_path.exists():
            try:
                with summ_path.open("r", encoding="utf-8") as f:
                    summaries[label] = json.load(f)
            except Exception as e:
                print(f"[warn] Could not read summary for {label}: {e}")
        else:
            print(f"[warn] No summary file for {label}; skipping.")
    return summaries

def main():
    args = parse_args()
    tasks = load_jsonl(args.data)
    ensure_dir(args.out_dir)

    # Build model list
    if args.models_from_swap:
        selected = load_models_from_swap_config(
            args.swap_config,
            include_aliases=args.use_aliases,
            include_patterns=args.include,
            exclude_patterns=args.exclude,
        )
    else:
        if not args.models:
            raise SystemExit("Provide --models or use --models-from-swap.")
        selected = list(args.models)

    models: List[ModelSpec] = [
        ModelSpec(name=m, display_name=None,
                pre_switch_cmd=(args.swap_cmd if args.swap_cmd else None))
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
    )
    run_cfg = RunConfig(
        data_path=args.data,
        out_dir=args.out_dir,
        models=models,
        judge_model=args.judge_model,
        client=client_cfg,
        judge_client=None,
        llama_guard=args.llama_guard,
        kill_llama_on_exit=args.kill_llama_on_exit,
    )

    # --- in main() ---
    bench = Benchmark(run_cfg)

    def _cleanup():
        if run_cfg.kill_llama_on_exit:
            print("[info] Cleaning up llama-server processes…")
            kill_all_llama_servers()

    # atexit + signals
    atexit.register(_cleanup)

    def _signal_handler(signum, frame):
        _cleanup()
        # exit immediately with non-zero for SIGINT/SIGTERM
        raise SystemExit(130 if signum == sig.SIGINT else 143)

    for s in (sig.SIGINT, sig.SIGTERM):
        try:
            signal_prev = sig.getsignal(s)
            sig.signal(s, _signal_handler)
        except Exception:
            pass

    try:
        all_summaries = asyncio.run(run_all(bench, run_cfg, tasks))
    finally:
        _cleanup()

    # Write combined summary
    with (run_cfg.out_dir / "summary_all.json").open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    # Optional: quick scores.csv (if you added the flags earlier)
    if hasattr(args, "summary_metric"):
        scores_path = run_cfg.out_dir / "scores.csv"
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

    print(f"[done] Wrote results to: {args.out_dir.resolve()}")

if __name__ == "__main__":
    main()
