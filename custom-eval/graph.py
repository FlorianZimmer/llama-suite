#!/usr/bin/env python3
"""
graph.py — Plot summary_all.json from the custom LLM bench.

Features
- Standalone CLI *or* importable helper `plot_summary(...)`.
- Metric selection: auto | composite | mcq_accuracy | short_em | short_f1 |
                    gen_judge_score | latency_s_avg |
                    tokens_prompt_avg | tokens_completion_avg | tokens_total_avg
- Dynamic figure height, readable labels, nice grid.

Requires: matplotlib (pure Matplotlib, no seaborn)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


PCT_METRICS = {"mcq_accuracy", "short_em", "short_f1"}   # [0..1] -> show as %
LOWER_IS_BETTER = {"latency_s_avg", "tokens_prompt_avg", "tokens_completion_avg", "tokens_total_avg"}


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _pick_score_row(summary: Dict[str, float], metric: str) -> Optional[float]:
    """
    Return a score for the requested metric.
    - 'auto': prefer gen_judge_score, then short_f1, mcq_accuracy, short_em.
    - 'composite': mean of available {mcq_accuracy, short_f1, gen_judge_score/10}.
    """
    if metric == "auto":
        for k in ("gen_judge_score", "short_f1", "mcq_accuracy", "short_em"):
            v = _safe_float(summary.get(k))
            if v is not None:
                return v

        # As a last resort, try tokens_total_avg (higher means more usage)
        v = _safe_float(summary.get("tokens_total_avg"))
        return v

    if metric == "composite":
        parts = []
        v = _safe_float(summary.get("mcq_accuracy"))
        if v is not None:
            parts.append(v)          # already 0..1
        v = _safe_float(summary.get("short_f1"))
        if v is not None:
            parts.append(v)          # already 0..1
        v = _safe_float(summary.get("gen_judge_score"))
        if v is not None:
            parts.append(v / 10.0)   # normalize 0..10 -> 0..1
        if parts:
            return sum(parts) / len(parts)
        return None

    # Specific metric
    return _safe_float(summary.get(metric))


def _format_value(metric: str, v: float) -> str:
    if metric in PCT_METRICS or metric == "composite":
        return f"{v*100:.1f}%"
    if metric == "gen_judge_score":
        return f"{v:.2f}"
    if metric.startswith("tokens_"):
        return f"{v:.0f}"
    if metric == "latency_s_avg":
        return f"{v:.2f}s"
    return f"{v:.4f}"


def plot_summary(
    summary_path: Path,
    out_path: Optional[Path] = None,
    metric: str = "auto",
    top: Optional[int] = None,
    ascending: Optional[bool] = None,
    title: Optional[str] = None,
    width: float = 10.0,
    min_height: float = 3.0,
    bar_height: float = 0.5,
) -> Path:
    """
    Render a horizontal bar chart for model scores in summary_all.json.

    Returns the output image path.
    """
    summary_path = Path(summary_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"summary file not found: {summary_path}")

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError("summary_all.json has no models")

    # Build (model, score) rows
    rows: List[Tuple[str, float]] = []
    for model, summ in data.items():
        if not isinstance(summ, dict):
            continue
        val = _pick_score_row(summ, metric)
        if val is None:
            continue
        rows.append((model, float(val)))

    if not rows:
        raise ValueError(f"No models had a value for metric '{metric}'")

    # Sorting defaults: descending unless lower-is-better
    if ascending is None:
        ascending = metric in LOWER_IS_BETTER

    rows.sort(key=lambda x: x[1], reverse=not ascending)

    if top is not None and top > 0:
        rows = rows[:top]

    labels = [m for m, _ in rows]
    values = [v for _, v in rows]

    # Figure size: dynamic height based on number of models
    height = max(min_height, bar_height * len(rows) + 1.5)
    plt.close("all")
    fig, ax = plt.subplots(figsize=(width, height), layout="constrained")

    # Draw bars
    ax.barh(labels, values)

    # Put the best at the top for readability when descending
    if not ascending:
        ax.invert_yaxis()

    # Labels on bars
    for i, v in enumerate(values):
        ax.text(
            v,
            i,
            "  " + _format_value(metric, v),
            va="center",
            ha="left",
            fontsize=9,
        )

    # Titles / axes
    pretty_metric = {
        "auto": "Auto-selected metric",
        "composite": "Composite (avg of mcq_acc, short_f1, judge/10)",
        "mcq_accuracy": "MCQ Accuracy",
        "short_em": "Short EM",
        "short_f1": "Short F1",
        "gen_judge_score": "Judge Score (0–10)",
        "latency_s_avg": "Latency (avg sec)",
        "tokens_prompt_avg": "Prompt Tokens (avg)",
        "tokens_completion_avg": "Completion Tokens (avg)",
        "tokens_total_avg": "Total Tokens (avg)",
    }.get(metric, metric)

    ax.set_xlabel(pretty_metric)
    ttl = title or f"Model comparison — {pretty_metric}"
    ax.set_title(ttl, pad=10, fontsize=12)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Make long labels readable
    ax.tick_params(axis="y", labelsize=9)

    # Output
    if out_path is None:
        out_path = summary_path.parent / f"plot_{metric}.png"

    fig.savefig(out_path, dpi=200)
    print(f"[plot] Wrote: {out_path.resolve()}")
    return out_path


def _cli():
    ap = argparse.ArgumentParser(description="Plot summary_all.json as a comparison chart.")
    ap.add_argument("--summary", type=Path, default=Path("summary_all.json"),
                    help="Path to summary_all.json (default: ./summary_all.json)")
    ap.add_argument("--out", type=Path, default=None, help="Path to save PNG (default: next to summary)")
    ap.add_argument("--metric", type=str, default="auto",
                    choices=[
                        "auto", "composite",
                        "mcq_accuracy", "short_em", "short_f1",
                        "gen_judge_score",
                        "latency_s_avg", "tokens_prompt_avg",
                        "tokens_completion_avg", "tokens_total_avg",
                    ])
    ap.add_argument("--top", type=int, default=None, help="Show only top N models")
    sgroup = ap.add_mutually_exclusive_group()
    sgroup.add_argument("--ascending", action="store_true", help="Sort ascending")
    sgroup.add_argument("--descending", action="store_true", help="Sort descending")
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--width", type=float, default=10.0)
    ap.add_argument("--bar-height", type=float, default=0.5)
    ap.add_argument("--min-height", type=float, default=3.0)
    ap.add_argument("--open", action="store_true", help="Open the image after saving")
    args = ap.parse_args()

    asc = True if args.ascending else False if args.descending else None
    out = plot_summary(
        summary_path=args.summary,
        out_path=args.out,
        metric=args.metric,
        top=args.top,
        ascending=asc,
        title=args.title,
        width=args.width,
        bar_height=args.bar_height,
        min_height=args.min_height,
    )

    if args.open:
        try:
            if sys.platform.startswith("win"):
                os.startfile(out)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                os.system(f'open "{out}"')
            else:
                os.system(f'xdg-open "{out}"')
        except Exception as e:
            print(f"[warn] Could not open image: {e}")


if __name__ == "__main__":
    _cli()
