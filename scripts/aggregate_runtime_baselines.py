#!/usr/bin/env python3
"""Aggregate runtime baseline artifacts across model runs.

This reads the `manifest.json`, `events.jsonl`, and `summary.json` shape
written by `memory-moe-mvp/llama_runtime_probe.py`. It intentionally reports
observable runtime evidence only; semantic expert IDs require hook-level traces.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


JSONDict = dict[str, Any]


def read_json(path: Path) -> JSONDict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[JSONDict]:
    rows: list[JSONDict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def rounded(value: float | None, digits: int = 3) -> float | None:
    return round(value, digits) if value is not None else None


def summarize_run(run_dir: Path) -> JSONDict:
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    events_path = run_dir / "events.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary.json: {run_dir}")

    summary = read_json(summary_path)
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    events = read_jsonl(events_path)

    latencies: list[float] = []
    changed_metric_counts: list[int] = []
    prompt_repeats: dict[str, set[int]] = defaultdict(set)
    observability_available: Counter[str] = Counter()
    changed_metric_frequency: Counter[str] = Counter()

    for event in events:
        latency_total = (event.get("latency_ms") or {}).get("total")
        if isinstance(latency_total, (int, float)):
            latencies.append(float(latency_total))

        case = event.get("case") or {}
        probe_id = case.get("probe_id")
        repeat = case.get("repeat")
        if isinstance(probe_id, str) and isinstance(repeat, int):
            prompt_repeats[probe_id].add(repeat)

        observability = event.get("observability") or {}
        after = observability.get("after") or {}
        for path_name in ("metrics", "props", "slots"):
            if (after.get(path_name) or {}).get("available") is True:
                observability_available[path_name] += 1

        metrics_delta = observability.get("metrics_delta") or {}
        changed_metrics = metrics_delta.get("changed_metrics") or {}
        changed_metric_counts.append(int(metrics_delta.get("changed_metric_count") or 0))
        changed_metric_frequency.update(changed_metrics.keys())

    config = summary.get("config") or manifest.get("config") or {}
    totals = summary.get("totals") or {}
    breakdowns = summary.get("breakdowns") or {}
    repeat_counts = [len(repeats) for repeats in prompt_repeats.values()]

    return {
        "run_dir": str(run_dir),
        "run_id": summary.get("run_id") or manifest.get("run_id") or run_dir.name,
        "backend_family": config.get("backend_family"),
        "model": config.get("model"),
        "label": config.get("label"),
        "request_count": totals.get("request_count", len(events)),
        "failure_count": totals.get("failure_count"),
        "latency_observation_count": totals.get("latency_observation_count", len(latencies)),
        "latency_ms": {
            "mean": totals.get("mean_latency_ms"),
            "min": rounded(min(latencies) if latencies else None),
            "median": rounded(statistics.median(latencies) if latencies else None),
            "p90": rounded(quantile(latencies, 0.9)),
            "max": rounded(max(latencies) if latencies else None),
        },
        "families": breakdowns.get("by_family") or {},
        "finish_reasons": breakdowns.get("by_finish_reason") or {},
        "unique_prompt_count": len(prompt_repeats),
        "prompt_repeat_count_min": min(repeat_counts) if repeat_counts else 0,
        "prompt_repeat_count_max": max(repeat_counts) if repeat_counts else 0,
        "observability_available": dict(sorted(observability_available.items())),
        "changed_metric_count": {
            "min": min(changed_metric_counts) if changed_metric_counts else None,
            "median": rounded(statistics.median(changed_metric_counts) if changed_metric_counts else None),
            "max": max(changed_metric_counts) if changed_metric_counts else None,
        },
        "top_changed_metrics": changed_metric_frequency.most_common(12),
    }


def build_markdown(report: JSONDict) -> str:
    lines = [
        "# Runtime Baseline Aggregate",
        "",
        f"Run count: {len(report['runs'])}",
        "",
        "| Model | Backend | Requests | Failures | Mean ms | Median ms | Prompts | Repeats | Observability |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for run in report["runs"]:
        latency = run["latency_ms"]
        repeat_range = f"{run['prompt_repeat_count_min']}-{run['prompt_repeat_count_max']}"
        observability = ", ".join(run["observability_available"].keys()) or "none"
        lines.append(
            "| {model} | {backend} | {requests} | {failures} | {mean} | {median} | {prompts} | {repeats} | {observability} |".format(
                model=run.get("model") or "",
                backend=run.get("backend_family") or "",
                requests=run.get("request_count"),
                failures=run.get("failure_count"),
                mean=latency.get("mean"),
                median=latency.get("median"),
                prompts=run.get("unique_prompt_count"),
                repeats=repeat_range,
                observability=observability,
            )
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("- These are runtime/request artifacts, not semantic expert-route traces.")
    lines.append("- `props` and `slots` are llama.cpp-sidecar surfaces; vLLM usually exposes `/metrics` only.")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+", help="runtime-probe artifact directories")
    parser.add_argument("--output-json", type=Path, help="write aggregate JSON to this path")
    parser.add_argument("--output-md", type=Path, help="write aggregate Markdown to this path")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    runs = [summarize_run(path) for path in args.run_dirs]
    report = {
        "mode": "runtime_baseline_aggregate",
        "runs": runs,
        "notes": [
            "Runtime/request evidence only; semantic expert IDs require hook-level traces.",
            "Use this aggregate to compare launch cards and runtime observability before controller work.",
        ],
    }

    json_payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json_payload + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(build_markdown(report), encoding="utf-8")
    if not args.output_json and not args.output_md:
        print(json_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
