#!/usr/bin/env python3
"""Active llama.cpp runtime probe built on stock llama-server observability.

This probe is intentionally separate from the passive sidecar. It does not sit
in the request path. Instead it:

- submits requests directly to llama-server
- snapshots built-in observability endpoints such as /metrics and /slots
- optionally slices a configured server log file per request
- writes correlated run artifacts for later analysis

This is the appropriate Probe 2 surface for stock llama.cpp before moving to a
custom libllama harness or source patch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error, request

from moe_shared_contract import attach_shared_contract, build_shared_contract, contract_shell


JSONDict = dict[str, Any]


@dataclass(frozen=True)
class RuntimeProbeConfig:
    base_url: str
    output_dir: Path
    label: str
    request_timeout_seconds: float
    model: str
    backend_family: str = "llama_cpp"
    request_max_tokens: int | None = None
    metrics_path: str = "/metrics"
    slots_path: str = "/slots"
    props_path: str = "/props"
    log_file_path: Path | None = None
    max_log_bytes_per_event: int = 16_384


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: JSONDict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def round3(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 3)


def preview_text(text: str, limit: int = 160) -> str:
    squashed = " ".join(text.split())
    if len(squashed) <= limit:
        return squashed
    return f"{squashed[: limit - 3]}..."


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def parse_prometheus_line(line: str) -> JSONDict | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    try:
        lhs, rhs = stripped.rsplit(" ", 1)
        value = float(rhs)
    except ValueError:
        return None

    labels: dict[str, str] = {}
    metric_name = lhs
    if "{" in lhs and lhs.endswith("}"):
        metric_name, raw_labels = lhs.split("{", 1)
        raw_labels = raw_labels[:-1]
        if raw_labels:
            for pair in raw_labels.split(","):
                if "=" not in pair:
                    continue
                key, raw_value = pair.split("=", 1)
                labels[key.strip()] = raw_value.strip().strip('"')

    return {
        "name": metric_name,
        "labels": labels,
        "value": value,
    }


def parse_prometheus_metrics(text: str) -> JSONDict:
    entries: list[JSONDict] = []
    sums: dict[str, float] = {}
    for line in text.splitlines():
        parsed = parse_prometheus_line(line)
        if parsed is None:
            continue
        entries.append(parsed)
        sums[parsed["name"]] = sums.get(parsed["name"], 0.0) + parsed["value"]
    return {
        "metric_count": len(entries),
        "metrics_by_name_sum": {name: round3(value) for name, value in sorted(sums.items())},
        "entries": entries,
    }


def diff_metric_summaries(before: JSONDict | None, after: JSONDict | None) -> JSONDict:
    before_map = (before or {}).get("metrics_by_name_sum", {})
    after_map = (after or {}).get("metrics_by_name_sum", {})
    keys = sorted(set(before_map) | set(after_map))
    changed: dict[str, float] = {}
    for key in keys:
        delta = float(after_map.get(key, 0.0)) - float(before_map.get(key, 0.0))
        if abs(delta) > 1e-12:
            changed[key] = round3(delta)
    return {
        "changed_metric_count": len(changed),
        "changed_metrics": changed,
    }


def summarize_slots_payload(payload: Any) -> JSONDict:
    if isinstance(payload, list):
        states: dict[str, int] = {}
        slot_ids: list[int] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            state = str(item.get("state", "unknown"))
            states[state] = states.get(state, 0) + 1
            slot_id = item.get("id")
            if isinstance(slot_id, int):
                slot_ids.append(slot_id)
        return {
            "slot_count": len(payload),
            "states": states,
            "slot_ids": slot_ids,
        }

    if isinstance(payload, dict):
        return {
            "slot_count": len(payload.get("slots", [])) if isinstance(payload.get("slots"), list) else 0,
            "keys": sorted(payload.keys()),
        }

    return {"slot_count": 0}


def summarize_props_payload(payload: Any) -> JSONDict:
    if isinstance(payload, dict):
        return {
            "keys": sorted(payload.keys()),
            "key_count": len(payload),
        }
    return {"key_count": 0}


def http_get_json(base_url: str, path: str, timeout_seconds: float) -> JSONDict:
    url = f"{base_url.rstrip('/')}{path}"
    req = request.Request(url, method="GET")
    with request.urlopen(req, timeout=timeout_seconds) as response:
        body = response.read().decode("utf-8")
        content_type = response.headers.get("Content-Type", "")
        payload = json.loads(body) if body else None
        return {
            "available": True,
            "status_code": response.status,
            "content_type": content_type,
            "payload": payload,
        }


def http_get_text(base_url: str, path: str, timeout_seconds: float) -> JSONDict:
    url = f"{base_url.rstrip('/')}{path}"
    req = request.Request(url, method="GET")
    with request.urlopen(req, timeout=timeout_seconds) as response:
        body = response.read().decode("utf-8")
        return {
            "available": True,
            "status_code": response.status,
            "content_type": response.headers.get("Content-Type", ""),
            "text": body,
        }


def http_post_json(base_url: str, path: str, payload: JSONDict, timeout_seconds: float) -> JSONDict:
    url = f"{base_url.rstrip('/')}{path}"
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with request.urlopen(req, timeout=timeout_seconds) as response:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        raw = response.read().decode("utf-8")
        parsed = json.loads(raw) if raw else {}
        return {
            "status_code": response.status,
            "content_type": response.headers.get("Content-Type", ""),
            "payload": parsed,
            "elapsed_ms": round3(elapsed_ms),
        }


def fetch_optional_json_endpoint(base_url: str, path: str, timeout_seconds: float) -> JSONDict:
    try:
        response = http_get_json(base_url, path, timeout_seconds)
        payload = response["payload"]
        if path == "/slots":
            response["summary"] = summarize_slots_payload(payload)
        elif path == "/props":
            response["summary"] = summarize_props_payload(payload)
        else:
            response["summary"] = {}
        return response
    except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "error": str(exc),
        }


def fetch_optional_metrics(base_url: str, path: str, timeout_seconds: float) -> JSONDict:
    try:
        response = http_get_text(base_url, path, timeout_seconds)
        parsed = parse_prometheus_metrics(response["text"])
        return {
            "available": True,
            "status_code": response["status_code"],
            "content_type": response["content_type"],
            "summary": {
                "metric_count": parsed["metric_count"],
                "metrics_by_name_sum": parsed["metrics_by_name_sum"],
            },
        }
    except (error.HTTPError, error.URLError, TimeoutError) as exc:
        return {
            "available": False,
            "error": str(exc),
        }


def read_log_growth(log_file_path: Path | None, offset: int, max_bytes: int) -> JSONDict:
    if log_file_path is None or not log_file_path.exists():
        return {
            "available": False,
            "next_offset": offset,
        }

    with log_file_path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        end_offset = handle.tell()
        if end_offset < offset:
            offset = 0
        size = max(0, end_offset - offset)
        read_offset = max(offset, end_offset - max_bytes)
        handle.seek(read_offset)
        chunk = handle.read(end_offset - read_offset)

    text = chunk.decode("utf-8", errors="replace")
    return {
        "available": True,
        "next_offset": end_offset,
        "captured_bytes": len(chunk),
        "truncated": size > len(chunk),
        "text_sha256": sha256_text(text) if text else None,
        "text_preview": preview_text(text, limit=320) if text else None,
    }


def capture_observability_snapshot(config: RuntimeProbeConfig) -> JSONDict:
    metrics = fetch_optional_metrics(config.base_url, config.metrics_path, config.request_timeout_seconds)
    slots = fetch_optional_json_endpoint(config.base_url, config.slots_path, config.request_timeout_seconds)
    props = fetch_optional_json_endpoint(config.base_url, config.props_path, config.request_timeout_seconds)
    return {
        "timestamp": now_iso(),
        "metrics": metrics,
        "slots": slots,
        "props": props,
    }


def summarize_chat_response(payload: JSONDict) -> JSONDict:
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    timings = payload.get("timings") if isinstance(payload.get("timings"), dict) else {}
    response_text = ""
    message_content = ""
    reasoning_content = ""
    finish_reason = None
    if isinstance(payload.get("choices"), list) and payload["choices"]:
        choice = payload["choices"][0]
        if isinstance(choice, dict):
            finish_reason = choice.get("finish_reason")
            message = choice.get("message")
            if isinstance(message, dict):
                if isinstance(message.get("content"), str):
                    message_content = message["content"]
                    response_text = message_content
                if isinstance(message.get("reasoning_content"), str):
                    reasoning_content = message["reasoning_content"]
                elif isinstance(message.get("reasoning"), str):
                    reasoning_content = message["reasoning"]
            elif isinstance(choice.get("text"), str):
                response_text = choice["text"]
    return {
        "finish_reason": finish_reason,
        "response_chars": len(response_text),
        "response_preview": preview_text(response_text) if response_text else None,
        "message_content_chars": len(message_content),
        "message_content_preview": preview_text(message_content) if message_content else None,
        "reasoning_content_chars": len(reasoning_content),
        "reasoning_content_preview": preview_text(reasoning_content) if reasoning_content else None,
        "reasoning_content_present": bool(reasoning_content),
        "usage": usage,
        "timings": timings,
    }


def load_prompt_suite(path: Path) -> JSONDict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_request_cases_from_suite(
    suite_path: Path,
    model: str,
    max_prompts: int,
    repeats: int,
    request_max_tokens: int | None = None,
) -> list[JSONDict]:
    suite = load_prompt_suite(suite_path)
    cases: list[JSONDict] = []
    max_tokens = request_max_tokens or suite["default_request"]["max_tokens"]
    selected_prompt_count = 0
    for family in suite["families"]:
        for prompt in family["prompts"]:
            if max_prompts > 0 and selected_prompt_count >= max_prompts:
                return cases
            selected_prompt_count += 1
            for repeat in range(1, repeats + 1):
                cases.append(
                    {
                        "family_id": family["family_id"],
                        "probe_id": prompt["probe_id"],
                        "title": prompt["title"],
                        "repeat": repeat,
                        "body": {
                            "model": model,
                            "messages": prompt["messages"],
                            "temperature": suite["default_request"]["temperature"],
                            "top_p": suite["default_request"]["top_p"],
                            "max_tokens": max_tokens,
                            "stream": suite["default_request"]["stream"],
                        },
                    }
                )
    return cases


class RuntimeProbeAccumulator:
    def __init__(self, run_id: str, config: RuntimeProbeConfig) -> None:
        self.run_id = run_id
        self.config = config
        self.request_count = 0
        self.failure_count = 0
        self.by_family: dict[str, int] = {}
        self.by_finish_reason: dict[str, int] = {}
        self.changed_metric_counts: dict[str, int] = {}
        self.total_latency_ms = 0.0
        self.latency_observation_count = 0

    def ingest(self, event: JSONDict) -> None:
        self.request_count += 1
        error_text = event.get("error")
        if error_text is not None:
            self.failure_count += 1
        family_id = event["case"]["family_id"]
        self.by_family[family_id] = self.by_family.get(family_id, 0) + 1
        finish_reason = str(event.get("response", {}).get("summary", {}).get("finish_reason") or "none")
        if finish_reason == "none" and error_text is not None:
            finish_reason = "error"
        self.by_finish_reason[finish_reason] = self.by_finish_reason.get(finish_reason, 0) + 1
        latency_payload = event.get("latency_ms") or {}
        latency = latency_payload.get("total") if isinstance(latency_payload, dict) else None
        if isinstance(latency, (float, int)):
            self.total_latency_ms += float(latency)
            self.latency_observation_count += 1
        changed_metrics = event.get("observability", {}).get("metrics_delta", {}).get("changed_metrics", {})
        for metric_name in changed_metrics.keys():
            self.changed_metric_counts[metric_name] = self.changed_metric_counts.get(metric_name, 0) + 1

    def snapshot(self) -> JSONDict:
        return {
            "created_at": now_iso(),
            "run_id": self.run_id,
            "mode": "llama_runtime_probe",
            "shared_contract_template": contract_shell(
                probe_tier="internal_runtime",
                backend_family=self.config.backend_family,
                default_baseline_kind="observational",
            ),
            "config": {
                "base_url": self.config.base_url,
                "label": self.config.label,
                "model": self.config.model,
                "backend_family": self.config.backend_family,
                "log_file_path": str(self.config.log_file_path) if self.config.log_file_path else None,
                "request_max_tokens": self.config.request_max_tokens,
            },
            "totals": {
                "request_count": self.request_count,
                "failure_count": self.failure_count,
                "latency_observation_count": self.latency_observation_count,
                "mean_latency_ms": (
                    round3(safe_div(self.total_latency_ms, self.latency_observation_count))
                    if self.latency_observation_count
                    else None
                ),
            },
            "breakdowns": {
                "by_family": self.by_family,
                "by_finish_reason": self.by_finish_reason,
                "changed_metric_frequency": dict(sorted(self.changed_metric_counts.items())),
            },
        }


class LlamaRuntimeProbe:
    def __init__(self, config: RuntimeProbeConfig) -> None:
        self.config = config
        self.run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{config.label}"
        self.run_dir = config.output_dir / self.run_id
        ensure_dir(self.run_dir)
        self.manifest_path = self.run_dir / "manifest.json"
        self.events_path = self.run_dir / "events.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.accumulator = RuntimeProbeAccumulator(run_id=self.run_id, config=config)
        self._log_offset = 0
        if config.log_file_path and config.log_file_path.exists():
            self._log_offset = config.log_file_path.stat().st_size
        self._write_manifest()

    def _write_manifest(self) -> None:
        write_json(
            self.manifest_path,
            {
                "created_at": now_iso(),
                "run_id": self.run_id,
                "mode": "llama_runtime_probe",
                "shared_contract_template": contract_shell(
                    probe_tier="internal_runtime",
                    backend_family=self.config.backend_family,
                    default_baseline_kind="observational",
                ),
                "config": {
                    "base_url": self.config.base_url,
                    "label": self.config.label,
                    "model": self.config.model,
                    "backend_family": self.config.backend_family,
                    "metrics_path": self.config.metrics_path,
                    "slots_path": self.config.slots_path,
                    "props_path": self.config.props_path,
                    "log_file_path": str(self.config.log_file_path) if self.config.log_file_path else None,
                    "request_max_tokens": self.config.request_max_tokens,
                },
            },
        )

    def run_case(self, case: JSONDict) -> JSONDict:
        before = capture_observability_snapshot(self.config)
        response_payload = None
        response_summary = {}
        request_error = None
        latency_ms = None
        try:
            response = http_post_json(
                base_url=self.config.base_url,
                path="/v1/chat/completions",
                payload=case["body"],
                timeout_seconds=self.config.request_timeout_seconds,
            )
            response_payload = response["payload"]
            response_summary = summarize_chat_response(response_payload)
            latency_ms = {
                "total": response["elapsed_ms"],
                "server_prompt_ms": response_summary["timings"].get("prompt_ms"),
                "server_predicted_ms": response_summary["timings"].get("predicted_ms"),
            }
        except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            request_error = str(exc)

        after = capture_observability_snapshot(self.config)
        metrics_delta = diff_metric_summaries(
            before.get("metrics", {}).get("summary"),
            after.get("metrics", {}).get("summary"),
        )
        log_growth = read_log_growth(
            log_file_path=self.config.log_file_path,
            offset=self._log_offset,
            max_bytes=self.config.max_log_bytes_per_event,
        )
        self._log_offset = int(log_growth["next_offset"])

        event = attach_shared_contract(
            {
            "timestamp": now_iso(),
            "event_type": "runtime_probe_request",
            "case": {
                "family_id": case["family_id"],
                "probe_id": case["probe_id"],
                "title": case["title"],
                "repeat": case["repeat"],
            },
            "request": {
                "model": case["body"]["model"],
                "max_tokens": case["body"]["max_tokens"],
                "message_count": len(case["body"]["messages"]),
                "text_preview": preview_text(case["body"]["messages"][0]["content"]),
            },
            "response": {
                "summary": response_summary,
            },
            "latency_ms": latency_ms,
            "error": request_error,
            "observability": {
                "before": before,
                "after": after,
                "metrics_delta": metrics_delta,
                "log_growth": log_growth,
            },
            },
            build_shared_contract(
                probe_tier="internal_runtime",
                backend_family=self.config.backend_family,
                prompt_family=case["family_id"],
                baseline_kind="observational",
            ),
        )
        append_jsonl(self.events_path, event)
        self.accumulator.ingest(event)
        write_json(self.summary_path, self.accumulator.snapshot())
        return event


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--label", default="llama-runtime-probe")
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend-family", default="llama_cpp")
    parser.add_argument("--suite-path", type=Path)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--request-max-tokens",
        type=int,
        help="override the prompt suite max_tokens for this run",
    )
    parser.add_argument("--log-file-path", type=Path)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = RuntimeProbeConfig(
        base_url=args.base_url,
        output_dir=args.output_dir,
        label=args.label,
        request_timeout_seconds=args.timeout_seconds,
        model=args.model,
        backend_family=args.backend_family,
        request_max_tokens=args.request_max_tokens,
        log_file_path=args.log_file_path,
    )
    probe = LlamaRuntimeProbe(config=config)
    print(json.dumps({"run_dir": str(probe.run_dir), "run_id": probe.run_id}, indent=2))
    if args.suite_path:
        cases = build_request_cases_from_suite(
            suite_path=args.suite_path,
            model=args.model,
            max_prompts=args.max_prompts,
            repeats=args.repeats,
            request_max_tokens=args.request_max_tokens,
        )
        for case in cases:
            probe.run_case(case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
