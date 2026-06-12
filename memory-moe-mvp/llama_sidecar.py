#!/usr/bin/env python3
"""Passive sidecar proxy for collecting llama.cpp request and telemetry data.

This script is intentionally non-invasive:

- it forwards requests to an upstream OpenAI-compatible or llama-server endpoint
- it records request and response summaries
- it samples before/after system telemetry when available
- it writes inspectable run artifacts for later analysis

It does not modify model execution, routing, or residency. Its job is to show
what information is already visible at the process boundary before we consider
runtime hooks or a fork.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import http.client
import json
import os
import shlex
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib import error, parse, request

from moe_shared_contract import attach_shared_contract, build_shared_contract, contract_shell


JSONDict = dict[str, Any]
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "proxy-connection",
    "host",
}
REDACTED_HEADERS = {"authorization", "proxy-authorization", "cookie", "set-cookie", "x-api-key"}


@dataclass(frozen=True)
class SidecarConfig:
    listen_host: str
    listen_port: int
    upstream_base_url: str
    request_timeout_seconds: float
    output_dir: Path
    label: str
    store_request_body: bool
    store_response_body: bool
    capture_gpu: bool
    gpu_query_command: list[str] | None
    capture_upstream_observability: bool
    metrics_path: str
    slots_path: str
    props_path: str


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: JSONDict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def round3(value: float) -> float:
    return round(value, 3)


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def split_command(command: str) -> list[str]:
    return shlex.split(command, posix=(os.name != "nt"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def preview_text(text: str, limit: int = 160) -> str:
    squashed = " ".join(text.split())
    if len(squashed) <= limit:
        return squashed
    return f"{squashed[: limit - 3]}..."


def default_gpu_query_command() -> list[str] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    return [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]


def get_system_memory_snapshot() -> JSONDict | None:
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        memory_status = MEMORYSTATUSEX()
        memory_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status)):
            return None
        total_mb = int(memory_status.ullTotalPhys / (1024 * 1024))
        available_mb = int(memory_status.ullAvailPhys / (1024 * 1024))
        return {
            "os": "windows",
            "total_mb": total_mb,
            "available_mb": available_mb,
            "used_mb": total_mb - available_mb,
            "load_percent": int(memory_status.dwMemoryLoad),
        }

    if os.path.exists("/proc/meminfo"):
        info: dict[str, int] = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                key, value = line.split(":", 1)
                info[key] = int(value.strip().split()[0])
        total_mb = int(info.get("MemTotal", 0) / 1024)
        available_mb = int(info.get("MemAvailable", 0) / 1024)
        used_mb = max(0, total_mb - available_mb)
        return {
            "os": "linux",
            "total_mb": total_mb,
            "available_mb": available_mb,
            "used_mb": used_mb,
            "load_percent": round3(safe_div(used_mb, total_mb) * 100.0),
        }

    return None


def get_gpu_snapshot(command: list[str] | None) -> JSONDict | None:
    if not command:
        return None
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=2.5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if completed.returncode != 0:
        return {
            "available": False,
            "returncode": completed.returncode,
            "stderr": preview_text(completed.stderr.strip()),
        }

    rows = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": int(float(parts[2])),
                    "memory_used_mb": int(float(parts[3])),
                    "utilization_gpu_percent": float(parts[4]),
                }
            )
        except ValueError:
            continue

    if not rows:
        return {
            "available": False,
            "stderr": "nvidia-smi returned no parseable rows",
        }

    total_mb = sum(item["memory_total_mb"] for item in rows)
    used_mb = sum(item["memory_used_mb"] for item in rows)
    return {
        "available": True,
        "gpu_count": len(rows),
        "total_memory_mb": total_mb,
        "used_memory_mb": used_mb,
        "utilization_gpu_percent_mean": round3(
            sum(item["utilization_gpu_percent"] for item in rows) / len(rows)
        ),
        "gpus": rows,
    }


def get_telemetry_snapshot(config: SidecarConfig) -> JSONDict:
    return {
        "timestamp": now_iso(),
        "system_memory": get_system_memory_snapshot(),
        "gpu": get_gpu_snapshot(config.gpu_query_command) if config.capture_gpu else None,
    }


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
            "key_count": len(payload),
            "keys": sorted(payload.keys()),
        }
    return {"key_count": 0}


def build_upstream_url(base_url: str, request_path: str) -> str:
    upstream_url = parse.urlsplit(base_url)
    upstream_path = build_upstream_path(upstream_url.path, request_path)
    return parse.urlunsplit((upstream_url.scheme, upstream_url.netloc, upstream_path, "", ""))


def fetch_optional_json_endpoint(config: SidecarConfig, path: str) -> JSONDict:
    started = time.perf_counter()
    url = build_upstream_url(config.upstream_base_url, path)
    try:
        with request.urlopen(url, timeout=config.request_timeout_seconds) as response:
            body = response.read()
            payload = maybe_parse_json_bytes(body)
            summary: JSONDict = {}
            if path == config.slots_path:
                summary = summarize_slots_payload(payload)
            elif path == config.props_path:
                summary = summarize_props_payload(payload)
            return {
                "available": True,
                "status_code": response.status,
                "elapsed_ms": round3((time.perf_counter() - started) * 1000.0),
                "content_type": response.headers.get("Content-Type"),
                "summary": summary,
            }
    except Exception as exc:
        return {
            "available": False,
            "elapsed_ms": round3((time.perf_counter() - started) * 1000.0),
            "error": preview_text(str(exc)),
        }


def fetch_optional_metrics(config: SidecarConfig) -> JSONDict:
    started = time.perf_counter()
    url = build_upstream_url(config.upstream_base_url, config.metrics_path)
    try:
        with request.urlopen(url, timeout=config.request_timeout_seconds) as response:
            text = response.read().decode("utf-8", errors="replace")
            parsed = parse_prometheus_metrics(text)
            return {
                "available": True,
                "status_code": response.status,
                "elapsed_ms": round3((time.perf_counter() - started) * 1000.0),
                "content_type": response.headers.get("Content-Type"),
                "summary": parsed,
            }
    except Exception as exc:
        return {
            "available": False,
            "elapsed_ms": round3((time.perf_counter() - started) * 1000.0),
            "error": preview_text(str(exc)),
        }


def capture_upstream_observability(config: SidecarConfig) -> JSONDict:
    return {
        "timestamp": now_iso(),
        "metrics": fetch_optional_metrics(config),
        "slots": fetch_optional_json_endpoint(config, config.slots_path),
        "props": fetch_optional_json_endpoint(config, config.props_path),
    }


def maybe_parse_json_bytes(body: bytes) -> Any | None:
    if not body:
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def summarize_text_inputs(payload: JSONDict) -> tuple[str, JSONDict]:
    texts: list[str] = []
    role_counts: dict[str, int] = {}
    if isinstance(payload.get("messages"), list):
        for message in payload["messages"]:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "unknown"))
            role_counts[role] = role_counts.get(role, 0) + 1
            content = message.get("content")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        maybe_text = item.get("text")
                        if isinstance(maybe_text, str):
                            texts.append(maybe_text)
    prompt = payload.get("prompt")
    if isinstance(prompt, str):
        texts.append(prompt)
    input_value = payload.get("input")
    if isinstance(input_value, str):
        texts.append(input_value)
    elif isinstance(input_value, list):
        texts.extend(str(item) for item in input_value if isinstance(item, str))

    joined = "\n".join(texts)
    text_meta = {
        "message_count": len(payload["messages"]) if isinstance(payload.get("messages"), list) else 0,
        "role_counts": role_counts,
        "text_chars": len(joined),
        "text_lines": joined.count("\n") + (1 if joined else 0),
        "text_sha256": sha256_text(joined) if joined else None,
        "text_preview": preview_text(joined) if joined else None,
    }
    return joined, text_meta


def classify_endpoint(path: str) -> str:
    lowered = path.lower()
    if "/chat/completions" in lowered:
        return "chat_completions"
    if lowered.endswith("/completion") or "/completion" in lowered:
        return "completion"
    if "/embeddings" in lowered:
        return "embeddings"
    if "/slots" in lowered:
        return "slots"
    if "/models" in lowered:
        return "models"
    return "other"


def summarize_request_payload(path: str, body: bytes, store_body: bool) -> JSONDict:
    payload = maybe_parse_json_bytes(body)
    if not isinstance(payload, dict):
        return {
            "path": path,
            "endpoint_kind": classify_endpoint(path),
            "content_length_bytes": len(body),
            "body_json": False,
            "body_sha256": sha256_text(body.decode("utf-8", errors="replace")) if body else None,
            "raw_body": body.decode("utf-8", errors="replace") if store_body and body else None,
        }

    _, text_meta = summarize_text_inputs(payload)
    model = payload.get("model")
    slot_id = payload.get("id_slot")
    if slot_id is None:
        slot_id = payload.get("slot_id")
    n_predict = payload.get("n_predict")
    if n_predict is None:
        n_predict = payload.get("max_tokens")
    return {
        "path": path,
        "endpoint_kind": classify_endpoint(path),
        "body_json": True,
        "content_length_bytes": len(body),
        "json_keys": sorted(payload.keys()),
        "model": str(model) if model is not None else None,
        "stream": bool(payload.get("stream", False)),
        "cache_prompt": payload.get("cache_prompt"),
        "slot_id": slot_id,
        "max_tokens": n_predict,
        "temperature": payload.get("temperature"),
        "seed": payload.get("seed"),
        "text_meta": text_meta,
        "raw_body": payload if store_body else None,
    }


def summarize_response_body(
    body: bytes,
    headers: JSONDict,
    streaming: bool,
    store_body: bool,
) -> JSONDict:
    content_type = str(headers.get("Content-Type", ""))
    payload = maybe_parse_json_bytes(body)
    response: JSONDict = {
        "content_type": content_type,
        "streaming": streaming,
        "body_bytes": len(body),
        "body_sha256": sha256_text(body.decode("utf-8", errors="replace")) if body else None,
    }
    if streaming or not isinstance(payload, dict):
        response["raw_body"] = body.decode("utf-8", errors="replace") if store_body and body else None
        return response

    choices = payload.get("choices")
    first_choice = choices[0] if isinstance(choices, list) and choices else {}
    message = first_choice.get("message") if isinstance(first_choice, dict) else {}
    content = ""
    if isinstance(message, dict):
        maybe_content = message.get("content")
        if isinstance(maybe_content, str):
            content = maybe_content
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    timings = payload.get("timings") if isinstance(payload.get("timings"), dict) else {}
    response.update(
        {
            "finish_reason": first_choice.get("finish_reason") if isinstance(first_choice, dict) else None,
            "response_chars": len(content),
            "response_preview": preview_text(content) if content else None,
            "usage": usage,
            "timings": timings,
            "raw_body": payload if store_body else None,
        }
    )
    return response


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    redacted: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in REDACTED_HEADERS:
            redacted[key] = "<redacted>"
        else:
            redacted[key] = value
    return redacted


class RunAccumulator:
    def __init__(self, run_id: str, config: SidecarConfig):
        self.run_id = run_id
        self.config = config
        self.started_at = now_iso()
        self.request_count = 0
        self.failure_count = 0
        self.streaming_request_count = 0
        self.by_status: dict[str, int] = {}
        self.by_path: dict[str, int] = {}
        self.by_endpoint_kind: dict[str, int] = {}
        self.by_model: dict[str, int] = {}
        self.total_duration_ms = 0.0
        self.total_ttfb_ms = 0.0
        self.max_duration_ms = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def ingest(self, event: JSONDict) -> None:
        self.request_count += 1
        path = str(event["request"]["path"])
        endpoint_kind = str(event["request"]["summary"].get("endpoint_kind", "other"))
        status_code = str(event["response"]["status_code"])
        model = event["request"]["summary"].get("model")
        if event["request"]["summary"].get("stream"):
            self.streaming_request_count += 1
        if event.get("error"):
            self.failure_count += 1

        self.by_path[path] = self.by_path.get(path, 0) + 1
        self.by_endpoint_kind[endpoint_kind] = self.by_endpoint_kind.get(endpoint_kind, 0) + 1
        self.by_status[status_code] = self.by_status.get(status_code, 0) + 1
        if model is not None:
            model_key = str(model)
            self.by_model[model_key] = self.by_model.get(model_key, 0) + 1

        total_ms = float(event["latency_ms"]["total"])
        ttfb_ms = float(event["latency_ms"]["ttfb"])
        self.total_duration_ms += total_ms
        self.total_ttfb_ms += ttfb_ms
        self.max_duration_ms = max(self.max_duration_ms, total_ms)

        usage = event["response"]["summary"].get("usage")
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens") or usage.get("prompt_token_count") or 0
            completion_tokens = usage.get("completion_tokens") or usage.get("completion_token_count") or 0
            if isinstance(prompt_tokens, int):
                self.total_prompt_tokens += prompt_tokens
            if isinstance(completion_tokens, int):
                self.total_completion_tokens += completion_tokens

    def snapshot(self) -> JSONDict:
        return {
            "created_at": self.started_at,
            "updated_at": now_iso(),
            "run_id": self.run_id,
            "mode": "passive_llama_sidecar",
            "shared_contract_template": contract_shell(
                probe_tier="passive_external",
                backend_family="llama_cpp",
                default_baseline_kind="observational",
            ),
            "config": {
                "listen_host": self.config.listen_host,
                "listen_port": self.config.listen_port,
                "upstream_base_url": self.config.upstream_base_url,
                "request_timeout_seconds": self.config.request_timeout_seconds,
                "label": self.config.label,
            },
            "totals": {
                "request_count": self.request_count,
                "failure_count": self.failure_count,
                "streaming_request_count": self.streaming_request_count,
                "mean_total_latency_ms": round3(safe_div(self.total_duration_ms, self.request_count)),
                "mean_ttfb_ms": round3(safe_div(self.total_ttfb_ms, self.request_count)),
                "max_total_latency_ms": round3(self.max_duration_ms),
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
            },
            "breakdowns": {
                "by_status": self.by_status,
                "by_path": self.by_path,
                "by_endpoint_kind": self.by_endpoint_kind,
                "by_model": self.by_model,
            },
        }


def probe_upstream(url: str, timeout_seconds: float) -> JSONDict:
    started = time.perf_counter()
    try:
        with request.urlopen(url, timeout=timeout_seconds) as response:
            body = response.read()
            elapsed_ms = round3((time.perf_counter() - started) * 1000.0)
            return {
                "ok": True,
                "status_code": response.status,
                "elapsed_ms": elapsed_ms,
                "content_type": response.headers.get("Content-Type"),
                "body_json": maybe_parse_json_bytes(body),
            }
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "status_code": exc.code,
            "elapsed_ms": round3((time.perf_counter() - started) * 1000.0),
            "error": preview_text(body),
        }
    except Exception as exc:
        return {
            "ok": False,
            "elapsed_ms": round3((time.perf_counter() - started) * 1000.0),
            "error": preview_text(str(exc)),
        }


class SidecarContext:
    def __init__(self, config: SidecarConfig):
        self.config = config
        self.run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{config.label}"
        self.run_dir = config.output_dir / self.run_id
        self.events_path = self.run_dir / "events.jsonl"
        self.manifest_path = self.run_dir / "manifest.json"
        self.summary_path = self.run_dir / "summary.json"
        self.lock = threading.Lock()
        self.request_counter = 0
        self.accumulator = RunAccumulator(self.run_id, config)

        ensure_dir(self.run_dir)
        manifest = {
            "created_at": now_iso(),
            "run_id": self.run_id,
            "mode": "passive_llama_sidecar",
            "shared_contract_template": contract_shell(
                probe_tier="passive_external",
                backend_family="llama_cpp",
                default_baseline_kind="observational",
            ),
            "purpose": (
                "Collect request boundary, timing, and coarse telemetry information "
                "from a live llama.cpp-compatible backend before any runtime fork."
            ),
            "config": {
                "listen_host": config.listen_host,
                "listen_port": config.listen_port,
                "upstream_base_url": config.upstream_base_url,
                "request_timeout_seconds": config.request_timeout_seconds,
                "label": config.label,
                "store_request_body": config.store_request_body,
                "store_response_body": config.store_response_body,
                "capture_gpu": config.capture_gpu,
                "gpu_query_command": config.gpu_query_command,
                "capture_upstream_observability": config.capture_upstream_observability,
                "metrics_path": config.metrics_path,
                "slots_path": config.slots_path,
                "props_path": config.props_path,
            },
            "initial_telemetry": get_telemetry_snapshot(config),
            "upstream_probes": {
                "root": probe_upstream(build_upstream_url(config.upstream_base_url, "/"), 5.0),
                "slots": probe_upstream(build_upstream_url(config.upstream_base_url, config.slots_path), 5.0),
                "health": probe_upstream(build_upstream_url(config.upstream_base_url, "/health"), 5.0),
                "metrics": probe_upstream(build_upstream_url(config.upstream_base_url, config.metrics_path), 5.0),
                "props": probe_upstream(build_upstream_url(config.upstream_base_url, config.props_path), 5.0),
            },
            "limitations": [
                "This sidecar only observes request boundary information.",
                "It does not expose router outputs or expert-level activation details.",
                "Streaming responses are proxied, but the sidecar does not reconstruct full structured JSON summaries from SSE chunks.",
            ],
        }
        write_json(self.manifest_path, manifest)
        write_json(self.summary_path, self.accumulator.snapshot())

    def next_request_id(self) -> str:
        with self.lock:
            self.request_counter += 1
            return f"req-{self.request_counter:06d}"

    def record_event(self, event: JSONDict) -> None:
        with self.lock:
            append_jsonl(self.events_path, event)
            self.accumulator.ingest(event)
            write_json(self.summary_path, self.accumulator.snapshot())


def build_upstream_path(base_path: str, request_path: str) -> str:
    prefix = base_path.rstrip("/")
    if not prefix:
        return request_path
    if request_path.startswith("/"):
        return prefix + request_path
    return prefix + "/" + request_path


def strip_hop_by_hop_headers(headers: list[tuple[str, str]]) -> list[tuple[str, str]]:
    filtered = []
    for key, value in headers:
        if key.lower() in HOP_BY_HOP_HEADERS:
            continue
        filtered.append((key, value))
    return filtered


class SidecarHandler(BaseHTTPRequestHandler):
    context: SidecarContext | None = None
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        self._proxy_request()

    def do_POST(self) -> None:  # noqa: N802
        self._proxy_request()

    def do_DELETE(self) -> None:  # noqa: N802
        self._proxy_request()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._proxy_request()

    def do_PUT(self) -> None:  # noqa: N802
        self._proxy_request()

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _proxy_request(self) -> None:
        assert self.context is not None
        config = self.context.config
        request_id = self.context.next_request_id()
        started = time.perf_counter()
        telemetry_before = get_telemetry_snapshot(config)
        upstream_before = capture_upstream_observability(config) if config.capture_upstream_observability else None

        body = self._read_body()
        request_summary = summarize_request_payload(self.path, body, config.store_request_body)
        request_headers = self._collect_forward_headers()
        request_meta = {
            "id": request_id,
            "timestamp": now_iso(),
            "method": self.command,
            "path": self.path,
            "client_address": self.client_address[0] if self.client_address else None,
            "headers": redact_headers(request_headers),
            "summary": request_summary,
        }

        upstream_url = parse.urlsplit(config.upstream_base_url)
        connection_cls = http.client.HTTPSConnection if upstream_url.scheme == "https" else http.client.HTTPConnection
        upstream_path = build_upstream_path(upstream_url.path, self.path)
        response_status = 502
        response_reason = "Bad Gateway"
        response_headers: JSONDict = {}
        response_summary: JSONDict = {}
        error_text: str | None = None
        ttfb_ms = 0.0
        total_ms = 0.0

        try:
            connection = connection_cls(
                upstream_url.hostname,
                upstream_url.port,
                timeout=config.request_timeout_seconds,
            )
            try:
                connection.request(
                    self.command,
                    upstream_path,
                    body=body,
                    headers=request_headers,
                )
                upstream_response = connection.getresponse()
                ttfb_ms = round3((time.perf_counter() - started) * 1000.0)
                response_status = upstream_response.status
                response_reason = upstream_response.reason
                response_headers = {key: value for key, value in upstream_response.getheaders()}
                content_type = str(response_headers.get("Content-Type", ""))
                streaming = bool(request_summary.get("stream")) or "text/event-stream" in content_type.lower()

                self.send_response(response_status, response_reason)
                for key, value in strip_hop_by_hop_headers(upstream_response.getheaders()):
                    if streaming and key.lower() in {"content-length", "transfer-encoding"}:
                        continue
                    self.send_header(key, value)
                if streaming:
                    self.send_header("Connection", "close")
                self.end_headers()

                if streaming:
                    total_bytes = 0
                    while True:
                        chunk = upstream_response.read(8192)
                        if not chunk:
                            break
                        total_bytes += len(chunk)
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    total_ms = round3((time.perf_counter() - started) * 1000.0)
                    response_summary = summarize_response_body(
                        body=b"",
                        headers=response_headers,
                        streaming=True,
                        store_body=False,
                    )
                    response_summary["body_bytes"] = total_bytes
                else:
                    upstream_body = upstream_response.read()
                    total_ms = round3((time.perf_counter() - started) * 1000.0)
                    if self.command != "HEAD":
                        self.wfile.write(upstream_body)
                    response_summary = summarize_response_body(
                        body=upstream_body,
                        headers=response_headers,
                        streaming=False,
                        store_body=config.store_response_body,
                    )
            finally:
                connection.close()
        except Exception as exc:
            error_text = str(exc)
            ttfb_ms = round3((time.perf_counter() - started) * 1000.0)
            total_ms = ttfb_ms
            fallback = {
                "error": "upstream_request_failed",
                "detail": error_text,
                "request_id": request_id,
            }
            encoded = json.dumps(fallback).encode("utf-8")
            self.send_response(502, "Bad Gateway")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            response_headers = {"Content-Type": "application/json"}
            response_summary = summarize_response_body(
                body=encoded,
                headers=response_headers,
                streaming=False,
                store_body=True,
            )

        telemetry_after = get_telemetry_snapshot(config)
        upstream_after = capture_upstream_observability(config) if config.capture_upstream_observability else None
        event = attach_shared_contract(
            {
            "timestamp": now_iso(),
            "event_type": "proxy_request",
            "request": request_meta,
            "response": {
                "status_code": response_status,
                "reason": response_reason,
                "headers": response_headers,
                "summary": response_summary,
            },
            "latency_ms": {
                "ttfb": ttfb_ms,
                "total": total_ms,
            },
            "telemetry": {
                "before": telemetry_before,
                "after": telemetry_after,
            },
            "upstream_observability": {
                "before": upstream_before,
                "after": upstream_after,
                "metrics_delta": diff_metric_summaries(
                    upstream_before.get("metrics", {}).get("summary") if upstream_before else None,
                    upstream_after.get("metrics", {}).get("summary") if upstream_after else None,
                ),
            }
            if config.capture_upstream_observability
            else None,
            "error": error_text,
            },
            build_shared_contract(
                probe_tier="passive_external",
                backend_family="llama_cpp",
                baseline_kind="observational",
            ),
        )
        self.context.record_event(event)

    def _read_body(self) -> bytes:
        length_text = self.headers.get("Content-Length")
        if not length_text:
            return b""
        try:
            length = int(length_text)
        except ValueError:
            return b""
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def _collect_forward_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        for key, value in self.headers.items():
            if key.lower() in HOP_BY_HOP_HEADERS:
                continue
            headers[key] = value
        upstream = parse.urlsplit(self.context.config.upstream_base_url)
        headers["Host"] = upstream.netloc
        return headers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Passive sidecar proxy for collecting llama.cpp request data.")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=8091)
    parser.add_argument("--upstream-base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--request-timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "sidecar-runs",
    )
    parser.add_argument("--label", default="llama-sidecar")
    parser.add_argument("--store-request-body", action="store_true")
    parser.add_argument("--store-response-body", action="store_true")
    parser.add_argument(
        "--capture-gpu",
        action="store_true",
        help="Capture GPU memory snapshots with nvidia-smi when available.",
    )
    parser.add_argument(
        "--gpu-query-command",
        help="Optional command string to use instead of the default nvidia-smi query.",
    )
    parser.add_argument(
        "--capture-upstream-observability",
        action="store_true",
        help="Capture built-in llama-server observability endpoints such as /metrics, /slots, and /props before and after proxied requests.",
    )
    parser.add_argument("--metrics-path", default="/metrics")
    parser.add_argument("--slots-path", default="/slots")
    parser.add_argument("--props-path", default="/props")
    return parser


def build_config(args: argparse.Namespace) -> SidecarConfig:
    if args.gpu_query_command:
        gpu_query_command = split_command(args.gpu_query_command)
    else:
        gpu_query_command = default_gpu_query_command()

    return SidecarConfig(
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        upstream_base_url=args.upstream_base_url,
        request_timeout_seconds=args.request_timeout_seconds,
        output_dir=args.output_dir,
        label=args.label,
        store_request_body=args.store_request_body,
        store_response_body=args.store_response_body,
        capture_gpu=bool(args.capture_gpu),
        gpu_query_command=gpu_query_command,
        capture_upstream_observability=bool(args.capture_upstream_observability),
        metrics_path=args.metrics_path,
        slots_path=args.slots_path,
        props_path=args.props_path,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = build_config(args)
    context = SidecarContext(config)
    SidecarHandler.context = context

    server = ThreadingHTTPServer((config.listen_host, config.listen_port), SidecarHandler)
    print(f"Run folder: {context.run_dir}")
    print(
        f"Listening on http://{config.listen_host}:{config.listen_port} "
        f"-> {config.upstream_base_url}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        write_json(context.summary_path, context.accumulator.snapshot())


if __name__ == "__main__":
    main()
