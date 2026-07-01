#!/usr/bin/env python3
"""Run an approved Phase 3 dense/full-runtime output capture.

This is the first executable Phase 3 artifact writer. It sends the approved
prompt set to an already-running OpenAI-compatible local backend and writes the
exact dense-output summary artifact requested by the runtime-capture request.
It does not launch model servers, download models, authenticate, mutate expert
residency, or claim router/expert trace support.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

import build_phase3_output_summary


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUEST_PATH = ROOT / "memory-moe-mvp" / "phase3-real-evidence" / "pc_mixtral_phase3_real_evidence_bundle.runtime-capture-request.json"
SUPPORTED_SCHEMA_VERSION = "moe-phase3-dense-output-capture-runner-v1"
ARTIFACT_ID = "dense_output_summary_fill"
CAPTURE_KIND = "dense_output_summary_json"
RECEIPT_KIND = "embedded_output_capture_receipt"
JSONDict = dict[str, Any]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_repo_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def normalized_path_text(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def same_path(left: Path, right: Path) -> bool:
    return normalized_path_text(left) == normalized_path_text(right)


def load_json(path: Path) -> JSONDict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{display_path(path) or path} must be a JSON object")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def list_prompt_rows(prompt_set: JSONDict) -> list[JSONDict]:
    rows = prompt_set.get("prompts")
    if not isinstance(rows, list) or not rows:
        raise ValueError("prompt set must contain a non-empty prompts list")
    result: list[JSONDict] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"prompt set prompts[{index}] must be an object")
        prompt_id = row.get("prompt_id") or row.get("probe_id") or row.get("id")
        prompt_text = row.get("prompt")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"prompt set prompts[{index}] missing prompt_id")
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            raise ValueError(f"prompt set prompts[{index}] missing prompt text")
        result.append(row)
    return result


def http_get_available(base_url: str, path: str, timeout_seconds: float) -> bool:
    url = f"{base_url.rstrip('/')}{path}"
    req = request.Request(url, method="GET")
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            response.read(1024)
            return 200 <= response.status < 500
    except (error.HTTPError, error.URLError, TimeoutError, OSError):
        return False


def preflight_runtime(base_url: str, timeout_seconds: float) -> JSONDict:
    paths = ["/v1/models", "/models"]
    available = [path for path in paths if http_get_available(base_url, path, timeout_seconds)]
    return {
        "base_url": base_url.rstrip("/"),
        "readiness_paths": paths,
        "available_readiness_paths": available,
        "traffic_allowed": bool(available),
    }


def http_post_json(base_url: str, path: str, payload: JSONDict, timeout_seconds: float) -> JSONDict:
    url = f"{base_url.rstrip('/')}{path}"
    started = time.perf_counter()
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=timeout_seconds) as response:
        raw = response.read().decode("utf-8")
        parsed = json.loads(raw) if raw else {}
    return {
        "status_code": response.status,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "payload": parsed,
    }


def response_text(payload: JSONDict) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(choice.get("text"), str):
                return choice["text"]
    if isinstance(payload.get("response"), str):
        return payload["response"]
    return ""


def request_payload(prompt_row: JSONDict, *, model: str, max_tokens: int) -> JSONDict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": str(prompt_row["prompt"])}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "stream": False,
    }


def capture_outputs(prompt_set: JSONDict, *, base_url: str, model: str, max_tokens: int, timeout_seconds: float) -> list[JSONDict]:
    rows: list[JSONDict] = []
    for prompt_row in list_prompt_rows(prompt_set):
        sent = request_payload(prompt_row, model=model, max_tokens=max_tokens)
        try:
            response = http_post_json(base_url, "/v1/chat/completions", sent, timeout_seconds)
            text = response_text(response["payload"])
            error_text = None if text.strip() else "empty_response"
        except (error.HTTPError, error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            response = {"error": str(exc), "elapsed_ms": None, "payload": {}}
            text = ""
            error_text = str(exc)
        rows.append(
            {
                "prompt_id": prompt_row.get("prompt_id") or prompt_row.get("probe_id") or prompt_row.get("id"),
                "group_id": prompt_row.get("group_id"),
                "prompt_repeat": prompt_row.get("repeat"),
                "output": text,
                "error": error_text,
                "output_label": "dense",
                "ready": error_text is None,
                "latency_ms": response.get("elapsed_ms"),
                "notes": "captured by approved Phase 3 dense/full-runtime output writer",
            }
        )
    return rows


def build_artifact(
    *,
    request_payload: JSONDict,
    request_path: Path,
    prompt_set: JSONDict,
    prompt_set_path: Path,
    rows: list[JSONDict],
    backend_family: str,
    base_url: str,
    model: str,
    operator_notes: str | None,
) -> JSONDict:
    return {
        "schema_version": build_phase3_output_summary.SUPPORTED_SCHEMA_VERSION,
        "source_prompt_set_path": display_path(prompt_set_path),
        "model_id": prompt_set.get("model_id"),
        "runtime_model": model,
        "backend_family": prompt_set.get("backend_family") or backend_family,
        "prompt_family": prompt_set.get("prompt_family"),
        "output_label": "dense",
        "output_ready": all(row.get("ready") is True for row in rows),
        "capture_receipt": {
            "receipt_ready": all(row.get("ready") is True for row in rows),
            "source_request_path": display_path(request_path),
            "source_prompt_set_path": display_path(prompt_set_path),
            "runtime_capture_approved": True,
            "runtime_prompt_traffic_approved": True,
            "captured_at": now_iso(),
            "capture_host": socket.gethostname(),
            "runtime_backend": request_payload.get("backend_family") or backend_family,
            "model_id": prompt_set.get("model_id"),
            "prompt_family": prompt_set.get("prompt_family"),
            "output_label": "dense",
            "base_url": base_url.rstrip("/"),
            "runtime_model": model,
            "operator_notes": operator_notes or "approved Phase 3 dense/full-runtime output capture",
        },
        "outputs": rows,
        "safety_contract": safety_contract(),
    }


def safety_contract() -> list[str]:
    return [
        "dense-output capture sends prompt traffic only with explicit approval flags",
        "dense-output capture writes exactly the requested output summary path",
        "dense-output capture embeds the receipt in the output summary artifact",
        "dense-output capture does not launch model servers",
        "dense-output capture does not run Docker",
        "dense-output capture does not download models",
        "dense-output capture does not inspect private tokens",
        "dense-output capture does not mutate runtime residency",
        "dense-output capture does not claim live expert paging",
        "dense-output capture does not claim semantic router trace support",
    ]


def validate_paths(prompt_set_path: Path, artifact_output_path: Path, receipt_output_path: Path) -> list[str]:
    errors: list[str] = []
    if not prompt_set_path.exists():
        errors.append("prompt_set_path does not exist")
    if not same_path(artifact_output_path, receipt_output_path):
        errors.append("dense output summaries use an embedded receipt, so receipt_output_path must equal artifact_output_path")
    return errors


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime_capture_request_path", nargs="?", type=Path, default=DEFAULT_REQUEST_PATH)
    parser.add_argument("--prompt-set-path", required=True, type=Path)
    parser.add_argument("--artifact-output", required=True, type=Path)
    parser.add_argument("--receipt-output", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--backend-family", default="llama_cpp")
    parser.add_argument("--model", required=True)
    parser.add_argument("--request-max-tokens", type=int, default=220)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--preflight-timeout-seconds", type=float, default=3.0)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--approved-runtime-prompt-traffic", action="store_true")
    parser.add_argument("--approved-dense-output-capture", action="store_true")
    parser.add_argument("--operator-notes")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs without sending prompt traffic or writing artifacts")
    parser.add_argument("--json", action="store_true")
    return parser


def run_capture(args: argparse.Namespace) -> tuple[int, JSONDict]:
    request_path = resolve_repo_path(str(args.runtime_capture_request_path)) or args.runtime_capture_request_path
    prompt_set_path = resolve_repo_path(str(args.prompt_set_path)) or args.prompt_set_path
    artifact_output_path = resolve_repo_path(str(args.artifact_output)) or args.artifact_output
    receipt_output_path = resolve_repo_path(str(args.receipt_output)) or args.receipt_output
    errors = validate_paths(prompt_set_path, artifact_output_path, receipt_output_path)
    if not args.approved_runtime_prompt_traffic:
        errors.append("approved-runtime-prompt-traffic is required")
    if not args.approved_dense_output_capture:
        errors.append("approved-dense-output-capture is required")

    request_json: JSONDict = {}
    prompt_set: JSONDict = {}
    try:
        request_json = load_json(request_path)
        prompt_set = load_json(prompt_set_path)
        list_prompt_rows(prompt_set)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(str(exc))

    preflight = None
    if not args.skip_preflight:
        preflight = preflight_runtime(args.base_url, args.preflight_timeout_seconds)
        if preflight.get("traffic_allowed") is not True:
            errors.append("runtime readiness endpoint is not available")

    base_summary = {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "mode": "phase3_dense_output_capture",
        "artifact_id": ARTIFACT_ID,
        "capture_kind": CAPTURE_KIND,
        "receipt_kind": RECEIPT_KIND,
        "runtime_capture_request_path": display_path(request_path),
        "prompt_set_path": display_path(prompt_set_path),
        "artifact_output_path": display_path(artifact_output_path),
        "receipt_output_path": display_path(receipt_output_path),
        "base_url": args.base_url.rstrip("/"),
        "backend_family": args.backend_family,
        "model": args.model,
        "preflight": preflight,
        "dry_run": args.dry_run,
        "approved_runtime_prompt_traffic": args.approved_runtime_prompt_traffic,
        "approved_dense_output_capture": args.approved_dense_output_capture,
        "prompt_traffic_sent": False,
        "write_performed": False,
        "artifact_valid": False,
        "summary_ready": False,
        "errors": errors,
        "safety_contract": safety_contract(),
    }
    if errors or args.dry_run:
        base_summary["ok"] = not errors
        base_summary["ready_to_execute"] = not errors
        return (0 if not errors else 2), base_summary

    rows = capture_outputs(
        prompt_set,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.request_max_tokens,
        timeout_seconds=args.timeout_seconds,
    )
    artifact = build_artifact(
        request_payload=request_json,
        request_path=request_path,
        prompt_set=prompt_set,
        prompt_set_path=prompt_set_path,
        rows=rows,
        backend_family=args.backend_family,
        base_url=args.base_url,
        model=args.model,
        operator_notes=args.operator_notes,
    )
    builder_summary = build_phase3_output_summary.build_summary(
        artifact,
        prompt_set_path=prompt_set_path,
        output_path=artifact_output_path,
        expected_label="dense",
    )
    base_summary.update(
        {
            "prompt_traffic_sent": True,
            "artifact_valid": builder_summary.get("valid") is True,
            "summary_ready": builder_summary.get("summary_ready") is True,
            "output_present_count": builder_summary.get("output_present_count"),
            "missing_output_count": builder_summary.get("missing_output_count"),
            "builder_summary": builder_summary,
            "errors": builder_summary.get("errors", []),
        }
    )
    if builder_summary.get("valid") is True and builder_summary.get("summary_ready") is True:
        write_json(artifact_output_path, artifact)
        base_summary["write_performed"] = True
        base_summary["ok"] = True
        return 0, base_summary
    base_summary["ok"] = False
    return 2, base_summary


def print_human(summary: JSONDict) -> None:
    print("MoE Run Anyway Phase 3 dense output capture")
    print(f"OK: {summary.get('ok')}")
    print(f"Prompt traffic sent: {summary.get('prompt_traffic_sent')}")
    print(f"Write performed: {summary.get('write_performed')}")
    print(f"Output path: {summary.get('artifact_output_path')}")
    if summary.get("errors"):
        print("Errors:")
        for item in summary.get("errors", []):
            print(f"  - {item}")


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary = run_capture(args)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_human(summary)
    return status


if __name__ == "__main__":
    raise SystemExit(main())