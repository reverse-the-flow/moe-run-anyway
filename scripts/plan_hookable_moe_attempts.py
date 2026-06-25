#!/usr/bin/env python3
"""Validate and summarize hookability attempts across available MoE targets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX_PATH = ROOT / "memory-moe-mvp" / "data" / "hookable_moe_attempt_matrix.json"
SUPPORTED_SCHEMA_VERSION = "hookable-moe-attempt-matrix-v1"
HOOK_STATUSES = {
    "screened_non_moe",
    "engine_hook_candidate_uninstrumented",
    "blocked_missing_hook_runtime_dependencies",
    "semantic_trace_captured",
}
RUNTIME_ONLY_SURFACES = {
    "ollama_gguf",
    "llama_cpp_gguf",
    "vllm_openai_compatible",
}
TRANSFORMERS_SURFACES = {"hf_transformers_directory"}

JSONDict = dict[str, Any]


def load_matrix(path: Path) -> JSONDict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("matrix root must be a JSON object")
    return data


def validate_attempt(attempt: JSONDict) -> list[str]:
    errors: list[str] = []
    attempt_id = str(attempt.get("attempt_id") or "<missing>")
    for field in ("attempt_id", "host", "model", "runtime_surface", "hook_attempt_status"):
        if not isinstance(attempt.get(field), str) or not str(attempt.get(field)).strip():
            errors.append(f"{attempt_id}: {field} must be a non-empty string")

    status = attempt.get("hook_attempt_status")
    if status not in HOOK_STATUSES:
        errors.append(f"{attempt_id}: unsupported hook_attempt_status {status!r}")

    blockers = attempt.get("blockers")
    if status != "semantic_trace_captured" and not (isinstance(blockers, list) and blockers):
        errors.append(f"{attempt_id}: non-success attempts must record blockers")

    surface = attempt.get("runtime_surface")
    if status == "engine_hook_candidate_uninstrumented" and surface not in RUNTIME_ONLY_SURFACES:
        errors.append(f"{attempt_id}: engine-hook candidates must use a runtime-only surface")

    moe_metadata = attempt.get("moe_metadata")
    if not isinstance(moe_metadata, dict):
        errors.append(f"{attempt_id}: moe_metadata must be an object")
    elif status == "screened_non_moe" and moe_metadata.get("is_moe") is not False:
        errors.append(f"{attempt_id}: screened_non_moe attempts must set moe_metadata.is_moe=false")

    captured = attempt.get("real_model_semantic_trace_captured")
    if status != "semantic_trace_captured" and captured is not False:
        errors.append(f"{attempt_id}: non-success attempts must set real_model_semantic_trace_captured=false")
    if status == "semantic_trace_captured" and captured is not True:
        errors.append(f"{attempt_id}: semantic_trace_captured status requires real_model_semantic_trace_captured=true")

    evidence = attempt.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        errors.append(f"{attempt_id}: evidence must be a non-empty list")

    return errors


def validate_matrix(matrix: JSONDict) -> list[str]:
    errors: list[str] = []
    if matrix.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        errors.append(f"schema_version must be {SUPPORTED_SCHEMA_VERSION!r}")
    attempts = matrix.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        errors.append("attempts must be a non-empty list")
        return errors

    ids: set[str] = set()
    hosts: set[str] = set()
    for attempt in attempts:
        if not isinstance(attempt, dict):
            errors.append("each attempt must be an object")
            continue
        attempt_id = str(attempt.get("attempt_id") or "")
        if attempt_id in ids:
            errors.append(f"duplicate attempt_id {attempt_id!r}")
        ids.add(attempt_id)
        host = attempt.get("host")
        if isinstance(host, str):
            hosts.add(host)
        errors.extend(validate_attempt(attempt))

    for required_host in ("pc", "gx10"):
        if required_host not in hosts:
            errors.append(f"missing attempts for host {required_host!r}")

    summary = matrix.get("summary")
    if isinstance(summary, dict):
        success_count = sum(
            1
            for attempt in attempts
            if isinstance(attempt, dict)
            and attempt.get("hook_attempt_status") == "semantic_trace_captured"
        )
        if summary.get("real_model_semantic_hook_success_count") != success_count:
            errors.append("summary.real_model_semantic_hook_success_count does not match attempts")
    return errors


def summarize_matrix(matrix: JSONDict) -> JSONDict:
    attempts = [attempt for attempt in matrix.get("attempts", []) if isinstance(attempt, dict)]
    by_host = Counter(str(attempt.get("host") or "unknown") for attempt in attempts)
    by_status = Counter(str(attempt.get("hook_attempt_status") or "unknown") for attempt in attempts)
    by_surface = Counter(str(attempt.get("runtime_surface") or "unknown") for attempt in attempts)
    real_successes = [
        attempt["attempt_id"]
        for attempt in attempts
        if attempt.get("hook_attempt_status") == "semantic_trace_captured"
    ]
    hookable_blockers = [
        {
            "attempt_id": attempt.get("attempt_id"),
            "model": attempt.get("model"),
            "blockers": attempt.get("blockers", []),
        }
        for attempt in attempts
        if attempt.get("runtime_surface") in TRANSFORMERS_SURFACES
        and attempt.get("moe_metadata", {}).get("is_moe") is True
        and attempt.get("hook_attempt_status") != "semantic_trace_captured"
    ]
    return {
        "mode": "hookable_moe_attempt_matrix_summary",
        "attempt_count": len(attempts),
        "by_host": dict(sorted(by_host.items())),
        "by_status": dict(sorted(by_status.items())),
        "by_runtime_surface": dict(sorted(by_surface.items())),
        "real_model_semantic_hook_success_count": len(real_successes),
        "real_model_semantic_hook_successes": real_successes,
        "hookable_transformers_moe_blockers": hookable_blockers,
        "honesty_note": (
            "Runtime GGUF/Ollama/vLLM entries can be engine-hook candidates, but they are not semantic hook traces until the inference engine is instrumented."
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-path", type=Path, default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    matrix = load_matrix(args.matrix_path)
    errors = validate_matrix(matrix)
    summary = summarize_matrix(matrix)
    payload = {
        "valid": not errors,
        "errors": errors,
        "matrix_path": str(args.matrix_path),
        **summary,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Hookable MoE attempt matrix: {args.matrix_path}")
        print(f"Valid: {payload['valid']}")
        print(f"Attempts: {payload['attempt_count']}")
        print(f"By status: {payload['by_status']}")
        print(f"Real semantic hook successes: {payload['real_model_semantic_hook_success_count']}")
        if errors:
            print("Errors:")
            for error in errors:
                print(f"  - {error}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
