#!/usr/bin/env python3
"""Validate saved Phase 3 runtime-capture request artifacts.

This planner audits generated runtime-capture request JSON files before an
operator uses them for approved runtime work. It reloads each saved request,
rebuilds the expected request from current local metadata, and reports drift in
paths, approvals, embedded summaries, artifact statuses, readiness flags,
source metadata, validator commands, or required artifact ids.
It does not launch runtimes, run Docker, call endpoints, inspect secrets, mutate
residency, or send prompt traffic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import build_phase3_runtime_capture_request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUEST_PATH = (
    ROOT
    / "memory-moe-mvp"
    / "phase3-real-evidence"
    / "pc_mixtral_phase3_real_evidence_bundle.runtime-capture-request.json"
)
DEFAULT_ROOT = ROOT / "memory-moe-mvp" / "phase3-real-evidence"
REQUEST_GLOB = "*.runtime-capture-request.json"
SUPPORTED_SCHEMA_VERSION = "moe-phase3-runtime-capture-request-audit-v1"

JSONDict = dict[str, Any]

RUNTIME_APPROVAL_KEYS = (
    "runtime_prompt_traffic_approved",
    "router_trace_capture_approved",
    "managed_output_capture_approved",
    "dense_output_capture_approved",
)
APPROVAL_FLAG_BY_KEY = {
    "runtime_prompt_traffic_approved": "--runtime-prompt-traffic-approved",
    "router_trace_capture_approved": "--router-trace-capture-approved",
    "managed_output_capture_approved": "--managed-output-capture-approved",
    "dense_output_capture_approved": "--dense-output-capture-approved",
}
APPROVAL_KEYS_BY_ARTIFACT_ID = {
    "candidate_router_trace": ("runtime_prompt_traffic_approved", "router_trace_capture_approved"),
    "managed_output_summary_fill": ("runtime_prompt_traffic_approved", "managed_output_capture_approved"),
    "dense_output_summary_fill": ("runtime_prompt_traffic_approved", "dense_output_capture_approved"),
}
COMPLETE_ARTIFACT_STATUSES = {"already_satisfied"}

def display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_repo_path(value: Any) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_request(path: Path) -> JSONDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{display_path(path) or path} must be a JSON object")
    return payload


def artifact_by_id(items: Any) -> dict[str, JSONDict]:
    result: dict[str, JSONDict] = {}
    if not isinstance(items, list):
        return result
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        if isinstance(item_id, str):
            result[item_id] = item
    return result


def artifact_path(summary: JSONDict, collection: str, artifact_id: str) -> Path | None:
    item = artifact_by_id(summary.get(collection)).get(artifact_id)
    if item is None:
        return None
    return resolve_repo_path(item.get("path"))


def artifact_source_path(summary: JSONDict, collection: str, artifact_id: str, source_key: str) -> Path | None:
    item = artifact_by_id(summary.get(collection)).get(artifact_id)
    if item is None:
        return None
    source = item.get("source")
    if not isinstance(source, dict):
        return None
    return resolve_repo_path(source.get(source_key))


def approvals(summary: JSONDict) -> JSONDict:
    value = summary.get("approvals")
    return value if isinstance(value, dict) else {}


def boolean_approval(summary: JSONDict, key: str) -> bool:
    return approvals(summary).get(key) is True


def rebuild_expected(saved: JSONDict) -> JSONDict:
    bundle_path = resolve_repo_path(saved.get("bundle_path"))
    if bundle_path is None:
        raise ValueError("saved request is missing bundle_path")
    return build_phase3_runtime_capture_request.build_request(
        bundle_path,
        prompt_set_path=resolve_repo_path(saved.get("prompt_set", {}).get("path"))
        if isinstance(saved.get("prompt_set"), dict)
        else None,
        candidate_trace_path=artifact_path(saved, "requested_artifacts", "candidate_router_trace"),
        candidate_trace_receipt_path=artifact_source_path(
            saved,
            "requested_artifacts",
            "candidate_router_trace",
            "capture_receipt_path",
        ),
        managed_output_path=artifact_path(saved, "requested_artifacts", "managed_output_summary_fill"),
        dense_output_path=artifact_path(saved, "requested_artifacts", "dense_output_summary_fill"),
        live_proof_template_path=artifact_path(saved, "future_artifacts", "live_capability_proof_fill"),
        router_trace_capture_approved=boolean_approval(saved, "router_trace_capture_approved"),
        managed_output_capture_approved=boolean_approval(saved, "managed_output_capture_approved"),
        dense_output_capture_approved=boolean_approval(saved, "dense_output_capture_approved"),
        runtime_prompt_traffic_approved=boolean_approval(saved, "runtime_prompt_traffic_approved"),
    )


def compare_field(errors: list[str], saved: JSONDict, expected: JSONDict, field: str) -> None:
    if saved.get(field) != expected.get(field):
        errors.append(f"{field} drifted: saved={saved.get(field)!r} expected={expected.get(field)!r}")


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def canonical_path_string(value: str) -> str:
    try:
        path = Path(value)
    except (TypeError, ValueError):
        return value.replace("\\", "/")
    try:
        display = display_path(path) if path.is_absolute() else display_path(ROOT / path)
    except (OSError, ValueError):
        display = None
    return (display or value).replace("\\", "/")


def canonical_compare_value(value: Any, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {str(child_key): canonical_compare_value(child_value, str(child_key)) for child_key, child_value in value.items()}
    if isinstance(value, list):
        return [canonical_compare_value(item, key) for item in value]
    if isinstance(value, str) and key and "path" in key:
        return canonical_path_string(value)
    return value


def compact_summary(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    keys = (
        "path",
        "exists",
        "valid",
        "ready",
        "prompt_count",
        "output_label",
        "output_present_count",
        "missing_output_count",
        "capture_receipt_ready",
        "proof_available",
        "proof_ready",
        "capture_complete",
        "proof_artifact_path",
        "capture_receipt_path",
        "source_request_path",
    )
    compact: JSONDict = {key: value.get(key) for key in keys if key in value}
    errors = value.get("errors")
    blockers = value.get("blockers")
    outputs = value.get("outputs")
    prompts = value.get("prompts")
    if isinstance(errors, list):
        compact["error_count"] = len(errors)
    if isinstance(blockers, list):
        compact["blocker_count"] = len(blockers)
    if isinstance(outputs, list):
        compact["output_count"] = len(outputs)
    if isinstance(prompts, list):
        compact["prompt_row_count"] = len(prompts)
    return compact or {"keys": sorted(str(key) for key in value)}


def compare_summary_field(errors: list[str], saved: JSONDict, expected: JSONDict, field: str) -> None:
    saved_value = canonical_compare_value(saved.get(field))
    expected_value = canonical_compare_value(expected.get(field))
    if stable_json(saved_value) != stable_json(expected_value):
        errors.append(
            f"{field} drifted: "
            f"saved={compact_summary(saved.get(field))!r} "
            f"expected={compact_summary(expected.get(field))!r}"
        )


def compare_artifacts(
    errors: list[str],
    saved: JSONDict,
    expected: JSONDict,
    collection: str,
    required_ids: list[str],
) -> None:
    saved_items = artifact_by_id(saved.get(collection))
    expected_items = artifact_by_id(expected.get(collection))
    for artifact_id in required_ids:
        saved_item = saved_items.get(artifact_id)
        expected_item = expected_items.get(artifact_id)
        if saved_item is None:
            errors.append(f"{collection} missing {artifact_id}")
            continue
        if expected_item is None:
            errors.append(f"expected {collection} missing {artifact_id}")
            continue
        for field in ("path", "status", "approval_required", "description", "source", "validator_commands"):
            if saved_item.get(field) != expected_item.get(field):
                errors.append(
                    f"{collection}.{artifact_id}.{field} drifted: "
                    f"saved={saved_item.get(field)!r} expected={expected_item.get(field)!r}"
                )


def status_counts(items: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not isinstance(items, list):
        return counts
    for item in items:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))

def validator_count(item: JSONDict) -> int:
    commands = item.get("validator_commands")
    return len(commands) if isinstance(commands, list) else 0


def validator_command_coverage(requested: Any, future: Any) -> JSONDict:
    required: list[tuple[str, str, JSONDict | None]] = []
    for stage, items in (("requested_artifacts", requested), ("future_artifacts", future)):
        by_id = artifact_by_id(items)
        required_ids = (
            ["candidate_router_trace", "managed_output_summary_fill", "dense_output_summary_fill"]
            if stage == "requested_artifacts"
            else ["live_capability_proof_fill"]
        )
        for artifact_id in required_ids:
            required.append((stage, artifact_id, by_id.get(artifact_id)))

    artifacts: list[JSONDict] = []
    missing: list[str] = []
    total_commands = 0
    for stage, artifact_id, artifact in required:
        count = validator_count(artifact) if isinstance(artifact, dict) else 0
        total_commands += count
        ready = count > 0
        if not ready:
            missing.append(artifact_id)
        artifacts.append(
            {
                "id": artifact_id,
                "stage": stage,
                "path": artifact.get("path") if isinstance(artifact, dict) else None,
                "status": artifact.get("status") if isinstance(artifact, dict) else "missing",
                "validator_command_count": count,
                "validator_commands_present": ready,
            }
        )
    return {
        "artifact_count": len(artifacts),
        "covered_artifact_count": sum(1 for artifact in artifacts if artifact["validator_commands_present"]),
        "validator_command_count": total_commands,
        "missing_validator_command_artifact_ids": missing,
        "all_required_validator_commands_present": len(artifacts) > 0 and not missing,
        "artifacts": artifacts,
    }


def aggregate_validator_command_coverage(requests: list[JSONDict]) -> JSONDict:
    coverages = [request.get("validator_command_coverage") for request in requests if isinstance(request, dict)]
    coverage_items = [coverage for coverage in coverages if isinstance(coverage, dict)]
    missing_by_request: list[JSONDict] = []
    for request, coverage in zip(requests, coverages):
        if not isinstance(request, dict) or not isinstance(coverage, dict):
            continue
        missing = coverage.get("missing_validator_command_artifact_ids")
        if isinstance(missing, list) and missing:
            missing_by_request.append(
                {
                    "request_name": request.get("name") or request.get("path"),
                    "missing_validator_command_artifact_ids": missing,
                }
            )
    return {
        "request_count": len(requests),
        "request_with_complete_validator_commands_count": sum(
            1 for coverage in coverage_items if coverage.get("all_required_validator_commands_present") is True
        ),
        "artifact_count": sum(int(coverage.get("artifact_count", 0) or 0) for coverage in coverage_items),
        "covered_artifact_count": sum(int(coverage.get("covered_artifact_count", 0) or 0) for coverage in coverage_items),
        "validator_command_count": sum(int(coverage.get("validator_command_count", 0) or 0) for coverage in coverage_items),
        "missing_request_count": len(missing_by_request),
        "missing_by_request": missing_by_request,
        "all_required_validator_commands_present": bool(requests) and not missing_by_request,
    }


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def command_to_text(command: Any) -> str:
    if isinstance(command, list):
        return " ".join(str(part) for part in command)
    return str(command)


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def source_blocker_summary(source: Any) -> str:
    if not isinstance(source, dict):
        return "none"
    blockers: list[str] = []
    receipt_blockers = source.get("capture_receipt_blockers")
    if isinstance(receipt_blockers, list):
        blockers.extend(str(item) for item in receipt_blockers if item)
    receipt_errors = source.get("capture_receipt_errors")
    if isinstance(receipt_errors, list) and receipt_errors:
        blockers.append(f"capture_receipt_errors={len(receipt_errors)}")
    prompt_coverage = source.get("prompt_coverage")
    if isinstance(prompt_coverage, dict):
        if prompt_coverage.get("ready") is False:
            blockers.append("prompt_coverage_not_ready")
        missing = prompt_coverage.get("missing_prompt_ids")
        extra = prompt_coverage.get("extra_output_ids")
        if isinstance(missing, list) and missing:
            blockers.append(f"missing_prompt_ids={len(missing)}")
        if isinstance(extra, list) and extra:
            blockers.append(f"extra_output_ids={len(extra)}")
    live_blockers = source.get("blockers")
    if isinstance(live_blockers, list):
        for item in live_blockers:
            if isinstance(item, dict) and item.get("id"):
                blockers.append(str(item["id"]))
            elif item:
                blockers.append(str(item))
    return ", ".join(blockers) if blockers else "none"


def receipt_requirement_artifacts(items: Any) -> list[JSONDict]:
    if not isinstance(items, list):
        return []
    result: list[JSONDict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        source = item.get("source") if isinstance(item.get("source"), dict) else {}
        if source.get("capture_receipt_required") is True or source.get("receipt_fill_note"):
            result.append(item)
    return result


def handoff_artifacts(items: Any) -> list[JSONDict]:
    if not isinstance(items, list):
        return []
    artifacts: list[JSONDict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        validators = item.get("validator_commands")
        artifacts.append(
            {
                "id": item.get("id"),
                "status": item.get("status"),
                "approval_required": item.get("approval_required") is True,
                "path": item.get("path"),
                "description": item.get("description"),
                "validator_commands": validators if isinstance(validators, list) else [],
                "source": item.get("source") if isinstance(item.get("source"), dict) else {},
            }
        )
    return artifacts


def approval_summary(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    return ", ".join(f"{key}={str(val).lower()}" for key, val in sorted(value.items()))


def pending_runtime_artifacts(request: JSONDict) -> list[JSONDict]:
    handoff = request.get("operator_handoff") if isinstance(request.get("operator_handoff"), dict) else {}
    requested = handoff.get("requested_artifacts") if isinstance(handoff.get("requested_artifacts"), list) else []
    pending: list[JSONDict] = []
    for artifact in requested:
        if not isinstance(artifact, dict):
            continue
        if artifact.get("status") not in COMPLETE_ARTIFACT_STATUSES:
            pending.append(artifact)
    return pending


def missing_runtime_approval_keys(request: JSONDict) -> list[str]:
    handoff = request.get("operator_handoff") if isinstance(request.get("operator_handoff"), dict) else {}
    approval_values = handoff.get("approvals") if isinstance(handoff.get("approvals"), dict) else {}
    return [key for key in RUNTIME_APPROVAL_KEYS if approval_values.get(key) is not True]


def artifact_map(items: Any) -> dict[str, JSONDict]:
    result: dict[str, JSONDict] = {}
    if not isinstance(items, list):
        return result
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        if isinstance(item_id, str):
            result[item_id] = item
    return result


def command_path(value: Any, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value
    return fallback


def source_value(artifact: JSONDict | None, key: str) -> Any:
    if not isinstance(artifact, dict):
        return None
    source = artifact.get("source") if isinstance(artifact.get("source"), dict) else {}
    return source.get(key)


def approval_rebuild_command(
    request: JSONDict,
    missing_keys: list[str],
    *,
    approval_scope: str = "all_missing_runtime_approvals",
    approval_scope_artifact_id: str | None = None,
    requested_keys: list[str] | None = None,
) -> JSONDict:
    handoff = request.get("operator_handoff") if isinstance(request.get("operator_handoff"), dict) else {}
    requested_by_id = artifact_map(handoff.get("requested_artifacts"))
    future_by_id = artifact_map(handoff.get("future_artifacts"))
    candidate = requested_by_id.get("candidate_router_trace")
    managed = requested_by_id.get("managed_output_summary_fill")
    dense = requested_by_id.get("dense_output_summary_fill")
    live = future_by_id.get("live_capability_proof_fill")
    approvals_value = handoff.get("approvals") if isinstance(handoff.get("approvals"), dict) else {}
    desired_keys = missing_keys if requested_keys is None else requested_keys
    keys_to_record = [
        key
        for key in RUNTIME_APPROVAL_KEYS
        if key in desired_keys or approvals_value.get(key) is True
    ]
    command = [
        "uv",
        "run",
        "--managed-python",
        "--python",
        "3.13",
        "scripts/build_phase3_runtime_capture_request.py",
        command_path(request.get("bundle_path"), "<bundle_path>"),
        "--prompt-set-path",
        command_path(handoff.get("prompt_set_path"), "<prompt_set_path>"),
        "--candidate-trace-path",
        command_path(candidate.get("path") if isinstance(candidate, dict) else None, "<candidate_trace_path>"),
        "--candidate-trace-receipt-path",
        command_path(source_value(candidate, "capture_receipt_path"), "<candidate_trace_receipt_path>"),
        "--managed-output-path",
        command_path(managed.get("path") if isinstance(managed, dict) else None, "<managed_output_summary_path>"),
        "--dense-output-path",
        command_path(dense.get("path") if isinstance(dense, dict) else None, "<dense_output_summary_path>"),
        "--live-proof-template-path",
        command_path(live.get("path") if isinstance(live, dict) else None, "<live_proof_template_path>"),
        "--output",
        command_path(request.get("path"), "<runtime_capture_request_path>"),
        "--json",
    ]
    for key in keys_to_record:
        command.append(APPROVAL_FLAG_BY_KEY[key])
    return {
        "command_class": "phase3_runtime_capture_request_approval_rebuild",
        "command": command,
        "records_approval_keys": keys_to_record,
        "writes_request_path": request.get("path"),
        "approval_scope": approval_scope,
        "approval_scope_artifact_id": approval_scope_artifact_id,
        "records_all_missing_approvals": set(missing_keys).issubset(set(keys_to_record)),
        "requires_explicit_user_approval": True,
        "metadata_only": True,
        "safety_contract": [
            "command rebuilds the saved runtime-capture request metadata only",
            "command does not launch model servers",
            "command does not run Docker",
            "command does not call endpoints",
            "command does not send prompt traffic",
        ],
    }


def next_artifact_approval_keys(artifact_id: Any, missing_keys: list[str]) -> list[str]:
    scoped = APPROVAL_KEYS_BY_ARTIFACT_ID.get(str(artifact_id or ""), ())
    return [key for key in scoped if key in missing_keys]


def next_artifact_approval_rebuild_command(request: JSONDict, missing_keys: list[str], artifact_id: Any) -> JSONDict:
    scoped_keys = next_artifact_approval_keys(artifact_id, missing_keys)
    if not scoped_keys:
        return {}
    return approval_rebuild_command(
        request,
        missing_keys,
        approval_scope="next_runtime_artifact",
        approval_scope_artifact_id=str(artifact_id or ""),
        requested_keys=scoped_keys,
    )


def capture_sequence_step(
    step_id: str,
    *,
    stage: str,
    status: str,
    action: str,
    path: Any = None,
    approval_keys: list[str] | None = None,
    validator_commands: Any = None,
) -> JSONDict:
    step: JSONDict = {
        "id": step_id,
        "stage": stage,
        "status": status,
        "action": action,
    }
    if path:
        step["path"] = path
    if approval_keys:
        step["approval_keys"] = approval_keys
    if isinstance(validator_commands, list) and validator_commands:
        step["validator_command_count"] = len(validator_commands)
    return step


def artifact_step_status(artifact: JSONDict | None) -> str:
    if not isinstance(artifact, dict):
        return "missing"
    return str(artifact.get("status") or "unknown")


def build_capture_sequence(request: JSONDict, missing_keys: list[str]) -> list[JSONDict]:
    handoff = request.get("operator_handoff") if isinstance(request.get("operator_handoff"), dict) else {}
    requested_by_id = artifact_map(handoff.get("requested_artifacts"))
    future_by_id = artifact_map(handoff.get("future_artifacts"))
    candidate = requested_by_id.get("candidate_router_trace")
    managed = requested_by_id.get("managed_output_summary_fill")
    dense = requested_by_id.get("dense_output_summary_fill")
    live = future_by_id.get("live_capability_proof_fill")
    return [
        capture_sequence_step(
            "record_runtime_approvals",
            stage="approval",
            status="approval_required" if missing_keys else "satisfied",
            action="Record explicit approval for prompt traffic, router trace capture, managed output capture, and dense output capture before runtime work.",
            approval_keys=missing_keys,
        ),
        capture_sequence_step(
            "capture_candidate_router_trace",
            stage="runtime_capture",
            status=artifact_step_status(candidate),
            action="Capture the candidate router trace with the shared prompt set and fill its trace capture_receipt.",
            path=candidate.get("path") if isinstance(candidate, dict) else None,
            validator_commands=candidate.get("validator_commands") if isinstance(candidate, dict) else None,
        ),
        capture_sequence_step(
            "fill_managed_output_summary",
            stage="runtime_capture",
            status=artifact_step_status(managed),
            action="Fill the managed output summary rows and capture_receipt from the approved managed run outputs.",
            path=managed.get("path") if isinstance(managed, dict) else None,
            validator_commands=managed.get("validator_commands") if isinstance(managed, dict) else None,
        ),
        capture_sequence_step(
            "fill_dense_output_summary",
            stage="runtime_capture",
            status=artifact_step_status(dense),
            action="Fill the dense/full-runtime output summary rows and capture_receipt from the same prompt ids.",
            path=dense.get("path") if isinstance(dense, dict) else None,
            validator_commands=dense.get("validator_commands") if isinstance(dense, dict) else None,
        ),
        capture_sequence_step(
            "run_capture_result_intake",
            stage="intake",
            status="pending_artifacts" if pending_runtime_artifacts(request) else "ready",
            action="Run scripts/plan_phase3_capture_result_intake.py --show-queue --output-md after the capture artifacts are filled.",
        ),
        capture_sequence_step(
            "defer_live_capability_proof",
            stage="future_adapter",
            status=artifact_step_status(live),
            action="Keep live residency observation/control and cleanup proof future-bound until an approved backend adapter exists.",
            path=live.get("path") if isinstance(live, dict) else None,
            validator_commands=live.get("validator_commands") if isinstance(live, dict) else None,
        ),
    ]


def approval_queue_status(request: JSONDict, missing_keys: list[str], pending: list[JSONDict]) -> str:
    if request.get("capture_complete") is True:
        return "complete"
    if request.get("valid") is not True or int(request.get("drift_count", 0) or 0) > 0:
        return "blocked_by_drift"
    if request.get("ready_for_operator_capture") is True:
        return "ready_for_operator_capture"
    if missing_keys:
        return "approval_required"
    if pending:
        return "artifact_capture_pending"
    return "blocked"


def capture_queue_haystack(item: JSONDict) -> str:
    return " ".join(
        str(item.get(key) or "")
        for key in ("request_name", "path", "bundle_path", "model_id", "prompt_family")
    ).lower()


def capture_queue_rank(item: JSONDict) -> tuple[int, str]:
    haystack = capture_queue_haystack(item)
    return (0 if "mixtral" in haystack else 1, str(item.get("request_name") or item.get("path") or ""))


def capture_queue_rank_reason(item: JSONDict) -> str:
    haystack = capture_queue_haystack(item)
    if "mixtral" in haystack:
        return "mixtral_first_known_sparse_baseline"
    return "stable_name_order_after_mixtral_baseline"


def capture_queue_summary(queue: list[JSONDict]) -> JSONDict:
    status_counts: dict[str, int] = {}
    backend_counts: dict[str, int] = {}
    prompt_counts: list[int] = []
    ranked_requests: list[JSONDict] = []
    for item in queue:
        status = str(item.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        backend = str(item.get("backend_family") or "unknown")
        backend_counts[backend] = backend_counts.get(backend, 0) + 1
        try:
            prompt_counts.append(int(item.get("prompt_count") or 0))
        except (TypeError, ValueError):
            pass
        missing = item.get("missing_approval_keys") if isinstance(item.get("missing_approval_keys"), list) else []
        pending = item.get("pending_artifact_ids") if isinstance(item.get("pending_artifact_ids"), list) else []
        ranked_requests.append(
            {
                "rank": item.get("queue_rank"),
                "request_name": item.get("request_name"),
                "status": item.get("status"),
                "selection_rationale": item.get("selection_rationale"),
                "next_artifact_id": item.get("next_artifact_id"),
                "pending_artifact_count": len(pending),
                "missing_approval_count": len(missing),
                "prompt_count": item.get("prompt_count"),
                "backend_family": item.get("backend_family"),
            }
        )
    return {
        "queue_count": len(queue),
        "status_counts": dict(sorted(status_counts.items())),
        "approval_required_count": status_counts.get("approval_required", 0),
        "ready_for_operator_capture_count": status_counts.get("ready_for_operator_capture", 0),
        "artifact_capture_pending_count": status_counts.get("artifact_capture_pending", 0),
        "approved_but_capture_incomplete_count": status_counts.get("ready_for_operator_capture", 0)
        + status_counts.get("artifact_capture_pending", 0),
        "blocked_by_drift_count": status_counts.get("blocked_by_drift", 0),
        "backend_family_counts": dict(sorted(backend_counts.items())),
        "prompt_count_min": min(prompt_counts) if prompt_counts else None,
        "prompt_count_max": max(prompt_counts) if prompt_counts else None,
        "recommended_rank": queue[0].get("queue_rank") if queue else None,
        "selection_contract": [
            "Prefer the known Mixtral sparse baseline first when all capture requests are otherwise equivalent.",
            "Keep non-Mixtral requests in stable name order so the queue is reproducible.",
            "Do not skip drift, approval, receipt, or validator gates when overriding the recommended request.",
        ],
        "ranked_requests": ranked_requests,
    }


def build_approval_queue(requests: list[JSONDict]) -> list[JSONDict]:
    queue: list[JSONDict] = []
    for request in requests:
        if not isinstance(request, dict):
            continue
        if request.get("capture_complete") is True:
            continue
        handoff = request.get("operator_handoff") if isinstance(request.get("operator_handoff"), dict) else {}
        pending = pending_runtime_artifacts(request)
        missing_keys = missing_runtime_approval_keys(request)
        next_artifact = pending[0] if pending else {}
        queue.append(
            {
                "request_name": request.get("name") or request.get("path") or "unnamed request",
                "path": request.get("path"),
                "bundle_path": request.get("bundle_path"),
                "model_id": request.get("model_id"),
                "backend_family": request.get("backend_family"),
                "prompt_family": request.get("prompt_family"),
                "prompt_set_path": handoff.get("prompt_set_path"),
                "prompt_count": handoff.get("prompt_count"),
                "status": approval_queue_status(request, missing_keys, pending),
                "approval_required": bool(missing_keys),
                "missing_approval_keys": missing_keys,
                "pending_artifact_ids": [str(artifact.get("id")) for artifact in pending if artifact.get("id")],
                "next_artifact_id": next_artifact.get("id"),
                "next_artifact_path": next_artifact.get("path"),
                "ready_for_operator_capture": request.get("ready_for_operator_capture") is True,
                "drift_count": int(request.get("drift_count", 0) or 0),
                "capture_sequence": build_capture_sequence(request, missing_keys),
                "approval_rebuild_command": approval_rebuild_command(request, missing_keys),
                "next_artifact_approval_rebuild_command": next_artifact_approval_rebuild_command(
                    request,
                    missing_keys,
                    next_artifact.get("id"),
                ),
            }
        )
    ranked = sorted(queue, key=capture_queue_rank)
    for index, item in enumerate(ranked, start=1):
        item["queue_rank"] = index
        item["selection_rationale"] = capture_queue_rank_reason(item)
    return ranked


def approval_rebuild_command_manifest(queue: list[JSONDict]) -> list[JSONDict]:
    manifest: list[JSONDict] = []
    for item in queue:
        command = item.get("approval_rebuild_command") if isinstance(item.get("approval_rebuild_command"), dict) else {}
        if not command.get("command"):
            continue
        manifest.append(
            {
                "queue_rank": item.get("queue_rank"),
                "request_name": item.get("request_name"),
                "request_path": item.get("path"),
                "bundle_path": item.get("bundle_path"),
                "status": item.get("status"),
                "next_artifact_id": item.get("next_artifact_id"),
                "command_class": command.get("command_class"),
                "writes_request_path": command.get("writes_request_path"),
                "records_approval_keys": string_list(command.get("records_approval_keys")),
                "approval_scope": command.get("approval_scope"),
                "approval_scope_artifact_id": command.get("approval_scope_artifact_id"),
                "records_all_missing_approvals": command.get("records_all_missing_approvals") is True,
                "requires_explicit_user_approval": command.get("requires_explicit_user_approval") is True,
                "metadata_only": command.get("metadata_only") is True,
                "command": string_list(command.get("command")),
            }
        )
    return manifest


def next_artifact_approval_rebuild_command_manifest(queue: list[JSONDict]) -> list[JSONDict]:
    manifest: list[JSONDict] = []
    for item in queue:
        command = item.get("next_artifact_approval_rebuild_command") if isinstance(item.get("next_artifact_approval_rebuild_command"), dict) else {}
        if not command.get("command"):
            continue
        manifest.append(
            {
                "queue_rank": item.get("queue_rank"),
                "request_name": item.get("request_name"),
                "request_path": item.get("path"),
                "bundle_path": item.get("bundle_path"),
                "status": item.get("status"),
                "next_artifact_id": item.get("next_artifact_id"),
                "command_class": command.get("command_class"),
                "writes_request_path": command.get("writes_request_path"),
                "records_approval_keys": string_list(command.get("records_approval_keys")),
                "approval_scope": command.get("approval_scope"),
                "approval_scope_artifact_id": command.get("approval_scope_artifact_id"),
                "records_all_missing_approvals": command.get("records_all_missing_approvals") is True,
                "requires_explicit_user_approval": command.get("requires_explicit_user_approval") is True,
                "metadata_only": command.get("metadata_only") is True,
                "command": string_list(command.get("command")),
            }
        )
    return manifest


def recommended_capture_request(queue: list[JSONDict]) -> JSONDict | None:
    for item in queue:
        if item.get("status") not in {"complete", "blocked_by_drift"}:
            return item
    return None


def request_handoff(saved: JSONDict) -> JSONDict:
    prompt_set = saved.get("prompt_set") if isinstance(saved.get("prompt_set"), dict) else {}
    return {
        "approvals": approvals(saved),
        "approval_summary": approval_summary(approvals(saved)),
        "prompt_set_path": prompt_set.get("path"),
        "prompt_count": prompt_set.get("prompt_count"),
        "requested_artifacts": handoff_artifacts(saved.get("requested_artifacts")),
        "future_artifacts": handoff_artifacts(saved.get("future_artifacts")),
        "next_actions": saved.get("next_actions") if isinstance(saved.get("next_actions"), list) else [],
    }


def request_preview_artifact_path(artifact: JSONDict | None) -> Path | None:
    if not isinstance(artifact, dict):
        return None
    return resolve_repo_path(artifact.get("path"))


def request_preview_source_path(artifact: JSONDict | None, key: str) -> Path | None:
    if not isinstance(artifact, dict):
        return None
    source = artifact.get("source") if isinstance(artifact.get("source"), dict) else {}
    return resolve_repo_path(source.get(key))


def post_approval_preview(request: JSONDict) -> JSONDict:
    handoff = request.get("operator_handoff") if isinstance(request.get("operator_handoff"), dict) else {}
    requested_by_id = artifact_map(handoff.get("requested_artifacts"))
    future_by_id = artifact_map(handoff.get("future_artifacts"))
    bundle_path = resolve_repo_path(request.get("bundle_path"))
    if bundle_path is None:
        return {
            "preview_only": True,
            "valid": False,
            "errors": ["request bundle_path is missing"],
            "request_name": request.get("name") or request.get("path"),
            "request_path": request.get("path"),
            "mutates_request": False,
        }

    try:
        approved = build_phase3_runtime_capture_request.build_request(
            bundle_path,
            prompt_set_path=resolve_repo_path(handoff.get("prompt_set_path")),
            candidate_trace_path=request_preview_artifact_path(requested_by_id.get("candidate_router_trace")),
            candidate_trace_receipt_path=request_preview_source_path(
                requested_by_id.get("candidate_router_trace"),
                "capture_receipt_path",
            ),
            managed_output_path=request_preview_artifact_path(requested_by_id.get("managed_output_summary_fill")),
            dense_output_path=request_preview_artifact_path(requested_by_id.get("dense_output_summary_fill")),
            live_proof_template_path=request_preview_artifact_path(future_by_id.get("live_capability_proof_fill")),
            router_trace_capture_approved=True,
            managed_output_capture_approved=True,
            dense_output_capture_approved=True,
            runtime_prompt_traffic_approved=True,
        )
        validation_errors = build_phase3_runtime_capture_request.validate_request(approved)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {
            "preview_only": True,
            "valid": False,
            "errors": [str(exc)],
            "request_name": request.get("name") or request.get("path"),
            "request_path": request.get("path"),
            "mutates_request": False,
        }

    errors = [str(error) for error in approved.get("errors", [])]
    errors.extend(str(error) for error in validation_errors if str(error) not in errors)
    audit_like = {
        "path": request.get("path"),
        "valid": not errors,
        "name": approved.get("name"),
        "bundle_path": approved.get("bundle_path"),
        "model_id": approved.get("model_id"),
        "backend_family": approved.get("backend_family"),
        "prompt_family": approved.get("prompt_family"),
        "ready_for_operator_capture": approved.get("ready_for_operator_capture") is True,
        "capture_complete": approved.get("capture_complete") is True,
        "drift_count": request.get("drift_count", 0),
        "operator_handoff": request_handoff(approved),
    }
    pending = pending_runtime_artifacts(audit_like)
    missing_keys = missing_runtime_approval_keys(audit_like)
    next_artifact = pending[0] if pending else {}
    requested_artifacts = approved.get("requested_artifacts") if isinstance(approved.get("requested_artifacts"), list) else []
    return {
        "preview_only": True,
        "valid": not errors,
        "errors": errors,
        "request_name": request.get("name") or approved.get("name") or request.get("path"),
        "request_path": request.get("path"),
        "status": approval_queue_status(audit_like, missing_keys, pending),
        "ready_for_operator_capture": approved.get("ready_for_operator_capture") is True,
        "capture_complete": approved.get("capture_complete") is True,
        "missing_approval_keys_after_preview": missing_keys,
        "records_approval_keys": list(RUNTIME_APPROVAL_KEYS),
        "requested_status_counts": status_counts(requested_artifacts),
        "pending_artifact_ids": [str(artifact.get("id")) for artifact in pending if artifact.get("id")],
        "pending_artifact_count": len(pending),
        "next_artifact_id": next_artifact.get("id"),
        "next_artifact_path": next_artifact.get("path"),
        "mutates_request": False,
        "still_requires_capture_artifacts": bool(pending),
        "artifact_statuses": [
            {
                "id": item.get("id"),
                "status": item.get("status"),
                "path": item.get("path"),
            }
            for item in requested_artifacts
            if isinstance(item, dict)
        ],
        "safety_contract": [
            "preview rebuilds the recommended request in memory only",
            "preview does not write the runtime-capture request",
            "preview does not launch model servers",
            "preview does not run Docker",
            "preview does not call endpoints",
            "preview does not send prompt traffic",
        ],
    }


def recommended_post_approval_preview(requests: list[JSONDict], recommended: JSONDict | None) -> JSONDict | None:
    if not isinstance(recommended, dict):
        return None
    recommended_path = recommended.get("path")
    for request in requests:
        if isinstance(request, dict) and request.get("path") == recommended_path:
            return post_approval_preview(request)
    return None


def format_markdown_report(summary: JSONDict) -> str:
    lines = [
        "# Phase 3 Runtime-Capture Request Audit",
        "",
        f"- Valid: `{summary.get('valid')}`",
        f"- Requests: `{summary.get('request_count')}`",
        f"- Valid requests: `{summary.get('valid_request_count')}`",
        f"- Ready for operator capture: `{summary.get('ready_for_operator_capture_count')}`",
        f"- Capture complete: `{summary.get('capture_complete_count')}`",
        f"- Drifted requests: `{summary.get('drifted_request_count')}`",
        f"- Validator commands: `{summary.get('validator_command_count')}`",
        f"- Requests missing validator commands: `{summary.get('validator_command_missing_request_count')}`",
        f"- All required validator commands present: `{summary.get('all_required_validator_commands_present')}`",
        "",
    ]
    recommended = summary.get("recommended_runtime_capture_request")
    if isinstance(recommended, dict):
        missing = recommended.get("missing_approval_keys") if isinstance(recommended.get("missing_approval_keys"), list) else []
        pending = recommended.get("pending_artifact_ids") if isinstance(recommended.get("pending_artifact_ids"), list) else []
        lines.extend(
            [
                "## Recommended First Capture",
                "",
                f"- Request: `{markdown_escape(recommended.get('request_name') or 'missing')}`",
                f"- Status: `{markdown_escape(recommended.get('status') or 'unknown')}`",
                f"- Bundle: `{markdown_escape(recommended.get('bundle_path') or 'missing')}`",
                f"- Model: `{markdown_escape(recommended.get('model_id') or 'missing')}`",
                f"- Prompt set: `{markdown_escape(recommended.get('prompt_set_path') or 'missing')}`",
                f"- Prompt count: `{markdown_escape(recommended.get('prompt_count') or 'unknown')}`",
                f"- Missing approvals: `{markdown_escape(', '.join(str(item) for item in missing) if missing else 'none')}`",
                f"- Next artifact: `{markdown_escape(recommended.get('next_artifact_id') or 'none')}`",
                f"- Next artifact path: `{markdown_escape(recommended.get('next_artifact_path') or 'missing')}`",
                f"- Pending artifacts: `{markdown_escape(', '.join(str(item) for item in pending) if pending else 'none')}`",
                f"- Queue rank: `{markdown_escape(recommended.get('queue_rank') or 'unknown')}`",
                f"- Selection rationale: `{markdown_escape(recommended.get('selection_rationale') or 'unknown')}`",
                "",
            ]
        )
        approval_command = recommended.get("approval_rebuild_command") if isinstance(recommended.get("approval_rebuild_command"), dict) else {}
        if approval_command:
            records = approval_command.get("records_approval_keys") if isinstance(approval_command.get("records_approval_keys"), list) else []
            lines.extend(
                [
                    "### Approval Metadata Rebuild",
                    "",
                    f"- Command class: `{markdown_escape(approval_command.get('command_class') or 'unknown')}`",
                    f"- Records approvals: `{markdown_escape(', '.join(str(item) for item in records) if records else 'none')}`",
                    f"- Writes request: `{markdown_escape(approval_command.get('writes_request_path') or 'missing')}`",
                    f"- Requires explicit user approval: `{markdown_escape(approval_command.get('requires_explicit_user_approval'))}`",
                    "```sh",
                    command_to_text(approval_command.get("command")),
                    "```",
                    "",
                ]
            )
        lines.extend(
            [
                "### Recommended Capture Sequence",
                "",
                "| Step | Stage | Status | Path / approval |",
                "| --- | --- | --- | --- |",
            ]
        )
        sequence = recommended.get("capture_sequence") if isinstance(recommended.get("capture_sequence"), list) else []
        for step in sequence:
            if not isinstance(step, dict):
                continue
            approval_keys = step.get("approval_keys") if isinstance(step.get("approval_keys"), list) else []
            path_or_approval = step.get("path") or (", ".join(str(item) for item in approval_keys) if approval_keys else "none")
            lines.append(
                "| "
                + " | ".join(
                    markdown_escape(value)
                    for value in (
                        step.get("id") or "unknown",
                        step.get("stage") or "unknown",
                        step.get("status") or "unknown",
                        path_or_approval,
                    )
                )
                + " |"
            )
        lines.append("")
    post_approval = summary.get("recommended_post_approval_preview") if isinstance(summary.get("recommended_post_approval_preview"), dict) else {}
    if post_approval:
        pending = post_approval.get("pending_artifact_ids") if isinstance(post_approval.get("pending_artifact_ids"), list) else []
        records = post_approval.get("records_approval_keys") if isinstance(post_approval.get("records_approval_keys"), list) else []
        lines.extend(
            [
                "### Post-Approval Preview",
                "",
                f"- Preview only: `{markdown_escape(post_approval.get('preview_only'))}`",
                f"- Mutates request: `{markdown_escape(post_approval.get('mutates_request'))}`",
                f"- Valid: `{markdown_escape(post_approval.get('valid'))}`",
                f"- Status after approval: `{markdown_escape(post_approval.get('status') or 'unknown')}`",
                f"- Ready for operator capture after approval: `{markdown_escape(post_approval.get('ready_for_operator_capture'))}`",
                f"- Capture complete after approval: `{markdown_escape(post_approval.get('capture_complete'))}`",
                f"- Records approvals: `{markdown_escape(', '.join(str(item) for item in records) if records else 'none')}`",
                f"- Requested status counts: `{markdown_escape(json.dumps(post_approval.get('requested_status_counts', {}), sort_keys=True))}`",
                f"- Pending artifacts after approval: `{markdown_escape(', '.join(str(item) for item in pending) if pending else 'none')}`",
                f"- Next artifact after approval: `{markdown_escape(post_approval.get('next_artifact_id') or 'none')}`",
                f"- Next artifact path: `{markdown_escape(post_approval.get('next_artifact_path') or 'none')}`",
                "",
            ]
        )
    queue_summary = summary.get("capture_queue_summary") if isinstance(summary.get("capture_queue_summary"), dict) else {}
    ranked_requests = queue_summary.get("ranked_requests") if isinstance(queue_summary.get("ranked_requests"), list) else []
    if queue_summary:
        lines.extend(
            [
                "## Capture Queue Selection",
                "",
                f"- Queue count: `{markdown_escape(queue_summary.get('queue_count'))}`",
                f"- Status counts: `{markdown_escape(json.dumps(queue_summary.get('status_counts', {}), sort_keys=True))}`",
                f"- Approval required: `{markdown_escape(queue_summary.get('approval_required_count'))}`",
                f"- Ready for operator capture: `{markdown_escape(queue_summary.get('ready_for_operator_capture_count'))}`",
                f"- Approved but capture incomplete: `{markdown_escape(queue_summary.get('approved_but_capture_incomplete_count'))}`",
                f"- Backend counts: `{markdown_escape(json.dumps(queue_summary.get('backend_family_counts', {}), sort_keys=True))}`",
                f"- Prompt count range: `{markdown_escape(queue_summary.get('prompt_count_min'))}` to `{markdown_escape(queue_summary.get('prompt_count_max'))}`",
                "",
            ]
        )
        selection_contract = queue_summary.get("selection_contract") if isinstance(queue_summary.get("selection_contract"), list) else []
        if selection_contract:
            lines.append("Selection contract:")
            for item in selection_contract:
                lines.append(f"- {markdown_escape(item)}")
            lines.append("")
        if ranked_requests:
            lines.extend(
                [
                    "| Rank | Request | Status | Rationale | Next Artifact | Pending | Missing Approvals |",
                    "| --- | --- | --- | --- | --- | --- | --- |",
                ]
            )
            for item in ranked_requests:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        markdown_escape(value)
                        for value in (
                            item.get("rank"),
                            item.get("request_name") or "missing",
                            item.get("status") or "unknown",
                            item.get("selection_rationale") or "unknown",
                            item.get("next_artifact_id") or "none",
                            item.get("pending_artifact_count"),
                            item.get("missing_approval_count"),
                        )
                    )
                    + " |"
                )
            lines.append("")
    approval_manifest = summary.get("approval_rebuild_command_manifest") if isinstance(summary.get("approval_rebuild_command_manifest"), list) else []
    if approval_manifest:
        lines.extend(
            [
                "## Approval Rebuild Command Manifest",
                "",
                "| Rank | Request | Status | Writes Request | Records | Explicit Approval | Metadata Only |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for item in approval_manifest:
            if not isinstance(item, dict):
                continue
            records = item.get("records_approval_keys") if isinstance(item.get("records_approval_keys"), list) else []
            lines.append(
                "| "
                + " | ".join(
                    markdown_escape(value)
                    for value in (
                        item.get("queue_rank"),
                        item.get("request_name") or "missing",
                        item.get("status") or "unknown",
                        item.get("writes_request_path") or "missing",
                        ", ".join(str(key) for key in records) if records else "none",
                        item.get("requires_explicit_user_approval"),
                        item.get("metadata_only"),
                    )
                )
                + " |"
            )
        lines.append("")
    lines.extend(
        [
            "## Operator Handoff Queue",
            "",
            "| Request | Ready | Complete | Drift | Validators | Approvals | Prompt Set |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    requests = summary.get("requests") if isinstance(summary.get("requests"), list) else []
    for request in requests:
        if not isinstance(request, dict):
            continue
        handoff = request.get("operator_handoff") if isinstance(request.get("operator_handoff"), dict) else {}
        lines.append(
            "| "
            + " | ".join(
                markdown_escape(value)
                for value in (
                    request.get("name") or request.get("path") or "unnamed request",
                    request.get("ready_for_operator_capture"),
                    request.get("capture_complete"),
                    request.get("drift_count", 0),
                    request.get("validator_command_coverage", {}).get("validator_command_count", 0),
                    handoff.get("approval_summary") or "none",
                    handoff.get("prompt_set_path") or "missing",
                )
            )
            + " |"
        )

    for request in requests:
        if not isinstance(request, dict):
            continue
        handoff = request.get("operator_handoff") if isinstance(request.get("operator_handoff"), dict) else {}
        request_name = request.get("name") or request.get("path") or "Unnamed request"
        lines.extend(
            [
                "",
                f"## {markdown_escape(request_name)}",
                "",
                f"- Bundle: `{markdown_escape(request.get('bundle_path') or 'missing')}`",
                f"- Model: `{markdown_escape(request.get('model_id') or 'missing')}`",
                f"- Backend: `{markdown_escape(request.get('backend_family') or 'missing')}`",
                f"- Prompt family: `{markdown_escape(request.get('prompt_family') or 'missing')}`",
                f"- Prompt set: `{markdown_escape(handoff.get('prompt_set_path') or 'missing')}`",
                f"- Prompt count: `{markdown_escape(handoff.get('prompt_count') or 'unknown')}`",
                f"- Approvals: `{markdown_escape(handoff.get('approval_summary') or 'none')}`",
                f"- Validator commands: `{markdown_escape(request.get('validator_command_coverage', {}).get('validator_command_count', 0))}`",
                f"- Required validator commands present: `{markdown_escape(request.get('validator_command_coverage', {}).get('all_required_validator_commands_present'))}`",
                "",
                "### Requested Artifacts",
                "",
                "| Artifact | Status | Approval | Path | Validators |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for artifact in handoff.get("requested_artifacts", []):
            if not isinstance(artifact, dict):
                continue
            approval = "required" if artifact.get("approval_required") is True else "not required"
            validators = artifact.get("validator_commands") if isinstance(artifact.get("validator_commands"), list) else []
            lines.append(
                "| "
                + " | ".join(
                    markdown_escape(value)
                    for value in (
                        artifact.get("id") or "unknown",
                        artifact.get("status") or "unknown",
                        approval,
                        artifact.get("path") or "missing",
                        len(validators),
                    )
                )
                + " |"
            )
        receipt_artifacts = receipt_requirement_artifacts(handoff.get("requested_artifacts"))
        if receipt_artifacts:
            lines.extend(
                [
                    "",
                    "### Capture Receipt Requirements",
                    "",
                    "| Artifact | Receipt | Ready | Required | Prompt Set | Blockers | Fill Note |",
                    "| --- | --- | --- | --- | --- | --- | --- |",
                ]
            )
            for artifact in receipt_artifacts:
                source = artifact.get("source") if isinstance(artifact.get("source"), dict) else {}
                receipt_path = source.get("capture_receipt_path") or artifact.get("path") or "embedded in artifact"
                prompt_set = source.get("prompt_set_path") or handoff.get("prompt_set_path") or "missing"
                lines.append(
                    "| "
                    + " | ".join(
                        markdown_escape(value)
                        for value in (
                            artifact.get("id") or "unknown",
                            receipt_path,
                            source.get("capture_receipt_ready"),
                            source.get("capture_receipt_required"),
                            prompt_set,
                            source_blocker_summary(source),
                            source.get("receipt_fill_note") or "none",
                        )
                    )
                    + " |"
                )
        future_artifacts = handoff.get("future_artifacts") if isinstance(handoff.get("future_artifacts"), list) else []
        if future_artifacts:
            lines.extend(["", "### Future Adapter Artifacts", "", "| Artifact | Status | Approval | Path | Validators |", "| --- | --- | --- | --- | --- |"])
            for artifact in future_artifacts:
                if not isinstance(artifact, dict):
                    continue
                approval = "required" if artifact.get("approval_required") is True else "not required"
                validators = artifact.get("validator_commands") if isinstance(artifact.get("validator_commands"), list) else []
                lines.append(
                    "| "
                    + " | ".join(
                        markdown_escape(value)
                        for value in (
                            artifact.get("id") or "unknown",
                            artifact.get("status") or "unknown",
                            approval,
                            artifact.get("path") or "missing",
                            len(validators),
                        )
                    )
                    + " |"
                )
        lines.extend(["", "### Validator Commands", ""])
        for artifact in [*handoff.get("requested_artifacts", []), *future_artifacts]:
            if not isinstance(artifact, dict):
                continue
            validators = artifact.get("validator_commands") if isinstance(artifact.get("validator_commands"), list) else []
            for command in validators:
                lines.append(f"- `{markdown_escape(artifact.get('id') or 'unknown')}`")
                lines.append("```sh")
                lines.append(command_to_text(command))
                lines.append("```")
        next_actions = handoff.get("next_actions") if isinstance(handoff.get("next_actions"), list) else []
        if next_actions:
            lines.extend(["", "### Next Actions", ""])
            for action in next_actions:
                lines.append(f"- {markdown_escape(action)}")

    lines.extend(["", "## Safety Contract", ""])
    for item in summary.get("safety_contract", []):
        lines.append(f"- {markdown_escape(item)}")
    return "\n".join(lines) + "\n"


def write_markdown_report(summary: JSONDict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_markdown_report(summary), encoding="utf-8")


def summarize_request(path: Path) -> JSONDict:
    errors: list[str] = []
    if not path.exists():
        return {
            "path": display_path(path),
            "exists": False,
            "valid": False,
            "errors": [f"runtime-capture request does not exist: {display_path(path)}"],
            "drift_count": 0,
        }

    saved = load_request(path)
    errors.extend(build_phase3_runtime_capture_request.validate_request(saved))
    if saved.get("schema_version") != build_phase3_runtime_capture_request.SUPPORTED_SCHEMA_VERSION:
        errors.append(
            "schema_version must be "
            f"{build_phase3_runtime_capture_request.SUPPORTED_SCHEMA_VERSION!r}, got {saved.get('schema_version')!r}"
        )

    expected: JSONDict | None = None
    drift_errors: list[str] = []
    try:
        expected = rebuild_expected(saved)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        drift_errors.append(f"could not rebuild expected request: {exc}")
    if expected is not None:
        for field in (
            "valid",
            "bundle_path",
            "model_id",
            "backend_family",
            "prompt_family",
            "trace_path",
            "inventory_path",
            "policies_path",
            "ready_for_operator_capture",
            "capture_complete",
            "runtime_prompt_traffic_approved",
        ):
            compare_field(drift_errors, saved, expected, field)
        if saved.get("approvals") != expected.get("approvals"):
            drift_errors.append(
                f"approvals drifted: saved={saved.get('approvals')!r} expected={expected.get('approvals')!r}"
            )
        for field in (
            "prompt_set",
            "managed_output_summary",
            "dense_output_summary",
            "live_capability_proof",
            "prompt_coverage",
            "candidate_trace_receipt",
            "candidate_trace_reuse",
        ):
            compare_summary_field(drift_errors, saved, expected, field)
        compare_artifacts(
            drift_errors,
            saved,
            expected,
            "requested_artifacts",
            ["candidate_router_trace", "managed_output_summary_fill", "dense_output_summary_fill"],
        )
        compare_artifacts(
            drift_errors,
            saved,
            expected,
            "future_artifacts",
            ["live_capability_proof_fill"],
        )
    errors.extend(drift_errors)

    requested = saved.get("requested_artifacts", [])
    future = saved.get("future_artifacts", [])
    command_coverage = validator_command_coverage(requested, future)
    return {
        "path": display_path(path),
        "exists": True,
        "valid": not errors,
        "errors": errors,
        "drift_count": len(drift_errors),
        "bundle_path": saved.get("bundle_path"),
        "name": saved.get("name"),
        "model_id": saved.get("model_id"),
        "backend_family": saved.get("backend_family"),
        "prompt_family": saved.get("prompt_family"),
        "ready_for_operator_capture": saved.get("ready_for_operator_capture") is True,
        "capture_complete": saved.get("capture_complete") is True,
        "requested_status_counts": status_counts(requested),
        "future_status_counts": status_counts(future),
        "validator_command_coverage": command_coverage,
        "requested_artifact_ids": sorted(artifact_by_id(requested)),
        "future_artifact_ids": sorted(artifact_by_id(future)),
        "operator_handoff": request_handoff(saved),
    }


def build_root_summary(root: Path = DEFAULT_ROOT) -> JSONDict:
    errors: list[str] = []
    if not root.exists():
        errors.append(f"runtime-capture request root does not exist: {display_path(root)}")
        request_paths: list[Path] = []
    else:
        request_paths = sorted(root.glob(REQUEST_GLOB))

    requests: list[JSONDict] = []
    for path in request_paths:
        try:
            requests.append(summarize_request(path))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            requests.append(
                {
                    "path": display_path(path),
                    "exists": path.exists(),
                    "valid": False,
                    "errors": [str(exc)],
                    "drift_count": 0,
                }
            )

    approval_queue = build_approval_queue(requests)
    recommended = recommended_capture_request(approval_queue)
    post_approval = recommended_post_approval_preview(requests, recommended)
    approval_manifest = approval_rebuild_command_manifest(approval_queue)
    next_artifact_approval_manifest = next_artifact_approval_rebuild_command_manifest(approval_queue)
    command_coverage = aggregate_validator_command_coverage(requests)

    return {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "mode": "phase3_runtime_capture_request_audit",
        "root": display_path(root),
        "valid": not errors and all(item.get("valid") is True for item in requests),
        "errors": errors,
        "request_count": len(requests),
        "valid_request_count": sum(1 for item in requests if item.get("valid") is True),
        "ready_for_operator_capture_count": sum(
            1 for item in requests if item.get("ready_for_operator_capture") is True
        ),
        "capture_complete_count": sum(1 for item in requests if item.get("capture_complete") is True),
        "drifted_request_count": sum(1 for item in requests if int(item.get("drift_count", 0) or 0) > 0),
        "validator_command_coverage": command_coverage,
        "validator_command_count": command_coverage["validator_command_count"],
        "validator_command_missing_request_count": command_coverage["missing_request_count"],
        "all_required_validator_commands_present": command_coverage["all_required_validator_commands_present"],
        "approval_queue_count": len(approval_queue),
        "capture_queue_summary": capture_queue_summary(approval_queue),
        "approval_rebuild_command_manifest_count": len(approval_manifest),
        "approval_rebuild_command_manifest": approval_manifest,
        "next_artifact_approval_rebuild_command_manifest_count": len(next_artifact_approval_manifest),
        "next_artifact_approval_rebuild_command_manifest": next_artifact_approval_manifest,
        "recommended_runtime_capture_request": recommended,
        "recommended_post_approval_preview": post_approval,
        "approval_queue": approval_queue,
        "requests": requests,
        "safety_contract": [
            "runtime-capture request audit reads local metadata only",
            "runtime-capture request audit does not launch model servers",
            "runtime-capture request audit does not run Docker",
            "runtime-capture request audit does not call endpoints",
            "runtime-capture request audit does not download models",
            "runtime-capture request audit does not inspect private tokens",
            "runtime-capture request audit does not send prompt traffic",
            "runtime-capture request audit does not mutate runtime residency",
            "runtime-capture request audit does not claim live expert paging",
        ],
    }


def print_human_summary(summary: JSONDict) -> None:
    print("MoE Run Anyway Phase 3 runtime-capture request audit")
    print(f"Valid: {summary['valid']}")
    print(f"Requests: {summary['request_count']}")
    print(f"Valid requests: {summary['valid_request_count']}")
    print(f"Ready for operator capture: {summary['ready_for_operator_capture_count']}")
    print(f"Capture complete: {summary['capture_complete_count']}")
    print(f"Drifted requests: {summary['drifted_request_count']}")
    print(f"Validator commands: {summary.get('validator_command_count')}")
    print(f"Requests missing validator commands: {summary.get('validator_command_missing_request_count')}")
    print(f"All required validator commands present: {summary.get('all_required_validator_commands_present')}")
    recommended = summary.get("recommended_runtime_capture_request")
    if isinstance(recommended, dict):
        missing = recommended.get("missing_approval_keys") if isinstance(recommended.get("missing_approval_keys"), list) else []
        print(
            "Recommended first capture: "
            f"{recommended.get('request_name')} status={recommended.get('status')} "
            f"next={recommended.get('next_artifact_id')} "
            f"missing_approvals={', '.join(str(item) for item in missing) if missing else 'none'}"
        )
    post_approval = summary.get("recommended_post_approval_preview")
    if isinstance(post_approval, dict):
        pending = post_approval.get("pending_artifact_ids") if isinstance(post_approval.get("pending_artifact_ids"), list) else []
        print(
            "Post-approval preview: "
            f"status={post_approval.get('status')} "
            f"ready={post_approval.get('ready_for_operator_capture')} "
            f"complete={post_approval.get('capture_complete')} "
            f"next={post_approval.get('next_artifact_id')} "
            f"pending={', '.join(str(item) for item in pending) if pending else 'none'}"
        )
    if summary["errors"]:
        print("Errors:")
        for error in summary["errors"]:
            print(f"  - {error}")
    print("Requests:")
    for item in summary["requests"]:
        print(
            "  - "
            f"{item.get('name') or item.get('path')}: valid={item.get('valid')} "
            f"ready={item.get('ready_for_operator_capture')} complete={item.get('capture_complete')} "
            f"drift={item.get('drift_count', 0)} "
            f"validators={item.get('validator_command_coverage', {}).get('validator_command_count', 0)}"
        )
        for error in item.get("errors", []):
            print(f"    - {error}")
    print("Safety contract:")
    for item in summary["safety_contract"]:
        print(f"  - {item}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request_path", nargs="?", type=Path)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    parser.add_argument("--output-md", type=Path, help="write a Markdown operator handoff report")
    return parser


def plan_paths(request_path: Path | None = None, *, root: Path = DEFAULT_ROOT) -> tuple[int, JSONDict | None, str | None]:
    try:
        if request_path is not None:
            request = summarize_request(request_path)
            command_coverage = aggregate_validator_command_coverage([request])
            approval_queue = build_approval_queue([request])
            recommended = recommended_capture_request(approval_queue)
            post_approval = recommended_post_approval_preview([request], recommended)
            summary = {
                "schema_version": SUPPORTED_SCHEMA_VERSION,
                "mode": "phase3_runtime_capture_request_audit",
                "root": None,
                "valid": request.get("valid") is True,
                "errors": [],
                "request_count": 1,
                "valid_request_count": 1 if request.get("valid") is True else 0,
                "ready_for_operator_capture_count": 1 if request.get("ready_for_operator_capture") is True else 0,
                "capture_complete_count": 1 if request.get("capture_complete") is True else 0,
                "drifted_request_count": 1 if int(request.get("drift_count", 0) or 0) > 0 else 0,
                "validator_command_coverage": command_coverage,
                "validator_command_count": command_coverage["validator_command_count"],
                "validator_command_missing_request_count": command_coverage["missing_request_count"],
                "all_required_validator_commands_present": command_coverage["all_required_validator_commands_present"],
                "approval_queue_count": len(approval_queue),
                "capture_queue_summary": capture_queue_summary(approval_queue),
                "approval_rebuild_command_manifest_count": len(approval_rebuild_command_manifest(approval_queue)),
                "approval_rebuild_command_manifest": approval_rebuild_command_manifest(approval_queue),
                "recommended_runtime_capture_request": recommended,
                "recommended_post_approval_preview": post_approval,
                "approval_queue": approval_queue,
                "requests": [request],
                "safety_contract": build_root_summary(root)["safety_contract"],
            }
        else:
            summary = build_root_summary(root)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not build Phase 3 runtime-capture request audit: {exc}"
    return (0 if summary["valid"] else 2), summary, None


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary, error_message = plan_paths(args.request_path, root=args.root)
    if error_message:
        print(error_message, file=sys.stderr)
        return status
    assert summary is not None
    if args.output_md is not None:
        write_markdown_report(summary, args.output_md)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_human_summary(summary)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
