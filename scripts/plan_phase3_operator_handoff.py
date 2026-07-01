#!/usr/bin/env python3
"""Validate a metadata-only Phase 3 operator handoff package.

The validator reads an operator handoff directory produced by
`scripts/plan_phase3_evidence_packet.py --output-operator-handoff-dir`. When no
path is provided it generates a temporary package from the current evidence
packet and validates that package. It never launches runtimes, runs Docker,
reads secrets, mutates residency, or sends prompt traffic.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any

import plan_phase3_evidence_packet
import plan_phase3_launch_card_library
import plan_phase3_runtime_capture_commands


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_SCHEMA_VERSION = "moe-phase3-operator-handoff-validation-v1"
REQUIRED_FILES = {
    "README.md": "operator_handoff_readme",
    "operator-handoff-manifest.json": "operator_handoff_manifest",
    "phase3-evidence-packet.json": "phase3_evidence_packet_json",
    "phase3-evidence-packet.md": "phase3_evidence_packet_markdown",
    "launch-card-binding-worksheet.json": "phase3_launch_card_binding_worksheet",
    "model-plane-artifact-writer-contract-request.json": "model_plane_artifact_writer_contract_request",
    "recommended-runtime-capture-launch-card.template.json": "recommended_runtime_capture_launch_card_template",
    "recommended-runtime-capture-preflight.json": "recommended_runtime_capture_preflight_manifest",
    "recommended-runtime-capture-command-contract.json": "recommended_runtime_capture_command_contract",
    "recommended-runtime-capture-execution-coverage.json": "recommended_runtime_capture_execution_coverage_manifest",
    "all-request-runtime-capture-execution-coverage.json": "all_request_runtime_capture_execution_coverage_manifest",
    "runtime-capture-launch-card-directory.json": "runtime_capture_launch_card_directory_manifest",
    "reuse-evidence-capture-plan.json": "phase3_reuse_evidence_capture_plan",
    "manual-capture-runbook.json": "all_request_manual_capture_runbook_manifest",
    "post-capture-intake-runbook.json": "all_request_post_capture_intake_runbook_manifest",
    "recommended-manual-capture-runbook.json": "recommended_manual_capture_runbook_manifest",
    "recommended-post-capture-intake-runbook.json": "recommended_post_capture_intake_runbook_manifest",
    "recommended-runtime-capture-work-order.json": "recommended_runtime_capture_work_order_manifest",
    "recommended-runtime-capture-completion-receipt.template.json": "recommended_runtime_capture_completion_receipt_template",
    "next-unblocked-operator-handoff.json": "phase3_next_unblocked_operator_handoff",
    "capture-queue.json": "all_request_capture_queue_manifest",
    "approval-command-manifest.json": "all_request_approval_command_manifest",
    "validator-command-manifest.json": "all_request_validator_command_manifest",
    "receipt-fill-manifest.json": "all_request_receipt_fill_manifest",
    "receipt-fill-command-manifest.json": "all_request_receipt_fill_command_manifest",
    "downstream-handoff-manifest.json": "all_request_downstream_handoff_manifest",
    "blocker-closure-manifest.json": "phase3_blocker_closure_manifest",
    "blocker-evidence-ledger.json": "phase3_blocker_evidence_ledger_manifest",
    "blocker-resolution-queue.json": "phase3_blocker_resolution_queue_manifest",
}
RUNTIME_ACTUATOR_SPIKE_PACKAGE_ID = "runtime_actuator_spike"
RUNTIME_ACTUATOR_EXPECTED_PROOF_REQUIREMENT_IDS = [
    "expert_inventory",
    "routing_visibility",
    "residency_observation",
    "policy_application",
    "dense_fallback",
    "artifact_export",
    "residency_control",
    "cleanup_restore",
]
RUNTIME_ACTUATOR_CONTROL_BLOCKER_IDS = [
    "residency_observation",
    "residency_control",
    "cleanup_restore",
]
RUNTIME_ACTUATOR_CONTROL_DEPENDENCIES = {
    "residency_control": {"residency_observation", "dense_fallback", "artifact_export"},
    "cleanup_restore": {"residency_control", "dense_fallback", "artifact_export"},
}
APPROVAL_COMMAND_CLASS = "phase3_runtime_capture_request_approval_rebuild"
COMPLETION_RECEIPT_VALIDATION_COMMAND_CLASS = "phase3_capture_completion_receipt_validation"
COMPLETION_RECEIPT_VALIDATION_SCRIPT = "scripts/plan_phase3_capture_completion_receipt.py"
COMPLETION_RECEIPT_TEMPLATE_FILE = "recommended-runtime-capture-completion-receipt.template.json"
COMPLETION_RECEIPT_WORK_ORDER_FILE = "recommended-runtime-capture-work-order.json"
NEXT_UNBLOCKED_HANDOFF_FILE = "next-unblocked-operator-handoff.json"
NEXT_UNBLOCKED_HANDOFF_SCHEMA_VERSION = "moe-phase3-next-unblocked-operator-handoff-v1"
APPROVAL_KEY_FLAGS = {
    "runtime_prompt_traffic_approved": "--runtime-prompt-traffic-approved",
    "router_trace_capture_approved": "--router-trace-capture-approved",
    "managed_output_capture_approved": "--managed-output-capture-approved",
    "dense_output_capture_approved": "--dense-output-capture-approved",
}
REPO_PATH_KEYS = {
    "artifact_output_path",
    "artifact_path",
    "bundle_path",
    "candidate_trace_path",
    "dense_output_path",
    "inventory_path",
    "launch_card_path",
    "live_proof_template_path",
    "managed_output_path",
    "next_artifact_path",
    "path",
    "primary_path",
    "prompt_set_path",
    "receipt_output_path",
    "receipt_path",
    "request_path",
    "secondary_path",
    "source_bundle_path",
    "source_prompt_set_path",
    "source_request_path",
    "trace_path",
    "writes_artifact_path",
    "writes_receipt_path",
    "writes_request_path",
}

JSONDict = dict[str, Any]


def load_json(path: Path, errors: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(f"missing_json:{path.name}")
    except json.JSONDecodeError as exc:
        errors.append(f"invalid_json:{path.name}:{exc.msg}")
    return None


def list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def int_count(value: Any) -> int:
    return value if isinstance(value, int) else 0


def normalized_manifest_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().replace("\\", "/")


def manifest_output_files(manifest: JSONDict) -> dict[str, str]:
    result: dict[str, str] = {}
    output_files = manifest.get("output_files")
    if not isinstance(output_files, list):
        return result
    for item in output_files:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        kind = item.get("kind")
        if isinstance(path, str) and isinstance(kind, str):
            result[path] = kind
    return result


def path_stays_in_root(root: Path, relative_path: str) -> bool:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        return False
    try:
        (root / candidate).resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def path_stays_in_repo(relative_path: str) -> bool:
    if "://" in relative_path:
        return False
    candidate = Path(relative_path)
    if candidate.is_absolute():
        return False
    try:
        (ROOT / candidate).resolve().relative_to(ROOT.resolve())
    except ValueError:
        return False
    return True


def collect_repo_path_fields(value: Any, *, prefix: str) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            item_prefix = f"{prefix}.{key}" if prefix else str(key)
            if key in REPO_PATH_KEYS and isinstance(item, str) and item.strip():
                fields.append((item_prefix, item))
            fields.extend(collect_repo_path_fields(item, prefix=item_prefix))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            fields.extend(collect_repo_path_fields(item, prefix=f"{prefix}[{index}]"))
    return fields


def repo_path_safety_errors(surfaces: list[tuple[str, Any]]) -> tuple[list[str], int]:
    errors: list[str] = []
    count = 0
    for name, surface in surfaces:
        for field, value in collect_repo_path_fields(surface, prefix=name):
            count += 1
            if not path_stays_in_repo(value):
                errors.append(f"repo_path_unsafe:{field}")
    return errors, count


def command_contract_path_safety_surface(command_contract: Any) -> Any:
    if not isinstance(command_contract, dict):
        return command_contract
    result = json.loads(json.dumps(command_contract))
    binding_summary = result.get("launch_card_binding_summary")
    if isinstance(binding_summary, dict):
        binding_summary.pop("launch_card_path", None)
    return result


def all_request_execution_path_safety_surface(execution: Any) -> Any:
    if not isinstance(execution, dict):
        return execution
    result = dict(execution)
    result.pop("launch_card_directory", None)
    return result


def count_validator_commands(manifest: Any) -> int:
    if not isinstance(manifest, list):
        return 0
    total = 0
    for item in manifest:
        if not isinstance(item, dict):
            continue
        commands = item.get("validator_commands")
        if isinstance(commands, list):
            command_count = len([command for command in commands if isinstance(command, (list, str))])
            declared_count = item.get("validator_command_count")
            total += declared_count if isinstance(declared_count, int) and declared_count == command_count else command_count
    return total


def ready_receipt_count(manifest: Any) -> int:
    if not isinstance(manifest, list):
        return 0
    return sum(1 for item in manifest if isinstance(item, dict) and item.get("receipt_ready") is True)


def ready_after_approval_count(manifest: Any) -> int:
    if not isinstance(manifest, list):
        return 0
    return sum(1 for item in manifest if isinstance(item, dict) and item.get("ready_after_approval") is True)

RECEIPT_FILL_ARTIFACT_COUNT_KEYS = {
    "candidate_router_trace": "candidate_router_trace",
    "managed_output_summary_fill": "managed_output",
    "dense_output_summary_fill": "dense_output",
}


def receipt_fill_artifact_counts(manifest: Any) -> dict[str, JSONDict]:
    counts: dict[str, JSONDict] = {
        artifact_id: {"entry_count": 0, "ready_count": 0, "missing_count": 0}
        for artifact_id in RECEIPT_FILL_ARTIFACT_COUNT_KEYS
    }
    if not isinstance(manifest, list):
        return counts
    for item in manifest:
        if not isinstance(item, dict):
            continue
        artifact_id = item.get("artifact_id")
        if artifact_id not in counts:
            continue
        counts[artifact_id]["entry_count"] += 1
        if item.get("ready_after_approval") is True:
            counts[artifact_id]["ready_count"] += 1
        else:
            counts[artifact_id]["missing_count"] += 1
    return counts


def receipt_fill_artifact_count_errors(counts: JSONDict, artifact_counts: dict[str, JSONDict]) -> list[str]:
    errors: list[str] = []
    for artifact_id, count_key_prefix in RECEIPT_FILL_ARTIFACT_COUNT_KEYS.items():
        actual = artifact_counts.get(artifact_id, {})
        for metric in ("entry_count", "ready_count", "missing_count"):
            handoff_key = f"receipt_fill_{count_key_prefix}_{metric}"
            if handoff_key in counts and int_count(counts.get(handoff_key)) != int_count(actual.get(metric)):
                errors.append(f"{handoff_key}_mismatch")
    return errors


def artifact_key(item: JSONDict) -> str | None:
    command_key = item.get("command_manifest_key")
    if isinstance(command_key, str) and "::" in command_key:
        return command_key
    request_path = item.get("request_path")
    artifact_id = item.get("artifact_id")
    if isinstance(request_path, str) and request_path and isinstance(artifact_id, str) and artifact_id:
        return f"{request_path}::{artifact_id}"
    return None


def artifact_keys(manifest: Any) -> set[str]:
    if not isinstance(manifest, list):
        return set()
    keys: set[str] = set()
    for item in manifest:
        if not isinstance(item, dict):
            continue
        key = artifact_key(item)
        if key is not None:
            keys.add(key)
    return keys


def capture_queue_artifact_keys(queue: Any) -> set[str]:
    if not isinstance(queue, list):
        return set()
    keys: set[str] = set()
    for request in queue:
        if not isinstance(request, dict):
            continue
        request_path = request.get("request_path")
        if not isinstance(request_path, str) or not request_path:
            continue
        steps = request.get("capture_fill_steps")
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            artifact_id = step.get("artifact_id")
            if isinstance(artifact_id, str) and artifact_id:
                keys.add(f"{request_path}::{artifact_id}")
    return keys


def validator_commands_for_item(item: JSONDict) -> list[Any]:
    commands = item.get("validator_commands")
    return commands if isinstance(commands, list) else []


def validator_command_count_for_item(item: JSONDict) -> int:
    commands = validator_commands_for_item(item)
    declared_count = item.get("validator_command_count")
    actual_count = len([command for command in commands if isinstance(command, (list, str))])
    if isinstance(declared_count, int) and declared_count == actual_count:
        return declared_count
    return actual_count


def validator_command_count_by_artifact_key(manifest: Any) -> dict[str, int]:
    if not isinstance(manifest, list):
        return {}
    counts: dict[str, int] = {}
    for item in manifest:
        if not isinstance(item, dict):
            continue
        key = artifact_key(item)
        if key is not None:
            counts[key] = validator_command_count_for_item(item)
    return counts


def capture_queue_validator_count_by_artifact_key(queue: Any) -> dict[str, int]:
    if not isinstance(queue, list):
        return {}
    counts: dict[str, int] = {}
    for request in queue:
        if not isinstance(request, dict):
            continue
        request_path = request.get("request_path")
        steps = request.get("capture_fill_steps")
        if not isinstance(request_path, str) or not request_path or not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            artifact_id = step.get("artifact_id")
            if isinstance(artifact_id, str) and artifact_id:
                counts[f"{request_path}::{artifact_id}"] = int_count(step.get("validator_command_count"))
    return counts


def validator_manifest_stage_items(manifest: Any, stage: str) -> list[JSONDict]:
    if not isinstance(manifest, list):
        return []
    return [item for item in manifest if isinstance(item, dict) and item.get("artifact_stage") == stage]


def request_paths_from_capture_queue(queue: Any) -> set[str]:
    if not isinstance(queue, list):
        return set()
    paths: set[str] = set()
    for item in queue:
        if not isinstance(item, dict):
            continue
        request_path = item.get("request_path")
        if isinstance(request_path, str) and request_path:
            paths.add(request_path)
    return paths


def validator_manifest_coverage_errors(
    validator_manifest: Any,
    receipt_command_manifest: Any,
    capture_queue: Any,
) -> tuple[list[str], int, int, int, int, int, int]:
    errors: list[str] = []
    runtime_items = validator_manifest_stage_items(validator_manifest, "runtime_capture")
    future_items = validator_manifest_stage_items(validator_manifest, "future_adapter")
    runtime_counts = validator_command_count_by_artifact_key(runtime_items)
    future_counts = validator_command_count_by_artifact_key(future_items)
    receipt_counts = validator_command_count_by_artifact_key(receipt_command_manifest)
    capture_counts = capture_queue_validator_count_by_artifact_key(capture_queue)
    runtime_missing_validator_count = 0
    future_missing_validator_count = 0

    for key, count in runtime_counts.items():
        if count == 0:
            runtime_missing_validator_count += 1
            errors.append(f"validator_manifest_runtime_commands_missing:{key}")
        if key in receipt_counts and receipt_counts[key] != count:
            errors.append(f"validator_manifest_receipt_command_count_mismatch:{key}")
        if key in capture_counts and capture_counts[key] != count:
            errors.append(f"validator_manifest_capture_queue_count_mismatch:{key}")
    for key in sorted(set(receipt_counts) - set(runtime_counts)):
        errors.append(f"validator_manifest_runtime_entry_missing_for_receipt:{key}")
    for key in sorted(set(capture_counts) - set(runtime_counts)):
        errors.append(f"validator_manifest_runtime_entry_missing_for_capture_queue:{key}")
    for key in sorted(set(runtime_counts) - set(capture_counts)):
        errors.append(f"validator_manifest_runtime_entry_extra:{key}")

    capture_request_paths = request_paths_from_capture_queue(capture_queue)
    future_request_paths: set[str] = set()
    for item in future_items:
        request_path = item.get("request_path")
        key = artifact_key(item) or str(request_path or "unknown_future")
        if isinstance(request_path, str) and request_path:
            if request_path in future_request_paths:
                errors.append(f"validator_manifest_future_duplicate_request:{request_path}")
            future_request_paths.add(request_path)
        if item.get("artifact_id") != "live_capability_proof_fill":
            errors.append(f"validator_manifest_future_unexpected_artifact:{key}")
        if item.get("status") != "future_adapter_required":
            errors.append(f"validator_manifest_future_status_mismatch:{key}")
        if validator_command_count_for_item(item) == 0:
            future_missing_validator_count += 1
            errors.append(f"validator_manifest_future_commands_missing:{key}")
    for request_path in sorted(capture_request_paths - future_request_paths):
        errors.append(f"validator_manifest_future_entry_missing_for_request:{request_path}")
    for request_path in sorted(future_request_paths - capture_request_paths):
        errors.append(f"validator_manifest_future_entry_extra:{request_path}")

    runtime_validator_count = sum(runtime_counts.values())
    future_validator_count = sum(future_counts.values())
    return (
        errors,
        len(runtime_counts),
        len(future_counts),
        runtime_validator_count,
        future_validator_count,
        runtime_missing_validator_count,
        future_missing_validator_count,
    )

def receipt_command_validator_errors(receipt_command_manifest: Any, capture_queue: Any) -> tuple[list[str], int, int, int, int]:
    errors: list[str] = []
    if not isinstance(receipt_command_manifest, list):
        return errors, 0, 0, 0, 0
    receipt_counts = validator_command_count_by_artifact_key(receipt_command_manifest)
    capture_counts = capture_queue_validator_count_by_artifact_key(capture_queue)
    missing_command_count = 0
    declared_count_mismatch_count = 0
    for index, item in enumerate(receipt_command_manifest):
        if not isinstance(item, dict):
            continue
        key = artifact_key(item) or f"index_{index}"
        commands = validator_commands_for_item(item)
        actual_count = len([command for command in commands if isinstance(command, (list, str))])
        declared_count = item.get("validator_command_count")
        if isinstance(declared_count, int) and declared_count != actual_count:
            declared_count_mismatch_count += 1
            errors.append(f"receipt_fill_command_validator_declared_count_mismatch:{key}")
        if actual_count == 0:
            missing_command_count += 1
            errors.append(f"receipt_fill_command_validator_commands_missing:{key}")
        expected_count = capture_counts.get(key)
        if expected_count is not None and expected_count != actual_count:
            errors.append(f"receipt_fill_command_capture_queue_validator_count_mismatch:{key}")
    for key in sorted(set(capture_counts) - set(receipt_counts)):
        errors.append(f"receipt_fill_command_validator_count_missing_for_capture_queue:{key}")
    total_receipt_validator_count = sum(receipt_counts.values())
    total_capture_validator_count = sum(capture_counts.values())
    return errors, total_receipt_validator_count, total_capture_validator_count, missing_command_count, declared_count_mismatch_count



def blocker_closure_manifest_errors(manifest: Any, remaining_blockers: list[str], remaining_gaps: JSONDict) -> tuple[list[str], JSONDict]:
    errors: list[str] = []
    counts: JSONDict = {
        "reason_id_count": 0,
        "unmapped_count": 0,
        "missing_action_count": 0,
        "missing_primary_path_count": 0,
        "runtime_capture_required_count": 0,
        "future_adapter_required_count": 0,
        "validator_command_count": 0,
        "missing_evidence_count": 0,
    }
    if not isinstance(manifest, list):
        errors.append("blocker_closure_manifest_not_list")
        return errors, counts

    allowed_states = {"blocked", "closed", "future_adapter_required", "metadata_build_required", "runtime_capture_required"}
    allowed_stages = {"capture_receipt_required", "future_adapter_required", "runtime_capture_required", "saved_output_required"}
    reason_ids: list[str] = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    for index, item in enumerate(manifest):
        if not isinstance(item, dict):
            errors.append(f"blocker_closure_entry_not_object:index_{index}")
            continue
        reason_id = item.get("reason_id")
        key = reason_id if isinstance(reason_id, str) and reason_id else f"index_{index}"
        if not isinstance(reason_id, str) or not reason_id:
            errors.append(f"blocker_closure_reason_id_missing:{key}")
        else:
            reason_ids.append(reason_id)
            if reason_id in seen:
                duplicates.add(reason_id)
            seen.add(reason_id)
        if item.get("mapped") is not True:
            counts["unmapped_count"] += 1
            errors.append(f"blocker_closure_unmapped:{key}")
        closure_gate = item.get("closure_gate")
        if not isinstance(closure_gate, str) or not closure_gate or closure_gate == "unmapped_no_go_reason":
            errors.append(f"blocker_closure_gate_missing:{key}")
        closure_class = item.get("closure_class")
        if not isinstance(closure_class, str) or not closure_class or closure_class == "unknown":
            errors.append(f"blocker_closure_class_missing:{key}")
        closure_state = item.get("closure_state")
        if closure_state not in allowed_states:
            errors.append(f"blocker_closure_state_invalid:{key}")
        approval_stage = item.get("approval_stage")
        if approval_stage not in allowed_stages:
            errors.append(f"blocker_closure_approval_stage_invalid:{key}")
        next_action = item.get("next_action")
        if not isinstance(next_action, str) or not next_action.strip():
            counts["missing_action_count"] += 1
            errors.append(f"blocker_closure_next_action_missing:{key}")
        primary_path = item.get("primary_path")
        if not isinstance(primary_path, str) or not primary_path.strip():
            counts["missing_primary_path_count"] += 1
            errors.append(f"blocker_closure_primary_path_missing:{key}")
        for count_key in ("pending_artifact_count", "ready_evidence_count", "missing_evidence_count", "validator_command_count"):
            if not isinstance(item.get(count_key), int) or item.get(count_key) < 0:
                errors.append(f"blocker_closure_{count_key}_invalid:{key}")
        if approval_stage == "runtime_capture_required":
            counts["runtime_capture_required_count"] += 1
        if approval_stage == "future_adapter_required":
            counts["future_adapter_required_count"] += 1
        validator_count = int_count(item.get("validator_command_count"))
        counts["validator_command_count"] += validator_count
        counts["missing_evidence_count"] += int_count(item.get("missing_evidence_count"))
        if item.get("closure_class") != "runtime_actuator" and validator_count == 0:
            errors.append(f"blocker_closure_validator_commands_missing:{key}")

    counts["reason_id_count"] = len(reason_ids)
    if duplicates:
        errors.append("blocker_closure_duplicate_reason_ids")
    if set(remaining_blockers) != set(reason_ids):
        errors.append("blocker_closure_remaining_blocker_mismatch")

    gap_to_count = {
        "phase3_blocker_closure_reason_count": "reason_id_count",
        "phase3_blocker_closure_mapped_reason_count": "reason_id_count",
        "phase3_blocker_closure_unmapped_reason_count": "unmapped_count",
        "phase3_blocker_closure_runtime_capture_required_count": "runtime_capture_required_count",
        "phase3_blocker_closure_future_adapter_required_count": "future_adapter_required_count",
        "phase3_blocker_closure_validator_command_count": "validator_command_count",
        "phase3_blocker_closure_missing_evidence_count": "missing_evidence_count",
    }
    for gap_key, count_key in gap_to_count.items():
        expected = remaining_gaps.get(gap_key)
        if isinstance(expected, int) and expected != counts[count_key]:
            errors.append(f"blocker_closure_{gap_key}_mismatch")
    return errors, counts




def blocker_evidence_ledger_errors(
    ledger: Any,
    packet: JSONDict,
    blocker_manifest: Any,
    remaining_gaps: JSONDict,
    handoff_counts: JSONDict,
) -> tuple[list[str], JSONDict]:
    errors: list[str] = []
    counts: JSONDict = {
        "ready": False,
        "reason_count": 0,
        "row_count": 0,
        "expected_missing_evidence_count": 0,
        "runtime_capture_required_row_count": 0,
        "future_adapter_required_row_count": 0,
        "receipt_bound_row_count": 0,
        "dense_output_row_count": 0,
        "live_proof_row_count": 0,
        "runtime_actuator_row_count": 0,
        "validator_command_count": 0,
        "missing_item_count": 0,
    }
    if not isinstance(ledger, dict):
        errors.append("blocker_evidence_ledger_not_object")
        return errors, counts

    packet_ledger = packet.get("phase3_blocker_evidence_ledger_manifest") if isinstance(packet, dict) else None
    if packet_ledger != ledger:
        errors.append("blocker_evidence_ledger_packet_mismatch")

    if ledger.get("ready") is not True:
        errors.append("blocker_evidence_ledger_not_ready")
    counts["ready"] = ledger.get("ready") is True

    rows = ledger.get("evidence_rows")
    if not isinstance(rows, list):
        errors.append("blocker_evidence_ledger_rows_not_list")
        rows = []
    blocker_rows = blocker_manifest if isinstance(blocker_manifest, list) else []
    reasons = {
        item.get("reason_id"): item
        for item in blocker_rows
        if isinstance(item, dict) and isinstance(item.get("reason_id"), str) and item.get("reason_id")
    }
    rows_by_reason: dict[str, int] = {}
    validator_sum = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"blocker_evidence_ledger_row_not_object:index_{index}")
            continue
        reason_id = row.get("reason_id")
        key = reason_id if isinstance(reason_id, str) and reason_id else f"index_{index}"
        if reason_id not in reasons:
            errors.append(f"blocker_evidence_ledger_unknown_reason:{key}")
        else:
            rows_by_reason[reason_id] = rows_by_reason.get(reason_id, 0) + 1
        if not isinstance(row.get("evidence_kind"), str) or not row.get("evidence_kind"):
            errors.append(f"blocker_evidence_ledger_kind_missing:{key}")
        if not isinstance(row.get("evidence_id"), str) or not row.get("evidence_id"):
            errors.append(f"blocker_evidence_ledger_evidence_id_missing:{key}")
        if not isinstance(row.get("status"), str) or row.get("status") != "missing":
            errors.append(f"blocker_evidence_ledger_status_invalid:{key}")
        if not isinstance(row.get("validator_command_count"), int) or row.get("validator_command_count") < 0:
            errors.append(f"blocker_evidence_ledger_validator_count_invalid:{key}")
        validator_sum += int_count(row.get("validator_command_count"))

    expected_missing = sum(int_count(item.get("missing_evidence_count")) for item in blocker_rows if isinstance(item, dict))
    counts["reason_count"] = len(reasons)
    counts["row_count"] = len(rows)
    counts["expected_missing_evidence_count"] = expected_missing
    counts["runtime_capture_required_row_count"] = sum(1 for row in rows if isinstance(row, dict) and row.get("approval_stage") == "runtime_capture_required")
    counts["future_adapter_required_row_count"] = sum(1 for row in rows if isinstance(row, dict) and row.get("approval_stage") == "future_adapter_required")
    counts["receipt_bound_row_count"] = sum(1 for row in rows if isinstance(row, dict) and row.get("evidence_kind") == "runtime_capture_receipt_gate")
    counts["dense_output_row_count"] = sum(1 for row in rows if isinstance(row, dict) and str(row.get("evidence_kind") or "").endswith("_fallback_output_row"))
    counts["live_proof_row_count"] = sum(1 for row in rows if isinstance(row, dict) and row.get("evidence_kind") == "live_capability_proof_blocker")
    counts["runtime_actuator_row_count"] = sum(1 for row in rows if isinstance(row, dict) and row.get("evidence_kind") == "runtime_actuator_capability")
    counts["validator_command_count"] = sum(int_count(item.get("validator_command_count")) for item in blocker_rows if isinstance(item, dict))
    counts["missing_item_count"] = int_count(ledger.get("missing_item_count"))

    if int_count(ledger.get("reason_count")) != counts["reason_count"]:
        errors.append("blocker_evidence_ledger_reason_count_mismatch")
    if int_count(ledger.get("evidence_row_count")) != counts["row_count"]:
        errors.append("blocker_evidence_ledger_row_count_mismatch")
    if counts["row_count"] != expected_missing:
        errors.append("blocker_evidence_ledger_expected_row_count_mismatch")
    if int_count(ledger.get("expected_missing_evidence_count")) != expected_missing:
        errors.append("blocker_evidence_ledger_expected_missing_count_mismatch")
    if int_count(ledger.get("validator_command_count")) != counts["validator_command_count"]:
        errors.append("blocker_evidence_ledger_validator_count_mismatch")
    if int_count(ledger.get("missing_item_count")) != len(ledger.get("missing_items") if isinstance(ledger.get("missing_items"), list) else []):
        errors.append("blocker_evidence_ledger_missing_item_count_mismatch")
    if counts["missing_item_count"] != 0:
        errors.append("blocker_evidence_ledger_missing_items_present")
    if validator_sum > counts["validator_command_count"]:
        errors.append("blocker_evidence_ledger_row_validator_count_exceeds_closure")

    ledger_count_map = {
        "runtime_capture_required_row_count": "runtime_capture_required_row_count",
        "future_adapter_required_row_count": "future_adapter_required_row_count",
        "receipt_bound_row_count": "receipt_bound_row_count",
        "dense_output_row_count": "dense_output_row_count",
        "live_proof_row_count": "live_proof_row_count",
        "runtime_actuator_row_count": "runtime_actuator_row_count",
    }
    for ledger_key, count_key in ledger_count_map.items():
        if int_count(ledger.get(ledger_key)) != counts[count_key]:
            errors.append(f"blocker_evidence_ledger_{ledger_key}_mismatch")

    for reason_id, reason in reasons.items():
        expected = int_count(reason.get("missing_evidence_count"))
        observed = rows_by_reason.get(reason_id, 0)
        if observed != expected:
            errors.append(f"blocker_evidence_ledger_reason_row_count_mismatch:{reason_id}:{observed}/{expected}")
    declared_rows_by_reason = ledger.get("rows_by_reason")
    if isinstance(declared_rows_by_reason, dict):
        for reason_id, observed in rows_by_reason.items():
            if int_count(declared_rows_by_reason.get(reason_id)) != observed:
                errors.append(f"blocker_evidence_ledger_declared_reason_count_mismatch:{reason_id}")
    else:
        errors.append("blocker_evidence_ledger_rows_by_reason_missing")

    gap_to_count = {
        "phase3_blocker_evidence_ledger_ready": "ready",
        "phase3_blocker_evidence_ledger_reason_count": "reason_count",
        "phase3_blocker_evidence_ledger_row_count": "row_count",
        "phase3_blocker_evidence_ledger_expected_missing_evidence_count": "expected_missing_evidence_count",
        "phase3_blocker_evidence_ledger_runtime_capture_required_row_count": "runtime_capture_required_row_count",
        "phase3_blocker_evidence_ledger_future_adapter_required_row_count": "future_adapter_required_row_count",
        "phase3_blocker_evidence_ledger_receipt_bound_row_count": "receipt_bound_row_count",
        "phase3_blocker_evidence_ledger_dense_output_row_count": "dense_output_row_count",
        "phase3_blocker_evidence_ledger_live_proof_row_count": "live_proof_row_count",
        "phase3_blocker_evidence_ledger_runtime_actuator_row_count": "runtime_actuator_row_count",
        "phase3_blocker_evidence_ledger_validator_command_count": "validator_command_count",
        "phase3_blocker_evidence_ledger_missing_item_count": "missing_item_count",
    }
    for gap_key, count_key in gap_to_count.items():
        expected = remaining_gaps.get(gap_key)
        if expected != counts[count_key]:
            errors.append(f"blocker_evidence_ledger_{gap_key}_mismatch")

    handoff_to_count = {
        "blocker_evidence_ledger_ready": "ready",
        "blocker_evidence_ledger_reason_count": "reason_count",
        "blocker_evidence_ledger_row_count": "row_count",
        "blocker_evidence_ledger_expected_missing_evidence_count": "expected_missing_evidence_count",
        "blocker_evidence_ledger_runtime_capture_required_row_count": "runtime_capture_required_row_count",
        "blocker_evidence_ledger_future_adapter_required_row_count": "future_adapter_required_row_count",
        "blocker_evidence_ledger_receipt_bound_row_count": "receipt_bound_row_count",
        "blocker_evidence_ledger_dense_output_row_count": "dense_output_row_count",
        "blocker_evidence_ledger_live_proof_row_count": "live_proof_row_count",
        "blocker_evidence_ledger_runtime_actuator_row_count": "runtime_actuator_row_count",
        "blocker_evidence_ledger_validator_command_count": "validator_command_count",
        "blocker_evidence_ledger_missing_item_count": "missing_item_count",
    }
    for handoff_key, count_key in handoff_to_count.items():
        expected = handoff_counts.get(handoff_key)
        if expected != counts[count_key]:
            errors.append(f"blocker_evidence_ledger_handoff_{handoff_key}_mismatch")
    return errors, counts



def runtime_actuator_proof_handoff_errors(package: JSONDict, key: str) -> tuple[list[str], JSONDict]:
    errors: list[str] = []
    counts: JSONDict = {
        "handoff_ready": False,
        "live_spike_ready": False,
        "proof_requirement_count": 0,
        "proof_requirement_id_count": 0,
        "proof_artifact_count": 0,
        "dependency_edge_count": 0,
        "blocking_capability_count": 0,
        "control_blocker_count": 0,
    }
    proof_handoff = package.get("proof_handoff")
    if not isinstance(proof_handoff, dict):
        errors.append(f"blocker_resolution_queue_runtime_actuator_proof_handoff_missing:{key}")
        return errors, counts

    counts["handoff_ready"] = proof_handoff.get("handoff_ready") is True
    counts["live_spike_ready"] = proof_handoff.get("live_spike_ready") is True
    if proof_handoff.get("handoff_ready") is not True:
        errors.append(f"blocker_resolution_queue_runtime_actuator_proof_handoff_not_ready:{key}")
    if proof_handoff.get("live_spike_ready") is True:
        errors.append(f"blocker_resolution_queue_runtime_actuator_live_spike_ready_too_early:{key}")
    elif proof_handoff.get("live_spike_ready") is not False:
        errors.append(f"blocker_resolution_queue_runtime_actuator_live_spike_ready_not_bool:{key}")
    if proof_handoff.get("backend_family") != "llama_cpp":
        errors.append(f"blocker_resolution_queue_runtime_actuator_backend_family_mismatch:{key}")

    proof_requirement_ids = list_of_strings(proof_handoff.get("proof_requirement_ids"))
    counts["proof_requirement_id_count"] = len(proof_requirement_ids)
    if proof_requirement_ids != RUNTIME_ACTUATOR_EXPECTED_PROOF_REQUIREMENT_IDS:
        errors.append(f"blocker_resolution_queue_runtime_actuator_proof_requirement_ids_mismatch:{key}")

    proof_requirements = proof_handoff.get("proof_requirements")
    if not isinstance(proof_requirements, list):
        errors.append(f"blocker_resolution_queue_runtime_actuator_proof_requirements_not_list:{key}")
        proof_requirements = []

    requirement_ids: list[str] = []
    proof_artifact_count = 0
    dependency_edge_count = 0
    for index, requirement in enumerate(proof_requirements):
        if not isinstance(requirement, dict):
            errors.append(f"blocker_resolution_queue_runtime_actuator_proof_requirement_not_object:{key}:index_{index}")
            continue
        capability_id = requirement.get("capability_id")
        item_key = capability_id if isinstance(capability_id, str) and capability_id else f"index_{index}"
        if not isinstance(capability_id, str) or not capability_id:
            errors.append(f"blocker_resolution_queue_runtime_actuator_proof_requirement_id_missing:{key}:{item_key}")
        else:
            requirement_ids.append(capability_id)
        if not isinstance(requirement.get("current_status"), str) or not requirement.get("current_status"):
            errors.append(f"blocker_resolution_queue_runtime_actuator_proof_requirement_status_missing:{key}:{item_key}")
        if not isinstance(requirement.get("spike_stage"), str) or not requirement.get("spike_stage"):
            errors.append(f"blocker_resolution_queue_runtime_actuator_proof_requirement_stage_missing:{key}:{item_key}")
        if not isinstance(requirement.get("completion_gate"), str) or not requirement.get("completion_gate"):
            errors.append(f"blocker_resolution_queue_runtime_actuator_proof_requirement_gate_missing:{key}:{item_key}")
        if requirement.get("live_ready") is not True and requirement.get("live_ready") is not False:
            errors.append(f"blocker_resolution_queue_runtime_actuator_proof_requirement_live_ready_not_bool:{key}:{item_key}")

        artifact_value = requirement.get("proof_artifacts")
        proof_artifacts = list_of_strings(artifact_value)
        if not isinstance(artifact_value, list) or len(proof_artifacts) != len(artifact_value):
            errors.append(f"blocker_resolution_queue_runtime_actuator_proof_artifacts_invalid:{key}:{item_key}")
        if not proof_artifacts:
            errors.append(f"blocker_resolution_queue_runtime_actuator_proof_artifacts_missing:{key}:{item_key}")
        proof_artifact_count += len(proof_artifacts)
        if int_count(requirement.get("proof_artifact_count")) != len(proof_artifacts):
            errors.append(f"blocker_resolution_queue_runtime_actuator_proof_artifact_count_mismatch:{key}:{item_key}")

        dependency_value = requirement.get("dependency_ids")
        dependency_ids = list_of_strings(dependency_value)
        if not isinstance(dependency_value, list) or len(dependency_ids) != len(dependency_value):
            errors.append(f"blocker_resolution_queue_runtime_actuator_dependencies_invalid:{key}:{item_key}")
        dependency_edge_count += len(dependency_ids)
        if int_count(requirement.get("dependency_count")) != len(dependency_ids):
            errors.append(f"blocker_resolution_queue_runtime_actuator_dependency_count_mismatch:{key}:{item_key}")
        for dependency_id in dependency_ids:
            if dependency_id not in RUNTIME_ACTUATOR_EXPECTED_PROOF_REQUIREMENT_IDS:
                errors.append(f"blocker_resolution_queue_runtime_actuator_dependency_unknown:{key}:{item_key}:{dependency_id}")

        expected_control_dependencies = RUNTIME_ACTUATOR_CONTROL_DEPENDENCIES.get(str(capability_id))
        if expected_control_dependencies is not None:
            if set(dependency_ids) != expected_control_dependencies:
                errors.append(f"blocker_resolution_queue_runtime_actuator_control_dependencies_mismatch:{key}:{item_key}")
            if requirement.get("requires_explicit_runtime_approval_before_live") is not True:
                errors.append(f"blocker_resolution_queue_runtime_actuator_runtime_approval_missing:{key}:{item_key}")
        elif requirement.get("requires_explicit_runtime_approval_before_live") is not False:
            errors.append(f"blocker_resolution_queue_runtime_actuator_runtime_approval_flag_invalid:{key}:{item_key}")

    counts["proof_requirement_count"] = len(proof_requirements)
    counts["proof_artifact_count"] = proof_artifact_count
    counts["dependency_edge_count"] = dependency_edge_count
    if requirement_ids != RUNTIME_ACTUATOR_EXPECTED_PROOF_REQUIREMENT_IDS:
        errors.append(f"blocker_resolution_queue_runtime_actuator_proof_requirement_order_mismatch:{key}")
    if len(set(requirement_ids)) != len(requirement_ids):
        errors.append(f"blocker_resolution_queue_runtime_actuator_duplicate_proof_requirements:{key}")
    if int_count(proof_handoff.get("proof_requirement_count")) != len(proof_requirements):
        errors.append(f"blocker_resolution_queue_runtime_actuator_proof_requirement_count_mismatch:{key}")
    if int_count(proof_handoff.get("proof_artifact_count")) != proof_artifact_count:
        errors.append(f"blocker_resolution_queue_runtime_actuator_proof_artifact_total_mismatch:{key}")
    if int_count(proof_handoff.get("dependency_edge_count")) != dependency_edge_count:
        errors.append(f"blocker_resolution_queue_runtime_actuator_dependency_edge_total_mismatch:{key}")

    blocking_capabilities = list_of_strings(proof_handoff.get("blocking_capabilities"))
    counts["blocking_capability_count"] = len(blocking_capabilities)
    if int_count(proof_handoff.get("blocking_capability_count")) != len(blocking_capabilities):
        errors.append(f"blocker_resolution_queue_runtime_actuator_blocking_capability_count_mismatch:{key}")
    control_blockers = list_of_strings(proof_handoff.get("control_blockers"))
    counts["control_blocker_count"] = len(control_blockers)
    if control_blockers != RUNTIME_ACTUATOR_CONTROL_BLOCKER_IDS:
        errors.append(f"blocker_resolution_queue_runtime_actuator_control_blockers_mismatch:{key}")
    if int_count(proof_handoff.get("control_blocker_count")) != len(control_blockers):
        errors.append(f"blocker_resolution_queue_runtime_actuator_control_blocker_count_mismatch:{key}")
    implementation_sequence = list_of_strings(proof_handoff.get("implementation_sequence"))
    if not implementation_sequence:
        errors.append(f"blocker_resolution_queue_runtime_actuator_implementation_sequence_missing:{key}")
    return errors, counts


def blocker_resolution_queue_errors(
    queue: Any,
    packet: JSONDict,
    ledger: Any,
    remaining_gaps: JSONDict,
    handoff_counts: JSONDict,
) -> tuple[list[str], JSONDict]:
    errors: list[str] = []
    counts: JSONDict = {
        "ready": False,
        "work_package_count": 0,
        "queue_row_count": 0,
        "expected_ledger_row_count": 0,
        "runtime_capture_package_count": 0,
        "future_adapter_package_count": 0,
        "runtime_capture_required_row_count": 0,
        "future_adapter_required_row_count": 0,
        "receipt_bound_row_count": 0,
        "validator_command_count": 0,
        "completion_gate_count": 0,
        "dependency_edge_count": 0,
        "missing_item_count": 0,
        "runtime_actuator_proof_handoff_ready": False,
        "runtime_actuator_live_spike_ready": False,
        "runtime_actuator_proof_requirement_count": 0,
        "runtime_actuator_proof_requirement_id_count": 0,
        "runtime_actuator_proof_artifact_count": 0,
        "runtime_actuator_dependency_edge_count": 0,
        "runtime_actuator_blocking_capability_count": 0,
        "runtime_actuator_control_blocker_count": 0,
    }
    if not isinstance(queue, dict):
        errors.append("blocker_resolution_queue_not_object")
        return errors, counts
    packet_queue = packet.get("phase3_blocker_resolution_queue_manifest") if isinstance(packet, dict) else None
    if packet_queue != queue:
        errors.append("blocker_resolution_queue_packet_mismatch")
    if queue.get("ready") is not True:
        errors.append("blocker_resolution_queue_not_ready")
    counts["ready"] = queue.get("ready") is True

    packages = queue.get("work_packages")
    if not isinstance(packages, list):
        errors.append("blocker_resolution_queue_packages_not_list")
        packages = []
    package_ids: list[str] = []
    row_sum = 0
    runtime_package_count = 0
    future_package_count = 0
    runtime_rows = 0
    future_rows = 0
    receipt_rows = 0
    validator_count = 0
    completion_gate_count = 0
    dependency_edge_count = 0
    package_ranks: dict[str, int] = {}
    package_dependency_pairs: list[tuple[str, list[str]]] = []
    runtime_actuator_proof_counts: JSONDict = {}
    for index, package in enumerate(packages):
        if not isinstance(package, dict):
            errors.append(f"blocker_resolution_queue_package_not_object:index_{index}")
            continue
        package_id = package.get("work_package_id")
        key = package_id if isinstance(package_id, str) and package_id else f"index_{index}"
        if not isinstance(package_id, str) or not package_id:
            errors.append(f"blocker_resolution_queue_package_id_missing:{key}")
        else:
            package_ids.append(package_id)
            package_ranks[package_id] = int_count(package.get("sequence_rank"))
        dependencies = package.get("depends_on_work_package_ids")
        if not isinstance(dependencies, list):
            errors.append(f"blocker_resolution_queue_dependencies_not_list:{key}")
            dependencies = []
        elif any(not isinstance(dependency, str) or not dependency for dependency in dependencies):
            errors.append(f"blocker_resolution_queue_dependency_invalid:{key}")
        dependency_strings = [dependency for dependency in dependencies if isinstance(dependency, str) and dependency]
        dependency_edge_count += len(dependency_strings)
        if int_count(package.get("dependency_count")) != len(dependency_strings):
            errors.append(f"blocker_resolution_queue_dependency_count_mismatch:{key}")
        package_dependency_pairs.append((key, dependency_strings))
        if not isinstance(package.get("operator_stage"), str) or not package.get("operator_stage"):
            errors.append(f"blocker_resolution_queue_stage_missing:{key}")
        if not isinstance(package.get("next_action"), str) or not package.get("next_action"):
            errors.append(f"blocker_resolution_queue_next_action_missing:{key}")
        completion_gates = package.get("completion_gates")
        if not isinstance(completion_gates, list) or not completion_gates:
            errors.append(f"blocker_resolution_queue_completion_gates_missing:{key}")
            completion_gates = []
        elif any(not isinstance(gate, str) or not gate for gate in completion_gates):
            errors.append(f"blocker_resolution_queue_completion_gate_invalid:{key}")
        completion_gate_count += len(completion_gates)
        if int_count(package.get("completion_gate_count")) != len(completion_gates):
            errors.append(f"blocker_resolution_queue_completion_gate_count_mismatch:{key}")
        row_sum += int_count(package.get("row_count"))
        runtime_rows += int_count(package.get("runtime_capture_required_row_count"))
        future_rows += int_count(package.get("future_adapter_required_row_count"))
        receipt_rows += int_count(package.get("receipt_bound_row_count"))
        validator_count += int_count(package.get("validator_command_count"))
        if package.get("operator_stage") in {"approved_runtime_capture", "post_capture_receipt_intake", "fallback_quality_bounds"}:
            runtime_package_count += 1
        if package.get("operator_stage") in {"future_adapter_proof", "phase4_adapter_design"}:
            future_package_count += 1
        if package_id == RUNTIME_ACTUATOR_SPIKE_PACKAGE_ID:
            proof_errors, runtime_actuator_proof_counts = runtime_actuator_proof_handoff_errors(package, key)
            errors.extend(proof_errors)
    if len(set(package_ids)) != len(package_ids):
        errors.append("blocker_resolution_queue_duplicate_package_ids")
    for package_id, dependencies in package_dependency_pairs:
        package_rank = package_ranks.get(package_id, 0)
        for dependency_id in dependencies:
            dependency_rank = package_ranks.get(dependency_id)
            if dependency_rank is None:
                errors.append(f"blocker_resolution_queue_dependency_unknown:{package_id}:{dependency_id}")
            elif dependency_rank >= package_rank:
                errors.append(f"blocker_resolution_queue_dependency_order_invalid:{package_id}:{dependency_id}")

    ledger_dict = ledger if isinstance(ledger, dict) else {}
    expected_ledger_rows = int_count(ledger_dict.get("evidence_row_count"))
    counts.update(
        {
            "work_package_count": len(packages),
            "queue_row_count": row_sum,
            "expected_ledger_row_count": expected_ledger_rows,
            "runtime_capture_package_count": runtime_package_count,
            "future_adapter_package_count": future_package_count,
            "runtime_capture_required_row_count": runtime_rows,
            "future_adapter_required_row_count": future_rows,
            "receipt_bound_row_count": receipt_rows,
            "validator_command_count": validator_count,
            "completion_gate_count": completion_gate_count,
            "dependency_edge_count": dependency_edge_count,
            "missing_item_count": int_count(queue.get("missing_item_count")),
            "runtime_actuator_proof_handoff_ready": runtime_actuator_proof_counts.get("handoff_ready", False),
            "runtime_actuator_live_spike_ready": runtime_actuator_proof_counts.get("live_spike_ready", False),
            "runtime_actuator_proof_requirement_count": runtime_actuator_proof_counts.get("proof_requirement_count", 0),
            "runtime_actuator_proof_requirement_id_count": runtime_actuator_proof_counts.get("proof_requirement_id_count", 0),
            "runtime_actuator_proof_artifact_count": runtime_actuator_proof_counts.get("proof_artifact_count", 0),
            "runtime_actuator_dependency_edge_count": runtime_actuator_proof_counts.get("dependency_edge_count", 0),
            "runtime_actuator_blocking_capability_count": runtime_actuator_proof_counts.get("blocking_capability_count", 0),
            "runtime_actuator_control_blocker_count": runtime_actuator_proof_counts.get("control_blocker_count", 0),
        }
    )
    if row_sum != expected_ledger_rows:
        errors.append("blocker_resolution_queue_ledger_row_count_mismatch")
    queue_count_map = {
        "work_package_count": "work_package_count",
        "queue_row_count": "queue_row_count",
        "expected_ledger_row_count": "expected_ledger_row_count",
        "runtime_capture_package_count": "runtime_capture_package_count",
        "future_adapter_package_count": "future_adapter_package_count",
        "runtime_capture_required_row_count": "runtime_capture_required_row_count",
        "future_adapter_required_row_count": "future_adapter_required_row_count",
        "receipt_bound_row_count": "receipt_bound_row_count",
        "validator_command_count": "validator_command_count",
        "completion_gate_count": "completion_gate_count",
        "dependency_edge_count": "dependency_edge_count",
        "missing_item_count": "missing_item_count",
    }
    for queue_key, count_key in queue_count_map.items():
        if int_count(queue.get(queue_key)) != counts[count_key]:
            errors.append(f"blocker_resolution_queue_{queue_key}_mismatch")
    if counts["missing_item_count"] != 0:
        errors.append("blocker_resolution_queue_missing_items_present")

    gap_to_count = {
        "phase3_blocker_resolution_queue_ready": "ready",
        "phase3_blocker_resolution_queue_work_package_count": "work_package_count",
        "phase3_blocker_resolution_queue_row_count": "queue_row_count",
        "phase3_blocker_resolution_queue_expected_ledger_row_count": "expected_ledger_row_count",
        "phase3_blocker_resolution_queue_runtime_capture_package_count": "runtime_capture_package_count",
        "phase3_blocker_resolution_queue_future_adapter_package_count": "future_adapter_package_count",
        "phase3_blocker_resolution_queue_runtime_capture_required_row_count": "runtime_capture_required_row_count",
        "phase3_blocker_resolution_queue_future_adapter_required_row_count": "future_adapter_required_row_count",
        "phase3_blocker_resolution_queue_receipt_bound_row_count": "receipt_bound_row_count",
        "phase3_blocker_resolution_queue_validator_command_count": "validator_command_count",
        "phase3_blocker_resolution_queue_completion_gate_count": "completion_gate_count",
        "phase3_blocker_resolution_queue_dependency_edge_count": "dependency_edge_count",
        "phase3_blocker_resolution_queue_missing_item_count": "missing_item_count",
    }
    for gap_key, count_key in gap_to_count.items():
        if remaining_gaps.get(gap_key) != counts[count_key]:
            errors.append(f"blocker_resolution_queue_{gap_key}_mismatch")
    optional_gap_to_count = {
        "runtime_actuator_spike_handoff_ready": "runtime_actuator_proof_handoff_ready",
        "runtime_actuator_spike_live_ready": "runtime_actuator_live_spike_ready",
        "runtime_actuator_spike_proof_requirement_count": "runtime_actuator_proof_requirement_count",
        "runtime_actuator_spike_proof_artifact_count": "runtime_actuator_proof_artifact_count",
        "runtime_actuator_spike_dependency_edge_count": "runtime_actuator_dependency_edge_count",
        "runtime_actuator_spike_blocking_capability_count": "runtime_actuator_blocking_capability_count",
    }
    for gap_key, count_key in optional_gap_to_count.items():
        if gap_key in remaining_gaps:
            if remaining_gaps.get(gap_key) != counts[count_key]:
                errors.append(f"blocker_resolution_queue_{gap_key}_mismatch")
        elif counts["runtime_actuator_proof_requirement_count"]:
            errors.append(f"blocker_resolution_queue_{gap_key}_missing")

    handoff_to_count = {
        "blocker_resolution_queue_ready": "ready",
        "blocker_resolution_queue_work_package_count": "work_package_count",
        "blocker_resolution_queue_row_count": "queue_row_count",
        "blocker_resolution_queue_expected_ledger_row_count": "expected_ledger_row_count",
        "blocker_resolution_queue_runtime_capture_package_count": "runtime_capture_package_count",
        "blocker_resolution_queue_future_adapter_package_count": "future_adapter_package_count",
        "blocker_resolution_queue_runtime_capture_required_row_count": "runtime_capture_required_row_count",
        "blocker_resolution_queue_future_adapter_required_row_count": "future_adapter_required_row_count",
        "blocker_resolution_queue_receipt_bound_row_count": "receipt_bound_row_count",
        "blocker_resolution_queue_validator_command_count": "validator_command_count",
        "blocker_resolution_queue_completion_gate_count": "completion_gate_count",
        "blocker_resolution_queue_dependency_edge_count": "dependency_edge_count",
        "blocker_resolution_queue_missing_item_count": "missing_item_count",
    }
    for handoff_key, count_key in handoff_to_count.items():
        if handoff_counts.get(handoff_key) != counts[count_key]:
            errors.append(f"blocker_resolution_queue_handoff_{handoff_key}_mismatch")
    optional_handoff_to_count = {
        "blocker_resolution_queue_runtime_actuator_proof_handoff_ready": "runtime_actuator_proof_handoff_ready",
        "blocker_resolution_queue_runtime_actuator_live_spike_ready": "runtime_actuator_live_spike_ready",
        "blocker_resolution_queue_runtime_actuator_proof_requirement_count": "runtime_actuator_proof_requirement_count",
        "blocker_resolution_queue_runtime_actuator_proof_requirement_id_count": "runtime_actuator_proof_requirement_id_count",
        "blocker_resolution_queue_runtime_actuator_proof_artifact_count": "runtime_actuator_proof_artifact_count",
        "blocker_resolution_queue_runtime_actuator_dependency_edge_count": "runtime_actuator_dependency_edge_count",
        "blocker_resolution_queue_runtime_actuator_blocking_capability_count": "runtime_actuator_blocking_capability_count",
        "blocker_resolution_queue_runtime_actuator_control_blocker_count": "runtime_actuator_control_blocker_count",
    }
    for handoff_key, count_key in optional_handoff_to_count.items():
        if handoff_key in handoff_counts:
            if handoff_counts.get(handoff_key) != counts[count_key]:
                errors.append(f"blocker_resolution_queue_handoff_{handoff_key}_mismatch")
        elif counts["runtime_actuator_proof_requirement_count"]:
            errors.append(f"blocker_resolution_queue_handoff_{handoff_key}_missing")
    return errors, counts
def downstream_handoff_manifest_errors(manifest: Any, capture_queue: Any, remaining_gaps: JSONDict) -> tuple[list[str], JSONDict]:
    errors: list[str] = []
    counts: JSONDict = {
        "request_count": 0,
        "policy_ready_count": 0,
        "dense_ready_count": 0,
        "live_ready_count": 0,
        "all_ready_count": 0,
        "missing_section_count": 0,
        "missing_path_count": 0,
        "missing_validator_count": 0,
    }
    if not isinstance(manifest, list):
        errors.append("downstream_handoff_manifest_not_list")
        return errors, counts

    queue_paths = request_paths_from_capture_queue(capture_queue)
    request_paths: set[str] = set()
    for index, item in enumerate(manifest):
        if not isinstance(item, dict):
            errors.append(f"downstream_handoff_entry_not_object:index_{index}")
            continue
        request_path = item.get("request_path")
        key = request_path if isinstance(request_path, str) and request_path else f"index_{index}"
        if not isinstance(request_path, str) or not request_path:
            errors.append(f"downstream_handoff_request_path_missing:{key}")
        else:
            request_paths.add(request_path)
        for flag_name, count_key in (
            ("policy_candidate_trace_handoff_ready", "policy_ready_count"),
            ("dense_fallback_capture_handoff_ready", "dense_ready_count"),
            ("live_capability_proof_handoff_ready", "live_ready_count"),
            ("all_downstream_handoffs_ready", "all_ready_count"),
        ):
            if item.get(flag_name) is True:
                counts[count_key] += 1
            elif item.get(flag_name) is not False:
                errors.append(f"downstream_handoff_{flag_name}_not_bool:{key}")

        section_specs = {
            "policy_candidate_trace": (
                "policy_candidate_trace_handoff_ready",
                ("prompt_set_path", "candidate_trace_path", "candidate_trace_receipt_path"),
            ),
            "dense_fallback_capture": (
                "dense_fallback_capture_handoff_ready",
                ("prompt_set_path", "managed_output_path", "dense_output_path", "fallback_artifact_path"),
            ),
            "live_capability_proof": (
                "live_capability_proof_handoff_ready",
                ("proof_artifact_path",),
            ),
        }
        for section_name, (ready_flag, required_paths) in section_specs.items():
            section = item.get(section_name)
            if not isinstance(section, dict):
                counts["missing_section_count"] += 1
                errors.append(f"downstream_handoff_section_missing:{key}:{section_name}")
                continue
            if section.get("handoff_ready") != item.get(ready_flag):
                errors.append(f"downstream_handoff_section_ready_mismatch:{key}:{section_name}")
            for path_key in required_paths:
                value = section.get(path_key)
                if not isinstance(value, str) or not value.strip():
                    counts["missing_path_count"] += 1
                    errors.append(f"downstream_handoff_path_missing:{key}:{section_name}:{path_key}")
            if int_count(section.get("validator_command_count")) == 0:
                counts["missing_validator_count"] += 1
                errors.append(f"downstream_handoff_validator_commands_missing:{key}:{section_name}")

    counts["request_count"] = len(request_paths)
    if queue_paths and request_paths != queue_paths:
        errors.append("downstream_handoff_capture_queue_request_mismatch")

    gap_to_count = {
        "all_request_downstream_handoff_manifest_request_count": "request_count",
        "all_request_downstream_handoff_policy_ready_count": "policy_ready_count",
        "all_request_downstream_handoff_dense_ready_count": "dense_ready_count",
        "all_request_downstream_handoff_live_ready_count": "live_ready_count",
        "all_request_downstream_handoff_all_ready_count": "all_ready_count",
    }
    for gap_key, count_key in gap_to_count.items():
        expected = remaining_gaps.get(gap_key)
        if isinstance(expected, int) and expected != counts[count_key]:
            errors.append(f"downstream_handoff_{gap_key}_mismatch")
    expected_ready = remaining_gaps.get("all_request_downstream_handoff_manifest_ready")
    if isinstance(expected_ready, bool):
        actual_ready = bool(request_paths) and counts["all_ready_count"] == len(request_paths)
        if actual_ready != expected_ready:
            errors.append("downstream_handoff_ready_mismatch")
    return errors, counts
def duplicate_artifact_keys(manifest: Any) -> list[str]:
    if not isinstance(manifest, list):
        return []
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in manifest:
        if not isinstance(item, dict):
            continue
        key = artifact_key(item)
        if key is None:
            continue
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    return sorted(duplicates)


def command_tokens(value: Any) -> list[str]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    if isinstance(value, str):
        try:
            return shlex.split(value)
        except ValueError:
            return []
    return []


def first_capture_queue_item(queue: Any) -> JSONDict:
    if not isinstance(queue, list) or not queue:
        return {}
    first = queue[0]
    return first if isinstance(first, dict) else {}


def sorted_string_tuple(value: Any) -> tuple[str, ...]:
    return tuple(sorted(list_of_strings(value)))


def launch_card_task_keys(card: JSONDict) -> set[tuple[str, str, str, str, str, tuple[str, ...]]]:
    request_path = card.get("request_path")
    if not isinstance(request_path, str) or not request_path:
        return set()
    tasks = card.get("tasks")
    if not isinstance(tasks, list):
        return set()
    keys: set[tuple[str, str, str, str, str, tuple[str, ...]]] = set()
    for task in tasks:
        if not isinstance(task, dict):
            continue
        artifact_id = task.get("artifact_id")
        artifact_path = task.get("artifact_output_path")
        receipt_path = task.get("receipt_output_path")
        prompt_set_path = task.get("prompt_set_path")
        if all(isinstance(value, str) and value for value in (artifact_id, artifact_path, receipt_path, prompt_set_path)):
            keys.add((request_path, artifact_id, artifact_path, receipt_path, prompt_set_path, sorted_string_tuple(task.get("approval_keys"))))
    return keys


def worksheet_task_keys(
    worksheet: JSONDict,
    *,
    request_path: Any,
    launch_card_path: Any,
) -> set[tuple[str, str, str, str, str, tuple[str, ...]]]:
    if not isinstance(request_path, str) or not request_path:
        return set()
    if not isinstance(launch_card_path, str) or not launch_card_path:
        return set()
    tasks = worksheet.get("tasks")
    if not isinstance(tasks, list):
        return set()
    keys: set[tuple[str, str, str, str, str, tuple[str, ...]]] = set()
    for task in tasks:
        if not isinstance(task, dict):
            continue
        if task.get("request_path") != request_path or task.get("launch_card_path") != launch_card_path:
            continue
        artifact_id = task.get("artifact_id")
        artifact_path = task.get("artifact_output_path")
        receipt_path = task.get("receipt_output_path")
        prompt_set_path = task.get("prompt_set_path")
        if all(isinstance(value, str) and value for value in (artifact_id, artifact_path, receipt_path, prompt_set_path)):
            keys.add((request_path, artifact_id, artifact_path, receipt_path, prompt_set_path, sorted_string_tuple(task.get("approval_keys"))))
    return keys


def launch_card_queue_task_keys(card: JSONDict) -> set[tuple[str, str, str, str]]:
    request_path = card.get("request_path")
    if not isinstance(request_path, str) or not request_path:
        return set()
    tasks = card.get("tasks")
    if not isinstance(tasks, list):
        return set()
    keys: set[tuple[str, str, str, str]] = set()
    for task in tasks:
        if not isinstance(task, dict):
            continue
        artifact_id = task.get("artifact_id")
        artifact_path = task.get("artifact_output_path")
        receipt_path = task.get("receipt_output_path")
        if all(isinstance(value, str) and value for value in (artifact_id, artifact_path, receipt_path)):
            keys.add((request_path, artifact_id, artifact_path, receipt_path))
    return keys


def capture_queue_task_keys(queue_item: JSONDict) -> set[tuple[str, str, str, str]]:
    request_path = queue_item.get("request_path")
    if not isinstance(request_path, str) or not request_path:
        return set()
    steps = queue_item.get("capture_fill_steps")
    if not isinstance(steps, list):
        return set()
    keys: set[tuple[str, str, str, str]] = set()
    for step in steps:
        if not isinstance(step, dict):
            continue
        artifact_id = step.get("artifact_id")
        artifact_path = step.get("artifact_path")
        receipt_path = step.get("receipt_path")
        if all(isinstance(value, str) and value for value in (artifact_id, artifact_path, receipt_path)):
            keys.add((request_path, artifact_id, artifact_path, receipt_path))
    return keys


def capture_queue_all_task_keys(queue: Any) -> set[tuple[str, str, str, str]]:
    if not isinstance(queue, list):
        return set()
    keys: set[tuple[str, str, str, str]] = set()
    for item in queue:
        if isinstance(item, dict):
            keys.update(capture_queue_task_keys(item))
    return keys


def worksheet_queue_task_keys(worksheet: JSONDict) -> set[tuple[str, str, str, str]]:
    tasks = worksheet.get("tasks")
    if not isinstance(tasks, list):
        return set()
    keys: set[tuple[str, str, str, str]] = set()
    for task in tasks:
        if not isinstance(task, dict):
            continue
        request_path = task.get("request_path")
        artifact_id = task.get("artifact_id")
        artifact_path = task.get("artifact_output_path")
        receipt_path = task.get("receipt_output_path")
        if all(isinstance(value, str) and value for value in (request_path, artifact_id, artifact_path, receipt_path)):
            keys.add((request_path, artifact_id, artifact_path, receipt_path))
    return keys


def model_plane_contract_request_task_keys(contract_request: JSONDict) -> set[tuple[str, str, str, str]]:
    tasks = contract_request.get("tasks")
    if not isinstance(tasks, list):
        return set()
    keys: set[tuple[str, str, str, str]] = set()
    for task in tasks:
        if not isinstance(task, dict):
            continue
        request_path = task.get("request_path")
        artifact_id = task.get("artifact_id")
        artifact_path = task.get("artifact_output_path")
        receipt_path = task.get("receipt_output_path")
        if all(isinstance(value, str) and value for value in (request_path, artifact_id, artifact_path, receipt_path)):
            keys.add((request_path, artifact_id, artifact_path, receipt_path))
    return keys


def approval_command_validation_errors(queue_item: JSONDict) -> list[str]:
    request_path = queue_item.get("request_path")
    request_label = request_path if isinstance(request_path, str) and request_path else "unknown_request"
    errors: list[str] = []
    approval = queue_item.get("approval_rebuild_command")
    if not isinstance(approval, dict):
        return [f"all_request_approval_command_missing:{request_label}"]

    missing_approval_keys = set(list_of_strings(queue_item.get("missing_approval_keys")))
    recorded_approval_keys = set(list_of_strings(approval.get("records_approval_keys")))
    approval_command_tokens = command_tokens(approval.get("command"))
    if approval.get("command_class") != APPROVAL_COMMAND_CLASS:
        errors.append(f"all_request_approval_command_class_mismatch:{request_label}")
    if approval.get("metadata_only") is not True:
        errors.append(f"all_request_approval_command_not_metadata_only:{request_label}")
    if approval.get("requires_explicit_user_approval") is not True:
        errors.append(f"all_request_approval_command_missing_explicit_approval_gate:{request_label}")
    if approval.get("writes_request_path") != request_path:
        errors.append(f"all_request_approval_command_writes_request_mismatch:{request_label}")
    if not approval_command_tokens:
        errors.append(f"all_request_approval_command_missing:{request_label}")
    elif "scripts/build_phase3_runtime_capture_request.py" not in approval_command_tokens:
        errors.append(f"all_request_approval_command_unexpected_script:{request_label}")
    if "--output" in approval_command_tokens:
        output_index = approval_command_tokens.index("--output") + 1
        if output_index >= len(approval_command_tokens) or approval_command_tokens[output_index] != request_path:
            errors.append(f"all_request_approval_command_output_mismatch:{request_label}")
    else:
        errors.append(f"all_request_approval_command_output_missing:{request_label}")
    if missing_approval_keys != recorded_approval_keys:
        errors.append(f"all_request_approval_recorded_key_mismatch:{request_label}")
    unknown_keys = sorted((missing_approval_keys | recorded_approval_keys) - set(APPROVAL_KEY_FLAGS))
    if unknown_keys:
        errors.append(f"all_request_approval_unknown_keys:{request_label}")
    for key in sorted(missing_approval_keys):
        expected_flag = APPROVAL_KEY_FLAGS.get(key)
        if expected_flag and expected_flag not in approval_command_tokens:
            errors.append(f"all_request_approval_command_missing_flag:{request_label}:{expected_flag}")
    return errors



def capture_queue_by_request_path(queue: Any) -> dict[str, JSONDict]:
    if not isinstance(queue, list):
        return {}
    result: dict[str, JSONDict] = {}
    for item in queue:
        if not isinstance(item, dict):
            continue
        request_path = item.get("request_path")
        if isinstance(request_path, str) and request_path:
            result[request_path] = item
    return result


def approval_command_manifest_errors(manifest: Any, capture_queue: Any, remaining_gaps: JSONDict) -> tuple[list[str], int, int]:
    errors: list[str] = []
    if not isinstance(manifest, list):
        return ["approval_command_manifest_not_list"], 0, 0
    queue_by_path = capture_queue_by_request_path(capture_queue)
    manifest_paths: set[str] = set()
    ready_count = 0
    for index, item in enumerate(manifest):
        if not isinstance(item, dict):
            errors.append(f"approval_command_manifest_entry_not_object:index_{index}")
            continue
        request_path = item.get("request_path")
        request_label = request_path if isinstance(request_path, str) and request_path else f"index_{index}"
        if not isinstance(request_path, str) or not request_path:
            errors.append(f"approval_command_manifest_request_path_missing:{request_label}")
        else:
            manifest_paths.add(request_path)
        queue_item = queue_by_path.get(request_path or "")
        if queue_item is None:
            errors.append(f"approval_command_manifest_extra_request:{request_label}")
            missing_approval_keys = list_of_strings(item.get("records_approval_keys"))
        else:
            missing_approval_keys = list_of_strings(queue_item.get("missing_approval_keys"))
            embedded = queue_item.get("approval_rebuild_command")
            if isinstance(embedded, dict):
                for key in ("command_class", "writes_request_path", "metadata_only", "requires_explicit_user_approval"):
                    if embedded.get(key) != item.get(key):
                        errors.append(f"approval_command_manifest_embedded_{key}_mismatch:{request_label}")
                if command_tokens(embedded.get("command")) != command_tokens(item.get("command")):
                    errors.append(f"approval_command_manifest_embedded_command_mismatch:{request_label}")
                if sorted(list_of_strings(embedded.get("records_approval_keys"))) != sorted(list_of_strings(item.get("records_approval_keys"))):
                    errors.append(f"approval_command_manifest_embedded_keys_mismatch:{request_label}")
        shim = {
            "request_path": request_path,
            "missing_approval_keys": missing_approval_keys,
            "approval_rebuild_command": item,
        }
        item_errors = approval_command_validation_errors(shim)
        if item_errors:
            errors.extend(error.replace("all_request_approval_", "approval_command_manifest_") for error in item_errors)
        else:
            ready_count += 1
    queue_paths = set(queue_by_path)
    for request_path in sorted(queue_paths - manifest_paths):
        errors.append(f"approval_command_manifest_missing_request:{request_path}")
    if queue_paths and manifest_paths != queue_paths:
        errors.append("approval_command_manifest_capture_queue_request_mismatch")
    expected_count = remaining_gaps.get("runtime_capture_approval_rebuild_command_manifest_count")
    if isinstance(expected_count, int) and expected_count != len(manifest_paths):
        errors.append("approval_command_manifest_count_mismatch")
    return errors, len(manifest_paths), ready_count


def artifact_ids_from_preflight(preflight: Any) -> set[str]:
    if not isinstance(preflight, dict):
        return set()
    checks = preflight.get("artifact_checks")
    if not isinstance(checks, list):
        return set()
    return {
        str(item.get("artifact_id"))
        for item in checks
        if isinstance(item, dict) and isinstance(item.get("artifact_id"), str) and item.get("artifact_id")
    }


def receipt_artifact_ids_for_request(manifest: Any, request_path: Any) -> set[str]:
    if not isinstance(request_path, str) or not request_path or not isinstance(manifest, list):
        return set()
    artifact_ids: set[str] = set()
    for item in manifest:
        if not isinstance(item, dict) or item.get("request_path") != request_path:
            continue
        artifact_id = item.get("artifact_id")
        if isinstance(artifact_id, str) and artifact_id:
            artifact_ids.add(artifact_id)
    return artifact_ids


def runtime_capture_preflight_manifest_errors(
    preflight: Any,
    packet: JSONDict,
    request: JSONDict,
    queue_item: JSONDict,
    receipt_command_manifest: Any,
    remaining_gaps: JSONDict,
) -> tuple[list[str], JSONDict]:
    errors: list[str] = []
    counts: JSONDict = {
        "ready": False,
        "pending_artifact_count": 0,
        "artifact_check_count": 0,
        "receipt_entry_count": 0,
        "validator_command_count": 0,
        "runtime_closure_reason_count": 0,
        "missing_item_count": 0,
    }
    if not isinstance(preflight, dict):
        return ["runtime_capture_preflight_not_object"], counts
    packet_preflight = packet.get("recommended_runtime_capture_preflight_manifest")
    if not isinstance(packet_preflight, dict) or not packet_preflight:
        errors.append("runtime_capture_preflight_packet_missing")
    elif preflight != packet_preflight:
        errors.append("runtime_capture_preflight_packet_mismatch")
    counts.update(
        {
            "ready": preflight.get("ready") is True,
            "pending_artifact_count": int_count(preflight.get("pending_artifact_count")),
            "artifact_check_count": int_count(preflight.get("artifact_check_count")),
            "receipt_entry_count": int_count(preflight.get("receipt_entry_count")),
            "validator_command_count": int_count(preflight.get("validator_command_count")),
            "runtime_closure_reason_count": int_count(preflight.get("runtime_closure_reason_count")),
            "missing_item_count": int_count(preflight.get("missing_preflight_item_count")),
        }
    )
    if preflight.get("ready") is not True:
        errors.append("runtime_capture_preflight_not_ready")
    if preflight.get("request_path") != request.get("request_path"):
        errors.append("runtime_capture_preflight_request_path_mismatch")
    pending_ids = set(list_of_strings(preflight.get("pending_artifact_ids")))
    if counts["pending_artifact_count"] != len(pending_ids):
        errors.append("runtime_capture_preflight_pending_count_mismatch")
    artifact_checks = preflight.get("artifact_checks") if isinstance(preflight.get("artifact_checks"), list) else []
    artifact_check_count = len([item for item in artifact_checks if isinstance(item, dict)])
    if counts["artifact_check_count"] != artifact_check_count:
        errors.append("runtime_capture_preflight_artifact_check_count_mismatch")
    artifact_ids = artifact_ids_from_preflight(preflight)
    if pending_ids and artifact_ids != pending_ids:
        errors.append("runtime_capture_preflight_artifact_id_mismatch")
    queue_artifact_ids = {key[1] for key in capture_queue_task_keys(queue_item)}
    if queue_artifact_ids and artifact_ids != queue_artifact_ids:
        errors.append("runtime_capture_preflight_capture_queue_mismatch")
    receipt_artifact_ids = receipt_artifact_ids_for_request(receipt_command_manifest, request.get("request_path"))
    if receipt_artifact_ids and artifact_ids != receipt_artifact_ids:
        errors.append("runtime_capture_preflight_receipt_manifest_mismatch")
    validator_count = sum(
        int_count(item.get("validator_command_count"))
        for item in artifact_checks
        if isinstance(item, dict)
    )
    if counts["validator_command_count"] != validator_count:
        errors.append("runtime_capture_preflight_validator_count_mismatch")
    missing_items = list_of_strings(preflight.get("missing_preflight_items"))
    if counts["missing_item_count"] != len(missing_items):
        errors.append("runtime_capture_preflight_missing_item_count_mismatch")
    for key, summary_key, error_name in (
        ("pending_artifact_count", "recommended_runtime_capture_preflight_pending_artifact_count", "runtime_capture_preflight_gap_pending_count_mismatch"),
        ("receipt_entry_count", "recommended_runtime_capture_preflight_receipt_entry_count", "runtime_capture_preflight_gap_receipt_count_mismatch"),
        ("validator_command_count", "recommended_runtime_capture_preflight_validator_command_count", "runtime_capture_preflight_gap_validator_count_mismatch"),
        ("runtime_closure_reason_count", "recommended_runtime_capture_preflight_runtime_closure_reason_count", "runtime_capture_preflight_gap_closure_count_mismatch"),
        ("missing_item_count", "recommended_runtime_capture_preflight_missing_item_count", "runtime_capture_preflight_gap_missing_count_mismatch"),
    ):
        expected = remaining_gaps.get(summary_key)
        if isinstance(expected, int) and counts[key] != expected:
            errors.append(error_name)
    return errors, counts


def command_contract_task_keys(contract: Any) -> set[tuple[str, str, str, str]]:
    if not isinstance(contract, dict):
        return set()
    request_path = contract.get("request_path")
    if not isinstance(request_path, str) or not request_path:
        return set()
    tasks = contract.get("capture_tasks")
    if not isinstance(tasks, list):
        return set()
    keys: set[tuple[str, str, str, str]] = set()
    for task in tasks:
        if not isinstance(task, dict):
            continue
        artifact_id = task.get("artifact_id")
        artifact_path = task.get("artifact_path")
        receipt_path = task.get("receipt_path")
        if all(isinstance(value, str) and value for value in (artifact_id, artifact_path, receipt_path)):
            keys.add((request_path, artifact_id, artifact_path, receipt_path))
    return keys


def runtime_capture_command_contract_errors(
    contract: Any,
    packet: JSONDict,
    request: JSONDict,
    launch_card: JSONDict,
    remaining_gaps: JSONDict,
) -> tuple[list[str], JSONDict]:
    errors: list[str] = []
    counts: JSONDict = {
        "ready": False,
        "runtime_ready": False,
        "planned_capture_count": 0,
        "runtime_command_count": 0,
        "missing_runtime_command_count": 0,
    }
    if not isinstance(contract, dict):
        return ["runtime_capture_command_contract_not_object"], counts
    packet_contract = packet.get("recommended_runtime_capture_command_contract")
    if not isinstance(packet_contract, dict) or not packet_contract:
        errors.append("runtime_capture_command_contract_packet_missing")
    elif contract != packet_contract:
        errors.append("runtime_capture_command_contract_packet_mismatch")
    counts.update(
        {
            "ready": contract.get("command_contract_ready") is True,
            "runtime_ready": contract.get("runtime_capture_command_ready") is True,
            "planned_capture_count": int_count(contract.get("planned_capture_count")),
            "runtime_command_count": int_count(contract.get("runtime_command_option_count")),
            "missing_runtime_command_count": int_count(contract.get("missing_runtime_command_count")),
        }
    )
    if contract.get("schema_version") != plan_phase3_runtime_capture_commands.SUPPORTED_SCHEMA_VERSION:
        errors.append("runtime_capture_command_contract_schema_mismatch")
    if contract.get("mode") != "phase3_runtime_capture_command_contract":
        errors.append("runtime_capture_command_contract_mode_mismatch")
    if contract.get("valid") is not True:
        errors.append("runtime_capture_command_contract_invalid")
    if contract.get("command_contract_ready") is not True:
        errors.append("runtime_capture_command_contract_not_ready")
    if contract.get("request_path") != request.get("request_path"):
        errors.append("runtime_capture_command_contract_request_path_mismatch")
    tasks = contract.get("capture_tasks") if isinstance(contract.get("capture_tasks"), list) else []
    task_count = len([item for item in tasks if isinstance(item, dict)])
    if counts["planned_capture_count"] != task_count:
        errors.append("runtime_capture_command_contract_planned_count_mismatch")
    missing_ids = set(list_of_strings(contract.get("missing_runtime_command_artifact_ids")))
    if counts["missing_runtime_command_count"] != len(missing_ids):
        errors.append("runtime_capture_command_contract_missing_count_mismatch")
    contract_task_keys = command_contract_task_keys(contract)
    launch_card_keys = launch_card_queue_task_keys(launch_card)
    if launch_card_keys and contract_task_keys != launch_card_keys:
        errors.append("runtime_capture_command_contract_launch_card_mismatch")
    for key, summary_key, error_name in (
        ("planned_capture_count", "recommended_runtime_capture_command_contract_planned_capture_count", "runtime_capture_command_contract_gap_planned_count_mismatch"),
        ("runtime_command_count", "recommended_runtime_capture_command_contract_runtime_command_count", "runtime_capture_command_contract_gap_runtime_command_count_mismatch"),
        ("missing_runtime_command_count", "recommended_runtime_capture_command_contract_missing_runtime_command_count", "runtime_capture_command_contract_gap_missing_count_mismatch"),
    ):
        expected = remaining_gaps.get(summary_key)
        if isinstance(expected, int) and counts[key] != expected:
            errors.append(error_name)
    expected_ready = remaining_gaps.get("recommended_runtime_capture_command_contract_ready")
    if isinstance(expected_ready, bool) and counts["ready"] is not expected_ready:
        errors.append("runtime_capture_command_contract_gap_ready_mismatch")
    expected_runtime_ready = remaining_gaps.get("recommended_runtime_capture_command_contract_runtime_ready")
    if isinstance(expected_runtime_ready, bool) and counts["runtime_ready"] is not expected_runtime_ready:
        errors.append("runtime_capture_command_contract_gap_runtime_ready_mismatch")
    return errors, counts

def preflight_task_keys(preflight: Any) -> set[tuple[str, str, str, str]]:
    if not isinstance(preflight, dict):
        return set()
    request_path = preflight.get("request_path")
    if not isinstance(request_path, str) or not request_path:
        return set()
    checks = preflight.get("artifact_checks")
    if not isinstance(checks, list):
        return set()
    keys: set[tuple[str, str, str, str]] = set()
    for item in checks:
        if not isinstance(item, dict):
            continue
        artifact_id = item.get("artifact_id")
        artifact_path = item.get("artifact_path")
        receipt_path = item.get("receipt_path")
        if all(isinstance(value, str) and value for value in (artifact_id, artifact_path, receipt_path)):
            keys.add((request_path, artifact_id, artifact_path, receipt_path))
    return keys


def execution_coverage_task_keys(execution: Any) -> set[tuple[str, str, str, str]]:
    if not isinstance(execution, dict):
        return set()
    request_path = execution.get("request_path")
    if not isinstance(request_path, str) or not request_path:
        return set()
    rows = execution.get("artifact_execution")
    if not isinstance(rows, list):
        return set()
    keys: set[tuple[str, str, str, str]] = set()
    for item in rows:
        if not isinstance(item, dict):
            continue
        artifact_id = item.get("artifact_id")
        artifact_path = item.get("artifact_path")
        receipt_path = item.get("receipt_path")
        if all(isinstance(value, str) and value for value in (artifact_id, artifact_path, receipt_path)):
            keys.add((request_path, artifact_id, artifact_path, receipt_path))
    return keys


def runtime_capture_execution_coverage_errors(
    execution: Any,
    packet: JSONDict,
    request: JSONDict,
    preflight: Any,
    command_contract: Any,
    remaining_gaps: JSONDict,
) -> tuple[list[str], JSONDict]:
    errors: list[str] = []
    counts: JSONDict = {
        "ready": False,
        "manual_ready": False,
        "automated_ready": False,
        "pending_artifact_count": 0,
        "artifact_execution_count": 0,
        "capture_command_count": 0,
        "operator_command_count": 0,
        "metadata_command_count": 0,
        "artifacts_with_command_count": 0,
        "manual_capture_required_count": 0,
        "missing_capture_command_count": 0,
        "missing_item_count": 0,
    }
    if not isinstance(execution, dict):
        return ["runtime_capture_execution_coverage_not_object"], counts
    packet_execution = packet.get("recommended_runtime_capture_execution_coverage_manifest")
    if not isinstance(packet_execution, dict) or not packet_execution:
        errors.append("runtime_capture_execution_coverage_packet_missing")
    elif execution != packet_execution:
        errors.append("runtime_capture_execution_coverage_packet_mismatch")
    counts.update(
        {
            "ready": execution.get("ready") is True,
            "manual_ready": execution.get("manual_operator_capture_ready") is True,
            "automated_ready": execution.get("automated_capture_ready") is True,
            "pending_artifact_count": int_count(execution.get("pending_artifact_count")),
            "artifact_execution_count": int_count(execution.get("artifact_execution_count")),
            "capture_command_count": int_count(execution.get("capture_command_option_count")),
            "operator_command_count": int_count(execution.get("operator_command_option_count")),
            "metadata_command_count": int_count(execution.get("metadata_command_option_count")),
            "artifacts_with_command_count": int_count(execution.get("artifacts_with_capture_command_count")),
            "manual_capture_required_count": int_count(execution.get("manual_capture_required_count")),
            "missing_capture_command_count": int_count(execution.get("missing_capture_command_count")),
            "missing_item_count": int_count(execution.get("missing_execution_item_count")),
        }
    )
    if execution.get("manual_operator_capture_ready") is not True:
        errors.append("runtime_capture_execution_manual_not_ready")
    if execution.get("request_path") != request.get("request_path"):
        errors.append("runtime_capture_execution_request_path_mismatch")
    rows = execution.get("artifact_execution") if isinstance(execution.get("artifact_execution"), list) else []
    row_count = len([item for item in rows if isinstance(item, dict)])
    if counts["artifact_execution_count"] != row_count:
        errors.append("runtime_capture_execution_artifact_count_mismatch")
    if counts["pending_artifact_count"] != row_count:
        errors.append("runtime_capture_execution_pending_count_mismatch")
    row_capture_command_count = sum(int_count(item.get("capture_command_option_count")) for item in rows if isinstance(item, dict))
    row_operator_command_count = sum(int_count(item.get("operator_command_option_count")) for item in rows if isinstance(item, dict))
    row_metadata_command_count = sum(int_count(item.get("metadata_command_option_count")) for item in rows if isinstance(item, dict))
    row_artifacts_with_command = sum(1 for item in rows if isinstance(item, dict) and item.get("has_capture_command") is True)
    if counts["capture_command_count"] != row_capture_command_count:
        errors.append("runtime_capture_execution_capture_command_count_mismatch")
    if counts["operator_command_count"] != row_operator_command_count:
        errors.append("runtime_capture_execution_operator_command_count_mismatch")
    if counts["metadata_command_count"] != row_metadata_command_count:
        errors.append("runtime_capture_execution_metadata_command_count_mismatch")
    if counts["artifacts_with_command_count"] != row_artifacts_with_command:
        errors.append("runtime_capture_execution_artifacts_with_command_count_mismatch")
    missing_ids = set(list_of_strings(execution.get("missing_capture_command_artifact_ids")))
    if counts["missing_capture_command_count"] != len(missing_ids):
        errors.append("runtime_capture_execution_missing_capture_command_count_mismatch")
    if counts["manual_capture_required_count"] != len(missing_ids):
        errors.append("runtime_capture_execution_manual_required_count_mismatch")
    missing_items = list_of_strings(execution.get("missing_execution_items"))
    if counts["missing_item_count"] != len(missing_items):
        errors.append("runtime_capture_execution_missing_item_count_mismatch")
    execution_keys = execution_coverage_task_keys(execution)
    preflight_keys = preflight_task_keys(preflight)
    if preflight_keys and execution_keys != preflight_keys:
        errors.append("runtime_capture_execution_preflight_mismatch")
    contract_keys = command_contract_task_keys(command_contract)
    if contract_keys and execution_keys != contract_keys:
        errors.append("runtime_capture_execution_command_contract_mismatch")
    for key, summary_key, error_name in (
        ("manual_ready", "recommended_runtime_capture_execution_manual_ready", "runtime_capture_execution_gap_manual_ready_mismatch"),
        ("automated_ready", "recommended_runtime_capture_execution_automated_ready", "runtime_capture_execution_gap_automated_ready_mismatch"),
        ("capture_command_count", "recommended_runtime_capture_execution_command_option_count", "runtime_capture_execution_gap_command_count_mismatch"),
        ("operator_command_count", "recommended_runtime_capture_execution_operator_command_option_count", "runtime_capture_execution_gap_operator_count_mismatch"),
        ("metadata_command_count", "recommended_runtime_capture_execution_metadata_command_option_count", "runtime_capture_execution_gap_metadata_count_mismatch"),
        ("artifacts_with_command_count", "recommended_runtime_capture_execution_artifacts_with_command_count", "runtime_capture_execution_gap_artifacts_with_command_mismatch"),
        ("manual_capture_required_count", "recommended_runtime_capture_execution_manual_capture_required_count", "runtime_capture_execution_gap_manual_required_mismatch"),
        ("missing_capture_command_count", "recommended_runtime_capture_execution_missing_capture_command_count", "runtime_capture_execution_gap_missing_capture_command_mismatch"),
        ("missing_item_count", "recommended_runtime_capture_execution_missing_item_count", "runtime_capture_execution_gap_missing_item_mismatch"),
    ):
        expected = remaining_gaps.get(summary_key)
        if isinstance(expected, bool) and counts[key] is not expected:
            errors.append(error_name)
        elif isinstance(expected, int) and counts[key] != expected:
            errors.append(error_name)
    return errors, counts
def all_execution_coverage_task_keys(execution: Any) -> set[tuple[str, str, str, str]]:
    if not isinstance(execution, dict):
        return set()
    requests = execution.get("requests")
    if not isinstance(requests, list):
        return set()
    keys: set[tuple[str, str, str, str]] = set()
    for request in requests:
        if not isinstance(request, dict):
            continue
        request_path = request.get("request_path")
        if not isinstance(request_path, str) or not request_path:
            continue
        rows = request.get("artifact_execution")
        if not isinstance(rows, list):
            continue
        for item in rows:
            if not isinstance(item, dict):
                continue
            artifact_id = item.get("artifact_id")
            artifact_path = item.get("artifact_path")
            receipt_path = item.get("receipt_path")
            if all(isinstance(value, str) and value for value in (artifact_id, artifact_path, receipt_path)):
                keys.add((request_path, artifact_id, artifact_path, receipt_path))
    return keys


def all_request_runtime_capture_execution_coverage_errors(
    execution: Any,
    packet: JSONDict,
    capture_queue: Any,
    remaining_gaps: JSONDict,
) -> tuple[list[str], JSONDict]:
    errors: list[str] = []
    counts: JSONDict = {
        "ready": False,
        "manual_ready": False,
        "automated_ready": False,
        "request_count": 0,
        "manual_ready_count": 0,
        "automated_ready_count": 0,
        "pending_artifact_count": 0,
        "artifact_execution_count": 0,
        "capture_command_count": 0,
        "operator_command_count": 0,
        "metadata_command_count": 0,
        "artifacts_with_command_count": 0,
        "manual_capture_required_count": 0,
        "missing_capture_command_count": 0,
        "missing_item_count": 0,
    }
    if not isinstance(execution, dict):
        return ["all_request_runtime_capture_execution_coverage_not_object"], counts
    packet_execution = packet.get("all_request_runtime_capture_execution_coverage_manifest")
    if not isinstance(packet_execution, dict) or not packet_execution:
        errors.append("all_request_runtime_capture_execution_coverage_packet_missing")
    elif execution != packet_execution:
        errors.append("all_request_runtime_capture_execution_coverage_packet_mismatch")

    requests = execution.get("requests") if isinstance(execution.get("requests"), list) else []
    request_rows = [item for item in requests if isinstance(item, dict)]
    counts.update(
        {
            "ready": execution.get("ready") is True,
            "manual_ready": execution.get("manual_operator_capture_ready") is True,
            "automated_ready": execution.get("automated_capture_ready") is True,
            "request_count": int_count(execution.get("request_count")),
            "manual_ready_count": int_count(execution.get("manual_operator_capture_ready_count")),
            "automated_ready_count": int_count(execution.get("automated_capture_ready_count")),
            "pending_artifact_count": int_count(execution.get("pending_artifact_count")),
            "artifact_execution_count": int_count(execution.get("artifact_execution_count")),
            "capture_command_count": int_count(execution.get("capture_command_option_count")),
            "operator_command_count": int_count(execution.get("operator_command_option_count")),
            "metadata_command_count": int_count(execution.get("metadata_command_option_count")),
            "artifacts_with_command_count": int_count(execution.get("artifacts_with_capture_command_count")),
            "manual_capture_required_count": int_count(execution.get("manual_capture_required_count")),
            "missing_capture_command_count": int_count(execution.get("missing_capture_command_count")),
            "missing_item_count": int_count(execution.get("missing_execution_item_count")),
        }
    )
    if execution.get("manual_operator_capture_ready") is not True:
        errors.append("all_request_runtime_capture_execution_manual_not_ready")
    if counts["request_count"] != len(request_rows):
        errors.append("all_request_runtime_capture_execution_request_count_mismatch")
    expected_request_count = len(capture_queue) if isinstance(capture_queue, list) else 0
    if counts["request_count"] != expected_request_count:
        errors.append("all_request_runtime_capture_execution_capture_queue_request_count_mismatch")
    row_pending_count = sum(int_count(item.get("pending_artifact_count")) for item in request_rows)
    row_artifact_execution_count = sum(int_count(item.get("artifact_execution_count")) for item in request_rows)
    row_capture_command_count = sum(int_count(item.get("capture_command_option_count")) for item in request_rows)
    row_operator_command_count = sum(int_count(item.get("operator_command_option_count")) for item in request_rows)
    row_metadata_command_count = sum(int_count(item.get("metadata_command_option_count")) for item in request_rows)
    row_artifacts_with_command = sum(int_count(item.get("artifacts_with_capture_command_count")) for item in request_rows)
    row_manual_required = sum(int_count(item.get("manual_capture_required_count")) for item in request_rows)
    row_missing_commands = sum(int_count(item.get("missing_capture_command_count")) for item in request_rows)
    row_missing_items = sum(int_count(item.get("missing_execution_item_count")) for item in request_rows)
    row_manual_ready_count = sum(1 for item in request_rows if item.get("manual_operator_capture_ready") is True)
    row_automated_ready_count = sum(1 for item in request_rows if item.get("automated_capture_ready") is True)
    for count_key, row_value, error_name in (
        ("pending_artifact_count", row_pending_count, "all_request_runtime_capture_execution_pending_count_mismatch"),
        ("artifact_execution_count", row_artifact_execution_count, "all_request_runtime_capture_execution_artifact_count_mismatch"),
        ("capture_command_count", row_capture_command_count, "all_request_runtime_capture_execution_capture_command_count_mismatch"),
        ("operator_command_count", row_operator_command_count, "all_request_runtime_capture_execution_operator_command_count_mismatch"),
        ("metadata_command_count", row_metadata_command_count, "all_request_runtime_capture_execution_metadata_command_count_mismatch"),
        ("artifacts_with_command_count", row_artifacts_with_command, "all_request_runtime_capture_execution_artifacts_with_command_count_mismatch"),
        ("manual_capture_required_count", row_manual_required, "all_request_runtime_capture_execution_manual_required_count_mismatch"),
        ("missing_capture_command_count", row_missing_commands, "all_request_runtime_capture_execution_missing_capture_command_count_mismatch"),
        ("missing_item_count", row_missing_items, "all_request_runtime_capture_execution_missing_item_count_mismatch"),
        ("manual_ready_count", row_manual_ready_count, "all_request_runtime_capture_execution_manual_ready_count_mismatch"),
        ("automated_ready_count", row_automated_ready_count, "all_request_runtime_capture_execution_automated_ready_count_mismatch"),
    ):
        if counts[count_key] != row_value:
            errors.append(error_name)
    execution_keys = all_execution_coverage_task_keys(execution)
    queue_keys = capture_queue_all_task_keys(capture_queue)
    if queue_keys and execution_keys != queue_keys:
        errors.append("all_request_runtime_capture_execution_capture_queue_mismatch")
    for key, summary_key, error_name in (
        ("manual_ready", "all_request_runtime_capture_execution_manual_ready", "all_request_runtime_capture_execution_gap_manual_ready_mismatch"),
        ("automated_ready", "all_request_runtime_capture_execution_automated_ready", "all_request_runtime_capture_execution_gap_automated_ready_mismatch"),
        ("request_count", "all_request_runtime_capture_execution_request_count", "all_request_runtime_capture_execution_gap_request_count_mismatch"),
        ("manual_ready_count", "all_request_runtime_capture_execution_ready_request_count", "all_request_runtime_capture_execution_gap_ready_request_count_mismatch"),
        ("automated_ready_count", "all_request_runtime_capture_execution_automated_request_count", "all_request_runtime_capture_execution_gap_automated_request_count_mismatch"),
        ("pending_artifact_count", "all_request_runtime_capture_execution_pending_artifact_count", "all_request_runtime_capture_execution_gap_pending_count_mismatch"),
        ("artifact_execution_count", "all_request_runtime_capture_execution_artifact_execution_count", "all_request_runtime_capture_execution_gap_artifact_count_mismatch"),
        ("capture_command_count", "all_request_runtime_capture_execution_command_option_count", "all_request_runtime_capture_execution_gap_command_count_mismatch"),
        ("operator_command_count", "all_request_runtime_capture_execution_operator_command_option_count", "all_request_runtime_capture_execution_gap_operator_count_mismatch"),
        ("metadata_command_count", "all_request_runtime_capture_execution_metadata_command_option_count", "all_request_runtime_capture_execution_gap_metadata_count_mismatch"),
        ("artifacts_with_command_count", "all_request_runtime_capture_execution_artifacts_with_command_count", "all_request_runtime_capture_execution_gap_artifacts_with_command_mismatch"),
        ("manual_capture_required_count", "all_request_runtime_capture_execution_manual_capture_required_count", "all_request_runtime_capture_execution_gap_manual_required_mismatch"),
        ("missing_capture_command_count", "all_request_runtime_capture_execution_missing_capture_command_count", "all_request_runtime_capture_execution_gap_missing_capture_command_mismatch"),
        ("missing_item_count", "all_request_runtime_capture_execution_missing_item_count", "all_request_runtime_capture_execution_gap_missing_item_mismatch"),
    ):
        expected = remaining_gaps.get(summary_key)
        if isinstance(expected, bool) and counts[key] is not expected:
            errors.append(error_name)
        elif isinstance(expected, int) and counts[key] != expected:
            errors.append(error_name)
    return errors, counts


def runtime_capture_launch_card_directory_errors(
    directory: Any,
    packet: JSONDict,
    all_execution: Any,
    remaining_gaps: JSONDict,
    handoff_counts: JSONDict,
) -> tuple[list[str], JSONDict]:
    errors: list[str] = []
    counts: JSONDict = {
        "provided": False,
        "ready": False,
        "expected_card_count": 0,
        "matched_card_count": 0,
        "missing_card_count": 0,
        "card_count": 0,
        "path_count": 0,
    }
    if not isinstance(directory, dict):
        return ["runtime_capture_launch_card_directory_not_object"], counts
    packet_directory = packet.get("runtime_capture_launch_card_directory_manifest")
    if not isinstance(packet_directory, dict) or not packet_directory:
        errors.append("runtime_capture_launch_card_directory_packet_missing")
    elif directory != packet_directory:
        errors.append("runtime_capture_launch_card_directory_packet_mismatch")
    embedded_directory = all_execution.get("launch_card_directory") if isinstance(all_execution, dict) else None
    if not isinstance(embedded_directory, dict) or not embedded_directory:
        errors.append("runtime_capture_launch_card_directory_execution_missing")
    elif directory != embedded_directory:
        errors.append("runtime_capture_launch_card_directory_execution_mismatch")

    cards = directory.get("cards") if isinstance(directory.get("cards"), list) else []
    paths_by_request = directory.get("paths_by_request") if isinstance(directory.get("paths_by_request"), dict) else {}
    counts.update(
        {
            "provided": directory.get("provided") is True,
            "ready": directory.get("directory_ready") is True,
            "expected_card_count": int_count(directory.get("expected_card_count")),
            "matched_card_count": int_count(directory.get("matched_card_count")),
            "missing_card_count": int_count(directory.get("missing_card_count")),
            "card_count": len([item for item in cards if isinstance(item, dict)]),
            "path_count": len(paths_by_request),
        }
    )
    card_request_paths: set[str] = set()
    for index, card in enumerate(cards):
        if not isinstance(card, dict):
            continue
        request_path = normalized_manifest_path(card.get("request_path"))
        source_launch_card_path = normalized_manifest_path(card.get("source_launch_card_path"))
        if not request_path or not path_stays_in_repo(request_path):
            errors.append(f"runtime_capture_launch_card_directory_unsafe_request_path:{index}")
        else:
            card_request_paths.add(request_path)
        if source_launch_card_path and not path_stays_in_repo(source_launch_card_path):
            errors.append(f"runtime_capture_launch_card_directory_unsafe_source_launch_card_path:{index}")
    for request_path in paths_by_request:
        normalized_request_path = normalized_manifest_path(request_path)
        if not normalized_request_path or not path_stays_in_repo(normalized_request_path):
            errors.append("runtime_capture_launch_card_directory_unsafe_paths_by_request_key")
        elif card_request_paths and normalized_request_path not in card_request_paths:
            errors.append("runtime_capture_launch_card_directory_unknown_paths_by_request_key")

    if directory.get("valid") is not True:
        errors.append("runtime_capture_launch_card_directory_invalid")
    if counts["matched_card_count"] > counts["expected_card_count"]:
        errors.append("runtime_capture_launch_card_directory_matched_exceeds_expected")
    if counts["provided"]:
        if counts["expected_card_count"] != counts["card_count"]:
            errors.append("runtime_capture_launch_card_directory_card_count_mismatch")
        if counts["matched_card_count"] != counts["path_count"]:
            errors.append("runtime_capture_launch_card_directory_path_count_mismatch")
    if counts["ready"] and not counts["provided"]:
        errors.append("runtime_capture_launch_card_directory_ready_without_input")
    if counts["ready"] and counts["missing_card_count"] != 0:
        errors.append("runtime_capture_launch_card_directory_ready_with_missing_cards")
    for count_key, gap_key, error_name in (
        ("provided", "runtime_capture_launch_card_dir_provided", "runtime_capture_launch_card_directory_gap_provided_mismatch"),
        ("ready", "runtime_capture_launch_card_dir_ready", "runtime_capture_launch_card_directory_gap_ready_mismatch"),
        ("expected_card_count", "runtime_capture_launch_card_dir_expected_card_count", "runtime_capture_launch_card_directory_gap_expected_count_mismatch"),
        ("matched_card_count", "runtime_capture_launch_card_dir_matched_card_count", "runtime_capture_launch_card_directory_gap_matched_count_mismatch"),
        ("missing_card_count", "runtime_capture_launch_card_dir_missing_card_count", "runtime_capture_launch_card_directory_gap_missing_count_mismatch"),
    ):
        expected = remaining_gaps.get(gap_key)
        if isinstance(expected, bool) and counts[count_key] is not expected:
            errors.append(error_name)
        elif isinstance(expected, int) and counts[count_key] != expected:
            errors.append(error_name)
    for count_key, handoff_key, error_name in (
        ("provided", "runtime_capture_launch_card_dir_provided", "runtime_capture_launch_card_directory_handoff_provided_mismatch"),
        ("ready", "runtime_capture_launch_card_dir_ready", "runtime_capture_launch_card_directory_handoff_ready_mismatch"),
        ("expected_card_count", "runtime_capture_launch_card_dir_expected_card_count", "runtime_capture_launch_card_directory_handoff_expected_count_mismatch"),
        ("matched_card_count", "runtime_capture_launch_card_dir_matched_card_count", "runtime_capture_launch_card_directory_handoff_matched_count_mismatch"),
        ("missing_card_count", "runtime_capture_launch_card_dir_missing_card_count", "runtime_capture_launch_card_directory_handoff_missing_count_mismatch"),
    ):
        expected = handoff_counts.get(handoff_key)
        if isinstance(expected, bool) and counts[count_key] is not expected:
            errors.append(error_name)
        elif isinstance(expected, int) and counts[count_key] != expected:
            errors.append(error_name)
    return errors, counts

def reuse_evidence_capture_plan_errors(
    plan: Any,
    packet: JSONDict,
    remaining_gaps: JSONDict,
    handoff_counts: JSONDict,
    handoff_recommendation: JSONDict,
) -> tuple[list[str], JSONDict]:
    errors: list[str] = []
    counts: JSONDict = {
        "valid": False,
        "bundle_count": 0,
        "reuse_ready_count": 0,
        "reuse_blocked_count": 0,
        "candidate_trace_valid_count": 0,
        "candidate_trace_receipt_ready_count": 0,
        "no_reuse_distance_observation_count": 0,
        "prompt_identity_ready_count": 0,
        "prompt_identity_metadata_missing_count": 0,
        "has_recommended_capture": False,
        "manifest_recommendation_ready": False,
    }
    if not isinstance(plan, dict):
        return ["reuse_evidence_capture_plan_not_object"], counts

    packet_plan = packet.get("phase3_reuse_evidence_capture_plan") if isinstance(packet, dict) else None
    if not isinstance(packet_plan, dict) or not packet_plan:
        errors.append("reuse_evidence_capture_plan_packet_missing")
    elif plan != packet_plan:
        errors.append("reuse_evidence_capture_plan_packet_mismatch")

    counts.update(
        {
            "valid": plan.get("valid") is True,
            "bundle_count": int_count(plan.get("bundle_count")),
            "reuse_ready_count": int_count(plan.get("reuse_ready_count")),
            "reuse_blocked_count": int_count(plan.get("reuse_blocked_count")),
            "candidate_trace_valid_count": int_count(plan.get("candidate_trace_valid_count")),
            "candidate_trace_receipt_ready_count": int_count(plan.get("candidate_trace_receipt_ready_count")),
            "no_reuse_distance_observation_count": int_count(plan.get("no_reuse_distance_observation_count")),
            "prompt_identity_ready_count": int_count(plan.get("prompt_identity_ready_count")),
            "prompt_identity_metadata_missing_count": int_count(plan.get("prompt_identity_metadata_missing_count")),
        }
    )

    if plan.get("schema_version") != "moe-phase3-reuse-evidence-capture-plan-v1":
        errors.append("reuse_evidence_capture_plan_schema_mismatch")
    if plan.get("valid") is not True:
        errors.append("reuse_evidence_capture_plan_invalid")
    if counts["reuse_ready_count"] + counts["reuse_blocked_count"] != counts["bundle_count"]:
        errors.append("reuse_evidence_capture_plan_ready_blocked_count_mismatch")
    if counts["candidate_trace_valid_count"] > counts["bundle_count"]:
        errors.append("reuse_evidence_capture_plan_trace_valid_count_invalid")
    if counts["candidate_trace_receipt_ready_count"] > counts["bundle_count"]:
        errors.append("reuse_evidence_capture_plan_receipt_ready_count_invalid")

    recommendation = plan.get("recommended_next_capture")
    if isinstance(recommendation, dict):
        counts["has_recommended_capture"] = True
        if recommendation.get("required_result") != "capture_candidate_router_trace_with_reuse_distance":
            errors.append("reuse_evidence_capture_recommended_result_mismatch")
        for field in ("bundle_path", "candidate_trace_path", "prompt_set_path", "runtime_capture_request_path"):
            path_text = normalized_manifest_path(recommendation.get(field))
            if not path_text or not path_stays_in_repo(path_text):
                errors.append(f"reuse_evidence_capture_recommended_path_unsafe:{field}")
        policy_plan = packet.get("recommended_policy_candidate_trace_plan") if isinstance(packet, dict) else None
        if isinstance(policy_plan, dict):
            if recommendation.get("bundle_path") != policy_plan.get("bundle_path"):
                errors.append("reuse_evidence_capture_policy_bundle_mismatch")
            if recommendation.get("candidate_trace_path") != policy_plan.get("candidate_trace_path"):
                errors.append("reuse_evidence_capture_policy_trace_path_mismatch")
            if recommendation.get("prompt_set_path") != policy_plan.get("candidate_prompt_set_path"):
                errors.append("reuse_evidence_capture_policy_prompt_set_mismatch")
        packet_request = packet.get("recommended_runtime_capture_request") if isinstance(packet, dict) else None
        if isinstance(packet_request, dict) and recommendation.get("runtime_capture_request_path") != packet_request.get("request_path"):
            errors.append("reuse_evidence_capture_runtime_request_mismatch")
        manifest_recommendation_errors: list[str] = []
        if not isinstance(handoff_recommendation, dict) or not handoff_recommendation:
            manifest_recommendation_errors.append("reuse_evidence_capture_manifest_recommendation_missing")
        else:
            for field in (
                "bundle_name",
                "bundle_path",
                "model_id",
                "candidate_trace_path",
                "prompt_set_path",
                "runtime_capture_request_path",
                "selection_rationale",
                "required_result",
            ):
                if handoff_recommendation.get(field) != recommendation.get(field):
                    manifest_recommendation_errors.append(f"reuse_evidence_capture_manifest_{field}_mismatch")
            contract = plan.get("capture_contract") if isinstance(plan.get("capture_contract"), dict) else {}
            if int_count(handoff_recommendation.get("minimum_reuse_distance_observations")) != int_count(contract.get("minimum_reuse_distance_observations")):
                manifest_recommendation_errors.append("reuse_evidence_capture_manifest_min_observations_mismatch")
            if (handoff_recommendation.get("must_bind_prompt_identity_metadata") is True) != (contract.get("must_bind_prompt_identity_metadata") is True):
                manifest_recommendation_errors.append("reuse_evidence_capture_manifest_prompt_identity_contract_mismatch")
        errors.extend(manifest_recommendation_errors)
        counts["manifest_recommendation_ready"] = not manifest_recommendation_errors
    elif counts["reuse_blocked_count"]:
        errors.append("reuse_evidence_capture_recommendation_missing")

    for count_key, gap_key, error_name in (
        ("reuse_ready_count", "policy_candidate_ready_count", "reuse_evidence_capture_gap_ready_count_mismatch"),
        ("reuse_blocked_count", "policy_candidate_blocked_bundle_count", "reuse_evidence_capture_gap_blocked_count_mismatch"),
        ("no_reuse_distance_observation_count", "policy_candidate_no_reuse_distance_observation_count", "reuse_evidence_capture_gap_no_distance_count_mismatch"),
        ("prompt_identity_ready_count", "policy_candidate_prompt_identity_ready_count", "reuse_evidence_capture_gap_prompt_identity_ready_count_mismatch"),
        ("prompt_identity_metadata_missing_count", "policy_candidate_prompt_identity_metadata_missing_count", "reuse_evidence_capture_gap_prompt_identity_missing_count_mismatch"),
    ):
        expected = remaining_gaps.get(gap_key)
        if isinstance(expected, int) and counts[count_key] != expected:
            errors.append(error_name)

    for count_key, handoff_key, error_name in (
        ("valid", "reuse_evidence_capture_valid", "reuse_evidence_capture_handoff_valid_mismatch"),
        ("bundle_count", "reuse_evidence_capture_bundle_count", "reuse_evidence_capture_handoff_bundle_count_mismatch"),
        ("reuse_ready_count", "reuse_evidence_capture_ready_count", "reuse_evidence_capture_handoff_ready_count_mismatch"),
        ("reuse_blocked_count", "reuse_evidence_capture_blocked_count", "reuse_evidence_capture_handoff_blocked_count_mismatch"),
        ("candidate_trace_valid_count", "reuse_evidence_capture_trace_valid_count", "reuse_evidence_capture_handoff_trace_valid_count_mismatch"),
        ("candidate_trace_receipt_ready_count", "reuse_evidence_capture_receipt_ready_count", "reuse_evidence_capture_handoff_receipt_ready_count_mismatch"),
        ("no_reuse_distance_observation_count", "reuse_evidence_capture_no_reuse_distance_count", "reuse_evidence_capture_handoff_no_distance_count_mismatch"),
        ("prompt_identity_ready_count", "reuse_evidence_capture_prompt_identity_ready_count", "reuse_evidence_capture_handoff_prompt_identity_ready_count_mismatch"),
        ("prompt_identity_metadata_missing_count", "reuse_evidence_capture_prompt_identity_missing_count", "reuse_evidence_capture_handoff_prompt_identity_missing_count_mismatch"),
    ):
        expected = handoff_counts.get(handoff_key)
        if isinstance(expected, bool) and counts[count_key] is not expected:
            errors.append(error_name)
        elif isinstance(expected, int) and counts[count_key] != expected:
            errors.append(error_name)
    return errors, counts

def _task_tuple(request_path: Any, artifact_id: Any, artifact_path: Any, receipt_path: Any) -> tuple[str, str, str, str] | None:

    if all(isinstance(value, str) and value for value in (request_path, artifact_id, artifact_path, receipt_path)):
        return (request_path, artifact_id, artifact_path, receipt_path)
    return None


def receipt_command_task_keys(manifest: Any) -> set[tuple[str, str, str, str]]:
    if not isinstance(manifest, list):
        return set()
    keys: set[tuple[str, str, str, str]] = set()
    for item in manifest:
        if not isinstance(item, dict):
            continue
        key = _task_tuple(item.get("request_path"), item.get("artifact_id"), item.get("artifact_path"), item.get("receipt_path"))
        if key:
            keys.add(key)
    return keys


def all_execution_manual_task_keys(execution: Any) -> set[tuple[str, str, str, str]]:
    if not isinstance(execution, dict):
        return set()
    requests = execution.get("requests")
    if not isinstance(requests, list):
        return set()
    keys: set[tuple[str, str, str, str]] = set()
    for request in requests:
        if not isinstance(request, dict):
            continue
        request_path = request.get("request_path")
        rows = request.get("artifact_execution")
        if not isinstance(rows, list):
            continue
        for item in rows:
            if not isinstance(item, dict) or item.get("has_runtime_capture_command") is True:
                continue
            key = _task_tuple(request_path, item.get("artifact_id"), item.get("artifact_path"), item.get("receipt_path"))
            if key:
                keys.add(key)
    return keys


def manual_capture_runbook_task_keys(runbook: Any, *, include_runtime_command_tasks: bool = False) -> set[tuple[str, str, str, str]]:
    if not isinstance(runbook, dict):
        return set()
    requests = runbook.get("requests")
    if not isinstance(requests, list):
        return set()
    keys: set[tuple[str, str, str, str]] = set()
    for request in requests:
        if not isinstance(request, dict):
            continue
        request_path = request.get("request_path")
        task_groups = ["manual_tasks"]
        if include_runtime_command_tasks:
            task_groups.append("runtime_command_tasks")
        for group in task_groups:
            tasks = request.get(group)
            if not isinstance(tasks, list):
                continue
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                key = _task_tuple(
                    task.get("request_path") or request_path,
                    task.get("artifact_id"),
                    task.get("artifact_path"),
                    task.get("receipt_path"),
                )
                if key:
                    keys.add(key)
    return keys


def manual_capture_runbook_errors(
    runbook: Any,
    packet: JSONDict,
    all_execution: Any,
    capture_queue: Any,
    receipt_command_manifest: Any,
    remaining_gaps: JSONDict,
    handoff_counts: JSONDict,
) -> tuple[list[str], JSONDict]:
    errors: list[str] = []
    counts: JSONDict = {
        "ready": False,
        "request_count": 0,
        "ready_request_count": 0,
        "manual_task_count": 0,
        "runtime_command_task_count": 0,
        "validator_command_count": 0,
        "missing_item_count": 0,
        "missing_receipt_command_count": 0,
        "missing_validator_command_count": 0,
        "missing_approval_command_count": 0,
        "missing_source_request_path_count": 0,
        "missing_prompt_set_path_count": 0,
        "missing_explicit_approval_count": 0,
        "missing_prompt_traffic_ack_count": 0,
    }
    if not isinstance(runbook, dict):
        return ["manual_capture_runbook_not_object"], counts
    packet_runbook = packet.get("all_request_manual_capture_runbook_manifest")
    if not isinstance(packet_runbook, dict) or not packet_runbook:
        errors.append("manual_capture_runbook_packet_missing")
    elif runbook != packet_runbook:
        errors.append("manual_capture_runbook_packet_mismatch")

    requests = runbook.get("requests") if isinstance(runbook.get("requests"), list) else []
    request_rows = [item for item in requests if isinstance(item, dict)]
    counts.update(
        {
            "ready": runbook.get("ready") is True,
            "request_count": int_count(runbook.get("request_count")),
            "ready_request_count": int_count(runbook.get("ready_request_count")),
            "manual_task_count": int_count(runbook.get("manual_task_count")),
            "runtime_command_task_count": int_count(runbook.get("runtime_command_task_count")),
            "validator_command_count": int_count(runbook.get("validator_command_count")),
            "missing_item_count": int_count(runbook.get("missing_item_count")),
            "missing_receipt_command_count": int_count(runbook.get("missing_receipt_command_count")),
            "missing_validator_command_count": int_count(runbook.get("missing_validator_command_count")),
            "missing_approval_command_count": int_count(runbook.get("missing_approval_command_count")),
            "missing_source_request_path_count": int_count(runbook.get("missing_source_request_path_count")),
            "missing_prompt_set_path_count": int_count(runbook.get("missing_prompt_set_path_count")),
            "missing_explicit_approval_count": int_count(runbook.get("missing_explicit_approval_count")),
            "missing_prompt_traffic_ack_count": int_count(runbook.get("missing_prompt_traffic_ack_count")),
        }
    )
    if runbook.get("ready") is not True:
        errors.append("manual_capture_runbook_not_ready")
    if counts["request_count"] != len(request_rows):
        errors.append("manual_capture_runbook_request_count_mismatch")
    expected_request_count = len(capture_queue) if isinstance(capture_queue, list) else 0
    if counts["request_count"] != expected_request_count:
        errors.append("manual_capture_runbook_capture_queue_request_count_mismatch")
    row_ready_count = sum(1 for item in request_rows if item.get("manual_operator_capture_ready") is True)
    row_manual_task_count = sum(len(item.get("manual_tasks")) for item in request_rows if isinstance(item.get("manual_tasks"), list))
    row_runtime_task_count = sum(len(item.get("runtime_command_tasks")) for item in request_rows if isinstance(item.get("runtime_command_tasks"), list))
    row_validator_count = 0
    row_missing_item_count = 0
    row_missing_receipt_count = 0
    row_missing_validator_count = 0
    row_missing_approval_count = 0
    row_missing_source_request_path_count = 0
    row_missing_prompt_set_path_count = 0
    row_missing_explicit_approval_count = 0
    row_missing_prompt_traffic_ack_count = 0
    task_missing_source_request_path_count = 0
    task_missing_prompt_set_path_count = 0
    task_missing_explicit_approval_count = 0
    task_missing_prompt_traffic_ack_count = 0
    for item in request_rows:
        row_validator_count += int_count(item.get("validator_command_count"))
        row_missing_item_count += int_count(item.get("missing_item_count"))
        row_missing_receipt_count += int_count(item.get("missing_receipt_command_count"))
        row_missing_validator_count += int_count(item.get("missing_validator_command_count"))
        row_missing_source_request_path_count += int_count(item.get("missing_source_request_path_count"))
        row_missing_prompt_set_path_count += int_count(item.get("missing_prompt_set_path_count"))
        row_missing_explicit_approval_count += int_count(item.get("missing_explicit_approval_count"))
        row_missing_prompt_traffic_ack_count += int_count(item.get("missing_prompt_traffic_ack_count"))
        if item.get("approval_command_present") is not True:
            row_missing_approval_count += 1
        request_path = normalized_manifest_path(item.get("request_path"))
        for group_name in ("manual_tasks", "runtime_command_tasks"):
            tasks = item.get(group_name)
            if not isinstance(tasks, list):
                continue
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                source_request_path = normalized_manifest_path(task.get("source_request_path"))
                prompt_set_path = normalized_manifest_path(task.get("prompt_set_path"))
                source_prompt_set_path = normalized_manifest_path(task.get("source_prompt_set_path"))
                if source_request_path != request_path:
                    task_missing_source_request_path_count += 1
                if not prompt_set_path or source_prompt_set_path != prompt_set_path:
                    task_missing_prompt_set_path_count += 1
                if task.get("requires_explicit_user_approval") is not True:
                    task_missing_explicit_approval_count += 1
                if task.get("approval_records_prompt_traffic") is not True or task.get("may_send_prompt_traffic_after_approval") is not True:
                    task_missing_prompt_traffic_ack_count += 1
    for count_key, task_value, error_name in (
        ("missing_source_request_path_count", task_missing_source_request_path_count, "manual_capture_runbook_task_missing_source_request_path_count_mismatch"),
        ("missing_prompt_set_path_count", task_missing_prompt_set_path_count, "manual_capture_runbook_task_missing_prompt_set_path_count_mismatch"),
        ("missing_explicit_approval_count", task_missing_explicit_approval_count, "manual_capture_runbook_task_missing_explicit_approval_count_mismatch"),
        ("missing_prompt_traffic_ack_count", task_missing_prompt_traffic_ack_count, "manual_capture_runbook_task_missing_prompt_traffic_ack_count_mismatch"),
    ):
        if counts[count_key] != task_value:
            errors.append(error_name)

    for count_key, row_value, error_name in (
        ("ready_request_count", row_ready_count, "manual_capture_runbook_ready_request_count_mismatch"),
        ("manual_task_count", row_manual_task_count, "manual_capture_runbook_manual_task_count_mismatch"),
        ("runtime_command_task_count", row_runtime_task_count, "manual_capture_runbook_runtime_task_count_mismatch"),
        ("validator_command_count", row_validator_count, "manual_capture_runbook_validator_count_mismatch"),
        ("missing_item_count", row_missing_item_count, "manual_capture_runbook_missing_item_count_mismatch"),
        ("missing_receipt_command_count", row_missing_receipt_count, "manual_capture_runbook_missing_receipt_count_mismatch"),
        ("missing_validator_command_count", row_missing_validator_count, "manual_capture_runbook_missing_validator_count_mismatch"),
        ("missing_approval_command_count", row_missing_approval_count, "manual_capture_runbook_missing_approval_count_mismatch"),
        ("missing_source_request_path_count", row_missing_source_request_path_count, "manual_capture_runbook_missing_source_request_path_count_mismatch"),
        ("missing_prompt_set_path_count", row_missing_prompt_set_path_count, "manual_capture_runbook_missing_prompt_set_path_count_mismatch"),
        ("missing_explicit_approval_count", row_missing_explicit_approval_count, "manual_capture_runbook_missing_explicit_approval_count_mismatch"),
        ("missing_prompt_traffic_ack_count", row_missing_prompt_traffic_ack_count, "manual_capture_runbook_missing_prompt_traffic_ack_count_mismatch"),
    ):
        if counts[count_key] != row_value:
            errors.append(error_name)

    manual_keys = manual_capture_runbook_task_keys(runbook)
    all_runbook_keys = manual_capture_runbook_task_keys(runbook, include_runtime_command_tasks=True)
    execution_manual_keys = all_execution_manual_task_keys(all_execution)
    receipt_keys = receipt_command_task_keys(receipt_command_manifest)
    if execution_manual_keys and manual_keys != execution_manual_keys:
        errors.append("manual_capture_runbook_execution_manual_task_mismatch")
    if all_runbook_keys and not all_runbook_keys.issubset(receipt_keys):
        errors.append("manual_capture_runbook_receipt_command_task_mismatch")
    receipt_validator_count = count_validator_commands(receipt_command_manifest)
    if receipt_validator_count and counts["validator_command_count"] != receipt_validator_count:
        errors.append("manual_capture_runbook_receipt_validator_count_mismatch")
    if isinstance(all_execution, dict):
        expected_manual = int_count(all_execution.get("missing_capture_command_count"))
        expected_runtime = int_count(all_execution.get("artifacts_with_capture_command_count"))
        if counts["manual_task_count"] != expected_manual:
            errors.append("manual_capture_runbook_execution_manual_count_mismatch")
        if counts["runtime_command_task_count"] != expected_runtime:
            errors.append("manual_capture_runbook_execution_runtime_count_mismatch")
    for count_key, gap_key, error_name in (
        ("ready", "all_request_manual_capture_runbook_ready", "manual_capture_runbook_gap_ready_mismatch"),
        ("request_count", "all_request_manual_capture_runbook_request_count", "manual_capture_runbook_gap_request_count_mismatch"),
        ("ready_request_count", "all_request_manual_capture_runbook_ready_request_count", "manual_capture_runbook_gap_ready_request_count_mismatch"),
        ("manual_task_count", "all_request_manual_capture_runbook_manual_task_count", "manual_capture_runbook_gap_manual_task_count_mismatch"),
        ("runtime_command_task_count", "all_request_manual_capture_runbook_runtime_command_task_count", "manual_capture_runbook_gap_runtime_task_count_mismatch"),
        ("validator_command_count", "all_request_manual_capture_runbook_validator_command_count", "manual_capture_runbook_gap_validator_count_mismatch"),
        ("missing_item_count", "all_request_manual_capture_runbook_missing_item_count", "manual_capture_runbook_gap_missing_item_count_mismatch"),
        ("missing_receipt_command_count", "all_request_manual_capture_runbook_missing_receipt_command_count", "manual_capture_runbook_gap_missing_receipt_count_mismatch"),
        ("missing_validator_command_count", "all_request_manual_capture_runbook_missing_validator_command_count", "manual_capture_runbook_gap_missing_validator_count_mismatch"),
        ("missing_approval_command_count", "all_request_manual_capture_runbook_missing_approval_command_count", "manual_capture_runbook_gap_missing_approval_count_mismatch"),
        ("missing_source_request_path_count", "all_request_manual_capture_runbook_missing_source_request_path_count", "manual_capture_runbook_gap_missing_source_request_path_count_mismatch"),
        ("missing_prompt_set_path_count", "all_request_manual_capture_runbook_missing_prompt_set_path_count", "manual_capture_runbook_gap_missing_prompt_set_path_count_mismatch"),
        ("missing_explicit_approval_count", "all_request_manual_capture_runbook_missing_explicit_approval_count", "manual_capture_runbook_gap_missing_explicit_approval_count_mismatch"),
        ("missing_prompt_traffic_ack_count", "all_request_manual_capture_runbook_missing_prompt_traffic_ack_count", "manual_capture_runbook_gap_missing_prompt_traffic_ack_count_mismatch"),
    ):
        expected = remaining_gaps.get(gap_key)
        if isinstance(expected, bool) and counts[count_key] is not expected:
            errors.append(error_name)
        elif isinstance(expected, int) and counts[count_key] != expected:
            errors.append(error_name)
    for count_key, handoff_key, error_name in (
        ("ready", "manual_runbook_ready", "manual_capture_runbook_handoff_ready_mismatch"),
        ("request_count", "manual_runbook_request_count", "manual_capture_runbook_handoff_request_count_mismatch"),
        ("ready_request_count", "manual_runbook_ready_request_count", "manual_capture_runbook_handoff_ready_request_count_mismatch"),
        ("manual_task_count", "manual_runbook_manual_task_count", "manual_capture_runbook_handoff_manual_task_count_mismatch"),
        ("runtime_command_task_count", "manual_runbook_runtime_command_task_count", "manual_capture_runbook_handoff_runtime_task_count_mismatch"),
        ("validator_command_count", "manual_runbook_validator_command_count", "manual_capture_runbook_handoff_validator_count_mismatch"),
        ("missing_item_count", "manual_runbook_missing_item_count", "manual_capture_runbook_handoff_missing_item_count_mismatch"),
        ("missing_receipt_command_count", "manual_runbook_missing_receipt_command_count", "manual_capture_runbook_handoff_missing_receipt_count_mismatch"),
        ("missing_validator_command_count", "manual_runbook_missing_validator_command_count", "manual_capture_runbook_handoff_missing_validator_count_mismatch"),
        ("missing_approval_command_count", "manual_runbook_missing_approval_command_count", "manual_capture_runbook_handoff_missing_approval_count_mismatch"),
        ("missing_source_request_path_count", "manual_runbook_missing_source_request_path_count", "manual_capture_runbook_handoff_missing_source_request_path_count_mismatch"),
        ("missing_prompt_set_path_count", "manual_runbook_missing_prompt_set_path_count", "manual_capture_runbook_handoff_missing_prompt_set_path_count_mismatch"),
        ("missing_explicit_approval_count", "manual_runbook_missing_explicit_approval_count", "manual_capture_runbook_handoff_missing_explicit_approval_count_mismatch"),
        ("missing_prompt_traffic_ack_count", "manual_runbook_missing_prompt_traffic_ack_count", "manual_capture_runbook_handoff_missing_prompt_traffic_ack_count_mismatch"),
    ):
        expected = handoff_counts.get(handoff_key)
        if isinstance(expected, bool) and counts[count_key] is not expected:
            errors.append(error_name)
        elif isinstance(expected, int) and counts[count_key] != expected:
            errors.append(error_name)
    return errors, counts

def post_capture_intake_runbook_task_keys(runbook: Any) -> set[tuple[str, str, str, str]]:
    if not isinstance(runbook, dict):
        return set()
    requests = runbook.get("requests")
    if not isinstance(requests, list):
        return set()
    keys: set[tuple[str, str, str, str]] = set()
    for request in requests:
        if not isinstance(request, dict):
            continue
        request_path = request.get("request_path")
        gates = request.get("artifact_gates")
        if not isinstance(gates, list):
            continue
        for gate in gates:
            if not isinstance(gate, dict):
                continue
            key = _task_tuple(
                request_path,
                gate.get("artifact_id"),
                gate.get("artifact_path"),
                gate.get("receipt_path"),
            )
            if key:
                keys.add(key)
    return keys


def post_capture_intake_runbook_errors(
    runbook: Any,
    packet: JSONDict,
    manual_runbook: Any,
    capture_queue: Any,
    receipt_command_manifest: Any,
    remaining_gaps: JSONDict,
    handoff_counts: JSONDict,
) -> tuple[list[str], JSONDict]:
    errors: list[str] = []
    counts: JSONDict = {
        "ready": False,
        "request_count": 0,
        "artifact_gate_count": 0,
        "ready_after_current_intake_count": 0,
        "missing_after_current_intake_count": 0,
        "validator_command_count": 0,
        "ready_to_update_bundle_count": 0,
        "phase4_candidate_count": 0,
        "live_spike_candidate_count": 0,
        "missing_source_request_path_count": 0,
        "missing_prompt_set_path_count": 0,
        "missing_explicit_approval_count": 0,
        "missing_prompt_traffic_ack_count": 0,
        "missing_item_count": 0,
    }
    if not isinstance(runbook, dict):
        return ["post_capture_intake_runbook_not_object"], counts
    packet_runbook = packet.get("all_request_post_capture_intake_runbook_manifest")
    if not isinstance(packet_runbook, dict) or not packet_runbook:
        errors.append("post_capture_intake_runbook_packet_missing")
    elif runbook != packet_runbook:
        errors.append("post_capture_intake_runbook_packet_mismatch")

    requests = runbook.get("requests") if isinstance(runbook.get("requests"), list) else []
    request_rows = [item for item in requests if isinstance(item, dict)]
    counts.update(
        {
            "ready": runbook.get("ready") is True,
            "request_count": int_count(runbook.get("request_count")),
            "artifact_gate_count": int_count(runbook.get("artifact_gate_count")),
            "ready_after_current_intake_count": int_count(runbook.get("ready_after_current_intake_count")),
            "missing_after_current_intake_count": int_count(runbook.get("missing_after_current_intake_count")),
            "validator_command_count": int_count(runbook.get("validator_command_count")),
            "ready_to_update_bundle_count": int_count(runbook.get("ready_to_update_bundle_count")),
            "phase4_candidate_count": int_count(runbook.get("phase4_candidate_count")),
            "live_spike_candidate_count": int_count(runbook.get("live_spike_candidate_count")),
            "missing_source_request_path_count": int_count(runbook.get("missing_source_request_path_count")),
            "missing_prompt_set_path_count": int_count(runbook.get("missing_prompt_set_path_count")),
            "missing_explicit_approval_count": int_count(runbook.get("missing_explicit_approval_count")),
            "missing_prompt_traffic_ack_count": int_count(runbook.get("missing_prompt_traffic_ack_count")),
            "missing_item_count": int_count(runbook.get("missing_item_count")),
        }
    )
    if runbook.get("ready") is not True:
        errors.append("post_capture_intake_runbook_not_ready")
    if counts["request_count"] != len(request_rows):
        errors.append("post_capture_intake_runbook_request_count_mismatch")
    expected_request_count = len(capture_queue) if isinstance(capture_queue, list) else 0
    if counts["request_count"] != expected_request_count:
        errors.append("post_capture_intake_runbook_capture_queue_request_count_mismatch")

    row_artifact_gate_count = 0
    row_ready_count = 0
    row_missing_count = 0
    row_validator_count = 0
    row_ready_to_update_count = 0
    row_missing_source_request_path_count = 0
    row_missing_prompt_set_path_count = 0
    row_missing_explicit_approval_count = 0
    row_missing_prompt_traffic_ack_count = 0
    gate_missing_source_request_path_count = 0
    gate_missing_prompt_set_path_count = 0
    gate_missing_explicit_approval_count = 0
    gate_missing_prompt_traffic_ack_count = 0
    for request in request_rows:
        gates = request.get("artifact_gates") if isinstance(request.get("artifact_gates"), list) else []
        row_artifact_gate_count += len(gates)
        row_ready_count += int_count(request.get("ready_after_current_intake_count"))
        row_missing_count += int_count(request.get("missing_after_current_intake_count"))
        row_validator_count += int_count(request.get("validator_command_count"))
        row_missing_source_request_path_count += int_count(request.get("missing_source_request_path_count"))
        row_missing_prompt_set_path_count += int_count(request.get("missing_prompt_set_path_count"))
        row_missing_explicit_approval_count += int_count(request.get("missing_explicit_approval_count"))
        row_missing_prompt_traffic_ack_count += int_count(request.get("missing_prompt_traffic_ack_count"))
        request_path = normalized_manifest_path(request.get("request_path"))
        for gate in gates:
            if not isinstance(gate, dict):
                continue
            source_request_path = normalized_manifest_path(gate.get("source_request_path"))
            prompt_set_path = normalized_manifest_path(gate.get("prompt_set_path"))
            source_prompt_set_path = normalized_manifest_path(gate.get("source_prompt_set_path"))
            if source_request_path != request_path:
                gate_missing_source_request_path_count += 1
            if not prompt_set_path or source_prompt_set_path != prompt_set_path:
                gate_missing_prompt_set_path_count += 1
            if gate.get("requires_explicit_user_approval") is not True:
                gate_missing_explicit_approval_count += 1
            if gate.get("approval_records_prompt_traffic") is not True or gate.get("may_send_prompt_traffic_after_approval") is not True:
                gate_missing_prompt_traffic_ack_count += 1
        if request.get("ready_to_update_bundle_after_current_intake") is True:
            row_ready_to_update_count += 1
    for count_key, row_value, error_name in (
        ("artifact_gate_count", row_artifact_gate_count, "post_capture_intake_runbook_artifact_gate_count_mismatch"),
        ("ready_after_current_intake_count", row_ready_count, "post_capture_intake_runbook_ready_count_mismatch"),
        ("missing_after_current_intake_count", row_missing_count, "post_capture_intake_runbook_missing_count_mismatch"),
        ("validator_command_count", row_validator_count, "post_capture_intake_runbook_validator_count_mismatch"),
        ("ready_to_update_bundle_count", row_ready_to_update_count, "post_capture_intake_runbook_ready_to_update_count_mismatch"),
        ("missing_source_request_path_count", row_missing_source_request_path_count, "post_capture_intake_runbook_missing_source_request_path_count_mismatch"),
        ("missing_prompt_set_path_count", row_missing_prompt_set_path_count, "post_capture_intake_runbook_missing_prompt_set_path_count_mismatch"),
        ("missing_explicit_approval_count", row_missing_explicit_approval_count, "post_capture_intake_runbook_missing_explicit_approval_count_mismatch"),
        ("missing_prompt_traffic_ack_count", row_missing_prompt_traffic_ack_count, "post_capture_intake_runbook_missing_prompt_traffic_ack_count_mismatch"),
    ):
        if counts[count_key] != row_value:
            errors.append(error_name)

    for count_key, gate_value, error_name in (
        ("missing_source_request_path_count", gate_missing_source_request_path_count, "post_capture_intake_runbook_gate_missing_source_request_path_count_mismatch"),
        ("missing_prompt_set_path_count", gate_missing_prompt_set_path_count, "post_capture_intake_runbook_gate_missing_prompt_set_path_count_mismatch"),
        ("missing_explicit_approval_count", gate_missing_explicit_approval_count, "post_capture_intake_runbook_gate_missing_explicit_approval_count_mismatch"),
        ("missing_prompt_traffic_ack_count", gate_missing_prompt_traffic_ack_count, "post_capture_intake_runbook_gate_missing_prompt_traffic_ack_count_mismatch"),
    ):
        if counts[count_key] != gate_value:
            errors.append(error_name)

    missing_items = runbook.get("missing_items") if isinstance(runbook.get("missing_items"), list) else []
    if counts["missing_item_count"] != len(missing_items):
        errors.append("post_capture_intake_runbook_missing_item_count_mismatch")
    gate_keys = post_capture_intake_runbook_task_keys(runbook)
    manual_keys = manual_capture_runbook_task_keys(manual_runbook, include_runtime_command_tasks=True)
    receipt_keys = receipt_command_task_keys(receipt_command_manifest)
    if manual_keys and gate_keys != manual_keys:
        errors.append("post_capture_intake_runbook_manual_task_mismatch")
    if gate_keys and not gate_keys.issubset(receipt_keys):
        errors.append("post_capture_intake_runbook_receipt_command_task_mismatch")
    receipt_validator_count = count_validator_commands(receipt_command_manifest)
    if receipt_validator_count and counts["validator_command_count"] != receipt_validator_count:
        errors.append("post_capture_intake_runbook_receipt_validator_count_mismatch")

    for count_key, gap_key, error_name in (
        ("ready", "all_request_post_capture_intake_runbook_ready", "post_capture_intake_runbook_gap_ready_mismatch"),
        ("request_count", "all_request_post_capture_intake_runbook_request_count", "post_capture_intake_runbook_gap_request_count_mismatch"),
        ("artifact_gate_count", "all_request_post_capture_intake_runbook_artifact_gate_count", "post_capture_intake_runbook_gap_artifact_gate_count_mismatch"),
        ("ready_after_current_intake_count", "all_request_post_capture_intake_runbook_ready_after_current_intake_count", "post_capture_intake_runbook_gap_ready_count_mismatch"),
        ("missing_after_current_intake_count", "all_request_post_capture_intake_runbook_missing_after_current_intake_count", "post_capture_intake_runbook_gap_missing_count_mismatch"),
        ("validator_command_count", "all_request_post_capture_intake_runbook_validator_command_count", "post_capture_intake_runbook_gap_validator_count_mismatch"),
        ("ready_to_update_bundle_count", "all_request_post_capture_intake_runbook_ready_to_update_bundle_count", "post_capture_intake_runbook_gap_ready_to_update_count_mismatch"),
        ("phase4_candidate_count", "all_request_post_capture_intake_runbook_phase4_candidate_count", "post_capture_intake_runbook_gap_phase4_count_mismatch"),
        ("live_spike_candidate_count", "all_request_post_capture_intake_runbook_live_spike_candidate_count", "post_capture_intake_runbook_gap_live_spike_count_mismatch"),
        ("missing_source_request_path_count", "all_request_post_capture_intake_runbook_missing_source_request_path_count", "post_capture_intake_runbook_gap_missing_source_request_path_count_mismatch"),
        ("missing_prompt_set_path_count", "all_request_post_capture_intake_runbook_missing_prompt_set_path_count", "post_capture_intake_runbook_gap_missing_prompt_set_path_count_mismatch"),
        ("missing_explicit_approval_count", "all_request_post_capture_intake_runbook_missing_explicit_approval_count", "post_capture_intake_runbook_gap_missing_explicit_approval_count_mismatch"),
        ("missing_prompt_traffic_ack_count", "all_request_post_capture_intake_runbook_missing_prompt_traffic_ack_count", "post_capture_intake_runbook_gap_missing_prompt_traffic_ack_count_mismatch"),
        ("missing_item_count", "all_request_post_capture_intake_runbook_missing_item_count", "post_capture_intake_runbook_gap_missing_item_count_mismatch"),
    ):
        expected = remaining_gaps.get(gap_key)
        if isinstance(expected, bool) and counts[count_key] is not expected:
            errors.append(error_name)
        elif isinstance(expected, int) and counts[count_key] != expected:
            errors.append(error_name)
    for count_key, handoff_key, error_name in (
        ("ready", "post_capture_intake_runbook_ready", "post_capture_intake_runbook_handoff_ready_mismatch"),
        ("request_count", "post_capture_intake_runbook_request_count", "post_capture_intake_runbook_handoff_request_count_mismatch"),
        ("artifact_gate_count", "post_capture_intake_runbook_artifact_gate_count", "post_capture_intake_runbook_handoff_artifact_gate_count_mismatch"),
        ("ready_after_current_intake_count", "post_capture_intake_runbook_ready_after_current_intake_count", "post_capture_intake_runbook_handoff_ready_count_mismatch"),
        ("missing_after_current_intake_count", "post_capture_intake_runbook_missing_after_current_intake_count", "post_capture_intake_runbook_handoff_missing_count_mismatch"),
        ("validator_command_count", "post_capture_intake_runbook_validator_command_count", "post_capture_intake_runbook_handoff_validator_count_mismatch"),
        ("ready_to_update_bundle_count", "post_capture_intake_runbook_ready_to_update_bundle_count", "post_capture_intake_runbook_handoff_ready_to_update_count_mismatch"),
        ("phase4_candidate_count", "post_capture_intake_runbook_phase4_candidate_count", "post_capture_intake_runbook_handoff_phase4_count_mismatch"),
        ("live_spike_candidate_count", "post_capture_intake_runbook_live_spike_candidate_count", "post_capture_intake_runbook_handoff_live_spike_count_mismatch"),
        ("missing_source_request_path_count", "post_capture_intake_runbook_missing_source_request_path_count", "post_capture_intake_runbook_handoff_missing_source_request_path_count_mismatch"),
        ("missing_prompt_set_path_count", "post_capture_intake_runbook_missing_prompt_set_path_count", "post_capture_intake_runbook_handoff_missing_prompt_set_path_count_mismatch"),
        ("missing_explicit_approval_count", "post_capture_intake_runbook_missing_explicit_approval_count", "post_capture_intake_runbook_handoff_missing_explicit_approval_count_mismatch"),
        ("missing_prompt_traffic_ack_count", "post_capture_intake_runbook_missing_prompt_traffic_ack_count", "post_capture_intake_runbook_handoff_missing_prompt_traffic_ack_count_mismatch"),
        ("missing_item_count", "post_capture_intake_runbook_missing_item_count", "post_capture_intake_runbook_handoff_missing_item_count_mismatch"),
    ):
        expected = handoff_counts.get(handoff_key)
        if isinstance(expected, bool) and counts[count_key] is not expected:
            errors.append(error_name)
        elif isinstance(expected, int) and counts[count_key] != expected:
            errors.append(error_name)
    return errors, counts


def selected_source_rows(source_runbook: Any, request_path: str | None) -> list[JSONDict]:
    if not isinstance(source_runbook, dict) or not request_path:
        return []
    rows = source_runbook.get("requests")
    if not isinstance(rows, list):
        return []
    return [
        item
        for item in rows
        if isinstance(item, dict)
        and normalized_manifest_path(item.get("request_path") or item.get("path")) == request_path
    ]


def recommended_manual_capture_runbook_errors(
    runbook: Any,
    packet: JSONDict,
    source_runbook: Any,
    recommended_request: JSONDict,
    handoff_counts: JSONDict,
) -> tuple[list[str], JSONDict]:
    errors: list[str] = []
    counts: JSONDict = {
        "ready": False,
        "request_count": 0,
        "manual_task_count": 0,
        "runtime_command_task_count": 0,
        "validator_command_count": 0,
        "missing_item_count": 0,
    }
    if not isinstance(runbook, dict):
        return ["recommended_manual_capture_runbook_not_object"], counts
    packet_runbook = packet.get("recommended_manual_capture_runbook_manifest")
    if not isinstance(packet_runbook, dict) or not packet_runbook:
        errors.append("recommended_manual_capture_runbook_packet_missing")
    elif runbook != packet_runbook:
        errors.append("recommended_manual_capture_runbook_packet_mismatch")

    request_path = normalized_manifest_path(recommended_request.get("request_path"))
    selected_path = normalized_manifest_path(runbook.get("request_path"))
    requests = runbook.get("requests") if isinstance(runbook.get("requests"), list) else []
    request_rows = [item for item in requests if isinstance(item, dict)]
    source_rows = selected_source_rows(source_runbook, request_path)
    counts.update(
        {
            "ready": runbook.get("ready") is True,
            "request_count": int_count(runbook.get("request_count")),
            "manual_task_count": int_count(runbook.get("manual_task_count")),
            "runtime_command_task_count": int_count(runbook.get("runtime_command_task_count")),
            "validator_command_count": int_count(runbook.get("validator_command_count")),
            "missing_item_count": int_count(runbook.get("missing_item_count")),
        }
    )
    if runbook.get("selected") is not True:
        errors.append("recommended_manual_capture_runbook_not_selected")
    if runbook.get("metadata_only") is not True:
        errors.append("recommended_manual_capture_runbook_not_metadata_only")
    if runbook.get("ready") is not True:
        errors.append("recommended_manual_capture_runbook_not_ready")
    if selected_path != request_path:
        errors.append("recommended_manual_capture_runbook_request_path_mismatch")
    if counts["request_count"] != 1 or int_count(runbook.get("expected_request_count")) != 1 or len(request_rows) != 1:
        errors.append("recommended_manual_capture_runbook_request_count_mismatch")
    if int_count(runbook.get("source_request_count")) != int_count(source_runbook.get("request_count")) if isinstance(source_runbook, dict) else True:
        errors.append("recommended_manual_capture_runbook_source_request_count_mismatch")
    if len(source_rows) != 1:
        errors.append("recommended_manual_capture_runbook_source_row_mismatch")
    else:
        source_row = source_rows[0]
        if runbook.get("request") != source_row:
            errors.append("recommended_manual_capture_runbook_request_row_mismatch")
        if request_rows != [source_row]:
            errors.append("recommended_manual_capture_runbook_requests_mismatch")
        expected_counts = {
            "manual_task_count": int_count(source_row.get("manual_task_count")),
            "runtime_command_task_count": int_count(source_row.get("runtime_command_task_count")),
            "validator_command_count": int_count(source_row.get("validator_command_count")),
            "missing_item_count": int_count(source_row.get("missing_item_count")),
        }
        for key, expected in expected_counts.items():
            if counts[key] != expected:
                errors.append(f"recommended_manual_capture_runbook_{key}_mismatch")
    for count_key, handoff_key, error_name in (
        ("ready", "recommended_manual_runbook_ready", "recommended_manual_capture_runbook_handoff_ready_mismatch"),
        ("request_count", "recommended_manual_runbook_request_count", "recommended_manual_capture_runbook_handoff_request_count_mismatch"),
        ("manual_task_count", "recommended_manual_runbook_manual_task_count", "recommended_manual_capture_runbook_handoff_manual_task_count_mismatch"),
        ("runtime_command_task_count", "recommended_manual_runbook_runtime_command_task_count", "recommended_manual_capture_runbook_handoff_runtime_task_count_mismatch"),
        ("validator_command_count", "recommended_manual_runbook_validator_command_count", "recommended_manual_capture_runbook_handoff_validator_count_mismatch"),
        ("missing_item_count", "recommended_manual_runbook_missing_item_count", "recommended_manual_capture_runbook_handoff_missing_item_count_mismatch"),
    ):
        expected = handoff_counts.get(handoff_key)
        if isinstance(expected, bool) and counts[count_key] is not expected:
            errors.append(error_name)
        elif isinstance(expected, int) and counts[count_key] != expected:
            errors.append(error_name)
    return errors, counts


def recommended_post_capture_intake_runbook_errors(
    runbook: Any,
    packet: JSONDict,
    source_runbook: Any,
    selected_manual_runbook: Any,
    recommended_request: JSONDict,
    handoff_counts: JSONDict,
) -> tuple[list[str], JSONDict]:
    errors: list[str] = []
    counts: JSONDict = {
        "ready": False,
        "request_count": 0,
        "artifact_gate_count": 0,
        "ready_after_current_intake_count": 0,
        "missing_after_current_intake_count": 0,
        "validator_command_count": 0,
        "ready_to_update_bundle_count": 0,
        "missing_item_count": 0,
    }
    if not isinstance(runbook, dict):
        return ["recommended_post_capture_intake_runbook_not_object"], counts
    packet_runbook = packet.get("recommended_post_capture_intake_runbook_manifest")
    if not isinstance(packet_runbook, dict) or not packet_runbook:
        errors.append("recommended_post_capture_intake_runbook_packet_missing")
    elif runbook != packet_runbook:
        errors.append("recommended_post_capture_intake_runbook_packet_mismatch")

    request_path = normalized_manifest_path(recommended_request.get("request_path"))
    selected_path = normalized_manifest_path(runbook.get("request_path"))
    manual_selected_path = (
        normalized_manifest_path(selected_manual_runbook.get("request_path"))
        if isinstance(selected_manual_runbook, dict)
        else None
    )
    requests = runbook.get("requests") if isinstance(runbook.get("requests"), list) else []
    request_rows = [item for item in requests if isinstance(item, dict)]
    source_rows = selected_source_rows(source_runbook, request_path)
    counts.update(
        {
            "ready": runbook.get("ready") is True,
            "request_count": int_count(runbook.get("request_count")),
            "artifact_gate_count": int_count(runbook.get("artifact_gate_count")),
            "ready_after_current_intake_count": int_count(runbook.get("ready_after_current_intake_count")),
            "missing_after_current_intake_count": int_count(runbook.get("missing_after_current_intake_count")),
            "validator_command_count": int_count(runbook.get("validator_command_count")),
            "ready_to_update_bundle_count": int_count(runbook.get("ready_to_update_bundle_count")),
            "missing_item_count": int_count(runbook.get("missing_item_count")),
        }
    )
    if runbook.get("selected") is not True:
        errors.append("recommended_post_capture_intake_runbook_not_selected")
    if runbook.get("metadata_only") is not True:
        errors.append("recommended_post_capture_intake_runbook_not_metadata_only")
    if runbook.get("ready") is not True:
        errors.append("recommended_post_capture_intake_runbook_not_ready")
    if selected_path != request_path:
        errors.append("recommended_post_capture_intake_runbook_request_path_mismatch")
    if manual_selected_path and manual_selected_path != selected_path:
        errors.append("recommended_post_capture_intake_runbook_manual_request_path_mismatch")
    if counts["request_count"] != 1 or int_count(runbook.get("expected_request_count")) != 1 or len(request_rows) != 1:
        errors.append("recommended_post_capture_intake_runbook_request_count_mismatch")
    if int_count(runbook.get("source_request_count")) != int_count(source_runbook.get("request_count")) if isinstance(source_runbook, dict) else True:
        errors.append("recommended_post_capture_intake_runbook_source_request_count_mismatch")
    if len(source_rows) != 1:
        errors.append("recommended_post_capture_intake_runbook_source_row_mismatch")
    else:
        source_row = source_rows[0]
        if runbook.get("request") != source_row:
            errors.append("recommended_post_capture_intake_runbook_request_row_mismatch")
        if request_rows != [source_row]:
            errors.append("recommended_post_capture_intake_runbook_requests_mismatch")
        expected_counts = {
            "artifact_gate_count": int_count(source_row.get("artifact_gate_count")),
            "ready_after_current_intake_count": int_count(source_row.get("ready_after_current_intake_count")),
            "missing_after_current_intake_count": int_count(source_row.get("missing_after_current_intake_count")),
            "validator_command_count": int_count(source_row.get("validator_command_count")),
            "ready_to_update_bundle_count": 1 if source_row.get("ready_to_update_bundle_after_current_intake") is True else 0,
            "missing_item_count": int_count(source_row.get("missing_source_request_path_count"))
            + int_count(source_row.get("missing_prompt_set_path_count"))
            + int_count(source_row.get("missing_explicit_approval_count"))
            + int_count(source_row.get("missing_prompt_traffic_ack_count")),
        }
        for key, expected in expected_counts.items():
            if counts[key] != expected:
                errors.append(f"recommended_post_capture_intake_runbook_{key}_mismatch")
    for count_key, handoff_key, error_name in (
        ("ready", "recommended_post_capture_intake_runbook_ready", "recommended_post_capture_intake_runbook_handoff_ready_mismatch"),
        ("request_count", "recommended_post_capture_intake_runbook_request_count", "recommended_post_capture_intake_runbook_handoff_request_count_mismatch"),
        ("artifact_gate_count", "recommended_post_capture_intake_runbook_artifact_gate_count", "recommended_post_capture_intake_runbook_handoff_artifact_gate_count_mismatch"),
        ("ready_after_current_intake_count", "recommended_post_capture_intake_runbook_ready_after_current_intake_count", "recommended_post_capture_intake_runbook_handoff_ready_count_mismatch"),
        ("missing_after_current_intake_count", "recommended_post_capture_intake_runbook_missing_after_current_intake_count", "recommended_post_capture_intake_runbook_handoff_missing_count_mismatch"),
        ("validator_command_count", "recommended_post_capture_intake_runbook_validator_command_count", "recommended_post_capture_intake_runbook_handoff_validator_count_mismatch"),
        ("ready_to_update_bundle_count", "recommended_post_capture_intake_runbook_ready_to_update_bundle_count", "recommended_post_capture_intake_runbook_handoff_ready_to_update_count_mismatch"),
        ("missing_item_count", "recommended_post_capture_intake_runbook_missing_item_count", "recommended_post_capture_intake_runbook_handoff_missing_item_count_mismatch"),
    ):
        expected = handoff_counts.get(handoff_key)
        if isinstance(expected, bool) and counts[count_key] is not expected:
            errors.append(error_name)
        elif isinstance(expected, int) and counts[count_key] != expected:
            errors.append(error_name)
    return errors, counts

def runtime_capture_work_order_task_keys(work_order: Any) -> set[tuple[str, str, str, str]]:
    if not isinstance(work_order, dict):
        return set()
    steps = work_order.get("capture_steps")
    if not isinstance(steps, list):
        return set()
    keys: set[tuple[str, str, str, str]] = set()
    for step in steps:
        if not isinstance(step, dict):
            continue
        key = _task_tuple(
            step.get("request_path") or work_order.get("request_path"),
            step.get("artifact_id"),
            step.get("artifact_path"),
            step.get("receipt_path"),
        )
        if key:
            keys.add(key)
    return keys


def recommended_runtime_capture_work_order_errors(
    work_order: Any,
    packet: JSONDict,
    recommended_request: JSONDict,
    selected_manual_runbook: Any,
    selected_post_capture_runbook: Any,
    handoff_counts: JSONDict,
) -> tuple[list[str], JSONDict]:
    errors: list[str] = []
    counts: JSONDict = {
        "ready": False,
        "capture_step_count": 0,
        "artifact_gate_count": 0,
        "validator_command_count": 0,
        "missing_item_count": 0,
        "approval_command_ready": False,
        "intake_command_ready": False,
        "post_capture_sequence_ready": False,
        "post_capture_sequence_step_count": 0,
        "completion_validation_before_intake": False,
        "completion_validation_command_ready": False,
    }
    if not isinstance(work_order, dict):
        return ["recommended_runtime_capture_work_order_not_object"], counts
    packet_work_order = packet.get("recommended_runtime_capture_work_order_manifest")
    if not isinstance(packet_work_order, dict) or not packet_work_order:
        errors.append("recommended_runtime_capture_work_order_packet_missing")
    elif work_order != packet_work_order:
        errors.append("recommended_runtime_capture_work_order_packet_mismatch")

    request_path = normalized_manifest_path(recommended_request.get("request_path"))
    selected_path = normalized_manifest_path(work_order.get("request_path"))
    approval_step = work_order.get("approval_step") if isinstance(work_order.get("approval_step"), dict) else {}
    intake_step = work_order.get("intake_step") if isinstance(work_order.get("intake_step"), dict) else {}
    capture_steps = work_order.get("capture_steps") if isinstance(work_order.get("capture_steps"), list) else []
    post_capture_gates = work_order.get("post_capture_gates") if isinstance(work_order.get("post_capture_gates"), list) else []
    post_capture_sequence = work_order.get("post_capture_sequence") if isinstance(work_order.get("post_capture_sequence"), list) else []
    capture_rows = [item for item in capture_steps if isinstance(item, dict)]
    gate_rows = [item for item in post_capture_gates if isinstance(item, dict)]
    sequence_rows = [item for item in post_capture_sequence if isinstance(item, dict)]
    sequence_ids = [item.get("step_id") for item in sequence_rows]
    required_sequence_ids = [
        "record_runtime_approvals",
        "capture_runtime_artifacts",
        "fill_completion_receipt",
        "validate_completion_receipt",
        "run_capture_result_intake",
    ]
    approval_command = command_tokens(approval_step.get("command"))
    intake_command = command_tokens(intake_step.get("command"))
    validation_step = work_order.get("post_capture_validation_step") if isinstance(work_order.get("post_capture_validation_step"), dict) else {}
    validation_step_errors, validation_step_counts = completion_receipt_validation_step_errors(
        validation_step,
        prefix="recommended_runtime_capture_work_order_completion_receipt",
    )
    errors.extend(validation_step_errors)
    validation_sequence_step = next((item for item in sequence_rows if item.get("step_id") == "validate_completion_receipt"), {})
    intake_sequence_step = next((item for item in sequence_rows if item.get("step_id") == "run_capture_result_intake"), {})
    validator_command_count = sum(int_count(item.get("validator_command_count")) for item in capture_rows)
    counts.update(
        {
            "ready": work_order.get("ready") is True,
            "capture_step_count": int_count(work_order.get("capture_step_count")),
            "artifact_gate_count": int_count(work_order.get("artifact_gate_count")),
            "validator_command_count": int_count(work_order.get("validator_command_count")),
            "missing_item_count": int_count(work_order.get("missing_item_count")),
            "approval_command_ready": bool(approval_command) and approval_step.get("requires_explicit_user_approval") is True,
            "intake_command_ready": bool(intake_command) and "scripts/plan_phase3_capture_result_intake.py" in intake_command,
            "post_capture_sequence_ready": sequence_ids == required_sequence_ids,
            "post_capture_sequence_step_count": int_count(work_order.get("post_capture_sequence_step_count")),
            "completion_validation_before_intake": (
                "validate_completion_receipt" in sequence_ids
                and "run_capture_result_intake" in sequence_ids
                and sequence_ids.index("validate_completion_receipt") < sequence_ids.index("run_capture_result_intake")
            ),
            "completion_validation_command_ready": validation_step_counts.get("validation_command_ready") is True,
        }
    )
    if work_order.get("schema_version") != "moe-phase3-recommended-runtime-capture-work-order-v1":
        errors.append("recommended_runtime_capture_work_order_schema_mismatch")
    for key, expected in (
        ("selected", True),
        ("metadata_only", True),
        ("launches_runtimes", False),
        ("runs_docker", False),
        ("sends_prompt_traffic", False),
        ("reads_private_tokens", False),
        ("mutates_runtime_residency", False),
        ("execution_requires_explicit_approval", True),
    ):
        if work_order.get(key) is not expected:
            errors.append(f"recommended_runtime_capture_work_order_safety_flag_mismatch:{key}")
    if work_order.get("ready") is not True:
        errors.append("recommended_runtime_capture_work_order_not_ready")
    if selected_path != request_path:
        errors.append("recommended_runtime_capture_work_order_request_path_mismatch")
    if approval_command != command_tokens(recommended_request.get("approval_command")):
        errors.append("recommended_runtime_capture_work_order_approval_command_mismatch")
    if counts["capture_step_count"] != len(capture_rows):
        errors.append("recommended_runtime_capture_work_order_capture_step_count_mismatch")
    if counts["artifact_gate_count"] != len(gate_rows):
        errors.append("recommended_runtime_capture_work_order_artifact_gate_count_mismatch")
    if counts["validator_command_count"] != validator_command_count:
        errors.append("recommended_runtime_capture_work_order_validator_count_mismatch")
    if not counts["approval_command_ready"]:
        errors.append("recommended_runtime_capture_work_order_approval_command_not_ready")
    if not counts["intake_command_ready"]:
        errors.append("recommended_runtime_capture_work_order_intake_command_not_ready")
    if counts["post_capture_sequence_step_count"] != len(sequence_rows):
        errors.append("recommended_runtime_capture_work_order_post_capture_sequence_count_mismatch")
    if not counts["post_capture_sequence_ready"]:
        errors.append("recommended_runtime_capture_work_order_post_capture_sequence_mismatch")
    if not counts["completion_validation_before_intake"]:
        errors.append("recommended_runtime_capture_work_order_completion_validation_not_before_intake")
    if not counts["completion_validation_command_ready"]:
        errors.append("recommended_runtime_capture_work_order_completion_validation_command_not_ready")
    if command_tokens(validation_sequence_step.get("command")) != command_tokens(validation_step.get("command")):
        errors.append("recommended_runtime_capture_work_order_validation_sequence_command_mismatch")
    if command_tokens(intake_sequence_step.get("command")) != intake_command:
        errors.append("recommended_runtime_capture_work_order_intake_sequence_command_mismatch")
    if intake_step.get("requires_completion_receipt_validation") is not True:
        errors.append("recommended_runtime_capture_work_order_intake_missing_completion_validation_gate")
    manual_keys = manual_capture_runbook_task_keys(selected_manual_runbook, include_runtime_command_tasks=True)
    post_capture_keys = post_capture_intake_runbook_task_keys(selected_post_capture_runbook)
    work_order_keys = runtime_capture_work_order_task_keys(work_order)
    if manual_keys and work_order_keys != manual_keys:
        errors.append("recommended_runtime_capture_work_order_manual_task_mismatch")
    if post_capture_keys and work_order_keys != post_capture_keys:
        errors.append("recommended_runtime_capture_work_order_post_capture_gate_mismatch")
    if isinstance(selected_manual_runbook, dict):
        expected_capture_count = int_count(selected_manual_runbook.get("manual_task_count")) + int_count(selected_manual_runbook.get("runtime_command_task_count"))
        if counts["capture_step_count"] != expected_capture_count:
            errors.append("recommended_runtime_capture_work_order_manual_count_mismatch")
    if isinstance(selected_post_capture_runbook, dict):
        if counts["artifact_gate_count"] != int_count(selected_post_capture_runbook.get("artifact_gate_count")):
            errors.append("recommended_runtime_capture_work_order_selected_gate_count_mismatch")
        if counts["validator_command_count"] != int_count(selected_post_capture_runbook.get("validator_command_count")):
            errors.append("recommended_runtime_capture_work_order_selected_validator_count_mismatch")
    for count_key, handoff_key, error_name in (
        ("ready", "recommended_work_order_ready", "recommended_runtime_capture_work_order_handoff_ready_mismatch"),
        ("capture_step_count", "recommended_work_order_capture_step_count", "recommended_runtime_capture_work_order_handoff_capture_step_count_mismatch"),
        ("artifact_gate_count", "recommended_work_order_artifact_gate_count", "recommended_runtime_capture_work_order_handoff_artifact_gate_count_mismatch"),
        ("validator_command_count", "recommended_work_order_validator_command_count", "recommended_runtime_capture_work_order_handoff_validator_count_mismatch"),
        ("missing_item_count", "recommended_work_order_missing_item_count", "recommended_runtime_capture_work_order_handoff_missing_item_count_mismatch"),
        ("approval_command_ready", "recommended_work_order_approval_command_ready", "recommended_runtime_capture_work_order_handoff_approval_command_ready_mismatch"),
        ("intake_command_ready", "recommended_work_order_intake_command_ready", "recommended_runtime_capture_work_order_handoff_intake_command_ready_mismatch"),
        ("post_capture_sequence_ready", "recommended_work_order_post_capture_sequence_ready", "recommended_runtime_capture_work_order_handoff_post_capture_sequence_ready_mismatch"),
        ("post_capture_sequence_step_count", "recommended_work_order_post_capture_sequence_step_count", "recommended_runtime_capture_work_order_handoff_post_capture_sequence_step_count_mismatch"),
        ("completion_validation_before_intake", "recommended_work_order_completion_validation_before_intake", "recommended_runtime_capture_work_order_handoff_completion_validation_before_intake_mismatch"),
        ("completion_validation_command_ready", "recommended_work_order_completion_validation_command_ready", "recommended_runtime_capture_work_order_handoff_completion_validation_command_ready_mismatch"),
    ):
        expected = handoff_counts.get(handoff_key)
        if isinstance(expected, bool) and counts[count_key] is not expected:
            errors.append(error_name)
        elif isinstance(expected, int) and counts[count_key] != expected:
            errors.append(error_name)
    return errors, counts

def completion_receipt_validation_step_errors(step: Any, *, prefix: str = "recommended_completion_receipt") -> tuple[list[str], JSONDict]:
    errors: list[str] = []
    counts: JSONDict = {
        "validation_command_ready": False,
        "validation_command_token_count": 0,
    }
    if not isinstance(step, dict):
        return [f"{prefix}_validation_step_missing"], counts
    command = command_tokens(step.get("command"))
    counts["validation_command_token_count"] = len(command)
    if step.get("command_class") != COMPLETION_RECEIPT_VALIDATION_COMMAND_CLASS:
        errors.append(f"{prefix}_validation_command_class_mismatch")
    for key, expected in (
        ("metadata_only", True),
        ("launches_runtimes", False),
        ("runs_docker", False),
        ("sends_prompt_traffic", False),
        ("reads_private_tokens", False),
        ("mutates_runtime_residency", False),
        ("requires_filled_completion_receipt_for_intake", True),
    ):
        if step.get(key) is not expected:
            errors.append(f"{prefix}_validation_safety_flag_mismatch:{key}")
    if not command:
        errors.append(f"{prefix}_validation_command_missing")
    else:
        if COMPLETION_RECEIPT_VALIDATION_SCRIPT not in command:
            errors.append(f"{prefix}_validation_script_missing")
        if "--work-order" not in command:
            errors.append(f"{prefix}_validation_work_order_arg_missing")
        elif command.index("--work-order") + 1 >= len(command):
            errors.append(f"{prefix}_validation_work_order_path_missing")
        if not any(COMPLETION_RECEIPT_TEMPLATE_FILE in token for token in command):
            errors.append(f"{prefix}_validation_receipt_path_missing")
        if not any(COMPLETION_RECEIPT_WORK_ORDER_FILE in token for token in command):
            errors.append(f"{prefix}_validation_work_order_path_mismatch")
    counts["validation_command_ready"] = not errors
    return errors, counts

def runtime_capture_completion_receipt_task_keys(template: Any) -> set[tuple[str, str, str, str]]:
    if not isinstance(template, dict):
        return set()
    rows = template.get("capture_receipts")
    if not isinstance(rows, list):
        return set()
    keys: set[tuple[str, str, str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = _task_tuple(
            row.get("request_path") or template.get("request_path"),
            row.get("artifact_id"),
            row.get("artifact_path"),
            row.get("receipt_path"),
        )
        if key:
            keys.add(key)
    return keys


def recommended_runtime_capture_completion_receipt_template_errors(
    template: Any,
    packet: JSONDict,
    work_order: Any,
    recommended_request: JSONDict,
    handoff_counts: JSONDict,
) -> tuple[list[str], JSONDict]:
    errors: list[str] = []
    counts: JSONDict = {
        "template_ready": False,
        "receipt_complete": False,
        "ready_for_intake": False,
        "capture_receipt_count": 0,
        "expected_capture_step_count": 0,
        "validator_command_count": 0,
        "missing_item_count": 0,
        "validation_command_ready": False,
        "validation_command_token_count": 0,
    }
    if not isinstance(template, dict):
        return ["recommended_completion_receipt_template_not_object"], counts
    packet_template = packet.get("recommended_runtime_capture_completion_receipt_template_manifest")
    if not isinstance(packet_template, dict) or not packet_template:
        errors.append("recommended_completion_receipt_template_packet_missing")
    elif template != packet_template:
        errors.append("recommended_completion_receipt_template_packet_mismatch")

    request_path = normalized_manifest_path(recommended_request.get("request_path"))
    selected_path = normalized_manifest_path(template.get("request_path"))
    rows = template.get("capture_receipts") if isinstance(template.get("capture_receipts"), list) else []
    receipt_rows = [item for item in rows if isinstance(item, dict)]
    validator_command_count = sum(int_count(item.get("expected_validator_command_count")) for item in receipt_rows)
    counts.update(
        {
            "template_ready": template.get("template_ready") is True,
            "receipt_complete": template.get("receipt_complete") is True,
            "ready_for_intake": template.get("ready_for_capture_result_intake") is True,
            "capture_receipt_count": int_count(template.get("capture_receipt_count")),
            "expected_capture_step_count": int_count(template.get("expected_capture_step_count")),
            "validator_command_count": int_count(template.get("validator_command_count")),
            "missing_item_count": int_count(template.get("missing_item_count")),
        }
    )
    validation_step_errors, validation_step_counts = completion_receipt_validation_step_errors(
        template.get("completion_receipt_validation_step")
    )
    errors.extend(validation_step_errors)
    counts.update(validation_step_counts)
    if template.get("schema_version") != "moe-phase3-recommended-runtime-capture-completion-receipt-template-v1":
        errors.append("recommended_completion_receipt_template_schema_mismatch")
    for key, expected in (
        ("selected", True),
        ("metadata_only", True),
        ("launches_runtimes", False),
        ("runs_docker", False),
        ("sends_prompt_traffic", False),
        ("reads_private_tokens", False),
        ("mutates_runtime_residency", False),
        ("execution_requires_explicit_approval", True),
    ):
        if template.get(key) is not expected:
            errors.append(f"recommended_completion_receipt_template_safety_flag_mismatch:{key}")
    if template.get("template_ready") is not True or template.get("ready") is not True:
        errors.append("recommended_completion_receipt_template_not_ready")
    if template.get("receipt_complete") is not False:
        errors.append("recommended_completion_receipt_template_unexpectedly_complete")
    if template.get("ready_for_capture_result_intake") is not False:
        errors.append("recommended_completion_receipt_template_unexpectedly_ready_for_intake")
    if selected_path != request_path:
        errors.append("recommended_completion_receipt_template_request_path_mismatch")
    if counts["capture_receipt_count"] != len(receipt_rows):
        errors.append("recommended_completion_receipt_template_receipt_count_mismatch")
    if counts["validator_command_count"] != validator_command_count:
        errors.append("recommended_completion_receipt_template_validator_count_mismatch")
    if isinstance(work_order, dict):
        if counts["expected_capture_step_count"] != int_count(work_order.get("capture_step_count")):
            errors.append("recommended_completion_receipt_template_work_order_step_count_mismatch")
        if counts["capture_receipt_count"] != int_count(work_order.get("capture_step_count")):
            errors.append("recommended_completion_receipt_template_work_order_receipt_count_mismatch")
        if counts["validator_command_count"] != int_count(work_order.get("validator_command_count")):
            errors.append("recommended_completion_receipt_template_work_order_validator_count_mismatch")
        if command_tokens(template.get("intake_step", {}).get("command") if isinstance(template.get("intake_step"), dict) else []) != command_tokens(work_order.get("intake_step", {}).get("command") if isinstance(work_order.get("intake_step"), dict) else []):
            errors.append("recommended_completion_receipt_template_intake_command_mismatch")
    template_keys = runtime_capture_completion_receipt_task_keys(template)
    work_order_keys = runtime_capture_work_order_task_keys(work_order)
    if work_order_keys and template_keys != work_order_keys:
        errors.append("recommended_completion_receipt_template_work_order_task_mismatch")
    for row in receipt_rows:
        if row.get("capture_complete") is not False or row.get("receipt_filled") is not False or row.get("validator_passed") is not False or row.get("ready_for_intake") is not False:
            errors.append("recommended_completion_receipt_template_row_unexpectedly_complete")
            break
    for count_key, handoff_key, error_name in (
        ("template_ready", "recommended_completion_receipt_template_ready", "recommended_completion_receipt_template_handoff_ready_mismatch"),
        ("capture_receipt_count", "recommended_completion_receipt_capture_receipt_count", "recommended_completion_receipt_template_handoff_receipt_count_mismatch"),
        ("expected_capture_step_count", "recommended_completion_receipt_expected_capture_step_count", "recommended_completion_receipt_template_handoff_expected_step_count_mismatch"),
        ("validator_command_count", "recommended_completion_receipt_validator_command_count", "recommended_completion_receipt_template_handoff_validator_count_mismatch"),
        ("missing_item_count", "recommended_completion_receipt_missing_item_count", "recommended_completion_receipt_template_handoff_missing_item_count_mismatch"),
        ("receipt_complete", "recommended_completion_receipt_receipt_complete", "recommended_completion_receipt_template_handoff_receipt_complete_mismatch"),
        ("ready_for_intake", "recommended_completion_receipt_ready_for_intake", "recommended_completion_receipt_template_handoff_ready_for_intake_mismatch"),
        ("validation_command_ready", "recommended_completion_receipt_validation_command_ready", "recommended_completion_receipt_validation_command_handoff_mismatch"),
    ):
        expected = handoff_counts.get(handoff_key)
        if isinstance(expected, bool) and counts[count_key] is not expected:
            errors.append(error_name)
        elif isinstance(expected, int) and counts[count_key] != expected:
            errors.append(error_name)
    return errors, counts

def next_unblocked_operator_handoff_errors(
    handoff: Any,
    packet: JSONDict,
    manifest_handoff: JSONDict,
    queue: Any,
    work_order: Any,
    completion_receipt: Any,
    remaining_gaps: JSONDict,
    handoff_counts: JSONDict,
) -> tuple[list[str], JSONDict]:
    errors: list[str] = []
    counts: JSONDict = {
        "handoff_ready": False,
        "work_order_ready": False,
        "work_order_advances_next_package": False,
        "work_order_capture_step_count": 0,
        "work_order_validator_command_count": 0,
        "completion_receipt_template_ready": False,
        "completion_receipt_validation_command_ready": False,
        "work_order_bound_to_completion_receipt": False,
        "next_unblocked_work_package_id": None,
        "next_unblocked_row_count": 0,
        "next_unblocked_validator_command_count": 0,
    }
    if not isinstance(handoff, dict):
        return ["next_unblocked_operator_handoff_not_object"], counts
    packet_handoff = packet.get("phase3_next_unblocked_operator_handoff_summary") if isinstance(packet, dict) else None
    if packet_handoff != handoff:
        errors.append("next_unblocked_operator_handoff_packet_mismatch")
    if manifest_handoff != handoff:
        errors.append("next_unblocked_operator_handoff_manifest_mismatch")
    if handoff.get("schema_version") != NEXT_UNBLOCKED_HANDOFF_SCHEMA_VERSION:
        errors.append("next_unblocked_operator_handoff_schema_mismatch")
    for key, expected in (
        ("metadata_only", True),
        ("launches_runtimes", False),
        ("runs_docker", False),
        ("sends_prompt_traffic", False),
        ("reads_private_tokens", False),
        ("mutates_runtime_residency", False),
    ):
        if handoff.get(key) is not expected:
            errors.append(f"next_unblocked_operator_handoff_safety_flag_mismatch:{key}")

    queue_dict = queue if isinstance(queue, dict) else {}
    next_package = queue_dict.get("next_unblocked_work_package") if isinstance(queue_dict.get("next_unblocked_work_package"), dict) else {}
    work_order_dict = work_order if isinstance(work_order, dict) else {}
    receipt_dict = completion_receipt if isinstance(completion_receipt, dict) else {}
    counts.update(
        {
            "handoff_ready": handoff.get("handoff_ready") is True,
            "work_order_ready": handoff.get("work_order_ready") is True,
            "work_order_advances_next_package": handoff.get("work_order_advances_next_package") is True,
            "work_order_capture_step_count": int_count(handoff.get("work_order_capture_step_count")),
            "work_order_validator_command_count": int_count(handoff.get("work_order_validator_command_count")),
            "completion_receipt_template_ready": handoff.get("completion_receipt_template_ready") is True,
            "completion_receipt_validation_command_ready": handoff.get("completion_receipt_validation_command_ready") is True,
            "work_order_bound_to_completion_receipt": handoff.get("work_order_bound_to_completion_receipt") is True,
            "next_unblocked_work_package_id": handoff.get("next_unblocked_work_package_id"),
            "next_unblocked_row_count": int_count(handoff.get("next_unblocked_row_count")),
            "next_unblocked_validator_command_count": int_count(handoff.get("next_unblocked_validator_command_count")),
        }
    )
    if not counts["handoff_ready"]:
        errors.append("next_unblocked_operator_handoff_not_ready")
    if handoff.get("work_order_artifact") != COMPLETION_RECEIPT_WORK_ORDER_FILE:
        errors.append("next_unblocked_operator_handoff_work_order_artifact_mismatch")
    if handoff.get("completion_receipt_template_artifact") != COMPLETION_RECEIPT_TEMPLATE_FILE:
        errors.append("next_unblocked_operator_handoff_receipt_template_artifact_mismatch")
    if not next_package:
        errors.append("next_unblocked_operator_handoff_queue_package_missing")
    else:
        for handoff_key, package_key, error_name in (
            ("next_unblocked_work_package_id", "work_package_id", "next_unblocked_operator_handoff_package_id_mismatch"),
            ("next_unblocked_sequence_rank", "sequence_rank", "next_unblocked_operator_handoff_sequence_rank_mismatch"),
            ("next_unblocked_operator_stage", "operator_stage", "next_unblocked_operator_handoff_stage_mismatch"),
            ("next_unblocked_package_class", "package_class", "next_unblocked_operator_handoff_package_class_mismatch"),
            ("next_unblocked_row_count", "row_count", "next_unblocked_operator_handoff_row_count_mismatch"),
            ("next_unblocked_validator_command_count", "validator_command_count", "next_unblocked_operator_handoff_validator_count_mismatch"),
            ("next_unblocked_completion_gate_count", "completion_gate_count", "next_unblocked_operator_handoff_completion_gate_count_mismatch"),
        ):
            expected = next_package.get(package_key)
            actual = handoff.get(handoff_key)
            if isinstance(expected, int):
                if int_count(actual) != expected:
                    errors.append(error_name)
            elif actual != expected:
                errors.append(error_name)
    if handoff.get("work_order_request_path") != normalized_manifest_path(work_order_dict.get("request_path")):
        errors.append("next_unblocked_operator_handoff_work_order_request_path_mismatch")
    if handoff.get("work_order_next_artifact_id") != work_order_dict.get("next_artifact_id"):
        errors.append("next_unblocked_operator_handoff_work_order_next_artifact_mismatch")
    if counts["work_order_ready"] != (work_order_dict.get("ready") is True):
        errors.append("next_unblocked_operator_handoff_work_order_ready_mismatch")
    if counts["work_order_capture_step_count"] != int_count(work_order_dict.get("capture_step_count")):
        errors.append("next_unblocked_operator_handoff_work_order_capture_count_mismatch")
    if counts["work_order_validator_command_count"] != int_count(work_order_dict.get("validator_command_count")):
        errors.append("next_unblocked_operator_handoff_work_order_validator_count_mismatch")
    if not counts["work_order_advances_next_package"]:
        errors.append("next_unblocked_operator_handoff_work_order_does_not_advance_package")
    if handoff.get("next_unblocked_package_class") == "policy_candidate_replay" and handoff.get("work_order_next_artifact_id") != "candidate_router_trace":
        errors.append("next_unblocked_operator_handoff_policy_candidate_artifact_mismatch")
    if handoff.get("completion_receipt_request_path") != normalized_manifest_path(receipt_dict.get("request_path")):
        errors.append("next_unblocked_operator_handoff_receipt_request_path_mismatch")
    if counts["completion_receipt_template_ready"] != (receipt_dict.get("template_ready") is True):
        errors.append("next_unblocked_operator_handoff_receipt_template_ready_mismatch")
    validation_step = receipt_dict.get("completion_receipt_validation_step") if isinstance(receipt_dict.get("completion_receipt_validation_step"), dict) else {}
    validation_command_ready = bool(command_tokens(validation_step.get("command"))) and COMPLETION_RECEIPT_VALIDATION_SCRIPT in command_tokens(validation_step.get("command"))
    if counts["completion_receipt_validation_command_ready"] != validation_command_ready:
        errors.append("next_unblocked_operator_handoff_validation_command_ready_mismatch")
    if not counts["work_order_bound_to_completion_receipt"]:
        errors.append("next_unblocked_operator_handoff_work_order_not_bound_to_receipt")
    if handoff.get("work_order_request_path") != handoff.get("completion_receipt_request_path"):
        errors.append("next_unblocked_operator_handoff_request_path_binding_mismatch")

    gap_to_count = {
        "phase3_next_unblocked_operator_handoff_ready": "handoff_ready",
        "phase3_next_unblocked_operator_handoff_work_order_ready": "work_order_ready",
        "phase3_next_unblocked_operator_handoff_receipt_template_ready": "completion_receipt_template_ready",
        "phase3_next_unblocked_operator_handoff_validation_command_ready": "completion_receipt_validation_command_ready",
        "phase3_next_unblocked_operator_handoff_work_order_advances_next_package": "work_order_advances_next_package",
    }
    for gap_key, count_key in gap_to_count.items():
        if remaining_gaps.get(gap_key) != counts[count_key]:
            errors.append(f"next_unblocked_operator_handoff_{gap_key}_mismatch")
    if remaining_gaps.get("phase3_next_unblocked_operator_handoff_work_order_artifact") != handoff.get("work_order_artifact"):
        errors.append("next_unblocked_operator_handoff_gap_work_order_artifact_mismatch")

    handoff_to_count = {
        "next_unblocked_operator_handoff_ready": "handoff_ready",
        "next_unblocked_operator_handoff_work_order_ready": "work_order_ready",
        "next_unblocked_operator_handoff_work_order_capture_step_count": "work_order_capture_step_count",
        "next_unblocked_operator_handoff_work_order_validator_command_count": "work_order_validator_command_count",
        "next_unblocked_operator_handoff_receipt_template_ready": "completion_receipt_template_ready",
        "next_unblocked_operator_handoff_validation_command_ready": "completion_receipt_validation_command_ready",
        "next_unblocked_operator_handoff_work_order_advances_next_package": "work_order_advances_next_package",
    }
    for handoff_key, count_key in handoff_to_count.items():
        if handoff_counts.get(handoff_key) != counts[count_key]:
            errors.append(f"next_unblocked_operator_handoff_count_{handoff_key}_mismatch")
    for handoff_key, artifact_key in (
        ("next_unblocked_operator_handoff_work_order_artifact", "work_order_artifact"),
        ("next_unblocked_operator_handoff_work_order_next_artifact_id", "work_order_next_artifact_id"),
        ("next_unblocked_operator_handoff_receipt_template_artifact", "completion_receipt_template_artifact"),
    ):
        if handoff_counts.get(handoff_key) != handoff.get(artifact_key):
            errors.append(f"next_unblocked_operator_handoff_count_{handoff_key}_mismatch")
    return errors, counts
def all_request_approval_command_errors(queue: Any) -> tuple[list[str], int, int]:


    if not isinstance(queue, list):
        return [], 0, 0
    errors: list[str] = []
    command_count = 0
    ready_count = 0
    for item in queue:
        if not isinstance(item, dict):
            continue
        command_count += 1
        item_errors = approval_command_validation_errors(item)
        if item_errors:
            errors.extend(item_errors)
        else:
            ready_count += 1
    return errors, command_count, ready_count


def build_summary(root: Path) -> JSONDict:
    errors: list[str] = []
    warnings: list[str] = []
    root = root.resolve()
    if not root.exists():
        return {
            "schema_version": SUPPORTED_SCHEMA_VERSION,
            "mode": "phase3_operator_handoff_validation",
            "root": str(root),
            "valid": False,
            "handoff_package_ready": False,
            "errors": [f"operator_handoff_root_missing:{root}"],
            "warnings": [],
        }
    if not root.is_dir():
        return {
            "schema_version": SUPPORTED_SCHEMA_VERSION,
            "mode": "phase3_operator_handoff_validation",
            "root": str(root),
            "valid": False,
            "handoff_package_ready": False,
            "errors": [f"operator_handoff_root_not_directory:{root}"],
            "warnings": [],
        }

    missing_files = sorted(name for name in REQUIRED_FILES if not (root / name).exists())
    errors.extend(f"missing_required_file:{name}" for name in missing_files)

    manifest = load_json(root / "operator-handoff-manifest.json", errors)
    packet = load_json(root / "phase3-evidence-packet.json", errors)
    worksheet = load_json(root / "launch-card-binding-worksheet.json", errors)
    model_plane_contract_request = load_json(root / "model-plane-artifact-writer-contract-request.json", errors)
    launch_card = load_json(root / "recommended-runtime-capture-launch-card.template.json", errors)
    preflight_manifest = load_json(root / "recommended-runtime-capture-preflight.json", errors)
    command_contract = load_json(root / "recommended-runtime-capture-command-contract.json", errors)
    execution_coverage = load_json(root / "recommended-runtime-capture-execution-coverage.json", errors)
    all_execution_coverage = load_json(root / "all-request-runtime-capture-execution-coverage.json", errors)
    launch_card_directory = load_json(root / "runtime-capture-launch-card-directory.json", errors)
    reuse_capture_plan = load_json(root / "reuse-evidence-capture-plan.json", errors)
    manual_runbook = load_json(root / "manual-capture-runbook.json", errors)
    post_capture_runbook = load_json(root / "post-capture-intake-runbook.json", errors)
    recommended_manual_runbook = load_json(root / "recommended-manual-capture-runbook.json", errors)
    recommended_post_capture_runbook = load_json(root / "recommended-post-capture-intake-runbook.json", errors)
    recommended_work_order = load_json(root / "recommended-runtime-capture-work-order.json", errors)
    recommended_completion_receipt = load_json(root / "recommended-runtime-capture-completion-receipt.template.json", errors)
    next_unblocked_handoff = load_json(root / NEXT_UNBLOCKED_HANDOFF_FILE, errors)
    capture_queue = load_json(root / "capture-queue.json", errors)
    approval_manifest = load_json(root / "approval-command-manifest.json", errors)
    validator_manifest = load_json(root / "validator-command-manifest.json", errors)
    receipt_manifest = load_json(root / "receipt-fill-manifest.json", errors)
    receipt_command_manifest = load_json(root / "receipt-fill-command-manifest.json", errors)
    downstream_handoff_manifest = load_json(root / "downstream-handoff-manifest.json", errors)
    blocker_manifest = load_json(root / "blocker-closure-manifest.json", errors)
    blocker_ledger = load_json(root / "blocker-evidence-ledger.json", errors)
    blocker_resolution_queue = load_json(root / "blocker-resolution-queue.json", errors)
    readme_path = root / "README.md"
    readme_text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

    manifest_dict = manifest if isinstance(manifest, dict) else {}
    packet_dict = packet if isinstance(packet, dict) else {}
    worksheet_dict = worksheet if isinstance(worksheet, dict) else {}
    model_plane_contract_request_dict = model_plane_contract_request if isinstance(model_plane_contract_request, dict) else {}
    launch_card_dict = launch_card if isinstance(launch_card, dict) else {}
    counts = manifest_dict.get("handoff_counts") if isinstance(manifest_dict.get("handoff_counts"), dict) else {}
    request = manifest_dict.get("recommended_request") if isinstance(manifest_dict.get("recommended_request"), dict) else {}
    packet_request = packet_dict.get("recommended_runtime_capture_request") if isinstance(packet_dict.get("recommended_runtime_capture_request"), dict) else {}
    manifest_launch = manifest_dict.get("recommended_launch_card_template") if isinstance(manifest_dict.get("recommended_launch_card_template"), dict) else {}
    manifest_completion_validation = manifest_dict.get("completion_receipt_validation") if isinstance(manifest_dict.get("completion_receipt_validation"), dict) else {}
    manifest_next_unblocked_handoff = manifest_dict.get("next_unblocked_operator_handoff") if isinstance(manifest_dict.get("next_unblocked_operator_handoff"), dict) else {}
    manifest_reuse_capture = manifest_dict.get("recommended_reuse_capture") if isinstance(manifest_dict.get("recommended_reuse_capture"), dict) else {}
    remaining_gaps = packet_dict.get("remaining_gaps") if isinstance(packet_dict.get("remaining_gaps"), dict) else {}
    launch_library = packet_dict.get("phase3_launch_card_library_summary") if isinstance(packet_dict.get("phase3_launch_card_library_summary"), dict) else {}

    if manifest_dict.get("schema_version") != plan_phase3_evidence_packet.OPERATOR_HANDOFF_SCHEMA_VERSION:
        errors.append("operator_handoff_manifest_schema_mismatch")
    for key, expected in (
        ("metadata_only", True),
        ("launches_runtimes", False),
        ("runs_docker", False),
        ("sends_prompt_traffic", False),
        ("reads_private_tokens", False),
        ("mutates_runtime_residency", False),
    ):
        if manifest_dict.get(key) is not expected:
            errors.append(f"operator_handoff_safety_flag_mismatch:{key}")
    if manifest_dict.get("operator_handoff_ready") is not True:
        errors.append("operator_handoff_ready_false")
    if packet_dict.get("schema_version") != plan_phase3_evidence_packet.SUPPORTED_SCHEMA_VERSION:
        errors.append("evidence_packet_schema_mismatch")
    if packet_dict.get("valid") is not True:
        errors.append("evidence_packet_not_valid")
    if packet_dict.get("packet_ready") is not True:
        errors.append("evidence_packet_not_ready")
    if packet_dict.get("phase3_complete") is True:
        warnings.append("phase3_complete_true_in_handoff_package")

    launch_card_saved_handoff_errors: list[str] = []
    if launch_library:
        for count_key, gap_key, library_key in (
            (
                "launch_card_saved_handoff_artifacts_ready",
                "phase3_launch_card_saved_handoff_artifacts_ready",
                "saved_handoff_artifacts_ready",
            ),
            (
                "launch_card_saved_handoff_artifact_missing_count",
                "phase3_launch_card_saved_handoff_artifact_missing_count",
                "saved_handoff_artifact_missing_count",
            ),
            (
                "launch_card_saved_handoff_artifact_drifted_count",
                "phase3_launch_card_saved_handoff_artifact_drifted_count",
                "saved_handoff_artifact_drifted_count",
            ),
        ):
            if library_key not in launch_library:
                launch_card_saved_handoff_errors.append(f"launch_card_library_{library_key}_missing")
                continue
            expected = launch_library.get(library_key)
            if counts.get(count_key) != expected:
                launch_card_saved_handoff_errors.append(f"operator_handoff_{count_key}_mismatch")
            if remaining_gaps.get(gap_key) != expected:
                launch_card_saved_handoff_errors.append(f"operator_handoff_{gap_key}_mismatch")
    errors.extend(launch_card_saved_handoff_errors)

    repo_path_errors, repo_relative_path_count = repo_path_safety_errors(
        [
            ("operator_handoff_manifest", manifest_dict),
            ("launch_card_binding_worksheet", worksheet_dict),
            ("model_plane_artifact_writer_contract_request", model_plane_contract_request_dict),
            ("recommended_launch_card", launch_card_dict),
            ("recommended_runtime_capture_preflight", preflight_manifest),
            ("recommended_runtime_capture_command_contract", command_contract_path_safety_surface(command_contract)),
            ("recommended_runtime_capture_execution_coverage", execution_coverage),
            ("all_request_runtime_capture_execution_coverage", all_request_execution_path_safety_surface(all_execution_coverage)),
            ("reuse_evidence_capture_plan", reuse_capture_plan),
            ("manual_capture_runbook", manual_runbook),
            ("post_capture_intake_runbook", post_capture_runbook),
            ("recommended_manual_capture_runbook", recommended_manual_runbook),
            ("recommended_post_capture_intake_runbook", recommended_post_capture_runbook),
            ("recommended_runtime_capture_work_order", recommended_work_order),
            ("recommended_runtime_capture_completion_receipt", recommended_completion_receipt),
            ("next_unblocked_operator_handoff", next_unblocked_handoff),
            ("capture_queue", capture_queue),
            ("approval_command_manifest", approval_manifest),
            ("receipt_fill_manifest", receipt_manifest),
            ("receipt_fill_command_manifest", receipt_command_manifest),
            ("downstream_handoff_manifest", downstream_handoff_manifest),
            ("blocker_closure_manifest", blocker_manifest),
            ("blocker_evidence_ledger", blocker_ledger),
            ("blocker_resolution_queue", blocker_resolution_queue),
        ]
    )
    errors.extend(repo_path_errors)

    output_files = manifest_output_files(manifest_dict)
    for name, expected_kind in REQUIRED_FILES.items():
        if output_files.get(name) != expected_kind:
            errors.append(f"output_file_manifest_mismatch:{name}")
    for relative_path in output_files:
        if not path_stays_in_root(root, relative_path):
            errors.append(f"output_file_path_escapes_root:{relative_path}")
        elif not (root / relative_path).exists():
            errors.append(f"output_file_missing_on_disk:{relative_path}")
    disk_files = {path.name for path in root.iterdir() if path.is_file()}
    extra_files = sorted(disk_files - set(REQUIRED_FILES))

    if worksheet_dict.get("schema_version") != plan_phase3_launch_card_library.BINDING_WORKSHEET_SCHEMA_VERSION:
        errors.append("binding_worksheet_schema_mismatch")
    if worksheet_dict.get("worksheet_ready") is not True:
        errors.append("binding_worksheet_not_ready")
    if worksheet_dict.get("execution_ready") is not False:
        errors.append("binding_worksheet_execution_ready_not_false")
    if int_count(worksheet_dict.get("task_count")) != int_count(counts.get("launch_card_task_count")):
        errors.append("binding_worksheet_task_count_mismatch")
    if int_count(worksheet_dict.get("unbound_task_count")) != int_count(counts.get("launch_card_unbound_task_count")):
        errors.append("binding_worksheet_unbound_task_count_mismatch")

    worksheet_queue_task_key_set_for_contract = worksheet_queue_task_keys(worksheet_dict)
    model_plane_contract_task_key_set = model_plane_contract_request_task_keys(model_plane_contract_request_dict)
    if model_plane_contract_request_dict.get("schema_version") != plan_phase3_launch_card_library.MODEL_PLANE_ARTIFACT_WRITER_REQUEST_SCHEMA_VERSION:
        errors.append("model_plane_contract_request_schema_mismatch")
    if model_plane_contract_request_dict.get("request_ready") is not True:
        errors.append("model_plane_contract_request_not_ready")
    if model_plane_contract_request_dict.get("execution_ready") is not False:
        errors.append("model_plane_contract_request_execution_ready_not_false")
    if int_count(model_plane_contract_request_dict.get("task_count")) != int_count(counts.get("model_plane_artifact_writer_contract_request_task_count")):
        errors.append("model_plane_contract_request_task_count_mismatch")
    if int_count(model_plane_contract_request_dict.get("phase3_artifact_writer_contract_count")) != int_count(model_plane_contract_request_dict.get("task_count")):
        errors.append("model_plane_contract_request_contract_count_mismatch")
    if int_count(model_plane_contract_request_dict.get("phase3_artifact_writer_ready_count")) != 0:
        errors.append("model_plane_contract_request_runtime_writers_unexpectedly_ready")
    if int_count(model_plane_contract_request_dict.get("task_count")) and not model_plane_contract_task_key_set:
        errors.append("model_plane_contract_request_task_keys_missing")
    if model_plane_contract_task_key_set and worksheet_queue_task_key_set_for_contract != model_plane_contract_task_key_set:
        errors.append("model_plane_contract_request_worksheet_task_mismatch")

    if launch_card_dict.get("schema_version") != "moe-phase3-runtime-capture-launch-card-v1":
        errors.append("launch_card_schema_mismatch")
    if launch_card_dict.get("status") != "planned_only":
        errors.append("launch_card_status_not_planned_only")
    if launch_card_dict.get("executable") is not False:
        errors.append("launch_card_executable_not_false")
    if launch_card_dict.get("runtime_capture_command_ready") is True and launch_card_dict.get("status") != "planned_only":
        errors.append("launch_card_runtime_commands_ready_without_planned_status")
    if int_count(launch_card_dict.get("task_count")) != int_count(manifest_launch.get("task_count")):
        errors.append("launch_card_task_count_mismatch")
    launch_card_task_key_set = launch_card_task_keys(launch_card_dict)
    worksheet_recommended_task_key_set = worksheet_task_keys(
        worksheet_dict,
        request_path=launch_card_dict.get("request_path") or request.get("request_path"),
        launch_card_path=manifest_launch.get("path"),
    )
    if int_count(launch_card_dict.get("task_count")) and not launch_card_task_key_set:
        errors.append("launch_card_task_keys_missing")
    if launch_card_task_key_set and worksheet_recommended_task_key_set != launch_card_task_key_set:
        errors.append("launch_card_worksheet_task_mismatch")

    if request.get("request_path") != packet_request.get("request_path"):
        errors.append("recommended_request_path_mismatch")
    if request.get("next_artifact_id") != packet_request.get("next_artifact_id"):
        errors.append("recommended_next_artifact_mismatch")
    if manifest_launch.get("path") != remaining_gaps.get("recommended_runtime_capture_launch_card_template_path"):
        errors.append("recommended_launch_card_path_mismatch")
    if manifest_completion_validation != recommended_completion_receipt.get("completion_receipt_validation_step"):
        errors.append("completion_receipt_validation_manifest_mismatch")
    if "## Completion Receipt Validation" not in readme_text:
        errors.append("readme_completion_receipt_validation_section_missing")
    if COMPLETION_RECEIPT_VALIDATION_SCRIPT not in readme_text:
        errors.append("readme_completion_receipt_validation_command_missing")
    if "## Reuse Evidence Capture" not in readme_text:
        errors.append("readme_reuse_evidence_capture_section_missing")
    reuse_trace_path = manifest_reuse_capture.get("candidate_trace_path")
    if isinstance(reuse_trace_path, str) and reuse_trace_path and reuse_trace_path not in readme_text:
        errors.append("readme_reuse_evidence_capture_trace_path_missing")

    capture_queue_count = len(capture_queue) if isinstance(capture_queue, list) else 0
    if capture_queue_count != int_count(counts.get("capture_queue_request_count")):
        errors.append("capture_queue_count_mismatch")
    first_queue_item = first_capture_queue_item(capture_queue)
    if first_queue_item:
        if first_queue_item.get("request_path") != request.get("request_path"):
            errors.append("recommended_capture_queue_request_mismatch")
        if int_count(first_queue_item.get("rank")) != int_count(request.get("queue_rank")):
            errors.append("recommended_capture_queue_rank_mismatch")
        if first_queue_item.get("next_artifact_id") != request.get("next_artifact_id"):
            errors.append("recommended_capture_queue_next_artifact_mismatch")
    elif capture_queue_count:
        errors.append("recommended_capture_queue_missing")
    launch_card_queue_task_key_set = launch_card_queue_task_keys(launch_card_dict)
    capture_queue_recommended_task_key_set = capture_queue_task_keys(first_queue_item)
    worksheet_queue_task_key_set = worksheet_queue_task_keys(worksheet_dict)
    capture_queue_all_task_key_set = capture_queue_all_task_keys(capture_queue)
    if launch_card_queue_task_key_set and capture_queue_recommended_task_key_set != launch_card_queue_task_key_set:
        errors.append("launch_card_capture_queue_task_mismatch")
    if int_count(worksheet_dict.get("task_count")) and not worksheet_queue_task_key_set:
        errors.append("binding_worksheet_capture_queue_task_keys_missing")
    if worksheet_queue_task_key_set and capture_queue_all_task_key_set != worksheet_queue_task_key_set:
        errors.append("binding_worksheet_capture_queue_task_mismatch")
    preflight_errors_list, preflight_counts = runtime_capture_preflight_manifest_errors(
        preflight_manifest,
        packet_dict,
        request,
        first_queue_item,
        receipt_command_manifest,
        remaining_gaps,
    )
    command_contract_errors_list, command_contract_counts = runtime_capture_command_contract_errors(
        command_contract,
        packet_dict,
        request,
        launch_card_dict,
        remaining_gaps,
    )
    execution_coverage_errors_list, execution_coverage_counts = runtime_capture_execution_coverage_errors(
        execution_coverage,
        packet_dict,
        request,
        preflight_manifest,
        command_contract,
        remaining_gaps,
    )
    all_execution_coverage_errors_list, all_execution_coverage_counts = all_request_runtime_capture_execution_coverage_errors(
        all_execution_coverage,
        packet_dict,
        capture_queue,
        remaining_gaps,
    )
    launch_card_directory_errors_list, launch_card_directory_counts = runtime_capture_launch_card_directory_errors(
        launch_card_directory,
        packet_dict,
        all_execution_coverage,
        remaining_gaps,
        counts,
    )
    reuse_capture_plan_errors_list, reuse_capture_plan_counts = reuse_evidence_capture_plan_errors(
        reuse_capture_plan,
        packet_dict,
        remaining_gaps,
        counts,
        manifest_reuse_capture,
    )
    manual_runbook_errors_list, manual_runbook_counts = manual_capture_runbook_errors(
        manual_runbook,
        packet_dict,
        all_execution_coverage,
        capture_queue,
        receipt_command_manifest,
        remaining_gaps,
        counts,
    )
    post_capture_runbook_errors_list, post_capture_runbook_counts = post_capture_intake_runbook_errors(
        post_capture_runbook,
        packet_dict,
        manual_runbook,
        capture_queue,
        receipt_command_manifest,
        remaining_gaps,
        counts,
    )
    recommended_manual_runbook_errors_list, recommended_manual_runbook_counts = recommended_manual_capture_runbook_errors(
        recommended_manual_runbook,
        packet_dict,
        manual_runbook,
        request,
        counts,
    )
    recommended_post_capture_runbook_errors_list, recommended_post_capture_runbook_counts = recommended_post_capture_intake_runbook_errors(
        recommended_post_capture_runbook,
        packet_dict,
        post_capture_runbook,
        recommended_manual_runbook,
        request,
        counts,
    )
    recommended_work_order_errors_list, recommended_work_order_counts = recommended_runtime_capture_work_order_errors(
        recommended_work_order,
        packet_dict,
        request,
        recommended_manual_runbook,
        recommended_post_capture_runbook,
        counts,
    )
    recommended_completion_receipt_errors_list, recommended_completion_receipt_counts = recommended_runtime_capture_completion_receipt_template_errors(
        recommended_completion_receipt,
        packet_dict,
        recommended_work_order,
        request,
        counts,
    )
    next_unblocked_handoff_errors_list, next_unblocked_handoff_counts = next_unblocked_operator_handoff_errors(
        next_unblocked_handoff,
        packet_dict,
        manifest_next_unblocked_handoff,
        blocker_resolution_queue,
        recommended_work_order,
        recommended_completion_receipt,
        remaining_gaps,
        counts,
    )
    errors.extend(preflight_errors_list)
    errors.extend(command_contract_errors_list)
    errors.extend(execution_coverage_errors_list)
    errors.extend(all_execution_coverage_errors_list)
    errors.extend(launch_card_directory_errors_list)
    errors.extend(reuse_capture_plan_errors_list)
    errors.extend(manual_runbook_errors_list)
    errors.extend(post_capture_runbook_errors_list)
    errors.extend(recommended_manual_runbook_errors_list)
    errors.extend(recommended_post_capture_runbook_errors_list)
    errors.extend(recommended_work_order_errors_list)
    errors.extend(recommended_completion_receipt_errors_list)
    errors.extend(next_unblocked_handoff_errors_list)
    all_request_approval_errors, all_request_approval_command_count, all_request_approval_command_ready_count = all_request_approval_command_errors(capture_queue)
    approval_manifest_errors_list, approval_manifest_command_count, approval_manifest_ready_count = approval_command_manifest_errors(approval_manifest, capture_queue, remaining_gaps)
    errors.extend(approval_manifest_errors_list)
    errors.extend(all_request_approval_errors)
    missing_approval_keys = set(list_of_strings(request.get("missing_approval_keys")))
    recorded_approval_keys = set(list_of_strings(request.get("records_approval_keys")))
    approval_command_tokens = command_tokens(request.get("approval_command"))
    approval_required = bool(missing_approval_keys) or request.get("status") == "approval_required"
    if approval_required:
        if request.get("approval_command_class") != APPROVAL_COMMAND_CLASS:
            errors.append("recommended_approval_command_class_mismatch")
        if request.get("approval_metadata_only") is not True:
            errors.append("recommended_approval_command_not_metadata_only")
        if request.get("requires_explicit_user_approval") is not True:
            errors.append("recommended_approval_command_missing_explicit_approval_gate")
        if not approval_command_tokens:
            errors.append("recommended_approval_command_missing")
        elif "scripts/build_phase3_runtime_capture_request.py" not in approval_command_tokens:
            errors.append("recommended_approval_command_unexpected_script")
        if "--output" in approval_command_tokens:
            output_index = approval_command_tokens.index("--output") + 1
            if output_index >= len(approval_command_tokens) or approval_command_tokens[output_index] != request.get("request_path"):
                errors.append("recommended_approval_command_output_mismatch")
        else:
            errors.append("recommended_approval_command_output_missing")
        if missing_approval_keys != recorded_approval_keys:
            errors.append("recommended_approval_recorded_key_mismatch")
        unknown_keys = sorted((missing_approval_keys | recorded_approval_keys) - set(APPROVAL_KEY_FLAGS))
        if unknown_keys:
            errors.append("recommended_approval_unknown_keys")
        for key in sorted(missing_approval_keys):
            expected_flag = APPROVAL_KEY_FLAGS.get(key)
            if expected_flag and expected_flag not in approval_command_tokens:
                errors.append(f"recommended_approval_command_missing_flag:{expected_flag}")
    validator_command_count = count_validator_commands(validator_manifest)
    if validator_command_count != int_count(counts.get("validator_command_count")):
        errors.append("validator_command_count_mismatch")
    (
        validator_manifest_coverage_errors_list,
        validator_manifest_runtime_artifact_count,
        validator_manifest_future_artifact_count,
        validator_manifest_runtime_validator_command_count,
        validator_manifest_future_validator_command_count,
        validator_manifest_runtime_missing_validator_count,
        validator_manifest_future_missing_validator_count,
    ) = validator_manifest_coverage_errors(validator_manifest, receipt_command_manifest, capture_queue)
    errors.extend(validator_manifest_coverage_errors_list)
    receipt_entry_count = len(receipt_manifest) if isinstance(receipt_manifest, list) else 0
    if receipt_entry_count != int_count(counts.get("receipt_fill_entry_count")):
        errors.append("receipt_fill_count_mismatch")
    receipt_missing_count = receipt_entry_count - ready_after_approval_count(receipt_manifest)
    if receipt_missing_count != int_count(counts.get("receipt_fill_missing_count")):
        errors.append("receipt_fill_missing_count_mismatch")
    receipt_artifact_counts = receipt_fill_artifact_counts(receipt_manifest)
    errors.extend(receipt_fill_artifact_count_errors(counts, receipt_artifact_counts))
    receipt_command_entry_count = len(receipt_command_manifest) if isinstance(receipt_command_manifest, list) else 0
    if receipt_command_entry_count != receipt_entry_count:
        errors.append("receipt_command_entry_count_mismatch")
    receipt_keys = artifact_keys(receipt_manifest)
    receipt_command_keys = artifact_keys(receipt_command_manifest)
    capture_queue_keys = capture_queue_artifact_keys(capture_queue)
    receipt_duplicate_keys = duplicate_artifact_keys(receipt_manifest)
    receipt_command_duplicate_keys = duplicate_artifact_keys(receipt_command_manifest)
    if receipt_duplicate_keys:
        errors.append("receipt_fill_duplicate_artifact_keys")
    if receipt_command_duplicate_keys:
        errors.append("receipt_fill_command_duplicate_artifact_keys")
    if receipt_keys != receipt_command_keys:
        errors.append("receipt_fill_command_key_mismatch")
    if capture_queue_keys and receipt_keys != capture_queue_keys:
        errors.append("receipt_fill_capture_queue_key_mismatch")
    if not receipt_keys and receipt_entry_count:
        errors.append("receipt_fill_artifact_keys_missing")
    if not receipt_command_keys and receipt_command_entry_count:
        errors.append("receipt_fill_command_artifact_keys_missing")
    if not capture_queue_keys and capture_queue_count:
        errors.append("capture_queue_artifact_keys_missing")
    receipt_command_validator_errors_list, receipt_command_validator_count, capture_queue_validator_command_count, receipt_command_missing_validator_count, receipt_command_declared_count_mismatch_count = receipt_command_validator_errors(
        receipt_command_manifest,
        capture_queue,
    )
    errors.extend(receipt_command_validator_errors_list)
    downstream_handoff_errors_list, downstream_handoff_counts = downstream_handoff_manifest_errors(
        downstream_handoff_manifest,
        capture_queue,
        remaining_gaps,
    )
    errors.extend(downstream_handoff_errors_list)
    blocker_count = len(blocker_manifest) if isinstance(blocker_manifest, list) else 0
    blocker_summary_count = int_count(remaining_gaps.get("phase3_blocker_closure_reason_count"))
    if blocker_summary_count and blocker_count != blocker_summary_count:
        errors.append("blocker_closure_count_mismatch")
    remaining_blockers = list_of_strings(manifest_dict.get("remaining_blockers"))
    blocker_closure_errors_list, blocker_closure_counts = blocker_closure_manifest_errors(
        blocker_manifest,
        remaining_blockers,
        remaining_gaps,
    )
    errors.extend(blocker_closure_errors_list)
    blocker_evidence_ledger_errors_list, blocker_evidence_ledger_counts = blocker_evidence_ledger_errors(
        blocker_ledger,
        packet_dict,
        blocker_manifest,
        remaining_gaps,
        counts,
    )
    errors.extend(blocker_evidence_ledger_errors_list)
    blocker_resolution_queue_errors_list, blocker_resolution_queue_counts = blocker_resolution_queue_errors(
        blocker_resolution_queue,
        packet_dict,
        blocker_ledger,
        remaining_gaps,
        counts,
    )
    errors.extend(blocker_resolution_queue_errors_list)
    if "Metadata only: `True`" not in readme_text:
        errors.append("readme_missing_metadata_only_marker")
    if "# Phase 3 Operator Handoff" not in readme_text:
        errors.append("readme_missing_title")

    handoff_package_ready = not errors and manifest_dict.get("operator_handoff_ready") is True
    return {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "mode": "phase3_operator_handoff_validation",
        "root": str(root),
        "valid": not errors,
        "handoff_package_ready": handoff_package_ready,
        "operator_handoff_ready": manifest_dict.get("operator_handoff_ready"),
        "metadata_only": manifest_dict.get("metadata_only"),
        "phase3_complete": manifest_dict.get("phase3_complete"),
        "decision": manifest_dict.get("decision"),
        "required_file_count": len(REQUIRED_FILES),
        "missing_file_count": len(missing_files),
        "extra_file_count": len(extra_files),
        "capture_queue_request_count": capture_queue_count,
        "runtime_capture_preflight_ready": preflight_counts.get("ready"),
        "runtime_capture_preflight_pending_artifact_count": preflight_counts.get("pending_artifact_count"),
        "runtime_capture_preflight_artifact_check_count": preflight_counts.get("artifact_check_count"),
        "runtime_capture_preflight_receipt_entry_count": preflight_counts.get("receipt_entry_count"),
        "runtime_capture_preflight_validator_command_count": preflight_counts.get("validator_command_count"),
        "runtime_capture_preflight_runtime_closure_reason_count": preflight_counts.get("runtime_closure_reason_count"),
        "runtime_capture_preflight_missing_item_count": preflight_counts.get("missing_item_count"),
        "runtime_capture_preflight_error_count": len(preflight_errors_list),
        "runtime_capture_command_contract_ready": command_contract_counts.get("ready"),
        "runtime_capture_command_contract_runtime_ready": command_contract_counts.get("runtime_ready"),
        "runtime_capture_command_contract_planned_capture_count": command_contract_counts.get("planned_capture_count"),
        "runtime_capture_command_contract_runtime_command_count": command_contract_counts.get("runtime_command_count"),
        "runtime_capture_command_contract_missing_runtime_command_count": command_contract_counts.get("missing_runtime_command_count"),
        "runtime_capture_command_contract_error_count": len(command_contract_errors_list),
        "runtime_capture_execution_manual_ready": execution_coverage_counts.get("manual_ready"),
        "runtime_capture_execution_automated_ready": execution_coverage_counts.get("automated_ready"),
        "runtime_capture_execution_pending_artifact_count": execution_coverage_counts.get("pending_artifact_count"),
        "runtime_capture_execution_artifact_execution_count": execution_coverage_counts.get("artifact_execution_count"),
        "runtime_capture_execution_command_option_count": execution_coverage_counts.get("capture_command_count"),
        "runtime_capture_execution_operator_command_option_count": execution_coverage_counts.get("operator_command_count"),
        "runtime_capture_execution_metadata_command_option_count": execution_coverage_counts.get("metadata_command_count"),
        "runtime_capture_execution_artifacts_with_command_count": execution_coverage_counts.get("artifacts_with_command_count"),
        "runtime_capture_execution_manual_capture_required_count": execution_coverage_counts.get("manual_capture_required_count"),
        "runtime_capture_execution_missing_capture_command_count": execution_coverage_counts.get("missing_capture_command_count"),
        "runtime_capture_execution_missing_item_count": execution_coverage_counts.get("missing_item_count"),
        "runtime_capture_execution_error_count": len(execution_coverage_errors_list),
        "all_request_runtime_capture_execution_manual_ready": all_execution_coverage_counts.get("manual_ready"),
        "all_request_runtime_capture_execution_automated_ready": all_execution_coverage_counts.get("automated_ready"),
        "all_request_runtime_capture_execution_request_count": all_execution_coverage_counts.get("request_count"),
        "all_request_runtime_capture_execution_ready_request_count": all_execution_coverage_counts.get("manual_ready_count"),
        "all_request_runtime_capture_execution_automated_request_count": all_execution_coverage_counts.get("automated_ready_count"),
        "all_request_runtime_capture_execution_pending_artifact_count": all_execution_coverage_counts.get("pending_artifact_count"),
        "all_request_runtime_capture_execution_artifact_execution_count": all_execution_coverage_counts.get("artifact_execution_count"),
        "all_request_runtime_capture_execution_command_option_count": all_execution_coverage_counts.get("capture_command_count"),
        "all_request_runtime_capture_execution_operator_command_option_count": all_execution_coverage_counts.get("operator_command_count"),
        "all_request_runtime_capture_execution_metadata_command_option_count": all_execution_coverage_counts.get("metadata_command_count"),
        "all_request_runtime_capture_execution_artifacts_with_command_count": all_execution_coverage_counts.get("artifacts_with_command_count"),
        "all_request_runtime_capture_execution_manual_capture_required_count": all_execution_coverage_counts.get("manual_capture_required_count"),
        "all_request_runtime_capture_execution_missing_capture_command_count": all_execution_coverage_counts.get("missing_capture_command_count"),
        "all_request_runtime_capture_execution_missing_item_count": all_execution_coverage_counts.get("missing_item_count"),
        "all_request_runtime_capture_execution_error_count": len(all_execution_coverage_errors_list),
        "runtime_capture_launch_card_dir_provided": launch_card_directory_counts.get("provided"),
        "runtime_capture_launch_card_dir_ready": launch_card_directory_counts.get("ready"),
        "runtime_capture_launch_card_dir_expected_card_count": launch_card_directory_counts.get("expected_card_count"),
        "runtime_capture_launch_card_dir_matched_card_count": launch_card_directory_counts.get("matched_card_count"),
        "runtime_capture_launch_card_dir_missing_card_count": launch_card_directory_counts.get("missing_card_count"),
        "runtime_capture_launch_card_dir_error_count": len(launch_card_directory_errors_list),
        "reuse_evidence_capture_valid": reuse_capture_plan_counts.get("valid"),
        "reuse_evidence_capture_bundle_count": reuse_capture_plan_counts.get("bundle_count"),
        "reuse_evidence_capture_ready_count": reuse_capture_plan_counts.get("reuse_ready_count"),
        "reuse_evidence_capture_blocked_count": reuse_capture_plan_counts.get("reuse_blocked_count"),
        "reuse_evidence_capture_trace_valid_count": reuse_capture_plan_counts.get("candidate_trace_valid_count"),
        "reuse_evidence_capture_receipt_ready_count": reuse_capture_plan_counts.get("candidate_trace_receipt_ready_count"),
        "reuse_evidence_capture_no_reuse_distance_count": reuse_capture_plan_counts.get("no_reuse_distance_observation_count"),
        "reuse_evidence_capture_prompt_identity_ready_count": reuse_capture_plan_counts.get("prompt_identity_ready_count"),
        "reuse_evidence_capture_prompt_identity_missing_count": reuse_capture_plan_counts.get("prompt_identity_metadata_missing_count"),
        "reuse_evidence_capture_has_recommended_capture": reuse_capture_plan_counts.get("has_recommended_capture"),
        "reuse_evidence_capture_manifest_recommendation_ready": reuse_capture_plan_counts.get("manifest_recommendation_ready"),
        "reuse_evidence_capture_error_count": len(reuse_capture_plan_errors_list),
        "manual_capture_runbook_ready": manual_runbook_counts.get("ready"),
        "manual_capture_runbook_request_count": manual_runbook_counts.get("request_count"),
        "manual_capture_runbook_ready_request_count": manual_runbook_counts.get("ready_request_count"),
        "manual_capture_runbook_manual_task_count": manual_runbook_counts.get("manual_task_count"),
        "manual_capture_runbook_runtime_command_task_count": manual_runbook_counts.get("runtime_command_task_count"),
        "manual_capture_runbook_validator_command_count": manual_runbook_counts.get("validator_command_count"),
        "manual_capture_runbook_missing_item_count": manual_runbook_counts.get("missing_item_count"),
        "manual_capture_runbook_missing_receipt_command_count": manual_runbook_counts.get("missing_receipt_command_count"),
        "manual_capture_runbook_missing_validator_command_count": manual_runbook_counts.get("missing_validator_command_count"),
        "manual_capture_runbook_missing_approval_command_count": manual_runbook_counts.get("missing_approval_command_count"),
        "manual_capture_runbook_missing_source_request_path_count": manual_runbook_counts.get("missing_source_request_path_count"),
        "manual_capture_runbook_missing_prompt_set_path_count": manual_runbook_counts.get("missing_prompt_set_path_count"),
        "manual_capture_runbook_missing_explicit_approval_count": manual_runbook_counts.get("missing_explicit_approval_count"),
        "manual_capture_runbook_missing_prompt_traffic_ack_count": manual_runbook_counts.get("missing_prompt_traffic_ack_count"),
        "manual_capture_runbook_error_count": len(manual_runbook_errors_list),
        "post_capture_intake_runbook_ready": post_capture_runbook_counts.get("ready"),
        "post_capture_intake_runbook_request_count": post_capture_runbook_counts.get("request_count"),
        "post_capture_intake_runbook_artifact_gate_count": post_capture_runbook_counts.get("artifact_gate_count"),
        "post_capture_intake_runbook_ready_after_current_intake_count": post_capture_runbook_counts.get("ready_after_current_intake_count"),
        "post_capture_intake_runbook_missing_after_current_intake_count": post_capture_runbook_counts.get("missing_after_current_intake_count"),
        "post_capture_intake_runbook_validator_command_count": post_capture_runbook_counts.get("validator_command_count"),
        "post_capture_intake_runbook_ready_to_update_bundle_count": post_capture_runbook_counts.get("ready_to_update_bundle_count"),
        "post_capture_intake_runbook_phase4_candidate_count": post_capture_runbook_counts.get("phase4_candidate_count"),
        "post_capture_intake_runbook_live_spike_candidate_count": post_capture_runbook_counts.get("live_spike_candidate_count"),
        "post_capture_intake_runbook_missing_source_request_path_count": post_capture_runbook_counts.get("missing_source_request_path_count"),
        "post_capture_intake_runbook_missing_prompt_set_path_count": post_capture_runbook_counts.get("missing_prompt_set_path_count"),
        "post_capture_intake_runbook_missing_explicit_approval_count": post_capture_runbook_counts.get("missing_explicit_approval_count"),
        "post_capture_intake_runbook_missing_prompt_traffic_ack_count": post_capture_runbook_counts.get("missing_prompt_traffic_ack_count"),
        "post_capture_intake_runbook_missing_item_count": post_capture_runbook_counts.get("missing_item_count"),
        "post_capture_intake_runbook_error_count": len(post_capture_runbook_errors_list),
        "recommended_manual_capture_runbook_ready": recommended_manual_runbook_counts.get("ready"),
        "recommended_manual_capture_runbook_request_count": recommended_manual_runbook_counts.get("request_count"),
        "recommended_manual_capture_runbook_manual_task_count": recommended_manual_runbook_counts.get("manual_task_count"),
        "recommended_manual_capture_runbook_runtime_command_task_count": recommended_manual_runbook_counts.get("runtime_command_task_count"),
        "recommended_manual_capture_runbook_validator_command_count": recommended_manual_runbook_counts.get("validator_command_count"),
        "recommended_manual_capture_runbook_missing_item_count": recommended_manual_runbook_counts.get("missing_item_count"),
        "recommended_manual_capture_runbook_error_count": len(recommended_manual_runbook_errors_list),
        "recommended_post_capture_intake_runbook_ready": recommended_post_capture_runbook_counts.get("ready"),
        "recommended_post_capture_intake_runbook_request_count": recommended_post_capture_runbook_counts.get("request_count"),
        "recommended_post_capture_intake_runbook_artifact_gate_count": recommended_post_capture_runbook_counts.get("artifact_gate_count"),
        "recommended_post_capture_intake_runbook_ready_after_current_intake_count": recommended_post_capture_runbook_counts.get("ready_after_current_intake_count"),
        "recommended_post_capture_intake_runbook_missing_after_current_intake_count": recommended_post_capture_runbook_counts.get("missing_after_current_intake_count"),
        "recommended_post_capture_intake_runbook_validator_command_count": recommended_post_capture_runbook_counts.get("validator_command_count"),
        "recommended_post_capture_intake_runbook_ready_to_update_bundle_count": recommended_post_capture_runbook_counts.get("ready_to_update_bundle_count"),
        "recommended_post_capture_intake_runbook_missing_item_count": recommended_post_capture_runbook_counts.get("missing_item_count"),
        "recommended_post_capture_intake_runbook_error_count": len(recommended_post_capture_runbook_errors_list),
        "recommended_runtime_capture_work_order_ready": recommended_work_order_counts.get("ready"),
        "recommended_runtime_capture_work_order_capture_step_count": recommended_work_order_counts.get("capture_step_count"),
        "recommended_runtime_capture_work_order_artifact_gate_count": recommended_work_order_counts.get("artifact_gate_count"),
        "recommended_runtime_capture_work_order_validator_command_count": recommended_work_order_counts.get("validator_command_count"),
        "recommended_runtime_capture_work_order_missing_item_count": recommended_work_order_counts.get("missing_item_count"),
        "recommended_runtime_capture_work_order_approval_command_ready": recommended_work_order_counts.get("approval_command_ready"),
        "recommended_runtime_capture_work_order_intake_command_ready": recommended_work_order_counts.get("intake_command_ready"),
        "recommended_runtime_capture_work_order_post_capture_sequence_ready": recommended_work_order_counts.get("post_capture_sequence_ready"),
        "recommended_runtime_capture_work_order_post_capture_sequence_step_count": recommended_work_order_counts.get("post_capture_sequence_step_count"),
        "recommended_runtime_capture_work_order_completion_validation_before_intake": recommended_work_order_counts.get("completion_validation_before_intake"),
        "recommended_runtime_capture_work_order_completion_validation_command_ready": recommended_work_order_counts.get("completion_validation_command_ready"),
        "recommended_runtime_capture_work_order_error_count": len(recommended_work_order_errors_list),
        "recommended_runtime_capture_completion_receipt_template_ready": recommended_completion_receipt_counts.get("template_ready"),
        "recommended_runtime_capture_completion_receipt_complete": recommended_completion_receipt_counts.get("receipt_complete"),
        "recommended_runtime_capture_completion_receipt_ready_for_intake": recommended_completion_receipt_counts.get("ready_for_intake"),
        "recommended_runtime_capture_completion_receipt_count": recommended_completion_receipt_counts.get("capture_receipt_count"),
        "recommended_runtime_capture_completion_receipt_expected_capture_step_count": recommended_completion_receipt_counts.get("expected_capture_step_count"),
        "recommended_runtime_capture_completion_receipt_validator_command_count": recommended_completion_receipt_counts.get("validator_command_count"),
        "recommended_runtime_capture_completion_receipt_missing_item_count": recommended_completion_receipt_counts.get("missing_item_count"),
        "recommended_runtime_capture_completion_receipt_validation_command_ready": recommended_completion_receipt_counts.get("validation_command_ready"),
        "recommended_runtime_capture_completion_receipt_error_count": len(recommended_completion_receipt_errors_list),
        "next_unblocked_operator_handoff_ready": next_unblocked_handoff_counts.get("handoff_ready"),
        "next_unblocked_operator_handoff_work_order_ready": next_unblocked_handoff_counts.get("work_order_ready"),
        "next_unblocked_operator_handoff_work_order_advances_next_package": next_unblocked_handoff_counts.get("work_order_advances_next_package"),
        "next_unblocked_operator_handoff_work_order_capture_step_count": next_unblocked_handoff_counts.get("work_order_capture_step_count"),
        "next_unblocked_operator_handoff_work_order_validator_command_count": next_unblocked_handoff_counts.get("work_order_validator_command_count"),
        "next_unblocked_operator_handoff_receipt_template_ready": next_unblocked_handoff_counts.get("completion_receipt_template_ready"),
        "next_unblocked_operator_handoff_validation_command_ready": next_unblocked_handoff_counts.get("completion_receipt_validation_command_ready"),
        "next_unblocked_operator_handoff_bound_to_receipt": next_unblocked_handoff_counts.get("work_order_bound_to_completion_receipt"),
        "next_unblocked_operator_handoff_package_id": next_unblocked_handoff_counts.get("next_unblocked_work_package_id"),
        "next_unblocked_operator_handoff_row_count": next_unblocked_handoff_counts.get("next_unblocked_row_count"),
        "next_unblocked_operator_handoff_validator_command_count": next_unblocked_handoff_counts.get("next_unblocked_validator_command_count"),
        "next_unblocked_operator_handoff_error_count": len(next_unblocked_handoff_errors_list),
        "validator_command_count": validator_command_count,
        "validator_manifest_runtime_artifact_count": validator_manifest_runtime_artifact_count,
        "validator_manifest_future_artifact_count": validator_manifest_future_artifact_count,
        "validator_manifest_runtime_validator_command_count": validator_manifest_runtime_validator_command_count,
        "validator_manifest_future_validator_command_count": validator_manifest_future_validator_command_count,
        "validator_manifest_runtime_missing_validator_count": validator_manifest_runtime_missing_validator_count,
        "validator_manifest_future_missing_validator_count": validator_manifest_future_missing_validator_count,
        "validator_manifest_coverage_ready": not validator_manifest_coverage_errors_list,
        "receipt_fill_entry_count": receipt_entry_count,
        "receipt_fill_missing_count": receipt_missing_count,
        "receipt_fill_candidate_router_trace_entry_count": receipt_artifact_counts["candidate_router_trace"]["entry_count"],
        "receipt_fill_candidate_router_trace_ready_count": receipt_artifact_counts["candidate_router_trace"]["ready_count"],
        "receipt_fill_candidate_router_trace_missing_count": receipt_artifact_counts["candidate_router_trace"]["missing_count"],
        "receipt_fill_managed_output_entry_count": receipt_artifact_counts["managed_output_summary_fill"]["entry_count"],
        "receipt_fill_managed_output_ready_count": receipt_artifact_counts["managed_output_summary_fill"]["ready_count"],
        "receipt_fill_managed_output_missing_count": receipt_artifact_counts["managed_output_summary_fill"]["missing_count"],
        "receipt_fill_dense_output_entry_count": receipt_artifact_counts["dense_output_summary_fill"]["entry_count"],
        "receipt_fill_dense_output_ready_count": receipt_artifact_counts["dense_output_summary_fill"]["ready_count"],
        "receipt_fill_dense_output_missing_count": receipt_artifact_counts["dense_output_summary_fill"]["missing_count"],
        "receipt_fill_key_count": len(receipt_keys),
        "receipt_fill_command_key_count": len(receipt_command_keys),
        "capture_queue_artifact_key_count": len(capture_queue_keys),
        "receipt_fill_command_validator_command_count": receipt_command_validator_count,
        "capture_queue_validator_command_count": capture_queue_validator_command_count,
        "receipt_fill_command_missing_validator_count": receipt_command_missing_validator_count,
        "receipt_fill_command_declared_count_mismatch_count": receipt_command_declared_count_mismatch_count,
        "receipt_fill_command_key_parity_ready": receipt_keys == receipt_command_keys and not receipt_duplicate_keys and not receipt_command_duplicate_keys,
        "receipt_fill_capture_queue_key_parity_ready": bool(capture_queue_keys) and receipt_keys == capture_queue_keys,
        "receipt_fill_command_validator_coverage_ready": bool(capture_queue_keys) and not receipt_command_validator_errors_list and receipt_command_validator_count == capture_queue_validator_command_count,
        "downstream_handoff_request_count": downstream_handoff_counts.get("request_count"),
        "downstream_handoff_policy_ready_count": downstream_handoff_counts.get("policy_ready_count"),
        "downstream_handoff_dense_ready_count": downstream_handoff_counts.get("dense_ready_count"),
        "downstream_handoff_live_ready_count": downstream_handoff_counts.get("live_ready_count"),
        "downstream_handoff_all_ready_count": downstream_handoff_counts.get("all_ready_count"),
        "downstream_handoff_missing_section_count": downstream_handoff_counts.get("missing_section_count"),
        "downstream_handoff_missing_path_count": downstream_handoff_counts.get("missing_path_count"),
        "downstream_handoff_missing_validator_count": downstream_handoff_counts.get("missing_validator_count"),
        "downstream_handoff_manifest_coverage_ready": not downstream_handoff_errors_list,
        "recommended_approval_command_ready": approval_required and APPROVAL_COMMAND_CLASS == request.get("approval_command_class") and request.get("approval_metadata_only") is True and request.get("requires_explicit_user_approval") is True and bool(approval_command_tokens) and missing_approval_keys == recorded_approval_keys,
        "all_request_approval_command_count": all_request_approval_command_count,
        "all_request_approval_command_ready_count": all_request_approval_command_ready_count,
        "all_request_approval_commands_ready": bool(capture_queue_count) and all_request_approval_command_ready_count == all_request_approval_command_count and not all_request_approval_errors,
        "all_request_approval_command_error_count": len(all_request_approval_errors),
        "approval_command_manifest_count": approval_manifest_command_count,
        "approval_command_manifest_ready_count": approval_manifest_ready_count,
        "approval_command_manifest_coverage_ready": bool(capture_queue_count) and approval_manifest_ready_count == approval_manifest_command_count == capture_queue_count and not approval_manifest_errors_list,
        "approval_command_manifest_error_count": len(approval_manifest_errors_list),
        "recommended_capture_queue_ready": bool(first_queue_item) and first_queue_item.get("request_path") == request.get("request_path") and int_count(first_queue_item.get("rank")) == int_count(request.get("queue_rank")) and first_queue_item.get("next_artifact_id") == request.get("next_artifact_id"),
        "launch_card_task_key_count": len(launch_card_task_key_set),
        "worksheet_recommended_task_key_count": len(worksheet_recommended_task_key_set),
        "capture_queue_recommended_task_key_count": len(capture_queue_recommended_task_key_set),
        "binding_worksheet_queue_task_key_count": len(worksheet_queue_task_key_set),
        "capture_queue_task_key_count": len(capture_queue_all_task_key_set),
        "launch_card_worksheet_task_parity_ready": bool(launch_card_task_key_set) and worksheet_recommended_task_key_set == launch_card_task_key_set,
        "launch_card_capture_queue_task_parity_ready": bool(launch_card_queue_task_key_set) and capture_queue_recommended_task_key_set == launch_card_queue_task_key_set,
        "binding_worksheet_capture_queue_task_parity_ready": bool(worksheet_queue_task_key_set) and capture_queue_all_task_key_set == worksheet_queue_task_key_set,
        "model_plane_contract_request_task_key_count": len(model_plane_contract_task_key_set),
        "model_plane_contract_request_parity_ready": bool(model_plane_contract_task_key_set) and model_plane_contract_task_key_set == worksheet_queue_task_key_set,
        "launch_card_saved_handoff_artifacts_ready": counts.get("launch_card_saved_handoff_artifacts_ready"),
        "launch_card_saved_handoff_artifact_missing_count": counts.get("launch_card_saved_handoff_artifact_missing_count"),
        "launch_card_saved_handoff_artifact_drifted_count": counts.get("launch_card_saved_handoff_artifact_drifted_count"),
        "launch_card_saved_handoff_parity_ready": not launch_card_saved_handoff_errors,
        "launch_card_saved_handoff_error_count": len(launch_card_saved_handoff_errors),
        "repo_relative_path_count": repo_relative_path_count,
        "repo_path_safety_ready": not repo_path_errors,
        "repo_path_unsafe_count": len(repo_path_errors),
        "blocker_count": blocker_count,
        "blocker_closure_reason_id_count": blocker_closure_counts.get("reason_id_count"),
        "blocker_closure_unmapped_count": blocker_closure_counts.get("unmapped_count"),
        "blocker_closure_missing_action_count": blocker_closure_counts.get("missing_action_count"),
        "blocker_closure_missing_primary_path_count": blocker_closure_counts.get("missing_primary_path_count"),
        "blocker_closure_runtime_capture_required_count": blocker_closure_counts.get("runtime_capture_required_count"),
        "blocker_closure_future_adapter_required_count": blocker_closure_counts.get("future_adapter_required_count"),
        "blocker_closure_validator_command_count": blocker_closure_counts.get("validator_command_count"),
        "blocker_closure_missing_evidence_count": blocker_closure_counts.get("missing_evidence_count"),
        "blocker_closure_manifest_coverage_ready": not blocker_closure_errors_list,
        "blocker_evidence_ledger_ready": blocker_evidence_ledger_counts.get("ready"),
        "blocker_evidence_ledger_reason_count": blocker_evidence_ledger_counts.get("reason_count"),
        "blocker_evidence_ledger_row_count": blocker_evidence_ledger_counts.get("row_count"),
        "blocker_evidence_ledger_expected_missing_evidence_count": blocker_evidence_ledger_counts.get("expected_missing_evidence_count"),
        "blocker_evidence_ledger_runtime_capture_required_row_count": blocker_evidence_ledger_counts.get("runtime_capture_required_row_count"),
        "blocker_evidence_ledger_future_adapter_required_row_count": blocker_evidence_ledger_counts.get("future_adapter_required_row_count"),
        "blocker_evidence_ledger_receipt_bound_row_count": blocker_evidence_ledger_counts.get("receipt_bound_row_count"),
        "blocker_evidence_ledger_dense_output_row_count": blocker_evidence_ledger_counts.get("dense_output_row_count"),
        "blocker_evidence_ledger_live_proof_row_count": blocker_evidence_ledger_counts.get("live_proof_row_count"),
        "blocker_evidence_ledger_runtime_actuator_row_count": blocker_evidence_ledger_counts.get("runtime_actuator_row_count"),
        "blocker_evidence_ledger_validator_command_count": blocker_evidence_ledger_counts.get("validator_command_count"),
        "blocker_evidence_ledger_missing_item_count": blocker_evidence_ledger_counts.get("missing_item_count"),
        "blocker_evidence_ledger_coverage_ready": not blocker_evidence_ledger_errors_list,
        "blocker_evidence_ledger_error_count": len(blocker_evidence_ledger_errors_list),
        "blocker_resolution_queue_ready": blocker_resolution_queue_counts.get("ready"),
        "blocker_resolution_queue_work_package_count": blocker_resolution_queue_counts.get("work_package_count"),
        "blocker_resolution_queue_row_count": blocker_resolution_queue_counts.get("queue_row_count"),
        "blocker_resolution_queue_expected_ledger_row_count": blocker_resolution_queue_counts.get("expected_ledger_row_count"),
        "blocker_resolution_queue_runtime_capture_package_count": blocker_resolution_queue_counts.get("runtime_capture_package_count"),
        "blocker_resolution_queue_future_adapter_package_count": blocker_resolution_queue_counts.get("future_adapter_package_count"),
        "blocker_resolution_queue_runtime_capture_required_row_count": blocker_resolution_queue_counts.get("runtime_capture_required_row_count"),
        "blocker_resolution_queue_future_adapter_required_row_count": blocker_resolution_queue_counts.get("future_adapter_required_row_count"),
        "blocker_resolution_queue_receipt_bound_row_count": blocker_resolution_queue_counts.get("receipt_bound_row_count"),
        "blocker_resolution_queue_validator_command_count": blocker_resolution_queue_counts.get("validator_command_count"),
        "blocker_resolution_queue_completion_gate_count": blocker_resolution_queue_counts.get("completion_gate_count"),
        "blocker_resolution_queue_dependency_edge_count": blocker_resolution_queue_counts.get("dependency_edge_count"),
        "blocker_resolution_queue_missing_item_count": blocker_resolution_queue_counts.get("missing_item_count"),
        "blocker_resolution_queue_runtime_actuator_proof_handoff_ready": blocker_resolution_queue_counts.get("runtime_actuator_proof_handoff_ready"),
        "blocker_resolution_queue_runtime_actuator_live_spike_ready": blocker_resolution_queue_counts.get("runtime_actuator_live_spike_ready"),
        "blocker_resolution_queue_runtime_actuator_proof_requirement_count": blocker_resolution_queue_counts.get("runtime_actuator_proof_requirement_count"),
        "blocker_resolution_queue_runtime_actuator_proof_requirement_id_count": blocker_resolution_queue_counts.get("runtime_actuator_proof_requirement_id_count"),
        "blocker_resolution_queue_runtime_actuator_proof_artifact_count": blocker_resolution_queue_counts.get("runtime_actuator_proof_artifact_count"),
        "blocker_resolution_queue_runtime_actuator_dependency_edge_count": blocker_resolution_queue_counts.get("runtime_actuator_dependency_edge_count"),
        "blocker_resolution_queue_runtime_actuator_blocking_capability_count": blocker_resolution_queue_counts.get("runtime_actuator_blocking_capability_count"),
        "blocker_resolution_queue_runtime_actuator_control_blocker_count": blocker_resolution_queue_counts.get("runtime_actuator_control_blocker_count"),
        "blocker_resolution_queue_coverage_ready": not blocker_resolution_queue_errors_list,
        "blocker_resolution_queue_error_count": len(blocker_resolution_queue_errors_list),
        "remaining_blockers": remaining_blockers,
        "errors": errors,
        "warnings": warnings,
        "safety_contract": [
            "operator handoff validation reads local files only",
            "operator handoff validation does not launch model servers",
            "operator handoff validation does not run Docker",
            "operator handoff validation does not call endpoints",
            "operator handoff validation does not inspect private tokens",
            "operator handoff validation does not send prompt traffic",
            "operator handoff validation does not mutate runtime residency",
            "operator handoff validation does not claim live expert paging",
        ],
    }


def build_current_temp_summary() -> JSONDict:
    packet = plan_phase3_evidence_packet.build_packet_summary(
        plan_phase3_evidence_packet.DEFAULT_TRACE_PATH,
        plan_phase3_evidence_packet.DEFAULT_INVENTORY_PATH,
        plan_phase3_evidence_packet.DEFAULT_POLICIES_PATH,
        plan_phase3_evidence_packet.DEFAULT_MANAGED_PLAN_PATH,
    )
    with tempfile.TemporaryDirectory(prefix="phase3-operator-handoff-") as temp_dir:
        output_dir = Path(temp_dir)
        plan_phase3_evidence_packet.write_operator_handoff_dir(packet, output_dir)
        summary = build_summary(output_dir)
        summary["generated_temp_package"] = True
        return summary


def format_markdown_report(summary: JSONDict) -> str:
    lines = [
        "# Phase 3 Operator Handoff Validation",
        "",
        f"- Valid: `{summary.get('valid')}`",
        f"- Handoff package ready: `{summary.get('handoff_package_ready')}`",
        f"- Metadata only: `{summary.get('metadata_only')}`",
        f"- Phase 3 complete: `{summary.get('phase3_complete')}`",
        f"- Decision: `{summary.get('decision')}`",
        f"- Required files: `{summary.get('required_file_count')}`",
        f"- Missing files: `{summary.get('missing_file_count')}`",
        f"- Extra files: `{summary.get('extra_file_count')}`",
        f"- Capture requests: `{summary.get('capture_queue_request_count')}`",
        f"- Runtime preflight ready: `{summary.get('runtime_capture_preflight_ready')}`",
        f"- Runtime preflight pending artifacts: `{summary.get('runtime_capture_preflight_pending_artifact_count')}`",
        f"- Runtime preflight validator commands: `{summary.get('runtime_capture_preflight_validator_command_count')}`",
        f"- Runtime preflight missing items: `{summary.get('runtime_capture_preflight_missing_item_count')}`",
        f"- Runtime preflight errors: `{summary.get('runtime_capture_preflight_error_count')}`",
        f"- Runtime command contract ready: `{summary.get('runtime_capture_command_contract_ready')}`",
        f"- Runtime command contract runtime ready: `{summary.get('runtime_capture_command_contract_runtime_ready')}`",
        f"- Runtime command contract planned captures: `{summary.get('runtime_capture_command_contract_planned_capture_count')}`",
        f"- Runtime command contract missing runtime commands: `{summary.get('runtime_capture_command_contract_missing_runtime_command_count')}`",
        f"- Runtime command contract errors: `{summary.get('runtime_capture_command_contract_error_count')}`",
        f"- Runtime execution manual ready: `{summary.get('runtime_capture_execution_manual_ready')}`",
        f"- Runtime execution automated ready: `{summary.get('runtime_capture_execution_automated_ready')}`",
        f"- Runtime execution command options: `{summary.get('runtime_capture_execution_command_option_count')}`",
        f"- Runtime execution manual captures required: `{summary.get('runtime_capture_execution_manual_capture_required_count')}`",
        f"- Runtime execution missing capture commands: `{summary.get('runtime_capture_execution_missing_capture_command_count')}`",
        f"- Runtime execution missing items: `{summary.get('runtime_capture_execution_missing_item_count')}`",
        f"- Runtime execution errors: `{summary.get('runtime_capture_execution_error_count')}`",
        f"- All-request runtime execution manual ready: `{summary.get('all_request_runtime_capture_execution_manual_ready')}`",
        f"- All-request runtime execution automated ready: `{summary.get('all_request_runtime_capture_execution_automated_ready')}`",
        f"- All-request runtime execution ready requests: `{summary.get('all_request_runtime_capture_execution_ready_request_count')}` / `{summary.get('all_request_runtime_capture_execution_request_count')}`",
        f"- All-request runtime execution pending artifacts: `{summary.get('all_request_runtime_capture_execution_pending_artifact_count')}`",
        f"- All-request runtime execution command options: `{summary.get('all_request_runtime_capture_execution_command_option_count')}`",
        f"- All-request runtime execution manual captures required: `{summary.get('all_request_runtime_capture_execution_manual_capture_required_count')}`",
        f"- All-request runtime execution missing capture commands: `{summary.get('all_request_runtime_capture_execution_missing_capture_command_count')}`",
        f"- All-request runtime execution missing items: `{summary.get('all_request_runtime_capture_execution_missing_item_count')}`",
        f"- All-request runtime execution errors: `{summary.get('all_request_runtime_capture_execution_error_count')}`",
        f"- Filled launch-card directory provided: `{summary.get('runtime_capture_launch_card_dir_provided')}`",
        f"- Filled launch-card directory ready: `{summary.get('runtime_capture_launch_card_dir_ready')}`",
        f"- Filled launch-card directory matched cards: `{summary.get('runtime_capture_launch_card_dir_matched_card_count')}` / `{summary.get('runtime_capture_launch_card_dir_expected_card_count')}`",
        f"- Filled launch-card directory missing cards: `{summary.get('runtime_capture_launch_card_dir_missing_card_count')}`",
        f"- Filled launch-card directory errors: `{summary.get('runtime_capture_launch_card_dir_error_count')}`",
        f"- Reuse capture valid: `{summary.get('reuse_evidence_capture_valid')}`",
        f"- Reuse-ready traces: `{summary.get('reuse_evidence_capture_ready_count')}` / `{summary.get('reuse_evidence_capture_bundle_count')}`",
        f"- Reuse capture blocked traces: `{summary.get('reuse_evidence_capture_blocked_count')}`",
        f"- Reuse capture no-distance blockers: `{summary.get('reuse_evidence_capture_no_reuse_distance_count')}`",
        f"- Reuse capture prompt-identity blockers: `{summary.get('reuse_evidence_capture_prompt_identity_missing_count')}`",
        f"- Reuse capture manifest recommendation ready: `{summary.get('reuse_evidence_capture_manifest_recommendation_ready')}`",
        f"- Reuse capture errors: `{summary.get('reuse_evidence_capture_error_count')}`",
        f"- Validator commands: `{summary.get('validator_command_count')}`",
        f"- Validator manifest runtime artifacts: `{summary.get('validator_manifest_runtime_artifact_count')}`",
        f"- Validator manifest future artifacts: `{summary.get('validator_manifest_future_artifact_count')}`",
        f"- Validator manifest runtime commands: `{summary.get('validator_manifest_runtime_validator_command_count')}`",
        f"- Validator manifest future commands: `{summary.get('validator_manifest_future_validator_command_count')}`",
        f"- Validator manifest coverage ready: `{summary.get('validator_manifest_coverage_ready')}`",
        f"- Receipt fills missing: `{summary.get('receipt_fill_missing_count')}`",
        f"- Receipt-fill candidate router trace ready: `{summary.get('receipt_fill_candidate_router_trace_ready_count')}` / `{summary.get('receipt_fill_candidate_router_trace_entry_count')}`",
        f"- Receipt-fill managed outputs ready: `{summary.get('receipt_fill_managed_output_ready_count')}` / `{summary.get('receipt_fill_managed_output_entry_count')}`",
        f"- Receipt-fill dense outputs ready: `{summary.get('receipt_fill_dense_output_ready_count')}` / `{summary.get('receipt_fill_dense_output_entry_count')}`",
        f"- Receipt fill keys: `{summary.get('receipt_fill_key_count')}`",
        f"- Receipt command keys: `{summary.get('receipt_fill_command_key_count')}`",
        f"- Capture queue artifact keys: `{summary.get('capture_queue_artifact_key_count')}`",
        f"- Receipt command validator commands: `{summary.get('receipt_fill_command_validator_command_count')}`",
        f"- Capture queue validator commands: `{summary.get('capture_queue_validator_command_count')}`",
        f"- Receipt command missing validators: `{summary.get('receipt_fill_command_missing_validator_count')}`",
        f"- Receipt command count mismatches: `{summary.get('receipt_fill_command_declared_count_mismatch_count')}`",
        f"- Receipt command parity ready: `{summary.get('receipt_fill_command_key_parity_ready')}`",
        f"- Capture queue parity ready: `{summary.get('receipt_fill_capture_queue_key_parity_ready')}`",
        f"- Receipt command validator coverage ready: `{summary.get('receipt_fill_command_validator_coverage_ready')}`",
        f"- Next unblocked handoff ready: `{summary.get('next_unblocked_operator_handoff_ready')}`",
        f"- Next unblocked handoff package: `{summary.get('next_unblocked_operator_handoff_package_id')}`",
        f"- Next unblocked handoff work order ready: `{summary.get('next_unblocked_operator_handoff_work_order_ready')}`",
        f"- Next unblocked handoff errors: `{summary.get('next_unblocked_operator_handoff_error_count')}`",
        f"- Downstream handoff requests: `{summary.get('downstream_handoff_request_count')}`",
        f"- Downstream policy handoffs ready: `{summary.get('downstream_handoff_policy_ready_count')}`",
        f"- Downstream dense handoffs ready: `{summary.get('downstream_handoff_dense_ready_count')}`",
        f"- Downstream live handoffs ready: `{summary.get('downstream_handoff_live_ready_count')}`",
        f"- Downstream all handoffs ready: `{summary.get('downstream_handoff_all_ready_count')}`",
        f"- Downstream missing sections: `{summary.get('downstream_handoff_missing_section_count')}`",
        f"- Downstream missing paths: `{summary.get('downstream_handoff_missing_path_count')}`",
        f"- Downstream missing validators: `{summary.get('downstream_handoff_missing_validator_count')}`",
        f"- Downstream handoff manifest coverage ready: `{summary.get('downstream_handoff_manifest_coverage_ready')}`",
        f"- Recommended approval command ready: `{summary.get('recommended_approval_command_ready')}`",
        f"- All-request approval commands: `{summary.get('all_request_approval_command_count')}`",
        f"- All-request approval commands ready: `{summary.get('all_request_approval_command_ready_count')}`",
        f"- All-request approval command parity ready: `{summary.get('all_request_approval_commands_ready')}`",
        f"- Approval command manifest entries: `{summary.get('approval_command_manifest_count')}`",
        f"- Approval command manifest ready entries: `{summary.get('approval_command_manifest_ready_count')}`",
        f"- Approval command manifest coverage ready: `{summary.get('approval_command_manifest_coverage_ready')}`",
        f"- Approval command manifest errors: `{summary.get('approval_command_manifest_error_count')}`",
        f"- Recommended capture queue ready: `{summary.get('recommended_capture_queue_ready')}`",
        f"- Launch-card task keys: `{summary.get('launch_card_task_key_count')}`",
        f"- Worksheet recommended task keys: `{summary.get('worksheet_recommended_task_key_count')}`",
        f"- Capture queue recommended task keys: `{summary.get('capture_queue_recommended_task_key_count')}`",
        f"- Binding worksheet task keys: `{summary.get('binding_worksheet_queue_task_key_count')}`",
        f"- Capture queue task keys: `{summary.get('capture_queue_task_key_count')}`",
        f"- Launch-card worksheet task parity ready: `{summary.get('launch_card_worksheet_task_parity_ready')}`",
        f"- Launch-card capture queue task parity ready: `{summary.get('launch_card_capture_queue_task_parity_ready')}`",
        f"- Binding worksheet/capture queue parity ready: `{summary.get('binding_worksheet_capture_queue_task_parity_ready')}`",
        f"- Saved launch-card handoff artifacts ready: `{summary.get('launch_card_saved_handoff_artifacts_ready')}`",
        f"- Saved launch-card handoff artifacts missing: `{summary.get('launch_card_saved_handoff_artifact_missing_count')}`",
        f"- Saved launch-card handoff artifacts drifted: `{summary.get('launch_card_saved_handoff_artifact_drifted_count')}`",
        f"- Saved launch-card handoff parity ready: `{summary.get('launch_card_saved_handoff_parity_ready')}`",
        f"- Repo-relative paths checked: `{summary.get('repo_relative_path_count')}`",
        f"- Repo path safety ready: `{summary.get('repo_path_safety_ready')}`",
        f"- Unsafe repo paths: `{summary.get('repo_path_unsafe_count')}`",
        f"- Blocker closure reason ids: `{summary.get('blocker_closure_reason_id_count')}`",
        f"- Blocker closure unmapped rows: `{summary.get('blocker_closure_unmapped_count')}`",
        f"- Blocker closure missing actions: `{summary.get('blocker_closure_missing_action_count')}`",
        f"- Blocker closure missing primary paths: `{summary.get('blocker_closure_missing_primary_path_count')}`",
        f"- Blocker closure runtime-capture rows: `{summary.get('blocker_closure_runtime_capture_required_count')}`",
        f"- Blocker closure future-adapter rows: `{summary.get('blocker_closure_future_adapter_required_count')}`",
        f"- Blocker closure validator commands: `{summary.get('blocker_closure_validator_command_count')}`",
        f"- Blocker closure missing evidence: `{summary.get('blocker_closure_missing_evidence_count')}`",
        f"- Blocker closure manifest coverage ready: `{summary.get('blocker_closure_manifest_coverage_ready')}`",
        f"- Blocker evidence ledger ready: `{summary.get('blocker_evidence_ledger_ready')}`",
        f"- Blocker evidence ledger rows: `{summary.get('blocker_evidence_ledger_row_count')}` / `{summary.get('blocker_evidence_ledger_expected_missing_evidence_count')}`",
        f"- Blocker evidence ledger runtime-capture rows: `{summary.get('blocker_evidence_ledger_runtime_capture_required_row_count')}`",
        f"- Blocker evidence ledger future-adapter rows: `{summary.get('blocker_evidence_ledger_future_adapter_required_row_count')}`",
        f"- Blocker evidence ledger validator commands: `{summary.get('blocker_evidence_ledger_validator_command_count')}`",
        f"- Blocker evidence ledger coverage ready: `{summary.get('blocker_evidence_ledger_coverage_ready')}`",
        f"- Blocker resolution queue ready: `{summary.get('blocker_resolution_queue_ready')}`",
        f"- Blocker resolution queue packages: `{summary.get('blocker_resolution_queue_work_package_count')}`",
        f"- Blocker resolution queue rows: `{summary.get('blocker_resolution_queue_row_count')}` / `{summary.get('blocker_resolution_queue_expected_ledger_row_count')}`",
        f"- Blocker resolution queue completion gates: `{summary.get('blocker_resolution_queue_completion_gate_count')}`",
        f"- Blocker resolution queue dependency edges: `{summary.get('blocker_resolution_queue_dependency_edge_count')}`",
        f"- Blocker resolution queue coverage ready: `{summary.get('blocker_resolution_queue_coverage_ready')}`",
        "",
        "## Remaining Blockers",
        "",
    ]
    blockers = list_of_strings(summary.get("remaining_blockers"))
    if blockers:
        for blocker in blockers:
            lines.append(f"- `{blocker}`")
    else:
        lines.append("- none")
    errors = list_of_strings(summary.get("errors"))
    lines.extend(["", "## Errors", ""])
    if errors:
        for error in errors:
            lines.append(f"- `{error}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Safety Contract", ""])
    for item in list_of_strings(summary.get("safety_contract")):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def write_markdown_report(summary: JSONDict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_markdown_report(summary), encoding="utf-8")


def print_human_summary(summary: JSONDict) -> None:
    print("MoE Run Anyway Phase 3 operator handoff validation")
    print(f"Valid: {summary['valid']}")
    print(f"Handoff package ready: {summary['handoff_package_ready']}")
    print(f"Metadata only: {summary.get('metadata_only')}")
    print(f"Phase 3 complete: {summary.get('phase3_complete')}")
    print(f"Decision: {summary.get('decision')}")
    print(f"Required files: {summary.get('required_file_count')}")
    print(f"Missing files: {summary.get('missing_file_count')}")
    print(f"Capture requests: {summary.get('capture_queue_request_count')}")
    print(f"Runtime preflight ready: {summary.get('runtime_capture_preflight_ready')}")
    print(f"Runtime preflight pending artifacts: {summary.get('runtime_capture_preflight_pending_artifact_count')}")
    print(f"Runtime preflight validator commands: {summary.get('runtime_capture_preflight_validator_command_count')}")
    print(f"Runtime preflight missing items: {summary.get('runtime_capture_preflight_missing_item_count')}")
    print(f"Runtime preflight errors: {summary.get('runtime_capture_preflight_error_count')}")
    print(f"Runtime command contract ready: {summary.get('runtime_capture_command_contract_ready')}")
    print(f"Runtime command contract runtime ready: {summary.get('runtime_capture_command_contract_runtime_ready')}")
    print(f"Runtime command contract planned captures: {summary.get('runtime_capture_command_contract_planned_capture_count')}")
    print(f"Runtime command contract missing runtime commands: {summary.get('runtime_capture_command_contract_missing_runtime_command_count')}")
    print(f"Runtime command contract errors: {summary.get('runtime_capture_command_contract_error_count')}")
    print(f"Runtime execution manual ready: {summary.get('runtime_capture_execution_manual_ready')}")
    print(f"Runtime execution automated ready: {summary.get('runtime_capture_execution_automated_ready')}")
    print(f"Runtime execution command options: {summary.get('runtime_capture_execution_command_option_count')}")
    print(f"Runtime execution manual captures required: {summary.get('runtime_capture_execution_manual_capture_required_count')}")
    print(f"Runtime execution missing capture commands: {summary.get('runtime_capture_execution_missing_capture_command_count')}")
    print(f"Runtime execution missing items: {summary.get('runtime_capture_execution_missing_item_count')}")
    print(f"Runtime execution errors: {summary.get('runtime_capture_execution_error_count')}")
    print(f"All-request runtime execution manual ready: {summary.get('all_request_runtime_capture_execution_manual_ready')}")
    print(f"All-request runtime execution automated ready: {summary.get('all_request_runtime_capture_execution_automated_ready')}")
    print(f"All-request runtime execution ready requests: {summary.get('all_request_runtime_capture_execution_ready_request_count')} / {summary.get('all_request_runtime_capture_execution_request_count')}")
    print(f"All-request runtime execution pending artifacts: {summary.get('all_request_runtime_capture_execution_pending_artifact_count')}")
    print(f"All-request runtime execution command options: {summary.get('all_request_runtime_capture_execution_command_option_count')}")
    print(f"All-request runtime execution manual captures required: {summary.get('all_request_runtime_capture_execution_manual_capture_required_count')}")
    print(f"All-request runtime execution missing capture commands: {summary.get('all_request_runtime_capture_execution_missing_capture_command_count')}")
    print(f"All-request runtime execution missing items: {summary.get('all_request_runtime_capture_execution_missing_item_count')}")
    print(f"All-request runtime execution errors: {summary.get('all_request_runtime_capture_execution_error_count')}")
    print(f"Filled launch-card directory provided: {summary.get('runtime_capture_launch_card_dir_provided')}")
    print(f"Filled launch-card directory ready: {summary.get('runtime_capture_launch_card_dir_ready')}")
    print(f"Filled launch-card directory matched cards: {summary.get('runtime_capture_launch_card_dir_matched_card_count')} / {summary.get('runtime_capture_launch_card_dir_expected_card_count')}")
    print(f"Filled launch-card directory missing cards: {summary.get('runtime_capture_launch_card_dir_missing_card_count')}")
    print(f"Filled launch-card directory errors: {summary.get('runtime_capture_launch_card_dir_error_count')}")
    print(f"Validator commands: {summary.get('validator_command_count')}")
    print(f"Validator manifest runtime artifacts: {summary.get('validator_manifest_runtime_artifact_count')}")
    print(f"Validator manifest future artifacts: {summary.get('validator_manifest_future_artifact_count')}")
    print(f"Validator manifest runtime commands: {summary.get('validator_manifest_runtime_validator_command_count')}")
    print(f"Validator manifest future commands: {summary.get('validator_manifest_future_validator_command_count')}")
    print(f"Validator manifest coverage ready: {summary.get('validator_manifest_coverage_ready')}")
    print(f"Receipt fills missing: {summary.get('receipt_fill_missing_count')}")
    print(f"Receipt-fill candidate router trace ready: {summary.get('receipt_fill_candidate_router_trace_ready_count')} / {summary.get('receipt_fill_candidate_router_trace_entry_count')}")
    print(f"Receipt-fill managed outputs ready: {summary.get('receipt_fill_managed_output_ready_count')} / {summary.get('receipt_fill_managed_output_entry_count')}")
    print(f"Receipt-fill dense outputs ready: {summary.get('receipt_fill_dense_output_ready_count')} / {summary.get('receipt_fill_dense_output_entry_count')}")
    print(f"Receipt fill keys: {summary.get('receipt_fill_key_count')}")
    print(f"Receipt command keys: {summary.get('receipt_fill_command_key_count')}")
    print(f"Capture queue artifact keys: {summary.get('capture_queue_artifact_key_count')}")
    print(f"Receipt command validator commands: {summary.get('receipt_fill_command_validator_command_count')}")
    print(f"Capture queue validator commands: {summary.get('capture_queue_validator_command_count')}")
    print(f"Receipt command missing validators: {summary.get('receipt_fill_command_missing_validator_count')}")
    print(f"Receipt command count mismatches: {summary.get('receipt_fill_command_declared_count_mismatch_count')}")
    print(f"Receipt command parity ready: {summary.get('receipt_fill_command_key_parity_ready')}")
    print(f"Capture queue parity ready: {summary.get('receipt_fill_capture_queue_key_parity_ready')}")
    print(f"Receipt command validator coverage ready: {summary.get('receipt_fill_command_validator_coverage_ready')}")
    print(f"Next unblocked handoff ready: {summary.get('next_unblocked_operator_handoff_ready')}")
    print(f"Next unblocked handoff package: {summary.get('next_unblocked_operator_handoff_package_id')}")
    print(f"Next unblocked handoff errors: {summary.get('next_unblocked_operator_handoff_error_count')}")
    print(f"Downstream handoff requests: {summary.get('downstream_handoff_request_count')}")
    print(f"Downstream policy handoffs ready: {summary.get('downstream_handoff_policy_ready_count')}")
    print(f"Downstream dense handoffs ready: {summary.get('downstream_handoff_dense_ready_count')}")
    print(f"Downstream live handoffs ready: {summary.get('downstream_handoff_live_ready_count')}")
    print(f"Downstream all handoffs ready: {summary.get('downstream_handoff_all_ready_count')}")
    print(f"Downstream missing sections: {summary.get('downstream_handoff_missing_section_count')}")
    print(f"Downstream missing paths: {summary.get('downstream_handoff_missing_path_count')}")
    print(f"Downstream missing validators: {summary.get('downstream_handoff_missing_validator_count')}")
    print(f"Downstream handoff manifest coverage ready: {summary.get('downstream_handoff_manifest_coverage_ready')}")
    print(f"Recommended approval command ready: {summary.get('recommended_approval_command_ready')}")
    print(f"All-request approval commands: {summary.get('all_request_approval_command_count')}")
    print(f"All-request approval commands ready: {summary.get('all_request_approval_command_ready_count')}")
    print(f"All-request approval command parity ready: {summary.get('all_request_approval_commands_ready')}")
    print(f"Approval command manifest entries: {summary.get('approval_command_manifest_count')}")
    print(f"Approval command manifest ready entries: {summary.get('approval_command_manifest_ready_count')}")
    print(f"Approval command manifest coverage ready: {summary.get('approval_command_manifest_coverage_ready')}")
    print(f"Approval command manifest errors: {summary.get('approval_command_manifest_error_count')}")
    print(f"Recommended capture queue ready: {summary.get('recommended_capture_queue_ready')}")
    print(f"Launch-card task keys: {summary.get('launch_card_task_key_count')}")
    print(f"Worksheet recommended task keys: {summary.get('worksheet_recommended_task_key_count')}")
    print(f"Capture queue recommended task keys: {summary.get('capture_queue_recommended_task_key_count')}")
    print(f"Binding worksheet task keys: {summary.get('binding_worksheet_queue_task_key_count')}")
    print(f"Capture queue task keys: {summary.get('capture_queue_task_key_count')}")
    print(f"Launch-card worksheet task parity ready: {summary.get('launch_card_worksheet_task_parity_ready')}")
    print(f"Launch-card capture queue task parity ready: {summary.get('launch_card_capture_queue_task_parity_ready')}")
    print(f"Binding worksheet/capture queue parity ready: {summary.get('binding_worksheet_capture_queue_task_parity_ready')}")
    print(f"Saved launch-card handoff artifacts ready: {summary.get('launch_card_saved_handoff_artifacts_ready')}")
    print(f"Saved launch-card handoff artifacts missing: {summary.get('launch_card_saved_handoff_artifact_missing_count')}")
    print(f"Saved launch-card handoff artifacts drifted: {summary.get('launch_card_saved_handoff_artifact_drifted_count')}")
    print(f"Saved launch-card handoff parity ready: {summary.get('launch_card_saved_handoff_parity_ready')}")
    print(f"Repo-relative paths checked: {summary.get('repo_relative_path_count')}")
    print(f"Repo path safety ready: {summary.get('repo_path_safety_ready')}")
    print(f"Unsafe repo paths: {summary.get('repo_path_unsafe_count')}")
    print(f"Blocker closure reason ids: {summary.get('blocker_closure_reason_id_count')}")
    print(f"Blocker closure unmapped rows: {summary.get('blocker_closure_unmapped_count')}")
    print(f"Blocker closure missing actions: {summary.get('blocker_closure_missing_action_count')}")
    print(f"Blocker closure missing primary paths: {summary.get('blocker_closure_missing_primary_path_count')}")
    print(f"Blocker closure runtime-capture rows: {summary.get('blocker_closure_runtime_capture_required_count')}")
    print(f"Blocker closure future-adapter rows: {summary.get('blocker_closure_future_adapter_required_count')}")
    print(f"Blocker closure validator commands: {summary.get('blocker_closure_validator_command_count')}")
    print(f"Blocker closure missing evidence: {summary.get('blocker_closure_missing_evidence_count')}")
    print(f"Blocker closure manifest coverage ready: {summary.get('blocker_closure_manifest_coverage_ready')}")
    print(f"Blocker evidence ledger ready: {summary.get('blocker_evidence_ledger_ready')}")
    print(f"Blocker evidence ledger rows: {summary.get('blocker_evidence_ledger_row_count')} / {summary.get('blocker_evidence_ledger_expected_missing_evidence_count')}")
    print(f"Blocker evidence ledger runtime-capture rows: {summary.get('blocker_evidence_ledger_runtime_capture_required_row_count')}")
    print(f"Blocker evidence ledger future-adapter rows: {summary.get('blocker_evidence_ledger_future_adapter_required_row_count')}")
    print(f"Blocker evidence ledger validator commands: {summary.get('blocker_evidence_ledger_validator_command_count')}")
    print(f"Blocker evidence ledger coverage ready: {summary.get('blocker_evidence_ledger_coverage_ready')}")
    print(f"Blocker resolution queue ready: {summary.get('blocker_resolution_queue_ready')}")
    print(f"Blocker resolution queue packages: {summary.get('blocker_resolution_queue_work_package_count')}")
    print(f"Blocker resolution queue rows: {summary.get('blocker_resolution_queue_row_count')} / {summary.get('blocker_resolution_queue_expected_ledger_row_count')}")
    print(f"Blocker resolution queue completion gates: {summary.get('blocker_resolution_queue_completion_gate_count')}")
    print(f"Blocker resolution queue dependency edges: {summary.get('blocker_resolution_queue_dependency_edge_count')}")
    print(f"Blocker resolution queue coverage ready: {summary.get('blocker_resolution_queue_coverage_ready')}")
    blockers = ", ".join(list_of_strings(summary.get("remaining_blockers"))) or "none"
    print(f"Remaining blockers: {blockers}")
    if summary["errors"]:
        print("Errors:")
        for error in summary["errors"]:
            print(f"  - {error}")
    print("Safety contract:")
    for item in summary["safety_contract"]:
        print(f"  - {item}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("handoff_dir", nargs="?", type=Path, help="existing operator handoff directory to validate")
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    parser.add_argument("--output-md", type=Path, help="write a Markdown validation report")
    return parser


def plan_paths(handoff_dir: Path | None = None) -> tuple[int, JSONDict | None, str | None]:
    try:
        summary = build_summary(handoff_dir) if handoff_dir is not None else build_current_temp_summary()
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not validate Phase 3 operator handoff package: {exc}"
    return (0 if summary["valid"] else 2), summary, None


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary, error_message = plan_paths(args.handoff_dir)
    if error_message:
        print(error_message, file=sys.stderr)
        return status
    assert summary is not None
    if args.output_md:
        write_markdown_report(summary, args.output_md)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_human_summary(summary)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
