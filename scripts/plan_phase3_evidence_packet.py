#!/usr/bin/env python3
"""Assemble a reproducible Phase 3 evidence packet.

The packet indexes every offline Phase 3 gate in one place: inventory, expert
store layout, trace/inventory join, real-model pairing, policy definitions,
policy replay, dense fallback comparison, and the go/no-go decision. It does
not launch runtimes, read tensor values, write packed stores, mutate residency,
or send prompt traffic.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
import sys
from pathlib import Path
from typing import Any

import plan_baseline_policy_replay
import plan_baseline_replay_policies
import plan_dense_fallback_comparison
import plan_expert_inventory
import plan_expert_store_layout
import plan_phase3_go_no_go
import plan_phase3_dense_fallback_capture
import plan_phase3_live_capability_proof
import plan_phase3_launch_card_library
import plan_phase3_policy_candidate_trace
import plan_phase3_runtime_actuator_design
import plan_phase3_runtime_actuator_spike
import plan_phase3_runtime_capture_commands
import plan_real_model_trace_inventory_pairing
import plan_trace_inventory_replay


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_PATH = ROOT / "memory-moe-mvp" / "data" / "llama_cpp_router_trace.fixture.jsonl"
DEFAULT_INVENTORY_PATH = ROOT / "memory-moe-mvp" / "data" / "expert_inventory_manifest.fixture.json"
DEFAULT_POLICIES_PATH = ROOT / "memory-moe-mvp" / "data" / "baseline_replay_policies.json"
DEFAULT_MANAGED_PLAN_PATH = ROOT / "memory-moe-mvp" / "data" / "managed_expert_loading_plan.json"
SUPPORTED_SCHEMA_VERSION = "moe-phase3-evidence-packet-v1"
OPERATOR_HANDOFF_SCHEMA_VERSION = "moe-phase3-operator-handoff-v1"

JSONDict = dict[str, Any]


def evidence_item(
    item_id: str,
    *,
    status: str,
    summary: str,
    source: str,
    details: JSONDict | None = None,
) -> JSONDict:
    result: JSONDict = {
        "id": item_id,
        "status": status,
        "summary": summary,
        "source": source,
    }
    if details is not None:
        result["details"] = details
    return result


def status_from_bool(value: bool, *, false_status: str = "blocked") -> str:
    return "proven" if value else false_status


def promotion_checklist_item(
    item_id: str,
    *,
    status: str,
    summary: str,
    next_action: str,
    evidence: JSONDict | None = None,
) -> JSONDict:
    item: JSONDict = {
        "id": item_id,
        "status": status,
        "summary": summary,
        "next_action": next_action,
    }
    if evidence is not None:
        item["evidence"] = evidence
    return item


def int_count(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def reuse_summary_from_real_matrix(real_matrix_summary: JSONDict) -> JSONDict:
    blocker_counts = real_matrix_summary.get("policy_candidate_blocker_counts", {})
    if not isinstance(blocker_counts, dict):
        blocker_counts = {}
    bundle_count = int_count(real_matrix_summary.get("bundle_count"))
    prompt_identity_missing = int_count(
        real_matrix_summary.get("policy_candidate_prompt_identity_metadata_missing_count")
    )
    if prompt_identity_missing == 0:
        prompt_identity_missing = int_count(blocker_counts.get("prompt_identity_metadata_missing"))
    return {
        "valid": real_matrix_summary.get("valid", True),
        "errors": real_matrix_summary.get("errors", []),
        "root": real_matrix_summary.get("root"),
        "bundle_count": bundle_count,
        "reuse_ready_count": int_count(real_matrix_summary.get("policy_candidate_ready_count")),
        "reuse_blocked_count": int_count(real_matrix_summary.get("policy_candidate_blocked_bundle_count")),
        "candidate_trace_valid_count": bundle_count,
        "candidate_trace_receipt_ready_count": int_count(real_matrix_summary.get("policy_candidate_trace_receipt_ready_count")),
        "no_reuse_distance_observation_count": int_count(real_matrix_summary.get("policy_candidate_no_reuse_distance_observation_count")),
        "prompt_identity_ready_count": int_count(real_matrix_summary.get("policy_candidate_prompt_identity_ready_count")),
        "prompt_identity_metadata_missing_count": prompt_identity_missing,
        "recommended_next_capture": real_matrix_summary.get("policy_candidate_next_operator_step"),
        "blocker_counts": dict(blocker_counts),
        "warning_counts": {},
    }


METADATA_ONLY_COMMAND_CLASSES = {"phase3_runtime_capture_request_approval_rebuild"}


def compact_command_options(value: Any) -> list[JSONDict]:
    if not isinstance(value, list):
        return []
    result: list[JSONDict] = []
    for option in value:
        if not isinstance(option, dict) or not option.get("command"):
            continue
        command = option.get("command")
        result.append(
            {
                "command_class": option.get("command_class"),
                "requires_runtime": option.get("requires_runtime") is True,
                "requires_prompt_traffic": option.get("requires_prompt_traffic") is True,
                "may_send_prompt_traffic": option.get("may_send_prompt_traffic") is True,
                "runtime_capture_request_path": normalize_manifest_path(option.get("runtime_capture_request_path")),
                "requires_explicit_user_approval": option.get("requires_explicit_user_approval") is True,
                "metadata_only": option.get("metadata_only") is True,
                "command": [str(part) for part in command] if isinstance(command, list) else [str(command)],
                "model_plane_callable_id": option.get("model_plane_callable_id"),
                "source": option.get("source"),
            }
        )
    return result


def command_option_is_metadata_only(option: JSONDict) -> bool:
    command_class = str(option.get("command_class") or "")
    return option.get("metadata_only") is True or command_class in METADATA_ONLY_COMMAND_CLASSES


def command_option_is_runtime_capture(option: JSONDict) -> bool:
    if not isinstance(option, dict) or not option.get("command"):
        return False
    if command_option_is_metadata_only(option):
        return False
    if option.get("requires_runtime") is True or option.get("requires_prompt_traffic") is True:
        return True
    command_class = str(option.get("command_class") or "").lower()
    if "approval" in command_class or "validator" in command_class:
        return False
    return any(
        marker in command_class
        for marker in (
            "runtime_capture",
            "capture_runtime",
            "capture_output",
            "output_capture",
            "router_trace_capture",
            "model_plane_launch_card",
            "launch_card",
        )
    )


def bound_runtime_capture_commands_by_artifact(command_contract: JSONDict | None) -> dict[str, JSONDict]:
    if not isinstance(command_contract, dict):
        return {}
    request_path = normalize_manifest_path(command_contract.get("request_path"))
    result: dict[str, JSONDict] = {}
    tasks = command_contract.get("capture_tasks") if isinstance(command_contract.get("capture_tasks"), list) else []
    for task in tasks:
        if not isinstance(task, dict) or task.get("command_binding_ready") is not True:
            continue
        artifact_id = str(task.get("artifact_id") or "").strip()
        if not artifact_id:
            continue
        planned = task.get("planned_command_binding") if isinstance(task.get("planned_command_binding"), dict) else {}
        callable_id = str(planned.get("model_plane_callable_id") or "").strip()
        launch_command = [part for part in list_of_strings(planned.get("launch_command")) if part.strip()]
        command = launch_command or (["model-plane-callable", callable_id] if callable_id else [])
        if not command:
            continue
        result[artifact_id] = {
            "command_class": planned.get("command_class") or "phase3_model_plane_launch_card_runtime_capture",
            "requires_runtime": True,
            "requires_prompt_traffic": True,
            "may_send_prompt_traffic": planned.get("may_send_prompt_traffic") is True or planned.get("requires_prompt_traffic") is True,
            "runtime_capture_request_path": normalize_manifest_path(planned.get("runtime_capture_request_path") or request_path),
            "requires_explicit_user_approval": True,
            "metadata_only": False,
            "command": command,
            "model_plane_callable_id": callable_id or None,
            "source": "launch_card_binding",
        }
    return result

def compact_capture_sequence(value: Any) -> list[JSONDict]:
    if not isinstance(value, list):
        return []
    sequence: list[JSONDict] = []
    for step in value:
        if not isinstance(step, dict):
            continue
        sequence.append(
            {
                "id": step.get("id"),
                "stage": step.get("stage"),
                "status": step.get("status"),
                "path": step.get("path"),
                "approval_keys": list_of_strings(step.get("approval_keys")),
                "validator_command_count": int_count(step.get("validator_command_count")),
            }
        )
    return sequence


def compact_approval_rebuild_command(value: Any) -> JSONDict:
    if not isinstance(value, dict):
        return {}
    return {
        "command_class": value.get("command_class"),
        "command": list_of_strings(value.get("command")),
        "records_approval_keys": list_of_strings(value.get("records_approval_keys")),
        "writes_request_path": value.get("writes_request_path"),
        "requires_explicit_user_approval": value.get("requires_explicit_user_approval") is True,
        "metadata_only": value.get("metadata_only") is True,
    }


def compact_post_approval_preview(value: Any) -> JSONDict | None:
    if not isinstance(value, dict):
        return None
    return {
        "preview_only": value.get("preview_only") is True,
        "valid": value.get("valid") is True,
        "status": value.get("status"),
        "ready_for_operator_capture": value.get("ready_for_operator_capture") is True,
        "capture_complete": value.get("capture_complete") is True,
        "mutates_request": value.get("mutates_request") is True,
        "still_requires_capture_artifacts": value.get("still_requires_capture_artifacts") is True,
        "pending_artifact_count": int_count(value.get("pending_artifact_count")),
        "pending_artifact_ids": list_of_strings(value.get("pending_artifact_ids")),
        "next_artifact_id": value.get("next_artifact_id"),
        "next_artifact_path": value.get("next_artifact_path"),
        "requested_status_counts": value.get("requested_status_counts") if isinstance(value.get("requested_status_counts"), dict) else {},
        "records_approval_keys": list_of_strings(value.get("records_approval_keys")),
    }


def compact_step_preview(value: Any) -> JSONDict:
    if not isinstance(value, dict):
        return {}
    return {
        "id": value.get("id"),
        "stage": value.get("stage"),
        "status": value.get("status"),
        "approval_state": value.get("approval_state"),
        "path": value.get("path"),
    }


def compact_receipt_fill_preview(value: Any) -> JSONDict:
    if not isinstance(value, dict):
        return {}
    return {
        "entry_count": int_count(value.get("entry_count")),
        "ready_after_approval_count": int_count(value.get("ready_after_approval_count")),
        "missing_after_approval_count": int_count(value.get("missing_after_approval_count")),
        "approval_missing_after_approval_count": int_count(value.get("approval_missing_after_approval_count")),
        "all_ready_after_approval": value.get("all_ready_after_approval") is True,
    }


def compact_approval_transition_preview(value: Any) -> JSONDict | None:
    if not isinstance(value, dict):
        return None
    return {
        "preview_only": value.get("preview_only") is True,
        "valid": value.get("valid") is True,
        "request_name": value.get("request_name"),
        "request_path": value.get("request_path"),
        "prompt_set_path": normalize_manifest_path(value.get("prompt_set_path")),
        "rank": value.get("rank"),
        "selection_rationale": value.get("selection_rationale"),
        "current_next_step": compact_step_preview(value.get("current_next_step")),
        "next_step_after_approval": compact_step_preview(value.get("next_step_after_approval")),
        "records_approval_keys": list_of_strings(value.get("records_approval_keys")),
        "requires_explicit_user_approval": value.get("requires_explicit_user_approval") is True,
        "metadata_only": value.get("metadata_only") is True,
        "mutates_request": value.get("mutates_request") is True,
        "capture_complete_after_approval": value.get("capture_complete_after_approval") is True,
        "approved_but_capture_incomplete_after_approval": value.get("approved_but_capture_incomplete_after_approval") is True,
        "ready_to_update_bundle_after_approval": value.get("ready_to_update_bundle_after_approval") is True,
        "receipt_fill_preview": compact_receipt_fill_preview(value.get("receipt_fill_preview")),
    }


def compact_post_approval_capture_fill_step(value: Any) -> JSONDict:
    if not isinstance(value, dict):
        return {}
    return {
        "artifact_id": value.get("artifact_id"),
        "request_path": normalize_manifest_path(value.get("request_path")),
        "prompt_set_path": normalize_manifest_path(value.get("prompt_set_path")),
        "source_request_path": normalize_manifest_path(value.get("source_request_path")),
        "source_prompt_set_path": normalize_manifest_path(value.get("source_prompt_set_path")),
        "queue_step_id": value.get("queue_step_id"),
        "receipt_kind": value.get("receipt_kind"),
        "artifact_path": value.get("artifact_path"),
        "receipt_path": value.get("receipt_path"),
        "current_step": compact_step_preview(value.get("current_step")),
        "step_after_approval": compact_step_preview(value.get("step_after_approval")),
        "fill_status_after_approval": value.get("fill_status_after_approval"),
        "ready_after_approval": value.get("ready_after_approval") is True,
        "receipt_ready": value.get("receipt_ready") is True,
        "validator_command_count": int_count(value.get("validator_command_count")),
        "command_option_count": int_count(value.get("command_option_count")),
        "command_options": compact_command_options(value.get("command_options")),
        "blockers_after_approval": list_of_strings(value.get("blockers_after_approval")),
    }


def compact_post_approval_capture_fill_plan(value: Any) -> JSONDict | None:
    if not isinstance(value, dict):
        return None
    steps = value.get("capture_fill_steps") if isinstance(value.get("capture_fill_steps"), list) else []
    return {
        "preview_only": value.get("preview_only") is True,
        "valid": value.get("valid") is True,
        "request_name": value.get("request_name"),
        "request_path": value.get("request_path"),
        "prompt_set_path": normalize_manifest_path(value.get("prompt_set_path")),
        "rank": value.get("rank"),
        "selection_rationale": value.get("selection_rationale"),
        "records_approval_keys": list_of_strings(value.get("records_approval_keys")),
        "requires_explicit_user_approval": value.get("requires_explicit_user_approval") is True,
        "metadata_only": value.get("metadata_only") is True,
        "mutates_request": value.get("mutates_request") is True,
        "artifact_step_count": int_count(value.get("artifact_step_count")),
        "runtime_capture_step_count": int_count(value.get("runtime_capture_step_count")),
        "ready_after_approval_count": int_count(value.get("ready_after_approval_count")),
        "missing_after_approval_count": int_count(value.get("missing_after_approval_count")),
        "validator_command_count": int_count(value.get("validator_command_count")),
        "command_option_count": int_count(value.get("command_option_count")),
        "all_receipts_ready_after_approval": value.get("all_receipts_ready_after_approval") is True,
        "ready_to_update_bundle_after_approval": value.get("ready_to_update_bundle_after_approval") is True,
        "capture_fill_steps": [compact_post_approval_capture_fill_step(step) for step in steps if isinstance(step, dict)],
    }


def normalize_manifest_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().replace("\\", "/")


def repo_path(value: Any) -> Path | None:
    path_text = normalize_manifest_path(value)
    if path_text is None:
        return None
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def command_option_value(command: Any, option: str) -> str | None:
    parts = list_of_strings(command)
    for index, part in enumerate(parts[:-1]):
        if part == option:
            return parts[index + 1]
    return None


def compact_artifact_request_statuses(value: Any) -> dict[str, str]:
    if not isinstance(value, list):
        return {}
    statuses: dict[str, str] = {}
    for item in value:
        if not isinstance(item, dict):
            continue
        request_id = item.get("id")
        status = item.get("status")
        if isinstance(request_id, str) and request_id.strip() and isinstance(status, str):
            statuses[request_id.strip()] = status
    return statuses


def compact_command_classes(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    classes: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        command_class = item.get("command_class")
        if isinstance(command_class, str) and command_class.strip():
            classes.append(command_class.strip())
    return classes


def compact_dense_fallback_capture_plan(value: JSONDict) -> JSONDict:
    prompt_set = value.get("prompt_set") if isinstance(value.get("prompt_set"), dict) else {}
    managed = value.get("managed_output_summary") if isinstance(value.get("managed_output_summary"), dict) else {}
    dense = value.get("dense_output_summary") if isinstance(value.get("dense_output_summary"), dict) else {}
    coverage = value.get("prompt_coverage") if isinstance(value.get("prompt_coverage"), dict) else {}
    managed_coverage = coverage.get("managed") if isinstance(coverage.get("managed"), dict) else {}
    dense_coverage = coverage.get("dense") if isinstance(coverage.get("dense"), dict) else {}
    runtime_contract = (
        value.get("runtime_capture_contract")
        if isinstance(value.get("runtime_capture_contract"), dict)
        else {}
    )
    approvals = value.get("approvals") if isinstance(value.get("approvals"), dict) else {}
    policy_warning = value.get("policy_warning") if isinstance(value.get("policy_warning"), dict) else {}
    command_classes = compact_command_classes(value.get("commands"))
    return {
        "valid": value.get("valid") is True,
        "errors": list_of_strings(value.get("errors")),
        "bundle_path": value.get("bundle_path"),
        "model_id": value.get("model_id"),
        "backend_family": value.get("backend_family"),
        "prompt_family": value.get("prompt_family"),
        "prompt_set_path": value.get("prompt_set_path"),
        "managed_output_path": value.get("managed_output_path"),
        "dense_output_path": value.get("dense_output_path"),
        "fallback_artifact_path": value.get("fallback_artifact_path"),
        "updated_bundle_path": value.get("updated_bundle_path"),
        "prompt_set_ready": prompt_set.get("ready") is True,
        "managed_output_exists": managed.get("exists") is True,
        "managed_output_valid": managed.get("valid") is True,
        "managed_output_ready": managed.get("ready") is True,
        "managed_output_capture_receipt_ready": managed.get("capture_receipt_ready") is True,
        "managed_output_present_count": int_count(managed.get("output_present_count")),
        "managed_output_missing_count": int_count(managed.get("missing_output_count")),
        "managed_output_prompt_ids": list_of_strings(managed.get("prompt_ids")),
        "managed_output_blocker": managed.get("blocker"),
        "dense_output_exists": dense.get("exists") is True,
        "dense_output_valid": dense.get("valid") is True,
        "dense_output_ready": dense.get("ready") is True,
        "dense_output_capture_receipt_ready": dense.get("capture_receipt_ready") is True,
        "dense_output_present_count": int_count(dense.get("output_present_count")),
        "dense_output_missing_count": int_count(dense.get("missing_output_count")),
        "dense_output_prompt_ids": list_of_strings(dense.get("prompt_ids")),
        "dense_output_blocker": dense.get("blocker"),
        "managed_prompt_coverage_ready": managed_coverage.get("ready") is True,
        "dense_prompt_coverage_ready": dense_coverage.get("ready") is True,
        "metadata_ready_to_build_comparison": value.get("metadata_ready_to_build_comparison") is True,
        "comparison_ready": value.get("comparison_ready") is True,
        "phase3_fallback_ready": value.get("phase3_fallback_ready") is True,
        "artifact_request_statuses": compact_artifact_request_statuses(value.get("artifact_requests")),
        "command_classes": command_classes,
        "command_count": len(command_classes),
        "runtime_requires_explicit_approval": runtime_contract.get("requires_explicit_approval") is True,
        "dense_fallback_capture_approved": approvals.get("dense_fallback_capture_approved") is True,
        "runtime_prompt_traffic_approved": approvals.get("runtime_prompt_traffic_approved") is True,
        "policy_warning_id": policy_warning.get("id"),
    }


def dense_fallback_capture_handoff_ready(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    command_classes = set(list_of_strings(value.get("command_classes")))
    required_commands = {
        "managed_output_summary_template_builder",
        "dense_output_summary_template_builder",
        "dense_fallback_comparison_builder",
        "dense_fallback_comparison_validator",
    }
    return (
        value.get("valid") is True
        and value.get("prompt_set_ready") is True
        and bool(value.get("managed_output_path"))
        and bool(value.get("dense_output_path"))
        and bool(value.get("fallback_artifact_path"))
        and required_commands.issubset(command_classes)
    )


def build_recommended_dense_fallback_capture_plan(recommendation: JSONDict | None) -> JSONDict | None:
    if not isinstance(recommendation, dict):
        return None
    bundle_path = repo_path(recommendation.get("bundle_path"))
    if bundle_path is None or not bundle_path.exists():
        return None
    approval_command = (
        recommendation.get("approval_rebuild_command", {}).get("command")
        if isinstance(recommendation.get("approval_rebuild_command"), dict)
        else []
    )
    prompt_set_path = repo_path(
        command_option_value(approval_command, "--prompt-set-path") or recommendation.get("prompt_set_path")
    )
    managed_output_path = repo_path(command_option_value(approval_command, "--managed-output-path") or recommendation.get("managed_output_path"))
    dense_output_path = repo_path(command_option_value(approval_command, "--dense-output-path") or recommendation.get("dense_output_path"))
    missing_approvals = set(list_of_strings(recommendation.get("missing_approval_keys")))
    try:
        plan = plan_phase3_dense_fallback_capture.build_capture_plan(
            bundle_path,
            prompt_set_path=prompt_set_path,
            managed_output_path=managed_output_path,
            dense_output_path=dense_output_path,
            dense_fallback_capture_approved=(
                "managed_output_capture_approved" not in missing_approvals
                and "dense_output_capture_approved" not in missing_approvals
            ),
            runtime_prompt_traffic_approved="runtime_prompt_traffic_approved" not in missing_approvals,
        )
    except Exception as exc:  # pragma: no cover - preserved as packet evidence if local artifacts drift.
        return {
            "valid": False,
            "errors": [str(exc)],
            "bundle_path": normalize_manifest_path(recommendation.get("bundle_path")),
            "prompt_set_path": normalize_manifest_path(recommendation.get("prompt_set_path")),
            "managed_output_path": normalize_manifest_path(command_option_value(approval_command, "--managed-output-path")),
            "dense_output_path": normalize_manifest_path(command_option_value(approval_command, "--dense-output-path")),
            "fallback_artifact_path": None,
            "prompt_set_ready": False,
            "metadata_ready_to_build_comparison": False,
            "comparison_ready": False,
            "phase3_fallback_ready": False,
            "command_classes": [],
            "command_count": 0,
        }
    return compact_dense_fallback_capture_plan(plan)


def compact_live_capability_proof_handoff(
    value: JSONDict,
    *,
    proof_artifact_path: str | None = None,
    source_bundle_path: str | None = None,
) -> JSONDict:
    section_status = value.get("section_status") if isinstance(value.get("section_status"), dict) else {}
    phase_gate = value.get("phase_3_gate") if isinstance(value.get("phase_3_gate"), dict) else {}
    context_binding = value.get("context_binding") if isinstance(value.get("context_binding"), dict) else {}
    blockers = value.get("blockers") if isinstance(value.get("blockers"), list) else []

    def section_ready(section_id: str) -> bool:
        section = section_status.get(section_id) if isinstance(section_status.get(section_id), dict) else {}
        return section.get("ready") is True

    return {
        "valid": value.get("valid") is True,
        "errors": list_of_strings(value.get("errors")),
        "proof_artifact_path": normalize_manifest_path(proof_artifact_path) or normalize_manifest_path(value.get("proof_artifact_path")),
        "proof_available": value.get("proof_available") is True,
        "proof_ready": value.get("proof_ready") is True,
        "schema_version": value.get("schema_version"),
        "name": value.get("name"),
        "model_id": value.get("model_id"),
        "backend_family": value.get("backend_family"),
        "prompt_family": value.get("prompt_family"),
        "source_bundle_path": normalize_manifest_path(source_bundle_path) or normalize_manifest_path(value.get("source_bundle_path")),
        "proof_scope": value.get("proof_scope"),
        "context_binding_ready": context_binding.get("ready") is True,
        "residency_observation_ready": section_ready("residency_observation"),
        "residency_control_ready": section_ready("residency_control"),
        "cleanup_restore_ready": section_ready("cleanup_restore"),
        "artifact_export_ready": section_ready("artifact_export"),
        "live_residency_observation_and_control_ready": phase_gate.get("live_residency_observation_and_control_ready") is True,
        "cleanup_restore_proof_ready": phase_gate.get("cleanup_restore_proof_ready") is True,
        "ready_for_live_spike": phase_gate.get("ready_for_live_spike") is True,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "future_adapter_required": value.get("proof_ready") is not True,
    }


def live_capability_proof_handoff_ready(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        value.get("valid") is True
        and value.get("proof_available") is True
        and bool(value.get("proof_artifact_path"))
        and bool(value.get("source_bundle_path"))
        and value.get("context_binding_ready") is True
    )


def build_recommended_live_capability_proof_handoff(recommendation: JSONDict | None) -> JSONDict | None:
    if not isinstance(recommendation, dict):
        return None
    bundle_path = repo_path(recommendation.get("bundle_path"))
    if bundle_path is None or not bundle_path.exists():
        return None
    approval_command = (
        recommendation.get("approval_rebuild_command", {}).get("command")
        if isinstance(recommendation.get("approval_rebuild_command"), dict)
        else []
    )
    live_proof_path_text = command_option_value(approval_command, "--live-proof-template-path") or recommendation.get("live_proof_template_path")
    live_proof_path = repo_path(live_proof_path_text)
    try:
        manifest = json.loads(bundle_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            manifest = {}
        summary = plan_phase3_live_capability_proof.build_summary(
            live_proof_path,
            expected_model_id=str(manifest.get("model_id")) if manifest.get("model_id") is not None else recommendation.get("model_id"),
            expected_backend_family=str(manifest.get("backend_family")) if manifest.get("backend_family") is not None else None,
            expected_prompt_family=str(manifest.get("prompt_family")) if manifest.get("prompt_family") is not None else None,
            expected_source_bundle_path=bundle_path,
        )
    except Exception as exc:  # pragma: no cover - preserved as packet evidence if local artifacts drift.
        return {
            "valid": False,
            "errors": [str(exc)],
            "proof_artifact_path": normalize_manifest_path(live_proof_path_text),
            "proof_available": False,
            "proof_ready": False,
            "source_bundle_path": normalize_manifest_path(recommendation.get("bundle_path")),
            "context_binding_ready": False,
            "residency_observation_ready": False,
            "residency_control_ready": False,
            "cleanup_restore_ready": False,
            "artifact_export_ready": False,
            "ready_for_live_spike": False,
            "blocker_count": 0,
            "blockers": [],
            "future_adapter_required": True,
        }
    return compact_live_capability_proof_handoff(
        summary,
        proof_artifact_path=live_proof_path_text,
        source_bundle_path=recommendation.get("bundle_path"),
    )

def compact_policy_candidate_trace_plan(value: JSONDict) -> JSONDict:
    candidate_trace = value.get("candidate_trace") if isinstance(value.get("candidate_trace"), dict) else {}
    capture_receipt = (
        candidate_trace.get("capture_receipt")
        if isinstance(candidate_trace.get("capture_receipt"), dict)
        else {}
    )
    prompt_set = value.get("prompt_set") if isinstance(value.get("prompt_set"), dict) else {}
    current_replay = value.get("current_replay") if isinstance(value.get("current_replay"), dict) else {}
    runtime_contract = (
        value.get("runtime_capture_contract")
        if isinstance(value.get("runtime_capture_contract"), dict)
        else {}
    )
    trace_contract = (
        value.get("candidate_trace_contract")
        if isinstance(value.get("candidate_trace_contract"), dict)
        else {}
    )
    approvals = value.get("approvals") if isinstance(value.get("approvals"), dict) else {}
    command_classes = compact_command_classes(value.get("commands"))
    return {
        "valid": value.get("valid") is True,
        "errors": list_of_strings(value.get("errors")),
        "bundle_path": value.get("bundle_path"),
        "model_id": value.get("model_id"),
        "backend_family": value.get("backend_family"),
        "prompt_family": value.get("prompt_family"),
        "candidate_prompt_set_path": value.get("candidate_prompt_set_path"),
        "candidate_trace_path": value.get("candidate_trace_path"),
        "candidate_trace_receipt_path": value.get("candidate_trace_receipt_path"),
        "updated_bundle_path": value.get("updated_bundle_path"),
        "current_policy_candidate_ready": current_replay.get("policy_candidate_ready") is True,
        "current_reuse_distance_observations": int_count(current_replay.get("reuse_distance_observations")),
        "prompt_set_ready": prompt_set.get("ready") is True,
        "candidate_trace_exists": candidate_trace.get("exists") is True,
        "candidate_trace_valid": candidate_trace.get("valid") is True,
        "candidate_trace_replay_valid": candidate_trace.get("replay_valid") is True,
        "candidate_trace_policy_candidate_ready": candidate_trace.get("policy_candidate_ready") is True,
        "candidate_trace_reuse_distance_observations": int_count(candidate_trace.get("reuse_distance_observations")),
        "candidate_trace_blocker": candidate_trace.get("blocker"),
        "capture_receipt_ready": capture_receipt.get("ready") is True,
        "capture_receipt_path_matches": capture_receipt.get("candidate_trace_path_matches") is True,
        "capture_receipt_prompt_set_matches": capture_receipt.get("prompt_set_path_matches") is True,
        "capture_receipt_request_matches": capture_receipt.get("request_path_matches") is True,
        "policy_candidate_ready_after_plan": value.get("policy_candidate_ready_after_plan") is True,
        "artifact_request_statuses": compact_artifact_request_statuses(value.get("artifact_requests")),
        "command_classes": command_classes,
        "command_count": len(command_classes),
        "runtime_requires_explicit_approval": runtime_contract.get("requires_explicit_approval") is True,
        "policy_candidate_trace_capture_approved": approvals.get("policy_candidate_trace_capture_approved") is True,
        "runtime_prompt_traffic_approved": approvals.get("runtime_prompt_traffic_approved") is True,
        "min_reuse_distance_observations": int_count(trace_contract.get("min_reuse_distance_observations")),
    }


def policy_candidate_trace_handoff_ready(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    command_classes = set(list_of_strings(value.get("command_classes")))
    required_commands = {
        "candidate_trace_receipt_validator",
        "candidate_trace_contract_validator",
        "candidate_policy_replay",
    }
    return (
        value.get("valid") is True
        and value.get("prompt_set_ready") is True
        and bool(value.get("candidate_trace_path"))
        and bool(value.get("candidate_trace_receipt_path"))
        and value.get("min_reuse_distance_observations") == 1
        and required_commands.issubset(command_classes)
    )


def build_recommended_policy_candidate_trace_plan(recommendation: JSONDict | None) -> JSONDict | None:
    if not isinstance(recommendation, dict):
        return None
    bundle_path = repo_path(recommendation.get("bundle_path"))
    if bundle_path is None or not bundle_path.exists():
        return None
    approval_command = (
        recommendation.get("approval_rebuild_command", {}).get("command")
        if isinstance(recommendation.get("approval_rebuild_command"), dict)
        else []
    )
    prompt_set_path = repo_path(
        command_option_value(approval_command, "--prompt-set-path") or recommendation.get("prompt_set_path")
    )
    candidate_trace_path = repo_path(
        command_option_value(approval_command, "--candidate-trace-path") or recommendation.get("next_artifact_path")
    )
    candidate_trace_receipt_path = repo_path(command_option_value(approval_command, "--candidate-trace-receipt-path") or recommendation.get("candidate_trace_receipt_path"))
    source_request_path = repo_path(recommendation.get("request_path"))
    missing_approvals = set(list_of_strings(recommendation.get("missing_approval_keys")))
    try:
        plan = plan_phase3_policy_candidate_trace.build_capture_plan(
            bundle_path,
            candidate_prompt_set_path=prompt_set_path,
            candidate_trace_path=candidate_trace_path,
            candidate_trace_receipt_path=candidate_trace_receipt_path,
            source_request_path=source_request_path,
            policy_candidate_trace_capture_approved="router_trace_capture_approved" not in missing_approvals,
            runtime_prompt_traffic_approved="runtime_prompt_traffic_approved" not in missing_approvals,
        )
    except Exception as exc:  # pragma: no cover - preserved as packet evidence if local artifacts drift.
        return {
            "valid": False,
            "errors": [str(exc)],
            "bundle_path": normalize_manifest_path(recommendation.get("bundle_path")),
            "candidate_trace_path": normalize_manifest_path(recommendation.get("next_artifact_path")),
            "candidate_trace_receipt_path": normalize_manifest_path(
                command_option_value(approval_command, "--candidate-trace-receipt-path")
            ),
            "prompt_set_ready": False,
            "policy_candidate_ready_after_plan": False,
            "command_classes": [],
            "command_count": 0,
        }
    return compact_policy_candidate_trace_plan(plan)


def approval_manifest_request_path(item: JSONDict) -> str | None:
    return normalize_manifest_path(item.get("writes_request_path") or item.get("request_path"))


def approval_manifest_by_request_path(value: Any) -> dict[str, JSONDict]:
    if not isinstance(value, list):
        return {}
    result: dict[str, JSONDict] = {}
    for item in value:
        if not isinstance(item, dict):
            continue
        request_path = approval_manifest_request_path(item)
        if request_path is not None:
            result[request_path] = item
    return result


def approval_manifest_metadata_mismatch(runtime_item: JSONDict, intake_item: JSONDict) -> bool:
    for key in ("command_class", "requires_explicit_user_approval", "metadata_only"):
        if runtime_item.get(key) != intake_item.get(key):
            return True
    runtime_writes = normalize_manifest_path(runtime_item.get("writes_request_path"))
    intake_writes = normalize_manifest_path(intake_item.get("writes_request_path"))
    if runtime_writes != intake_writes:
        return True
    if sorted(list_of_strings(runtime_item.get("records_approval_keys"))) != sorted(
        list_of_strings(intake_item.get("records_approval_keys"))
    ):
        return True
    return False


def approval_manifest_parity_summary(runtime_request_summary: JSONDict, capture_intake_summary: JSONDict) -> JSONDict:
    runtime_manifest = runtime_request_summary.get("approval_rebuild_command_manifest")
    capture_manifest = capture_intake_summary.get("approval_rebuild_command_manifest")
    runtime_by_path = approval_manifest_by_request_path(runtime_manifest)
    capture_by_path = approval_manifest_by_request_path(capture_manifest)
    runtime_paths = set(runtime_by_path)
    capture_paths = set(capture_by_path)
    shared_paths = sorted(runtime_paths & capture_paths)
    metadata_mismatch_paths = [
        path
        for path in shared_paths
        if approval_manifest_metadata_mismatch(runtime_by_path[path], capture_by_path[path])
    ]
    command_mismatch_paths = [
        path
        for path in shared_paths
        if list_of_strings(runtime_by_path[path].get("command")) != list_of_strings(capture_by_path[path].get("command"))
    ]
    runtime_count = len(runtime_manifest) if isinstance(runtime_manifest, list) else 0
    capture_count = len(capture_manifest) if isinstance(capture_manifest, list) else 0
    ready = (
        runtime_count > 0
        and capture_count > 0
        and runtime_count == capture_count
        and runtime_paths == capture_paths
        and not metadata_mismatch_paths
        and not command_mismatch_paths
    )
    return {
        "ready": ready,
        "comparison_basis": "writes_request_path_or_request_path",
        "runtime_manifest_count": runtime_count,
        "capture_result_manifest_count": capture_count,
        "matched_request_count": len(shared_paths),
        "missing_from_capture_result_request_paths": sorted(runtime_paths - capture_paths),
        "missing_from_runtime_request_paths": sorted(capture_paths - runtime_paths),
        "metadata_mismatch_request_paths": metadata_mismatch_paths,
        "command_mismatch_request_paths": command_mismatch_paths,
    }



def receipt_requirement_key(item: JSONDict) -> str | None:
    request_path = normalize_manifest_path(item.get("request_path"))
    artifact_id = item.get("artifact_id")
    if not request_path or not isinstance(artifact_id, str) or not artifact_id.strip():
        return None
    return f"{request_path}::{artifact_id.strip()}"


def receipt_requirement_item(
    *,
    request_path: str | None,
    artifact_id: Any,
    artifact_path: Any,
    receipt_path: Any,
) -> JSONDict | None:
    normalized_request_path = normalize_manifest_path(request_path)
    if not normalized_request_path or not isinstance(artifact_id, str) or not artifact_id.strip():
        return None
    normalized_artifact_path = normalize_manifest_path(artifact_path)
    normalized_receipt_path = normalize_manifest_path(receipt_path) or normalized_artifact_path
    return {
        "request_path": normalized_request_path,
        "artifact_id": artifact_id.strip(),
        "artifact_path": normalized_artifact_path,
        "receipt_path": normalized_receipt_path,
    }


def runtime_receipt_requirement_manifest(runtime_request_summary: JSONDict) -> list[JSONDict]:
    requests = runtime_request_summary.get("requests")
    if not isinstance(requests, list):
        return []
    manifest: list[JSONDict] = []
    for request in requests:
        if not isinstance(request, dict):
            continue
        request_path = normalize_manifest_path(request.get("path"))
        handoff = request.get("operator_handoff") if isinstance(request.get("operator_handoff"), dict) else {}
        requested = handoff.get("requested_artifacts") if isinstance(handoff.get("requested_artifacts"), list) else []
        for artifact in requested:
            if not isinstance(artifact, dict):
                continue
            source = artifact.get("source") if isinstance(artifact.get("source"), dict) else {}
            if source.get("capture_receipt_required") is not True and not source.get("receipt_fill_note"):
                continue
            item = receipt_requirement_item(
                request_path=request_path,
                artifact_id=artifact.get("id"),
                artifact_path=artifact.get("path"),
                receipt_path=source.get("capture_receipt_path"),
            )
            if item is not None:
                manifest.append(item)
    return manifest


def intake_receipt_requirement_manifest(capture_intake_summary: JSONDict) -> list[JSONDict]:
    requests = capture_intake_summary.get("requests")
    if not isinstance(requests, list):
        return []
    manifest: list[JSONDict] = []
    for request in requests:
        if not isinstance(request, dict):
            continue
        request_path = normalize_manifest_path(request.get("path"))
        policy = request.get("policy_candidate") if isinstance(request.get("policy_candidate"), dict) else {}
        trace_receipt = (
            policy.get("candidate_trace_capture_receipt")
            if isinstance(policy.get("candidate_trace_capture_receipt"), dict)
            else {}
        )
        binding = request.get("capture_receipt_binding") if isinstance(request.get("capture_receipt_binding"), dict) else {}
        managed = binding.get("managed") if isinstance(binding.get("managed"), dict) else {}
        dense = binding.get("dense") if isinstance(binding.get("dense"), dict) else {}
        candidates = (
            (
                "candidate_router_trace",
                request.get("candidate_trace_path"),
                request.get("candidate_trace_receipt_path") or trace_receipt.get("path"),
            ),
            (
                "managed_output_summary_fill",
                request.get("managed_output_path") or managed.get("path"),
                request.get("managed_output_path") or managed.get("path"),
            ),
            (
                "dense_output_summary_fill",
                request.get("dense_output_path") or dense.get("path"),
                request.get("dense_output_path") or dense.get("path"),
            ),
        )
        for artifact_id, artifact_path, receipt_path in candidates:
            item = receipt_requirement_item(
                request_path=request_path,
                artifact_id=artifact_id,
                artifact_path=artifact_path,
                receipt_path=receipt_path,
            )
            if item is not None:
                manifest.append(item)
    return manifest


def receipt_requirement_by_key(manifest: list[JSONDict]) -> dict[str, JSONDict]:
    result: dict[str, JSONDict] = {}
    for item in manifest:
        key = receipt_requirement_key(item)
        if key is not None:
            result[key] = item
    return result


def receipt_requirement_mismatches(runtime_item: JSONDict, intake_item: JSONDict) -> list[str]:
    mismatches: list[str] = []
    for key in ("artifact_path", "receipt_path"):
        if normalize_manifest_path(runtime_item.get(key)) != normalize_manifest_path(intake_item.get(key)):
            mismatches.append(key)
    return mismatches


def receipt_requirement_parity_summary(runtime_request_summary: JSONDict, capture_intake_summary: JSONDict) -> JSONDict:
    runtime_manifest = runtime_receipt_requirement_manifest(runtime_request_summary)
    intake_manifest = intake_receipt_requirement_manifest(capture_intake_summary)
    runtime_by_key = receipt_requirement_by_key(runtime_manifest)
    intake_by_key = receipt_requirement_by_key(intake_manifest)
    runtime_keys = set(runtime_by_key)
    intake_keys = set(intake_by_key)
    shared_keys = sorted(runtime_keys & intake_keys)
    metadata_mismatches = []
    for key in shared_keys:
        mismatches = receipt_requirement_mismatches(runtime_by_key[key], intake_by_key[key])
        if mismatches:
            metadata_mismatches.append(
                {
                    "requirement_key": key,
                    "mismatch_fields": mismatches,
                }
            )
    ready = (
        bool(runtime_manifest)
        and bool(intake_manifest)
        and len(runtime_manifest) == len(intake_manifest)
        and runtime_keys == intake_keys
        and not metadata_mismatches
    )
    return {
        "ready": ready,
        "comparison_basis": "request_path_and_artifact_id",
        "runtime_requirement_count": len(runtime_manifest),
        "capture_result_requirement_count": len(intake_manifest),
        "matched_requirement_count": len(shared_keys),
        "missing_from_capture_result_requirement_keys": sorted(runtime_keys - intake_keys),
        "missing_from_runtime_request_requirement_keys": sorted(intake_keys - runtime_keys),
        "metadata_mismatch_requirements": metadata_mismatches,
    }

def runtime_operator_queue_manifest(runtime_request_summary: JSONDict) -> list[JSONDict]:
    queue = runtime_request_summary.get("capture_queue_summary")
    if not isinstance(queue, dict):
        return []
    ranked = queue.get("ranked_requests")
    if not isinstance(ranked, list):
        return []
    requests_by_name: dict[str, JSONDict] = {}
    requests = runtime_request_summary.get("requests")
    if isinstance(requests, list):
        for request in requests:
            if not isinstance(request, dict):
                continue
            request_name = request.get("name") or request.get("path")
            if isinstance(request_name, str) and request_name.strip():
                requests_by_name[request_name.strip()] = request
    manifest: list[JSONDict] = []
    for item in ranked:
        if not isinstance(item, dict):
            continue
        request_name = item.get("request_name")
        if not isinstance(request_name, str) or not request_name.strip():
            continue
        next_step_id = item.get("next_artifact_id")
        next_step_path = None
        request = requests_by_name.get(request_name.strip())
        if isinstance(request, dict):
            handoff = request.get("operator_handoff") if isinstance(request.get("operator_handoff"), dict) else {}
            artifacts = handoff.get("requested_artifacts") if isinstance(handoff.get("requested_artifacts"), list) else []
            for artifact in artifacts:
                if isinstance(artifact, dict) and artifact.get("id") == next_step_id:
                    next_step_path = normalize_manifest_path(artifact.get("path"))
                    break
        manifest.append(
            {
                "request_name": request_name.strip(),
                "rank": item.get("rank"),
                "status": item.get("status"),
                "next_step_id": next_step_id,
                "path": next_step_path,
                "pending_artifact_count": int_count(item.get("pending_artifact_count")),
                "missing_approval_count": int_count(item.get("missing_approval_count")),
            }
        )
    return manifest


def intake_operator_queue_manifest(capture_intake_summary: JSONDict) -> list[JSONDict]:
    requests = capture_intake_summary.get("requests")
    if not isinstance(requests, list):
        return []
    manifest: list[JSONDict] = []
    for request in requests:
        if not isinstance(request, dict):
            continue
        request_name = request.get("name") or request.get("path")
        if not isinstance(request_name, str) or not request_name.strip():
            continue
        step = request.get("next_operator_step") if isinstance(request.get("next_operator_step"), dict) else {}
        command_options = step.get("command_options") if isinstance(step.get("command_options"), list) else []
        manifest.append(
            {
                "request_name": request_name.strip(),
                "request_path": normalize_manifest_path(request.get("path")),
                "status": step.get("status"),
                "next_step_id": step.get("id"),
                "path": normalize_manifest_path(step.get("path")),
                "approval_state": step.get("approval_state"),
                "command_option_count": len(command_options),
            }
        )
    return manifest


def operator_queue_by_request_name(manifest: list[JSONDict]) -> dict[str, JSONDict]:
    result: dict[str, JSONDict] = {}
    for item in manifest:
        request_name = item.get("request_name")
        if isinstance(request_name, str) and request_name.strip():
            result[request_name.strip()] = item
    return result


def operator_queue_mismatches(runtime_item: JSONDict, intake_item: JSONDict) -> list[str]:
    mismatches: list[str] = []
    for runtime_key, intake_key in (("next_step_id", "next_step_id"), ("status", "status"), ("path", "path")):
        if runtime_item.get(runtime_key) != intake_item.get(intake_key):
            mismatches.append(runtime_key)
    return mismatches


def operator_queue_parity_summary(runtime_request_summary: JSONDict, capture_intake_summary: JSONDict) -> JSONDict:
    runtime_manifest = runtime_operator_queue_manifest(runtime_request_summary)
    intake_manifest = intake_operator_queue_manifest(capture_intake_summary)
    runtime_by_name = operator_queue_by_request_name(runtime_manifest)
    intake_by_name = operator_queue_by_request_name(intake_manifest)
    runtime_names = set(runtime_by_name)
    intake_names = set(intake_by_name)
    shared_names = sorted(runtime_names & intake_names)
    metadata_mismatches = []
    for name in shared_names:
        mismatches = operator_queue_mismatches(runtime_by_name[name], intake_by_name[name])
        if mismatches:
            metadata_mismatches.append(
                {
                    "request_name": name,
                    "mismatch_fields": mismatches,
                }
            )
    ready = (
        bool(runtime_manifest)
        and bool(intake_manifest)
        and len(runtime_manifest) == len(intake_manifest)
        and runtime_names == intake_names
        and not metadata_mismatches
    )
    return {
        "ready": ready,
        "comparison_basis": "request_name_next_step_id_status_path",
        "runtime_queue_count": len(runtime_manifest),
        "capture_result_queue_count": len(intake_manifest),
        "matched_request_count": len(shared_names),
        "missing_from_capture_result_request_names": sorted(runtime_names - intake_names),
        "missing_from_runtime_request_names": sorted(intake_names - runtime_names),
        "metadata_mismatch_requests": metadata_mismatches,
    }

def runtime_request_for_recommendation(runtime_request_summary: JSONDict) -> JSONDict:
    recommendation = runtime_request_summary.get("recommended_runtime_capture_request")
    request_path = None
    request_name = None
    if isinstance(recommendation, dict):
        request_path = normalize_manifest_path(recommendation.get("path") or recommendation.get("request_path"))
        request_name = recommendation.get("request_name")
    requests = runtime_request_summary.get("requests")
    if not isinstance(requests, list):
        return {}
    for request in requests:
        if not isinstance(request, dict):
            continue
        if request_path and normalize_manifest_path(request.get("path")) == request_path:
            return request
        if isinstance(request_name, str) and request_name.strip() and request.get("name") == request_name:
            return request
    return {}


def runtime_post_approval_pending_manifest(runtime_request_summary: JSONDict) -> list[JSONDict]:
    preview = runtime_request_summary.get("recommended_post_approval_preview")
    if not isinstance(preview, dict):
        return []
    pending_ids = list_of_strings(preview.get("pending_artifact_ids"))
    if not pending_ids:
        return []
    recommendation = runtime_request_summary.get("recommended_runtime_capture_request")
    recommendation = recommendation if isinstance(recommendation, dict) else {}
    request = runtime_request_for_recommendation(runtime_request_summary)
    request_path = normalize_manifest_path(preview.get("request_path") or recommendation.get("path") or recommendation.get("request_path") or request.get("path"))
    request_name = preview.get("request_name") or recommendation.get("request_name") or request.get("name") or request_path
    handoff = request.get("operator_handoff") if isinstance(request.get("operator_handoff"), dict) else {}
    requested = handoff.get("requested_artifacts") if isinstance(handoff.get("requested_artifacts"), list) else []
    requested_by_id = {artifact.get("id"): artifact for artifact in requested if isinstance(artifact, dict)}
    preview_statuses = preview.get("artifact_statuses") if isinstance(preview.get("artifact_statuses"), list) else []
    status_by_id = {item.get("id"): item for item in preview_statuses if isinstance(item, dict)}
    manifest: list[JSONDict] = []
    for artifact_id in pending_ids:
        artifact = requested_by_id.get(artifact_id, {})
        preview_status = status_by_id.get(artifact_id, {})
        validators = artifact.get("validator_commands") if isinstance(artifact.get("validator_commands"), list) else []
        manifest.append(
            {
                "request_name": request_name,
                "request_path": request_path,
                "artifact_id": artifact_id,
                "artifact_path": normalize_manifest_path(preview_status.get("path") or artifact.get("path")),
                "status_after_approval": preview_status.get("status") or "ready_for_operator_capture",
                "validator_command_count": len(validators),
            }
        )
    return manifest


def intake_post_approval_capture_fill_manifest(capture_intake_summary: JSONDict) -> list[JSONDict]:
    plan = capture_intake_summary.get("recommended_post_approval_capture_fill_plan")
    if not isinstance(plan, dict):
        return []
    request_path = normalize_manifest_path(plan.get("request_path"))
    request_name = plan.get("request_name") or request_path
    steps = plan.get("capture_fill_steps") if isinstance(plan.get("capture_fill_steps"), list) else []
    manifest: list[JSONDict] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        artifact_id = step.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            continue
        after = step.get("step_after_approval") if isinstance(step.get("step_after_approval"), dict) else {}
        manifest.append(
            {
                "request_name": request_name,
                "request_path": request_path,
                "artifact_id": artifact_id.strip(),
                "artifact_path": normalize_manifest_path(step.get("artifact_path") or after.get("path")),
                "status_after_approval": after.get("status"),
                "fill_status_after_approval": step.get("fill_status_after_approval"),
                "validator_command_count": int_count(step.get("validator_command_count")),
            }
        )
    return manifest


def post_approval_fill_by_artifact(manifest: list[JSONDict]) -> dict[str, JSONDict]:
    result: dict[str, JSONDict] = {}
    for item in manifest:
        artifact_id = item.get("artifact_id")
        if isinstance(artifact_id, str) and artifact_id.strip():
            result[artifact_id.strip()] = item
    return result


def post_approval_capture_fill_mismatches(runtime_item: JSONDict, intake_item: JSONDict) -> list[str]:
    mismatches: list[str] = []
    for key in ("artifact_path", "status_after_approval", "validator_command_count"):
        if runtime_item.get(key) != intake_item.get(key):
            mismatches.append(key)
    return mismatches


def post_approval_capture_fill_parity_summary(runtime_request_summary: JSONDict, capture_intake_summary: JSONDict) -> JSONDict:
    runtime_manifest = runtime_post_approval_pending_manifest(runtime_request_summary)
    intake_manifest = intake_post_approval_capture_fill_manifest(capture_intake_summary)
    runtime_by_id = post_approval_fill_by_artifact(runtime_manifest)
    intake_by_id = post_approval_fill_by_artifact(intake_manifest)
    runtime_ids = set(runtime_by_id)
    intake_ids = set(intake_by_id)
    shared_ids = sorted(runtime_ids & intake_ids)
    metadata_mismatches = []
    for artifact_id in shared_ids:
        mismatches = post_approval_capture_fill_mismatches(runtime_by_id[artifact_id], intake_by_id[artifact_id])
        if mismatches:
            metadata_mismatches.append(
                {
                    "artifact_id": artifact_id,
                    "mismatch_fields": mismatches,
                }
            )
    request_path_match = True
    if runtime_manifest and intake_manifest:
        runtime_paths = {item.get("request_path") for item in runtime_manifest if item.get("request_path")}
        intake_paths = {item.get("request_path") for item in intake_manifest if item.get("request_path")}
        request_path_match = runtime_paths == intake_paths
    ready = (
        bool(runtime_manifest)
        and bool(intake_manifest)
        and len(runtime_manifest) == len(intake_manifest)
        and runtime_ids == intake_ids
        and request_path_match
        and not metadata_mismatches
    )
    return {
        "ready": ready,
        "comparison_basis": "recommended_request_pending_artifact_id_path_status_validator_count",
        "runtime_pending_artifact_count": len(runtime_manifest),
        "capture_result_fill_step_count": len(intake_manifest),
        "matched_artifact_count": len(shared_ids),
        "request_path_match": request_path_match,
        "missing_from_capture_result_artifact_ids": sorted(runtime_ids - intake_ids),
        "missing_from_runtime_preview_artifact_ids": sorted(intake_ids - runtime_ids),
        "metadata_mismatch_artifacts": metadata_mismatches,
    }

def request_lookup_by_name_and_path(summary: JSONDict) -> tuple[dict[str, JSONDict], dict[str, JSONDict]]:
    requests = summary.get("requests")
    by_name: dict[str, JSONDict] = {}
    by_path: dict[str, JSONDict] = {}
    if not isinstance(requests, list):
        return by_name, by_path
    for request in requests:
        if not isinstance(request, dict):
            continue
        request_name = request.get("name")
        if isinstance(request_name, str) and request_name.strip():
            by_name[request_name.strip()] = request
        request_path = normalize_manifest_path(request.get("path"))
        if request_path:
            by_path[request_path] = request
    return by_name, by_path


def artifact_status_after_approval(value: Any) -> str | None:
    status = value if isinstance(value, str) and value.strip() else None
    if status == "approval_required":
        return "ready_for_operator_capture"
    return status


def runtime_all_post_approval_pending_manifest(runtime_request_summary: JSONDict) -> list[JSONDict]:
    queue = runtime_request_summary.get("approval_queue")
    if not isinstance(queue, list):
        return []
    requests_by_name, requests_by_path = request_lookup_by_name_and_path(runtime_request_summary)
    manifest: list[JSONDict] = []
    for item in queue:
        if not isinstance(item, dict):
            continue
        request_path = normalize_manifest_path(item.get("path") or item.get("request_path"))
        request_name = item.get("request_name")
        request = {}
        if request_path and request_path in requests_by_path:
            request = requests_by_path[request_path]
        elif isinstance(request_name, str) and request_name.strip():
            request = requests_by_name.get(request_name.strip(), {})
        if not request_path:
            request_path = normalize_manifest_path(request.get("path"))
        if not isinstance(request_name, str) or not request_name.strip():
            request_name = request.get("name") or request_path
        pending_ids = list_of_strings(item.get("pending_artifact_ids"))
        if not pending_ids:
            continue
        handoff = request.get("operator_handoff") if isinstance(request.get("operator_handoff"), dict) else {}
        requested = handoff.get("requested_artifacts") if isinstance(handoff.get("requested_artifacts"), list) else []
        requested_by_id = {artifact.get("id"): artifact for artifact in requested if isinstance(artifact, dict)}
        for artifact_id in pending_ids:
            artifact = requested_by_id.get(artifact_id, {})
            validators = artifact.get("validator_commands") if isinstance(artifact.get("validator_commands"), list) else []
            manifest.append(
                {
                    "request_name": request_name,
                    "request_path": request_path,
                    "artifact_id": artifact_id,
                    "artifact_path": normalize_manifest_path(artifact.get("path")),
                    "status_after_approval": artifact_status_after_approval(artifact.get("status") or item.get("status")),
                    "validator_command_count": len(validators),
                }
            )
    return manifest


def intake_all_post_approval_capture_fill_manifest(capture_intake_summary: JSONDict) -> list[JSONDict]:
    plans = capture_intake_summary.get("post_approval_capture_fill_plan_manifest")
    if not isinstance(plans, list):
        plans = []
    manifest: list[JSONDict] = []
    for plan in plans:
        if not isinstance(plan, dict):
            continue
        request_path = normalize_manifest_path(plan.get("request_path"))
        request_name = plan.get("request_name") or request_path
        steps = plan.get("capture_fill_steps") if isinstance(plan.get("capture_fill_steps"), list) else []
        for step in steps:
            if not isinstance(step, dict):
                continue
            artifact_id = step.get("artifact_id")
            if not isinstance(artifact_id, str) or not artifact_id.strip():
                continue
            after = step.get("step_after_approval") if isinstance(step.get("step_after_approval"), dict) else {}
            manifest.append(
                {
                    "request_name": request_name,
                    "request_path": request_path,
                    "artifact_id": artifact_id.strip(),
                    "artifact_path": normalize_manifest_path(step.get("artifact_path") or after.get("path")),
                    "status_after_approval": after.get("status"),
                    "fill_status_after_approval": step.get("fill_status_after_approval"),
                    "validator_command_count": int_count(step.get("validator_command_count")),
                }
            )
    return manifest


def post_approval_fill_key(item: JSONDict) -> str | None:
    request_path = normalize_manifest_path(item.get("request_path"))
    artifact_id = item.get("artifact_id")
    if not request_path or not isinstance(artifact_id, str) or not artifact_id.strip():
        return None
    return f"{request_path}::{artifact_id.strip()}"


def post_approval_fill_by_key(manifest: list[JSONDict]) -> dict[str, JSONDict]:
    result: dict[str, JSONDict] = {}
    for item in manifest:
        key = post_approval_fill_key(item)
        if key is not None:
            result[key] = item
    return result


def all_post_approval_capture_fill_parity_summary(
    runtime_request_summary: JSONDict,
    capture_intake_summary: JSONDict,
) -> JSONDict:
    runtime_manifest = runtime_all_post_approval_pending_manifest(runtime_request_summary)
    intake_manifest = intake_all_post_approval_capture_fill_manifest(capture_intake_summary)
    runtime_by_key = post_approval_fill_by_key(runtime_manifest)
    intake_by_key = post_approval_fill_by_key(intake_manifest)
    runtime_keys = set(runtime_by_key)
    intake_keys = set(intake_by_key)
    shared_keys = sorted(runtime_keys & intake_keys)
    metadata_mismatches = []
    for key in shared_keys:
        mismatches = post_approval_capture_fill_mismatches(runtime_by_key[key], intake_by_key[key])
        if mismatches:
            metadata_mismatches.append(
                {
                    "artifact_key": key,
                    "request_path": runtime_by_key[key].get("request_path"),
                    "artifact_id": runtime_by_key[key].get("artifact_id"),
                    "mismatch_fields": mismatches,
                }
            )
    runtime_request_paths = {
        item.get("request_path")
        for item in runtime_manifest
        if isinstance(item.get("request_path"), str) and item.get("request_path")
    }
    intake_request_paths = {
        item.get("request_path")
        for item in intake_manifest
        if isinstance(item.get("request_path"), str) and item.get("request_path")
    }
    ready = (
        bool(runtime_manifest)
        and bool(intake_manifest)
        and len(runtime_manifest) == len(intake_manifest)
        and runtime_keys == intake_keys
        and runtime_request_paths == intake_request_paths
        and not metadata_mismatches
    )
    return {
        "ready": ready,
        "comparison_basis": "all_requests_pending_artifact_id_path_status_validator_count",
        "runtime_request_count": len(runtime_request_paths),
        "capture_result_plan_request_count": len(intake_request_paths),
        "matched_request_count": len(runtime_request_paths & intake_request_paths),
        "runtime_pending_artifact_count": len(runtime_manifest),
        "capture_result_fill_step_count": len(intake_manifest),
        "matched_artifact_count": len(shared_keys),
        "missing_from_capture_result_artifact_keys": sorted(runtime_keys - intake_keys),
        "missing_from_runtime_request_artifact_keys": sorted(intake_keys - runtime_keys),
        "missing_from_capture_result_request_paths": sorted(runtime_request_paths - intake_request_paths),
        "missing_from_runtime_request_paths": sorted(intake_request_paths - runtime_request_paths),
        "metadata_mismatch_artifacts": metadata_mismatches,
    }

def compact_receipt_gate_coverage(value: Any) -> JSONDict:
    if not isinstance(value, dict):
        return {}
    keys = (
        "request_count",
        "capture_receipt_required_count",
        "capture_receipt_ready_count",
        "approvals_ready_count",
        "candidate_trace_receipt_ready_count",
        "managed_output_receipt_ready_count",
        "dense_output_receipt_ready_count",
        "output_receipt_binding_ready_count",
        "live_capability_proof_ready_count",
        "missing_request_count",
        "all_capture_receipts_ready",
        "all_output_receipt_bindings_ready",
    )
    return {key: value.get(key) for key in keys if key in value}

def compact_receipt_fill_artifact_counts(value: Any, artifact_id: str) -> JSONDict:
    if not isinstance(value, dict):
        return {}
    by_artifact = value.get("by_artifact")
    if not isinstance(by_artifact, dict):
        by_artifact = value.get("artifact_class_counts")
    if not isinstance(by_artifact, dict):
        return {}
    artifact = by_artifact.get(artifact_id)
    if not isinstance(artifact, dict):
        return {}
    keys = ("entry_count", "ready_count", "missing_count")
    return {key: artifact.get(key) for key in keys if key in artifact}

def command_to_text(command: Any) -> str:
    if isinstance(command, list):
        return " ".join(str(part) for part in command)
    return str(command)


def compact_validator_commands(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [command_to_text(command) for command in value]


def artifact_command_manifest(items: Any, *, artifact_stage: str) -> list[JSONDict]:
    if not isinstance(items, list):
        return []
    manifest: list[JSONDict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        commands = compact_validator_commands(item.get("validator_commands"))
        if not commands:
            continue
        manifest.append(
            {
                "artifact_id": item.get("id"),
                "artifact_stage": artifact_stage,
                "status": item.get("status"),
                "path": item.get("path"),
                "validator_commands": commands,
            }
        )
    return manifest


def recommended_capture_command_manifest(
    runtime_request_summary: JSONDict,
    recommendation: JSONDict,
) -> list[JSONDict]:
    requests = runtime_request_summary.get("requests")
    if not isinstance(requests, list):
        return []
    request_path = recommendation.get("request_path")
    request_name = recommendation.get("request_name")
    bundle_path = recommendation.get("bundle_path")
    for request in requests:
        if not isinstance(request, dict):
            continue
        if not (
            (request_path and request.get("path") == request_path)
            or (request_name and request.get("name") == request_name)
            or (bundle_path and request.get("bundle_path") == bundle_path)
        ):
            continue
        handoff = request.get("operator_handoff") if isinstance(request.get("operator_handoff"), dict) else {}
        return [
            *artifact_command_manifest(handoff.get("requested_artifacts"), artifact_stage="runtime_capture"),
            *artifact_command_manifest(handoff.get("future_artifacts"), artifact_stage="future_adapter"),
        ]
    return []


def all_request_validator_command_manifest(runtime_request_summary: JSONDict) -> list[JSONDict]:
    queue = runtime_request_summary.get("approval_queue")
    if not isinstance(queue, list):
        return []
    requests_by_name, requests_by_path = request_lookup_by_name_and_path(runtime_request_summary)
    manifest: list[JSONDict] = []
    for item in queue:
        if not isinstance(item, dict):
            continue
        rank = int_count(item.get("queue_rank") or item.get("rank"))
        request_path = normalize_manifest_path(item.get("path") or item.get("request_path"))
        request_name = item.get("request_name")
        request = {}
        if request_path and request_path in requests_by_path:
            request = requests_by_path[request_path]
        elif isinstance(request_name, str) and request_name.strip():
            request = requests_by_name.get(request_name.strip(), {})
        if not request_path:
            request_path = normalize_manifest_path(request.get("path"))
        if not isinstance(request_name, str) or not request_name.strip():
            request_name = request.get("name") or request_path
        handoff = request.get("operator_handoff") if isinstance(request.get("operator_handoff"), dict) else {}
        for artifact in [
            *artifact_command_manifest(handoff.get("requested_artifacts"), artifact_stage="runtime_capture"),
            *artifact_command_manifest(handoff.get("future_artifacts"), artifact_stage="future_adapter"),
        ]:
            commands = artifact.get("validator_commands") if isinstance(artifact.get("validator_commands"), list) else []
            manifest.append(
                {
                    "rank": rank,
                    "request_name": request_name,
                    "request_path": request_path,
                    "artifact_id": artifact.get("artifact_id"),
                    "artifact_stage": artifact.get("artifact_stage"),
                    "status": artifact.get("status"),
                    "path": normalize_manifest_path(artifact.get("path")),
                    "validator_command_count": len(commands),
                    "validator_commands": commands,
                }
            )
    stage_order = {"runtime_capture": 0, "future_adapter": 1}
    return sorted(
        manifest,
        key=lambda item: (
            int_count(item.get("rank")),
            str(item.get("request_name") or ""),
            stage_order.get(str(item.get("artifact_stage")), 99),
            str(item.get("artifact_id") or ""),
        ),
    )


def all_request_validator_command_summary(
    manifest: list[JSONDict],
    *,
    expected_request_count: int,
) -> JSONDict:
    runtime_entries = [item for item in manifest if item.get("artifact_stage") == "runtime_capture"]
    future_entries = [item for item in manifest if item.get("artifact_stage") == "future_adapter"]
    request_paths = {
        item.get("request_path")
        for item in manifest
        if isinstance(item.get("request_path"), str) and item.get("request_path")
    }
    missing_command_entries = [
        item
        for item in manifest
        if int_count(item.get("validator_command_count")) == 0
    ]
    runtime_command_count = sum(int_count(item.get("validator_command_count")) for item in runtime_entries)
    future_command_count = sum(int_count(item.get("validator_command_count")) for item in future_entries)
    return {
        "ready": (
            expected_request_count > 0
            and len(request_paths) == expected_request_count
            and len(manifest) > 0
            and not missing_command_entries
        ),
        "request_count": len(request_paths),
        "expected_request_count": expected_request_count,
        "artifact_entry_count": len(manifest),
        "runtime_artifact_count": len(runtime_entries),
        "future_artifact_count": len(future_entries),
        "runtime_validator_command_count": runtime_command_count,
        "future_validator_command_count": future_command_count,
        "total_validator_command_count": runtime_command_count + future_command_count,
        "missing_command_entry_count": len(missing_command_entries),
    }

def artifact_by_id(items: Any) -> dict[str, JSONDict]:
    if not isinstance(items, list):
        return {}
    result: dict[str, JSONDict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        artifact_id = item.get("id")
        if isinstance(artifact_id, str) and artifact_id.strip():
            result[artifact_id.strip()] = item
    return result


def runtime_capture_queue_recommendation(
    item: JSONDict,
    *,
    request: JSONDict | None = None,
    approval_item: JSONDict | None = None,
) -> JSONDict:
    request = request if isinstance(request, dict) else {}
    approval_item = approval_item if isinstance(approval_item, dict) else {}
    request_path = normalize_manifest_path(item.get("path") or item.get("request_path") or request.get("path"))
    request_name = item.get("request_name") or request.get("name") or request_path
    handoff = request.get("operator_handoff") if isinstance(request.get("operator_handoff"), dict) else {}
    requested_artifacts = artifact_by_id(handoff.get("requested_artifacts"))
    future_artifacts = artifact_by_id(handoff.get("future_artifacts"))
    candidate_trace = requested_artifacts.get("candidate_router_trace", {})
    candidate_source = candidate_trace.get("source") if isinstance(candidate_trace.get("source"), dict) else {}
    managed_output = requested_artifacts.get("managed_output_summary_fill", {})
    dense_output = requested_artifacts.get("dense_output_summary_fill", {})
    live_proof = future_artifacts.get("live_capability_proof_fill", {})
    approval_rebuild = item.get("approval_rebuild_command") if isinstance(item.get("approval_rebuild_command"), dict) else approval_item
    return {
        "request_name": request_name,
        "request_path": request_path,
        "status": item.get("status"),
        "bundle_path": normalize_manifest_path(item.get("bundle_path") or request.get("bundle_path")),
        "model_id": item.get("model_id"),
        "prompt_set_path": normalize_manifest_path(item.get("prompt_set_path")),
        "prompt_count": int_count(item.get("prompt_count")),
        "missing_approval_keys": list_of_strings(item.get("missing_approval_keys")),
        "pending_artifact_ids": list_of_strings(item.get("pending_artifact_ids")),
        "next_artifact_id": item.get("next_artifact_id"),
        "next_artifact_path": normalize_manifest_path(item.get("next_artifact_path") or candidate_trace.get("path")),
        "candidate_trace_receipt_path": normalize_manifest_path(candidate_source.get("capture_receipt_path")),
        "managed_output_path": normalize_manifest_path(managed_output.get("path")),
        "dense_output_path": normalize_manifest_path(dense_output.get("path")),
        "live_proof_template_path": normalize_manifest_path(live_proof.get("path")),
        "queue_rank": int_count(item.get("queue_rank") or item.get("rank")),
        "selection_rationale": item.get("selection_rationale"),
        "approval_rebuild_command": compact_approval_rebuild_command(approval_rebuild),
    }


def repo_path_exists(value: Any) -> bool:
    path = repo_path(value)
    return path is not None and path.exists()


def sibling_manifest_path(value: str | None, filename: str) -> str | None:
    path = normalize_manifest_path(value)
    if not path or "/" not in path:
        return None
    return path.rsplit("/", 1)[0] + "/" + filename


def validator_command_count_for_artifact(value: JSONDict) -> int:
    commands = value.get("validator_commands") if isinstance(value.get("validator_commands"), list) else []
    return len(commands)


def all_request_downstream_handoff_manifest(runtime_request_summary: JSONDict) -> list[JSONDict]:
    queue = runtime_request_summary.get("approval_queue")
    if not isinstance(queue, list):
        return []
    requests_by_name, requests_by_path = request_lookup_by_name_and_path(runtime_request_summary)
    approval_by_path = approval_manifest_by_request_path(runtime_request_summary.get("approval_rebuild_command_manifest"))
    manifest: list[JSONDict] = []
    for item in queue:
        if not isinstance(item, dict):
            continue
        request_path = normalize_manifest_path(item.get("path") or item.get("request_path"))
        request_name = item.get("request_name")
        request = {}
        if request_path and request_path in requests_by_path:
            request = requests_by_path[request_path]
        elif isinstance(request_name, str) and request_name.strip():
            request = requests_by_name.get(request_name.strip(), {})
        if not request_path:
            request_path = normalize_manifest_path(request.get("path"))
        recommendation = runtime_capture_queue_recommendation(
            item,
            request=request,
            approval_item=approval_by_path.get(request_path or ""),
        )
        handoff = request.get("operator_handoff") if isinstance(request.get("operator_handoff"), dict) else {}
        requested_artifacts = artifact_by_id(handoff.get("requested_artifacts"))
        future_artifacts = artifact_by_id(handoff.get("future_artifacts"))
        candidate_trace = requested_artifacts.get("candidate_router_trace", {})
        managed_output = requested_artifacts.get("managed_output_summary_fill", {})
        dense_output = requested_artifacts.get("dense_output_summary_fill", {})
        live_proof = future_artifacts.get("live_capability_proof_fill", {})
        prompt_set_path = normalize_manifest_path(recommendation.get("prompt_set_path"))
        candidate_trace_path = normalize_manifest_path(recommendation.get("next_artifact_path"))
        candidate_trace_receipt_path = normalize_manifest_path(recommendation.get("candidate_trace_receipt_path"))
        managed_output_path = normalize_manifest_path(recommendation.get("managed_output_path"))
        dense_output_path = normalize_manifest_path(recommendation.get("dense_output_path"))
        live_proof_path = normalize_manifest_path(recommendation.get("live_proof_template_path"))
        fallback_artifact_path = sibling_manifest_path(dense_output_path, "dense-fallback-comparison.json")
        policy_validator_count = validator_command_count_for_artifact(candidate_trace)
        dense_validator_count = validator_command_count_for_artifact(managed_output) + validator_command_count_for_artifact(dense_output)
        live_validator_count = validator_command_count_for_artifact(live_proof)
        prompt_set_exists = repo_path_exists(prompt_set_path)
        policy_ready = (
            prompt_set_exists
            and bool(candidate_trace_path)
            and bool(candidate_trace_receipt_path)
            and repo_path_exists(candidate_trace_receipt_path)
            and policy_validator_count >= 3
        )
        dense_ready = (
            prompt_set_exists
            and bool(managed_output_path)
            and repo_path_exists(managed_output_path)
            and bool(dense_output_path)
            and repo_path_exists(dense_output_path)
            and bool(fallback_artifact_path)
            and dense_validator_count >= 2
        )
        live_ready = bool(live_proof_path) and repo_path_exists(live_proof_path) and live_validator_count >= 1
        manifest.append(
            {
                "rank": recommendation.get("queue_rank"),
                "request_name": recommendation.get("request_name"),
                "request_path": recommendation.get("request_path"),
                "policy_candidate_trace_handoff_ready": policy_ready,
                "dense_fallback_capture_handoff_ready": dense_ready,
                "live_capability_proof_handoff_ready": live_ready,
                "all_downstream_handoffs_ready": policy_ready and dense_ready and live_ready,
                "policy_candidate_trace": {
                    "prompt_set_path": prompt_set_path,
                    "prompt_set_exists": prompt_set_exists,
                    "candidate_trace_path": candidate_trace_path,
                    "candidate_trace_receipt_path": candidate_trace_receipt_path,
                    "candidate_trace_receipt_template_exists": repo_path_exists(candidate_trace_receipt_path),
                    "validator_command_count": policy_validator_count,
                    "runtime_requires_explicit_approval": bool(recommendation.get("missing_approval_keys")),
                    "handoff_ready": policy_ready,
                },
                "dense_fallback_capture": {
                    "prompt_set_path": prompt_set_path,
                    "prompt_set_exists": prompt_set_exists,
                    "managed_output_path": managed_output_path,
                    "managed_output_template_exists": repo_path_exists(managed_output_path),
                    "dense_output_path": dense_output_path,
                    "dense_output_template_exists": repo_path_exists(dense_output_path),
                    "fallback_artifact_path": fallback_artifact_path,
                    "validator_command_count": dense_validator_count,
                    "runtime_requires_explicit_approval": bool(recommendation.get("missing_approval_keys")),
                    "handoff_ready": dense_ready,
                },
                "live_capability_proof": {
                    "proof_artifact_path": live_proof_path,
                    "proof_template_exists": repo_path_exists(live_proof_path),
                    "validator_command_count": live_validator_count,
                    "future_adapter_required": True,
                    "handoff_ready": live_ready,
                },
            }
        )
    return sorted(manifest, key=lambda value: (int_count(value.get("rank")), str(value.get("request_name") or "")))

def all_request_downstream_handoff_summary(
    manifest: list[JSONDict],
    *,
    expected_request_count: int,
) -> JSONDict:
    request_paths = {
        item.get("request_path")
        for item in manifest
        if isinstance(item.get("request_path"), str) and item.get("request_path")
    }
    policy_ready_count = sum(1 for item in manifest if item.get("policy_candidate_trace_handoff_ready") is True)
    dense_ready_count = sum(1 for item in manifest if item.get("dense_fallback_capture_handoff_ready") is True)
    live_ready_count = sum(1 for item in manifest if item.get("live_capability_proof_handoff_ready") is True)
    all_ready_count = sum(1 for item in manifest if item.get("all_downstream_handoffs_ready") is True)
    return {
        "ready": expected_request_count > 0 and len(request_paths) == expected_request_count and all_ready_count == expected_request_count,
        "request_count": len(request_paths),
        "expected_request_count": expected_request_count,
        "policy_candidate_trace_handoff_ready_count": policy_ready_count,
        "dense_fallback_capture_handoff_ready_count": dense_ready_count,
        "live_capability_proof_handoff_ready_count": live_ready_count,
        "all_downstream_handoff_ready_count": all_ready_count,
    }


def all_request_receipt_fill_manifest(capture_intake_summary: JSONDict, capture_queue_manifest: list[JSONDict] | None = None) -> list[JSONDict]:
    plans = capture_intake_summary.get("post_approval_capture_fill_plan_manifest")
    if not isinstance(plans, list):
        plans = []
    manifest: list[JSONDict] = []
    for plan in plans:
        if not isinstance(plan, dict):
            continue
        request_path = normalize_manifest_path(plan.get("request_path"))
        request_name = plan.get("request_name") or request_path
        rank = int_count(plan.get("rank"))
        records_approval_keys = list_of_strings(plan.get("records_approval_keys"))
        approval_records_prompt_traffic = "runtime_prompt_traffic_approved" in records_approval_keys
        plan_prompt_set_path = normalize_manifest_path(plan.get("prompt_set_path"))
        steps = plan.get("capture_fill_steps") if isinstance(plan.get("capture_fill_steps"), list) else []
        for raw_step in steps:
            if not isinstance(raw_step, dict):
                continue
            step = compact_post_approval_capture_fill_step(raw_step)
            artifact_id = step.get("artifact_id")
            if not isinstance(artifact_id, str) or not artifact_id.strip():
                continue
            current = step.get("current_step") if isinstance(step.get("current_step"), dict) else {}
            after = step.get("step_after_approval") if isinstance(step.get("step_after_approval"), dict) else {}
            manifest.append(
                {
                    "rank": rank,
                    "request_name": request_name,
                    "request_path": request_path,
                    "source_request_path": normalize_manifest_path(step.get("source_request_path") or step.get("request_path") or request_path),
                    "prompt_set_path": normalize_manifest_path(step.get("prompt_set_path") or plan_prompt_set_path),
                    "source_prompt_set_path": normalize_manifest_path(step.get("source_prompt_set_path") or step.get("prompt_set_path") or plan_prompt_set_path),
                    "records_approval_keys": records_approval_keys,
                    "requires_explicit_user_approval": plan.get("requires_explicit_user_approval") is True,
                    "approval_records_prompt_traffic": approval_records_prompt_traffic,
                    "may_send_prompt_traffic_after_approval": plan.get("requires_explicit_user_approval") is True and approval_records_prompt_traffic,
                    "artifact_id": artifact_id.strip(),
                    "queue_step_id": step.get("queue_step_id"),
                    "receipt_kind": step.get("receipt_kind"),
                    "artifact_path": normalize_manifest_path(step.get("artifact_path") or after.get("path")),
                    "receipt_path": normalize_manifest_path(step.get("receipt_path")),
                    "status_before_approval": current.get("status"),
                    "approval_state_before_approval": current.get("approval_state"),
                    "status_after_approval": after.get("status"),
                    "approval_state_after_approval": after.get("approval_state"),
                    "fill_status_after_approval": step.get("fill_status_after_approval"),
                    "ready_after_approval": step.get("ready_after_approval") is True,
                    "receipt_ready": step.get("receipt_ready") is True,
                    "validator_command_count": int_count(step.get("validator_command_count")),
                    "blockers_after_approval": list_of_strings(step.get("blockers_after_approval")),
                }
            )
    observed_keys = {post_approval_fill_key(item) for item in manifest}
    if isinstance(capture_queue_manifest, list):
        for queue_item in capture_queue_manifest:
            if not isinstance(queue_item, dict):
                continue
            request_path = normalize_manifest_path(queue_item.get("request_path"))
            request_name = queue_item.get("request_name") or request_path
            rank = int_count(queue_item.get("rank"))
            records_approval_keys = list_of_strings(queue_item.get("approval_rebuild_command", {}).get("records_approval_keys"))
            approval_records_prompt_traffic = "runtime_prompt_traffic_approved" in records_approval_keys
            requires_explicit_user_approval = queue_item.get("approval_rebuild_command", {}).get("requires_explicit_user_approval") is True
            plan_prompt_set_path = normalize_manifest_path(queue_item.get("prompt_set_path"))
            for raw_step in queue_item.get("capture_fill_steps") if isinstance(queue_item.get("capture_fill_steps"), list) else []:
                if not isinstance(raw_step, dict):
                    continue
                step = compact_post_approval_capture_fill_step(raw_step)
                artifact_id = step.get("artifact_id")
                if not isinstance(artifact_id, str) or not artifact_id.strip():
                    continue
                key = post_approval_fill_key({"request_path": request_path, "artifact_id": artifact_id})
                if key in observed_keys:
                    continue
                current = step.get("current_step") if isinstance(step.get("current_step"), dict) else {}
                after = step.get("step_after_approval") if isinstance(step.get("step_after_approval"), dict) else {}
                row = {
                    "rank": rank,
                    "request_name": request_name,
                    "request_path": request_path,
                    "source_request_path": normalize_manifest_path(step.get("source_request_path") or step.get("request_path") or request_path),
                    "prompt_set_path": normalize_manifest_path(step.get("prompt_set_path") or plan_prompt_set_path),
                    "source_prompt_set_path": normalize_manifest_path(step.get("source_prompt_set_path") or step.get("prompt_set_path") or plan_prompt_set_path),
                    "records_approval_keys": records_approval_keys,
                    "requires_explicit_user_approval": requires_explicit_user_approval,
                    "approval_records_prompt_traffic": approval_records_prompt_traffic,
                    "may_send_prompt_traffic_after_approval": requires_explicit_user_approval and approval_records_prompt_traffic,
                    "artifact_id": artifact_id.strip(),
                    "queue_step_id": step.get("queue_step_id"),
                    "receipt_kind": step.get("receipt_kind"),
                    "artifact_path": normalize_manifest_path(step.get("artifact_path") or after.get("path")),
                    "receipt_path": normalize_manifest_path(step.get("receipt_path")),
                    "status_before_approval": current.get("status"),
                    "approval_state_before_approval": current.get("approval_state"),
                    "status_after_approval": after.get("status"),
                    "approval_state_after_approval": after.get("approval_state"),
                    "fill_status_after_approval": step.get("fill_status_after_approval"),
                    "ready_after_approval": step.get("ready_after_approval") is True,
                    "receipt_ready": step.get("receipt_ready") is True,
                    "validator_command_count": int_count(step.get("validator_command_count")),
                    "blockers_after_approval": list_of_strings(step.get("blockers_after_approval")),
                }
                manifest.append(row)
                observed_keys.add(key)
    return sorted(
        manifest,
        key=lambda item: (
            int_count(item.get("rank")),
            str(item.get("request_name") or ""),
            str(item.get("artifact_id") or ""),
        ),
    )


def all_request_receipt_fill_summary(
    manifest: list[JSONDict],
    *,
    expected_request_count: int,
) -> JSONDict:
    request_paths = {
        item.get("request_path")
        for item in manifest
        if isinstance(item.get("request_path"), str) and item.get("request_path")
    }
    ready_count = sum(
        1
        for item in manifest
        if item.get("ready_after_approval") is True and item.get("receipt_ready") is True
    )
    missing_count = len(manifest) - ready_count
    approval_recorded_after_count = sum(
        1 for item in manifest if item.get("approval_state_after_approval") == "recorded"
    )
    approval_missing_after_count = sum(
        1 for item in manifest if item.get("approval_state_after_approval") == "missing"
    )
    blocked_after_approval_count = sum(
        1 for item in manifest if item.get("fill_status_after_approval") == "blocked_after_approval"
    )
    validator_command_count = sum(int_count(item.get("validator_command_count")) for item in manifest)
    artifact_class_counts: JSONDict = {
        "candidate_router_trace": {"entry_count": 0, "ready_count": 0, "missing_count": 0},
        "managed_output_summary_fill": {"entry_count": 0, "ready_count": 0, "missing_count": 0},
        "dense_output_summary_fill": {"entry_count": 0, "ready_count": 0, "missing_count": 0},
    }
    for item in manifest:
        artifact_id = item.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            continue
        counts = artifact_class_counts.setdefault(
            artifact_id.strip(),
            {"entry_count": 0, "ready_count": 0, "missing_count": 0},
        )
        counts["entry_count"] += 1
        if item.get("ready_after_approval") is True and item.get("receipt_ready") is True:
            counts["ready_count"] += 1
        else:
            counts["missing_count"] += 1
    return {
        "ready": (
            expected_request_count > 0
            and len(request_paths) == expected_request_count
            and len(manifest) > 0
            and missing_count == 0
        ),
        "request_count": len(request_paths),
        "expected_request_count": expected_request_count,
        "entry_count": len(manifest),
        "ready_count": ready_count,
        "missing_count": missing_count,
        "approval_recorded_after_count": approval_recorded_after_count,
        "approval_missing_after_count": approval_missing_after_count,
        "blocked_after_approval_count": blocked_after_approval_count,
        "validator_command_count": validator_command_count,
        "artifact_class_counts": artifact_class_counts,
    }


def all_request_receipt_validator_parity_summary(
    validator_manifest: list[JSONDict],
    receipt_fill_manifest: list[JSONDict],
) -> JSONDict:
    runtime_entries = [item for item in validator_manifest if item.get("artifact_stage") == "runtime_capture"]
    runtime_by_key = post_approval_fill_by_key(runtime_entries)
    receipt_by_key = post_approval_fill_by_key(receipt_fill_manifest)
    runtime_keys = set(runtime_by_key)
    receipt_keys = set(receipt_by_key)
    shared_keys = sorted(runtime_keys & receipt_keys)
    metadata_mismatches: list[JSONDict] = []
    for key in shared_keys:
        runtime_item = runtime_by_key[key]
        receipt_item = receipt_by_key[key]
        mismatches: list[str] = []
        if normalize_manifest_path(runtime_item.get("path")) != normalize_manifest_path(receipt_item.get("artifact_path")):
            mismatches.append("artifact_path")
        receipt_status_before = receipt_item.get("status_before_approval")
        if receipt_status_before and runtime_item.get("status") != receipt_status_before:
            mismatches.append("status_before_approval")
        if int_count(runtime_item.get("validator_command_count")) != int_count(receipt_item.get("validator_command_count")):
            mismatches.append("validator_command_count")
        if mismatches:
            metadata_mismatches.append(
                {
                    "artifact_key": key,
                    "request_path": runtime_item.get("request_path"),
                    "artifact_id": runtime_item.get("artifact_id"),
                    "mismatch_fields": mismatches,
                }
            )
    runtime_request_paths = {
        item.get("request_path")
        for item in runtime_entries
        if isinstance(item.get("request_path"), str) and item.get("request_path")
    }
    receipt_request_paths = {
        item.get("request_path")
        for item in receipt_fill_manifest
        if isinstance(item.get("request_path"), str) and item.get("request_path")
    }
    return {
        "ready": (
            bool(runtime_entries)
            and bool(receipt_fill_manifest)
            and runtime_keys == receipt_keys
            and runtime_request_paths == receipt_request_paths
            and not metadata_mismatches
        ),
        "comparison_basis": "runtime_capture_artifact_key_path_status_validator_count",
        "runtime_request_count": len(runtime_request_paths),
        "receipt_fill_request_count": len(receipt_request_paths),
        "matched_request_count": len(runtime_request_paths & receipt_request_paths),
        "runtime_artifact_count": len(runtime_entries),
        "receipt_fill_entry_count": len(receipt_fill_manifest),
        "matched_artifact_count": len(shared_keys),
        "missing_from_receipt_fill_artifact_keys": sorted(runtime_keys - receipt_keys),
        "missing_from_runtime_validator_artifact_keys": sorted(receipt_keys - runtime_keys),
        "missing_from_receipt_fill_request_paths": sorted(runtime_request_paths - receipt_request_paths),
        "missing_from_runtime_validator_request_paths": sorted(receipt_request_paths - runtime_request_paths),
        "metadata_mismatch_artifacts": metadata_mismatches,
    }



def all_request_receipt_fill_command_manifest(
    validator_manifest: list[JSONDict],
    receipt_fill_manifest: list[JSONDict],
) -> list[JSONDict]:
    runtime_validators = [item for item in validator_manifest if item.get("artifact_stage") == "runtime_capture"]
    validators_by_key = post_approval_fill_by_key(runtime_validators)
    manifest: list[JSONDict] = []
    for receipt in receipt_fill_manifest:
        key = post_approval_fill_key(receipt)
        validator = validators_by_key.get(key or "", {})
        commands = validator.get("validator_commands") if isinstance(validator.get("validator_commands"), list) else []
        manifest.append(
            {
                "rank": receipt.get("rank"),
                "request_name": receipt.get("request_name"),
                "request_path": receipt.get("request_path"),
                "source_request_path": receipt.get("source_request_path"),
                "prompt_set_path": receipt.get("prompt_set_path"),
                "source_prompt_set_path": receipt.get("source_prompt_set_path"),
                "records_approval_keys": list_of_strings(receipt.get("records_approval_keys")),
                "requires_explicit_user_approval": receipt.get("requires_explicit_user_approval") is True,
                "approval_records_prompt_traffic": receipt.get("approval_records_prompt_traffic") is True,
                "may_send_prompt_traffic_after_approval": receipt.get("may_send_prompt_traffic_after_approval") is True,
                "artifact_id": receipt.get("artifact_id"),
                "artifact_path": receipt.get("artifact_path"),
                "receipt_path": receipt.get("receipt_path"),
                "receipt_kind": receipt.get("receipt_kind"),
                "status_before_approval": receipt.get("status_before_approval") or validator.get("status"),
                "approval_state_after_approval": receipt.get("approval_state_after_approval"),
                "fill_status_after_approval": receipt.get("fill_status_after_approval"),
                "receipt_ready": receipt.get("receipt_ready") is True,
                "validator_command_count": len(commands),
                "validator_commands": commands,
                "blockers_after_approval": list_of_strings(receipt.get("blockers_after_approval")),
                "command_manifest_key": key,
            }
        )
    return sorted(
        manifest,
        key=lambda item: (
            int_count(item.get("rank")),
            str(item.get("request_name") or ""),
            str(item.get("artifact_id") or ""),
        ),
    )



def all_request_receipt_fill_command_summary(
    manifest: list[JSONDict],
    *,
    expected_request_count: int,
) -> JSONDict:
    request_paths = {
        item.get("request_path")
        for item in manifest
        if isinstance(item.get("request_path"), str) and item.get("request_path")
    }
    missing_command_entries = [item for item in manifest if int_count(item.get("validator_command_count")) == 0]
    missing_receipt_paths = [item for item in manifest if not item.get("receipt_path")]
    missing_source_request_paths = [
        item
        for item in manifest
        if not item.get("source_request_path") or normalize_manifest_path(item.get("source_request_path")) != normalize_manifest_path(item.get("request_path"))
    ]
    missing_prompt_set_paths = [
        item
        for item in manifest
        if not item.get("prompt_set_path") or normalize_manifest_path(item.get("source_prompt_set_path")) != normalize_manifest_path(item.get("prompt_set_path"))
    ]
    missing_explicit_approval = [item for item in manifest if item.get("requires_explicit_user_approval") is not True]
    missing_prompt_traffic_ack = [item for item in manifest if item.get("approval_records_prompt_traffic") is not True or item.get("may_send_prompt_traffic_after_approval") is not True]
    ready_receipt_count = sum(1 for item in manifest if item.get("receipt_ready") is True)
    blocked_after_approval_count = sum(
        1 for item in manifest if item.get("fill_status_after_approval") == "blocked_after_approval"
    )
    validator_command_count = sum(int_count(item.get("validator_command_count")) for item in manifest)
    return {
        "ready": (
            expected_request_count > 0
            and len(request_paths) == expected_request_count
            and len(manifest) > 0
            and not missing_command_entries
            and not missing_receipt_paths
            and not missing_source_request_paths
            and not missing_prompt_set_paths
            and not missing_explicit_approval
            and not missing_prompt_traffic_ack
        ),
        "request_count": len(request_paths),
        "expected_request_count": expected_request_count,
        "entry_count": len(manifest),
        "ready_receipt_count": ready_receipt_count,
        "blocked_after_approval_count": blocked_after_approval_count,
        "validator_command_count": validator_command_count,
        "missing_command_entry_count": len(missing_command_entries),
        "missing_receipt_path_count": len(missing_receipt_paths),
        "missing_source_request_path_count": len(missing_source_request_paths),
        "missing_prompt_set_path_count": len(missing_prompt_set_paths),
        "missing_explicit_approval_count": len(missing_explicit_approval),
        "missing_prompt_traffic_ack_count": len(missing_prompt_traffic_ack),
    }


def checklist_items_by_id(checklist: list[JSONDict]) -> dict[str, JSONDict]:
    return {
        str(item.get("id")): item
        for item in checklist
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }


def closure_state_from_status(status: Any) -> str:
    value = str(status or "unknown")
    if value in {"satisfied", "proven"}:
        return "closed"
    if value in {"approval_required", "ready_for_operator_capture", "pending_capture"}:
        return "runtime_capture_required"
    if value in {"ready_to_build_comparison"}:
        return "metadata_build_required"
    if value in {"future_phase_4", "future_adapter_required"}:
        return "future_adapter_required"
    if value == "unmapped":
        return "unmapped"
    return "blocked"


def checklist_status(checklist_by_id: dict[str, JSONDict], gate_id: str, *, default: str = "blocked") -> str:
    item = checklist_by_id.get(gate_id)
    if not isinstance(item, dict):
        return default
    return str(item.get("status") or default)


def checklist_next_action(checklist_by_id: dict[str, JSONDict], gate_id: str, *, default: str = "Rerun the Phase 3 evidence packet after repairing this gate.") -> str:
    item = checklist_by_id.get(gate_id)
    if not isinstance(item, dict):
        return default
    value = item.get("next_action")
    return str(value) if value else default


def phase3_blocker_closure_manifest(
    *,
    decision_summary: JSONDict,
    promotion_checklist: list[JSONDict],
    recommended_runtime_capture: JSONDict | None,
    policy_candidate_trace_plan: JSONDict | None,
    dense_fallback_capture_plan: JSONDict | None,
    live_capability_proof_handoff: JSONDict | None,
    runtime_actuator_design: JSONDict | None,
    runtime_actuator_spike: JSONDict | None,
    all_downstream_handoff_manifest: list[JSONDict],
    all_capture_queue_summary: JSONDict,
    all_receipt_fill_summary: JSONDict,
    receipt_command_summary: JSONDict,
) -> list[JSONDict]:
    reasons = decision_summary.get("no_go_reasons")
    if not isinstance(reasons, list):
        return []
    checklist_by_id = checklist_items_by_id(promotion_checklist)
    runtime_capture = recommended_runtime_capture if isinstance(recommended_runtime_capture, dict) else {}
    policy_plan = policy_candidate_trace_plan if isinstance(policy_candidate_trace_plan, dict) else {}
    dense_plan = dense_fallback_capture_plan if isinstance(dense_fallback_capture_plan, dict) else {}
    live_handoff = live_capability_proof_handoff if isinstance(live_capability_proof_handoff, dict) else {}
    runtime_actuator = runtime_actuator_design if isinstance(runtime_actuator_design, dict) else {}
    runtime_spike = runtime_actuator_spike if isinstance(runtime_actuator_spike, dict) else {}
    downstream_handoff = next((item for item in all_downstream_handoff_manifest if isinstance(item, dict)), {})
    downstream_policy = downstream_handoff.get("policy_candidate_trace") if isinstance(downstream_handoff.get("policy_candidate_trace"), dict) else {}
    downstream_dense = downstream_handoff.get("dense_fallback_capture") if isinstance(downstream_handoff.get("dense_fallback_capture"), dict) else {}
    downstream_live = downstream_handoff.get("live_capability_proof") if isinstance(downstream_handoff.get("live_capability_proof"), dict) else {}

    manifest: list[JSONDict] = []
    for rank, raw_reason in enumerate(reasons, start=1):
        if not isinstance(raw_reason, dict) or not raw_reason.get("id"):
            continue
        reason_id = str(raw_reason["id"])
        reason_summary = str(raw_reason.get("summary") or raw_reason.get("reason") or "")
        row: JSONDict = {
            "rank": rank,
            "reason_id": reason_id,
            "reason_summary": reason_summary,
            "mapped": True,
        }

        if reason_id in {"no_replay_policy_candidate", "reuse_evidence_capture_not_ready"}:
            gate = "policy_candidate_trace_capture"
            status = checklist_status(checklist_by_id, gate)
            row.update(
                {
                    "closure_gate": gate,
                    "supporting_gate": "approved_runtime_capture",
                    "closure_class": "policy_candidate_replay",
                    "closure_status": status,
                    "closure_state": closure_state_from_status(status),
                    "approval_stage": "runtime_capture_required"
                    if policy_plan.get("runtime_requires_explicit_approval") is True or downstream_policy.get("runtime_requires_explicit_approval") is True
                    else "capture_receipt_required",
                    "handoff_ready": policy_candidate_trace_handoff_ready(policy_plan) or downstream_policy.get("handoff_ready") is True,
                    "primary_path": policy_plan.get("candidate_trace_path") or downstream_policy.get("candidate_trace_path"),
                    "secondary_path": policy_plan.get("candidate_prompt_set_path") or downstream_policy.get("prompt_set_path"),
                    "receipt_path": policy_plan.get("candidate_trace_receipt_path") or downstream_policy.get("candidate_trace_receipt_path"),
                    "pending_artifact_count": 0 if policy_plan.get("candidate_trace_exists") is True else 1,
                    "ready_evidence_count": 1 if policy_plan.get("policy_candidate_ready_after_plan") is True else 0,
                    "missing_evidence_count": 0 if policy_plan.get("policy_candidate_ready_after_plan") is True else 1,
                    "validator_command_count": int_count(downstream_policy.get("validator_command_count") or policy_plan.get("command_count")),
                    "next_action": checklist_next_action(checklist_by_id, gate),
                }
            )
        elif reason_id == "capture_result_intake_not_ready":
            gate = "capture_result_intake"
            status = checklist_status(checklist_by_id, gate)
            row.update(
                {
                    "closure_gate": gate,
                    "supporting_gate": "approved_runtime_capture",
                    "closure_class": "receipt_bound_runtime_capture",
                    "closure_status": status,
                    "closure_state": closure_state_from_status(status),
                    "approval_stage": "runtime_capture_required",
                    "handoff_ready": receipt_command_summary.get("ready") is True,
                    "primary_path": runtime_capture.get("request_path"),
                    "secondary_path": runtime_capture.get("next_artifact_path"),
                    "receipt_path": None,
                    "pending_artifact_count": int_count(all_capture_queue_summary.get("pending_artifact_count")),
                    "ready_evidence_count": int_count(all_receipt_fill_summary.get("ready_count")),
                    "missing_evidence_count": int_count(all_receipt_fill_summary.get("missing_count")),
                    "validator_command_count": int_count(receipt_command_summary.get("validator_command_count")),
                    "next_action": checklist_next_action(checklist_by_id, gate),
                }
            )
        elif reason_id == "dense_fallback_comparison_not_ready":
            gate = "dense_fallback_capture"
            status = checklist_status(checklist_by_id, gate)
            managed_missing = int_count(dense_plan.get("managed_output_missing_count"))
            dense_missing = int_count(dense_plan.get("dense_output_missing_count"))
            if managed_missing == 0 and dense_missing == 0 and dense_plan.get("comparison_ready") is not True:
                managed_missing = 1 if downstream_dense.get("managed_output_path") else 0
                dense_missing = 1 if downstream_dense.get("dense_output_path") else 0
            row.update(
                {
                    "closure_gate": gate,
                    "supporting_gate": "dense_fallback_quality_bounds",
                    "closure_class": "dense_fallback_quality_bounds",
                    "closure_status": status,
                    "closure_state": closure_state_from_status(status),
                    "approval_stage": "runtime_capture_required"
                    if dense_plan.get("runtime_requires_explicit_approval") is True or downstream_dense.get("runtime_requires_explicit_approval") is True
                    else "saved_output_required",
                    "handoff_ready": dense_fallback_capture_handoff_ready(dense_plan) or downstream_dense.get("handoff_ready") is True,
                    "primary_path": dense_plan.get("fallback_artifact_path") or downstream_dense.get("fallback_artifact_path"),
                    "secondary_path": dense_plan.get("managed_output_path") or downstream_dense.get("managed_output_path"),
                    "receipt_path": dense_plan.get("dense_output_path") or downstream_dense.get("dense_output_path"),
                    "pending_artifact_count": managed_missing + dense_missing,
                    "ready_evidence_count": 1 if dense_plan.get("comparison_ready") is True else 0,
                    "missing_evidence_count": 0 if dense_plan.get("comparison_ready") is True else max(1, managed_missing + dense_missing),
                    "validator_command_count": int_count(downstream_dense.get("validator_command_count") or dense_plan.get("command_count")),
                    "next_action": checklist_next_action(checklist_by_id, gate),
                }
            )
        elif reason_id == "live_capability_proof_not_ready":
            gate = "live_capability_proof_handoff"
            status = checklist_status(checklist_by_id, gate)
            row.update(
                {
                    "closure_gate": gate,
                    "supporting_gate": "live_capability_proof",
                    "closure_class": "live_capability_proof",
                    "closure_status": status,
                    "closure_state": closure_state_from_status(status),
                    "approval_stage": "future_adapter_required",
                    "handoff_ready": live_capability_proof_handoff_ready(live_handoff) or downstream_live.get("handoff_ready") is True,
                    "primary_path": live_handoff.get("proof_artifact_path") or downstream_live.get("proof_artifact_path"),
                    "secondary_path": live_handoff.get("source_bundle_path") or downstream_handoff.get("bundle_path"),
                    "receipt_path": None,
                    "pending_artifact_count": int_count(live_handoff.get("blocker_count")) or 1,
                    "ready_evidence_count": 1 if live_handoff.get("proof_ready") is True else 0,
                    "missing_evidence_count": int_count(live_handoff.get("blocker_count")) or 1,
                    "validator_command_count": int_count(downstream_live.get("validator_command_count")),
                    "next_action": checklist_next_action(checklist_by_id, gate),
                }
            )
        elif reason_id in {"live_actuator_missing", "live_actuator_capabilities_unavailable"}:
            gate = "runtime_actuator_spike"
            status = checklist_status(checklist_by_id, "live_capability_proof", default="future_phase_4")
            spike_ready = runtime_spike.get("spike_handoff_ready") is True
            blocker_count = int_count(runtime_spike.get("blocking_capability_count")) or int_count(runtime_actuator.get("live_actuator_blocker_count")) or 1
            row.update(
                {
                    "closure_gate": gate,
                    "supporting_gate": "phase4_promotion_decision",
                    "closure_class": "runtime_actuator",
                    "closure_status": status,
                    "closure_state": "future_adapter_required",
                    "approval_stage": "future_adapter_required",
                    "handoff_ready": spike_ready,
                    "primary_path": runtime_spike.get("managed_plan_path") or runtime_actuator.get("managed_plan_path") or "memory-moe-mvp/data/managed_expert_loading_plan.json",
                    "secondary_path": live_handoff.get("proof_artifact_path") or downstream_live.get("proof_artifact_path") or downstream_handoff.get("bundle_path"),
                    "receipt_path": None,
                    "pending_artifact_count": blocker_count,
                    "ready_evidence_count": 1 if spike_ready else 0,
                    "missing_evidence_count": 1,
                    "validator_command_count": 1 if runtime_spike.get("valid") is True else 0,
                    "next_action": "Use the runtime actuator spike planner to prove inventory, routing visibility, residency observation, fallback, artifact export, residency control, and cleanup gates before any live mutation claim.",
                }
            )
        else:
            row.update(
                {
                    "mapped": False,
                    "closure_gate": "unmapped_no_go_reason",
                    "supporting_gate": None,
                    "closure_class": "unknown",
                    "closure_status": "unmapped",
                    "closure_state": "unmapped",
                    "approval_stage": "unknown",
                    "handoff_ready": False,
                    "primary_path": None,
                    "secondary_path": None,
                    "receipt_path": None,
                    "pending_artifact_count": 0,
                    "ready_evidence_count": 0,
                    "missing_evidence_count": 1,
                    "validator_command_count": 0,
                    "next_action": "Add an explicit blocker-closure mapping for this no-go reason before treating the packet as operator-ready.",
                }
            )
        manifest.append(row)
    return manifest


def phase3_blocker_closure_summary(manifest: list[JSONDict]) -> JSONDict:
    reason_count = len(manifest)
    mapped_reason_count = sum(1 for item in manifest if item.get("mapped") is True)
    handoff_ready_count = sum(1 for item in manifest if item.get("handoff_ready") is True)
    runtime_capture_required_count = sum(
        1 for item in manifest if item.get("approval_stage") == "runtime_capture_required"
    )
    future_adapter_required_count = sum(
        1 for item in manifest if item.get("approval_stage") == "future_adapter_required"
    )
    unresolved_reason_count = sum(1 for item in manifest if item.get("closure_state") != "closed")
    missing_next_action_count = sum(1 for item in manifest if not item.get("next_action"))
    missing_evidence_count = sum(int_count(item.get("missing_evidence_count")) for item in manifest)
    pending_artifact_count = sum(int_count(item.get("pending_artifact_count")) for item in manifest)
    mapping_ready = reason_count > 0 and mapped_reason_count == reason_count and missing_next_action_count == 0
    evidence_complete = reason_count > 0 and unresolved_reason_count == 0 and missing_evidence_count == 0
    if evidence_complete:
        ready_scope = "closure_complete"
    elif mapping_ready:
        ready_scope = "mapping_ready_only"
    else:
        ready_scope = "mapping_incomplete"
    return {
        "ready": mapping_ready,
        "mapping_ready": mapping_ready,
        "evidence_complete": evidence_complete,
        "closure_complete": evidence_complete,
        "ready_scope": ready_scope,
        "reason_count": reason_count,
        "mapped_reason_count": mapped_reason_count,
        "unmapped_reason_count": reason_count - mapped_reason_count,
        "unresolved_reason_count": unresolved_reason_count,
        "handoff_ready_count": handoff_ready_count,
        "runtime_capture_required_count": runtime_capture_required_count,
        "future_adapter_required_count": future_adapter_required_count,
        "pending_artifact_count": pending_artifact_count,
        "missing_evidence_count": missing_evidence_count,
        "validator_command_count": sum(int_count(item.get("validator_command_count")) for item in manifest),
        "missing_next_action_count": missing_next_action_count,
    }


def _closure_rows_by_reason(manifest: Any) -> dict[str, JSONDict]:
    if not isinstance(manifest, list):
        return {}
    rows: dict[str, JSONDict] = {}
    for item in manifest:
        if isinstance(item, dict) and isinstance(item.get("reason_id"), str) and item.get("reason_id"):
            rows[str(item["reason_id"])] = item
    return rows


def _add_blocker_ledger_row(
    rows: list[JSONDict],
    *,
    reason: JSONDict,
    evidence_kind: str,
    evidence_id: str,
    artifact_path: Any = None,
    receipt_path: Any = None,
    prompt_id: Any = None,
    section: Any = None,
    blocker_id: Any = None,
    validator_command_count: int = 0,
    next_action: str | None = None,
) -> None:
    rows.append(
        {
            "rank": len(rows) + 1,
            "reason_id": reason.get("reason_id"),
            "closure_gate": reason.get("closure_gate"),
            "closure_class": reason.get("closure_class"),
            "approval_stage": reason.get("approval_stage"),
            "evidence_kind": evidence_kind,
            "evidence_id": evidence_id,
            "status": "missing",
            "artifact_path": normalize_manifest_path(artifact_path),
            "receipt_path": normalize_manifest_path(receipt_path),
            "prompt_id": prompt_id,
            "section": section,
            "blocker_id": blocker_id,
            "validator_command_count": validator_command_count,
            "closure_validator_command_count": int_count(reason.get("validator_command_count")),
            "next_action": next_action or str(reason.get("next_action") or ""),
        }
    )


def _dense_missing_prompt_ids(plan: JSONDict, label: str) -> list[str]:
    missing_count = int_count(plan.get(f"{label}_output_missing_count"))
    prompt_ids = list_of_strings(plan.get(f"{label}_output_prompt_ids"))
    if missing_count <= 0:
        return []
    if len(prompt_ids) == missing_count:
        return prompt_ids
    return [f"{label}_missing_output_{index}" for index in range(1, missing_count + 1)]


def phase3_blocker_evidence_ledger_manifest(
    blocker_closure_manifest: list[JSONDict],
    post_capture_intake_runbook_manifest: JSONDict,
    dense_fallback_capture_plan: JSONDict | None,
    live_capability_proof_handoff: JSONDict | None,
) -> JSONDict:
    reasons = _closure_rows_by_reason(blocker_closure_manifest)
    dense_plan = dense_fallback_capture_plan if isinstance(dense_fallback_capture_plan, dict) else {}
    live_handoff = live_capability_proof_handoff if isinstance(live_capability_proof_handoff, dict) else {}
    rows: list[JSONDict] = []
    missing_items: list[str] = []

    reason = reasons.get("no_replay_policy_candidate", {})
    if reason and int_count(reason.get("missing_evidence_count")) > 0:
        _add_blocker_ledger_row(
            rows,
            reason=reason,
            evidence_kind="policy_candidate_trace_replay",
            evidence_id="candidate_router_trace_policy_replay",
            artifact_path=reason.get("primary_path"),
            receipt_path=reason.get("receipt_path"),
            validator_command_count=int_count(reason.get("validator_command_count")),
        )

    reason = reasons.get("capture_result_intake_not_ready", {})
    if reason and isinstance(post_capture_intake_runbook_manifest, dict):
        for request in post_capture_intake_runbook_manifest.get("requests", []):
            if not isinstance(request, dict):
                continue
            request_path = request.get("request_path")
            for gate in request.get("artifact_gates", []):
                if not isinstance(gate, dict) or gate.get("ready_after_current_intake") is True:
                    continue
                artifact_id = str(gate.get("artifact_id") or "unknown_artifact")
                _add_blocker_ledger_row(
                    rows,
                    reason=reason,
                    evidence_kind="runtime_capture_receipt_gate",
                    evidence_id=f"{request_path or 'unknown_request'}::{artifact_id}",
                    artifact_path=gate.get("artifact_path"),
                    receipt_path=gate.get("receipt_path"),
                    validator_command_count=int_count(gate.get("validator_command_count")),
                )

    reason = reasons.get("dense_fallback_comparison_not_ready", {})
    if reason:
        for label, artifact_path in (
            ("managed", dense_plan.get("managed_output_path") or reason.get("secondary_path")),
            ("dense", dense_plan.get("dense_output_path") or reason.get("receipt_path")),
        ):
            for prompt_id in _dense_missing_prompt_ids(dense_plan, label):
                _add_blocker_ledger_row(
                    rows,
                    reason=reason,
                    evidence_kind=f"{label}_fallback_output_row",
                    evidence_id=f"{label}::{prompt_id}",
                    artifact_path=artifact_path,
                    prompt_id=prompt_id,
                    validator_command_count=0,
                )

    reason = reasons.get("live_capability_proof_not_ready", {})
    if reason:
        blockers = live_handoff.get("blockers") if isinstance(live_handoff.get("blockers"), list) else []
        for blocker in blockers:
            if not isinstance(blocker, dict):
                continue
            blocker_id = str(blocker.get("id") or "unknown_live_blocker")
            _add_blocker_ledger_row(
                rows,
                reason=reason,
                evidence_kind="live_capability_proof_blocker",
                evidence_id=blocker_id,
                artifact_path=reason.get("primary_path"),
                section=blocker.get("section"),
                blocker_id=blocker_id,
                validator_command_count=0,
            )

    for reason_id in ("live_actuator_missing", "live_actuator_capabilities_unavailable"):
        reason = reasons.get(reason_id, {})
        if not reason:
            continue
        _add_blocker_ledger_row(
            rows,
            reason=reason,
            evidence_kind="runtime_actuator_capability",
            evidence_id=reason_id,
            artifact_path=reason.get("primary_path"),
            validator_command_count=int_count(reason.get("validator_command_count")),
        )

    provisional_rows_by_reason: dict[str, int] = {}
    for row in rows:
        reason_id = str(row.get("reason_id") or "unknown")
        provisional_rows_by_reason[reason_id] = provisional_rows_by_reason.get(reason_id, 0) + 1
    for reason in blocker_closure_manifest:
        if not isinstance(reason, dict):
            continue
        reason_id = str(reason.get("reason_id") or "unknown")
        expected = int_count(reason.get("missing_evidence_count"))
        observed = provisional_rows_by_reason.get(reason_id, 0)
        for index in range(observed + 1, expected + 1):
            _add_blocker_ledger_row(
                rows,
                reason=reason,
                evidence_kind=f"{reason.get('closure_gate') or reason_id}_missing_evidence",
                evidence_id=f"{reason_id}::missing_evidence_{index}",
                artifact_path=reason.get("primary_path"),
                receipt_path=reason.get("receipt_path"),
                validator_command_count=0,
            )
            provisional_rows_by_reason[reason_id] = provisional_rows_by_reason.get(reason_id, 0) + 1
    rows_by_reason: dict[str, int] = {}
    for row in rows:
        reason_id = str(row.get("reason_id") or "unknown")
        rows_by_reason[reason_id] = rows_by_reason.get(reason_id, 0) + 1
    for reason in blocker_closure_manifest:
        if not isinstance(reason, dict):
            continue
        reason_id = str(reason.get("reason_id") or "unknown")
        expected = int_count(reason.get("missing_evidence_count"))
        observed = rows_by_reason.get(reason_id, 0)
        if expected != observed:
            missing_items.append(f"blocker_evidence_count_mismatch:{reason_id}:{observed}/{expected}")
    expected_missing = sum(int_count(item.get("missing_evidence_count")) for item in blocker_closure_manifest if isinstance(item, dict))
    ready = bool(blocker_closure_manifest) and len(rows) == expected_missing and not missing_items
    return {
        "ready": ready,
        "metadata_only": True,
        "reason_count": len([item for item in blocker_closure_manifest if isinstance(item, dict)]),
        "evidence_row_count": len(rows),
        "expected_missing_evidence_count": expected_missing,
        "runtime_capture_required_row_count": sum(1 for item in rows if item.get("approval_stage") == "runtime_capture_required"),
        "future_adapter_required_row_count": sum(1 for item in rows if item.get("approval_stage") == "future_adapter_required"),
        "receipt_bound_row_count": sum(1 for item in rows if item.get("evidence_kind") == "runtime_capture_receipt_gate"),
        "dense_output_row_count": sum(1 for item in rows if str(item.get("evidence_kind") or "").endswith("_fallback_output_row")),
        "live_proof_row_count": sum(1 for item in rows if item.get("evidence_kind") == "live_capability_proof_blocker"),
        "runtime_actuator_row_count": sum(1 for item in rows if item.get("evidence_kind") == "runtime_actuator_capability"),
        "validator_command_count": sum(int_count(item.get("validator_command_count")) for item in blocker_closure_manifest if isinstance(item, dict)),
        "missing_item_count": len(missing_items),
        "missing_items": sorted(set(missing_items)),
        "rows_by_reason": dict(sorted(rows_by_reason.items())),
        "evidence_rows": rows,
        "safety_contract": [
            "blocker evidence ledger is metadata only",
            "blocker evidence ledger generation does not launch runtimes",
            "blocker evidence ledger generation does not run Docker",
            "blocker evidence ledger generation does not call endpoints",
            "blocker evidence ledger generation does not inspect private tokens",
            "blocker evidence ledger generation does not send prompt traffic",
            "blocker evidence ledger generation does not mutate runtime residency",
        ],
    }


def phase3_blocker_evidence_ledger_summary(manifest: JSONDict) -> JSONDict:
    if not isinstance(manifest, dict) or not manifest:
        return {
            "ready": False,
            "reason_count": 0,
            "evidence_row_count": 0,
            "expected_missing_evidence_count": 0,
            "runtime_capture_required_row_count": 0,
            "future_adapter_required_row_count": 0,
            "receipt_bound_row_count": 0,
            "dense_output_row_count": 0,
            "live_proof_row_count": 0,
            "runtime_actuator_row_count": 0,
            "validator_command_count": 0,
            "completion_gate_count": 0,
            "dependency_edge_count": 0,
            "missing_item_count": 1,
        }
    return {
        "ready": manifest.get("ready") is True,
        "reason_count": int_count(manifest.get("reason_count")),
        "evidence_row_count": int_count(manifest.get("evidence_row_count")),
        "expected_missing_evidence_count": int_count(manifest.get("expected_missing_evidence_count")),
        "runtime_capture_required_row_count": int_count(manifest.get("runtime_capture_required_row_count")),
        "future_adapter_required_row_count": int_count(manifest.get("future_adapter_required_row_count")),
        "receipt_bound_row_count": int_count(manifest.get("receipt_bound_row_count")),
        "dense_output_row_count": int_count(manifest.get("dense_output_row_count")),
        "live_proof_row_count": int_count(manifest.get("live_proof_row_count")),
        "runtime_actuator_row_count": int_count(manifest.get("runtime_actuator_row_count")),
        "validator_command_count": int_count(manifest.get("validator_command_count")),
        "completion_gate_count": int_count(manifest.get("completion_gate_count")),
        "dependency_edge_count": int_count(manifest.get("dependency_edge_count")),
        "missing_item_count": int_count(manifest.get("missing_item_count")),
    }

BLOCKER_RESOLUTION_PACKAGE_ORDER = {
    "policy_candidate_trace_capture": 1,
    "capture_result_receipt_intake": 2,
    "dense_fallback_output_capture": 3,
    "runtime_actuator_spike": 4,
    "live_capability_proof_fill": 5,
    "unclassified_blocker_evidence": 99,
}


def blocker_resolution_package_id(row: JSONDict) -> str:
    reason_id = str(row.get("reason_id") or "")
    evidence_kind = str(row.get("evidence_kind") or "")
    if evidence_kind == "policy_candidate_trace_replay" or reason_id in {"no_replay_policy_candidate", "reuse_evidence_capture_not_ready"}:
        return "policy_candidate_trace_capture"
    if evidence_kind == "runtime_capture_receipt_gate" or reason_id == "capture_result_intake_not_ready":
        return "capture_result_receipt_intake"
    if evidence_kind.endswith("_fallback_output_row") or reason_id == "dense_fallback_comparison_not_ready":
        return "dense_fallback_output_capture"
    if evidence_kind == "live_capability_proof_blocker" or reason_id == "live_capability_proof_not_ready":
        return "live_capability_proof_fill"
    if evidence_kind == "runtime_actuator_capability" or reason_id.startswith("live_actuator_"):
        return "runtime_actuator_spike"
    return "unclassified_blocker_evidence"


def blocker_resolution_package_metadata(package_id: str) -> JSONDict:
    metadata = {
        "policy_candidate_trace_capture": {
            "operator_stage": "approved_runtime_capture",
            "package_class": "policy_candidate_replay",
            "summary": "Capture a candidate router trace, fill its receipt, and replay it for reuse-distance observations.",
            "next_action": "Record explicit approval, capture the candidate router trace, fill the trace receipt, and run policy-candidate replay validation.",
            "depends_on_work_package_ids": [],
            "completion_gates": [
                "runtime approvals are recorded on the selected request",
                "candidate router trace artifact exists at the planned path",
                "trace capture receipt is ready and bound to the approved request",
                "policy replay either produces reuse-distance observations or preserves the no-go reason",
            ],
        },
        "capture_result_receipt_intake": {
            "operator_stage": "post_capture_receipt_intake",
            "package_class": "runtime_capture_receipts",
            "summary": "Fill receipt-bound runtime artifacts and rerun capture-result intake before bundle promotion.",
            "next_action": "Complete approved runtime captures, fill their receipts, then rerun capture-result intake until receipt gates are ready.",
            "depends_on_work_package_ids": ["policy_candidate_trace_capture"],
            "completion_gates": [
                "approved runtime capture artifacts are filled at their planned paths",
                "capture receipts are ready and request-bound",
                "capture-result intake reports zero missing receipt gates for completed captures",
                "bundle update readiness is evaluated without promoting live-spike readiness",
            ],
        },
        "dense_fallback_output_capture": {
            "operator_stage": "fallback_quality_bounds",
            "package_class": "dense_fallback_comparison",
            "summary": "Capture managed and dense outputs for the shared prompt set, then build the dense fallback comparison.",
            "next_action": "Fill managed and dense output summaries with approved captures and ready receipts, then build and validate the fallback comparison artifact.",
            "depends_on_work_package_ids": ["capture_result_receipt_intake"],
            "completion_gates": [
                "managed output summary is filled from the approved request and prompt set",
                "dense output summary is filled from the same approved request and prompt set",
                "embedded output receipts are ready for both outputs",
                "dense fallback comparison artifact validates with pair-consistent provenance",
            ],
        },
        "live_capability_proof_fill": {
            "operator_stage": "future_adapter_proof",
            "package_class": "live_capability_proof",
            "summary": "Fill the live proof sections for residency observation, residency control, cleanup, artifact export, and context binding.",
            "next_action": "Populate live-capability proof sections only after the adapter surface can produce approved residency/control and cleanup evidence.",
            "depends_on_work_package_ids": ["runtime_actuator_spike"],
            "completion_gates": [
                "live proof artifact matches the expected bundle, model, backend, and prompt context",
                "residency observation evidence is filled from an approved adapter path",
                "residency control evidence is filled from an approved adapter path",
                "cleanup/restore and artifact-export proof sections are filled",
            ],
        },
        "runtime_actuator_spike": {
            "operator_stage": "phase4_adapter_spike_handoff",
            "package_class": "runtime_actuator",
            "summary": "Use the actuator-spike proof handoff before any live managed expert loading claim.",
            "next_action": "Keep these rows future-adapter bound until proof requirements, dependency order, cleanup path, and capability artifacts validate.",
            "depends_on_work_package_ids": ["dense_fallback_output_capture"],
            "completion_gates": [
                "runtime actuator spike handoff validates for the selected backend",
                "proof requirements cover every live-actuator capability",
                "residency observation proof exists before residency-control mutation",
                "dense fallback and artifact export proofs exist before residency-control mutation",
                "cleanup/restore proof exists before live managed-loading readiness",
            ],
        },
        "unclassified_blocker_evidence": {
            "operator_stage": "manual_triage",
            "package_class": "unclassified",
            "summary": "Triage blocker rows that do not match a known Phase 3 work package.",
            "next_action": "Classify these rows before operator handoff is considered solid.",
            "depends_on_work_package_ids": [],
            "completion_gates": [
                "every row is assigned to a known work package",
                "each known package has explicit completion gates",
            ],
        },
    }
    return metadata.get(package_id, metadata["unclassified_blocker_evidence"])




def runtime_actuator_spike_package_handoff(runtime_actuator_spike: JSONDict | None) -> JSONDict:
    spike = runtime_actuator_spike if isinstance(runtime_actuator_spike, dict) else {}
    requirements = spike.get("proof_requirements") if isinstance(spike.get("proof_requirements"), list) else []
    compact_requirements: list[JSONDict] = []
    for requirement in requirements:
        if not isinstance(requirement, dict):
            continue
        compact_requirements.append(
            {
                "capability_id": requirement.get("capability_id"),
                "current_status": requirement.get("current_status"),
                "spike_stage": requirement.get("spike_stage"),
                "live_ready": requirement.get("live_ready") is True,
                "proof_artifacts": list_of_strings(requirement.get("proof_artifacts")),
                "proof_artifact_count": int_count(requirement.get("proof_artifact_count")),
                "dependency_ids": list_of_strings(requirement.get("dependency_ids")),
                "dependency_count": int_count(requirement.get("dependency_count")),
                "completion_gate": requirement.get("completion_gate"),
                "requires_explicit_runtime_approval_before_live": requirement.get("requires_explicit_runtime_approval_before_live") is True,
            }
        )
    return {
        "handoff_ready": spike.get("spike_handoff_ready") is True,
        "live_spike_ready": spike.get("live_spike_ready") is True,
        "backend_family": spike.get("backend_family"),
        "proof_requirement_count": int_count(spike.get("proof_requirement_count")),
        "proof_artifact_count": int_count(spike.get("proof_artifact_count")),
        "dependency_edge_count": int_count(spike.get("dependency_edge_count")),
        "blocking_capability_count": int_count(spike.get("blocking_capability_count")),
        "control_blocker_count": int_count(spike.get("control_blocker_count")),
        "proof_requirement_ids": [str(item.get("capability_id")) for item in compact_requirements if item.get("capability_id")],
        "blocking_capabilities": list_of_strings(spike.get("blocking_capabilities")),
        "control_blockers": list_of_strings(spike.get("control_blockers")),
        "implementation_sequence": list_of_strings(spike.get("implementation_sequence")),
        "proof_requirements": compact_requirements,
    }
def phase3_blocker_resolution_queue_manifest(
    blocker_evidence_ledger_manifest: JSONDict,
    runtime_actuator_spike: JSONDict | None = None,
) -> JSONDict:
    ledger = blocker_evidence_ledger_manifest if isinstance(blocker_evidence_ledger_manifest, dict) else {}
    runtime_spike = runtime_actuator_spike if isinstance(runtime_actuator_spike, dict) else {}
    rows = ledger.get("evidence_rows") if isinstance(ledger.get("evidence_rows"), list) else []
    grouped: dict[str, list[JSONDict]] = {}
    missing_items: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            missing_items.append("resolution_queue_row_not_object")
            continue
        package_id = blocker_resolution_package_id(row)
        grouped.setdefault(package_id, []).append(row)

    packages: list[JSONDict] = []
    present_package_ids = set(grouped)
    for package_id in sorted(grouped, key=lambda value: BLOCKER_RESOLUTION_PACKAGE_ORDER.get(value, 99)):
        package_rows = grouped[package_id]
        metadata = blocker_resolution_package_metadata(package_id)
        reason_ids = sorted({str(row.get("reason_id")) for row in package_rows if row.get("reason_id")})
        evidence_kinds = sorted({str(row.get("evidence_kind")) for row in package_rows if row.get("evidence_kind")})
        artifact_paths = sorted({str(row.get("artifact_path")) for row in package_rows if row.get("artifact_path")})
        receipt_paths = sorted({str(row.get("receipt_path")) for row in package_rows if row.get("receipt_path")})
        prompt_ids = sorted({str(row.get("prompt_id")) for row in package_rows if row.get("prompt_id")})
        blocker_ids = sorted({str(row.get("blocker_id")) for row in package_rows if row.get("blocker_id")})
        declared_dependencies = list_of_strings(metadata.get("depends_on_work_package_ids"))
        depends_on_work_package_ids = [dependency for dependency in declared_dependencies if dependency in present_package_ids]
        completion_gates = list_of_strings(metadata.get("completion_gates"))
        if not completion_gates:
            missing_items.append(f"resolution_queue_completion_gates_missing:{package_id}")
        row_validator_command_count = sum(int_count(row.get("validator_command_count")) for row in package_rows)
        reason_validator_command_counts = {
            str(row.get("reason_id")): int_count(row.get("closure_validator_command_count"))
            for row in package_rows
            if row.get("reason_id")
        }
        reason_validator_command_count = sum(reason_validator_command_counts.values())
        package: JSONDict = {
            "sequence_rank": BLOCKER_RESOLUTION_PACKAGE_ORDER.get(package_id, 99),
            "work_package_id": package_id,
            "operator_stage": metadata["operator_stage"],
            "package_class": metadata["package_class"],
            "summary": metadata["summary"],
            "next_action": metadata["next_action"],
            "depends_on_work_package_ids": depends_on_work_package_ids,
            "dependency_count": len(depends_on_work_package_ids),
            "completion_gates": completion_gates,
            "completion_gate_count": len(completion_gates),
            "reason_ids": reason_ids,
            "evidence_kinds": evidence_kinds,
            "row_count": len(package_rows),
            "runtime_capture_required_row_count": sum(1 for row in package_rows if row.get("approval_stage") == "runtime_capture_required"),
            "future_adapter_required_row_count": sum(1 for row in package_rows if row.get("approval_stage") == "future_adapter_required"),
            "receipt_bound_row_count": sum(1 for row in package_rows if row.get("evidence_kind") == "runtime_capture_receipt_gate"),
            "row_validator_command_count": row_validator_command_count,
            "reason_validator_command_count": reason_validator_command_count,
            "validator_command_count": max(row_validator_command_count, reason_validator_command_count),
            "artifact_paths": artifact_paths,
            "receipt_paths": receipt_paths,
            "prompt_ids": prompt_ids,
            "blocker_ids": blocker_ids,
        }
        if package_id == "runtime_actuator_spike":
            package["proof_handoff"] = runtime_actuator_spike_package_handoff(runtime_spike)
        packages.append(package)

    package_ranks = {str(package.get("work_package_id")): int_count(package.get("sequence_rank")) for package in packages if package.get("work_package_id")}
    for package in packages:
        package_id = str(package.get("work_package_id") or "unknown")
        package_rank = int_count(package.get("sequence_rank"))
        for dependency_id in list_of_strings(package.get("depends_on_work_package_ids")):
            dependency_rank = package_ranks.get(dependency_id)
            if dependency_rank is None:
                missing_items.append(f"resolution_queue_dependency_unknown:{package_id}:{dependency_id}")
            elif dependency_rank >= package_rank:
                missing_items.append(f"resolution_queue_dependency_order_invalid:{package_id}:{dependency_id}")
    queue_row_count = sum(int_count(package.get("row_count")) for package in packages)
    expected_rows = int_count(ledger.get("evidence_row_count"))
    if expected_rows != queue_row_count:
        missing_items.append(f"resolution_queue_row_count_mismatch:{queue_row_count}/{expected_rows}")
    if "unclassified_blocker_evidence" in grouped:
        missing_items.append("resolution_queue_unclassified_rows_present")
    ready = ledger.get("ready") is True and expected_rows == queue_row_count and not missing_items
    ordered_packages = sorted(packages, key=lambda item: int_count(item.get("sequence_rank")))
    next_unblocked_work_package = next(
        (package for package in ordered_packages if int_count(package.get("dependency_count")) == 0),
        {},
    )
    return {
        "ready": ready,
        "metadata_only": True,
        "work_package_count": len(packages),
        "queue_row_count": queue_row_count,
        "expected_ledger_row_count": expected_rows,
        "runtime_capture_package_count": sum(1 for package in packages if package.get("operator_stage") in {"approved_runtime_capture", "post_capture_receipt_intake", "fallback_quality_bounds"}),
        "future_adapter_package_count": sum(1 for package in packages if package.get("operator_stage") in {"future_adapter_proof", "phase4_adapter_design"}),
        "runtime_capture_required_row_count": sum(int_count(package.get("runtime_capture_required_row_count")) for package in packages),
        "future_adapter_required_row_count": sum(int_count(package.get("future_adapter_required_row_count")) for package in packages),
        "receipt_bound_row_count": sum(int_count(package.get("receipt_bound_row_count")) for package in packages),
        "validator_command_count": sum(int_count(package.get("validator_command_count")) for package in packages),
        "completion_gate_count": sum(int_count(package.get("completion_gate_count")) for package in packages),
        "dependency_edge_count": sum(int_count(package.get("dependency_count")) for package in packages),
        "missing_item_count": len(missing_items),
        "missing_items": sorted(set(missing_items)),
        "work_packages": packages,
        "next_unblocked_work_package": next_unblocked_work_package,
        "safety_contract": [
            "blocker resolution queue is metadata only",
            "blocker resolution queue generation does not launch runtimes",
            "blocker resolution queue generation does not run Docker",
            "blocker resolution queue generation does not call endpoints",
            "blocker resolution queue generation does not inspect private tokens",
            "blocker resolution queue generation does not send prompt traffic",
            "blocker resolution queue generation does not mutate runtime residency",
        ],
    }


def phase3_blocker_resolution_queue_summary(manifest: JSONDict) -> JSONDict:
    if not isinstance(manifest, dict) or not manifest:
        return {
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
            "missing_item_count": 1,
            "next_unblocked_work_package_id": None,
            "next_unblocked_sequence_rank": None,
            "next_unblocked_operator_stage": None,
            "next_unblocked_package_class": None,
            "next_unblocked_summary": None,
            "next_unblocked_next_action": None,
            "next_unblocked_row_count": 0,
            "next_unblocked_runtime_capture_required_row_count": 0,
            "next_unblocked_future_adapter_required_row_count": 0,
            "next_unblocked_validator_command_count": 0,
            "next_unblocked_completion_gate_count": 0,
        }
    next_unblocked = manifest.get("next_unblocked_work_package") if isinstance(manifest.get("next_unblocked_work_package"), dict) else {}
    return {
        "ready": manifest.get("ready") is True,
        "work_package_count": int_count(manifest.get("work_package_count")),
        "queue_row_count": int_count(manifest.get("queue_row_count")),
        "expected_ledger_row_count": int_count(manifest.get("expected_ledger_row_count")),
        "runtime_capture_package_count": int_count(manifest.get("runtime_capture_package_count")),
        "future_adapter_package_count": int_count(manifest.get("future_adapter_package_count")),
        "runtime_capture_required_row_count": int_count(manifest.get("runtime_capture_required_row_count")),
        "future_adapter_required_row_count": int_count(manifest.get("future_adapter_required_row_count")),
        "receipt_bound_row_count": int_count(manifest.get("receipt_bound_row_count")),
        "validator_command_count": int_count(manifest.get("validator_command_count")),
        "completion_gate_count": int_count(manifest.get("completion_gate_count")),
        "dependency_edge_count": int_count(manifest.get("dependency_edge_count")),
        "missing_item_count": int_count(manifest.get("missing_item_count")),
        "next_unblocked_work_package_id": next_unblocked.get("work_package_id"),
        "next_unblocked_sequence_rank": next_unblocked.get("sequence_rank"),
        "next_unblocked_operator_stage": next_unblocked.get("operator_stage"),
        "next_unblocked_package_class": next_unblocked.get("package_class"),
        "next_unblocked_summary": next_unblocked.get("summary"),
        "next_unblocked_next_action": next_unblocked.get("next_action"),
        "next_unblocked_row_count": int_count(next_unblocked.get("row_count")),
        "next_unblocked_runtime_capture_required_row_count": int_count(next_unblocked.get("runtime_capture_required_row_count")),
        "next_unblocked_future_adapter_required_row_count": int_count(next_unblocked.get("future_adapter_required_row_count")),
        "next_unblocked_validator_command_count": int_count(next_unblocked.get("validator_command_count")),
        "next_unblocked_completion_gate_count": int_count(next_unblocked.get("completion_gate_count")),
    }

def matching_request_item(manifest: list[JSONDict], request_path: str | None, request_name: str | None) -> JSONDict:
    for item in manifest:
        if not isinstance(item, dict):
            continue
        item_path = normalize_manifest_path(item.get("request_path"))
        if request_path and item_path == request_path:
            return item
        if request_name and item.get("request_name") == request_name:
            return item
    return {}


def recommended_runtime_capture_preflight_manifest(
    recommendation: JSONDict | None,
    post_approval_preview: JSONDict | None,
    all_capture_queue_manifest: list[JSONDict],
    receipt_command_manifest: list[JSONDict],
    blocker_closure_manifest: list[JSONDict],
) -> JSONDict:
    if not isinstance(recommendation, dict) or not recommendation:
        return {}
    post_approval_preview = post_approval_preview if isinstance(post_approval_preview, dict) else {}
    request_path = normalize_manifest_path(recommendation.get("request_path"))
    request_name = str(recommendation.get("request_name") or "")
    queue_entry = matching_request_item(all_capture_queue_manifest, request_path, request_name)
    missing_approval_keys = list_of_strings(recommendation.get("missing_approval_keys"))
    pending_artifact_ids = list_of_strings(recommendation.get("pending_artifact_ids"))
    if not pending_artifact_ids:
        pending_artifact_ids = list_of_strings(queue_entry.get("pending_artifact_ids"))
    receipt_entries = [
        item
        for item in receipt_command_manifest
        if isinstance(item, dict)
        and (
            (request_path and normalize_manifest_path(item.get("request_path")) == request_path)
            or (request_name and item.get("request_name") == request_name)
        )
    ]
    receipt_by_artifact = {
        str(item.get("artifact_id")): item
        for item in receipt_entries
        if isinstance(item.get("artifact_id"), str)
    }
    artifact_checks: list[JSONDict] = []
    missing_preflight_items: list[str] = []
    for artifact_id in pending_artifact_ids:
        receipt = receipt_by_artifact.get(artifact_id, {})
        validator_count = int_count(receipt.get("validator_command_count"))
        artifact_path = normalize_manifest_path(receipt.get("artifact_path"))
        receipt_path = normalize_manifest_path(receipt.get("receipt_path"))
        artifact_checks.append(
            {
                "artifact_id": artifact_id,
                "artifact_path": artifact_path,
                "receipt_path": receipt_path,
                "receipt_kind": receipt.get("receipt_kind"),
                "receipt_ready": receipt.get("receipt_ready") is True,
                "fill_status_after_approval": receipt.get("fill_status_after_approval"),
                "validator_command_count": validator_count,
                "validator_commands": receipt.get("validator_commands") if isinstance(receipt.get("validator_commands"), list) else [],
            }
        )
        if not receipt:
            missing_preflight_items.append(f"receipt_entry_missing:{artifact_id}")
        if not artifact_path:
            missing_preflight_items.append(f"artifact_path_missing:{artifact_id}")
        if not receipt_path:
            missing_preflight_items.append(f"receipt_path_missing:{artifact_id}")
        if validator_count == 0:
            missing_preflight_items.append(f"validator_commands_missing:{artifact_id}")

    approval_rebuild = recommendation.get("approval_rebuild_command") if isinstance(recommendation.get("approval_rebuild_command"), dict) else {}
    approval_command = approval_rebuild.get("command") if isinstance(approval_rebuild.get("command"), list) else []
    capture_sequence = recommendation.get("capture_sequence") if isinstance(recommendation.get("capture_sequence"), list) else []
    sequence_ids = [str(step.get("id")) for step in capture_sequence if isinstance(step, dict) and step.get("id")]
    required_sequence_ids = ["record_runtime_approvals", "capture_candidate_router_trace", "run_capture_result_intake"]
    missing_sequence_ids = [step_id for step_id in required_sequence_ids if step_id not in sequence_ids]
    runtime_closure_rows = [
        item
        for item in blocker_closure_manifest
        if isinstance(item, dict) and item.get("approval_stage") == "runtime_capture_required"
    ]
    if not request_path:
        missing_preflight_items.append("request_path_missing")
    if not pending_artifact_ids:
        missing_preflight_items.append("pending_artifacts_missing")
    if len(artifact_checks) != len(pending_artifact_ids):
        missing_preflight_items.append("artifact_check_count_mismatch")
    if missing_approval_keys and not approval_command:
        missing_preflight_items.append("approval_rebuild_command_missing")
    if post_approval_preview.get("valid") is not True:
        missing_preflight_items.append("post_approval_preview_invalid")
    if post_approval_preview.get("ready_for_operator_capture") is not True:
        missing_preflight_items.append("post_approval_preview_not_ready_for_operator")
    if post_approval_preview.get("capture_complete") is True:
        missing_preflight_items.append("post_approval_preview_already_capture_complete")
    if post_approval_preview.get("mutates_request") is not False:
        missing_preflight_items.append("post_approval_preview_mutation_risk")
    if missing_sequence_ids:
        missing_preflight_items.append("capture_sequence_missing_steps")
    if not runtime_closure_rows:
        missing_preflight_items.append("runtime_closure_rows_missing")

    validator_command_count = sum(int_count(item.get("validator_command_count")) for item in artifact_checks)
    missing_preflight_items = sorted(set(missing_preflight_items))
    return {
        "ready": not missing_preflight_items,
        "request_name": recommendation.get("request_name"),
        "request_path": request_path,
        "status": recommendation.get("status"),
        "queue_rank": recommendation.get("queue_rank"),
        "selection_rationale": recommendation.get("selection_rationale"),
        "approval_required": bool(missing_approval_keys),
        "missing_approval_keys": missing_approval_keys,
        "approval_rebuild_command_present": bool(approval_command),
        "approval_rebuild_command": approval_rebuild,
        "post_approval_preview_valid": post_approval_preview.get("valid") is True,
        "post_approval_ready_for_operator": post_approval_preview.get("ready_for_operator_capture") is True,
        "post_approval_capture_complete": post_approval_preview.get("capture_complete") is True,
        "post_approval_mutates_request": post_approval_preview.get("mutates_request") is True,
        "capture_sequence_step_count": len(capture_sequence),
        "missing_capture_sequence_ids": missing_sequence_ids,
        "pending_artifact_ids": pending_artifact_ids,
        "pending_artifact_count": len(pending_artifact_ids),
        "artifact_check_count": len(artifact_checks),
        "artifact_checks": artifact_checks,
        "receipt_entry_count": len(receipt_entries),
        "receipt_ready_count": sum(1 for item in artifact_checks if item.get("receipt_ready") is True),
        "validator_command_count": validator_command_count,
        "runtime_closure_reason_count": len(runtime_closure_rows),
        "runtime_closure_reason_ids": [str(item.get("reason_id")) for item in runtime_closure_rows if item.get("reason_id")],
        "runtime_closure_validator_command_count": sum(int_count(item.get("validator_command_count")) for item in runtime_closure_rows),
        "missing_preflight_item_count": len(missing_preflight_items),
        "missing_preflight_items": missing_preflight_items,
    }


def recommended_runtime_capture_preflight_summary(manifest: JSONDict) -> JSONDict:
    if not isinstance(manifest, dict) or not manifest:
        return {
            "ready": False,
            "request_name": None,
            "missing_preflight_item_count": 1,
            "missing_preflight_items": ["recommended_runtime_capture_missing"],
        }
    return {
        "ready": manifest.get("ready") is True,
        "request_name": manifest.get("request_name"),
        "request_path": manifest.get("request_path"),
        "approval_required": manifest.get("approval_required") is True,
        "pending_artifact_count": int_count(manifest.get("pending_artifact_count")),
        "artifact_check_count": int_count(manifest.get("artifact_check_count")),
        "receipt_entry_count": int_count(manifest.get("receipt_entry_count")),
        "receipt_ready_count": int_count(manifest.get("receipt_ready_count")),
        "validator_command_count": int_count(manifest.get("validator_command_count")),
        "runtime_closure_reason_count": int_count(manifest.get("runtime_closure_reason_count")),
        "runtime_closure_validator_command_count": int_count(manifest.get("runtime_closure_validator_command_count")),
        "missing_preflight_item_count": int_count(manifest.get("missing_preflight_item_count")),
        "missing_preflight_items": list_of_strings(manifest.get("missing_preflight_items")),
    }


def recommended_runtime_capture_execution_coverage_manifest(
    recommendation: JSONDict | None,
    preflight_manifest: JSONDict | None,
    all_capture_queue_manifest: list[JSONDict],
    command_contract: JSONDict | None = None,
) -> JSONDict:
    if not isinstance(recommendation, dict) or not recommendation:
        return {}
    preflight_manifest = preflight_manifest if isinstance(preflight_manifest, dict) else {}
    request_path = normalize_manifest_path(recommendation.get("request_path"))
    request_name = str(recommendation.get("request_name") or "")
    queue_entry = matching_request_item(all_capture_queue_manifest, request_path, request_name)
    pending_artifact_ids = list_of_strings(preflight_manifest.get("pending_artifact_ids"))
    if not pending_artifact_ids:
        pending_artifact_ids = list_of_strings(recommendation.get("pending_artifact_ids"))
    if not pending_artifact_ids:
        pending_artifact_ids = list_of_strings(queue_entry.get("pending_artifact_ids"))
    preflight_checks = preflight_manifest.get("artifact_checks") if isinstance(preflight_manifest.get("artifact_checks"), list) else []
    preflight_by_artifact = {
        str(item.get("artifact_id")): item
        for item in preflight_checks
        if isinstance(item, dict) and item.get("artifact_id")
    }
    queue_steps = queue_entry.get("capture_fill_steps") if isinstance(queue_entry.get("capture_fill_steps"), list) else []
    queue_steps_by_artifact = {
        str(item.get("artifact_id")): item
        for item in queue_steps
        if isinstance(item, dict) and item.get("artifact_id")
    }
    runtime_commands_by_artifact = bound_runtime_capture_commands_by_artifact(command_contract)
    artifact_execution: list[JSONDict] = []
    missing_execution_items: list[str] = []
    missing_capture_command_artifact_ids: list[str] = []
    for artifact_id in pending_artifact_ids:
        preflight = preflight_by_artifact.get(artifact_id, {})
        step = queue_steps_by_artifact.get(artifact_id, {})
        command_options = compact_command_options(step.get("command_options"))
        bound_runtime_command = runtime_commands_by_artifact.get(artifact_id)
        if isinstance(bound_runtime_command, dict):
            command_options.append(bound_runtime_command)
        operator_command_count = len(command_options) or int_count(step.get("command_option_count"))
        runtime_capture_options = [option for option in command_options if command_option_is_runtime_capture(option)]
        runtime_capture_command_count = len(runtime_capture_options)
        metadata_command_count = sum(1 for option in command_options if command_option_is_metadata_only(option))
        runtime_command_contract_errors: list[str] = []
        for option in runtime_capture_options:
            if option.get("requires_prompt_traffic") is True and option.get("may_send_prompt_traffic") is not True:
                runtime_command_contract_errors.append(f"prompt_traffic_ack_missing:{artifact_id}")
            option_request_path = normalize_manifest_path(option.get("runtime_capture_request_path"))
            if option_request_path != request_path:
                runtime_command_contract_errors.append(f"runtime_capture_request_path_missing:{artifact_id}")
        has_capture_command = runtime_capture_command_count > 0 and not runtime_command_contract_errors
        if not step:
            missing_execution_items.append(f"capture_fill_step_missing:{artifact_id}")
        if runtime_command_contract_errors:
            missing_execution_items.extend(runtime_command_contract_errors)
        if not has_capture_command:
            missing_capture_command_artifact_ids.append(artifact_id)
        step_after_approval = step.get("step_after_approval") if isinstance(step.get("step_after_approval"), dict) else {}
        current_step = step.get("current_step") if isinstance(step.get("current_step"), dict) else {}
        artifact_execution.append(
            {
                "artifact_id": artifact_id,
                "artifact_path": preflight.get("artifact_path") or normalize_manifest_path(step.get("artifact_path")),
                "receipt_path": preflight.get("receipt_path") or normalize_manifest_path(step.get("receipt_path")),
                "receipt_kind": preflight.get("receipt_kind") or step.get("receipt_kind"),
                "current_status": current_step.get("status"),
                "status_after_approval": step_after_approval.get("status"),
                "fill_status_after_approval": step.get("fill_status_after_approval"),
                "receipt_ready": preflight.get("receipt_ready") is True or step.get("receipt_ready") is True,
                "validator_command_count": int_count(preflight.get("validator_command_count") or step.get("validator_command_count")),
                "operator_command_option_count": operator_command_count,
                "metadata_command_option_count": metadata_command_count,
                "capture_command_option_count": runtime_capture_command_count,
                "has_capture_command": has_capture_command,
                "has_runtime_capture_command": has_capture_command,
                "runtime_command_contract_error_count": len(runtime_command_contract_errors),
                "runtime_command_contract_errors": runtime_command_contract_errors,
                "launch_card_binding_ready": isinstance(bound_runtime_command, dict),
                "capture_mode": "runtime_command_available" if has_capture_command else "manual_runtime_capture_required",
                "command_options": command_options,
                "blockers_after_approval": list_of_strings(step.get("blockers_after_approval")),
            }
        )

    missing_execution_items = sorted(set(missing_execution_items))
    missing_capture_command_artifact_ids = sorted(set(missing_capture_command_artifact_ids))
    command_option_count = sum(int_count(item.get("capture_command_option_count")) for item in artifact_execution)
    operator_command_option_count = sum(int_count(item.get("operator_command_option_count")) for item in artifact_execution)
    metadata_command_option_count = sum(int_count(item.get("metadata_command_option_count")) for item in artifact_execution)
    artifacts_with_capture_command_count = sum(1 for item in artifact_execution if item.get("has_capture_command") is True)
    manual_capture_required_count = len(missing_capture_command_artifact_ids)
    manual_operator_capture_ready = (
        preflight_manifest.get("ready") is True
        and bool(pending_artifact_ids)
        and not missing_execution_items
        and len(artifact_execution) == len(pending_artifact_ids)
    )
    automated_capture_ready = manual_operator_capture_ready and manual_capture_required_count == 0
    return {
        "ready": manual_operator_capture_ready,
        "automated_capture_ready": automated_capture_ready,
        "manual_operator_capture_ready": manual_operator_capture_ready,
        "request_name": recommendation.get("request_name"),
        "request_path": request_path,
        "pending_artifact_count": len(pending_artifact_ids),
        "artifact_execution_count": len(artifact_execution),
        "capture_command_option_count": command_option_count,
        "operator_command_option_count": operator_command_option_count,
        "metadata_command_option_count": metadata_command_option_count,
        "artifacts_with_capture_command_count": artifacts_with_capture_command_count,
        "runtime_command_contract_error_count": sum(int_count(item.get("runtime_command_contract_error_count")) for item in artifact_execution),
        "manual_capture_required_count": manual_capture_required_count,
        "missing_capture_command_count": manual_capture_required_count,
        "missing_capture_command_artifact_ids": missing_capture_command_artifact_ids,
        "missing_execution_item_count": len(missing_execution_items),
        "missing_execution_items": missing_execution_items,
        "artifact_execution": artifact_execution,
        "automation_blockers": [
            f"runtime_capture_command_option_missing:{artifact_id}"
            for artifact_id in missing_capture_command_artifact_ids
        ],
    }


def recommended_runtime_capture_execution_coverage_summary(manifest: JSONDict) -> JSONDict:
    if not isinstance(manifest, dict) or not manifest:
        return {
            "ready": False,
            "automated_capture_ready": False,
            "manual_operator_capture_ready": False,
            "request_name": None,
            "missing_execution_item_count": 1,
            "missing_execution_items": ["recommended_runtime_capture_missing"],
        }
    return {
        "ready": manifest.get("ready") is True,
        "automated_capture_ready": manifest.get("automated_capture_ready") is True,
        "manual_operator_capture_ready": manifest.get("manual_operator_capture_ready") is True,
        "request_name": manifest.get("request_name"),
        "request_path": manifest.get("request_path"),
        "pending_artifact_count": int_count(manifest.get("pending_artifact_count")),
        "artifact_execution_count": int_count(manifest.get("artifact_execution_count")),
        "capture_command_option_count": int_count(manifest.get("capture_command_option_count")),
        "operator_command_option_count": int_count(manifest.get("operator_command_option_count")),
        "metadata_command_option_count": int_count(manifest.get("metadata_command_option_count")),
        "artifacts_with_capture_command_count": int_count(manifest.get("artifacts_with_capture_command_count")),
        "manual_capture_required_count": int_count(manifest.get("manual_capture_required_count")),
        "missing_capture_command_count": int_count(manifest.get("missing_capture_command_count")),
        "missing_capture_command_artifact_ids": list_of_strings(manifest.get("missing_capture_command_artifact_ids")),
        "missing_execution_item_count": int_count(manifest.get("missing_execution_item_count")),
        "missing_execution_items": list_of_strings(manifest.get("missing_execution_items")),
    }


def bound_runtime_capture_commands_from_launch_card_library(
    launch_card_library_summary: JSONDict,
) -> dict[tuple[str, str], JSONDict]:
    tasks = launch_card_library_summary.get("binding_tasks")
    if not isinstance(tasks, list):
        return {}
    result: dict[tuple[str, str], JSONDict] = {}
    for task in tasks:
        if not isinstance(task, dict) or task.get("command_option_ready") is not True:
            continue
        request_path = normalize_manifest_path(task.get("request_path"))
        artifact_id = str(task.get("artifact_id") or "").strip()
        if not request_path or not artifact_id:
            continue
        launch_command = [part for part in list_of_strings(task.get("launch_command")) if part.strip()]
        callable_id = str(task.get("model_plane_callable_id") or "").strip()
        command = launch_command or (["model-plane-callable", callable_id] if callable_id else [])
        if not command:
            continue
        result[(request_path, artifact_id)] = {
            "command_class": "phase3_model_plane_launch_card_runtime_capture",
            "requires_runtime": True,
            "requires_prompt_traffic": True,
            "may_send_prompt_traffic": task.get("requires_prompt_traffic") is True,
            "runtime_capture_request_path": request_path,
            "requires_explicit_user_approval": True,
            "metadata_only": False,
            "command": command,
            "model_plane_callable_id": callable_id or None,
            "source": "launch_card_library_binding",
        }
    return result


def add_bound_runtime_capture_commands_from_contract(
    runtime_commands_by_key: dict[tuple[str, str], JSONDict],
    command_contract: JSONDict | None,
) -> None:
    request_path = normalize_manifest_path(
        command_contract.get("request_path") if isinstance(command_contract, dict) else None
    )
    if not request_path:
        return
    for artifact_id, command in bound_runtime_capture_commands_by_artifact(command_contract).items():
        runtime_commands_by_key[(request_path, artifact_id)] = command


def runtime_capture_launch_card_dir_manifest(
    launch_card_library_summary: JSONDict,
    launch_card_dir: Path | None,
) -> JSONDict:
    cards = launch_card_library_summary.get("cards") if isinstance(launch_card_library_summary.get("cards"), list) else []
    binding_tasks = launch_card_library_summary.get("binding_tasks") if isinstance(launch_card_library_summary.get("binding_tasks"), list) else []
    expected_cards: list[JSONDict] = []
    seen_expected_cards: set[tuple[str, str]] = set()
    for source in [item for item in cards if isinstance(item, dict)] + [item for item in binding_tasks if isinstance(item, dict)]:
        request_path = normalize_manifest_path(source.get("request_path"))
        launch_card_path = normalize_manifest_path(source.get("launch_card_path"))
        if not request_path or not launch_card_path:
            continue
        key = (request_path, launch_card_path)
        if key in seen_expected_cards:
            continue
        seen_expected_cards.add(key)
        expected_cards.append(
            {
                "request_name": source.get("request_name"),
                "request_path": request_path,
                "launch_card_path": launch_card_path,
            }
        )
    manifest: JSONDict = {
        "provided": launch_card_dir is not None,
        "path": str(launch_card_dir) if launch_card_dir is not None else None,
        "valid": True,
        "directory_ready": False,
        "expected_card_count": len(expected_cards),
        "matched_card_count": 0,
        "missing_card_count": 0,
        "cards": [],
        "paths_by_request": {},
        "errors": [],
        "blockers": [],
    }
    if launch_card_dir is None:
        manifest["blockers"] = ["runtime_capture_launch_card_dir_not_provided"]
        return manifest
    if not launch_card_dir.exists():
        manifest["valid"] = False
        manifest["errors"] = [f"runtime_capture_launch_card_dir_missing:{launch_card_dir}"]
        manifest["blockers"] = ["runtime_capture_launch_card_dir_missing"]
        manifest["missing_card_count"] = len(expected_cards)
        return manifest
    if not launch_card_dir.is_dir():
        manifest["valid"] = False
        manifest["errors"] = [f"runtime_capture_launch_card_dir_not_directory:{launch_card_dir}"]
        manifest["blockers"] = ["runtime_capture_launch_card_dir_not_directory"]
        manifest["missing_card_count"] = len(expected_cards)
        return manifest

    card_rows: list[JSONDict] = []
    paths_by_request: dict[str, str] = {}
    for expected_card in expected_cards:
        request_path = normalize_manifest_path(expected_card.get("request_path"))
        launch_card_path = normalize_manifest_path(expected_card.get("launch_card_path"))
        expected_file = Path(launch_card_path or "runtime-capture-launch-card.template.json").name
        filled_path = launch_card_dir / expected_file
        exists = filled_path.exists()
        if request_path and exists:
            paths_by_request[request_path] = str(filled_path)
        card_rows.append(
            {
                "request_name": expected_card.get("request_name"),
                "request_path": request_path,
                "source_launch_card_path": launch_card_path,
                "expected_filled_launch_card_path": str(filled_path),
                "filled_launch_card_exists": exists,
            }
        )

    missing_cards = [row for row in card_rows if row.get("filled_launch_card_exists") is not True]
    manifest["cards"] = card_rows
    manifest["paths_by_request"] = paths_by_request
    manifest["matched_card_count"] = len(paths_by_request)
    manifest["missing_card_count"] = len(missing_cards)
    manifest["directory_ready"] = bool(card_rows) and not missing_cards
    if missing_cards:
        manifest["blockers"] = ["runtime_capture_launch_card_dir_incomplete"]
    return manifest


def runtime_capture_launch_card_path_from_dir_manifest(
    launch_card_dir_manifest: JSONDict,
    request_path: Any,
) -> Path | None:
    normalized_request_path = normalize_manifest_path(request_path)
    paths_by_request = launch_card_dir_manifest.get("paths_by_request")
    if not normalized_request_path or not isinstance(paths_by_request, dict):
        return None
    path_text = paths_by_request.get(normalized_request_path)
    return Path(path_text) if isinstance(path_text, str) and path_text.strip() else None


def runtime_capture_command_contracts_from_launch_card_dir(
    all_capture_queue_manifest: list[JSONDict],
    launch_card_dir_manifest: JSONDict,
) -> dict[str, JSONDict]:
    result: dict[str, JSONDict] = {}
    paths_by_request = launch_card_dir_manifest.get("paths_by_request")
    if not isinstance(paths_by_request, dict):
        return result
    for item in all_capture_queue_manifest:
        if not isinstance(item, dict):
            continue
        request_path_text = normalize_manifest_path(item.get("request_path"))
        if not request_path_text or request_path_text not in paths_by_request:
            continue
        request_file = repo_path(request_path_text)
        launch_card_path = runtime_capture_launch_card_path_from_dir_manifest(
            launch_card_dir_manifest,
            request_path_text,
        )
        if request_file is None or launch_card_path is None:
            continue
        result[request_path_text] = plan_phase3_runtime_capture_commands.build_summary(
            request_file,
            launch_card_path=launch_card_path,
        )
    return result


def queue_entry_runtime_capture_execution_coverage(
    queue_entry: JSONDict,
    *,
    runtime_commands_by_key: dict[tuple[str, str], JSONDict] | None = None,
) -> JSONDict:
    runtime_commands_by_key = runtime_commands_by_key or {}
    request_path = normalize_manifest_path(queue_entry.get("request_path"))
    request_name = str(queue_entry.get("request_name") or "")
    pending_artifact_ids = list_of_strings(queue_entry.get("pending_artifact_ids"))
    queue_steps = queue_entry.get("capture_fill_steps") if isinstance(queue_entry.get("capture_fill_steps"), list) else []
    queue_steps_by_artifact = {
        str(item.get("artifact_id")): item
        for item in queue_steps
        if isinstance(item, dict) and item.get("artifact_id")
    }
    artifact_execution: list[JSONDict] = []
    missing_execution_items: list[str] = []
    missing_capture_command_artifact_ids: list[str] = []
    for artifact_id in pending_artifact_ids:
        step = queue_steps_by_artifact.get(artifact_id, {})
        command_options = compact_command_options(step.get("command_options"))
        bound_runtime_command = runtime_commands_by_key.get((request_path or "", artifact_id))
        if isinstance(bound_runtime_command, dict):
            command_options.append(bound_runtime_command)
        operator_command_count = len(command_options) or int_count(step.get("command_option_count"))
        runtime_capture_options = [option for option in command_options if command_option_is_runtime_capture(option)]
        runtime_capture_command_count = len(runtime_capture_options)
        metadata_command_count = sum(1 for option in command_options if command_option_is_metadata_only(option))
        runtime_command_contract_errors: list[str] = []
        for option in runtime_capture_options:
            if option.get("requires_prompt_traffic") is True and option.get("may_send_prompt_traffic") is not True:
                runtime_command_contract_errors.append(f"prompt_traffic_ack_missing:{artifact_id}")
            option_request_path = normalize_manifest_path(option.get("runtime_capture_request_path"))
            if option_request_path != request_path:
                runtime_command_contract_errors.append(f"runtime_capture_request_path_missing:{artifact_id}")
        has_capture_command = runtime_capture_command_count > 0 and not runtime_command_contract_errors
        if not step:
            missing_execution_items.append(f"capture_fill_step_missing:{artifact_id}")
        if runtime_command_contract_errors:
            missing_execution_items.extend(runtime_command_contract_errors)
        if not has_capture_command:
            missing_capture_command_artifact_ids.append(artifact_id)
        step_after_approval = step.get("step_after_approval") if isinstance(step.get("step_after_approval"), dict) else {}
        current_step = step.get("current_step") if isinstance(step.get("current_step"), dict) else {}
        artifact_execution.append(
            {
                "request_name": request_name,
                "request_path": request_path,
                "artifact_id": artifact_id,
                "artifact_path": normalize_manifest_path(step.get("artifact_path")),
                "receipt_path": normalize_manifest_path(step.get("receipt_path")),
                "receipt_kind": step.get("receipt_kind"),
                "current_status": current_step.get("status"),
                "status_after_approval": step_after_approval.get("status"),
                "fill_status_after_approval": step.get("fill_status_after_approval"),
                "receipt_ready": step.get("receipt_ready") is True,
                "validator_command_count": int_count(step.get("validator_command_count")),
                "operator_command_option_count": operator_command_count,
                "metadata_command_option_count": metadata_command_count,
                "capture_command_option_count": runtime_capture_command_count,
                "has_capture_command": has_capture_command,
                "has_runtime_capture_command": has_capture_command,
                "runtime_command_contract_error_count": len(runtime_command_contract_errors),
                "runtime_command_contract_errors": runtime_command_contract_errors,
                "launch_card_binding_ready": isinstance(bound_runtime_command, dict),
                "capture_mode": "runtime_command_available" if has_capture_command else "manual_runtime_capture_required",
                "command_options": command_options,
                "blockers_after_approval": list_of_strings(step.get("blockers_after_approval")),
            }
        )

    missing_execution_items = sorted(set(missing_execution_items))
    missing_capture_command_artifact_ids = sorted(set(missing_capture_command_artifact_ids))
    command_option_count = sum(int_count(item.get("capture_command_option_count")) for item in artifact_execution)
    operator_command_option_count = sum(int_count(item.get("operator_command_option_count")) for item in artifact_execution)
    metadata_command_option_count = sum(int_count(item.get("metadata_command_option_count")) for item in artifact_execution)
    artifacts_with_capture_command_count = sum(1 for item in artifact_execution if item.get("has_capture_command") is True)
    manual_capture_required_count = len(missing_capture_command_artifact_ids)
    manual_operator_capture_ready = (
        bool(request_path)
        and bool(pending_artifact_ids)
        and not missing_execution_items
        and len(artifact_execution) == len(pending_artifact_ids)
    )
    automated_capture_ready = manual_operator_capture_ready and manual_capture_required_count == 0
    return {
        "ready": manual_operator_capture_ready,
        "automated_capture_ready": automated_capture_ready,
        "manual_operator_capture_ready": manual_operator_capture_ready,
        "request_name": request_name,
        "request_path": request_path,
        "rank": int_count(queue_entry.get("rank")),
        "status": queue_entry.get("status"),
        "pending_artifact_count": len(pending_artifact_ids),
        "artifact_execution_count": len(artifact_execution),
        "capture_command_option_count": command_option_count,
        "operator_command_option_count": operator_command_option_count,
        "metadata_command_option_count": metadata_command_option_count,
        "artifacts_with_capture_command_count": artifacts_with_capture_command_count,
        "runtime_command_contract_error_count": sum(int_count(item.get("runtime_command_contract_error_count")) for item in artifact_execution),
        "manual_capture_required_count": manual_capture_required_count,
        "missing_capture_command_count": manual_capture_required_count,
        "missing_capture_command_artifact_ids": missing_capture_command_artifact_ids,
        "missing_execution_item_count": len(missing_execution_items),
        "missing_execution_items": missing_execution_items,
        "artifact_execution": artifact_execution,
        "automation_blockers": [
            f"runtime_capture_command_option_missing:{request_path}::{artifact_id}"
            for artifact_id in missing_capture_command_artifact_ids
        ],
    }


def all_request_runtime_capture_execution_coverage_manifest(
    all_capture_queue_manifest: list[JSONDict],
    launch_card_library_summary: JSONDict,
    selected_command_contract: JSONDict | None = None,
    command_contracts_by_request: dict[str, JSONDict] | None = None,
    launch_card_dir_manifest: JSONDict | None = None,
) -> JSONDict:
    runtime_commands_by_key = bound_runtime_capture_commands_from_launch_card_library(launch_card_library_summary)
    if isinstance(command_contracts_by_request, dict):
        for command_contract in command_contracts_by_request.values():
            add_bound_runtime_capture_commands_from_contract(runtime_commands_by_key, command_contract)
    add_bound_runtime_capture_commands_from_contract(runtime_commands_by_key, selected_command_contract)
    requests = [
        queue_entry_runtime_capture_execution_coverage(
            item,
            runtime_commands_by_key=runtime_commands_by_key,
        )
        for item in all_capture_queue_manifest
        if isinstance(item, dict)
    ]
    summary = all_request_runtime_capture_execution_coverage_summary_from_requests(
        requests,
        expected_request_count=len(all_capture_queue_manifest),
    )
    return {
        "ready": summary["manual_operator_capture_ready"],
        "automated_capture_ready": summary["automated_capture_ready"],
        "manual_operator_capture_ready": summary["manual_operator_capture_ready"],
        "request_count": summary["request_count"],
        "expected_request_count": summary["expected_request_count"],
        "manual_operator_capture_ready_count": summary["manual_operator_capture_ready_count"],
        "automated_capture_ready_count": summary["automated_capture_ready_count"],
        "pending_artifact_count": summary["pending_artifact_count"],
        "artifact_execution_count": summary["artifact_execution_count"],
        "capture_command_option_count": summary["capture_command_option_count"],
        "operator_command_option_count": summary["operator_command_option_count"],
        "metadata_command_option_count": summary["metadata_command_option_count"],
        "artifacts_with_capture_command_count": summary["artifacts_with_capture_command_count"],
        "manual_capture_required_count": summary["manual_capture_required_count"],
        "missing_capture_command_count": summary["missing_capture_command_count"],
        "missing_execution_item_count": summary["missing_execution_item_count"],
        "requests_with_missing_execution_items": summary["requests_with_missing_execution_items"],
        "requests": requests,
        "launch_card_directory": launch_card_dir_manifest or {},
        "automation_blockers": summary["automation_blockers"],
    }


def all_request_runtime_capture_execution_coverage_summary_from_requests(
    requests: list[JSONDict],
    *,
    expected_request_count: int,
) -> JSONDict:
    request_count = len(requests)
    manual_ready_count = sum(1 for item in requests if item.get("manual_operator_capture_ready") is True)
    automated_ready_count = sum(1 for item in requests if item.get("automated_capture_ready") is True)
    missing_request_count = max(expected_request_count - request_count, 0)
    automation_blockers = sorted(
        {
            blocker
            for request in requests
            for blocker in list_of_strings(request.get("automation_blockers"))
        }
    )
    return {
        "ready": request_count > 0 and request_count == expected_request_count and manual_ready_count == request_count,
        "automated_capture_ready": request_count > 0 and request_count == expected_request_count and automated_ready_count == request_count,
        "manual_operator_capture_ready": request_count > 0 and request_count == expected_request_count and manual_ready_count == request_count,
        "request_count": request_count,
        "expected_request_count": expected_request_count,
        "missing_request_count": missing_request_count,
        "manual_operator_capture_ready_count": manual_ready_count,
        "automated_capture_ready_count": automated_ready_count,
        "pending_artifact_count": sum(int_count(item.get("pending_artifact_count")) for item in requests),
        "artifact_execution_count": sum(int_count(item.get("artifact_execution_count")) for item in requests),
        "capture_command_option_count": sum(int_count(item.get("capture_command_option_count")) for item in requests),
        "operator_command_option_count": sum(int_count(item.get("operator_command_option_count")) for item in requests),
        "metadata_command_option_count": sum(int_count(item.get("metadata_command_option_count")) for item in requests),
        "artifacts_with_capture_command_count": sum(int_count(item.get("artifacts_with_capture_command_count")) for item in requests),
        "runtime_command_contract_error_count": sum(int_count(item.get("runtime_command_contract_error_count")) for item in requests),
        "manual_capture_required_count": sum(int_count(item.get("manual_capture_required_count")) for item in requests),
        "missing_capture_command_count": sum(int_count(item.get("missing_capture_command_count")) for item in requests),
        "missing_execution_item_count": sum(int_count(item.get("missing_execution_item_count")) for item in requests),
        "requests_with_missing_execution_items": sum(1 for item in requests if int_count(item.get("missing_execution_item_count"))),
        "automation_blockers": automation_blockers,
    }


def all_request_runtime_capture_execution_coverage_summary(manifest: JSONDict) -> JSONDict:
    if not isinstance(manifest, dict) or not manifest:
        return {
            "ready": False,
            "automated_capture_ready": False,
            "manual_operator_capture_ready": False,
            "request_count": 0,
            "expected_request_count": 0,
            "missing_execution_item_count": 1,
            "automation_blockers": ["all_request_runtime_capture_execution_coverage_missing"],
        }
    requests = manifest.get("requests") if isinstance(manifest.get("requests"), list) else []
    summary = all_request_runtime_capture_execution_coverage_summary_from_requests(
        [item for item in requests if isinstance(item, dict)],
        expected_request_count=int_count(manifest.get("expected_request_count")),
    )
    summary["automation_blockers"] = list_of_strings(manifest.get("automation_blockers")) or summary["automation_blockers"]
    return summary

def _artifact_key_tuple(request_path: Any, artifact_id: Any, artifact_path: Any, receipt_path: Any) -> tuple[str, str, str, str] | None:
    if all(isinstance(value, str) and value for value in (request_path, artifact_id, artifact_path, receipt_path)):
        return (request_path, artifact_id, artifact_path, receipt_path)
    return None


def all_request_manual_capture_runbook_manifest(
    all_capture_queue_manifest: list[JSONDict],
    receipt_command_manifest: list[JSONDict],
    all_execution_coverage_manifest: JSONDict,
) -> JSONDict:
    receipt_by_key: dict[tuple[str, str, str, str], JSONDict] = {}
    for receipt in receipt_command_manifest:
        if not isinstance(receipt, dict):
            continue
        key = _artifact_key_tuple(
            receipt.get("request_path"),
            receipt.get("artifact_id"),
            receipt.get("artifact_path"),
            receipt.get("receipt_path"),
        )
        if key:
            receipt_by_key[key] = receipt

    queue_by_path = {
        str(item.get("request_path")): item
        for item in all_capture_queue_manifest
        if isinstance(item, dict) and isinstance(item.get("request_path"), str) and item.get("request_path")
    }
    execution_requests = all_execution_coverage_manifest.get("requests")
    if not isinstance(execution_requests, list):
        execution_requests = []

    requests: list[JSONDict] = []
    missing_items: list[str] = []
    manual_task_count = 0
    automated_task_count = 0
    validator_command_count = 0
    missing_receipt_command_count = 0
    missing_validator_command_count = 0
    missing_approval_command_count = 0
    missing_source_request_path_count = 0
    missing_prompt_set_path_count = 0
    missing_explicit_approval_count = 0
    missing_prompt_traffic_ack_count = 0

    for execution_request in execution_requests:
        if not isinstance(execution_request, dict):
            continue
        request_path = normalize_manifest_path(execution_request.get("request_path"))
        queue_item = queue_by_path.get(request_path or "", {})
        if not isinstance(queue_item, dict):
            queue_item = {}
            missing_items.append(f"capture_queue_request_missing:{request_path or 'unknown_request'}")
        approval_command = compact_approval_rebuild_command(queue_item.get("approval_rebuild_command"))
        if not approval_command:
            missing_approval_command_count += 1
            missing_items.append(f"approval_command_missing:{request_path or 'unknown_request'}")
        artifacts = execution_request.get("artifact_execution")
        if not isinstance(artifacts, list):
            artifacts = []
            missing_items.append(f"artifact_execution_missing:{request_path or 'unknown_request'}")

        manual_tasks: list[JSONDict] = []
        runtime_command_tasks: list[JSONDict] = []
        request_validator_count = 0
        request_missing_item_count = 0
        request_missing_receipt_command_count = 0
        request_missing_validator_command_count = 0
        request_missing_source_request_path_count = 0
        request_missing_prompt_set_path_count = 0
        request_missing_explicit_approval_count = 0
        request_missing_prompt_traffic_ack_count = 0

        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            artifact_id = artifact.get("artifact_id")
            artifact_path = normalize_manifest_path(artifact.get("artifact_path"))
            receipt_path = normalize_manifest_path(artifact.get("receipt_path"))
            key = _artifact_key_tuple(request_path, artifact_id, artifact_path, receipt_path)
            if key is None:
                request_missing_item_count += 1
                missing_items.append(f"manual_runbook_key_missing:{request_path or 'unknown_request'}::{artifact_id or 'unknown_artifact'}")
                continue
            receipt_row = receipt_by_key.get(key, {})
            if not receipt_row:
                request_missing_receipt_command_count += 1
                missing_items.append(f"receipt_command_missing:{request_path}::{artifact_id}")
            validator_commands = receipt_row.get("validator_commands") if isinstance(receipt_row.get("validator_commands"), list) else []
            if not validator_commands:
                request_missing_validator_command_count += 1
                missing_items.append(f"validator_command_missing:{request_path}::{artifact_id}")
            source_request_path = normalize_manifest_path(receipt_row.get("source_request_path"))
            prompt_set_path = normalize_manifest_path(receipt_row.get("prompt_set_path"))
            source_prompt_set_path = normalize_manifest_path(receipt_row.get("source_prompt_set_path"))
            records_approval_keys = list_of_strings(receipt_row.get("records_approval_keys"))
            requires_explicit_user_approval = receipt_row.get("requires_explicit_user_approval") is True
            approval_records_prompt_traffic = receipt_row.get("approval_records_prompt_traffic") is True
            may_send_prompt_traffic_after_approval = receipt_row.get("may_send_prompt_traffic_after_approval") is True
            if source_request_path != request_path:
                request_missing_source_request_path_count += 1
                missing_items.append(f"source_request_path_missing:{request_path}::{artifact_id}")
            if not prompt_set_path or source_prompt_set_path != prompt_set_path:
                request_missing_prompt_set_path_count += 1
                missing_items.append(f"prompt_set_path_missing:{request_path}::{artifact_id}")
            if not requires_explicit_user_approval:
                request_missing_explicit_approval_count += 1
                missing_items.append(f"explicit_approval_missing:{request_path}::{artifact_id}")
            if not approval_records_prompt_traffic or not may_send_prompt_traffic_after_approval:
                request_missing_prompt_traffic_ack_count += 1
                missing_items.append(f"prompt_traffic_ack_missing:{request_path}::{artifact_id}")
            task = {
                "request_name": execution_request.get("request_name"),
                "request_path": request_path,
                "source_request_path": source_request_path,
                "prompt_set_path": prompt_set_path,
                "source_prompt_set_path": source_prompt_set_path,
                "records_approval_keys": records_approval_keys,
                "requires_explicit_user_approval": requires_explicit_user_approval,
                "approval_records_prompt_traffic": approval_records_prompt_traffic,
                "may_send_prompt_traffic_after_approval": may_send_prompt_traffic_after_approval,
                "rank": int_count(execution_request.get("rank")),
                "artifact_id": artifact_id,
                "artifact_path": artifact_path,
                "receipt_path": receipt_path,
                "receipt_kind": artifact.get("receipt_kind") or receipt_row.get("receipt_kind"),
                "capture_mode": artifact.get("capture_mode"),
                "status_after_approval": artifact.get("status_after_approval"),
                "fill_status_after_approval": artifact.get("fill_status_after_approval"),
                "receipt_ready": artifact.get("receipt_ready") is True,
                "has_runtime_capture_command": artifact.get("has_runtime_capture_command") is True,
                "operator_command_option_count": int_count(artifact.get("operator_command_option_count")),
                "metadata_command_option_count": int_count(artifact.get("metadata_command_option_count")),
                "capture_command_option_count": int_count(artifact.get("capture_command_option_count")),
                "receipt_command_manifest_key": receipt_row.get("command_manifest_key"),
                "validator_command_count": len(validator_commands),
                "validator_commands": validator_commands,
                "blockers_after_approval": list_of_strings(artifact.get("blockers_after_approval") or receipt_row.get("blockers_after_approval")),
                "operator_action": "capture_runtime_artifact_then_fill_receipt",
                "next_action": "After explicit approval, capture this artifact, fill the receipt, then run the listed validators.",
            }
            request_validator_count += len(validator_commands)
            if artifact.get("has_runtime_capture_command") is True:
                runtime_command_tasks.append(task)
                automated_task_count += 1
            else:
                manual_tasks.append(task)
                manual_task_count += 1

        missing_receipt_command_count += request_missing_receipt_command_count
        missing_validator_command_count += request_missing_validator_command_count
        missing_source_request_path_count += request_missing_source_request_path_count
        missing_prompt_set_path_count += request_missing_prompt_set_path_count
        missing_explicit_approval_count += request_missing_explicit_approval_count
        missing_prompt_traffic_ack_count += request_missing_prompt_traffic_ack_count
        validator_command_count += request_validator_count
        requests.append(
            {
                "rank": int_count(execution_request.get("rank")),
                "request_name": execution_request.get("request_name"),
                "request_path": request_path,
                "status": execution_request.get("status"),
                "missing_approval_keys": list_of_strings(queue_item.get("missing_approval_keys")),
                "approval_command": approval_command,
                "approval_command_present": bool(approval_command),
                "manual_operator_capture_ready": execution_request.get("manual_operator_capture_ready") is True,
                "automated_capture_ready": execution_request.get("automated_capture_ready") is True,
                "pending_artifact_count": int_count(execution_request.get("pending_artifact_count")),
                "manual_task_count": len(manual_tasks),
                "runtime_command_task_count": len(runtime_command_tasks),
                "validator_command_count": request_validator_count,
                "missing_item_count": request_missing_item_count + request_missing_receipt_command_count + request_missing_validator_command_count + request_missing_source_request_path_count + request_missing_prompt_set_path_count + request_missing_explicit_approval_count + request_missing_prompt_traffic_ack_count + (0 if approval_command else 1),
                "missing_receipt_command_count": request_missing_receipt_command_count,
                "missing_validator_command_count": request_missing_validator_command_count,
                "missing_source_request_path_count": request_missing_source_request_path_count,
                "missing_prompt_set_path_count": request_missing_prompt_set_path_count,
                "missing_explicit_approval_count": request_missing_explicit_approval_count,
                "missing_prompt_traffic_ack_count": request_missing_prompt_traffic_ack_count,
                "manual_tasks": manual_tasks,
                "runtime_command_tasks": runtime_command_tasks,
                "next_action": "Record approval metadata, run manual tasks that still lack runtime commands, fill receipts, and run validators.",
            }
        )

    request_count = len(requests)
    expected_request_count = len(all_capture_queue_manifest)
    request_ready_count = sum(1 for item in requests if item.get("manual_operator_capture_ready") is True)
    missing_items = sorted(set(missing_items))
    ready = request_count > 0 and request_count == expected_request_count and request_ready_count == request_count and not missing_items
    return {
        "ready": ready,
        "manual_operator_capture_ready": ready,
        "request_count": request_count,
        "expected_request_count": expected_request_count,
        "ready_request_count": request_ready_count,
        "manual_task_count": manual_task_count,
        "runtime_command_task_count": automated_task_count,
        "validator_command_count": validator_command_count,
        "missing_item_count": len(missing_items),
        "missing_items": missing_items,
        "missing_receipt_command_count": missing_receipt_command_count,
        "missing_validator_command_count": missing_validator_command_count,
        "missing_approval_command_count": missing_approval_command_count,
        "missing_source_request_path_count": missing_source_request_path_count,
        "missing_prompt_set_path_count": missing_prompt_set_path_count,
        "missing_explicit_approval_count": missing_explicit_approval_count,
        "missing_prompt_traffic_ack_count": missing_prompt_traffic_ack_count,
        "requests": sorted(requests, key=lambda item: (int_count(item.get("rank")), str(item.get("request_name") or ""))),
        "safety_contract": [
            "manual capture runbook is metadata only",
            "manual capture runbook generation does not launch runtimes",
            "manual capture runbook generation does not run Docker",
            "manual capture runbook generation does not call endpoints",
            "manual capture runbook generation does not read private tokens",
            "manual capture runbook generation does not send prompt traffic",
        ],
    }


def all_request_manual_capture_runbook_summary(manifest: JSONDict) -> JSONDict:
    if not isinstance(manifest, dict) or not manifest:
        return {
            "ready": False,
            "manual_operator_capture_ready": False,
            "request_count": 0,
            "manual_task_count": 0,
            "runtime_command_task_count": 0,
            "validator_command_count": 0,
            "completion_gate_count": 0,
            "dependency_edge_count": 0,
            "missing_item_count": 1,
            "missing_receipt_command_count": 0,
            "missing_validator_command_count": 0,
            "missing_approval_command_count": 0,
            "missing_source_request_path_count": 0,
            "missing_prompt_set_path_count": 0,
            "missing_explicit_approval_count": 0,
            "missing_prompt_traffic_ack_count": 0,
        }
    requests = manifest.get("requests") if isinstance(manifest.get("requests"), list) else []
    return {
        "ready": manifest.get("ready") is True,
        "manual_operator_capture_ready": manifest.get("manual_operator_capture_ready") is True,
        "request_count": int_count(manifest.get("request_count")),
        "expected_request_count": int_count(manifest.get("expected_request_count")),
        "ready_request_count": int_count(manifest.get("ready_request_count")),
        "manual_task_count": int_count(manifest.get("manual_task_count")),
        "runtime_command_task_count": int_count(manifest.get("runtime_command_task_count")),
        "validator_command_count": int_count(manifest.get("validator_command_count")),
        "completion_gate_count": int_count(manifest.get("completion_gate_count")),
        "dependency_edge_count": int_count(manifest.get("dependency_edge_count")),
        "missing_item_count": int_count(manifest.get("missing_item_count")),
        "missing_receipt_command_count": int_count(manifest.get("missing_receipt_command_count")),
        "missing_validator_command_count": int_count(manifest.get("missing_validator_command_count")),
        "missing_approval_command_count": int_count(manifest.get("missing_approval_command_count")),
        "missing_source_request_path_count": int_count(manifest.get("missing_source_request_path_count")),
        "missing_prompt_set_path_count": int_count(manifest.get("missing_prompt_set_path_count")),
        "missing_explicit_approval_count": int_count(manifest.get("missing_explicit_approval_count")),
        "missing_prompt_traffic_ack_count": int_count(manifest.get("missing_prompt_traffic_ack_count")),
        "request_rows": len([item for item in requests if isinstance(item, dict)]),
    }


def _rows_by_request_path(rows: Any) -> dict[str, JSONDict]:
    if not isinstance(rows, list):
        return {}
    result: dict[str, JSONDict] = {}
    for row in rows:
        if isinstance(row, dict):
            request_path = normalize_manifest_path(row.get("request_path") or row.get("path"))
            if request_path:
                result[request_path] = row
    return result


def _artifact_rows_by_id(rows: Any) -> dict[str, JSONDict]:
    if not isinstance(rows, list):
        return {}
    result: dict[str, JSONDict] = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("artifact_id"), str) and row.get("artifact_id"):
            result[str(row["artifact_id"])] = row
    return result


def all_request_post_capture_intake_runbook_manifest(
    manual_runbook_manifest: JSONDict,
    capture_intake_summary: JSONDict,
) -> JSONDict:
    manual_requests = manual_runbook_manifest.get("requests") if isinstance(manual_runbook_manifest.get("requests"), list) else []
    fill_plans_by_path = _rows_by_request_path(capture_intake_summary.get("post_approval_capture_fill_plan_manifest"))
    transition_by_path = _rows_by_request_path(capture_intake_summary.get("approval_transition_preview_manifest"))
    intake_requests_by_path = _rows_by_request_path(capture_intake_summary.get("requests"))
    requests: list[JSONDict] = []
    missing_items: list[str] = []
    artifact_gate_count = 0
    ready_after_current_intake_count = 0
    missing_after_current_intake_count = 0
    validator_command_count = 0
    ready_to_update_bundle_count = 0
    missing_source_request_path_count = 0
    missing_prompt_set_path_count = 0
    missing_explicit_approval_count = 0
    missing_prompt_traffic_ack_count = 0

    for manual_request in manual_requests:
        if not isinstance(manual_request, dict):
            continue
        request_path = normalize_manifest_path(manual_request.get("request_path"))
        request_name = manual_request.get("request_name")
        fill_plan = fill_plans_by_path.get(request_path or "", {})
        transition = transition_by_path.get(request_path or "", {})
        intake_request = intake_requests_by_path.get(request_path or "", {})
        if not fill_plan:
            missing_items.append(f"post_capture_fill_plan_missing:{request_path or 'unknown_request'}")
        plan_steps = fill_plan.get("capture_fill_steps") if isinstance(fill_plan.get("capture_fill_steps"), list) else []
        if not transition:
            derived_next_step: JSONDict = {}
            for plan_step in plan_steps:
                if not isinstance(plan_step, dict):
                    continue
                step_after_approval = plan_step.get("step_after_approval")
                if isinstance(step_after_approval, dict) and step_after_approval:
                    derived_next_step = step_after_approval
                    break
            if derived_next_step:
                transition = {
                    "next_step_after_approval": derived_next_step,
                    "ready_to_update_bundle_after_approval": fill_plan.get("ready_to_update_bundle_after_approval") is True,
                }
            else:
                missing_items.append(f"approval_transition_preview_missing:{request_path or 'unknown_request'}")
        plan_steps_by_id = _artifact_rows_by_id(plan_steps)
        manual_tasks = []
        for group in ("manual_tasks", "runtime_command_tasks"):
            group_tasks = manual_request.get(group)
            if isinstance(group_tasks, list):
                manual_tasks.extend(item for item in group_tasks if isinstance(item, dict))
        artifact_gates: list[JSONDict] = []
        request_validator_count = 0
        request_ready_count = 0
        request_missing_count = 0
        request_missing_source_request_path_count = 0
        request_missing_prompt_set_path_count = 0
        request_missing_explicit_approval_count = 0
        request_missing_prompt_traffic_ack_count = 0
        for task in manual_tasks:
            artifact_id = task.get("artifact_id")
            step = plan_steps_by_id.get(str(artifact_id or ""), {})
            if not step:
                missing_items.append(f"post_capture_step_missing:{request_path or 'unknown_request'}::{artifact_id or 'unknown_artifact'}")
            validator_count = int_count(step.get("validator_command_count") if step else task.get("validator_command_count"))
            ready_after_current = step.get("ready_after_approval") is True
            receipt_ready = step.get("receipt_ready") is True
            fill_status = step.get("fill_status_after_approval") or task.get("fill_status_after_approval")
            blockers = list_of_strings(step.get("blockers_after_approval") or task.get("blockers_after_approval"))
            source_request_path = normalize_manifest_path(step.get("source_request_path") or task.get("source_request_path"))
            prompt_set_path = normalize_manifest_path(step.get("prompt_set_path") or task.get("prompt_set_path"))
            source_prompt_set_path = normalize_manifest_path(step.get("source_prompt_set_path") or task.get("source_prompt_set_path"))
            records_approval_keys = list_of_strings(task.get("records_approval_keys"))
            requires_explicit_user_approval = task.get("requires_explicit_user_approval") is True
            approval_records_prompt_traffic = task.get("approval_records_prompt_traffic") is True
            may_send_prompt_traffic_after_approval = task.get("may_send_prompt_traffic_after_approval") is True
            if source_request_path != request_path:
                request_missing_source_request_path_count += 1
                missing_items.append(f"post_capture_source_request_path_missing:{request_path or 'unknown_request'}::{artifact_id or 'unknown_artifact'}")
            if not prompt_set_path or source_prompt_set_path != prompt_set_path:
                request_missing_prompt_set_path_count += 1
                missing_items.append(f"post_capture_prompt_set_path_missing:{request_path or 'unknown_request'}::{artifact_id or 'unknown_artifact'}")
            if not requires_explicit_user_approval:
                request_missing_explicit_approval_count += 1
                missing_items.append(f"post_capture_explicit_approval_missing:{request_path or 'unknown_request'}::{artifact_id or 'unknown_artifact'}")
            if not approval_records_prompt_traffic or not may_send_prompt_traffic_after_approval:
                request_missing_prompt_traffic_ack_count += 1
                missing_items.append(f"post_capture_prompt_traffic_ack_missing:{request_path or 'unknown_request'}::{artifact_id or 'unknown_artifact'}")
            artifact_gates.append(
                {
                    "artifact_id": artifact_id,
                    "source_request_path": source_request_path,
                    "prompt_set_path": prompt_set_path,
                    "source_prompt_set_path": source_prompt_set_path,
                    "records_approval_keys": records_approval_keys,
                    "requires_explicit_user_approval": requires_explicit_user_approval,
                    "approval_records_prompt_traffic": approval_records_prompt_traffic,
                    "may_send_prompt_traffic_after_approval": may_send_prompt_traffic_after_approval,
                    "artifact_path": normalize_manifest_path(step.get("artifact_path") if step else task.get("artifact_path")),
                    "receipt_path": normalize_manifest_path(step.get("receipt_path") if step else task.get("receipt_path")),
                    "receipt_kind": step.get("receipt_kind") or task.get("receipt_kind"),
                    "receipt_ready": receipt_ready,
                    "ready_after_current_intake": ready_after_current,
                    "fill_status_after_approval": fill_status,
                    "validator_command_count": validator_count,
                    "blockers_after_approval": blockers,
                    "next_action": "After capture and receipt fill, rerun capture-result intake and require this gate to become ready.",
                }
            )
            request_validator_count += validator_count
            if ready_after_current:
                request_ready_count += 1
            else:
                request_missing_count += 1
        artifact_gate_count += len(artifact_gates)
        validator_command_count += request_validator_count
        ready_after_current_intake_count += request_ready_count
        missing_after_current_intake_count += request_missing_count
        missing_source_request_path_count += request_missing_source_request_path_count
        missing_prompt_set_path_count += request_missing_prompt_set_path_count
        missing_explicit_approval_count += request_missing_explicit_approval_count
        missing_prompt_traffic_ack_count += request_missing_prompt_traffic_ack_count
        ready_to_update = fill_plan.get("ready_to_update_bundle_after_approval") is True or intake_request.get("ready_to_update_bundle") is True
        if ready_to_update:
            ready_to_update_bundle_count += 1
        expected_steps = int_count(fill_plan.get("artifact_step_count")) if fill_plan else 0
        if expected_steps and expected_steps != len(artifact_gates):
            missing_items.append(f"post_capture_step_count_mismatch:{request_path or 'unknown_request'}")
        requests.append(
            {
                "rank": int_count(manual_request.get("rank") or fill_plan.get("rank")),
                "request_name": request_name or fill_plan.get("request_name"),
                "request_path": request_path,
                "manual_runbook_ready": manual_request.get("manual_operator_capture_ready") is True,
                "approval_transition_ready_for_operator": isinstance(transition.get("next_step_after_approval"), dict)
                and transition["next_step_after_approval"].get("status") == "ready_for_operator_capture",
                "approval_transition_ready_to_update_bundle": transition.get("ready_to_update_bundle_after_approval") is True,
                "artifact_gate_count": len(artifact_gates),
                "ready_after_current_intake_count": request_ready_count,
                "missing_after_current_intake_count": request_missing_count,
                "validator_command_count": request_validator_count,
                "missing_source_request_path_count": request_missing_source_request_path_count,
                "missing_prompt_set_path_count": request_missing_prompt_set_path_count,
                "missing_explicit_approval_count": request_missing_explicit_approval_count,
                "missing_prompt_traffic_ack_count": request_missing_prompt_traffic_ack_count,
                "ready_to_update_bundle_after_current_intake": ready_to_update,
                "next_operator_step": intake_request.get("next_operator_step") if isinstance(intake_request.get("next_operator_step"), dict) else {},
                "artifact_gates": artifact_gates,
                "post_capture_intake_command": [
                    "uv",
                    "run",
                    "--managed-python",
                    "--python",
                    "3.13",
                    "scripts/plan_phase3_capture_result_intake.py",
                    "--json",
                ],
                "next_action": "Rerun capture-result intake after all listed artifact receipts are ready; update bundles only when intake reports ready_to_update_bundle.",
            }
        )

    request_count = len(requests)
    expected_request_count = int_count(manual_runbook_manifest.get("request_count"))
    missing_items = sorted(set(missing_items))
    runbook_ready = (
        manual_runbook_manifest.get("ready") is True
        and capture_intake_summary.get("valid") is True
        and request_count > 0
        and request_count == expected_request_count
        and artifact_gate_count == int_count(manual_runbook_manifest.get("manual_task_count")) + int_count(manual_runbook_manifest.get("runtime_command_task_count"))
        and not missing_items
    )
    return {
        "ready": runbook_ready,
        "metadata_only": True,
        "request_count": request_count,
        "expected_request_count": expected_request_count,
        "artifact_gate_count": artifact_gate_count,
        "ready_after_current_intake_count": ready_after_current_intake_count,
        "missing_after_current_intake_count": missing_after_current_intake_count,
        "validator_command_count": validator_command_count,
        "ready_to_update_bundle_count": ready_to_update_bundle_count,
        "phase4_candidate_count": int_count(capture_intake_summary.get("phase4_candidate_ready_count")),
        "live_spike_candidate_count": int_count(capture_intake_summary.get("live_spike_candidate_ready_count")),
        "missing_source_request_path_count": missing_source_request_path_count,
        "missing_prompt_set_path_count": missing_prompt_set_path_count,
        "missing_explicit_approval_count": missing_explicit_approval_count,
        "missing_prompt_traffic_ack_count": missing_prompt_traffic_ack_count,
        "missing_item_count": len(missing_items),
        "missing_items": missing_items,
        "intake_json_command": [
            "uv",
            "run",
            "--managed-python",
            "--python",
            "3.13",
            "scripts/plan_phase3_capture_result_intake.py",
            "--json",
        ],
        "intake_queue_command": [
            "uv",
            "run",
            "--managed-python",
            "--python",
            "3.13",
            "scripts/plan_phase3_capture_result_intake.py",
            "--show-queue",
        ],
        "requests": sorted(requests, key=lambda item: (int_count(item.get("rank")), str(item.get("request_name") or ""))),
        "safety_contract": [
            "post-capture intake runbook is metadata only",
            "post-capture intake runbook generation does not launch runtimes",
            "post-capture intake runbook generation does not run Docker",
            "post-capture intake runbook generation does not call endpoints",
            "post-capture intake runbook generation does not read private tokens",
            "post-capture intake runbook generation does not send prompt traffic",
            "post-capture intake runbook generation does not mutate runtime residency",
        ],
    }


def all_request_post_capture_intake_runbook_summary(manifest: JSONDict) -> JSONDict:
    if not isinstance(manifest, dict) or not manifest:
        return {
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
            "missing_item_count": 1,
        }
    return {
        "ready": manifest.get("ready") is True,
        "request_count": int_count(manifest.get("request_count")),
        "expected_request_count": int_count(manifest.get("expected_request_count")),
        "artifact_gate_count": int_count(manifest.get("artifact_gate_count")),
        "ready_after_current_intake_count": int_count(manifest.get("ready_after_current_intake_count")),
        "missing_after_current_intake_count": int_count(manifest.get("missing_after_current_intake_count")),
        "validator_command_count": int_count(manifest.get("validator_command_count")),
        "ready_to_update_bundle_count": int_count(manifest.get("ready_to_update_bundle_count")),
        "phase4_candidate_count": int_count(manifest.get("phase4_candidate_count")),
        "live_spike_candidate_count": int_count(manifest.get("live_spike_candidate_count")),
        "missing_source_request_path_count": int_count(manifest.get("missing_source_request_path_count")),
        "missing_prompt_set_path_count": int_count(manifest.get("missing_prompt_set_path_count")),
        "missing_explicit_approval_count": int_count(manifest.get("missing_explicit_approval_count")),
        "missing_prompt_traffic_ack_count": int_count(manifest.get("missing_prompt_traffic_ack_count")),
        "missing_item_count": int_count(manifest.get("missing_item_count")),
    }

def selected_runbook_rows(manifest: JSONDict, recommended_request: JSONDict) -> tuple[str | None, list[JSONDict]]:
    request_path = normalize_manifest_path(
        recommended_request.get("request_path") or recommended_request.get("path")
    )
    requests = manifest.get("requests") if isinstance(manifest.get("requests"), list) else []
    if not request_path:
        return None, []
    return (
        request_path,
        [
            item
            for item in requests
            if isinstance(item, dict)
            and normalize_manifest_path(item.get("request_path") or item.get("path")) == request_path
        ],
    )


def recommended_manual_capture_runbook_manifest(
    all_runbook_manifest: JSONDict,
    recommended_request: JSONDict,
) -> JSONDict:
    request_path, rows = selected_runbook_rows(all_runbook_manifest, recommended_request)
    row = rows[0] if len(rows) == 1 else {}
    selection_missing_items: list[str] = []
    if not request_path:
        selection_missing_items.append("recommended_request_path_missing")
    if len(rows) != 1:
        selection_missing_items.append(
            f"recommended_manual_capture_runbook_selection_count:{len(rows)}"
        )
    row_missing_item_count = int_count(row.get("missing_item_count")) if isinstance(row, dict) else 0
    missing_item_count = row_missing_item_count + len(selection_missing_items)
    ready_request_count = 1 if isinstance(row, dict) and row.get("manual_operator_capture_ready") is True else 0
    return {
        "selected": True,
        "metadata_only": True,
        "source_manifest_kind": "all_request_manual_capture_runbook_manifest",
        "source_manifest_ready": all_runbook_manifest.get("ready") is True,
        "request_name": row.get("request_name") if isinstance(row, dict) else recommended_request.get("request_name"),
        "request_path": request_path,
        "ready": (
            all_runbook_manifest.get("ready") is True
            and len(rows) == 1
            and ready_request_count == 1
            and missing_item_count == 0
        ),
        "manual_operator_capture_ready": ready_request_count == 1 and missing_item_count == 0,
        "request_count": len(rows),
        "expected_request_count": 1,
        "source_request_count": int_count(all_runbook_manifest.get("request_count")),
        "ready_request_count": ready_request_count,
        "manual_task_count": int_count(row.get("manual_task_count")) if isinstance(row, dict) else 0,
        "runtime_command_task_count": int_count(row.get("runtime_command_task_count")) if isinstance(row, dict) else 0,
        "validator_command_count": int_count(row.get("validator_command_count")) if isinstance(row, dict) else 0,
        "missing_item_count": missing_item_count,
        "missing_items": selection_missing_items,
        "missing_receipt_command_count": int_count(row.get("missing_receipt_command_count")) if isinstance(row, dict) else 0,
        "missing_validator_command_count": int_count(row.get("missing_validator_command_count")) if isinstance(row, dict) else 0,
        "missing_approval_command_count": 0
        if isinstance(row, dict) and row.get("approval_command_present") is True
        else 1
        if isinstance(row, dict) and row
        else 0,
        "missing_source_request_path_count": int_count(row.get("missing_source_request_path_count")) if isinstance(row, dict) else 0,
        "missing_prompt_set_path_count": int_count(row.get("missing_prompt_set_path_count")) if isinstance(row, dict) else 0,
        "missing_explicit_approval_count": int_count(row.get("missing_explicit_approval_count")) if isinstance(row, dict) else 0,
        "missing_prompt_traffic_ack_count": int_count(row.get("missing_prompt_traffic_ack_count")) if isinstance(row, dict) else 0,
        "request": row,
        "requests": rows,
        "safety_contract": [
            "recommended manual capture runbook is metadata only",
            "recommended manual capture runbook generation does not launch runtimes",
            "recommended manual capture runbook generation does not run Docker",
            "recommended manual capture runbook generation does not call endpoints",
            "recommended manual capture runbook generation does not read private tokens",
            "recommended manual capture runbook generation does not send prompt traffic",
        ],
    }


def recommended_post_capture_intake_runbook_manifest(
    all_runbook_manifest: JSONDict,
    recommended_request: JSONDict,
) -> JSONDict:
    request_path, rows = selected_runbook_rows(all_runbook_manifest, recommended_request)
    row = rows[0] if len(rows) == 1 else {}
    selection_missing_items: list[str] = []
    if not request_path:
        selection_missing_items.append("recommended_request_path_missing")
    if len(rows) != 1:
        selection_missing_items.append(
            f"recommended_post_capture_intake_runbook_selection_count:{len(rows)}"
        )
    structural_missing_count = (
        int_count(row.get("missing_source_request_path_count"))
        + int_count(row.get("missing_prompt_set_path_count"))
        + int_count(row.get("missing_explicit_approval_count"))
        + int_count(row.get("missing_prompt_traffic_ack_count"))
        if isinstance(row, dict)
        else 0
    )
    missing_item_count = structural_missing_count + len(selection_missing_items)
    ready_to_update_bundle_count = (
        1
        if isinstance(row, dict) and row.get("ready_to_update_bundle_after_current_intake") is True
        else 0
    )
    return {
        "selected": True,
        "metadata_only": True,
        "source_manifest_kind": "all_request_post_capture_intake_runbook_manifest",
        "source_manifest_ready": all_runbook_manifest.get("ready") is True,
        "request_name": row.get("request_name") if isinstance(row, dict) else recommended_request.get("request_name"),
        "request_path": request_path,
        "ready": all_runbook_manifest.get("ready") is True and len(rows) == 1 and missing_item_count == 0,
        "request_count": len(rows),
        "expected_request_count": 1,
        "source_request_count": int_count(all_runbook_manifest.get("request_count")),
        "artifact_gate_count": int_count(row.get("artifact_gate_count")) if isinstance(row, dict) else 0,
        "ready_after_current_intake_count": int_count(row.get("ready_after_current_intake_count")) if isinstance(row, dict) else 0,
        "missing_after_current_intake_count": int_count(row.get("missing_after_current_intake_count")) if isinstance(row, dict) else 0,
        "validator_command_count": int_count(row.get("validator_command_count")) if isinstance(row, dict) else 0,
        "ready_to_update_bundle_count": ready_to_update_bundle_count,
        "missing_source_request_path_count": int_count(row.get("missing_source_request_path_count")) if isinstance(row, dict) else 0,
        "missing_prompt_set_path_count": int_count(row.get("missing_prompt_set_path_count")) if isinstance(row, dict) else 0,
        "missing_explicit_approval_count": int_count(row.get("missing_explicit_approval_count")) if isinstance(row, dict) else 0,
        "missing_prompt_traffic_ack_count": int_count(row.get("missing_prompt_traffic_ack_count")) if isinstance(row, dict) else 0,
        "missing_item_count": missing_item_count,
        "missing_items": selection_missing_items,
        "intake_json_command": all_runbook_manifest.get("intake_json_command") if isinstance(all_runbook_manifest.get("intake_json_command"), list) else [],
        "intake_queue_command": all_runbook_manifest.get("intake_queue_command") if isinstance(all_runbook_manifest.get("intake_queue_command"), list) else [],
        "request": row,
        "requests": rows,
        "safety_contract": [
            "recommended post-capture intake runbook is metadata only",
            "recommended post-capture intake runbook generation does not launch runtimes",
            "recommended post-capture intake runbook generation does not run Docker",
            "recommended post-capture intake runbook generation does not call endpoints",
            "recommended post-capture intake runbook generation does not read private tokens",
            "recommended post-capture intake runbook generation does not send prompt traffic",
            "recommended post-capture intake runbook generation does not mutate runtime residency",
        ],
    }

def recommended_runtime_capture_work_order_manifest(
    recommended_request: JSONDict,
    preflight_manifest: JSONDict,
    selected_manual_runbook: JSONDict,
    selected_post_capture_runbook: JSONDict,
) -> JSONDict:
    request_path = normalize_manifest_path(
        recommended_request.get("request_path") or recommended_request.get("path") or selected_manual_runbook.get("request_path")
    )
    manual_request = selected_manual_runbook.get("request") if isinstance(selected_manual_runbook.get("request"), dict) else {}
    post_request = selected_post_capture_runbook.get("request") if isinstance(selected_post_capture_runbook.get("request"), dict) else {}
    approval_command = compact_approval_rebuild_command(
        recommended_request.get("approval_rebuild_command") or manual_request.get("approval_command")
    )
    capture_tasks: list[JSONDict] = []
    for group_name, execution_mode in (
        ("manual_tasks", "manual_runtime_capture_required"),
        ("runtime_command_tasks", "runtime_command_capture_available"),
    ):
        for task in manual_request.get(group_name, []) if isinstance(manual_request.get(group_name), list) else []:
            if not isinstance(task, dict):
                continue
            capture_tasks.append(
                {
                    "step_rank": len(capture_tasks) + 1,
                    "step_id": f"capture_{task.get('artifact_id') or 'unknown_artifact'}",
                    "execution_mode": execution_mode,
                    "request_name": task.get("request_name") or manual_request.get("request_name"),
                    "request_path": normalize_manifest_path(task.get("request_path") or request_path),
                    "source_request_path": normalize_manifest_path(task.get("source_request_path")),
                    "prompt_set_path": normalize_manifest_path(task.get("prompt_set_path")),
                    "source_prompt_set_path": normalize_manifest_path(task.get("source_prompt_set_path")),
                    "artifact_id": task.get("artifact_id"),
                    "artifact_path": normalize_manifest_path(task.get("artifact_path")),
                    "receipt_path": normalize_manifest_path(task.get("receipt_path")),
                    "receipt_kind": task.get("receipt_kind"),
                    "capture_mode": task.get("capture_mode"),
                    "status_after_approval": task.get("status_after_approval"),
                    "fill_status_after_approval": task.get("fill_status_after_approval"),
                    "has_runtime_capture_command": task.get("has_runtime_capture_command") is True,
                    "requires_explicit_user_approval": task.get("requires_explicit_user_approval") is True,
                    "approval_records_prompt_traffic": task.get("approval_records_prompt_traffic") is True,
                    "may_send_prompt_traffic_after_approval": task.get("may_send_prompt_traffic_after_approval") is True,
                    "records_approval_keys": list_of_strings(task.get("records_approval_keys")),
                    "validator_command_count": int_count(task.get("validator_command_count")),
                    "validator_commands": task.get("validator_commands") if isinstance(task.get("validator_commands"), list) else [],
                    "blockers_after_approval": list_of_strings(task.get("blockers_after_approval")),
                    "operator_action": task.get("operator_action") or "capture_runtime_artifact_then_fill_receipt",
                    "next_action": task.get("next_action") or "After explicit approval, capture this artifact, fill the receipt, then run validators.",
                }
            )
    artifact_gates = [
        gate
        for gate in post_request.get("artifact_gates", [])
        if isinstance(gate, dict)
    ]
    intake_command = post_request.get("post_capture_intake_command")
    if not isinstance(intake_command, list):
        intake_command = selected_post_capture_runbook.get("intake_json_command") if isinstance(selected_post_capture_runbook.get("intake_json_command"), list) else []
    missing_items: list[str] = []
    if not request_path:
        missing_items.append("recommended_work_order_request_path_missing")
    if not approval_command.get("command"):
        missing_items.append("recommended_work_order_approval_command_missing")
    if not capture_tasks:
        missing_items.append("recommended_work_order_capture_steps_missing")
    if len(capture_tasks) != len(artifact_gates):
        missing_items.append("recommended_work_order_capture_gate_count_mismatch")
    if not intake_command:
        missing_items.append("recommended_work_order_intake_command_missing")
    validator_command_count = sum(int_count(step.get("validator_command_count")) for step in capture_tasks)
    completion_validation_step = recommended_completion_receipt_validation_step()
    post_capture_sequence = [
        {
            "sequence_rank": 1,
            "step_id": "record_runtime_approvals",
            "stage": "approval_metadata_rebuild",
            "command": approval_command.get("command") if isinstance(approval_command.get("command"), list) else [],
            "metadata_only": True,
            "requires_explicit_user_approval": True,
            "next_step_id": "capture_runtime_artifacts",
        },
        {
            "sequence_rank": 2,
            "step_id": "capture_runtime_artifacts",
            "stage": "approved_runtime_capture",
            "capture_step_count": len(capture_tasks),
            "manual_capture_step_count": int_count(selected_manual_runbook.get("manual_task_count")),
            "runtime_command_capture_step_count": int_count(selected_manual_runbook.get("runtime_command_task_count")),
            "metadata_only": False,
            "requires_explicit_user_approval": True,
            "may_send_prompt_traffic_after_approval": True,
            "next_step_id": "fill_completion_receipt",
        },
        {
            "sequence_rank": 3,
            "step_id": "fill_completion_receipt",
            "stage": "completion_receipt_fill",
            "receipt_path": "<operator-handoff-dir>/recommended-runtime-capture-completion-receipt.template.json",
            "metadata_only": True,
            "requires_completed_capture_artifacts": True,
            "next_step_id": "validate_completion_receipt",
        },
        {
            "sequence_rank": 4,
            "step_id": "validate_completion_receipt",
            "stage": "completion_receipt_validation",
            "command": completion_validation_step.get("command") if isinstance(completion_validation_step.get("command"), list) else [],
            "metadata_only": True,
            "requires_filled_completion_receipt_for_intake": True,
            "next_step_id": "run_capture_result_intake",
        },
        {
            "sequence_rank": 5,
            "step_id": "run_capture_result_intake",
            "stage": "capture_result_intake",
            "command": intake_command,
            "metadata_only": True,
            "requires_completed_capture_artifacts": True,
            "requires_completion_receipt_validation": True,
            "next_step_id": "bundle_promotion_decision",
        },
    ]
    ready = (
        selected_manual_runbook.get("ready") is True
        and selected_post_capture_runbook.get("ready") is True
        and not missing_items
    )
    return {
        "schema_version": "moe-phase3-recommended-runtime-capture-work-order-v1",
        "selected": True,
        "metadata_only": True,
        "launches_runtimes": False,
        "runs_docker": False,
        "sends_prompt_traffic": False,
        "reads_private_tokens": False,
        "mutates_runtime_residency": False,
        "execution_requires_explicit_approval": True,
        "ready": ready,
        "request_name": recommended_request.get("request_name") or manual_request.get("request_name"),
        "request_path": request_path,
        "status": recommended_request.get("status") or manual_request.get("status"),
        "queue_rank": int_count(recommended_request.get("queue_rank") or recommended_request.get("rank") or manual_request.get("rank")),
        "next_artifact_id": recommended_request.get("next_artifact_id"),
        "preflight_ready": preflight_manifest.get("ready") is True,
        "manual_runbook_ready": selected_manual_runbook.get("ready") is True,
        "post_capture_runbook_ready": selected_post_capture_runbook.get("ready") is True,
        "approval_step": {
            "stage": "approval_metadata_rebuild",
            "command_class": approval_command.get("command_class"),
            "command": approval_command.get("command") if isinstance(approval_command.get("command"), list) else [],
            "metadata_only": approval_command.get("metadata_only") is True,
            "records_approval_keys": list_of_strings(approval_command.get("records_approval_keys")),
            "requires_explicit_user_approval": approval_command.get("requires_explicit_user_approval") is True,
            "writes_request_path": normalize_manifest_path(approval_command.get("writes_request_path") or request_path),
        },
        "capture_step_count": len(capture_tasks),
        "manual_capture_step_count": int_count(selected_manual_runbook.get("manual_task_count")),
        "runtime_command_capture_step_count": int_count(selected_manual_runbook.get("runtime_command_task_count")),
        "artifact_gate_count": len(artifact_gates),
        "validator_command_count": validator_command_count,
        "missing_item_count": len(missing_items),
        "missing_items": missing_items,
        "capture_steps": capture_tasks,
        "post_capture_gates": artifact_gates,
        "post_capture_validation_step": completion_validation_step,
        "post_capture_sequence": post_capture_sequence,
        "post_capture_sequence_step_count": len(post_capture_sequence),
        "completion_receipt_validation_command_ready": bool(completion_validation_step.get("command")),
        "intake_step": {
            "stage": "capture_result_intake",
            "command": intake_command,
            "metadata_only": True,
            "requires_completed_capture_artifacts": True,
            "requires_completion_receipt_validation": True,
        },
        "completion_gates": [
            "explicit approval metadata is recorded before prompt traffic",
            "all capture steps write their artifact paths and receipt paths",
            "all validator commands pass for captured artifacts",
            "completion receipt validator passes before capture-result intake",
            "capture-result intake reports the selected request ready to update before bundle promotion",
        ],
        "safety_contract": [
            "recommended runtime capture work order is metadata only",
            "work order generation does not launch runtimes",
            "work order generation does not run Docker",
            "work order generation does not call endpoints",
            "work order generation does not read private tokens",
            "work order generation does not send prompt traffic",
            "work order generation does not mutate runtime residency",
        ],
    }


def recommended_runtime_capture_work_order_sequence_summary(manifest: JSONDict) -> JSONDict:
    sequence = manifest.get("post_capture_sequence") if isinstance(manifest.get("post_capture_sequence"), list) else []
    sequence_ids = [item.get("step_id") for item in sequence if isinstance(item, dict)]
    required_ids = [
        "record_runtime_approvals",
        "capture_runtime_artifacts",
        "fill_completion_receipt",
        "validate_completion_receipt",
        "run_capture_result_intake",
    ]
    return {
        "post_capture_sequence_step_count": len(sequence),
        "post_capture_sequence_ids": sequence_ids,
        "post_capture_sequence_ready": sequence_ids == required_ids,
        "completion_validation_before_intake": (
            "validate_completion_receipt" in sequence_ids
            and "run_capture_result_intake" in sequence_ids
            and sequence_ids.index("validate_completion_receipt") < sequence_ids.index("run_capture_result_intake")
        ),
    }


def recommended_runtime_capture_work_order_summary(manifest: JSONDict) -> JSONDict:
    if not isinstance(manifest, dict) or not manifest:
        return {
            "ready": False,
            "capture_step_count": 0,
            "artifact_gate_count": 0,
            "validator_command_count": 0,
            "missing_item_count": 1,
            "post_capture_sequence_step_count": 0,
            "post_capture_sequence_ready": False,
            "completion_validation_before_intake": False,
            "completion_validation_command_ready": False,
        }
    sequence_summary = recommended_runtime_capture_work_order_sequence_summary(manifest)
    return {
        "ready": manifest.get("ready") is True,
        "request_path": normalize_manifest_path(manifest.get("request_path")),
        "preflight_ready": manifest.get("preflight_ready") is True,
        "manual_runbook_ready": manifest.get("manual_runbook_ready") is True,
        "post_capture_runbook_ready": manifest.get("post_capture_runbook_ready") is True,
        "capture_step_count": int_count(manifest.get("capture_step_count")),
        "manual_capture_step_count": int_count(manifest.get("manual_capture_step_count")),
        "runtime_command_capture_step_count": int_count(manifest.get("runtime_command_capture_step_count")),
        "artifact_gate_count": int_count(manifest.get("artifact_gate_count")),
        "validator_command_count": int_count(manifest.get("validator_command_count")),
        "missing_item_count": int_count(manifest.get("missing_item_count")),
        "post_capture_sequence_step_count": sequence_summary["post_capture_sequence_step_count"],
        "post_capture_sequence_ready": sequence_summary["post_capture_sequence_ready"],
        "completion_validation_before_intake": sequence_summary["completion_validation_before_intake"],
        "completion_validation_command_ready": manifest.get("completion_receipt_validation_command_ready") is True,
        "approval_command_ready": isinstance(manifest.get("approval_step"), dict)
        and bool(manifest["approval_step"].get("command"))
        and manifest["approval_step"].get("requires_explicit_user_approval") is True,
        "intake_command_ready": isinstance(manifest.get("intake_step"), dict)
        and isinstance(manifest["intake_step"].get("command"), list)
        and bool(manifest["intake_step"].get("command")),
    }

def recommended_completion_receipt_validation_step(
    *,
    receipt_path: str = "<operator-handoff-dir>/recommended-runtime-capture-completion-receipt.template.json",
    work_order_path: str = "<operator-handoff-dir>/recommended-runtime-capture-work-order.json",
) -> JSONDict:
    return {
        "stage": "completion_receipt_validation",
        "command_class": "phase3_capture_completion_receipt_validation",
        "metadata_only": True,
        "launches_runtimes": False,
        "runs_docker": False,
        "sends_prompt_traffic": False,
        "reads_private_tokens": False,
        "mutates_runtime_residency": False,
        "requires_filled_completion_receipt_for_intake": True,
        "run_from": "repo_root",
        "receipt_path_argument": receipt_path,
        "work_order_path_argument": work_order_path,
        "command": [
            "uv",
            "run",
            "--managed-python",
            "--python",
            "3.13",
            "scripts/plan_phase3_capture_completion_receipt.py",
            receipt_path,
            "--work-order",
            work_order_path,
            "--json",
        ],
    }

def recommended_runtime_capture_completion_receipt_template_manifest(work_order: JSONDict) -> JSONDict:
    capture_steps = work_order.get("capture_steps") if isinstance(work_order.get("capture_steps"), list) else []
    receipt_rows: list[JSONDict] = []
    for step in capture_steps:
        if not isinstance(step, dict):
            continue
        receipt_rows.append(
            {
                "receipt_rank": len(receipt_rows) + 1,
                "capture_step_id": step.get("step_id"),
                "artifact_id": step.get("artifact_id"),
                "artifact_path": normalize_manifest_path(step.get("artifact_path")),
                "receipt_path": normalize_manifest_path(step.get("receipt_path")),
                "receipt_kind": step.get("receipt_kind"),
                "request_path": normalize_manifest_path(step.get("request_path") or work_order.get("request_path")),
                "source_request_path": normalize_manifest_path(step.get("source_request_path")),
                "prompt_set_path": normalize_manifest_path(step.get("prompt_set_path")),
                "source_prompt_set_path": normalize_manifest_path(step.get("source_prompt_set_path")),
                "records_approval_keys": list_of_strings(step.get("records_approval_keys")),
                "requires_explicit_user_approval": step.get("requires_explicit_user_approval") is True,
                "approval_records_prompt_traffic": step.get("approval_records_prompt_traffic") is True,
                "may_send_prompt_traffic_after_approval": step.get("may_send_prompt_traffic_after_approval") is True,
                "expected_validator_command_count": int_count(step.get("validator_command_count")),
                "expected_validator_commands": step.get("validator_commands") if isinstance(step.get("validator_commands"), list) else [],
                "capture_complete": False,
                "receipt_filled": False,
                "validator_passed": False,
                "ready_for_intake": False,
                "observed_at": None,
                "operator_notes": "",
            }
        )
    missing_items: list[str] = []
    request_path = normalize_manifest_path(work_order.get("request_path"))
    if not request_path:
        missing_items.append("completion_receipt_request_path_missing")
    if work_order.get("ready") is not True:
        missing_items.append("completion_receipt_work_order_not_ready")
    if not receipt_rows:
        missing_items.append("completion_receipt_rows_missing")
    if int_count(work_order.get("capture_step_count")) != len(receipt_rows):
        missing_items.append("completion_receipt_capture_step_count_mismatch")
    intake_step = work_order.get("intake_step") if isinstance(work_order.get("intake_step"), dict) else {}
    if not isinstance(intake_step.get("command"), list) or not intake_step.get("command"):
        missing_items.append("completion_receipt_intake_command_missing")
    validator_command_count = sum(int_count(row.get("expected_validator_command_count")) for row in receipt_rows)
    template_ready = not missing_items
    return {
        "schema_version": "moe-phase3-recommended-runtime-capture-completion-receipt-template-v1",
        "selected": True,
        "metadata_only": True,
        "launches_runtimes": False,
        "runs_docker": False,
        "sends_prompt_traffic": False,
        "reads_private_tokens": False,
        "mutates_runtime_residency": False,
        "template_ready": template_ready,
        "ready": template_ready,
        "receipt_complete": False,
        "ready_for_capture_result_intake": False,
        "execution_requires_explicit_approval": True,
        "request_name": work_order.get("request_name"),
        "request_path": request_path,
        "queue_rank": int_count(work_order.get("queue_rank")),
        "source_work_order_schema_version": work_order.get("schema_version"),
        "source_work_order_ready": work_order.get("ready") is True,
        "capture_receipt_count": len(receipt_rows),
        "expected_capture_step_count": int_count(work_order.get("capture_step_count")),
        "validator_command_count": validator_command_count,
        "missing_item_count": len(missing_items),
        "missing_items": missing_items,
        "approval_step": work_order.get("approval_step") if isinstance(work_order.get("approval_step"), dict) else {},
        "capture_receipts": receipt_rows,
        "intake_step": intake_step,
        "completion_receipt_validation_step": recommended_completion_receipt_validation_step(),
        "completion_gates": [
            "approval metadata was recorded before prompt traffic",
            "each artifact path exists and matches the work order",
            "each receipt path records source request, prompt set, approval, prompt-traffic acknowledgement, and validator results",
            "all validators pass before capture-result intake",
            "capture-result intake reports the selected request ready to update before bundle promotion",
        ],
        "safety_contract": [
            "completion receipt template is metadata only",
            "template generation does not launch runtimes",
            "template generation does not run Docker",
            "template generation does not call endpoints",
            "template generation does not read private tokens",
            "template generation does not send prompt traffic",
            "template generation does not mutate runtime residency",
        ],
    }


def recommended_runtime_capture_completion_receipt_template_summary(manifest: JSONDict) -> JSONDict:
    if not isinstance(manifest, dict) or not manifest:
        return {
            "ready": False,
            "template_ready": False,
            "receipt_complete": False,
            "ready_for_capture_result_intake": False,
            "capture_receipt_count": 0,
            "validator_command_count": 0,
            "missing_item_count": 1,
        }
    return {
        "ready": manifest.get("ready") is True,
        "template_ready": manifest.get("template_ready") is True,
        "receipt_complete": manifest.get("receipt_complete") is True,
        "ready_for_capture_result_intake": manifest.get("ready_for_capture_result_intake") is True,
        "request_path": normalize_manifest_path(manifest.get("request_path")),
        "capture_receipt_count": int_count(manifest.get("capture_receipt_count")),
        "expected_capture_step_count": int_count(manifest.get("expected_capture_step_count")),
        "validator_command_count": int_count(manifest.get("validator_command_count")),
        "missing_item_count": int_count(manifest.get("missing_item_count")),
        "validation_command_ready": isinstance(manifest.get("completion_receipt_validation_step"), dict)
        and isinstance(manifest["completion_receipt_validation_step"].get("command"), list)
        and "scripts/plan_phase3_capture_completion_receipt.py" in manifest["completion_receipt_validation_step"].get("command", [])
        and "--work-order" in manifest["completion_receipt_validation_step"].get("command", []),
    }

def next_unblocked_operator_handoff_summary(
    blocker_resolution_queue_summary: JSONDict,
    blocker_resolution_queue_manifest: JSONDict,
    recommended_work_order_summary: JSONDict,
    recommended_work_order_manifest: JSONDict,
    completion_receipt_summary: JSONDict,
    completion_receipt_manifest: JSONDict,
) -> JSONDict:
    next_package = (
        blocker_resolution_queue_manifest.get("next_unblocked_work_package")
        if isinstance(blocker_resolution_queue_manifest.get("next_unblocked_work_package"), dict)
        else {}
    )
    package_id = next_package.get("work_package_id") or blocker_resolution_queue_summary.get("next_unblocked_work_package_id")
    package_class = next_package.get("package_class") or blocker_resolution_queue_summary.get("next_unblocked_package_class")
    operator_stage = next_package.get("operator_stage") or blocker_resolution_queue_summary.get("next_unblocked_operator_stage")
    expected_next_artifact_id = "candidate_router_trace" if package_class == "policy_candidate_replay" else None
    work_order_next_artifact_id = recommended_work_order_manifest.get("next_artifact_id")
    work_order_request_path = normalize_manifest_path(
        recommended_work_order_manifest.get("request_path") or recommended_work_order_summary.get("request_path")
    )
    completion_receipt_request_path = normalize_manifest_path(
        completion_receipt_manifest.get("request_path") or completion_receipt_summary.get("request_path")
    )
    work_order_bound_to_receipt = bool(work_order_request_path) and work_order_request_path == completion_receipt_request_path
    work_order_advances_next_package = (
        operator_stage == "approved_runtime_capture"
        and recommended_work_order_summary.get("ready") is True
        and bool(work_order_next_artifact_id)
        and (expected_next_artifact_id is None or work_order_next_artifact_id == expected_next_artifact_id)
    )
    handoff_ready = (
        blocker_resolution_queue_summary.get("ready") is True
        and bool(package_id)
        and work_order_advances_next_package
        and recommended_work_order_summary.get("approval_command_ready") is True
        and completion_receipt_summary.get("template_ready") is True
        and completion_receipt_summary.get("validation_command_ready") is True
        and work_order_bound_to_receipt
    )
    return {
        "schema_version": "moe-phase3-next-unblocked-operator-handoff-v1",
        "handoff_ready": handoff_ready,
        "metadata_only": True,
        "launches_runtimes": False,
        "runs_docker": False,
        "sends_prompt_traffic": False,
        "reads_private_tokens": False,
        "mutates_runtime_residency": False,
        "next_unblocked_work_package_id": package_id,
        "next_unblocked_sequence_rank": next_package.get("sequence_rank")
        or blocker_resolution_queue_summary.get("next_unblocked_sequence_rank"),
        "next_unblocked_operator_stage": operator_stage,
        "next_unblocked_package_class": package_class,
        "next_unblocked_summary": next_package.get("summary") or blocker_resolution_queue_summary.get("next_unblocked_summary"),
        "next_unblocked_next_action": next_package.get("next_action")
        or blocker_resolution_queue_summary.get("next_unblocked_next_action"),
        "next_unblocked_row_count": int_count(
            next_package.get("row_count") or blocker_resolution_queue_summary.get("next_unblocked_row_count")
        ),
        "next_unblocked_validator_command_count": int_count(
            next_package.get("validator_command_count")
            or blocker_resolution_queue_summary.get("next_unblocked_validator_command_count")
        ),
        "next_unblocked_completion_gate_count": int_count(
            next_package.get("completion_gate_count")
            or blocker_resolution_queue_summary.get("next_unblocked_completion_gate_count")
        ),
        "work_order_artifact": "recommended-runtime-capture-work-order.json",
        "work_order_ready": recommended_work_order_summary.get("ready") is True,
        "work_order_request_path": work_order_request_path,
        "work_order_next_artifact_id": work_order_next_artifact_id,
        "work_order_capture_step_count": int_count(recommended_work_order_summary.get("capture_step_count")),
        "work_order_validator_command_count": int_count(recommended_work_order_summary.get("validator_command_count")),
        "work_order_approval_command_ready": recommended_work_order_summary.get("approval_command_ready") is True,
        "work_order_intake_command_ready": recommended_work_order_summary.get("intake_command_ready") is True,
        "work_order_completion_validation_command_ready": recommended_work_order_summary.get(
            "completion_validation_command_ready"
        )
        is True,
        "work_order_advances_next_package": work_order_advances_next_package,
        "completion_receipt_template_artifact": "recommended-runtime-capture-completion-receipt.template.json",
        "completion_receipt_template_ready": completion_receipt_summary.get("template_ready") is True,
        "completion_receipt_request_path": completion_receipt_request_path,
        "completion_receipt_validation_command_ready": completion_receipt_summary.get("validation_command_ready") is True,
        "work_order_bound_to_completion_receipt": work_order_bound_to_receipt,
        "operator_artifact_sequence": [
            {
                "sequence_rank": 1,
                "path": "blocker-resolution-queue.json",
                "kind": "next_unblocked_queue_package",
            },
            {
                "sequence_rank": 2,
                "path": "recommended-runtime-capture-work-order.json",
                "kind": "approved_runtime_capture_work_order",
            },
            {
                "sequence_rank": 3,
                "path": "recommended-runtime-capture-completion-receipt.template.json",
                "kind": "completion_receipt_template",
            },
        ],
        "granularity_notes": [
            "blocker resolution queue packages can include replay or intake validators outside the capture work order",
            "recommended runtime capture work order starts the next unblocked package with the selected request artifact",
            "completion receipt validation must pass before capture-result intake or promotion",
        ],
        "operator_next_action": (
            "Open recommended-runtime-capture-work-order.json, record the required approval metadata, capture the "
            "selected artifact, fill recommended-runtime-capture-completion-receipt.template.json, then run its "
            "validation command before intake."
            if handoff_ready
            else "Repair the next-unblocked queue package, recommended work order, or completion receipt template before runtime capture."
        ),
        "safety_contract": [
            "next unblocked operator handoff is metadata only",
            "handoff generation does not launch runtimes",
            "handoff generation does not run Docker",
            "handoff generation does not call endpoints",
            "handoff generation does not read private tokens",
            "handoff generation does not send prompt traffic",
            "handoff generation does not mutate runtime residency",
        ],
    }
def runtime_capture_recommendation(runtime_request_summary: JSONDict) -> JSONDict | None:
    value = runtime_request_summary.get("recommended_runtime_capture_request")
    if not isinstance(value, dict):
        return None
    recommendation: JSONDict = {
        "request_name": value.get("request_name"),
        "request_path": value.get("path"),
        "status": value.get("status"),
        "bundle_path": value.get("bundle_path"),
        "model_id": value.get("model_id"),
        "prompt_set_path": value.get("prompt_set_path"),
        "prompt_count": int_count(value.get("prompt_count")),
        "missing_approval_keys": list_of_strings(value.get("missing_approval_keys")),
        "pending_artifact_ids": list_of_strings(value.get("pending_artifact_ids")),
        "next_artifact_id": value.get("next_artifact_id"),
        "next_artifact_path": value.get("next_artifact_path"),
        "queue_rank": value.get("queue_rank"),
        "selection_rationale": value.get("selection_rationale"),
        "capture_sequence": compact_capture_sequence(value.get("capture_sequence")),
        "approval_rebuild_command": compact_approval_rebuild_command(value.get("approval_rebuild_command")),
    }
    command_manifest = recommended_capture_command_manifest(runtime_request_summary, recommendation)
    recommendation["validator_command_manifest"] = command_manifest
    recommendation["validator_command_count"] = sum(
        len(item.get("validator_commands", [])) for item in command_manifest
    )
    return recommendation


def compact_launch_card_library_entry(card: JSONDict, *, match_basis: str) -> JSONDict:
    task_count = int_count(card.get("task_count"))
    handoff_task_count = int_count(card.get("binding_handoff_task_count"))
    handoff_ready_count = int_count(card.get("binding_handoff_ready_count"))
    handoff_missing_field_count = int_count(card.get("binding_handoff_missing_field_count"))
    binding_handoff_ready = (
        task_count > 0
        and handoff_task_count == task_count
        and handoff_ready_count == task_count
        and handoff_missing_field_count == 0
    )
    return {
        "match_basis": match_basis,
        "request_name": card.get("request_name"),
        "request_path": normalize_manifest_path(card.get("request_path")),
        "launch_card_path": normalize_manifest_path(card.get("launch_card_path")),
        "template_ready": card.get("template_ready") is True,
        "planned_only": card.get("planned_only") is True,
        "model_plane_binding_ready": card.get("model_plane_binding_ready") is True,
        "binding_ready": card.get("binding_ready") is True,
        "runtime_capture_command_ready": card.get("runtime_capture_command_ready") is True,
        "task_count": task_count,
        "binding_handoff_ready": binding_handoff_ready,
        "binding_handoff_task_count": handoff_task_count,
        "binding_handoff_ready_count": handoff_ready_count,
        "binding_handoff_missing_field_count": handoff_missing_field_count,
        "unbound_task_count": int_count(card.get("unbound_task_count")),
        "missing_runtime_command_count": int_count(card.get("missing_runtime_command_count")),
        "blockers": list_of_strings(card.get("blockers")),
    }


def recommended_launch_card_library_entry(
    launch_card_library_summary: JSONDict,
    recommendation: JSONDict | None,
) -> JSONDict:
    if not isinstance(recommendation, dict):
        return {}
    cards = launch_card_library_summary.get("cards")
    if not isinstance(cards, list):
        return {}
    request_path = normalize_manifest_path(recommendation.get("request_path"))
    request_name = recommendation.get("request_name")
    for card in cards:
        if not isinstance(card, dict):
            continue
        if request_path and normalize_manifest_path(card.get("request_path")) == request_path:
            return compact_launch_card_library_entry(card, match_basis="request_path")
    for card in cards:
        if not isinstance(card, dict):
            continue
        if request_name and card.get("request_name") == request_name:
            return compact_launch_card_library_entry(card, match_basis="request_name")
    return {}


def post_approval_capture_fill_plans_by_request_path(capture_intake_summary: JSONDict) -> dict[str, JSONDict]:
    plans = capture_intake_summary.get("post_approval_capture_fill_plan_manifest")
    result: dict[str, JSONDict] = {}
    if not isinstance(plans, list):
        return result
    for plan in plans:
        if not isinstance(plan, dict):
            continue
        request_path = normalize_manifest_path(plan.get("request_path"))
        if request_path:
            result[request_path] = plan
    return result


def capture_sequence_by_step_id(item: JSONDict) -> dict[str, JSONDict]:
    sequence = compact_capture_sequence(item.get("capture_sequence"))
    return {
        str(step.get("id")): step
        for step in sequence
        if isinstance(step, dict) and isinstance(step.get("id"), str) and step.get("id")
    }


CAPTURE_STEP_ID_BY_ARTIFACT_ID = {
    "candidate_router_trace": "capture_candidate_router_trace",
    "managed_output_summary_fill": "fill_managed_output_summary",
    "dense_output_summary_fill": "fill_dense_output_summary",
}


RECEIPT_KIND_BY_ARTIFACT_ID = {
    "candidate_router_trace": "candidate_router_trace_capture_receipt",
    "managed_output_summary_fill": "embedded_output_capture_receipt",
    "dense_output_summary_fill": "embedded_output_capture_receipt",
}


def synthetic_receipt_path_for_artifact(artifact_id: str, artifact_path: str | None, approval_command: JSONDict) -> str | None:
    if artifact_id == "candidate_router_trace":
        command = approval_command.get("command") if isinstance(approval_command.get("command"), list) else []
        return normalize_manifest_path(command_option_value(command, "--candidate-trace-receipt-path"))
    return artifact_path


def synthetic_capture_fill_steps_from_queue_item(item: JSONDict, pending_ids: list[str], approval_command: JSONDict) -> list[JSONDict]:
    sequence_by_id = capture_sequence_by_step_id(item)
    request_path = normalize_manifest_path(item.get("path") or item.get("request_path"))
    prompt_set_path = normalize_manifest_path(item.get("prompt_set_path"))
    steps: list[JSONDict] = []
    for artifact_id in pending_ids:
        sequence_id = CAPTURE_STEP_ID_BY_ARTIFACT_ID.get(artifact_id, artifact_id)
        sequence_step = sequence_by_id.get(sequence_id, {})
        artifact_path = normalize_manifest_path(sequence_step.get("path"))
        if artifact_id == item.get("next_artifact_id") and not artifact_path:
            artifact_path = normalize_manifest_path(item.get("next_artifact_path"))
        receipt_path = synthetic_receipt_path_for_artifact(artifact_id, artifact_path, approval_command)
        validator_count = int_count(sequence_step.get("validator_command_count"))
        steps.append(
            {
                "artifact_id": artifact_id,
                "request_path": request_path,
                "prompt_set_path": prompt_set_path,
                "source_request_path": request_path,
                "source_prompt_set_path": prompt_set_path,
                "queue_step_id": sequence_id,
                "receipt_kind": RECEIPT_KIND_BY_ARTIFACT_ID.get(artifact_id, "runtime_capture_receipt"),
                "artifact_path": artifact_path,
                "receipt_path": receipt_path,
                "current_step": {
                    "id": sequence_id,
                    "stage": sequence_step.get("stage") or "runtime_capture",
                    "status": sequence_step.get("status"),
                    "path": artifact_path,
                    "approval_state": "missing" if item.get("status") == "approval_required" else "recorded",
                },
                "step_after_approval": {
                    "id": sequence_id,
                    "stage": sequence_step.get("stage") or "runtime_capture",
                    "status": "ready_for_operator_capture",
                    "path": artifact_path,
                    "approval_state": "recorded",
                },
                "fill_status_after_approval": "blocked_after_approval",
                "ready_after_approval": False,
                "receipt_ready": False,
                "validator_command_count": validator_count,
                "command_option_count": 0,
                "command_options": [],
                "blockers_after_approval": ["runtime_capture_artifact_not_filled"],
            }
        )
    return steps

def compact_capture_queue_item(item: JSONDict, fill_plan: JSONDict | None = None) -> JSONDict:
    fill_plan = fill_plan if isinstance(fill_plan, dict) else {}
    steps = fill_plan.get("capture_fill_steps") if isinstance(fill_plan.get("capture_fill_steps"), list) else []
    compact_steps = [compact_post_approval_capture_fill_step(step) for step in steps if isinstance(step, dict)]
    pending_ids = list_of_strings(item.get("pending_artifact_ids"))
    if not pending_ids:
        pending_ids = [str(step.get("artifact_id")) for step in compact_steps if step.get("artifact_id")]
    approval_rebuild_command = compact_approval_rebuild_command(item.get("approval_rebuild_command"))
    if pending_ids and not compact_steps:
        compact_steps = synthetic_capture_fill_steps_from_queue_item(item, pending_ids, approval_rebuild_command)
    validator_count = int_count(fill_plan.get("validator_command_count"))
    if validator_count == 0:
        validator_count = sum(int_count(step.get("validator_command_count")) for step in compact_steps)
    return {
        "rank": int_count(item.get("queue_rank") or item.get("rank")),
        "request_name": item.get("request_name"),
        "request_path": normalize_manifest_path(item.get("path") or item.get("request_path")),
        "status": item.get("status"),
        "selection_rationale": item.get("selection_rationale"),
        "missing_approval_keys": list_of_strings(item.get("missing_approval_keys")),
        "pending_artifact_ids": pending_ids,
        "pending_artifact_count": len(pending_ids),
        "next_artifact_id": item.get("next_artifact_id"),
        "next_artifact_path": normalize_manifest_path(item.get("next_artifact_path")),
        "capture_fill_step_count": len(compact_steps),
        "ready_after_approval_count": int_count(fill_plan.get("ready_after_approval_count")),
        "missing_after_approval_count": int_count(fill_plan.get("missing_after_approval_count")),
        "validator_command_count": validator_count,
        "ready_to_update_bundle_after_approval": fill_plan.get("ready_to_update_bundle_after_approval") is True,
        "prompt_set_path": normalize_manifest_path(item.get("prompt_set_path")),
        "approval_rebuild_command": approval_rebuild_command,
        "capture_sequence": compact_capture_sequence(item.get("capture_sequence")),
        "capture_fill_steps": compact_steps,
    }


def all_request_capture_queue_manifest(
    runtime_request_summary: JSONDict,
    capture_intake_summary: JSONDict,
) -> list[JSONDict]:
    queue = runtime_request_summary.get("approval_queue")
    if not isinstance(queue, list):
        return []
    fill_plans_by_path = post_approval_capture_fill_plans_by_request_path(capture_intake_summary)
    manifest: list[JSONDict] = []
    for item in queue:
        if not isinstance(item, dict):
            continue
        request_path = normalize_manifest_path(item.get("path") or item.get("request_path"))
        manifest.append(compact_capture_queue_item(item, fill_plans_by_path.get(request_path or "")))
    return sorted(manifest, key=lambda item: (int_count(item.get("rank")), str(item.get("request_name") or "")))


def all_request_capture_queue_summary(manifest: list[JSONDict], *, expected_request_count: int) -> JSONDict:
    request_count = len(manifest)
    pending_artifact_count = sum(int_count(item.get("pending_artifact_count")) for item in manifest)
    fill_step_count = sum(int_count(item.get("capture_fill_step_count")) for item in manifest)
    validator_command_count = sum(int_count(item.get("validator_command_count")) for item in manifest)
    requests_with_complete_fill_plan = sum(
        1
        for item in manifest
        if int_count(item.get("pending_artifact_count")) > 0
        and int_count(item.get("pending_artifact_count")) == int_count(item.get("capture_fill_step_count"))
    )
    return {
        "ready": request_count > 0 and request_count == expected_request_count and requests_with_complete_fill_plan == request_count,
        "request_count": request_count,
        "expected_request_count": expected_request_count,
        "requests_with_complete_fill_plan": requests_with_complete_fill_plan,
        "pending_artifact_count": pending_artifact_count,
        "capture_fill_step_count": fill_step_count,
        "validator_command_count": validator_command_count,
    }


def build_promotion_checklist(
    *,
    real_matrix_summary: JSONDict,
    handoff_summary: JSONDict,
    runtime_request_summary: JSONDict,
    capture_intake_summary: JSONDict,
    fallback_summary: JSONDict,
    live_proof_summary: JSONDict,
    decision_summary: JSONDict,
    policy_candidate_trace_plan: JSONDict | None = None,
    dense_fallback_capture_plan: JSONDict | None = None,
    live_capability_proof_handoff: JSONDict | None = None,
    launch_card_library_summary: JSONDict | None = None,
) -> list[JSONDict]:
    bundle_count = int_count(real_matrix_summary.get("bundle_count"))
    real_matrix_ready = (
        bool(real_matrix_summary.get("valid"))
        and bundle_count > 0
        and real_matrix_summary.get("real_model_pair_ready_count") == bundle_count
        and real_matrix_summary.get("replay_valid_count") == bundle_count
    )
    handoff_ready = bool(handoff_summary.get("valid")) and bool(
        handoff_summary.get("all_handoff_scaffolds_ready")
    )
    request_count = int_count(runtime_request_summary.get("request_count"))
    ready_for_operator_count = int_count(runtime_request_summary.get("ready_for_operator_capture_count"))
    capture_complete_count = int_count(runtime_request_summary.get("capture_complete_count"))
    ready_to_update_bundle_count = int_count(capture_intake_summary.get("ready_to_update_bundle_count"))
    phase4_candidate_count = int_count(capture_intake_summary.get("phase4_candidate_ready_count"))
    request_drift_free_count = int_count(capture_intake_summary.get("request_drift_free_count"))
    request_drifted_count = int_count(capture_intake_summary.get("request_drifted_count"))
    request_audit_valid_count = int_count(capture_intake_summary.get("request_audit_valid_count"))
    request_ready_for_operator_flag_count = int_count(capture_intake_summary.get("request_ready_for_operator_capture_flag_count"))
    approved_but_capture_incomplete_count = int_count(capture_intake_summary.get("approved_but_capture_incomplete_request_count"))
    runtime_approval_missing_count = int_count(capture_intake_summary.get("runtime_approval_missing_request_count"))
    approval_rebuild_command_available_count = int_count(capture_intake_summary.get("approval_rebuild_command_available_request_count"))
    approval_rebuild_command_manifest = capture_intake_summary.get("approval_rebuild_command_manifest")
    approval_rebuild_command_manifest_count = len(approval_rebuild_command_manifest) if isinstance(approval_rebuild_command_manifest, list) else 0
    approved_runtime_capture_pending_count = int_count(capture_intake_summary.get("approved_runtime_capture_pending_request_count"))
    capture_receipt_required_count = int_count(capture_intake_summary.get("capture_receipt_required_count"))
    capture_receipt_ready_count = int_count(capture_intake_summary.get("capture_receipt_ready_count"))
    capture_receipt_missing_request_count = int_count(capture_intake_summary.get("capture_receipt_missing_request_count"))
    output_receipt_binding_ready_count = int_count(capture_intake_summary.get("output_receipt_binding_ready_count"))
    receipt_gate_coverage = compact_receipt_gate_coverage(capture_intake_summary.get("receipt_gate_coverage"))
    receipt_fill_summary = capture_intake_summary.get("receipt_fill_manifest_summary")
    if not isinstance(receipt_fill_summary, dict):
        receipt_fill_summary = {}
    receipt_fill_entry_count = int_count(capture_intake_summary.get("receipt_fill_entry_count") or receipt_fill_summary.get("entry_count"))
    receipt_fill_ready_count = int_count(capture_intake_summary.get("receipt_fill_ready_count") or receipt_fill_summary.get("ready_count"))
    receipt_fill_missing_count = int_count(capture_intake_summary.get("receipt_fill_missing_count") or receipt_fill_summary.get("missing_count"))
    receipt_fill_approval_missing_count = int_count(capture_intake_summary.get("receipt_fill_approval_missing_count") or receipt_fill_summary.get("approval_missing_count"))
    intake_transition_preview = compact_approval_transition_preview(
        capture_intake_summary.get("recommended_approval_transition_preview")
    )
    intake_capture_fill_plan = compact_post_approval_capture_fill_plan(
        capture_intake_summary.get("recommended_post_approval_capture_fill_plan")
    )
    next_step_counts = capture_intake_summary.get("next_operator_step_counts")
    if not isinstance(next_step_counts, dict):
        next_step_counts = {}
    recommendation = runtime_capture_recommendation(runtime_request_summary)
    post_approval_preview = compact_post_approval_preview(
        runtime_request_summary.get("recommended_post_approval_preview")
    )
    approval_manifest_parity = approval_manifest_parity_summary(runtime_request_summary, capture_intake_summary)
    receipt_requirement_parity = receipt_requirement_parity_summary(runtime_request_summary, capture_intake_summary)
    operator_queue_parity = operator_queue_parity_summary(runtime_request_summary, capture_intake_summary)
    post_approval_capture_fill_parity = post_approval_capture_fill_parity_summary(runtime_request_summary, capture_intake_summary)
    all_post_approval_capture_fill_parity = all_post_approval_capture_fill_parity_summary(runtime_request_summary, capture_intake_summary)
    all_capture_queue_manifest = all_request_capture_queue_manifest(runtime_request_summary, capture_intake_summary)
    all_capture_queue_summary = all_request_capture_queue_summary(all_capture_queue_manifest, expected_request_count=request_count)
    all_validator_command_manifest = all_request_validator_command_manifest(runtime_request_summary)
    all_validator_command_summary = all_request_validator_command_summary(
        all_validator_command_manifest,
        expected_request_count=request_count,
    )
    all_downstream_handoff_manifest = all_request_downstream_handoff_manifest(runtime_request_summary)
    all_downstream_handoff_summary = all_request_downstream_handoff_summary(
        all_downstream_handoff_manifest,
        expected_request_count=request_count,
    )
    all_receipt_fill_manifest = all_request_receipt_fill_manifest(capture_intake_summary, all_capture_queue_manifest)
    all_receipt_fill_summary = all_request_receipt_fill_summary(
        all_receipt_fill_manifest,
        expected_request_count=request_count,
    )
    receipt_validator_parity = all_request_receipt_validator_parity_summary(
        all_validator_command_manifest,
        all_receipt_fill_manifest,
    )
    receipt_command_manifest = all_request_receipt_fill_command_manifest(
        all_validator_command_manifest,
        all_receipt_fill_manifest,
    )
    receipt_command_summary = all_request_receipt_fill_command_summary(
        receipt_command_manifest,
        expected_request_count=request_count,
    )
    policy_trace_plan = policy_candidate_trace_plan if isinstance(policy_candidate_trace_plan, dict) else {}
    dense_capture_plan = dense_fallback_capture_plan if isinstance(dense_fallback_capture_plan, dict) else {}
    live_proof_handoff = (
        live_capability_proof_handoff
        if isinstance(live_capability_proof_handoff, dict)
        else {}
    )
    policy_trace_handoff_ready = policy_candidate_trace_handoff_ready(policy_trace_plan)
    dense_capture_handoff_ready = dense_fallback_capture_handoff_ready(dense_capture_plan)
    live_proof_handoff_ready = live_capability_proof_handoff_ready(live_proof_handoff)
    launch_card_library = launch_card_library_summary if isinstance(launch_card_library_summary, dict) else {}
    launch_card_library_ready = launch_card_library.get("library_ready") is True
    launch_card_binding_handoff_ready = launch_card_library.get("binding_handoff_ready") is True
    launch_card_execution_ready = launch_card_library.get("execution_ready") is True
    if launch_card_execution_ready:
        launch_card_library_status = "satisfied"
    elif launch_card_library_ready and launch_card_binding_handoff_ready:
        launch_card_library_status = "approval_required"
    else:
        launch_card_library_status = "blocked"
    policy_trace_ready_after_plan = policy_trace_plan.get("policy_candidate_ready_after_plan") is True
    if policy_trace_ready_after_plan:
        policy_trace_status = "satisfied"
    elif policy_trace_handoff_ready and policy_trace_plan.get("runtime_requires_explicit_approval") is True:
        policy_trace_status = "approval_required"
    elif policy_trace_handoff_ready:
        policy_trace_status = "pending_capture"
    else:
        policy_trace_status = "blocked"

    if dense_capture_plan.get("phase3_fallback_ready") is True:
        dense_capture_status = "satisfied"
    elif dense_capture_plan.get("metadata_ready_to_build_comparison") is True:
        dense_capture_status = "ready_to_build_comparison"
    elif dense_capture_handoff_ready and dense_capture_plan.get("runtime_requires_explicit_approval") is True:
        dense_capture_status = "approval_required"
    elif dense_capture_handoff_ready:
        dense_capture_status = "pending_capture"
    else:
        dense_capture_status = "blocked"

    if live_proof_handoff.get("proof_ready") is True:
        live_handoff_status = "satisfied"
    elif live_proof_handoff_ready:
        live_handoff_status = "future_adapter_required"
    else:
        live_handoff_status = "blocked"

    runtime_status = "satisfied" if capture_complete_count > 0 else "blocked"
    if ready_for_operator_count > 0 and capture_complete_count == 0:
        runtime_status = "ready_for_operator_capture"
    elif request_count > 0 and capture_complete_count == 0:
        runtime_status = "approval_required"

    return [
        promotion_checklist_item(
            "repo_real_evidence_matrix",
            status="satisfied" if real_matrix_ready else "blocked",
            summary="Six-bundle real trace/inventory/replay matrix is the repo-level evidence base.",
            next_action="Repair the real-evidence matrix before relying on Phase 3 promotion state."
            if not real_matrix_ready
            else "Keep matrix evidence current when adding or replacing model targets.",
            evidence={
                "bundle_count": bundle_count,
                "real_model_pair_ready_count": real_matrix_summary.get("real_model_pair_ready_count"),
                "replay_valid_count": real_matrix_summary.get("replay_valid_count"),
                "phase4_ready_bundle_count": real_matrix_summary.get("phase4_ready_bundle_count"),
            },
        ),
        promotion_checklist_item(
            "handoff_scaffold_coverage",
            status="satisfied" if handoff_ready else "blocked",
            summary="Prompt sets, managed/dense output templates, runtime requests, and live-proof templates exist for every real bundle.",
            next_action="Build missing scaffold artifacts before approving runtime capture."
            if not handoff_ready
            else "Use the saved runtime-capture request audit before any approved capture.",
            evidence={
                "bundle_count": handoff_summary.get("bundle_count"),
                "handoff_scaffold_ready_count": handoff_summary.get("handoff_scaffold_ready_count"),
                "missing_artifact_counts": handoff_summary.get("missing_artifact_counts", {}),
            },
        ),
        promotion_checklist_item(
            "phase3_launch_card_library",
            status=launch_card_library_status,
            summary="Repo-level planned launch-card library must expose every runtime-capture command binding gap before approved capture.",
            next_action="Repair or regenerate planned launch-card templates before using Model Plane handoff."
            if not launch_card_library_ready
            else "Repair launch-card binding handoff rows before asking for runtime approval."
            if not launch_card_binding_handoff_ready
            else "Bind Model Plane callable ids or launch commands after explicit approval, then rerun the launch-card library planner."
            if not launch_card_execution_ready
            else "Use capture-result intake after approved runtime capture fills the named artifacts.",
            evidence={
                "library_ready": launch_card_library.get("library_ready"),
                "execution_ready": launch_card_library.get("execution_ready"),
                "card_count": launch_card_library.get("card_count"),
                "template_ready_count": launch_card_library.get("template_ready_count"),
                "model_plane_binding_ready_count": launch_card_library.get("model_plane_binding_ready_count"),
                "binding_ready_count": launch_card_library.get("binding_ready_count"),
                "runtime_capture_command_ready_count": launch_card_library.get("runtime_capture_command_ready_count"),
                "task_count": launch_card_library.get("task_count"),
                "missing_runtime_command_count": launch_card_library.get("missing_runtime_command_count"),
                "binding_handoff_ready": launch_card_library.get("binding_handoff_ready"),
                "binding_handoff_task_count": launch_card_library.get("binding_handoff_task_count"),
                "binding_handoff_ready_count": launch_card_library.get("binding_handoff_ready_count"),
                "binding_handoff_missing_field_count": launch_card_library.get("binding_handoff_missing_field_count"),
                "unbound_task_count": launch_card_library.get("unbound_task_count"),
                "model_plane_artifact_writer_contract_request_ready": launch_card_library.get("model_plane_artifact_writer_contract_request_ready"),
                "model_plane_artifact_writer_contract_request_task_count": launch_card_library.get("model_plane_artifact_writer_contract_request_task_count"),
                "saved_handoff_artifacts_ready": launch_card_library.get("saved_handoff_artifacts_ready"),
                "saved_handoff_artifact_missing_count": launch_card_library.get("saved_handoff_artifact_missing_count"),
                "saved_handoff_artifact_drifted_count": launch_card_library.get("saved_handoff_artifact_drifted_count"),
                "blockers": launch_card_library.get("blockers", []),
            },
        ),
        promotion_checklist_item(
            "approved_runtime_capture",
            status=runtime_status,
            summary="A request must produce candidate router traces plus managed and dense output summaries before bundle intake can update evidence.",
            next_action="Generate the pre-capture handoff with scripts/plan_phase3_runtime_capture_request.py --output-md, approve one request, then capture the named artifacts."
            if capture_complete_count == 0
            else "Run capture-result intake against the filled artifacts before rebuilding any bundle.",
            evidence={
                "request_count": request_count,
                "ready_for_operator_capture_count": ready_for_operator_count,
                "capture_complete_count": capture_complete_count,
                "next_operator_step_counts": next_step_counts,
                "capture_queue_summary": runtime_request_summary.get("capture_queue_summary", {}),
                "approval_rebuild_command_manifest_count": runtime_request_summary.get("approval_rebuild_command_manifest_count"),
                "recommended_runtime_capture_request": recommendation,
                "recommended_post_approval_preview": post_approval_preview,
            },
        ),
        promotion_checklist_item(
            "policy_candidate_trace_capture",
            status=policy_trace_status,
            summary="The policy-candidate blocker must resolve through the same candidate trace path named by the runtime-capture request.",
            next_action="Repair the policy-candidate trace handoff until it has a prompt set, candidate trace path, receipt path, and validator commands."
            if not policy_trace_handoff_ready
            else "Approve the recommended runtime-capture request, capture the candidate router trace, fill its receipt, then replay it for a policy candidate."
            if policy_trace_plan.get("runtime_requires_explicit_approval") is True
            else "Capture the candidate router trace and fill its receipt before rebuilding the bundle."
            if not policy_trace_plan.get("candidate_trace_exists")
            else "Fill or repair the candidate trace capture receipt before treating replay as promotion evidence."
            if not policy_trace_plan.get("capture_receipt_ready")
            else "Replay the candidate trace and rebuild the bundle only when replay produces a policy candidate or stronger no-go evidence.",
            evidence=policy_trace_plan,
        ),
        promotion_checklist_item(
            "capture_result_intake",
            status="satisfied" if ready_to_update_bundle_count > 0 else "blocked",
            summary="Filled capture artifacts and drift-free saved requests must pass intake before a bundle update or Phase 4 claim.",
            next_action="Regenerate drifted runtime-capture requests, then rerun capture-result intake."
            if request_drifted_count > 0
            else "Run scripts/plan_phase3_capture_result_intake.py --show-queue --output-md after capture artifacts are filled."
            if ready_to_update_bundle_count == 0
            else "Build the updated bundle from the intake-approved artifacts and re-run the packet.",
            evidence={
                "request_audit_valid_count": request_audit_valid_count,
                "request_drift_free_count": request_drift_free_count,
                "request_drifted_count": request_drifted_count,
                "request_ready_for_operator_capture_flag_count": request_ready_for_operator_flag_count,
                "approved_but_capture_incomplete_request_count": approved_but_capture_incomplete_count,
                "runtime_approval_missing_request_count": runtime_approval_missing_count,
                "approval_rebuild_command_available_request_count": approval_rebuild_command_available_count,
                "approval_rebuild_command_manifest_count": approval_rebuild_command_manifest_count,
                "approved_runtime_capture_pending_request_count": approved_runtime_capture_pending_count,
                "capture_receipt_required_count": capture_receipt_required_count,
                "capture_receipt_ready_count": capture_receipt_ready_count,
                "capture_receipt_missing_request_count": capture_receipt_missing_request_count,
                "output_receipt_binding_ready_count": output_receipt_binding_ready_count,
                "receipt_gate_coverage": receipt_gate_coverage,
                "receipt_fill_entry_count": receipt_fill_entry_count,
                "receipt_fill_ready_count": receipt_fill_ready_count,
                "receipt_fill_missing_count": receipt_fill_missing_count,
                "receipt_fill_approval_missing_count": receipt_fill_approval_missing_count,
                "receipt_fill_manifest_summary": receipt_fill_summary,
                "approval_transition_preview_count": capture_intake_summary.get("approval_transition_preview_count"),
                "approval_transition_ready_for_operator_count": capture_intake_summary.get("approval_transition_ready_for_operator_count"),
                "approval_transition_ready_to_update_bundle_count": capture_intake_summary.get("approval_transition_ready_to_update_bundle_count"),
                "recommended_approval_transition_preview": intake_transition_preview,
                "post_approval_capture_fill_plan_count": capture_intake_summary.get("post_approval_capture_fill_plan_count"),
                "post_approval_capture_fill_artifact_step_count": capture_intake_summary.get("post_approval_capture_fill_artifact_step_count"),
                "post_approval_capture_fill_runtime_step_count": capture_intake_summary.get("post_approval_capture_fill_runtime_step_count"),
                "post_approval_capture_fill_ready_count": capture_intake_summary.get("post_approval_capture_fill_ready_count"),
                "post_approval_capture_fill_missing_count": capture_intake_summary.get("post_approval_capture_fill_missing_count"),
                "post_approval_capture_fill_validator_command_count": capture_intake_summary.get("post_approval_capture_fill_validator_command_count"),
                "post_approval_capture_fill_ready_to_update_bundle_count": capture_intake_summary.get("post_approval_capture_fill_ready_to_update_bundle_count"),
                "recommended_post_approval_capture_fill_plan": intake_capture_fill_plan,
                "ready_to_update_bundle_count": ready_to_update_bundle_count,
                "phase4_candidate_ready_count": phase4_candidate_count,
                "remaining_blockers_after_intake": capture_intake_summary.get("remaining_blockers_after_intake", []),
            },
        ),
        promotion_checklist_item(
            "approval_manifest_parity",
            status="satisfied" if approval_manifest_parity.get("ready") is True else "blocked",
            summary="Pre-capture and capture-result approval rebuild command manifests must describe the same request paths and commands.",
            next_action="Repair runtime-capture request or capture-result intake manifests until request paths, approval metadata, and commands match.",
            evidence=approval_manifest_parity,
        ),
        promotion_checklist_item(
            "receipt_requirement_parity",
            status="satisfied" if receipt_requirement_parity.get("ready") is True else "blocked",
            summary="Runtime-capture requests and capture-result intake must require the same receipt-bound artifacts.",
            next_action="Repair runtime-capture request or capture-result intake receipt paths until request/artifact ids and receipt targets match.",
            evidence=receipt_requirement_parity,
        ),
        promotion_checklist_item(
            "operator_queue_parity",
            status="satisfied" if operator_queue_parity.get("ready") is True else "blocked",
            summary="Runtime-capture queue ranking and capture-result intake next steps must agree per request.",
            next_action="Repair runtime-capture request audit or capture-result intake next-step computation until request names, statuses, and next step ids match.",
            evidence=operator_queue_parity,
        ),
        promotion_checklist_item(
            "post_approval_capture_fill_parity",
            status="satisfied" if post_approval_capture_fill_parity.get("ready") is True else "blocked",
            summary="Runtime post-approval pending artifacts and capture-result fill plan must agree before approved capture.",
            next_action="Repair runtime-capture request audit or capture-result intake fill planning until pending artifact ids, paths, statuses, and validator counts match.",
            evidence=post_approval_capture_fill_parity,
        ),
        promotion_checklist_item(
            "all_post_approval_capture_fill_parity",
            status="satisfied" if all_post_approval_capture_fill_parity.get("ready") is True else "blocked",
            summary="Every runtime-capture request must have the same post-approval artifact fill requirements as capture-result intake.",
            next_action="Repair runtime-capture request audit or capture-result intake fill planning until all queued request/artifact keys, paths, statuses, and validator counts match.",
            evidence=all_post_approval_capture_fill_parity,
        ),
        promotion_checklist_item(
            "all_request_capture_queue_manifest",
            status="satisfied" if all_capture_queue_summary.get("ready") is True else "blocked",
            summary="The evidence packet must expose a complete metadata-only queue for every approval-bound runtime-capture request.",
            next_action="Regenerate runtime-capture requests or capture-result fill plans until every queued request has matching capture-fill steps and validator counts.",
            evidence={
                **all_capture_queue_summary,
                "requests": all_capture_queue_manifest,
            },
        ),
        promotion_checklist_item(
            "all_request_validator_command_manifest",
            status="satisfied" if all_validator_command_summary.get("ready") is True else "blocked",
            summary="Every queued runtime-capture request must expose its artifact validator commands without opening the request JSON by hand.",
            next_action="Regenerate runtime-capture requests until every queued request has runtime-capture and future-adapter validator command entries.",
            evidence={
                **all_validator_command_summary,
                "artifact_commands": all_validator_command_manifest,
            },
        ),
        promotion_checklist_item(
            "all_request_downstream_handoff_manifest",
            status="satisfied" if all_downstream_handoff_summary.get("ready") is True else "blocked",
            summary="Every queued request must expose policy-candidate, dense-fallback, and live-proof handoff paths before capture work fans out.",
            next_action="Repair request scaffolds until every queued request has policy-candidate, dense-fallback, and live-proof handoffs with validator commands.",
            evidence={
                **all_downstream_handoff_summary,
                "requests": all_downstream_handoff_manifest,
            },
        ),
        promotion_checklist_item(
            "all_request_receipt_fill_manifest",
            status="satisfied" if all_receipt_fill_summary.get("ready") is True else "blocked",
            summary="Every queued runtime-capture artifact must expose its receipt-fill row, approval-after state, and validator count.",
            next_action="Approve the runtime-capture request, capture the named artifacts, fill every receipt row, then rerun capture-result intake.",
            evidence={
                **all_receipt_fill_summary,
                "receipt_fills": all_receipt_fill_manifest,
            },
        ),
        promotion_checklist_item(
            "all_request_receipt_validator_parity",
            status="satisfied" if receipt_validator_parity.get("ready") is True else "blocked",
            summary="Runtime-capture validator artifacts and capture-result receipt-fill rows must agree before operator capture relies on the ledger.",
            next_action="Repair runtime requests or capture-result intake until request/artifact keys, paths, statuses, and validator counts match.",
            evidence=receipt_validator_parity,
        ),
        promotion_checklist_item(
            "all_request_receipt_fill_command_manifest",
            status="satisfied" if receipt_command_summary.get("ready") is True else "blocked",
            summary="Every receipt-fill row must carry its receipt path plus the validator commands needed after approved capture.",
            next_action="Repair receipt-fill rows or runtime validator commands until every runtime artifact has a receipt path and command list.",
            evidence={
                **receipt_command_summary,
                "receipt_fill_commands": receipt_command_manifest,
            },
        ),
        promotion_checklist_item(
            "dense_fallback_capture",
            status=dense_capture_status,
            summary="Managed and dense/full-runtime saved outputs must use the same runtime-capture request paths before fallback comparison can clear quality bounds.",
            next_action="Repair the dense fallback handoff until it has prompt, managed output, dense output, comparison, and validator paths."
            if not dense_capture_handoff_ready
            else "Approve the recommended runtime-capture request, fill managed and dense output summaries, then build the fallback comparison."
            if dense_capture_plan.get("runtime_requires_explicit_approval") is True
            else "Fill managed and dense output summaries before building the fallback comparison."
            if not dense_capture_plan.get("metadata_ready_to_build_comparison")
            else "Build and validate the dense fallback comparison from the ready saved outputs.",
            evidence=dense_capture_plan,
        ),
        promotion_checklist_item(
            "dense_fallback_quality_bounds",
            status="satisfied" if fallback_summary.get("comparison_ready") is True else "blocked",
            summary="Managed and dense/full-runtime outputs need a comparison artifact for quality or behavior bounds.",
            next_action="Fill managed and dense output summaries for the same prompt set, then build and validate the dense fallback comparison.",
            evidence={
                "comparison_available": fallback_summary.get("comparison_available"),
                "comparison_ready": fallback_summary.get("comparison_ready"),
                "blocker": fallback_summary.get("blocker"),
            },
        ),
        promotion_checklist_item(
            "live_capability_proof_handoff",
            status=live_handoff_status,
            summary="The future live capability proof template must stay context-bound to the recommended runtime-capture bundle before any live adapter claim.",
            next_action="Repair or rebuild the live capability proof template until model, backend, prompt family, and source bundle match the recommended request."
            if not live_proof_handoff_ready
            else "Fill the context-bound live proof after approved residency observation, residency control, cleanup/restore, and artifact export exist."
            if live_proof_handoff.get("proof_ready") is not True
            else "Attach the validated live proof artifact and rerun the packet before promoting a live spike.",
            evidence=live_proof_handoff,
        ),
        promotion_checklist_item(
            "live_capability_proof",
            status="satisfied" if live_proof_summary.get("proof_ready") is True else "future_phase_4",
            summary="Residency observation/control plus cleanup/restore proof is required before live-spike readiness.",
            next_action="Fill the live-capability proof template only after an approved backend adapter can observe/control residency and prove cleanup.",
            evidence={
                "proof_available": live_proof_summary.get("proof_available"),
                "proof_ready": live_proof_summary.get("proof_ready"),
                "blockers": live_proof_summary.get("blockers", []),
            },
        ),
        promotion_checklist_item(
            "phase4_promotion_decision",
            status="satisfied" if decision_summary.get("ready_for_phase4_adapter_spike") is True else "blocked",
            summary="The matrix-aware go/no-go decision must clear before Phase 4 adapter work is promoted.",
            next_action="Clear blocked checklist items and rerun scripts/plan_phase3_evidence_packet.py.",
            evidence={
                "decision": decision_summary.get("decision"),
                "ready_for_phase4_adapter_spike": decision_summary.get("ready_for_phase4_adapter_spike"),
                "ready_for_live_spike": decision_summary.get("ready_for_live_spike"),
            },
        ),
    ]


@lru_cache(maxsize=1)
def build_repo_gate_summaries() -> tuple[JSONDict, JSONDict, JSONDict, JSONDict, JSONDict, JSONDict]:
    import plan_phase3_capture_result_intake
    import plan_phase3_handoff_coverage
    import plan_phase3_real_evidence_matrix
    import plan_phase3_reuse_evidence_capture
    import plan_phase3_runtime_capture_request

    return (
        plan_phase3_real_evidence_matrix.build_matrix(plan_phase3_real_evidence_matrix.DEFAULT_ROOT),
        plan_phase3_handoff_coverage.build_coverage(),
        plan_phase3_launch_card_library.build_library(),
        plan_phase3_runtime_capture_request.build_root_summary(),
        plan_phase3_capture_result_intake.build_root_summary(),
        plan_phase3_reuse_evidence_capture.build_root_summary(),
    )


def build_packet_summary(
    trace_path: Path,
    inventory_path: Path,
    policies_path: Path,
    managed_plan_path: Path,
    *,
    fallback_artifact_path: Path | None = None,
    live_proof_artifact_path: Path | None = None,
    runtime_capture_launch_card_path: Path | None = None,
    runtime_capture_launch_card_dir: Path | None = None,
    policy_candidate_trace_receipt_path: Path | None = None,
    include_repo_gates: bool = True,
    expected_model_id: str | None = None,
    expected_backend_family: str | None = None,
    expected_prompt_family: str | None = None,
    expected_source_bundle_path: Path | None = None,
) -> JSONDict:
    inventory_manifest = plan_expert_inventory.load_manifest(inventory_path)
    inventory_summary = plan_expert_inventory.build_summary(inventory_manifest, inventory_path)
    layout_summary = plan_expert_store_layout.build_layout_summary(inventory_path)
    join_summary = plan_trace_inventory_replay.build_join_summary(trace_path, inventory_path)
    pairing_summary = plan_real_model_trace_inventory_pairing.build_pairing_summary(trace_path, inventory_path)
    policies_plan = plan_baseline_replay_policies.load_policies(policies_path)
    policies_summary = plan_baseline_replay_policies.build_summary(policies_plan, policies_path)
    replay_summary = plan_baseline_policy_replay.build_replay_summary(trace_path, inventory_path, policies_path)
    fallback_summary = plan_dense_fallback_comparison.build_summary(fallback_artifact_path)
    expected_model_id = expected_model_id or inventory_manifest.get("model_id")
    expected_backend_family = expected_backend_family or inventory_manifest.get("backend_family")
    live_proof_summary = plan_phase3_live_capability_proof.build_summary(
        live_proof_artifact_path,
        expected_model_id=expected_model_id,
        expected_backend_family=expected_backend_family,
        expected_prompt_family=expected_prompt_family,
        expected_source_bundle_path=expected_source_bundle_path,
    )
    managed_plan = plan_phase3_runtime_actuator_design.load_plan(managed_plan_path)
    actuator_backend_family = str(expected_backend_family or plan_phase3_runtime_actuator_design.DEFAULT_BACKEND_FAMILY)
    runtime_actuator_design_summary = plan_phase3_runtime_actuator_design.build_summary(
        managed_plan,
        managed_plan_path,
        backend_family=actuator_backend_family,
    )
    runtime_actuator_spike_summary = plan_phase3_runtime_actuator_spike.build_summary(
        managed_plan,
        managed_plan_path,
        backend_family=actuator_backend_family,
    )
    if include_repo_gates:
        repo_gate_summaries = build_repo_gate_summaries()
        if len(repo_gate_summaries) == 6:
            (
                real_matrix_summary,
                handoff_summary,
                launch_card_library_summary,
                runtime_request_summary,
                capture_intake_summary,
                reuse_summary,
            ) = repo_gate_summaries
        elif len(repo_gate_summaries) == 5:
            (
                real_matrix_summary,
                handoff_summary,
                launch_card_library_summary,
                runtime_request_summary,
                capture_intake_summary,
            ) = repo_gate_summaries
            reuse_summary = reuse_summary_from_real_matrix(real_matrix_summary)
        else:
            raise ValueError("build_repo_gate_summaries must return five legacy summaries or six summaries including reuse evidence")
    else:
        real_matrix_summary = {"valid": True, "errors": []}
        handoff_summary = {"valid": True, "errors": []}
        launch_card_library_summary = {
            "valid": True,
            "errors": [],
            "library_ready": False,
            "execution_ready": False,
            "binding_handoff_ready": False,
            "binding_handoff_task_count": 0,
            "binding_handoff_ready_count": 0,
            "binding_handoff_missing_field_count": 0,
            "unbound_task_count": 0,
            "model_plane_artifact_writer_contract_request_ready": False,
            "model_plane_artifact_writer_contract_request_task_count": 0,
        }
        runtime_request_summary = {"valid": True, "errors": []}
        capture_intake_summary = {"valid": True, "errors": []}
        reuse_summary = reuse_summary_from_real_matrix(real_matrix_summary)
    decision_summary = plan_phase3_go_no_go.build_decision_summary(
        trace_path,
        inventory_path,
        policies_path,
        managed_plan_path,
        fallback_artifact_path=fallback_artifact_path,
        live_proof_artifact_path=live_proof_artifact_path,
        policy_candidate_trace_receipt_path=policy_candidate_trace_receipt_path,
        real_evidence_root=plan_phase3_go_no_go.DEFAULT_REAL_EVIDENCE_ROOT if include_repo_gates else None,
        expected_live_proof_model_id=expected_model_id,
        expected_live_proof_backend_family=expected_backend_family,
        expected_live_proof_prompt_family=expected_prompt_family,
        expected_live_proof_source_bundle_path=expected_source_bundle_path,
        repo_gate_summaries=(
            real_matrix_summary,
            handoff_summary,
            runtime_request_summary,
            capture_intake_summary,
            reuse_summary,
        ) if include_repo_gates and reuse_summary is not None else (
            real_matrix_summary,
            handoff_summary,
            runtime_request_summary,
            capture_intake_summary,
        ) if include_repo_gates else None,
    )

    policy_candidate_gate_evidence = decision_summary.get("policy_candidate_evidence", {})
    approval_manifest_parity = approval_manifest_parity_summary(runtime_request_summary, capture_intake_summary)
    receipt_requirement_parity = receipt_requirement_parity_summary(runtime_request_summary, capture_intake_summary)
    operator_queue_parity = operator_queue_parity_summary(runtime_request_summary, capture_intake_summary)
    post_approval_capture_fill_parity = post_approval_capture_fill_parity_summary(runtime_request_summary, capture_intake_summary)
    all_post_approval_capture_fill_parity = all_post_approval_capture_fill_parity_summary(runtime_request_summary, capture_intake_summary)
    all_capture_queue_manifest = all_request_capture_queue_manifest(runtime_request_summary, capture_intake_summary)
    expected_runtime_request_count = int_count(runtime_request_summary.get("request_count"))
    all_capture_queue_summary = all_request_capture_queue_summary(
        all_capture_queue_manifest,
        expected_request_count=expected_runtime_request_count,
    )
    all_validator_command_manifest = all_request_validator_command_manifest(runtime_request_summary)
    all_validator_command_summary = all_request_validator_command_summary(
        all_validator_command_manifest,
        expected_request_count=expected_runtime_request_count,
    )
    all_downstream_handoff_manifest = all_request_downstream_handoff_manifest(runtime_request_summary)
    all_downstream_handoff_summary = all_request_downstream_handoff_summary(
        all_downstream_handoff_manifest,
        expected_request_count=expected_runtime_request_count,
    )
    all_receipt_fill_manifest = all_request_receipt_fill_manifest(capture_intake_summary, all_capture_queue_manifest)
    all_receipt_fill_summary = all_request_receipt_fill_summary(
        all_receipt_fill_manifest,
        expected_request_count=expected_runtime_request_count,
    )
    receipt_validator_parity = all_request_receipt_validator_parity_summary(
        all_validator_command_manifest,
        all_receipt_fill_manifest,
    )
    receipt_command_manifest = all_request_receipt_fill_command_manifest(
        all_validator_command_manifest,
        all_receipt_fill_manifest,
    )
    receipt_command_summary = all_request_receipt_fill_command_summary(
        receipt_command_manifest,
        expected_request_count=expected_runtime_request_count,
    )
    recommended_runtime_capture = (
        runtime_capture_recommendation(runtime_request_summary) if include_repo_gates else None
    )
    recommended_launch_card_template = (
        recommended_launch_card_library_entry(launch_card_library_summary, recommended_runtime_capture)
        if include_repo_gates
        else {}
    )
    runtime_capture_launch_card_directory_manifest = runtime_capture_launch_card_dir_manifest(
        launch_card_library_summary,
        runtime_capture_launch_card_dir,
    )
    effective_runtime_capture_launch_card_path = runtime_capture_launch_card_path
    if effective_runtime_capture_launch_card_path is None and isinstance(recommended_runtime_capture, dict):
        effective_runtime_capture_launch_card_path = runtime_capture_launch_card_path_from_dir_manifest(
            runtime_capture_launch_card_directory_manifest,
            recommended_runtime_capture.get("request_path"),
        )
    recommended_post_approval_preview = (
        compact_post_approval_preview(runtime_request_summary.get("recommended_post_approval_preview"))
        if include_repo_gates
        else None
    )
    recommended_policy_candidate_trace_plan = (
        build_recommended_policy_candidate_trace_plan(recommended_runtime_capture)
        if include_repo_gates
        else None
    )
    recommended_dense_fallback_capture_plan = (
        build_recommended_dense_fallback_capture_plan(recommended_runtime_capture)
        if include_repo_gates
        else None
    )
    recommended_live_capability_proof_handoff = (
        build_recommended_live_capability_proof_handoff(recommended_runtime_capture)
        if include_repo_gates
        else None
    )
    recommended_runtime_capture_command_contract: JSONDict = {}
    if include_repo_gates and isinstance(recommended_runtime_capture, dict):
        request_path = repo_path(recommended_runtime_capture.get("request_path"))
        if request_path is not None and request_path.exists():
            if effective_runtime_capture_launch_card_path is not None:
                recommended_runtime_capture_command_contract = plan_phase3_runtime_capture_commands.build_summary(
                    request_path,
                    launch_card_path=effective_runtime_capture_launch_card_path,
                )
            else:
                recommended_runtime_capture_command_contract = plan_phase3_runtime_capture_commands.build_summary(request_path)
    launch_card_binding_summary = (
        recommended_runtime_capture_command_contract.get("launch_card_binding_summary")
        if isinstance(recommended_runtime_capture_command_contract.get("launch_card_binding_summary"), dict)
        else {}
    )
    recommended_runtime_capture_command_contract_summary: JSONDict = {
        "ready": recommended_runtime_capture_command_contract.get("command_contract_ready") is True,
        "runtime_capture_command_ready": recommended_runtime_capture_command_contract.get("runtime_capture_command_ready") is True,
        "planned_capture_count": int_count(recommended_runtime_capture_command_contract.get("planned_capture_count")),
        "runtime_command_option_count": int_count(recommended_runtime_capture_command_contract.get("runtime_command_option_count")),
        "missing_runtime_command_count": int_count(recommended_runtime_capture_command_contract.get("missing_runtime_command_count")),
        "missing_runtime_command_artifact_ids": list_of_strings(recommended_runtime_capture_command_contract.get("missing_runtime_command_artifact_ids")),
        "launch_card_binding_valid": launch_card_binding_summary.get("valid") is True,
        "launch_card_binding_ready": launch_card_binding_summary.get("binding_ready") is True,
        "launch_card_binding_path": launch_card_binding_summary.get("launch_card_path"),
        "launch_card_binding_bound_task_count": int_count(launch_card_binding_summary.get("bound_task_count")),
        "launch_card_binding_command_option_count": int_count(launch_card_binding_summary.get("command_option_count")),
        "launch_card_binding_missing_runtime_command_count": int_count(launch_card_binding_summary.get("missing_runtime_command_count")),
        "launch_card_binding_blockers": list_of_strings(launch_card_binding_summary.get("blockers")),
    }
    repo_bundle_count = int_count(real_matrix_summary.get("bundle_count")) if include_repo_gates else 0
    repo_real_model_pair_ready = (
        include_repo_gates
        and repo_bundle_count > 0
        and real_matrix_summary.get("real_model_pair_ready_count") == repo_bundle_count
    )
    local_real_model_pair_ready = pairing_summary.get("real_model_pair_ready") is True
    pairing_item_ready = repo_real_model_pair_ready or local_real_model_pair_ready
    pairing_decision_basis = (
        "repo_real_evidence_matrix"
        if repo_real_model_pair_ready
        else "default_trace_inventory_pairing"
    )

    errors: list[str] = []
    for label, summary in (
        ("expert inventory", inventory_summary),
        ("expert store layout", layout_summary),
        ("trace inventory join", join_summary),
        ("real-model trace inventory pairing", pairing_summary),
        ("baseline policies", policies_summary),
        ("baseline policy replay", replay_summary),
        ("dense fallback comparison", fallback_summary),
        ("live capability proof", live_proof_summary),
        ("phase 3 runtime-actuator spike", runtime_actuator_spike_summary),
        ("phase 3 real-evidence matrix", real_matrix_summary),
        ("phase 3 handoff coverage", handoff_summary),
        ("phase 3 launch-card library", launch_card_library_summary),
        ("phase 3 runtime-capture request audit", runtime_request_summary),
        ("phase 3 runtime-capture command contract", recommended_runtime_capture_command_contract or {"valid": True, "errors": []}),
        ("phase 3 capture-result intake", capture_intake_summary),
        ("phase 3 go/no-go", decision_summary),
    ):
        if not summary.get("valid"):
            errors.extend(f"{label}: {error}" for error in summary.get("errors", []))

    decision_gate_status = decision_summary.get("gate_status") if isinstance(decision_summary.get("gate_status"), dict) else {}

    items = [
        evidence_item(
            "expert_inventory_manifest",
            status=status_from_bool(bool(inventory_summary.get("valid"))),
            summary="Expert inventory manifest validates and exposes layer/expert/component byte estimates.",
            source="scripts/plan_expert_inventory.py",
            details={
                "expert_count": inventory_summary.get("expert_count"),
                "component_count": inventory_summary.get("component_count"),
                "total_estimated_residency_bytes": inventory_summary.get("total_estimated_residency_bytes"),
                "phase_3_replay_ready": inventory_summary.get("phase_3_replay_ready"),
            },
        ),
        evidence_item(
            "expert_store_layout_plan",
            status=status_from_bool(bool(layout_summary.get("valid")) and bool(layout_summary.get("can_stream_without_full_model_load"))),
            summary="Dry-run expert-store layout validates target files, source ranges, and disk estimate without writing tensors.",
            source="scripts/plan_expert_store_layout.py",
            details={
                "required_disk_bytes": layout_summary.get("required_disk_bytes"),
                "write_path_implemented": layout_summary.get("write_path_implemented"),
                "can_stream_without_full_model_load": layout_summary.get("can_stream_without_full_model_load"),
            },
        ),
        evidence_item(
            "trace_inventory_join",
            status=status_from_bool(bool(join_summary.get("valid")) and bool(join_summary.get("joined_route_count"))),
            summary="Semantic trace joins to inventory and reports routes, unique experts, bytes, and reuse distance.",
            source="scripts/plan_trace_inventory_replay.py",
            details={
                "joined_route_count": join_summary.get("joined_route_count"),
                "unique_expert_count": join_summary.get("unique_expert_count"),
                "reuse_distance": join_summary.get("reuse_distance"),
            },
        ),
        evidence_item(
            "real_model_trace_inventory_pairing",
            status=status_from_bool(pairing_item_ready, false_status="fixture_only"),
            summary="Pairing gate distinguishes the default fixture replay input from scanner-derived repo-wide real-model evidence.",
            source="scripts/plan_real_model_trace_inventory_pairing.py",
            details={
                "local_fixture_only_pair": pairing_summary.get("fixture_only_pair"),
                "local_real_model_pair_ready": local_real_model_pair_ready,
                "repo_real_model_pair_ready": repo_real_model_pair_ready,
                "repo_real_model_pair_ready_count": real_matrix_summary.get("real_model_pair_ready_count") if include_repo_gates else None,
                "repo_bundle_count": repo_bundle_count if include_repo_gates else None,
                "decision_pairing_basis": pairing_decision_basis,
                "blockers": pairing_summary.get("blockers", []),
            },
        ),
        evidence_item(
            "baseline_replay_policies",
            status=status_from_bool(bool(policies_summary.get("valid")) and not bool(policies_summary.get("may_mutate_runtime"))),
            summary="Baseline policy definitions validate and remain non-mutating.",
            source="scripts/plan_baseline_replay_policies.py",
            details={
                "policy_ids": policies_summary.get("policy_ids", []),
                "may_mutate_runtime": policies_summary.get("may_mutate_runtime"),
            },
        ),
        evidence_item(
            "baseline_policy_replay",
            status=status_from_bool(bool(replay_summary.get("phase_3_gate", {}).get("policy_replay_metrics_ready"))),
            summary="Policy replay reports miss rate, warm-hit rate, churn, fallback frequency, and rejection reasons.",
            source="scripts/plan_baseline_policy_replay.py",
            details={
                "candidate_policy_ids": replay_summary.get("candidate_policy_ids", []),
                "missing_for_phase_3_completion": replay_summary.get("phase_3_gate", {}).get("missing_for_phase_3_completion", []),
            },
        ),
        evidence_item(
            "dense_fallback_comparison",
            status=status_from_bool(bool(fallback_summary.get("comparison_ready")), false_status="blocked"),
            summary="Dense/full-runtime fallback comparison validates or records the exact missing-output blocker.",
            source="scripts/plan_dense_fallback_comparison.py",
            details={
                "comparison_available": fallback_summary.get("comparison_available"),
                "comparison_ready": fallback_summary.get("comparison_ready"),
                "blocker": fallback_summary.get("blocker"),
            },
        ),
        evidence_item(
            "live_capability_proof",
            status=status_from_bool(bool(live_proof_summary.get("proof_ready")), false_status="future_phase_4"),
            summary="Live residency observation/control, cleanup/restore, and artifact-export proof validates or remains a future live-spike gate.",
            source="scripts/plan_phase3_live_capability_proof.py",
            details={
                "proof_available": live_proof_summary.get("proof_available"),
                "proof_ready": live_proof_summary.get("proof_ready"),
                "blockers": live_proof_summary.get("blockers", []),
            },
        ),
        evidence_item(
            "runtime_actuator_spike_handoff",
            status=status_from_bool(runtime_actuator_spike_summary.get("spike_handoff_ready") is True, false_status="future_phase_4"),
            summary="Runtime actuator spike handoff names proof requirements, dependency order, and completion gates before live residency mutation.",
            source="scripts/plan_phase3_runtime_actuator_spike.py",
            details={
                "spike_handoff_ready": runtime_actuator_spike_summary.get("spike_handoff_ready"),
                "live_spike_ready": runtime_actuator_spike_summary.get("live_spike_ready"),
                "proof_requirement_count": runtime_actuator_spike_summary.get("proof_requirement_count"),
                "proof_artifact_count": runtime_actuator_spike_summary.get("proof_artifact_count"),
                "dependency_edge_count": runtime_actuator_spike_summary.get("dependency_edge_count"),
                "blocking_capability_count": runtime_actuator_spike_summary.get("blocking_capability_count"),
                "control_blocker_count": runtime_actuator_spike_summary.get("control_blocker_count"),
            },
        ),
        evidence_item(
            "phase3_go_no_go_decision",
            status=status_from_bool(bool(decision_summary.get("valid"))),
            summary="Decision packet aggregates Phase 3 evidence into a go/no-go outcome.",
            source="scripts/plan_phase3_go_no_go.py",
            details={
                "decision": decision_summary.get("decision"),
                "ready_for_phase4_adapter_spike": decision_summary.get("ready_for_phase4_adapter_spike"),
                "ready_for_live_spike": decision_summary.get("ready_for_live_spike"),
                "no_go_reasons": decision_summary.get("no_go_reasons", []),
                "gate_status": decision_summary.get("gate_status", {}),
                "policy_candidate_evidence": policy_candidate_gate_evidence,
            },
        ),
    ]

    if include_repo_gates:
        items.extend(
            [
                evidence_item(
                    "repo_real_evidence_matrix",
                    status=status_from_bool(
                        bool(real_matrix_summary.get("valid"))
                        and bool(real_matrix_summary.get("bundle_count"))
                        and real_matrix_summary.get("real_model_pair_ready_count") == real_matrix_summary.get("bundle_count")
                        and real_matrix_summary.get("replay_valid_count") == real_matrix_summary.get("bundle_count")
                    ),
                    summary="Repo-local matrix validates real trace/inventory/replay bundles and keeps Phase 4 readiness separate.",
                    source="scripts/plan_phase3_real_evidence_matrix.py",
                    details={
                        "bundle_count": real_matrix_summary.get("bundle_count"),
                        "real_model_pair_ready_count": real_matrix_summary.get("real_model_pair_ready_count"),
                        "replay_valid_count": real_matrix_summary.get("replay_valid_count"),
                        "policy_candidate_ready_count": real_matrix_summary.get("policy_candidate_ready_count"),
                        "policy_candidate_blocked_bundle_count": policy_candidate_gate_evidence.get("blocked_bundle_count"),
                        "policy_candidate_no_reuse_distance_observation_count": policy_candidate_gate_evidence.get("no_reuse_distance_observation_count"),
                        "policy_candidate_prompt_identity_ready_count": decision_gate_status.get("reuse_evidence_prompt_identity_ready_count"),
                        "policy_candidate_prompt_identity_metadata_missing_count": decision_gate_status.get("reuse_evidence_prompt_identity_metadata_missing_count"),
                        "policy_candidate_blocker_counts": policy_candidate_gate_evidence.get("blocker_counts", {}),
                        "phase4_ready_bundle_count": real_matrix_summary.get("phase4_ready_bundle_count"),
                        "remaining_blockers": real_matrix_summary.get("remaining_blockers", []),
                    },
                ),
                evidence_item(
                    "handoff_scaffold_coverage",
                    status=status_from_bool(
                        bool(handoff_summary.get("valid"))
                        and bool(handoff_summary.get("all_handoff_scaffolds_ready"))
                    ),
                    summary="All repo-local real bundles have prompt sets, output templates, runtime requests, and live-proof templates before capture.",
                    source="scripts/plan_phase3_handoff_coverage.py",
                    details={
                        "bundle_count": handoff_summary.get("bundle_count"),
                        "handoff_scaffold_ready_count": handoff_summary.get("handoff_scaffold_ready_count"),
                        "all_handoff_scaffolds_ready": handoff_summary.get("all_handoff_scaffolds_ready"),
                        "missing_artifact_counts": handoff_summary.get("missing_artifact_counts", {}),
                    },
                ),
                evidence_item(
                    "phase3_launch_card_library",
                    status=status_from_bool(
                        bool(launch_card_library_summary.get("valid"))
                        and bool(launch_card_library_summary.get("library_ready"))
                    ),
                    summary="Repo-level launch-card library indexes all planned runtime-capture cards and keeps execution binding gaps visible.",
                    source="scripts/plan_phase3_launch_card_library.py",
                    details={
                        "library_ready": launch_card_library_summary.get("library_ready"),
                        "execution_ready": launch_card_library_summary.get("execution_ready"),
                        "card_count": launch_card_library_summary.get("card_count"),
                        "template_ready_count": launch_card_library_summary.get("template_ready_count"),
                        "model_plane_binding_ready_count": launch_card_library_summary.get("model_plane_binding_ready_count"),
                        "binding_ready_count": launch_card_library_summary.get("binding_ready_count"),
                        "runtime_capture_command_ready_count": launch_card_library_summary.get("runtime_capture_command_ready_count"),
                        "task_count": launch_card_library_summary.get("task_count"),
                        "missing_runtime_command_count": launch_card_library_summary.get("missing_runtime_command_count"),
                        "binding_handoff_ready": launch_card_library_summary.get("binding_handoff_ready"),
                        "binding_handoff_task_count": launch_card_library_summary.get("binding_handoff_task_count"),
                        "binding_handoff_ready_count": launch_card_library_summary.get("binding_handoff_ready_count"),
                        "binding_handoff_missing_field_count": launch_card_library_summary.get("binding_handoff_missing_field_count"),
                        "unbound_task_count": launch_card_library_summary.get("unbound_task_count"),
                        "model_plane_artifact_writer_contract_request_ready": launch_card_library_summary.get("model_plane_artifact_writer_contract_request_ready"),
                        "model_plane_artifact_writer_contract_request_task_count": launch_card_library_summary.get("model_plane_artifact_writer_contract_request_task_count"),
                        "saved_handoff_artifacts_ready": launch_card_library_summary.get("saved_handoff_artifacts_ready"),
                        "saved_handoff_artifact_missing_count": launch_card_library_summary.get("saved_handoff_artifact_missing_count"),
                        "saved_handoff_artifact_drifted_count": launch_card_library_summary.get("saved_handoff_artifact_drifted_count"),
                        "blockers": launch_card_library_summary.get("blockers", []),
                    },
                ),
                evidence_item(
                    "runtime_capture_request_audit",
                    status=status_from_bool(
                        bool(runtime_request_summary.get("valid"))
                        and int(runtime_request_summary.get("drifted_request_count", 0) or 0) == 0
                    ),
                    summary="Saved runtime-capture requests are valid and drift-free before approved operator capture.",
                    source="scripts/plan_phase3_runtime_capture_request.py",
                    details={
                        "request_count": runtime_request_summary.get("request_count"),
                        "valid_request_count": runtime_request_summary.get("valid_request_count"),
                        "ready_for_operator_capture_count": runtime_request_summary.get("ready_for_operator_capture_count"),
                        "capture_complete_count": runtime_request_summary.get("capture_complete_count"),
                        "drifted_request_count": runtime_request_summary.get("drifted_request_count"),
                        "capture_queue_summary": runtime_request_summary.get("capture_queue_summary", {}),
                        "approval_rebuild_command_manifest_count": runtime_request_summary.get("approval_rebuild_command_manifest_count"),
                        "recommended_post_approval_preview": recommended_post_approval_preview,
                    },
                ),
                evidence_item(
                    "capture_result_intake",
                    status=status_from_bool(
                        bool(capture_intake_summary.get("valid"))
                        and int(capture_intake_summary.get("ready_to_update_bundle_count", 0) or 0) > 0,
                        false_status="blocked",
                    ),
                    summary="Filled runtime-capture artifacts are audited before any bundle update or Phase 4 promotion.",
                    source="scripts/plan_phase3_capture_result_intake.py",
                    details={
                        "request_count": capture_intake_summary.get("request_count"),
                        "request_audit_valid_count": capture_intake_summary.get("request_audit_valid_count"),
                        "request_drift_free_count": capture_intake_summary.get("request_drift_free_count"),
                        "request_drifted_count": capture_intake_summary.get("request_drifted_count"),
                        "capture_complete_count": capture_intake_summary.get("request_capture_complete_flag_count"),
                        "request_ready_for_operator_capture_flag_count": capture_intake_summary.get("request_ready_for_operator_capture_flag_count"),
                        "approved_but_capture_incomplete_request_count": capture_intake_summary.get("approved_but_capture_incomplete_request_count"),
                        "runtime_approval_missing_request_count": capture_intake_summary.get("runtime_approval_missing_request_count"),
                        "approval_rebuild_command_available_request_count": capture_intake_summary.get("approval_rebuild_command_available_request_count"),
                        "approval_rebuild_command_manifest_count": len(capture_intake_summary.get("approval_rebuild_command_manifest", [])) if isinstance(capture_intake_summary.get("approval_rebuild_command_manifest"), list) else None,
                        "approved_runtime_capture_pending_request_count": capture_intake_summary.get("approved_runtime_capture_pending_request_count"),
                        "capture_receipt_required_count": capture_intake_summary.get("capture_receipt_required_count"),
                        "capture_receipt_ready_count": capture_intake_summary.get("capture_receipt_ready_count"),
                        "capture_receipt_missing_request_count": capture_intake_summary.get("capture_receipt_missing_request_count"),
                        "output_receipt_binding_ready_count": capture_intake_summary.get("output_receipt_binding_ready_count"),
                        "receipt_gate_coverage": compact_receipt_gate_coverage(capture_intake_summary.get("receipt_gate_coverage")),
                        "receipt_fill_entry_count": capture_intake_summary.get("receipt_fill_entry_count"),
                        "receipt_fill_ready_count": capture_intake_summary.get("receipt_fill_ready_count"),
                        "receipt_fill_missing_count": capture_intake_summary.get("receipt_fill_missing_count"),
                        "receipt_fill_approval_missing_count": capture_intake_summary.get("receipt_fill_approval_missing_count"),
                        "receipt_fill_manifest_summary": capture_intake_summary.get("receipt_fill_manifest_summary", {}),
                        "approval_transition_preview_count": capture_intake_summary.get("approval_transition_preview_count"),
                        "approval_transition_ready_for_operator_count": capture_intake_summary.get("approval_transition_ready_for_operator_count"),
                        "approval_transition_ready_to_update_bundle_count": capture_intake_summary.get("approval_transition_ready_to_update_bundle_count"),
                        "recommended_approval_transition_preview": compact_approval_transition_preview(capture_intake_summary.get("recommended_approval_transition_preview")),
                        "post_approval_capture_fill_plan_count": capture_intake_summary.get("post_approval_capture_fill_plan_count"),
                        "post_approval_capture_fill_artifact_step_count": capture_intake_summary.get("post_approval_capture_fill_artifact_step_count"),
                        "post_approval_capture_fill_runtime_step_count": capture_intake_summary.get("post_approval_capture_fill_runtime_step_count"),
                        "post_approval_capture_fill_ready_count": capture_intake_summary.get("post_approval_capture_fill_ready_count"),
                        "post_approval_capture_fill_missing_count": capture_intake_summary.get("post_approval_capture_fill_missing_count"),
                        "post_approval_capture_fill_validator_command_count": capture_intake_summary.get("post_approval_capture_fill_validator_command_count"),
                        "post_approval_capture_fill_ready_to_update_bundle_count": capture_intake_summary.get("post_approval_capture_fill_ready_to_update_bundle_count"),
                        "recommended_post_approval_capture_fill_plan": compact_post_approval_capture_fill_plan(capture_intake_summary.get("recommended_post_approval_capture_fill_plan")),
                        "ready_to_update_bundle_count": capture_intake_summary.get("ready_to_update_bundle_count"),
                        "phase4_candidate_ready_count": capture_intake_summary.get("phase4_candidate_ready_count"),
                        "live_spike_candidate_ready_count": capture_intake_summary.get("live_spike_candidate_ready_count"),
                        "next_operator_step_counts": capture_intake_summary.get("next_operator_step_counts", {}),
                        "remaining_blockers_after_intake": capture_intake_summary.get("remaining_blockers_after_intake", []),
                    },
                ),
                evidence_item(
                    "approval_manifest_parity",
                    status=status_from_bool(approval_manifest_parity.get("ready") is True),
                    summary="Pre-capture and capture-result approval rebuild command manifests agree by request path and command metadata.",
                    source="scripts/plan_phase3_evidence_packet.py",
                    details=approval_manifest_parity,
                ),
                evidence_item(
                    "receipt_requirement_parity",
                    status=status_from_bool(receipt_requirement_parity.get("ready") is True),
                    summary="Runtime-capture request receipt requirements agree with capture-result intake receipt gates.",
                    source="scripts/plan_phase3_evidence_packet.py",
                    details=receipt_requirement_parity,
                ),
                evidence_item(
                    "operator_queue_parity",
                    status=status_from_bool(operator_queue_parity.get("ready") is True),
                    summary="Runtime-capture queue ranking agrees with capture-result intake next operator steps.",
                    source="scripts/plan_phase3_evidence_packet.py",
                    details=operator_queue_parity,
                ),
                evidence_item(
                    "post_approval_capture_fill_parity",
                    status=status_from_bool(post_approval_capture_fill_parity.get("ready") is True),
                    summary="Runtime post-approval pending artifacts agree with capture-result fill planning.",
                    source="scripts/plan_phase3_evidence_packet.py",
                    details=post_approval_capture_fill_parity,
                ),
                evidence_item(
                    "all_post_approval_capture_fill_parity",
                    status=status_from_bool(all_post_approval_capture_fill_parity.get("ready") is True),
                    summary="All queued runtime post-approval pending artifacts agree with capture-result fill planning.",
                    source="scripts/plan_phase3_evidence_packet.py",
                    details=all_post_approval_capture_fill_parity,
                ),
                evidence_item(
                    "all_request_capture_queue_manifest",
                    status=status_from_bool(all_capture_queue_summary.get("ready") is True),
                    summary="All queued runtime-capture requests are exposed as a metadata-only operator queue with fill steps and validator counts.",
                    source="scripts/plan_phase3_evidence_packet.py",
                    details={
                        **all_capture_queue_summary,
                        "requests": all_capture_queue_manifest,
                    },
                ),
                evidence_item(
                    "all_request_validator_command_manifest",
                    status=status_from_bool(all_validator_command_summary.get("ready") is True),
                    summary="All queued runtime-capture requests expose artifact validator commands for operator capture and post-capture checks.",
                    source="scripts/plan_phase3_evidence_packet.py",
                    details={
                        **all_validator_command_summary,
                        "artifact_commands": all_validator_command_manifest,
                    },
                ),
                evidence_item(
                    "all_request_downstream_handoff_manifest",
                    status=status_from_bool(all_downstream_handoff_summary.get("ready") is True),
                    summary="All queued runtime-capture requests expose policy-candidate, dense-fallback, and live-proof handoff paths.",
                    source="scripts/plan_phase3_evidence_packet.py",
                    details={
                        **all_downstream_handoff_summary,
                        "requests": all_downstream_handoff_manifest,
                    },
                ),
                evidence_item(
                    "all_request_receipt_fill_manifest",
                    status=status_from_bool(all_receipt_fill_summary.get("ready") is True, false_status="blocked"),
                    summary="All queued runtime-capture artifact receipt fills are listed with approval-after state, readiness, and validator counts.",
                    source="scripts/plan_phase3_evidence_packet.py",
                    details={
                        **all_receipt_fill_summary,
                        "receipt_fills": all_receipt_fill_manifest,
                    },
                ),
                evidence_item(
                    "all_request_receipt_validator_parity",
                    status=status_from_bool(receipt_validator_parity.get("ready") is True),
                    summary="All runtime-capture validator artifacts match capture-result receipt-fill rows by request, artifact, path, status, and validator count.",
                    source="scripts/plan_phase3_evidence_packet.py",
                    details=receipt_validator_parity,
                ),
                evidence_item(
                    "all_request_receipt_fill_command_manifest",
                    status=status_from_bool(receipt_command_summary.get("ready") is True),
                    summary="All receipt-fill rows carry their receipt paths plus post-capture validator commands.",
                    source="scripts/plan_phase3_evidence_packet.py",
                    details={
                        **receipt_command_summary,
                        "receipt_fill_commands": receipt_command_manifest,
                    },
                ),
            ]
        )
        if recommended_policy_candidate_trace_plan is not None:
            items.append(
                evidence_item(
                    "policy_candidate_trace_handoff",
                    status=status_from_bool(
                        policy_candidate_trace_handoff_ready(recommended_policy_candidate_trace_plan)
                    ),
                    summary="Recommended policy-candidate trace handoff is bound to the runtime-capture request path, receipt path, prompt set, and validator commands.",
                    source="scripts/plan_phase3_policy_candidate_trace.py",
                    details=recommended_policy_candidate_trace_plan,
                )
            )
        if recommended_dense_fallback_capture_plan is not None:
            items.append(
                evidence_item(
                    "dense_fallback_capture_handoff",
                    status=status_from_bool(
                        dense_fallback_capture_handoff_ready(recommended_dense_fallback_capture_plan)
                    ),
                    summary="Recommended dense fallback capture handoff is bound to the runtime-capture request prompt/output paths and fallback comparison commands.",
                    source="scripts/plan_phase3_dense_fallback_capture.py",
                    details=recommended_dense_fallback_capture_plan,
                )
            )
        if recommended_live_capability_proof_handoff is not None:
            items.append(
                evidence_item(
                    "live_capability_proof_handoff",
                    status=status_from_bool(
                        live_capability_proof_handoff_ready(recommended_live_capability_proof_handoff)
                    ),
                    summary="Recommended live capability proof handoff keeps the future proof template bound to the selected runtime-capture bundle before live adapter work.",
                    source="scripts/plan_phase3_live_capability_proof.py",
                    details=recommended_live_capability_proof_handoff,
                )
            )



    promotion_checklist = (
        build_promotion_checklist(
            real_matrix_summary=real_matrix_summary,
            handoff_summary=handoff_summary,
            runtime_request_summary=runtime_request_summary,
            capture_intake_summary=capture_intake_summary,
            fallback_summary=fallback_summary,
            live_proof_summary=live_proof_summary,
            decision_summary=decision_summary,
            policy_candidate_trace_plan=recommended_policy_candidate_trace_plan,
            dense_fallback_capture_plan=recommended_dense_fallback_capture_plan,
            live_capability_proof_handoff=recommended_live_capability_proof_handoff,
            launch_card_library_summary=launch_card_library_summary,
        )
        if include_repo_gates
        else []
    )
    blocker_closure_manifest = phase3_blocker_closure_manifest(
        decision_summary=decision_summary,
        promotion_checklist=promotion_checklist,
        recommended_runtime_capture=recommended_runtime_capture,
        policy_candidate_trace_plan=recommended_policy_candidate_trace_plan,
        dense_fallback_capture_plan=recommended_dense_fallback_capture_plan,
        live_capability_proof_handoff=recommended_live_capability_proof_handoff,
        runtime_actuator_design=runtime_actuator_design_summary,
        runtime_actuator_spike=runtime_actuator_spike_summary,
        all_downstream_handoff_manifest=all_downstream_handoff_manifest,
        all_capture_queue_summary=all_capture_queue_summary,
        all_receipt_fill_summary=all_receipt_fill_summary,
        receipt_command_summary=receipt_command_summary,
    )
    blocker_closure_summary = phase3_blocker_closure_summary(blocker_closure_manifest)
    recommended_capture_preflight_manifest = recommended_runtime_capture_preflight_manifest(
        recommended_runtime_capture,
        recommended_post_approval_preview,
        all_capture_queue_manifest,
        receipt_command_manifest,
        blocker_closure_manifest,
    )
    recommended_capture_preflight_summary = recommended_runtime_capture_preflight_summary(
        recommended_capture_preflight_manifest
    )
    recommended_execution_coverage_manifest = recommended_runtime_capture_execution_coverage_manifest(
        recommended_runtime_capture,
        recommended_capture_preflight_manifest,
        all_capture_queue_manifest,
        recommended_runtime_capture_command_contract,
    )
    recommended_execution_coverage_summary = recommended_runtime_capture_execution_coverage_summary(
        recommended_execution_coverage_manifest
    )
    all_runtime_capture_command_contracts_by_request = runtime_capture_command_contracts_from_launch_card_dir(
        all_capture_queue_manifest,
        runtime_capture_launch_card_directory_manifest,
    )
    all_execution_coverage_manifest = all_request_runtime_capture_execution_coverage_manifest(
        all_capture_queue_manifest,
        launch_card_library_summary,
        recommended_runtime_capture_command_contract,
        command_contracts_by_request=all_runtime_capture_command_contracts_by_request,
        launch_card_dir_manifest=runtime_capture_launch_card_directory_manifest,
    )
    all_execution_coverage_summary = all_request_runtime_capture_execution_coverage_summary(
        all_execution_coverage_manifest
    )
    manual_capture_runbook_manifest = all_request_manual_capture_runbook_manifest(
        all_capture_queue_manifest,
        receipt_command_manifest,
        all_execution_coverage_manifest,
    )
    manual_capture_runbook_summary = all_request_manual_capture_runbook_summary(manual_capture_runbook_manifest)
    post_capture_intake_runbook_manifest = all_request_post_capture_intake_runbook_manifest(
        manual_capture_runbook_manifest,
        capture_intake_summary,
    )
    post_capture_intake_runbook_summary = all_request_post_capture_intake_runbook_summary(post_capture_intake_runbook_manifest)
    recommended_manual_capture_runbook = recommended_manual_capture_runbook_manifest(
        manual_capture_runbook_manifest,
        recommended_runtime_capture or {},
    )
    recommended_manual_capture_runbook_summary = all_request_manual_capture_runbook_summary(
        recommended_manual_capture_runbook
    )
    recommended_post_capture_intake_runbook = recommended_post_capture_intake_runbook_manifest(
        post_capture_intake_runbook_manifest,
        recommended_runtime_capture or {},
    )
    recommended_post_capture_intake_runbook_summary = all_request_post_capture_intake_runbook_summary(
        recommended_post_capture_intake_runbook
    )
    recommended_runtime_capture_work_order = recommended_runtime_capture_work_order_manifest(
        recommended_runtime_capture or {},
        recommended_capture_preflight_manifest,
        recommended_manual_capture_runbook,
        recommended_post_capture_intake_runbook,
    )
    recommended_runtime_capture_work_order_stats = recommended_runtime_capture_work_order_summary(
        recommended_runtime_capture_work_order
    )
    recommended_completion_receipt_template = recommended_runtime_capture_completion_receipt_template_manifest(
        recommended_runtime_capture_work_order
    )
    recommended_completion_receipt_template_stats = recommended_runtime_capture_completion_receipt_template_summary(
        recommended_completion_receipt_template
    )
    blocker_evidence_ledger_manifest = phase3_blocker_evidence_ledger_manifest(
        blocker_closure_manifest,
        post_capture_intake_runbook_manifest,
        recommended_dense_fallback_capture_plan,
        recommended_live_capability_proof_handoff,
    )
    blocker_evidence_ledger_summary = phase3_blocker_evidence_ledger_summary(blocker_evidence_ledger_manifest)
    blocker_resolution_queue_manifest = phase3_blocker_resolution_queue_manifest(
        blocker_evidence_ledger_manifest,
        runtime_actuator_spike_summary,
    )
    blocker_resolution_queue_summary = phase3_blocker_resolution_queue_summary(blocker_resolution_queue_manifest)
    next_unblocked_operator_handoff = next_unblocked_operator_handoff_summary(
        blocker_resolution_queue_summary,
        blocker_resolution_queue_manifest,
        recommended_runtime_capture_work_order_stats,
        recommended_runtime_capture_work_order,
        recommended_completion_receipt_template_stats,
        recommended_completion_receipt_template,
    )
    if include_repo_gates and recommended_capture_preflight_manifest:
        promotion_checklist.append(
            promotion_checklist_item(
                "recommended_runtime_capture_preflight",
                status="satisfied" if recommended_capture_preflight_summary.get("ready") is True else "blocked",
                summary="The selected runtime-capture request has approval metadata, non-mutating post-approval preview, receipt-fill rows, validators, and runtime blocker closures.",
                next_action="Request explicit approval, then capture the pending artifacts and fill their receipts."
                if recommended_capture_preflight_summary.get("ready") is True
                else "Repair the selected request preflight before asking for runtime capture approval.",
                evidence={
                    **recommended_capture_preflight_summary,
                    "manifest": recommended_capture_preflight_manifest,
                },
            )
        )
    if include_repo_gates and recommended_runtime_capture_command_contract:
        promotion_checklist.append(
            promotion_checklist_item(
                "recommended_runtime_capture_command_contract",
                status="satisfied" if recommended_runtime_capture_command_contract_summary.get("ready") is True else "blocked",
                summary="The selected runtime-capture request has planned Model Plane/launch-card binding requirements for each pending runtime artifact.",
                next_action="Bind the planned tasks to real Model Plane callable or launch-card commands; keep execution coverage blocked until command_ready=true."
                if recommended_runtime_capture_command_contract_summary.get("ready") is True
                else "Repair the command contract before runtime capture approval.",
                evidence={
                    **recommended_runtime_capture_command_contract_summary,
                    "contract": recommended_runtime_capture_command_contract,
                },
            )
        )
    if include_repo_gates and recommended_execution_coverage_manifest:
        promotion_checklist.append(
            promotion_checklist_item(
                "recommended_runtime_capture_execution_coverage",
                status="satisfied" if recommended_execution_coverage_summary.get("automated_capture_ready") is True else "blocked",
                summary="The selected runtime-capture request classifies capture-command coverage for every pending artifact.",
                next_action="Run the approved manual capture plan or add model-plane/launch-card runtime capture commands for manual-only artifacts."
                if recommended_execution_coverage_summary.get("manual_operator_capture_ready") is True
                else "Repair execution coverage before asking for runtime capture approval.",
                evidence={
                    **recommended_execution_coverage_summary,
                    "manifest": recommended_execution_coverage_manifest,
                },
            )
        )
    if include_repo_gates and all_execution_coverage_manifest:
        promotion_checklist.append(
            promotion_checklist_item(
                "all_request_runtime_capture_execution_coverage",
                status="satisfied" if all_execution_coverage_summary.get("automated_capture_ready") is True else "blocked",
                summary="Every queued runtime-capture request has classified manual/runtime-command execution coverage before approved prompt traffic.",
                next_action="Fill launch-card callable ids or launch commands for all manual-only capture tasks, or proceed with approved manual capture using the classified queue."
                if all_execution_coverage_summary.get("manual_operator_capture_ready") is True
                else "Repair all-request execution coverage before asking for runtime capture approval.",
                evidence={
                    **all_execution_coverage_summary,
                    "manifest": all_execution_coverage_manifest,
                },
            )
        )
    if include_repo_gates and manual_capture_runbook_manifest:
        promotion_checklist.append(
            promotion_checklist_item(
                "all_request_manual_capture_runbook",
                status="satisfied" if manual_capture_runbook_summary.get("ready") is True else "blocked",
                summary="Manual-only runtime captures are grouped into request-level approval, artifact, receipt, and validator steps before approved prompt traffic.",
                next_action="Use the manual runbook after explicit approval, or replace manual rows with bound launch-card runtime commands before capture.",
                evidence={
                    **manual_capture_runbook_summary,
                    "manifest": manual_capture_runbook_manifest,
                },
            )
        )
    if include_repo_gates and post_capture_intake_runbook_manifest:
        promotion_checklist.append(
            promotion_checklist_item(
                "all_request_post_capture_intake_runbook",
                status="satisfied" if post_capture_intake_runbook_summary.get("ready") is True else "blocked",
                summary="After approved capture, every queued artifact has a receipt gate, validator count, and capture-result intake command before bundle promotion.",
                next_action="Fill the approved capture receipts, rerun capture-result intake, and only promote bundles after the intake gates become ready.",
                evidence={
                    **post_capture_intake_runbook_summary,
                    "manifest": post_capture_intake_runbook_manifest,
                },
            )
        )
    if include_repo_gates:
        promotion_checklist.append(
            promotion_checklist_item(
                "phase3_blocker_closure_manifest",
                status="satisfied" if blocker_closure_summary.get("ready") is True else "blocked",
                summary="Every current no-go reason is mapped to a closure gate, approval stage, handoff path, and next action.",
                next_action="Use the closure manifest to clear runtime-capture, fallback, and future-adapter blockers in order."
                if blocker_closure_summary.get("ready") is True
                else "Add closure mappings for unmapped no-go reasons before operator capture work continues.",
                evidence={
                    **blocker_closure_summary,
                    "reasons": blocker_closure_manifest,
                },
            )
        )
        promotion_checklist.append(
            promotion_checklist_item(
                "phase3_blocker_evidence_ledger",
                status="satisfied" if blocker_evidence_ledger_summary.get("ready") is True else "blocked",
                summary="Every current no-go reason's missing evidence is expanded into a row-level evidence ledger.",
                next_action="Close ledger rows in order, starting with approved runtime captures, then dense fallback outputs, then live proof and actuator rows."
                if blocker_evidence_ledger_summary.get("ready") is True
                else "Repair blocker evidence ledger row counts before operator capture work continues.",
                evidence={
                    **blocker_evidence_ledger_summary,
                    "manifest": blocker_evidence_ledger_manifest,
                },
            )
        )
        promotion_checklist.append(
            promotion_checklist_item(
                "phase3_blocker_resolution_queue",
                status="satisfied" if blocker_resolution_queue_summary.get("ready") is True else "blocked",
                summary="The blocker evidence ledger is grouped into ordered operator work packages for clearing Phase 3 blockers.",
                next_action="Use the resolution queue to clear approved runtime-capture, receipt-intake, dense-fallback, live-proof, and actuator-design packages in order."
                if blocker_resolution_queue_summary.get("ready") is True
                else "Repair blocker resolution queue grouping before operator handoff work continues.",
                evidence={
                    **blocker_resolution_queue_summary,
                    "manifest": blocker_resolution_queue_manifest,
                },
            )
        )
    if include_repo_gates and recommended_capture_preflight_manifest:
        items.append(
            evidence_item(
                "recommended_runtime_capture_preflight",
                status=status_from_bool(recommended_capture_preflight_summary.get("ready") is True),
                summary="The selected runtime-capture request passes metadata preflight before any approved prompt traffic.",
                source="scripts/plan_phase3_evidence_packet.py",
                details={
                    **recommended_capture_preflight_summary,
                    "manifest": recommended_capture_preflight_manifest,
                },
            )
        )
    if include_repo_gates and recommended_runtime_capture_command_contract:
        items.append(
            evidence_item(
                "recommended_runtime_capture_command_contract",
                status=status_from_bool(recommended_runtime_capture_command_contract_summary.get("ready") is True),
                summary="The selected runtime-capture request has a planned launch-card/callable binding contract before executable runtime commands are attached.",
                source="scripts/plan_phase3_runtime_capture_commands.py",
                details={
                    **recommended_runtime_capture_command_contract_summary,
                    "contract": recommended_runtime_capture_command_contract,
                },
            )
        )
    if include_repo_gates and recommended_execution_coverage_manifest:
        items.append(
            evidence_item(
                "recommended_runtime_capture_execution_coverage",
                status=status_from_bool(recommended_execution_coverage_summary.get("automated_capture_ready") is True),
                summary="The selected runtime-capture request has classified runtime-command/manual execution coverage before approved prompt traffic.",
                source="scripts/plan_phase3_evidence_packet.py",
                details={
                    **recommended_execution_coverage_summary,
                    "manifest": recommended_execution_coverage_manifest,
                },
            )
        )
    if include_repo_gates and all_execution_coverage_manifest:
        items.append(
            evidence_item(
                "all_request_runtime_capture_execution_coverage",
                status=status_from_bool(all_execution_coverage_summary.get("automated_capture_ready") is True),
                summary="All queued runtime-capture requests have classified runtime-command/manual execution coverage before approved prompt traffic.",
                source="scripts/plan_phase3_evidence_packet.py",
                details={
                    **all_execution_coverage_summary,
                    "manifest": all_execution_coverage_manifest,
                },
            )
        )
    if include_repo_gates and manual_capture_runbook_manifest:
        items.append(
            evidence_item(
                "all_request_manual_capture_runbook",
                status=status_from_bool(manual_capture_runbook_summary.get("ready") is True),
                summary="All queued manual runtime captures have a request-level approval, artifact, receipt, and validator runbook.",
                source="scripts/plan_phase3_evidence_packet.py",
                details={
                    **manual_capture_runbook_summary,
                    "manifest": manual_capture_runbook_manifest,
                },
            )
        )
    if include_repo_gates and post_capture_intake_runbook_manifest:
        items.append(
            evidence_item(
                "all_request_post_capture_intake_runbook",
                status=status_from_bool(post_capture_intake_runbook_summary.get("ready") is True),
                summary="All queued runtime-capture artifacts have post-capture receipt gates and capture-result intake commands before bundle promotion.",
                source="scripts/plan_phase3_evidence_packet.py",
                details={
                    **post_capture_intake_runbook_summary,
                    "manifest": post_capture_intake_runbook_manifest,
                },
            )
        )
    if include_repo_gates and recommended_manual_capture_runbook:
        items.append(
            evidence_item(
                "recommended_manual_capture_runbook",
                status=status_from_bool(recommended_manual_capture_runbook_summary.get("ready") is True),
                summary="The recommended runtime-capture request has a single-request manual capture runbook derived from the all-request package.",
                source="scripts/plan_phase3_evidence_packet.py",
                details={
                    **recommended_manual_capture_runbook_summary,
                    "manifest": recommended_manual_capture_runbook,
                },
            )
        )
    if include_repo_gates and recommended_post_capture_intake_runbook:
        items.append(
            evidence_item(
                "recommended_post_capture_intake_runbook",
                status=status_from_bool(recommended_post_capture_intake_runbook_summary.get("ready") is True),
                summary="The recommended runtime-capture request has a single-request post-capture intake runbook derived from the all-request package.",
                source="scripts/plan_phase3_evidence_packet.py",
                details={
                    **recommended_post_capture_intake_runbook_summary,
                    "manifest": recommended_post_capture_intake_runbook,
                },
            )
        )
    if include_repo_gates and recommended_runtime_capture_work_order:
        items.append(
            evidence_item(
                "recommended_runtime_capture_work_order",
                status=status_from_bool(recommended_runtime_capture_work_order_stats.get("ready") is True),
                summary="The recommended runtime-capture request has one ordered approval, capture, validation, and intake work order.",
                source="scripts/plan_phase3_evidence_packet.py",
                details={
                    **recommended_runtime_capture_work_order_stats,
                    "manifest": recommended_runtime_capture_work_order,
                },
            )
        )
    if include_repo_gates and recommended_completion_receipt_template:
        items.append(
            evidence_item(
                "recommended_runtime_capture_completion_receipt_template",
                status=status_from_bool(recommended_completion_receipt_template_stats.get("template_ready") is True),
                summary="The recommended runtime-capture work order has a fillable completion receipt template for post-capture verification and intake.",
                source="scripts/plan_phase3_evidence_packet.py",
                details={
                    **recommended_completion_receipt_template_stats,
                    "manifest": recommended_completion_receipt_template,
                },
            )
        )
    if include_repo_gates and next_unblocked_operator_handoff:
        items.append(
            evidence_item(
                "phase3_next_unblocked_operator_handoff",
                status=status_from_bool(next_unblocked_operator_handoff.get("handoff_ready") is True),
                summary="The next unblocked blocker-resolution package is tied to the recommended runtime-capture work order and completion receipt template.",
                source="scripts/plan_phase3_evidence_packet.py",
                details=next_unblocked_operator_handoff,
            )
        )
    items.append(
        evidence_item(
            "phase3_blocker_closure_manifest",
            status=status_from_bool(blocker_closure_summary.get("ready") is True),
            summary="No-go reasons are mapped to concrete closure gates, approval stages, paths, validator counts, and next actions.",
            source="scripts/plan_phase3_evidence_packet.py",
            details={
                **blocker_closure_summary,
                "reasons": blocker_closure_manifest,
            },
        )
    )

    items.append(
        evidence_item(
            "phase3_blocker_evidence_ledger",
            status=status_from_bool(blocker_evidence_ledger_summary.get("ready") is True),
            summary="Remaining missing blocker evidence is enumerated as concrete rows with artifact paths, receipts, prompts, sections, and validator counts.",
            source="scripts/plan_phase3_evidence_packet.py",
            details={
                **blocker_evidence_ledger_summary,
                "manifest": blocker_evidence_ledger_manifest,
            },
        )
    )
    items.append(
        evidence_item(
            "phase3_blocker_resolution_queue",
            status=status_from_bool(blocker_resolution_queue_summary.get("ready") is True),
            summary="Remaining blocker evidence rows are grouped into ordered operator work packages.",
            source="scripts/plan_phase3_evidence_packet.py",
            details={
                **blocker_resolution_queue_summary,
                "manifest": blocker_resolution_queue_manifest,
            },
        )
    )
    status_counts: dict[str, int] = {}
    for item in items:
        status = str(item["status"])
        status_counts[status] = status_counts.get(status, 0) + 1

    decision_policy_candidate_trace_receipt_ready = (
        decision_gate_status.get("policy_candidate_trace_receipt_ready") is True
    )

    packet_ready = not errors and all(item["status"] in {"proven", "fixture_only", "blocked", "future_phase_4"} for item in items)
    phase3_complete = (
        packet_ready
        and all(
            item["status"] == "proven"
            for item in items
            if item["id"] != "live_capability_proof"
        )
        and decision_summary.get("ready_for_phase4_adapter_spike") is True
        and decision_policy_candidate_trace_receipt_ready
    )

    return {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "mode": "phase3_evidence_packet",
        "trace_path": str(trace_path),
        "inventory_path": str(inventory_path),
        "policies_path": str(policies_path),
        "managed_plan_path": str(managed_plan_path),
        "fallback_artifact_path": str(fallback_artifact_path) if fallback_artifact_path is not None else None,
        "live_proof_artifact_path": str(live_proof_artifact_path) if live_proof_artifact_path is not None else None,
        "runtime_capture_launch_card_path": str(runtime_capture_launch_card_path) if runtime_capture_launch_card_path is not None else None,
        "runtime_capture_launch_card_dir_path": str(runtime_capture_launch_card_dir) if runtime_capture_launch_card_dir is not None else None,
        "runtime_capture_launch_card_directory_manifest": runtime_capture_launch_card_directory_manifest,
        "phase3_reuse_evidence_capture_plan": reuse_summary,
        "policy_candidate_trace_receipt_path": str(policy_candidate_trace_receipt_path) if policy_candidate_trace_receipt_path is not None else None,
        "live_proof_context_binding": live_proof_summary.get("context_binding", {}),
        "valid": not errors,
        "errors": errors,
        "packet_ready": packet_ready,
        "phase3_complete": phase3_complete,
        "status_counts": dict(sorted(status_counts.items())),
        "evidence_items": items,
        "promotion_checklist": promotion_checklist,
        "recommended_runtime_capture_request": recommended_runtime_capture,
        "recommended_post_approval_preview": recommended_post_approval_preview,
        "recommended_runtime_capture_preflight_summary": recommended_capture_preflight_summary,
        "recommended_runtime_capture_preflight_manifest": recommended_capture_preflight_manifest,
        "recommended_runtime_capture_command_contract_summary": recommended_runtime_capture_command_contract_summary,
        "recommended_runtime_capture_command_contract": recommended_runtime_capture_command_contract,
        "recommended_runtime_capture_launch_card_template": recommended_launch_card_template,
        "recommended_runtime_capture_execution_coverage_summary": recommended_execution_coverage_summary,
        "recommended_runtime_capture_execution_coverage_manifest": recommended_execution_coverage_manifest,
        "all_request_runtime_capture_execution_coverage_summary": all_execution_coverage_summary,
        "all_request_runtime_capture_execution_coverage_manifest": all_execution_coverage_manifest,
        "all_request_manual_capture_runbook_summary": manual_capture_runbook_summary,
        "all_request_manual_capture_runbook_manifest": manual_capture_runbook_manifest,
        "all_request_post_capture_intake_runbook_summary": post_capture_intake_runbook_summary,
        "all_request_post_capture_intake_runbook_manifest": post_capture_intake_runbook_manifest,
        "recommended_manual_capture_runbook_summary": recommended_manual_capture_runbook_summary,
        "recommended_manual_capture_runbook_manifest": recommended_manual_capture_runbook,
        "recommended_post_capture_intake_runbook_summary": recommended_post_capture_intake_runbook_summary,
        "recommended_post_capture_intake_runbook_manifest": recommended_post_capture_intake_runbook,
        "recommended_runtime_capture_work_order_summary": recommended_runtime_capture_work_order_stats,
        "recommended_runtime_capture_work_order_manifest": recommended_runtime_capture_work_order,
        "recommended_runtime_capture_completion_receipt_template_summary": recommended_completion_receipt_template_stats,
        "recommended_runtime_capture_completion_receipt_template_manifest": recommended_completion_receipt_template,
        "phase3_next_unblocked_operator_handoff_summary": next_unblocked_operator_handoff,
        "phase3_launch_card_library_summary": launch_card_library_summary,
        "all_request_capture_queue_summary": all_capture_queue_summary,
        "all_request_capture_queue_manifest": all_capture_queue_manifest,
        "all_request_approval_command_manifest": runtime_request_summary.get("approval_rebuild_command_manifest", []),
        "all_request_validator_command_summary": all_validator_command_summary,
        "all_request_validator_command_manifest": all_validator_command_manifest,
        "all_request_downstream_handoff_summary": all_downstream_handoff_summary,
        "all_request_downstream_handoff_manifest": all_downstream_handoff_manifest,
        "all_request_receipt_fill_summary": all_receipt_fill_summary,
        "all_request_receipt_fill_manifest": all_receipt_fill_manifest,
        "all_request_receipt_validator_parity_summary": receipt_validator_parity,
        "all_request_receipt_fill_command_summary": receipt_command_summary,
        "all_request_receipt_fill_command_manifest": receipt_command_manifest,
        "phase3_blocker_closure_summary": blocker_closure_summary,
        "phase3_blocker_closure_manifest": blocker_closure_manifest,
        "phase3_blocker_evidence_ledger_summary": blocker_evidence_ledger_summary,
        "phase3_blocker_evidence_ledger_manifest": blocker_evidence_ledger_manifest,
        "phase3_blocker_resolution_queue_summary": blocker_resolution_queue_summary,
        "phase3_blocker_resolution_queue_manifest": blocker_resolution_queue_manifest,

        "recommended_policy_candidate_trace_plan": recommended_policy_candidate_trace_plan,
        "recommended_dense_fallback_capture_plan": recommended_dense_fallback_capture_plan,
        "recommended_live_capability_proof_handoff": recommended_live_capability_proof_handoff,
        "phase3_runtime_actuator_design_summary": runtime_actuator_design_summary,
        "phase3_runtime_actuator_spike_summary": runtime_actuator_spike_summary,
        "decision": decision_summary.get("decision"),
        "remaining_gaps": {
            "real_model_pair_ready": (
                real_matrix_summary.get("real_model_pair_ready_count") == real_matrix_summary.get("bundle_count")
                if include_repo_gates
                else pairing_summary.get("real_model_pair_ready")
            ),
            "policy_candidate_ready_count": policy_candidate_gate_evidence.get("policy_candidate_ready_count"),
            "policy_candidate_blocked_bundle_count": policy_candidate_gate_evidence.get("blocked_bundle_count"),
            "policy_candidate_no_reuse_distance_observation_count": policy_candidate_gate_evidence.get("no_reuse_distance_observation_count"),
            "policy_candidate_prompt_identity_ready_count": decision_gate_status.get("reuse_evidence_prompt_identity_ready_count"),
            "policy_candidate_prompt_identity_metadata_missing_count": decision_gate_status.get("reuse_evidence_prompt_identity_metadata_missing_count"),
            "policy_candidate_next_operator_step": policy_candidate_gate_evidence.get("next_operator_step"),
            "policy_candidate_trace_handoff_ready": policy_candidate_trace_handoff_ready(recommended_policy_candidate_trace_plan),
            "policy_candidate_trace_prompt_set_ready": recommended_policy_candidate_trace_plan.get("prompt_set_ready") if isinstance(recommended_policy_candidate_trace_plan, dict) else None,
            "policy_candidate_trace_exists": recommended_policy_candidate_trace_plan.get("candidate_trace_exists") if isinstance(recommended_policy_candidate_trace_plan, dict) else None,
            "policy_candidate_trace_receipt_ready": (
                recommended_policy_candidate_trace_plan.get("capture_receipt_ready")
                if isinstance(recommended_policy_candidate_trace_plan, dict)
                else decision_gate_status.get("policy_candidate_trace_receipt_ready")
            ),
            "policy_candidate_trace_decision_receipt_ready": decision_gate_status.get("policy_candidate_trace_receipt_ready"),
            "policy_candidate_trace_ready_after_plan": recommended_policy_candidate_trace_plan.get("policy_candidate_ready_after_plan") if isinstance(recommended_policy_candidate_trace_plan, dict) else None,
            "policy_candidate_trace_path": recommended_policy_candidate_trace_plan.get("candidate_trace_path") if isinstance(recommended_policy_candidate_trace_plan, dict) else None,
            "policy_candidate_trace_receipt_path": (
                recommended_policy_candidate_trace_plan.get("candidate_trace_receipt_path")
                if isinstance(recommended_policy_candidate_trace_plan, dict)
                else decision_gate_status.get("policy_candidate_trace_receipt_path")
            ),
            "policy_candidate_trace_decision_receipt_path": decision_gate_status.get("policy_candidate_trace_receipt_path"),
            "dense_fallback_capture_handoff_ready": dense_fallback_capture_handoff_ready(recommended_dense_fallback_capture_plan),
            "dense_fallback_prompt_set_ready": recommended_dense_fallback_capture_plan.get("prompt_set_ready") if isinstance(recommended_dense_fallback_capture_plan, dict) else None,
            "dense_fallback_managed_output_ready": recommended_dense_fallback_capture_plan.get("managed_output_ready") if isinstance(recommended_dense_fallback_capture_plan, dict) else None,
            "dense_fallback_dense_output_ready": recommended_dense_fallback_capture_plan.get("dense_output_ready") if isinstance(recommended_dense_fallback_capture_plan, dict) else None,
            "dense_fallback_metadata_ready_to_build_comparison": recommended_dense_fallback_capture_plan.get("metadata_ready_to_build_comparison") if isinstance(recommended_dense_fallback_capture_plan, dict) else None,
            "dense_fallback_managed_output_path": recommended_dense_fallback_capture_plan.get("managed_output_path") if isinstance(recommended_dense_fallback_capture_plan, dict) else None,
            "dense_fallback_dense_output_path": recommended_dense_fallback_capture_plan.get("dense_output_path") if isinstance(recommended_dense_fallback_capture_plan, dict) else None,
            "dense_fallback_capture_comparison_path": recommended_dense_fallback_capture_plan.get("fallback_artifact_path") if isinstance(recommended_dense_fallback_capture_plan, dict) else None,
            "dense_fallback_comparison_ready": fallback_summary.get("comparison_ready"),
            "live_capability_proof_handoff_ready": live_capability_proof_handoff_ready(recommended_live_capability_proof_handoff),
            "live_capability_proof_template_path": recommended_live_capability_proof_handoff.get("proof_artifact_path") if isinstance(recommended_live_capability_proof_handoff, dict) else None,
            "live_capability_proof_context_ready": recommended_live_capability_proof_handoff.get("context_binding_ready") if isinstance(recommended_live_capability_proof_handoff, dict) else None,
            "live_residency_observation_ready": recommended_live_capability_proof_handoff.get("residency_observation_ready") if isinstance(recommended_live_capability_proof_handoff, dict) else None,
            "live_residency_control_ready": recommended_live_capability_proof_handoff.get("residency_control_ready") if isinstance(recommended_live_capability_proof_handoff, dict) else None,
            "cleanup_restore_proof_ready": recommended_live_capability_proof_handoff.get("cleanup_restore_ready") if isinstance(recommended_live_capability_proof_handoff, dict) else None,
            "live_artifact_export_ready": recommended_live_capability_proof_handoff.get("artifact_export_ready") if isinstance(recommended_live_capability_proof_handoff, dict) else None,
            "live_capability_proof_ready": live_proof_summary.get("proof_ready"),
            "runtime_actuator_design_ready": runtime_actuator_design_summary.get("design_handoff_ready"),
            "runtime_actuator_backend_family": runtime_actuator_design_summary.get("backend_family"),
            "runtime_actuator_live_ready": runtime_actuator_design_summary.get("live_actuator_ready"),
            "runtime_actuator_blocking_capability_count": runtime_actuator_design_summary.get("blocking_capability_count"),
            "runtime_actuator_control_blocker_count": runtime_actuator_design_summary.get("control_blocker_count"),
            "runtime_actuator_spike_handoff_ready": runtime_actuator_spike_summary.get("spike_handoff_ready"),
            "runtime_actuator_spike_live_ready": runtime_actuator_spike_summary.get("live_spike_ready"),
            "runtime_actuator_spike_proof_requirement_count": runtime_actuator_spike_summary.get("proof_requirement_count"),
            "runtime_actuator_spike_proof_artifact_count": runtime_actuator_spike_summary.get("proof_artifact_count"),
            "runtime_actuator_spike_dependency_edge_count": runtime_actuator_spike_summary.get("dependency_edge_count"),
            "runtime_actuator_spike_blocking_capability_count": runtime_actuator_spike_summary.get("blocking_capability_count"),
            "ready_for_phase4_adapter_spike": decision_summary.get("ready_for_phase4_adapter_spike"),
            "ready_for_live_spike": decision_summary.get("ready_for_live_spike"),
            "handoff_scaffolds_ready": handoff_summary.get("all_handoff_scaffolds_ready"),
            "runtime_capture_request_count": runtime_request_summary.get("request_count"),
            "runtime_capture_ready_for_operator_count": runtime_request_summary.get("ready_for_operator_capture_count"),
            "runtime_capture_complete_count": runtime_request_summary.get("capture_complete_count"),
            "runtime_capture_queue_count": runtime_request_summary.get("capture_queue_summary", {}).get("queue_count"),
            "runtime_capture_recommended_rank": runtime_request_summary.get("capture_queue_summary", {}).get("recommended_rank"),
            "runtime_capture_approval_rebuild_command_manifest_count": runtime_request_summary.get("approval_rebuild_command_manifest_count"),
            "runtime_capture_post_approval_preview_valid": recommended_post_approval_preview.get("valid") if isinstance(recommended_post_approval_preview, dict) else None,
            "runtime_capture_post_approval_preview_status": recommended_post_approval_preview.get("status") if isinstance(recommended_post_approval_preview, dict) else None,
            "runtime_capture_post_approval_ready_for_operator": recommended_post_approval_preview.get("ready_for_operator_capture") if isinstance(recommended_post_approval_preview, dict) else None,
            "runtime_capture_post_approval_capture_complete": recommended_post_approval_preview.get("capture_complete") if isinstance(recommended_post_approval_preview, dict) else None,
            "runtime_capture_post_approval_pending_artifact_count": recommended_post_approval_preview.get("pending_artifact_count") if isinstance(recommended_post_approval_preview, dict) else None,
            "runtime_capture_post_approval_mutates_request": recommended_post_approval_preview.get("mutates_request") if isinstance(recommended_post_approval_preview, dict) else None,
            "recommended_runtime_capture_preflight_ready": recommended_capture_preflight_summary.get("ready"),
            "recommended_runtime_capture_preflight_pending_artifact_count": recommended_capture_preflight_summary.get("pending_artifact_count"),
            "recommended_runtime_capture_preflight_receipt_entry_count": recommended_capture_preflight_summary.get("receipt_entry_count"),
            "recommended_runtime_capture_preflight_validator_command_count": recommended_capture_preflight_summary.get("validator_command_count"),
            "recommended_runtime_capture_preflight_runtime_closure_reason_count": recommended_capture_preflight_summary.get("runtime_closure_reason_count"),
            "recommended_runtime_capture_preflight_missing_item_count": recommended_capture_preflight_summary.get("missing_preflight_item_count"),
            "recommended_runtime_capture_command_contract_ready": recommended_runtime_capture_command_contract_summary.get("ready"),
            "recommended_runtime_capture_command_contract_runtime_ready": recommended_runtime_capture_command_contract_summary.get("runtime_capture_command_ready"),
            "recommended_runtime_capture_command_contract_planned_capture_count": recommended_runtime_capture_command_contract_summary.get("planned_capture_count"),
            "recommended_runtime_capture_command_contract_runtime_command_count": recommended_runtime_capture_command_contract_summary.get("runtime_command_option_count"),
            "recommended_runtime_capture_command_contract_missing_runtime_command_count": recommended_runtime_capture_command_contract_summary.get("missing_runtime_command_count"),
            "recommended_runtime_capture_launch_card_binding_valid": recommended_runtime_capture_command_contract_summary.get("launch_card_binding_valid"),
            "recommended_runtime_capture_launch_card_binding_ready": recommended_runtime_capture_command_contract_summary.get("launch_card_binding_ready"),
            "recommended_runtime_capture_launch_card_binding_path": recommended_runtime_capture_command_contract_summary.get("launch_card_binding_path"),
            "recommended_runtime_capture_launch_card_template_path": recommended_launch_card_template.get("launch_card_path"),
            "recommended_runtime_capture_launch_card_template_match_basis": recommended_launch_card_template.get("match_basis"),
            "recommended_runtime_capture_launch_card_template_ready": recommended_launch_card_template.get("template_ready"),
            "recommended_runtime_capture_launch_card_template_binding_handoff_ready": recommended_launch_card_template.get("binding_handoff_ready"),
            "recommended_runtime_capture_launch_card_template_task_count": recommended_launch_card_template.get("task_count"),
            "recommended_runtime_capture_launch_card_template_unbound_task_count": recommended_launch_card_template.get("unbound_task_count"),
            "recommended_runtime_capture_launch_card_template_missing_runtime_command_count": recommended_launch_card_template.get("missing_runtime_command_count"),
            "recommended_runtime_capture_launch_card_binding_bound_task_count": recommended_runtime_capture_command_contract_summary.get("launch_card_binding_bound_task_count"),
            "recommended_runtime_capture_launch_card_binding_command_option_count": recommended_runtime_capture_command_contract_summary.get("launch_card_binding_command_option_count"),
            "recommended_runtime_capture_launch_card_binding_missing_runtime_command_count": recommended_runtime_capture_command_contract_summary.get("launch_card_binding_missing_runtime_command_count"),
            "recommended_runtime_capture_launch_card_binding_blockers": recommended_runtime_capture_command_contract_summary.get("launch_card_binding_blockers"),
            "runtime_capture_launch_card_dir_provided": runtime_capture_launch_card_directory_manifest.get("provided"),
            "runtime_capture_launch_card_dir_ready": runtime_capture_launch_card_directory_manifest.get("directory_ready"),
            "runtime_capture_launch_card_dir_expected_card_count": runtime_capture_launch_card_directory_manifest.get("expected_card_count"),
            "runtime_capture_launch_card_dir_matched_card_count": runtime_capture_launch_card_directory_manifest.get("matched_card_count"),
            "runtime_capture_launch_card_dir_missing_card_count": runtime_capture_launch_card_directory_manifest.get("missing_card_count"),
            "recommended_runtime_capture_execution_manual_ready": recommended_execution_coverage_summary.get("manual_operator_capture_ready"),
            "recommended_runtime_capture_execution_automated_ready": recommended_execution_coverage_summary.get("automated_capture_ready"),
            "recommended_runtime_capture_execution_command_option_count": recommended_execution_coverage_summary.get("capture_command_option_count"),
            "recommended_runtime_capture_execution_operator_command_option_count": recommended_execution_coverage_summary.get("operator_command_option_count"),
            "recommended_runtime_capture_execution_metadata_command_option_count": recommended_execution_coverage_summary.get("metadata_command_option_count"),
            "recommended_runtime_capture_execution_artifacts_with_command_count": recommended_execution_coverage_summary.get("artifacts_with_capture_command_count"),
            "recommended_runtime_capture_execution_manual_capture_required_count": recommended_execution_coverage_summary.get("manual_capture_required_count"),
            "recommended_runtime_capture_execution_missing_capture_command_count": recommended_execution_coverage_summary.get("missing_capture_command_count"),
            "recommended_runtime_capture_execution_missing_item_count": recommended_execution_coverage_summary.get("missing_execution_item_count"),
            "all_request_runtime_capture_execution_manual_ready": all_execution_coverage_summary.get("manual_operator_capture_ready"),
            "all_request_runtime_capture_execution_automated_ready": all_execution_coverage_summary.get("automated_capture_ready"),
            "all_request_runtime_capture_execution_request_count": all_execution_coverage_summary.get("request_count"),
            "all_request_runtime_capture_execution_ready_request_count": all_execution_coverage_summary.get("manual_operator_capture_ready_count"),
            "all_request_runtime_capture_execution_automated_request_count": all_execution_coverage_summary.get("automated_capture_ready_count"),
            "all_request_runtime_capture_execution_pending_artifact_count": all_execution_coverage_summary.get("pending_artifact_count"),
            "all_request_runtime_capture_execution_artifact_execution_count": all_execution_coverage_summary.get("artifact_execution_count"),
            "all_request_runtime_capture_execution_command_option_count": all_execution_coverage_summary.get("capture_command_option_count"),
            "all_request_runtime_capture_execution_operator_command_option_count": all_execution_coverage_summary.get("operator_command_option_count"),
            "all_request_runtime_capture_execution_metadata_command_option_count": all_execution_coverage_summary.get("metadata_command_option_count"),
            "all_request_runtime_capture_execution_artifacts_with_command_count": all_execution_coverage_summary.get("artifacts_with_capture_command_count"),
            "all_request_runtime_capture_execution_manual_capture_required_count": all_execution_coverage_summary.get("manual_capture_required_count"),
            "all_request_runtime_capture_execution_missing_capture_command_count": all_execution_coverage_summary.get("missing_capture_command_count"),
            "all_request_runtime_capture_execution_missing_item_count": all_execution_coverage_summary.get("missing_execution_item_count"),
            "all_request_manual_capture_runbook_ready": manual_capture_runbook_summary.get("ready"),
            "all_request_manual_capture_runbook_request_count": manual_capture_runbook_summary.get("request_count"),
            "all_request_manual_capture_runbook_ready_request_count": manual_capture_runbook_summary.get("ready_request_count"),
            "all_request_manual_capture_runbook_manual_task_count": manual_capture_runbook_summary.get("manual_task_count"),
            "all_request_manual_capture_runbook_runtime_command_task_count": manual_capture_runbook_summary.get("runtime_command_task_count"),
            "all_request_manual_capture_runbook_validator_command_count": manual_capture_runbook_summary.get("validator_command_count"),
            "all_request_manual_capture_runbook_missing_item_count": manual_capture_runbook_summary.get("missing_item_count"),
            "all_request_manual_capture_runbook_missing_receipt_command_count": manual_capture_runbook_summary.get("missing_receipt_command_count"),
            "all_request_manual_capture_runbook_missing_validator_command_count": manual_capture_runbook_summary.get("missing_validator_command_count"),
            "all_request_manual_capture_runbook_missing_approval_command_count": manual_capture_runbook_summary.get("missing_approval_command_count"),
            "all_request_manual_capture_runbook_missing_source_request_path_count": manual_capture_runbook_summary.get("missing_source_request_path_count"),
            "all_request_manual_capture_runbook_missing_prompt_set_path_count": manual_capture_runbook_summary.get("missing_prompt_set_path_count"),
            "all_request_manual_capture_runbook_missing_explicit_approval_count": manual_capture_runbook_summary.get("missing_explicit_approval_count"),
            "all_request_manual_capture_runbook_missing_prompt_traffic_ack_count": manual_capture_runbook_summary.get("missing_prompt_traffic_ack_count"),
            "all_request_post_capture_intake_runbook_ready": post_capture_intake_runbook_summary.get("ready"),
            "all_request_post_capture_intake_runbook_request_count": post_capture_intake_runbook_summary.get("request_count"),
            "all_request_post_capture_intake_runbook_artifact_gate_count": post_capture_intake_runbook_summary.get("artifact_gate_count"),
            "all_request_post_capture_intake_runbook_ready_after_current_intake_count": post_capture_intake_runbook_summary.get("ready_after_current_intake_count"),
            "all_request_post_capture_intake_runbook_missing_after_current_intake_count": post_capture_intake_runbook_summary.get("missing_after_current_intake_count"),
            "all_request_post_capture_intake_runbook_validator_command_count": post_capture_intake_runbook_summary.get("validator_command_count"),
            "all_request_post_capture_intake_runbook_ready_to_update_bundle_count": post_capture_intake_runbook_summary.get("ready_to_update_bundle_count"),
            "all_request_post_capture_intake_runbook_phase4_candidate_count": post_capture_intake_runbook_summary.get("phase4_candidate_count"),
            "all_request_post_capture_intake_runbook_live_spike_candidate_count": post_capture_intake_runbook_summary.get("live_spike_candidate_count"),
            "all_request_post_capture_intake_runbook_missing_source_request_path_count": post_capture_intake_runbook_summary.get("missing_source_request_path_count"),
            "all_request_post_capture_intake_runbook_missing_prompt_set_path_count": post_capture_intake_runbook_summary.get("missing_prompt_set_path_count"),
            "all_request_post_capture_intake_runbook_missing_explicit_approval_count": post_capture_intake_runbook_summary.get("missing_explicit_approval_count"),
            "all_request_post_capture_intake_runbook_missing_prompt_traffic_ack_count": post_capture_intake_runbook_summary.get("missing_prompt_traffic_ack_count"),
            "all_request_post_capture_intake_runbook_missing_item_count": post_capture_intake_runbook_summary.get("missing_item_count"),
            "phase3_launch_card_library_ready": launch_card_library_summary.get("library_ready"),
            "phase3_launch_card_execution_ready": launch_card_library_summary.get("execution_ready"),
            "phase3_launch_card_count": launch_card_library_summary.get("card_count"),
            "phase3_launch_card_template_ready_count": launch_card_library_summary.get("template_ready_count"),
            "phase3_launch_card_model_plane_binding_ready_count": launch_card_library_summary.get("model_plane_binding_ready_count"),
            "phase3_launch_card_binding_ready_count": launch_card_library_summary.get("binding_ready_count"),
            "phase3_launch_card_runtime_command_ready_count": launch_card_library_summary.get("runtime_capture_command_ready_count"),
            "phase3_launch_card_task_count": launch_card_library_summary.get("task_count"),
            "phase3_launch_card_missing_runtime_command_count": launch_card_library_summary.get("missing_runtime_command_count"),
            "phase3_launch_card_binding_handoff_ready": launch_card_library_summary.get("binding_handoff_ready"),
            "phase3_launch_card_binding_handoff_task_count": launch_card_library_summary.get("binding_handoff_task_count"),
            "phase3_launch_card_binding_handoff_ready_count": launch_card_library_summary.get("binding_handoff_ready_count"),
            "phase3_launch_card_binding_handoff_missing_field_count": launch_card_library_summary.get("binding_handoff_missing_field_count"),
            "phase3_launch_card_unbound_task_count": launch_card_library_summary.get("unbound_task_count"),
            "phase3_launch_card_saved_handoff_artifacts_ready": launch_card_library_summary.get("saved_handoff_artifacts_ready"),
            "phase3_launch_card_saved_handoff_artifact_missing_count": launch_card_library_summary.get("saved_handoff_artifact_missing_count"),
            "phase3_launch_card_saved_handoff_artifact_drifted_count": launch_card_library_summary.get("saved_handoff_artifact_drifted_count"),
            "phase3_launch_card_blockers": launch_card_library_summary.get("blockers", []),
            "approval_manifest_parity_ready": approval_manifest_parity.get("ready"),
            "approval_manifest_parity_matched_request_count": approval_manifest_parity.get("matched_request_count"),
            "receipt_requirement_parity_ready": receipt_requirement_parity.get("ready"),
            "receipt_requirement_parity_matched_requirement_count": receipt_requirement_parity.get("matched_requirement_count"),
            "operator_queue_parity_ready": operator_queue_parity.get("ready"),
            "operator_queue_parity_matched_request_count": operator_queue_parity.get("matched_request_count"),
            "post_approval_capture_fill_parity_ready": post_approval_capture_fill_parity.get("ready"),
            "post_approval_capture_fill_parity_matched_artifact_count": post_approval_capture_fill_parity.get("matched_artifact_count"),
            "post_approval_capture_fill_parity_runtime_pending_artifact_count": post_approval_capture_fill_parity.get("runtime_pending_artifact_count"),
            "post_approval_capture_fill_parity_capture_result_fill_step_count": post_approval_capture_fill_parity.get("capture_result_fill_step_count"),
            "post_approval_capture_fill_parity_request_path_match": post_approval_capture_fill_parity.get("request_path_match"),
            "all_post_approval_capture_fill_parity_ready": all_post_approval_capture_fill_parity.get("ready"),
            "all_post_approval_capture_fill_parity_matched_request_count": all_post_approval_capture_fill_parity.get("matched_request_count"),
            "all_post_approval_capture_fill_parity_matched_artifact_count": all_post_approval_capture_fill_parity.get("matched_artifact_count"),
            "all_post_approval_capture_fill_parity_runtime_pending_artifact_count": all_post_approval_capture_fill_parity.get("runtime_pending_artifact_count"),
            "all_post_approval_capture_fill_parity_capture_result_fill_step_count": all_post_approval_capture_fill_parity.get("capture_result_fill_step_count"),
            "all_request_capture_queue_ready": all_capture_queue_summary.get("ready"),
            "all_request_capture_queue_request_count": all_capture_queue_summary.get("request_count"),
            "all_request_capture_queue_complete_fill_plan_count": all_capture_queue_summary.get("requests_with_complete_fill_plan"),
            "all_request_capture_queue_pending_artifact_count": all_capture_queue_summary.get("pending_artifact_count"),
            "all_request_capture_queue_validator_command_count": all_capture_queue_summary.get("validator_command_count"),
            "all_request_validator_command_manifest_ready": all_validator_command_summary.get("ready"),
            "all_request_validator_command_manifest_request_count": all_validator_command_summary.get("request_count"),
            "all_request_validator_command_manifest_artifact_entry_count": all_validator_command_summary.get("artifact_entry_count"),
            "all_request_validator_command_manifest_runtime_artifact_count": all_validator_command_summary.get("runtime_artifact_count"),
            "all_request_validator_command_manifest_future_artifact_count": all_validator_command_summary.get("future_artifact_count"),
            "all_request_validator_command_manifest_runtime_command_count": all_validator_command_summary.get("runtime_validator_command_count"),
            "all_request_validator_command_manifest_future_command_count": all_validator_command_summary.get("future_validator_command_count"),
            "all_request_validator_command_manifest_total_command_count": all_validator_command_summary.get("total_validator_command_count"),
            "all_request_downstream_handoff_manifest_ready": all_downstream_handoff_summary.get("ready"),
            "all_request_downstream_handoff_manifest_request_count": all_downstream_handoff_summary.get("request_count"),
            "all_request_downstream_handoff_policy_ready_count": all_downstream_handoff_summary.get("policy_candidate_trace_handoff_ready_count"),
            "all_request_downstream_handoff_dense_ready_count": all_downstream_handoff_summary.get("dense_fallback_capture_handoff_ready_count"),
            "all_request_downstream_handoff_live_ready_count": all_downstream_handoff_summary.get("live_capability_proof_handoff_ready_count"),
            "all_request_downstream_handoff_all_ready_count": all_downstream_handoff_summary.get("all_downstream_handoff_ready_count"),
            "all_request_receipt_fill_manifest_ready": all_receipt_fill_summary.get("ready"),
            "all_request_receipt_fill_manifest_request_count": all_receipt_fill_summary.get("request_count"),
            "all_request_receipt_fill_manifest_entry_count": all_receipt_fill_summary.get("entry_count"),
            "all_request_receipt_fill_manifest_ready_count": all_receipt_fill_summary.get("ready_count"),
            "all_request_receipt_fill_manifest_missing_count": all_receipt_fill_summary.get("missing_count"),
            "all_request_receipt_fill_manifest_approval_recorded_after_count": all_receipt_fill_summary.get("approval_recorded_after_count"),
            "all_request_receipt_fill_manifest_approval_missing_after_count": all_receipt_fill_summary.get("approval_missing_after_count"),
            "all_request_receipt_fill_manifest_blocked_after_approval_count": all_receipt_fill_summary.get("blocked_after_approval_count"),
            "all_request_receipt_fill_manifest_validator_command_count": all_receipt_fill_summary.get("validator_command_count"),
            "all_request_receipt_fill_manifest_candidate_router_trace_entry_count": compact_receipt_fill_artifact_counts(all_receipt_fill_summary, "candidate_router_trace").get("entry_count"),
            "all_request_receipt_fill_manifest_candidate_router_trace_ready_count": compact_receipt_fill_artifact_counts(all_receipt_fill_summary, "candidate_router_trace").get("ready_count"),
            "all_request_receipt_fill_manifest_candidate_router_trace_missing_count": compact_receipt_fill_artifact_counts(all_receipt_fill_summary, "candidate_router_trace").get("missing_count"),
            "all_request_receipt_fill_manifest_managed_output_entry_count": compact_receipt_fill_artifact_counts(all_receipt_fill_summary, "managed_output_summary_fill").get("entry_count"),
            "all_request_receipt_fill_manifest_managed_output_ready_count": compact_receipt_fill_artifact_counts(all_receipt_fill_summary, "managed_output_summary_fill").get("ready_count"),
            "all_request_receipt_fill_manifest_managed_output_missing_count": compact_receipt_fill_artifact_counts(all_receipt_fill_summary, "managed_output_summary_fill").get("missing_count"),
            "all_request_receipt_fill_manifest_dense_output_entry_count": compact_receipt_fill_artifact_counts(all_receipt_fill_summary, "dense_output_summary_fill").get("entry_count"),
            "all_request_receipt_fill_manifest_dense_output_ready_count": compact_receipt_fill_artifact_counts(all_receipt_fill_summary, "dense_output_summary_fill").get("ready_count"),
            "all_request_receipt_fill_manifest_dense_output_missing_count": compact_receipt_fill_artifact_counts(all_receipt_fill_summary, "dense_output_summary_fill").get("missing_count"),
            "all_request_receipt_validator_parity_ready": receipt_validator_parity.get("ready"),
            "all_request_receipt_validator_parity_runtime_artifact_count": receipt_validator_parity.get("runtime_artifact_count"),
            "all_request_receipt_validator_parity_receipt_entry_count": receipt_validator_parity.get("receipt_fill_entry_count"),
            "all_request_receipt_validator_parity_matched_artifact_count": receipt_validator_parity.get("matched_artifact_count"),
            "all_request_receipt_validator_parity_metadata_mismatch_count": len(receipt_validator_parity.get("metadata_mismatch_artifacts", [])) if isinstance(receipt_validator_parity.get("metadata_mismatch_artifacts"), list) else None,
            "all_request_receipt_validator_parity_missing_from_receipt_count": len(receipt_validator_parity.get("missing_from_receipt_fill_artifact_keys", [])) if isinstance(receipt_validator_parity.get("missing_from_receipt_fill_artifact_keys"), list) else None,
            "all_request_receipt_validator_parity_missing_from_runtime_count": len(receipt_validator_parity.get("missing_from_runtime_validator_artifact_keys", [])) if isinstance(receipt_validator_parity.get("missing_from_runtime_validator_artifact_keys"), list) else None,
            "all_request_receipt_fill_command_manifest_ready": receipt_command_summary.get("ready"),
            "all_request_receipt_fill_command_manifest_request_count": receipt_command_summary.get("request_count"),
            "all_request_receipt_fill_command_manifest_entry_count": receipt_command_summary.get("entry_count"),
            "all_request_receipt_fill_command_manifest_ready_receipt_count": receipt_command_summary.get("ready_receipt_count"),
            "all_request_receipt_fill_command_manifest_blocked_after_approval_count": receipt_command_summary.get("blocked_after_approval_count"),
            "all_request_receipt_fill_command_manifest_validator_command_count": receipt_command_summary.get("validator_command_count"),
            "all_request_receipt_fill_command_manifest_missing_command_entry_count": receipt_command_summary.get("missing_command_entry_count"),
            "all_request_receipt_fill_command_manifest_missing_receipt_path_count": receipt_command_summary.get("missing_receipt_path_count"),
            "all_request_receipt_fill_command_manifest_missing_source_request_path_count": receipt_command_summary.get("missing_source_request_path_count"),
            "all_request_receipt_fill_command_manifest_missing_prompt_set_path_count": receipt_command_summary.get("missing_prompt_set_path_count"),
            "all_request_receipt_fill_command_manifest_missing_explicit_approval_count": receipt_command_summary.get("missing_explicit_approval_count"),
            "all_request_receipt_fill_command_manifest_missing_prompt_traffic_ack_count": receipt_command_summary.get("missing_prompt_traffic_ack_count"),
            "phase3_blocker_closure_manifest_ready": blocker_closure_summary.get("ready"),
            "phase3_blocker_closure_mapping_ready": blocker_closure_summary.get("mapping_ready"),
            "phase3_blocker_closure_evidence_complete": blocker_closure_summary.get("evidence_complete"),
            "phase3_blocker_closure_closure_complete": blocker_closure_summary.get("closure_complete"),
            "phase3_blocker_closure_ready_scope": blocker_closure_summary.get("ready_scope"),
            "phase3_blocker_closure_reason_count": blocker_closure_summary.get("reason_count"),
            "phase3_blocker_closure_mapped_reason_count": blocker_closure_summary.get("mapped_reason_count"),
            "phase3_blocker_closure_unmapped_reason_count": blocker_closure_summary.get("unmapped_reason_count"),
            "phase3_blocker_closure_unresolved_reason_count": blocker_closure_summary.get("unresolved_reason_count"),
            "phase3_blocker_closure_runtime_capture_required_count": blocker_closure_summary.get("runtime_capture_required_count"),
            "phase3_blocker_closure_future_adapter_required_count": blocker_closure_summary.get("future_adapter_required_count"),
            "phase3_blocker_closure_handoff_ready_count": blocker_closure_summary.get("handoff_ready_count"),
            "phase3_blocker_closure_validator_command_count": blocker_closure_summary.get("validator_command_count"),
            "phase3_blocker_closure_missing_evidence_count": blocker_closure_summary.get("missing_evidence_count"),
            "phase3_blocker_evidence_ledger_ready": blocker_evidence_ledger_summary.get("ready"),
            "phase3_blocker_evidence_ledger_reason_count": blocker_evidence_ledger_summary.get("reason_count"),
            "phase3_blocker_evidence_ledger_row_count": blocker_evidence_ledger_summary.get("evidence_row_count"),
            "phase3_blocker_evidence_ledger_expected_missing_evidence_count": blocker_evidence_ledger_summary.get("expected_missing_evidence_count"),
            "phase3_blocker_evidence_ledger_runtime_capture_required_row_count": blocker_evidence_ledger_summary.get("runtime_capture_required_row_count"),
            "phase3_blocker_evidence_ledger_future_adapter_required_row_count": blocker_evidence_ledger_summary.get("future_adapter_required_row_count"),
            "phase3_blocker_evidence_ledger_receipt_bound_row_count": blocker_evidence_ledger_summary.get("receipt_bound_row_count"),
            "phase3_blocker_evidence_ledger_dense_output_row_count": blocker_evidence_ledger_summary.get("dense_output_row_count"),
            "phase3_blocker_evidence_ledger_live_proof_row_count": blocker_evidence_ledger_summary.get("live_proof_row_count"),
            "phase3_blocker_evidence_ledger_runtime_actuator_row_count": blocker_evidence_ledger_summary.get("runtime_actuator_row_count"),
            "phase3_blocker_evidence_ledger_validator_command_count": blocker_evidence_ledger_summary.get("validator_command_count"),
            "phase3_blocker_evidence_ledger_missing_item_count": blocker_evidence_ledger_summary.get("missing_item_count"),
            "phase3_blocker_resolution_queue_ready": blocker_resolution_queue_summary.get("ready"),
            "phase3_blocker_resolution_queue_work_package_count": blocker_resolution_queue_summary.get("work_package_count"),
            "phase3_blocker_resolution_queue_row_count": blocker_resolution_queue_summary.get("queue_row_count"),
            "phase3_blocker_resolution_queue_expected_ledger_row_count": blocker_resolution_queue_summary.get("expected_ledger_row_count"),
            "phase3_blocker_resolution_queue_runtime_capture_package_count": blocker_resolution_queue_summary.get("runtime_capture_package_count"),
            "phase3_blocker_resolution_queue_future_adapter_package_count": blocker_resolution_queue_summary.get("future_adapter_package_count"),
            "phase3_blocker_resolution_queue_runtime_capture_required_row_count": blocker_resolution_queue_summary.get("runtime_capture_required_row_count"),
            "phase3_blocker_resolution_queue_future_adapter_required_row_count": blocker_resolution_queue_summary.get("future_adapter_required_row_count"),
            "phase3_blocker_resolution_queue_receipt_bound_row_count": blocker_resolution_queue_summary.get("receipt_bound_row_count"),
            "phase3_blocker_resolution_queue_validator_command_count": blocker_resolution_queue_summary.get("validator_command_count"),
            "phase3_blocker_resolution_queue_completion_gate_count": blocker_resolution_queue_summary.get("completion_gate_count"),
            "phase3_blocker_resolution_queue_dependency_edge_count": blocker_resolution_queue_summary.get("dependency_edge_count"),
            "phase3_blocker_resolution_queue_missing_item_count": blocker_resolution_queue_summary.get("missing_item_count"),
            "phase3_blocker_resolution_queue_next_unblocked_work_package_id": blocker_resolution_queue_summary.get("next_unblocked_work_package_id"),
            "phase3_blocker_resolution_queue_next_unblocked_sequence_rank": blocker_resolution_queue_summary.get("next_unblocked_sequence_rank"),
            "phase3_blocker_resolution_queue_next_unblocked_operator_stage": blocker_resolution_queue_summary.get("next_unblocked_operator_stage"),
            "phase3_blocker_resolution_queue_next_unblocked_package_class": blocker_resolution_queue_summary.get("next_unblocked_package_class"),
            "phase3_blocker_resolution_queue_next_unblocked_row_count": blocker_resolution_queue_summary.get("next_unblocked_row_count"),
            "phase3_blocker_resolution_queue_next_unblocked_validator_command_count": blocker_resolution_queue_summary.get("next_unblocked_validator_command_count"),
            "phase3_blocker_resolution_queue_next_unblocked_completion_gate_count": blocker_resolution_queue_summary.get("next_unblocked_completion_gate_count"),
            "phase3_blocker_resolution_queue_next_unblocked_next_action": blocker_resolution_queue_summary.get("next_unblocked_next_action"),
            "phase3_next_unblocked_operator_handoff_ready": next_unblocked_operator_handoff.get("handoff_ready"),
            "phase3_next_unblocked_operator_handoff_work_order_ready": next_unblocked_operator_handoff.get("work_order_ready"),
            "phase3_next_unblocked_operator_handoff_work_order_artifact": next_unblocked_operator_handoff.get("work_order_artifact"),
            "phase3_next_unblocked_operator_handoff_receipt_template_ready": next_unblocked_operator_handoff.get("completion_receipt_template_ready"),
            "phase3_next_unblocked_operator_handoff_validation_command_ready": next_unblocked_operator_handoff.get("completion_receipt_validation_command_ready"),
            "phase3_next_unblocked_operator_handoff_work_order_advances_next_package": next_unblocked_operator_handoff.get("work_order_advances_next_package"),
            "capture_result_request_drift_free_count": capture_intake_summary.get("request_drift_free_count"),
            "capture_result_request_drifted_count": capture_intake_summary.get("request_drifted_count"),
            "capture_result_ready_for_operator_capture_flag_count": capture_intake_summary.get("request_ready_for_operator_capture_flag_count"),
            "capture_result_approved_but_capture_incomplete_request_count": capture_intake_summary.get("approved_but_capture_incomplete_request_count"),
            "capture_result_runtime_approval_missing_request_count": capture_intake_summary.get("runtime_approval_missing_request_count"),
            "capture_result_approval_rebuild_command_available_request_count": capture_intake_summary.get("approval_rebuild_command_available_request_count"),
            "capture_result_approval_rebuild_command_manifest_count": len(capture_intake_summary.get("approval_rebuild_command_manifest", [])) if isinstance(capture_intake_summary.get("approval_rebuild_command_manifest"), list) else None,
            "capture_result_approved_runtime_capture_pending_request_count": capture_intake_summary.get("approved_runtime_capture_pending_request_count"),
            "capture_result_capture_receipts_required_count": capture_intake_summary.get("capture_receipt_required_count"),
            "capture_result_capture_receipts_ready_count": capture_intake_summary.get("capture_receipt_ready_count"),
            "capture_result_candidate_trace_receipt_ready_count": compact_receipt_gate_coverage(capture_intake_summary.get("receipt_gate_coverage")).get("candidate_trace_receipt_ready_count"),
            "capture_result_managed_output_receipt_ready_count": compact_receipt_gate_coverage(capture_intake_summary.get("receipt_gate_coverage")).get("managed_output_receipt_ready_count"),
            "capture_result_dense_output_receipt_ready_count": compact_receipt_gate_coverage(capture_intake_summary.get("receipt_gate_coverage")).get("dense_output_receipt_ready_count"),
            "capture_result_live_capability_proof_ready_count": compact_receipt_gate_coverage(capture_intake_summary.get("receipt_gate_coverage")).get("live_capability_proof_ready_count"),
            "capture_result_all_capture_receipts_ready": compact_receipt_gate_coverage(capture_intake_summary.get("receipt_gate_coverage")).get("all_capture_receipts_ready"),
            "capture_result_all_output_receipt_bindings_ready": compact_receipt_gate_coverage(capture_intake_summary.get("receipt_gate_coverage")).get("all_output_receipt_bindings_ready"),
            "capture_result_receipt_fill_entry_count": capture_intake_summary.get("receipt_fill_entry_count"),
            "capture_result_receipt_fill_ready_count": capture_intake_summary.get("receipt_fill_ready_count"),
            "capture_result_receipt_fill_missing_count": capture_intake_summary.get("receipt_fill_missing_count"),
            "capture_result_receipt_fill_approval_missing_count": capture_intake_summary.get("receipt_fill_approval_missing_count"),
            "capture_result_receipt_fill_candidate_router_trace_entry_count": compact_receipt_fill_artifact_counts(capture_intake_summary.get("receipt_fill_manifest_summary"), "candidate_router_trace").get("entry_count"),
            "capture_result_receipt_fill_candidate_router_trace_ready_count": compact_receipt_fill_artifact_counts(capture_intake_summary.get("receipt_fill_manifest_summary"), "candidate_router_trace").get("ready_count"),
            "capture_result_receipt_fill_candidate_router_trace_missing_count": compact_receipt_fill_artifact_counts(capture_intake_summary.get("receipt_fill_manifest_summary"), "candidate_router_trace").get("missing_count"),
            "capture_result_receipt_fill_managed_output_entry_count": compact_receipt_fill_artifact_counts(capture_intake_summary.get("receipt_fill_manifest_summary"), "managed_output_summary_fill").get("entry_count"),
            "capture_result_receipt_fill_managed_output_ready_count": compact_receipt_fill_artifact_counts(capture_intake_summary.get("receipt_fill_manifest_summary"), "managed_output_summary_fill").get("ready_count"),
            "capture_result_receipt_fill_managed_output_missing_count": compact_receipt_fill_artifact_counts(capture_intake_summary.get("receipt_fill_manifest_summary"), "managed_output_summary_fill").get("missing_count"),
            "capture_result_receipt_fill_dense_output_entry_count": compact_receipt_fill_artifact_counts(capture_intake_summary.get("receipt_fill_manifest_summary"), "dense_output_summary_fill").get("entry_count"),
            "capture_result_receipt_fill_dense_output_ready_count": compact_receipt_fill_artifact_counts(capture_intake_summary.get("receipt_fill_manifest_summary"), "dense_output_summary_fill").get("ready_count"),
            "capture_result_receipt_fill_dense_output_missing_count": compact_receipt_fill_artifact_counts(capture_intake_summary.get("receipt_fill_manifest_summary"), "dense_output_summary_fill").get("missing_count"),
            "capture_result_approval_transition_preview_count": capture_intake_summary.get("approval_transition_preview_count"),
            "capture_result_approval_transition_ready_for_operator_count": capture_intake_summary.get("approval_transition_ready_for_operator_count"),
            "capture_result_approval_transition_ready_to_update_bundle_count": capture_intake_summary.get("approval_transition_ready_to_update_bundle_count"),
            "capture_result_post_approval_capture_fill_plan_count": capture_intake_summary.get("post_approval_capture_fill_plan_count"),
            "capture_result_post_approval_capture_fill_artifact_step_count": capture_intake_summary.get("post_approval_capture_fill_artifact_step_count"),
            "capture_result_post_approval_capture_fill_ready_count": capture_intake_summary.get("post_approval_capture_fill_ready_count"),
            "capture_result_post_approval_capture_fill_missing_count": capture_intake_summary.get("post_approval_capture_fill_missing_count"),
            "capture_result_post_approval_capture_fill_validator_command_count": capture_intake_summary.get("post_approval_capture_fill_validator_command_count"),
            "capture_result_post_approval_capture_fill_ready_to_update_bundle_count": capture_intake_summary.get("post_approval_capture_fill_ready_to_update_bundle_count"),
            "capture_result_missing_receipt_gate_request_count": capture_intake_summary.get("capture_receipt_missing_request_count"),
            "capture_result_output_receipt_binding_ready_count": capture_intake_summary.get("output_receipt_binding_ready_count"),
            "capture_result_ready_to_update_bundle_count": capture_intake_summary.get("ready_to_update_bundle_count"),
            "capture_result_phase4_candidate_count": capture_intake_summary.get("phase4_candidate_ready_count"),
            "no_go_reasons": decision_summary.get("no_go_reasons", []),
        },
        "safety_contract": [
            "packet aggregates local planner outputs only",
            "packet does not launch model servers",
            "packet does not run Docker",
            "packet does not load tensor values",
            "packet does not write packed expert stores",
            "packet does not mutate runtime residency",
            "packet does not send prompt traffic",
            "packet does not claim live expert paging",
        ],
        "next_actions": [
            "Use this packet as the Phase 3 audit surface before any Phase 4 adapter work.",
            "Replace fixture trace/inventory inputs with scanner-derived real-model artifacts when available.",
            "Attach a dense/full-runtime fallback comparison artifact for the same prompt set.",
            "Keep the live-spike decision no-go until residency observation, residency control, and cleanup proof exist.",
        ],
    }


def markdown_escape(value: Any) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def format_no_go_reasons(reasons: Any) -> str:
    if not isinstance(reasons, list) or not reasons:
        return "none"
    ids = [str(item.get("id")) for item in reasons if isinstance(item, dict) and item.get("id")]
    return ", ".join(ids) if ids else "none"


def format_markdown_report(summary: JSONDict) -> str:
    lines = [
        "# MoE Run Anyway Phase 3 Evidence Packet",
        "",
        "## Decision",
        "",
        f"- Valid: {summary['valid']}",
        f"- Packet ready: {summary['packet_ready']}",
        f"- Phase 3 complete: {summary['phase3_complete']}",
        f"- Decision: `{summary['decision']}`",
        "",
        "## Evidence Items",
        "",
        "| Evidence item | Status | Source | Summary |",
        "| --- | --- | --- | --- |",
    ]
    for item in summary["evidence_items"]:
        lines.append(
            "| "
            f"`{markdown_escape(item['id'])}` | "
            f"`{markdown_escape(item['status'])}` | "
            f"`{markdown_escape(item['source'])}` | "
            f"{markdown_escape(item['summary'])} |"
        )

    checklist = summary.get("promotion_checklist", [])
    if isinstance(checklist, list) and checklist:
        lines.extend(
            [
                "",
                "## Promotion Checklist",
                "",
                "| Gate | Status | Summary | Next action |",
                "| --- | --- | --- | --- |",
            ]
        )
        for item in checklist:
            if not isinstance(item, dict):
                continue
            lines.append(
                "| "
                f"`{markdown_escape(item.get('id', 'unknown'))}` | "
                f"`{markdown_escape(item.get('status', 'unknown'))}` | "
                f"{markdown_escape(item.get('summary', ''))} | "
                f"{markdown_escape(item.get('next_action', ''))} |"
            )

    blocker_closure_manifest = summary.get("phase3_blocker_closure_manifest")
    blocker_closure_summary = summary.get("phase3_blocker_closure_summary")
    if isinstance(blocker_closure_manifest, list) and blocker_closure_manifest:
        blocker_closure_summary = blocker_closure_summary if isinstance(blocker_closure_summary, dict) else {}
        lines.extend(
            [
                "",
                "## Phase 3 Blocker Closure",
                "",
                f"- Ready: `{markdown_escape(blocker_closure_summary.get('ready'))}`",
                f"- Mapping ready: `{markdown_escape(blocker_closure_summary.get('mapping_ready'))}`",
                f"- Evidence complete: `{markdown_escape(blocker_closure_summary.get('evidence_complete'))}`",
                f"- Ready scope: `{markdown_escape(blocker_closure_summary.get('ready_scope'))}`",
                f"- Missing evidence: `{markdown_escape(blocker_closure_summary.get('missing_evidence_count'))}`",
                f"- Reasons: `{markdown_escape(blocker_closure_summary.get('reason_count'))}`",
                f"- Mapped reasons: `{markdown_escape(blocker_closure_summary.get('mapped_reason_count'))}`",
                f"- Unresolved reasons: `{markdown_escape(blocker_closure_summary.get('unresolved_reason_count'))}`",
                f"- Runtime-capture required: `{markdown_escape(blocker_closure_summary.get('runtime_capture_required_count'))}`",
                f"- Future-adapter required: `{markdown_escape(blocker_closure_summary.get('future_adapter_required_count'))}`",
                f"- Validator commands: `{markdown_escape(blocker_closure_summary.get('validator_command_count'))}`",
                "",
                "| Reason | Gate | Status | Approval stage | Handoff | Validators | Primary path | Next action |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for item in blocker_closure_manifest:
            if not isinstance(item, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    markdown_escape(value)
                    for value in (
                        item.get("reason_id") or "unknown",
                        item.get("closure_gate") or "missing",
                        item.get("closure_status") or "unknown",
                        item.get("approval_stage") or "unknown",
                        item.get("handoff_ready"),
                        item.get("validator_command_count"),
                        item.get("primary_path") or "missing",
                        item.get("next_action") or "missing",
                    )
                )
                + " |"
            )
    blocker_ledger_manifest = summary.get("phase3_blocker_evidence_ledger_manifest")
    blocker_ledger_summary = summary.get("phase3_blocker_evidence_ledger_summary")
    if isinstance(blocker_ledger_manifest, dict) and blocker_ledger_manifest:
        blocker_ledger_summary = blocker_ledger_summary if isinstance(blocker_ledger_summary, dict) else {}
        lines.extend(
            [
                "",
                "## Phase 3 Blocker Evidence Ledger",
                "",
                f"- Ready: `{markdown_escape(blocker_ledger_summary.get('ready'))}`",
                f"- Rows: `{markdown_escape(blocker_ledger_summary.get('evidence_row_count'))}` / `{markdown_escape(blocker_ledger_summary.get('expected_missing_evidence_count'))}`",
                f"- Reasons: `{markdown_escape(blocker_ledger_summary.get('reason_count'))}`",
                f"- Runtime-capture rows: `{markdown_escape(blocker_ledger_summary.get('runtime_capture_required_row_count'))}`",
                f"- Future-adapter rows: `{markdown_escape(blocker_ledger_summary.get('future_adapter_required_row_count'))}`",
                f"- Receipt-bound rows: `{markdown_escape(blocker_ledger_summary.get('receipt_bound_row_count'))}`",
                f"- Dense output rows: `{markdown_escape(blocker_ledger_summary.get('dense_output_row_count'))}`",
                f"- Live proof rows: `{markdown_escape(blocker_ledger_summary.get('live_proof_row_count'))}`",
                f"- Runtime actuator rows: `{markdown_escape(blocker_ledger_summary.get('runtime_actuator_row_count'))}`",
                f"- Validator commands: `{markdown_escape(blocker_ledger_summary.get('validator_command_count'))}`",
                f"- Missing ledger items: `{markdown_escape(blocker_ledger_summary.get('missing_item_count'))}`",
                "",
                "| Reason | Evidence kind | Evidence id | Artifact path | Receipt path | Prompt/section | Validators |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        rows = blocker_ledger_manifest.get("evidence_rows") if isinstance(blocker_ledger_manifest.get("evidence_rows"), list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            prompt_or_section = row.get("prompt_id") or row.get("section") or row.get("blocker_id") or ""
            lines.append(
                "| "
                + " | ".join(
                    markdown_escape(value)
                    for value in (
                        row.get("reason_id") or "unknown",
                        row.get("evidence_kind") or "unknown",
                        row.get("evidence_id") or "unknown",
                        row.get("artifact_path") or "missing",
                        row.get("receipt_path") or "missing",
                        prompt_or_section,
                        row.get("validator_command_count"),
                    )
                )
                + " |"
            )
    blocker_resolution_queue_manifest = summary.get("phase3_blocker_resolution_queue_manifest")
    blocker_resolution_queue_summary = summary.get("phase3_blocker_resolution_queue_summary")
    if isinstance(blocker_resolution_queue_manifest, dict) and blocker_resolution_queue_manifest:
        blocker_resolution_queue_summary = blocker_resolution_queue_summary if isinstance(blocker_resolution_queue_summary, dict) else {}
        lines.extend(
            [
                "",
                "## Phase 3 Blocker Resolution Queue",
                "",
                f"- Ready: `{markdown_escape(blocker_resolution_queue_summary.get('ready'))}`",
                f"- Work packages: `{markdown_escape(blocker_resolution_queue_summary.get('work_package_count'))}`",
                f"- Rows: `{markdown_escape(blocker_resolution_queue_summary.get('queue_row_count'))}` / `{markdown_escape(blocker_resolution_queue_summary.get('expected_ledger_row_count'))}`",
                f"- Runtime-capture packages: `{markdown_escape(blocker_resolution_queue_summary.get('runtime_capture_package_count'))}`",
                f"- Future-adapter packages: `{markdown_escape(blocker_resolution_queue_summary.get('future_adapter_package_count'))}`",
                f"- Validator commands: `{markdown_escape(blocker_resolution_queue_summary.get('validator_command_count'))}`",
                f"- Completion gates: `{markdown_escape(blocker_resolution_queue_summary.get('completion_gate_count'))}`",
                f"- Dependency edges: `{markdown_escape(blocker_resolution_queue_summary.get('dependency_edge_count'))}`",
                f"- Missing queue items: `{markdown_escape(blocker_resolution_queue_summary.get('missing_item_count'))}`",
                f"- Next unblocked package: `{markdown_escape(blocker_resolution_queue_summary.get('next_unblocked_work_package_id'))}`",
                f"- Next unblocked stage: `{markdown_escape(blocker_resolution_queue_summary.get('next_unblocked_operator_stage'))}`",
                f"- Next unblocked action: `{markdown_escape(blocker_resolution_queue_summary.get('next_unblocked_next_action'))}`",
                "",
                "| Rank | Work package | Depends on | Stage | Rows | Runtime rows | Future rows | Validators | Gates | Next action |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        packages = blocker_resolution_queue_manifest.get("work_packages") if isinstance(blocker_resolution_queue_manifest.get("work_packages"), list) else []
        for package in packages:
            if not isinstance(package, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    markdown_escape(value)
                    for value in (
                        package.get("sequence_rank"),
                        package.get("work_package_id") or "unknown",
                        ", ".join(list_of_strings(package.get("depends_on_work_package_ids"))) or "none",
                        package.get("operator_stage") or "unknown",
                        package.get("row_count"),
                        package.get("runtime_capture_required_row_count"),
                        package.get("future_adapter_required_row_count"),
                        package.get("validator_command_count"),
                        package.get("completion_gate_count"),
                        package.get("next_action") or "missing",
                    )
                )
                + " |"
            )
    launch_card_library = summary.get("phase3_launch_card_library_summary")
    if isinstance(launch_card_library, dict) and launch_card_library:
        recommended_launch_card = summary.get("recommended_runtime_capture_launch_card_template")
        recommended_launch_card = recommended_launch_card if isinstance(recommended_launch_card, dict) else {}
        lines.extend(
            [
                "",
                "## Phase 3 Launch-Card Library",
                "",
                f"- Library ready: `{markdown_escape(launch_card_library.get('library_ready'))}`",
                f"- Execution ready: `{markdown_escape(launch_card_library.get('execution_ready'))}`",
                f"- Cards: `{markdown_escape(launch_card_library.get('card_count'))}`",
                f"- Template-ready cards: `{markdown_escape(launch_card_library.get('template_ready_count'))}`",
                f"- Model Plane binding-ready cards: `{markdown_escape(launch_card_library.get('model_plane_binding_ready_count'))}`",
                f"- Runtime-command-ready cards: `{markdown_escape(launch_card_library.get('runtime_capture_command_ready_count'))}`",
                f"- Tasks: `{markdown_escape(launch_card_library.get('task_count'))}`",
                f"- Missing runtime commands: `{markdown_escape(launch_card_library.get('missing_runtime_command_count'))}`",
                f"- Binding handoff ready: `{markdown_escape(launch_card_library.get('binding_handoff_ready'))}`",
                f"- Binding handoff tasks: `{markdown_escape(launch_card_library.get('binding_handoff_task_count'))}`",
                f"- Binding handoff missing fields: `{markdown_escape(launch_card_library.get('binding_handoff_missing_field_count'))}`",
                f"- Unbound tasks: `{markdown_escape(launch_card_library.get('unbound_task_count'))}`",
                f"- Saved handoff artifacts ready: `{markdown_escape(launch_card_library.get('saved_handoff_artifacts_ready'))}`",
                f"- Saved handoff artifacts missing: `{markdown_escape(launch_card_library.get('saved_handoff_artifact_missing_count'))}`",
                f"- Saved handoff artifacts drifted: `{markdown_escape(launch_card_library.get('saved_handoff_artifact_drifted_count'))}`",
                f"- Recommended template path: `{markdown_escape(recommended_launch_card.get('launch_card_path') or 'missing')}`",
                f"- Recommended template ready: `{markdown_escape(recommended_launch_card.get('template_ready'))}`",
                f"- Recommended binding handoff ready: `{markdown_escape(recommended_launch_card.get('binding_handoff_ready'))}`",
                f"- Recommended template unbound tasks: `{markdown_escape(recommended_launch_card.get('unbound_task_count'))}`",
                "",
                "| Card | Template | Model Plane | Binding | Runtime Commands | Missing Commands |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        cards = launch_card_library.get("cards") if isinstance(launch_card_library.get("cards"), list) else []
        for card in cards:
            if not isinstance(card, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    markdown_escape(value)
                    for value in (
                        card.get("request_name") or card.get("bundle_path") or "unknown",
                        card.get("template_ready"),
                        card.get("model_plane_binding_ready"),
                        card.get("binding_ready"),
                        card.get("runtime_capture_command_ready"),
                        card.get("missing_runtime_command_count"),
                    )
                )
                + " |"
            )
    all_queue = summary.get("all_request_capture_queue_manifest")
    all_queue_summary = summary.get("all_request_capture_queue_summary")
    if isinstance(all_queue, list) and all_queue:
        all_queue_summary = all_queue_summary if isinstance(all_queue_summary, dict) else {}
        lines.extend(
            [
                "",
                "## All-Request Capture Queue",
                "",
                f"- Ready: `{markdown_escape(all_queue_summary.get('ready'))}`",
                f"- Requests: `{markdown_escape(all_queue_summary.get('request_count'))}`",
                f"- Pending artifact fills: `{markdown_escape(all_queue_summary.get('pending_artifact_count'))}`",
                f"- Validator commands: `{markdown_escape(all_queue_summary.get('validator_command_count'))}`",
                "",
                "| Rank | Request | Status | Next artifact | Pending fills | Validators | Request path |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for item in all_queue:
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
                        item.get("next_artifact_id") or "none",
                        item.get("pending_artifact_count"),
                        item.get("validator_command_count"),
                        item.get("request_path") or "missing",
                    )
                )
                + " |"
            )
    all_validator_commands = summary.get("all_request_validator_command_manifest")
    all_validator_summary = summary.get("all_request_validator_command_summary")
    if isinstance(all_validator_commands, list) and all_validator_commands:
        all_validator_summary = all_validator_summary if isinstance(all_validator_summary, dict) else {}
        lines.extend(
            [
                "",
                "## All-Request Validator Commands",
                "",
                f"- Ready: `{markdown_escape(all_validator_summary.get('ready'))}`",
                f"- Requests: `{markdown_escape(all_validator_summary.get('request_count'))}`",
                f"- Runtime artifacts: `{markdown_escape(all_validator_summary.get('runtime_artifact_count'))}`",
                f"- Runtime validator commands: `{markdown_escape(all_validator_summary.get('runtime_validator_command_count'))}`",
                f"- Future adapter artifacts: `{markdown_escape(all_validator_summary.get('future_artifact_count'))}`",
                f"- Future adapter validator commands: `{markdown_escape(all_validator_summary.get('future_validator_command_count'))}`",
                f"- Total validator commands: `{markdown_escape(all_validator_summary.get('total_validator_command_count'))}`",
                "",
                "| Rank | Request | Artifact | Stage | Status | Validators | Path |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for item in all_validator_commands:
            if not isinstance(item, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    markdown_escape(value)
                    for value in (
                        item.get("rank"),
                        item.get("request_name") or "missing",
                        item.get("artifact_id") or "unknown",
                        item.get("artifact_stage") or "unknown",
                        item.get("status") or "unknown",
                        item.get("validator_command_count"),
                        item.get("path") or "missing",
                    )
                )
                + " |"
            )
    all_downstream_handoffs = summary.get("all_request_downstream_handoff_manifest")
    all_downstream_summary = summary.get("all_request_downstream_handoff_summary")
    if isinstance(all_downstream_handoffs, list) and all_downstream_handoffs:
        all_downstream_summary = all_downstream_summary if isinstance(all_downstream_summary, dict) else {}
        lines.extend(
            [
                "",
                "## All-Request Downstream Handoffs",
                "",
                f"- Ready: `{markdown_escape(all_downstream_summary.get('ready'))}`",
                f"- Requests: `{markdown_escape(all_downstream_summary.get('request_count'))}`",
                f"- Policy-candidate handoffs ready: `{markdown_escape(all_downstream_summary.get('policy_candidate_trace_handoff_ready_count'))}`",
                f"- Dense-fallback handoffs ready: `{markdown_escape(all_downstream_summary.get('dense_fallback_capture_handoff_ready_count'))}`",
                f"- Live-proof handoffs ready: `{markdown_escape(all_downstream_summary.get('live_capability_proof_handoff_ready_count'))}`",
                f"- All downstream handoffs ready: `{markdown_escape(all_downstream_summary.get('all_downstream_handoff_ready_count'))}`",
                "",
                "| Rank | Request | Policy | Dense | Live | Candidate trace | Request path |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for item in all_downstream_handoffs:
            if not isinstance(item, dict):
                continue
            policy = item.get("policy_candidate_trace") if isinstance(item.get("policy_candidate_trace"), dict) else {}
            lines.append(
                "| "
                + " | ".join(
                    markdown_escape(value)
                    for value in (
                        item.get("rank"),
                        item.get("request_name") or "missing",
                        item.get("policy_candidate_trace_handoff_ready"),
                        item.get("dense_fallback_capture_handoff_ready"),
                        item.get("live_capability_proof_handoff_ready"),
                        policy.get("candidate_trace_path") or "missing",
                        item.get("request_path") or "missing",
                    )
                )
                + " |"
            )
    all_receipt_fills = summary.get("all_request_receipt_fill_manifest")
    all_receipt_summary = summary.get("all_request_receipt_fill_summary")
    if isinstance(all_receipt_fills, list) and all_receipt_fills:
        all_receipt_summary = all_receipt_summary if isinstance(all_receipt_summary, dict) else {}
        lines.extend(
            [
                "",
                "## All-Request Receipt Fills",
                "",
                f"- Ready: `{markdown_escape(all_receipt_summary.get('ready'))}`",
                f"- Requests: `{markdown_escape(all_receipt_summary.get('request_count'))}`",
                f"- Receipt-fill entries: `{markdown_escape(all_receipt_summary.get('entry_count'))}`",
                f"- Ready receipts: `{markdown_escape(all_receipt_summary.get('ready_count'))}`",
                f"- Missing receipts: `{markdown_escape(all_receipt_summary.get('missing_count'))}`",
                f"- Approvals recorded after approval: `{markdown_escape(all_receipt_summary.get('approval_recorded_after_count'))}`",
                f"- Blocked after approval: `{markdown_escape(all_receipt_summary.get('blocked_after_approval_count'))}`",
                f"- Validator commands: `{markdown_escape(all_receipt_summary.get('validator_command_count'))}`",
                "",
                "| Rank | Request | Artifact | Receipt ready | After approval | Validators | Receipt path |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for item in all_receipt_fills:
            if not isinstance(item, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    markdown_escape(value)
                    for value in (
                        item.get("rank"),
                        item.get("request_name") or "missing",
                        item.get("artifact_id") or "unknown",
                        item.get("receipt_ready"),
                        item.get("fill_status_after_approval") or "unknown",
                        item.get("validator_command_count"),
                        item.get("receipt_path") or "missing",
                    )
                )
                + " |"
            )
    receipt_validator_parity = summary.get("all_request_receipt_validator_parity_summary")
    if isinstance(receipt_validator_parity, dict) and receipt_validator_parity:
        metadata_mismatches = receipt_validator_parity.get("metadata_mismatch_artifacts")
        missing_from_receipts = receipt_validator_parity.get("missing_from_receipt_fill_artifact_keys")
        missing_from_runtime = receipt_validator_parity.get("missing_from_runtime_validator_artifact_keys")
        lines.extend(
            [
                "",
                "## All-Request Receipt/Validator Parity",
                "",
                f"- Ready: `{markdown_escape(receipt_validator_parity.get('ready'))}`",
                f"- Runtime artifacts: `{markdown_escape(receipt_validator_parity.get('runtime_artifact_count'))}`",
                f"- Receipt-fill entries: `{markdown_escape(receipt_validator_parity.get('receipt_fill_entry_count'))}`",
                f"- Matched artifacts: `{markdown_escape(receipt_validator_parity.get('matched_artifact_count'))}`",
                f"- Metadata mismatches: `{markdown_escape(len(metadata_mismatches) if isinstance(metadata_mismatches, list) else 0)}`",
                f"- Missing from receipt fills: `{markdown_escape(len(missing_from_receipts) if isinstance(missing_from_receipts, list) else 0)}`",
                f"- Missing from runtime validators: `{markdown_escape(len(missing_from_runtime) if isinstance(missing_from_runtime, list) else 0)}`",
            ]
        )
    receipt_command_manifest = summary.get("all_request_receipt_fill_command_manifest")
    receipt_command_summary = summary.get("all_request_receipt_fill_command_summary")
    if isinstance(receipt_command_manifest, list) and receipt_command_manifest:
        receipt_command_summary = receipt_command_summary if isinstance(receipt_command_summary, dict) else {}
        lines.extend(
            [
                "",
                "## All-Request Receipt Fill Commands",
                "",
                f"- Ready: `{markdown_escape(receipt_command_summary.get('ready'))}`",
                f"- Requests: `{markdown_escape(receipt_command_summary.get('request_count'))}`",
                f"- Entries: `{markdown_escape(receipt_command_summary.get('entry_count'))}`",
                f"- Validator commands: `{markdown_escape(receipt_command_summary.get('validator_command_count'))}`",
                f"- Missing command entries: `{markdown_escape(receipt_command_summary.get('missing_command_entry_count'))}`",
                f"- Missing receipt paths: `{markdown_escape(receipt_command_summary.get('missing_receipt_path_count'))}`",
                "",
                "| Rank | Request | Artifact | Validators | Receipt ready | Receipt path |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for item in receipt_command_manifest:
            if not isinstance(item, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    markdown_escape(value)
                    for value in (
                        item.get("rank"),
                        item.get("request_name") or "missing",
                        item.get("artifact_id") or "unknown",
                        item.get("validator_command_count"),
                        item.get("receipt_ready"),
                        item.get("receipt_path") or "missing",
                    )
                )
                + " |"
            )
    recommendation = summary.get("recommended_runtime_capture_request")
    if isinstance(recommendation, dict):
        missing = recommendation.get("missing_approval_keys") if isinstance(recommendation.get("missing_approval_keys"), list) else []
        pending = recommendation.get("pending_artifact_ids") if isinstance(recommendation.get("pending_artifact_ids"), list) else []
        lines.extend(
            [
                "",
                "## Recommended Runtime Capture",
                "",
                f"- Request: `{markdown_escape(recommendation.get('request_name') or 'missing')}`",
                f"- Status: `{markdown_escape(recommendation.get('status') or 'unknown')}`",
                f"- Prompt set: `{markdown_escape(recommendation.get('prompt_set_path') or 'missing')}`",
                f"- Missing approvals: `{markdown_escape(', '.join(str(item) for item in missing) if missing else 'none')}`",
                f"- Next artifact: `{markdown_escape(recommendation.get('next_artifact_id') or 'none')}`",
                f"- Pending artifacts: `{markdown_escape(', '.join(str(item) for item in pending) if pending else 'none')}`",
                f"- Validator commands: `{markdown_escape(recommendation.get('validator_command_count', 0))}`",
                "",
                "| Step | Stage | Status | Path / approval |",
                "| --- | --- | --- | --- |",
            ]
        )
        sequence = recommendation.get("capture_sequence") if isinstance(recommendation.get("capture_sequence"), list) else []
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
        command_manifest = recommendation.get("validator_command_manifest")
        if isinstance(command_manifest, list) and command_manifest:
            lines.extend(["", "### Recommended Validator Commands", ""])
            for artifact in command_manifest:
                if not isinstance(artifact, dict):
                    continue
                lines.append(
                    f"- `{markdown_escape(artifact.get('artifact_id') or 'unknown')}` "
                    f"({markdown_escape(artifact.get('artifact_stage') or 'unknown')}, "
                    f"{markdown_escape(artifact.get('status') or 'unknown')}, "
                    f"`{markdown_escape(artifact.get('path') or 'missing')}`)"
                )
                commands = artifact.get("validator_commands") if isinstance(artifact.get("validator_commands"), list) else []
                for command in commands:
                    lines.append("```sh")
                    lines.append(command)
                    lines.append("```")

    post_approval_preview = summary.get("recommended_post_approval_preview")
    if isinstance(post_approval_preview, dict):
        pending = post_approval_preview.get("pending_artifact_ids") if isinstance(post_approval_preview.get("pending_artifact_ids"), list) else []
        records = post_approval_preview.get("records_approval_keys") if isinstance(post_approval_preview.get("records_approval_keys"), list) else []
        lines.extend(
            [
                "",
                "### Runtime Capture Post-Approval Preview",
                "",
                f"- Preview only: `{markdown_escape(post_approval_preview.get('preview_only'))}`",
                f"- Mutates request: `{markdown_escape(post_approval_preview.get('mutates_request'))}`",
                f"- Valid: `{markdown_escape(post_approval_preview.get('valid'))}`",
                f"- Status after approval: `{markdown_escape(post_approval_preview.get('status') or 'unknown')}`",
                f"- Ready for operator capture after approval: `{markdown_escape(post_approval_preview.get('ready_for_operator_capture'))}`",
                f"- Capture complete after approval: `{markdown_escape(post_approval_preview.get('capture_complete'))}`",
                f"- Records approvals: `{markdown_escape(', '.join(str(item) for item in records) if records else 'none')}`",
                f"- Pending artifacts after approval: `{markdown_escape(', '.join(str(item) for item in pending) if pending else 'none')}`",
                f"- Next artifact after approval: `{markdown_escape(post_approval_preview.get('next_artifact_id') or 'none')}`",
            ]
        )

    preflight = summary.get("recommended_runtime_capture_preflight_manifest")
    preflight_summary = summary.get("recommended_runtime_capture_preflight_summary")
    if isinstance(preflight, dict) and preflight:
        preflight_summary = preflight_summary if isinstance(preflight_summary, dict) else {}
        missing_items = preflight_summary.get("missing_preflight_items") if isinstance(preflight_summary.get("missing_preflight_items"), list) else []
        artifact_checks = preflight.get("artifact_checks") if isinstance(preflight.get("artifact_checks"), list) else []
        lines.extend(
            [
                "",
                "## Recommended Runtime Capture Preflight",
                "",
                f"- Ready: `{markdown_escape(preflight_summary.get('ready'))}`",
                f"- Request: `{markdown_escape(preflight_summary.get('request_name') or 'missing')}`",
                f"- Approval required: `{markdown_escape(preflight_summary.get('approval_required'))}`",
                f"- Pending artifacts: `{markdown_escape(preflight_summary.get('pending_artifact_count'))}`",
                f"- Receipt entries: `{markdown_escape(preflight_summary.get('receipt_entry_count'))}`",
                f"- Validator commands: `{markdown_escape(preflight_summary.get('validator_command_count'))}`",
                f"- Runtime closure reasons: `{markdown_escape(preflight_summary.get('runtime_closure_reason_count'))}`",
                f"- Missing preflight items: `{markdown_escape(', '.join(str(item) for item in missing_items) if missing_items else 'none')}`",
                "",
                "| Artifact | Artifact path | Receipt path | Validators | Receipt ready |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for artifact in artifact_checks:
            if not isinstance(artifact, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    markdown_escape(value)
                    for value in (
                        artifact.get("artifact_id") or "unknown",
                        artifact.get("artifact_path") or "missing",
                        artifact.get("receipt_path") or "missing",
                        artifact.get("validator_command_count"),
                        artifact.get("receipt_ready"),
                    )
                )
                + " |"
            )

    command_contract = summary.get("recommended_runtime_capture_command_contract")
    command_contract_summary = summary.get("recommended_runtime_capture_command_contract_summary")
    if isinstance(command_contract, dict) and command_contract:
        command_contract_summary = command_contract_summary if isinstance(command_contract_summary, dict) else {}
        missing_runtime_commands = command_contract_summary.get("missing_runtime_command_artifact_ids") if isinstance(command_contract_summary.get("missing_runtime_command_artifact_ids"), list) else []
        capture_tasks = command_contract.get("capture_tasks") if isinstance(command_contract.get("capture_tasks"), list) else []
        lines.extend(
            [
                "",
                "## Recommended Runtime Capture Command Contract",
                "",
                f"- Contract ready: `{markdown_escape(command_contract_summary.get('ready'))}`",
                f"- Runtime commands ready: `{markdown_escape(command_contract_summary.get('runtime_capture_command_ready'))}`",
                f"- Planned captures: `{markdown_escape(command_contract_summary.get('planned_capture_count'))}`",
                f"- Runtime command options: `{markdown_escape(command_contract_summary.get('runtime_command_option_count'))}`",
                f"- Missing runtime commands: `{markdown_escape(command_contract_summary.get('missing_runtime_command_count'))}`",
                f"- Missing runtime command artifacts: `{markdown_escape(', '.join(str(item) for item in missing_runtime_commands) if missing_runtime_commands else 'none')}`",
                f"- Launch-card binding ready: `{markdown_escape(command_contract_summary.get('launch_card_binding_ready'))}`",
                f"- Launch-card binding valid: `{markdown_escape(command_contract_summary.get('launch_card_binding_valid'))}`",
                f"- Launch-card bound tasks: `{markdown_escape(command_contract_summary.get('launch_card_binding_bound_task_count'))}`",
                f"- Launch-card command options: `{markdown_escape(command_contract_summary.get('launch_card_binding_command_option_count'))}`",
                f"- Launch-card blockers: `{markdown_escape(', '.join(str(item) for item in command_contract_summary.get('launch_card_binding_blockers', [])) if command_contract_summary.get('launch_card_binding_blockers') else 'none')}`",
                "",
                "| Artifact | Capture kind | Artifact path | Receipt path | Command ready | Validators |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for task in capture_tasks:
            if not isinstance(task, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    markdown_escape(value)
                    for value in (
                        task.get("artifact_id") or "unknown",
                        task.get("capture_kind") or "unknown",
                        task.get("artifact_path") or "missing",
                        task.get("receipt_path") or "missing",
                        task.get("command_binding_ready"),
                        task.get("validator_command_count"),
                    )
                )
                + " |"
            )
    execution = summary.get("recommended_runtime_capture_execution_coverage_manifest")
    execution_summary = summary.get("recommended_runtime_capture_execution_coverage_summary")
    if isinstance(execution, dict) and execution:
        execution_summary = execution_summary if isinstance(execution_summary, dict) else {}
        missing_command_artifacts = execution_summary.get("missing_capture_command_artifact_ids") if isinstance(execution_summary.get("missing_capture_command_artifact_ids"), list) else []
        missing_execution_items = execution_summary.get("missing_execution_items") if isinstance(execution_summary.get("missing_execution_items"), list) else []
        artifact_execution = execution.get("artifact_execution") if isinstance(execution.get("artifact_execution"), list) else []
        lines.extend(
            [
                "",
                "## Recommended Runtime Capture Execution Coverage",
                "",
                f"- Manual operator capture ready: `{markdown_escape(execution_summary.get('manual_operator_capture_ready'))}`",
                f"- Automated capture ready: `{markdown_escape(execution_summary.get('automated_capture_ready'))}`",
                f"- Pending artifacts: `{markdown_escape(execution_summary.get('pending_artifact_count'))}`",
                f"- Runtime capture command options: `{markdown_escape(execution_summary.get('capture_command_option_count'))}`",
                f"- Operator command options: `{markdown_escape(execution_summary.get('operator_command_option_count'))}`",
                f"- Metadata command options: `{markdown_escape(execution_summary.get('metadata_command_option_count'))}`",
                f"- Artifacts with runtime capture commands: `{markdown_escape(execution_summary.get('artifacts_with_capture_command_count'))}`",
                f"- Manual capture required: `{markdown_escape(execution_summary.get('manual_capture_required_count'))}`",
                f"- Missing capture command artifacts: `{markdown_escape(', '.join(str(item) for item in missing_command_artifacts) if missing_command_artifacts else 'none')}`",
                f"- Missing execution items: `{markdown_escape(', '.join(str(item) for item in missing_execution_items) if missing_execution_items else 'none')}`",
                "",
                "| Artifact | Mode | Runtime commands | Operator commands | Metadata commands | Status after approval | Validators | Receipt ready |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for artifact in artifact_execution:
            if not isinstance(artifact, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    markdown_escape(value)
                    for value in (
                        artifact.get("artifact_id") or "unknown",
                        artifact.get("capture_mode") or "unknown",
                        artifact.get("capture_command_option_count"),
                        artifact.get("operator_command_option_count"),
                        artifact.get("metadata_command_option_count"),
                        artifact.get("status_after_approval") or "unknown",
                        artifact.get("validator_command_count"),
                        artifact.get("receipt_ready"),
                    )
                )
                + " |"
            )

    all_execution = summary.get("all_request_runtime_capture_execution_coverage_manifest")
    all_execution_summary = summary.get("all_request_runtime_capture_execution_coverage_summary")
    if isinstance(all_execution, dict) and all_execution:
        all_execution_summary = all_execution_summary if isinstance(all_execution_summary, dict) else {}
        lines.extend(
            [
                "",
                "## All-Request Runtime Capture Execution Coverage",
                "",
                f"- Manual operator capture ready: `{markdown_escape(all_execution_summary.get('manual_operator_capture_ready'))}`",
                f"- Automated capture ready: `{markdown_escape(all_execution_summary.get('automated_capture_ready'))}`",
                f"- Requests ready for manual capture: `{markdown_escape(all_execution_summary.get('manual_operator_capture_ready_count'))}` / `{markdown_escape(all_execution_summary.get('request_count'))}`",
                f"- Requests ready for automated capture: `{markdown_escape(all_execution_summary.get('automated_capture_ready_count'))}` / `{markdown_escape(all_execution_summary.get('request_count'))}`",
                f"- Pending artifacts: `{markdown_escape(all_execution_summary.get('pending_artifact_count'))}`",
                f"- Runtime capture command options: `{markdown_escape(all_execution_summary.get('capture_command_option_count'))}`",
                f"- Operator command options: `{markdown_escape(all_execution_summary.get('operator_command_option_count'))}`",
                f"- Metadata command options: `{markdown_escape(all_execution_summary.get('metadata_command_option_count'))}`",
                f"- Manual captures required: `{markdown_escape(all_execution_summary.get('manual_capture_required_count'))}`",
                f"- Missing execution items: `{markdown_escape(all_execution_summary.get('missing_execution_item_count'))}`",
                "",
                "| Request | Pending artifacts | Runtime commands | Manual captures required | Missing execution items |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for request_execution in all_execution.get("requests", []):
            if not isinstance(request_execution, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    markdown_escape(value)
                    for value in (
                        request_execution.get("request_name") or "unknown",
                        request_execution.get("pending_artifact_count"),
                        request_execution.get("capture_command_option_count"),
                        request_execution.get("manual_capture_required_count"),
                        request_execution.get("missing_execution_item_count"),
                    )
                )
                + " |"
            )

    manual_runbook = summary.get("all_request_manual_capture_runbook_manifest")
    manual_runbook_summary = summary.get("all_request_manual_capture_runbook_summary")
    if isinstance(manual_runbook, dict) and manual_runbook:
        manual_runbook_summary = manual_runbook_summary if isinstance(manual_runbook_summary, dict) else {}
        missing_items = manual_runbook.get("missing_items") if isinstance(manual_runbook.get("missing_items"), list) else []
        lines.extend(
            [
                "",
                "## All-Request Manual Capture Runbook",
                "",
                f"- Ready: `{markdown_escape(manual_runbook_summary.get('ready'))}`",
                f"- Requests ready: `{markdown_escape(manual_runbook_summary.get('ready_request_count'))}` / `{markdown_escape(manual_runbook_summary.get('request_count'))}`",
                f"- Manual tasks: `{markdown_escape(manual_runbook_summary.get('manual_task_count'))}`",
                f"- Runtime-command tasks: `{markdown_escape(manual_runbook_summary.get('runtime_command_task_count'))}`",
                f"- Validator commands: `{markdown_escape(manual_runbook_summary.get('validator_command_count'))}`",
                f"- Missing items: `{markdown_escape(', '.join(str(item) for item in missing_items) if missing_items else 'none')}`",
                "",
                "| Request | Artifact | Status after approval | Receipt ready | Validators | Receipt path |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for request in manual_runbook.get("requests", []):
            if not isinstance(request, dict):
                continue
            for task in request.get("manual_tasks", []):
                if not isinstance(task, dict):
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        markdown_escape(value)
                        for value in (
                            request.get("request_name") or "unknown",
                            task.get("artifact_id") or "unknown",
                            task.get("status_after_approval") or "unknown",
                            task.get("receipt_ready"),
                            task.get("validator_command_count"),
                            task.get("receipt_path") or "missing",
                        )
                    )
                    + " |"
                )

    post_capture_runbook = summary.get("all_request_post_capture_intake_runbook_manifest")
    post_capture_runbook_summary = summary.get("all_request_post_capture_intake_runbook_summary")
    if isinstance(post_capture_runbook, dict) and post_capture_runbook:
        post_capture_runbook_summary = post_capture_runbook_summary if isinstance(post_capture_runbook_summary, dict) else {}
        lines.extend(
            [
                "",
                "## All-Request Post-Capture Intake Runbook",
                "",
                f"- Ready: `{markdown_escape(post_capture_runbook_summary.get('ready'))}`",
                f"- Requests: `{markdown_escape(post_capture_runbook_summary.get('request_count'))}`",
                f"- Artifact gates: `{markdown_escape(post_capture_runbook_summary.get('artifact_gate_count'))}`",
                f"- Ready after current intake: `{markdown_escape(post_capture_runbook_summary.get('ready_after_current_intake_count'))}`",
                f"- Missing after current intake: `{markdown_escape(post_capture_runbook_summary.get('missing_after_current_intake_count'))}`",
                f"- Validator commands: `{markdown_escape(post_capture_runbook_summary.get('validator_command_count'))}`",
                f"- Ready to update bundles: `{markdown_escape(post_capture_runbook_summary.get('ready_to_update_bundle_count'))}`",
                "",
                "| Request | Artifact gates | Ready now | Missing now | Validators | Ready to update bundle |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for request in post_capture_runbook.get("requests", []):
            if not isinstance(request, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    markdown_escape(value)
                    for value in (
                        request.get("request_name") or "unknown",
                        request.get("artifact_gate_count"),
                        request.get("ready_after_current_intake_count"),
                        request.get("missing_after_current_intake_count"),
                        request.get("validator_command_count"),
                        request.get("ready_to_update_bundle_after_current_intake"),
                    )
                )
                + " |"
            )

    policy_candidate_trace_plan = summary.get("recommended_policy_candidate_trace_plan")
    if isinstance(policy_candidate_trace_plan, dict):
        command_classes = policy_candidate_trace_plan.get("command_classes")
        if not isinstance(command_classes, list):
            command_classes = []
        lines.extend(
            [
                "",
                "## Recommended Policy-Candidate Trace",
                "",
                f"- Bundle: `{markdown_escape(policy_candidate_trace_plan.get('bundle_path') or 'missing')}`",
                f"- Prompt set: `{markdown_escape(policy_candidate_trace_plan.get('candidate_prompt_set_path') or 'missing')}`",
                f"- Candidate trace: `{markdown_escape(policy_candidate_trace_plan.get('candidate_trace_path') or 'missing')}`",
                f"- Capture receipt: `{markdown_escape(policy_candidate_trace_plan.get('candidate_trace_receipt_path') or 'missing')}`",
                f"- Handoff ready: `{markdown_escape(policy_candidate_trace_handoff_ready(policy_candidate_trace_plan))}`",
                f"- Prompt set ready: `{markdown_escape(policy_candidate_trace_plan.get('prompt_set_ready'))}`",
                f"- Candidate trace exists: `{markdown_escape(policy_candidate_trace_plan.get('candidate_trace_exists'))}`",
                f"- Capture receipt ready: `{markdown_escape(policy_candidate_trace_plan.get('capture_receipt_ready'))}`",
                f"- Runtime approval required: `{markdown_escape(policy_candidate_trace_plan.get('runtime_requires_explicit_approval'))}`",
                f"- Policy candidate ready after plan: `{markdown_escape(policy_candidate_trace_plan.get('policy_candidate_ready_after_plan'))}`",
                f"- Validator command classes: `{markdown_escape(', '.join(str(item) for item in command_classes) if command_classes else 'none')}`",
            ]
        )

    dense_fallback_capture_plan = summary.get("recommended_dense_fallback_capture_plan")
    if isinstance(dense_fallback_capture_plan, dict):
        command_classes = dense_fallback_capture_plan.get("command_classes")
        if not isinstance(command_classes, list):
            command_classes = []
        lines.extend(
            [
                "",
                "## Recommended Dense Fallback Capture",
                "",
                f"- Bundle: `{markdown_escape(dense_fallback_capture_plan.get('bundle_path') or 'missing')}`",
                f"- Prompt set: `{markdown_escape(dense_fallback_capture_plan.get('prompt_set_path') or 'missing')}`",
                f"- Managed output: `{markdown_escape(dense_fallback_capture_plan.get('managed_output_path') or 'missing')}`",
                f"- Dense output: `{markdown_escape(dense_fallback_capture_plan.get('dense_output_path') or 'missing')}`",
                f"- Comparison artifact: `{markdown_escape(dense_fallback_capture_plan.get('fallback_artifact_path') or 'missing')}`",
                f"- Handoff ready: `{markdown_escape(dense_fallback_capture_handoff_ready(dense_fallback_capture_plan))}`",
                f"- Managed output ready: `{markdown_escape(dense_fallback_capture_plan.get('managed_output_ready'))}`",
                f"- Dense output ready: `{markdown_escape(dense_fallback_capture_plan.get('dense_output_ready'))}`",
                f"- Metadata ready to build comparison: `{markdown_escape(dense_fallback_capture_plan.get('metadata_ready_to_build_comparison'))}`",
                f"- Comparison ready: `{markdown_escape(dense_fallback_capture_plan.get('comparison_ready'))}`",
                f"- Runtime approval required: `{markdown_escape(dense_fallback_capture_plan.get('runtime_requires_explicit_approval'))}`",
                f"- Validator command classes: `{markdown_escape(', '.join(str(item) for item in command_classes) if command_classes else 'none')}`",
            ]
        )

    live_capability_proof_handoff = summary.get("recommended_live_capability_proof_handoff")
    if isinstance(live_capability_proof_handoff, dict):
        blockers = live_capability_proof_handoff.get("blockers")
        if not isinstance(blockers, list):
            blockers = []
        lines.extend(
            [
                "",
                "## Recommended Live Capability Proof",
                "",
                f"- Bundle: `{markdown_escape(live_capability_proof_handoff.get('source_bundle_path') or 'missing')}`",
                f"- Proof template: `{markdown_escape(live_capability_proof_handoff.get('proof_artifact_path') or 'missing')}`",
                f"- Handoff ready: `{markdown_escape(live_capability_proof_handoff_ready(live_capability_proof_handoff))}`",
                f"- Proof ready: `{markdown_escape(live_capability_proof_handoff.get('proof_ready'))}`",
                f"- Context binding ready: `{markdown_escape(live_capability_proof_handoff.get('context_binding_ready'))}`",
                f"- Residency observation ready: `{markdown_escape(live_capability_proof_handoff.get('residency_observation_ready'))}`",
                f"- Residency control ready: `{markdown_escape(live_capability_proof_handoff.get('residency_control_ready'))}`",
                f"- Cleanup/restore ready: `{markdown_escape(live_capability_proof_handoff.get('cleanup_restore_ready'))}`",
                f"- Artifact export ready: `{markdown_escape(live_capability_proof_handoff.get('artifact_export_ready'))}`",
                f"- Blocker count: `{markdown_escape(live_capability_proof_handoff.get('blocker_count'))}`",
                f"- Blockers: `{markdown_escape(', '.join(str(item) for item in blockers) if blockers else 'none')}`",
            ]
        )

    runtime_actuator_spike = summary.get("phase3_runtime_actuator_spike_summary")
    if isinstance(runtime_actuator_spike, dict):
        lines.extend(
            [
                "",
                "## Runtime Actuator Spike Handoff",
                "",
                f"- Handoff ready: `{markdown_escape(runtime_actuator_spike.get('spike_handoff_ready'))}`",
                f"- Live spike ready: `{markdown_escape(runtime_actuator_spike.get('live_spike_ready'))}`",
                f"- Proof requirements: `{markdown_escape(runtime_actuator_spike.get('proof_requirement_count'))}`",
                f"- Proof artifacts: `{markdown_escape(runtime_actuator_spike.get('proof_artifact_count'))}`",
                f"- Dependency edges: `{markdown_escape(runtime_actuator_spike.get('dependency_edge_count'))}`",
                f"- Blocking capabilities: `{markdown_escape(runtime_actuator_spike.get('blocking_capability_count'))}`",
                "",
                "| Capability | Status | Artifacts | Dependencies | Gate |",
                "| --- | --- | ---: | ---: | --- |",
            ]
        )
        requirements = runtime_actuator_spike.get("proof_requirements")
        if isinstance(requirements, list):
            for requirement in requirements:
                if not isinstance(requirement, dict):
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            markdown_escape(requirement.get("capability_id")),
                            markdown_escape(requirement.get("current_status")),
                            markdown_escape(requirement.get("proof_artifact_count")),
                            markdown_escape(requirement.get("dependency_count")),
                            markdown_escape(requirement.get("completion_gate")),
                        ]
                    )
                    + " |"
                )

    gaps = summary["remaining_gaps"]
    lines.extend(
        [
            "",
            "## Remaining Gaps",
            "",
            f"- Real-model pair ready: {gaps['real_model_pair_ready']}",
            f"- Dense fallback comparison ready: {gaps['dense_fallback_comparison_ready']}",
            f"- Dense fallback capture handoff ready: {gaps['dense_fallback_capture_handoff_ready']}",
            f"- Dense fallback prompt set ready: {gaps['dense_fallback_prompt_set_ready']}",
            f"- Dense fallback managed output ready: {gaps['dense_fallback_managed_output_ready']}",
            f"- Dense fallback dense output ready: {gaps['dense_fallback_dense_output_ready']}",
            f"- Dense fallback metadata ready to build comparison: {gaps['dense_fallback_metadata_ready_to_build_comparison']}",
            f"- Dense fallback managed output path: {gaps['dense_fallback_managed_output_path']}",
            f"- Dense fallback dense output path: {gaps['dense_fallback_dense_output_path']}",
            f"- Dense fallback comparison path: {gaps['dense_fallback_capture_comparison_path']}",
            f"- Live capability proof handoff ready: {gaps['live_capability_proof_handoff_ready']}",
            f"- Live capability proof template path: {gaps['live_capability_proof_template_path']}",
            f"- Live capability proof context ready: {gaps['live_capability_proof_context_ready']}",
            f"- Live residency observation ready: {gaps['live_residency_observation_ready']}",
            f"- Live residency control ready: {gaps['live_residency_control_ready']}",
            f"- Cleanup restore proof ready: {gaps['cleanup_restore_proof_ready']}",
            f"- Live artifact export ready: {gaps['live_artifact_export_ready']}",
            f"- Live capability proof ready: {gaps['live_capability_proof_ready']}",
            f"- Runtime actuator design ready: {gaps['runtime_actuator_design_ready']}",
            f"- Runtime actuator backend: {gaps['runtime_actuator_backend_family']}",
            f"- Runtime actuator live ready: {gaps['runtime_actuator_live_ready']}",
            f"- Runtime actuator blocking capabilities: {gaps['runtime_actuator_blocking_capability_count']}",
            f"- Runtime actuator control blockers: {gaps['runtime_actuator_control_blocker_count']}",
            f"- Runtime actuator spike handoff ready: {gaps['runtime_actuator_spike_handoff_ready']}",
            f"- Runtime actuator spike live ready: {gaps['runtime_actuator_spike_live_ready']}",
            f"- Runtime actuator spike proof requirements: {gaps['runtime_actuator_spike_proof_requirement_count']}",
            f"- Runtime actuator spike proof artifacts: {gaps['runtime_actuator_spike_proof_artifact_count']}",
            f"- Runtime actuator spike dependency edges: {gaps['runtime_actuator_spike_dependency_edge_count']}",
            f"- Runtime actuator spike blocking capabilities: {gaps['runtime_actuator_spike_blocking_capability_count']}",
            f"- Ready for Phase 4 adapter spike: {gaps['ready_for_phase4_adapter_spike']}",
            f"- Ready for live spike: {gaps['ready_for_live_spike']}",
            f"- Handoff scaffolds ready: {gaps['handoff_scaffolds_ready']}",
            f"- Runtime-capture requests: {gaps['runtime_capture_request_count']}",
            f"- Runtime-capture ready for operator: {gaps['runtime_capture_ready_for_operator_count']}",
            f"- Runtime-capture complete: {gaps['runtime_capture_complete_count']}",
            f"- Runtime-capture approval rebuild command manifest entries: {gaps['runtime_capture_approval_rebuild_command_manifest_count']}",
            f"- Runtime-capture post-approval preview valid: {gaps['runtime_capture_post_approval_preview_valid']}",
            f"- Runtime-capture post-approval preview status: {gaps['runtime_capture_post_approval_preview_status']}",
            f"- Runtime-capture post-approval ready for operator: {gaps['runtime_capture_post_approval_ready_for_operator']}",
            f"- Runtime-capture post-approval capture complete: {gaps['runtime_capture_post_approval_capture_complete']}",
            f"- Runtime-capture post-approval pending artifacts: {gaps['runtime_capture_post_approval_pending_artifact_count']}",
            f"- Runtime-capture post-approval mutates request: {gaps['runtime_capture_post_approval_mutates_request']}",
            f"- Recommended runtime-capture preflight ready: {gaps['recommended_runtime_capture_preflight_ready']}",
            f"- Recommended runtime-capture preflight pending artifacts: {gaps['recommended_runtime_capture_preflight_pending_artifact_count']}",
            f"- Recommended runtime-capture preflight receipt entries: {gaps['recommended_runtime_capture_preflight_receipt_entry_count']}",
            f"- Recommended runtime-capture preflight validator commands: {gaps['recommended_runtime_capture_preflight_validator_command_count']}",
            f"- Recommended runtime-capture preflight runtime closure reasons: {gaps['recommended_runtime_capture_preflight_runtime_closure_reason_count']}",
            f"- Recommended runtime-capture preflight missing items: {gaps['recommended_runtime_capture_preflight_missing_item_count']}",
            f"- Recommended runtime-capture command contract ready: {gaps['recommended_runtime_capture_command_contract_ready']}",
            f"- Recommended runtime-capture command contract runtime ready: {gaps['recommended_runtime_capture_command_contract_runtime_ready']}",
            f"- Recommended runtime-capture command contract planned captures: {gaps['recommended_runtime_capture_command_contract_planned_capture_count']}",
            f"- Recommended runtime-capture command contract runtime commands: {gaps['recommended_runtime_capture_command_contract_runtime_command_count']}",
            f"- Recommended runtime-capture command contract missing runtime commands: {gaps['recommended_runtime_capture_command_contract_missing_runtime_command_count']}",
            f"- Recommended runtime-capture launch-card binding ready: {gaps['recommended_runtime_capture_launch_card_binding_ready']}",
            f"- Recommended runtime-capture launch-card bound tasks: {gaps['recommended_runtime_capture_launch_card_binding_bound_task_count']}",
            f"- Recommended runtime-capture launch-card command options: {gaps['recommended_runtime_capture_launch_card_binding_command_option_count']}",
            f"- Recommended runtime-capture execution manual ready: {gaps['recommended_runtime_capture_execution_manual_ready']}",
            f"- Recommended runtime-capture execution automated ready: {gaps['recommended_runtime_capture_execution_automated_ready']}",
            f"- Recommended runtime-capture execution runtime command options: {gaps['recommended_runtime_capture_execution_command_option_count']}",
            f"- Recommended runtime-capture execution operator command options: {gaps['recommended_runtime_capture_execution_operator_command_option_count']}",
            f"- Recommended runtime-capture execution metadata command options: {gaps['recommended_runtime_capture_execution_metadata_command_option_count']}",
            f"- Recommended runtime-capture execution artifacts with runtime commands: {gaps['recommended_runtime_capture_execution_artifacts_with_command_count']}",
            f"- Recommended runtime-capture execution manual capture required: {gaps['recommended_runtime_capture_execution_manual_capture_required_count']}",
            f"- Recommended runtime-capture execution missing capture commands: {gaps['recommended_runtime_capture_execution_missing_capture_command_count']}",
            f"- Recommended runtime-capture execution missing items: {gaps['recommended_runtime_capture_execution_missing_item_count']}",
            f"- All-request runtime-capture execution manual ready: {gaps['all_request_runtime_capture_execution_manual_ready']}",
            f"- All-request runtime-capture execution automated ready: {gaps['all_request_runtime_capture_execution_automated_ready']}",
            f"- All-request runtime-capture execution ready requests: {gaps['all_request_runtime_capture_execution_ready_request_count']} / {gaps['all_request_runtime_capture_execution_request_count']}",
            f"- All-request runtime-capture execution pending artifacts: {gaps['all_request_runtime_capture_execution_pending_artifact_count']}",
            f"- All-request runtime-capture execution runtime command options: {gaps['all_request_runtime_capture_execution_command_option_count']}",
            f"- All-request runtime-capture execution manual captures required: {gaps['all_request_runtime_capture_execution_manual_capture_required_count']}",
            f"- All-request runtime-capture execution missing capture commands: {gaps['all_request_runtime_capture_execution_missing_capture_command_count']}",
            f"- All-request runtime-capture execution missing items: {gaps['all_request_runtime_capture_execution_missing_item_count']}",
            f"- All-request manual capture runbook ready: {gaps['all_request_manual_capture_runbook_ready']}",
            f"- All-request manual capture runbook requests: {gaps['all_request_manual_capture_runbook_ready_request_count']} / {gaps['all_request_manual_capture_runbook_request_count']}",
            f"- All-request manual capture runbook manual tasks: {gaps['all_request_manual_capture_runbook_manual_task_count']}",
            f"- All-request manual capture runbook runtime-command tasks: {gaps['all_request_manual_capture_runbook_runtime_command_task_count']}",
            f"- All-request manual capture runbook validator commands: {gaps['all_request_manual_capture_runbook_validator_command_count']}",
            f"- All-request manual capture runbook missing items: {gaps['all_request_manual_capture_runbook_missing_item_count']}",
            f"- All-request manual capture runbook missing receipt commands: {gaps['all_request_manual_capture_runbook_missing_receipt_command_count']}",
            f"- All-request manual capture runbook missing validator commands: {gaps['all_request_manual_capture_runbook_missing_validator_command_count']}",
            f"- All-request manual capture runbook missing approval commands: {gaps['all_request_manual_capture_runbook_missing_approval_command_count']}",
            f"- All-request manual capture runbook missing source request paths: {gaps['all_request_manual_capture_runbook_missing_source_request_path_count']}",
            f"- All-request manual capture runbook missing prompt-set paths: {gaps['all_request_manual_capture_runbook_missing_prompt_set_path_count']}",
            f"- All-request manual capture runbook missing explicit approvals: {gaps['all_request_manual_capture_runbook_missing_explicit_approval_count']}",
            f"- All-request manual capture runbook missing prompt-traffic acknowledgements: {gaps['all_request_manual_capture_runbook_missing_prompt_traffic_ack_count']}",
            f"- All-request post-capture intake runbook ready: {gaps['all_request_post_capture_intake_runbook_ready']}",
            f"- All-request post-capture intake runbook requests: {gaps['all_request_post_capture_intake_runbook_request_count']}",
            f"- All-request post-capture intake runbook artifact gates: {gaps['all_request_post_capture_intake_runbook_artifact_gate_count']}",
            f"- All-request post-capture intake runbook ready after current intake: {gaps['all_request_post_capture_intake_runbook_ready_after_current_intake_count']}",
            f"- All-request post-capture intake runbook missing after current intake: {gaps['all_request_post_capture_intake_runbook_missing_after_current_intake_count']}",
            f"- All-request post-capture intake runbook validator commands: {gaps['all_request_post_capture_intake_runbook_validator_command_count']}",
            f"- All-request post-capture intake runbook ready to update bundles: {gaps['all_request_post_capture_intake_runbook_ready_to_update_bundle_count']}",
            f"- All-request post-capture intake runbook phase4 candidates: {gaps['all_request_post_capture_intake_runbook_phase4_candidate_count']}",
            f"- All-request post-capture intake runbook live-spike candidates: {gaps['all_request_post_capture_intake_runbook_live_spike_candidate_count']}",
            f"- All-request post-capture intake runbook missing source request paths: {gaps['all_request_post_capture_intake_runbook_missing_source_request_path_count']}",
            f"- All-request post-capture intake runbook missing prompt-set paths: {gaps['all_request_post_capture_intake_runbook_missing_prompt_set_path_count']}",
            f"- All-request post-capture intake runbook missing explicit approvals: {gaps['all_request_post_capture_intake_runbook_missing_explicit_approval_count']}",
            f"- All-request post-capture intake runbook missing prompt-traffic acknowledgements: {gaps['all_request_post_capture_intake_runbook_missing_prompt_traffic_ack_count']}",
            f"- All-request post-capture intake runbook missing items: {gaps['all_request_post_capture_intake_runbook_missing_item_count']}",
            f"- Approval manifest parity ready: {gaps['approval_manifest_parity_ready']}",
            f"- Approval manifest parity matched requests: {gaps['approval_manifest_parity_matched_request_count']}",
            f"- Receipt requirement parity ready: {gaps['receipt_requirement_parity_ready']}",
            f"- Receipt requirement parity matched requirements: {gaps['receipt_requirement_parity_matched_requirement_count']}",
            f"- Operator queue parity ready: {gaps['operator_queue_parity_ready']}",
            f"- Operator queue parity matched requests: {gaps['operator_queue_parity_matched_request_count']}",
            f"- Post-approval capture-fill parity ready: {gaps['post_approval_capture_fill_parity_ready']}",
            f"- Post-approval capture-fill parity matched artifacts: {gaps['post_approval_capture_fill_parity_matched_artifact_count']}",
            f"- Post-approval capture-fill parity runtime pending artifacts: {gaps['post_approval_capture_fill_parity_runtime_pending_artifact_count']}",
            f"- Post-approval capture-fill parity capture-result fill steps: {gaps['post_approval_capture_fill_parity_capture_result_fill_step_count']}",
            f"- Post-approval capture-fill parity request path match: {gaps['post_approval_capture_fill_parity_request_path_match']}",
            f"- All-request post-approval capture-fill parity ready: {gaps['all_post_approval_capture_fill_parity_ready']}",
            f"- All-request post-approval capture-fill parity matched requests: {gaps['all_post_approval_capture_fill_parity_matched_request_count']}",
            f"- All-request post-approval capture-fill parity matched artifacts: {gaps['all_post_approval_capture_fill_parity_matched_artifact_count']}",
            f"- All-request post-approval capture-fill parity runtime pending artifacts: {gaps['all_post_approval_capture_fill_parity_runtime_pending_artifact_count']}",
            f"- All-request post-approval capture-fill parity capture-result fill steps: {gaps['all_post_approval_capture_fill_parity_capture_result_fill_step_count']}",
            f"- All-request capture queue ready: {gaps['all_request_capture_queue_ready']}",
            f"- All-request capture queue requests: {gaps['all_request_capture_queue_request_count']}",
            f"- All-request capture queue complete fill plans: {gaps['all_request_capture_queue_complete_fill_plan_count']}",
            f"- All-request capture queue pending artifacts: {gaps['all_request_capture_queue_pending_artifact_count']}",
            f"- All-request capture queue validator commands: {gaps['all_request_capture_queue_validator_command_count']}",
            f"- All-request validator command manifest ready: {gaps['all_request_validator_command_manifest_ready']}",
            f"- All-request validator command manifest requests: {gaps['all_request_validator_command_manifest_request_count']}",
            f"- All-request validator command manifest artifact entries: {gaps['all_request_validator_command_manifest_artifact_entry_count']}",
            f"- All-request validator command manifest runtime artifacts: {gaps['all_request_validator_command_manifest_runtime_artifact_count']}",
            f"- All-request validator command manifest future artifacts: {gaps['all_request_validator_command_manifest_future_artifact_count']}",
            f"- All-request validator command manifest runtime commands: {gaps['all_request_validator_command_manifest_runtime_command_count']}",
            f"- All-request validator command manifest future commands: {gaps['all_request_validator_command_manifest_future_command_count']}",
            f"- All-request validator command manifest total commands: {gaps['all_request_validator_command_manifest_total_command_count']}",
            f"- All-request downstream handoff manifest ready: {gaps['all_request_downstream_handoff_manifest_ready']}",
            f"- All-request downstream handoff manifest requests: {gaps['all_request_downstream_handoff_manifest_request_count']}",
            f"- All-request downstream handoff policy ready: {gaps['all_request_downstream_handoff_policy_ready_count']}",
            f"- All-request downstream handoff dense ready: {gaps['all_request_downstream_handoff_dense_ready_count']}",
            f"- All-request downstream handoff live ready: {gaps['all_request_downstream_handoff_live_ready_count']}",
            f"- All-request downstream handoff all ready: {gaps['all_request_downstream_handoff_all_ready_count']}",
            f"- All-request receipt-fill manifest ready: {gaps['all_request_receipt_fill_manifest_ready']}",
            f"- All-request receipt-fill manifest requests: {gaps['all_request_receipt_fill_manifest_request_count']}",
            f"- All-request receipt-fill entries ready: {gaps['all_request_receipt_fill_manifest_ready_count']} / {gaps['all_request_receipt_fill_manifest_entry_count']}",
            f"- All-request receipt-fill missing entries: {gaps['all_request_receipt_fill_manifest_missing_count']}",
            f"- All-request receipt-fill candidate router trace ready: {gaps['all_request_receipt_fill_manifest_candidate_router_trace_ready_count']} / {gaps['all_request_receipt_fill_manifest_candidate_router_trace_entry_count']}",
            f"- All-request receipt-fill candidate router trace missing: {gaps['all_request_receipt_fill_manifest_candidate_router_trace_missing_count']}",
            f"- All-request receipt-fill managed outputs ready: {gaps['all_request_receipt_fill_manifest_managed_output_ready_count']} / {gaps['all_request_receipt_fill_manifest_managed_output_entry_count']}",
            f"- All-request receipt-fill managed outputs missing: {gaps['all_request_receipt_fill_manifest_managed_output_missing_count']}",
            f"- All-request receipt-fill dense outputs ready: {gaps['all_request_receipt_fill_manifest_dense_output_ready_count']} / {gaps['all_request_receipt_fill_manifest_dense_output_entry_count']}",
            f"- All-request receipt-fill dense outputs missing: {gaps['all_request_receipt_fill_manifest_dense_output_missing_count']}",
            f"- All-request receipt-fill approvals recorded after approval: {gaps['all_request_receipt_fill_manifest_approval_recorded_after_count']}",
            f"- All-request receipt-fill approvals missing after approval: {gaps['all_request_receipt_fill_manifest_approval_missing_after_count']}",
            f"- All-request receipt-fill blocked after approval: {gaps['all_request_receipt_fill_manifest_blocked_after_approval_count']}",
            f"- All-request receipt-fill validator commands: {gaps['all_request_receipt_fill_manifest_validator_command_count']}",
            f"- All-request receipt/validator parity ready: {gaps['all_request_receipt_validator_parity_ready']}",
            f"- All-request receipt/validator parity matched artifacts: {gaps['all_request_receipt_validator_parity_matched_artifact_count']} / {gaps['all_request_receipt_validator_parity_runtime_artifact_count']}",
            f"- All-request receipt/validator parity receipt entries: {gaps['all_request_receipt_validator_parity_receipt_entry_count']}",
            f"- All-request receipt/validator parity metadata mismatches: {gaps['all_request_receipt_validator_parity_metadata_mismatch_count']}",
            f"- All-request receipt/validator parity missing from receipt fills: {gaps['all_request_receipt_validator_parity_missing_from_receipt_count']}",
            f"- All-request receipt/validator parity missing from runtime validators: {gaps['all_request_receipt_validator_parity_missing_from_runtime_count']}",
            f"- All-request receipt-fill command manifest ready: {gaps['all_request_receipt_fill_command_manifest_ready']}",
            f"- All-request receipt-fill command manifest requests: {gaps['all_request_receipt_fill_command_manifest_request_count']}",
            f"- All-request receipt-fill command entries: {gaps['all_request_receipt_fill_command_manifest_entry_count']}",
            f"- All-request receipt-fill command ready receipts: {gaps['all_request_receipt_fill_command_manifest_ready_receipt_count']}",
            f"- All-request receipt-fill command blocked after approval: {gaps['all_request_receipt_fill_command_manifest_blocked_after_approval_count']}",
            f"- All-request receipt-fill command validator commands: {gaps['all_request_receipt_fill_command_manifest_validator_command_count']}",
            f"- All-request receipt-fill command missing command entries: {gaps['all_request_receipt_fill_command_manifest_missing_command_entry_count']}",
            f"- All-request receipt-fill command missing receipt paths: {gaps['all_request_receipt_fill_command_manifest_missing_receipt_path_count']}",
            f"- All-request receipt-fill command missing source request paths: {gaps['all_request_receipt_fill_command_manifest_missing_source_request_path_count']}",
            f"- All-request receipt-fill command missing prompt-set paths: {gaps['all_request_receipt_fill_command_manifest_missing_prompt_set_path_count']}",
            f"- All-request receipt-fill command missing explicit approvals: {gaps['all_request_receipt_fill_command_manifest_missing_explicit_approval_count']}",
            f"- All-request receipt-fill command missing prompt-traffic acknowledgements: {gaps['all_request_receipt_fill_command_manifest_missing_prompt_traffic_ack_count']}",
            f"- Phase 3 blocker closure manifest ready: {gaps['phase3_blocker_closure_manifest_ready']}",
            f"- Phase 3 blocker closure mapping ready: {gaps['phase3_blocker_closure_mapping_ready']}",
            f"- Phase 3 blocker closure evidence complete: {gaps['phase3_blocker_closure_evidence_complete']}",
            f"- Phase 3 blocker closure ready scope: {gaps['phase3_blocker_closure_ready_scope']}",
            f"- Phase 3 blocker closure reasons: {gaps['phase3_blocker_closure_reason_count']}",
            f"- Phase 3 blocker closure mapped reasons: {gaps['phase3_blocker_closure_mapped_reason_count']}",
            f"- Phase 3 blocker closure unmapped reasons: {gaps['phase3_blocker_closure_unmapped_reason_count']}",
            f"- Phase 3 blocker closure unresolved reasons: {gaps['phase3_blocker_closure_unresolved_reason_count']}",
            f"- Phase 3 blocker closure runtime-capture required: {gaps['phase3_blocker_closure_runtime_capture_required_count']}",
            f"- Phase 3 blocker closure future-adapter required: {gaps['phase3_blocker_closure_future_adapter_required_count']}",
            f"- Phase 3 blocker closure handoffs ready: {gaps['phase3_blocker_closure_handoff_ready_count']}",
            f"- Phase 3 blocker closure validator commands: {gaps['phase3_blocker_closure_validator_command_count']}",
            f"- Phase 3 blocker closure missing evidence: {gaps['phase3_blocker_closure_missing_evidence_count']}",
            f"- Phase 3 blocker evidence ledger ready: {gaps['phase3_blocker_evidence_ledger_ready']}",
            f"- Phase 3 blocker evidence ledger rows: {gaps['phase3_blocker_evidence_ledger_row_count']} / {gaps['phase3_blocker_evidence_ledger_expected_missing_evidence_count']}",
            f"- Phase 3 blocker evidence ledger runtime-capture rows: {gaps['phase3_blocker_evidence_ledger_runtime_capture_required_row_count']}",
            f"- Phase 3 blocker evidence ledger future-adapter rows: {gaps['phase3_blocker_evidence_ledger_future_adapter_required_row_count']}",
            f"- Phase 3 blocker evidence ledger receipt-bound rows: {gaps['phase3_blocker_evidence_ledger_receipt_bound_row_count']}",
            f"- Phase 3 blocker evidence ledger dense output rows: {gaps['phase3_blocker_evidence_ledger_dense_output_row_count']}",
            f"- Phase 3 blocker evidence ledger live proof rows: {gaps['phase3_blocker_evidence_ledger_live_proof_row_count']}",
            f"- Phase 3 blocker evidence ledger runtime actuator rows: {gaps['phase3_blocker_evidence_ledger_runtime_actuator_row_count']}",
            f"- Phase 3 blocker evidence ledger validator commands: {gaps['phase3_blocker_evidence_ledger_validator_command_count']}",
            f"- Phase 3 blocker evidence ledger missing items: {gaps['phase3_blocker_evidence_ledger_missing_item_count']}",
            f"- Phase 3 blocker resolution queue ready: {gaps['phase3_blocker_resolution_queue_ready']}",
            f"- Phase 3 blocker resolution queue work packages: {gaps['phase3_blocker_resolution_queue_work_package_count']}",
            f"- Phase 3 blocker resolution queue rows: {gaps['phase3_blocker_resolution_queue_row_count']} / {gaps['phase3_blocker_resolution_queue_expected_ledger_row_count']}",
            f"- Phase 3 blocker resolution queue runtime-capture packages: {gaps['phase3_blocker_resolution_queue_runtime_capture_package_count']}",
            f"- Phase 3 blocker resolution queue future-adapter packages: {gaps['phase3_blocker_resolution_queue_future_adapter_package_count']}",
            f"- Phase 3 blocker resolution queue validator commands: {gaps['phase3_blocker_resolution_queue_validator_command_count']}",
            f"- Phase 3 blocker resolution queue completion gates: {gaps['phase3_blocker_resolution_queue_completion_gate_count']}",
            f"- Phase 3 blocker resolution queue dependency edges: {gaps['phase3_blocker_resolution_queue_dependency_edge_count']}",
            f"- Phase 3 blocker resolution queue missing items: {gaps['phase3_blocker_resolution_queue_missing_item_count']}",
            f"- Phase 3 blocker resolution queue next unblocked package: {gaps['phase3_blocker_resolution_queue_next_unblocked_work_package_id']}",
            f"- Phase 3 blocker resolution queue next unblocked stage: {gaps['phase3_blocker_resolution_queue_next_unblocked_operator_stage']}",
            f"- Phase 3 blocker resolution queue next unblocked action: {gaps['phase3_blocker_resolution_queue_next_unblocked_next_action']}",
            f"- Phase 3 next unblocked operator handoff ready: {gaps['phase3_next_unblocked_operator_handoff_ready']}",
            f"- Phase 3 next unblocked work order ready: {gaps['phase3_next_unblocked_operator_handoff_work_order_ready']}",
            f"- Phase 3 next unblocked work order artifact: {gaps['phase3_next_unblocked_operator_handoff_work_order_artifact']}",
            f"- Policy-candidate prompt-identity ready traces: {gaps['policy_candidate_prompt_identity_ready_count']}",
            f"- Policy-candidate prompt-identity blockers: {gaps['policy_candidate_prompt_identity_metadata_missing_count']}",
            f"- Policy-candidate trace handoff ready: {gaps['policy_candidate_trace_handoff_ready']}",
            f"- Policy-candidate trace prompt set ready: {gaps['policy_candidate_trace_prompt_set_ready']}",
            f"- Policy-candidate trace exists: {gaps['policy_candidate_trace_exists']}",
            f"- Policy-candidate trace receipt ready: {gaps['policy_candidate_trace_receipt_ready']}",
            f"- Policy-candidate trace ready after plan: {gaps['policy_candidate_trace_ready_after_plan']}",
            f"- Policy-candidate trace path: {gaps['policy_candidate_trace_path']}",
            f"- Policy-candidate trace receipt path: {gaps['policy_candidate_trace_receipt_path']}",
            f"- Capture-result drift-free saved requests: {gaps['capture_result_request_drift_free_count']}",
            f"- Capture-result drifted saved requests: {gaps['capture_result_request_drifted_count']}",
            f"- Capture-result ready-for-operator flags: {gaps['capture_result_ready_for_operator_capture_flag_count']}",
            f"- Capture-result approved but capture incomplete: {gaps['capture_result_approved_but_capture_incomplete_request_count']}",
            f"- Capture-result runtime approvals missing: {gaps['capture_result_runtime_approval_missing_request_count']}",
            f"- Capture-result approval rebuild commands available: {gaps['capture_result_approval_rebuild_command_available_request_count']}",
            f"- Capture-result approval rebuild command manifest entries: {gaps['capture_result_approval_rebuild_command_manifest_count']}",
            f"- Capture-result approved runtime-capture pending: {gaps['capture_result_approved_runtime_capture_pending_request_count']}",
            f"- Capture-result capture receipts ready: {gaps['capture_result_capture_receipts_ready_count']} / {gaps['capture_result_capture_receipts_required_count']}",
            f"- Capture-result candidate trace receipts ready: {gaps['capture_result_candidate_trace_receipt_ready_count']}",
            f"- Capture-result managed output receipts ready: {gaps['capture_result_managed_output_receipt_ready_count']}",
            f"- Capture-result dense output receipts ready: {gaps['capture_result_dense_output_receipt_ready_count']}",
            f"- Capture-result live proof receipts ready: {gaps['capture_result_live_capability_proof_ready_count']}",
            f"- Capture-result all capture receipts ready: {gaps['capture_result_all_capture_receipts_ready']}",
            f"- Capture-result all output receipt bindings ready: {gaps['capture_result_all_output_receipt_bindings_ready']}",
            f"- Capture-result receipt-fill manifest ready: {gaps['capture_result_receipt_fill_ready_count']} / {gaps['capture_result_receipt_fill_entry_count']}",
            f"- Capture-result receipt-fill missing entries: {gaps['capture_result_receipt_fill_missing_count']}",
            f"- Capture-result receipt-fill approvals missing: {gaps['capture_result_receipt_fill_approval_missing_count']}",
            f"- Capture-result receipt-fill candidate router trace ready: {gaps['capture_result_receipt_fill_candidate_router_trace_ready_count']} / {gaps['capture_result_receipt_fill_candidate_router_trace_entry_count']}",
            f"- Capture-result receipt-fill candidate router trace missing: {gaps['capture_result_receipt_fill_candidate_router_trace_missing_count']}",
            f"- Capture-result receipt-fill managed outputs ready: {gaps['capture_result_receipt_fill_managed_output_ready_count']} / {gaps['capture_result_receipt_fill_managed_output_entry_count']}",
            f"- Capture-result receipt-fill managed outputs missing: {gaps['capture_result_receipt_fill_managed_output_missing_count']}",
            f"- Capture-result receipt-fill dense outputs ready: {gaps['capture_result_receipt_fill_dense_output_ready_count']} / {gaps['capture_result_receipt_fill_dense_output_entry_count']}",
            f"- Capture-result receipt-fill dense outputs missing: {gaps['capture_result_receipt_fill_dense_output_missing_count']}",
            f"- Capture-result approval transition previews: {gaps['capture_result_approval_transition_preview_count']}",
            f"- Capture-result approval transitions ready for operator: {gaps['capture_result_approval_transition_ready_for_operator_count']}",
            f"- Capture-result approval transitions ready to update bundle: {gaps['capture_result_approval_transition_ready_to_update_bundle_count']}",
            f"- Capture-result post-approval capture-fill plans: {gaps['capture_result_post_approval_capture_fill_plan_count']}",
            f"- Capture-result post-approval capture-fill ready: {gaps['capture_result_post_approval_capture_fill_ready_count']} / {gaps['capture_result_post_approval_capture_fill_artifact_step_count']}",
            f"- Capture-result post-approval capture-fill missing: {gaps['capture_result_post_approval_capture_fill_missing_count']}",
            f"- Capture-result post-approval capture-fill validator commands: {gaps['capture_result_post_approval_capture_fill_validator_command_count']}",
            f"- Capture-result post-approval capture-fill ready to update bundle: {gaps['capture_result_post_approval_capture_fill_ready_to_update_bundle_count']}",
            f"- Capture-result missing receipt-gate requests: {gaps['capture_result_missing_receipt_gate_request_count']}",
            f"- Capture-result output receipt bindings ready: {gaps['capture_result_output_receipt_binding_ready_count']}",
            f"- Capture-result bundle updates ready: {gaps['capture_result_ready_to_update_bundle_count']}",
            f"- Capture-result Phase 4 candidates: {gaps['capture_result_phase4_candidate_count']}",
            f"- No-go reasons: {format_no_go_reasons(gaps.get('no_go_reasons'))}",
            "",
            "## Safety Contract",
            "",
        ]
    )
    for item in summary["safety_contract"]:
        lines.append(f"- {markdown_escape(item)}")
    lines.append("")
    return "\n".join(lines)


def write_json_artifact(output_dir: Path, filename: str, value: Any, kind: str, written_files: list[JSONDict]) -> Path:
    output_path = output_dir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written_files.append({"path": filename, "kind": kind})
    return output_path


def build_operator_handoff_manifest(
    summary: JSONDict,
    *,
    output_files: list[JSONDict],
    warnings: list[str] | None = None,
) -> JSONDict:
    gaps = summary.get("remaining_gaps") if isinstance(summary.get("remaining_gaps"), dict) else {}
    recommendation = summary.get("recommended_runtime_capture_request")
    recommendation = recommendation if isinstance(recommendation, dict) else {}
    approval_command = recommendation.get("approval_rebuild_command")
    approval_command = approval_command if isinstance(approval_command, dict) else {}
    launch_template = summary.get("recommended_runtime_capture_launch_card_template")
    launch_template = launch_template if isinstance(launch_template, dict) else {}
    launch_library = summary.get("phase3_launch_card_library_summary")
    launch_library = launch_library if isinstance(launch_library, dict) else {}
    blocker_closure = summary.get("phase3_blocker_closure_summary")
    blocker_closure = blocker_closure if isinstance(blocker_closure, dict) else {}
    blocker_ledger = summary.get("phase3_blocker_evidence_ledger_summary")
    blocker_ledger = blocker_ledger if isinstance(blocker_ledger, dict) else {}
    blocker_resolution_queue = summary.get("phase3_blocker_resolution_queue_summary")
    blocker_resolution_queue = blocker_resolution_queue if isinstance(blocker_resolution_queue, dict) else {}
    next_unblocked_handoff = summary.get("phase3_next_unblocked_operator_handoff_summary")
    next_unblocked_handoff = next_unblocked_handoff if isinstance(next_unblocked_handoff, dict) else {}
    runtime_actuator_spike = summary.get("phase3_runtime_actuator_spike_summary")
    runtime_actuator_spike = runtime_actuator_spike if isinstance(runtime_actuator_spike, dict) else {}
    runtime_actuator_proof_requirements = runtime_actuator_spike.get("proof_requirements")
    runtime_actuator_proof_requirements = runtime_actuator_proof_requirements if isinstance(runtime_actuator_proof_requirements, list) else []
    runtime_actuator_proof_requirement_id_count = len(
        [
            item
            for item in runtime_actuator_proof_requirements
            if isinstance(item, dict) and isinstance(item.get("capability_id"), str) and item.get("capability_id")
        ]
    )
    preflight = summary.get("recommended_runtime_capture_preflight_summary")
    preflight = preflight if isinstance(preflight, dict) else {}
    command_contract = summary.get("recommended_runtime_capture_command_contract_summary")
    command_contract = command_contract if isinstance(command_contract, dict) else {}
    execution = summary.get("recommended_runtime_capture_execution_coverage_summary")
    execution = execution if isinstance(execution, dict) else {}
    all_execution = summary.get("all_request_runtime_capture_execution_coverage_summary")
    all_execution = all_execution if isinstance(all_execution, dict) else {}
    launch_card_directory = summary.get("runtime_capture_launch_card_directory_manifest")
    launch_card_directory = launch_card_directory if isinstance(launch_card_directory, dict) else {}
    reuse_plan = summary.get("phase3_reuse_evidence_capture_plan")
    reuse_plan = reuse_plan if isinstance(reuse_plan, dict) else {}
    manual_runbook = summary.get("all_request_manual_capture_runbook_summary")
    manual_runbook = manual_runbook if isinstance(manual_runbook, dict) else {}
    post_capture_runbook = summary.get("all_request_post_capture_intake_runbook_summary")
    post_capture_runbook = post_capture_runbook if isinstance(post_capture_runbook, dict) else {}
    recommended_manual_runbook = summary.get("recommended_manual_capture_runbook_summary")
    recommended_manual_runbook = recommended_manual_runbook if isinstance(recommended_manual_runbook, dict) else {}
    recommended_post_capture_runbook = summary.get("recommended_post_capture_intake_runbook_summary")
    recommended_post_capture_runbook = recommended_post_capture_runbook if isinstance(recommended_post_capture_runbook, dict) else {}
    recommended_work_order = summary.get("recommended_runtime_capture_work_order_summary")
    recommended_work_order = recommended_work_order if isinstance(recommended_work_order, dict) else {}
    recommended_completion_receipt = summary.get("recommended_runtime_capture_completion_receipt_template_summary")
    recommended_completion_receipt = recommended_completion_receipt if isinstance(recommended_completion_receipt, dict) else {}
    recommended_completion_receipt_manifest = summary.get("recommended_runtime_capture_completion_receipt_template_manifest")
    recommended_completion_receipt_manifest = recommended_completion_receipt_manifest if isinstance(recommended_completion_receipt_manifest, dict) else {}
    recommended_reuse_capture = reuse_plan.get("recommended_next_capture")
    recommended_reuse_capture = recommended_reuse_capture if isinstance(recommended_reuse_capture, dict) else {}
    reuse_capture_contract = reuse_plan.get("capture_contract")
    reuse_capture_contract = reuse_capture_contract if isinstance(reuse_capture_contract, dict) else {}
    return {
        "schema_version": OPERATOR_HANDOFF_SCHEMA_VERSION,
        "mode": "phase3_operator_handoff_package",
        "valid": summary.get("valid") is True,
        "packet_ready": summary.get("packet_ready") is True,
        "phase3_complete": summary.get("phase3_complete") is True,
        "decision": summary.get("decision"),
        "metadata_only": True,
        "launches_runtimes": False,
        "runs_docker": False,
        "sends_prompt_traffic": False,
        "reads_private_tokens": False,
        "mutates_runtime_residency": False,
        "operator_handoff_ready": (
            summary.get("valid") is True
            and summary.get("packet_ready") is True
            and preflight.get("ready") is True
            and all_execution.get("manual_operator_capture_ready") is True
            and manual_runbook.get("ready") is True
            and post_capture_runbook.get("ready") is True
            and recommended_manual_runbook.get("ready") is True
            and recommended_post_capture_runbook.get("ready") is True
            and recommended_work_order.get("ready") is True
            and recommended_completion_receipt.get("template_ready") is True
            and launch_library.get("binding_handoff_ready") is True
            and gaps.get("all_request_capture_queue_ready") is True
            and gaps.get("phase3_blocker_closure_mapping_ready") is True
            and blocker_ledger.get("ready") is True
            and blocker_resolution_queue.get("ready") is True
            and next_unblocked_handoff.get("handoff_ready") is True
            and gaps.get("runtime_actuator_spike_handoff_ready") is True
        ),
        "recommended_request": {
            "request_name": recommendation.get("request_name"),
            "request_path": recommendation.get("request_path"),
            "status": recommendation.get("status"),
            "queue_rank": recommendation.get("queue_rank"),
            "selection_rationale": recommendation.get("selection_rationale"),
            "next_artifact_id": recommendation.get("next_artifact_id"),
            "next_artifact_path": recommendation.get("next_artifact_path"),
            "pending_artifact_ids": list_of_strings(recommendation.get("pending_artifact_ids")),
            "missing_approval_keys": list_of_strings(recommendation.get("missing_approval_keys")),
            "approval_command_class": approval_command.get("command_class"),
            "approval_command": list_of_strings(approval_command.get("command")),
            "records_approval_keys": list_of_strings(approval_command.get("records_approval_keys")),
            "approval_metadata_only": approval_command.get("metadata_only") is True,
            "requires_explicit_user_approval": approval_command.get("requires_explicit_user_approval") is True,
        },
        "recommended_reuse_capture": {
            "bundle_name": recommended_reuse_capture.get("bundle_name"),
            "bundle_path": recommended_reuse_capture.get("bundle_path"),
            "model_id": recommended_reuse_capture.get("model_id"),
            "candidate_trace_path": recommended_reuse_capture.get("candidate_trace_path"),
            "prompt_set_path": recommended_reuse_capture.get("prompt_set_path"),
            "runtime_capture_request_path": recommended_reuse_capture.get("runtime_capture_request_path"),
            "selection_rationale": recommended_reuse_capture.get("selection_rationale"),
            "required_result": recommended_reuse_capture.get("required_result"),
            "minimum_reuse_distance_observations": reuse_capture_contract.get("minimum_reuse_distance_observations"),
            "must_bind_prompt_identity_metadata": reuse_capture_contract.get("must_bind_prompt_identity_metadata") is True,
        },
        "recommended_launch_card_template": {
            "path": launch_template.get("launch_card_path"),
            "match_basis": launch_template.get("match_basis"),
            "template_ready": launch_template.get("template_ready"),
            "binding_handoff_ready": launch_template.get("binding_handoff_ready"),
            "binding_ready": launch_template.get("binding_ready"),
            "runtime_capture_command_ready": launch_template.get("runtime_capture_command_ready"),
            "task_count": launch_template.get("task_count"),
            "unbound_task_count": launch_template.get("unbound_task_count"),
            "missing_runtime_command_count": launch_template.get("missing_runtime_command_count"),
        },
        "completion_receipt_validation": recommended_completion_receipt_manifest.get("completion_receipt_validation_step") if isinstance(recommended_completion_receipt_manifest.get("completion_receipt_validation_step"), dict) else {},
        "next_unblocked_operator_handoff": next_unblocked_handoff,
        "handoff_counts": {
            "capture_queue_ready": gaps.get("all_request_capture_queue_ready"),
            "capture_queue_request_count": gaps.get("all_request_capture_queue_request_count"),
            "capture_queue_pending_artifact_count": gaps.get("all_request_capture_queue_pending_artifact_count"),
            "preflight_ready": gaps.get("recommended_runtime_capture_preflight_ready"),
            "preflight_pending_artifact_count": gaps.get("recommended_runtime_capture_preflight_pending_artifact_count"),
            "preflight_validator_command_count": gaps.get("recommended_runtime_capture_preflight_validator_command_count"),
            "preflight_runtime_closure_reason_count": gaps.get("recommended_runtime_capture_preflight_runtime_closure_reason_count"),
            "preflight_missing_item_count": gaps.get("recommended_runtime_capture_preflight_missing_item_count"),
            "approval_command_manifest_count": gaps.get("runtime_capture_approval_rebuild_command_manifest_count"),
            "approval_manifest_parity_ready": gaps.get("approval_manifest_parity_ready"),
            "approval_manifest_parity_matched_request_count": gaps.get("approval_manifest_parity_matched_request_count"),
            "validator_command_manifest_ready": gaps.get("all_request_validator_command_manifest_ready"),
            "validator_command_count": gaps.get("all_request_validator_command_manifest_total_command_count"),
            "receipt_fill_manifest_ready": gaps.get("all_request_receipt_fill_manifest_ready"),
            "receipt_fill_entry_count": gaps.get("all_request_receipt_fill_manifest_entry_count"),
            "receipt_fill_missing_count": gaps.get("all_request_receipt_fill_manifest_missing_count"),
            "receipt_fill_candidate_router_trace_entry_count": gaps.get("capture_result_receipt_fill_candidate_router_trace_entry_count"),
            "receipt_fill_candidate_router_trace_ready_count": gaps.get("capture_result_receipt_fill_candidate_router_trace_ready_count"),
            "receipt_fill_candidate_router_trace_missing_count": gaps.get("capture_result_receipt_fill_candidate_router_trace_missing_count"),
            "receipt_fill_managed_output_entry_count": gaps.get("capture_result_receipt_fill_managed_output_entry_count"),
            "receipt_fill_managed_output_ready_count": gaps.get("capture_result_receipt_fill_managed_output_ready_count"),
            "receipt_fill_managed_output_missing_count": gaps.get("capture_result_receipt_fill_managed_output_missing_count"),
            "receipt_fill_dense_output_entry_count": gaps.get("capture_result_receipt_fill_dense_output_entry_count"),
            "receipt_fill_dense_output_ready_count": gaps.get("capture_result_receipt_fill_dense_output_ready_count"),
            "receipt_fill_dense_output_missing_count": gaps.get("capture_result_receipt_fill_dense_output_missing_count"),
            "launch_card_count": gaps.get("phase3_launch_card_count"),
            "launch_card_task_count": gaps.get("phase3_launch_card_task_count"),
            "launch_card_unbound_task_count": gaps.get("phase3_launch_card_unbound_task_count"),
            "model_plane_artifact_writer_contract_request_ready": launch_library.get("model_plane_artifact_writer_contract_request_ready"),
            "model_plane_artifact_writer_contract_request_task_count": launch_library.get("model_plane_artifact_writer_contract_request_task_count"),
            "launch_card_saved_handoff_artifacts_ready": launch_library.get("saved_handoff_artifacts_ready"),
            "launch_card_saved_handoff_artifact_missing_count": launch_library.get("saved_handoff_artifact_missing_count"),
            "launch_card_saved_handoff_artifact_drifted_count": launch_library.get("saved_handoff_artifact_drifted_count"),
            "command_contract_ready": command_contract.get("ready"),
            "command_contract_runtime_ready": command_contract.get("runtime_capture_command_ready"),
            "command_contract_planned_capture_count": command_contract.get("planned_capture_count"),
            "command_contract_runtime_command_count": command_contract.get("runtime_command_option_count"),
            "command_contract_missing_runtime_command_count": command_contract.get("missing_runtime_command_count"),
            "execution_manual_ready": execution.get("manual_operator_capture_ready"),
            "execution_automated_ready": execution.get("automated_capture_ready"),
            "execution_command_option_count": execution.get("capture_command_option_count"),
            "execution_operator_command_option_count": execution.get("operator_command_option_count"),
            "execution_metadata_command_option_count": execution.get("metadata_command_option_count"),
            "execution_artifacts_with_command_count": execution.get("artifacts_with_capture_command_count"),
            "execution_manual_capture_required_count": execution.get("manual_capture_required_count"),
            "execution_missing_capture_command_count": execution.get("missing_capture_command_count"),
            "execution_missing_item_count": execution.get("missing_execution_item_count"),
            "all_execution_manual_ready": all_execution.get("manual_operator_capture_ready"),
            "all_execution_automated_ready": all_execution.get("automated_capture_ready"),
            "all_execution_request_count": all_execution.get("request_count"),
            "all_execution_ready_request_count": all_execution.get("manual_operator_capture_ready_count"),
            "all_execution_automated_request_count": all_execution.get("automated_capture_ready_count"),
            "all_execution_pending_artifact_count": all_execution.get("pending_artifact_count"),
            "all_execution_command_option_count": all_execution.get("capture_command_option_count"),
            "all_execution_operator_command_option_count": all_execution.get("operator_command_option_count"),
            "all_execution_metadata_command_option_count": all_execution.get("metadata_command_option_count"),
            "all_execution_artifacts_with_command_count": all_execution.get("artifacts_with_capture_command_count"),
            "all_execution_manual_capture_required_count": all_execution.get("manual_capture_required_count"),
            "all_execution_missing_capture_command_count": all_execution.get("missing_capture_command_count"),
            "all_execution_missing_item_count": all_execution.get("missing_execution_item_count"),
            "runtime_capture_launch_card_dir_provided": launch_card_directory.get("provided"),
            "runtime_capture_launch_card_dir_ready": launch_card_directory.get("directory_ready"),
            "runtime_capture_launch_card_dir_expected_card_count": launch_card_directory.get("expected_card_count"),
            "runtime_capture_launch_card_dir_matched_card_count": launch_card_directory.get("matched_card_count"),
            "runtime_capture_launch_card_dir_missing_card_count": launch_card_directory.get("missing_card_count"),
            "reuse_evidence_capture_valid": reuse_plan.get("valid"),
            "reuse_evidence_capture_bundle_count": reuse_plan.get("bundle_count"),
            "reuse_evidence_capture_ready_count": reuse_plan.get("reuse_ready_count"),
            "reuse_evidence_capture_blocked_count": reuse_plan.get("reuse_blocked_count"),
            "reuse_evidence_capture_trace_valid_count": reuse_plan.get("candidate_trace_valid_count"),
            "reuse_evidence_capture_receipt_ready_count": reuse_plan.get("candidate_trace_receipt_ready_count"),
            "reuse_evidence_capture_no_reuse_distance_count": reuse_plan.get("no_reuse_distance_observation_count"),
            "reuse_evidence_capture_prompt_identity_ready_count": reuse_plan.get("prompt_identity_ready_count"),
            "reuse_evidence_capture_prompt_identity_missing_count": reuse_plan.get("prompt_identity_metadata_missing_count"),
            "manual_runbook_ready": manual_runbook.get("ready"),
            "manual_runbook_request_count": manual_runbook.get("request_count"),
            "manual_runbook_ready_request_count": manual_runbook.get("ready_request_count"),
            "manual_runbook_manual_task_count": manual_runbook.get("manual_task_count"),
            "manual_runbook_runtime_command_task_count": manual_runbook.get("runtime_command_task_count"),
            "manual_runbook_validator_command_count": manual_runbook.get("validator_command_count"),
            "manual_runbook_missing_item_count": manual_runbook.get("missing_item_count"),
            "manual_runbook_missing_receipt_command_count": manual_runbook.get("missing_receipt_command_count"),
            "manual_runbook_missing_validator_command_count": manual_runbook.get("missing_validator_command_count"),
            "manual_runbook_missing_approval_command_count": manual_runbook.get("missing_approval_command_count"),
            "post_capture_intake_runbook_ready": post_capture_runbook.get("ready"),
            "post_capture_intake_runbook_request_count": post_capture_runbook.get("request_count"),
            "post_capture_intake_runbook_artifact_gate_count": post_capture_runbook.get("artifact_gate_count"),
            "post_capture_intake_runbook_ready_after_current_intake_count": post_capture_runbook.get("ready_after_current_intake_count"),
            "post_capture_intake_runbook_missing_after_current_intake_count": post_capture_runbook.get("missing_after_current_intake_count"),
            "post_capture_intake_runbook_validator_command_count": post_capture_runbook.get("validator_command_count"),
            "post_capture_intake_runbook_ready_to_update_bundle_count": post_capture_runbook.get("ready_to_update_bundle_count"),
            "post_capture_intake_runbook_phase4_candidate_count": post_capture_runbook.get("phase4_candidate_count"),
            "post_capture_intake_runbook_live_spike_candidate_count": post_capture_runbook.get("live_spike_candidate_count"),
            "post_capture_intake_runbook_missing_source_request_path_count": post_capture_runbook.get("missing_source_request_path_count"),
            "post_capture_intake_runbook_missing_prompt_set_path_count": post_capture_runbook.get("missing_prompt_set_path_count"),
            "post_capture_intake_runbook_missing_explicit_approval_count": post_capture_runbook.get("missing_explicit_approval_count"),
            "post_capture_intake_runbook_missing_prompt_traffic_ack_count": post_capture_runbook.get("missing_prompt_traffic_ack_count"),
            "post_capture_intake_runbook_missing_item_count": post_capture_runbook.get("missing_item_count"),
            "recommended_manual_runbook_ready": recommended_manual_runbook.get("ready"),
            "recommended_manual_runbook_request_count": recommended_manual_runbook.get("request_count"),
            "recommended_manual_runbook_manual_task_count": recommended_manual_runbook.get("manual_task_count"),
            "recommended_manual_runbook_runtime_command_task_count": recommended_manual_runbook.get("runtime_command_task_count"),
            "recommended_manual_runbook_validator_command_count": recommended_manual_runbook.get("validator_command_count"),
            "recommended_manual_runbook_missing_item_count": recommended_manual_runbook.get("missing_item_count"),
            "recommended_post_capture_intake_runbook_ready": recommended_post_capture_runbook.get("ready"),
            "recommended_post_capture_intake_runbook_request_count": recommended_post_capture_runbook.get("request_count"),
            "recommended_post_capture_intake_runbook_artifact_gate_count": recommended_post_capture_runbook.get("artifact_gate_count"),
            "recommended_post_capture_intake_runbook_ready_after_current_intake_count": recommended_post_capture_runbook.get("ready_after_current_intake_count"),
            "recommended_post_capture_intake_runbook_missing_after_current_intake_count": recommended_post_capture_runbook.get("missing_after_current_intake_count"),
            "recommended_post_capture_intake_runbook_validator_command_count": recommended_post_capture_runbook.get("validator_command_count"),
            "recommended_post_capture_intake_runbook_ready_to_update_bundle_count": recommended_post_capture_runbook.get("ready_to_update_bundle_count"),
            "recommended_post_capture_intake_runbook_missing_item_count": recommended_post_capture_runbook.get("missing_item_count"),
            "recommended_work_order_ready": recommended_work_order.get("ready"),
            "recommended_work_order_capture_step_count": recommended_work_order.get("capture_step_count"),
            "recommended_work_order_artifact_gate_count": recommended_work_order.get("artifact_gate_count"),
            "recommended_work_order_validator_command_count": recommended_work_order.get("validator_command_count"),
            "recommended_work_order_missing_item_count": recommended_work_order.get("missing_item_count"),
            "recommended_work_order_approval_command_ready": recommended_work_order.get("approval_command_ready"),
            "recommended_work_order_intake_command_ready": recommended_work_order.get("intake_command_ready"),
            "recommended_work_order_post_capture_sequence_ready": recommended_work_order.get("post_capture_sequence_ready"),
            "recommended_work_order_post_capture_sequence_step_count": recommended_work_order.get("post_capture_sequence_step_count"),
            "recommended_work_order_completion_validation_before_intake": recommended_work_order.get("completion_validation_before_intake"),
            "recommended_work_order_completion_validation_command_ready": recommended_work_order.get("completion_validation_command_ready"),
            "recommended_completion_receipt_template_ready": recommended_completion_receipt.get("template_ready"),
            "recommended_completion_receipt_capture_receipt_count": recommended_completion_receipt.get("capture_receipt_count"),
            "recommended_completion_receipt_expected_capture_step_count": recommended_completion_receipt.get("expected_capture_step_count"),
            "recommended_completion_receipt_validator_command_count": recommended_completion_receipt.get("validator_command_count"),
            "recommended_completion_receipt_missing_item_count": recommended_completion_receipt.get("missing_item_count"),
            "recommended_completion_receipt_receipt_complete": recommended_completion_receipt.get("receipt_complete"),
            "recommended_completion_receipt_ready_for_intake": recommended_completion_receipt.get("ready_for_capture_result_intake"),
            "recommended_completion_receipt_validation_command_ready": recommended_completion_receipt.get("validation_command_ready"),
            "blocker_closure_ready_scope": blocker_closure.get("ready_scope"),
            "blocker_closure_missing_evidence_count": blocker_closure.get("missing_evidence_count"),
            "blocker_evidence_ledger_ready": blocker_ledger.get("ready"),
            "blocker_evidence_ledger_reason_count": blocker_ledger.get("reason_count"),
            "blocker_evidence_ledger_row_count": blocker_ledger.get("evidence_row_count"),
            "blocker_evidence_ledger_expected_missing_evidence_count": blocker_ledger.get("expected_missing_evidence_count"),
            "blocker_evidence_ledger_runtime_capture_required_row_count": blocker_ledger.get("runtime_capture_required_row_count"),
            "blocker_evidence_ledger_future_adapter_required_row_count": blocker_ledger.get("future_adapter_required_row_count"),
            "blocker_evidence_ledger_receipt_bound_row_count": blocker_ledger.get("receipt_bound_row_count"),
            "blocker_evidence_ledger_dense_output_row_count": blocker_ledger.get("dense_output_row_count"),
            "blocker_evidence_ledger_live_proof_row_count": blocker_ledger.get("live_proof_row_count"),
            "blocker_evidence_ledger_runtime_actuator_row_count": blocker_ledger.get("runtime_actuator_row_count"),
            "blocker_evidence_ledger_validator_command_count": blocker_ledger.get("validator_command_count"),
            "blocker_evidence_ledger_missing_item_count": blocker_ledger.get("missing_item_count"),
            "blocker_resolution_queue_ready": blocker_resolution_queue.get("ready"),
            "blocker_resolution_queue_work_package_count": blocker_resolution_queue.get("work_package_count"),
            "blocker_resolution_queue_row_count": blocker_resolution_queue.get("queue_row_count"),
            "blocker_resolution_queue_expected_ledger_row_count": blocker_resolution_queue.get("expected_ledger_row_count"),
            "blocker_resolution_queue_runtime_capture_package_count": blocker_resolution_queue.get("runtime_capture_package_count"),
            "blocker_resolution_queue_future_adapter_package_count": blocker_resolution_queue.get("future_adapter_package_count"),
            "blocker_resolution_queue_runtime_capture_required_row_count": blocker_resolution_queue.get("runtime_capture_required_row_count"),
            "blocker_resolution_queue_future_adapter_required_row_count": blocker_resolution_queue.get("future_adapter_required_row_count"),
            "blocker_resolution_queue_receipt_bound_row_count": blocker_resolution_queue.get("receipt_bound_row_count"),
            "blocker_resolution_queue_validator_command_count": blocker_resolution_queue.get("validator_command_count"),
            "blocker_resolution_queue_completion_gate_count": blocker_resolution_queue.get("completion_gate_count"),
            "blocker_resolution_queue_dependency_edge_count": blocker_resolution_queue.get("dependency_edge_count"),
            "blocker_resolution_queue_missing_item_count": blocker_resolution_queue.get("missing_item_count"),
            "blocker_resolution_queue_next_unblocked_work_package_id": blocker_resolution_queue.get("next_unblocked_work_package_id"),
            "blocker_resolution_queue_next_unblocked_sequence_rank": blocker_resolution_queue.get("next_unblocked_sequence_rank"),
            "blocker_resolution_queue_next_unblocked_operator_stage": blocker_resolution_queue.get("next_unblocked_operator_stage"),
            "blocker_resolution_queue_next_unblocked_package_class": blocker_resolution_queue.get("next_unblocked_package_class"),
            "blocker_resolution_queue_next_unblocked_row_count": blocker_resolution_queue.get("next_unblocked_row_count"),
            "blocker_resolution_queue_next_unblocked_validator_command_count": blocker_resolution_queue.get("next_unblocked_validator_command_count"),
            "blocker_resolution_queue_next_unblocked_completion_gate_count": blocker_resolution_queue.get("next_unblocked_completion_gate_count"),
            "blocker_resolution_queue_next_unblocked_next_action": blocker_resolution_queue.get("next_unblocked_next_action"),
            "next_unblocked_operator_handoff_ready": next_unblocked_handoff.get("handoff_ready"),
            "next_unblocked_operator_handoff_work_order_ready": next_unblocked_handoff.get("work_order_ready"),
            "next_unblocked_operator_handoff_work_order_artifact": next_unblocked_handoff.get("work_order_artifact"),
            "next_unblocked_operator_handoff_work_order_next_artifact_id": next_unblocked_handoff.get("work_order_next_artifact_id"),
            "next_unblocked_operator_handoff_work_order_capture_step_count": next_unblocked_handoff.get("work_order_capture_step_count"),
            "next_unblocked_operator_handoff_work_order_validator_command_count": next_unblocked_handoff.get("work_order_validator_command_count"),
            "next_unblocked_operator_handoff_receipt_template_artifact": next_unblocked_handoff.get("completion_receipt_template_artifact"),
            "next_unblocked_operator_handoff_receipt_template_ready": next_unblocked_handoff.get("completion_receipt_template_ready"),
            "next_unblocked_operator_handoff_validation_command_ready": next_unblocked_handoff.get("completion_receipt_validation_command_ready"),
            "next_unblocked_operator_handoff_work_order_advances_next_package": next_unblocked_handoff.get("work_order_advances_next_package"),
            "blocker_resolution_queue_runtime_actuator_proof_handoff_ready": gaps.get("runtime_actuator_spike_handoff_ready"),
            "blocker_resolution_queue_runtime_actuator_live_spike_ready": gaps.get("runtime_actuator_spike_live_ready"),
            "blocker_resolution_queue_runtime_actuator_proof_requirement_count": gaps.get("runtime_actuator_spike_proof_requirement_count"),
            "blocker_resolution_queue_runtime_actuator_proof_requirement_id_count": runtime_actuator_proof_requirement_id_count,
            "blocker_resolution_queue_runtime_actuator_proof_artifact_count": gaps.get("runtime_actuator_spike_proof_artifact_count"),
            "blocker_resolution_queue_runtime_actuator_dependency_edge_count": gaps.get("runtime_actuator_spike_dependency_edge_count"),
            "blocker_resolution_queue_runtime_actuator_blocking_capability_count": gaps.get("runtime_actuator_spike_blocking_capability_count"),
            "blocker_resolution_queue_runtime_actuator_control_blocker_count": runtime_actuator_spike.get("control_blocker_count"),
        },
        "remaining_blockers": [item.get("id") for item in gaps.get("no_go_reasons", []) if isinstance(item, dict)],
        "next_actions": [
            "Review README.md in this handoff package before recording any approval metadata.",
            "Fill launch-card-binding-worksheet.json with real Model Plane callable ids or launch commands only after explicit approval.",
            "Use the approval command manifest to record approvals before runtime capture; this metadata step does not complete capture.",
            "After approved capture, fill the receipt-bound artifacts and run capture-result intake before bundle promotion.",
        ],
        "output_files": output_files,
        "warnings": warnings or [],
        "safety_contract": [
            "operator handoff package is metadata only",
            "operator handoff package generation does not launch model servers",
            "operator handoff package generation does not run Docker",
            "operator handoff package generation does not call endpoints",
            "operator handoff package generation does not inspect private tokens",
            "operator handoff package generation does not send prompt traffic",
            "operator handoff package generation does not mutate runtime residency",
            "operator handoff package generation does not claim live expert paging",
        ],
    }


def format_operator_handoff_readme(summary: JSONDict, manifest: JSONDict) -> str:
    request = manifest.get("recommended_request") if isinstance(manifest.get("recommended_request"), dict) else {}
    reuse_capture = manifest.get("recommended_reuse_capture") if isinstance(manifest.get("recommended_reuse_capture"), dict) else {}
    launch_card = manifest.get("recommended_launch_card_template") if isinstance(manifest.get("recommended_launch_card_template"), dict) else {}
    counts = manifest.get("handoff_counts") if isinstance(manifest.get("handoff_counts"), dict) else {}
    lines = [
        "# Phase 3 Operator Handoff",
        "",
        f"- Decision: `{markdown_escape(manifest.get('decision'))}`",
        f"- Packet ready: `{markdown_escape(manifest.get('packet_ready'))}`",
        f"- Phase 3 complete: `{markdown_escape(manifest.get('phase3_complete'))}`",
        f"- Operator handoff ready: `{markdown_escape(manifest.get('operator_handoff_ready'))}`",
        f"- Metadata only: `{markdown_escape(manifest.get('metadata_only'))}`",
        "",
        "## Recommended Capture",
        "",
        f"- Request: `{markdown_escape(request.get('request_name') or 'missing')}`",
        f"- Request path: `{markdown_escape(request.get('request_path') or 'missing')}`",
        f"- Next artifact: `{markdown_escape(request.get('next_artifact_id') or 'missing')}`",
        f"- Next artifact path: `{markdown_escape(request.get('next_artifact_path') or 'missing')}`",
        f"- Missing approval keys: `{markdown_escape(', '.join(list_of_strings(request.get('missing_approval_keys'))) or 'none')}`",
        f"- Approval command class: `{markdown_escape(request.get('approval_command_class') or 'missing')}`",
        "",
        "## Reuse Evidence Capture",
        "",
        f"- Bundle: `{markdown_escape(reuse_capture.get('bundle_name') or 'missing')}`",
        f"- Bundle path: `{markdown_escape(reuse_capture.get('bundle_path') or 'missing')}`",
        f"- Candidate trace: `{markdown_escape(reuse_capture.get('candidate_trace_path') or 'missing')}`",
        f"- Prompt set: `{markdown_escape(reuse_capture.get('prompt_set_path') or 'missing')}`",
        f"- Runtime request: `{markdown_escape(reuse_capture.get('runtime_capture_request_path') or 'missing')}`",
        f"- Required result: `{markdown_escape(reuse_capture.get('required_result') or 'missing')}`",
        f"- Minimum reuse-distance observations: `{markdown_escape(reuse_capture.get('minimum_reuse_distance_observations'))}`",
        f"- Prompt identity required: `{markdown_escape(reuse_capture.get('must_bind_prompt_identity_metadata'))}`",
        "",
        "## Launch Card",
        "",
        f"- Recommended template path: `{markdown_escape(launch_card.get('path') or 'missing')}`",
        f"- Template ready: `{markdown_escape(launch_card.get('template_ready'))}`",
        f"- Binding handoff ready: `{markdown_escape(launch_card.get('binding_handoff_ready'))}`",
        f"- Binding ready: `{markdown_escape(launch_card.get('binding_ready'))}`",
        f"- Runtime commands ready: `{markdown_escape(launch_card.get('runtime_capture_command_ready'))}`",
        f"- Tasks: `{markdown_escape(launch_card.get('task_count'))}`",
        f"- Unbound tasks: `{markdown_escape(launch_card.get('unbound_task_count'))}`",
        "",
        "## Counts",
        "",
        f"- Capture queue ready: `{markdown_escape(counts.get('capture_queue_ready'))}`",
        f"- Capture requests: `{markdown_escape(counts.get('capture_queue_request_count'))}`",
        f"- Pending runtime artifacts: `{markdown_escape(counts.get('capture_queue_pending_artifact_count'))}`",
        f"- Preflight ready: `{markdown_escape(counts.get('preflight_ready'))}`",
        f"- Preflight pending artifacts: `{markdown_escape(counts.get('preflight_pending_artifact_count'))}`",
        f"- Preflight validator commands: `{markdown_escape(counts.get('preflight_validator_command_count'))}`",
        f"- Preflight missing items: `{markdown_escape(counts.get('preflight_missing_item_count'))}`",
        f"- Approval commands: `{markdown_escape(counts.get('approval_command_manifest_count'))}`",
        f"- Approval parity ready: `{markdown_escape(counts.get('approval_manifest_parity_ready'))}`",
        f"- Approval parity matched requests: `{markdown_escape(counts.get('approval_manifest_parity_matched_request_count'))}`",
        f"- Validator commands: `{markdown_escape(counts.get('validator_command_count'))}`",
        f"- Receipt-fill ready: `{markdown_escape(counts.get('receipt_fill_manifest_ready'))}`",
        f"- Receipt fills missing: `{markdown_escape(counts.get('receipt_fill_missing_count'))}`",
        f"- Receipt-fill candidate router trace ready: `{markdown_escape(counts.get('receipt_fill_candidate_router_trace_ready_count'))}` / `{markdown_escape(counts.get('receipt_fill_candidate_router_trace_entry_count'))}`",
        f"- Receipt-fill candidate router trace missing: `{markdown_escape(counts.get('receipt_fill_candidate_router_trace_missing_count'))}`",
        f"- Receipt-fill managed outputs ready: `{markdown_escape(counts.get('receipt_fill_managed_output_ready_count'))}` / `{markdown_escape(counts.get('receipt_fill_managed_output_entry_count'))}`",
        f"- Receipt-fill managed outputs missing: `{markdown_escape(counts.get('receipt_fill_managed_output_missing_count'))}`",
        f"- Receipt-fill dense outputs ready: `{markdown_escape(counts.get('receipt_fill_dense_output_ready_count'))}` / `{markdown_escape(counts.get('receipt_fill_dense_output_entry_count'))}`",
        f"- Receipt-fill dense outputs missing: `{markdown_escape(counts.get('receipt_fill_dense_output_missing_count'))}`",
        f"- Command contract ready: `{markdown_escape(counts.get('command_contract_ready'))}`",
        f"- Command contract runtime ready: `{markdown_escape(counts.get('command_contract_runtime_ready'))}`",
        f"- Command contract planned captures: `{markdown_escape(counts.get('command_contract_planned_capture_count'))}`",
        f"- Command contract missing runtime commands: `{markdown_escape(counts.get('command_contract_missing_runtime_command_count'))}`",
        f"- Execution manual ready: `{markdown_escape(counts.get('execution_manual_ready'))}`",
        f"- Execution automated ready: `{markdown_escape(counts.get('execution_automated_ready'))}`",
        f"- Execution command options: `{markdown_escape(counts.get('execution_command_option_count'))}`",
        f"- Execution manual captures required: `{markdown_escape(counts.get('execution_manual_capture_required_count'))}`",
        f"- Execution missing capture commands: `{markdown_escape(counts.get('execution_missing_capture_command_count'))}`",
        f"- Execution missing items: `{markdown_escape(counts.get('execution_missing_item_count'))}`",
        f"- All-request execution manual ready: `{markdown_escape(counts.get('all_execution_manual_ready'))}`",
        f"- All-request execution automated ready: `{markdown_escape(counts.get('all_execution_automated_ready'))}`",
        f"- All-request execution ready requests: `{markdown_escape(counts.get('all_execution_ready_request_count'))}` / `{markdown_escape(counts.get('all_execution_request_count'))}`",
        f"- All-request execution pending artifacts: `{markdown_escape(counts.get('all_execution_pending_artifact_count'))}`",
        f"- All-request execution command options: `{markdown_escape(counts.get('all_execution_command_option_count'))}`",
        f"- All-request execution manual captures required: `{markdown_escape(counts.get('all_execution_manual_capture_required_count'))}`",
        f"- All-request execution missing capture commands: `{markdown_escape(counts.get('all_execution_missing_capture_command_count'))}`",
        f"- All-request execution missing items: `{markdown_escape(counts.get('all_execution_missing_item_count'))}`",
        f"- Filled launch-card directory provided: `{markdown_escape(counts.get('runtime_capture_launch_card_dir_provided'))}`",
        f"- Filled launch-card directory ready: `{markdown_escape(counts.get('runtime_capture_launch_card_dir_ready'))}`",
        f"- Filled launch-card directory matched cards: `{markdown_escape(counts.get('runtime_capture_launch_card_dir_matched_card_count'))}` / `{markdown_escape(counts.get('runtime_capture_launch_card_dir_expected_card_count'))}`",
        f"- Filled launch-card directory missing cards: `{markdown_escape(counts.get('runtime_capture_launch_card_dir_missing_card_count'))}`",
        f"- Reuse capture valid: `{markdown_escape(counts.get('reuse_evidence_capture_valid'))}`",
        f"- Reuse-ready traces: `{markdown_escape(counts.get('reuse_evidence_capture_ready_count'))}` / `{markdown_escape(counts.get('reuse_evidence_capture_bundle_count'))}`",
        f"- Reuse capture blocked traces: `{markdown_escape(counts.get('reuse_evidence_capture_blocked_count'))}`",
        f"- Reuse capture no-distance blockers: `{markdown_escape(counts.get('reuse_evidence_capture_no_reuse_distance_count'))}`",
        f"- Reuse capture prompt-identity blockers: `{markdown_escape(counts.get('reuse_evidence_capture_prompt_identity_missing_count'))}`",
        f"- Manual runbook ready: `{markdown_escape(counts.get('manual_runbook_ready'))}`",
        f"- Manual runbook ready requests: `{markdown_escape(counts.get('manual_runbook_ready_request_count'))}` / `{markdown_escape(counts.get('manual_runbook_request_count'))}`",
        f"- Manual runbook manual tasks: `{markdown_escape(counts.get('manual_runbook_manual_task_count'))}`",
        f"- Manual runbook validator commands: `{markdown_escape(counts.get('manual_runbook_validator_command_count'))}`",
        f"- Manual runbook missing items: `{markdown_escape(counts.get('manual_runbook_missing_item_count'))}`",
        f"- Post-capture intake runbook ready: `{markdown_escape(counts.get('post_capture_intake_runbook_ready'))}`",
        f"- Post-capture intake artifact gates: `{markdown_escape(counts.get('post_capture_intake_runbook_artifact_gate_count'))}`",
        f"- Post-capture intake missing after current intake: `{markdown_escape(counts.get('post_capture_intake_runbook_missing_after_current_intake_count'))}`",
        f"- Post-capture intake validator commands: `{markdown_escape(counts.get('post_capture_intake_runbook_validator_command_count'))}`",
        f"- Post-capture intake ready-to-update bundles: `{markdown_escape(counts.get('post_capture_intake_runbook_ready_to_update_bundle_count'))}`",
        f"- Post-capture intake missing source request paths: `{markdown_escape(counts.get('post_capture_intake_runbook_missing_source_request_path_count'))}`",
        f"- Post-capture intake missing prompt-set paths: `{markdown_escape(counts.get('post_capture_intake_runbook_missing_prompt_set_path_count'))}`",
        f"- Post-capture intake missing explicit approvals: `{markdown_escape(counts.get('post_capture_intake_runbook_missing_explicit_approval_count'))}`",
        f"- Post-capture intake missing prompt-traffic acknowledgements: `{markdown_escape(counts.get('post_capture_intake_runbook_missing_prompt_traffic_ack_count'))}`",
        f"- Recommended manual runbook ready: `{markdown_escape(counts.get('recommended_manual_runbook_ready'))}`",
        f"- Recommended manual runbook manual tasks: `{markdown_escape(counts.get('recommended_manual_runbook_manual_task_count'))}`",
        f"- Recommended post-capture runbook ready: `{markdown_escape(counts.get('recommended_post_capture_intake_runbook_ready'))}`",
        f"- Recommended post-capture artifact gates: `{markdown_escape(counts.get('recommended_post_capture_intake_runbook_artifact_gate_count'))}`",
        f"- Recommended post-capture missing after current intake: `{markdown_escape(counts.get('recommended_post_capture_intake_runbook_missing_after_current_intake_count'))}`",
        f"- Recommended work order ready: `{markdown_escape(counts.get('recommended_work_order_ready'))}`",
        f"- Recommended work order capture steps: `{markdown_escape(counts.get('recommended_work_order_capture_step_count'))}`",
        f"- Recommended work order validators: `{markdown_escape(counts.get('recommended_work_order_validator_command_count'))}`",
        f"- Recommended work order post-capture sequence ready: `{markdown_escape(counts.get('recommended_work_order_post_capture_sequence_ready'))}`",
        f"- Recommended work order post-capture sequence steps: `{markdown_escape(counts.get('recommended_work_order_post_capture_sequence_step_count'))}`",
        f"- Recommended work order validation before intake: `{markdown_escape(counts.get('recommended_work_order_completion_validation_before_intake'))}`",
        f"- Recommended completion receipt template ready: `{markdown_escape(counts.get('recommended_completion_receipt_template_ready'))}`",
        f"- Recommended completion receipt rows: `{markdown_escape(counts.get('recommended_completion_receipt_capture_receipt_count'))}`",
        f"- Recommended completion receipt ready for intake: `{markdown_escape(counts.get('recommended_completion_receipt_ready_for_intake'))}`",
        f"- Recommended completion receipt validation command ready: `{markdown_escape(counts.get('recommended_completion_receipt_validation_command_ready'))}`",
        f"- Next unblocked handoff ready: `{markdown_escape(counts.get('next_unblocked_operator_handoff_ready'))}`",
        f"- Next unblocked work order: `{markdown_escape(counts.get('next_unblocked_operator_handoff_work_order_artifact'))}`",
        f"- Next unblocked receipt template ready: `{markdown_escape(counts.get('next_unblocked_operator_handoff_receipt_template_ready'))}`",
        f"- Blocker closure scope: `{markdown_escape(counts.get('blocker_closure_ready_scope'))}`",
        f"- Missing evidence count: `{markdown_escape(counts.get('blocker_closure_missing_evidence_count'))}`",
        f"- Runtime actuator proof handoff ready: `{markdown_escape(counts.get('blocker_resolution_queue_runtime_actuator_proof_handoff_ready'))}`",
        f"- Runtime actuator proof requirements: `{markdown_escape(counts.get('blocker_resolution_queue_runtime_actuator_proof_requirement_count'))}`",
        f"- Runtime actuator proof artifacts: `{markdown_escape(counts.get('blocker_resolution_queue_runtime_actuator_proof_artifact_count'))}`",
        f"- Runtime actuator dependency edges: `{markdown_escape(counts.get('blocker_resolution_queue_runtime_actuator_dependency_edge_count'))}`",
        f"- Runtime actuator control blockers: `{markdown_escape(counts.get('blocker_resolution_queue_runtime_actuator_control_blocker_count'))}`",
        "",
        "## Completion Receipt Validation",
        "",
        "Run from repo root after filling a copy of `recommended-runtime-capture-completion-receipt.template.json`:",
        "",
        f"```text\n{markdown_escape(' '.join(list_of_strings((manifest.get('completion_receipt_validation') if isinstance(manifest.get('completion_receipt_validation'), dict) else {}).get('command'))))}\n```",
        "",
        "## Files",
        "",
        "| File | Kind |",
        "| --- | --- |",
    ]
    for item in manifest.get("output_files", []):
        if isinstance(item, dict):
            lines.append(f"| {markdown_escape(item.get('path') or 'missing')} | {markdown_escape(item.get('kind') or 'unknown')} |")
    blockers = list_of_strings(manifest.get("remaining_blockers"))
    lines.extend(["", "## Remaining Blockers", ""])
    if blockers:
        for blocker in blockers:
            lines.append(f"- `{markdown_escape(blocker)}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Safety Contract", ""])
    for item in manifest.get("safety_contract", []):
        lines.append(f"- {markdown_escape(item)}")
    lines.append("")
    return "\n".join(lines)


def write_operator_handoff_dir(summary: JSONDict, output_dir: Path) -> JSONDict:
    output_dir.mkdir(parents=True, exist_ok=True)
    written_files: list[JSONDict] = []
    warnings: list[str] = []
    write_json_artifact(output_dir, "phase3-evidence-packet.json", summary, "phase3_evidence_packet_json", written_files)
    (output_dir / "phase3-evidence-packet.md").write_text(format_markdown_report(summary), encoding="utf-8")
    written_files.append({"path": "phase3-evidence-packet.md", "kind": "phase3_evidence_packet_markdown"})

    launch_library = summary.get("phase3_launch_card_library_summary")
    if isinstance(launch_library, dict) and launch_library:
        worksheet = plan_phase3_launch_card_library.build_binding_worksheet(launch_library)
        write_json_artifact(output_dir, "launch-card-binding-worksheet.json", worksheet, "phase3_launch_card_binding_worksheet", written_files)
        contract_request = plan_phase3_launch_card_library.build_model_plane_artifact_writer_contract_request(launch_library)
        write_json_artifact(output_dir, "model-plane-artifact-writer-contract-request.json", contract_request, "model_plane_artifact_writer_contract_request", written_files)

    else:
        warnings.append("phase3_launch_card_library_summary_missing")

    try:
        launch_card = build_recommended_runtime_capture_launch_card(summary)
    except (KeyError, TypeError, ValueError) as exc:
        warnings.append(f"recommended_launch_card_template_not_written: {exc}")
    else:
        write_json_artifact(
            output_dir,
            "recommended-runtime-capture-launch-card.template.json",
            launch_card,
            "recommended_runtime_capture_launch_card_template",
            written_files,
        )

    write_json_artifact(output_dir, "recommended-runtime-capture-preflight.json", summary.get("recommended_runtime_capture_preflight_manifest", {}), "recommended_runtime_capture_preflight_manifest", written_files)
    write_json_artifact(output_dir, "recommended-runtime-capture-command-contract.json", summary.get("recommended_runtime_capture_command_contract", {}), "recommended_runtime_capture_command_contract", written_files)
    write_json_artifact(output_dir, "recommended-runtime-capture-execution-coverage.json", summary.get("recommended_runtime_capture_execution_coverage_manifest", {}), "recommended_runtime_capture_execution_coverage_manifest", written_files)
    write_json_artifact(output_dir, "all-request-runtime-capture-execution-coverage.json", summary.get("all_request_runtime_capture_execution_coverage_manifest", {}), "all_request_runtime_capture_execution_coverage_manifest", written_files)
    write_json_artifact(output_dir, "runtime-capture-launch-card-directory.json", summary.get("runtime_capture_launch_card_directory_manifest", {}), "runtime_capture_launch_card_directory_manifest", written_files)
    write_json_artifact(output_dir, "reuse-evidence-capture-plan.json", summary.get("phase3_reuse_evidence_capture_plan", {}), "phase3_reuse_evidence_capture_plan", written_files)
    write_json_artifact(output_dir, "manual-capture-runbook.json", summary.get("all_request_manual_capture_runbook_manifest", {}), "all_request_manual_capture_runbook_manifest", written_files)
    write_json_artifact(output_dir, "post-capture-intake-runbook.json", summary.get("all_request_post_capture_intake_runbook_manifest", {}), "all_request_post_capture_intake_runbook_manifest", written_files)
    write_json_artifact(output_dir, "recommended-manual-capture-runbook.json", summary.get("recommended_manual_capture_runbook_manifest", {}), "recommended_manual_capture_runbook_manifest", written_files)
    write_json_artifact(output_dir, "recommended-post-capture-intake-runbook.json", summary.get("recommended_post_capture_intake_runbook_manifest", {}), "recommended_post_capture_intake_runbook_manifest", written_files)
    write_json_artifact(output_dir, "recommended-runtime-capture-work-order.json", summary.get("recommended_runtime_capture_work_order_manifest", {}), "recommended_runtime_capture_work_order_manifest", written_files)
    write_json_artifact(output_dir, "recommended-runtime-capture-completion-receipt.template.json", summary.get("recommended_runtime_capture_completion_receipt_template_manifest", {}), "recommended_runtime_capture_completion_receipt_template", written_files)
    write_json_artifact(output_dir, "next-unblocked-operator-handoff.json", summary.get("phase3_next_unblocked_operator_handoff_summary", {}), "phase3_next_unblocked_operator_handoff", written_files)
    write_json_artifact(output_dir, "capture-queue.json", summary.get("all_request_capture_queue_manifest", []), "all_request_capture_queue_manifest", written_files)
    write_json_artifact(output_dir, "approval-command-manifest.json", summary.get("all_request_approval_command_manifest", []), "all_request_approval_command_manifest", written_files)
    write_json_artifact(output_dir, "validator-command-manifest.json", summary.get("all_request_validator_command_manifest", []), "all_request_validator_command_manifest", written_files)
    write_json_artifact(output_dir, "receipt-fill-manifest.json", summary.get("all_request_receipt_fill_manifest", []), "all_request_receipt_fill_manifest", written_files)
    write_json_artifact(output_dir, "receipt-fill-command-manifest.json", summary.get("all_request_receipt_fill_command_manifest", []), "all_request_receipt_fill_command_manifest", written_files)
    write_json_artifact(output_dir, "downstream-handoff-manifest.json", summary.get("all_request_downstream_handoff_manifest", []), "all_request_downstream_handoff_manifest", written_files)
    write_json_artifact(output_dir, "blocker-closure-manifest.json", summary.get("phase3_blocker_closure_manifest", []), "phase3_blocker_closure_manifest", written_files)
    write_json_artifact(output_dir, "blocker-evidence-ledger.json", summary.get("phase3_blocker_evidence_ledger_manifest", {}), "phase3_blocker_evidence_ledger_manifest", written_files)
    write_json_artifact(output_dir, "blocker-resolution-queue.json", summary.get("phase3_blocker_resolution_queue_manifest", {}), "phase3_blocker_resolution_queue_manifest", written_files)

    final_files = list(written_files) + [
        {"path": "operator-handoff-manifest.json", "kind": "operator_handoff_manifest"},
        {"path": "README.md", "kind": "operator_handoff_readme"},
    ]
    manifest = build_operator_handoff_manifest(summary, output_files=final_files, warnings=warnings)
    (output_dir / "operator-handoff-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(format_operator_handoff_readme(summary, manifest), encoding="utf-8")
    return manifest

def build_recommended_runtime_capture_launch_card(summary: JSONDict) -> JSONDict:
    command_contract = summary.get("recommended_runtime_capture_command_contract")
    if not isinstance(command_contract, dict) or not command_contract:
        raise ValueError("recommended runtime-capture command contract is missing")
    return plan_phase3_runtime_capture_commands.build_launch_card_template(command_contract)


def write_recommended_runtime_capture_launch_card(summary: JSONDict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    launch_card = build_recommended_runtime_capture_launch_card(summary)
    output_path.write_text(json.dumps(launch_card, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def write_markdown_report(summary: JSONDict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_markdown_report(summary), encoding="utf-8")


def print_human_summary(summary: JSONDict) -> None:
    print("MoE Run Anyway Phase 3 evidence packet")
    print(f"Valid: {summary['valid']}")
    print(f"Packet ready: {summary['packet_ready']}")
    print(f"Phase 3 complete: {summary['phase3_complete']}")
    print(f"Decision: {summary['decision']}")
    if summary["errors"]:
        print("Errors:")
        for error in summary["errors"]:
            print(f"  - {error}")
    print("Evidence items:")
    for item in summary["evidence_items"]:
        print(f"  - {item['id']}: {item['status']}")
    checklist = summary.get("promotion_checklist", [])
    if isinstance(checklist, list) and checklist:
        print("Promotion checklist:")
        for item in checklist:
            if isinstance(item, dict):
                print(f"  - {item.get('id')}: {item.get('status')}")
    blocker_closure_manifest = summary.get("phase3_blocker_closure_manifest")
    blocker_closure_summary = summary.get("phase3_blocker_closure_summary")
    if isinstance(blocker_closure_manifest, list) and blocker_closure_manifest:
        blocker_closure_summary = blocker_closure_summary if isinstance(blocker_closure_summary, dict) else {}
        print(
            "Phase 3 blocker closure: "
            f"ready={blocker_closure_summary.get('ready')} "
            f"mapping_ready={blocker_closure_summary.get('mapping_ready')} "
            f"evidence_complete={blocker_closure_summary.get('evidence_complete')} "
            f"scope={blocker_closure_summary.get('ready_scope')} "
            f"missing_evidence={blocker_closure_summary.get('missing_evidence_count')} "
            f"reasons={blocker_closure_summary.get('reason_count')} "
            f"mapped={blocker_closure_summary.get('mapped_reason_count')} "
            f"unresolved={blocker_closure_summary.get('unresolved_reason_count')} "
            f"runtime_capture={blocker_closure_summary.get('runtime_capture_required_count')} "
            f"future_adapter={blocker_closure_summary.get('future_adapter_required_count')} "
            f"commands={blocker_closure_summary.get('validator_command_count')}"
        )
    blocker_ledger = summary.get("phase3_blocker_evidence_ledger_summary")
    if isinstance(blocker_ledger, dict) and blocker_ledger:
        print(
            "Phase 3 blocker evidence ledger: "
            f"ready={blocker_ledger.get('ready')} "
            f"rows={blocker_ledger.get('evidence_row_count')}/"
            f"{blocker_ledger.get('expected_missing_evidence_count')} "
            f"reasons={blocker_ledger.get('reason_count')} "
            f"runtime_capture={blocker_ledger.get('runtime_capture_required_row_count')} "
            f"future_adapter={blocker_ledger.get('future_adapter_required_row_count')} "
            f"receipt_bound={blocker_ledger.get('receipt_bound_row_count')} "
            f"dense_outputs={blocker_ledger.get('dense_output_row_count')} "
            f"live_proof={blocker_ledger.get('live_proof_row_count')} "
            f"runtime_actuator={blocker_ledger.get('runtime_actuator_row_count')} "
            f"commands={blocker_ledger.get('validator_command_count')} "
            f"missing_items={blocker_ledger.get('missing_item_count')}"
        )
    blocker_resolution_queue = summary.get("phase3_blocker_resolution_queue_summary")
    if isinstance(blocker_resolution_queue, dict) and blocker_resolution_queue:
        print(
            "Phase 3 blocker resolution queue: "
            f"ready={blocker_resolution_queue.get('ready')} "
            f"packages={blocker_resolution_queue.get('work_package_count')} "
            f"rows={blocker_resolution_queue.get('queue_row_count')}/"
            f"{blocker_resolution_queue.get('expected_ledger_row_count')} "
            f"runtime_packages={blocker_resolution_queue.get('runtime_capture_package_count')} "
            f"future_packages={blocker_resolution_queue.get('future_adapter_package_count')} "
            f"commands={blocker_resolution_queue.get('validator_command_count')} "
            f"gates={blocker_resolution_queue.get('completion_gate_count')} "
            f"dependencies={blocker_resolution_queue.get('dependency_edge_count')} "
            f"missing_items={blocker_resolution_queue.get('missing_item_count')}"
        )
    launch_card_library = summary.get("phase3_launch_card_library_summary")
    if isinstance(launch_card_library, dict) and launch_card_library:
        print(
            "Phase 3 launch-card library: "
            f"library_ready={launch_card_library.get('library_ready')} "
            f"execution_ready={launch_card_library.get('execution_ready')} "
            f"cards={launch_card_library.get('card_count')} "
            f"template_ready={launch_card_library.get('template_ready_count')} "
            f"model_plane_ready={launch_card_library.get('model_plane_binding_ready_count')} "
            f"runtime_ready={launch_card_library.get('runtime_capture_command_ready_count')} "
            f"tasks={launch_card_library.get('task_count')} "
            f"missing_runtime_commands={launch_card_library.get('missing_runtime_command_count')} "
            f"handoff_ready={launch_card_library.get('binding_handoff_ready')} "
            f"handoff_tasks={launch_card_library.get('binding_handoff_task_count')} "
            f"unbound_tasks={launch_card_library.get('unbound_task_count')} "
            f"saved_handoff_ready={launch_card_library.get('saved_handoff_artifacts_ready')} "
            f"saved_handoff_missing={launch_card_library.get('saved_handoff_artifact_missing_count')} "
            f"saved_handoff_drifted={launch_card_library.get('saved_handoff_artifact_drifted_count')}"
        )
    all_queue = summary.get("all_request_capture_queue_manifest")
    all_queue_summary = summary.get("all_request_capture_queue_summary")
    if isinstance(all_queue, list) and all_queue:
        all_queue_summary = all_queue_summary if isinstance(all_queue_summary, dict) else {}
        print(
            "All-request capture queue: "
            f"ready={all_queue_summary.get('ready')} "
            f"requests={all_queue_summary.get('request_count')} "
            f"pending_artifacts={all_queue_summary.get('pending_artifact_count')} "
            f"validators={all_queue_summary.get('validator_command_count')}"
        )
    all_validator_commands = summary.get("all_request_validator_command_manifest")
    all_validator_summary = summary.get("all_request_validator_command_summary")
    if isinstance(all_validator_commands, list) and all_validator_commands:
        all_validator_summary = all_validator_summary if isinstance(all_validator_summary, dict) else {}
        print(
            "All-request validator commands: "
            f"ready={all_validator_summary.get('ready')} "
            f"requests={all_validator_summary.get('request_count')} "
            f"runtime_artifacts={all_validator_summary.get('runtime_artifact_count')} "
            f"future_artifacts={all_validator_summary.get('future_artifact_count')} "
            f"commands={all_validator_summary.get('total_validator_command_count')}"
        )
    all_downstream_handoffs = summary.get("all_request_downstream_handoff_manifest")
    all_downstream_summary = summary.get("all_request_downstream_handoff_summary")
    if isinstance(all_downstream_handoffs, list) and all_downstream_handoffs:
        all_downstream_summary = all_downstream_summary if isinstance(all_downstream_summary, dict) else {}
        print(
            "All-request downstream handoffs: "
            f"ready={all_downstream_summary.get('ready')} "
            f"requests={all_downstream_summary.get('request_count')} "
            f"policy={all_downstream_summary.get('policy_candidate_trace_handoff_ready_count')} "
            f"dense={all_downstream_summary.get('dense_fallback_capture_handoff_ready_count')} "
            f"live={all_downstream_summary.get('live_capability_proof_handoff_ready_count')} "
            f"all={all_downstream_summary.get('all_downstream_handoff_ready_count')}"
        )
    all_receipt_fills = summary.get("all_request_receipt_fill_manifest")
    all_receipt_summary = summary.get("all_request_receipt_fill_summary")
    if isinstance(all_receipt_fills, list) and all_receipt_fills:
        all_receipt_summary = all_receipt_summary if isinstance(all_receipt_summary, dict) else {}
        print(
            "All-request receipt fills: "
            f"ready={all_receipt_summary.get('ready')} "
            f"requests={all_receipt_summary.get('request_count')} "
            f"entries={all_receipt_summary.get('entry_count')} "
            f"ready_receipts={all_receipt_summary.get('ready_count')} "
            f"missing={all_receipt_summary.get('missing_count')} "
            f"validators={all_receipt_summary.get('validator_command_count')}"
        )
    receipt_validator_parity = summary.get("all_request_receipt_validator_parity_summary")
    if isinstance(receipt_validator_parity, dict) and receipt_validator_parity:
        metadata_mismatches = receipt_validator_parity.get("metadata_mismatch_artifacts")
        missing_from_receipts = receipt_validator_parity.get("missing_from_receipt_fill_artifact_keys")
        missing_from_runtime = receipt_validator_parity.get("missing_from_runtime_validator_artifact_keys")
        print(
            "All-request receipt/validator parity: "
            f"ready={receipt_validator_parity.get('ready')} "
            f"runtime_artifacts={receipt_validator_parity.get('runtime_artifact_count')} "
            f"receipt_entries={receipt_validator_parity.get('receipt_fill_entry_count')} "
            f"matched={receipt_validator_parity.get('matched_artifact_count')} "
            f"mismatches={len(metadata_mismatches) if isinstance(metadata_mismatches, list) else 0} "
            f"missing_receipts={len(missing_from_receipts) if isinstance(missing_from_receipts, list) else 0} "
            f"missing_runtime={len(missing_from_runtime) if isinstance(missing_from_runtime, list) else 0}"
        )
    receipt_command_manifest = summary.get("all_request_receipt_fill_command_manifest")
    receipt_command_summary = summary.get("all_request_receipt_fill_command_summary")
    if isinstance(receipt_command_manifest, list) and receipt_command_manifest:
        receipt_command_summary = receipt_command_summary if isinstance(receipt_command_summary, dict) else {}
        print(
            "All-request receipt fill commands: "
            f"ready={receipt_command_summary.get('ready')} "
            f"requests={receipt_command_summary.get('request_count')} "
            f"entries={receipt_command_summary.get('entry_count')} "
            f"commands={receipt_command_summary.get('validator_command_count')} "
            f"missing_commands={receipt_command_summary.get('missing_command_entry_count')} "
            f"missing_receipt_paths={receipt_command_summary.get('missing_receipt_path_count')}"
        )
    recommendation = summary.get("recommended_runtime_capture_request")
    if isinstance(recommendation, dict):
        print(
            "Recommended runtime capture: "
            f"{recommendation.get('request_name')} status={recommendation.get('status')} "
            f"next={recommendation.get('next_artifact_id')}"
        )
    preflight_summary = summary.get("recommended_runtime_capture_preflight_summary")
    if isinstance(preflight_summary, dict) and preflight_summary:
        print(
            "Recommended runtime capture preflight: "
            f"ready={preflight_summary.get('ready')} "
            f"request={preflight_summary.get('request_name')} "
            f"pending={preflight_summary.get('pending_artifact_count')} "
            f"receipts={preflight_summary.get('receipt_entry_count')} "
            f"commands={preflight_summary.get('validator_command_count')} "
            f"runtime_closures={preflight_summary.get('runtime_closure_reason_count')} "
            f"missing={preflight_summary.get('missing_preflight_item_count')}"
        )
    command_contract_summary = summary.get("recommended_runtime_capture_command_contract_summary")
    if isinstance(command_contract_summary, dict) and command_contract_summary:
        print(
            "Recommended runtime capture command contract: "
            f"ready={command_contract_summary.get('ready')} "
            f"runtime_ready={command_contract_summary.get('runtime_capture_command_ready')} "
            f"planned={command_contract_summary.get('planned_capture_count')} "
            f"runtime_commands={command_contract_summary.get('runtime_command_option_count')} "
            f"missing_runtime_commands={command_contract_summary.get('missing_runtime_command_count')} "
            f"launch_card_binding_ready={command_contract_summary.get('launch_card_binding_ready')} "
            f"launch_card_bound_tasks={command_contract_summary.get('launch_card_binding_bound_task_count')}"
        )
    execution_summary = summary.get("recommended_runtime_capture_execution_coverage_summary")
    if isinstance(execution_summary, dict) and execution_summary:
        print(
            "Recommended runtime capture execution: "
            f"manual_ready={execution_summary.get('manual_operator_capture_ready')} "
            f"automated_ready={execution_summary.get('automated_capture_ready')} "
            f"pending={execution_summary.get('pending_artifact_count')} "
            f"runtime_command_options={execution_summary.get('capture_command_option_count')} "
            f"operator_command_options={execution_summary.get('operator_command_option_count')} "
            f"metadata_command_options={execution_summary.get('metadata_command_option_count')} "
            f"with_runtime_commands={execution_summary.get('artifacts_with_capture_command_count')} "
            f"manual_required={execution_summary.get('manual_capture_required_count')} "
            f"missing_commands={execution_summary.get('missing_capture_command_count')}"
        )
    all_execution_summary = summary.get("all_request_runtime_capture_execution_coverage_summary")
    if isinstance(all_execution_summary, dict) and all_execution_summary:
        print(
            "All-request runtime capture execution: "
            f"manual_ready={all_execution_summary.get('manual_operator_capture_ready')} "
            f"automated_ready={all_execution_summary.get('automated_capture_ready')} "
            f"requests={all_execution_summary.get('request_count')} "
            f"ready_requests={all_execution_summary.get('manual_operator_capture_ready_count')} "
            f"pending={all_execution_summary.get('pending_artifact_count')} "
            f"runtime_command_options={all_execution_summary.get('capture_command_option_count')} "
            f"manual_required={all_execution_summary.get('manual_capture_required_count')} "
            f"missing_commands={all_execution_summary.get('missing_capture_command_count')}"
        )
    print("Remaining gaps:")
    for key, value in summary["remaining_gaps"].items():
        if key == "no_go_reasons":
            print(f"  - {key}: {', '.join(reason['id'] for reason in value)}")
        else:
            print(f"  - {key}: {value}")
    print("Safety contract:")
    for item in summary["safety_contract"]:
        print(f"  - {item}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_path", nargs="?", type=Path, default=DEFAULT_TRACE_PATH)
    parser.add_argument("inventory_path", nargs="?", type=Path, default=DEFAULT_INVENTORY_PATH)
    parser.add_argument("policies_path", nargs="?", type=Path, default=DEFAULT_POLICIES_PATH)
    parser.add_argument("managed_plan_path", nargs="?", type=Path, default=DEFAULT_MANAGED_PLAN_PATH)
    parser.add_argument("--fallback-artifact-path", type=Path, default=None)
    parser.add_argument("--live-proof-artifact-path", type=Path, default=None)
    parser.add_argument("--runtime-capture-launch-card", type=Path, default=None, help="validate a filled launch-card for the recommended runtime-capture request without executing it")
    parser.add_argument("--runtime-capture-launch-card-dir", type=Path, default=None, help="validate filled launch-cards for all runtime-capture requests without executing them")
    parser.add_argument("--policy-candidate-trace-receipt-path", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    parser.add_argument("--output-md", type=Path, help="write a Markdown evidence packet report")
    parser.add_argument("--output-runtime-capture-launch-card", type=Path, help="write the recommended planned-only runtime-capture launch-card template")
    parser.add_argument("--output-operator-handoff-dir", type=Path, help="write a metadata-only Phase 3 operator handoff package directory")
    return parser


def plan_paths(
    trace_path: Path,
    inventory_path: Path,
    policies_path: Path,
    managed_plan_path: Path,
    *,
    fallback_artifact_path: Path | None = None,
    live_proof_artifact_path: Path | None = None,
    runtime_capture_launch_card_path: Path | None = None,
    runtime_capture_launch_card_dir: Path | None = None,
    policy_candidate_trace_receipt_path: Path | None = None,
) -> tuple[int, JSONDict | None, str | None]:
    try:
        summary = build_packet_summary(
            trace_path,
            inventory_path,
            policies_path,
            managed_plan_path,
            fallback_artifact_path=fallback_artifact_path,
            live_proof_artifact_path=live_proof_artifact_path,
            runtime_capture_launch_card_path=runtime_capture_launch_card_path,
            runtime_capture_launch_card_dir=runtime_capture_launch_card_dir,
            policy_candidate_trace_receipt_path=policy_candidate_trace_receipt_path,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not build Phase 3 evidence packet: {exc}"
    return (0 if summary["valid"] else 2), summary, None


def main_from_test_paths(trace_path: Path, inventory_path: Path, policies_path: Path, managed_plan_path: Path) -> int:
    status, _, _ = plan_paths(trace_path, inventory_path, policies_path, managed_plan_path)
    return status


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary, error_message = plan_paths(
        args.trace_path,
        args.inventory_path,
        args.policies_path,
        args.managed_plan_path,
        fallback_artifact_path=args.fallback_artifact_path,
        live_proof_artifact_path=args.live_proof_artifact_path,
        runtime_capture_launch_card_path=args.runtime_capture_launch_card,
        runtime_capture_launch_card_dir=args.runtime_capture_launch_card_dir,
        policy_candidate_trace_receipt_path=args.policy_candidate_trace_receipt_path,
    )
    if error_message:
        print(error_message, file=sys.stderr)
        return status
    assert summary is not None
    if args.output_md:
        write_markdown_report(summary, args.output_md)
    if args.output_runtime_capture_launch_card:
        try:
            write_recommended_runtime_capture_launch_card(summary, args.output_runtime_capture_launch_card)
        except ValueError as exc:
            print(f"Could not write recommended runtime-capture launch card: {exc}", file=sys.stderr)
            return 2
    if args.output_operator_handoff_dir:
        write_operator_handoff_dir(summary, args.output_operator_handoff_dir)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_human_summary(summary)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
