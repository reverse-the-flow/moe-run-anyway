#!/usr/bin/env python3
"""Audit Phase 3 runtime-capture results after artifacts are filled.

This planner reads saved runtime-capture request JSON files and the local
artifacts those requests point at. It reuses the policy-candidate trace, dense
fallback, live-capability proof, and bundle validators to decide whether a
capture result can update the Phase 3 bundle or clear the next gate. It does
not launch runtimes, run Docker, call endpoints, inspect secrets, mutate
residency, or send prompt traffic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import build_phase3_runtime_capture_request
import plan_phase3_dense_fallback_capture
import plan_phase3_live_capability_proof
import plan_phase3_policy_candidate_trace
import plan_phase3_real_evidence_bundle
import plan_phase3_runtime_capture_request
import plan_phase3_capture_completion_receipt
import phase3_output_receipts


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "memory-moe-mvp" / "phase3-real-evidence"
REQUEST_GLOB = "*.runtime-capture-request.json"
SUPPORTED_SCHEMA_VERSION = "moe-phase3-capture-result-intake-v1"

JSONDict = dict[str, Any]


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


def load_json(path: Path) -> JSONDict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{display_path(path) or path} must be a JSON object")
    return data


def items_by_id(items: Any) -> dict[str, JSONDict]:
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


def request_artifact_path(request: JSONDict, artifact_id: str, *, future: bool = False) -> Path | None:
    collection = "future_artifacts" if future else "requested_artifacts"
    item = items_by_id(request.get(collection)).get(artifact_id)
    if item is None:
        return None
    return resolve_repo_path(item.get("path"))


def request_artifact_source_path(request: JSONDict, artifact_id: str, source_key: str, *, future: bool = False) -> Path | None:
    collection = "future_artifacts" if future else "requested_artifacts"
    item = items_by_id(request.get(collection)).get(artifact_id)
    if item is None:
        return None
    source = item.get("source")
    if not isinstance(source, dict):
        return None
    return resolve_repo_path(source.get(source_key))


def request_approval(request: JSONDict, key: str) -> bool:
    approvals = request.get("approvals")
    if not isinstance(approvals, dict):
        return False
    return approvals.get(key) is True


def request_approvals_ready(request: JSONDict) -> bool:
    return all(
        request_approval(request, key)
        for key in (
            "router_trace_capture_approved",
            "managed_output_capture_approved",
            "dense_output_capture_approved",
            "runtime_prompt_traffic_approved",
        )
    )

def normalized_path_text(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().replace("\\", "/")


def resolved_path_text(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def receipt_path_matches(value: Any, expected_path: Path | None) -> bool:
    if expected_path is None:
        return False
    actual = normalized_path_text(value)
    if actual is None:
        return False
    expected_display = normalized_path_text(display_path(expected_path))
    if expected_display is not None and actual == expected_display:
        return True
    actual_path = resolve_repo_path(actual)
    if actual_path is None:
        return False
    return resolved_path_text(actual_path) == resolved_path_text(expected_path)


def output_receipt_binding(
    label: str,
    output_path: Path | None,
    *,
    request_path: Path,
    prompt_set_path: Path | None,
) -> JSONDict:
    if output_path is None:
        return {
            "label": label,
            "path": None,
            "exists": False,
            "receipt_ready": False,
            "request_path_matches": False,
            "prompt_set_path_matches": False,
            "ready": False,
            "errors": [],
            "blockers": [f"{label}_output_summary_missing"],
        }
    if not output_path.exists():
        return {
            "label": label,
            "path": display_path(output_path),
            "exists": False,
            "receipt_ready": False,
            "request_path_matches": False,
            "prompt_set_path_matches": False,
            "ready": False,
            "errors": [],
            "blockers": [f"{label}_output_summary_missing"],
        }

    payload = load_json(output_path)
    receipt = payload.get("capture_receipt") if isinstance(payload, dict) else None
    validation = phase3_output_receipts.validate_capture_receipt_source_binding(
        receipt,
        repo_root=ROOT,
        expected_label=label,
        expected_source_request_path=request_path,
        expected_source_prompt_set_path=prompt_set_path,
    )
    errors = list(validation["errors"])
    blockers: list[str] = []
    receipt_ready = validation["receipt_ready"] is True
    request_path_matches = validation["source_request_path_matches"] is True
    prompt_set_path_matches = validation["source_prompt_set_path_matches"] is True

    if receipt_ready:
        if not request_path_matches:
            errors.append(f"{label} capture_receipt.source_request_path must match request path")
        if not prompt_set_path_matches:
            errors.append(f"{label} capture_receipt.source_prompt_set_path must match request prompt set path")
    elif validation["shape_valid"]:
        blockers.append(f"{label}_capture_receipt_not_ready")
    else:
        blockers.append(f"{label}_capture_receipt_invalid")

    ready = receipt_ready and not errors and request_path_matches and prompt_set_path_matches
    return {
        "label": label,
        "path": display_path(output_path),
        "exists": True,
        "receipt_ready": receipt_ready,
        "request_path_matches": request_path_matches,
        "prompt_set_path_matches": prompt_set_path_matches,
        "source_request_path_exists": validation.get("source_request_path_exists"),
        "source_prompt_set_path_exists": validation.get("source_prompt_set_path_exists"),
        "ready": ready,
        "errors": errors,
        "blockers": blockers,
    }


def capture_receipt_binding_for_request(
    request_path: Path,
    request: JSONDict,
    *,
    prompt_set_path: Path | None,
    managed_output_path: Path | None,
    dense_output_path: Path | None,
) -> JSONDict:
    managed = output_receipt_binding(
        "managed",
        managed_output_path,
        request_path=request_path,
        prompt_set_path=prompt_set_path,
    )
    dense = output_receipt_binding(
        "dense",
        dense_output_path,
        request_path=request_path,
        prompt_set_path=prompt_set_path,
    )
    approvals_ready = request_approvals_ready(request)
    errors = [
        f"{item['label']}: {error}"
        for item in (managed, dense)
        for error in item.get("errors", [])
        if isinstance(error, str)
    ]
    blockers = [
        blocker
        for item in (managed, dense)
        for blocker in item.get("blockers", [])
        if isinstance(blocker, str)
    ]
    if not approvals_ready:
        blockers.append("recorded_request_approvals_missing")
    if managed.get("ready") is not True:
        blockers.append("managed_capture_receipt_not_bound_to_request")
    if dense.get("ready") is not True:
        blockers.append("dense_capture_receipt_not_bound_to_request")
    return {
        "ready": approvals_ready and not errors and managed.get("ready") is True and dense.get("ready") is True,
        "approvals_ready": approvals_ready,
        "managed": managed,
        "dense": dense,
        "errors": errors,
        "blockers": sorted(set(blockers)),
    }

def artifact_request_status(plan: JSONDict, request_id: str) -> str | None:
    item = items_by_id(plan.get("artifact_requests")).get(request_id)
    if item is None:
        return None
    status = item.get("status")
    return str(status) if status is not None else None


COMPLETE_STEP_STATUSES = {"already_satisfied"}
RUNTIME_CAPTURE_STEP_STATUSES = {"approval_required", "ready_for_operator_capture", "needed"}
APPROVED_RUNTIME_CAPTURE_STEP_STATUSES = {"ready_for_operator_capture", "needed"}


def runtime_step_status(plan_status: str | None, request_item: JSONDict) -> str:
    status = str(plan_status or "unknown")
    request_status = str(request_item.get("status") or "")
    if status in {"approval_required", "needed", "present_but_not_candidate_ready"} and request_status in {
        "approval_required",
        "ready_for_operator_capture",
        "already_satisfied",
    }:
        return request_status
    return status


def approval_state_for_step(*, stage: str, status: str, approval_required: bool) -> str:
    if not approval_required:
        return "not_required"
    if status == "approval_required":
        return "missing"
    if stage == "runtime_capture" and status in APPROVED_RUNTIME_CAPTURE_STEP_STATUSES:
        return "recorded"
    return "required"


def step_has_command_class(step: JSONDict, command_class: str) -> bool:
    options = step.get("command_options")
    if not isinstance(options, list):
        return False
    return any(
        isinstance(option, dict)
        and option.get("command_class") == command_class
        and bool(option.get("command"))
        for option in options
    )


def step_has_approval_rebuild_command(step: JSONDict) -> bool:
    return step_has_command_class(step, "phase3_runtime_capture_request_approval_rebuild")


def request_next_step_has_approval_rebuild_command(request: JSONDict) -> bool:
    step = request.get("next_operator_step")
    return isinstance(step, dict) and step_has_approval_rebuild_command(step)


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def approval_rebuild_command_manifest(requests: list[JSONDict]) -> list[JSONDict]:
    manifest: list[JSONDict] = []
    for request in requests:
        step = request.get("next_operator_step")
        if not isinstance(step, dict):
            continue
        options = step.get("command_options")
        if not isinstance(options, list):
            continue
        for option in options:
            if (
                isinstance(option, dict)
                and option.get("command_class") == "phase3_runtime_capture_request_approval_rebuild"
                and option.get("command")
            ):
                manifest.append(
                    {
                        "request_name": request.get("name") or request.get("path") or "unnamed request",
                        "request_path": request.get("path"),
                        "bundle_path": request.get("bundle_path"),
                        "next_step_id": step.get("id"),
                        "command_class": option.get("command_class"),
                        "writes_request_path": option.get("writes_request_path"),
                        "records_approval_keys": string_list(option.get("records_approval_keys")),
                        "approval_scope": option.get("approval_scope"),
                        "approval_scope_artifact_id": option.get("approval_scope_artifact_id"),
                        "records_all_missing_approvals": option.get("records_all_missing_approvals") is True,
                        "requires_explicit_user_approval": option.get("requires_explicit_user_approval") is True,
                        "metadata_only": option.get("metadata_only") is True,
                        "command": string_list(option.get("command")),
                    }
                )
                break
    return manifest


def receipt_fill_entry_after_approval(item: JSONDict) -> JSONDict:
    blockers = [
        str(blocker)
        for blocker in item.get("blockers", [])
        if blocker and str(blocker) != "recorded_request_approvals_missing"
    ] if isinstance(item.get("blockers"), list) else []
    errors = [str(error) for error in item.get("errors", [])] if isinstance(item.get("errors"), list) else []
    ready_after_approval = item.get("ready") is True and not blockers and not errors
    return {
        "artifact_id": item.get("artifact_id"),
        "receipt_kind": item.get("receipt_kind"),
        "artifact_path": item.get("artifact_path"),
        "receipt_path": item.get("receipt_path"),
        "ready_after_approval": ready_after_approval,
        "receipt_ready": item.get("receipt_ready") is True,
        "approval_state_after_approval": "recorded",
        "blockers_after_approval": blockers,
        "errors": errors,
    }


def receipt_fill_preview_summary(entries: list[JSONDict]) -> JSONDict:
    ready_count = sum(1 for item in entries if item.get("ready_after_approval") is True)
    return {
        "entry_count": len(entries),
        "ready_after_approval_count": ready_count,
        "missing_after_approval_count": len(entries) - ready_count,
        "approval_missing_after_approval_count": 0,
        "all_ready_after_approval": bool(entries) and ready_count == len(entries),
    }


RECEIPT_ARTIFACT_TO_QUEUE_STEP = {
    "candidate_router_trace": "candidate_router_trace",
    "managed_output_summary_fill": "managed_output_summary",
    "dense_output_summary_fill": "dense_output_summary",
}


def command_option_count(step: JSONDict) -> int:
    options = step.get("command_options")
    return sum(1 for item in options if isinstance(item, dict) and item.get("command")) if isinstance(options, list) else 0


def command_option_manifest(step: JSONDict) -> list[JSONDict]:
    options = step.get("command_options")
    if not isinstance(options, list):
        return []
    result: list[JSONDict] = []
    for option in options:
        if not isinstance(option, dict) or not option.get("command"):
            continue
        command = option.get("command")
        result.append(
            {
                "command_class": option.get("command_class"),
                "requires_runtime": option.get("requires_runtime") is True,
                "requires_prompt_traffic": option.get("requires_prompt_traffic") is True,
                "requires_explicit_user_approval": option.get("requires_explicit_user_approval") is True,
                "metadata_only": option.get("metadata_only") is True,
                "command": [str(part) for part in command] if isinstance(command, list) else [str(command)],
            }
        )
    return result


def validator_command_count(step: JSONDict) -> int:
    validators = step.get("validator_commands")
    return len(validators) if isinstance(validators, list) else 0


def runtime_step_after_approval(step: JSONDict) -> JSONDict:
    status = step.get("status")
    approval_state = step.get("approval_state")
    if (
        step.get("stage") == "runtime_capture"
        and approval_state == "missing"
        and status in RUNTIME_CAPTURE_STEP_STATUSES
    ):
        status = "ready_for_operator_capture"
        approval_state = "recorded"
    return {
        "id": step.get("id"),
        "stage": step.get("stage"),
        "status": status,
        "approval_state": approval_state,
        "path": step.get("path"),
    }


def queue_step_by_id(request: JSONDict, step_id: str | None) -> JSONDict:
    if not isinstance(step_id, str):
        return {}
    queue = request.get("operator_queue")
    if not isinstance(queue, list):
        return {}
    for step in queue:
        if isinstance(step, dict) and step.get("id") == step_id:
            return step
    return {}


def post_approval_capture_fill_step(request: JSONDict, receipt_entry: JSONDict) -> JSONDict:
    artifact_id = str(receipt_entry.get("artifact_id") or "unknown")
    step_id = RECEIPT_ARTIFACT_TO_QUEUE_STEP.get(artifact_id)
    step = queue_step_by_id(request, step_id)
    after_receipt = receipt_fill_entry_after_approval(receipt_entry)
    blockers_after_approval = [
        str(item)
        for item in after_receipt.get("blockers_after_approval", [])
        if item
    ] if isinstance(after_receipt.get("blockers_after_approval"), list) else []
    errors = [str(item) for item in after_receipt.get("errors", []) if item] if isinstance(after_receipt.get("errors"), list) else []
    if after_receipt.get("ready_after_approval") is True:
        fill_status = "ready_after_approval"
    elif blockers_after_approval:
        fill_status = "blocked_after_approval"
    elif errors:
        fill_status = "invalid_after_approval"
    elif receipt_entry.get("receipt_ready") is not True:
        fill_status = "receipt_missing_or_not_ready"
    else:
        fill_status = "not_ready_after_approval"
    return {
        "artifact_id": artifact_id,
        "request_path": request.get("path"),
        "prompt_set_path": request.get("prompt_set_path"),
        "source_request_path": receipt_entry.get("request_path"),
        "source_prompt_set_path": receipt_entry.get("prompt_set_path"),
        "queue_step_id": step_id,
        "receipt_kind": receipt_entry.get("receipt_kind"),
        "artifact_path": receipt_entry.get("artifact_path"),
        "receipt_path": receipt_entry.get("receipt_path"),
        "current_step": {
            "id": step.get("id"),
            "stage": step.get("stage"),
            "status": step.get("status"),
            "approval_state": step.get("approval_state"),
            "path": step.get("path"),
        },
        "step_after_approval": runtime_step_after_approval(step),
        "fill_status_after_approval": fill_status,
        "ready_after_approval": after_receipt.get("ready_after_approval") is True,
        "receipt_ready": receipt_entry.get("receipt_ready") is True,
        "validator_command_count": validator_command_count(step),
        "command_option_count": command_option_count(step),
        "command_options": command_option_manifest(step),
        "blockers_after_approval": blockers_after_approval,
        "errors": errors,
    }


def post_approval_capture_fill_plan_for_request(request: JSONDict) -> JSONDict | None:
    transition = approval_transition_preview_for_request(request)
    if not isinstance(transition, dict):
        return None
    receipt_entries = receipt_fill_manifest([request])
    steps = [post_approval_capture_fill_step(request, item) for item in receipt_entries]
    ready_count = sum(1 for item in steps if item.get("ready_after_approval") is True)
    runtime_steps = [item for item in steps if item.get("current_step", {}).get("stage") == "runtime_capture"]
    bundle_update_statuses = request.get("bundle_update_statuses") if isinstance(request.get("bundle_update_statuses"), dict) else {}
    ready_bundle_update_status = any(status == "ready_to_build" for status in bundle_update_statuses.values())
    all_ready_after_approval = bool(steps) and ready_count == len(steps)
    return {
        "preview_only": True,
        "valid": transition.get("valid") is True,
        "request_name": request.get("name") or request.get("path") or "unnamed request",
        "request_path": request.get("path"),
        "prompt_set_path": request.get("prompt_set_path"),
        "bundle_path": request.get("bundle_path"),
        "rank": transition.get("rank"),
        "selection_rationale": transition.get("selection_rationale"),
        "records_approval_keys": string_list(transition.get("records_approval_keys")),
        "requires_explicit_user_approval": transition.get("requires_explicit_user_approval") is True,
        "metadata_only": True,
        "mutates_request": False,
        "artifact_step_count": len(steps),
        "runtime_capture_step_count": len(runtime_steps),
        "ready_after_approval_count": ready_count,
        "missing_after_approval_count": len(steps) - ready_count,
        "validator_command_count": sum(int(item.get("validator_command_count", 0) or 0) for item in steps),
        "command_option_count": sum(int(item.get("command_option_count", 0) or 0) for item in steps),
        "all_receipts_ready_after_approval": all_ready_after_approval,
        "ready_to_update_bundle_after_approval": all_ready_after_approval and ready_bundle_update_status,
        "capture_fill_steps": steps,
        "safety_contract": [
            "plan reads local metadata only",
            "plan does not write approval metadata",
            "plan does not create or fill capture artifacts",
            "plan does not launch model servers",
            "plan does not run Docker",
            "plan does not call endpoints",
            "plan does not send prompt traffic",
        ],
    }


def post_approval_capture_fill_plan_rank(item: JSONDict) -> tuple[int, str]:
    rank = item.get("rank")
    try:
        return (int(rank), str(item.get("request_name") or item.get("request_path") or ""))
    except (TypeError, ValueError):
        return approval_transition_preview_rank(item)


def post_approval_capture_fill_plan_manifest(requests: list[JSONDict]) -> list[JSONDict]:
    plans = [
        plan
        for request in requests
        if isinstance(request, dict)
        for plan in [post_approval_capture_fill_plan_for_request(request)]
        if isinstance(plan, dict)
    ]
    ranked = sorted(plans, key=post_approval_capture_fill_plan_rank)
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index
        if not item.get("selection_rationale"):
            item["selection_rationale"] = (
                "mixtral_first_known_sparse_baseline"
                if index == 1 and "mixtral" in str(item.get("request_name") or item.get("request_path") or "").lower()
                else "stable_name_order_after_mixtral_baseline"
            )
    return ranked


def approval_transition_preview_for_request(request: JSONDict) -> JSONDict | None:
    step = request.get("next_operator_step") if isinstance(request.get("next_operator_step"), dict) else None
    if step is None:
        return None
    if step.get("stage") != "runtime_capture" or step.get("approval_state") != "missing":
        return None
    options = step.get("command_options") if isinstance(step.get("command_options"), list) else []
    command = next(
        (
            option
            for option in options
            if isinstance(option, dict)
            and option.get("command_class") == "phase3_runtime_capture_request_approval_rebuild"
            and option.get("command")
        ),
        None,
    )
    if command is None:
        return None
    receipt_entries = [receipt_fill_entry_after_approval(item) for item in receipt_fill_manifest([request])]
    receipt_summary = receipt_fill_preview_summary(receipt_entries)
    bundle_update_statuses = request.get("bundle_update_statuses") if isinstance(request.get("bundle_update_statuses"), dict) else {}
    ready_bundle_update_status = any(status == "ready_to_build" for status in bundle_update_statuses.values())
    return {
        "preview_only": True,
        "valid": request.get("valid") is True and request.get("request_drift_free") is True,
        "request_name": request.get("name") or request.get("path") or "unnamed request",
        "request_path": request.get("path"),
        "bundle_path": request.get("bundle_path"),
        "current_next_step": {
            "id": step.get("id"),
            "stage": step.get("stage"),
            "status": step.get("status"),
            "approval_state": step.get("approval_state"),
            "path": step.get("path"),
        },
        "next_step_after_approval": {
            "id": step.get("id"),
            "stage": step.get("stage"),
            "status": "ready_for_operator_capture",
            "approval_state": "recorded",
            "path": step.get("path"),
        },
        "records_approval_keys": string_list(command.get("records_approval_keys")),
        "requires_explicit_user_approval": command.get("requires_explicit_user_approval") is True,
        "metadata_only": command.get("metadata_only") is True,
        "mutates_request": False,
        "capture_complete_after_approval": request.get("request_capture_complete_flag") is True,
        "approved_but_capture_incomplete_after_approval": request.get("request_capture_complete_flag") is not True,
        "ready_to_update_bundle_after_approval": (
            request.get("request_drift_free") is True
            and receipt_summary.get("all_ready_after_approval") is True
            and ready_bundle_update_status
        ),
        "receipt_fill_preview": receipt_summary,
        "receipt_fill_entries": receipt_entries,
        "remaining_blockers_after_approval": request.get("remaining_blockers_after_intake", []),
        "safety_contract": [
            "preview reads local metadata only",
            "preview does not write approval metadata",
            "preview does not launch model servers",
            "preview does not run Docker",
            "preview does not call endpoints",
            "preview does not send prompt traffic",
        ],
    }


def approval_transition_preview_rank(item: JSONDict) -> tuple[int, str]:
    haystack = " ".join(
        str(item.get(key) or "")
        for key in ("request_name", "request_path", "bundle_path")
    ).lower()
    return (0 if "mixtral" in haystack else 1, str(item.get("request_name") or item.get("request_path") or ""))


def approval_transition_preview_manifest(requests: list[JSONDict]) -> list[JSONDict]:
    previews = [
        preview
        for request in requests
        if isinstance(request, dict)
        for preview in [approval_transition_preview_for_request(request)]
        if isinstance(preview, dict)
    ]
    ranked = sorted(previews, key=approval_transition_preview_rank)
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index
        item["selection_rationale"] = (
            "mixtral_first_known_sparse_baseline"
            if index == 1 and "mixtral" in str(item.get("request_name") or item.get("request_path") or "").lower()
            else "stable_name_order_after_mixtral_baseline"
        )
    return ranked


def approval_label(step: JSONDict) -> str:
    state = step.get("approval_state")
    if state == "missing":
        if step_has_approval_rebuild_command(step):
            return "approval missing; approval command ready"
        return "approval missing"
    if state == "recorded":
        return "approval recorded"
    if state == "required":
        return "approval required"
    return "no runtime approval"


def request_artifact_item(request: JSONDict, artifact_id: str, *, future: bool = False) -> JSONDict:
    collection = "future_artifacts" if future else "requested_artifacts"
    return items_by_id(request.get(collection)).get(artifact_id, {})


def command_items(plan: JSONDict, *classes: str) -> list[JSONDict]:
    wanted = set(classes)
    result: list[JSONDict] = []
    commands = plan.get("commands")
    if not isinstance(commands, list):
        return result
    for item in commands:
        if not isinstance(item, dict):
            continue
        if item.get("command_class") in wanted:
            result.append(item)
    return result


def approval_rebuild_command_options(request_audit: JSONDict) -> list[JSONDict]:
    missing_keys = plan_phase3_runtime_capture_request.missing_runtime_approval_keys(request_audit)
    if not missing_keys:
        return []
    command = plan_phase3_runtime_capture_request.approval_rebuild_command(request_audit, missing_keys)
    options = [command] if command.get("command") else []
    pending = plan_phase3_runtime_capture_request.pending_runtime_artifacts(request_audit)
    next_artifact = pending[0] if pending else {}
    scoped = plan_phase3_runtime_capture_request.next_artifact_approval_rebuild_command(
        request_audit,
        missing_keys,
        next_artifact.get("id") if isinstance(next_artifact, dict) else None,
    )
    if scoped.get("command"):
        options.append(scoped)
    return options


def attach_approval_rebuild_command(queue: list[JSONDict], request_audit: JSONDict) -> None:
    options = approval_rebuild_command_options(request_audit)
    if not options:
        return
    for step in queue:
        if (
            step.get("stage") == "runtime_capture"
            and step.get("approval_state") == "missing"
            and step.get("status") not in COMPLETE_STEP_STATUSES
        ):
            existing = step.get("command_options") if isinstance(step.get("command_options"), list) else []
            step["command_options"] = [*options, *existing]
            return


def queue_step(
    step_id: str,
    *,
    stage: str,
    status: str | None,
    path: Path | None = None,
    approval_required: bool = False,
    validator_commands: Any = None,
    command_options: Any = None,
) -> JSONDict:
    normalized_status = status or "unknown"
    return {
        "id": step_id,
        "stage": stage,
        "status": normalized_status,
        "path": display_path(path),
        "approval_required": approval_required,
        "approval_state": approval_state_for_step(
            stage=stage,
            status=normalized_status,
            approval_required=approval_required,
        ),
        "validator_commands": validator_commands if isinstance(validator_commands, list) else [],
        "command_options": command_options if isinstance(command_options, list) else [],
    }


def first_incomplete_step(queue: list[JSONDict]) -> JSONDict | None:
    for step in queue:
        if step.get("status") not in COMPLETE_STEP_STATUSES:
            return step
    return None


def queue_counts_by_next_step(requests: list[JSONDict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for request in requests:
        step = request.get("next_operator_step")
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("id", "unknown"))
        counts[step_id] = counts.get(step_id, 0) + 1
    return dict(sorted(counts.items()))

def receipt_gate_status(request: JSONDict) -> JSONDict:
    policy = request.get("policy_candidate") if isinstance(request.get("policy_candidate"), dict) else {}
    trace_receipt = (
        policy.get("candidate_trace_capture_receipt")
        if isinstance(policy.get("candidate_trace_capture_receipt"), dict)
        else {}
    )
    binding = request.get("capture_receipt_binding") if isinstance(request.get("capture_receipt_binding"), dict) else {}
    managed = binding.get("managed") if isinstance(binding.get("managed"), dict) else {}
    dense = binding.get("dense") if isinstance(binding.get("dense"), dict) else {}
    live = request.get("live_capability_proof") if isinstance(request.get("live_capability_proof"), dict) else {}
    gates = {
        "approvals_ready": request.get("approvals_ready_for_recorded_capture") is True,
        "candidate_trace_receipt_ready": trace_receipt.get("ready") is True,
        "managed_output_receipt_ready": managed.get("ready") is True,
        "dense_output_receipt_ready": dense.get("ready") is True,
        "output_receipt_binding_ready": binding.get("ready") is True,
        "live_capability_proof_ready": live.get("proof_ready") is True,
    }
    capture_receipt_ready_count = sum(
        1
        for gate_id in (
            "candidate_trace_receipt_ready",
            "managed_output_receipt_ready",
            "dense_output_receipt_ready",
        )
        if gates[gate_id]
    )
    missing = [gate_id for gate_id, ready in gates.items() if ready is not True]
    return {
        **gates,
        "capture_receipt_required_count": 3,
        "capture_receipt_ready_count": capture_receipt_ready_count,
        "missing_gate_ids": missing,
    }


def aggregate_receipt_gate_coverage(requests: list[JSONDict]) -> JSONDict:
    statuses = [receipt_gate_status(request) for request in requests if isinstance(request, dict)]
    missing_by_request: list[JSONDict] = []
    for request, status in zip(requests, statuses):
        if not isinstance(request, dict):
            continue
        missing = status.get("missing_gate_ids") if isinstance(status.get("missing_gate_ids"), list) else []
        if missing:
            missing_by_request.append(
                {
                    "request_name": request.get("name") or request.get("path"),
                    "missing_gate_ids": missing,
                }
            )
    return {
        "request_count": len(statuses),
        "capture_receipt_required_count": sum(int(status.get("capture_receipt_required_count", 0) or 0) for status in statuses),
        "capture_receipt_ready_count": sum(int(status.get("capture_receipt_ready_count", 0) or 0) for status in statuses),
        "approvals_ready_count": sum(1 for status in statuses if status.get("approvals_ready") is True),
        "candidate_trace_receipt_ready_count": sum(1 for status in statuses if status.get("candidate_trace_receipt_ready") is True),
        "managed_output_receipt_ready_count": sum(1 for status in statuses if status.get("managed_output_receipt_ready") is True),
        "dense_output_receipt_ready_count": sum(1 for status in statuses if status.get("dense_output_receipt_ready") is True),
        "output_receipt_binding_ready_count": sum(1 for status in statuses if status.get("output_receipt_binding_ready") is True),
        "live_capability_proof_ready_count": sum(1 for status in statuses if status.get("live_capability_proof_ready") is True),
        "missing_request_count": len(missing_by_request),
        "missing_by_request": missing_by_request,
        "all_capture_receipts_ready": bool(statuses) and all(
            status.get("capture_receipt_ready_count") == status.get("capture_receipt_required_count")
            for status in statuses
        ),
        "all_output_receipt_bindings_ready": bool(statuses) and all(
            status.get("output_receipt_binding_ready") is True for status in statuses
        ),
    }


def receipt_summary_values(value: Any) -> JSONDict:
    return value if isinstance(value, dict) else {}


def receipt_blockers(value: JSONDict) -> list[str]:
    blockers = value.get("blockers")
    return [str(item) for item in blockers] if isinstance(blockers, list) else []


def receipt_errors(value: JSONDict) -> list[str]:
    errors = value.get("errors")
    return [str(item) for item in errors] if isinstance(errors, list) else []


def receipt_fill_entry(
    request: JSONDict,
    *,
    artifact_id: str,
    receipt_kind: str,
    artifact_path: Any,
    receipt_path: Any,
    receipt_summary: Any,
) -> JSONDict:
    summary = receipt_summary_values(receipt_summary)
    ready = summary.get("ready") is True
    receipt_ready = summary.get("receipt_ready") is True
    blockers = receipt_blockers(summary)
    errors = receipt_errors(summary)
    if request.get("approvals_ready_for_recorded_capture") is not True:
        blockers = sorted(set([*blockers, "recorded_request_approvals_missing"]))
    return {
        "request_name": request.get("name") or request.get("path") or "unnamed request",
        "request_path": request.get("path"),
        "bundle_path": request.get("bundle_path"),
        "model_id": request.get("model_id"),
        "prompt_family": request.get("prompt_family"),
        "prompt_set_path": request.get("prompt_set_path"),
        "artifact_id": artifact_id,
        "receipt_kind": receipt_kind,
        "artifact_path": artifact_path,
        "receipt_path": receipt_path,
        "exists": summary.get("exists") is True,
        "receipt_ready": receipt_ready,
        "ready": ready,
        "approvals_ready": request.get("approvals_ready_for_recorded_capture") is True,
        "approval_state": "recorded" if request.get("approvals_ready_for_recorded_capture") is True else "missing",
        "request_path_matches": summary.get("request_path_matches") is True,
        "prompt_set_path_matches": summary.get("prompt_set_path_matches") is True,
        "candidate_trace_path_matches": summary.get("candidate_trace_path_matches") if "candidate_trace_path_matches" in summary else None,
        "blockers": blockers,
        "errors": errors,
    }


def receipt_fill_manifest(requests: list[JSONDict]) -> list[JSONDict]:
    manifest: list[JSONDict] = []
    for request in requests:
        if not isinstance(request, dict):
            continue
        policy = request.get("policy_candidate") if isinstance(request.get("policy_candidate"), dict) else {}
        trace_receipt = policy.get("candidate_trace_capture_receipt") if isinstance(policy.get("candidate_trace_capture_receipt"), dict) else {}
        binding = request.get("capture_receipt_binding") if isinstance(request.get("capture_receipt_binding"), dict) else {}
        managed = binding.get("managed") if isinstance(binding.get("managed"), dict) else {}
        dense = binding.get("dense") if isinstance(binding.get("dense"), dict) else {}
        candidate_entry = receipt_fill_entry(
            request,
            artifact_id="candidate_router_trace",
            receipt_kind="trace_capture_receipt_json",
            artifact_path=request.get("candidate_trace_path"),
            receipt_path=request.get("candidate_trace_receipt_path") or trace_receipt.get("path"),
            receipt_summary=trace_receipt,
        )
        if policy.get("candidate_trace_ready") is not True:
            candidate_entry["ready"] = False
            blockers = candidate_entry.get("blockers") if isinstance(candidate_entry.get("blockers"), list) else []
            candidate_entry["blockers"] = sorted(set([*blockers, "candidate_trace_not_policy_candidate_ready"]))
        manifest.append(candidate_entry)
        manifest.append(
            receipt_fill_entry(
                request,
                artifact_id="managed_output_summary_fill",
                receipt_kind="embedded_output_capture_receipt",
                artifact_path=request.get("managed_output_path") or managed.get("path"),
                receipt_path=managed.get("path") or request.get("managed_output_path"),
                receipt_summary=managed,
            )
        )
        manifest.append(
            receipt_fill_entry(
                request,
                artifact_id="dense_output_summary_fill",
                receipt_kind="embedded_output_capture_receipt",
                artifact_path=request.get("dense_output_path") or dense.get("path"),
                receipt_path=dense.get("path") or request.get("dense_output_path"),
                receipt_summary=dense,
            )
        )
    return manifest


def receipt_fill_manifest_summary(manifest: list[JSONDict]) -> JSONDict:
    by_artifact: dict[str, JSONDict] = {}
    for item in manifest:
        artifact_id = str(item.get("artifact_id") or "unknown")
        current = by_artifact.setdefault(artifact_id, {"entry_count": 0, "ready_count": 0, "missing_count": 0})
        current["entry_count"] += 1
        if item.get("ready") is True:
            current["ready_count"] += 1
        else:
            current["missing_count"] += 1
    ready_count = sum(1 for item in manifest if item.get("ready") is True)
    return {
        "entry_count": len(manifest),
        "ready_count": ready_count,
        "missing_count": len(manifest) - ready_count,
        "approval_missing_count": sum(1 for item in manifest if item.get("approval_state") == "missing"),
        "by_artifact": dict(sorted(by_artifact.items())),
        "all_ready": bool(manifest) and ready_count == len(manifest),
    }


def request_queue_line(request: JSONDict) -> str:
    name = str(request.get("name") or request.get("path") or "unnamed request")
    step = request.get("next_operator_step")
    if not isinstance(step, dict):
        return f"{name}: no next operator step"
    step_id = str(step.get("id") or "unknown")
    status = str(step.get("status") or "unknown")
    stage = str(step.get("stage") or "unknown")
    path = str(step.get("path") or "stdout")
    approval = approval_label(step)
    return f"{name}: {step_id} {status} ({stage}, {approval}) -> {path}"

def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def command_to_text(command: Any) -> str:
    if isinstance(command, list):
        return " ".join(str(part) for part in command)
    return str(command)


def step_command_count(step: JSONDict) -> int:
    count = 0
    validators = step.get("validator_commands")
    if isinstance(validators, list):
        count += len(validators)
    options = step.get("command_options")
    if isinstance(options, list):
        count += sum(1 for item in options if isinstance(item, dict) and item.get("command"))
    return count


def format_markdown_report(summary: JSONDict) -> str:
    lines = [
        "# Phase 3 Capture-Result Intake",
        "",
        f"- Valid: `{summary.get('valid')}`",
        f"- Completion receipt gate provided: `{summary.get('completion_receipt_gate_provided')}`",
        f"- Completion receipt gate ready: `{summary.get('completion_receipt_gate_ready')}`",
        f"- Completion receipt gate complete: `{summary.get('completion_receipt_gate_complete')}`",
        f"- Completion receipt gate errors: `{summary.get('completion_receipt_gate_error_count')}`",
        f"- Requests: `{summary.get('request_count')}`",
        f"- Drift-free saved requests: `{summary.get('request_drift_free_count')}`",
        f"- Drifted saved requests: `{summary.get('request_drifted_count')}`",
        f"- Capture-complete flags: `{summary.get('request_capture_complete_flag_count')}`",
        f"- Ready-for-operator flags: `{summary.get('request_ready_for_operator_capture_flag_count')}`",
        f"- Approved but capture incomplete: `{summary.get('approved_but_capture_incomplete_request_count')}`",
        f"- Runtime approvals still missing: `{summary.get('runtime_approval_missing_request_count')}`",
        f"- Approval rebuild commands available: `{summary.get('approval_rebuild_command_available_request_count')}`",
        f"- Approval transition previews: `{summary.get('approval_transition_preview_count')}`",
        f"- Approval previews ready for operator: `{summary.get('approval_transition_ready_for_operator_count')}`",
        f"- Approval previews ready to update bundle: `{summary.get('approval_transition_ready_to_update_bundle_count')}`",
        f"- Post-approval capture-fill plans: `{summary.get('post_approval_capture_fill_plan_count')}`",
        f"- Post-approval capture-fill ready: `{summary.get('post_approval_capture_fill_ready_count')}` / `{summary.get('post_approval_capture_fill_artifact_step_count')}`",
        f"- Post-approval capture-fill missing: `{summary.get('post_approval_capture_fill_missing_count')}`",
        f"- Post-approval capture-fill validator commands: `{summary.get('post_approval_capture_fill_validator_command_count')}`",
        f"- Approved runtime-capture pending: `{summary.get('approved_runtime_capture_pending_request_count')}`",
        f"- Capture receipts ready: `{summary.get('capture_receipt_ready_count')}` / `{summary.get('capture_receipt_required_count')}`",
        f"- Receipt-fill manifest ready: `{summary.get('receipt_fill_ready_count')}` / `{summary.get('receipt_fill_entry_count')}`",
        f"- Receipt-fill approvals missing: `{summary.get('receipt_fill_approval_missing_count')}`",
        f"- Requests missing receipt gates: `{summary.get('capture_receipt_missing_request_count')}`",
        f"- Output receipt bindings ready: `{summary.get('output_receipt_binding_ready_count')}`",
        f"- Ready to update bundle: `{summary.get('ready_to_update_bundle_count')}`",
        f"- Phase 4 candidates: `{summary.get('phase4_candidate_ready_count')}`",
        f"- Live-spike candidates: `{summary.get('live_spike_candidate_ready_count')}`",
        f"- Pending runtime-capture next steps: `{summary.get('pending_runtime_capture_request_count')}`",
        f"- Pending bundle-update next steps: `{summary.get('pending_bundle_update_request_count')}`",
        "",
        "## Next Operator Steps",
        "",
        "| Request | Step | Status | Stage | Approval | Path |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    approval_manifest = (
        summary.get("approval_rebuild_command_manifest")
        if isinstance(summary.get("approval_rebuild_command_manifest"), list)
        else []
    )
    if approval_manifest:
        lines.extend(["", "## Approval Rebuild Command Manifest", ""])
        for item in approval_manifest:
            if not isinstance(item, dict):
                continue
            records = item.get("records_approval_keys") if isinstance(item.get("records_approval_keys"), list) else []
            lines.extend(
                [
                    f"- Request: `{markdown_escape(item.get('request_name') or 'unnamed request')}`",
                    f"  - Records: `{markdown_escape(', '.join(str(key) for key in records) if records else 'none')}`",
                    f"  - Writes: `{markdown_escape(item.get('writes_request_path') or 'missing')}`",
                    "```sh",
                    command_to_text(item.get("command")),
                    "```",
                ]
            )
    transition_manifest = (
        summary.get("approval_transition_preview_manifest")
        if isinstance(summary.get("approval_transition_preview_manifest"), list)
        else []
    )
    if transition_manifest:
        lines.extend(
            [
                "",
                "## Approval Transition Preview",
                "",
                "| Rank | Request | Current step | After approval | Receipt fills ready | Receipt fills missing | Mutates request |",
                "| --- | --- | --- | --- | ---: | ---: | --- |",
            ]
        )
        for item in transition_manifest:
            if not isinstance(item, dict):
                continue
            current = item.get("current_next_step") if isinstance(item.get("current_next_step"), dict) else {}
            after = item.get("next_step_after_approval") if isinstance(item.get("next_step_after_approval"), dict) else {}
            receipt_preview = item.get("receipt_fill_preview") if isinstance(item.get("receipt_fill_preview"), dict) else {}
            lines.append(
                "| "
                + " | ".join(
                    markdown_escape(value)
                    for value in (
                        item.get("rank"),
                        item.get("request_name") or item.get("request_path") or "unnamed request",
                        f"{current.get('id') or 'unknown'}:{current.get('status') or 'unknown'}:{current.get('approval_state') or 'unknown'}",
                        f"{after.get('id') or 'unknown'}:{after.get('status') or 'unknown'}:{after.get('approval_state') or 'unknown'}",
                        receipt_preview.get("ready_after_approval_count"),
                        receipt_preview.get("missing_after_approval_count"),
                        item.get("mutates_request"),
                    )
                )
                + " |"
            )
    capture_fill_plans = (
        summary.get("post_approval_capture_fill_plan_manifest")
        if isinstance(summary.get("post_approval_capture_fill_plan_manifest"), list)
        else []
    )
    if capture_fill_plans:
        lines.extend(
            [
                "",
                "## Post-Approval Capture Fill Plan",
                "",
                "| Rank | Request | Artifact steps | Ready after approval | Missing after approval | Validator commands | Ready to update bundle |",
                "| --- | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for item in capture_fill_plans:
            if not isinstance(item, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    markdown_escape(value)
                    for value in (
                        item.get("rank"),
                        item.get("request_name") or item.get("request_path") or "unnamed request",
                        item.get("artifact_step_count"),
                        item.get("ready_after_approval_count"),
                        item.get("missing_after_approval_count"),
                        item.get("validator_command_count"),
                        item.get("ready_to_update_bundle_after_approval"),
                    )
                )
                + " |"
            )
    recommended_fill_plan = summary.get("recommended_post_approval_capture_fill_plan")
    if isinstance(recommended_fill_plan, dict):
        steps = recommended_fill_plan.get("capture_fill_steps") if isinstance(recommended_fill_plan.get("capture_fill_steps"), list) else []
        lines.extend(
            [
                "",
                "## Recommended Post-Approval Capture Steps",
                "",
                "| Artifact | Step after approval | Fill status | Artifact path | Receipt path | Validators | Blockers after approval |",
                "| --- | --- | --- | --- | --- | ---: | --- |",
            ]
        )
        for step in steps:
            if not isinstance(step, dict):
                continue
            after = step.get("step_after_approval") if isinstance(step.get("step_after_approval"), dict) else {}
            blockers = step.get("blockers_after_approval") if isinstance(step.get("blockers_after_approval"), list) else []
            lines.append(
                "| "
                + " | ".join(
                    markdown_escape(value)
                    for value in (
                        step.get("artifact_id") or "unknown",
                        f"{after.get('id') or 'unknown'}:{after.get('status') or 'unknown'}:{after.get('approval_state') or 'unknown'}",
                        step.get("fill_status_after_approval") or "unknown",
                        step.get("artifact_path") or "missing",
                        step.get("receipt_path") or "missing",
                        step.get("validator_command_count"),
                        ", ".join(str(blocker) for blocker in blockers) if blockers else "none",
                    )
                )
                + " |"
            )
    for request in summary.get("requests", []):
        if not isinstance(request, dict):
            continue
        step = request.get("next_operator_step") if isinstance(request.get("next_operator_step"), dict) else {}
        approval = step.get("approval_state") or ("missing" if step.get("approval_required") is True else "not_required")
        lines.append(
            "| "
            + " | ".join(
                markdown_escape(value)
                for value in (
                    request.get("name") or request.get("path") or "unnamed request",
                    step.get("id") or "unknown",
                    step.get("status") or "unknown",
                    step.get("stage") or "unknown",
                    approval,
                    step.get("path") or "stdout",
                )
            )
            + " |"
        )
    receipt_manifest = summary.get("receipt_fill_manifest") if isinstance(summary.get("receipt_fill_manifest"), list) else []
    if receipt_manifest:
        lines.extend(
            [
                "",
                "## Receipt Fill Manifest",
                "",
                "| Request | Artifact | Receipt kind | Ready | Approval | Artifact path | Receipt path | Blockers |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for item in receipt_manifest:
            if not isinstance(item, dict):
                continue
            blockers = item.get("blockers") if isinstance(item.get("blockers"), list) else []
            lines.append(
                "| "
                + " | ".join(
                    markdown_escape(value)
                    for value in (
                        item.get("request_name") or "unnamed request",
                        item.get("artifact_id") or "unknown",
                        item.get("receipt_kind") or "unknown",
                        item.get("ready"),
                        item.get("approval_state") or "unknown",
                        item.get("artifact_path") or "missing",
                        item.get("receipt_path") or "missing",
                        ", ".join(str(blocker) for blocker in blockers) if blockers else "none",
                    )
                )
                + " |"
            )
    queue_requests = [request for request in summary.get("requests", []) if isinstance(request, dict) and isinstance(request.get("operator_queue"), list)]
    if queue_requests:
        lines.extend(["", "## Detailed Operator Queues", ""])
        for request in queue_requests:
            request_name = request.get("name") or request.get("path") or "Unnamed request"
            lines.extend(
                [
                    f"### {markdown_escape(request_name)}",
                    "",
                    "| Step | Stage | Status | Approval | Path | Commands |",
                    "| --- | --- | --- | --- | --- | --- |",
                ]
            )
            for step in request.get("operator_queue", []):
                if not isinstance(step, dict):
                    continue
                approval = step.get("approval_state") or ("missing" if step.get("approval_required") is True else "not_required")
                lines.append(
                    "| "
                    + " | ".join(
                        markdown_escape(value)
                        for value in (
                            step.get("id") or "unknown",
                            step.get("stage") or "unknown",
                            step.get("status") or "unknown",
                            approval,
                            step.get("path") or "stdout",
                            step_command_count(step),
                        )
                    )
                    + " |"
                )
            command_lines: list[str] = []
            for step in request.get("operator_queue", []):
                if not isinstance(step, dict):
                    continue
                step_id = step.get("id") or "unknown"
                validators = step.get("validator_commands") if isinstance(step.get("validator_commands"), list) else []
                for command in validators:
                    command_lines.extend([f"- `{markdown_escape(step_id)}` validator", "```sh", command_to_text(command), "```"])
                options = step.get("command_options") if isinstance(step.get("command_options"), list) else []
                for option in options:
                    if not isinstance(option, dict) or not option.get("command"):
                        continue
                    command_class = option.get("command_class") or step_id
                    command_lines.extend(
                        [
                            f"- `{markdown_escape(step_id)}` `{markdown_escape(command_class)}`",
                            "```sh",
                            command_to_text(option.get("command")),
                            "```",
                        ]
                    )
            if command_lines:
                lines.extend(["", "#### Commands", "", *command_lines, ""])
    blockers = summary.get("remaining_blockers_after_intake", [])
    if isinstance(blockers, list) and blockers:
        lines.extend(["", "## Remaining Blockers", ""])
        for blocker in blockers:
            lines.append(f"- `{markdown_escape(blocker)}`")
    lines.extend(["", "## Safety Contract", ""])
    for item in summary.get("safety_contract", []):
        lines.append(f"- {markdown_escape(item)}")
    return "\n".join(lines) + "\n"


def write_markdown_report(summary: JSONDict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_markdown_report(summary), encoding="utf-8")


def blockers_for_intake(
    *,
    request_current: bool = True,
    policy_ready: bool,
    fallback_ready: bool,
    live_gate: JSONDict,
) -> list[str]:
    blockers: list[str] = []
    if not request_current:
        blockers.append("runtime_capture_request_drift")
    if not policy_ready:
        blockers.append("no_replay_policy_candidate")
    if not fallback_ready:
        blockers.append("dense_fallback_output_artifact_for_quality_bounds")
    if live_gate.get("live_residency_observation_and_control_ready") is not True:
        blockers.append("live_residency_observation_and_control")
    if live_gate.get("cleanup_restore_proof_ready") is not True:
        blockers.append("cleanup_restore_proof")
    return blockers


def summarize_request(path: Path) -> JSONDict:
    if not path.exists():
        return {
            "path": display_path(path),
            "exists": False,
            "valid": False,
            "errors": [f"runtime-capture request does not exist: {display_path(path)}"],
        }

    errors: list[str] = []
    request = load_json(path)
    request_errors = build_phase3_runtime_capture_request.validate_request(request)
    errors.extend(f"request: {error}" for error in request_errors)

    bundle_path = resolve_repo_path(request.get("bundle_path"))
    if bundle_path is None:
        errors.append("request.bundle_path is missing")
        return {
            "path": display_path(path),
            "exists": True,
            "valid": False,
            "errors": errors,
        }
    if not bundle_path.exists():
        errors.append(f"request bundle does not exist: {display_path(bundle_path)}")
        return {
            "path": display_path(path),
            "exists": True,
            "valid": False,
            "errors": errors,
            "bundle_path": display_path(bundle_path),
        }

    request_audit = plan_phase3_runtime_capture_request.summarize_request(path)
    request_drift_count = int(request_audit.get("drift_count", 0) or 0)
    request_current = request_audit.get("valid") is True and request_drift_count == 0
    if request_audit.get("valid") is not True:
        errors.extend(f"request audit: {error}" for error in request_audit.get("errors", []))
    if request_drift_count:
        errors.append(f"request audit drift_count must be 0, got {request_drift_count}")

    candidate_trace_path = request_artifact_path(request, "candidate_router_trace")
    candidate_trace_receipt_path = request_artifact_source_path(
        request,
        "candidate_router_trace",
        "capture_receipt_path",
    )
    managed_output_path = request_artifact_path(request, "managed_output_summary_fill")
    dense_output_path = request_artifact_path(request, "dense_output_summary_fill")
    live_proof_path = request_artifact_path(request, "live_capability_proof_fill", future=True)
    prompt_set = request.get("prompt_set") if isinstance(request.get("prompt_set"), dict) else {}
    prompt_set_path = resolve_repo_path(prompt_set.get("path")) if isinstance(prompt_set, dict) else None
    capture_receipt_binding = capture_receipt_binding_for_request(
        path,
        request,
        prompt_set_path=prompt_set_path,
        managed_output_path=managed_output_path,
        dense_output_path=dense_output_path,
    )

    policy_plan = plan_phase3_policy_candidate_trace.build_capture_plan(
        bundle_path,
        candidate_prompt_set_path=prompt_set_path,
        candidate_trace_path=candidate_trace_path,
        candidate_trace_receipt_path=candidate_trace_receipt_path,
        source_request_path=path,
        policy_candidate_trace_capture_approved=request_approval(request, "router_trace_capture_approved"),
        runtime_prompt_traffic_approved=request_approval(request, "runtime_prompt_traffic_approved"),
    )
    fallback_plan = plan_phase3_dense_fallback_capture.build_capture_plan(
        bundle_path,
        prompt_set_path=prompt_set_path,
        managed_output_path=managed_output_path,
        dense_output_path=dense_output_path,
        output_dir=managed_output_path.parent if managed_output_path is not None else None,
        dense_fallback_capture_approved=(
            request_approval(request, "managed_output_capture_approved")
            and request_approval(request, "dense_output_capture_approved")
        ),
        runtime_prompt_traffic_approved=request_approval(request, "runtime_prompt_traffic_approved"),
    )
    if live_proof_path is not None and live_proof_path.exists():
        live_summary = plan_phase3_live_capability_proof.build_summary(live_proof_path)
    else:
        live_summary = plan_phase3_live_capability_proof.missing_artifact_summary(live_proof_path)
    bundle_summary = plan_phase3_real_evidence_bundle.build_summary(bundle_path)

    policy_ready = policy_plan.get("policy_candidate_ready_after_plan") is True
    fallback_ready = fallback_plan.get("comparison_ready") is True
    capture_binding_ready = capture_receipt_binding.get("ready") is True
    fallback_ready_after_intake = fallback_ready and capture_binding_ready
    live_gate = live_summary.get("phase_3_gate", {}) if isinstance(live_summary.get("phase_3_gate"), dict) else {}
    live_ready = live_summary.get("proof_ready") is True
    phase4_candidate_ready = request_current and policy_ready and fallback_ready_after_intake and bundle_summary.get("valid") is True
    live_spike_candidate_ready = phase4_candidate_ready and live_ready
    bundle_update_statuses = {
        "policy_candidate_bundle": artifact_request_status(policy_plan, "phase3_bundle_with_candidate_trace"),
        "fallback_bundle": artifact_request_status(fallback_plan, "phase3_bundle_with_fallback"),
    }
    ready_to_update_bundle = request_current and capture_binding_ready and any(status == "ready_to_build" for status in bundle_update_statuses.values())
    requested_items = items_by_id(request.get("requested_artifacts"))
    future_items = items_by_id(request.get("future_artifacts"))
    candidate_item = requested_items.get("candidate_router_trace", {})
    managed_item = requested_items.get("managed_output_summary_fill", {})
    dense_item = requested_items.get("dense_output_summary_fill", {})
    live_item = future_items.get("live_capability_proof_fill", {})
    candidate_trace_status = runtime_step_status(
        artifact_request_status(policy_plan, "candidate_trace_artifact"),
        candidate_item,
    )
    candidate_replay_status = artifact_request_status(policy_plan, "candidate_policy_replay")
    managed_output_status = runtime_step_status(
        artifact_request_status(fallback_plan, "managed_output_summary"),
        managed_item,
    )
    dense_output_status = runtime_step_status(
        artifact_request_status(fallback_plan, "dense_output_summary"),
        dense_item,
    )
    comparison_status = artifact_request_status(fallback_plan, "dense_fallback_comparison_artifact")
    live_status = str(live_item.get("status", "already_satisfied" if live_ready else "future_adapter_required"))
    operator_queue = [
        queue_step(
            "candidate_router_trace",
            stage="runtime_capture",
            status=candidate_trace_status,
            path=candidate_trace_path,
            approval_required=candidate_trace_status in RUNTIME_CAPTURE_STEP_STATUSES,
            validator_commands=candidate_item.get("validator_commands"),
        ),
        queue_step(
            "candidate_policy_replay",
            stage="offline_validation",
            status=candidate_replay_status,
            path=candidate_trace_path,
            approval_required=False,
            command_options=command_items(policy_plan, "candidate_policy_replay"),
        ),
        queue_step(
            "managed_output_summary",
            stage="runtime_capture",
            status=managed_output_status,
            path=managed_output_path,
            approval_required=managed_output_status in RUNTIME_CAPTURE_STEP_STATUSES,
            validator_commands=managed_item.get("validator_commands"),
        ),
        queue_step(
            "dense_output_summary",
            stage="runtime_capture",
            status=dense_output_status,
            path=dense_output_path,
            approval_required=dense_output_status in RUNTIME_CAPTURE_STEP_STATUSES,
            validator_commands=dense_item.get("validator_commands"),
        ),
        queue_step(
            "dense_fallback_comparison_artifact",
            stage="offline_build",
            status=comparison_status,
            path=resolve_repo_path(fallback_plan.get("fallback_artifact_path")),
            approval_required=False,
            command_options=command_items(
                fallback_plan,
                "dense_fallback_comparison_builder",
                "dense_fallback_comparison_validator",
            ),
        ),
        queue_step(
            "phase3_bundle_with_candidate_trace",
            stage="bundle_update",
            status=bundle_update_statuses["policy_candidate_bundle"],
            path=resolve_repo_path(policy_plan.get("updated_bundle_path")),
            approval_required=False,
            command_options=command_items(
                policy_plan,
                "phase3_bundle_builder_with_candidate_trace",
                "phase3_bundle_validator",
            ),
        ),
        queue_step(
            "phase3_bundle_with_fallback",
            stage="bundle_update",
            status=bundle_update_statuses["fallback_bundle"],
            path=resolve_repo_path(fallback_plan.get("updated_bundle_path")),
            approval_required=False,
            command_options=command_items(
                fallback_plan,
                "phase3_bundle_builder_with_fallback",
                "phase3_bundle_validator",
            ),
        ),
        queue_step(
            "live_capability_proof_fill",
            stage="future_adapter_proof",
            status=live_status,
            path=live_proof_path,
            approval_required=live_status != "already_satisfied",
            validator_commands=live_item.get("validator_commands"),
        ),
    ]
    if request_current:
        attach_approval_rebuild_command(operator_queue, request_audit)
    if request.get("capture_complete") is True:
        operator_queue.insert(
            4,
            queue_step(
                "record_runtime_approvals",
                stage="runtime_capture",
                status="already_satisfied" if capture_receipt_binding.get("approvals_ready") is True else "approval_required",
                path=path,
                approval_required=capture_receipt_binding.get("approvals_ready") is not True,
            ),
        )
    if not request_current:
        operator_queue.insert(
            0,
            queue_step(
                "regenerate_runtime_capture_request",
                stage="request_audit",
                status="blocked_by_drift",
                path=path,
                approval_required=False,
            ),
        )
    next_operator_step = first_incomplete_step(operator_queue)

    nonfatal_policy_fragments = ("candidate trace artifact does not exist:",)
    errors.extend(
        f"policy candidate: {error}"
        for error in policy_plan.get("errors", [])
        if not any(fragment in str(error) for fragment in nonfatal_policy_fragments)
    )
    errors.extend(f"dense fallback: {error}" for error in fallback_plan.get("errors", []))
    errors.extend(f"live capability proof: {error}" for error in live_summary.get("errors", []))
    errors.extend(f"bundle: {error}" for error in bundle_summary.get("errors", []))
    errors.extend(f"capture receipt binding: {error}" for error in capture_receipt_binding.get("errors", []))

    return {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "mode": "phase3_capture_result_intake_item",
        "path": display_path(path),
        "exists": True,
        "valid": not errors,
        "errors": errors,
        "request_valid": not request_errors,
        "request_audit": {
            "valid": request_audit.get("valid") is True,
            "drift_count": request_drift_count,
            "errors": request_audit.get("errors", []),
        },
        "request_drift_count": request_drift_count,
        "request_drift_free": request_current,
        "request_capture_complete_flag": request.get("capture_complete") is True,
        "request_ready_for_operator_capture_flag": request.get("ready_for_operator_capture") is True,
        "approved_but_capture_incomplete": request.get("ready_for_operator_capture") is True
        and request.get("capture_complete") is not True,
        "approvals_ready_for_recorded_capture": request_approvals_ready(request),
        "bundle_path": display_path(bundle_path),
        "name": request.get("name"),
        "model_id": request.get("model_id"),
        "backend_family": request.get("backend_family"),
        "prompt_family": request.get("prompt_family"),
        "prompt_set_path": display_path(prompt_set_path),
        "candidate_trace_path": display_path(candidate_trace_path),
        "candidate_trace_receipt_path": display_path(candidate_trace_receipt_path),
        "managed_output_path": display_path(managed_output_path),
        "dense_output_path": display_path(dense_output_path),
        "live_proof_path": display_path(live_proof_path),
        "policy_candidate": {
            "valid": policy_plan.get("valid"),
            "current_ready": policy_plan.get("current_replay", {}).get("policy_candidate_ready"),
            "candidate_trace_ready": policy_plan.get("candidate_trace", {}).get("policy_candidate_ready"),
            "candidate_trace_capture_receipt": policy_plan.get("candidate_trace", {}).get("capture_receipt", {}),
            "ready_after_plan": policy_ready,
            "candidate_policy_ids": policy_plan.get("candidate_trace", {}).get("candidate_policy_ids", []),
            "candidate_trace_status": artifact_request_status(policy_plan, "candidate_trace_artifact"),
            "candidate_replay_status": artifact_request_status(policy_plan, "candidate_policy_replay"),
            "bundle_update_status": bundle_update_statuses["policy_candidate_bundle"],
        },
        "dense_fallback": {
            "valid": fallback_plan.get("valid"),
            "metadata_ready_to_build_comparison": fallback_plan.get("metadata_ready_to_build_comparison") is True,
            "comparison_ready": fallback_ready_after_intake,
            "comparison_artifact_ready": fallback_ready,
            "capture_receipt_binding_ready": capture_binding_ready,
            "capture_receipt_binding_blockers": capture_receipt_binding.get("blockers", []),
            "managed_output_status": artifact_request_status(fallback_plan, "managed_output_summary"),
            "dense_output_status": artifact_request_status(fallback_plan, "dense_output_summary"),
            "comparison_status": artifact_request_status(fallback_plan, "dense_fallback_comparison_artifact"),
            "bundle_update_status": bundle_update_statuses["fallback_bundle"],
        },
        "capture_receipt_binding": capture_receipt_binding,
        "live_capability_proof": {
            "valid": live_summary.get("valid"),
            "proof_available": live_summary.get("proof_available") is True,
            "proof_ready": live_ready,
            "phase_3_gate": live_gate,
            "blocker_ids": [
                blocker.get("id")
                for blocker in live_summary.get("blockers", [])
                if isinstance(blocker, dict) and blocker.get("id")
            ],
        },
        "current_bundle": {
            "valid": bundle_summary.get("valid"),
            "phase3_complete": bundle_summary.get("phase3_complete") is True,
            "bundle_ready_for_phase4": bundle_summary.get("bundle_ready_for_phase4") is True,
            "no_go_reason_ids": bundle_summary.get("no_go_reason_ids", []),
        },
        "operator_queue": operator_queue,
        "next_operator_step": next_operator_step,
        "runtime_capture_step_count": sum(1 for step in operator_queue if step.get("stage") == "runtime_capture" and step.get("status") not in COMPLETE_STEP_STATUSES),
        "runtime_approval_missing_step_count": sum(
            1
            for step in operator_queue
            if step.get("stage") == "runtime_capture"
            and step.get("approval_state") == "missing"
            and step.get("status") not in COMPLETE_STEP_STATUSES
        ),
        "approved_runtime_capture_step_count": sum(
            1
            for step in operator_queue
            if step.get("stage") == "runtime_capture"
            and step.get("approval_state") == "recorded"
            and step.get("status") not in COMPLETE_STEP_STATUSES
        ),
        "approval_required_step_count": sum(1 for step in operator_queue if step.get("approval_required") is True and step.get("status") not in COMPLETE_STEP_STATUSES),
        "ready_to_update_bundle": ready_to_update_bundle,
        "bundle_update_statuses": bundle_update_statuses,
        "phase4_candidate_ready_after_intake": phase4_candidate_ready,
        "live_spike_candidate_ready_after_intake": live_spike_candidate_ready,
        "remaining_blockers_after_intake": blockers_for_intake(
            request_current=request_current,
            policy_ready=policy_ready,
            fallback_ready=fallback_ready_after_intake,
            live_gate=live_gate,
        ),
    }


def completion_receipt_intake_gate(
    completion_receipt_path: Path | None = None,
    completion_work_order_path: Path | None = None,
) -> JSONDict:
    if completion_receipt_path is None and completion_work_order_path is None:
        return {
            "provided": False,
            "valid": True,
            "ready_for_capture_result_intake": False,
            "receipt_complete": False,
            "errors": [],
            "blockers": [],
        }
    errors: list[str] = []
    receipt: JSONDict | None = None
    work_order: JSONDict | None = None
    if completion_receipt_path is None:
        errors.append("completion_receipt_path_missing")
    elif not completion_receipt_path.exists():
        errors.append(f"completion_receipt_path_missing_on_disk:{display_path(completion_receipt_path)}")
    else:
        receipt = load_json(completion_receipt_path)
    if completion_work_order_path is None:
        errors.append("completion_work_order_path_missing")
    elif not completion_work_order_path.exists():
        errors.append(f"completion_work_order_path_missing_on_disk:{display_path(completion_work_order_path)}")
    else:
        work_order = load_json(completion_work_order_path)
    validation = (
        plan_phase3_capture_completion_receipt.validate_completion_receipt(receipt, work_order)
        if receipt is not None and work_order is not None
        else {
            "valid": False,
            "receipt_complete": False,
            "ready_for_capture_result_intake": False,
            "errors": [],
        }
    )
    validation_errors = [str(error) for error in validation.get("errors", []) if isinstance(error, str)]
    errors.extend(validation_errors)
    ready = validation.get("ready_for_capture_result_intake") is True and not errors
    blockers = [] if ready else ["completion_receipt_validation_not_ready"]
    return {
        "provided": True,
        "valid": not errors and validation.get("valid") is True,
        "ready_for_capture_result_intake": ready,
        "receipt_complete": validation.get("receipt_complete") is True,
        "receipt_path": display_path(completion_receipt_path),
        "work_order_path": display_path(completion_work_order_path),
        "row_count": validation.get("row_count", 0),
        "complete_row_count": validation.get("complete_row_count", 0),
        "validator_passed_count": validation.get("validator_passed_count", 0),
        "missing_item_count": validation.get("missing_item_count", 0),
        "errors": errors,
        "blockers": blockers,
        "validation_summary": validation,
    }

def build_root_summary(root: Path = DEFAULT_ROOT, request_glob: str = REQUEST_GLOB, completion_receipt_path: Path | None = None, completion_work_order_path: Path | None = None) -> JSONDict:
    errors: list[str] = []
    completion_gate = completion_receipt_intake_gate(completion_receipt_path, completion_work_order_path)
    if completion_gate.get("provided") is True:
        errors.extend(f"completion receipt: {error}" for error in completion_gate.get("errors", []))
        if completion_gate.get("ready_for_capture_result_intake") is not True:
            errors.append("completion receipt validation not ready for capture-result intake")
    if not root.exists():
        errors.append(f"capture-result root does not exist: {display_path(root)}")
        request_paths: list[Path] = []
    else:
        request_paths = sorted(root.glob(request_glob))

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
                }
            )

    remaining_blockers = sorted(
        {
            blocker
            for request in requests
            for blocker in request.get("remaining_blockers_after_intake", [])
            if isinstance(blocker, str)
        }
        | {
            blocker
            for blocker in completion_gate.get("blockers", [])
            if isinstance(blocker, str)
        }
    )
    errors.extend(
        f"{request.get('path')}: {error}"
        for request in requests
        if request.get("valid") is not True
        for error in request.get("errors", [])
    )
    request_drift_free_count = sum(1 for request in requests if request.get("request_drift_free") is True)
    request_drifted_count = sum(
        1
        for request in requests
        if request.get("request_drift_free") is False or int(request.get("request_drift_count", 0) or 0) > 0
    )
    request_audit_valid_count = sum(
        1
        for request in requests
        if isinstance(request.get("request_audit"), dict) and request["request_audit"].get("valid") is True
    )
    receipt_gate_coverage = aggregate_receipt_gate_coverage(requests)
    receipt_manifest = receipt_fill_manifest(requests)
    receipt_manifest_summary = receipt_fill_manifest_summary(receipt_manifest)
    transition_manifest = approval_transition_preview_manifest(requests)
    recommended_transition_preview = transition_manifest[0] if transition_manifest else None
    capture_fill_plan_manifest = post_approval_capture_fill_plan_manifest(requests)
    recommended_capture_fill_plan = capture_fill_plan_manifest[0] if capture_fill_plan_manifest else None

    return {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "mode": "phase3_capture_result_intake",
        "valid": not errors,
        "errors": errors,
        "root": display_path(root),
        "request_glob": request_glob,
        "completion_receipt_gate": completion_gate,
        "completion_receipt_gate_provided": completion_gate.get("provided") is True,
        "completion_receipt_gate_ready": completion_gate.get("ready_for_capture_result_intake") is True,
        "completion_receipt_gate_complete": completion_gate.get("receipt_complete") is True,
        "completion_receipt_gate_error_count": len(completion_gate.get("errors", [])) if isinstance(completion_gate.get("errors"), list) else 0,
        "request_count": len(requests),
        "valid_request_count": sum(1 for request in requests if request.get("valid") is True),
        "request_audit_valid_count": request_audit_valid_count,
        "request_drift_free_count": request_drift_free_count,
        "request_drifted_count": request_drifted_count,
        "request_capture_complete_flag_count": sum(
            1 for request in requests if request.get("request_capture_complete_flag") is True
        ),
        "request_ready_for_operator_capture_flag_count": sum(
            1 for request in requests if request.get("request_ready_for_operator_capture_flag") is True
        ),
        "approved_but_capture_incomplete_request_count": sum(
            1 for request in requests if request.get("approved_but_capture_incomplete") is True
        ),
        "approvals_ready_for_recorded_capture_count": sum(
            1 for request in requests if request.get("approvals_ready_for_recorded_capture") is True
        ),
        "receipt_gate_coverage": receipt_gate_coverage,
        "receipt_fill_manifest": receipt_manifest,
        "receipt_fill_manifest_summary": receipt_manifest_summary,
        "receipt_fill_entry_count": receipt_manifest_summary["entry_count"],
        "receipt_fill_ready_count": receipt_manifest_summary["ready_count"],
        "receipt_fill_missing_count": receipt_manifest_summary["missing_count"],
        "receipt_fill_approval_missing_count": receipt_manifest_summary["approval_missing_count"],
        "capture_receipt_required_count": receipt_gate_coverage["capture_receipt_required_count"],
        "capture_receipt_ready_count": receipt_gate_coverage["capture_receipt_ready_count"],
        "capture_receipt_missing_request_count": receipt_gate_coverage["missing_request_count"],
        "output_receipt_binding_ready_count": receipt_gate_coverage["output_receipt_binding_ready_count"],
        "ready_to_update_bundle_count": sum(1 for request in requests if request.get("ready_to_update_bundle") is True),
        "phase4_candidate_ready_count": sum(
            1 for request in requests if request.get("phase4_candidate_ready_after_intake") is True
        ),
        "live_spike_candidate_ready_count": sum(
            1 for request in requests if request.get("live_spike_candidate_ready_after_intake") is True
        ),
        "next_operator_step_counts": queue_counts_by_next_step(requests),
        "runtime_approval_missing_request_count": sum(
            1
            for request in requests
            if isinstance(request.get("next_operator_step"), dict)
            and request["next_operator_step"].get("stage") == "runtime_capture"
            and request["next_operator_step"].get("approval_state") == "missing"
        ),
        "approved_runtime_capture_pending_request_count": sum(
            1
            for request in requests
            if isinstance(request.get("next_operator_step"), dict)
            and request["next_operator_step"].get("stage") == "runtime_capture"
            and request["next_operator_step"].get("approval_state") == "recorded"
        ),
        "approval_rebuild_command_available_request_count": sum(
            1 for request in requests if request_next_step_has_approval_rebuild_command(request)
        ),
        "approval_rebuild_command_manifest": approval_rebuild_command_manifest(requests),
        "approval_transition_preview_count": len(transition_manifest),
        "approval_transition_preview_manifest": transition_manifest,
        "recommended_approval_transition_preview": recommended_transition_preview,
        "approval_transition_ready_for_operator_count": sum(
            1
            for preview in transition_manifest
            if isinstance(preview.get("next_step_after_approval"), dict)
            and preview["next_step_after_approval"].get("status") == "ready_for_operator_capture"
        ),
        "approval_transition_ready_to_update_bundle_count": sum(
            1 for preview in transition_manifest if preview.get("ready_to_update_bundle_after_approval") is True
        ),
        "post_approval_capture_fill_plan_count": len(capture_fill_plan_manifest),
        "post_approval_capture_fill_plan_manifest": capture_fill_plan_manifest,
        "recommended_post_approval_capture_fill_plan": recommended_capture_fill_plan,
        "post_approval_capture_fill_artifact_step_count": sum(
            int(plan.get("artifact_step_count", 0) or 0) for plan in capture_fill_plan_manifest
        ),
        "post_approval_capture_fill_runtime_step_count": sum(
            int(plan.get("runtime_capture_step_count", 0) or 0) for plan in capture_fill_plan_manifest
        ),
        "post_approval_capture_fill_ready_count": sum(
            int(plan.get("ready_after_approval_count", 0) or 0) for plan in capture_fill_plan_manifest
        ),
        "post_approval_capture_fill_missing_count": sum(
            int(plan.get("missing_after_approval_count", 0) or 0) for plan in capture_fill_plan_manifest
        ),
        "post_approval_capture_fill_validator_command_count": sum(
            int(plan.get("validator_command_count", 0) or 0) for plan in capture_fill_plan_manifest
        ),
        "post_approval_capture_fill_ready_to_update_bundle_count": sum(
            1 for plan in capture_fill_plan_manifest if plan.get("ready_to_update_bundle_after_approval") is True
        ),
        "pending_runtime_capture_request_count": sum(
            1
            for request in requests
            if isinstance(request.get("next_operator_step"), dict)
            and request["next_operator_step"].get("stage") == "runtime_capture"
        ),
        "pending_bundle_update_request_count": sum(
            1
            for request in requests
            if isinstance(request.get("next_operator_step"), dict)
            and request["next_operator_step"].get("stage") == "bundle_update"
        ),
        "remaining_blockers_after_intake": remaining_blockers,
        "requests": requests,
        "safety_contract": [
            "intake reads saved runtime-capture requests and local artifacts only",
            "intake does not launch model servers",
            "intake does not run Docker",
            "intake does not call endpoints",
            "intake does not download models",
            "intake does not inspect private tokens",
            "intake does not mutate runtime residency",
            "intake does not send prompt traffic",
            "intake does not claim live expert paging",
        ],
        "next_actions": [
            "Fill capture artifacts only after explicit runtime approval.",
            "Validate the filled completion receipt before capture-result intake when a handoff completion receipt is supplied.",
            "Rerun this intake before rebuilding a Phase 3 bundle from captured artifacts.",
            "Update the bundle only when an intake request reports ready_to_update_bundle and supplied completion receipt evidence is ready.",
            "Treat phase4_candidate_ready and live_spike_candidate_ready as separate gates.",
        ],
    }


def print_human_summary(summary: JSONDict, *, show_queue: bool = False, queue_limit: int = 6) -> None:
    print("MoE Run Anyway Phase 3 capture-result intake")
    print(f"Valid: {summary['valid']}")
    print(f"Requests: {summary['request_count']}")
    print(f"Completion receipt gate provided: {summary.get('completion_receipt_gate_provided')}")
    print(f"Completion receipt gate ready: {summary.get('completion_receipt_gate_ready')}")
    print(f"Completion receipt gate complete: {summary.get('completion_receipt_gate_complete')}")
    print(f"Completion receipt gate errors: {summary.get('completion_receipt_gate_error_count')}")
    print(f"Drift-free saved requests: {summary['request_drift_free_count']}")
    print(f"Drifted saved requests: {summary['request_drifted_count']}")
    print(f"Capture-complete flags: {summary['request_capture_complete_flag_count']}")
    print(f"Ready-for-operator flags: {summary.get('request_ready_for_operator_capture_flag_count')}")
    print(f"Approved but capture incomplete: {summary.get('approved_but_capture_incomplete_request_count')}")
    print(f"Runtime approvals still missing: {summary.get('runtime_approval_missing_request_count')}")
    print(f"Approval rebuild commands available: {summary.get('approval_rebuild_command_available_request_count')}")
    print(f"Approval transition previews: {summary.get('approval_transition_preview_count')}")
    print(f"Approval previews ready for operator: {summary.get('approval_transition_ready_for_operator_count')}")
    print(f"Approval previews ready to update bundle: {summary.get('approval_transition_ready_to_update_bundle_count')}")
    print(f"Post-approval capture-fill plans: {summary.get('post_approval_capture_fill_plan_count')}")
    print(
        "Post-approval capture-fill ready: "
        f"{summary.get('post_approval_capture_fill_ready_count')}/"
        f"{summary.get('post_approval_capture_fill_artifact_step_count')}"
    )
    print(f"Post-approval capture-fill missing: {summary.get('post_approval_capture_fill_missing_count')}")
    print(f"Post-approval capture-fill validator commands: {summary.get('post_approval_capture_fill_validator_command_count')}")
    print(f"Approved runtime-capture pending: {summary.get('approved_runtime_capture_pending_request_count')}")
    print(f"Capture receipts ready: {summary.get('capture_receipt_ready_count')}/{summary.get('capture_receipt_required_count')}")
    print(f"Receipt-fill manifest ready: {summary.get('receipt_fill_ready_count')}/{summary.get('receipt_fill_entry_count')}")
    print(f"Receipt-fill approvals missing: {summary.get('receipt_fill_approval_missing_count')}")
    print(f"Requests missing receipt gates: {summary.get('capture_receipt_missing_request_count')}")
    print(f"Output receipt bindings ready: {summary.get('output_receipt_binding_ready_count')}")
    print(f"Ready to update bundle: {summary['ready_to_update_bundle_count']}")
    print(f"Phase 4 candidates: {summary['phase4_candidate_ready_count']}")
    print(f"Live-spike candidates: {summary['live_spike_candidate_ready_count']}")
    print(f"Pending runtime-capture next steps: {summary['pending_runtime_capture_request_count']}")
    print(f"Pending bundle-update next steps: {summary['pending_bundle_update_request_count']}")
    if show_queue:
        print("Next operator steps:")
        for request in summary.get("requests", [])[: max(queue_limit, 0)]:
            if isinstance(request, dict):
                print(f"  - {request_queue_line(request)}")
    if summary["remaining_blockers_after_intake"]:
        print("Remaining blockers after intake:")
        for blocker in summary["remaining_blockers_after_intake"]:
            print(f"  - {blocker}")
    print("Safety contract:")
    for item in summary["safety_contract"]:
        print(f"  - {item}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--request-glob", default=REQUEST_GLOB)
    parser.add_argument("--completion-receipt", type=Path, help="optional filled completion receipt JSON to validate before intake promotion")
    parser.add_argument("--completion-work-order", type=Path, help="recommended runtime-capture work order JSON for completion receipt binding")
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    parser.add_argument("--output-md", type=Path, help="write a Markdown intake queue report")
    parser.add_argument("--show-queue", action="store_true", help="print per-request next operator steps")
    parser.add_argument("--queue-limit", type=int, default=6, help="maximum request queue lines to print")
    return parser


def plan_path(
    root: Path = DEFAULT_ROOT,
    request_glob: str = REQUEST_GLOB,
    completion_receipt_path: Path | None = None,
    completion_work_order_path: Path | None = None,
) -> tuple[int, JSONDict | None, str | None]:
    try:
        summary = build_root_summary(
            root=root,
            request_glob=request_glob,
            completion_receipt_path=completion_receipt_path,
            completion_work_order_path=completion_work_order_path,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not build Phase 3 capture-result intake: {exc}"
    return (0 if summary["valid"] else 2), summary, None


def main_from_test_path(root: Path = DEFAULT_ROOT) -> int:
    status, _, _ = plan_path(root=root)
    return status


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary, error_message = plan_path(
        root=args.root,
        request_glob=args.request_glob,
        completion_receipt_path=args.completion_receipt,
        completion_work_order_path=args.completion_work_order,
    )
    if error_message:
        print(error_message, file=sys.stderr)
        return status
    assert summary is not None
    if args.output_md:
        write_markdown_report(summary, args.output_md)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_human_summary(summary, show_queue=args.show_queue, queue_limit=args.queue_limit)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
