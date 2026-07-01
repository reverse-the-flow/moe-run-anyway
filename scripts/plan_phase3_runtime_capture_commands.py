#!/usr/bin/env python3
"""Plan Phase 3 runtime-capture command bindings for a saved request.

This planner turns a saved runtime-capture request into a launch-card/callable
command contract for the artifacts that still require approved runtime work. It
only reads local JSON metadata. It does not launch runtimes, run Docker, call
endpoints, inspect secrets, mutate residency, or send prompt traffic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import plan_moe_probe_manifest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUEST_PATH = (
    ROOT
    / "memory-moe-mvp"
    / "phase3-real-evidence"
    / "pc_mixtral_phase3_real_evidence_bundle.runtime-capture-request.json"
)
SUPPORTED_SCHEMA_VERSION = "moe-phase3-runtime-capture-command-contract-v1"
LAUNCH_CARD_SCHEMA_VERSION = "moe-phase3-runtime-capture-launch-card-v1"
LAUNCH_CARD_BINDING_SCHEMA_VERSION = "moe-phase3-runtime-capture-launch-card-binding-v1"
PLACEHOLDER_COMMAND = "model-plane-callable-or-launch-card"
RUNTIME_ARTIFACT_IDS = (
    "candidate_router_trace",
    "managed_output_summary_fill",
    "dense_output_summary_fill",
)
CAPTURE_KIND_BY_ARTIFACT = {
    "candidate_router_trace": "llama_cpp_router_trace_jsonl",
    "managed_output_summary_fill": "managed_output_summary_json",
    "dense_output_summary_fill": "dense_output_summary_json",
}
RECEIPT_KIND_BY_ARTIFACT = {
    "candidate_router_trace": "trace_capture_receipt_json",
    "managed_output_summary_fill": "embedded_output_capture_receipt",
    "dense_output_summary_fill": "embedded_output_capture_receipt",
}
APPROVAL_KEY_BY_ARTIFACT = {
    "candidate_router_trace": "router_trace_capture_approved",
    "managed_output_summary_fill": "managed_output_capture_approved",
    "dense_output_summary_fill": "dense_output_capture_approved",
}
JSONDict = dict[str, Any]


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


def load_json(path: Path) -> JSONDict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{display_path(path) or path} must be a JSON object")
    return payload


def string_value(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def items_by_id(items: Any) -> dict[str, JSONDict]:
    result: dict[str, JSONDict] = {}
    if not isinstance(items, list):
        return result
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            result[str(item["id"])] = item
    return result


def request_prompt_set_path(request: JSONDict, artifact: JSONDict) -> str | None:
    source = artifact.get("source") if isinstance(artifact.get("source"), dict) else {}
    prompt_path = source.get("prompt_set_path")
    if isinstance(prompt_path, str) and prompt_path.strip():
        return prompt_path
    prompt_set = request.get("prompt_set") if isinstance(request.get("prompt_set"), dict) else {}
    prompt_path = prompt_set.get("path")
    return prompt_path if isinstance(prompt_path, str) and prompt_path.strip() else None


def receipt_path_for_artifact(artifact_id: str, artifact: JSONDict) -> str | None:
    source = artifact.get("source") if isinstance(artifact.get("source"), dict) else {}
    receipt_path = source.get("capture_receipt_path")
    if isinstance(receipt_path, str) and receipt_path.strip():
        return receipt_path
    artifact_path = artifact.get("path")
    if artifact_id in {"managed_output_summary_fill", "dense_output_summary_fill"} and isinstance(artifact_path, str):
        return artifact_path
    return None


def compact_model_plane_plan(manifest_path: Path | None) -> JSONDict:
    if manifest_path is None:
        return {
            "provided": False,
            "valid": False,
            "ready_for_binding": False,
            "blockers": ["model_plane_probe_manifest_missing"],
        }
    try:
        manifest = plan_moe_probe_manifest.load_manifest(manifest_path)
        plan = plan_moe_probe_manifest.build_plan(manifest, manifest_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {
            "provided": True,
            "path": display_path(manifest_path),
            "valid": False,
            "ready_for_binding": False,
            "blockers": ["model_plane_probe_manifest_invalid"],
            "errors": [str(exc)],
        }
    request = plan.get("planned_harness_run_request") if isinstance(plan.get("planned_harness_run_request"), dict) else {}
    runtime_observability = manifest.get("runtime_observability") if isinstance(manifest.get("runtime_observability"), dict) else {}
    valid = plan.get("valid") is True
    return {
        "provided": True,
        "path": display_path(manifest_path),
        "valid": valid,
        "ready_for_binding": valid and bool(request),
        "profile_id": plan.get("profile_id"),
        "model_id": plan.get("model_id"),
        "model_path": manifest.get("model_path"),
        "backend_family": plan.get("backend_family"),
        "base_url": manifest.get("base_url"),
        "health_url": manifest.get("health_url"),
        "container_name": manifest.get("container_name"),
        "log_file_path": manifest.get("log_file_path"),
        "runtime_observability": runtime_observability,
        "target_class": plan.get("target_class"),
        "planned_stage": request.get("stage"),
        "planned_status": request.get("status"),
        "expected_artifact_class": request.get("expected_artifact_class"),
        "safe_command_count": len(request.get("safe_commands", [])) if isinstance(request.get("safe_commands"), list) else 0,
        "deferred_live_command_count": len(request.get("deferred_live_commands", [])) if isinstance(request.get("deferred_live_commands"), list) else 0,
        "errors": plan.get("errors", []),
        "blockers": [] if valid and request else ["model_plane_harness_plan_missing"],
    }


def command_binding_template(request_path: str, artifact_id: str, artifact_path: str | None, receipt_path: str | None) -> JSONDict:
    return {
        "command_class": "phase3_model_plane_launch_card_runtime_capture",
        "planned_only": True,
        "requires_runtime": True,
        "requires_prompt_traffic": True,
        "requires_explicit_user_approval": True,
        "may_send_prompt_traffic": True,
        "runtime_capture_request_path": request_path,
        "metadata_only": False,
        "command_ready": False,
        "command": [],
        "command_shape": [
            "model-plane-callable-or-launch-card",
            "--runtime-capture-request",
            request_path,
            "--artifact-id",
            artifact_id,
            "--artifact-output",
            artifact_path or "<artifact_path>",
            "--receipt-output",
            receipt_path or "<receipt_path>",
        ],
        "missing_binding": "model_plane_launch_card_command_missing",
    }


def capture_task_for_artifact(request: JSONDict, request_path: str, artifact_id: str, artifact: JSONDict) -> JSONDict:
    artifact_path = artifact.get("path") if isinstance(artifact.get("path"), str) else None
    receipt_path = receipt_path_for_artifact(artifact_id, artifact)
    prompt_set_path = request_prompt_set_path(request, artifact)
    approval_keys = ["runtime_prompt_traffic_approved", APPROVAL_KEY_BY_ARTIFACT[artifact_id]]
    blockers: list[str] = []
    if not artifact_path:
        blockers.append("artifact_path_missing")
    if not receipt_path:
        blockers.append("receipt_path_missing")
    if not prompt_set_path:
        blockers.append("prompt_set_path_missing")
    if artifact.get("status") == "already_satisfied":
        blockers.append("artifact_already_satisfied")
    return {
        "artifact_id": artifact_id,
        "capture_kind": CAPTURE_KIND_BY_ARTIFACT[artifact_id],
        "receipt_kind": RECEIPT_KIND_BY_ARTIFACT[artifact_id],
        "status": artifact.get("status"),
        "approval_required": artifact.get("approval_required") is True,
        "artifact_path": artifact_path,
        "receipt_path": receipt_path,
        "prompt_set_path": prompt_set_path,
        "approval_keys": approval_keys,
        "validator_command_count": len(artifact.get("validator_commands", [])) if isinstance(artifact.get("validator_commands"), list) else 0,
        "planned_command_binding": command_binding_template(request_path, artifact_id, artifact_path, receipt_path),
        "command_binding_ready": False,
        "blockers": blockers,
        "next_actions": [
            "Bind this task to a Model Plane callable or launch card before treating it as executable.",
            "Run only after explicit approval for runtime prompt traffic and this artifact capture kind.",
            "Fill the artifact and receipt paths named here, then rerun capture-result intake.",
        ],
    }


def build_summary(request_path: Path = DEFAULT_REQUEST_PATH, *, model_plane_manifest_path: Path | None = None, launch_card_path: Path | None = None) -> JSONDict:
    request = load_json(request_path)
    request_path_text = display_path(request_path) or str(request_path)
    artifacts = items_by_id(request.get("requested_artifacts"))
    errors: list[str] = []
    tasks: list[JSONDict] = []
    blocking_task_errors = {"artifact_path_missing", "receipt_path_missing", "prompt_set_path_missing"}
    for artifact_id in RUNTIME_ARTIFACT_IDS:
        artifact = artifacts.get(artifact_id)
        if artifact is None:
            errors.append(f"requested_artifacts missing {artifact_id}")
            continue
        task = capture_task_for_artifact(request, request_path_text, artifact_id, artifact)
        tasks.append(task)
        errors.extend(
            f"{artifact_id}: {blocker}"
            for blocker in task.get("blockers", [])
            if blocker in blocking_task_errors
        )
    model_plane_plan = compact_model_plane_plan(model_plane_manifest_path)
    missing_bindings = [task["artifact_id"] for task in tasks if task.get("command_binding_ready") is not True]
    # The contract can be ready while executable commands are still absent.
    command_contract_ready = len(tasks) == len(RUNTIME_ARTIFACT_IDS) and not errors
    prompt_set = request.get("prompt_set") if isinstance(request.get("prompt_set"), dict) else {}
    prompt_set_path = prompt_set.get("path") if isinstance(prompt_set.get("path"), str) else None
    summary = {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "mode": "phase3_runtime_capture_command_contract",
        "valid": not errors,
        "errors": errors,
        "request_path": request_path_text,
        "request_name": request.get("name"),
        "bundle_path": request.get("bundle_path"),
        "model_id": request.get("model_id"),
        "backend_family": request.get("backend_family"),
        "prompt_family": request.get("prompt_family"),
        "prompt_set_path": prompt_set_path,
        "runtime_prompt_traffic_approved": request.get("runtime_prompt_traffic_approved") is True,
        "command_contract_ready": command_contract_ready,
        "runtime_capture_command_ready": False,
        "planned_capture_count": len(tasks),
        "runtime_command_option_count": 0,
        "missing_runtime_command_count": len(missing_bindings),
        "missing_runtime_command_artifact_ids": missing_bindings,
        "model_plane_manifest": model_plane_plan,
        "capture_tasks": tasks,
        "binding_requirements": [
            "Model Plane profile or launch card selected for the target host.",
            "Callable command records explicit runtime approval and artifact-specific approval.",
            "Command writes exactly the artifact_path and receipt_path listed for the task.",
            "Command returns a run-scoped artifact manifest suitable for capture-result intake.",
            "Cleanup/restore evidence remains future-bound until live actuator proof exists.",
        ],
        "next_actions": [
            "Bind each planned task to a real Model Plane callable or launch-card command.",
            "Keep these planned bindings out of executable command coverage until command_ready=true.",
            "After approved capture, rerun scripts/plan_phase3_capture_result_intake.py before updating bundles.",
        ],
        "safety_contract": [
            "command-contract planner reads local metadata only",
            "command-contract planner does not launch model servers",
            "command-contract planner does not run Docker",
            "command-contract planner does not call endpoints",
            "command-contract planner does not inspect private tokens",
            "command-contract planner does not send prompt traffic",
            "command-contract planner does not mutate runtime residency",
            "command-contract planner does not claim live expert paging",
        ],
    }
    summary["launch_card_binding_summary"] = empty_launch_card_binding_summary()
    if launch_card_path is not None:
        binding_summary = validate_launch_card_bindings(
            summary,
            load_json(launch_card_path),
            launch_card_path=launch_card_path,
        )
        summary["launch_card_binding_summary"] = binding_summary
        if binding_summary.get("valid") is not True:
            summary["valid"] = False
            summary["errors"].extend(f"launch_card_binding: {error}" for error in binding_summary.get("errors", []))
        if binding_summary.get("binding_ready") is True:
            summary["runtime_capture_command_ready"] = True
            summary["runtime_command_option_count"] = binding_summary.get("command_option_count", 0)
            summary["missing_runtime_command_count"] = 0
            summary["missing_runtime_command_artifact_ids"] = []
            ready_by_artifact = {
                item.get("artifact_id"): item
                for item in binding_summary.get("task_bindings", [])
                if isinstance(item, dict) and item.get("binding_ready") is True
            }
            for task in summary["capture_tasks"]:
                binding = ready_by_artifact.get(task.get("artifact_id"))
                if not isinstance(binding, dict):
                    continue
                task["command_binding_ready"] = True
                task["bound_runtime_command"] = binding
                planned = task.get("planned_command_binding") if isinstance(task.get("planned_command_binding"), dict) else {}
                planned["command_ready"] = True
                planned["missing_binding"] = None
                planned["model_plane_callable_id"] = binding.get("model_plane_callable_id")
                planned["launch_command"] = binding.get("launch_command")
                task["planned_command_binding"] = planned
    launch_card = build_launch_card_template(summary)
    summary["launch_card_template_summary"] = {
        "template_ready": launch_card["template_ready"],
        "model_plane_binding_ready": launch_card["model_plane_binding_ready"],
        "runtime_capture_command_ready": launch_card["runtime_capture_command_ready"],
        "task_count": launch_card["task_count"],
        "missing_runtime_command_count": launch_card["missing_runtime_command_count"],
    }
    return summary


def build_launch_card_task(task: JSONDict) -> JSONDict:
    binding = task.get("planned_command_binding") if isinstance(task.get("planned_command_binding"), dict) else {}
    artifact_id = task.get("artifact_id")
    return {
        "task_id": f"phase3_capture_{artifact_id}",
        "artifact_id": artifact_id,
        "capture_kind": task.get("capture_kind"),
        "receipt_kind": task.get("receipt_kind"),
        "status": "planned_only",
        "approval_keys": task.get("approval_keys", []),
        "requires_explicit_user_approval": True,
        "requires_prompt_traffic": True,
        "requires_runtime": True,
        "prompt_set_path": task.get("prompt_set_path"),
        "artifact_output_path": task.get("artifact_path"),
        "receipt_output_path": task.get("receipt_path"),
        "validator_command_count": task.get("validator_command_count"),
        "command_binding_ready": task.get("command_binding_ready") is True,
        "command_placeholder": binding.get("command_shape", []),
        "binding_template": {
            "binding_kind": "model_plane_callable_or_launch_command",
            "model_plane_callable_id": binding.get("model_plane_callable_id") or "",
            "launch_command": list_of_strings(binding.get("launch_command")),
            "command_ready": binding.get("command_ready") is True,
            "requires_explicit_user_approval": True,
            "may_send_prompt_traffic": True,
            "runtime_capture_request_path": binding.get("runtime_capture_request_path"),
            "reads_prompt_set_path": task.get("prompt_set_path"),
            "approval_keys": task.get("approval_keys", []),
            "writes_artifact_path": task.get("artifact_path"),
            "writes_receipt_path": task.get("receipt_path"),
        },
        "expected_result": {
            "reads_prompt_set_path": task.get("prompt_set_path"),
            "writes_artifact_path": task.get("artifact_path"),
            "writes_receipt_path": task.get("receipt_path"),
            "rerun_after_capture": [
                "scripts/plan_phase3_capture_result_intake.py",
                "scripts/plan_phase3_evidence_packet.py",
            ],
        },
        "post_capture_next_actions": [
            "Validate the artifact and receipt paths named in this task.",
            "Run capture-result intake before updating any Phase 3 bundle.",
            "Keep live residency/control proof separate from this runtime capture card.",
        ],
    }


def build_launch_card_template(summary: JSONDict) -> JSONDict:
    model_plane = summary.get("model_plane_manifest") if isinstance(summary.get("model_plane_manifest"), dict) else {}
    tasks = [build_launch_card_task(task) for task in summary.get("capture_tasks", []) if isinstance(task, dict)]
    template_ready = summary.get("valid") is True and summary.get("command_contract_ready") is True and len(tasks) > 0
    model_plane_binding_ready = model_plane.get("ready_for_binding") is True
    blockers: list[str] = []
    if not template_ready:
        blockers.append("command_contract_not_ready")
    if not model_plane_binding_ready:
        blockers.append("model_plane_binding_context_missing")
    if summary.get("runtime_capture_command_ready") is not True:
        blockers.append("runtime_capture_commands_unbound")
    return {
        "schema_version": LAUNCH_CARD_SCHEMA_VERSION,
        "mode": "phase3_runtime_capture_launch_card_template",
        "status": "planned_only",
        "valid": template_ready,
        "template_ready": template_ready,
        "model_plane_binding_ready": model_plane_binding_ready,
        "runtime_capture_command_ready": summary.get("runtime_capture_command_ready") is True,
        "executable": False,
        "request_path": summary.get("request_path"),
        "request_name": summary.get("request_name"),
        "bundle_path": summary.get("bundle_path"),
        "model_id": summary.get("model_id"),
        "backend_family": summary.get("backend_family"),
        "prompt_family": summary.get("prompt_family"),
        "prompt_set_path": summary.get("prompt_set_path"),
        "model_plane_profile": {
            "provided": model_plane.get("provided") is True,
            "ready_for_binding": model_plane_binding_ready,
            "profile_id": model_plane.get("profile_id"),
            "model_id": model_plane.get("model_id"),
            "model_path": model_plane.get("model_path"),
            "backend_family": model_plane.get("backend_family"),
            "base_url": model_plane.get("base_url"),
            "health_url": model_plane.get("health_url"),
            "container_name": model_plane.get("container_name"),
            "log_file_path": model_plane.get("log_file_path"),
            "runtime_observability": model_plane.get("runtime_observability", {}),
            "blockers": model_plane.get("blockers", []),
        },
        "preflight_gates": [
            "explicit_user_approval_for_runtime_prompt_traffic",
            "model_plane_profile_validated_for_target_host",
            "runtime_health_check_passed_before_prompt_traffic",
            "artifact_and_receipt_outputs_are_repo_relative_and_exact",
            "capture_result_intake_runs_before_bundle_update",
        ],
        "task_count": len(tasks),
        "missing_runtime_command_count": summary.get("missing_runtime_command_count"),
        "missing_runtime_command_artifact_ids": summary.get("missing_runtime_command_artifact_ids", []),
        "tasks": tasks,
        "blockers": blockers,
        "next_actions": [
            "Attach real Model Plane callable ids or launch-card commands for each planned task.",
            "Do not execute runtime prompt traffic from this template until explicit approval is recorded.",
            "After capture, fill receipts and rerun capture-result intake before bundle promotion.",
        ],
        "safety_contract": [
            "launch-card template is planned-only",
            "launch-card template does not launch model servers",
            "launch-card template does not run Docker",
            "launch-card template does not call endpoints",
            "launch-card template does not inspect private tokens",
            "launch-card template does not send prompt traffic",
            "launch-card template does not mutate runtime residency",
            "launch-card template does not claim live expert paging",
        ],
    }


def empty_launch_card_binding_summary() -> JSONDict:
    return {
        "schema_version": LAUNCH_CARD_BINDING_SCHEMA_VERSION,
        "valid": True,
        "binding_ready": False,
        "launch_card_path": None,
        "task_count": 0,
        "bound_task_count": 0,
        "command_option_count": 0,
        "missing_runtime_command_count": 0,
        "missing_runtime_command_artifact_ids": [],
        "errors": [],
        "blockers": ["launch_card_not_provided"],
        "task_bindings": [],
    }


def binding_payload_for_task(card_task: JSONDict) -> JSONDict:
    binding = card_task.get("binding")
    if isinstance(binding, dict):
        return binding
    template = card_task.get("binding_template")
    return template if isinstance(template, dict) else {}


def launch_command_ready(command: list[str]) -> bool:
    return bool(command) and command[0] != PLACEHOLDER_COMMAND


def binding_has_command_option(binding: JSONDict) -> bool:
    return string_value(binding.get("model_plane_callable_id")) is not None or launch_command_ready(
        list_of_strings(binding.get("launch_command"))
    )


def binding_command_ready(binding: JSONDict) -> bool:
    return binding_has_command_option(binding) and binding.get("command_ready") is True


def validate_launch_card_bindings(summary: JSONDict, launch_card: JSONDict, *, launch_card_path: Path | None = None) -> JSONDict:
    errors: list[str] = []
    raw_tasks = launch_card.get("tasks")
    card_tasks: dict[str, JSONDict] = {}
    if launch_card.get("schema_version") != LAUNCH_CARD_SCHEMA_VERSION:
        errors.append("launch_card_schema_version_mismatch")
    if launch_card.get("request_path") != summary.get("request_path"):
        errors.append("launch_card_request_path_mismatch")
    if not isinstance(raw_tasks, list):
        errors.append("launch_card_tasks_missing")
        raw_tasks = []
    for raw_task in raw_tasks:
        if not isinstance(raw_task, dict):
            errors.append("launch_card_task_must_be_object")
            continue
        artifact_id = raw_task.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            errors.append("launch_card_task_artifact_id_missing")
            continue
        if artifact_id in card_tasks:
            errors.append(f"{artifact_id}: launch_card_task_duplicate")
            continue
        card_tasks[artifact_id] = raw_task

    task_bindings: list[JSONDict] = []
    expected_tasks = [task for task in summary.get("capture_tasks", []) if isinstance(task, dict)]
    for task in expected_tasks:
        artifact_id = str(task.get("artifact_id"))
        card_task = card_tasks.get(artifact_id)
        item_errors: list[str] = []
        blockers: list[str] = []
        binding_ready = False
        model_plane_callable_id: str | None = None
        launch_command: list[str] = []
        command_option_ready = False
        command_ready = False
        if card_task is None:
            blockers.append("launch_card_task_missing")
        else:
            expected_paths = {
                "artifact_output_path": task.get("artifact_path"),
                "receipt_output_path": task.get("receipt_path"),
                "prompt_set_path": task.get("prompt_set_path"),
            }
            for field, expected in expected_paths.items():
                if card_task.get(field) != expected:
                    item_errors.append(f"{field}_mismatch")
            binding = binding_payload_for_task(card_task)
            model_plane_callable_id = string_value(binding.get("model_plane_callable_id"))
            raw_launch_command = binding.get("launch_command")
            launch_command = list_of_strings(raw_launch_command)
            if isinstance(raw_launch_command, list) and len(raw_launch_command) != len(launch_command):
                item_errors.append("launch_command_must_be_strings")
            command_option_ready = model_plane_callable_id is not None or launch_command_ready(launch_command)
            command_ready = command_option_ready and binding.get("command_ready") is True
            if not command_option_ready:
                blockers.append("runtime_command_binding_missing")
            elif binding.get("command_ready") is not True:
                blockers.append("runtime_command_ready_flag_missing")
            if binding.get("requires_explicit_user_approval") is not True:
                blockers.append("explicit_user_approval_gate_missing")
            if binding.get("may_send_prompt_traffic") is not True:
                blockers.append("prompt_traffic_ack_missing")
            if binding.get("runtime_capture_request_path") != summary.get("request_path"):
                blockers.append("runtime_capture_request_path_missing")
            if binding.get("reads_prompt_set_path") != task.get("prompt_set_path"):
                blockers.append("reads_prompt_set_path_missing")
            approval_keys = set(list_of_strings(binding.get("approval_keys")))
            expected_approval_keys = set(list_of_strings(task.get("approval_keys")))
            if not expected_approval_keys.issubset(approval_keys):
                blockers.append("approval_keys_missing")
            if binding.get("writes_artifact_path") != task.get("artifact_path"):
                blockers.append("writes_artifact_path_missing")
            if binding.get("writes_receipt_path") != task.get("receipt_path"):
                blockers.append("writes_receipt_path_missing")
            binding_ready = not item_errors and not blockers
            errors.extend(f"{artifact_id}: {error}" for error in item_errors)
        task_bindings.append(
            {
                "artifact_id": artifact_id,
                "binding_ready": binding_ready,
                "command_option_ready": command_option_ready,
                "command_ready": command_ready,
                "model_plane_callable_id": model_plane_callable_id,
                "launch_command": launch_command,
                "runtime_capture_request_path": binding.get("runtime_capture_request_path") if card_task is not None else None,
                "reads_prompt_set_path": binding.get("reads_prompt_set_path") if card_task is not None else None,
                "may_send_prompt_traffic": binding.get("may_send_prompt_traffic") is True if card_task is not None else False,
                "artifact_output_path": task.get("artifact_path"),
                "receipt_output_path": task.get("receipt_path"),
                "prompt_set_path": task.get("prompt_set_path"),
                "errors": item_errors,
                "blockers": blockers,
            }
        )

    missing_artifact_ids = [item["artifact_id"] for item in task_bindings if item.get("binding_ready") is not True]
    blockers = sorted({blocker for item in task_bindings for blocker in item.get("blockers", [])})
    binding_ready = not errors and not missing_artifact_ids and len(task_bindings) == len(RUNTIME_ARTIFACT_IDS)
    return {
        "schema_version": LAUNCH_CARD_BINDING_SCHEMA_VERSION,
        "valid": not errors,
        "binding_ready": binding_ready,
        "launch_card_path": display_path(launch_card_path),
        "task_count": len(task_bindings),
        "bound_task_count": sum(1 for item in task_bindings if item.get("binding_ready") is True),
        "command_option_count": sum(1 for item in task_bindings if item.get("command_option_ready") is True),
        "command_ready_count": sum(1 for item in task_bindings if item.get("command_ready") is True),
        "missing_runtime_command_count": len(missing_artifact_ids),
        "missing_runtime_command_artifact_ids": missing_artifact_ids,
        "errors": errors,
        "blockers": blockers,
        "task_bindings": task_bindings,
    }

def write_launch_card_template(summary: JSONDict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_launch_card_template(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")

def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def format_markdown_report(summary: JSONDict) -> str:
    lines = [
        "# Phase 3 Runtime-Capture Command Contract",
        "",
        f"- Valid: `{summary.get('valid')}`",
        f"- Request: `{markdown_escape(summary.get('request_name') or 'missing')}`",
        f"- Contract ready: `{summary.get('command_contract_ready')}`",
        f"- Runtime capture commands ready: `{summary.get('runtime_capture_command_ready')}`",
        f"- Planned captures: `{summary.get('planned_capture_count')}`",
        f"- Runtime command options: `{summary.get('runtime_command_option_count')}`",
        f"- Missing runtime commands: `{summary.get('missing_runtime_command_count')}`",
        "",
        "| Artifact | Capture kind | Artifact path | Receipt path | Command ready | Validators |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for task in summary.get("capture_tasks", []):
        if not isinstance(task, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                markdown_escape(value)
                for value in (
                    task.get("artifact_id"),
                    task.get("capture_kind"),
                    task.get("artifact_path"),
                    task.get("receipt_path"),
                    task.get("command_binding_ready"),
                    task.get("validator_command_count"),
                )
            )
            + " |"
        )
    launch_card_summary = summary.get("launch_card_template_summary") if isinstance(summary.get("launch_card_template_summary"), dict) else {}
    lines.extend(
        [
            "",
            "## Launch-Card Template",
            "",
            f"- Template ready: `{launch_card_summary.get('template_ready')}`",
            f"- Model Plane binding ready: `{launch_card_summary.get('model_plane_binding_ready')}`",
            f"- Runtime capture commands ready: `{launch_card_summary.get('runtime_capture_command_ready')}`",
            f"- Tasks: `{launch_card_summary.get('task_count')}`",
            f"- Missing runtime commands: `{launch_card_summary.get('missing_runtime_command_count')}`",
        ]
    )
    binding_summary = summary.get("launch_card_binding_summary") if isinstance(summary.get("launch_card_binding_summary"), dict) else {}
    binding_blockers = binding_summary.get("blockers", []) if isinstance(binding_summary.get("blockers"), list) else []
    lines.extend(
        [
            "",
            "## Launch-Card Binding Intake",
            "",
            f"- Binding ready: `{binding_summary.get('binding_ready')}`",
            f"- Valid: `{binding_summary.get('valid')}`",
            f"- Bound tasks: `{binding_summary.get('bound_task_count')}`",
            f"- Command options: `{binding_summary.get('command_option_count')}`",
            f"- Missing runtime commands: `{binding_summary.get('missing_runtime_command_count')}`",
        ]
    )
    if binding_blockers:
        lines.append(f"- Blockers: `{markdown_escape(', '.join(str(item) for item in binding_blockers))}`")
    lines.extend(["", "## Binding Requirements", ""])
    for item in summary.get("binding_requirements", []):
        lines.append(f"- {markdown_escape(item)}")
    lines.extend(["", "## Safety Contract", ""])
    for item in summary.get("safety_contract", []):
        lines.append(f"- {markdown_escape(item)}")
    return "\n".join(lines) + "\n"


def write_markdown_report(summary: JSONDict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_markdown_report(summary), encoding="utf-8")


def print_human_summary(summary: JSONDict) -> None:
    print("MoE Run Anyway Phase 3 runtime-capture command contract")
    print(f"Valid: {summary['valid']}")
    print(f"Contract ready: {summary['command_contract_ready']}")
    print(f"Runtime capture commands ready: {summary['runtime_capture_command_ready']}")
    print(f"Request: {summary.get('request_name')}")
    print(f"Planned captures: {summary['planned_capture_count']}")
    print(f"Runtime command options: {summary['runtime_command_option_count']}")
    print(f"Missing runtime commands: {summary['missing_runtime_command_count']}")
    binding_summary = summary.get("launch_card_binding_summary") if isinstance(summary.get("launch_card_binding_summary"), dict) else {}
    if binding_summary:
        print(
            "Launch-card binding intake: "
            f"ready={binding_summary.get('binding_ready')} "
            f"valid={binding_summary.get('valid')} "
            f"bound={binding_summary.get('bound_task_count')} "
            f"command_options={binding_summary.get('command_option_count')} "
            f"missing={binding_summary.get('missing_runtime_command_count')}"
        )
    for task in summary.get("capture_tasks", []):
        print(
            "  - "
            f"{task.get('artifact_id')}: kind={task.get('capture_kind')} "
            f"command_ready={task.get('command_binding_ready')} "
            f"artifact={task.get('artifact_path')}"
        )
    print("Safety contract:")
    for item in summary["safety_contract"]:
        print(f"  - {item}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request_path", nargs="?", type=Path, default=DEFAULT_REQUEST_PATH)
    parser.add_argument("--model-plane-manifest", type=Path)
    parser.add_argument("--launch-card", type=Path, help="validate a filled launch-card JSON without executing it")
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    parser.add_argument("--output-md", type=Path, help="write a Markdown command-contract report")
    parser.add_argument("--output-launch-card", type=Path, help="write a planned-only launch-card JSON template")
    return parser


def plan_paths(request_path: Path, *, model_plane_manifest_path: Path | None = None, launch_card_path: Path | None = None) -> tuple[int, JSONDict | None, str | None]:
    try:
        summary = build_summary(request_path, model_plane_manifest_path=model_plane_manifest_path, launch_card_path=launch_card_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not build Phase 3 runtime-capture command contract: {exc}"
    return (0 if summary["valid"] else 2), summary, None


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary, error_message = plan_paths(args.request_path, model_plane_manifest_path=args.model_plane_manifest, launch_card_path=args.launch_card)
    if error_message:
        print(error_message, file=sys.stderr)
        return status
    assert summary is not None
    if args.output_md:
        write_markdown_report(summary, args.output_md)
    if args.output_launch_card:
        write_launch_card_template(summary, args.output_launch_card)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_human_summary(summary)
    return status


if __name__ == "__main__":
    raise SystemExit(main())