#!/usr/bin/env python3
"""Plan the real-evidence captures needed to finish Phase 3.

This planner turns the Phase 3 evidence packet's remaining gaps into concrete
artifact requests and validator commands. It does not launch runtimes, download
models, read tensor values, mutate residency, inspect secrets, or send prompt
traffic.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import plan_phase3_evidence_packet
import plan_phase3_live_capability_proof

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_PATH = plan_phase3_evidence_packet.DEFAULT_TRACE_PATH
DEFAULT_INVENTORY_PATH = plan_phase3_evidence_packet.DEFAULT_INVENTORY_PATH
DEFAULT_POLICIES_PATH = plan_phase3_evidence_packet.DEFAULT_POLICIES_PATH
DEFAULT_MANAGED_PLAN_PATH = plan_phase3_evidence_packet.DEFAULT_MANAGED_PLAN_PATH
DEFAULT_OUTPUT_DIR = ROOT / "memory-moe-mvp" / "phase3-real-evidence"
SUPPORTED_SCHEMA_VERSION = "moe-phase3-real-evidence-capture-plan-v1"
SUPPORTED_SOURCE_FORMATS = {"gguf", "safetensors"}
NON_BLOCKING_PHASE3_STATUSES = {"already_satisfied", "ready_metadata_only", "future_phase_4"}

JSONDict = dict[str, Any]


def display_path(path: Path | str) -> str:
    if isinstance(path, str):
        return path
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def uv_command(*parts: Path | str) -> list[str]:
    return [
        "uv",
        "run",
        "--managed-python",
        "--python",
        "3.13",
        *[display_path(part) if isinstance(part, Path) else part for part in parts],
    ]


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._-") or "real_model"


def infer_source_format(model_path: Path | None, source_format: str | None) -> str | None:
    if source_format:
        return source_format
    if model_path is None:
        return None
    suffix = model_path.suffix.lower()
    if suffix == ".gguf":
        return "gguf"
    if suffix == ".safetensors":
        return "safetensors"
    return None


def planned_model_id(model_path: Path | None, model_id: str | None) -> str | None:
    if model_id:
        return model_id
    if model_path is not None:
        return model_path.name
    return None


def planned_inventory_path(output_dir: Path, model_path: Path | None, model_id: str | None) -> Path:
    label = planned_model_id(model_path, model_id) or "real_model"
    return output_dir / f"{slugify(label)}.expert_inventory.json"


def evidence_item(packet: JSONDict, item_id: str) -> JSONDict:
    for item in packet.get("evidence_items", []):
        if isinstance(item, dict) and item.get("id") == item_id:
            return item
    return {}


def planned_trace_path(trace_path: Path, packet: JSONDict, output_dir: Path) -> Path:
    pairing = evidence_item(packet, "real_model_trace_inventory_pairing")
    details = pairing.get("details", {}) if pairing else {}
    if details.get("fixture_only_pair") is True:
        return output_dir / "router_trace.jsonl"
    return trace_path


def no_go_reason_ids(packet: JSONDict) -> set[str]:
    reasons = packet.get("remaining_gaps", {}).get("no_go_reasons", [])
    if not isinstance(reasons, list):
        return set()
    return {
        str(reason["id"])
        for reason in reasons
        if isinstance(reason, dict) and isinstance(reason.get("id"), str)
    }


def request_counts(requests: list[JSONDict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for request in requests:
        status = str(request.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def scanner_commands(
    *,
    model_path: Path | None,
    model_id: str | None,
    source_format: str | None,
    inventory_path: Path,
) -> list[JSONDict]:
    path_arg = display_path(model_path) if model_path is not None else "<model_path>"
    model_id_arg = model_id or "<model_id>"
    output_arg = display_path(inventory_path)
    formats = [source_format] if source_format else ["gguf", "safetensors"]
    commands: list[JSONDict] = []

    for fmt in formats:
        if fmt == "gguf":
            script = "scripts/scan_gguf_expert_inventory.py"
            optional_inputs = [
                "experts_per_layer when GGUF metadata does not expose expert count for stacked tensors"
            ]
        else:
            script = "scripts/scan_safetensors_expert_inventory.py"
            optional_inputs = [
                "model directory is preferred when a safetensors shard index is available"
            ]
        commands.append(
            {
                "source_format": fmt,
                "command": uv_command(
                    script,
                    path_arg,
                    "--model-id",
                    model_id_arg,
                    "--output",
                    output_arg,
                    "--json",
                ),
                "optional_inputs": optional_inputs,
            }
        )
    return commands


def bundle_builder_commands(
    *,
    trace_path: Path,
    trace_receipt_path: Path | None,
    inventory_path: Path,
    policies_path: Path,
    managed_plan_path: Path,
    fallback_artifact_path: Path | None,
    model_id: str | None,
    source_format: str | None,
    output_bundle_path: Path,
) -> list[JSONDict]:
    command = uv_command(
        "scripts/build_phase3_real_evidence_bundle.py",
        "--trace-path",
        display_path(trace_path),
        "--inventory-path",
        display_path(inventory_path),
        "--policies-path",
        display_path(policies_path),
        "--managed-plan-path",
        display_path(managed_plan_path),
        "--model-id",
        model_id or "<model_id>",
        "--source-format",
        source_format or "<source_format>",
        "--backend-family",
        "<backend_family>",
        "--prompt-family",
        "<prompt_family>",
    )
    notes = [
        "uses selected local artifact paths only",
        "does not run models or prompts",
        "keeps approval flags false unless matching captures were explicitly approved",
    ]
    if trace_receipt_path is not None:
        command.extend(["--trace-receipt-path", display_path(trace_receipt_path)])
    else:
        notes.append("rerun with --trace-receipt-path after candidate trace capture receipt is filled")
    if fallback_artifact_path is not None:
        command.extend(["--fallback-artifact-path", display_path(fallback_artifact_path)])
    else:
        notes.append("rerun with --fallback-artifact-path after dense fallback comparison is captured")
    command.extend(["--output", display_path(output_bundle_path), "--json"])
    return [
        {
            "command_class": "local_phase3_bundle_builder",
            "command": command,
            "notes": notes,
        }
    ]

def artifact_request(
    request_id: str,
    *,
    status: str,
    command_class: str,
    summary: str,
    approval_required: bool,
    metadata_only: bool,
    required_inputs: list[str],
    planned_outputs: list[JSONDict],
    validator_commands: list[list[str]] | None = None,
    command_options: list[JSONDict] | None = None,
    details: JSONDict | None = None,
) -> JSONDict:
    result: JSONDict = {
        "id": request_id,
        "status": status,
        "command_class": command_class,
        "summary": summary,
        "approval_required": approval_required,
        "metadata_only": metadata_only,
        "required_inputs": required_inputs,
        "planned_outputs": planned_outputs,
        "validator_commands": validator_commands or [],
    }
    if command_options is not None:
        result["command_options"] = command_options
    if details is not None:
        result["details"] = details
    return result


def build_capture_plan(
    trace_path: Path,
    inventory_path: Path,
    policies_path: Path,
    managed_plan_path: Path,
    *,
    fallback_artifact_path: Path | None = None,
    policy_candidate_trace_receipt_path: Path | None = None,
    live_proof_artifact_path: Path | None = None,
    model_path: Path | None = None,
    model_id: str | None = None,
    source_format: str | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> JSONDict:
    resolved_source_format = infer_source_format(model_path, source_format)
    resolved_model_id = planned_model_id(model_path, model_id)
    output_inventory_path = planned_inventory_path(output_dir, model_path, model_id)
    output_trace_path = output_dir / "router_trace.jsonl"
    output_trace_receipt_path = policy_candidate_trace_receipt_path or output_dir / "router_trace.capture-receipt.json"
    output_fallback_path = fallback_artifact_path or output_dir / "dense_fallback_comparison.json"
    output_live_proof_path = live_proof_artifact_path or output_dir / "live_capability_proof.json"
    output_bundle_path = output_dir / "phase3_real_evidence_bundle.json"

    live_proof = plan_phase3_live_capability_proof.build_summary(live_proof_artifact_path)
    live_proof_ready = live_proof.get("proof_ready") is True
    packet = plan_phase3_evidence_packet.build_packet_summary(
        trace_path,
        inventory_path,
        policies_path,
        managed_plan_path,
        fallback_artifact_path=fallback_artifact_path,
        policy_candidate_trace_receipt_path=policy_candidate_trace_receipt_path,
        include_repo_gates=False,
    )
    reasons = no_go_reason_ids(packet)
    gaps = packet.get("remaining_gaps", {})
    real_model_pair_ready = gaps.get("real_model_pair_ready") is True
    fallback_ready = gaps.get("dense_fallback_comparison_ready") is True
    ready_for_live_spike = gaps.get("ready_for_live_spike") is True
    trace_for_pairing = planned_trace_path(trace_path, packet, output_dir)
    inventory_for_pairing = inventory_path if real_model_pair_ready else output_inventory_path

    trace_status = "already_satisfied" if real_model_pair_ready else "needed"
    scanner_inputs_ready = bool(model_path and resolved_model_id and resolved_source_format)
    scanner_status = "already_satisfied"
    if not real_model_pair_ready:
        scanner_status = "ready_metadata_only" if scanner_inputs_ready else "needed"
    pairing_status = "already_satisfied" if real_model_pair_ready else "blocked_by_required_artifacts"
    fallback_status = "already_satisfied" if fallback_ready else "needed"
    if not fallback_ready and fallback_artifact_path is not None:
        fallback_status = "needs_valid_artifact"
    replay_status = "already_satisfied"
    if "no_replay_policy_candidate" in reasons:
        replay_status = "needs_real_trace_window"
    if not real_model_pair_ready:
        replay_status = "blocked_by_real_model_pairing"
    live_status = "already_satisfied" if live_proof_ready else "future_phase_4"
    if not live_proof_ready and live_proof_artifact_path is not None:
        live_status = "needs_valid_artifact"

    requests = [
        artifact_request(
            "real_semantic_trace_artifact",
            status=trace_status,
            command_class="approved_semantic_router_trace_capture_or_existing_trace",
            summary=(
                "Provide a non-fixture semantic routing trace from the same model that will be scanned "
                "for inventory evidence."
            ),
            approval_required=not real_model_pair_ready,
            metadata_only=False,
            required_inputs=[
                "approved hookable llama.cpp, PyTorch, or equivalent MoE runtime",
                "prompt set and model label recorded in the trace artifact",
                "router events containing selected_experts for at least one routed layer",
            ],
            planned_outputs=[
                {
                    "path": display_path(output_trace_path if not real_model_pair_ready else trace_path),
                    "schema": "memory-moe-bridge-v1 JSONL router trace",
                }
            ],
            validator_commands=[
                uv_command(
                    "scripts/validate_llama_cpp_router_trace.py",
                    display_path(output_trace_path if not real_model_pair_ready else trace_path),
                    "--require-kind",
                    "selected_experts",
                    "--require-kind",
                    "selected_weights",
                    "--require-kind",
                    "selected_weights_norm",
                    "--json",
                )
            ],
            details={
                "current_trace_path": display_path(trace_path),
                "current_pairing_status": evidence_item(packet, "real_model_trace_inventory_pairing").get("status"),
            },
        ),
        artifact_request(
            "scanner_derived_inventory",
            status=scanner_status,
            command_class="metadata_only_inventory_scan",
            summary="Scan GGUF or safetensors metadata into an expert inventory manifest without loading tensor values.",
            approval_required=False,
            metadata_only=True,
            required_inputs=[
                "model_path",
                "model_id matching the semantic trace model name or source file name",
                "source_format gguf or safetensors",
            ],
            planned_outputs=[
                {
                    "path": display_path(inventory_for_pairing),
                    "schema": "expert inventory manifest",
                }
            ],
            command_options=scanner_commands(
                model_path=model_path,
                model_id=resolved_model_id,
                source_format=resolved_source_format,
                inventory_path=inventory_for_pairing,
            ),
            details={
                "input_status": {
                    "model_path": model_path is not None,
                    "model_id": resolved_model_id is not None,
                    "source_format": resolved_source_format is not None,
                },
                "inferred_source_format": resolved_source_format,
            },
        ),
        artifact_request(
            "real_trace_inventory_pairing",
            status=pairing_status,
            command_class="offline_pairing_validation",
            summary="Validate that the semantic trace and scanner-derived inventory describe the same real model.",
            approval_required=False,
            metadata_only=True,
            required_inputs=[
                "real semantic trace artifact",
                "scanner-derived inventory manifest for the same model",
                "complete component coverage for routed experts",
            ],
            planned_outputs=[
                {
                    "path": "stdout",
                    "schema": "real-model trace/inventory pairing summary",
                }
            ],
            validator_commands=[
                uv_command(
                    "scripts/plan_trace_inventory_replay.py",
                    display_path(trace_for_pairing),
                    display_path(inventory_for_pairing),
                    "--json",
                ),
                uv_command(
                    "scripts/plan_real_model_trace_inventory_pairing.py",
                    display_path(trace_for_pairing),
                    display_path(inventory_for_pairing),
                    "--require-real-model",
                    "--json",
                ),
                uv_command(
                    "scripts/plan_expert_store_layout.py",
                    display_path(inventory_for_pairing),
                    "--json",
                ),
            ],
        ),
        artifact_request(
            "dense_fallback_comparison_artifact",
            status=fallback_status,
            command_class="approved_dense_or_full_runtime_baseline_run",
            summary=(
                "Capture dense or full-runtime output for the same prompt set, then validate the "
                "comparison artifact."
            ),
            approval_required=not fallback_ready,
            metadata_only=False,
            required_inputs=[
                "same model target and prompt family as the managed replay trace",
                "managed output artifact",
                "dense or full-runtime baseline output artifact",
                "quality_delta_label for each prompt comparison",
            ],
            planned_outputs=[
                {
                    "path": display_path(output_fallback_path),
                    "schema": "moe-dense-fallback-comparison-v1",
                }
            ],
            validator_commands=[
                uv_command(
                    "scripts/plan_dense_fallback_comparison.py",
                    display_path(output_fallback_path),
                    "--json",
                )
            ],
            command_options=[
                {
                    "command_class": "local_comparison_artifact_builder",
                    "command": uv_command(
                        "scripts/build_dense_fallback_comparison.py",
                        "--managed-artifact",
                        "<managed_output_artifact>",
                        "--dense-artifact",
                        "<dense_or_full_runtime_output_artifact>",
                        "--model-id",
                        resolved_model_id or "<model_id>",
                        "--prompt-family",
                        "<prompt_family>",
                        "--managed-policy-id",
                        "<managed_policy_id>",
                        "--output",
                        display_path(output_fallback_path),
                        "--json",
                    ),
                    "notes": [
                        "uses already-saved outputs only",
                        "keeps quality_delta_label unknown unless exact comparable text is available or reviewed",
                    ],
                }
            ],
            details={
                "current_fallback_status": evidence_item(packet, "dense_fallback_comparison").get("status"),
                "current_fallback_artifact_path": display_path(output_fallback_path),
            },
        ),
        artifact_request(
            "policy_replay_candidate_window",
            status=replay_status,
            command_class="offline_policy_replay_over_real_trace_inventory",
            summary=(
                "Replay the baseline residency policies over real trace and inventory artifacts, "
                "then keep rejected policies explicit."
            ),
            approval_required=False,
            metadata_only=True,
            required_inputs=[
                "real semantic trace artifact",
                "scanner-derived inventory manifest",
                "baseline replay policies JSON",
            ],
            planned_outputs=[
                {
                    "path": "stdout",
                    "schema": "baseline policy replay summary",
                }
            ],
            validator_commands=[
                uv_command(
                    "scripts/plan_baseline_policy_replay.py",
                    display_path(trace_for_pairing),
                    display_path(inventory_for_pairing),
                    display_path(policies_path),
                    "--json",
                )
            ],
            details={
                "current_no_go_reason_present": "no_replay_policy_candidate" in reasons,
            },
        ),
        artifact_request(
            "phase3_real_evidence_bundle_manifest",
            status="already_satisfied" if packet.get("phase3_complete") is True else "needed",
            command_class="offline_phase3_bundle_manifest_validation",
            summary="Bind trace, inventory, policies, managed-loading, and fallback comparison artifacts into one Phase 3 handoff bundle.",
            approval_required=False,
            metadata_only=True,
            required_inputs=[
                "real semantic trace artifact",
                "scanner-derived inventory manifest",
                "baseline replay policies JSON",
                "managed expert loading plan",
                "dense fallback comparison artifact when available",
                "candidate trace capture receipt when a policy candidate is produced",
            ],
            planned_outputs=[
                {
                    "path": display_path(output_bundle_path),
                    "schema": "moe-phase3-real-evidence-bundle-v1",
                }
            ],
            validator_commands=[
                uv_command(
                    "scripts/plan_phase3_real_evidence_bundle.py",
                    display_path(output_bundle_path),
                    "--json",
                )
            ],
            command_options=bundle_builder_commands(
                trace_path=trace_for_pairing,
                trace_receipt_path=policy_candidate_trace_receipt_path,
                inventory_path=inventory_for_pairing,
                policies_path=policies_path,
                managed_plan_path=managed_plan_path,
                fallback_artifact_path=fallback_artifact_path,
                model_id=resolved_model_id,
                source_format=resolved_source_format,
                output_bundle_path=output_bundle_path,
            ),
            details={
                "fixture_template": "memory-moe-mvp/data/phase3_real_evidence_bundle.fixture.json",
                "current_packet_ready": packet.get("packet_ready"),
                "current_phase3_complete": packet.get("phase3_complete"),
                "current_trace_receipt_path": display_path(output_trace_receipt_path),
                "current_trace_receipt_ready": gaps.get("policy_candidate_trace_receipt_ready"),
            },
        ),
        artifact_request(
            "live_residency_capability_proof",
            status=live_status,
            command_class="future_backend_adapter_or_actuator_spike",
            summary=(
                "Prove residency observation, residency control, cleanup, fallback, and artifact export "
                "before any live expert-paging claim."
            ),
            approval_required=not live_proof_ready,
            metadata_only=False,
            required_inputs=[
                "backend control point that can observe expert residency",
                "backend control point that can preload, pin, evict, or demote expert residency",
                "cleanup and rollback proof",
                "dense fallback still available under failure",
                "run-scoped artifact export",
            ],
            planned_outputs=[
                {
                    "path": display_path(output_live_proof_path),
                    "schema": plan_phase3_live_capability_proof.SUPPORTED_SCHEMA_VERSION,
                }
            ],
            validator_commands=[
                uv_command("scripts/plan_phase3_live_capability_proof.py", display_path(output_live_proof_path), "--json"),
                uv_command("scripts/plan_managed_expert_loading.py", display_path(managed_plan_path), "--json"),
            ],
            details={
                "current_live_proof_ready": live_proof_ready,
                "current_live_proof_artifact_path": display_path(output_live_proof_path),
                "current_live_proof_blockers": live_proof.get("blockers", []),
                "unavailable_live_capabilities": [
                    reason
                    for reason in packet.get("remaining_gaps", {}).get("no_go_reasons", [])
                    if isinstance(reason, dict)
                    and reason.get("id") == "live_actuator_capabilities_unavailable"
                ],
            },
        ),
    ]

    blocking_request_ids = [
        request["id"]
        for request in requests
        if request["status"] not in NON_BLOCKING_PHASE3_STATUSES
    ]
    future_request_ids = [
        request["id"]
        for request in requests
        if request["status"] == "future_phase_4"
    ]
    metadata_only_steps_ready = any(
        request["id"] == "scanner_derived_inventory" and request["status"] == "ready_metadata_only"
        for request in requests
    )
    prompt_or_runtime_approval_required = any(
        bool(request.get("approval_required")) and not bool(request.get("metadata_only"))
        for request in requests
        if request["status"] not in {"already_satisfied", "future_phase_4"}
    )

    return {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "mode": "phase3_real_evidence_capture_plan",
        "valid": packet.get("valid") is True,
        "errors": packet.get("errors", []),
        "packet_decision": packet.get("decision"),
        "packet_ready": packet.get("packet_ready"),
        "phase3_complete": packet.get("phase3_complete"),
        "ready_to_execute_without_approval": False if prompt_or_runtime_approval_required else metadata_only_steps_ready,
        "metadata_only_steps_ready": metadata_only_steps_ready,
        "prompt_or_runtime_approval_required": prompt_or_runtime_approval_required,
        "trace_path": display_path(trace_path),
        "inventory_path": display_path(inventory_path),
        "policies_path": display_path(policies_path),
        "managed_plan_path": display_path(managed_plan_path),
        "fallback_artifact_path": display_path(fallback_artifact_path) if fallback_artifact_path is not None else None,
        "policy_candidate_trace_receipt_path": display_path(policy_candidate_trace_receipt_path) if policy_candidate_trace_receipt_path is not None else None,
        "planned_trace_receipt_path": display_path(output_trace_receipt_path),
        "live_proof_artifact_path": display_path(live_proof_artifact_path) if live_proof_artifact_path is not None else None,
        "model_path": display_path(model_path) if model_path is not None else None,
        "model_id": resolved_model_id,
        "source_format": resolved_source_format,
        "output_dir": display_path(output_dir),
        "planned_bundle_path": display_path(output_bundle_path),
        "request_status_counts": request_counts(requests),
        "blocking_request_ids": blocking_request_ids,
        "future_request_ids": future_request_ids,
        "artifact_requests": requests,
        "safety_contract": [
            "planner emits artifact requests and validator commands only",
            "planner does not launch model servers",
            "planner does not download models",
            "planner does not read tensor values",
            "planner does not inspect private tokens",
            "planner does not mutate runtime residency",
            "planner does not send prompt traffic",
            "planner does not claim live expert paging",
        ],
        "next_actions": [
            "Use metadata-only scanner commands once a local GGUF or safetensors target path is selected.",
            "Pair scanner-derived inventory with a non-fixture semantic trace using --require-real-model.",
            "Capture dense/full-runtime comparison output only after explicit runtime and prompt approval.",
            "Keep live residency control in Phase 4 until a backend control point and cleanup proof exist.",
        ],
    }


def command_to_text(command: Any) -> str:
    if isinstance(command, list):
        return " ".join(str(part) for part in command)
    return str(command)


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def planned_output_text(request: JSONDict) -> str:
    outputs = request.get("planned_outputs") if isinstance(request.get("planned_outputs"), list) else []
    values = []
    for output in outputs:
        if isinstance(output, dict):
            path = output.get("path") or "missing"
            schema = output.get("schema") or "unknown"
            values.append(f"{path} ({schema})")
        else:
            values.append(str(output))
    return "; ".join(values) if values else "none"


def format_markdown_report(summary: JSONDict) -> str:
    counts = summary.get("request_status_counts") if isinstance(summary.get("request_status_counts"), dict) else {}
    lines = [
        "# Phase 3 Real-Evidence Capture Plan",
        "",
        f"- Valid: `{summary.get('valid')}`",
        f"- Packet decision: `{markdown_escape(summary.get('packet_decision'))}`",
        f"- Phase 3 complete: `{summary.get('phase3_complete')}`",
        f"- Metadata-only steps ready: `{summary.get('metadata_only_steps_ready')}`",
        f"- Prompt/runtime approval required: `{summary.get('prompt_or_runtime_approval_required')}`",
        f"- Ready to execute without approval: `{summary.get('ready_to_execute_without_approval')}`",
        f"- Output directory: `{markdown_escape(summary.get('output_dir'))}`",
        f"- Planned bundle: `{markdown_escape(summary.get('planned_bundle_path'))}`",
        "",
        "## Request Status Counts",
        "",
    ]
    if counts:
        for status, count in counts.items():
            lines.append(f"- `{markdown_escape(status)}`: `{count}`")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Artifact Requests",
            "",
            "| Request | Status | Approval | Metadata Only | Planned Outputs | Summary |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for request in summary.get("artifact_requests", []):
        if not isinstance(request, dict):
            continue
        approval = "required" if request.get("approval_required") is True else "not required"
        metadata_only = "yes" if request.get("metadata_only") is True else "no"
        lines.append(
            "| "
            + " | ".join(
                markdown_escape(value)
                for value in (
                    request.get("id") or "unknown",
                    request.get("status") or "unknown",
                    approval,
                    metadata_only,
                    planned_output_text(request),
                    request.get("summary") or "",
                )
            )
            + " |"
        )
    blocking = summary.get("blocking_request_ids") if isinstance(summary.get("blocking_request_ids"), list) else []
    if blocking:
        lines.extend(["", "## Phase 3 Blocking Requests", ""])
        for request_id in blocking:
            lines.append(f"- `{markdown_escape(request_id)}`")
    future = summary.get("future_request_ids") if isinstance(summary.get("future_request_ids"), list) else []
    if future:
        lines.extend(["", "## Future Requests", ""])
        for request_id in future:
            lines.append(f"- `{markdown_escape(request_id)}`")
    command_requests = [
        request
        for request in summary.get("artifact_requests", [])
        if isinstance(request, dict)
        and (request.get("command_options") or request.get("validator_commands"))
    ]
    if command_requests:
        lines.extend(["", "## Commands", ""])
        for request in command_requests:
            lines.append(f"### {markdown_escape(request.get('id') or 'unknown')}")
            for option in request.get("command_options", []) or []:
                if not isinstance(option, dict):
                    continue
                lines.append(f"- `{markdown_escape(option.get('command_class') or option.get('source_format') or 'command_option')}`")
                lines.append("```sh")
                lines.append(command_to_text(option.get("command")))
                lines.append("```")
            validators = request.get("validator_commands") if isinstance(request.get("validator_commands"), list) else []
            for command in validators:
                lines.append("- `validator`")
                lines.append("```sh")
                lines.append(command_to_text(command))
                lines.append("```")
    next_actions = summary.get("next_actions") if isinstance(summary.get("next_actions"), list) else []
    if next_actions:
        lines.extend(["", "## Next Actions", ""])
        for action in next_actions:
            lines.append(f"- {markdown_escape(action)}")
    lines.extend(["", "## Safety Contract", ""])
    for item in summary.get("safety_contract", []):
        lines.append(f"- {markdown_escape(item)}")
    return "\n".join(lines) + "\n"


def write_markdown_report(summary: JSONDict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_markdown_report(summary), encoding="utf-8")

def print_human_summary(summary: JSONDict) -> None:
    print("MoE Run Anyway Phase 3 real-evidence capture plan")
    print(f"Valid: {summary['valid']}")
    print(f"Packet decision: {summary['packet_decision']}")
    print(f"Phase 3 complete: {summary['phase3_complete']}")
    print(f"Metadata-only steps ready: {summary['metadata_only_steps_ready']}")
    print(f"Prompt/runtime approval required: {summary['prompt_or_runtime_approval_required']}")
    if summary["errors"]:
        print("Errors:")
        for error in summary["errors"]:
            print(f"  - {error}")
    print("Artifact requests:")
    for request in summary["artifact_requests"]:
        approval = "approval required" if request["approval_required"] else "no runtime approval"
        print(f"  - {request['id']}: {request['status']} ({approval})")
    if summary["blocking_request_ids"]:
        print(f"Phase 3 blockers: {', '.join(summary['blocking_request_ids'])}")
    if summary["future_request_ids"]:
        print(f"Future requests: {', '.join(summary['future_request_ids'])}")
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
    parser.add_argument("--policy-candidate-trace-receipt-path", type=Path, default=None)
    parser.add_argument("--live-proof-artifact-path", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--source-format", choices=sorted(SUPPORTED_SOURCE_FORMATS), default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    parser.add_argument("--output-md", type=Path, help="write a Markdown real-evidence capture handoff report")
    return parser


def plan_paths(
    trace_path: Path,
    inventory_path: Path,
    policies_path: Path,
    managed_plan_path: Path,
    *,
    fallback_artifact_path: Path | None = None,
    policy_candidate_trace_receipt_path: Path | None = None,
    live_proof_artifact_path: Path | None = None,
    model_path: Path | None = None,
    model_id: str | None = None,
    source_format: str | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[int, JSONDict | None, str | None]:
    try:
        summary = build_capture_plan(
            trace_path,
            inventory_path,
            policies_path,
            managed_plan_path,
            fallback_artifact_path=fallback_artifact_path,
            policy_candidate_trace_receipt_path=policy_candidate_trace_receipt_path,
            live_proof_artifact_path=live_proof_artifact_path,
            model_path=model_path,
            model_id=model_id,
            source_format=source_format,
            output_dir=output_dir,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not build Phase 3 real-evidence capture plan: {exc}"
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
        policy_candidate_trace_receipt_path=args.policy_candidate_trace_receipt_path,
        live_proof_artifact_path=args.live_proof_artifact_path,
        model_path=args.model_path,
        model_id=args.model_id,
        source_format=args.source_format,
        output_dir=args.output_dir,
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
        print_human_summary(summary)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
