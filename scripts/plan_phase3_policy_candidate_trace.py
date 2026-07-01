#!/usr/bin/env python3
"""Plan trace capture to clear the Phase 3 policy-candidate blocker.

This planner turns the current `no_replay_policy_candidate` gate into concrete
offline artifact requests. It reads a Phase 3 real-evidence bundle, replays the
current trace/inventory/policy artifacts, and optionally evaluates a newly saved
candidate trace. It does not launch runtimes, run Docker, download models,
inspect secrets, mutate residency, or send prompt traffic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import plan_baseline_policy_replay
import plan_phase3_dense_fallback_capture
import plan_phase3_real_evidence_bundle
import phase3_trace_receipts


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_PATH = ROOT / "memory-moe-mvp" / "phase3-real-evidence" / "pc_mixtral_phase3_real_evidence_bundle.json"
SUPPORTED_SCHEMA_VERSION = "moe-phase3-policy-candidate-trace-plan-v1"

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
    if path.is_absolute():
        return path
    return ROOT / path


def load_json(path: Path) -> JSONDict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{display_path(path) or path} must be a JSON object")
    return data


def uv_command(*args: str) -> list[str]:
    return ["uv", "run", "--managed-python", "--python", "3.13", *args]


def canonical_prompt_set_path(bundle_path: Path) -> Path:
    return bundle_path.with_name(f"{bundle_path.stem}.prompt-set.json")


def default_candidate_trace_receipt_path(candidate_trace_path: Path) -> Path:
    return candidate_trace_path.with_name(f"{candidate_trace_path.stem}.capture-receipt.json")


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


def artifact_path(manifest: JSONDict, key: str) -> Path | None:
    paths = manifest.get("artifact_paths")
    if not isinstance(paths, dict):
        return None
    return resolve_repo_path(paths.get(key))


def artifact_request(
    request_id: str,
    *,
    status: str,
    approval_required: bool,
    description: str,
    details: JSONDict | None = None,
) -> JSONDict:
    item: JSONDict = {
        "id": request_id,
        "status": status,
        "approval_required": approval_required,
        "description": description,
    }
    if details is not None:
        item["details"] = details
    return item


def replay_policy_diagnostics(replay_summary: JSONDict | None) -> JSONDict:
    if replay_summary is None:
        return {
            "valid": False,
            "policy_candidate_ready": False,
            "candidate_policy_ids": [],
            "reuse_distance_observations": 0,
            "blockers": [{"id": "policy_replay_missing", "description": "Policy replay summary is unavailable."}],
            "rejected_policy_reasons_by_policy": {},
        }
    diagnostics = replay_summary.get("policy_candidate_diagnostics", {})
    phase_gate = replay_summary.get("phase_3_gate", {})
    reuse_distance = replay_summary.get("reuse_distance", {})
    blockers = diagnostics.get("blockers", []) if isinstance(diagnostics, dict) else []
    return {
        "valid": replay_summary.get("valid") is True,
        "policy_candidate_ready": phase_gate.get("policy_candidate_ready") is True,
        "candidate_policy_ids": replay_summary.get("candidate_policy_ids", []),
        "reuse_distance_observations": int(reuse_distance.get("observations", 0) or 0),
        "reuse_distance": reuse_distance,
        "blockers": blockers,
        "rejected_policy_reasons_by_policy": diagnostics.get("rejected_policy_reasons_by_policy", {}) if isinstance(diagnostics, dict) else {},
        "joined_route_count": replay_summary.get("joined_route_count", 0),
        "unique_expert_count": replay_summary.get("unique_expert_count", 0),
        "real_model_pair_ready": phase_gate.get("real_model_pair_ready") is True,
    }


def build_replay(trace_path: Path, inventory_path: Path, policies_path: Path) -> tuple[JSONDict | None, list[str]]:
    try:
        return plan_baseline_policy_replay.build_replay_summary(trace_path, inventory_path, policies_path), []
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return None, [str(exc)]


def summarize_trace_receipt(
    trace_receipt_path: Path | None,
    *,
    candidate_trace_path: Path | None,
    expected_prompt_set_path: Path | None,
    expected_request_path: Path | None = None,
    expected_model_id: str | None = None,
    expected_backend_family: str | None = None,
    expected_prompt_family: str | None = None,
) -> JSONDict:
    if trace_receipt_path is None:
        return {
            "path": None,
            "exists": False,
            "valid": False,
            "receipt_ready": False,
            "ready": False,
            "candidate_trace_path_matches": False,
            "prompt_set_path_matches": False,
            "request_path_matches": expected_request_path is None,
            "errors": [],
            "blockers": ["candidate_trace_capture_receipt_missing"],
        }
    if not trace_receipt_path.exists():
        return {
            "path": display_path(trace_receipt_path),
            "exists": False,
            "valid": False,
            "receipt_ready": False,
            "ready": False,
            "candidate_trace_path_matches": False,
            "prompt_set_path_matches": False,
            "request_path_matches": expected_request_path is None,
            "errors": [],
            "blockers": ["candidate_trace_capture_receipt_missing"],
        }

    receipt = load_json(trace_receipt_path)
    validation = phase3_trace_receipts.validate_trace_receipt_source_binding(
        receipt,
        repo_root=ROOT,
        expected_source_request_path=expected_request_path,
        expected_source_prompt_set_path=expected_prompt_set_path,
        expected_candidate_trace_path=candidate_trace_path,
    )
    errors = list(validation["errors"])
    blockers: list[str] = []
    receipt_ready = validation["receipt_ready"] is True
    candidate_trace_path_matches = validation["candidate_trace_path_matches"] is True
    prompt_set_path_matches = validation["source_prompt_set_path_matches"] is True
    request_path_matches = validation["source_request_path_matches"] is True

    if validation["shape_valid"] and not receipt_ready:
        blockers.append("candidate_trace_capture_receipt_not_ready")
    if validation["shape_valid"] is not True:
        blockers.append("candidate_trace_capture_receipt_invalid")

    if receipt_ready:
        if expected_model_id is not None and receipt.get("model_id") != expected_model_id:
            errors.append("trace_capture_receipt.model_id must match bundle model_id")
        if expected_backend_family is not None and receipt.get("runtime_backend") != expected_backend_family:
            errors.append("trace_capture_receipt.runtime_backend must match bundle backend_family")
        if expected_prompt_family is not None and receipt.get("prompt_family") != expected_prompt_family:
            errors.append("trace_capture_receipt.prompt_family must match bundle prompt_family")

    ready = receipt_ready and not errors and candidate_trace_path_matches and prompt_set_path_matches and request_path_matches
    return {
        "path": display_path(trace_receipt_path),
        "exists": True,
        "valid": not errors and (receipt_ready or bool(blockers)),
        "receipt_ready": receipt_ready,
        "ready": ready,
        "candidate_trace_path_matches": candidate_trace_path_matches,
        "prompt_set_path_matches": prompt_set_path_matches,
        "request_path_matches": request_path_matches,
        "candidate_trace_path_exists": validation.get("candidate_trace_path_exists"),
        "source_prompt_set_path_exists": validation.get("source_prompt_set_path_exists"),
        "source_request_path_exists": validation.get("source_request_path_exists"),
        "errors": errors,
        "blockers": blockers,
    }

def summarize_candidate_trace(
    candidate_trace_path: Path | None,
    *,
    inventory_path: Path | None,
    policies_path: Path | None,
    trace_receipt_path: Path | None,
    expected_prompt_set_path: Path | None,
    expected_request_path: Path | None = None,
    expected_model_id: str | None = None,
    expected_backend_family: str | None = None,
    expected_prompt_family: str | None = None,
) -> JSONDict:
    if candidate_trace_path is None:
        return {
            "path": None,
            "provided": False,
            "exists": False,
            "valid": True,
            "replay_valid": False,
            "policy_candidate_ready": False,
            "candidate_policy_ids": [],
            "reuse_distance_observations": 0,
            "errors": [],
            "capture_receipt": summarize_trace_receipt(
                trace_receipt_path,
                candidate_trace_path=candidate_trace_path,
                expected_prompt_set_path=expected_prompt_set_path,
                expected_request_path=expected_request_path,
                expected_model_id=expected_model_id,
                expected_backend_family=expected_backend_family,
                expected_prompt_family=expected_prompt_family,
            ),
            "blocker": "candidate_trace_artifact_missing",
        }
    if not candidate_trace_path.exists():
        return {
            "path": display_path(candidate_trace_path),
            "provided": True,
            "exists": False,
            "valid": False,
            "replay_valid": False,
            "policy_candidate_ready": False,
            "candidate_policy_ids": [],
            "reuse_distance_observations": 0,
            "errors": [f"candidate trace artifact does not exist: {display_path(candidate_trace_path)}"],
            "capture_receipt": summarize_trace_receipt(
                trace_receipt_path,
                candidate_trace_path=candidate_trace_path,
                expected_prompt_set_path=expected_prompt_set_path,
                expected_request_path=expected_request_path,
                expected_model_id=expected_model_id,
                expected_backend_family=expected_backend_family,
                expected_prompt_family=expected_prompt_family,
            ),
            "blocker": "candidate_trace_artifact_missing",
        }
    errors: list[str] = []
    if inventory_path is None or not inventory_path.exists():
        errors.append("inventory artifact is missing; cannot replay candidate trace")
    if policies_path is None or not policies_path.exists():
        errors.append("policies artifact is missing; cannot replay candidate trace")
    replay_summary: JSONDict | None = None
    replay_errors: list[str] = []
    if not errors and inventory_path is not None and policies_path is not None:
        replay_summary, replay_errors = build_replay(candidate_trace_path, inventory_path, policies_path)
        errors.extend(f"candidate replay: {error}" for error in replay_errors)
    diagnostics = replay_policy_diagnostics(replay_summary)
    receipt_summary = summarize_trace_receipt(
        trace_receipt_path,
        candidate_trace_path=candidate_trace_path,
        expected_prompt_set_path=expected_prompt_set_path,
        expected_request_path=expected_request_path,
        expected_model_id=expected_model_id,
        expected_backend_family=expected_backend_family,
        expected_prompt_family=expected_prompt_family,
    )
    errors.extend(receipt_summary.get("errors", []))
    receipt_blockers = [
        {"id": blocker, "section": "candidate_trace_capture_receipt"}
        for blocker in receipt_summary.get("blockers", [])
        if isinstance(blocker, str)
    ]
    replay_valid = diagnostics["valid"]
    receipt_ready = receipt_summary.get("ready") is True
    candidate_ready = diagnostics["policy_candidate_ready"] and receipt_ready
    candidate_blocker = None
    if not candidate_ready:
        candidate_blocker = (
            "candidate_trace_capture_receipt_not_ready"
            if receipt_ready is not True
            else "candidate_trace_not_policy_candidate_ready"
        )
    return {
        "path": display_path(candidate_trace_path),
        "provided": True,
        "exists": True,
        "valid": not errors and replay_valid and receipt_ready,
        "replay_valid": replay_valid,
        "policy_candidate_ready": candidate_ready,
        "candidate_policy_ids": diagnostics["candidate_policy_ids"],
        "reuse_distance_observations": diagnostics["reuse_distance_observations"],
        "blockers": [*diagnostics["blockers"], *receipt_blockers],
        "errors": errors,
        "capture_receipt": receipt_summary,
        "blocker": candidate_blocker,
    }

def trace_request_status(
    *,
    current_candidate_ready: bool,
    candidate_trace: JSONDict,
    prompt_ready: bool,
    approvals_ready: bool,
) -> str:
    if current_candidate_ready or candidate_trace.get("policy_candidate_ready") is True:
        return "already_satisfied"
    if candidate_trace.get("exists") and candidate_trace.get("valid") is not True:
        return "invalid"
    if candidate_trace.get("exists") and candidate_trace.get("replay_valid") is True:
        return "present_but_not_candidate_ready"
    if not prompt_ready:
        return "blocked_by_prompt_set"
    if not approvals_ready:
        return "approval_required"
    return "needed"


def replay_request_status(*, current_candidate_ready: bool, candidate_trace: JSONDict) -> str:
    if current_candidate_ready or candidate_trace.get("policy_candidate_ready") is True:
        return "already_satisfied"
    if candidate_trace.get("exists") and candidate_trace.get("valid") is not True:
        return "invalid_trace_or_replay"
    if candidate_trace.get("exists") and candidate_trace.get("replay_valid") is True:
        return "no_candidate_found"
    return "blocked_by_candidate_trace"


def build_capture_plan(
    bundle_path: Path,
    *,
    output_dir: Path | None = None,
    candidate_prompt_set_path: Path | None = None,
    candidate_trace_path: Path | None = None,
    candidate_trace_receipt_path: Path | None = None,
    source_request_path: Path | None = None,
    updated_bundle_path: Path | None = None,
    policy_candidate_trace_capture_approved: bool = False,
    runtime_prompt_traffic_approved: bool = False,
) -> JSONDict:
    manifest = plan_phase3_real_evidence_bundle.load_manifest(bundle_path)
    bundle_summary = plan_phase3_real_evidence_bundle.build_summary(bundle_path)
    trace_path = artifact_path(manifest, "trace_path")
    inventory_path = artifact_path(manifest, "inventory_path")
    policies_path = artifact_path(manifest, "policies_path")
    managed_plan_path = artifact_path(manifest, "managed_plan_path")
    fallback_artifact_path = artifact_path(manifest, "fallback_artifact_path")
    live_proof_artifact_path = artifact_path(manifest, "live_proof_artifact_path")

    output_root = output_dir or (ROOT / "memory-moe-mvp" / "phase3-real-evidence" / f"{bundle_path.stem}-policy-candidate")
    shared_prompt_set = canonical_prompt_set_path(bundle_path)
    planned_prompt_set = candidate_prompt_set_path or (shared_prompt_set if shared_prompt_set.exists() else output_root / "candidate-prompt-set.json")
    planned_candidate_trace = candidate_trace_path or (output_root / "candidate-router-events.jsonl")
    planned_candidate_trace_receipt = candidate_trace_receipt_path or default_candidate_trace_receipt_path(planned_candidate_trace)
    planned_updated_bundle = updated_bundle_path or (output_root / f"{bundle_path.stem}.policy-candidate.json")

    prompt_summary_path = candidate_prompt_set_path or (planned_prompt_set if planned_prompt_set.exists() else None)
    prompt_summary = plan_phase3_dense_fallback_capture.summarize_prompt_set(prompt_summary_path)

    current_replay: JSONDict | None = None
    current_replay_errors: list[str] = []
    if trace_path is None or inventory_path is None or policies_path is None:
        current_replay_errors.append("bundle is missing trace, inventory, or policies path")
    elif not trace_path.exists() or not inventory_path.exists() or not policies_path.exists():
        current_replay_errors.append("bundle trace, inventory, or policies path does not exist")
    else:
        current_replay, current_replay_errors = build_replay(trace_path, inventory_path, policies_path)
    current_diagnostics = replay_policy_diagnostics(current_replay)

    candidate_trace_for_replay = candidate_trace_path if candidate_trace_path is not None and candidate_trace_path.exists() else (planned_candidate_trace if planned_candidate_trace.exists() else None)
    candidate_trace_summary = summarize_candidate_trace(
        candidate_trace_for_replay,
        inventory_path=inventory_path,
        policies_path=policies_path,
        trace_receipt_path=planned_candidate_trace_receipt,
        expected_prompt_set_path=prompt_summary_path,
        expected_request_path=source_request_path,
        expected_model_id=str(manifest.get("model_id")) if manifest.get("model_id") is not None else None,
        expected_backend_family=str(manifest.get("backend_family")) if manifest.get("backend_family") is not None else None,
        expected_prompt_family=str(manifest.get("prompt_family")) if manifest.get("prompt_family") is not None else None,
    )

    prompt_ready = prompt_summary.get("ready") is True
    approvals_ready = policy_candidate_trace_capture_approved and runtime_prompt_traffic_approved
    current_candidate_ready = current_diagnostics["policy_candidate_ready"]
    policy_candidate_ready_after_plan = current_candidate_ready or candidate_trace_summary.get("policy_candidate_ready") is True

    candidate_trace_status = trace_request_status(
        current_candidate_ready=current_candidate_ready,
        candidate_trace=candidate_trace_summary,
        prompt_ready=prompt_ready,
        approvals_ready=approvals_ready,
    )
    replay_status = replay_request_status(
        current_candidate_ready=current_candidate_ready,
        candidate_trace=candidate_trace_summary,
    )

    artifact_requests = [
        artifact_request(
            "candidate_prompt_set_artifact",
            status="already_satisfied" if prompt_ready else "needed",
            approval_required=False,
            description="Exact prompt set designed to produce repeated routed expert observations for the same model/backend.",
            details={"path": display_path(planned_prompt_set), "current": prompt_summary},
        ),
        artifact_request(
            "candidate_trace_artifact",
            status=candidate_trace_status,
            approval_required=candidate_trace_status in {"approval_required", "needed"},
            description="Semantic router trace captured from the same model/backend with repeated prompt cases and selected expert ids/weights.",
            details={
                "path": display_path(planned_candidate_trace),
                "capture_receipt_path": display_path(planned_candidate_trace_receipt),
                "current": candidate_trace_summary,
            },
        ),
        artifact_request(
            "candidate_policy_replay",
            status=replay_status,
            approval_required=False,
            description="Offline replay proving at least one managed-loading policy is a candidate, or explaining why no-go remains correct.",
            details={"candidate_trace": candidate_trace_summary, "current_replay": current_diagnostics},
        ),
        artifact_request(
            "phase3_bundle_with_candidate_trace",
            status="ready_to_build" if policy_candidate_ready_after_plan else "blocked_by_candidate_replay",
            approval_required=False,
            description="Updated Phase 3 bundle manifest that points at the policy-candidate trace once replay validates it.",
            details={"path": display_path(planned_updated_bundle)},
        ),
    ]

    replay_trace_for_commands = planned_candidate_trace
    commands = [
        {
            "command_class": "candidate_trace_receipt_validator",
            "requires_runtime": False,
            "command": uv_command(
                "scripts/plan_phase3_policy_candidate_trace.py",
                display_path(bundle_path) or str(bundle_path),
                "--candidate-prompt-set-path",
                display_path(planned_prompt_set) or str(planned_prompt_set),
                "--candidate-trace-path",
                display_path(replay_trace_for_commands) or str(replay_trace_for_commands),
                "--candidate-trace-receipt-path",
                display_path(planned_candidate_trace_receipt) or str(planned_candidate_trace_receipt),
                "--json",
            ),
        },
        {
            "command_class": "candidate_trace_contract_validator",
            "requires_runtime": False,
            "command": uv_command(
                "scripts/validate_llama_cpp_router_trace.py",
                display_path(replay_trace_for_commands) or str(replay_trace_for_commands),
                "--require-kind",
                "selected_experts",
                "--require-kind",
                "selected_weights",
                "--require-kind",
                "selected_weights_norm",
                "--json",
            ),
        },
        {
            "command_class": "candidate_policy_replay",
            "requires_runtime": False,
            "command": uv_command(
                "scripts/plan_baseline_policy_replay.py",
                display_path(replay_trace_for_commands) or str(replay_trace_for_commands),
                display_path(inventory_path) or "<inventory_path>",
                display_path(policies_path) or "<policies_path>",
                "--json",
            ),
        },
    ]
    bundle_command = uv_command(
        "scripts/build_phase3_real_evidence_bundle.py",
        "--name",
        str(manifest.get("name", "Phase 3 real-evidence bundle")),
        "--model-id",
        str(manifest.get("model_id", "replace-with-model-id")),
        "--source-format",
        str(manifest.get("source_format", "gguf")),
        "--backend-family",
        str(manifest.get("backend_family", "llama_cpp")),
        "--prompt-family",
        str(manifest.get("prompt_family", "replace-with-prompt-family")),
        "--trace-path",
        display_path(replay_trace_for_commands) or str(replay_trace_for_commands),
        "--inventory-path",
        display_path(inventory_path) or "<inventory_path>",
        "--policies-path",
        display_path(policies_path) or "<policies_path>",
        "--managed-plan-path",
        display_path(managed_plan_path) or "<managed_plan_path>",
        "--output",
        display_path(planned_updated_bundle) or str(planned_updated_bundle),
        "--json",
    )
    if fallback_artifact_path is not None:
        bundle_command.extend(["--fallback-artifact-path", display_path(fallback_artifact_path) or str(fallback_artifact_path)])
    if live_proof_artifact_path is not None:
        bundle_command.extend(["--live-proof-artifact-path", display_path(live_proof_artifact_path) or str(live_proof_artifact_path)])
    if policy_candidate_trace_capture_approved:
        bundle_command.append("--real-model-trace-capture-approved")
    if runtime_prompt_traffic_approved:
        bundle_command.append("--runtime-prompt-traffic-approved")
    commands.extend(
        [
            {
                "command_class": "phase3_bundle_builder_with_candidate_trace",
                "requires_runtime": False,
                "command": bundle_command,
            },
            {
                "command_class": "phase3_bundle_validator",
                "requires_runtime": False,
                "command": uv_command(
                    "scripts/plan_phase3_real_evidence_bundle.py",
                    display_path(planned_updated_bundle) or str(planned_updated_bundle),
                    "--json",
                ),
            },
        ]
    )

    errors = [*current_replay_errors, *prompt_summary.get("errors", []), *candidate_trace_summary.get("errors", [])]
    return {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "mode": "phase3_policy_candidate_trace_plan",
        "valid": not errors,
        "errors": errors,
        "bundle_path": display_path(bundle_path),
        "bundle_valid": bundle_summary.get("valid"),
        "bundle_ready_for_phase4": bundle_summary.get("bundle_ready_for_phase4"),
        "model_id": manifest.get("model_id"),
        "backend_family": manifest.get("backend_family"),
        "prompt_family": manifest.get("prompt_family"),
        "output_dir": display_path(output_root),
        "candidate_prompt_set_path": display_path(planned_prompt_set),
        "candidate_trace_path": display_path(planned_candidate_trace),
        "candidate_trace_receipt_path": display_path(planned_candidate_trace_receipt),
        "updated_bundle_path": display_path(planned_updated_bundle),
        "approvals": {
            "policy_candidate_trace_capture_approved": policy_candidate_trace_capture_approved,
            "runtime_prompt_traffic_approved": runtime_prompt_traffic_approved,
        },
        "current_replay": current_diagnostics,
        "candidate_trace": candidate_trace_summary,
        "prompt_set": prompt_summary,
        "policy_candidate_ready_after_plan": policy_candidate_ready_after_plan,
        "artifact_requests": artifact_requests,
        "commands": commands,
        "candidate_trace_contract": {
            "must_use_same_model_backend_and_inventory": True,
            "must_include_ready_capture_receipt": True,
            "must_include_selected_experts_and_weights": True,
            "min_reuse_distance_observations": 1,
            "receipt_required_fields": list(phase3_trace_receipts.READY_TEXT_FIELDS)
            + list(phase3_trace_receipts.READY_BOOLEAN_FIELDS),
            "prompt_strategy": [
                "Use the exact prompt set artifact for provenance.",
                "Include repeated or closely related prompt cases instead of only one-off diverse prompts.",
                "Keep prompt ids stable so output and trace artifacts can be joined later.",
                "Preserve llama.cpp router trace events; endpoint timing is not a substitute.",
            ],
        },
        "runtime_capture_contract": {
            "requires_explicit_approval": not approvals_ready,
            "must_write_saved_trace_artifact_only": True,
            "must_write_trace_capture_receipt": True,
            "trace_capture_receipt_path": display_path(planned_candidate_trace_receipt),
            "must_not_store_private_tokens": True,
            "must_not_claim_live_paging": True,
        },
        "safety_contract": [
            "planner reads local metadata and optional saved trace artifacts only",
            "planner does not launch model servers",
            "planner does not run Docker",
            "planner does not download models",
            "planner does not inspect private tokens",
            "planner does not send prompt traffic",
            "planner does not mutate runtime residency",
            "planner does not claim live expert paging",
        ],
        "next_actions": [
            "Attach or write a candidate prompt set with repeated routed-expert opportunities.",
            "After approval, capture a semantic router trace for that prompt set on the same model/backend.",
            "Fill the candidate trace capture receipt with the approved request path, prompt set path, trace path, model/backend, host, timestamp, and approval flags.",
            "Run policy replay over the candidate trace, existing inventory, and baseline policies.",
            "Only rebuild the Phase 3 bundle with the candidate trace if replay produces a policy candidate or documents a stronger no-go.",
        ],
    }


def command_to_text(command: Any) -> str:
    if isinstance(command, list):
        return " ".join(str(part) for part in command)
    return str(command)


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def artifact_detail_value(request: JSONDict, key: str) -> Any:
    details = request.get("details") if isinstance(request.get("details"), dict) else {}
    current = details.get("current") if isinstance(details.get("current"), dict) else {}
    if key == "path":
        return details.get("path") or current.get("path") or "missing"
    if key == "ready":
        if "ready" in current:
            return current.get("ready")
        if "policy_candidate_ready" in current:
            return current.get("policy_candidate_ready")
        return "unknown"
    if key == "blocker":
        blocker = current.get("blocker")
        if isinstance(blocker, dict):
            return blocker.get("id") or "unknown"
        return blocker or "none"
    return current.get(key, "unknown")


def format_markdown_report(summary: JSONDict) -> str:
    contract = summary.get("runtime_capture_contract") if isinstance(summary.get("runtime_capture_contract"), dict) else {}
    trace_contract = summary.get("candidate_trace_contract") if isinstance(summary.get("candidate_trace_contract"), dict) else {}
    receipt_fields = trace_contract.get("receipt_required_fields") if isinstance(trace_contract.get("receipt_required_fields"), list) else []
    current_replay = summary.get("current_replay") if isinstance(summary.get("current_replay"), dict) else {}
    prompt_set = summary.get("prompt_set") if isinstance(summary.get("prompt_set"), dict) else {}
    candidate_trace = summary.get("candidate_trace") if isinstance(summary.get("candidate_trace"), dict) else {}
    capture_receipt = candidate_trace.get("capture_receipt") if isinstance(candidate_trace.get("capture_receipt"), dict) else {}
    lines = [
        "# Phase 3 Policy-Candidate Trace",
        "",
        f"- Valid: `{summary.get('valid')}`",
        f"- Bundle: `{markdown_escape(summary.get('bundle_path'))}`",
        f"- Prompt family: `{markdown_escape(summary.get('prompt_family'))}`",
        f"- Current policy candidate ready: `{current_replay.get('policy_candidate_ready', 'unknown')}`",
        f"- Current reuse-distance observations: `{current_replay.get('reuse_distance_observations', 'unknown')}`",
        f"- Prompt set ready: `{prompt_set.get('ready', 'unknown')}`",
        f"- Candidate trace ready: `{candidate_trace.get('policy_candidate_ready', 'unknown')}`",
        f"- Candidate trace receipt ready: `{capture_receipt.get('ready', 'unknown')}`",
        f"- Policy candidate ready after plan: `{summary.get('policy_candidate_ready_after_plan')}`",
        f"- Runtime approval required: `{contract.get('requires_explicit_approval')}`",
        "",
        "## Artifact Requests",
        "",
        "| Artifact | Status | Approval | Ready | Path | Blocker |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for request in summary.get("artifact_requests", []):
        if not isinstance(request, dict):
            continue
        approval = "required" if request.get("approval_required") is True else "not required"
        lines.append(
            "| "
            + " | ".join(
                markdown_escape(value)
                for value in (
                    request.get("id") or "unknown",
                    request.get("status") or "unknown",
                    approval,
                    artifact_detail_value(request, "ready"),
                    artifact_detail_value(request, "path"),
                    artifact_detail_value(request, "blocker"),
                )
            )
            + " |"
        )
    blockers = current_replay.get("blockers") if isinstance(current_replay.get("blockers"), list) else []
    if blockers:
        lines.extend(["", "## Current Replay Blockers", ""])
        for blocker in blockers:
            if isinstance(blocker, dict):
                lines.append(f"- `{markdown_escape(blocker.get('id') or 'unknown')}`")
            else:
                lines.append(f"- `{markdown_escape(blocker)}`")
    if trace_contract:
        prompt_strategy = trace_contract.get("prompt_strategy") if isinstance(trace_contract.get("prompt_strategy"), list) else []
        lines.extend(
            [
                "",
                "## Candidate Trace Contract",
                "",
                f"- Must use same model/backend/inventory: `{trace_contract.get('must_use_same_model_backend_and_inventory')}`",
                f"- Must include ready capture receipt: `{trace_contract.get('must_include_ready_capture_receipt')}`",
                f"- Must include selected experts and weights: `{trace_contract.get('must_include_selected_experts_and_weights')}`",
                f"- Minimum reuse-distance observations: `{trace_contract.get('min_reuse_distance_observations')}`",
                f"- Receipt required fields: `{markdown_escape(', '.join(str(item) for item in receipt_fields))}`",
            ]
        )
        if prompt_strategy:
            lines.extend(["", "Prompt strategy:"])
            for item in prompt_strategy:
                lines.append(f"- {markdown_escape(item)}")
    if contract:
        lines.extend(
            [
                "",
                "## Runtime Capture Contract",
                "",
                f"- Requires explicit approval: `{contract.get('requires_explicit_approval')}`",
                f"- Saved trace artifact only: `{contract.get('must_write_saved_trace_artifact_only')}`",
                f"- Must write trace capture receipt: `{contract.get('must_write_trace_capture_receipt')}`",
                f"- Trace capture receipt path: `{markdown_escape(contract.get('trace_capture_receipt_path'))}`",
                f"- Must not store private tokens: `{contract.get('must_not_store_private_tokens')}`",
            ]
        )
    commands = summary.get("commands") if isinstance(summary.get("commands"), list) else []
    if commands:
        lines.extend(["", "## Commands", ""])
        for item in commands:
            if not isinstance(item, dict):
                continue
            lines.append(f"- `{markdown_escape(item.get('command_class') or 'unknown')}`")
            lines.append("```sh")
            lines.append(command_to_text(item.get("command")))
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
    print("MoE Run Anyway Phase 3 policy-candidate trace plan")
    print(f"Valid: {summary['valid']}")
    print(f"Bundle: {summary['bundle_path']}")
    print(f"Prompt family: {summary['prompt_family']}")
    print(f"Current policy candidate ready: {summary['current_replay']['policy_candidate_ready']}")
    print(f"Current reuse-distance observations: {summary['current_replay']['reuse_distance_observations']}")
    print(f"Candidate trace ready: {summary['candidate_trace']['policy_candidate_ready']}")
    print(f"Candidate trace receipt ready: {summary['candidate_trace'].get('capture_receipt', {}).get('ready')}")
    print(f"Policy candidate ready after plan: {summary['policy_candidate_ready_after_plan']}")
    if summary["errors"]:
        print("Errors:")
        for error in summary["errors"]:
            print(f"  - {error}")
    if summary["current_replay"].get("blockers"):
        print("Current replay blockers:")
        for blocker in summary["current_replay"]["blockers"]:
            print(f"  - {blocker['id']}")
    print("Artifact requests:")
    for request in summary["artifact_requests"]:
        print(f"  - {request['id']}: {request['status']}")
    print("Safety contract:")
    for item in summary["safety_contract"]:
        print(f"  - {item}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_path", nargs="?", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--candidate-prompt-set-path", type=Path)
    parser.add_argument("--candidate-trace-path", type=Path)
    parser.add_argument("--candidate-trace-receipt-path", type=Path)
    parser.add_argument("--source-request-path", type=Path)
    parser.add_argument("--updated-bundle-path", type=Path)
    parser.add_argument("--policy-candidate-trace-capture-approved", action="store_true")
    parser.add_argument("--runtime-prompt-traffic-approved", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    parser.add_argument("--output-md", type=Path, help="write a Markdown policy-candidate trace handoff report")
    return parser


def plan_path(
    bundle_path: Path = DEFAULT_BUNDLE_PATH,
    **kwargs: Any,
) -> tuple[int, JSONDict | None, str | None]:
    try:
        summary = build_capture_plan(bundle_path, **kwargs)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not build Phase 3 policy-candidate trace plan: {exc}"
    return (0 if summary["valid"] else 2), summary, None


def main_from_test_path(bundle_path: Path = DEFAULT_BUNDLE_PATH) -> int:
    status, _, _ = plan_path(bundle_path)
    return status


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary, error_message = plan_path(
        args.bundle_path,
        output_dir=args.output_dir,
        candidate_prompt_set_path=args.candidate_prompt_set_path,
        candidate_trace_path=args.candidate_trace_path,
        candidate_trace_receipt_path=args.candidate_trace_receipt_path,
        source_request_path=args.source_request_path,
        updated_bundle_path=args.updated_bundle_path,
        policy_candidate_trace_capture_approved=args.policy_candidate_trace_capture_approved,
        runtime_prompt_traffic_approved=args.runtime_prompt_traffic_approved,
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