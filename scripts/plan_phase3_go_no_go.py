#!/usr/bin/env python3
"""Build a Phase 3 go/no-go decision packet for live expert-paging work.

This planner aggregates the offline Phase 3 gates: replay metrics, real-model
trace/inventory pairing, capture-result intake, dense fallback comparison,
managed-loading capability gaps, runtime-actuator spike proof handoff, and optional live-capability proof. It does
not launch models, run Docker, load tensor values, mutate residency, or send
prompt traffic.
"""

from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import json
import sys
from pathlib import Path
from typing import Any

import plan_baseline_policy_replay
import plan_dense_fallback_comparison
import plan_managed_expert_loading
import plan_phase3_capture_result_intake
import plan_phase3_handoff_coverage
import plan_phase3_launch_card_library
import plan_phase3_live_capability_proof
import plan_phase3_real_evidence_matrix
import plan_phase3_runtime_actuator_spike
import plan_phase3_runtime_capture_request
import plan_phase3_reuse_evidence_capture
import plan_real_model_trace_inventory_pairing
import phase3_trace_receipts


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_PATH = ROOT / "memory-moe-mvp" / "data" / "llama_cpp_router_trace.fixture.jsonl"
DEFAULT_INVENTORY_PATH = ROOT / "memory-moe-mvp" / "data" / "expert_inventory_manifest.fixture.json"
DEFAULT_POLICIES_PATH = ROOT / "memory-moe-mvp" / "data" / "baseline_replay_policies.json"
DEFAULT_MANAGED_PLAN_PATH = ROOT / "memory-moe-mvp" / "data" / "managed_expert_loading_plan.json"
DEFAULT_REAL_EVIDENCE_ROOT = ROOT / "memory-moe-mvp" / "phase3-real-evidence"
REQUIRED_LIVE_CAPABILITIES = {
    "expert_inventory",
    "routing_visibility",
    "residency_observation",
    "residency_control",
    "policy_application",
    "dense_fallback",
    "cleanup_restore",
    "artifact_export",
}

JSONDict = dict[str, Any]
RepoGateSummaries = tuple[JSONDict, ...]


def int_count(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def normalize_path_text(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().replace("\\", "/")


def resolve_repo_path(value: Any) -> Path | None:
    text = normalize_path_text(value)
    if text is None:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return ROOT / path


def resolved_path_text(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def path_matches(value: Any, expected_path: Path | None) -> bool:
    if expected_path is None:
        return False
    actual = normalize_path_text(value)
    if actual is None:
        return False
    try:
        expected_display = expected_path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        expected_display = resolved_path_text(expected_path)
    if actual == expected_display:
        return True
    actual_path = resolve_repo_path(actual)
    return actual_path is not None and resolved_path_text(actual_path) == resolved_path_text(expected_path)


def policy_candidate_trace_receipt_summary(
    receipt_path: Path | None,
    *,
    trace_path: Path,
    expected_model_id: str | None = None,
    expected_backend_family: str | None = None,
    expected_prompt_family: str | None = None,
) -> JSONDict:
    if receipt_path is None:
        return {
            "path": None,
            "exists": False,
            "valid": False,
            "receipt_ready": False,
            "ready": False,
            "candidate_trace_path_matches": False,
            "errors": [],
            "blockers": ["policy_candidate_trace_receipt_missing"],
        }
    display = None
    try:
        display = receipt_path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        display = str(receipt_path)
    if not receipt_path.exists():
        return {
            "path": display,
            "exists": False,
            "valid": False,
            "receipt_ready": False,
            "ready": False,
            "candidate_trace_path_matches": False,
            "errors": [],
            "blockers": ["policy_candidate_trace_receipt_missing"],
        }
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "path": display,
            "exists": True,
            "valid": False,
            "receipt_ready": False,
            "ready": False,
            "candidate_trace_path_matches": False,
            "errors": [str(exc)],
            "blockers": ["policy_candidate_trace_receipt_invalid"],
        }
    validation = phase3_trace_receipts.validate_trace_receipt_source_binding(
        receipt,
        repo_root=ROOT,
        expected_candidate_trace_path=trace_path,
    )
    errors = list(validation.get("errors", []))
    receipt_ready = validation.get("receipt_ready") is True
    candidate_trace_path_matches = validation.get("candidate_trace_path_matches") is True
    if receipt_ready:
        if expected_model_id is not None and receipt.get("model_id") != expected_model_id:
            errors.append("trace_capture_receipt.model_id must match expected model_id")
        if expected_backend_family is not None and receipt.get("runtime_backend") != expected_backend_family:
            errors.append("trace_capture_receipt.runtime_backend must match expected backend_family")
        if expected_prompt_family is not None and receipt.get("prompt_family") != expected_prompt_family:
            errors.append("trace_capture_receipt.prompt_family must match expected prompt_family")
    blockers: list[str] = []
    if not receipt_ready:
        blockers.append("policy_candidate_trace_receipt_not_ready")
    if errors:
        blockers.append("policy_candidate_trace_receipt_invalid")
    ready = receipt_ready and not errors and candidate_trace_path_matches
    return {
        "path": display,
        "exists": True,
        "valid": not errors and (receipt_ready or bool(blockers)),
        "receipt_ready": receipt_ready,
        "ready": ready,
        "candidate_trace_path_matches": candidate_trace_path_matches,
        "candidate_trace_path_exists": validation.get("candidate_trace_path_exists"),
        "source_request_path_exists": validation.get("source_request_path_exists"),
        "source_prompt_set_path_exists": validation.get("source_prompt_set_path_exists"),
        "source_request_path_matches": validation.get("source_request_path_matches"),
        "source_prompt_set_path_matches": validation.get("source_prompt_set_path_matches"),
        "errors": errors,
        "blockers": sorted(set(blockers)),
    }


def candidate_trace_receipt_scaffold_summary(reuse_summary: JSONDict | None) -> JSONDict:
    if reuse_summary is None:
        return {
            "source": None,
            "ready": None,
            "required_count": None,
            "ready_count": None,
            "valid_count": None,
            "blocked_count": None,
        }
    required_count = int_count(reuse_summary.get("bundle_count"))
    ready_count = int_count(reuse_summary.get("candidate_trace_receipt_ready_count"))
    valid_count = int_count(reuse_summary.get("candidate_trace_valid_count"))
    return {
        "source": "repo_reuse_evidence_capture",
        "ready": required_count > 0 and ready_count == required_count,
        "required_count": required_count,
        "ready_count": ready_count,
        "valid_count": valid_count,
        "blocked_count": max(required_count - ready_count, 0),
    }


def policy_candidate_trace_receipt_gate_summary(
    receipt_summary: JSONDict,
    real_evidence_summary: JSONDict | None,
    reuse_summary: JSONDict | None = None,
) -> JSONDict:
    scaffold = candidate_trace_receipt_scaffold_summary(reuse_summary)
    scaffold_fields = {
        "scaffold_source": scaffold.get("source"),
        "scaffold_ready": scaffold.get("ready"),
        "scaffold_required_count": scaffold.get("required_count"),
        "scaffold_ready_count": scaffold.get("ready_count"),
        "scaffold_valid_count": scaffold.get("valid_count"),
        "scaffold_blocked_count": scaffold.get("blocked_count"),
    }
    if real_evidence_summary is not None:
        required_count = int_count(real_evidence_summary.get("policy_candidate_trace_receipt_required_count"))
        ready_count = int_count(real_evidence_summary.get("policy_candidate_trace_receipt_ready_count"))
        attached_count = int_count(real_evidence_summary.get("policy_candidate_trace_receipt_attached_count"))
        blocked_count = int_count(real_evidence_summary.get("policy_candidate_trace_receipt_blocked_bundle_count"))
        return {
            "source": "repo_real_evidence_matrix",
            "path": None,
            "exists": attached_count > 0,
            "valid": real_evidence_summary.get("valid") is True,
            "receipt_ready": ready_count > 0,
            "ready": ready_count > 0,
            "required_count": required_count,
            "ready_count": ready_count,
            "attached_count": attached_count,
            "blocked_bundle_count": blocked_count,
            **scaffold_fields,
            "errors": [],
            "blockers": ["policy_candidate_trace_receipt_not_ready"] if required_count and not ready_count else [],
        }
    return {
        **receipt_summary,
        "source": "direct_receipt_path",
        "required_count": 1 if receipt_summary.get("path") is not None else 0,
        "ready_count": 1 if receipt_summary.get("ready") is True else 0,
        "attached_count": 1 if receipt_summary.get("exists") is True else 0,
        "blocked_bundle_count": 1 if receipt_summary.get("path") is not None and receipt_summary.get("ready") is not True else 0,
        **scaffold_fields,
    }


def reason(reason_id: str, description: str, evidence: Any = None) -> JSONDict:
    item: JSONDict = {"id": reason_id, "description": description}
    if evidence is not None:
        item["evidence"] = evidence
    return item


def blocker_id(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("id") or "unknown")
    return str(item)


def policy_candidate_evidence(real_evidence_summary: JSONDict | None, replay_summary: JSONDict) -> JSONDict:
    if real_evidence_summary is not None:
        bundles = [item for item in real_evidence_summary.get("bundles", []) if isinstance(item, dict)]
        blocked_bundles: list[JSONDict] = []
        blocker_counts: Counter[str] = Counter()
        candidate_policy_ids = sorted(
            {
                str(policy_id)
                for bundle in bundles
                for policy_id in bundle.get("candidate_policy_ids", [])
            }
        )
        for bundle in bundles:
            raw_blocker_ids = bundle.get("policy_candidate_blocker_ids")
            if isinstance(raw_blocker_ids, list):
                bundle_blocker_ids = [str(item) for item in raw_blocker_ids]
            else:
                bundle_blocker_ids = [blocker_id(item) for item in bundle.get("policy_candidate_blockers", [])]
            blocker_counts.update(bundle_blocker_ids)
            if bundle.get("policy_candidate_ready") is not True:
                blocked_bundles.append(
                    {
                        "name": bundle.get("name"),
                        "path": bundle.get("path"),
                        "model_id": bundle.get("model_id"),
                        "replay_valid": bundle.get("replay_valid"),
                        "reuse_distance_observations": int_count(bundle.get("reuse_distance_observations")),
                        "blocker_ids": sorted(set(bundle_blocker_ids)),
                    }
                )
        no_reuse_count = sum(
            1
            for bundle in blocked_bundles
            if "no_reuse_distance_observations" in bundle.get("blocker_ids", [])
        )
        next_operator_step = (
            "capture_candidate_router_trace_with_reuse_distance"
            if blocked_bundles and no_reuse_count
            else "review_policy_replay_outputs"
        )
        return {
            "source": "repo_real_evidence_matrix",
            "bundle_count": real_evidence_summary.get("bundle_count"),
            "policy_candidate_ready_count": real_evidence_summary.get("policy_candidate_ready_count"),
            "blocked_bundle_count": len(blocked_bundles),
            "candidate_policy_ids": candidate_policy_ids,
            "blocker_counts": dict(sorted(blocker_counts.items())),
            "no_reuse_distance_observation_count": no_reuse_count,
            "next_operator_step": next_operator_step,
            "blocked_bundles": blocked_bundles,
        }

    diagnostics = replay_summary.get("policy_candidate_diagnostics", {})
    raw_blockers = diagnostics.get("blockers", []) if isinstance(diagnostics, dict) else []
    blocker_ids = [blocker_id(item) for item in raw_blockers]
    reuse_distance = replay_summary.get("reuse_distance", {})
    reuse_distance_observations = int_count(reuse_distance.get("observations")) if isinstance(reuse_distance, dict) else 0
    return {
        "source": "baseline_policy_replay",
        "bundle_count": None,
        "policy_candidate_ready_count": 1 if replay_summary.get("candidate_policy_ids") else 0,
        "blocked_bundle_count": 0 if replay_summary.get("candidate_policy_ids") else 1,
        "candidate_policy_ids": replay_summary.get("candidate_policy_ids", []),
        "blocker_counts": dict(sorted(Counter(blocker_ids).items())),
        "no_reuse_distance_observation_count": 1 if "no_reuse_distance_observations" in blocker_ids else 0,
        "next_operator_step": (
            "capture_candidate_router_trace_with_reuse_distance"
            if "no_reuse_distance_observations" in blocker_ids
            else "review_policy_replay_outputs"
        ),
        "blocked_bundles": [
            {
                "name": "default_trace_inventory_pair",
                "path": replay_summary.get("trace_path"),
                "model_id": None,
                "replay_valid": replay_summary.get("valid"),
                "reuse_distance_observations": reuse_distance_observations,
                "blocker_ids": sorted(set(blocker_ids)),
            }
        ] if not replay_summary.get("candidate_policy_ids") else [],
    }


def unavailable_live_capabilities(managed_summary: JSONDict) -> dict[str, str]:
    statuses = managed_summary.get("capability_statuses", {})
    if not isinstance(statuses, dict):
        return {}
    return {
        capability_id: str(status)
        for capability_id, status in sorted(statuses.items())
        if capability_id in REQUIRED_LIVE_CAPABILITIES and status != "available"
    }


def capture_intake_ready(capture_intake_summary: JSONDict | None) -> bool | None:
    if capture_intake_summary is None:
        return None
    return (
        capture_intake_summary.get("valid") is True
        and int_count(capture_intake_summary.get("ready_to_update_bundle_count")) > 0
    )


def capture_intake_evidence(capture_intake_summary: JSONDict | None) -> JSONDict:
    if capture_intake_summary is None:
        return {
            "root": None,
            "request_count": None,
            "request_drift_free_count": None,
            "request_drifted_count": None,
            "request_ready_for_operator_capture_flag_count": None,
            "approved_but_capture_incomplete_request_count": None,
            "runtime_approval_missing_request_count": None,
            "approval_rebuild_command_available_request_count": None,
            "approval_rebuild_command_manifest_count": None,
            "approved_runtime_capture_pending_request_count": None,
            "capture_receipt_ready_count": None,
            "capture_receipt_required_count": None,
            "capture_receipt_missing_request_count": None,
            "candidate_trace_receipt_ready_count": None,
            "managed_output_receipt_ready_count": None,
            "dense_output_receipt_ready_count": None,
            "live_capability_proof_ready_count": None,
            "all_capture_receipts_ready": None,
            "all_output_receipt_bindings_ready": None,
            "receipt_fill_candidate_router_trace_entry_count": None,
            "receipt_fill_candidate_router_trace_ready_count": None,
            "receipt_fill_candidate_router_trace_missing_count": None,
            "receipt_fill_managed_output_entry_count": None,
            "receipt_fill_managed_output_ready_count": None,
            "receipt_fill_managed_output_missing_count": None,
            "receipt_fill_dense_output_entry_count": None,
            "receipt_fill_dense_output_ready_count": None,
            "receipt_fill_dense_output_missing_count": None,
            "output_receipt_binding_ready_count": None,
            "ready_to_update_bundle_count": None,
            "phase4_candidate_ready_count": None,
            "next_operator_step_counts": {},
            "remaining_blockers_after_intake": [],
        }
    receipt_gate_coverage = capture_intake_summary.get("receipt_gate_coverage")
    receipt_gate_coverage = receipt_gate_coverage if isinstance(receipt_gate_coverage, dict) else {}
    receipt_fill_summary = capture_intake_summary.get("receipt_fill_manifest_summary")
    receipt_fill_summary = receipt_fill_summary if isinstance(receipt_fill_summary, dict) else {}
    receipt_fill_by_artifact = receipt_fill_summary.get("by_artifact")
    receipt_fill_by_artifact = receipt_fill_by_artifact if isinstance(receipt_fill_by_artifact, dict) else {}
    candidate_fill = receipt_fill_by_artifact.get("candidate_router_trace")
    candidate_fill = candidate_fill if isinstance(candidate_fill, dict) else {}
    managed_fill = receipt_fill_by_artifact.get("managed_output_summary_fill")
    managed_fill = managed_fill if isinstance(managed_fill, dict) else {}
    dense_fill = receipt_fill_by_artifact.get("dense_output_summary_fill")
    dense_fill = dense_fill if isinstance(dense_fill, dict) else {}
    return {
        "root": capture_intake_summary.get("root"),
        "request_count": capture_intake_summary.get("request_count"),
        "request_drift_free_count": capture_intake_summary.get("request_drift_free_count"),
        "request_drifted_count": capture_intake_summary.get("request_drifted_count"),
        "request_ready_for_operator_capture_flag_count": capture_intake_summary.get("request_ready_for_operator_capture_flag_count"),
        "approved_but_capture_incomplete_request_count": capture_intake_summary.get("approved_but_capture_incomplete_request_count"),
        "runtime_approval_missing_request_count": capture_intake_summary.get("runtime_approval_missing_request_count"),
        "approval_rebuild_command_available_request_count": capture_intake_summary.get("approval_rebuild_command_available_request_count"),
        "approval_rebuild_command_manifest_count": len(capture_intake_summary.get("approval_rebuild_command_manifest", [])) if isinstance(capture_intake_summary.get("approval_rebuild_command_manifest"), list) else None,
        "approved_runtime_capture_pending_request_count": capture_intake_summary.get("approved_runtime_capture_pending_request_count"),
        "capture_receipt_ready_count": capture_intake_summary.get("capture_receipt_ready_count"),
        "capture_receipt_required_count": capture_intake_summary.get("capture_receipt_required_count"),
        "capture_receipt_missing_request_count": capture_intake_summary.get("capture_receipt_missing_request_count"),
        "candidate_trace_receipt_ready_count": receipt_gate_coverage.get("candidate_trace_receipt_ready_count"),
        "managed_output_receipt_ready_count": receipt_gate_coverage.get("managed_output_receipt_ready_count"),
        "dense_output_receipt_ready_count": receipt_gate_coverage.get("dense_output_receipt_ready_count"),
        "live_capability_proof_ready_count": receipt_gate_coverage.get("live_capability_proof_ready_count"),
        "all_capture_receipts_ready": receipt_gate_coverage.get("all_capture_receipts_ready"),
        "all_output_receipt_bindings_ready": receipt_gate_coverage.get("all_output_receipt_bindings_ready"),
        "receipt_fill_candidate_router_trace_entry_count": candidate_fill.get("entry_count"),
        "receipt_fill_candidate_router_trace_ready_count": candidate_fill.get("ready_count"),
        "receipt_fill_candidate_router_trace_missing_count": candidate_fill.get("missing_count"),
        "receipt_fill_managed_output_entry_count": managed_fill.get("entry_count"),
        "receipt_fill_managed_output_ready_count": managed_fill.get("ready_count"),
        "receipt_fill_managed_output_missing_count": managed_fill.get("missing_count"),
        "receipt_fill_dense_output_entry_count": dense_fill.get("entry_count"),
        "receipt_fill_dense_output_ready_count": dense_fill.get("ready_count"),
        "receipt_fill_dense_output_missing_count": dense_fill.get("missing_count"),

        "output_receipt_binding_ready_count": capture_intake_summary.get("output_receipt_binding_ready_count"),
        "ready_to_update_bundle_count": capture_intake_summary.get("ready_to_update_bundle_count"),
        "phase4_candidate_ready_count": capture_intake_summary.get("phase4_candidate_ready_count"),
        "next_operator_step_counts": capture_intake_summary.get("next_operator_step_counts", {}),
        "remaining_blockers_after_intake": capture_intake_summary.get("remaining_blockers_after_intake", []),
    }


def handoff_ready(handoff_summary: JSONDict | None) -> bool | None:
    if handoff_summary is None:
        return None
    return handoff_summary.get("valid") is True and handoff_summary.get("all_handoff_scaffolds_ready") is True


def launch_card_library_ready(launch_card_summary: JSONDict | None) -> bool | None:
    if launch_card_summary is None:
        return None
    return (
        launch_card_summary.get("valid") is True
        and launch_card_summary.get("library_ready") is True
        and launch_card_summary.get("saved_handoff_artifacts_ready") is True
    )


def launch_card_library_evidence(launch_card_summary: JSONDict | None) -> JSONDict:
    if launch_card_summary is None:
        return {
            "root": None,
            "library_ready": None,
            "execution_ready": None,
            "card_count": None,
            "task_count": None,
            "binding_handoff_ready": None,
            "binding_handoff_task_count": None,
            "model_plane_artifact_writer_contract_request_ready": None,
            "model_plane_artifact_writer_contract_request_task_count": None,
            "saved_handoff_artifacts_ready": None,
            "saved_handoff_artifact_missing_count": None,
            "saved_handoff_artifact_drifted_count": None,
            "missing_runtime_command_count": None,
            "blockers": [],
        }
    return {
        "root": launch_card_summary.get("root"),
        "library_ready": launch_card_summary.get("library_ready"),
        "execution_ready": launch_card_summary.get("execution_ready"),
        "card_count": launch_card_summary.get("card_count"),
        "task_count": launch_card_summary.get("task_count"),
        "binding_handoff_ready": launch_card_summary.get("binding_handoff_ready"),
        "binding_handoff_task_count": launch_card_summary.get("binding_handoff_task_count"),
        "model_plane_artifact_writer_contract_request_ready": launch_card_summary.get("model_plane_artifact_writer_contract_request_ready"),
        "model_plane_artifact_writer_contract_request_task_count": launch_card_summary.get("model_plane_artifact_writer_contract_request_task_count"),
        "saved_handoff_artifacts_ready": launch_card_summary.get("saved_handoff_artifacts_ready"),
        "saved_handoff_artifact_missing_count": launch_card_summary.get("saved_handoff_artifact_missing_count"),
        "saved_handoff_artifact_drifted_count": launch_card_summary.get("saved_handoff_artifact_drifted_count"),
        "missing_runtime_command_count": launch_card_summary.get("missing_runtime_command_count"),
        "blockers": launch_card_summary.get("blockers", []),
    }


def handoff_evidence(handoff_summary: JSONDict | None) -> JSONDict:
    if handoff_summary is None:
        return {
            "root": None,
            "bundle_count": None,
            "handoff_scaffold_ready_count": None,
            "all_handoff_scaffolds_ready": None,
            "missing_artifact_counts": {},
            "runtime_capture_launch_card_template_ready_count": None,
            "runtime_capture_launch_card_binding_ready_count": None,
            "runtime_capture_launch_card_runtime_command_ready_count": None,
            "runtime_capture_launch_card_command_option_count": None,
            "runtime_capture_launch_card_missing_runtime_command_count": None,
            "runtime_capture_launch_card_missing_runtime_command_artifact_ids": [],
        }

    bundles = [bundle for bundle in handoff_summary.get("bundles", []) if isinstance(bundle, dict)]
    launch_cards = [
        bundle.get("runtime_capture_launch_card_template")
        for bundle in bundles
        if isinstance(bundle.get("runtime_capture_launch_card_template"), dict)
    ]
    missing_runtime_command_ids = sorted(
        {
            str(artifact_id)
            for card in launch_cards
            for artifact_id in card.get("missing_runtime_command_artifact_ids", [])
            if isinstance(artifact_id, str)
        }
    )
    return {
        "root": handoff_summary.get("root"),
        "bundle_count": handoff_summary.get("bundle_count"),
        "handoff_scaffold_ready_count": handoff_summary.get("handoff_scaffold_ready_count"),
        "all_handoff_scaffolds_ready": handoff_summary.get("all_handoff_scaffolds_ready"),
        "missing_artifact_counts": handoff_summary.get("missing_artifact_counts", {}),
        "runtime_capture_launch_card_template_ready_count": sum(
            1 for card in launch_cards if card.get("ready") is True
        ),
        "runtime_capture_launch_card_binding_ready_count": sum(
            1 for card in launch_cards if card.get("binding_ready") is True
        ),
        "runtime_capture_launch_card_runtime_command_ready_count": sum(
            1 for card in launch_cards if card.get("runtime_capture_command_ready") is True
        ),
        "runtime_capture_launch_card_command_option_count": sum(
            int_count(card.get("command_option_count")) for card in launch_cards
        ),
        "runtime_capture_launch_card_missing_runtime_command_count": sum(
            int_count(card.get("missing_runtime_command_count")) for card in launch_cards
        ),
        "runtime_capture_launch_card_missing_runtime_command_artifact_ids": missing_runtime_command_ids,
    }

def reuse_evidence_ready(reuse_summary: JSONDict | None) -> bool | None:
    if reuse_summary is None:
        return None
    return reuse_summary.get("valid") is True and int_count(reuse_summary.get("reuse_ready_count")) > 0


def reuse_evidence(reuse_summary: JSONDict | None) -> JSONDict:
    if reuse_summary is None:
        return {
            "root": None,
            "bundle_count": None,
            "reuse_ready_count": None,
            "reuse_blocked_count": None,
            "candidate_trace_valid_count": None,
            "candidate_trace_receipt_ready_count": None,
            "no_reuse_distance_observation_count": None,
            "prompt_identity_ready_count": None,
            "prompt_identity_metadata_missing_count": None,
            "recommended_next_capture": None,
            "blocker_counts": {},
            "warning_counts": {},
        }
    return {
        "root": reuse_summary.get("root"),
        "bundle_count": reuse_summary.get("bundle_count"),
        "reuse_ready_count": reuse_summary.get("reuse_ready_count"),
        "reuse_blocked_count": reuse_summary.get("reuse_blocked_count"),
        "candidate_trace_valid_count": reuse_summary.get("candidate_trace_valid_count"),
        "candidate_trace_receipt_ready_count": reuse_summary.get("candidate_trace_receipt_ready_count"),
        "no_reuse_distance_observation_count": reuse_summary.get("no_reuse_distance_observation_count"),
        "prompt_identity_ready_count": reuse_summary.get("prompt_identity_ready_count"),
        "prompt_identity_metadata_missing_count": reuse_summary.get("prompt_identity_metadata_missing_count"),
        "recommended_next_capture": reuse_summary.get("recommended_next_capture"),
        "blocker_counts": reuse_summary.get("blocker_counts", {}),
        "warning_counts": reuse_summary.get("warning_counts", {}),
    }

def runtime_request_audit_ready(runtime_request_summary: JSONDict | None) -> bool | None:
    if runtime_request_summary is None:
        return None
    return (
        runtime_request_summary.get("valid") is True
        and int_count(runtime_request_summary.get("request_count")) > 0
        and int_count(runtime_request_summary.get("drifted_request_count")) == 0
        and runtime_request_summary.get("all_required_validator_commands_present") is True
    )


@lru_cache(maxsize=4)
def build_repo_gate_summaries(real_evidence_root: Path) -> RepoGateSummaries:
    return (
        plan_phase3_real_evidence_matrix.build_matrix(real_evidence_root),
        plan_phase3_handoff_coverage.build_coverage(real_evidence_root),
        plan_phase3_launch_card_library.build_library(real_evidence_root),
        plan_phase3_runtime_capture_request.build_root_summary(real_evidence_root),
        plan_phase3_capture_result_intake.build_root_summary(real_evidence_root),
        plan_phase3_reuse_evidence_capture.build_root_summary(real_evidence_root),
    )


def runtime_request_evidence(runtime_request_summary: JSONDict | None) -> JSONDict:
    if runtime_request_summary is None:
        return {
            "root": None,
            "request_count": None,
            "valid_request_count": None,
            "ready_for_operator_capture_count": None,
            "capture_complete_count": None,
            "drifted_request_count": None,
            "validator_command_count": None,
            "validator_command_missing_request_count": None,
            "all_required_validator_commands_present": None,
        }
    return {
        "root": runtime_request_summary.get("root"),
        "request_count": runtime_request_summary.get("request_count"),
        "valid_request_count": runtime_request_summary.get("valid_request_count"),
        "ready_for_operator_capture_count": runtime_request_summary.get("ready_for_operator_capture_count"),
        "capture_complete_count": runtime_request_summary.get("capture_complete_count"),
        "drifted_request_count": runtime_request_summary.get("drifted_request_count"),
        "validator_command_count": runtime_request_summary.get("validator_command_count"),
        "validator_command_missing_request_count": runtime_request_summary.get("validator_command_missing_request_count"),
        "all_required_validator_commands_present": runtime_request_summary.get("all_required_validator_commands_present"),
    }


def collect_no_go_reasons(
    replay_summary: JSONDict,
    pairing_summary: JSONDict,
    fallback_summary: JSONDict,
    managed_summary: JSONDict,
    live_proof_summary: JSONDict,
    real_evidence_summary: JSONDict | None = None,
    capture_intake_summary: JSONDict | None = None,
    handoff_summary: JSONDict | None = None,
    runtime_request_summary: JSONDict | None = None,
    policy_trace_receipt_summary: JSONDict | None = None,
    reuse_summary: JSONDict | None = None,
    launch_card_summary: JSONDict | None = None,
) -> list[JSONDict]:
    reasons: list[JSONDict] = []
    if not replay_summary.get("valid"):
        reasons.append(reason("policy_replay_invalid", "Policy replay did not validate.", replay_summary.get("errors", [])))

    if real_evidence_summary is not None:
        if real_evidence_summary.get("valid") is not True:
            reasons.append(
                reason(
                    "real_evidence_matrix_invalid",
                    "Repo-local real-evidence matrix did not validate.",
                    real_evidence_summary.get("errors", []),
                )
            )
        if real_evidence_summary.get("policy_candidate_ready_count", 0) < 1:
            reasons.append(
                reason(
                    "no_replay_policy_candidate",
                    "No real trace/inventory/replay bundle has produced a policy candidate worth a live spike.",
                    policy_candidate_evidence(real_evidence_summary, replay_summary),
                )
            )
        real_bundle_count = real_evidence_summary.get("bundle_count", 0)
        real_pair_count = real_evidence_summary.get("real_model_pair_ready_count", 0)
        if real_bundle_count < 1 or real_pair_count != real_bundle_count:
            reasons.append(
                reason(
                    "real_model_trace_inventory_pairing_not_ready",
                    "Not every repo-local real-evidence bundle has scanner-derived trace/inventory pairing.",
                    {
                        "bundle_count": real_bundle_count,
                        "real_model_pair_ready_count": real_pair_count,
                    },
                )
            )
    else:
        if not replay_summary.get("candidate_policy_ids"):
            reasons.append(
                reason(
                    "no_replay_policy_candidate",
                    "Replay did not produce a policy candidate worth a live spike.",
                    policy_candidate_evidence(None, replay_summary),
                )
            )
        if not pairing_summary.get("real_model_pair_ready"):
            reasons.append(
                reason(
                    "real_model_trace_inventory_pairing_not_ready",
                    "Trace/inventory evidence is not scanner-derived real-model evidence yet.",
                    pairing_summary.get("blockers", []),
                )
            )

    candidate_policy_ids = policy_candidate_evidence(real_evidence_summary, replay_summary).get("candidate_policy_ids", [])
    if candidate_policy_ids and (
        policy_trace_receipt_summary is None or policy_trace_receipt_summary.get("ready") is not True
    ):
        reasons.append(
            reason(
                "policy_candidate_trace_receipt_not_ready",
                "Policy-candidate trace capture receipt is missing or not ready for the trace used in the decision.",
                policy_trace_receipt_summary or {"path": None, "ready": False},
            )
        )

    if reuse_summary is not None:
        if reuse_summary.get("valid") is not True:
            reasons.append(
                reason(
                    "reuse_evidence_capture_invalid",
                    "Reuse-evidence capture planner did not validate.",
                    reuse_summary.get("errors", []),
                )
            )
        if reuse_evidence_ready(reuse_summary) is not True:
            reasons.append(
                reason(
                    "reuse_evidence_capture_not_ready",
                    "No candidate router trace currently proves repeated routed expert reuse.",
                    reuse_evidence(reuse_summary),
                )
            )
    if capture_intake_summary is not None:
        capture_intake_gate_evidence = capture_intake_evidence(capture_intake_summary)
        if capture_intake_summary.get("valid") is not True:
            reasons.append(
                reason(
                    "capture_result_intake_invalid",
                    "Capture-result intake did not validate.",
                    capture_intake_summary.get("errors", []),
                )
            )
        if capture_intake_ready(capture_intake_summary) is not True:
            reasons.append(
                reason(
                    "capture_result_intake_not_ready",
                    "No saved runtime-capture request has filled receipt-bound artifacts ready for bundle update.",
                    {
                        "request_count": capture_intake_summary.get("request_count"),
                        "request_ready_for_operator_capture_flag_count": capture_intake_summary.get("request_ready_for_operator_capture_flag_count"),
                        "approved_but_capture_incomplete_request_count": capture_intake_summary.get("approved_but_capture_incomplete_request_count"),
                        "runtime_approval_missing_request_count": capture_intake_summary.get("runtime_approval_missing_request_count"),
                        "approval_rebuild_command_available_request_count": capture_intake_summary.get("approval_rebuild_command_available_request_count"),
                        "approval_rebuild_command_manifest_count": len(capture_intake_summary.get("approval_rebuild_command_manifest", [])) if isinstance(capture_intake_summary.get("approval_rebuild_command_manifest"), list) else None,
                        "approved_runtime_capture_pending_request_count": capture_intake_summary.get("approved_runtime_capture_pending_request_count"),
                        "capture_receipt_ready_count": capture_intake_summary.get("capture_receipt_ready_count"),
                        "capture_receipt_required_count": capture_intake_summary.get("capture_receipt_required_count"),
                        "capture_receipt_missing_request_count": capture_intake_summary.get("capture_receipt_missing_request_count"),
                        "candidate_trace_receipt_ready_count": capture_intake_gate_evidence["candidate_trace_receipt_ready_count"],
                        "managed_output_receipt_ready_count": capture_intake_gate_evidence["managed_output_receipt_ready_count"],
                        "dense_output_receipt_ready_count": capture_intake_gate_evidence["dense_output_receipt_ready_count"],
                        "live_capability_proof_ready_count": capture_intake_gate_evidence["live_capability_proof_ready_count"],
                        "all_capture_receipts_ready": capture_intake_gate_evidence["all_capture_receipts_ready"],
                        "all_output_receipt_bindings_ready": capture_intake_gate_evidence["all_output_receipt_bindings_ready"],
                        "receipt_fill_candidate_router_trace_entry_count": capture_intake_gate_evidence["receipt_fill_candidate_router_trace_entry_count"],
                        "receipt_fill_candidate_router_trace_ready_count": capture_intake_gate_evidence["receipt_fill_candidate_router_trace_ready_count"],
                        "receipt_fill_candidate_router_trace_missing_count": capture_intake_gate_evidence["receipt_fill_candidate_router_trace_missing_count"],
                        "receipt_fill_managed_output_entry_count": capture_intake_gate_evidence["receipt_fill_managed_output_entry_count"],
                        "receipt_fill_managed_output_ready_count": capture_intake_gate_evidence["receipt_fill_managed_output_ready_count"],
                        "receipt_fill_managed_output_missing_count": capture_intake_gate_evidence["receipt_fill_managed_output_missing_count"],
                        "receipt_fill_dense_output_entry_count": capture_intake_gate_evidence["receipt_fill_dense_output_entry_count"],
                        "receipt_fill_dense_output_ready_count": capture_intake_gate_evidence["receipt_fill_dense_output_ready_count"],
                        "receipt_fill_dense_output_missing_count": capture_intake_gate_evidence["receipt_fill_dense_output_missing_count"],

                        "output_receipt_binding_ready_count": capture_intake_summary.get("output_receipt_binding_ready_count"),
                        "ready_to_update_bundle_count": capture_intake_summary.get("ready_to_update_bundle_count"),
                    },
                )
            )

    if handoff_summary is not None:
        if handoff_summary.get("valid") is not True:
            reasons.append(
                reason(
                    "handoff_scaffold_coverage_invalid",
                    "Handoff scaffold coverage did not validate.",
                    handoff_summary.get("errors", []),
                )
            )
        if handoff_ready(handoff_summary) is not True:
            reasons.append(
                reason(
                    "handoff_scaffold_coverage_not_ready",
                    "Not every repo-local real bundle has complete operator-handoff scaffolds.",
                    handoff_evidence(handoff_summary),
                )
            )

    if launch_card_summary is not None:
        if launch_card_summary.get("valid") is not True:
            reasons.append(
                reason(
                    "launch_card_library_invalid",
                    "Launch-card library or saved bridge handoff artifacts did not validate.",
                    launch_card_summary.get("errors", []),
                )
            )
        if launch_card_library_ready(launch_card_summary) is not True:
            reasons.append(
                reason(
                    "launch_card_bridge_handoff_not_ready",
                    "Launch-card bridge handoff metadata is missing, stale, or incomplete.",
                    launch_card_library_evidence(launch_card_summary),
                )
            )
    if runtime_request_summary is not None:
        if runtime_request_summary.get("valid") is not True:
            reasons.append(
                reason(
                    "runtime_capture_request_audit_invalid",
                    "Saved runtime-capture request audit did not validate.",
                    runtime_request_summary.get("errors", []),
                )
            )
        if runtime_request_audit_ready(runtime_request_summary) is not True:
            reasons.append(
                reason(
                    "runtime_capture_request_audit_not_ready",
                    "Saved runtime-capture requests are missing, drifted, invalid, or missing validator-command coverage.",
                    runtime_request_evidence(runtime_request_summary),
                )
            )

    if not fallback_summary.get("comparison_ready"):
        reasons.append(
            reason(
                "dense_fallback_comparison_not_ready",
                "Dense or full-runtime fallback behavior is not bounded for the same prompt set.",
                fallback_summary.get("blocker"),
            )
        )

    live_proof_ready = live_proof_summary.get("proof_ready") is True
    if not live_proof_ready:
        reasons.append(
            reason(
                "live_capability_proof_not_ready",
                "Live residency observation/control and cleanup/restore proof is not ready.",
                live_proof_summary.get("blockers", []),
            )
        )
    if managed_summary.get("live_expert_loading_implemented") is not True and not live_proof_ready:
        reasons.append(reason("live_actuator_missing", "No live expert residency actuator is implemented."))

    unavailable = unavailable_live_capabilities(managed_summary)
    if unavailable and not live_proof_ready:
        reasons.append(
            reason(
                "live_actuator_capabilities_unavailable",
                "Required live-actuator capabilities are not all available.",
                unavailable,
            )
        )
    return reasons


def build_decision_summary(
    trace_path: Path,
    inventory_path: Path,
    policies_path: Path,
    managed_plan_path: Path,
    fallback_artifact_path: Path | None = None,
    live_proof_artifact_path: Path | None = None,
    policy_candidate_trace_receipt_path: Path | None = None,
    real_evidence_root: Path | None = DEFAULT_REAL_EVIDENCE_ROOT,
    expected_live_proof_model_id: str | None = None,
    expected_live_proof_backend_family: str | None = None,
    expected_live_proof_prompt_family: str | None = None,
    expected_live_proof_source_bundle_path: Path | None = None,
    repo_gate_summaries: RepoGateSummaries | None = None,
) -> JSONDict:
    replay_summary = plan_baseline_policy_replay.build_replay_summary(
        trace_path,
        inventory_path,
        policies_path,
    )
    pairing_summary = plan_real_model_trace_inventory_pairing.build_pairing_summary(
        trace_path,
        inventory_path,
    )
    fallback_summary = plan_dense_fallback_comparison.build_summary(fallback_artifact_path)
    live_proof_summary = plan_phase3_live_capability_proof.build_summary(
        live_proof_artifact_path,
        expected_model_id=expected_live_proof_model_id,
        expected_backend_family=expected_live_proof_backend_family,
        expected_prompt_family=expected_live_proof_prompt_family,
        expected_source_bundle_path=expected_live_proof_source_bundle_path,
    )
    if repo_gate_summaries is not None:
        if len(repo_gate_summaries) == 6:
            (
                real_evidence_summary,
                handoff_summary,
                launch_card_summary,
                runtime_request_summary,
                capture_intake_summary,
                reuse_summary,
            ) = repo_gate_summaries
        elif len(repo_gate_summaries) == 5:
            real_evidence_summary, handoff_summary, runtime_request_summary, capture_intake_summary, reuse_summary = repo_gate_summaries
            launch_card_summary = (
                plan_phase3_launch_card_library.build_library(real_evidence_root)
                if real_evidence_root is not None
                else None
            )
        elif len(repo_gate_summaries) == 4:
            real_evidence_summary, handoff_summary, runtime_request_summary, capture_intake_summary = repo_gate_summaries
            reuse_summary = (
                plan_phase3_reuse_evidence_capture.build_root_summary(real_evidence_root)
                if real_evidence_root is not None
                else None
            )
            launch_card_summary = (
                plan_phase3_launch_card_library.build_library(real_evidence_root)
                if real_evidence_root is not None
                else None
            )
        else:
            raise ValueError("repo_gate_summaries must contain four legacy summaries, five summaries including reuse evidence, or six summaries including launch-card library")
    elif real_evidence_root is not None:
        (
            real_evidence_summary,
            handoff_summary,
            launch_card_summary,
            runtime_request_summary,
            capture_intake_summary,
            reuse_summary,
        ) = build_repo_gate_summaries(real_evidence_root)
    else:
        real_evidence_summary = None
        handoff_summary = None
        launch_card_summary = None
        runtime_request_summary = None
        capture_intake_summary = None
        reuse_summary = None
    managed_plan = plan_managed_expert_loading.load_plan(managed_plan_path)
    managed_summary = plan_managed_expert_loading.build_summary(managed_plan, managed_plan_path)
    runtime_actuator_spike_summary = plan_phase3_runtime_actuator_spike.build_summary(
        managed_plan,
        managed_plan_path,
    )
    policy_trace_receipt_summary = policy_candidate_trace_receipt_summary(
        policy_candidate_trace_receipt_path,
        trace_path=trace_path,
        expected_model_id=expected_live_proof_model_id,
        expected_backend_family=expected_live_proof_backend_family,
        expected_prompt_family=expected_live_proof_prompt_family,
    )
    policy_trace_receipt_gate_summary = policy_candidate_trace_receipt_gate_summary(
        policy_trace_receipt_summary,
        real_evidence_summary,
        reuse_summary,
    )

    errors: list[str] = []
    for label, summary in (
        ("policy replay", replay_summary),
        ("trace inventory pairing", pairing_summary),
        ("fallback comparison", fallback_summary),
        ("managed expert loading", managed_summary),
        ("runtime actuator spike", runtime_actuator_spike_summary),
        ("live capability proof", live_proof_summary),
    ):
        if not summary.get("valid"):
            errors.extend(f"{label}: {error}" for error in summary.get("errors", []))
    if real_evidence_summary is not None and real_evidence_summary.get("valid") is not True:
        errors.extend(f"real evidence matrix: {error}" for error in real_evidence_summary.get("errors", []))
    if capture_intake_summary is not None and capture_intake_summary.get("valid") is not True:
        errors.extend(f"capture-result intake: {error}" for error in capture_intake_summary.get("errors", []))
    if handoff_summary is not None and handoff_summary.get("valid") is not True:
        errors.extend(f"handoff coverage: {error}" for error in handoff_summary.get("errors", []))
    if launch_card_summary is not None and launch_card_summary.get("valid") is not True:
        errors.extend(f"launch-card library: {error}" for error in launch_card_summary.get("errors", []))
    if runtime_request_summary is not None and runtime_request_summary.get("valid") is not True:
        errors.extend(f"runtime-capture request audit: {error}" for error in runtime_request_summary.get("errors", []))
    if reuse_summary is not None and reuse_summary.get("valid") is not True:
        errors.extend(f"reuse-evidence capture: {error}" for error in reuse_summary.get("errors", []))
    if policy_trace_receipt_summary.get("errors"):
        errors.extend(
            f"policy candidate trace receipt: {error}"
            for error in policy_trace_receipt_summary.get("errors", [])
        )

    no_go_reasons = collect_no_go_reasons(
        replay_summary,
        pairing_summary,
        fallback_summary,
        managed_summary,
        live_proof_summary,
        real_evidence_summary,
        capture_intake_summary,
        handoff_summary,
        runtime_request_summary,
        policy_trace_receipt_gate_summary,
        reuse_summary,
        launch_card_summary,
    )
    if real_evidence_summary is not None:
        real_bundle_count = real_evidence_summary.get("bundle_count", 0)
        real_model_pair_ready = real_bundle_count > 0 and real_evidence_summary.get("real_model_pair_ready_count") == real_bundle_count
        policy_candidate_ready = real_evidence_summary.get("policy_candidate_ready_count", 0) > 0
        policy_replay_metrics_ready = real_bundle_count > 0 and real_evidence_summary.get("replay_valid_count") == real_bundle_count
        candidate_policy_ids = sorted(
            {
                policy_id
                for bundle in real_evidence_summary.get("bundles", [])
                for policy_id in bundle.get("candidate_policy_ids", [])
            }
        )
    else:
        real_model_pair_ready = bool(pairing_summary.get("real_model_pair_ready"))
        policy_candidate_ready = bool(replay_summary.get("candidate_policy_ids"))
        policy_replay_metrics_ready = replay_summary.get("phase_3_gate", {}).get("policy_replay_metrics_ready", False)
        candidate_policy_ids = replay_summary.get("candidate_policy_ids", [])

    policy_candidate_gate_evidence = policy_candidate_evidence(real_evidence_summary, replay_summary)
    intake_ready = capture_intake_ready(capture_intake_summary)
    handoff_gate_ready = handoff_ready(handoff_summary)
    runtime_gate_ready = runtime_request_audit_ready(runtime_request_summary)
    launch_card_gate_ready = launch_card_library_ready(launch_card_summary)
    policy_candidate_trace_receipt_ready = policy_trace_receipt_gate_summary.get("ready") is True
    ready_for_phase4_adapter_spike = (
        not errors
        and policy_candidate_ready
        and real_model_pair_ready
        and policy_candidate_trace_receipt_ready
        and bool(fallback_summary.get("comparison_ready"))
        and (intake_ready is not False)
        and (handoff_gate_ready is not False)
        and (launch_card_gate_ready is not False)
        and (runtime_gate_ready is not False)
    )
    live_capability_ready = (
        live_proof_summary.get("proof_ready") is True
        or (
            managed_summary.get("live_expert_loading_implemented") is True
            and not unavailable_live_capabilities(managed_summary)
        )
    )
    ready_for_live_spike = ready_for_phase4_adapter_spike and live_capability_ready
    decision = "go_live_spike" if ready_for_live_spike else "no_go_live_spike"
    intake_evidence = capture_intake_evidence(capture_intake_summary)
    handoff_gate_evidence = handoff_evidence(handoff_summary)
    launch_card_gate_evidence = launch_card_library_evidence(launch_card_summary)
    runtime_gate_evidence = runtime_request_evidence(runtime_request_summary)
    reuse_gate_ready = reuse_evidence_ready(reuse_summary)
    reuse_gate_evidence = reuse_evidence(reuse_summary)
    runtime_actuator_spike_handoff_ready = runtime_actuator_spike_summary.get("spike_handoff_ready") is True

    return {
        "mode": "phase3_go_no_go_decision",
        "trace_path": str(trace_path),
        "inventory_path": str(inventory_path),
        "policies_path": str(policies_path),
        "managed_plan_path": str(managed_plan_path),
        "real_evidence_root": str(real_evidence_root) if real_evidence_root is not None else None,
        "fallback_artifact_path": str(fallback_artifact_path) if fallback_artifact_path is not None else None,
        "live_proof_artifact_path": str(live_proof_artifact_path) if live_proof_artifact_path is not None else None,
        "policy_candidate_trace_receipt_path": str(policy_candidate_trace_receipt_path) if policy_candidate_trace_receipt_path is not None else None,
        "valid": not errors,
        "errors": errors,
        "decision": decision,
        "ready_for_phase4_adapter_spike": ready_for_phase4_adapter_spike,
        "ready_for_live_spike": ready_for_live_spike,
        "candidate_policy_ids": candidate_policy_ids,
        "policy_candidate_evidence": policy_candidate_gate_evidence,
        "no_go_reasons": no_go_reasons,
        "gate_status": {
            "policy_replay_metrics_ready": policy_replay_metrics_ready,
            "real_model_pair_ready": real_model_pair_ready,
            "policy_candidate_trace_receipt_ready": policy_candidate_trace_receipt_ready,
            "policy_candidate_trace_receipt_source": policy_trace_receipt_gate_summary.get("source"),
            "policy_candidate_trace_receipt_path": policy_trace_receipt_gate_summary.get("path"),
            "policy_candidate_trace_receipt_exists": policy_trace_receipt_gate_summary.get("exists"),
            "policy_candidate_trace_receipt_required_count": policy_trace_receipt_gate_summary.get("required_count"),
            "policy_candidate_trace_receipt_ready_count": policy_trace_receipt_gate_summary.get("ready_count"),
            "policy_candidate_trace_receipt_attached_count": policy_trace_receipt_gate_summary.get("attached_count"),
            "policy_candidate_trace_receipt_blocked_bundle_count": policy_trace_receipt_gate_summary.get("blocked_bundle_count"),
            "policy_candidate_trace_receipt_error_count": len(policy_trace_receipt_gate_summary.get("errors", [])),
            "policy_candidate_trace_receipt_scaffold_ready": policy_trace_receipt_gate_summary.get("scaffold_ready"),
            "policy_candidate_trace_receipt_scaffold_source": policy_trace_receipt_gate_summary.get("scaffold_source"),
            "policy_candidate_trace_receipt_scaffold_required_count": policy_trace_receipt_gate_summary.get("scaffold_required_count"),
            "policy_candidate_trace_receipt_scaffold_ready_count": policy_trace_receipt_gate_summary.get("scaffold_ready_count"),
            "policy_candidate_trace_receipt_scaffold_valid_count": policy_trace_receipt_gate_summary.get("scaffold_valid_count"),
            "policy_candidate_trace_receipt_scaffold_blocked_count": policy_trace_receipt_gate_summary.get("scaffold_blocked_count"),
            "reuse_evidence_capture_ready": reuse_gate_ready,
            "reuse_evidence_reuse_ready_count": reuse_gate_evidence["reuse_ready_count"],
            "reuse_evidence_reuse_blocked_count": reuse_gate_evidence["reuse_blocked_count"],
            "reuse_evidence_candidate_trace_valid_count": reuse_gate_evidence["candidate_trace_valid_count"],
            "reuse_evidence_candidate_trace_receipt_ready_count": reuse_gate_evidence["candidate_trace_receipt_ready_count"],
            "reuse_evidence_no_reuse_distance_observation_count": reuse_gate_evidence["no_reuse_distance_observation_count"],
            "reuse_evidence_prompt_identity_ready_count": reuse_gate_evidence["prompt_identity_ready_count"],
            "reuse_evidence_prompt_identity_metadata_missing_count": reuse_gate_evidence["prompt_identity_metadata_missing_count"],
            "capture_result_intake_ready": intake_ready,
            "capture_result_ready_to_update_bundle_count": intake_evidence["ready_to_update_bundle_count"],
            "capture_result_ready_for_operator_capture_flag_count": intake_evidence["request_ready_for_operator_capture_flag_count"],
            "capture_result_approved_but_capture_incomplete_request_count": intake_evidence["approved_but_capture_incomplete_request_count"],
            "capture_result_runtime_approval_missing_request_count": intake_evidence["runtime_approval_missing_request_count"],
            "capture_result_approval_rebuild_command_available_request_count": intake_evidence["approval_rebuild_command_available_request_count"],
            "capture_result_approval_rebuild_command_manifest_count": intake_evidence["approval_rebuild_command_manifest_count"],
            "capture_result_approved_runtime_capture_pending_request_count": intake_evidence["approved_runtime_capture_pending_request_count"],
            "capture_result_capture_receipts_ready_count": intake_evidence["capture_receipt_ready_count"],
            "capture_result_capture_receipts_required_count": intake_evidence["capture_receipt_required_count"],
            "capture_result_missing_receipt_gate_request_count": intake_evidence["capture_receipt_missing_request_count"],
            "capture_result_candidate_trace_receipt_ready_count": intake_evidence["candidate_trace_receipt_ready_count"],
            "capture_result_managed_output_receipt_ready_count": intake_evidence["managed_output_receipt_ready_count"],
            "capture_result_dense_output_receipt_ready_count": intake_evidence["dense_output_receipt_ready_count"],
            "capture_result_live_capability_proof_ready_count": intake_evidence["live_capability_proof_ready_count"],
            "capture_result_all_capture_receipts_ready": intake_evidence["all_capture_receipts_ready"],
            "capture_result_all_output_receipt_bindings_ready": intake_evidence["all_output_receipt_bindings_ready"],
            "capture_result_receipt_fill_candidate_router_trace_entry_count": intake_evidence["receipt_fill_candidate_router_trace_entry_count"],
            "capture_result_receipt_fill_candidate_router_trace_ready_count": intake_evidence["receipt_fill_candidate_router_trace_ready_count"],
            "capture_result_receipt_fill_candidate_router_trace_missing_count": intake_evidence["receipt_fill_candidate_router_trace_missing_count"],
            "capture_result_receipt_fill_managed_output_entry_count": intake_evidence["receipt_fill_managed_output_entry_count"],
            "capture_result_receipt_fill_managed_output_ready_count": intake_evidence["receipt_fill_managed_output_ready_count"],
            "capture_result_receipt_fill_managed_output_missing_count": intake_evidence["receipt_fill_managed_output_missing_count"],
            "capture_result_receipt_fill_dense_output_entry_count": intake_evidence["receipt_fill_dense_output_entry_count"],
            "capture_result_receipt_fill_dense_output_ready_count": intake_evidence["receipt_fill_dense_output_ready_count"],
            "capture_result_receipt_fill_dense_output_missing_count": intake_evidence["receipt_fill_dense_output_missing_count"],
            "capture_result_output_receipt_binding_ready_count": intake_evidence["output_receipt_binding_ready_count"],
            "handoff_scaffold_coverage_ready": handoff_gate_ready,
            "handoff_scaffold_ready_count": handoff_gate_evidence["handoff_scaffold_ready_count"],
            "handoff_bundle_count": handoff_gate_evidence["bundle_count"],
            "handoff_runtime_capture_launch_card_template_ready_count": handoff_gate_evidence["runtime_capture_launch_card_template_ready_count"],
            "handoff_runtime_capture_launch_card_binding_ready_count": handoff_gate_evidence["runtime_capture_launch_card_binding_ready_count"],
            "handoff_runtime_capture_launch_card_runtime_command_ready_count": handoff_gate_evidence["runtime_capture_launch_card_runtime_command_ready_count"],
            "handoff_runtime_capture_launch_card_command_option_count": handoff_gate_evidence["runtime_capture_launch_card_command_option_count"],
            "handoff_runtime_capture_launch_card_missing_runtime_command_count": handoff_gate_evidence["runtime_capture_launch_card_missing_runtime_command_count"],
            "launch_card_library_ready": launch_card_gate_ready,
            "launch_card_library_card_count": launch_card_gate_evidence["card_count"],
            "launch_card_library_task_count": launch_card_gate_evidence["task_count"],
            "launch_card_binding_handoff_ready": launch_card_gate_evidence["binding_handoff_ready"],
            "launch_card_binding_handoff_task_count": launch_card_gate_evidence["binding_handoff_task_count"],
            "launch_card_model_plane_contract_request_ready": launch_card_gate_evidence["model_plane_artifact_writer_contract_request_ready"],
            "launch_card_model_plane_contract_request_task_count": launch_card_gate_evidence["model_plane_artifact_writer_contract_request_task_count"],
            "launch_card_saved_handoff_artifacts_ready": launch_card_gate_evidence["saved_handoff_artifacts_ready"],
            "launch_card_saved_handoff_artifact_missing_count": launch_card_gate_evidence["saved_handoff_artifact_missing_count"],
            "launch_card_saved_handoff_artifact_drifted_count": launch_card_gate_evidence["saved_handoff_artifact_drifted_count"],
            "launch_card_missing_runtime_command_count": launch_card_gate_evidence["missing_runtime_command_count"],
            "runtime_capture_request_audit_ready": runtime_gate_ready,
            "runtime_capture_request_count": runtime_gate_evidence["request_count"],
            "runtime_capture_drifted_request_count": runtime_gate_evidence["drifted_request_count"],
            "runtime_capture_validator_command_count": runtime_gate_evidence["validator_command_count"],
            "runtime_capture_validator_command_missing_request_count": runtime_gate_evidence["validator_command_missing_request_count"],
            "dense_fallback_comparison_ready": fallback_summary.get("comparison_ready", False),
            "live_actuator_available": managed_summary.get("live_expert_loading_implemented") is True,
            "runtime_actuator_spike_handoff_ready": runtime_actuator_spike_handoff_ready,
            "runtime_actuator_spike_live_ready": runtime_actuator_spike_summary.get("live_spike_ready") is True,
            "runtime_actuator_spike_proof_requirement_count": runtime_actuator_spike_summary.get("proof_requirement_count"),
            "runtime_actuator_spike_proof_artifact_count": runtime_actuator_spike_summary.get("proof_artifact_count"),
            "runtime_actuator_spike_dependency_edge_count": runtime_actuator_spike_summary.get("dependency_edge_count"),
            "runtime_actuator_spike_blocking_capability_count": runtime_actuator_spike_summary.get("blocking_capability_count"),
            "runtime_actuator_spike_control_blocker_count": runtime_actuator_spike_summary.get("control_blocker_count"),
            "live_capability_proof_ready": live_proof_summary.get("proof_ready") is True,
            "unavailable_live_capabilities": unavailable_live_capabilities(managed_summary),
            "real_evidence_bundle_count": real_evidence_summary.get("bundle_count", 0) if real_evidence_summary else None,
            "real_evidence_policy_candidate_ready_count": (
                real_evidence_summary.get("policy_candidate_ready_count", 0) if real_evidence_summary else None
            ),
            "policy_candidate_blocked_bundle_count": policy_candidate_gate_evidence.get("blocked_bundle_count"),
            "policy_candidate_no_reuse_distance_observation_count": policy_candidate_gate_evidence.get("no_reuse_distance_observation_count"),
        },
        "input_summaries": {
            "replay": {
                "joined_route_count": replay_summary.get("joined_route_count"),
                "unique_expert_count": replay_summary.get("unique_expert_count"),
                "candidate_policy_ids": replay_summary.get("candidate_policy_ids", []),
            },
            "pairing": {
                "decision_pairing_basis": (
                    "repo_real_evidence_matrix" if real_evidence_summary is not None else "default_trace_inventory_pairing"
                ),
                "decision_real_model_pair_ready": real_model_pair_ready,
                "fixture_only_pair": pairing_summary.get("fixture_only_pair"),
                "default_trace_inventory_pair_ready": pairing_summary.get("real_model_pair_ready"),
                "blockers": pairing_summary.get("blockers", []),
            },
            "real_evidence_matrix": {
                "root": real_evidence_summary.get("root") if real_evidence_summary else None,
                "bundle_count": real_evidence_summary.get("bundle_count") if real_evidence_summary else None,
                "real_model_pair_ready_count": (
                    real_evidence_summary.get("real_model_pair_ready_count") if real_evidence_summary else None
                ),
                "replay_valid_count": real_evidence_summary.get("replay_valid_count") if real_evidence_summary else None,
                "policy_candidate_ready_count": (
                    real_evidence_summary.get("policy_candidate_ready_count") if real_evidence_summary else None
                ),
                "policy_candidate_blocked_bundle_count": policy_candidate_gate_evidence.get("blocked_bundle_count"),
                "policy_candidate_blocker_counts": policy_candidate_gate_evidence.get("blocker_counts", {}),
                "remaining_blockers": real_evidence_summary.get("remaining_blockers", []) if real_evidence_summary else [],
            },
            "policy_candidate": policy_candidate_gate_evidence,
            "policy_candidate_trace_receipt": policy_trace_receipt_gate_summary,
            "candidate_trace_receipt_path_validation": policy_trace_receipt_summary,
            "reuse_evidence_capture": reuse_gate_evidence,
            "capture_result_intake": intake_evidence,
            "handoff_scaffold_coverage": handoff_gate_evidence,
            "launch_card_library": launch_card_gate_evidence,
            "runtime_capture_request_audit": runtime_gate_evidence,
            "fallback": {
                "comparison_available": fallback_summary.get("comparison_available"),
                "comparison_ready": fallback_summary.get("comparison_ready"),
                "blocker": fallback_summary.get("blocker"),
            },
            "managed_loading": {
                "live_expert_loading_implemented": managed_summary.get("live_expert_loading_implemented"),
                "capability_statuses": managed_summary.get("capability_statuses", {}),
            },
            "runtime_actuator_spike": {
                "spike_handoff_ready": runtime_actuator_spike_summary.get("spike_handoff_ready"),
                "live_spike_ready": runtime_actuator_spike_summary.get("live_spike_ready"),
                "proof_requirement_count": runtime_actuator_spike_summary.get("proof_requirement_count"),
                "proof_artifact_count": runtime_actuator_spike_summary.get("proof_artifact_count"),
                "dependency_edge_count": runtime_actuator_spike_summary.get("dependency_edge_count"),
                "blocking_capability_count": runtime_actuator_spike_summary.get("blocking_capability_count"),
                "control_blocker_count": runtime_actuator_spike_summary.get("control_blocker_count"),
            },
            "live_capability_proof": {
                "proof_available": live_proof_summary.get("proof_available"),
                "proof_ready": live_proof_summary.get("proof_ready"),
                "blockers": live_proof_summary.get("blockers", []),
                "context_binding": live_proof_summary.get("context_binding", {}),
            },
        },
        "safety_contract": [
            "planner aggregates local Phase 3 evidence only",
            "planner does not launch model servers",
            "planner does not run Docker",
            "planner does not load tensor values",
            "planner does not mutate runtime residency",
            "planner does not send prompt traffic",
            "planner does not claim live expert paging",
        ],
        "next_actions": [
            "Keep the current decision as no-go for live expert paging.",
            "Capture an approved candidate router trace for one real bundle that uses the generated prompt set and produces reuse-distance observations.",
            "Capture dense/full-runtime fallback outputs for the same prompt set after approval.",
            "Run capture-result intake and rebuild the bundle only after receipt-bound artifacts are filled.",
            "Use the runtime-actuator spike handoff to prove inventory, routing visibility, residency observation, fallback, artifact export, residency control, and cleanup before live mutation.",
        ],
    }


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def evidence_to_text(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def format_markdown_report(summary: JSONDict) -> str:
    candidate_policy_ids = summary.get("candidate_policy_ids") if isinstance(summary.get("candidate_policy_ids"), list) else []
    candidate_text = ", ".join(str(item) for item in candidate_policy_ids) if candidate_policy_ids else "none"
    lines = [
        "# Phase 3 Go/No-Go Decision",
        "",
        f"- Valid: `{summary.get('valid')}`",
        f"- Decision: `{markdown_escape(summary.get('decision'))}`",
        f"- Ready for Phase 4 adapter spike: `{summary.get('ready_for_phase4_adapter_spike')}`",
        f"- Ready for live spike: `{summary.get('ready_for_live_spike')}`",
        f"- Candidate policies: `{markdown_escape(candidate_text)}`",
        f"- Real evidence root: `{markdown_escape(summary.get('real_evidence_root'))}`",
        "",
        "## Gate Status",
        "",
        "| Gate | Status |",
        "| --- | --- |",
    ]
    gate_status = summary.get("gate_status") if isinstance(summary.get("gate_status"), dict) else {}
    for key, value in gate_status.items():
        lines.append(f"| `{markdown_escape(key)}` | `{markdown_escape(evidence_to_text(value))}` |")

    errors = summary.get("errors") if isinstance(summary.get("errors"), list) else []
    if errors:
        lines.extend(["", "## Errors", ""])
        for error in errors:
            lines.append(f"- {markdown_escape(error)}")

    lines.extend(["", "## No-Go Reasons", ""])
    no_go_reasons = summary.get("no_go_reasons") if isinstance(summary.get("no_go_reasons"), list) else []
    if no_go_reasons:
        lines.extend(["| Reason | Description | Evidence |", "| --- | --- | --- |"])
        for item in no_go_reasons:
            if not isinstance(item, dict):
                lines.append(f"| `{markdown_escape(item)}` |  |  |")
                continue
            evidence = evidence_to_text(item.get("evidence"))
            lines.append(
                "| "
                f"`{markdown_escape(item.get('id') or 'unknown')}` | "
                f"{markdown_escape(item.get('description') or '')} | "
                f"{markdown_escape(evidence)} |"
            )
    else:
        lines.append("- none")

    input_summaries = summary.get("input_summaries") if isinstance(summary.get("input_summaries"), dict) else {}
    lines.extend(["", "## Evidence Summary", "", "| Source | Key Status |", "| --- | --- |"])
    for source_id, source_summary in input_summaries.items():
        lines.append(f"| `{markdown_escape(source_id)}` | `{markdown_escape(evidence_to_text(source_summary))}` |")

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
    print("MoE Run Anyway Phase 3 go/no-go decision")
    print(f"Decision: {summary['decision']}")
    print(f"Valid: {summary['valid']}")
    print(f"Ready for Phase 4 adapter spike: {summary['ready_for_phase4_adapter_spike']}")
    print(f"Ready for live spike: {summary['ready_for_live_spike']}")
    if summary["errors"]:
        print("Errors:")
        for error in summary["errors"]:
            print(f"  - {error}")
    print("No-go reasons:")
    for item in summary["no_go_reasons"]:
        print(f"  - {item['id']}")
    print("Gate status:")
    for key, value in summary["gate_status"].items():
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
    parser.add_argument("--policy-candidate-trace-receipt-path", type=Path, default=None)
    parser.add_argument("--real-evidence-root", type=Path, default=DEFAULT_REAL_EVIDENCE_ROOT)
    parser.add_argument("--no-real-evidence-matrix", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    parser.add_argument("--output-md", type=Path, help="write a Markdown go/no-go decision report")
    return parser


def plan_paths(
    trace_path: Path,
    inventory_path: Path,
    policies_path: Path,
    managed_plan_path: Path,
    *,
    fallback_artifact_path: Path | None = None,
    live_proof_artifact_path: Path | None = None,
    policy_candidate_trace_receipt_path: Path | None = None,
    real_evidence_root: Path | None = DEFAULT_REAL_EVIDENCE_ROOT,
) -> tuple[int, JSONDict | None, str | None]:
    try:
        summary = build_decision_summary(
            trace_path,
            inventory_path,
            policies_path,
            managed_plan_path,
            fallback_artifact_path=fallback_artifact_path,
            live_proof_artifact_path=live_proof_artifact_path,
            policy_candidate_trace_receipt_path=policy_candidate_trace_receipt_path,
            real_evidence_root=real_evidence_root,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not build Phase 3 go/no-go decision: {exc}"
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
        policy_candidate_trace_receipt_path=args.policy_candidate_trace_receipt_path,
        real_evidence_root=None if args.no_real_evidence_matrix else args.real_evidence_root,
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

