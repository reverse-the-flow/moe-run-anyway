import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_phase3_evidence_packet.py"
SPEC = importlib.util.spec_from_file_location("plan_phase3_evidence_packet", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)

TRACE_FIXTURE = ROOT / "memory-moe-mvp" / "data" / "llama_cpp_router_trace.fixture.jsonl"
INVENTORY_FIXTURE = ROOT / "memory-moe-mvp" / "data" / "expert_inventory_manifest.fixture.json"
POLICIES_FIXTURE = ROOT / "memory-moe-mvp" / "data" / "baseline_replay_policies.json"
MANAGED_PLAN = ROOT / "memory-moe-mvp" / "data" / "managed_expert_loading_plan.json"
MIXTRAL_BUNDLE_PATH = "memory-moe-mvp/phase3-real-evidence/pc_mixtral_phase3_real_evidence_bundle.json"
MIXTRAL_REQUEST_PATH = "memory-moe-mvp/phase3-real-evidence/pc_mixtral_phase3_real_evidence_bundle.runtime-capture-request.json"
MIXTRAL_PROMPT_SET_PATH = "memory-moe-mvp/phase3-real-evidence/pc_mixtral_phase3_real_evidence_bundle.prompt-set.json"
MIXTRAL_CANDIDATE_TRACE_PATH = "memory-moe-mvp/phase3-real-evidence/pc_mixtral_phase3_real_evidence_bundle-policy-candidate/candidate-router-events.jsonl"
MIXTRAL_CANDIDATE_RECEIPT_PATH = "memory-moe-mvp/phase3-real-evidence/pc_mixtral_phase3_real_evidence_bundle-policy-candidate/candidate-router-events.capture-receipt.json"
MIXTRAL_MANAGED_OUTPUT_PATH = "memory-moe-mvp/phase3-real-evidence/pc_mixtral_phase3_real_evidence_bundle-fallback/managed-output-summary.json"
MIXTRAL_DENSE_OUTPUT_PATH = "memory-moe-mvp/phase3-real-evidence/pc_mixtral_phase3_real_evidence_bundle-fallback/dense-output-summary.json"
MIXTRAL_FALLBACK_COMPARISON_PATH = "memory-moe-mvp/phase3-real-evidence/pc_mixtral_phase3_real_evidence_bundle-fallback/dense-fallback-comparison.json"
MIXTRAL_LIVE_PROOF_TEMPLATE_PATH = "memory-moe-mvp/phase3-real-evidence/pc_mixtral_phase3_real_evidence_bundle.live-capability-proof.template.json"


def fallback_artifact():
    return {
        "schema_version": "moe-dense-fallback-comparison-v1",
        "model_id": "fixture-mixtral.gguf",
        "prompt_family": "fixture",
        "managed_policy_id": "preload_shortlist",
        "managed_artifact": "memory-moe-mvp/replay/managed.json",
        "dense_artifact": "memory-moe-mvp/replay/dense.json",
        "builder": {
            "schema_version": "moe-dense-fallback-comparison-builder-v1",
            "mode": "paired_output_summary",
            "input_receipts": {
                "managed_capture_receipt_ready": True,
                "dense_capture_receipt_ready": True,
                "receipt_pair_consistent": True,
                "receipt_gate": "required_before_write",
            },
        },
        "comparisons": [
            {
                "prompt_id": "case-001",
                "managed_output_present": True,
                "dense_output_present": True,
                "quality_delta_label": "same",
            }
        ],
    }


def live_proof_artifact(*, artifact_paths: list[str] | None = None):
    return {
        "schema_version": "moe-phase3-live-capability-proof-v1",
        "name": "Fixture live capability proof",
        "model_id": "fixture-mixtral.gguf",
        "backend_family": "llama_cpp",
        "prompt_family": "fixture",
        "source_bundle_path": "memory-moe-mvp/data/phase3_real_evidence_bundle.fixture.json",
        "proof_scope": "fixture-adapter-smoke",
        "residency_observation": {
            "status": "available",
            "before_state_captured": True,
            "after_state_captured": True,
            "evidence_fields": ["resident_expert_count", "resident_bytes", "layer_expert_keys"],
        },
        "residency_control": {
            "status": "available",
            "supported_actions": ["observe", "preload", "evict", "restore_dense"],
            "actuator_boundary": "fixture-adapter",
            "dry_run_only": False,
        },
        "cleanup_restore": {
            "status": "available",
            "restore_verified": True,
            "cleanup_actions": ["restore_dense", "clear_policy_state"],
            "failure_path_tested": True,
        },
        "artifact_export": {
            "status": "available",
            "artifact_paths": artifact_paths or ["memory-moe-mvp/data/phase3_real_evidence_bundle.fixture.json"],
        },
        "safety_contract": ["Fixture validates metadata only."],
        "next_actions": ["Use a real backend proof before live mutation."],
    }



def approval_manifest_entries() -> list[dict]:
    approval_keys = [
        "runtime_prompt_traffic_approved",
        "router_trace_capture_approved",
    ]
    entries: list[dict] = []
    for index in range(6):
        request_path = f"memory-moe-mvp/phase3-real-evidence/fixture-{index}.runtime-capture-request.json"
        bundle_path = f"memory-moe-mvp/phase3-real-evidence/fixture-{index}.json"
        entries.append(
            {
                "request_name": f"Fixture request {index}",
                "request_path": request_path,
                "bundle_path": bundle_path,
                "status": "approval_required",
                "next_artifact_id": "candidate_router_trace",
                "command_class": "phase3_runtime_capture_request_approval_rebuild",
                "writes_request_path": request_path,
                "records_approval_keys": list(approval_keys),
                "requires_explicit_user_approval": True,
                "metadata_only": True,
                "command": [
                    "uv",
                    "run",
                    "--managed-python",
                    "--python",
                    "3.13",
                    "scripts/build_phase3_runtime_capture_request.py",
                    bundle_path,
                    "--output",
                    request_path,
                    "--json",
                    "--runtime-prompt-traffic-approved",
                    "--router-trace-capture-approved",
                ],
            }
        )
    return entries



def fixture_paths(index: int) -> dict:
    if index == 0:
        slug = "fixture-mixtral"
        artifact_dir = "fixture"
        name = "Fixture Mixtral Phase 3 real evidence"
        rationale = "mixtral_first_known_sparse_baseline"
    else:
        slug = f"fixture-{index}"
        artifact_dir = slug
        name = f"Fixture request {index}"
        rationale = "stable_name_order"
    root = "memory-moe-mvp/phase3-real-evidence"
    return {
        "name": name,
        "slug": slug,
        "artifact_dir": artifact_dir,
        "rationale": rationale,
        "request_path": f"{root}/{slug}.runtime-capture-request.json",
        "bundle_path": f"{root}/{slug}.json",
        "prompt_set_path": f"{root}/{slug}.prompt-set.json",
        "candidate_trace_path": f"{root}/{artifact_dir}/candidate-router-events.jsonl",
        "candidate_receipt_path": f"{root}/{artifact_dir}/candidate-router-events.capture-receipt.json",
        "managed_output_path": f"{root}/{artifact_dir}/managed-output-summary.json",
        "dense_output_path": f"{root}/{artifact_dir}/dense-output-summary.json",
        "live_proof_path": f"{root}/{artifact_dir}-live-proof.template.json",
    }


def fixture_launch_card_binding_tasks() -> list[dict]:
    tasks: list[dict] = []
    for index in range(6):
        paths = fixture_paths(index)
        launch_card_path = f"memory-moe-mvp/phase3-real-evidence/fixture-{index}.runtime-capture-launch-card.template.json"
        for task_id, artifact_id, capture_kind, artifact_path, receipt_path in (
            (
                "capture_candidate_router_trace",
                "candidate_router_trace",
                "llama_cpp_router_trace_jsonl",
                paths["candidate_trace_path"],
                paths["candidate_receipt_path"],
            ),
            (
                "fill_managed_output_summary",
                "managed_output_summary_fill",
                "managed_output_summary_json",
                paths["managed_output_path"],
                paths["managed_output_path"],
            ),
            (
                "fill_dense_output_summary",
                "dense_output_summary_fill",
                "dense_output_summary_json",
                paths["dense_output_path"],
                paths["dense_output_path"],
            ),
        ):
            tasks.append(
                {
                    "request_name": paths["name"],
                    "request_path": paths["request_path"],
                    "launch_card_path": launch_card_path,
                    "model_id": "fixture-mixtral.gguf",
                    "backend_family": "llama_cpp",
                    "prompt_family": "fixture",
                    "task_id": task_id,
                    "artifact_id": artifact_id,
                    "capture_kind": capture_kind,
                    "receipt_kind": "trace_capture_receipt_json" if artifact_id == "candidate_router_trace" else "embedded_output_capture_receipt",
                    "artifact_output_path": artifact_path,
                    "receipt_output_path": receipt_path,
                    "prompt_set_path": paths["prompt_set_path"],
                    "approval_keys": [
                        "runtime_prompt_traffic_approved",
                        "router_trace_capture_approved",
                    ],
                    "requires_explicit_user_approval": True,
                    "requires_prompt_traffic": True,
                    "requires_runtime": True,
                    "validator_command_count": 1,
                    "binding_handoff_ready": True,
                    "binding_ready": False,
                    "command_option_ready": False,
                    "model_plane_callable_id": None,
                    "launch_command": [],
                    "missing_fields": [],
                    "binding_template": {
                        "binding_kind": "model_plane_callable_or_launch_command",
                        "requires_explicit_user_approval": True,
                        "may_send_prompt_traffic": True,
                        "approval_keys": [
                            "runtime_prompt_traffic_approved",
                            "router_trace_capture_approved",
                        ],
                        "writes_artifact_path": artifact_path,
                        "writes_receipt_path": receipt_path,
                    },
                }
            )
    return tasks


def fixture_approval_command_options(paths: dict) -> list[dict]:
    return [
        {
            "command_class": "phase3_runtime_capture_request_approval_rebuild",
            "requires_explicit_user_approval": True,
            "metadata_only": True,
            "command": [
                "uv",
                "run",
                "--managed-python",
                "--python",
                "3.13",
                "scripts/build_phase3_runtime_capture_request.py",
                paths["bundle_path"],
                "--output",
                paths["request_path"],
                "--json",
                "--runtime-prompt-traffic-approved",
                "--router-trace-capture-approved",
            ],
        }
    ]


def fixture_requested_artifacts(index: int) -> list[dict]:
    paths = fixture_paths(index)
    prompt_set_path = paths["prompt_set_path"]
    candidate_trace_path = paths["candidate_trace_path"]
    candidate_receipt_path = paths["candidate_receipt_path"]
    return [
        {
            "id": "candidate_router_trace",
            "status": "approval_required",
            "path": candidate_trace_path,
            "validator_commands": [
                ["uv", "run", "--managed-python", "--python", "3.13", "scripts/validate_llama_cpp_router_trace.py", candidate_trace_path, "--json"],
                ["uv", "run", "--managed-python", "--python", "3.13", "scripts/phase3_trace_receipts.py", candidate_receipt_path, "--json"],
                ["uv", "run", "--managed-python", "--python", "3.13", "scripts/plan_baseline_policy_replay.py", candidate_trace_path, "--json"],
            ],
            "source": {
                "capture_receipt_required": True,
                "capture_receipt_path": candidate_receipt_path,
                "receipt_fill_note": "trace receipt required",
            },
        },
        {
            "id": "managed_output_summary_fill",
            "status": "approval_required",
            "path": paths["managed_output_path"],
            "validator_commands": [
                ["uv", "run", "--managed-python", "--python", "3.13", "scripts/build_phase3_output_summary.py", prompt_set_path, "--output-label", "managed", "--json"],
            ],
            "source": {
                "capture_receipt_required": True,
                "receipt_fill_note": "managed output receipt required",
            },
        },
        {
            "id": "dense_output_summary_fill",
            "status": "approval_required",
            "path": paths["dense_output_path"],
            "validator_commands": [
                ["uv", "run", "--managed-python", "--python", "3.13", "scripts/build_phase3_output_summary.py", prompt_set_path, "--output-label", "dense", "--json"],
            ],
            "source": {
                "capture_receipt_required": True,
                "receipt_fill_note": "dense output receipt required",
            },
        },
    ]


def fixture_runtime_request(index: int) -> dict:
    paths = fixture_paths(index)
    return {
        "name": paths["name"],
        "path": paths["request_path"],
        "bundle_path": paths["bundle_path"],
        "operator_handoff": {
            "requested_artifacts": fixture_requested_artifacts(index),
            "future_artifacts": [
                {
                    "id": "live_capability_proof_fill",
                    "status": "future_adapter_required",
                    "path": paths["live_proof_path"],
                    "validator_commands": [
                        ["uv", "run", "--managed-python", "--python", "3.13", "scripts/plan_phase3_live_capability_proof.py", paths["live_proof_path"], "--json"],
                    ],
                },
            ],
        },
    }


def fixture_runtime_queue_item(index: int) -> dict:
    paths = fixture_paths(index)
    return {
        "request_name": paths["name"],
        "path": paths["request_path"],
        "status": "approval_required",
        "bundle_path": paths["bundle_path"],
        "model_id": f"fixture-model-{index}.gguf",
        "prompt_set_path": paths["prompt_set_path"],
        "prompt_count": 8,
        "backend_family": "llama_cpp",
        "missing_approval_keys": [
            "runtime_prompt_traffic_approved",
            "router_trace_capture_approved",
        ],
        "pending_artifact_ids": [
            "candidate_router_trace",
            "managed_output_summary_fill",
            "dense_output_summary_fill",
        ],
        "pending_artifact_count": 3,
        "missing_approval_count": 2,
        "next_artifact_id": "candidate_router_trace",
        "next_artifact_path": paths["candidate_trace_path"],
        "queue_rank": index + 1,
        "rank": index + 1,
        "selection_rationale": paths["rationale"],
    }


def fixture_capture_fill_plan(index: int) -> dict:
    paths = fixture_paths(index)
    return {
        "preview_only": True,
        "valid": True,
        "request_name": paths["name"],
        "request_path": paths["request_path"],
        "prompt_set_path": paths["prompt_set_path"],
        "rank": index + 1,
        "selection_rationale": paths["rationale"],
        "records_approval_keys": [
            "runtime_prompt_traffic_approved",
            "router_trace_capture_approved",
            "managed_output_capture_approved",
            "dense_output_capture_approved",
        ],
        "requires_explicit_user_approval": True,
        "metadata_only": True,
        "mutates_request": False,
        "artifact_step_count": 3,
        "runtime_capture_step_count": 3,
        "ready_after_approval_count": 0,
        "missing_after_approval_count": 3,
        "validator_command_count": 5,
        "command_option_count": 1,
        "all_receipts_ready_after_approval": False,
        "ready_to_update_bundle_after_approval": False,
        "capture_fill_steps": [
            {
                "artifact_id": "candidate_router_trace",
                "queue_step_id": "candidate_router_trace",
                "receipt_kind": "trace_capture_receipt_json",
                "artifact_path": paths["candidate_trace_path"],
                "receipt_path": paths["candidate_receipt_path"],
                "current_step": {
                    "id": "candidate_router_trace",
                    "stage": "runtime_capture",
                    "status": "approval_required",
                    "approval_state": "missing",
                    "path": paths["candidate_trace_path"],
                },
                "step_after_approval": {
                    "id": "candidate_router_trace",
                    "stage": "runtime_capture",
                    "status": "ready_for_operator_capture",
                    "approval_state": "recorded",
                    "path": paths["candidate_trace_path"],
                },
                "fill_status_after_approval": "blocked_after_approval",
                "ready_after_approval": False,
                "receipt_ready": False,
                "validator_command_count": 3,
                "command_option_count": 1,
                "command_options": fixture_approval_command_options(paths),
                "blockers_after_approval": ["candidate_trace_capture_receipt_not_ready"],
            },
            {
                "artifact_id": "managed_output_summary_fill",
                "queue_step_id": "managed_output_summary",
                "receipt_kind": "embedded_output_capture_receipt",
                "artifact_path": paths["managed_output_path"],
                "receipt_path": paths["managed_output_path"],
                "step_after_approval": {
                    "id": "managed_output_summary",
                    "stage": "runtime_capture",
                    "status": "ready_for_operator_capture",
                    "approval_state": "recorded",
                    "path": paths["managed_output_path"],
                },
                "fill_status_after_approval": "blocked_after_approval",
                "ready_after_approval": False,
                "receipt_ready": False,
                "validator_command_count": 1,
                "command_option_count": 0,
                "blockers_after_approval": ["managed_capture_receipt_not_ready"],
            },
            {
                "artifact_id": "dense_output_summary_fill",
                "queue_step_id": "dense_output_summary",
                "receipt_kind": "embedded_output_capture_receipt",
                "artifact_path": paths["dense_output_path"],
                "receipt_path": paths["dense_output_path"],
                "step_after_approval": {
                    "id": "dense_output_summary",
                    "stage": "runtime_capture",
                    "status": "ready_for_operator_capture",
                    "approval_state": "recorded",
                    "path": paths["dense_output_path"],
                },
                "fill_status_after_approval": "blocked_after_approval",
                "ready_after_approval": False,
                "receipt_ready": False,
                "validator_command_count": 1,
                "command_option_count": 0,
                "blockers_after_approval": ["dense_capture_receipt_not_ready"],
            },
        ],
    }


def fixture_intake_request(index: int) -> dict:
    paths = fixture_paths(index)
    return {
        "name": paths["name"],
        "path": paths["request_path"],
        "candidate_trace_path": paths["candidate_trace_path"],
        "candidate_trace_receipt_path": paths["candidate_receipt_path"],
        "managed_output_path": paths["managed_output_path"],
        "dense_output_path": paths["dense_output_path"],
        "policy_candidate": {
            "candidate_trace_capture_receipt": {
                "path": paths["candidate_receipt_path"],
            },
        },
        "capture_receipt_binding": {
            "managed": {
                "path": paths["managed_output_path"],
            },
            "dense": {
                "path": paths["dense_output_path"],
            },
        },
        "next_operator_step": {
            "id": "candidate_router_trace",
            "stage": "runtime_capture",
            "status": "approval_required",
            "path": paths["candidate_trace_path"],
            "approval_state": "missing",
        },
    }

def repo_gate_summaries_fixture():
    return (
        {
            "valid": True,
            "errors": [],
            "bundle_count": 6,
            "real_model_pair_ready_count": 6,
            "replay_valid_count": 6,
            "policy_candidate_ready_count": 0,
            "policy_candidate_blocked_bundle_count": 6,
            "policy_candidate_no_reuse_distance_observation_count": 6,
            "policy_candidate_prompt_identity_ready_count": 0,
            "policy_candidate_prompt_identity_metadata_missing_count": 6,
            "policy_candidate_blocker_counts": {
                "no_replay_policy_candidate": 6,
                "no_reuse_distance_observations": 6,
                "prompt_identity_metadata_missing": 6,
            },
            "candidate_policy_ids": [],
            "phase4_ready_bundle_count": 0,
            "remaining_blockers": ["no_replay_policy_candidate", "dense_fallback_output_artifact_for_quality_bounds"],
            "bundles": [
                {
                    "name": f"Fixture bundle {index}",
                    "path": f"memory-moe-mvp/phase3-real-evidence/fixture-{index}.json",
                    "model_id": f"fixture-model-{index}.gguf",
                    "replay_valid": True,
                    "reuse_distance_observations": 0,
                    "candidate_policy_ids": [],
                    "policy_candidate_ready": False,
                    "policy_candidate_blockers": [
                        {"id": "no_replay_policy_candidate"},
                        {"id": "no_reuse_distance_observations"},
                        {"id": "prompt_identity_metadata_missing"},
                    ],
                    "policy_candidate_blocker_ids": [
                        "no_replay_policy_candidate",
                        "no_reuse_distance_observations",
                        "prompt_identity_metadata_missing",
                    ],
                }
                for index in range(6)
            ],
        },
        {
            "valid": True,
            "errors": [],
            "bundle_count": 6,
            "handoff_scaffold_ready_count": 6,
            "all_handoff_scaffolds_ready": True,
            "missing_artifact_counts": {},
            "bundles": [
                {
                    "name": f"fixture-launch-card-{index}",
                    "runtime_capture_launch_card_template": {
                        "ready": True,
                        "binding_ready": False,
                        "runtime_capture_command_ready": False,
                        "command_option_count": 0,
                        "missing_runtime_command_count": 3,
                        "missing_runtime_command_artifact_ids": [
                            "candidate_router_trace",
                            "managed_output_summary_fill",
                            "dense_output_summary_fill",
                        ],
                    },
                }
                for index in range(6)
            ],
        },
        {
            "valid": True,
            "errors": [],
            "library_ready": True,
            "execution_ready": False,
            "card_count": 6,
            "template_ready_count": 6,
            "model_plane_binding_ready_count": 0,
            "binding_ready_count": 0,
            "runtime_capture_command_ready_count": 0,
            "task_count": 18,
            "bound_task_count": 0,
            "command_option_count": 0,
            "missing_runtime_command_count": 18,
            "binding_handoff_ready": True,
            "binding_handoff_task_count": 18,
            "binding_handoff_ready_count": 18,
            "binding_handoff_missing_field_count": 0,
            "unbound_task_count": 18,
            "model_plane_artifact_writer_contract_request_ready": True,
            "model_plane_artifact_writer_contract_request_task_count": 18,
            "saved_handoff_artifacts_ready": True,
            "saved_handoff_artifact_missing_count": 0,
            "saved_handoff_artifact_drifted_count": 0,
            "binding_tasks": fixture_launch_card_binding_tasks(),
            "blockers": [
                "launch_card_runtime_command_bindings_missing",
                "model_plane_binding_context_missing",
                "runtime_capture_commands_unbound",
            ],
            "cards": [
                {
                    "request_name": f"Fixture request {index}" if index else "Fixture Mixtral Phase 3 real evidence",
                    "template_ready": True,
                    "model_plane_binding_ready": False,
                    "binding_ready": False,
                    "runtime_capture_command_ready": False,
                    "task_count": 3,
                    "binding_handoff_task_count": 3,
                    "binding_handoff_ready_count": 3,
                    "binding_handoff_missing_field_count": 0,
                    "unbound_task_count": 3,
                    "missing_runtime_command_count": 3,
                    "launch_card_path": f"memory-moe-mvp/phase3-real-evidence/fixture-{index}.runtime-capture-launch-card.template.json",
                }
                for index in range(6)
            ],
        },
        {
            "valid": True,
            "errors": [],
            "request_count": 6,
            "valid_request_count": 6,
            "ready_for_operator_capture_count": 0,
            "capture_complete_count": 0,
            "approval_rebuild_command_manifest_count": 6,
            "approval_rebuild_command_manifest": approval_manifest_entries(),
            "drifted_request_count": 0,
            "validator_command_count": 5,
            "validator_command_missing_request_count": 0,
            "all_required_validator_commands_present": True,
            "requests": [fixture_runtime_request(index) for index in range(6)],
            "approval_queue": [fixture_runtime_queue_item(index) for index in range(6)],
            "recommended_runtime_capture_request": {
                "request_name": "Fixture Mixtral Phase 3 real evidence",
                "path": "memory-moe-mvp/phase3-real-evidence/fixture-mixtral.runtime-capture-request.json",
                "status": "approval_required",
                "bundle_path": "memory-moe-mvp/phase3-real-evidence/fixture-mixtral.json",
                "model_id": "fixture-mixtral.gguf",
                "prompt_set_path": "memory-moe-mvp/phase3-real-evidence/fixture-mixtral.prompt-set.json",
                "prompt_count": 8,
                "missing_approval_keys": [
                    "runtime_prompt_traffic_approved",
                    "router_trace_capture_approved",
                ],
                "pending_artifact_ids": [
                    "candidate_router_trace",
                    "managed_output_summary_fill",
                    "dense_output_summary_fill",
                ],
                "next_artifact_id": "candidate_router_trace",
                "next_artifact_path": "memory-moe-mvp/phase3-real-evidence/fixture/candidate-router-events.jsonl",
                "queue_rank": 1,
                "selection_rationale": "mixtral_first_known_sparse_baseline",
                "approval_rebuild_command": {
                    "command_class": "phase3_runtime_capture_request_approval_rebuild",
                    "command": [
                        "uv",
                        "run",
                        "--managed-python",
                        "--python",
                        "3.13",
                        "scripts/build_phase3_runtime_capture_request.py",
                        "memory-moe-mvp/phase3-real-evidence/fixture-mixtral.json",
                        "--output",
                        "memory-moe-mvp/phase3-real-evidence/fixture-mixtral.runtime-capture-request.json",
                        "--json",
                        "--runtime-prompt-traffic-approved",
                        "--router-trace-capture-approved",
                    ],
                    "records_approval_keys": [
                        "runtime_prompt_traffic_approved",
                        "router_trace_capture_approved",
                    ],
                    "writes_request_path": "memory-moe-mvp/phase3-real-evidence/fixture-mixtral.runtime-capture-request.json",
                    "requires_explicit_user_approval": True,
                    "metadata_only": True,
                },
                "capture_sequence": [
                    {
                        "id": "record_runtime_approvals",
                        "stage": "approval",
                        "status": "approval_required",
                        "approval_keys": [
                            "runtime_prompt_traffic_approved",
                            "router_trace_capture_approved",
                        ],
                    },
                    {
                        "id": "capture_candidate_router_trace",
                        "stage": "runtime_capture",
                        "status": "approval_required",
                        "path": "memory-moe-mvp/phase3-real-evidence/fixture/candidate-router-events.jsonl",
                        "validator_command_count": 3,
                    },
                    {
                        "id": "run_capture_result_intake",
                        "stage": "intake",
                        "status": "pending_artifacts",
                    },
                ],
            },
            "recommended_post_approval_preview": {
                "preview_only": True,
                "valid": True,
                "status": "ready_for_operator_capture",
                "ready_for_operator_capture": True,
                "capture_complete": False,
                "mutates_request": False,
                "still_requires_capture_artifacts": True,
                "pending_artifact_count": 3,
                "pending_artifact_ids": [
                    "candidate_router_trace",
                    "managed_output_summary_fill",
                    "dense_output_summary_fill",
                ],
                "next_artifact_id": "candidate_router_trace",
                "next_artifact_path": "memory-moe-mvp/phase3-real-evidence/fixture/candidate-router-events.jsonl",
                "requested_status_counts": {"ready_for_operator_capture": 3},
                "records_approval_keys": [
                    "runtime_prompt_traffic_approved",
                    "router_trace_capture_approved",
                ],
            },
            "capture_queue_summary": {
                "queue_count": 6,
                "status_counts": {"approval_required": 6},
                "backend_family_counts": {"llama_cpp": 6},
                "prompt_count_min": 8,
                "prompt_count_max": 8,
                "recommended_rank": 1,
                "selection_contract": [
                    "Prefer the known Mixtral sparse baseline first when all capture requests are otherwise equivalent."
                ],
                "ranked_requests": [fixture_runtime_queue_item(index) for index in range(6)],
            },
        },
        {
            "valid": True,
            "errors": [],
            "request_count": 6,
            "request_audit_valid_count": 6,
            "request_drift_free_count": 6,
            "request_drifted_count": 0,
            "request_capture_complete_flag_count": 0,
            "request_ready_for_operator_capture_flag_count": 0,
            "approved_but_capture_incomplete_request_count": 0,
            "runtime_approval_missing_request_count": 6,
            "approval_rebuild_command_available_request_count": 6,
            "approval_rebuild_command_manifest": approval_manifest_entries(),
            "approved_runtime_capture_pending_request_count": 0,
            "capture_receipt_required_count": 18,
            "capture_receipt_ready_count": 0,
            "receipt_fill_entry_count": 18,
            "receipt_fill_ready_count": 0,
            "receipt_fill_missing_count": 18,
            "receipt_fill_approval_missing_count": 18,
            "approval_transition_preview_count": 6,
            "approval_transition_ready_for_operator_count": 6,
            "approval_transition_ready_to_update_bundle_count": 0,
            "recommended_approval_transition_preview": {
                "preview_only": True,
                "valid": True,
                "request_name": "Fixture Mixtral Phase 3 real evidence",
                "request_path": "memory-moe-mvp/phase3-real-evidence/fixture-mixtral.runtime-capture-request.json",
                "rank": 1,
                "selection_rationale": "mixtral_first_known_sparse_baseline",
                "current_next_step": {
                    "id": "candidate_router_trace",
                    "stage": "runtime_capture",
                    "status": "approval_required",
                    "approval_state": "missing",
                    "path": "memory-moe-mvp/phase3-real-evidence/fixture/candidate-router-events.jsonl",
                },
                "next_step_after_approval": {
                    "id": "candidate_router_trace",
                    "stage": "runtime_capture",
                    "status": "ready_for_operator_capture",
                    "approval_state": "recorded",
                    "path": "memory-moe-mvp/phase3-real-evidence/fixture/candidate-router-events.jsonl",
                },
                "records_approval_keys": [
                    "runtime_prompt_traffic_approved",
                    "router_trace_capture_approved",
                ],
                "requires_explicit_user_approval": True,
                "metadata_only": True,
                "mutates_request": False,
                "capture_complete_after_approval": False,
                "approved_but_capture_incomplete_after_approval": True,
                "ready_to_update_bundle_after_approval": False,
                "receipt_fill_preview": {
                    "entry_count": 3,
                    "ready_after_approval_count": 0,
                    "missing_after_approval_count": 3,
                    "approval_missing_after_approval_count": 0,
                    "all_ready_after_approval": False,
                },
            },
            "post_approval_capture_fill_plan_count": 6,
            "post_approval_capture_fill_artifact_step_count": 18,
            "post_approval_capture_fill_runtime_step_count": 18,
            "post_approval_capture_fill_ready_count": 0,
            "post_approval_capture_fill_missing_count": 18,
            "post_approval_capture_fill_validator_command_count": 30,
            "post_approval_capture_fill_ready_to_update_bundle_count": 0,
            "post_approval_capture_fill_plan_manifest": [fixture_capture_fill_plan(index) for index in range(6)],
            "recommended_post_approval_capture_fill_plan": {
                "preview_only": True,
                "valid": True,
                "request_name": "Fixture Mixtral Phase 3 real evidence",
                "request_path": "memory-moe-mvp/phase3-real-evidence/fixture-mixtral.runtime-capture-request.json",
                "prompt_set_path": "memory-moe-mvp/phase3-real-evidence/fixture-mixtral.prompt-set.json",
                "rank": 1,
                "selection_rationale": "mixtral_first_known_sparse_baseline",
                "records_approval_keys": [
                    "runtime_prompt_traffic_approved",
                    "router_trace_capture_approved",
                    "managed_output_capture_approved",
                    "dense_output_capture_approved",
                ],
                "requires_explicit_user_approval": True,
                "metadata_only": True,
                "mutates_request": False,
                "artifact_step_count": 3,
                "runtime_capture_step_count": 3,
                "ready_after_approval_count": 0,
                "missing_after_approval_count": 3,
                "validator_command_count": 5,
                "command_option_count": 1,
                "all_receipts_ready_after_approval": False,
                "ready_to_update_bundle_after_approval": False,
                "capture_fill_steps": [
                    {
                        "artifact_id": "candidate_router_trace",
                        "queue_step_id": "candidate_router_trace",
                        "receipt_kind": "trace_capture_receipt_json",
                        "artifact_path": "memory-moe-mvp/phase3-real-evidence/fixture/candidate-router-events.jsonl",
                        "receipt_path": "memory-moe-mvp/phase3-real-evidence/fixture/candidate-router-events.capture-receipt.json",
                        "current_step": {
                            "id": "candidate_router_trace",
                            "stage": "runtime_capture",
                            "status": "approval_required",
                            "approval_state": "missing",
                            "path": "memory-moe-mvp/phase3-real-evidence/fixture/candidate-router-events.jsonl",
                        },
                        "step_after_approval": {
                            "id": "candidate_router_trace",
                            "stage": "runtime_capture",
                            "status": "ready_for_operator_capture",
                            "approval_state": "recorded",
                            "path": "memory-moe-mvp/phase3-real-evidence/fixture/candidate-router-events.jsonl",
                        },
                        "fill_status_after_approval": "blocked_after_approval",
                        "ready_after_approval": False,
                        "receipt_ready": False,
                        "validator_command_count": 3,
                        "command_option_count": 1,
                        "command_options": fixture_approval_command_options(fixture_paths(0)),
                        "blockers_after_approval": ["candidate_trace_capture_receipt_not_ready"],
                    },
                    {
                        "artifact_id": "managed_output_summary_fill",
                        "queue_step_id": "managed_output_summary",
                        "receipt_kind": "embedded_output_capture_receipt",
                        "artifact_path": "memory-moe-mvp/phase3-real-evidence/fixture/managed-output-summary.json",
                        "receipt_path": "memory-moe-mvp/phase3-real-evidence/fixture/managed-output-summary.json",
                        "step_after_approval": {
                            "id": "managed_output_summary",
                            "stage": "runtime_capture",
                            "status": "ready_for_operator_capture",
                            "approval_state": "recorded",
                            "path": "memory-moe-mvp/phase3-real-evidence/fixture/managed-output-summary.json",
                        },
                        "fill_status_after_approval": "blocked_after_approval",
                        "ready_after_approval": False,
                        "receipt_ready": False,
                        "validator_command_count": 1,
                        "command_option_count": 0,
                        "blockers_after_approval": ["managed_capture_receipt_not_ready"],
                    },
                    {
                        "artifact_id": "dense_output_summary_fill",
                        "queue_step_id": "dense_output_summary",
                        "receipt_kind": "embedded_output_capture_receipt",
                        "artifact_path": "memory-moe-mvp/phase3-real-evidence/fixture/dense-output-summary.json",
                        "receipt_path": "memory-moe-mvp/phase3-real-evidence/fixture/dense-output-summary.json",
                        "step_after_approval": {
                            "id": "dense_output_summary",
                            "stage": "runtime_capture",
                            "status": "ready_for_operator_capture",
                            "approval_state": "recorded",
                            "path": "memory-moe-mvp/phase3-real-evidence/fixture/dense-output-summary.json",
                        },
                        "fill_status_after_approval": "blocked_after_approval",
                        "ready_after_approval": False,
                        "receipt_ready": False,
                        "validator_command_count": 1,
                        "command_option_count": 0,
                        "blockers_after_approval": ["dense_capture_receipt_not_ready"],
                    },
                ],
            },
            "receipt_fill_manifest_summary": {
                "entry_count": 18,
                "ready_count": 0,
                "missing_count": 18,
                "approval_missing_count": 18,
                "by_artifact": {
                    "candidate_router_trace": {"entry_count": 6, "ready_count": 0, "missing_count": 6},
                    "managed_output_summary_fill": {"entry_count": 6, "ready_count": 0, "missing_count": 6},
                    "dense_output_summary_fill": {"entry_count": 6, "ready_count": 0, "missing_count": 6},
                },
                "all_ready": False,
            },
            "capture_receipt_missing_request_count": 6,
            "output_receipt_binding_ready_count": 0,
            "receipt_gate_coverage": {
                "request_count": 6,
                "capture_receipt_required_count": 18,
                "capture_receipt_ready_count": 0,
                "approvals_ready_count": 0,
                "candidate_trace_receipt_ready_count": 0,
                "managed_output_receipt_ready_count": 0,
                "dense_output_receipt_ready_count": 0,
                "output_receipt_binding_ready_count": 0,
                "live_capability_proof_ready_count": 0,
                "missing_request_count": 6,
                "all_capture_receipts_ready": False,
                "all_output_receipt_bindings_ready": False,
            },
            "ready_to_update_bundle_count": 0,
            "phase4_candidate_ready_count": 0,
            "live_spike_candidate_ready_count": 0,
            "requests": [fixture_intake_request(index) for index in range(6)],
            "next_operator_step_counts": {"candidate_router_trace": 6},
            "remaining_blockers_after_intake": ["no_replay_policy_candidate"],
        },

    )

class Phase3EvidencePacketRepoGateWiringTests(unittest.TestCase):
    def test_repo_gate_summaries_include_dedicated_reuse_evidence(self) -> None:
        real_matrix = __import__("plan_phase3_real_evidence_matrix")
        handoff = __import__("plan_phase3_handoff_coverage")
        launch_library = planner.plan_phase3_launch_card_library
        runtime_request = __import__("plan_phase3_runtime_capture_request")
        capture_intake = __import__("plan_phase3_capture_result_intake")
        reuse_capture = __import__("plan_phase3_reuse_evidence_capture")
        originals = {
            "real_matrix": real_matrix.build_matrix,
            "handoff": handoff.build_coverage,
            "launch_library": launch_library.build_library,
            "runtime_request": runtime_request.build_root_summary,
            "capture_intake": capture_intake.build_root_summary,
            "reuse_capture": reuse_capture.build_root_summary,
        }
        real_matrix.build_matrix = lambda _root: {"id": "real_matrix"}
        handoff.build_coverage = lambda: {"id": "handoff"}
        launch_library.build_library = lambda: {"id": "launch_library"}
        runtime_request.build_root_summary = lambda: {"id": "runtime_request"}
        capture_intake.build_root_summary = lambda: {"id": "capture_intake"}
        reuse_capture.build_root_summary = lambda: {
            "id": "reuse_capture",
            "prompt_identity_metadata_missing_count": 6,
        }
        planner.build_repo_gate_summaries.cache_clear()
        try:
            summaries = planner.build_repo_gate_summaries()
        finally:
            real_matrix.build_matrix = originals["real_matrix"]
            handoff.build_coverage = originals["handoff"]
            launch_library.build_library = originals["launch_library"]
            runtime_request.build_root_summary = originals["runtime_request"]
            capture_intake.build_root_summary = originals["capture_intake"]
            reuse_capture.build_root_summary = originals["reuse_capture"]
            planner.build_repo_gate_summaries.cache_clear()

        self.assertEqual(len(summaries), 6)
        self.assertEqual(
            [summary["id"] for summary in summaries],
            ["real_matrix", "handoff", "launch_library", "runtime_request", "capture_intake", "reuse_capture"],
        )
        self.assertEqual(summaries[5]["prompt_identity_metadata_missing_count"], 6)


class Phase3EvidencePacketFilledLaunchCardHandoffTests(unittest.TestCase):
    def test_model_plane_fulfillment_filled_card_dir_validates_in_operator_handoff(self) -> None:
        import plan_phase3_operator_handoff

        launch_library = planner.plan_phase3_launch_card_library.build_library()
        fulfillment_tasks = []
        for index, task in enumerate(launch_library["binding_tasks"]):
            fulfillment_tasks.append(
                {
                    "request_name": task["request_name"],
                    "request_path": task["request_path"],
                    "launch_card_path": task["launch_card_path"],
                    "model_id": task["model_id"],
                    "backend_family": task["backend_family"],
                    "prompt_family": task["prompt_family"],
                    "task_id": task["task_id"],
                    "artifact_id": task["artifact_id"],
                    "capture_kind": task["capture_kind"],
                    "receipt_kind": task["receipt_kind"],
                    "artifact_output_path": task["artifact_output_path"],
                    "receipt_output_path": task["receipt_output_path"],
                    "prompt_set_path": task["prompt_set_path"],
                    "approval_keys": task["approval_keys"],
                    "binding": {
                        "binding_kind": "model_plane_callable_or_launch_command",
                        "model_plane_callable_id": f"fixture-callable-{index:02d}",
                        "launch_command": [],
                        "command_ready": True,
                        "requires_explicit_user_approval": True,
                        "may_send_prompt_traffic": True,
                        "runtime_capture_request_path": task["request_path"],
                        "approval_keys": task["approval_keys"],
                        "writes_artifact_path": task["artifact_output_path"],
                        "writes_receipt_path": task["receipt_output_path"],
                    },
                }
            )
        fulfillment = {
            "schema_version": planner.plan_phase3_launch_card_library.MODEL_PLANE_ARTIFACT_WRITER_FULFILLMENT_SCHEMA_VERSION,
            "request_schema_version": planner.plan_phase3_launch_card_library.MODEL_PLANE_ARTIFACT_WRITER_REQUEST_SCHEMA_VERSION,
            "mode": "phase3_model_plane_artifact_writer_fulfillment",
            "tasks": fulfillment_tasks,
        }
        package = planner.plan_phase3_launch_card_library.build_filled_launch_card_package_from_model_plane_fulfillment(
            launch_library,
            fulfillment,
        )
        self.assertTrue(package["valid"], package["errors"])
        self.assertTrue(package["binding_ready"], package["blockers"])

        try:
            planner.build_repo_gate_summaries.cache_clear()
        except AttributeError:
            pass
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                filled_dir = Path(temp_dir) / "filled-launch-cards"
                outputs = planner.plan_phase3_launch_card_library.write_filled_launch_cards(package, filled_dir)
                self.assertEqual(len(outputs), 6)
                summary = planner.build_packet_summary(
                    TRACE_FIXTURE,
                    INVENTORY_FIXTURE,
                    POLICIES_FIXTURE,
                    MANAGED_PLAN,
                    runtime_capture_launch_card_dir=filled_dir,
                )
                handoff_dir = Path(temp_dir) / "handoff"
                planner.write_operator_handoff_dir(summary, handoff_dir)
                handoff_summary = plan_phase3_operator_handoff.build_summary(handoff_dir)
        finally:
            try:
                planner.build_repo_gate_summaries.cache_clear()
            except AttributeError:
                pass

        self.assertTrue(summary["runtime_capture_launch_card_directory_manifest"]["directory_ready"])
        self.assertTrue(summary["all_request_runtime_capture_execution_coverage_summary"]["automated_capture_ready"])
        self.assertTrue(handoff_summary["valid"], handoff_summary["errors"])
        self.assertTrue(handoff_summary["handoff_package_ready"])
        self.assertTrue(handoff_summary["runtime_capture_launch_card_dir_provided"])
        self.assertTrue(handoff_summary["runtime_capture_launch_card_dir_ready"])
        self.assertEqual(handoff_summary["runtime_capture_launch_card_dir_expected_card_count"], 6)
        self.assertEqual(handoff_summary["runtime_capture_launch_card_dir_matched_card_count"], 6)
        self.assertEqual(handoff_summary["runtime_capture_launch_card_dir_missing_card_count"], 0)
        self.assertEqual(handoff_summary["runtime_capture_launch_card_dir_error_count"], 0)
        self.assertTrue(handoff_summary["all_request_runtime_capture_execution_automated_ready"])

class Phase3EvidencePacketPlannerTests(unittest.TestCase):

    def setUp(self) -> None:
        self._original_repo_gate_summaries = planner.build_repo_gate_summaries
        self._original_real_evidence_root = planner.plan_phase3_go_no_go.DEFAULT_REAL_EVIDENCE_ROOT
        planner.build_repo_gate_summaries = repo_gate_summaries_fixture
        planner.plan_phase3_go_no_go.DEFAULT_REAL_EVIDENCE_ROOT = None

    def tearDown(self) -> None:
        planner.build_repo_gate_summaries = self._original_repo_gate_summaries
        planner.plan_phase3_go_no_go.DEFAULT_REAL_EVIDENCE_ROOT = self._original_real_evidence_root


    def test_default_packet_is_valid_but_not_phase3_complete(self) -> None:
        summary = planner.build_packet_summary(
            TRACE_FIXTURE,
            INVENTORY_FIXTURE,
            POLICIES_FIXTURE,
            MANAGED_PLAN,
        )

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertTrue(summary["packet_ready"])
        self.assertFalse(summary["phase3_complete"])
        self.assertEqual(summary["schema_version"], planner.SUPPORTED_SCHEMA_VERSION)
        self.assertEqual(summary["decision"], "no_go_live_spike")
        by_item = {item["id"]: item for item in summary["evidence_items"]}
        self.assertEqual(by_item["expert_inventory_manifest"]["status"], "proven")
        self.assertEqual(by_item["expert_store_layout_plan"]["status"], "proven")
        self.assertEqual(by_item["trace_inventory_join"]["status"], "proven")
        self.assertEqual(by_item["real_model_trace_inventory_pairing"]["status"], "proven")
        self.assertTrue(by_item["real_model_trace_inventory_pairing"]["details"]["local_fixture_only_pair"])
        self.assertTrue(by_item["real_model_trace_inventory_pairing"]["details"]["repo_real_model_pair_ready"])
        self.assertEqual(by_item["real_model_trace_inventory_pairing"]["details"]["decision_pairing_basis"], "repo_real_evidence_matrix")
        self.assertEqual(by_item["dense_fallback_comparison"]["status"], "blocked")
        self.assertEqual(by_item["live_capability_proof"]["status"], "future_phase_4")
        self.assertEqual(by_item["runtime_actuator_spike_handoff"]["status"], "proven")
        spike_details = by_item["runtime_actuator_spike_handoff"]["details"]
        self.assertTrue(spike_details["spike_handoff_ready"])
        self.assertFalse(spike_details["live_spike_ready"])
        self.assertEqual(spike_details["proof_requirement_count"], 8)
        self.assertEqual(spike_details["proof_artifact_count"], 20)
        self.assertEqual(spike_details["dependency_edge_count"], 14)
        self.assertEqual(spike_details["blocking_capability_count"], 7)
        self.assertEqual(by_item["repo_real_evidence_matrix"]["status"], "proven")
        self.assertEqual(by_item["handoff_scaffold_coverage"]["status"], "proven")
        self.assertEqual(by_item["phase3_launch_card_library"]["status"], "proven")
        launch_library = by_item["phase3_launch_card_library"]["details"]
        self.assertTrue(launch_library["library_ready"])
        self.assertFalse(launch_library["execution_ready"])
        self.assertEqual(launch_library["card_count"], 6)
        self.assertEqual(launch_library["template_ready_count"], 6)
        self.assertEqual(launch_library["model_plane_binding_ready_count"], 0)
        self.assertEqual(launch_library["binding_ready_count"], 0)
        self.assertEqual(launch_library["runtime_capture_command_ready_count"], 0)
        self.assertEqual(launch_library["task_count"], 18)
        self.assertEqual(launch_library["missing_runtime_command_count"], 18)
        self.assertTrue(launch_library["binding_handoff_ready"])
        self.assertEqual(launch_library["binding_handoff_task_count"], 18)
        self.assertEqual(launch_library["binding_handoff_ready_count"], 18)
        self.assertEqual(launch_library["binding_handoff_missing_field_count"], 0)
        self.assertEqual(launch_library["unbound_task_count"], 18)
        self.assertTrue(launch_library["model_plane_artifact_writer_contract_request_ready"])
        self.assertEqual(launch_library["model_plane_artifact_writer_contract_request_task_count"], 18)
        self.assertTrue(launch_library["saved_handoff_artifacts_ready"])
        self.assertEqual(launch_library["saved_handoff_artifact_missing_count"], 0)
        self.assertEqual(launch_library["saved_handoff_artifact_drifted_count"], 0)
        recommended_launch_card = summary["recommended_runtime_capture_launch_card_template"]
        self.assertEqual(recommended_launch_card["match_basis"], "request_name")
        self.assertEqual(recommended_launch_card["request_name"], "Fixture Mixtral Phase 3 real evidence")
        self.assertEqual(
            recommended_launch_card["launch_card_path"],
            "memory-moe-mvp/phase3-real-evidence/fixture-0.runtime-capture-launch-card.template.json",
        )
        self.assertTrue(recommended_launch_card["template_ready"])
        self.assertTrue(recommended_launch_card["binding_handoff_ready"])
        self.assertFalse(recommended_launch_card["binding_ready"])
        self.assertFalse(recommended_launch_card["runtime_capture_command_ready"])
        self.assertEqual(recommended_launch_card["task_count"], 3)
        self.assertEqual(recommended_launch_card["unbound_task_count"], 3)
        self.assertEqual(recommended_launch_card["missing_runtime_command_count"], 3)
        self.assertEqual(by_item["runtime_capture_request_audit"]["status"], "proven")
        self.assertEqual(by_item["capture_result_intake"]["status"], "blocked")
        self.assertEqual(by_item["approval_manifest_parity"]["status"], "proven")
        self.assertEqual(by_item["receipt_requirement_parity"]["status"], "proven")
        self.assertEqual(by_item["operator_queue_parity"]["status"], "proven")
        self.assertEqual(by_item["post_approval_capture_fill_parity"]["status"], "proven")
        self.assertEqual(by_item["all_post_approval_capture_fill_parity"]["status"], "proven")
        self.assertEqual(by_item["all_request_capture_queue_manifest"]["status"], "proven")
        self.assertEqual(by_item["all_request_validator_command_manifest"]["status"], "proven")
        self.assertEqual(by_item["all_request_downstream_handoff_manifest"]["status"], "blocked")
        self.assertEqual(by_item["all_request_receipt_fill_manifest"]["status"], "blocked")
        self.assertEqual(by_item["all_request_receipt_validator_parity"]["status"], "proven")
        self.assertEqual(by_item["all_request_receipt_fill_command_manifest"]["status"], "proven")
        self.assertEqual(by_item["recommended_runtime_capture_preflight"]["status"], "proven")
        self.assertNotIn("recommended_runtime_capture_command_contract", by_item)
        self.assertEqual(by_item["recommended_runtime_capture_execution_coverage"]["status"], "blocked")
        self.assertEqual(by_item["all_request_runtime_capture_execution_coverage"]["status"], "blocked")
        self.assertEqual(by_item["all_request_manual_capture_runbook"]["status"], "proven")
        self.assertEqual(by_item["all_request_post_capture_intake_runbook"]["status"], "proven")
        post_capture_details = by_item["all_request_post_capture_intake_runbook"]["details"]
        self.assertTrue(post_capture_details["ready"])
        self.assertEqual(post_capture_details["request_count"], 6)
        self.assertEqual(post_capture_details["artifact_gate_count"], 18)
        self.assertEqual(post_capture_details["ready_after_current_intake_count"], 0)
        self.assertEqual(post_capture_details["missing_after_current_intake_count"], 18)
        self.assertEqual(post_capture_details["validator_command_count"], 30)
        self.assertEqual(post_capture_details["ready_to_update_bundle_count"], 0)
        self.assertEqual(post_capture_details["phase4_candidate_count"], 0)
        self.assertEqual(post_capture_details["live_spike_candidate_count"], 0)
        self.assertEqual(post_capture_details["missing_item_count"], 0)
        self.assertEqual(by_item["phase3_blocker_closure_manifest"]["status"], "proven")
        closure_details = by_item["phase3_blocker_closure_manifest"]["details"]
        self.assertTrue(closure_details["ready"])
        self.assertTrue(closure_details["mapping_ready"])
        self.assertFalse(closure_details["evidence_complete"])
        self.assertFalse(closure_details["closure_complete"])
        self.assertEqual(closure_details["ready_scope"], "mapping_ready_only")
        self.assertEqual(closure_details["reason_count"], 7)
        self.assertEqual(closure_details["mapped_reason_count"], 7)
        self.assertEqual(closure_details["unmapped_reason_count"], 0)
        self.assertEqual(closure_details["missing_evidence_count"], 25)
        self.assertEqual(closure_details["runtime_capture_required_count"], 4)
        self.assertEqual(closure_details["future_adapter_required_count"], 3)
        self.assertEqual(closure_details["validator_command_count"], 41)
        closure_by_reason = {item["reason_id"]: item for item in closure_details["reasons"]}
        self.assertEqual(closure_by_reason["no_replay_policy_candidate"]["closure_gate"], "policy_candidate_trace_capture")
        self.assertEqual(closure_by_reason["capture_result_intake_not_ready"]["validator_command_count"], 30)
        self.assertEqual(closure_by_reason["dense_fallback_comparison_not_ready"]["closure_gate"], "dense_fallback_capture")
        self.assertEqual(closure_by_reason["live_actuator_missing"]["closure_gate"], "runtime_actuator_spike")
        self.assertTrue(closure_by_reason["live_actuator_missing"]["handoff_ready"])
        self.assertEqual(closure_by_reason["live_actuator_missing"]["validator_command_count"], 1)
        self.assertEqual(closure_by_reason["live_actuator_capabilities_unavailable"]["closure_gate"], "runtime_actuator_spike")
        self.assertTrue(closure_by_reason["live_actuator_capabilities_unavailable"]["handoff_ready"])
        self.assertEqual(closure_by_reason["live_actuator_capabilities_unavailable"]["validator_command_count"], 1)
        self.assertEqual(by_item["phase3_blocker_evidence_ledger"]["status"], "proven")
        ledger_details = by_item["phase3_blocker_evidence_ledger"]["details"]
        self.assertTrue(ledger_details["ready"])
        self.assertEqual(ledger_details["reason_count"], closure_details["reason_count"])
        self.assertEqual(ledger_details["evidence_row_count"], closure_details["missing_evidence_count"])
        self.assertEqual(ledger_details["expected_missing_evidence_count"], closure_details["missing_evidence_count"])
        self.assertEqual(ledger_details["runtime_capture_required_row_count"], 22)
        self.assertEqual(ledger_details["future_adapter_required_row_count"], 3)
        self.assertEqual(ledger_details["receipt_bound_row_count"], 18)
        self.assertEqual(ledger_details["dense_output_row_count"], 0)
        self.assertEqual(ledger_details["live_proof_row_count"], 0)
        self.assertEqual(ledger_details["runtime_actuator_row_count"], 2)
        self.assertEqual(ledger_details["validator_command_count"], closure_details["validator_command_count"])
        self.assertEqual(ledger_details["missing_item_count"], 0)
        self.assertEqual(len(ledger_details["manifest"]["evidence_rows"]), 25)
        self.assertEqual(by_item["phase3_blocker_resolution_queue"]["status"], "proven")
        queue_details = by_item["phase3_blocker_resolution_queue"]["details"]
        self.assertTrue(queue_details["ready"])
        self.assertEqual(queue_details["work_package_count"], 5)
        self.assertEqual(queue_details["queue_row_count"], ledger_details["evidence_row_count"])
        self.assertEqual(queue_details["expected_ledger_row_count"], ledger_details["evidence_row_count"])
        self.assertEqual(queue_details["runtime_capture_package_count"], 3)
        self.assertEqual(queue_details["future_adapter_package_count"], 1)
        self.assertEqual(queue_details["runtime_capture_required_row_count"], ledger_details["runtime_capture_required_row_count"])
        self.assertEqual(queue_details["future_adapter_required_row_count"], ledger_details["future_adapter_required_row_count"])
        self.assertEqual(queue_details["receipt_bound_row_count"], ledger_details["receipt_bound_row_count"])
        self.assertEqual(queue_details["validator_command_count"], closure_details["validator_command_count"])
        self.assertEqual(queue_details["completion_gate_count"], 21)
        self.assertEqual(queue_details["dependency_edge_count"], 4)
        self.assertEqual(queue_details["missing_item_count"], 0)
        self.assertEqual(queue_details["next_unblocked_work_package_id"], "policy_candidate_trace_capture")
        self.assertEqual(queue_details["next_unblocked_sequence_rank"], 1)
        self.assertEqual(queue_details["next_unblocked_operator_stage"], "approved_runtime_capture")
        self.assertEqual(queue_details["next_unblocked_package_class"], "policy_candidate_replay")
        self.assertEqual(queue_details["next_unblocked_row_count"], 2)
        self.assertEqual(queue_details["next_unblocked_validator_command_count"], 6)
        self.assertIn("capture the candidate router trace", queue_details["next_unblocked_next_action"])
        queue_order = [item["work_package_id"] for item in queue_details["manifest"]["work_packages"]]
        self.assertEqual(
            queue_details["manifest"]["next_unblocked_work_package"]["work_package_id"],
            "policy_candidate_trace_capture",
        )
        self.assertEqual(
            queue_order,
            [
                "policy_candidate_trace_capture",
                "capture_result_receipt_intake",
                "dense_fallback_output_capture",
                "runtime_actuator_spike",
                "live_capability_proof_fill",
            ],
        )
        queue_packages = {item["work_package_id"]: item for item in queue_details["manifest"]["work_packages"]}
        self.assertEqual(queue_packages["policy_candidate_trace_capture"]["depends_on_work_package_ids"], [])
        self.assertEqual(queue_packages["policy_candidate_trace_capture"]["validator_command_count"], 6)
        self.assertEqual(
            queue_packages["capture_result_receipt_intake"]["depends_on_work_package_ids"],
            ["policy_candidate_trace_capture"],
        )
        self.assertEqual(queue_packages["capture_result_receipt_intake"]["row_count"], 18)
        self.assertEqual(
            queue_packages["dense_fallback_output_capture"]["depends_on_work_package_ids"],
            ["capture_result_receipt_intake"],
        )
        self.assertEqual(queue_packages["dense_fallback_output_capture"]["row_validator_command_count"], 0)
        self.assertEqual(queue_packages["dense_fallback_output_capture"]["reason_validator_command_count"], 2)
        self.assertEqual(queue_packages["dense_fallback_output_capture"]["completion_gate_count"], 4)
        self.assertIn("dense fallback comparison artifact validates", " ".join(queue_packages["dense_fallback_output_capture"]["completion_gates"]))
        self.assertEqual(queue_packages["runtime_actuator_spike"]["sequence_rank"], 4)
        self.assertEqual(
            queue_packages["runtime_actuator_spike"]["depends_on_work_package_ids"],
            ["dense_fallback_output_capture"],
        )
        self.assertEqual(queue_packages["runtime_actuator_spike"]["completion_gate_count"], 5)
        runtime_proof = queue_packages["runtime_actuator_spike"]["proof_handoff"]
        self.assertTrue(runtime_proof["handoff_ready"])
        self.assertFalse(runtime_proof["live_spike_ready"])
        self.assertEqual(runtime_proof["backend_family"], "llama_cpp")
        self.assertEqual(runtime_proof["proof_requirement_count"], 8)
        self.assertEqual(runtime_proof["proof_artifact_count"], 20)
        self.assertEqual(runtime_proof["dependency_edge_count"], 14)
        self.assertEqual(runtime_proof["blocking_capability_count"], 7)
        self.assertEqual(runtime_proof["control_blocker_count"], 3)
        self.assertEqual(
            runtime_proof["proof_requirement_ids"],
            [
                "expert_inventory",
                "routing_visibility",
                "residency_observation",
                "policy_application",
                "dense_fallback",
                "artifact_export",
                "residency_control",
                "cleanup_restore",
            ],
        )
        self.assertEqual(len(runtime_proof["proof_requirements"]), 8)
        self.assertIn("residency_control", runtime_proof["control_blockers"])
        control_requirement = next(item for item in runtime_proof["proof_requirements"] if item["capability_id"] == "residency_control")
        self.assertTrue(control_requirement["requires_explicit_runtime_approval_before_live"])
        self.assertEqual(control_requirement["dependency_ids"], ["residency_observation", "dense_fallback", "artifact_export"])
        self.assertEqual(queue_packages["live_capability_proof_fill"]["validator_command_count"], 1)
        self.assertEqual(queue_packages["live_capability_proof_fill"]["sequence_rank"], 5)
        self.assertEqual(
            queue_packages["live_capability_proof_fill"]["depends_on_work_package_ids"],
            ["runtime_actuator_spike"],
        )
        preflight_details = by_item["recommended_runtime_capture_preflight"]["details"]
        self.assertTrue(preflight_details["ready"])
        self.assertEqual(preflight_details["request_name"], "Fixture Mixtral Phase 3 real evidence")
        self.assertTrue(preflight_details["approval_required"])
        self.assertEqual(preflight_details["pending_artifact_count"], 3)
        self.assertEqual(preflight_details["artifact_check_count"], 3)
        self.assertEqual(preflight_details["receipt_entry_count"], 3)
        self.assertEqual(preflight_details["receipt_ready_count"], 0)
        self.assertEqual(preflight_details["validator_command_count"], 5)
        self.assertEqual(preflight_details["runtime_closure_reason_count"], 4)
        self.assertEqual(preflight_details["runtime_closure_validator_command_count"], 38)
        self.assertEqual(preflight_details["missing_preflight_item_count"], 0)
        self.assertEqual(preflight_details["missing_preflight_items"], [])
        self.assertEqual(preflight_details["manifest"]["missing_capture_sequence_ids"], [])
        self.assertEqual(
            [item["artifact_id"] for item in preflight_details["manifest"]["artifact_checks"]],
            ["candidate_router_trace", "managed_output_summary_fill", "dense_output_summary_fill"],
        )
        self.assertFalse(summary["recommended_runtime_capture_command_contract_summary"]["ready"])
        self.assertEqual(summary["recommended_runtime_capture_command_contract"], {})
        execution_details = by_item["recommended_runtime_capture_execution_coverage"]["details"]
        self.assertTrue(execution_details["ready"])
        self.assertTrue(execution_details["manual_operator_capture_ready"])
        self.assertFalse(execution_details["automated_capture_ready"])
        self.assertEqual(execution_details["pending_artifact_count"], 3)
        self.assertEqual(execution_details["artifact_execution_count"], 3)
        self.assertEqual(execution_details["capture_command_option_count"], 0)
        self.assertEqual(execution_details["operator_command_option_count"], 1)
        self.assertEqual(execution_details["metadata_command_option_count"], 1)
        self.assertEqual(execution_details["artifacts_with_capture_command_count"], 0)
        self.assertEqual(execution_details["manual_capture_required_count"], 3)
        self.assertEqual(execution_details["missing_capture_command_count"], 3)
        self.assertEqual(
            execution_details["missing_capture_command_artifact_ids"],
            ["candidate_router_trace", "dense_output_summary_fill", "managed_output_summary_fill"],
        )
        self.assertEqual(execution_details["missing_execution_item_count"], 0)
        execution_by_artifact = {item["artifact_id"]: item for item in execution_details["manifest"]["artifact_execution"]}
        self.assertEqual(execution_by_artifact["candidate_router_trace"]["capture_mode"], "manual_runtime_capture_required")
        self.assertEqual(execution_by_artifact["candidate_router_trace"]["capture_command_option_count"], 0)
        self.assertEqual(execution_by_artifact["candidate_router_trace"]["operator_command_option_count"], 1)
        self.assertEqual(execution_by_artifact["candidate_router_trace"]["metadata_command_option_count"], 1)
        self.assertFalse(execution_by_artifact["candidate_router_trace"]["has_runtime_capture_command"])
        self.assertEqual(execution_by_artifact["managed_output_summary_fill"]["capture_mode"], "manual_runtime_capture_required")
        self.assertEqual(execution_by_artifact["dense_output_summary_fill"]["capture_command_option_count"], 0)
        all_execution_details = by_item["all_request_runtime_capture_execution_coverage"]["details"]
        self.assertTrue(all_execution_details["ready"])
        self.assertTrue(all_execution_details["manual_operator_capture_ready"])
        self.assertFalse(all_execution_details["automated_capture_ready"])
        self.assertEqual(all_execution_details["request_count"], 6)
        self.assertEqual(all_execution_details["manual_operator_capture_ready_count"], 6)
        self.assertEqual(all_execution_details["automated_capture_ready_count"], 0)
        self.assertEqual(all_execution_details["pending_artifact_count"], 18)
        self.assertEqual(all_execution_details["artifact_execution_count"], 18)
        self.assertEqual(all_execution_details["capture_command_option_count"], 0)
        self.assertEqual(all_execution_details["manual_capture_required_count"], 18)
        self.assertEqual(all_execution_details["missing_capture_command_count"], 18)
        self.assertEqual(all_execution_details["missing_execution_item_count"], 0)
        self.assertEqual(len(all_execution_details["manifest"]["requests"]), 6)
        self.assertEqual(by_item["repo_real_evidence_matrix"]["details"]["bundle_count"], 6)
        self.assertEqual(by_item["repo_real_evidence_matrix"]["details"]["policy_candidate_blocked_bundle_count"], 6)
        self.assertEqual(by_item["repo_real_evidence_matrix"]["details"]["policy_candidate_no_reuse_distance_observation_count"], 6)
        self.assertEqual(by_item["repo_real_evidence_matrix"]["details"]["policy_candidate_prompt_identity_ready_count"], 0)
        self.assertEqual(by_item["repo_real_evidence_matrix"]["details"]["policy_candidate_prompt_identity_metadata_missing_count"], 6)
        self.assertEqual(
            by_item["repo_real_evidence_matrix"]["details"]["policy_candidate_blocker_counts"]["no_reuse_distance_observations"],
            6,
        )
        self.assertEqual(by_item["phase3_go_no_go_decision"]["details"]["policy_candidate_evidence"]["blocked_bundle_count"], 6)
        go_no_go_gate_status = by_item["phase3_go_no_go_decision"]["details"]["gate_status"]
        self.assertEqual(go_no_go_gate_status["handoff_runtime_capture_launch_card_template_ready_count"], 6)
        self.assertEqual(go_no_go_gate_status["handoff_runtime_capture_launch_card_binding_ready_count"], 0)
        self.assertEqual(go_no_go_gate_status["handoff_runtime_capture_launch_card_runtime_command_ready_count"], 0)
        self.assertEqual(go_no_go_gate_status["handoff_runtime_capture_launch_card_missing_runtime_command_count"], 18)
        self.assertEqual(by_item["runtime_capture_request_audit"]["details"]["request_count"], 6)
        self.assertEqual(by_item["runtime_capture_request_audit"]["details"]["capture_queue_summary"]["queue_count"], 6)
        self.assertEqual(by_item["runtime_capture_request_audit"]["details"]["capture_queue_summary"]["recommended_rank"], 1)
        self.assertEqual(by_item["runtime_capture_request_audit"]["details"]["approval_rebuild_command_manifest_count"], 6)
        runtime_preview = by_item["runtime_capture_request_audit"]["details"]["recommended_post_approval_preview"]
        self.assertTrue(runtime_preview["preview_only"])
        self.assertTrue(runtime_preview["valid"])
        self.assertEqual(runtime_preview["status"], "ready_for_operator_capture")
        self.assertTrue(runtime_preview["ready_for_operator_capture"])
        self.assertFalse(runtime_preview["capture_complete"])
        self.assertFalse(runtime_preview["mutates_request"])
        self.assertEqual(runtime_preview["pending_artifact_count"], 3)
        self.assertEqual(by_item["capture_result_intake"]["details"]["request_drift_free_count"], 6)
        self.assertEqual(by_item["capture_result_intake"]["details"]["request_drifted_count"], 0)
        self.assertEqual(by_item["capture_result_intake"]["details"]["capture_receipt_required_count"], 18)
        self.assertEqual(by_item["capture_result_intake"]["details"]["capture_receipt_ready_count"], 0)
        self.assertEqual(by_item["capture_result_intake"]["details"]["capture_receipt_missing_request_count"], 6)
        self.assertEqual(by_item["capture_result_intake"]["details"]["output_receipt_binding_ready_count"], 0)
        self.assertEqual(by_item["capture_result_intake"]["details"]["approval_rebuild_command_available_request_count"], 6)
        self.assertEqual(by_item["capture_result_intake"]["details"]["approval_rebuild_command_manifest_count"], 6)
        self.assertEqual(by_item["capture_result_intake"]["details"]["approval_transition_preview_count"], 6)
        self.assertEqual(by_item["capture_result_intake"]["details"]["approval_transition_ready_for_operator_count"], 6)
        self.assertEqual(by_item["capture_result_intake"]["details"]["approval_transition_ready_to_update_bundle_count"], 0)
        intake_preview = by_item["capture_result_intake"]["details"]["recommended_approval_transition_preview"]
        self.assertEqual(intake_preview["request_name"], "Fixture Mixtral Phase 3 real evidence")
        self.assertEqual(intake_preview["next_step_after_approval"]["status"], "ready_for_operator_capture")
        self.assertEqual(intake_preview["receipt_fill_preview"]["missing_after_approval_count"], 3)
        self.assertEqual(by_item["capture_result_intake"]["details"]["post_approval_capture_fill_plan_count"], 6)
        self.assertEqual(by_item["capture_result_intake"]["details"]["post_approval_capture_fill_artifact_step_count"], 18)
        self.assertEqual(by_item["capture_result_intake"]["details"]["post_approval_capture_fill_ready_count"], 0)
        self.assertEqual(by_item["capture_result_intake"]["details"]["post_approval_capture_fill_missing_count"], 18)
        self.assertEqual(by_item["capture_result_intake"]["details"]["post_approval_capture_fill_validator_command_count"], 30)
        self.assertEqual(by_item["capture_result_intake"]["details"]["post_approval_capture_fill_ready_to_update_bundle_count"], 0)
        intake_fill = by_item["capture_result_intake"]["details"]["recommended_post_approval_capture_fill_plan"]
        self.assertEqual(intake_fill["request_name"], "Fixture Mixtral Phase 3 real evidence")
        self.assertEqual(intake_fill["rank"], 1)
        self.assertEqual(intake_fill["artifact_step_count"], 3)
        self.assertEqual(intake_fill["runtime_capture_step_count"], 3)
        self.assertEqual(intake_fill["missing_after_approval_count"], 3)
        self.assertEqual(intake_fill["validator_command_count"], 5)
        self.assertFalse(intake_fill["ready_to_update_bundle_after_approval"])
        self.assertEqual(
            [step["artifact_id"] for step in intake_fill["capture_fill_steps"]],
            ["candidate_router_trace", "managed_output_summary_fill", "dense_output_summary_fill"],
        )
        self.assertEqual(
            [step["step_after_approval"]["approval_state"] for step in intake_fill["capture_fill_steps"]],
            ["recorded", "recorded", "recorded"],
        )
        self.assertEqual(
            by_item["capture_result_intake"]["details"]["receipt_gate_coverage"]["missing_request_count"],
            6,
        )
        self.assertEqual(by_item["capture_result_intake"]["details"]["ready_to_update_bundle_count"], 0)
        approval_parity = by_item["approval_manifest_parity"]["details"]
        self.assertTrue(approval_parity["ready"])
        self.assertEqual(approval_parity["runtime_manifest_count"], 6)
        self.assertEqual(approval_parity["capture_result_manifest_count"], 6)
        self.assertEqual(approval_parity["matched_request_count"], 6)
        self.assertEqual(approval_parity["missing_from_capture_result_request_paths"], [])
        self.assertEqual(approval_parity["missing_from_runtime_request_paths"], [])
        self.assertEqual(approval_parity["metadata_mismatch_request_paths"], [])
        self.assertEqual(approval_parity["command_mismatch_request_paths"], [])
        receipt_parity = by_item["receipt_requirement_parity"]["details"]
        self.assertTrue(receipt_parity["ready"])
        self.assertEqual(receipt_parity["runtime_requirement_count"], 18)
        self.assertEqual(receipt_parity["capture_result_requirement_count"], 18)
        self.assertEqual(receipt_parity["matched_requirement_count"], 18)
        self.assertEqual(receipt_parity["missing_from_capture_result_requirement_keys"], [])
        self.assertEqual(receipt_parity["missing_from_runtime_request_requirement_keys"], [])
        self.assertEqual(receipt_parity["metadata_mismatch_requirements"], [])
        operator_parity = by_item["operator_queue_parity"]["details"]
        self.assertTrue(operator_parity["ready"])
        self.assertEqual(operator_parity["comparison_basis"], "request_name_next_step_id_status_path")
        self.assertEqual(operator_parity["runtime_queue_count"], 6)
        self.assertEqual(operator_parity["capture_result_queue_count"], 6)
        self.assertEqual(operator_parity["matched_request_count"], 6)
        self.assertEqual(operator_parity["missing_from_capture_result_request_names"], [])
        self.assertEqual(operator_parity["missing_from_runtime_request_names"], [])
        self.assertEqual(operator_parity["metadata_mismatch_requests"], [])
        fill_parity = by_item["post_approval_capture_fill_parity"]["details"]
        self.assertTrue(fill_parity["ready"])
        self.assertEqual(fill_parity["comparison_basis"], "recommended_request_pending_artifact_id_path_status_validator_count")
        self.assertEqual(fill_parity["runtime_pending_artifact_count"], 3)
        self.assertEqual(fill_parity["capture_result_fill_step_count"], 3)
        self.assertEqual(fill_parity["matched_artifact_count"], 3)
        self.assertTrue(fill_parity["request_path_match"])
        self.assertEqual(fill_parity["missing_from_capture_result_artifact_ids"], [])
        self.assertEqual(fill_parity["missing_from_runtime_preview_artifact_ids"], [])
        self.assertEqual(fill_parity["metadata_mismatch_artifacts"], [])
        queue_manifest = by_item["all_request_capture_queue_manifest"]["details"]
        self.assertTrue(queue_manifest["ready"])
        self.assertEqual(queue_manifest["request_count"], 6)
        self.assertEqual(queue_manifest["requests_with_complete_fill_plan"], 6)
        self.assertEqual(queue_manifest["pending_artifact_count"], 18)
        self.assertEqual(queue_manifest["capture_fill_step_count"], 18)
        self.assertEqual(queue_manifest["validator_command_count"], 30)
        self.assertEqual(queue_manifest["requests"][0]["request_name"], "Fixture Mixtral Phase 3 real evidence")
        self.assertEqual(queue_manifest["requests"][0]["capture_fill_step_count"], 3)
        self.assertEqual(queue_manifest["requests"][0]["pending_artifact_ids"], ["candidate_router_trace", "managed_output_summary_fill", "dense_output_summary_fill"])
        validator_manifest = by_item["all_request_validator_command_manifest"]["details"]
        self.assertTrue(validator_manifest["ready"])
        self.assertEqual(validator_manifest["request_count"], 6)
        self.assertEqual(validator_manifest["artifact_entry_count"], 24)
        self.assertEqual(validator_manifest["runtime_artifact_count"], 18)
        self.assertEqual(validator_manifest["future_artifact_count"], 6)
        self.assertEqual(validator_manifest["runtime_validator_command_count"], 30)
        self.assertEqual(validator_manifest["future_validator_command_count"], 6)
        self.assertEqual(validator_manifest["total_validator_command_count"], 36)
        self.assertEqual(validator_manifest["missing_command_entry_count"], 0)
        first_validator = validator_manifest["artifact_commands"][0]
        self.assertEqual(first_validator["request_name"], "Fixture Mixtral Phase 3 real evidence")
        self.assertEqual(first_validator["artifact_id"], "candidate_router_trace")
        self.assertEqual(first_validator["artifact_stage"], "runtime_capture")
        self.assertEqual(len(first_validator["validator_commands"]), 3)
        self.assertEqual(by_item["phase3_go_no_go_decision"]["status"], "proven")
        by_check = {item["id"]: item for item in summary["promotion_checklist"]}
        self.assertEqual(by_check["repo_real_evidence_matrix"]["status"], "satisfied")
        self.assertEqual(by_check["handoff_scaffold_coverage"]["status"], "satisfied")
        self.assertEqual(by_check["phase3_launch_card_library"]["status"], "approval_required")
        self.assertEqual(by_check["approved_runtime_capture"]["status"], "approval_required")
        self.assertEqual(by_check["capture_result_intake"]["status"], "blocked")
        self.assertEqual(by_check["approval_manifest_parity"]["status"], "satisfied")
        self.assertEqual(by_check["receipt_requirement_parity"]["status"], "satisfied")
        self.assertEqual(by_check["operator_queue_parity"]["status"], "satisfied")
        self.assertEqual(by_check["post_approval_capture_fill_parity"]["status"], "satisfied")
        self.assertEqual(by_check["all_post_approval_capture_fill_parity"]["status"], "satisfied")
        self.assertEqual(by_check["all_request_capture_queue_manifest"]["status"], "satisfied")
        self.assertEqual(by_check["all_request_validator_command_manifest"]["status"], "satisfied")
        self.assertEqual(by_check["all_request_downstream_handoff_manifest"]["status"], "blocked")
        self.assertEqual(by_check["all_request_receipt_fill_manifest"]["status"], "blocked")
        self.assertEqual(by_check["all_request_receipt_validator_parity"]["status"], "satisfied")
        self.assertEqual(by_check["all_request_receipt_fill_command_manifest"]["status"], "satisfied")
        self.assertEqual(by_check["recommended_runtime_capture_preflight"]["status"], "satisfied")
        self.assertNotIn("recommended_runtime_capture_command_contract", by_check)
        self.assertEqual(by_check["recommended_runtime_capture_execution_coverage"]["status"], "blocked")
        self.assertEqual(by_check["all_request_runtime_capture_execution_coverage"]["status"], "blocked")
        self.assertEqual(by_check["all_request_manual_capture_runbook"]["status"], "satisfied")
        self.assertEqual(by_check["all_request_post_capture_intake_runbook"]["status"], "satisfied")
        self.assertEqual(by_check["phase3_blocker_closure_manifest"]["status"], "satisfied")
        self.assertEqual(by_check["phase3_blocker_evidence_ledger"]["status"], "satisfied")
        self.assertEqual(by_check["phase3_blocker_resolution_queue"]["status"], "satisfied")
        self.assertEqual(by_check["dense_fallback_quality_bounds"]["status"], "blocked")
        self.assertEqual(by_check["live_capability_proof_handoff"]["status"], "blocked")
        self.assertEqual(by_check["live_capability_proof"]["status"], "future_phase_4")
        self.assertEqual(by_check["phase4_promotion_decision"]["status"], "blocked")
        launch_library_check = by_check["phase3_launch_card_library"]["evidence"]
        self.assertTrue(launch_library_check["library_ready"])
        self.assertFalse(launch_library_check["execution_ready"])
        self.assertEqual(launch_library_check["card_count"], 6)
        self.assertEqual(launch_library_check["template_ready_count"], 6)
        self.assertEqual(launch_library_check["model_plane_binding_ready_count"], 0)
        self.assertEqual(launch_library_check["binding_ready_count"], 0)
        self.assertEqual(launch_library_check["runtime_capture_command_ready_count"], 0)
        self.assertEqual(launch_library_check["task_count"], 18)
        self.assertEqual(launch_library_check["missing_runtime_command_count"], 18)
        self.assertTrue(launch_library_check["binding_handoff_ready"])
        self.assertEqual(launch_library_check["binding_handoff_task_count"], 18)
        self.assertEqual(launch_library_check["binding_handoff_ready_count"], 18)
        self.assertEqual(launch_library_check["binding_handoff_missing_field_count"], 0)
        self.assertEqual(launch_library_check["unbound_task_count"], 18)
        self.assertTrue(launch_library_check["model_plane_artifact_writer_contract_request_ready"])
        self.assertEqual(launch_library_check["model_plane_artifact_writer_contract_request_task_count"], 18)
        self.assertTrue(launch_library_check["saved_handoff_artifacts_ready"])
        self.assertEqual(launch_library_check["saved_handoff_artifact_missing_count"], 0)
        self.assertEqual(launch_library_check["saved_handoff_artifact_drifted_count"], 0)
        self.assertEqual(by_check["approved_runtime_capture"]["evidence"]["request_count"], 6)
        self.assertEqual(by_check["approved_runtime_capture"]["evidence"]["capture_complete_count"], 0)
        self.assertEqual(by_check["approved_runtime_capture"]["evidence"]["capture_queue_summary"]["queue_count"], 6)
        self.assertEqual(by_check["approved_runtime_capture"]["evidence"]["approval_rebuild_command_manifest_count"], 6)
        self.assertEqual(
            by_check["approved_runtime_capture"]["evidence"]["recommended_post_approval_preview"]["status"],
            "ready_for_operator_capture",
        )
        self.assertFalse(by_check["approved_runtime_capture"]["evidence"]["recommended_post_approval_preview"]["capture_complete"])
        self.assertEqual(by_check["approved_runtime_capture"]["evidence"]["capture_queue_summary"]["ranked_requests"][0]["selection_rationale"], "mixtral_first_known_sparse_baseline")
        self.assertEqual(by_check["capture_result_intake"]["evidence"]["request_drift_free_count"], 6)
        self.assertEqual(by_check["capture_result_intake"]["evidence"]["request_drifted_count"], 0)
        self.assertEqual(by_check["capture_result_intake"]["evidence"]["capture_receipt_required_count"], 18)
        self.assertEqual(by_check["capture_result_intake"]["evidence"]["capture_receipt_ready_count"], 0)
        self.assertEqual(by_check["capture_result_intake"]["evidence"]["capture_receipt_missing_request_count"], 6)
        self.assertEqual(by_check["capture_result_intake"]["evidence"]["receipt_fill_entry_count"], 18)
        self.assertEqual(by_check["capture_result_intake"]["evidence"]["receipt_fill_ready_count"], 0)
        self.assertEqual(by_check["capture_result_intake"]["evidence"]["receipt_fill_missing_count"], 18)
        self.assertEqual(by_check["capture_result_intake"]["evidence"]["receipt_fill_approval_missing_count"], 18)
        self.assertEqual(
            by_check["capture_result_intake"]["evidence"]["receipt_fill_manifest_summary"]["by_artifact"]["candidate_router_trace"],
            {"entry_count": 6, "ready_count": 0, "missing_count": 6},
        )
        self.assertEqual(by_check["capture_result_intake"]["evidence"]["output_receipt_binding_ready_count"], 0)
        self.assertEqual(by_check["capture_result_intake"]["evidence"]["approval_rebuild_command_available_request_count"], 6)
        self.assertEqual(by_check["capture_result_intake"]["evidence"]["approval_rebuild_command_manifest_count"], 6)
        self.assertEqual(by_check["capture_result_intake"]["evidence"]["approval_transition_preview_count"], 6)
        self.assertEqual(by_check["capture_result_intake"]["evidence"]["approval_transition_ready_for_operator_count"], 6)
        self.assertEqual(by_check["capture_result_intake"]["evidence"]["approval_transition_ready_to_update_bundle_count"], 0)
        self.assertEqual(
            by_check["capture_result_intake"]["evidence"]["recommended_approval_transition_preview"]["next_step_after_approval"]["approval_state"],
            "recorded",
        )
        self.assertEqual(by_check["capture_result_intake"]["evidence"]["post_approval_capture_fill_plan_count"], 6)
        self.assertEqual(by_check["capture_result_intake"]["evidence"]["post_approval_capture_fill_artifact_step_count"], 18)
        self.assertEqual(by_check["capture_result_intake"]["evidence"]["post_approval_capture_fill_ready_count"], 0)
        self.assertEqual(by_check["capture_result_intake"]["evidence"]["post_approval_capture_fill_missing_count"], 18)
        self.assertEqual(by_check["capture_result_intake"]["evidence"]["post_approval_capture_fill_validator_command_count"], 30)
        self.assertEqual(by_check["capture_result_intake"]["evidence"]["post_approval_capture_fill_ready_to_update_bundle_count"], 0)
        checklist_fill = by_check["capture_result_intake"]["evidence"]["recommended_post_approval_capture_fill_plan"]
        self.assertEqual(checklist_fill["request_name"], "Fixture Mixtral Phase 3 real evidence")
        self.assertEqual(checklist_fill["missing_after_approval_count"], 3)
        self.assertEqual(checklist_fill["validator_command_count"], 5)
        self.assertEqual(checklist_fill["capture_fill_steps"][0]["fill_status_after_approval"], "blocked_after_approval")
        self.assertTrue(by_check["approval_manifest_parity"]["evidence"]["ready"])
        self.assertEqual(by_check["approval_manifest_parity"]["evidence"]["matched_request_count"], 6)
        self.assertTrue(by_check["receipt_requirement_parity"]["evidence"]["ready"])
        self.assertEqual(by_check["receipt_requirement_parity"]["evidence"]["matched_requirement_count"], 18)
        self.assertTrue(by_check["operator_queue_parity"]["evidence"]["ready"])
        self.assertEqual(by_check["operator_queue_parity"]["evidence"]["matched_request_count"], 6)
        self.assertTrue(by_check["post_approval_capture_fill_parity"]["evidence"]["ready"])
        self.assertEqual(by_check["post_approval_capture_fill_parity"]["evidence"]["matched_artifact_count"], 3)
        self.assertTrue(by_check["all_post_approval_capture_fill_parity"]["evidence"]["ready"])
        self.assertEqual(by_check["all_post_approval_capture_fill_parity"]["evidence"]["matched_request_count"], 6)
        self.assertEqual(by_check["all_post_approval_capture_fill_parity"]["evidence"]["matched_artifact_count"], 18)
        self.assertTrue(by_check["all_request_capture_queue_manifest"]["evidence"]["ready"])
        self.assertEqual(by_check["all_request_capture_queue_manifest"]["evidence"]["request_count"], 6)
        self.assertEqual(by_check["all_request_capture_queue_manifest"]["evidence"]["pending_artifact_count"], 18)
        self.assertTrue(by_check["all_request_validator_command_manifest"]["evidence"]["ready"])
        self.assertEqual(by_check["all_request_validator_command_manifest"]["evidence"]["artifact_entry_count"], 24)
        self.assertEqual(by_check["all_request_validator_command_manifest"]["evidence"]["total_validator_command_count"], 36)
        self.assertEqual(summary["all_request_capture_queue_summary"]["request_count"], 6)
        self.assertEqual(summary["all_request_capture_queue_summary"]["pending_artifact_count"], 18)
        self.assertEqual(len(summary["all_request_capture_queue_manifest"]), 6)
        self.assertEqual(summary["all_request_validator_command_summary"]["request_count"], 6)
        self.assertEqual(summary["all_request_validator_command_summary"]["artifact_entry_count"], 24)
        self.assertEqual(summary["all_request_validator_command_summary"]["runtime_validator_command_count"], 30)
        self.assertEqual(summary["all_request_validator_command_summary"]["future_validator_command_count"], 6)
        self.assertEqual(summary["all_request_validator_command_summary"]["total_validator_command_count"], 36)
        self.assertEqual(len(summary["all_request_validator_command_manifest"]), 24)
        self.assertFalse(summary["all_request_downstream_handoff_summary"]["ready"])
        self.assertEqual(summary["all_request_downstream_handoff_summary"]["request_count"], 6)
        self.assertEqual(summary["all_request_downstream_handoff_summary"]["all_downstream_handoff_ready_count"], 0)
        self.assertEqual(len(summary["all_request_downstream_handoff_manifest"]), 6)
        self.assertFalse(summary["all_request_receipt_fill_summary"]["ready"])
        self.assertEqual(summary["all_request_receipt_fill_summary"]["request_count"], 6)
        self.assertEqual(summary["all_request_receipt_fill_summary"]["entry_count"], 18)
        self.assertEqual(summary["all_request_receipt_fill_summary"]["ready_count"], 0)
        self.assertEqual(summary["all_request_receipt_fill_summary"]["missing_count"], 18)
        self.assertEqual(summary["all_request_receipt_fill_summary"]["approval_recorded_after_count"], 18)
        self.assertEqual(summary["all_request_receipt_fill_summary"]["approval_missing_after_count"], 0)
        self.assertEqual(summary["all_request_receipt_fill_summary"]["blocked_after_approval_count"], 18)
        self.assertEqual(summary["all_request_receipt_fill_summary"]["validator_command_count"], 30)
        artifact_class_counts = summary["all_request_receipt_fill_summary"]["artifact_class_counts"]
        self.assertEqual(artifact_class_counts["candidate_router_trace"], {"entry_count": 6, "ready_count": 0, "missing_count": 6})
        self.assertEqual(artifact_class_counts["managed_output_summary_fill"], {"entry_count": 6, "ready_count": 0, "missing_count": 6})
        self.assertEqual(artifact_class_counts["dense_output_summary_fill"], {"entry_count": 6, "ready_count": 0, "missing_count": 6})
        self.assertEqual(len(summary["all_request_receipt_fill_manifest"]), 18)
        first_receipt = summary["all_request_receipt_fill_manifest"][0]
        self.assertEqual(first_receipt["request_name"], "Fixture Mixtral Phase 3 real evidence")
        self.assertEqual(first_receipt["artifact_id"], "candidate_router_trace")
        self.assertEqual(first_receipt["approval_state_after_approval"], "recorded")
        self.assertFalse(first_receipt["receipt_ready"])
        receipt_validator_parity = summary["all_request_receipt_validator_parity_summary"]
        self.assertTrue(receipt_validator_parity["ready"])
        self.assertEqual(receipt_validator_parity["comparison_basis"], "runtime_capture_artifact_key_path_status_validator_count")
        self.assertEqual(receipt_validator_parity["runtime_artifact_count"], 18)
        self.assertEqual(receipt_validator_parity["receipt_fill_entry_count"], 18)
        self.assertEqual(receipt_validator_parity["matched_artifact_count"], 18)
        self.assertEqual(receipt_validator_parity["metadata_mismatch_artifacts"], [])
        self.assertEqual(receipt_validator_parity["missing_from_receipt_fill_artifact_keys"], [])
        self.assertEqual(receipt_validator_parity["missing_from_runtime_validator_artifact_keys"], [])
        receipt_commands = summary["all_request_receipt_fill_command_summary"]
        self.assertTrue(receipt_commands["ready"])
        self.assertEqual(receipt_commands["request_count"], 6)
        self.assertEqual(receipt_commands["entry_count"], 18)
        self.assertEqual(receipt_commands["ready_receipt_count"], 0)
        self.assertEqual(receipt_commands["blocked_after_approval_count"], 18)
        self.assertEqual(receipt_commands["validator_command_count"], 30)
        self.assertEqual(receipt_commands["missing_command_entry_count"], 0)
        self.assertEqual(receipt_commands["missing_receipt_path_count"], 0)
        self.assertEqual(len(summary["all_request_receipt_fill_command_manifest"]), 18)
        first_receipt_command = summary["all_request_receipt_fill_command_manifest"][0]
        self.assertEqual(first_receipt_command["artifact_id"], "candidate_router_trace")
        self.assertEqual(first_receipt_command["receipt_path"], "memory-moe-mvp/phase3-real-evidence/fixture/candidate-router-events.capture-receipt.json")
        self.assertEqual(len(first_receipt_command["validator_commands"]), 3)
        blocker_closure = summary["phase3_blocker_closure_summary"]
        self.assertTrue(blocker_closure["ready"])
        self.assertTrue(blocker_closure["mapping_ready"])
        self.assertFalse(blocker_closure["evidence_complete"])
        self.assertFalse(blocker_closure["closure_complete"])
        self.assertEqual(blocker_closure["ready_scope"], "mapping_ready_only")
        self.assertEqual(blocker_closure["reason_count"], 7)
        self.assertEqual(blocker_closure["mapped_reason_count"], 7)
        self.assertEqual(blocker_closure["unresolved_reason_count"], 7)
        self.assertEqual(blocker_closure["missing_evidence_count"], 25)
        self.assertEqual(blocker_closure["runtime_capture_required_count"], 4)
        self.assertEqual(blocker_closure["future_adapter_required_count"], 3)
        self.assertEqual(blocker_closure["handoff_ready_count"], 3)
        self.assertEqual(blocker_closure["validator_command_count"], 41)
        self.assertEqual(len(summary["phase3_blocker_closure_manifest"]), 7)
        self.assertEqual(summary["phase3_blocker_closure_manifest"][0]["reason_id"], "no_replay_policy_candidate")
        self.assertEqual(summary["phase3_blocker_closure_manifest"][0]["validator_command_count"], 3)
        blocker_ledger = summary["phase3_blocker_evidence_ledger_summary"]
        self.assertTrue(blocker_ledger["ready"])
        self.assertEqual(blocker_ledger["reason_count"], blocker_closure["reason_count"])
        self.assertEqual(blocker_ledger["evidence_row_count"], blocker_closure["missing_evidence_count"])
        self.assertEqual(blocker_ledger["expected_missing_evidence_count"], blocker_closure["missing_evidence_count"])
        self.assertEqual(blocker_ledger["runtime_capture_required_row_count"], 22)
        self.assertEqual(blocker_ledger["future_adapter_required_row_count"], 3)
        self.assertEqual(blocker_ledger["receipt_bound_row_count"], 18)
        self.assertEqual(blocker_ledger["dense_output_row_count"], 0)
        self.assertEqual(blocker_ledger["live_proof_row_count"], 0)
        self.assertEqual(blocker_ledger["runtime_actuator_row_count"], 2)
        self.assertEqual(blocker_ledger["validator_command_count"], blocker_closure["validator_command_count"])
        self.assertEqual(blocker_ledger["missing_item_count"], 0)
        self.assertEqual(len(summary["phase3_blocker_evidence_ledger_manifest"]["evidence_rows"]), 25)
        blocker_queue = summary["phase3_blocker_resolution_queue_summary"]
        self.assertTrue(blocker_queue["ready"])
        self.assertEqual(blocker_queue["work_package_count"], 5)
        self.assertEqual(blocker_queue["queue_row_count"], blocker_ledger["evidence_row_count"])
        self.assertEqual(blocker_queue["expected_ledger_row_count"], blocker_ledger["evidence_row_count"])
        self.assertEqual(blocker_queue["runtime_capture_package_count"], 3)
        self.assertEqual(blocker_queue["future_adapter_package_count"], 1)
        self.assertEqual(blocker_queue["runtime_capture_required_row_count"], blocker_ledger["runtime_capture_required_row_count"])
        self.assertEqual(blocker_queue["future_adapter_required_row_count"], blocker_ledger["future_adapter_required_row_count"])
        self.assertEqual(blocker_queue["receipt_bound_row_count"], blocker_ledger["receipt_bound_row_count"])
        self.assertEqual(blocker_queue["validator_command_count"], blocker_closure["validator_command_count"])
        self.assertEqual(blocker_queue["completion_gate_count"], 21)
        self.assertEqual(blocker_queue["dependency_edge_count"], 4)
        self.assertEqual(blocker_queue["missing_item_count"], 0)
        self.assertEqual(blocker_queue["next_unblocked_work_package_id"], "policy_candidate_trace_capture")
        self.assertEqual(blocker_queue["next_unblocked_sequence_rank"], 1)
        self.assertEqual(blocker_queue["next_unblocked_operator_stage"], "approved_runtime_capture")
        self.assertEqual(blocker_queue["next_unblocked_package_class"], "policy_candidate_replay")
        self.assertEqual(blocker_queue["next_unblocked_row_count"], 2)
        self.assertEqual(blocker_queue["next_unblocked_validator_command_count"], 6)
        self.assertEqual(blocker_queue["next_unblocked_completion_gate_count"], 4)
        self.assertIn("capture the candidate router trace", blocker_queue["next_unblocked_next_action"])
        next_handoff = summary["phase3_next_unblocked_operator_handoff_summary"]
        self.assertTrue(next_handoff["handoff_ready"])
        self.assertEqual(next_handoff["next_unblocked_work_package_id"], "policy_candidate_trace_capture")
        self.assertEqual(next_handoff["next_unblocked_operator_stage"], "approved_runtime_capture")
        self.assertEqual(next_handoff["work_order_artifact"], "recommended-runtime-capture-work-order.json")
        self.assertEqual(next_handoff["work_order_next_artifact_id"], "candidate_router_trace")
        self.assertTrue(next_handoff["work_order_advances_next_package"])
        self.assertTrue(next_handoff["work_order_approval_command_ready"])
        self.assertTrue(next_handoff["completion_receipt_template_ready"])
        self.assertTrue(next_handoff["completion_receipt_validation_command_ready"])
        self.assertTrue(next_handoff["work_order_bound_to_completion_receipt"])
        self.assertEqual(next_handoff["next_unblocked_row_count"], 2)
        self.assertEqual(next_handoff["next_unblocked_validator_command_count"], 6)
        self.assertEqual(next_handoff["work_order_capture_step_count"], 3)
        self.assertEqual(next_handoff["work_order_validator_command_count"], 5)
        self.assertEqual(
            [item["path"] for item in next_handoff["operator_artifact_sequence"]],
            [
                "blocker-resolution-queue.json",
                "recommended-runtime-capture-work-order.json",
                "recommended-runtime-capture-completion-receipt.template.json",
            ],
        )
        self.assertIn("Open recommended-runtime-capture-work-order.json", next_handoff["operator_next_action"])
        self.assertEqual(len(summary["phase3_blocker_resolution_queue_manifest"]["work_packages"]), 5)
        self.assertEqual(summary["phase3_blocker_resolution_queue_manifest"]["dependency_edge_count"], 4)
        recommended = summary["recommended_runtime_capture_request"]
        self.assertEqual(recommended["request_name"], "Fixture Mixtral Phase 3 real evidence")
        self.assertEqual(recommended["next_artifact_id"], "candidate_router_trace")
        self.assertEqual(
            [step["id"] for step in recommended["capture_sequence"]],
            ["record_runtime_approvals", "capture_candidate_router_trace", "run_capture_result_intake"],
        )
        self.assertEqual(
            by_check["approved_runtime_capture"]["evidence"]["recommended_runtime_capture_request"]["request_name"],
            "Fixture Mixtral Phase 3 real evidence",
        )
        self.assertEqual(recommended["validator_command_count"], 6)
        self.assertEqual(recommended["queue_rank"], 1)
        self.assertEqual(recommended["selection_rationale"], "mixtral_first_known_sparse_baseline")
        self.assertEqual(recommended["approval_rebuild_command"]["command_class"], "phase3_runtime_capture_request_approval_rebuild")
        self.assertTrue(recommended["approval_rebuild_command"]["metadata_only"])
        self.assertEqual(
            recommended["approval_rebuild_command"]["records_approval_keys"],
            ["runtime_prompt_traffic_approved", "router_trace_capture_approved"],
        )
        self.assertIn("--router-trace-capture-approved", recommended["approval_rebuild_command"]["command"])
        command_manifest = {item["artifact_id"]: item for item in recommended["validator_command_manifest"]}
        self.assertEqual(len(command_manifest["candidate_router_trace"]["validator_commands"]), 3)
        self.assertEqual(command_manifest["live_capability_proof_fill"]["artifact_stage"], "future_adapter")
        preflight_summary = summary["recommended_runtime_capture_preflight_summary"]
        self.assertTrue(preflight_summary["ready"])
        self.assertEqual(preflight_summary["request_name"], "Fixture Mixtral Phase 3 real evidence")
        self.assertEqual(preflight_summary["pending_artifact_count"], 3)
        self.assertEqual(preflight_summary["receipt_entry_count"], 3)
        self.assertEqual(preflight_summary["validator_command_count"], 5)
        self.assertEqual(preflight_summary["runtime_closure_reason_count"], 4)
        self.assertEqual(preflight_summary["runtime_closure_validator_command_count"], 38)
        self.assertEqual(preflight_summary["missing_preflight_item_count"], 0)
        self.assertEqual(
            summary["recommended_runtime_capture_preflight_manifest"]["runtime_closure_reason_ids"],
            [
                "no_replay_policy_candidate",
                "reuse_evidence_capture_not_ready",
                "capture_result_intake_not_ready",
                "dense_fallback_comparison_not_ready",
            ],
        )
        command_contract_summary = summary["recommended_runtime_capture_command_contract_summary"]
        self.assertFalse(command_contract_summary["ready"])
        self.assertFalse(command_contract_summary["runtime_capture_command_ready"])
        self.assertEqual(command_contract_summary["planned_capture_count"], 0)
        self.assertEqual(command_contract_summary["runtime_command_option_count"], 0)
        self.assertEqual(command_contract_summary["missing_runtime_command_count"], 0)
        self.assertEqual(command_contract_summary["missing_runtime_command_artifact_ids"], [])
        execution_summary = summary["recommended_runtime_capture_execution_coverage_summary"]
        self.assertTrue(execution_summary["ready"])
        self.assertTrue(execution_summary["manual_operator_capture_ready"])
        self.assertFalse(execution_summary["automated_capture_ready"])
        self.assertEqual(execution_summary["capture_command_option_count"], 0)
        self.assertEqual(execution_summary["operator_command_option_count"], 1)
        self.assertEqual(execution_summary["metadata_command_option_count"], 1)
        self.assertEqual(execution_summary["artifacts_with_capture_command_count"], 0)
        self.assertEqual(execution_summary["manual_capture_required_count"], 3)
        self.assertEqual(execution_summary["missing_capture_command_count"], 3)
        self.assertEqual(
            execution_summary["missing_capture_command_artifact_ids"],
            ["candidate_router_trace", "dense_output_summary_fill", "managed_output_summary_fill"],
        )
        all_execution_summary = summary["all_request_runtime_capture_execution_coverage_summary"]
        self.assertTrue(all_execution_summary["ready"])
        self.assertTrue(all_execution_summary["manual_operator_capture_ready"])
        self.assertFalse(all_execution_summary["automated_capture_ready"])
        self.assertEqual(all_execution_summary["request_count"], 6)
        self.assertEqual(all_execution_summary["manual_operator_capture_ready_count"], 6)
        self.assertEqual(all_execution_summary["automated_capture_ready_count"], 0)
        self.assertEqual(all_execution_summary["pending_artifact_count"], 18)
        self.assertEqual(all_execution_summary["artifact_execution_count"], 18)
        self.assertEqual(all_execution_summary["capture_command_option_count"], 0)
        self.assertEqual(all_execution_summary["manual_capture_required_count"], 18)
        self.assertEqual(all_execution_summary["missing_capture_command_count"], 18)
        self.assertEqual(all_execution_summary["missing_execution_item_count"], 0)
        manual_runbook_summary = summary["all_request_manual_capture_runbook_summary"]
        self.assertTrue(manual_runbook_summary["ready"])
        self.assertEqual(manual_runbook_summary["request_count"], 6)
        self.assertEqual(manual_runbook_summary["manual_task_count"], 18)
        self.assertEqual(manual_runbook_summary["runtime_command_task_count"], 0)
        self.assertEqual(manual_runbook_summary["validator_command_count"], 30)
        self.assertEqual(manual_runbook_summary["missing_item_count"], 0)
        post_capture_summary = summary["all_request_post_capture_intake_runbook_summary"]
        self.assertTrue(post_capture_summary["ready"])
        self.assertEqual(post_capture_summary["request_count"], 6)
        self.assertEqual(post_capture_summary["artifact_gate_count"], 18)
        self.assertEqual(post_capture_summary["ready_after_current_intake_count"], 0)
        self.assertEqual(post_capture_summary["missing_after_current_intake_count"], 18)
        self.assertEqual(post_capture_summary["validator_command_count"], 30)
        self.assertEqual(post_capture_summary["ready_to_update_bundle_count"], 0)
        self.assertEqual(post_capture_summary["phase4_candidate_count"], 0)
        self.assertEqual(post_capture_summary["live_spike_candidate_count"], 0)
        self.assertEqual(post_capture_summary["missing_source_request_path_count"], 0)
        self.assertEqual(post_capture_summary["missing_prompt_set_path_count"], 0)
        self.assertEqual(post_capture_summary["missing_explicit_approval_count"], 0)
        self.assertEqual(post_capture_summary["missing_prompt_traffic_ack_count"], 0)
        self.assertEqual(post_capture_summary["missing_item_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["policy_candidate_ready_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["policy_candidate_blocked_bundle_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["policy_candidate_no_reuse_distance_observation_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["policy_candidate_prompt_identity_ready_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["policy_candidate_prompt_identity_metadata_missing_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["policy_candidate_next_operator_step"], "capture_candidate_router_trace_with_reuse_distance")
        self.assertFalse(summary["remaining_gaps"]["live_capability_proof_handoff_ready"])
        self.assertIsNone(summary["remaining_gaps"]["live_capability_proof_template_path"])
        self.assertIsNone(summary["remaining_gaps"]["live_capability_proof_context_ready"])
        self.assertIsNone(summary["remaining_gaps"]["live_residency_observation_ready"])
        self.assertIsNone(summary["remaining_gaps"]["live_residency_control_ready"])
        self.assertIsNone(summary["remaining_gaps"]["cleanup_restore_proof_ready"])
        self.assertIsNone(summary["remaining_gaps"]["live_artifact_export_ready"])
        self.assertFalse(summary["remaining_gaps"]["live_capability_proof_ready"])
        self.assertTrue(summary["remaining_gaps"]["runtime_actuator_design_ready"])
        self.assertEqual(summary["remaining_gaps"]["runtime_actuator_backend_family"], "llama_cpp")
        self.assertFalse(summary["remaining_gaps"]["runtime_actuator_live_ready"])
        self.assertEqual(summary["remaining_gaps"]["runtime_actuator_control_blocker_count"], 3)
        self.assertTrue(summary["remaining_gaps"]["runtime_actuator_spike_handoff_ready"])
        self.assertFalse(summary["remaining_gaps"]["runtime_actuator_spike_live_ready"])
        self.assertEqual(summary["remaining_gaps"]["runtime_actuator_spike_proof_requirement_count"], 8)
        self.assertEqual(summary["remaining_gaps"]["runtime_actuator_spike_proof_artifact_count"], 20)
        self.assertEqual(summary["remaining_gaps"]["runtime_actuator_spike_dependency_edge_count"], 14)
        self.assertEqual(summary["remaining_gaps"]["runtime_actuator_spike_blocking_capability_count"], 7)
        self.assertFalse(summary["remaining_gaps"]["ready_for_live_spike"])
        self.assertTrue(summary["remaining_gaps"]["handoff_scaffolds_ready"])
        self.assertEqual(summary["remaining_gaps"]["runtime_capture_request_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["runtime_capture_complete_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["runtime_capture_queue_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["runtime_capture_recommended_rank"], 1)
        self.assertTrue(summary["remaining_gaps"]["runtime_capture_post_approval_preview_valid"])
        self.assertEqual(summary["remaining_gaps"]["runtime_capture_post_approval_preview_status"], "ready_for_operator_capture")
        self.assertTrue(summary["remaining_gaps"]["runtime_capture_post_approval_ready_for_operator"])
        self.assertFalse(summary["remaining_gaps"]["runtime_capture_post_approval_capture_complete"])
        self.assertEqual(summary["remaining_gaps"]["runtime_capture_post_approval_pending_artifact_count"], 3)
        self.assertFalse(summary["remaining_gaps"]["runtime_capture_post_approval_mutates_request"])
        self.assertTrue(summary["remaining_gaps"]["recommended_runtime_capture_preflight_ready"])
        self.assertEqual(summary["remaining_gaps"]["recommended_runtime_capture_preflight_pending_artifact_count"], 3)
        self.assertEqual(summary["remaining_gaps"]["recommended_runtime_capture_preflight_receipt_entry_count"], 3)
        self.assertEqual(summary["remaining_gaps"]["recommended_runtime_capture_preflight_validator_command_count"], 5)
        self.assertEqual(summary["remaining_gaps"]["recommended_runtime_capture_preflight_runtime_closure_reason_count"], 4)
        self.assertEqual(summary["remaining_gaps"]["recommended_runtime_capture_preflight_missing_item_count"], 0)
        self.assertFalse(summary["remaining_gaps"]["recommended_runtime_capture_command_contract_ready"])
        self.assertFalse(summary["remaining_gaps"]["recommended_runtime_capture_command_contract_runtime_ready"])
        self.assertEqual(summary["remaining_gaps"]["recommended_runtime_capture_command_contract_planned_capture_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["recommended_runtime_capture_command_contract_runtime_command_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["recommended_runtime_capture_command_contract_missing_runtime_command_count"], 0)
        self.assertEqual(
            summary["remaining_gaps"]["recommended_runtime_capture_launch_card_template_path"],
            "memory-moe-mvp/phase3-real-evidence/fixture-0.runtime-capture-launch-card.template.json",
        )
        self.assertEqual(summary["remaining_gaps"]["recommended_runtime_capture_launch_card_template_match_basis"], "request_name")
        self.assertTrue(summary["remaining_gaps"]["recommended_runtime_capture_launch_card_template_ready"])
        self.assertTrue(summary["remaining_gaps"]["recommended_runtime_capture_launch_card_template_binding_handoff_ready"])
        self.assertEqual(summary["remaining_gaps"]["recommended_runtime_capture_launch_card_template_task_count"], 3)
        self.assertEqual(summary["remaining_gaps"]["recommended_runtime_capture_launch_card_template_unbound_task_count"], 3)
        self.assertEqual(summary["remaining_gaps"]["recommended_runtime_capture_launch_card_template_missing_runtime_command_count"], 3)
        self.assertTrue(summary["remaining_gaps"]["recommended_runtime_capture_execution_manual_ready"])
        self.assertFalse(summary["remaining_gaps"]["recommended_runtime_capture_execution_automated_ready"])
        self.assertEqual(summary["remaining_gaps"]["recommended_runtime_capture_execution_command_option_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["recommended_runtime_capture_execution_operator_command_option_count"], 1)
        self.assertEqual(summary["remaining_gaps"]["recommended_runtime_capture_execution_metadata_command_option_count"], 1)
        self.assertEqual(summary["remaining_gaps"]["recommended_runtime_capture_execution_artifacts_with_command_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["recommended_runtime_capture_execution_manual_capture_required_count"], 3)
        self.assertEqual(summary["remaining_gaps"]["recommended_runtime_capture_execution_missing_capture_command_count"], 3)
        self.assertEqual(summary["remaining_gaps"]["recommended_runtime_capture_execution_missing_item_count"], 0)
        self.assertTrue(summary["remaining_gaps"]["all_request_runtime_capture_execution_manual_ready"])
        self.assertFalse(summary["remaining_gaps"]["all_request_runtime_capture_execution_automated_ready"])
        self.assertEqual(summary["remaining_gaps"]["all_request_runtime_capture_execution_request_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["all_request_runtime_capture_execution_ready_request_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["all_request_runtime_capture_execution_automated_request_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["all_request_runtime_capture_execution_pending_artifact_count"], 18)
        self.assertEqual(summary["remaining_gaps"]["all_request_runtime_capture_execution_artifact_execution_count"], 18)
        self.assertEqual(summary["remaining_gaps"]["all_request_runtime_capture_execution_command_option_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["all_request_runtime_capture_execution_manual_capture_required_count"], 18)
        self.assertEqual(summary["remaining_gaps"]["all_request_runtime_capture_execution_missing_capture_command_count"], 18)
        self.assertEqual(summary["remaining_gaps"]["all_request_runtime_capture_execution_missing_item_count"], 0)
        self.assertTrue(summary["remaining_gaps"]["all_request_manual_capture_runbook_ready"])
        self.assertEqual(summary["remaining_gaps"]["all_request_manual_capture_runbook_request_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["all_request_manual_capture_runbook_manual_task_count"], 18)
        self.assertEqual(summary["remaining_gaps"]["all_request_manual_capture_runbook_runtime_command_task_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["all_request_manual_capture_runbook_validator_command_count"], 30)
        self.assertEqual(summary["remaining_gaps"]["all_request_manual_capture_runbook_missing_item_count"], 0)
        self.assertTrue(summary["remaining_gaps"]["all_request_post_capture_intake_runbook_ready"])
        self.assertEqual(summary["remaining_gaps"]["all_request_post_capture_intake_runbook_request_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["all_request_post_capture_intake_runbook_artifact_gate_count"], 18)
        self.assertEqual(summary["remaining_gaps"]["all_request_post_capture_intake_runbook_ready_after_current_intake_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["all_request_post_capture_intake_runbook_missing_after_current_intake_count"], 18)
        self.assertEqual(summary["remaining_gaps"]["all_request_post_capture_intake_runbook_validator_command_count"], 30)
        self.assertEqual(summary["remaining_gaps"]["all_request_post_capture_intake_runbook_ready_to_update_bundle_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["all_request_post_capture_intake_runbook_phase4_candidate_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["all_request_post_capture_intake_runbook_live_spike_candidate_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["all_request_post_capture_intake_runbook_missing_source_request_path_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["all_request_post_capture_intake_runbook_missing_prompt_set_path_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["all_request_post_capture_intake_runbook_missing_explicit_approval_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["all_request_post_capture_intake_runbook_missing_prompt_traffic_ack_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["all_request_post_capture_intake_runbook_missing_item_count"], 0)
        self.assertTrue(summary["remaining_gaps"]["phase3_launch_card_library_ready"])
        self.assertFalse(summary["remaining_gaps"]["phase3_launch_card_execution_ready"])
        self.assertEqual(summary["remaining_gaps"]["phase3_launch_card_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["phase3_launch_card_template_ready_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["phase3_launch_card_model_plane_binding_ready_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["phase3_launch_card_binding_ready_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["phase3_launch_card_runtime_command_ready_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["phase3_launch_card_task_count"], 18)
        self.assertEqual(summary["remaining_gaps"]["phase3_launch_card_missing_runtime_command_count"], 18)
        self.assertTrue(summary["remaining_gaps"]["phase3_launch_card_binding_handoff_ready"])
        self.assertEqual(summary["remaining_gaps"]["phase3_launch_card_binding_handoff_task_count"], 18)
        self.assertEqual(summary["remaining_gaps"]["phase3_launch_card_binding_handoff_ready_count"], 18)
        self.assertEqual(summary["remaining_gaps"]["phase3_launch_card_binding_handoff_missing_field_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["phase3_launch_card_unbound_task_count"], 18)
        self.assertTrue(summary["remaining_gaps"]["phase3_launch_card_saved_handoff_artifacts_ready"])
        self.assertEqual(summary["remaining_gaps"]["phase3_launch_card_saved_handoff_artifact_missing_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["phase3_launch_card_saved_handoff_artifact_drifted_count"], 0)
        self.assertTrue(summary["remaining_gaps"]["approval_manifest_parity_ready"])
        self.assertEqual(summary["remaining_gaps"]["approval_manifest_parity_matched_request_count"], 6)
        self.assertTrue(summary["remaining_gaps"]["receipt_requirement_parity_ready"])
        self.assertEqual(summary["remaining_gaps"]["receipt_requirement_parity_matched_requirement_count"], 18)
        self.assertTrue(summary["remaining_gaps"]["operator_queue_parity_ready"])
        self.assertEqual(summary["remaining_gaps"]["operator_queue_parity_matched_request_count"], 6)
        self.assertTrue(summary["remaining_gaps"]["post_approval_capture_fill_parity_ready"])
        self.assertEqual(summary["remaining_gaps"]["post_approval_capture_fill_parity_matched_artifact_count"], 3)
        self.assertEqual(summary["remaining_gaps"]["post_approval_capture_fill_parity_runtime_pending_artifact_count"], 3)
        self.assertEqual(summary["remaining_gaps"]["post_approval_capture_fill_parity_capture_result_fill_step_count"], 3)
        self.assertTrue(summary["remaining_gaps"]["post_approval_capture_fill_parity_request_path_match"])
        self.assertTrue(summary["remaining_gaps"]["all_post_approval_capture_fill_parity_ready"])
        self.assertEqual(summary["remaining_gaps"]["all_post_approval_capture_fill_parity_matched_request_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["all_post_approval_capture_fill_parity_matched_artifact_count"], 18)
        self.assertEqual(summary["remaining_gaps"]["all_post_approval_capture_fill_parity_runtime_pending_artifact_count"], 18)
        self.assertEqual(summary["remaining_gaps"]["all_post_approval_capture_fill_parity_capture_result_fill_step_count"], 18)
        self.assertTrue(summary["remaining_gaps"]["all_request_capture_queue_ready"])
        self.assertEqual(summary["remaining_gaps"]["all_request_capture_queue_request_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["all_request_capture_queue_complete_fill_plan_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["all_request_capture_queue_pending_artifact_count"], 18)
        self.assertEqual(summary["remaining_gaps"]["all_request_capture_queue_validator_command_count"], 30)
        self.assertTrue(summary["remaining_gaps"]["all_request_validator_command_manifest_ready"])
        self.assertEqual(summary["remaining_gaps"]["all_request_validator_command_manifest_request_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["all_request_validator_command_manifest_artifact_entry_count"], 24)
        self.assertEqual(summary["remaining_gaps"]["all_request_validator_command_manifest_runtime_artifact_count"], 18)
        self.assertEqual(summary["remaining_gaps"]["all_request_validator_command_manifest_future_artifact_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["all_request_validator_command_manifest_runtime_command_count"], 30)
        self.assertEqual(summary["remaining_gaps"]["all_request_validator_command_manifest_future_command_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["all_request_validator_command_manifest_total_command_count"], 36)
        self.assertFalse(summary["remaining_gaps"]["all_request_downstream_handoff_manifest_ready"])
        self.assertEqual(summary["remaining_gaps"]["all_request_downstream_handoff_manifest_request_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["all_request_downstream_handoff_policy_ready_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["all_request_downstream_handoff_dense_ready_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["all_request_downstream_handoff_live_ready_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["all_request_downstream_handoff_all_ready_count"], 0)
        self.assertFalse(summary["remaining_gaps"]["all_request_receipt_fill_manifest_ready"])
        self.assertEqual(summary["remaining_gaps"]["all_request_receipt_fill_manifest_request_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["all_request_receipt_fill_manifest_entry_count"], 18)
        self.assertEqual(summary["remaining_gaps"]["all_request_receipt_fill_manifest_ready_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["all_request_receipt_fill_manifest_missing_count"], 18)
        self.assertEqual(summary["remaining_gaps"]["all_request_receipt_fill_manifest_approval_recorded_after_count"], 18)
        self.assertEqual(summary["remaining_gaps"]["all_request_receipt_fill_manifest_approval_missing_after_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["all_request_receipt_fill_manifest_blocked_after_approval_count"], 18)
        self.assertEqual(summary["remaining_gaps"]["all_request_receipt_fill_manifest_validator_command_count"], 30)
        self.assertEqual(summary["remaining_gaps"]["all_request_receipt_fill_manifest_candidate_router_trace_entry_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["all_request_receipt_fill_manifest_candidate_router_trace_ready_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["all_request_receipt_fill_manifest_candidate_router_trace_missing_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["all_request_receipt_fill_manifest_managed_output_entry_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["all_request_receipt_fill_manifest_managed_output_ready_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["all_request_receipt_fill_manifest_managed_output_missing_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["all_request_receipt_fill_manifest_dense_output_entry_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["all_request_receipt_fill_manifest_dense_output_ready_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["all_request_receipt_fill_manifest_dense_output_missing_count"], 6)
        self.assertTrue(summary["remaining_gaps"]["all_request_receipt_validator_parity_ready"])
        self.assertEqual(summary["remaining_gaps"]["all_request_receipt_validator_parity_runtime_artifact_count"], 18)
        self.assertEqual(summary["remaining_gaps"]["all_request_receipt_validator_parity_receipt_entry_count"], 18)
        self.assertEqual(summary["remaining_gaps"]["all_request_receipt_validator_parity_matched_artifact_count"], 18)
        self.assertEqual(summary["remaining_gaps"]["all_request_receipt_validator_parity_metadata_mismatch_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["all_request_receipt_validator_parity_missing_from_receipt_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["all_request_receipt_validator_parity_missing_from_runtime_count"], 0)
        self.assertTrue(summary["remaining_gaps"]["all_request_receipt_fill_command_manifest_ready"])
        self.assertEqual(summary["remaining_gaps"]["all_request_receipt_fill_command_manifest_request_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["all_request_receipt_fill_command_manifest_entry_count"], 18)
        self.assertEqual(summary["remaining_gaps"]["all_request_receipt_fill_command_manifest_ready_receipt_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["all_request_receipt_fill_command_manifest_blocked_after_approval_count"], 18)
        self.assertEqual(summary["remaining_gaps"]["all_request_receipt_fill_command_manifest_validator_command_count"], 30)
        self.assertEqual(summary["remaining_gaps"]["all_request_receipt_fill_command_manifest_missing_command_entry_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["all_request_receipt_fill_command_manifest_missing_receipt_path_count"], 0)
        self.assertTrue(summary["remaining_gaps"]["phase3_blocker_closure_manifest_ready"])
        self.assertTrue(summary["remaining_gaps"]["phase3_blocker_closure_mapping_ready"])
        self.assertFalse(summary["remaining_gaps"]["phase3_blocker_closure_evidence_complete"])
        self.assertFalse(summary["remaining_gaps"]["phase3_blocker_closure_closure_complete"])
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_closure_ready_scope"], "mapping_ready_only")
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_closure_reason_count"], 7)
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_closure_mapped_reason_count"], 7)
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_closure_unmapped_reason_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_closure_runtime_capture_required_count"], 4)
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_closure_future_adapter_required_count"], 3)
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_closure_validator_command_count"], 41)
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_closure_missing_evidence_count"], 25)
        self.assertTrue(summary["remaining_gaps"]["phase3_blocker_evidence_ledger_ready"])
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_evidence_ledger_reason_count"], 7)
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_evidence_ledger_row_count"], 25)
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_evidence_ledger_expected_missing_evidence_count"], 25)
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_evidence_ledger_runtime_capture_required_row_count"], 22)
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_evidence_ledger_future_adapter_required_row_count"], 3)
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_evidence_ledger_receipt_bound_row_count"], 18)
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_evidence_ledger_dense_output_row_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_evidence_ledger_live_proof_row_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_evidence_ledger_runtime_actuator_row_count"], 2)
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_evidence_ledger_validator_command_count"], 41)
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_evidence_ledger_missing_item_count"], 0)
        self.assertTrue(summary["remaining_gaps"]["phase3_blocker_resolution_queue_ready"])
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_resolution_queue_work_package_count"], 5)
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_resolution_queue_row_count"], 25)
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_resolution_queue_expected_ledger_row_count"], 25)
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_resolution_queue_runtime_capture_package_count"], 3)
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_resolution_queue_future_adapter_package_count"], 1)
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_resolution_queue_runtime_capture_required_row_count"], 22)
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_resolution_queue_future_adapter_required_row_count"], 3)
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_resolution_queue_receipt_bound_row_count"], 18)
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_resolution_queue_validator_command_count"], 41)
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_resolution_queue_completion_gate_count"], 21)
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_resolution_queue_dependency_edge_count"], 4)
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_resolution_queue_missing_item_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_resolution_queue_next_unblocked_work_package_id"], "policy_candidate_trace_capture")
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_resolution_queue_next_unblocked_sequence_rank"], 1)
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_resolution_queue_next_unblocked_operator_stage"], "approved_runtime_capture")
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_resolution_queue_next_unblocked_package_class"], "policy_candidate_replay")
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_resolution_queue_next_unblocked_row_count"], 2)
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_resolution_queue_next_unblocked_validator_command_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["phase3_blocker_resolution_queue_next_unblocked_completion_gate_count"], 4)
        self.assertIn("capture the candidate router trace", summary["remaining_gaps"]["phase3_blocker_resolution_queue_next_unblocked_next_action"])
        self.assertTrue(summary["remaining_gaps"]["phase3_next_unblocked_operator_handoff_ready"])
        self.assertTrue(summary["remaining_gaps"]["phase3_next_unblocked_operator_handoff_work_order_ready"])
        self.assertEqual(
            summary["remaining_gaps"]["phase3_next_unblocked_operator_handoff_work_order_artifact"],
            "recommended-runtime-capture-work-order.json",
        )
        self.assertTrue(summary["remaining_gaps"]["phase3_next_unblocked_operator_handoff_receipt_template_ready"])
        self.assertTrue(summary["remaining_gaps"]["phase3_next_unblocked_operator_handoff_validation_command_ready"])
        self.assertTrue(summary["remaining_gaps"]["phase3_next_unblocked_operator_handoff_work_order_advances_next_package"])
        self.assertEqual(summary["remaining_gaps"]["capture_result_request_drift_free_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["capture_result_request_drifted_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["capture_result_capture_receipts_required_count"], 18)
        self.assertEqual(summary["remaining_gaps"]["capture_result_capture_receipts_ready_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["capture_result_candidate_trace_receipt_ready_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["capture_result_managed_output_receipt_ready_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["capture_result_dense_output_receipt_ready_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["capture_result_live_capability_proof_ready_count"], 0)
        self.assertFalse(summary["remaining_gaps"]["capture_result_all_capture_receipts_ready"])
        self.assertFalse(summary["remaining_gaps"]["capture_result_all_output_receipt_bindings_ready"])
        self.assertEqual(summary["remaining_gaps"]["capture_result_receipt_fill_entry_count"], 18)
        self.assertEqual(summary["remaining_gaps"]["capture_result_receipt_fill_ready_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["capture_result_receipt_fill_missing_count"], 18)
        self.assertEqual(summary["remaining_gaps"]["capture_result_receipt_fill_approval_missing_count"], 18)
        self.assertEqual(summary["remaining_gaps"]["capture_result_receipt_fill_candidate_router_trace_entry_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["capture_result_receipt_fill_candidate_router_trace_ready_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["capture_result_receipt_fill_candidate_router_trace_missing_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["capture_result_receipt_fill_managed_output_entry_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["capture_result_receipt_fill_managed_output_ready_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["capture_result_receipt_fill_managed_output_missing_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["capture_result_receipt_fill_dense_output_entry_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["capture_result_receipt_fill_dense_output_ready_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["capture_result_receipt_fill_dense_output_missing_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["capture_result_approval_transition_preview_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["capture_result_approval_transition_ready_for_operator_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["capture_result_approval_transition_ready_to_update_bundle_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["capture_result_post_approval_capture_fill_plan_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["capture_result_post_approval_capture_fill_artifact_step_count"], 18)
        self.assertEqual(summary["remaining_gaps"]["capture_result_post_approval_capture_fill_ready_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["capture_result_post_approval_capture_fill_missing_count"], 18)
        self.assertEqual(summary["remaining_gaps"]["capture_result_post_approval_capture_fill_validator_command_count"], 30)
        self.assertEqual(summary["remaining_gaps"]["capture_result_post_approval_capture_fill_ready_to_update_bundle_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["capture_result_missing_receipt_gate_request_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["capture_result_output_receipt_binding_ready_count"], 0)
        self.assertEqual(summary["remaining_gaps"]["capture_result_ready_to_update_bundle_count"], 0)
        self.assertIn("packet does not claim live expert paging", summary["safety_contract"])

    def test_command_contract_gate_uses_saved_request_file_when_present(self) -> None:
        paths = fixture_paths(0)
        contract = {
            "schema_version": "moe-phase3-runtime-capture-command-contract-v1",
            "mode": "phase3_runtime_capture_command_contract",
            "valid": True,
            "errors": [],
            "request_path": paths["request_path"],
            "request_name": paths["name"],
            "command_contract_ready": True,
            "runtime_capture_command_ready": False,
            "planned_capture_count": 3,
            "runtime_command_option_count": 0,
            "missing_runtime_command_count": 3,
            "missing_runtime_command_artifact_ids": [
                "candidate_router_trace",
                "managed_output_summary_fill",
                "dense_output_summary_fill",
            ],
            "capture_tasks": [
                {
                    "artifact_id": "candidate_router_trace",
                    "capture_kind": "llama_cpp_router_trace_jsonl",
                    "artifact_path": paths["candidate_trace_path"],
                    "receipt_path": paths["candidate_receipt_path"],
                    "validator_command_count": 3,
                    "command_binding_ready": False,
                    "planned_command_binding": {
                        "command_class": "phase3_model_plane_launch_card_runtime_capture",
                        "planned_only": True,
                        "metadata_only": False,
                        "command_ready": False,
                        "command": [],
                        "missing_binding": "model_plane_launch_card_command_missing",
                    },
                },
                {
                    "artifact_id": "managed_output_summary_fill",
                    "capture_kind": "managed_output_summary_json",
                    "artifact_path": paths["managed_output_path"],
                    "receipt_path": paths["managed_output_path"],
                    "validator_command_count": 1,
                    "command_binding_ready": False,
                    "planned_command_binding": {"command_ready": False},
                },
                {
                    "artifact_id": "dense_output_summary_fill",
                    "capture_kind": "dense_output_summary_json",
                    "artifact_path": paths["dense_output_path"],
                    "receipt_path": paths["dense_output_path"],
                    "validator_command_count": 1,
                    "command_binding_ready": False,
                    "planned_command_binding": {"command_ready": False},
                },
            ],
        }
        original_root = planner.ROOT
        original_build_summary = planner.plan_phase3_runtime_capture_commands.build_summary
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                planner.ROOT = Path(temp_dir)
                request_path = planner.ROOT / paths["request_path"]
                request_path.parent.mkdir(parents=True, exist_ok=True)
                request_path.write_text("{}", encoding="utf-8")
                planner.plan_phase3_runtime_capture_commands.build_summary = lambda _path: contract

                summary = planner.build_packet_summary(
                    TRACE_FIXTURE,
                    INVENTORY_FIXTURE,
                    POLICIES_FIXTURE,
                    MANAGED_PLAN,
                )
        finally:
            planner.ROOT = original_root
            planner.plan_phase3_runtime_capture_commands.build_summary = original_build_summary

        by_item = {item["id"]: item for item in summary["evidence_items"]}
        by_check = {item["id"]: item for item in summary["promotion_checklist"]}
        self.assertEqual(by_item["recommended_runtime_capture_command_contract"]["status"], "proven")
        self.assertEqual(by_check["recommended_runtime_capture_command_contract"]["status"], "satisfied")
        details = by_item["recommended_runtime_capture_command_contract"]["details"]
        self.assertTrue(details["ready"])
        self.assertFalse(details["runtime_capture_command_ready"])
        self.assertEqual(details["planned_capture_count"], 3)
        self.assertEqual(details["runtime_command_option_count"], 0)
        self.assertEqual(details["missing_runtime_command_count"], 3)
        self.assertEqual(
            details["missing_runtime_command_artifact_ids"],
            ["candidate_router_trace", "managed_output_summary_fill", "dense_output_summary_fill"],
        )
        contract_by_artifact = {item["artifact_id"]: item for item in details["contract"]["capture_tasks"]}
        self.assertEqual(
            contract_by_artifact["candidate_router_trace"]["capture_kind"],
            "llama_cpp_router_trace_jsonl",
        )
        candidate_binding = contract_by_artifact["candidate_router_trace"]["planned_command_binding"]
        self.assertEqual(candidate_binding["command_class"], "phase3_model_plane_launch_card_runtime_capture")
        self.assertFalse(candidate_binding["metadata_only"])
        self.assertFalse(candidate_binding["command_ready"])
        self.assertEqual(candidate_binding["command"], [])
        self.assertTrue(summary["remaining_gaps"]["recommended_runtime_capture_command_contract_ready"])
        self.assertFalse(summary["remaining_gaps"]["recommended_runtime_capture_command_contract_runtime_ready"])
        report = planner.format_markdown_report(summary)
        self.assertIn("## Recommended Runtime Capture Command Contract", report)
        self.assertIn("Contract ready: `True`", report)
        self.assertIn("Runtime commands ready: `False`", report)
        self.assertIn("Missing runtime commands: `3`", report)
        self.assertIn("| candidate_router_trace | llama_cpp_router_trace_jsonl | memory-moe-mvp/phase3-real-evidence/fixture/candidate-router-events.jsonl | memory-moe-mvp/phase3-real-evidence/fixture/candidate-router-events.capture-receipt.json | False | 3 |", report)
        with tempfile.TemporaryDirectory() as output_dir:
            launch_card_path = Path(output_dir) / "recommended-launch-card.json"
            planner.write_recommended_runtime_capture_launch_card(summary, launch_card_path)
            launch_card = json.loads(launch_card_path.read_text(encoding="utf-8"))

        self.assertEqual(launch_card["schema_version"], "moe-phase3-runtime-capture-launch-card-v1")
        self.assertEqual(launch_card["status"], "planned_only")
        self.assertFalse(launch_card["executable"])
        self.assertEqual(launch_card["task_count"], 3)
        self.assertFalse(launch_card["runtime_capture_command_ready"])
        self.assertIn("binding_template", launch_card["tasks"][0])

        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir)
            handoff_manifest = planner.write_operator_handoff_dir(summary, output_path)
            written_names = {item["path"] for item in handoff_manifest["output_files"]}
            expected_names = {
                "README.md",
                "operator-handoff-manifest.json",
                "phase3-evidence-packet.json",
                "phase3-evidence-packet.md",
                "launch-card-binding-worksheet.json",
                "model-plane-artifact-writer-contract-request.json",
                "recommended-runtime-capture-launch-card.template.json",
                "recommended-runtime-capture-preflight.json",
                "recommended-runtime-capture-command-contract.json",
                "recommended-runtime-capture-execution-coverage.json",
                "all-request-runtime-capture-execution-coverage.json",
                "manual-capture-runbook.json",
                "post-capture-intake-runbook.json",
                "recommended-manual-capture-runbook.json",
                "recommended-post-capture-intake-runbook.json",
                "recommended-runtime-capture-work-order.json",
                "recommended-runtime-capture-completion-receipt.template.json",
                "capture-queue.json",
                "approval-command-manifest.json",
                "validator-command-manifest.json",
                "receipt-fill-manifest.json",
                "receipt-fill-command-manifest.json",
                "downstream-handoff-manifest.json",
                "blocker-closure-manifest.json",
                "blocker-evidence-ledger.json",
                "blocker-resolution-queue.json",
            }
            self.assertTrue(expected_names.issubset(written_names))
            for name in expected_names:
                self.assertTrue((output_path / name).exists(), name)
            saved_manifest = json.loads((output_path / "operator-handoff-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_manifest["schema_version"], planner.OPERATOR_HANDOFF_SCHEMA_VERSION)
            self.assertTrue(saved_manifest["metadata_only"])
            self.assertFalse(saved_manifest["launches_runtimes"])
            self.assertFalse(saved_manifest["sends_prompt_traffic"])
            self.assertTrue(saved_manifest["operator_handoff_ready"])
            self.assertFalse(saved_manifest["phase3_complete"])
            self.assertEqual(saved_manifest["recommended_request"]["request_name"], "Fixture Mixtral Phase 3 real evidence")
            self.assertEqual(saved_manifest["recommended_request"]["next_artifact_id"], "candidate_router_trace")
            self.assertEqual(saved_manifest["recommended_launch_card_template"]["task_count"], 3)
            self.assertTrue(saved_manifest["handoff_counts"]["model_plane_artifact_writer_contract_request_ready"])
            self.assertEqual(saved_manifest["handoff_counts"]["model_plane_artifact_writer_contract_request_task_count"], 18)
            self.assertTrue(saved_manifest["handoff_counts"]["launch_card_saved_handoff_artifacts_ready"])
            self.assertEqual(saved_manifest["handoff_counts"]["launch_card_saved_handoff_artifact_missing_count"], 0)
            self.assertEqual(saved_manifest["handoff_counts"]["launch_card_saved_handoff_artifact_drifted_count"], 0)
            self.assertEqual(saved_manifest["handoff_counts"]["capture_queue_request_count"], 6)
            self.assertEqual(saved_manifest["handoff_counts"]["receipt_fill_missing_count"], 18)
            self.assertEqual(saved_manifest["handoff_counts"]["receipt_fill_candidate_router_trace_entry_count"], 6)
            self.assertEqual(saved_manifest["handoff_counts"]["receipt_fill_candidate_router_trace_ready_count"], 0)
            self.assertEqual(saved_manifest["handoff_counts"]["receipt_fill_candidate_router_trace_missing_count"], 6)
            self.assertEqual(saved_manifest["handoff_counts"]["receipt_fill_managed_output_entry_count"], 6)
            self.assertEqual(saved_manifest["handoff_counts"]["receipt_fill_managed_output_ready_count"], 0)
            self.assertEqual(saved_manifest["handoff_counts"]["receipt_fill_managed_output_missing_count"], 6)
            self.assertEqual(saved_manifest["handoff_counts"]["receipt_fill_dense_output_entry_count"], 6)
            self.assertEqual(saved_manifest["handoff_counts"]["receipt_fill_dense_output_ready_count"], 0)
            self.assertEqual(saved_manifest["handoff_counts"]["receipt_fill_dense_output_missing_count"], 6)
            self.assertTrue(saved_manifest["handoff_counts"]["all_execution_manual_ready"])
            self.assertFalse(saved_manifest["handoff_counts"]["all_execution_automated_ready"])
            self.assertEqual(saved_manifest["handoff_counts"]["all_execution_request_count"], 6)
            self.assertEqual(saved_manifest["handoff_counts"]["all_execution_pending_artifact_count"], 18)
            self.assertEqual(saved_manifest["handoff_counts"]["all_execution_missing_capture_command_count"], 18)
            self.assertTrue(saved_manifest["handoff_counts"]["post_capture_intake_runbook_ready"])
            self.assertEqual(saved_manifest["handoff_counts"]["post_capture_intake_runbook_artifact_gate_count"], 18)
            self.assertEqual(saved_manifest["handoff_counts"]["post_capture_intake_runbook_missing_after_current_intake_count"], 18)
            self.assertEqual(saved_manifest["handoff_counts"]["post_capture_intake_runbook_validator_command_count"], 30)
            self.assertTrue(saved_manifest["handoff_counts"]["blocker_evidence_ledger_ready"])
            self.assertEqual(saved_manifest["handoff_counts"]["blocker_evidence_ledger_row_count"], 25)
            self.assertEqual(saved_manifest["handoff_counts"]["blocker_evidence_ledger_expected_missing_evidence_count"], 25)
            self.assertEqual(saved_manifest["handoff_counts"]["blocker_evidence_ledger_validator_command_count"], 41)
            self.assertTrue(saved_manifest["handoff_counts"]["blocker_resolution_queue_ready"])
            self.assertEqual(saved_manifest["handoff_counts"]["blocker_resolution_queue_work_package_count"], 5)
            self.assertEqual(saved_manifest["handoff_counts"]["blocker_resolution_queue_row_count"], 25)
            self.assertEqual(saved_manifest["handoff_counts"]["blocker_resolution_queue_expected_ledger_row_count"], 25)
            self.assertEqual(saved_manifest["handoff_counts"]["blocker_resolution_queue_validator_command_count"], 41)
            self.assertEqual(saved_manifest["handoff_counts"]["blocker_resolution_queue_completion_gate_count"], 21)
            self.assertEqual(saved_manifest["handoff_counts"]["blocker_resolution_queue_dependency_edge_count"], 4)
            self.assertEqual(
                saved_manifest["handoff_counts"]["blocker_resolution_queue_next_unblocked_work_package_id"],
                "policy_candidate_trace_capture",
            )
            self.assertEqual(saved_manifest["handoff_counts"]["blocker_resolution_queue_next_unblocked_sequence_rank"], 1)
            self.assertEqual(saved_manifest["handoff_counts"]["blocker_resolution_queue_next_unblocked_operator_stage"], "approved_runtime_capture")
            self.assertEqual(saved_manifest["handoff_counts"]["blocker_resolution_queue_next_unblocked_validator_command_count"], 6)
            self.assertTrue(saved_manifest["handoff_counts"]["blocker_resolution_queue_runtime_actuator_proof_handoff_ready"])
            self.assertFalse(saved_manifest["handoff_counts"]["blocker_resolution_queue_runtime_actuator_live_spike_ready"])
            self.assertEqual(
                saved_manifest["handoff_counts"]["blocker_resolution_queue_runtime_actuator_proof_requirement_count"],
                8,
            )
            self.assertEqual(
                saved_manifest["handoff_counts"]["blocker_resolution_queue_runtime_actuator_proof_requirement_id_count"],
                8,
            )
            self.assertEqual(saved_manifest["handoff_counts"]["blocker_resolution_queue_runtime_actuator_proof_artifact_count"], 20)
            self.assertEqual(saved_manifest["handoff_counts"]["blocker_resolution_queue_runtime_actuator_dependency_edge_count"], 14)
            self.assertEqual(saved_manifest["handoff_counts"]["blocker_resolution_queue_runtime_actuator_blocking_capability_count"], 7)
            self.assertEqual(saved_manifest["handoff_counts"]["blocker_resolution_queue_runtime_actuator_control_blocker_count"], 3)
            blocker_ledger_file = json.loads((output_path / "blocker-evidence-ledger.json").read_text(encoding="utf-8"))
            self.assertTrue(blocker_ledger_file["ready"])
            self.assertEqual(blocker_ledger_file["evidence_row_count"], 25)
            blocker_queue_file = json.loads((output_path / "blocker-resolution-queue.json").read_text(encoding="utf-8"))
            self.assertTrue(blocker_queue_file["ready"])
            self.assertEqual(blocker_queue_file["queue_row_count"], 25)
            self.assertEqual(blocker_queue_file["validator_command_count"], 41)
            self.assertEqual(blocker_queue_file["completion_gate_count"], 21)
            self.assertEqual(blocker_queue_file["dependency_edge_count"], 4)
            self.assertEqual(blocker_queue_file["next_unblocked_work_package"]["work_package_id"], "policy_candidate_trace_capture")
            blocker_queue_packages = {
                item["work_package_id"]: item
                for item in blocker_queue_file["work_packages"]
            }
            self.assertEqual(blocker_queue_packages["runtime_actuator_spike"]["proof_handoff"]["proof_requirement_count"], 8)
            self.assertEqual(blocker_queue_packages["runtime_actuator_spike"]["proof_handoff"]["proof_artifact_count"], 20)
            all_execution_file = json.loads((output_path / "all-request-runtime-capture-execution-coverage.json").read_text(encoding="utf-8"))
            self.assertEqual(all_execution_file["request_count"], 6)
            self.assertEqual(all_execution_file["pending_artifact_count"], 18)
            post_capture_file = json.loads((output_path / "post-capture-intake-runbook.json").read_text(encoding="utf-8"))
            self.assertTrue(post_capture_file["ready"])
            self.assertEqual(post_capture_file["artifact_gate_count"], 18)
            self.assertEqual(post_capture_file["missing_after_current_intake_count"], 18)
            recommended_manual_file = json.loads((output_path / "recommended-manual-capture-runbook.json").read_text(encoding="utf-8"))
            self.assertTrue(recommended_manual_file["ready"])
            self.assertEqual(recommended_manual_file["request_count"], 1)
            self.assertEqual(recommended_manual_file["request_path"], saved_manifest["recommended_request"]["request_path"])
            self.assertEqual(
                saved_manifest["handoff_counts"]["recommended_manual_runbook_manual_task_count"],
                recommended_manual_file["manual_task_count"],
            )
            recommended_post_capture_file = json.loads((output_path / "recommended-post-capture-intake-runbook.json").read_text(encoding="utf-8"))
            self.assertTrue(recommended_post_capture_file["ready"])
            self.assertEqual(recommended_post_capture_file["request_count"], 1)
            self.assertEqual(recommended_post_capture_file["request_path"], saved_manifest["recommended_request"]["request_path"])
            self.assertEqual(
                saved_manifest["handoff_counts"]["recommended_post_capture_intake_runbook_artifact_gate_count"],
                recommended_post_capture_file["artifact_gate_count"],
            )
            recommended_work_order_file = json.loads((output_path / "recommended-runtime-capture-work-order.json").read_text(encoding="utf-8"))
            self.assertTrue(recommended_work_order_file["ready"])
            self.assertEqual(recommended_work_order_file["request_path"], saved_manifest["recommended_request"]["request_path"])
            self.assertEqual(recommended_work_order_file["capture_step_count"], recommended_manual_file["manual_task_count"])
            self.assertEqual(recommended_work_order_file["artifact_gate_count"], recommended_post_capture_file["artifact_gate_count"])
            self.assertEqual(
                [step["step_id"] for step in recommended_work_order_file["post_capture_sequence"]],
                [
                    "record_runtime_approvals",
                    "capture_runtime_artifacts",
                    "fill_completion_receipt",
                    "validate_completion_receipt",
                    "run_capture_result_intake",
                ],
            )
            self.assertTrue(recommended_work_order_file["completion_receipt_validation_command_ready"])
            self.assertTrue(recommended_work_order_file["intake_step"]["requires_completion_receipt_validation"])
            self.assertEqual(
                saved_manifest["handoff_counts"]["recommended_work_order_capture_step_count"],
                recommended_work_order_file["capture_step_count"],
            )
            self.assertTrue(saved_manifest["handoff_counts"]["recommended_work_order_post_capture_sequence_ready"])
            self.assertEqual(saved_manifest["handoff_counts"]["recommended_work_order_post_capture_sequence_step_count"], 5)
            self.assertTrue(saved_manifest["handoff_counts"]["recommended_work_order_completion_validation_before_intake"])
            completion_receipt_file = json.loads((output_path / "recommended-runtime-capture-completion-receipt.template.json").read_text(encoding="utf-8"))
            self.assertTrue(completion_receipt_file["template_ready"])
            self.assertFalse(completion_receipt_file["receipt_complete"])
            self.assertFalse(completion_receipt_file["ready_for_capture_result_intake"])
            self.assertEqual(completion_receipt_file["request_path"], saved_manifest["recommended_request"]["request_path"])
            self.assertEqual(completion_receipt_file["capture_receipt_count"], recommended_work_order_file["capture_step_count"])
            self.assertEqual(completion_receipt_file["validator_command_count"], recommended_work_order_file["validator_command_count"])
            validation_step = completion_receipt_file["completion_receipt_validation_step"]
            self.assertEqual(validation_step["command_class"], "phase3_capture_completion_receipt_validation")
            self.assertIn("scripts/plan_phase3_capture_completion_receipt.py", validation_step["command"])
            self.assertIn("--work-order", validation_step["command"])
            self.assertEqual(saved_manifest["completion_receipt_validation"], validation_step)
            self.assertTrue(saved_manifest["handoff_counts"]["recommended_completion_receipt_validation_command_ready"])
            self.assertEqual(
                saved_manifest["handoff_counts"]["recommended_completion_receipt_capture_receipt_count"],
                completion_receipt_file["capture_receipt_count"],
            )
            next_handoff_file = json.loads((output_path / "next-unblocked-operator-handoff.json").read_text(encoding="utf-8"))
            self.assertTrue(next_handoff_file["handoff_ready"])
            self.assertEqual(next_handoff_file["next_unblocked_work_package_id"], "policy_candidate_trace_capture")
            self.assertEqual(next_handoff_file["work_order_artifact"], "recommended-runtime-capture-work-order.json")
            self.assertEqual(next_handoff_file["completion_receipt_template_artifact"], "recommended-runtime-capture-completion-receipt.template.json")
            self.assertTrue(next_handoff_file["work_order_advances_next_package"])
            self.assertTrue(next_handoff_file["work_order_bound_to_completion_receipt"])
            self.assertEqual(saved_manifest["next_unblocked_operator_handoff"], next_handoff_file)
            self.assertTrue(saved_manifest["handoff_counts"]["next_unblocked_operator_handoff_ready"])
            self.assertTrue(saved_manifest["handoff_counts"]["next_unblocked_operator_handoff_work_order_ready"])
            self.assertEqual(
                saved_manifest["handoff_counts"]["next_unblocked_operator_handoff_work_order_artifact"],
                "recommended-runtime-capture-work-order.json",
            )
            self.assertTrue(saved_manifest["handoff_counts"]["next_unblocked_operator_handoff_validation_command_ready"])
            self.assertEqual(saved_manifest["warnings"], [])
            worksheet = json.loads((output_path / "launch-card-binding-worksheet.json").read_text(encoding="utf-8"))
            model_plane_contract_request = json.loads((output_path / "model-plane-artifact-writer-contract-request.json").read_text(encoding="utf-8"))
            self.assertEqual(worksheet["schema_version"], planner.plan_phase3_launch_card_library.BINDING_WORKSHEET_SCHEMA_VERSION)
            self.assertEqual(worksheet["task_count"], 18)
            self.assertEqual(worksheet["command_option_count"], 0)
            self.assertEqual(
                model_plane_contract_request["schema_version"],
                planner.plan_phase3_launch_card_library.MODEL_PLANE_ARTIFACT_WRITER_REQUEST_SCHEMA_VERSION,
            )
            self.assertTrue(model_plane_contract_request["request_ready"])
            self.assertFalse(model_plane_contract_request["execution_ready"])
            self.assertEqual(model_plane_contract_request["task_count"], 18)
            self.assertEqual(model_plane_contract_request["phase3_artifact_writer_contract_count"], 18)
            self.assertEqual(model_plane_contract_request["phase3_artifact_writer_ready_count"], 0)
            handoff_card = json.loads((output_path / "recommended-runtime-capture-launch-card.template.json").read_text(encoding="utf-8"))
            self.assertEqual(handoff_card["schema_version"], "moe-phase3-runtime-capture-launch-card-v1")
            self.assertFalse(handoff_card["executable"])
            readme = (output_path / "README.md").read_text(encoding="utf-8")
            self.assertIn("# Phase 3 Operator Handoff", readme)
            self.assertIn("Metadata only: `True`", readme)
            self.assertIn("Receipt-fill ready: `False`", readme)
            self.assertIn("Recommended work order post-capture sequence ready: `True`", readme)
            self.assertIn("Recommended work order validation before intake: `True`", readme)
            self.assertIn("## Completion Receipt Validation", readme)
            self.assertIn("scripts/plan_phase3_capture_completion_receipt.py", readme)
            self.assertIn("Post-capture intake runbook ready: `True`", readme)

    def test_runtime_launch_card_binding_promotes_execution_coverage(self) -> None:
        paths = fixture_paths(0)
        capture_tasks = []
        for artifact_id, capture_kind, artifact_path, receipt_path in (
            ("candidate_router_trace", "llama_cpp_router_trace_jsonl", paths["candidate_trace_path"], paths["candidate_receipt_path"]),
            ("managed_output_summary_fill", "managed_output_summary_json", paths["managed_output_path"], paths["managed_output_path"]),
            ("dense_output_summary_fill", "dense_output_summary_json", paths["dense_output_path"], paths["dense_output_path"]),
        ):
            capture_tasks.append(
                {
                    "artifact_id": artifact_id,
                    "capture_kind": capture_kind,
                    "artifact_path": artifact_path,
                    "receipt_path": receipt_path,
                    "validator_command_count": 1,
                    "command_binding_ready": True,
                    "planned_command_binding": {
                        "command_class": "phase3_model_plane_launch_card_runtime_capture",
                        "command_ready": True,
                        "requires_runtime": True,
                        "requires_prompt_traffic": True,
                        "metadata_only": False,
                        "model_plane_callable_id": f"callable-{artifact_id}",
                        "launch_command": ["model-plane-callable", f"callable-{artifact_id}"],
                        "missing_binding": None,
                    },
                }
            )
        contract = {
            "schema_version": "moe-phase3-runtime-capture-command-contract-v1",
            "mode": "phase3_runtime_capture_command_contract",
            "valid": True,
            "errors": [],
            "request_path": paths["request_path"],
            "request_name": paths["name"],
            "command_contract_ready": True,
            "runtime_capture_command_ready": True,
            "planned_capture_count": 3,
            "runtime_command_option_count": 3,
            "missing_runtime_command_count": 0,
            "missing_runtime_command_artifact_ids": [],
            "launch_card_binding_summary": {
                "valid": True,
                "binding_ready": True,
                "launch_card_path": "memory-moe-mvp/phase3-real-evidence/fixture-filled-launch-card.json",
                "bound_task_count": 3,
                "command_option_count": 3,
                "missing_runtime_command_count": 0,
                "blockers": [],
            },
            "capture_tasks": capture_tasks,
        }
        original_root = planner.ROOT
        original_build_summary = planner.plan_phase3_runtime_capture_commands.build_summary
        captured_kwargs = {}
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                planner.ROOT = Path(temp_dir)
                request_path = planner.ROOT / paths["request_path"]
                request_path.parent.mkdir(parents=True, exist_ok=True)
                request_path.write_text("{}", encoding="utf-8")
                launch_card_path = Path(temp_dir) / "filled-launch-card.json"
                launch_card_path.write_text("{}", encoding="utf-8")

                def fake_build_summary(_path, **kwargs):
                    captured_kwargs.update(kwargs)
                    return contract

                planner.plan_phase3_runtime_capture_commands.build_summary = fake_build_summary
                summary = planner.build_packet_summary(
                    TRACE_FIXTURE,
                    INVENTORY_FIXTURE,
                    POLICIES_FIXTURE,
                    MANAGED_PLAN,
                    runtime_capture_launch_card_path=launch_card_path,
                )
        finally:
            planner.ROOT = original_root
            planner.plan_phase3_runtime_capture_commands.build_summary = original_build_summary

        self.assertEqual(captured_kwargs["launch_card_path"], launch_card_path)
        contract_summary = summary["recommended_runtime_capture_command_contract_summary"]
        self.assertTrue(contract_summary["ready"])
        self.assertTrue(contract_summary["runtime_capture_command_ready"])
        self.assertTrue(contract_summary["launch_card_binding_ready"])
        self.assertEqual(contract_summary["launch_card_binding_bound_task_count"], 3)
        self.assertEqual(contract_summary["launch_card_binding_command_option_count"], 3)
        execution_summary = summary["recommended_runtime_capture_execution_coverage_summary"]
        self.assertTrue(execution_summary["manual_operator_capture_ready"])
        self.assertTrue(execution_summary["automated_capture_ready"])
        self.assertEqual(execution_summary["capture_command_option_count"], 3)
        self.assertEqual(execution_summary["manual_capture_required_count"], 0)
        self.assertEqual(execution_summary["missing_capture_command_count"], 0)
        self.assertEqual(execution_summary["artifacts_with_capture_command_count"], 3)
        self.assertTrue(summary["remaining_gaps"]["recommended_runtime_capture_launch_card_binding_ready"])
        self.assertEqual(summary["remaining_gaps"]["recommended_runtime_capture_launch_card_binding_bound_task_count"], 3)
        self.assertEqual(summary["remaining_gaps"]["recommended_runtime_capture_execution_missing_capture_command_count"], 0)
        all_execution_summary = summary["all_request_runtime_capture_execution_coverage_summary"]
        self.assertTrue(all_execution_summary["manual_operator_capture_ready"])
        self.assertFalse(all_execution_summary["automated_capture_ready"])
        self.assertEqual(all_execution_summary["request_count"], 6)
        self.assertEqual(all_execution_summary["manual_operator_capture_ready_count"], 6)
        self.assertEqual(all_execution_summary["automated_capture_ready_count"], 1)
        self.assertEqual(all_execution_summary["pending_artifact_count"], 18)
        self.assertEqual(all_execution_summary["capture_command_option_count"], 3)
        self.assertEqual(all_execution_summary["manual_capture_required_count"], 15)
        self.assertEqual(all_execution_summary["missing_capture_command_count"], 15)
        self.assertTrue(summary["remaining_gaps"]["all_request_runtime_capture_execution_manual_ready"])
        self.assertFalse(summary["remaining_gaps"]["all_request_runtime_capture_execution_automated_ready"])
        self.assertEqual(summary["remaining_gaps"]["all_request_runtime_capture_execution_automated_request_count"], 1)
        self.assertEqual(summary["remaining_gaps"]["all_request_runtime_capture_execution_missing_capture_command_count"], 15)
        artifact_execution = summary["recommended_runtime_capture_execution_coverage_manifest"]["artifact_execution"]
        self.assertEqual({item["capture_mode"] for item in artifact_execution}, {"runtime_command_available"})
        self.assertTrue(all(item["launch_card_binding_ready"] for item in artifact_execution))
        report = planner.format_markdown_report(summary)
        self.assertIn("Launch-card binding ready: `True`", report)
        self.assertIn("Automated capture ready: `True`", report)

    def test_runtime_launch_card_directory_promotes_all_request_execution_coverage(self) -> None:
        captured_launch_card_paths = []

        def fake_build_summary(request_path, **kwargs):
            request_path_text = request_path.resolve().relative_to(planner.ROOT).as_posix()
            captured_launch_card_paths.append(kwargs.get("launch_card_path"))
            index = 0 if "fixture-mixtral" in request_path_text else int(request_path_text.split("fixture-")[1].split(".")[0])
            paths = fixture_paths(index)
            capture_tasks = []
            for artifact_id, capture_kind, artifact_path, receipt_path in (
                ("candidate_router_trace", "llama_cpp_router_trace_jsonl", paths["candidate_trace_path"], paths["candidate_receipt_path"]),
                ("managed_output_summary_fill", "managed_output_summary_json", paths["managed_output_path"], paths["managed_output_path"]),
                ("dense_output_summary_fill", "dense_output_summary_json", paths["dense_output_path"], paths["dense_output_path"]),
            ):
                capture_tasks.append(
                    {
                        "artifact_id": artifact_id,
                        "capture_kind": capture_kind,
                        "artifact_path": artifact_path,
                        "receipt_path": receipt_path,
                        "validator_command_count": 1,
                        "command_binding_ready": True,
                        "planned_command_binding": {
                            "command_class": "phase3_model_plane_launch_card_runtime_capture",
                            "command_ready": True,
                            "requires_runtime": True,
                            "requires_prompt_traffic": True,
                            "may_send_prompt_traffic": True,
                            "metadata_only": False,
                            "runtime_capture_request_path": request_path_text,
                            "model_plane_callable_id": f"callable-{index}-{artifact_id}",
                            "launch_command": ["model-plane-callable", f"callable-{index}-{artifact_id}"],
                            "missing_binding": None,
                        },
                    }
                )
            return {
                "schema_version": "moe-phase3-runtime-capture-command-contract-v1",
                "mode": "phase3_runtime_capture_command_contract",
                "valid": True,
                "errors": [],
                "request_path": request_path_text,
                "request_name": paths["name"],
                "command_contract_ready": True,
                "runtime_capture_command_ready": True,
                "planned_capture_count": 3,
                "runtime_command_option_count": 3,
                "missing_runtime_command_count": 0,
                "missing_runtime_command_artifact_ids": [],
                "launch_card_binding_summary": {
                    "valid": True,
                    "binding_ready": True,
                    "launch_card_path": str(kwargs.get("launch_card_path")),
                    "bound_task_count": 3,
                    "command_option_count": 3,
                    "missing_runtime_command_count": 0,
                    "blockers": [],
                },
                "capture_tasks": capture_tasks,
            }

        original_build_summary = planner.plan_phase3_runtime_capture_commands.build_summary
        original_root = planner.ROOT
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                planner.ROOT = Path(temp_dir)
                launch_card_dir = planner.ROOT / "filled-launch-cards"
                launch_card_dir.mkdir(parents=True)
                for index in range(6):
                    paths = fixture_paths(index)
                    request_path = planner.ROOT / paths["request_path"]
                    request_path.parent.mkdir(parents=True, exist_ok=True)
                    request_path.write_text("{}", encoding="utf-8")
                    (launch_card_dir / f"fixture-{index}.runtime-capture-launch-card.template.json").write_text(
                        "{}",
                        encoding="utf-8",
                    )
                planner.plan_phase3_runtime_capture_commands.build_summary = fake_build_summary
                summary = planner.build_packet_summary(
                    TRACE_FIXTURE,
                    INVENTORY_FIXTURE,
                    POLICIES_FIXTURE,
                    MANAGED_PLAN,
                    runtime_capture_launch_card_dir=launch_card_dir,
                )
        finally:
            planner.plan_phase3_runtime_capture_commands.build_summary = original_build_summary
            planner.ROOT = original_root

        directory_manifest = summary["runtime_capture_launch_card_directory_manifest"]
        self.assertTrue(directory_manifest["provided"])
        self.assertTrue(directory_manifest["directory_ready"])
        self.assertEqual(directory_manifest["expected_card_count"], 6)
        self.assertEqual(directory_manifest["matched_card_count"], 6)
        self.assertEqual(directory_manifest["missing_card_count"], 0)
        self.assertGreaterEqual(len(captured_launch_card_paths), 6)
        self.assertTrue(all(path is not None for path in captured_launch_card_paths))
        contract_summary = summary["recommended_runtime_capture_command_contract_summary"]
        self.assertTrue(contract_summary["launch_card_binding_ready"])
        self.assertEqual(contract_summary["launch_card_binding_command_option_count"], 3)
        execution_summary = summary["recommended_runtime_capture_execution_coverage_summary"]
        self.assertTrue(execution_summary["automated_capture_ready"])
        all_execution_summary = summary["all_request_runtime_capture_execution_coverage_summary"]
        self.assertTrue(all_execution_summary["manual_operator_capture_ready"])
        self.assertTrue(all_execution_summary["automated_capture_ready"])
        self.assertEqual(all_execution_summary["automated_capture_ready_count"], 6)
        self.assertEqual(all_execution_summary["capture_command_option_count"], 18)
        self.assertEqual(all_execution_summary["manual_capture_required_count"], 0)
        self.assertEqual(all_execution_summary["missing_capture_command_count"], 0)
        self.assertTrue(summary["remaining_gaps"]["runtime_capture_launch_card_dir_ready"])
        self.assertEqual(summary["remaining_gaps"]["runtime_capture_launch_card_dir_matched_card_count"], 6)
        self.assertTrue(summary["remaining_gaps"]["all_request_runtime_capture_execution_automated_ready"])
        self.assertEqual(summary["remaining_gaps"]["all_request_runtime_capture_execution_automated_request_count"], 6)
        self.assertEqual(summary["remaining_gaps"]["all_request_runtime_capture_execution_missing_capture_command_count"], 0)
        by_item = {item["id"]: item for item in summary["evidence_items"]}
        self.assertEqual(by_item["all_request_runtime_capture_execution_coverage"]["status"], "proven")

    def test_operator_queue_parity_detects_path_mismatch(self) -> None:
        _, _, _, runtime_summary, intake_summary = repo_gate_summaries_fixture()
        mutated_intake = json.loads(json.dumps(intake_summary))
        mutated_intake["requests"][0]["next_operator_step"]["path"] = (
            "memory-moe-mvp/phase3-real-evidence/fixture/wrong-router-events.jsonl"
        )

        summary = planner.operator_queue_parity_summary(runtime_summary, mutated_intake)

        self.assertFalse(summary["ready"])
        self.assertEqual(summary["matched_request_count"], 6)
        self.assertEqual(
            summary["metadata_mismatch_requests"],
            [
                {
                    "request_name": "Fixture Mixtral Phase 3 real evidence",
                    "mismatch_fields": ["path"],
                }
            ],
        )

    def test_post_approval_capture_fill_parity_detects_path_mismatch(self) -> None:
        _, _, _, runtime_summary, intake_summary = repo_gate_summaries_fixture()
        mutated_intake = json.loads(json.dumps(intake_summary))
        mutated_intake["recommended_post_approval_capture_fill_plan"]["capture_fill_steps"][1]["artifact_path"] = (
            "memory-moe-mvp/phase3-real-evidence/fixture/wrong-managed-output-summary.json"
        )

        summary = planner.post_approval_capture_fill_parity_summary(runtime_summary, mutated_intake)

        self.assertFalse(summary["ready"])
        self.assertEqual(summary["runtime_pending_artifact_count"], 3)
        self.assertEqual(summary["capture_result_fill_step_count"], 3)
        self.assertEqual(summary["matched_artifact_count"], 3)
        self.assertTrue(summary["request_path_match"])
        self.assertEqual(summary["missing_from_capture_result_artifact_ids"], [])
        self.assertEqual(summary["missing_from_runtime_preview_artifact_ids"], [])
        self.assertEqual(
            summary["metadata_mismatch_artifacts"],
            [
                {
                    "artifact_id": "managed_output_summary_fill",
                    "mismatch_fields": ["artifact_path"],
                }
            ],
        )

    def test_all_post_approval_capture_fill_parity_detects_path_mismatch(self) -> None:
        _, _, _, runtime_summary, intake_summary = repo_gate_summaries_fixture()
        mutated_intake = json.loads(json.dumps(intake_summary))
        mutated_intake["post_approval_capture_fill_plan_manifest"][0]["capture_fill_steps"][1]["artifact_path"] = (
            "memory-moe-mvp/phase3-real-evidence/fixture/wrong-managed-output-summary.json"
        )

        summary = planner.all_post_approval_capture_fill_parity_summary(runtime_summary, mutated_intake)

        self.assertFalse(summary["ready"])
        self.assertEqual(summary["runtime_pending_artifact_count"], 18)
        self.assertEqual(summary["capture_result_fill_step_count"], 18)
        self.assertEqual(summary["matched_request_count"], 6)
        self.assertEqual(summary["matched_artifact_count"], 18)
        self.assertEqual(summary["missing_from_capture_result_artifact_keys"], [])
        self.assertEqual(summary["missing_from_runtime_request_artifact_keys"], [])
        self.assertEqual(
            summary["metadata_mismatch_artifacts"],
            [
                {
                    "artifact_key": "memory-moe-mvp/phase3-real-evidence/fixture-mixtral.runtime-capture-request.json::managed_output_summary_fill",
                    "request_path": "memory-moe-mvp/phase3-real-evidence/fixture-mixtral.runtime-capture-request.json",
                    "artifact_id": "managed_output_summary_fill",
                    "mismatch_fields": ["artifact_path"],
                }
            ],
        )
    def test_all_request_receipt_validator_parity_detects_validator_mismatch(self) -> None:
        _, _, _, runtime_summary, intake_summary = repo_gate_summaries_fixture()
        mutated_intake = json.loads(json.dumps(intake_summary))
        mutated_intake["post_approval_capture_fill_plan_manifest"][0]["capture_fill_steps"][0]["validator_command_count"] = 99
        validator_manifest = planner.all_request_validator_command_manifest(runtime_summary)
        receipt_fill_manifest = planner.all_request_receipt_fill_manifest(mutated_intake)

        summary = planner.all_request_receipt_validator_parity_summary(validator_manifest, receipt_fill_manifest)

        self.assertFalse(summary["ready"])
        self.assertEqual(summary["runtime_artifact_count"], 18)
        self.assertEqual(summary["receipt_fill_entry_count"], 18)
        self.assertEqual(summary["matched_artifact_count"], 18)
        self.assertEqual(summary["missing_from_receipt_fill_artifact_keys"], [])
        self.assertEqual(summary["missing_from_runtime_validator_artifact_keys"], [])
        self.assertEqual(
            summary["metadata_mismatch_artifacts"],
            [
                {
                    "artifact_key": "memory-moe-mvp/phase3-real-evidence/fixture-mixtral.runtime-capture-request.json::candidate_router_trace",
                    "request_path": "memory-moe-mvp/phase3-real-evidence/fixture-mixtral.runtime-capture-request.json",
                    "artifact_id": "candidate_router_trace",
                    "mismatch_fields": ["validator_command_count"],
                }
            ],
        )

    def test_all_request_receipt_fill_command_manifest_detects_missing_receipt_path(self) -> None:
        _, _, _, runtime_summary, intake_summary = repo_gate_summaries_fixture()
        mutated_intake = json.loads(json.dumps(intake_summary))
        mutated_intake["post_approval_capture_fill_plan_manifest"][0]["capture_fill_steps"][0]["receipt_path"] = ""
        validator_manifest = planner.all_request_validator_command_manifest(runtime_summary)
        receipt_fill_manifest = planner.all_request_receipt_fill_manifest(mutated_intake)

        manifest = planner.all_request_receipt_fill_command_manifest(validator_manifest, receipt_fill_manifest)
        summary = planner.all_request_receipt_fill_command_summary(manifest, expected_request_count=6)

        self.assertFalse(summary["ready"])
        self.assertEqual(summary["request_count"], 6)
        self.assertEqual(summary["entry_count"], 18)
        self.assertEqual(summary["validator_command_count"], 30)
        self.assertEqual(summary["missing_command_entry_count"], 0)
        self.assertEqual(summary["missing_receipt_path_count"], 1)

    def test_all_request_receipt_fill_command_manifest_requires_source_prompt_and_prompt_traffic_ack(self) -> None:
        _, _, _, runtime_summary, intake_summary = repo_gate_summaries_fixture()
        mutated_intake = json.loads(json.dumps(intake_summary))
        first_plan = mutated_intake["post_approval_capture_fill_plan_manifest"][0]
        first_plan["prompt_set_path"] = ""
        first_plan["records_approval_keys"] = []
        first_plan["capture_fill_steps"][0]["source_request_path"] = "memory-moe-mvp/phase3-real-evidence/wrong.runtime-capture-request.json"
        validator_manifest = planner.all_request_validator_command_manifest(runtime_summary)
        receipt_fill_manifest = planner.all_request_receipt_fill_manifest(mutated_intake)

        manifest = planner.all_request_receipt_fill_command_manifest(validator_manifest, receipt_fill_manifest)
        summary = planner.all_request_receipt_fill_command_summary(manifest, expected_request_count=6)

        self.assertFalse(summary["ready"])
        self.assertEqual(summary["missing_source_request_path_count"], 1)
        self.assertEqual(summary["missing_prompt_set_path_count"], 3)
        self.assertEqual(summary["missing_prompt_traffic_ack_count"], 3)
        self.assertEqual(summary["missing_explicit_approval_count"], 0)

    def test_all_request_manual_capture_runbook_requires_source_prompt_and_prompt_traffic_ack(self) -> None:
        _, _, _, runtime_summary, intake_summary = repo_gate_summaries_fixture()
        capture_queue = planner.all_request_capture_queue_manifest(runtime_summary, intake_summary)
        validator_manifest = planner.all_request_validator_command_manifest(runtime_summary)
        receipt_fill_manifest = planner.all_request_receipt_fill_manifest(intake_summary)
        receipt_command_manifest = planner.all_request_receipt_fill_command_manifest(validator_manifest, receipt_fill_manifest)
        receipt_command_manifest[0]["source_request_path"] = "memory-moe-mvp/phase3-real-evidence/wrong.runtime-capture-request.json"
        receipt_command_manifest[0]["prompt_set_path"] = ""
        receipt_command_manifest[0]["may_send_prompt_traffic_after_approval"] = False
        execution_coverage = planner.all_request_runtime_capture_execution_coverage_manifest(capture_queue, {"binding_tasks": []})

        manifest = planner.all_request_manual_capture_runbook_manifest(capture_queue, receipt_command_manifest, execution_coverage)
        summary = planner.all_request_manual_capture_runbook_summary(manifest)

        self.assertFalse(summary["ready"])
        self.assertEqual(summary["missing_source_request_path_count"], 1)
        self.assertEqual(summary["missing_prompt_set_path_count"], 1)
        self.assertEqual(summary["missing_prompt_traffic_ack_count"], 1)
        self.assertEqual(summary["missing_explicit_approval_count"], 0)
        self.assertIn(
            "prompt_traffic_ack_missing:memory-moe-mvp/phase3-real-evidence/fixture-mixtral.runtime-capture-request.json::candidate_router_trace",
            manifest["missing_items"],
        )

    def test_all_request_post_capture_intake_runbook_requires_source_prompt_and_prompt_traffic_ack(self) -> None:
        _, _, _, runtime_summary, intake_summary = repo_gate_summaries_fixture()
        mutated_intake = json.loads(json.dumps(intake_summary))
        first_plan = mutated_intake["post_approval_capture_fill_plan_manifest"][0]
        first_step = first_plan["capture_fill_steps"][0]
        first_step["source_request_path"] = "memory-moe-mvp/phase3-real-evidence/wrong.runtime-capture-request.json"
        first_step["source_prompt_set_path"] = "memory-moe-mvp/phase3-real-evidence/wrong.prompt-set.json"
        capture_queue = planner.all_request_capture_queue_manifest(runtime_summary, mutated_intake)
        validator_manifest = planner.all_request_validator_command_manifest(runtime_summary)
        receipt_fill_manifest = planner.all_request_receipt_fill_manifest(mutated_intake)
        receipt_command_manifest = planner.all_request_receipt_fill_command_manifest(validator_manifest, receipt_fill_manifest)
        execution_coverage = planner.all_request_runtime_capture_execution_coverage_manifest(capture_queue, {"binding_tasks": []})
        manual_runbook = planner.all_request_manual_capture_runbook_manifest(capture_queue, receipt_command_manifest, execution_coverage)
        first_task = manual_runbook["requests"][0]["manual_tasks"][0]
        first_task["requires_explicit_user_approval"] = False
        first_task["may_send_prompt_traffic_after_approval"] = False

        manifest = planner.all_request_post_capture_intake_runbook_manifest(manual_runbook, mutated_intake)
        summary = planner.all_request_post_capture_intake_runbook_summary(manifest)

        self.assertFalse(summary["ready"])
        self.assertEqual(summary["missing_source_request_path_count"], 1)
        self.assertEqual(summary["missing_prompt_set_path_count"], 1)
        self.assertEqual(summary["missing_explicit_approval_count"], 1)
        self.assertEqual(summary["missing_prompt_traffic_ack_count"], 1)
        self.assertIn(
            "post_capture_prompt_traffic_ack_missing:memory-moe-mvp/phase3-real-evidence/fixture-mixtral.runtime-capture-request.json::candidate_router_trace",
            manifest["missing_items"],
        )

    def test_policy_candidate_trace_handoff_uses_runtime_capture_paths(self) -> None:
        recommendation = {
            "request_path": MIXTRAL_REQUEST_PATH,
            "bundle_path": MIXTRAL_BUNDLE_PATH,
            "prompt_set_path": MIXTRAL_PROMPT_SET_PATH,
            "missing_approval_keys": [
                "runtime_prompt_traffic_approved",
                "router_trace_capture_approved",
            ],
            "next_artifact_id": "candidate_router_trace",
            "next_artifact_path": MIXTRAL_CANDIDATE_TRACE_PATH,
            "approval_rebuild_command": {
                "command": [
                    "uv",
                    "run",
                    "--managed-python",
                    "--python",
                    "3.13",
                    "scripts/build_phase3_runtime_capture_request.py",
                    MIXTRAL_BUNDLE_PATH,
                    "--prompt-set-path",
                    MIXTRAL_PROMPT_SET_PATH,
                    "--candidate-trace-path",
                    MIXTRAL_CANDIDATE_TRACE_PATH,
                    "--candidate-trace-receipt-path",
                    MIXTRAL_CANDIDATE_RECEIPT_PATH,
                    "--output",
                    MIXTRAL_REQUEST_PATH,
                    "--json",
                ],
            },
        }

        plan = planner.build_recommended_policy_candidate_trace_plan(recommendation)

        assert plan is not None
        self.assertTrue(plan["valid"], plan["errors"])
        self.assertTrue(planner.policy_candidate_trace_handoff_ready(plan))
        self.assertEqual(plan["bundle_path"], MIXTRAL_BUNDLE_PATH)
        self.assertEqual(plan["candidate_prompt_set_path"], MIXTRAL_PROMPT_SET_PATH)
        self.assertEqual(plan["candidate_trace_path"], MIXTRAL_CANDIDATE_TRACE_PATH)
        self.assertEqual(plan["candidate_trace_receipt_path"], MIXTRAL_CANDIDATE_RECEIPT_PATH)
        self.assertTrue(plan["prompt_set_ready"])
        self.assertTrue(plan["candidate_trace_exists"])
        self.assertTrue(plan["capture_receipt_ready"])
        self.assertFalse(plan["policy_candidate_ready_after_plan"])
        self.assertTrue(plan["runtime_requires_explicit_approval"])
        self.assertEqual(plan["artifact_request_statuses"]["candidate_trace_artifact"], "present_but_not_candidate_ready")
        self.assertIn("candidate_trace_contract_validator", plan["command_classes"])
        self.assertIn("candidate_policy_replay", plan["command_classes"])

    def test_dense_fallback_capture_handoff_uses_runtime_capture_paths(self) -> None:
        recommendation = {
            "request_path": MIXTRAL_REQUEST_PATH,
            "bundle_path": MIXTRAL_BUNDLE_PATH,
            "prompt_set_path": MIXTRAL_PROMPT_SET_PATH,
            "missing_approval_keys": [
                "runtime_prompt_traffic_approved",
                "managed_output_capture_approved",
                "dense_output_capture_approved",
            ],
            "approval_rebuild_command": {
                "command": [
                    "uv",
                    "run",
                    "--managed-python",
                    "--python",
                    "3.13",
                    "scripts/build_phase3_runtime_capture_request.py",
                    MIXTRAL_BUNDLE_PATH,
                    "--prompt-set-path",
                    MIXTRAL_PROMPT_SET_PATH,
                    "--managed-output-path",
                    MIXTRAL_MANAGED_OUTPUT_PATH,
                    "--dense-output-path",
                    MIXTRAL_DENSE_OUTPUT_PATH,
                    "--output",
                    MIXTRAL_REQUEST_PATH,
                    "--json",
                ],
            },
        }

        plan = planner.build_recommended_dense_fallback_capture_plan(recommendation)

        assert plan is not None
        self.assertTrue(plan["valid"], plan["errors"])
        self.assertTrue(planner.dense_fallback_capture_handoff_ready(plan))
        self.assertEqual(plan["bundle_path"], MIXTRAL_BUNDLE_PATH)
        self.assertEqual(plan["prompt_set_path"], MIXTRAL_PROMPT_SET_PATH)
        self.assertEqual(plan["managed_output_path"], MIXTRAL_MANAGED_OUTPUT_PATH)
        self.assertEqual(plan["dense_output_path"], MIXTRAL_DENSE_OUTPUT_PATH)
        self.assertEqual(plan["fallback_artifact_path"], MIXTRAL_FALLBACK_COMPARISON_PATH)
        self.assertTrue(plan["prompt_set_ready"])
        self.assertTrue(plan["managed_output_exists"])
        self.assertTrue(plan["dense_output_exists"])
        self.assertFalse(plan["managed_output_ready"])
        self.assertFalse(plan["dense_output_ready"])
        self.assertFalse(plan["metadata_ready_to_build_comparison"])
        self.assertFalse(plan["comparison_ready"])
        self.assertTrue(plan["runtime_requires_explicit_approval"])
        self.assertEqual(plan["artifact_request_statuses"]["managed_output_summary"], "approval_required")
        self.assertEqual(plan["artifact_request_statuses"]["dense_output_summary"], "approval_required")
        self.assertEqual(plan["artifact_request_statuses"]["dense_fallback_comparison_artifact"], "blocked_by_saved_outputs")
        self.assertIn("dense_fallback_comparison_builder", plan["command_classes"])
        self.assertIn("dense_fallback_comparison_validator", plan["command_classes"])

    def test_live_capability_proof_handoff_uses_runtime_capture_path(self) -> None:
        recommendation = {
            "request_path": MIXTRAL_REQUEST_PATH,
            "bundle_path": MIXTRAL_BUNDLE_PATH,
            "model_id": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
            "approval_rebuild_command": {
                "command": [
                    "uv",
                    "run",
                    "--managed-python",
                    "--python",
                    "3.13",
                    "scripts/build_phase3_runtime_capture_request.py",
                    MIXTRAL_BUNDLE_PATH,
                    "--live-proof-template-path",
                    MIXTRAL_LIVE_PROOF_TEMPLATE_PATH,
                    "--output",
                    MIXTRAL_REQUEST_PATH,
                    "--json",
                ],
            },
        }

        plan = planner.build_recommended_live_capability_proof_handoff(recommendation)

        assert plan is not None
        self.assertTrue(plan["valid"], plan["errors"])
        self.assertTrue(planner.live_capability_proof_handoff_ready(plan))
        self.assertEqual(plan["proof_artifact_path"], MIXTRAL_LIVE_PROOF_TEMPLATE_PATH)
        self.assertEqual(plan["source_bundle_path"], MIXTRAL_BUNDLE_PATH)
        self.assertTrue(plan["proof_available"])
        self.assertFalse(plan["proof_ready"])
        self.assertTrue(plan["context_binding_ready"])
        self.assertFalse(plan["residency_observation_ready"])
        self.assertFalse(plan["residency_control_ready"])
        self.assertFalse(plan["cleanup_restore_ready"])
        self.assertFalse(plan["artifact_export_ready"])
        self.assertGreater(plan["blocker_count"], 0)
        self.assertFalse(plan["ready_for_live_spike"])

    def test_all_request_downstream_handoff_manifest_goes_ready_for_real_mixtral_scaffold(self) -> None:
        request = json.loads((ROOT / MIXTRAL_REQUEST_PATH).read_text(encoding="utf-8"))
        candidate = next(item for item in request["requested_artifacts"] if item["id"] == "candidate_router_trace")
        approvals = request.get("approvals", {})
        runtime_summary = {
            "request_count": 1,
            "requests": [
                {
                    "name": request["name"],
                    "path": MIXTRAL_REQUEST_PATH,
                    "bundle_path": request["bundle_path"],
                    "operator_handoff": {
                        "requested_artifacts": request["requested_artifacts"],
                        "future_artifacts": request["future_artifacts"],
                    },
                }
            ],
            "approval_queue": [
                {
                    "request_name": request["name"],
                    "path": MIXTRAL_REQUEST_PATH,
                    "status": "approval_required",
                    "bundle_path": request["bundle_path"],
                    "model_id": request["model_id"],
                    "prompt_set_path": request["prompt_set"]["path"],
                    "prompt_count": request["prompt_set"]["prompt_count"],
                    "missing_approval_keys": [key for key, value in approvals.items() if value is False],
                    "pending_artifact_ids": [item["id"] for item in request["requested_artifacts"]],
                    "next_artifact_id": "candidate_router_trace",
                    "next_artifact_path": candidate["path"],
                    "queue_rank": 1,
                    "selection_rationale": "mixtral_first_known_sparse_baseline",
                }
            ],
            "approval_rebuild_command_manifest": [],
        }

        manifest = planner.all_request_downstream_handoff_manifest(runtime_summary)
        summary = planner.all_request_downstream_handoff_summary(manifest, expected_request_count=1)

        self.assertTrue(summary["ready"])
        self.assertEqual(summary["policy_candidate_trace_handoff_ready_count"], 1)
        self.assertEqual(summary["dense_fallback_capture_handoff_ready_count"], 1)
        self.assertEqual(summary["live_capability_proof_handoff_ready_count"], 1)
        self.assertEqual(summary["all_downstream_handoff_ready_count"], 1)
        self.assertEqual(manifest[0]["request_name"], "PC Mixtral Phase 3 real evidence")
        self.assertEqual(manifest[0]["policy_candidate_trace"]["candidate_trace_path"], MIXTRAL_CANDIDATE_TRACE_PATH)
        self.assertTrue(manifest[0]["policy_candidate_trace"]["candidate_trace_receipt_template_exists"])
        self.assertTrue(manifest[0]["dense_fallback_capture"]["managed_output_template_exists"])
        self.assertEqual(manifest[0]["dense_fallback_capture"]["fallback_artifact_path"], MIXTRAL_FALLBACK_COMPARISON_PATH)
        self.assertTrue(manifest[0]["live_capability_proof"]["proof_template_exists"])

    def test_valid_fallback_artifact_updates_evidence_item_without_completing_phase3(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fallback_path = Path(temp_dir) / "fallback.json"
            fallback_path.write_text(json.dumps(fallback_artifact()), encoding="utf-8")
            summary = planner.build_packet_summary(
                TRACE_FIXTURE,
                INVENTORY_FIXTURE,
                POLICIES_FIXTURE,
                MANAGED_PLAN,
                fallback_artifact_path=fallback_path,
            )

        self.assertTrue(summary["valid"], summary["errors"])
        by_item = {item["id"]: item for item in summary["evidence_items"]}
        self.assertEqual(by_item["dense_fallback_comparison"]["status"], "proven")
        self.assertEqual(by_item["real_model_trace_inventory_pairing"]["status"], "proven")
        self.assertEqual(by_item["real_model_trace_inventory_pairing"]["details"]["decision_pairing_basis"], "repo_real_evidence_matrix")
        self.assertFalse(summary["phase3_complete"])
        self.assertFalse(summary["remaining_gaps"]["ready_for_live_spike"])

    def test_valid_live_proof_artifact_updates_future_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            live_path = Path(temp_dir) / "live-proof.json"
            export_path = Path(temp_dir) / "live-proof-export.json"
            export_path.write_text("{}\n", encoding="utf-8")
            live_path.write_text(json.dumps(live_proof_artifact(artifact_paths=[str(export_path)])), encoding="utf-8")
            summary = planner.build_packet_summary(
                TRACE_FIXTURE,
                INVENTORY_FIXTURE,
                POLICIES_FIXTURE,
                MANAGED_PLAN,
                live_proof_artifact_path=live_path,
            )

        self.assertTrue(summary["valid"], summary["errors"])
        by_item = {item["id"]: item for item in summary["evidence_items"]}
        self.assertEqual(by_item["live_capability_proof"]["status"], "proven")
        self.assertTrue(summary["remaining_gaps"]["live_capability_proof_ready"])
        reason_ids = {item["id"] for item in summary["remaining_gaps"]["no_go_reasons"]}
        self.assertNotIn("live_capability_proof_not_ready", reason_ids)
        self.assertNotIn("live_actuator_missing", reason_ids)
        self.assertFalse(summary["phase3_complete"])
        self.assertFalse(summary["remaining_gaps"]["ready_for_live_spike"])

    def test_invalid_input_makes_packet_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bad_policies = Path(temp_dir) / "policies.json"
            bad_policies.write_text("{", encoding="utf-8")
            status, summary, error = planner.plan_paths(
                TRACE_FIXTURE,
                INVENTORY_FIXTURE,
                bad_policies,
                MANAGED_PLAN,
            )

        self.assertEqual(status, 2)
        self.assertIsNone(summary)
        assert error is not None
        self.assertIn("Could not build Phase 3 evidence packet", error)

    def test_packet_indexes_no_go_reason_ids(self) -> None:
        summary = planner.build_packet_summary(
            TRACE_FIXTURE,
            INVENTORY_FIXTURE,
            POLICIES_FIXTURE,
            MANAGED_PLAN,
        )

        reason_ids = {item["id"] for item in summary["remaining_gaps"]["no_go_reasons"]}
        self.assertNotIn("real_model_trace_inventory_pairing_not_ready", reason_ids)
        self.assertNotIn("runtime_capture_request_audit_not_ready", reason_ids)
        self.assertIn("capture_result_intake_not_ready", reason_ids)
        self.assertIn("dense_fallback_comparison_not_ready", reason_ids)
        self.assertIn("live_capability_proof_not_ready", reason_ids)
        self.assertIn("live_actuator_missing", reason_ids)

    def test_markdown_report_contains_status_table_and_gaps(self) -> None:
        summary = planner.build_packet_summary(
            TRACE_FIXTURE,
            INVENTORY_FIXTURE,
            POLICIES_FIXTURE,
            MANAGED_PLAN,
        )

        report = planner.format_markdown_report(summary)

        self.assertIn("# MoE Run Anyway Phase 3 Evidence Packet", report)
        self.assertIn("| Evidence item | Status | Source | Summary |", report)
        self.assertIn("`real_model_trace_inventory_pairing` | `proven`", report)
        self.assertIn("`dense_fallback_comparison` | `blocked`", report)
        self.assertIn("`live_capability_proof` | `future_phase_4`", report)
        self.assertIn("Live capability proof handoff ready: False", report)
        self.assertIn("Live capability proof ready: False", report)
        self.assertIn("Runtime actuator design ready: True", report)
        self.assertIn("Runtime actuator live ready: False", report)
        self.assertIn("Runtime actuator control blockers: 3", report)
        self.assertIn("`repo_real_evidence_matrix` | `proven`", report)
        self.assertIn("`phase3_launch_card_library` | `proven`", report)
        self.assertIn("`capture_result_intake` | `blocked`", report)
        self.assertIn("`approval_manifest_parity` | `proven`", report)
        self.assertIn("`receipt_requirement_parity` | `proven`", report)
        self.assertIn("`operator_queue_parity` | `proven`", report)
        self.assertIn("`post_approval_capture_fill_parity` | `proven`", report)
        self.assertIn("`all_post_approval_capture_fill_parity` | `proven`", report)
        self.assertIn("`all_request_capture_queue_manifest` | `proven`", report)
        self.assertIn("`all_request_validator_command_manifest` | `proven`", report)
        self.assertIn("`all_request_downstream_handoff_manifest` | `blocked`", report)
        self.assertIn("`all_request_receipt_fill_manifest` | `blocked`", report)
        self.assertIn("`all_request_receipt_validator_parity` | `proven`", report)
        self.assertIn("`all_request_receipt_fill_command_manifest` | `proven`", report)
        self.assertIn("`recommended_runtime_capture_preflight` | `proven`", report)
        self.assertNotIn("`recommended_runtime_capture_command_contract` | `proven`", report)
        self.assertIn("`recommended_runtime_capture_execution_coverage` | `blocked`", report)
        self.assertIn("`phase3_blocker_closure_manifest` | `proven`", report)
        self.assertIn("`phase3_blocker_evidence_ledger` | `proven`", report)
        self.assertIn("`phase3_blocker_resolution_queue` | `proven`", report)
        self.assertIn("## Phase 3 Blocker Evidence Ledger", report)
        self.assertIn("Rows: `25` / `25`", report)
        self.assertIn("Dense output rows: `0`", report)
        self.assertIn("Live proof rows: `0`", report)
        self.assertIn("Runtime actuator rows: `2`", report)
        self.assertIn("## Phase 3 Blocker Resolution Queue", report)
        self.assertIn("Work packages: `5`", report)
        self.assertIn("Rows: `25` / `25`", report)
        self.assertIn("Runtime-capture packages: `3`", report)
        self.assertIn("Future-adapter packages: `1`", report)
        self.assertIn("Validator commands: `41`", report)
        self.assertIn("Completion gates: `21`", report)
        self.assertIn("Dependency edges: `4`", report)
        self.assertIn("| 3 | dense_fallback_output_capture | capture_result_receipt_intake | fallback_quality_bounds | 2 | 2 | 0 | 2 | 4 |", report)
        self.assertIn("## Phase 3 Blocker Closure", report)
        self.assertIn("Mapping ready: `True`", report)
        self.assertIn("Evidence complete: `False`", report)
        self.assertIn("Ready scope: `mapping_ready_only`", report)
        self.assertIn("Missing evidence: `25`", report)
        self.assertIn("## Recommended Runtime Capture Preflight", report)
        self.assertIn("Ready: `True`", report)
        self.assertIn("Runtime closure reasons: `4`", report)
        self.assertIn("| candidate_router_trace | memory-moe-mvp/phase3-real-evidence/fixture/candidate-router-events.jsonl | memory-moe-mvp/phase3-real-evidence/fixture/candidate-router-events.capture-receipt.json | 3 | False |", report)
        self.assertIn("Recommended runtime-capture preflight ready: True", report)
        self.assertIn("Recommended runtime-capture preflight validator commands: 5", report)
        self.assertNotIn("## Recommended Runtime Capture Command Contract", report)
        self.assertIn("Recommended runtime-capture command contract ready: False", report)
        self.assertIn("Recommended runtime-capture command contract runtime ready: False", report)
        self.assertIn("Recommended runtime-capture command contract missing runtime commands: 0", report)
        self.assertIn("## Recommended Runtime Capture Execution Coverage", report)
        self.assertIn("Manual operator capture ready: `True`", report)
        self.assertIn("Automated capture ready: `False`", report)
        self.assertIn("Missing capture command artifacts: `candidate_router_trace, dense_output_summary_fill, managed_output_summary_fill`", report)
        self.assertIn("| managed_output_summary_fill | manual_runtime_capture_required | 0 | 0 | 0 | ready_for_operator_capture | 1 | False |", report)
        self.assertIn("Recommended runtime-capture execution automated ready: False", report)
        self.assertIn("Recommended runtime-capture execution runtime command options: 0", report)
        self.assertIn("Recommended runtime-capture execution operator command options: 1", report)
        self.assertIn("Recommended runtime-capture execution metadata command options: 1", report)
        self.assertIn("Recommended runtime-capture execution missing capture commands: 3", report)
        self.assertIn("## Phase 3 Launch-Card Library", report)
        self.assertIn("Library ready: `True`", report)
        self.assertIn("Execution ready: `False`", report)
        self.assertIn("Missing runtime commands: `18`", report)
        self.assertIn("Binding handoff ready: `True`", report)
        self.assertIn("Binding handoff tasks: `18`", report)
        self.assertIn("Unbound tasks: `18`", report)
        self.assertIn("Saved handoff artifacts ready: `True`", report)
        self.assertIn("Saved handoff artifacts missing: `0`", report)
        self.assertIn("Saved handoff artifacts drifted: `0`", report)
        self.assertIn("Recommended template path: `memory-moe-mvp/phase3-real-evidence/fixture-0.runtime-capture-launch-card.template.json`", report)
        self.assertIn("Recommended template ready: `True`", report)
        self.assertIn("Recommended binding handoff ready: `True`", report)
        self.assertIn("Recommended template unbound tasks: `3`", report)
        self.assertIn("| Fixture Mixtral Phase 3 real evidence | True | False | False | False | 3 |", report)
        self.assertIn("Capture-result approval rebuild command manifest entries: 6", report)
        self.assertIn("Approval manifest parity ready: True", report)
        self.assertIn("Approval manifest parity matched requests: 6", report)
        self.assertIn("Receipt requirement parity ready: True", report)
        self.assertIn("Receipt requirement parity matched requirements: 18", report)
        self.assertIn("Operator queue parity ready: True", report)
        self.assertIn("Operator queue parity matched requests: 6", report)
        self.assertIn("Post-approval capture-fill parity ready: True", report)
        self.assertIn("Post-approval capture-fill parity matched artifacts: 3", report)
        self.assertIn("Post-approval capture-fill parity runtime pending artifacts: 3", report)
        self.assertIn("Post-approval capture-fill parity capture-result fill steps: 3", report)
        self.assertIn("Post-approval capture-fill parity request path match: True", report)
        self.assertIn("All-request post-approval capture-fill parity ready: True", report)
        self.assertIn("All-request post-approval capture-fill parity matched requests: 6", report)
        self.assertIn("All-request post-approval capture-fill parity matched artifacts: 18", report)
        self.assertIn("All-request post-approval capture-fill parity runtime pending artifacts: 18", report)
        self.assertIn("All-request post-approval capture-fill parity capture-result fill steps: 18", report)
        self.assertIn("## Phase 3 Blocker Closure", report)
        self.assertIn("Runtime-capture required: `4`", report)
        self.assertIn("Future-adapter required: `3`", report)
        self.assertIn("## Runtime Actuator Spike Handoff", report)
        self.assertIn("Handoff ready: `True`", report)
        self.assertIn("Live spike ready: `False`", report)
        self.assertIn("Proof requirements: `8`", report)
        self.assertIn("Proof artifacts: `20`", report)
        self.assertIn("Dependency edges: `14`", report)
        self.assertIn("Blocking capabilities: `7`", report)
        self.assertIn("| no_replay_policy_candidate | policy_candidate_trace_capture | blocked | runtime_capture_required |", report)
        self.assertIn("| live_actuator_capabilities_unavailable | runtime_actuator_spike | future_phase_4 | future_adapter_required |", report)
        self.assertIn("## All-Request Capture Queue", report)
        self.assertIn("Pending artifact fills: `18`", report)
        self.assertIn("| 1 | Fixture Mixtral Phase 3 real evidence | approval_required | candidate_router_trace | 3 | 5 | memory-moe-mvp/phase3-real-evidence/fixture-mixtral.runtime-capture-request.json |", report)
        self.assertIn("All-request capture queue ready: True", report)
        self.assertIn("All-request capture queue requests: 6", report)
        self.assertIn("All-request capture queue pending artifacts: 18", report)
        self.assertIn("All-request capture queue validator commands: 30", report)
        self.assertIn("## All-Request Validator Commands", report)
        self.assertIn("Runtime validator commands: `30`", report)
        self.assertIn("Future adapter validator commands: `6`", report)
        self.assertIn("Total validator commands: `36`", report)
        self.assertIn("| 1 | Fixture Mixtral Phase 3 real evidence | candidate_router_trace | runtime_capture | approval_required | 3 | memory-moe-mvp/phase3-real-evidence/fixture/candidate-router-events.jsonl |", report)
        self.assertIn("## Promotion Checklist", report)
        self.assertIn("| Gate | Status | Summary | Next action |", report)
        self.assertIn("`approved_runtime_capture` | `approval_required`", report)
        self.assertIn("## Recommended Runtime Capture", report)
        self.assertIn("Request: `Fixture Mixtral Phase 3 real evidence`", report)
        self.assertIn("| record_runtime_approvals | approval | approval_required", report)
        self.assertIn("| capture_candidate_router_trace | runtime_capture | approval_required", report)
        self.assertIn("Validator commands: `6`", report)
        self.assertIn("Runtime-capture approval rebuild command manifest entries: 6", report)
        self.assertIn("### Runtime Capture Post-Approval Preview", report)
        self.assertIn("Status after approval: `ready_for_operator_capture`", report)
        self.assertIn("Ready for operator capture after approval: `True`", report)
        self.assertIn("Capture complete after approval: `False`", report)
        self.assertIn("Pending artifacts after approval: `candidate_router_trace, managed_output_summary_fill, dense_output_summary_fill`", report)
        self.assertIn("Runtime-capture post-approval preview status: ready_for_operator_capture", report)
        self.assertIn("Runtime-capture post-approval mutates request: False", report)
        self.assertIn("### Recommended Validator Commands", report)
        self.assertIn("scripts/validate_llama_cpp_router_trace.py", report)
        self.assertIn("scripts/plan_phase3_live_capability_proof.py", report)
        self.assertIn("Handoff scaffolds ready: True", report)
        self.assertIn("Runtime-capture complete: 0", report)
        self.assertIn("Capture-result drift-free saved requests: 6", report)
        self.assertIn("Capture-result drifted saved requests: 0", report)
        self.assertIn("Capture-result ready-for-operator flags: 0", report)
        self.assertIn("Capture-result approved but capture incomplete: 0", report)
        self.assertIn("Capture-result runtime approvals missing: 6", report)
        self.assertIn("Capture-result approval rebuild commands available: 6", report)
        self.assertIn("Capture-result approved runtime-capture pending: 0", report)
        self.assertIn("Capture-result capture receipts ready: 0 / 18", report)
        self.assertIn("Capture-result candidate trace receipts ready: 0", report)
        self.assertIn("Capture-result managed output receipts ready: 0", report)
        self.assertIn("Capture-result dense output receipts ready: 0", report)
        self.assertIn("Capture-result live proof receipts ready: 0", report)
        self.assertIn("Capture-result all capture receipts ready: False", report)
        self.assertIn("Capture-result all output receipt bindings ready: False", report)
        self.assertIn("Capture-result receipt-fill manifest ready: 0 / 18", report)
        self.assertIn("Capture-result receipt-fill missing entries: 18", report)
        self.assertIn("Capture-result receipt-fill approvals missing: 18", report)
        self.assertIn("Capture-result receipt-fill candidate router trace ready: 0 / 6", report)
        self.assertIn("Capture-result receipt-fill candidate router trace missing: 6", report)
        self.assertIn("Capture-result receipt-fill managed outputs ready: 0 / 6", report)
        self.assertIn("Capture-result receipt-fill managed outputs missing: 6", report)
        self.assertIn("Capture-result receipt-fill dense outputs ready: 0 / 6", report)
        self.assertIn("Capture-result receipt-fill dense outputs missing: 6", report)
        self.assertIn("Capture-result approval transition previews: 6", report)
        self.assertIn("Capture-result approval transitions ready for operator: 6", report)
        self.assertIn("Capture-result approval transitions ready to update bundle: 0", report)
        self.assertIn("Capture-result post-approval capture-fill plans: 6", report)
        self.assertIn("Capture-result post-approval capture-fill ready: 0 / 18", report)
        self.assertIn("Capture-result post-approval capture-fill missing: 18", report)
        self.assertIn("Capture-result post-approval capture-fill validator commands: 30", report)
        self.assertIn("All-request validator command manifest ready: True", report)
        self.assertIn("All-request validator command manifest artifact entries: 24", report)
        self.assertIn("All-request validator command manifest total commands: 36", report)
        self.assertIn("## All-Request Downstream Handoffs", report)
        self.assertIn("All downstream handoffs ready: `0`", report)
        self.assertIn("All-request downstream handoff manifest ready: False", report)
        self.assertIn("## All-Request Receipt Fills", report)
        self.assertIn("Receipt-fill entries: `18`", report)
        self.assertIn("Missing receipts: `18`", report)
        self.assertIn("Approvals recorded after approval: `18`", report)
        self.assertIn("| 1 | Fixture Mixtral Phase 3 real evidence | candidate_router_trace | False | blocked_after_approval | 3 | memory-moe-mvp/phase3-real-evidence/fixture/candidate-router-events.capture-receipt.json |", report)
        self.assertIn("All-request receipt-fill manifest ready: False", report)
        self.assertIn("All-request receipt-fill entries ready: 0 / 18", report)
        self.assertIn("All-request receipt-fill blocked after approval: 18", report)
        self.assertIn("## All-Request Receipt/Validator Parity", report)
        self.assertIn("Matched artifacts: `18`", report)
        self.assertIn("Metadata mismatches: `0`", report)
        self.assertIn("All-request receipt/validator parity ready: True", report)
        self.assertIn("All-request receipt/validator parity matched artifacts: 18 / 18", report)
        self.assertIn("All-request receipt/validator parity missing from receipt fills: 0", report)
        self.assertIn("## All-Request Receipt Fill Commands", report)
        self.assertIn("Missing command entries: `0`", report)
        self.assertIn("Missing receipt paths: `0`", report)
        self.assertIn("| 1 | Fixture Mixtral Phase 3 real evidence | candidate_router_trace | 3 | False | memory-moe-mvp/phase3-real-evidence/fixture/candidate-router-events.capture-receipt.json |", report)
        self.assertIn("All-request receipt-fill command manifest ready: True", report)
        self.assertIn("All-request receipt-fill command entries: 18", report)
        self.assertIn("All-request receipt-fill command missing command entries: 0", report)
        self.assertIn("## All-Request Manual Capture Runbook", report)
        self.assertIn("Manual tasks: `18`", report)
        self.assertIn("Validator commands: `30`", report)
        self.assertIn("All-request manual capture runbook ready: True", report)
        self.assertIn("All-request manual capture runbook manual tasks: 18", report)
        self.assertIn("All-request manual capture runbook missing items: 0", report)
        self.assertIn("## All-Request Post-Capture Intake Runbook", report)
        self.assertIn("Artifact gates: `18`", report)
        self.assertIn("Missing after current intake: `18`", report)
        self.assertIn("All-request post-capture intake runbook ready: True", report)
        self.assertIn("All-request post-capture intake runbook artifact gates: 18", report)
        self.assertIn("All-request post-capture intake runbook missing after current intake: 18", report)
        self.assertIn("Phase 3 blocker closure manifest ready: True", report)
        self.assertIn("Phase 3 blocker closure mapping ready: True", report)
        self.assertIn("Phase 3 blocker closure evidence complete: False", report)
        self.assertIn("Phase 3 blocker closure ready scope: mapping_ready_only", report)
        self.assertIn("Phase 3 blocker closure mapped reasons: 7", report)
        self.assertIn("Phase 3 blocker closure validator commands: 41", report)
        self.assertIn("Phase 3 blocker closure missing evidence: 25", report)
        self.assertIn("Phase 3 blocker evidence ledger ready: True", report)
        self.assertIn("Phase 3 blocker evidence ledger rows: 25 / 25", report)
        self.assertIn("Phase 3 blocker evidence ledger validator commands: 41", report)
        self.assertIn("Phase 3 blocker resolution queue ready: True", report)
        self.assertIn("Phase 3 blocker resolution queue work packages: 5", report)
        self.assertIn("Phase 3 blocker resolution queue rows: 25 / 25", report)
        self.assertIn("Phase 3 blocker resolution queue validator commands: 41", report)
        self.assertIn("Phase 3 blocker resolution queue completion gates: 21", report)
        self.assertIn("Phase 3 blocker resolution queue dependency edges: 4", report)
        self.assertIn("Capture-result post-approval capture-fill ready to update bundle: 0", report)
        self.assertIn("Capture-result missing receipt-gate requests: 6", report)
        self.assertIn("Capture-result output receipt bindings ready: 0", report)
        self.assertIn("Capture-result bundle updates ready: 0", report)
        self.assertIn("No-go reasons: no_replay_policy_candidate", report)
        self.assertIn("packet does not claim live expert paging", report)

    def test_human_summary_includes_all_request_capture_queue(self) -> None:
        summary = planner.build_packet_summary(
            TRACE_FIXTURE,
            INVENTORY_FIXTURE,
            POLICIES_FIXTURE,
            MANAGED_PLAN,
        )
        output = io.StringIO()

        with redirect_stdout(output):
            planner.print_human_summary(summary)

        report = output.getvalue()
        self.assertIn("All-request capture queue: ready=True requests=6 pending_artifacts=18 validators=30", report)
        self.assertIn("All-request validator commands: ready=True requests=6 runtime_artifacts=18 future_artifacts=6 commands=36", report)
        self.assertIn("All-request downstream handoffs: ready=False requests=6 policy=0 dense=0 live=0 all=0", report)
        self.assertIn("All-request receipt fills: ready=False requests=6 entries=18 ready_receipts=0 missing=18 validators=30", report)
        self.assertIn("All-request receipt/validator parity: ready=True runtime_artifacts=18 receipt_entries=18 matched=18 mismatches=0 missing_receipts=0 missing_runtime=0", report)
        self.assertIn("Phase 3 blocker closure: ready=True mapping_ready=True evidence_complete=False scope=mapping_ready_only missing_evidence=25 reasons=7 mapped=7 unresolved=7 runtime_capture=4 future_adapter=3 commands=41", report)
        self.assertIn("Phase 3 blocker evidence ledger: ready=True rows=25/25 reasons=7 runtime_capture=22 future_adapter=3 receipt_bound=18 dense_outputs=0 live_proof=0 runtime_actuator=2 commands=41 missing_items=0", report)
        self.assertIn("Phase 3 blocker resolution queue: ready=True packages=5 rows=25/25 runtime_packages=3 future_packages=1 commands=41 gates=21 dependencies=4 missing_items=0", report)
        self.assertIn("All-request receipt fill commands: ready=True requests=6 entries=18 commands=30 missing_commands=0 missing_receipt_paths=0", report)
        self.assertIn("Recommended runtime capture preflight: ready=True request=Fixture Mixtral Phase 3 real evidence pending=3 receipts=3 commands=5 runtime_closures=4 missing=0", report)
        self.assertIn("Recommended runtime capture command contract: ready=False runtime_ready=False planned=0 runtime_commands=0 missing_runtime_commands=0", report)
        self.assertIn("Recommended runtime capture execution: manual_ready=True automated_ready=False pending=3 runtime_command_options=0 operator_command_options=1 metadata_command_options=1 with_runtime_commands=0 manual_required=3 missing_commands=3", report)
        self.assertIn("Phase 3 launch-card library: library_ready=True execution_ready=False cards=6 template_ready=6 model_plane_ready=0 runtime_ready=0 tasks=18 missing_runtime_commands=18 handoff_ready=True handoff_tasks=18 unbound_tasks=18 saved_handoff_ready=True saved_handoff_missing=0 saved_handoff_drifted=0", report)
        self.assertIn("all_request_capture_queue_manifest: proven", report)
        self.assertIn("all_request_validator_command_manifest: proven", report)
        self.assertIn("all_request_receipt_fill_manifest: blocked", report)
        self.assertIn("all_request_receipt_validator_parity: proven", report)
        self.assertIn("all_request_receipt_fill_command_manifest: proven", report)
        self.assertIn("recommended_runtime_capture_preflight: proven", report)
        self.assertNotIn("recommended_runtime_capture_command_contract: proven", report)
        self.assertIn("recommended_runtime_capture_execution_coverage: blocked", report)
        self.assertIn("phase3_blocker_closure_manifest: proven", report)
        self.assertIn("phase3_blocker_resolution_queue: proven", report)
        self.assertIn("Phase 3 complete: False", report)
    def test_markdown_report_writes_requested_path(self) -> None:
        summary = planner.build_packet_summary(
            TRACE_FIXTURE,
            INVENTORY_FIXTURE,
            POLICIES_FIXTURE,
            MANAGED_PLAN,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "reports" / "phase3.md"
            planner.write_markdown_report(summary, output_path)
            report = output_path.read_text(encoding="utf-8")

        self.assertIn("Phase 3 complete: False", report)
        self.assertIn("Decision: `no_go_live_spike`", report)


if __name__ == "__main__":
    unittest.main()
