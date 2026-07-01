import copy
import importlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = ROOT / "scripts" / "plan_phase3_capture_result_intake.py"
SPEC = importlib.util.spec_from_file_location("plan_phase3_capture_result_intake", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TESTS_DIR))
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)

build_phase3_runtime_capture_request = importlib.import_module("build_phase3_runtime_capture_request")
build_phase3_output_summary = importlib.import_module("build_phase3_output_summary")
build_phase3_real_evidence_bundle = importlib.import_module("build_phase3_real_evidence_bundle")
plan_phase3_capture_completion_receipt = importlib.import_module("plan_phase3_capture_completion_receipt")
positive = importlib.import_module("test_phase3_positive_path")


POLICIES_FIXTURE = ROOT / "memory-moe-mvp" / "data" / "baseline_replay_policies.json"
MANAGED_PLAN = ROOT / "memory-moe-mvp" / "data" / "managed_expert_loading_plan.json"
REQUEST = ROOT / "memory-moe-mvp" / "phase3-real-evidence" / "pc_mixtral_phase3_real_evidence_bundle.runtime-capture-request.json"


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def prompt_set() -> dict:
    return {
        "model_id": positive.MODEL_ID,
        "backend_family": "llama_cpp",
        "prompt_family": "phase3-intake-test",
        "prompt_count": 1,
        "prompts": [{"prompt_id": "case-001", "prompt": "Say one concise thing about expert caching."}],
        "safety_contract": ["test prompt set only"],
        "next_actions": ["use saved outputs only"],
    }


def noncandidate_trace_events() -> list[dict]:
    rows = copy.deepcopy(positive.local_trace_events())
    for row in rows:
        row["shape"] = [1, 1, 1, 1]
        row["value_count"] = 1
        row["values"] = [0] if row["tensor_kind"] == "selected_experts" else [1.0]
    return rows


def ready_receipt(
    label: str,
    *,
    request_path: Path | str = "phase3-bundle.runtime-capture-request.json",
    prompt_path: Path | str = "phase3-bundle.prompt-set.json",
    overrides: dict | None = None,
) -> dict:
    receipt = {
        "receipt_ready": True,
        "source_request_path": str(request_path),
        "source_prompt_set_path": str(prompt_path),
        "runtime_capture_approved": True,
        "runtime_prompt_traffic_approved": True,
        "captured_at": "2026-06-28T21:00:00Z",
        "capture_host": "pc-intake-test",
        "runtime_backend": "llama_cpp",
        "model_id": positive.MODEL_ID,
        "prompt_family": "phase3-intake-test",
        "output_label": label,
        "operator_notes": "test receipt",
    }
    if overrides:
        receipt.update(overrides)
    return receipt


def trace_receipt(
    candidate_path: Path,
    prompt_path: Path,
    request_path: Path,
    *,
    overrides: dict | None = None,
) -> dict:
    receipt = {
        "schema_version": "moe-phase3-trace-capture-receipt-v1",
        "receipt_ready": True,
        "source_request_path": str(request_path),
        "source_prompt_set_path": str(prompt_path),
        "candidate_trace_path": str(candidate_path),
        "router_trace_capture_approved": True,
        "runtime_prompt_traffic_approved": True,
        "captured_at": "2026-06-28T21:00:00Z",
        "capture_host": "pc-intake-test",
        "runtime_backend": "llama_cpp",
        "model_id": positive.MODEL_ID,
        "prompt_family": "phase3-intake-test",
        "operator_notes": "test trace receipt",
    }
    if overrides:
        receipt.update(overrides)
    return receipt


def output_summary(
    label: str,
    *,
    request_path: Path | str = "phase3-bundle.runtime-capture-request.json",
    prompt_path: Path | str = "phase3-bundle.prompt-set.json",
    receipt_overrides: dict | None = None,
) -> dict:
    return {
        "schema_version": "moe-phase3-output-summary-v1",
        "model_id": positive.MODEL_ID,
        "backend_family": "llama_cpp",
        "prompt_family": "phase3-intake-test",
        "output_label": label,
        "output_ready": True,
        "capture_receipt": ready_receipt(
            label,
            request_path=request_path,
            prompt_path=prompt_path,
            overrides=receipt_overrides,
        ),
        "outputs": [
            {
                "prompt_id": "case-001",
                "output_label": label,
                "output": "same answer",
                "ready": True,
                "error": None,
            }
        ],
        "safety_contract": ["test output summary only"],
        "next_actions": ["build comparison artifact"],
    }


def minimal_intake_summary() -> dict:
    return {
        "valid": True,
        "request_count": 1,
        "request_audit_valid_count": 1,
        "request_drift_free_count": 1,
        "request_drifted_count": 0,
        "request_capture_complete_flag_count": 0,
        "request_ready_for_operator_capture_flag_count": 0,
        "approved_but_capture_incomplete_request_count": 0,
        "runtime_approval_missing_request_count": 1,
        "approval_rebuild_command_available_request_count": 1,
        "approval_rebuild_command_manifest": [
            {
                "request_name": "Fixture request",
                "request_path": "fixture-bundle.runtime-capture-request.json",
                "bundle_path": "fixture-bundle.json",
                "next_step_id": "candidate_router_trace",
                "command_class": "phase3_runtime_capture_request_approval_rebuild",
                "writes_request_path": "fixture-bundle.runtime-capture-request.json",
                "records_approval_keys": ["runtime_prompt_traffic_approved"],
                "requires_explicit_user_approval": True,
                "metadata_only": True,
                "command": [
                    "uv",
                    "run",
                    "--managed-python",
                    "--python",
                    "3.13",
                    "scripts/build_phase3_runtime_capture_request.py",
                    "fixture-bundle.json",
                    "--json",
                    "--runtime-prompt-traffic-approved",
                ],
            }
        ],
        "approved_runtime_capture_pending_request_count": 0,
        "approval_transition_preview_count": 1,
        "approval_transition_ready_for_operator_count": 1,
        "approval_transition_ready_to_update_bundle_count": 0,
        "recommended_approval_transition_preview": {
            "preview_only": True,
            "valid": True,
            "request_name": "Fixture request",
            "request_path": "fixture-bundle.runtime-capture-request.json",
            "rank": 1,
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
            "records_approval_keys": ["runtime_prompt_traffic_approved"],
            "requires_explicit_user_approval": True,
            "metadata_only": True,
            "mutates_request": False,
            "capture_complete_after_approval": False,
            "approved_but_capture_incomplete_after_approval": True,
            "ready_to_update_bundle_after_approval": False,
            "receipt_fill_preview": {
                "entry_count": 1,
                "ready_after_approval_count": 0,
                "missing_after_approval_count": 1,
                "approval_missing_after_approval_count": 0,
                "all_ready_after_approval": False,
            },
        },
        "approval_transition_preview_manifest": [
            {
                "preview_only": True,
                "valid": True,
                "request_name": "Fixture request",
                "request_path": "fixture-bundle.runtime-capture-request.json",
                "rank": 1,
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
                "records_approval_keys": ["runtime_prompt_traffic_approved"],
                "requires_explicit_user_approval": True,
                "metadata_only": True,
                "mutates_request": False,
                "capture_complete_after_approval": False,
                "approved_but_capture_incomplete_after_approval": True,
                "ready_to_update_bundle_after_approval": False,
                "receipt_fill_preview": {
                    "entry_count": 1,
                    "ready_after_approval_count": 0,
                    "missing_after_approval_count": 1,
                    "approval_missing_after_approval_count": 0,
                    "all_ready_after_approval": False,
                },
            }
        ],
        "post_approval_capture_fill_plan_count": 1,
        "post_approval_capture_fill_artifact_step_count": 1,
        "post_approval_capture_fill_runtime_step_count": 1,
        "post_approval_capture_fill_ready_count": 0,
        "post_approval_capture_fill_missing_count": 1,
        "post_approval_capture_fill_validator_command_count": 1,
        "post_approval_capture_fill_ready_to_update_bundle_count": 0,
        "recommended_post_approval_capture_fill_plan": {
            "preview_only": True,
            "valid": True,
            "request_name": "Fixture request",
            "request_path": "fixture-bundle.runtime-capture-request.json",
            "rank": 1,
            "selection_rationale": "stable_name_order_after_mixtral_baseline",
            "requires_explicit_user_approval": True,
            "metadata_only": True,
            "mutates_request": False,
            "artifact_step_count": 1,
            "runtime_capture_step_count": 1,
            "ready_after_approval_count": 0,
            "missing_after_approval_count": 1,
            "validator_command_count": 1,
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
                    "validator_command_count": 1,
                    "command_option_count": 1,
                    "blockers_after_approval": ["candidate_trace_capture_receipt_not_ready"],
                    "errors": [],
                }
            ],
        },
        "post_approval_capture_fill_plan_manifest": [
            {
                "preview_only": True,
                "valid": True,
                "request_name": "Fixture request",
                "request_path": "fixture-bundle.runtime-capture-request.json",
                "rank": 1,
                "selection_rationale": "stable_name_order_after_mixtral_baseline",
                "requires_explicit_user_approval": True,
                "metadata_only": True,
                "mutates_request": False,
                "artifact_step_count": 1,
                "runtime_capture_step_count": 1,
                "ready_after_approval_count": 0,
                "missing_after_approval_count": 1,
                "validator_command_count": 1,
                "command_option_count": 1,
                "all_receipts_ready_after_approval": False,
                "ready_to_update_bundle_after_approval": False,
                "capture_fill_steps": [],
            }
        ],
        "capture_receipt_required_count": 3,
        "capture_receipt_ready_count": 0,
        "receipt_fill_entry_count": 1,
        "receipt_fill_ready_count": 0,
        "receipt_fill_missing_count": 1,
        "receipt_fill_approval_missing_count": 1,
        "receipt_fill_manifest": [
            {
                "request_name": "Fixture request",
                "request_path": "fixture-bundle.runtime-capture-request.json",
                "bundle_path": "fixture-bundle.json",
                "artifact_id": "candidate_router_trace",
                "receipt_kind": "trace_capture_receipt_json",
                "artifact_path": "memory-moe-mvp/phase3-real-evidence/fixture/candidate-router-events.jsonl",
                "receipt_path": "memory-moe-mvp/phase3-real-evidence/fixture/candidate-router-events.capture-receipt.json",
                "ready": False,
                "receipt_ready": False,
                "approvals_ready": False,
                "approval_state": "missing",
                "blockers": ["recorded_request_approvals_missing", "candidate_trace_capture_receipt_not_ready"],
                "errors": [],
            }
        ],
        "receipt_fill_manifest_summary": {
            "entry_count": 1,
            "ready_count": 0,
            "missing_count": 1,
            "approval_missing_count": 1,
            "by_artifact": {
                "candidate_router_trace": {"entry_count": 1, "ready_count": 0, "missing_count": 1},
            },
            "all_ready": False,
        },
        "capture_receipt_missing_request_count": 1,
        "output_receipt_binding_ready_count": 0,
        "receipt_gate_coverage": {
            "request_count": 1,
            "capture_receipt_required_count": 3,
            "capture_receipt_ready_count": 0,
            "approvals_ready_count": 0,
            "candidate_trace_receipt_ready_count": 0,
            "managed_output_receipt_ready_count": 0,
            "dense_output_receipt_ready_count": 0,
            "output_receipt_binding_ready_count": 0,
            "live_capability_proof_ready_count": 0,
            "missing_request_count": 1,
            "missing_by_request": [
                {
                    "request_name": "Fixture request",
                    "missing_gate_ids": [
                        "approvals_ready",
                        "candidate_trace_receipt_ready",
                        "managed_output_receipt_ready",
                        "dense_output_receipt_ready",
                        "output_receipt_binding_ready",
                        "live_capability_proof_ready",
                    ],
                }
            ],
            "all_capture_receipts_ready": False,
            "all_output_receipt_bindings_ready": False,
        },
        "ready_to_update_bundle_count": 0,
        "phase4_candidate_ready_count": 0,
        "live_spike_candidate_ready_count": 0,
        "pending_runtime_capture_request_count": 1,
        "pending_bundle_update_request_count": 0,
        "remaining_blockers_after_intake": ["no_replay_policy_candidate"],
        "safety_contract": ["metadata only"],
        "requests": [
            {
                "name": "Fixture request",
                "next_operator_step": {
                    "id": "candidate_router_trace",
                    "status": "approval_required",
                    "stage": "runtime_capture",
                    "approval_required": True,
                    "approval_state": "missing",
                    "path": "memory-moe-mvp/phase3-real-evidence/fixture/candidate-router-events.jsonl",
                    "command_options": [
                        {
                            "command_class": "phase3_runtime_capture_request_approval_rebuild",
                            "command": [
                                "uv",
                                "run",
                                "--managed-python",
                                "--python",
                                "3.13",
                                "scripts/build_phase3_runtime_capture_request.py",
                                "fixture-bundle.json",
                                "--json",
                                "--runtime-prompt-traffic-approved",
                            ],
                            "records_approval_keys": ["runtime_prompt_traffic_approved"],
                            "writes_request_path": "fixture-bundle.runtime-capture-request.json",
                            "requires_explicit_user_approval": True,
                            "metadata_only": True,
                        }
                    ],
                },
                "operator_queue": [
                    {
                        "id": "candidate_router_trace",
                        "stage": "runtime_capture",
                        "status": "approval_required",
                        "approval_required": True,
                        "approval_state": "missing",
                        "path": "memory-moe-mvp/phase3-real-evidence/fixture/candidate-router-events.jsonl",
                        "command_options": [
                            {
                                "command_class": "phase3_runtime_capture_request_approval_rebuild",
                                "command": [
                                    "uv",
                                    "run",
                                    "--managed-python",
                                    "--python",
                                    "3.13",
                                    "scripts/build_phase3_runtime_capture_request.py",
                                    "fixture-bundle.json",
                                    "--json",
                                    "--runtime-prompt-traffic-approved",
                                ],
                                "metadata_only": True,
                            }
                        ],
                        "validator_commands": [
                            [
                                "uv",
                                "run",
                                "--managed-python",
                                "--python",
                                "3.13",
                                "scripts/validate_llama_cpp_router_trace.py",
                                "fixture.jsonl",
                            ]
                        ],
                    },
                    {
                        "id": "phase3_bundle_with_candidate_trace",
                        "stage": "bundle_update",
                        "status": "ready_to_build",
                        "approval_required": False,
                        "approval_state": "not_required",
                        "path": "memory-moe-mvp/phase3-real-evidence/fixture.updated.json",
                        "validator_commands": [],
                        "command_options": [
                            {
                                "command_class": "phase3_bundle_builder_with_candidate_trace",
                                "command": [
                                    "uv",
                                    "run",
                                    "--managed-python",
                                    "--python",
                                    "3.13",
                                    "scripts/build_phase3_real_evidence_bundle.py",
                                    "--json",
                                ],
                            }
                        ],
                    },
                ],
            }
        ],
    }

def write_filled_request_fixture(
    temp_path: Path,
    *,
    approvals: bool = True,
    trace_receipt_overrides: dict | None = None,
    managed_receipt_overrides: dict | None = None,
    dense_receipt_overrides: dict | None = None,
) -> Path:
    bundle_path = temp_path / "phase3-bundle.json"
    trace_path = temp_path / "router-events.jsonl"
    inventory_path = temp_path / "expert-inventory.json"
    prompt_path = temp_path / "phase3-bundle.prompt-set.json"
    candidate_path = temp_path / "phase3-bundle-policy-candidate" / "candidate-router-events.jsonl"
    candidate_receipt_path = candidate_path.with_name("candidate-router-events.capture-receipt.json")
    fallback_dir = temp_path / "phase3-bundle-fallback"
    managed_path = fallback_dir / "managed-output-summary.json"
    dense_path = fallback_dir / "dense-output-summary.json"
    fallback_path = fallback_dir / "dense-fallback-comparison.json"
    request_path = temp_path / "phase3-bundle.runtime-capture-request.json"

    write_jsonl(trace_path, noncandidate_trace_events())
    write_jsonl(candidate_path, positive.local_trace_events())
    write_json(request_path, {"schema_version": "pending-runtime-capture-request"})
    write_json(candidate_receipt_path, trace_receipt(candidate_path, prompt_path, request_path, overrides=trace_receipt_overrides))
    write_json(inventory_path, positive.local_inventory())
    write_json(prompt_path, prompt_set())
    write_json(
        managed_path,
        output_summary(
            "managed",
            request_path=request_path,
            prompt_path=prompt_path,
            receipt_overrides=managed_receipt_overrides,
        ),
    )
    write_json(
        dense_path,
        output_summary(
            "dense",
            request_path=request_path,
            prompt_path=prompt_path,
            receipt_overrides=dense_receipt_overrides,
        ),
    )
    write_json(fallback_path, positive.fallback_artifact())

    manifest = build_phase3_real_evidence_bundle.build_manifest(
        name="Intake positive bundle",
        model_id=positive.MODEL_ID,
        source_format="gguf",
        backend_family="llama_cpp",
        prompt_family="phase3-intake-test",
        trace_path=trace_path,
        inventory_path=inventory_path,
        policies_path=POLICIES_FIXTURE,
        managed_plan_path=MANAGED_PLAN,
        fallback_artifact_path=None,
        real_model_trace_capture_approved=approvals,
        dense_fallback_capture_approved=False,
        runtime_prompt_traffic_approved=False,
        approval_notes=["Intake test artifact only."],
    )
    write_json(bundle_path, manifest)

    request = build_phase3_runtime_capture_request.build_request(
        bundle_path,
        prompt_set_path=prompt_path,
        candidate_trace_path=candidate_path,
        candidate_trace_receipt_path=candidate_receipt_path,
        managed_output_path=managed_path,
        dense_output_path=dense_path,
        router_trace_capture_approved=approvals,
        managed_output_capture_approved=approvals,
        dense_output_capture_approved=approvals,
        runtime_prompt_traffic_approved=approvals,
    )
    write_json(request_path, request)
    return request_path


def filled_completion_receipt_from_work_order(work_order: dict) -> dict:
    receipt = plan_phase3_capture_completion_receipt.completion_receipt_template_from_work_order(work_order)
    receipt["receipt_complete"] = True
    receipt["ready_for_capture_result_intake"] = True
    for row in receipt["capture_receipts"]:
        row["capture_complete"] = True
        row["receipt_filled"] = True
        row["validator_passed"] = True
        row["ready_for_intake"] = True
        row["observed_at"] = "2026-06-30T21:00:00Z"
    return receipt


def write_completion_receipt_pair(temp_path: Path, *, filled: bool = True) -> tuple[Path, Path]:
    work_order = plan_phase3_capture_completion_receipt.fixture_work_order()
    receipt = (
        filled_completion_receipt_from_work_order(work_order)
        if filled
        else plan_phase3_capture_completion_receipt.completion_receipt_template_from_work_order(work_order)
    )
    work_order_path = temp_path / "recommended-runtime-capture-work-order.json"
    receipt_path = temp_path / "recommended-runtime-capture-completion-receipt.json"
    write_json(work_order_path, work_order)
    write_json(receipt_path, receipt)
    return receipt_path, work_order_path

def write_approved_unfilled_request_fixture(temp_path: Path) -> Path:
    bundle_path = temp_path / "phase3-bundle.json"
    trace_path = temp_path / "router-events.jsonl"
    inventory_path = temp_path / "expert-inventory.json"
    prompt_path = temp_path / "phase3-bundle.prompt-set.json"
    candidate_path = temp_path / "phase3-bundle-policy-candidate" / "candidate-router-events.jsonl"
    candidate_receipt_path = candidate_path.with_name("candidate-router-events.capture-receipt.json")
    fallback_dir = temp_path / "phase3-bundle-fallback"
    managed_path = fallback_dir / "managed-output-summary.json"
    dense_path = fallback_dir / "dense-output-summary.json"
    request_path = temp_path / "phase3-bundle.runtime-capture-request.json"

    write_jsonl(trace_path, noncandidate_trace_events())
    write_json(inventory_path, positive.local_inventory())
    write_json(prompt_path, prompt_set())
    write_json(managed_path, build_phase3_output_summary.build_template_artifact(prompt_path, output_label="managed"))
    write_json(dense_path, build_phase3_output_summary.build_template_artifact(prompt_path, output_label="dense"))

    manifest = build_phase3_real_evidence_bundle.build_manifest(
        name="Intake approved unfilled bundle",
        model_id=positive.MODEL_ID,
        source_format="gguf",
        backend_family="llama_cpp",
        prompt_family="phase3-intake-test",
        trace_path=trace_path,
        inventory_path=inventory_path,
        policies_path=POLICIES_FIXTURE,
        managed_plan_path=MANAGED_PLAN,
        fallback_artifact_path=None,
        real_model_trace_capture_approved=True,
        dense_fallback_capture_approved=False,
        runtime_prompt_traffic_approved=True,
        approval_notes=["Intake approved-unfilled test artifact only."],
    )
    write_json(bundle_path, manifest)

    request = build_phase3_runtime_capture_request.build_request(
        bundle_path,
        prompt_set_path=prompt_path,
        candidate_trace_path=candidate_path,
        candidate_trace_receipt_path=candidate_receipt_path,
        managed_output_path=managed_path,
        dense_output_path=dense_path,
        router_trace_capture_approved=True,
        managed_output_capture_approved=True,
        dense_output_capture_approved=True,
        runtime_prompt_traffic_approved=True,
    )
    write_json(request_path, request)
    return request_path


class Phase3CaptureResultIntakePlannerTests(unittest.TestCase):
    def test_current_repo_requests_are_valid_but_not_ready_to_update(self) -> None:
        summary = planner.build_root_summary()

        self.assertTrue(summary["valid"], json.dumps(summary["errors"], indent=2))
        self.assertEqual(summary["request_count"], 6)
        self.assertEqual(summary["valid_request_count"], 6)
        self.assertEqual(summary["request_audit_valid_count"], 6)
        self.assertEqual(summary["request_drift_free_count"], 6)
        self.assertEqual(summary["request_drifted_count"], 0)
        self.assertEqual(summary["request_capture_complete_flag_count"], 0)
        self.assertEqual(summary["request_ready_for_operator_capture_flag_count"], 0)
        self.assertEqual(summary["approved_but_capture_incomplete_request_count"], 0)
        self.assertEqual(summary["runtime_approval_missing_request_count"], 6)
        self.assertEqual(summary["approval_rebuild_command_available_request_count"], 6)
        approval_manifest = summary["approval_rebuild_command_manifest"]
        self.assertEqual(len(approval_manifest), 6)
        self.assertEqual(approval_manifest[0]["command_class"], "phase3_runtime_capture_request_approval_rebuild")
        self.assertTrue(approval_manifest[0]["requires_explicit_user_approval"])
        self.assertTrue(approval_manifest[0]["metadata_only"])
        self.assertEqual(summary["approved_runtime_capture_pending_request_count"], 0)
        self.assertEqual(summary["approval_transition_preview_count"], 6)
        self.assertEqual(summary["approval_transition_ready_for_operator_count"], 6)
        self.assertEqual(summary["approval_transition_ready_to_update_bundle_count"], 0)
        self.assertIsNotNone(summary["recommended_approval_transition_preview"])
        self.assertEqual(
            summary["recommended_approval_transition_preview"]["next_step_after_approval"]["status"],
            "ready_for_operator_capture",
        )
        self.assertEqual(summary["post_approval_capture_fill_plan_count"], 6)
        self.assertEqual(summary["post_approval_capture_fill_artifact_step_count"], 18)
        self.assertEqual(summary["post_approval_capture_fill_runtime_step_count"], 18)
        self.assertEqual(summary["post_approval_capture_fill_ready_count"], 0)
        self.assertEqual(summary["post_approval_capture_fill_missing_count"], 18)
        self.assertEqual(summary["post_approval_capture_fill_validator_command_count"], 30)
        self.assertEqual(summary["post_approval_capture_fill_ready_to_update_bundle_count"], 0)
        self.assertIsNotNone(summary["recommended_post_approval_capture_fill_plan"])
        self.assertEqual(summary["capture_receipt_required_count"], 18)
        self.assertEqual(summary["capture_receipt_ready_count"], 6)
        self.assertEqual(summary["receipt_fill_entry_count"], 18)
        self.assertEqual(summary["receipt_fill_ready_count"], 0)
        self.assertEqual(summary["receipt_fill_missing_count"], 18)
        self.assertEqual(summary["receipt_fill_approval_missing_count"], 18)
        receipt_manifest = summary["receipt_fill_manifest"]
        self.assertEqual(len(receipt_manifest), 18)
        self.assertEqual(
            {item["artifact_id"] for item in receipt_manifest},
            {"candidate_router_trace", "managed_output_summary_fill", "dense_output_summary_fill"},
        )
        self.assertEqual(
            summary["receipt_fill_manifest_summary"]["by_artifact"]["candidate_router_trace"],
            {"entry_count": 6, "ready_count": 0, "missing_count": 6},
        )
        self.assertEqual(summary["capture_receipt_missing_request_count"], 6)
        self.assertEqual(summary["output_receipt_binding_ready_count"], 0)
        coverage = summary["receipt_gate_coverage"]
        self.assertEqual(coverage["candidate_trace_receipt_ready_count"], 6)
        self.assertEqual(coverage["managed_output_receipt_ready_count"], 0)
        self.assertEqual(coverage["dense_output_receipt_ready_count"], 0)
        self.assertEqual(coverage["live_capability_proof_ready_count"], 0)
        self.assertFalse(coverage["all_capture_receipts_ready"])
        self.assertFalse(coverage["all_output_receipt_bindings_ready"])
        self.assertEqual(summary["ready_to_update_bundle_count"], 0)
        self.assertEqual(summary["phase4_candidate_ready_count"], 0)
        self.assertEqual(summary["pending_runtime_capture_request_count"], 6)
        self.assertEqual(summary["pending_bundle_update_request_count"], 0)
        self.assertEqual(summary["next_operator_step_counts"], {"candidate_router_trace": 6})
        next_step = summary["requests"][0]["next_operator_step"]
        self.assertEqual(next_step["id"], "candidate_router_trace")
        self.assertEqual(next_step["stage"], "runtime_capture")
        self.assertEqual(next_step["status"], "approval_required")
        self.assertTrue(next_step["approval_required"])
        self.assertEqual(next_step["approval_state"], "missing")
        self.assertEqual(len(next_step["command_options"]), 2)
        self.assertIn("no_replay_policy_candidate", summary["remaining_blockers_after_intake"])
        self.assertIn("dense_fallback_output_artifact_for_quality_bounds", summary["remaining_blockers_after_intake"])
        self.assertIn("live_residency_observation_and_control", summary["remaining_blockers_after_intake"])

    def test_queue_line_summarizes_next_operator_step(self) -> None:
        summary = minimal_intake_summary()

        line = planner.request_queue_line(summary["requests"][0])

        self.assertIn("candidate_router_trace approval_required", line)
        self.assertIn("runtime_capture", line)
        self.assertIn("approval missing; approval command ready", line)
        self.assertIn("candidate-router-events.jsonl", line)

    def test_markdown_report_contains_queue_table(self) -> None:
        summary = minimal_intake_summary()

        report = planner.format_markdown_report(summary)

        self.assertIn("# Phase 3 Capture-Result Intake", report)
        self.assertIn("Drift-free saved requests", report)
        self.assertIn("Drifted saved requests", report)
        self.assertIn("Ready-for-operator flags: `0`", report)
        self.assertIn("Approved but capture incomplete: `0`", report)
        self.assertIn("Runtime approvals still missing: `1`", report)
        self.assertIn("Approval rebuild commands available: `1`", report)
        self.assertIn("## Approval Rebuild Command Manifest", report)
        self.assertIn("Records: `runtime_prompt_traffic_approved`", report)
        self.assertIn("fixture-bundle.runtime-capture-request.json", report)
        self.assertIn("Approved runtime-capture pending: `0`", report)
        self.assertIn("Approval transition previews: `1`", report)
        self.assertIn("Approval previews ready for operator: `1`", report)
        self.assertIn("Approval previews ready to update bundle: `0`", report)
        self.assertIn("Post-approval capture-fill plans: `1`", report)
        self.assertIn("Post-approval capture-fill ready: `0` / `1`", report)
        self.assertIn("Post-approval capture-fill validator commands: `1`", report)
        self.assertIn("## Approval Transition Preview", report)
        self.assertIn("candidate_router_trace:approval_required:missing", report)
        self.assertIn("candidate_router_trace:ready_for_operator_capture:recorded", report)
        self.assertIn("## Post-Approval Capture Fill Plan", report)
        self.assertIn("## Recommended Post-Approval Capture Steps", report)
        self.assertIn("blocked_after_approval", report)
        self.assertIn("candidate_trace_capture_receipt_not_ready", report)
        self.assertIn("Capture receipts ready: `0` / `3`", report)
        self.assertIn("Receipt-fill manifest ready: `0` / `1`", report)
        self.assertIn("Receipt-fill approvals missing: `1`", report)
        self.assertIn("Requests missing receipt gates: `1`", report)
        self.assertIn("Output receipt bindings ready: `0`", report)
        self.assertIn("## Receipt Fill Manifest", report)
        self.assertIn("trace_capture_receipt_json", report)
        self.assertIn("recorded_request_approvals_missing", report)
        self.assertIn("| Request | Step | Status | Stage | Approval | Path |", report)
        self.assertIn("candidate_router_trace", report)
        self.assertIn("approval_required", report)
        self.assertIn("## Detailed Operator Queues", report)
        self.assertIn("phase3_bundle_with_candidate_trace", report)
        self.assertIn("uv run --managed-python --python 3.13 scripts/validate_llama_cpp_router_trace.py fixture.jsonl", report)
        self.assertIn("uv run --managed-python --python 3.13 scripts/build_phase3_real_evidence_bundle.py --json", report)
        self.assertIn("## Remaining Blockers", report)
        self.assertIn("no_replay_policy_candidate", report)

    def test_markdown_report_writes_requested_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "reports" / "intake.md"
            summary = minimal_intake_summary()

            planner.write_markdown_report(summary, output_path)
            written = output_path.read_text(encoding="utf-8")

        self.assertIn("# Phase 3 Capture-Result Intake", written)
        self.assertIn("candidate_router_trace", written)

    def test_missing_bundle_is_a_request_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            request_path = Path(temp_dir) / REQUEST.name
            payload = json.loads(REQUEST.read_text(encoding="utf-8"))
            drifted = copy.deepcopy(payload)
            drifted["bundle_path"] = "memory-moe-mvp/phase3-real-evidence/missing-bundle.json"
            write_json(request_path, drifted)

            summary = planner.summarize_request(request_path)

        self.assertFalse(summary["valid"])
        self.assertTrue(any("request bundle does not exist" in error for error in summary["errors"]))

    def test_approved_unfilled_request_is_pending_capture_not_missing_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            request_path = write_approved_unfilled_request_fixture(temp_path)

            summary = planner.summarize_request(request_path)
            root_summary = planner.build_root_summary(root=temp_path)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertFalse(summary["request_capture_complete_flag"])
        self.assertTrue(summary["request_ready_for_operator_capture_flag"])
        self.assertTrue(summary["approved_but_capture_incomplete"])
        self.assertTrue(summary["approvals_ready_for_recorded_capture"])
        self.assertEqual(summary["runtime_capture_step_count"], 3)
        self.assertEqual(summary["runtime_approval_missing_step_count"], 0)
        self.assertEqual(summary["approved_runtime_capture_step_count"], 3)
        self.assertEqual(summary["approval_required_step_count"], 4)
        next_step = summary["next_operator_step"]
        self.assertEqual(next_step["id"], "candidate_router_trace")
        self.assertEqual(next_step["stage"], "runtime_capture")
        self.assertEqual(next_step["status"], "ready_for_operator_capture")
        self.assertTrue(next_step["approval_required"])
        self.assertEqual(next_step["approval_state"], "recorded")
        self.assertEqual(next_step["command_options"], [])
        self.assertEqual(
            [step["status"] for step in summary["operator_queue"][:4]],
            [
                "ready_for_operator_capture",
                "blocked_by_candidate_trace",
                "ready_for_operator_capture",
                "ready_for_operator_capture",
            ],
        )
        self.assertTrue(root_summary["valid"], root_summary["errors"])
        self.assertEqual(root_summary["request_ready_for_operator_capture_flag_count"], 1)
        self.assertEqual(root_summary["approved_but_capture_incomplete_request_count"], 1)
        self.assertEqual(root_summary["runtime_approval_missing_request_count"], 0)
        self.assertEqual(root_summary["approval_rebuild_command_available_request_count"], 0)
        self.assertEqual(root_summary["approved_runtime_capture_pending_request_count"], 1)
        self.assertEqual(root_summary["approval_transition_preview_count"], 0)
        self.assertEqual(root_summary["approval_transition_ready_for_operator_count"], 0)
        self.assertEqual(root_summary["approval_transition_ready_to_update_bundle_count"], 0)
        self.assertIsNone(root_summary["recommended_approval_transition_preview"])
        self.assertEqual(root_summary["post_approval_capture_fill_plan_count"], 0)
        self.assertEqual(root_summary["post_approval_capture_fill_artifact_step_count"], 0)
        self.assertEqual(root_summary["post_approval_capture_fill_ready_count"], 0)
        self.assertEqual(root_summary["post_approval_capture_fill_missing_count"], 0)
        self.assertEqual(root_summary["post_approval_capture_fill_validator_command_count"], 0)
        self.assertEqual(root_summary["post_approval_capture_fill_ready_to_update_bundle_count"], 0)
        self.assertIsNone(root_summary["recommended_post_approval_capture_fill_plan"])
        self.assertEqual(root_summary["pending_runtime_capture_request_count"], 1)
        self.assertEqual(root_summary["next_operator_step_counts"], {"candidate_router_trace": 1})

    def test_filled_capture_artifacts_are_ready_to_update_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            request_path = write_filled_request_fixture(Path(temp_dir))

            summary = planner.summarize_request(request_path)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertTrue(summary["request_capture_complete_flag"])
        self.assertTrue(summary["approvals_ready_for_recorded_capture"])
        self.assertTrue(summary["capture_receipt_binding"]["ready"])
        receipt_manifest = planner.receipt_fill_manifest([summary])
        receipt_summary = planner.receipt_fill_manifest_summary(receipt_manifest)
        self.assertEqual(receipt_summary["entry_count"], 3)
        self.assertEqual(receipt_summary["ready_count"], 3)
        self.assertEqual(receipt_summary["missing_count"], 0)
        self.assertEqual(receipt_summary["approval_missing_count"], 0)
        self.assertTrue(all(item["ready"] is True for item in receipt_manifest))
        self.assertEqual(
            {item["receipt_kind"] for item in receipt_manifest},
            {"trace_capture_receipt_json", "embedded_output_capture_receipt"},
        )
        receipt_status = planner.receipt_gate_status(summary)
        self.assertEqual(receipt_status["capture_receipt_required_count"], 3)
        self.assertEqual(receipt_status["capture_receipt_ready_count"], 3)
        self.assertTrue(receipt_status["output_receipt_binding_ready"])
        self.assertFalse(receipt_status["live_capability_proof_ready"])
        self.assertIn("live_capability_proof_ready", receipt_status["missing_gate_ids"])
        self.assertTrue(summary["ready_to_update_bundle"])
        self.assertTrue(summary["phase4_candidate_ready_after_intake"])
        self.assertFalse(summary["live_spike_candidate_ready_after_intake"])
        self.assertEqual(summary["runtime_capture_step_count"], 0)
        self.assertEqual(summary["next_operator_step"]["stage"], "bundle_update")
        self.assertIn(
            summary["next_operator_step"]["id"],
            {"phase3_bundle_with_candidate_trace", "phase3_bundle_with_fallback"},
        )
        self.assertFalse(summary["next_operator_step"]["approval_required"])
        self.assertEqual(summary["policy_candidate"]["bundle_update_status"], "ready_to_build")
        self.assertEqual(summary["dense_fallback"]["bundle_update_status"], "ready_to_build")
        self.assertTrue(summary["dense_fallback"]["capture_receipt_binding_ready"])
        self.assertNotIn("no_replay_policy_candidate", summary["remaining_blockers_after_intake"])
        self.assertNotIn("dense_fallback_output_artifact_for_quality_bounds", summary["remaining_blockers_after_intake"])
        self.assertIn("live_residency_observation_and_control", summary["remaining_blockers_after_intake"])

    def test_root_summary_accepts_ready_completion_receipt_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            write_filled_request_fixture(temp_path)
            receipt_path, work_order_path = write_completion_receipt_pair(temp_path, filled=True)

            summary = planner.build_root_summary(
                root=temp_path,
                completion_receipt_path=receipt_path,
                completion_work_order_path=work_order_path,
            )

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertTrue(summary["completion_receipt_gate_provided"])
        self.assertTrue(summary["completion_receipt_gate_ready"])
        self.assertTrue(summary["completion_receipt_gate_complete"])
        self.assertEqual(summary["completion_receipt_gate_error_count"], 0)
        self.assertNotIn("completion_receipt_validation_not_ready", summary["remaining_blockers_after_intake"])

    def test_root_summary_rejects_unfilled_completion_receipt_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            write_filled_request_fixture(temp_path)
            receipt_path, work_order_path = write_completion_receipt_pair(temp_path, filled=False)

            summary = planner.build_root_summary(
                root=temp_path,
                completion_receipt_path=receipt_path,
                completion_work_order_path=work_order_path,
            )

        self.assertFalse(summary["valid"])
        self.assertTrue(summary["completion_receipt_gate_provided"])
        self.assertFalse(summary["completion_receipt_gate_ready"])
        self.assertFalse(summary["completion_receipt_gate_complete"])
        self.assertIn("completion_receipt_validation_not_ready", summary["remaining_blockers_after_intake"])
        self.assertIn("completion receipt validation not ready for capture-result intake", summary["errors"])

    def test_completion_receipt_gate_requires_work_order_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            write_filled_request_fixture(temp_path)
            receipt_path, _ = write_completion_receipt_pair(temp_path, filled=True)

            summary = planner.build_root_summary(root=temp_path, completion_receipt_path=receipt_path)

        self.assertFalse(summary["valid"])
        self.assertTrue(summary["completion_receipt_gate_provided"])
        self.assertFalse(summary["completion_receipt_gate_ready"])
        self.assertIn("completion_work_order_path_missing", summary["completion_receipt_gate"]["errors"])
        self.assertIn("completion_receipt_validation_not_ready", summary["remaining_blockers_after_intake"])
    def test_drifted_saved_request_does_not_update_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            request_path = write_filled_request_fixture(Path(temp_dir))
            drifted = json.loads(request_path.read_text(encoding="utf-8"))
            drifted["prompt_set"]["prompt_count"] = 999
            write_json(request_path, drifted)

            summary = planner.summarize_request(request_path)

        self.assertFalse(summary["valid"])
        self.assertEqual(summary["request_drift_count"], 1)
        self.assertFalse(summary["request_drift_free"])
        self.assertFalse(summary["ready_to_update_bundle"])
        self.assertFalse(summary["phase4_candidate_ready_after_intake"])
        self.assertEqual(summary["next_operator_step"]["id"], "regenerate_runtime_capture_request")
        self.assertIn("runtime_capture_request_drift", summary["remaining_blockers_after_intake"])
        self.assertTrue(any("request audit" in error and "prompt_set drifted" in error for error in summary["errors"]))

    def test_root_summary_counts_drifted_saved_requests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            request_path = write_filled_request_fixture(temp_path)
            drifted = json.loads(request_path.read_text(encoding="utf-8"))
            drifted["prompt_set"]["prompt_count"] = 999
            write_json(request_path, drifted)

            summary = planner.build_root_summary(root=temp_path)

        self.assertFalse(summary["valid"])
        self.assertEqual(summary["request_count"], 1)
        self.assertEqual(summary["request_drift_free_count"], 0)
        self.assertEqual(summary["request_drifted_count"], 1)
        self.assertEqual(summary["next_operator_step_counts"], {"regenerate_runtime_capture_request": 1})
        self.assertIn("runtime_capture_request_drift", summary["remaining_blockers_after_intake"])
        self.assertTrue(any("request audit" in error and "prompt_set drifted" in error for error in summary["errors"]))

    def test_filled_artifacts_without_recorded_approvals_do_not_update_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            request_path = write_filled_request_fixture(Path(temp_dir), approvals=False)

            summary = planner.summarize_request(request_path)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertTrue(summary["request_capture_complete_flag"])
        self.assertFalse(summary["approvals_ready_for_recorded_capture"])
        self.assertFalse(summary["capture_receipt_binding"]["ready"])
        self.assertIn("recorded_request_approvals_missing", summary["capture_receipt_binding"]["blockers"])
        self.assertFalse(summary["ready_to_update_bundle"])
        self.assertFalse(summary["phase4_candidate_ready_after_intake"])
        self.assertEqual(summary["next_operator_step"]["id"], "record_runtime_approvals")
        self.assertEqual(summary["next_operator_step"]["status"], "approval_required")
        self.assertTrue(summary["next_operator_step"]["approval_required"])
        self.assertIn("dense_fallback_output_artifact_for_quality_bounds", summary["remaining_blockers_after_intake"])

    def test_mismatched_capture_receipt_request_path_invalidates_intake(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            request_path = write_filled_request_fixture(
                temp_path,
                managed_receipt_overrides={
                    "source_request_path": str(temp_path / "other.runtime-capture-request.json"),
                },
            )

            summary = planner.summarize_request(request_path)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["capture_receipt_binding"]["ready"])
        self.assertFalse(summary["dense_fallback"]["comparison_ready"])
        self.assertFalse(summary["ready_to_update_bundle"])
        self.assertTrue(
            any("capture_receipt.source_request_path" in error and "match" in error for error in summary["errors"])
        )


    def test_mismatched_trace_receipt_request_path_invalidates_intake(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            request_path = write_filled_request_fixture(
                temp_path,
                trace_receipt_overrides={
                    "source_request_path": str(temp_path / "other.runtime-capture-request.json"),
                },
            )

            summary = planner.summarize_request(request_path)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["policy_candidate"]["ready_after_plan"])
        self.assertFalse(summary["policy_candidate"]["candidate_trace_capture_receipt"]["ready"])
        self.assertFalse(summary["phase4_candidate_ready_after_intake"])
        self.assertEqual(summary["policy_candidate"]["bundle_update_status"], "blocked_by_candidate_replay")
        self.assertTrue(
            any("trace_capture_receipt.source_request_path" in error and "match" in error for error in summary["errors"])
        )
if __name__ == "__main__":
    unittest.main()

