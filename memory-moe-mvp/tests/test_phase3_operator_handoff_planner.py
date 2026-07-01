import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_phase3_operator_handoff.py"
SPEC = importlib.util.spec_from_file_location("plan_phase3_operator_handoff", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


REQUIRED_KINDS = {
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


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_minimal_handoff(root: Path) -> None:
    request_path = "memory-moe-mvp/phase3-real-evidence/fixture.runtime-capture-request.json"
    launch_card_path = "memory-moe-mvp/phase3-real-evidence/fixture.runtime-capture-launch-card.template.json"
    prompt_set_path = "memory-moe-mvp/phase3-real-evidence/fixture.prompt-set.json"
    candidate_path = "memory-moe-mvp/phase3-real-evidence/fixture/candidate-router-events.jsonl"
    candidate_receipt_path = "memory-moe-mvp/phase3-real-evidence/fixture/candidate-router-events.capture-receipt.json"
    managed_path = "memory-moe-mvp/phase3-real-evidence/fixture/managed-output-summary.json"
    live_proof_path = "memory-moe-mvp/phase3-real-evidence/fixture.live-capability-proof.template.json"
    root.mkdir(parents=True, exist_ok=True)
    output_files = [{"path": name, "kind": kind} for name, kind in REQUIRED_KINDS.items()]
    manifest = {
        "schema_version": planner.plan_phase3_evidence_packet.OPERATOR_HANDOFF_SCHEMA_VERSION,
        "mode": "phase3_operator_handoff_package",
        "valid": True,
        "packet_ready": True,
        "phase3_complete": False,
        "decision": "no_go_live_spike",
        "metadata_only": True,
        "launches_runtimes": False,
        "runs_docker": False,
        "sends_prompt_traffic": False,
        "reads_private_tokens": False,
        "mutates_runtime_residency": False,
        "operator_handoff_ready": True,
        "recommended_request": {
            "request_name": "Fixture request",
            "request_path": request_path,
            "status": "approval_required",
            "queue_rank": 1,
            "selection_rationale": "fixture",
            "next_artifact_id": "candidate_router_trace",
            "next_artifact_path": candidate_path,
            "pending_artifact_ids": ["candidate_router_trace"],
            "missing_approval_keys": ["runtime_prompt_traffic_approved"],
            "approval_command_class": "phase3_runtime_capture_request_approval_rebuild",
            "approval_command": [
                "uv",
                "run",
                "--managed-python",
                "--python",
                "3.13",
                "scripts/build_phase3_runtime_capture_request.py",
                "memory-moe-mvp/phase3-real-evidence/fixture.json",
                "--output",
                request_path,
                "--json",
                "--runtime-prompt-traffic-approved",
            ],
            "records_approval_keys": ["runtime_prompt_traffic_approved"],
            "approval_metadata_only": True,
            "requires_explicit_user_approval": True,
        },
        "recommended_reuse_capture": {
            "bundle_name": "Fixture request",
            "bundle_path": "memory-moe-mvp/phase3-real-evidence/fixture.json",
            "model_id": "fixture-mixtral.gguf",
            "candidate_trace_path": candidate_path,
            "prompt_set_path": prompt_set_path,
            "runtime_capture_request_path": request_path,
            "selection_rationale": "fixture",
            "required_result": "capture_candidate_router_trace_with_reuse_distance",
            "minimum_reuse_distance_observations": 1,
            "must_bind_prompt_identity_metadata": True,
        },
        "recommended_launch_card_template": {
            "path": launch_card_path,
            "match_basis": "request_path",
            "template_ready": True,
            "binding_handoff_ready": True,
            "binding_ready": False,
            "runtime_capture_command_ready": False,
            "task_count": 2,
            "unbound_task_count": 2,
            "missing_runtime_command_count": 2,
        },
        "handoff_counts": {
            "capture_queue_ready": True,
            "capture_queue_request_count": 1,
            "capture_queue_pending_artifact_count": 1,
            "preflight_ready": True,
            "preflight_pending_artifact_count": 2,
            "preflight_validator_command_count": 2,
            "preflight_runtime_closure_reason_count": 1,
            "preflight_missing_item_count": 0,
            "approval_command_manifest_count": 1,
            "approval_manifest_parity_ready": True,
            "approval_manifest_parity_matched_request_count": 1,
            "validator_command_manifest_ready": True,
            "validator_command_count": 3,
            "receipt_fill_manifest_ready": False,
            "receipt_fill_entry_count": 2,
            "receipt_fill_missing_count": 2,
            "receipt_fill_candidate_router_trace_entry_count": 1,
            "receipt_fill_candidate_router_trace_ready_count": 0,
            "receipt_fill_candidate_router_trace_missing_count": 1,
            "receipt_fill_managed_output_entry_count": 1,
            "receipt_fill_managed_output_ready_count": 0,
            "receipt_fill_managed_output_missing_count": 1,
            "receipt_fill_dense_output_entry_count": 0,
            "receipt_fill_dense_output_ready_count": 0,
            "receipt_fill_dense_output_missing_count": 0,
            "launch_card_count": 1,
            "launch_card_task_count": 2,
            "launch_card_unbound_task_count": 2,
            "model_plane_artifact_writer_contract_request_ready": True,
            "model_plane_artifact_writer_contract_request_task_count": 2,
            "launch_card_saved_handoff_artifacts_ready": True,
            "launch_card_saved_handoff_artifact_missing_count": 0,
            "launch_card_saved_handoff_artifact_drifted_count": 0,
            "command_contract_ready": True,
            "command_contract_runtime_ready": False,
            "command_contract_planned_capture_count": 2,
            "command_contract_runtime_command_count": 0,
            "command_contract_missing_runtime_command_count": 2,
            "execution_manual_ready": True,
            "execution_automated_ready": False,
            "execution_command_option_count": 0,
            "execution_operator_command_option_count": 0,
            "execution_metadata_command_option_count": 0,
            "execution_artifacts_with_command_count": 0,
            "execution_manual_capture_required_count": 2,
            "execution_missing_capture_command_count": 2,
            "execution_missing_item_count": 0,
            "all_execution_manual_ready": True,
            "all_execution_automated_ready": False,
            "all_execution_request_count": 1,
            "all_execution_ready_request_count": 1,
            "all_execution_automated_request_count": 0,
            "all_execution_pending_artifact_count": 2,
            "all_execution_command_option_count": 0,
            "all_execution_operator_command_option_count": 0,
            "all_execution_metadata_command_option_count": 0,
            "all_execution_artifacts_with_command_count": 0,
            "all_execution_manual_capture_required_count": 2,
            "all_execution_missing_capture_command_count": 2,
            "all_execution_missing_item_count": 0,
            "runtime_capture_launch_card_dir_provided": False,
            "runtime_capture_launch_card_dir_ready": False,
            "runtime_capture_launch_card_dir_expected_card_count": 1,
            "runtime_capture_launch_card_dir_matched_card_count": 0,
            "runtime_capture_launch_card_dir_missing_card_count": 0,
            "reuse_evidence_capture_valid": True,
            "reuse_evidence_capture_bundle_count": 1,
            "reuse_evidence_capture_ready_count": 0,
            "reuse_evidence_capture_blocked_count": 1,
            "reuse_evidence_capture_trace_valid_count": 1,
            "reuse_evidence_capture_receipt_ready_count": 1,
            "reuse_evidence_capture_no_reuse_distance_count": 1,
            "reuse_evidence_capture_prompt_identity_ready_count": 0,
            "reuse_evidence_capture_prompt_identity_missing_count": 1,
            "manual_runbook_ready": True,
            "manual_runbook_request_count": 1,
            "manual_runbook_ready_request_count": 1,
            "manual_runbook_manual_task_count": 2,
            "manual_runbook_runtime_command_task_count": 0,
            "manual_runbook_validator_command_count": 2,
            "manual_runbook_missing_item_count": 0,
            "manual_runbook_missing_receipt_command_count": 0,
            "manual_runbook_missing_validator_command_count": 0,
            "manual_runbook_missing_approval_command_count": 0,
            "manual_runbook_missing_source_request_path_count": 0,
            "manual_runbook_missing_prompt_set_path_count": 0,
            "manual_runbook_missing_explicit_approval_count": 0,
            "manual_runbook_missing_prompt_traffic_ack_count": 0,
            "post_capture_intake_runbook_ready": True,
            "post_capture_intake_runbook_request_count": 1,
            "post_capture_intake_runbook_artifact_gate_count": 2,
            "post_capture_intake_runbook_ready_after_current_intake_count": 0,
            "post_capture_intake_runbook_missing_after_current_intake_count": 2,
            "post_capture_intake_runbook_validator_command_count": 2,
            "post_capture_intake_runbook_ready_to_update_bundle_count": 0,
            "post_capture_intake_runbook_phase4_candidate_count": 0,
            "post_capture_intake_runbook_live_spike_candidate_count": 0,
            "post_capture_intake_runbook_missing_source_request_path_count": 0,
            "post_capture_intake_runbook_missing_prompt_set_path_count": 0,
            "post_capture_intake_runbook_missing_explicit_approval_count": 0,
            "post_capture_intake_runbook_missing_prompt_traffic_ack_count": 0,
            "post_capture_intake_runbook_missing_item_count": 0,
            "blocker_closure_ready_scope": "mapping_ready_only",
            "blocker_closure_missing_evidence_count": 1,
            "blocker_evidence_ledger_ready": True,
            "blocker_evidence_ledger_reason_count": 1,
            "blocker_evidence_ledger_row_count": 1,
            "blocker_evidence_ledger_expected_missing_evidence_count": 1,
            "blocker_evidence_ledger_runtime_capture_required_row_count": 1,
            "blocker_evidence_ledger_future_adapter_required_row_count": 0,
            "blocker_evidence_ledger_receipt_bound_row_count": 0,
            "blocker_evidence_ledger_dense_output_row_count": 0,
            "blocker_evidence_ledger_live_proof_row_count": 0,
            "blocker_evidence_ledger_runtime_actuator_row_count": 0,
            "blocker_evidence_ledger_validator_command_count": 3,
            "blocker_evidence_ledger_missing_item_count": 0,
            "blocker_resolution_queue_ready": True,
            "blocker_resolution_queue_work_package_count": 1,
            "blocker_resolution_queue_row_count": 1,
            "blocker_resolution_queue_expected_ledger_row_count": 1,
            "blocker_resolution_queue_runtime_capture_package_count": 1,
            "blocker_resolution_queue_future_adapter_package_count": 0,
            "blocker_resolution_queue_runtime_capture_required_row_count": 1,
            "blocker_resolution_queue_future_adapter_required_row_count": 0,
            "blocker_resolution_queue_receipt_bound_row_count": 0,
            "blocker_resolution_queue_validator_command_count": 3,
            "blocker_resolution_queue_completion_gate_count": 4,
            "blocker_resolution_queue_dependency_edge_count": 0,
            "blocker_resolution_queue_missing_item_count": 0,
        },
        "remaining_blockers": ["no_replay_policy_candidate"],
        "output_files": output_files,
        "warnings": [],
        "safety_contract": ["operator handoff package is metadata only"],
    }
    reuse_capture_plan = {
        "schema_version": "moe-phase3-reuse-evidence-capture-plan-v1",
        "mode": "phase3_reuse_evidence_capture_plan",
        "valid": True,
        "errors": [],
        "root": "memory-moe-mvp/phase3-real-evidence",
        "bundle_count": 1,
        "reuse_ready_count": 0,
        "reuse_blocked_count": 1,
        "candidate_trace_valid_count": 1,
        "candidate_trace_receipt_ready_count": 1,
        "no_reuse_distance_observation_count": 1,
        "prompt_identity_ready_count": 0,
        "prompt_identity_metadata_missing_count": 1,
        "blocker_counts": {
            "no_reuse_distance_observations": 1,
            "prompt_identity_metadata_missing": 1,
        },
        "recommended_next_capture": {
            "bundle_name": "Fixture request",
            "bundle_path": "memory-moe-mvp/phase3-real-evidence/fixture.json",
            "model_id": "fixture-mixtral.gguf",
            "candidate_trace_path": candidate_path,
            "prompt_set_path": prompt_set_path,
            "runtime_capture_request_path": request_path,
            "selection_rationale": "fixture",
            "required_result": "capture_candidate_router_trace_with_reuse_distance",
        },
        "capture_contract": {
            "minimum_reuse_distance_observations": 1,
            "must_bind_prompt_identity_metadata": True,
        },
        "bundles": [],
        "safety_contract": ["reuse-evidence planner reads local metadata and JSONL traces only"],
        "next_actions": ["Recapture the recommended bundle with repeated prompt cases."],
    }
    packet = {
        "schema_version": planner.plan_phase3_evidence_packet.SUPPORTED_SCHEMA_VERSION,
        "mode": "phase3_evidence_packet",
        "valid": True,
        "packet_ready": True,
        "phase3_complete": False,
        "decision": "no_go_live_spike",
        "recommended_runtime_capture_request": {
            "request_name": "Fixture request",
            "request_path": request_path,
            "next_artifact_id": "candidate_router_trace",
        },
        "phase3_reuse_evidence_capture_plan": reuse_capture_plan,
        "recommended_policy_candidate_trace_plan": {
            "valid": True,
            "bundle_path": "memory-moe-mvp/phase3-real-evidence/fixture.json",
            "candidate_prompt_set_path": prompt_set_path,
            "candidate_trace_path": candidate_path,
            "candidate_trace_receipt_path": candidate_receipt_path,
            "prompt_set_ready": True,
            "min_reuse_distance_observations": 1,
        },
        "phase3_launch_card_library_summary": {
            "saved_handoff_artifacts_ready": True,
            "saved_handoff_artifact_missing_count": 0,
            "saved_handoff_artifact_drifted_count": 0,
        },
        "remaining_gaps": {
            "policy_candidate_ready_count": 0,
            "policy_candidate_blocked_bundle_count": 1,
            "policy_candidate_no_reuse_distance_observation_count": 1,
            "policy_candidate_prompt_identity_ready_count": 0,
            "policy_candidate_prompt_identity_metadata_missing_count": 1,
            "recommended_runtime_capture_launch_card_template_path": launch_card_path,
            "phase3_launch_card_saved_handoff_artifacts_ready": True,
            "phase3_launch_card_saved_handoff_artifact_missing_count": 0,
            "phase3_launch_card_saved_handoff_artifact_drifted_count": 0,
            "phase3_blocker_closure_reason_count": 1,
            "phase3_blocker_closure_mapped_reason_count": 1,
            "phase3_blocker_closure_unmapped_reason_count": 0,
            "phase3_blocker_closure_runtime_capture_required_count": 1,
            "phase3_blocker_closure_future_adapter_required_count": 0,
            "phase3_blocker_closure_validator_command_count": 3,
            "phase3_blocker_closure_missing_evidence_count": 1,
            "phase3_blocker_evidence_ledger_ready": True,
            "phase3_blocker_evidence_ledger_reason_count": 1,
            "phase3_blocker_evidence_ledger_row_count": 1,
            "phase3_blocker_evidence_ledger_expected_missing_evidence_count": 1,
            "phase3_blocker_evidence_ledger_runtime_capture_required_row_count": 1,
            "phase3_blocker_evidence_ledger_future_adapter_required_row_count": 0,
            "phase3_blocker_evidence_ledger_receipt_bound_row_count": 0,
            "phase3_blocker_evidence_ledger_dense_output_row_count": 0,
            "phase3_blocker_evidence_ledger_live_proof_row_count": 0,
            "phase3_blocker_evidence_ledger_runtime_actuator_row_count": 0,
            "phase3_blocker_evidence_ledger_validator_command_count": 3,
            "phase3_blocker_evidence_ledger_missing_item_count": 0,
            "phase3_blocker_resolution_queue_ready": True,
            "phase3_blocker_resolution_queue_work_package_count": 1,
            "phase3_blocker_resolution_queue_row_count": 1,
            "phase3_blocker_resolution_queue_expected_ledger_row_count": 1,
            "phase3_blocker_resolution_queue_runtime_capture_package_count": 1,
            "phase3_blocker_resolution_queue_future_adapter_package_count": 0,
            "phase3_blocker_resolution_queue_runtime_capture_required_row_count": 1,
            "phase3_blocker_resolution_queue_future_adapter_required_row_count": 0,
            "phase3_blocker_resolution_queue_receipt_bound_row_count": 0,
            "phase3_blocker_resolution_queue_validator_command_count": 3,
            "phase3_blocker_resolution_queue_completion_gate_count": 4,
            "phase3_blocker_resolution_queue_dependency_edge_count": 0,
            "phase3_blocker_resolution_queue_missing_item_count": 0,
            "all_request_downstream_handoff_manifest_ready": True,
            "all_request_downstream_handoff_manifest_request_count": 1,
            "all_request_downstream_handoff_policy_ready_count": 1,
            "all_request_downstream_handoff_dense_ready_count": 1,
            "all_request_downstream_handoff_live_ready_count": 1,
            "all_request_downstream_handoff_all_ready_count": 1,
            "recommended_runtime_capture_preflight_ready": True,
            "recommended_runtime_capture_preflight_pending_artifact_count": 2,
            "recommended_runtime_capture_preflight_receipt_entry_count": 2,
            "recommended_runtime_capture_preflight_validator_command_count": 2,
            "recommended_runtime_capture_preflight_runtime_closure_reason_count": 1,
            "recommended_runtime_capture_preflight_missing_item_count": 0,
            "recommended_runtime_capture_command_contract_ready": True,
            "recommended_runtime_capture_command_contract_runtime_ready": False,
            "recommended_runtime_capture_command_contract_planned_capture_count": 2,
            "recommended_runtime_capture_command_contract_runtime_command_count": 0,
            "recommended_runtime_capture_command_contract_missing_runtime_command_count": 2,
            "recommended_runtime_capture_execution_manual_ready": True,
            "recommended_runtime_capture_execution_automated_ready": False,
            "recommended_runtime_capture_execution_command_option_count": 0,
            "recommended_runtime_capture_execution_operator_command_option_count": 0,
            "recommended_runtime_capture_execution_metadata_command_option_count": 0,
            "recommended_runtime_capture_execution_artifacts_with_command_count": 0,
            "recommended_runtime_capture_execution_manual_capture_required_count": 2,
            "recommended_runtime_capture_execution_missing_capture_command_count": 2,
            "recommended_runtime_capture_execution_missing_item_count": 0,
            "all_request_runtime_capture_execution_manual_ready": True,
            "all_request_runtime_capture_execution_automated_ready": False,
            "all_request_runtime_capture_execution_request_count": 1,
            "all_request_runtime_capture_execution_ready_request_count": 1,
            "all_request_runtime_capture_execution_automated_request_count": 0,
            "all_request_runtime_capture_execution_pending_artifact_count": 2,
            "all_request_runtime_capture_execution_artifact_execution_count": 2,
            "all_request_runtime_capture_execution_command_option_count": 0,
            "all_request_runtime_capture_execution_operator_command_option_count": 0,
            "all_request_runtime_capture_execution_metadata_command_option_count": 0,
            "all_request_runtime_capture_execution_artifacts_with_command_count": 0,
            "all_request_runtime_capture_execution_manual_capture_required_count": 2,
            "all_request_runtime_capture_execution_missing_capture_command_count": 2,
            "all_request_runtime_capture_execution_missing_item_count": 0,
            "runtime_capture_launch_card_dir_provided": False,
            "runtime_capture_launch_card_dir_ready": False,
            "runtime_capture_launch_card_dir_expected_card_count": 1,
            "runtime_capture_launch_card_dir_matched_card_count": 0,
            "runtime_capture_launch_card_dir_missing_card_count": 0,
            "all_request_manual_capture_runbook_ready": True,
            "all_request_manual_capture_runbook_request_count": 1,
            "all_request_manual_capture_runbook_ready_request_count": 1,
            "all_request_manual_capture_runbook_manual_task_count": 2,
            "all_request_manual_capture_runbook_runtime_command_task_count": 0,
            "all_request_manual_capture_runbook_validator_command_count": 2,
            "all_request_manual_capture_runbook_missing_item_count": 0,
            "all_request_manual_capture_runbook_missing_receipt_command_count": 0,
            "all_request_manual_capture_runbook_missing_validator_command_count": 0,
            "all_request_manual_capture_runbook_missing_approval_command_count": 0,
            "all_request_manual_capture_runbook_missing_source_request_path_count": 0,
            "all_request_manual_capture_runbook_missing_prompt_set_path_count": 0,
            "all_request_manual_capture_runbook_missing_explicit_approval_count": 0,
            "all_request_manual_capture_runbook_missing_prompt_traffic_ack_count": 0,
            "all_request_post_capture_intake_runbook_ready": True,
            "all_request_post_capture_intake_runbook_request_count": 1,
            "all_request_post_capture_intake_runbook_artifact_gate_count": 2,
            "all_request_post_capture_intake_runbook_ready_after_current_intake_count": 0,
            "all_request_post_capture_intake_runbook_missing_after_current_intake_count": 2,
            "all_request_post_capture_intake_runbook_validator_command_count": 2,
            "all_request_post_capture_intake_runbook_ready_to_update_bundle_count": 0,
            "all_request_post_capture_intake_runbook_phase4_candidate_count": 0,
            "all_request_post_capture_intake_runbook_live_spike_candidate_count": 0,
            "all_request_post_capture_intake_runbook_missing_source_request_path_count": 0,
            "all_request_post_capture_intake_runbook_missing_prompt_set_path_count": 0,
            "all_request_post_capture_intake_runbook_missing_explicit_approval_count": 0,
            "all_request_post_capture_intake_runbook_missing_prompt_traffic_ack_count": 0,
            "all_request_post_capture_intake_runbook_missing_item_count": 0,
            "runtime_capture_approval_rebuild_command_manifest_count": 1,
            "approval_manifest_parity_ready": True,
            "approval_manifest_parity_matched_request_count": 1,
        },
    }
    worksheet_tasks = [
        {
            "request_path": request_path,
            "launch_card_path": launch_card_path,
            "task_id": "phase3_capture_candidate_router_trace",
            "artifact_id": "candidate_router_trace",
            "artifact_output_path": candidate_path,
            "receipt_output_path": candidate_receipt_path,
            "prompt_set_path": prompt_set_path,
            "approval_keys": ["runtime_prompt_traffic_approved", "router_trace_capture_approved"],
        },
        {
            "request_path": request_path,
            "launch_card_path": launch_card_path,
            "task_id": "phase3_capture_managed_output_summary_fill",
            "artifact_id": "managed_output_summary_fill",
            "artifact_output_path": managed_path,
            "receipt_output_path": managed_path,
            "prompt_set_path": prompt_set_path,
            "approval_keys": ["runtime_prompt_traffic_approved", "managed_output_capture_approved"],
        },
    ]
    card_tasks = [
        {
            "task_id": item["task_id"],
            "artifact_id": item["artifact_id"],
            "artifact_output_path": item["artifact_output_path"],
            "receipt_output_path": item["receipt_output_path"],
            "prompt_set_path": item["prompt_set_path"],
            "approval_keys": item["approval_keys"],
        }
        for item in worksheet_tasks
    ]
    worksheet = {
        "schema_version": planner.plan_phase3_launch_card_library.BINDING_WORKSHEET_SCHEMA_VERSION,
        "mode": "phase3_launch_card_binding_worksheet",
        "valid": True,
        "worksheet_ready": True,
        "execution_ready": False,
        "task_count": 2,
        "command_option_count": 0,
        "unbound_task_count": 2,
        "tasks": worksheet_tasks,
    }
    model_plane_contract_request = {
        "schema_version": planner.plan_phase3_launch_card_library.MODEL_PLANE_ARTIFACT_WRITER_REQUEST_SCHEMA_VERSION,
        "mode": "phase3_model_plane_artifact_writer_contract_request",
        "valid": True,
        "request_ready": True,
        "execution_ready": False,
        "task_count": 2,
        "phase3_artifact_writer_contract_count": 2,
        "phase3_artifact_writer_ready_count": 0,
        "model_plane_card_catalog_hint": "phase3_launch_card_library",
        "tasks": [
            {
                "task_id": item["task_id"],
                "request_path": item["request_path"],
                "launch_card_path": item["launch_card_path"],
                "artifact_id": item["artifact_id"],
                "artifact_output_path": item["artifact_output_path"],
                "receipt_output_path": item["receipt_output_path"],
                "prompt_set_path": item["prompt_set_path"],
                "required_phase3_artifact_writer": {
                    "artifact_id": item["artifact_id"],
                    "artifact_output_path": item["artifact_output_path"],
                    "receipt_output_path": item["receipt_output_path"],
                    "write_contract": "write artifact and receipt exactly to the requested repo-relative paths",
                    "metadata_only_until_bound": True,
                    "requires_explicit_runtime_approval": True,
                },
                "runtime_binding_template": {},
                "model_plane_card_catalog_hint": "phase3_launch_card_library",
            }
            for item in worksheet_tasks
        ],
    }
    launch_card = {
        "schema_version": "moe-phase3-runtime-capture-launch-card-v1",
        "status": "planned_only",
        "request_path": request_path,
        "prompt_set_path": prompt_set_path,
        "executable": False,
        "runtime_capture_command_ready": False,
        "task_count": 2,
        "tasks": card_tasks,
    }
    approval_rebuild_command = {
        "command": [
            "uv",
            "run",
            "--managed-python",
            "--python",
            "3.13",
            "scripts/build_phase3_runtime_capture_request.py",
            "memory-moe-mvp/phase3-real-evidence/fixture.json",
            "--output",
            request_path,
            "--json",
            "--runtime-prompt-traffic-approved",
        ],
        "command_class": "phase3_runtime_capture_request_approval_rebuild",
        "metadata_only": True,
        "records_approval_keys": ["runtime_prompt_traffic_approved"],
        "requires_explicit_user_approval": True,
        "writes_request_path": request_path,
    }
    capture_queue = [
        {
            "request_path": request_path,
            "status": "approval_required",
            "rank": 1,
            "next_artifact_id": "candidate_router_trace",
            "missing_approval_keys": ["runtime_prompt_traffic_approved"],
            "approval_rebuild_command": approval_rebuild_command,
            "capture_fill_steps": [
                {"artifact_id": "candidate_router_trace", "artifact_path": candidate_path, "receipt_path": candidate_receipt_path, "validator_command_count": 1},
                {"artifact_id": "managed_output_summary_fill", "artifact_path": managed_path, "receipt_path": managed_path, "validator_command_count": 1},
            ],
        }
    ]
    approval_command_manifest = [
        {
            "queue_rank": 1,
            "request_name": "Fixture request",
            "request_path": request_path,
            "bundle_path": "memory-moe-mvp/phase3-real-evidence/fixture.json",
            "status": "approval_required",
            "next_artifact_id": "candidate_router_trace",
            "command_class": approval_rebuild_command["command_class"],
            "writes_request_path": approval_rebuild_command["writes_request_path"],
            "records_approval_keys": approval_rebuild_command["records_approval_keys"],
            "requires_explicit_user_approval": approval_rebuild_command["requires_explicit_user_approval"],
            "metadata_only": approval_rebuild_command["metadata_only"],
            "command": approval_rebuild_command["command"],
        }
    ]
    validator_manifest = [
        {
            "request_path": request_path,
            "artifact_id": "candidate_router_trace",
            "artifact_stage": "runtime_capture",
            "status": "approval_required",
            "path": candidate_path,
            "validator_command_count": 1,
            "validator_commands": ["uv run --managed-python --python 3.13 scripts/one.py"],
        },
        {
            "request_path": request_path,
            "artifact_id": "managed_output_summary_fill",
            "artifact_stage": "runtime_capture",
            "status": "approval_required",
            "path": managed_path,
            "validator_command_count": 1,
            "validator_commands": [["uv", "run", "--managed-python", "--python", "3.13", "scripts/two.py"]],
        },
        {
            "request_path": request_path,
            "artifact_id": "live_capability_proof_fill",
            "artifact_stage": "future_adapter",
            "status": "future_adapter_required",
            "path": live_proof_path,
            "validator_command_count": 1,
            "validator_commands": ["uv run --managed-python --python 3.13 scripts/live.py"],
        },
    ]
    receipt_manifest = [
        {
            "request_path": request_path,
            "source_request_path": request_path,
            "prompt_set_path": prompt_set_path,
            "source_prompt_set_path": prompt_set_path,
            "records_approval_keys": ["runtime_prompt_traffic_approved"],
            "requires_explicit_user_approval": True,
            "approval_records_prompt_traffic": True,
            "may_send_prompt_traffic_after_approval": True,
            "artifact_id": "candidate_router_trace",
            "artifact_path": candidate_path,
            "receipt_path": candidate_receipt_path,
            "receipt_kind": "trace_capture_receipt_json",
            "receipt_ready": False,
        },
        {
            "request_path": request_path,
            "source_request_path": request_path,
            "prompt_set_path": prompt_set_path,
            "source_prompt_set_path": prompt_set_path,
            "records_approval_keys": ["runtime_prompt_traffic_approved"],
            "requires_explicit_user_approval": True,
            "approval_records_prompt_traffic": True,
            "may_send_prompt_traffic_after_approval": True,
            "artifact_id": "managed_output_summary_fill",
            "artifact_path": managed_path,
            "receipt_path": managed_path,
            "receipt_kind": "embedded_output_capture_receipt",
            "receipt_ready": False,
        },
    ]
    receipt_command_manifest = [
        {
            "request_path": request_path,
            "source_request_path": request_path,
            "prompt_set_path": prompt_set_path,
            "source_prompt_set_path": prompt_set_path,
            "records_approval_keys": ["runtime_prompt_traffic_approved"],
            "requires_explicit_user_approval": True,
            "approval_records_prompt_traffic": True,
            "may_send_prompt_traffic_after_approval": True,
            "artifact_id": "candidate_router_trace",
            "artifact_path": candidate_path,
            "receipt_path": candidate_receipt_path,
            "receipt_kind": "trace_capture_receipt_json",
            "command_manifest_key": f"{request_path}::candidate_router_trace",
            "validator_command_count": 1,
            "validator_commands": ["uv run --managed-python --python 3.13 scripts/one.py"],
        },
        {
            "request_path": request_path,
            "source_request_path": request_path,
            "prompt_set_path": prompt_set_path,
            "source_prompt_set_path": prompt_set_path,
            "records_approval_keys": ["runtime_prompt_traffic_approved"],
            "requires_explicit_user_approval": True,
            "approval_records_prompt_traffic": True,
            "may_send_prompt_traffic_after_approval": True,
            "artifact_id": "managed_output_summary_fill",
            "artifact_path": managed_path,
            "receipt_path": managed_path,
            "receipt_kind": "embedded_output_capture_receipt",
            "command_manifest_key": f"{request_path}::managed_output_summary_fill",
            "validator_command_count": 1,
            "validator_commands": ["uv run --managed-python --python 3.13 scripts/two.py"],
        },
    ]
    preflight_manifest = {
        "ready": True,
        "request_name": "Fixture request",
        "request_path": request_path,
        "status": "approval_required",
        "queue_rank": 1,
        "selection_rationale": "fixture",
        "approval_required": True,
        "missing_approval_keys": ["runtime_prompt_traffic_approved"],
        "approval_rebuild_command_present": True,
        "approval_rebuild_command": approval_rebuild_command,
        "post_approval_preview_valid": True,
        "post_approval_ready_for_operator": True,
        "post_approval_capture_complete": False,
        "post_approval_mutates_request": False,
        "capture_sequence_step_count": 3,
        "missing_capture_sequence_ids": [],
        "pending_artifact_ids": ["candidate_router_trace", "managed_output_summary_fill"],
        "pending_artifact_count": 2,
        "artifact_check_count": 2,
        "artifact_checks": [
            {
                "artifact_id": "candidate_router_trace",
                "artifact_path": candidate_path,
                "receipt_path": candidate_receipt_path,
                "receipt_kind": "trace_capture_receipt_json",
                "receipt_ready": False,
                "fill_status_after_approval": "pending_runtime_capture",
                "validator_command_count": 1,
                "validator_commands": ["uv run --managed-python --python 3.13 scripts/one.py"],
            },
            {
                "artifact_id": "managed_output_summary_fill",
                "artifact_path": managed_path,
                "receipt_path": managed_path,
                "receipt_kind": "embedded_output_capture_receipt",
                "receipt_ready": False,
                "fill_status_after_approval": "pending_runtime_capture",
                "validator_command_count": 1,
                "validator_commands": ["uv run --managed-python --python 3.13 scripts/two.py"],
            },
        ],
        "receipt_entry_count": 2,
        "receipt_ready_count": 0,
        "validator_command_count": 2,
        "runtime_closure_reason_count": 1,
        "runtime_closure_reason_ids": ["no_replay_policy_candidate"],
        "runtime_closure_validator_command_count": 3,
        "missing_preflight_item_count": 0,
        "missing_preflight_items": [],
    }
    command_contract = {
        "schema_version": planner.plan_phase3_runtime_capture_commands.SUPPORTED_SCHEMA_VERSION,
        "mode": "phase3_runtime_capture_command_contract",
        "valid": True,
        "errors": [],
        "request_path": request_path,
        "request_name": "Fixture request",
        "bundle_path": "memory-moe-mvp/phase3-real-evidence/fixture.json",
        "model_id": "fixture-model",
        "backend_family": "llama_cpp",
        "prompt_family": "fixture",
        "prompt_set_path": prompt_set_path,
        "runtime_prompt_traffic_approved": False,
        "command_contract_ready": True,
        "runtime_capture_command_ready": False,
        "planned_capture_count": 2,
        "runtime_command_option_count": 0,
        "missing_runtime_command_count": 2,
        "missing_runtime_command_artifact_ids": ["candidate_router_trace", "managed_output_summary_fill"],
        "model_plane_manifest": {"provided": False, "valid": False, "ready_for_binding": False, "blockers": ["model_plane_probe_manifest_missing"]},
        "capture_tasks": [
            {
                "artifact_id": "candidate_router_trace",
                "capture_kind": "llama_cpp_router_trace_jsonl",
                "receipt_kind": "trace_capture_receipt_json",
                "status": "pending_runtime_capture",
                "approval_required": True,
                "artifact_path": candidate_path,
                "receipt_path": candidate_receipt_path,
                "prompt_set_path": prompt_set_path,
                "approval_keys": ["runtime_prompt_traffic_approved", "router_trace_capture_approved"],
                "validator_command_count": 1,
                "command_binding_ready": False,
                "blockers": [],
            },
            {
                "artifact_id": "managed_output_summary_fill",
                "capture_kind": "managed_output_summary_json",
                "receipt_kind": "embedded_output_capture_receipt",
                "status": "pending_runtime_capture",
                "approval_required": True,
                "artifact_path": managed_path,
                "receipt_path": managed_path,
                "prompt_set_path": prompt_set_path,
                "approval_keys": ["runtime_prompt_traffic_approved", "managed_output_capture_approved"],
                "validator_command_count": 1,
                "command_binding_ready": False,
                "blockers": [],
            },
        ],
        "binding_requirements": ["Bind each task before treating it as executable."],
        "next_actions": ["Bind each planned task to a real callable or launch-card command."],
        "safety_contract": ["command-contract planner reads local metadata only"],
        "launch_card_binding_summary": {"valid": False, "binding_ready": False, "bound_task_count": 0, "command_option_count": 0, "missing_runtime_command_count": 0, "blockers": []},
    }
    packet["recommended_runtime_capture_preflight_manifest"] = preflight_manifest
    packet["recommended_runtime_capture_command_contract"] = command_contract
    execution_coverage = {
        "ready": True,
        "automated_capture_ready": False,
        "manual_operator_capture_ready": True,
        "request_name": "Fixture request",
        "request_path": request_path,
        "pending_artifact_count": 2,
        "artifact_execution_count": 2,
        "capture_command_option_count": 0,
        "operator_command_option_count": 0,
        "metadata_command_option_count": 0,
        "artifacts_with_capture_command_count": 0,
        "manual_capture_required_count": 2,
        "missing_capture_command_count": 2,
        "missing_capture_command_artifact_ids": ["candidate_router_trace", "managed_output_summary_fill"],
        "missing_execution_item_count": 0,
        "missing_execution_items": [],
        "artifact_execution": [
            {
                "artifact_id": "candidate_router_trace",
                "artifact_path": candidate_path,
                "receipt_path": candidate_receipt_path,
                "receipt_kind": "trace_capture_receipt_json",
                "current_status": "approval_required",
                "status_after_approval": "pending_runtime_capture",
                "fill_status_after_approval": "pending_runtime_capture",
                "receipt_ready": False,
                "validator_command_count": 1,
                "operator_command_option_count": 0,
                "metadata_command_option_count": 0,
                "capture_command_option_count": 0,
                "has_capture_command": False,
                "has_runtime_capture_command": False,
                "launch_card_binding_ready": False,
                "capture_mode": "manual_runtime_capture_required",
                "command_options": [],
                "blockers_after_approval": [],
            },
            {
                "artifact_id": "managed_output_summary_fill",
                "artifact_path": managed_path,
                "receipt_path": managed_path,
                "receipt_kind": "embedded_output_capture_receipt",
                "current_status": "approval_required",
                "status_after_approval": "pending_runtime_capture",
                "fill_status_after_approval": "pending_runtime_capture",
                "receipt_ready": False,
                "validator_command_count": 1,
                "operator_command_option_count": 0,
                "metadata_command_option_count": 0,
                "capture_command_option_count": 0,
                "has_capture_command": False,
                "has_runtime_capture_command": False,
                "launch_card_binding_ready": False,
                "capture_mode": "manual_runtime_capture_required",
                "command_options": [],
                "blockers_after_approval": [],
            },
        ],
        "automation_blockers": [
            "runtime_capture_command_option_missing:candidate_router_trace",
            "runtime_capture_command_option_missing:managed_output_summary_fill",
        ],
    }
    packet["recommended_runtime_capture_execution_coverage_manifest"] = execution_coverage
    all_execution_coverage = {
        "ready": True,
        "automated_capture_ready": False,
        "manual_operator_capture_ready": True,
        "request_count": 1,
        "expected_request_count": 1,
        "manual_operator_capture_ready_count": 1,
        "automated_capture_ready_count": 0,
        "pending_artifact_count": 2,
        "artifact_execution_count": 2,
        "capture_command_option_count": 0,
        "operator_command_option_count": 0,
        "metadata_command_option_count": 0,
        "artifacts_with_capture_command_count": 0,
        "manual_capture_required_count": 2,
        "missing_capture_command_count": 2,
        "missing_execution_item_count": 0,
        "requests_with_missing_execution_items": 0,
        "requests": [
            {
                "ready": True,
                "automated_capture_ready": False,
                "manual_operator_capture_ready": True,
                "request_name": "Fixture request",
                "request_path": request_path,
                "rank": 1,
                "status": "approval_required",
                "pending_artifact_count": 2,
                "artifact_execution_count": 2,
                "capture_command_option_count": 0,
                "operator_command_option_count": 0,
                "metadata_command_option_count": 0,
                "artifacts_with_capture_command_count": 0,
                "manual_capture_required_count": 2,
                "missing_capture_command_count": 2,
                "missing_capture_command_artifact_ids": ["candidate_router_trace", "managed_output_summary_fill"],
                "missing_execution_item_count": 0,
                "missing_execution_items": [],
                "artifact_execution": [
                    {**item, "request_name": "Fixture request", "request_path": request_path}
                    for item in execution_coverage["artifact_execution"]
                ],
                "automation_blockers": [
                    f"runtime_capture_command_option_missing:{request_path}::candidate_router_trace",
                    f"runtime_capture_command_option_missing:{request_path}::managed_output_summary_fill",
                ],
            }
        ],
        "automation_blockers": [
            f"runtime_capture_command_option_missing:{request_path}::candidate_router_trace",
            f"runtime_capture_command_option_missing:{request_path}::managed_output_summary_fill",
        ],
    }
    launch_card_directory = {
        "provided": False,
        "path": None,
        "valid": True,
        "directory_ready": False,
        "expected_card_count": 1,
        "matched_card_count": 0,
        "missing_card_count": 0,
        "cards": [],
        "paths_by_request": {},
        "errors": [],
        "blockers": ["runtime_capture_launch_card_dir_not_provided"],
    }
    all_execution_coverage["launch_card_directory"] = launch_card_directory
    packet["runtime_capture_launch_card_directory_manifest"] = launch_card_directory
    packet["all_request_runtime_capture_execution_coverage_manifest"] = all_execution_coverage
    manual_runbook = planner.plan_phase3_evidence_packet.all_request_manual_capture_runbook_manifest(
        capture_queue,
        receipt_command_manifest,
        all_execution_coverage,
    )
    packet["all_request_manual_capture_runbook_manifest"] = manual_runbook
    capture_intake_summary = {
        "valid": True,
        "post_approval_capture_fill_plan_manifest": [
            {
                "rank": 1,
                "request_name": "Fixture request",
                "request_path": request_path,
                "artifact_step_count": 2,
                "ready_after_approval_count": 0,
                "missing_after_approval_count": 2,
                "validator_command_count": 2,
                "ready_to_update_bundle_after_approval": False,
                "capture_fill_steps": [
                    {
                        "artifact_id": "candidate_router_trace",
                        "artifact_path": candidate_path,
                        "receipt_path": candidate_receipt_path,
                        "receipt_kind": "trace_capture_receipt_json",
                        "receipt_ready": False,
                        "ready_after_approval": False,
                        "fill_status_after_approval": "pending_runtime_capture",
                        "validator_command_count": 1,
                        "blockers_after_approval": [],
                    },
                    {
                        "artifact_id": "managed_output_summary_fill",
                        "artifact_path": managed_path,
                        "receipt_path": managed_path,
                        "receipt_kind": "embedded_output_capture_receipt",
                        "receipt_ready": False,
                        "ready_after_approval": False,
                        "fill_status_after_approval": "pending_runtime_capture",
                        "validator_command_count": 1,
                        "blockers_after_approval": [],
                    },
                ],
            }
        ],
        "approval_transition_preview_manifest": [
            {
                "rank": 1,
                "request_name": "Fixture request",
                "request_path": request_path,
                "next_step_after_approval": {"status": "ready_for_operator_capture"},
                "ready_to_update_bundle_after_approval": False,
            }
        ],
        "requests": [
            {
                "path": request_path,
                "request_path": request_path,
                "next_operator_step": {"id": "candidate_router_trace", "stage": "runtime_capture"},
                "ready_to_update_bundle": False,
            }
        ],
        "ready_to_update_bundle_count": 0,
        "phase4_candidate_ready_count": 0,
        "live_spike_candidate_ready_count": 0,
    }
    post_capture_runbook = planner.plan_phase3_evidence_packet.all_request_post_capture_intake_runbook_manifest(
        manual_runbook,
        capture_intake_summary,
    )
    packet["all_request_post_capture_intake_runbook_manifest"] = post_capture_runbook
    recommended_manual_runbook = planner.plan_phase3_evidence_packet.recommended_manual_capture_runbook_manifest(
        manual_runbook,
        packet["recommended_runtime_capture_request"],
    )
    recommended_post_capture_runbook = planner.plan_phase3_evidence_packet.recommended_post_capture_intake_runbook_manifest(
        post_capture_runbook,
        packet["recommended_runtime_capture_request"],
    )
    recommended_work_order = planner.plan_phase3_evidence_packet.recommended_runtime_capture_work_order_manifest(
        packet["recommended_runtime_capture_request"],
        preflight_manifest,
        recommended_manual_runbook,
        recommended_post_capture_runbook,
    )
    packet["recommended_manual_capture_runbook_manifest"] = recommended_manual_runbook
    packet["recommended_post_capture_intake_runbook_manifest"] = recommended_post_capture_runbook
    recommended_completion_receipt = planner.plan_phase3_evidence_packet.recommended_runtime_capture_completion_receipt_template_manifest(
        recommended_work_order
    )
    packet["recommended_runtime_capture_work_order_manifest"] = recommended_work_order
    packet["recommended_runtime_capture_completion_receipt_template_manifest"] = recommended_completion_receipt
    manifest["completion_receipt_validation"] = recommended_completion_receipt["completion_receipt_validation_step"]
    manifest["handoff_counts"].update(
        {
            "recommended_manual_runbook_ready": recommended_manual_runbook["ready"],
            "recommended_manual_runbook_request_count": recommended_manual_runbook["request_count"],
            "recommended_manual_runbook_manual_task_count": recommended_manual_runbook["manual_task_count"],
            "recommended_manual_runbook_runtime_command_task_count": recommended_manual_runbook["runtime_command_task_count"],
            "recommended_manual_runbook_validator_command_count": recommended_manual_runbook["validator_command_count"],
            "recommended_manual_runbook_missing_item_count": recommended_manual_runbook["missing_item_count"],
            "recommended_post_capture_intake_runbook_ready": recommended_post_capture_runbook["ready"],
            "recommended_post_capture_intake_runbook_request_count": recommended_post_capture_runbook["request_count"],
            "recommended_post_capture_intake_runbook_artifact_gate_count": recommended_post_capture_runbook["artifact_gate_count"],
            "recommended_post_capture_intake_runbook_ready_after_current_intake_count": recommended_post_capture_runbook["ready_after_current_intake_count"],
            "recommended_post_capture_intake_runbook_missing_after_current_intake_count": recommended_post_capture_runbook["missing_after_current_intake_count"],
            "recommended_post_capture_intake_runbook_validator_command_count": recommended_post_capture_runbook["validator_command_count"],
            "recommended_post_capture_intake_runbook_ready_to_update_bundle_count": recommended_post_capture_runbook["ready_to_update_bundle_count"],
            "recommended_post_capture_intake_runbook_missing_item_count": recommended_post_capture_runbook["missing_item_count"],
            "recommended_work_order_ready": recommended_work_order["ready"],
            "recommended_work_order_capture_step_count": recommended_work_order["capture_step_count"],
            "recommended_work_order_artifact_gate_count": recommended_work_order["artifact_gate_count"],
            "recommended_work_order_validator_command_count": recommended_work_order["validator_command_count"],
            "recommended_work_order_missing_item_count": recommended_work_order["missing_item_count"],
            "recommended_work_order_approval_command_ready": bool(recommended_work_order["approval_step"]["command"]),
            "recommended_work_order_intake_command_ready": bool(recommended_work_order["intake_step"]["command"]),
            "recommended_work_order_post_capture_sequence_ready": recommended_work_order["post_capture_sequence_step_count"] == 5,
            "recommended_work_order_post_capture_sequence_step_count": recommended_work_order["post_capture_sequence_step_count"],
            "recommended_work_order_completion_validation_before_intake": True,
            "recommended_work_order_completion_validation_command_ready": recommended_work_order["completion_receipt_validation_command_ready"],
            "recommended_completion_receipt_template_ready": recommended_completion_receipt["template_ready"],
            "recommended_completion_receipt_capture_receipt_count": recommended_completion_receipt["capture_receipt_count"],
            "recommended_completion_receipt_expected_capture_step_count": recommended_completion_receipt["expected_capture_step_count"],
            "recommended_completion_receipt_validator_command_count": recommended_completion_receipt["validator_command_count"],
            "recommended_completion_receipt_missing_item_count": recommended_completion_receipt["missing_item_count"],
            "recommended_completion_receipt_receipt_complete": recommended_completion_receipt["receipt_complete"],
            "recommended_completion_receipt_ready_for_intake": recommended_completion_receipt["ready_for_capture_result_intake"],
            "recommended_completion_receipt_validation_command_ready": True,
        }
    )
    downstream_handoff_manifest = [

        {
            "rank": 1,
            "request_name": "Fixture request",
            "request_path": request_path,
            "policy_candidate_trace_handoff_ready": True,
            "dense_fallback_capture_handoff_ready": True,
            "live_capability_proof_handoff_ready": True,
            "all_downstream_handoffs_ready": True,
            "policy_candidate_trace": {
                "prompt_set_path": prompt_set_path,
                "prompt_set_exists": True,
                "candidate_trace_path": candidate_path,
                "candidate_trace_receipt_path": candidate_receipt_path,
                "candidate_trace_receipt_template_exists": True,
                "validator_command_count": 1,
                "runtime_requires_explicit_approval": True,
                "handoff_ready": True,
            },
            "dense_fallback_capture": {
                "prompt_set_path": prompt_set_path,
                "prompt_set_exists": True,
                "managed_output_path": managed_path,
                "managed_output_template_exists": True,
                "dense_output_path": managed_path,
                "dense_output_template_exists": True,
                "fallback_artifact_path": "memory-moe-mvp/phase3-real-evidence/fixture/dense-fallback-comparison.json",
                "validator_command_count": 1,
                "runtime_requires_explicit_approval": True,
                "handoff_ready": True,
            },
            "live_capability_proof": {
                "proof_artifact_path": live_proof_path,
                "proof_template_exists": True,
                "validator_command_count": 1,
                "future_adapter_required": True,
                "handoff_ready": True,
            },
        }
    ]

    blocker_manifest = [        {
            "rank": 1,
            "reason_id": "no_replay_policy_candidate",
            "reason_summary": "fixture",
            "mapped": True,
            "closure_gate": "policy_candidate_trace_capture",
            "supporting_gate": "approved_runtime_capture",
            "closure_class": "policy_candidate_replay",
            "closure_status": "approval_required",
            "closure_state": "runtime_capture_required",
            "approval_stage": "runtime_capture_required",
            "handoff_ready": True,
            "primary_path": candidate_path,
            "secondary_path": prompt_set_path,
            "receipt_path": candidate_receipt_path,
            "pending_artifact_count": 1,
            "ready_evidence_count": 0,
            "missing_evidence_count": 1,
            "validator_command_count": 3,
            "next_action": "Approve the fixture capture and fill the trace receipt.",
        }
    ]


    blocker_ledger = planner.plan_phase3_evidence_packet.phase3_blocker_evidence_ledger_manifest(
        blocker_manifest,
        post_capture_runbook,
        {},
        {},
    )
    blocker_resolution_queue = planner.plan_phase3_evidence_packet.phase3_blocker_resolution_queue_manifest(blocker_ledger)
    blocker_resolution_queue_summary = planner.plan_phase3_evidence_packet.phase3_blocker_resolution_queue_summary(blocker_resolution_queue)
    recommended_work_order_summary = planner.plan_phase3_evidence_packet.recommended_runtime_capture_work_order_summary(recommended_work_order)
    recommended_completion_receipt_summary = planner.plan_phase3_evidence_packet.recommended_runtime_capture_completion_receipt_template_summary(recommended_completion_receipt)
    next_unblocked_handoff = planner.plan_phase3_evidence_packet.next_unblocked_operator_handoff_summary(
        blocker_resolution_queue_summary,
        blocker_resolution_queue,
        recommended_work_order_summary,
        recommended_work_order,
        recommended_completion_receipt_summary,
        recommended_completion_receipt,
    )
    packet["phase3_blocker_evidence_ledger_manifest"] = blocker_ledger
    packet["phase3_blocker_resolution_queue_summary"] = blocker_resolution_queue_summary
    packet["phase3_blocker_resolution_queue_manifest"] = blocker_resolution_queue
    packet["recommended_runtime_capture_work_order_summary"] = recommended_work_order_summary
    packet["recommended_runtime_capture_completion_receipt_template_summary"] = recommended_completion_receipt_summary
    packet["phase3_next_unblocked_operator_handoff_summary"] = next_unblocked_handoff
    packet["remaining_gaps"].update(
        {
            "phase3_blocker_resolution_queue_next_unblocked_work_package_id": blocker_resolution_queue_summary["next_unblocked_work_package_id"],
            "phase3_blocker_resolution_queue_next_unblocked_sequence_rank": blocker_resolution_queue_summary["next_unblocked_sequence_rank"],
            "phase3_blocker_resolution_queue_next_unblocked_operator_stage": blocker_resolution_queue_summary["next_unblocked_operator_stage"],
            "phase3_blocker_resolution_queue_next_unblocked_package_class": blocker_resolution_queue_summary["next_unblocked_package_class"],
            "phase3_blocker_resolution_queue_next_unblocked_row_count": blocker_resolution_queue_summary["next_unblocked_row_count"],
            "phase3_blocker_resolution_queue_next_unblocked_validator_command_count": blocker_resolution_queue_summary["next_unblocked_validator_command_count"],
            "phase3_blocker_resolution_queue_next_unblocked_completion_gate_count": blocker_resolution_queue_summary["next_unblocked_completion_gate_count"],
            "phase3_blocker_resolution_queue_next_unblocked_next_action": blocker_resolution_queue_summary["next_unblocked_next_action"],
            "phase3_next_unblocked_operator_handoff_ready": next_unblocked_handoff["handoff_ready"],
            "phase3_next_unblocked_operator_handoff_work_order_ready": next_unblocked_handoff["work_order_ready"],
            "phase3_next_unblocked_operator_handoff_work_order_artifact": next_unblocked_handoff["work_order_artifact"],
            "phase3_next_unblocked_operator_handoff_receipt_template_ready": next_unblocked_handoff["completion_receipt_template_ready"],
            "phase3_next_unblocked_operator_handoff_validation_command_ready": next_unblocked_handoff["completion_receipt_validation_command_ready"],
            "phase3_next_unblocked_operator_handoff_work_order_advances_next_package": next_unblocked_handoff["work_order_advances_next_package"],
        }
    )
    manifest["next_unblocked_operator_handoff"] = next_unblocked_handoff
    manifest["handoff_counts"].update(
        {
            "blocker_resolution_queue_next_unblocked_work_package_id": blocker_resolution_queue_summary["next_unblocked_work_package_id"],
            "blocker_resolution_queue_next_unblocked_sequence_rank": blocker_resolution_queue_summary["next_unblocked_sequence_rank"],
            "blocker_resolution_queue_next_unblocked_operator_stage": blocker_resolution_queue_summary["next_unblocked_operator_stage"],
            "blocker_resolution_queue_next_unblocked_package_class": blocker_resolution_queue_summary["next_unblocked_package_class"],
            "blocker_resolution_queue_next_unblocked_row_count": blocker_resolution_queue_summary["next_unblocked_row_count"],
            "blocker_resolution_queue_next_unblocked_validator_command_count": blocker_resolution_queue_summary["next_unblocked_validator_command_count"],
            "blocker_resolution_queue_next_unblocked_completion_gate_count": blocker_resolution_queue_summary["next_unblocked_completion_gate_count"],
            "blocker_resolution_queue_next_unblocked_next_action": blocker_resolution_queue_summary["next_unblocked_next_action"],
            "next_unblocked_operator_handoff_ready": next_unblocked_handoff["handoff_ready"],
            "next_unblocked_operator_handoff_work_order_ready": next_unblocked_handoff["work_order_ready"],
            "next_unblocked_operator_handoff_work_order_artifact": next_unblocked_handoff["work_order_artifact"],
            "next_unblocked_operator_handoff_work_order_next_artifact_id": next_unblocked_handoff["work_order_next_artifact_id"],
            "next_unblocked_operator_handoff_work_order_capture_step_count": next_unblocked_handoff["work_order_capture_step_count"],
            "next_unblocked_operator_handoff_work_order_validator_command_count": next_unblocked_handoff["work_order_validator_command_count"],
            "next_unblocked_operator_handoff_receipt_template_artifact": next_unblocked_handoff["completion_receipt_template_artifact"],
            "next_unblocked_operator_handoff_receipt_template_ready": next_unblocked_handoff["completion_receipt_template_ready"],
            "next_unblocked_operator_handoff_validation_command_ready": next_unblocked_handoff["completion_receipt_validation_command_ready"],
            "next_unblocked_operator_handoff_work_order_advances_next_package": next_unblocked_handoff["work_order_advances_next_package"],
        }
    )
    (root / "README.md").write_text(
        "# Phase 3 Operator Handoff\n\n"
        "- Metadata only: `True`\n\n"
        "## Reuse Evidence Capture\n\n"
        f"- Bundle: `Fixture request`\n"
        f"- Bundle path: `memory-moe-mvp/phase3-real-evidence/fixture.json`\n"
        f"- Candidate trace: `{candidate_path}`\n"
        f"- Prompt set: `{prompt_set_path}`\n"
        f"- Runtime request: `{request_path}`\n"
        f"- Required result: `capture_candidate_router_trace_with_reuse_distance`\n"
        f"- Minimum reuse-distance observations: `1`\n"
        f"- Prompt identity metadata required: `True`\n\n"
        "## Next Unblocked Handoff\n\n"
        f"- Ready: `{next_unblocked_handoff['handoff_ready']}`\n"
        f"- Package: `{next_unblocked_handoff['next_unblocked_work_package_id']}`\n"
        f"- Work order: `{next_unblocked_handoff['work_order_artifact']}`\n\n"
        "## Completion Receipt Validation\n\n"
        "```text\n"
        + " ".join(recommended_completion_receipt["completion_receipt_validation_step"]["command"])
        + "\n```\n",
        encoding="utf-8",
    )
    (root / "phase3-evidence-packet.md").write_text("# packet\n", encoding="utf-8")
    write_json(root / "operator-handoff-manifest.json", manifest)
    write_json(root / "phase3-evidence-packet.json", packet)
    write_json(root / "launch-card-binding-worksheet.json", worksheet)
    write_json(root / "model-plane-artifact-writer-contract-request.json", model_plane_contract_request)
    write_json(root / "recommended-runtime-capture-launch-card.template.json", launch_card)
    write_json(root / "recommended-runtime-capture-preflight.json", preflight_manifest)
    write_json(root / "recommended-runtime-capture-command-contract.json", command_contract)
    write_json(root / "recommended-runtime-capture-execution-coverage.json", execution_coverage)
    write_json(root / "all-request-runtime-capture-execution-coverage.json", all_execution_coverage)
    write_json(root / "runtime-capture-launch-card-directory.json", launch_card_directory)
    write_json(root / "reuse-evidence-capture-plan.json", reuse_capture_plan)
    write_json(root / "manual-capture-runbook.json", manual_runbook)
    write_json(root / "post-capture-intake-runbook.json", post_capture_runbook)
    write_json(root / "recommended-manual-capture-runbook.json", recommended_manual_runbook)
    write_json(root / "recommended-post-capture-intake-runbook.json", recommended_post_capture_runbook)
    write_json(root / "recommended-runtime-capture-work-order.json", recommended_work_order)
    write_json(root / "recommended-runtime-capture-completion-receipt.template.json", recommended_completion_receipt)
    write_json(root / "next-unblocked-operator-handoff.json", next_unblocked_handoff)
    write_json(root / "capture-queue.json", capture_queue)
    write_json(root / "approval-command-manifest.json", approval_command_manifest)
    write_json(root / "validator-command-manifest.json", validator_manifest)
    write_json(root / "receipt-fill-manifest.json", receipt_manifest)
    write_json(root / "receipt-fill-command-manifest.json", receipt_command_manifest)
    write_json(root / "downstream-handoff-manifest.json", downstream_handoff_manifest)
    write_json(root / "blocker-closure-manifest.json", blocker_manifest)
    write_json(root / "blocker-evidence-ledger.json", blocker_ledger)
    write_json(root / "blocker-resolution-queue.json", blocker_resolution_queue)


def runtime_actuator_proof_handoff_fixture() -> dict:
    spike_planner = planner.plan_phase3_evidence_packet.plan_phase3_runtime_actuator_spike
    spike_summary = spike_planner.build_summary(
        spike_planner.load_plan(spike_planner.DEFAULT_PLAN_PATH),
        spike_planner.DEFAULT_PLAN_PATH,
        backend_family=spike_planner.DEFAULT_BACKEND_FAMILY,
    )
    return planner.plan_phase3_evidence_packet.runtime_actuator_spike_package_handoff(spike_summary)


def add_runtime_actuator_package(root: Path) -> None:
    proof_handoff = runtime_actuator_proof_handoff_fixture()
    queue_path = root / "blocker-resolution-queue.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    runtime_package = {
        "sequence_rank": 4,
        "work_package_id": "runtime_actuator_spike",
        "operator_stage": "phase4_adapter_spike_handoff",
        "package_class": "runtime_actuator",
        "summary": "Fixture runtime actuator proof handoff.",
        "next_action": "Use the proof handoff before any live managed expert loading claim.",
        "depends_on_work_package_ids": ["policy_candidate_trace_capture"],
        "dependency_count": 1,
        "completion_gates": [
            "runtime actuator spike handoff validates for the selected backend",
            "proof requirements cover every live-actuator capability",
            "residency observation proof exists before residency-control mutation",
            "dense fallback and artifact export proofs exist before residency-control mutation",
            "cleanup/restore proof exists before live managed-loading readiness",
        ],
        "completion_gate_count": 5,
        "reason_ids": ["live_actuator_residency_control_not_ready"],
        "evidence_kinds": ["runtime_actuator_capability"],
        "row_count": 0,
        "runtime_capture_required_row_count": 0,
        "future_adapter_required_row_count": 0,
        "receipt_bound_row_count": 0,
        "row_validator_command_count": 0,
        "reason_validator_command_count": 0,
        "validator_command_count": 0,
        "artifact_paths": [],
        "receipt_paths": [],
        "prompt_ids": [],
        "blocker_ids": ["live_actuator_residency_control_not_ready"],
        "proof_handoff": proof_handoff,
    }
    queue["work_packages"].append(runtime_package)
    packages = queue["work_packages"]
    queue["work_package_count"] = len(packages)
    queue["queue_row_count"] = sum(item.get("row_count", 0) for item in packages)
    queue["runtime_capture_package_count"] = sum(
        1
        for item in packages
        if item.get("operator_stage") in {"approved_runtime_capture", "post_capture_receipt_intake", "fallback_quality_bounds"}
    )
    queue["future_adapter_package_count"] = sum(
        1 for item in packages if item.get("operator_stage") in {"future_adapter_proof", "phase4_adapter_design"}
    )
    queue["runtime_capture_required_row_count"] = sum(item.get("runtime_capture_required_row_count", 0) for item in packages)
    queue["future_adapter_required_row_count"] = sum(item.get("future_adapter_required_row_count", 0) for item in packages)
    queue["receipt_bound_row_count"] = sum(item.get("receipt_bound_row_count", 0) for item in packages)
    queue["validator_command_count"] = sum(item.get("validator_command_count", 0) for item in packages)
    queue["completion_gate_count"] = sum(item.get("completion_gate_count", 0) for item in packages)
    queue["dependency_edge_count"] = sum(item.get("dependency_count", 0) for item in packages)
    write_json(queue_path, queue)

    packet_path = root / "phase3-evidence-packet.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["phase3_blocker_resolution_queue_manifest"] = queue
    gaps = packet["remaining_gaps"]
    for queue_key, gap_key in {
        "work_package_count": "phase3_blocker_resolution_queue_work_package_count",
        "queue_row_count": "phase3_blocker_resolution_queue_row_count",
        "expected_ledger_row_count": "phase3_blocker_resolution_queue_expected_ledger_row_count",
        "runtime_capture_package_count": "phase3_blocker_resolution_queue_runtime_capture_package_count",
        "future_adapter_package_count": "phase3_blocker_resolution_queue_future_adapter_package_count",
        "runtime_capture_required_row_count": "phase3_blocker_resolution_queue_runtime_capture_required_row_count",
        "future_adapter_required_row_count": "phase3_blocker_resolution_queue_future_adapter_required_row_count",
        "receipt_bound_row_count": "phase3_blocker_resolution_queue_receipt_bound_row_count",
        "validator_command_count": "phase3_blocker_resolution_queue_validator_command_count",
        "completion_gate_count": "phase3_blocker_resolution_queue_completion_gate_count",
        "dependency_edge_count": "phase3_blocker_resolution_queue_dependency_edge_count",
        "missing_item_count": "phase3_blocker_resolution_queue_missing_item_count",
    }.items():
        gaps[gap_key] = queue[queue_key]
    gaps["runtime_actuator_spike_handoff_ready"] = proof_handoff["handoff_ready"]
    gaps["runtime_actuator_spike_live_ready"] = proof_handoff["live_spike_ready"]
    gaps["runtime_actuator_spike_proof_requirement_count"] = proof_handoff["proof_requirement_count"]
    gaps["runtime_actuator_spike_proof_artifact_count"] = proof_handoff["proof_artifact_count"]
    gaps["runtime_actuator_spike_dependency_edge_count"] = proof_handoff["dependency_edge_count"]
    gaps["runtime_actuator_spike_blocking_capability_count"] = proof_handoff["blocking_capability_count"]
    write_json(packet_path, packet)

    manifest_path = root / "operator-handoff-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    handoff_counts = manifest["handoff_counts"]
    for queue_key, handoff_key in {
        "work_package_count": "blocker_resolution_queue_work_package_count",
        "queue_row_count": "blocker_resolution_queue_row_count",
        "expected_ledger_row_count": "blocker_resolution_queue_expected_ledger_row_count",
        "runtime_capture_package_count": "blocker_resolution_queue_runtime_capture_package_count",
        "future_adapter_package_count": "blocker_resolution_queue_future_adapter_package_count",
        "runtime_capture_required_row_count": "blocker_resolution_queue_runtime_capture_required_row_count",
        "future_adapter_required_row_count": "blocker_resolution_queue_future_adapter_required_row_count",
        "receipt_bound_row_count": "blocker_resolution_queue_receipt_bound_row_count",
        "validator_command_count": "blocker_resolution_queue_validator_command_count",
        "completion_gate_count": "blocker_resolution_queue_completion_gate_count",
        "dependency_edge_count": "blocker_resolution_queue_dependency_edge_count",
        "missing_item_count": "blocker_resolution_queue_missing_item_count",
    }.items():
        handoff_counts[handoff_key] = queue[queue_key]
    handoff_counts["blocker_resolution_queue_runtime_actuator_proof_handoff_ready"] = proof_handoff["handoff_ready"]
    handoff_counts["blocker_resolution_queue_runtime_actuator_live_spike_ready"] = proof_handoff["live_spike_ready"]
    handoff_counts["blocker_resolution_queue_runtime_actuator_proof_requirement_count"] = proof_handoff[
        "proof_requirement_count"
    ]
    handoff_counts["blocker_resolution_queue_runtime_actuator_proof_requirement_id_count"] = len(
        proof_handoff["proof_requirement_ids"]
    )
    handoff_counts["blocker_resolution_queue_runtime_actuator_proof_artifact_count"] = proof_handoff["proof_artifact_count"]
    handoff_counts["blocker_resolution_queue_runtime_actuator_dependency_edge_count"] = proof_handoff["dependency_edge_count"]
    handoff_counts["blocker_resolution_queue_runtime_actuator_blocking_capability_count"] = proof_handoff[
        "blocking_capability_count"
    ]
    handoff_counts["blocker_resolution_queue_runtime_actuator_control_blocker_count"] = proof_handoff["control_blocker_count"]
    write_json(manifest_path, manifest)


class Phase3OperatorHandoffPlannerTests(unittest.TestCase):
    def test_valid_minimal_handoff_package(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            summary = planner.build_summary(root)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertTrue(summary["handoff_package_ready"])
        self.assertTrue(summary["metadata_only"])
        self.assertFalse(summary["phase3_complete"])
        self.assertEqual(summary["capture_queue_request_count"], 1)
        self.assertTrue(summary["runtime_capture_preflight_ready"])
        self.assertEqual(summary["runtime_capture_preflight_pending_artifact_count"], 2)
        self.assertEqual(summary["runtime_capture_preflight_validator_command_count"], 2)
        self.assertEqual(summary["runtime_capture_preflight_missing_item_count"], 0)
        self.assertEqual(summary["runtime_capture_preflight_error_count"], 0)
        self.assertTrue(summary["runtime_capture_command_contract_ready"])
        self.assertFalse(summary["runtime_capture_command_contract_runtime_ready"])
        self.assertEqual(summary["runtime_capture_command_contract_planned_capture_count"], 2)
        self.assertEqual(summary["runtime_capture_command_contract_missing_runtime_command_count"], 2)
        self.assertEqual(summary["runtime_capture_command_contract_error_count"], 0)
        self.assertTrue(summary["runtime_capture_execution_manual_ready"])
        self.assertFalse(summary["runtime_capture_execution_automated_ready"])
        self.assertEqual(summary["runtime_capture_execution_pending_artifact_count"], 2)
        self.assertEqual(summary["runtime_capture_execution_command_option_count"], 0)
        self.assertEqual(summary["runtime_capture_execution_manual_capture_required_count"], 2)
        self.assertEqual(summary["runtime_capture_execution_missing_capture_command_count"], 2)
        self.assertEqual(summary["runtime_capture_execution_missing_item_count"], 0)
        self.assertEqual(summary["runtime_capture_execution_error_count"], 0)
        self.assertTrue(summary["all_request_runtime_capture_execution_manual_ready"])
        self.assertFalse(summary["all_request_runtime_capture_execution_automated_ready"])
        self.assertEqual(summary["all_request_runtime_capture_execution_request_count"], 1)
        self.assertEqual(summary["all_request_runtime_capture_execution_ready_request_count"], 1)
        self.assertEqual(summary["all_request_runtime_capture_execution_pending_artifact_count"], 2)
        self.assertEqual(summary["all_request_runtime_capture_execution_command_option_count"], 0)
        self.assertEqual(summary["all_request_runtime_capture_execution_manual_capture_required_count"], 2)
        self.assertEqual(summary["all_request_runtime_capture_execution_missing_capture_command_count"], 2)
        self.assertEqual(summary["all_request_runtime_capture_execution_missing_item_count"], 0)
        self.assertEqual(summary["all_request_runtime_capture_execution_error_count"], 0)
        self.assertFalse(summary["runtime_capture_launch_card_dir_provided"])
        self.assertFalse(summary["runtime_capture_launch_card_dir_ready"])
        self.assertEqual(summary["runtime_capture_launch_card_dir_expected_card_count"], 1)
        self.assertEqual(summary["runtime_capture_launch_card_dir_matched_card_count"], 0)
        self.assertEqual(summary["runtime_capture_launch_card_dir_missing_card_count"], 0)
        self.assertEqual(summary["runtime_capture_launch_card_dir_error_count"], 0)
        self.assertTrue(summary["reuse_evidence_capture_valid"])
        self.assertEqual(summary["reuse_evidence_capture_bundle_count"], 1)
        self.assertEqual(summary["reuse_evidence_capture_ready_count"], 0)
        self.assertEqual(summary["reuse_evidence_capture_blocked_count"], 1)
        self.assertEqual(summary["reuse_evidence_capture_trace_valid_count"], 1)
        self.assertEqual(summary["reuse_evidence_capture_receipt_ready_count"], 1)
        self.assertEqual(summary["reuse_evidence_capture_no_reuse_distance_count"], 1)
        self.assertEqual(summary["reuse_evidence_capture_prompt_identity_ready_count"], 0)
        self.assertEqual(summary["reuse_evidence_capture_prompt_identity_missing_count"], 1)
        self.assertTrue(summary["reuse_evidence_capture_has_recommended_capture"])
        self.assertTrue(summary["reuse_evidence_capture_manifest_recommendation_ready"])
        self.assertEqual(summary["reuse_evidence_capture_error_count"], 0)
        self.assertTrue(summary["manual_capture_runbook_ready"])
        self.assertEqual(summary["manual_capture_runbook_request_count"], 1)
        self.assertEqual(summary["manual_capture_runbook_ready_request_count"], 1)
        self.assertEqual(summary["manual_capture_runbook_manual_task_count"], 2)
        self.assertEqual(summary["manual_capture_runbook_runtime_command_task_count"], 0)
        self.assertEqual(summary["manual_capture_runbook_validator_command_count"], 2)
        self.assertEqual(summary["manual_capture_runbook_missing_item_count"], 0)
        self.assertEqual(summary["manual_capture_runbook_missing_receipt_command_count"], 0)
        self.assertEqual(summary["manual_capture_runbook_missing_validator_command_count"], 0)
        self.assertEqual(summary["manual_capture_runbook_missing_approval_command_count"], 0)
        self.assertEqual(summary["manual_capture_runbook_missing_source_request_path_count"], 0)
        self.assertEqual(summary["manual_capture_runbook_missing_prompt_set_path_count"], 0)
        self.assertEqual(summary["manual_capture_runbook_missing_explicit_approval_count"], 0)
        self.assertEqual(summary["manual_capture_runbook_missing_prompt_traffic_ack_count"], 0)
        self.assertEqual(summary["manual_capture_runbook_error_count"], 0)
        self.assertTrue(summary["post_capture_intake_runbook_ready"])
        self.assertEqual(summary["post_capture_intake_runbook_request_count"], 1)
        self.assertEqual(summary["post_capture_intake_runbook_artifact_gate_count"], 2)
        self.assertEqual(summary["post_capture_intake_runbook_ready_after_current_intake_count"], 0)
        self.assertEqual(summary["post_capture_intake_runbook_missing_after_current_intake_count"], 2)
        self.assertEqual(summary["post_capture_intake_runbook_validator_command_count"], 2)
        self.assertEqual(summary["post_capture_intake_runbook_ready_to_update_bundle_count"], 0)
        self.assertEqual(summary["post_capture_intake_runbook_phase4_candidate_count"], 0)
        self.assertEqual(summary["post_capture_intake_runbook_live_spike_candidate_count"], 0)
        self.assertEqual(summary["post_capture_intake_runbook_missing_source_request_path_count"], 0)
        self.assertEqual(summary["post_capture_intake_runbook_missing_prompt_set_path_count"], 0)
        self.assertEqual(summary["post_capture_intake_runbook_missing_explicit_approval_count"], 0)
        self.assertEqual(summary["post_capture_intake_runbook_missing_prompt_traffic_ack_count"], 0)
        self.assertEqual(summary["post_capture_intake_runbook_missing_item_count"], 0)
        self.assertEqual(summary["post_capture_intake_runbook_error_count"], 0)
        self.assertTrue(summary["recommended_manual_capture_runbook_ready"])
        self.assertEqual(summary["recommended_manual_capture_runbook_request_count"], 1)
        self.assertEqual(summary["recommended_manual_capture_runbook_manual_task_count"], 2)
        self.assertEqual(summary["recommended_manual_capture_runbook_runtime_command_task_count"], 0)
        self.assertEqual(summary["recommended_manual_capture_runbook_validator_command_count"], 2)
        self.assertEqual(summary["recommended_manual_capture_runbook_missing_item_count"], 0)
        self.assertEqual(summary["recommended_manual_capture_runbook_error_count"], 0)
        self.assertTrue(summary["recommended_post_capture_intake_runbook_ready"])
        self.assertEqual(summary["recommended_post_capture_intake_runbook_request_count"], 1)
        self.assertEqual(summary["recommended_post_capture_intake_runbook_artifact_gate_count"], 2)
        self.assertEqual(summary["recommended_post_capture_intake_runbook_ready_after_current_intake_count"], 0)
        self.assertEqual(summary["recommended_post_capture_intake_runbook_missing_after_current_intake_count"], 2)
        self.assertEqual(summary["recommended_post_capture_intake_runbook_validator_command_count"], 2)
        self.assertEqual(summary["recommended_post_capture_intake_runbook_ready_to_update_bundle_count"], 0)
        self.assertEqual(summary["recommended_post_capture_intake_runbook_missing_item_count"], 0)
        self.assertEqual(summary["recommended_post_capture_intake_runbook_error_count"], 0)
        self.assertTrue(summary["recommended_runtime_capture_work_order_ready"])
        self.assertEqual(summary["recommended_runtime_capture_work_order_capture_step_count"], 2)
        self.assertEqual(summary["recommended_runtime_capture_work_order_artifact_gate_count"], 2)
        self.assertEqual(summary["recommended_runtime_capture_work_order_validator_command_count"], 2)
        self.assertEqual(summary["recommended_runtime_capture_work_order_missing_item_count"], 0)
        self.assertTrue(summary["recommended_runtime_capture_work_order_approval_command_ready"])
        self.assertTrue(summary["recommended_runtime_capture_work_order_intake_command_ready"])
        self.assertTrue(summary["recommended_runtime_capture_work_order_post_capture_sequence_ready"])
        self.assertEqual(summary["recommended_runtime_capture_work_order_post_capture_sequence_step_count"], 5)
        self.assertTrue(summary["recommended_runtime_capture_work_order_completion_validation_before_intake"])
        self.assertTrue(summary["recommended_runtime_capture_work_order_completion_validation_command_ready"])
        self.assertEqual(summary["recommended_runtime_capture_work_order_error_count"], 0)
        self.assertTrue(summary["recommended_runtime_capture_completion_receipt_template_ready"])
        self.assertFalse(summary["recommended_runtime_capture_completion_receipt_complete"])
        self.assertFalse(summary["recommended_runtime_capture_completion_receipt_ready_for_intake"])
        self.assertEqual(summary["recommended_runtime_capture_completion_receipt_count"], 2)
        self.assertEqual(summary["recommended_runtime_capture_completion_receipt_expected_capture_step_count"], 2)
        self.assertEqual(summary["recommended_runtime_capture_completion_receipt_validator_command_count"], 2)
        self.assertEqual(summary["recommended_runtime_capture_completion_receipt_missing_item_count"], 0)
        self.assertTrue(summary["recommended_runtime_capture_completion_receipt_validation_command_ready"])
        self.assertEqual(summary["recommended_runtime_capture_completion_receipt_error_count"], 0)
        self.assertTrue(summary["next_unblocked_operator_handoff_ready"])
        self.assertEqual(summary["next_unblocked_operator_handoff_package_id"], "policy_candidate_trace_capture")
        self.assertTrue(summary["next_unblocked_operator_handoff_work_order_ready"])
        self.assertTrue(summary["next_unblocked_operator_handoff_work_order_advances_next_package"])
        self.assertEqual(summary["next_unblocked_operator_handoff_work_order_capture_step_count"], 2)
        self.assertEqual(summary["next_unblocked_operator_handoff_work_order_validator_command_count"], 2)
        self.assertTrue(summary["next_unblocked_operator_handoff_receipt_template_ready"])
        self.assertTrue(summary["next_unblocked_operator_handoff_validation_command_ready"])
        self.assertTrue(summary["next_unblocked_operator_handoff_bound_to_receipt"])
        self.assertEqual(summary["next_unblocked_operator_handoff_row_count"], 1)
        self.assertEqual(summary["next_unblocked_operator_handoff_validator_command_count"], 3)
        self.assertEqual(summary["next_unblocked_operator_handoff_error_count"], 0)
        self.assertEqual(summary["validator_command_count"], 3)
        self.assertEqual(summary["validator_manifest_runtime_artifact_count"], 2)
        self.assertEqual(summary["validator_manifest_future_artifact_count"], 1)
        self.assertEqual(summary["validator_manifest_runtime_validator_command_count"], 2)
        self.assertEqual(summary["validator_manifest_future_validator_command_count"], 1)
        self.assertTrue(summary["validator_manifest_coverage_ready"])
        self.assertEqual(summary["receipt_fill_missing_count"], 2)
        self.assertEqual(summary["receipt_fill_candidate_router_trace_entry_count"], 1)
        self.assertEqual(summary["receipt_fill_candidate_router_trace_ready_count"], 0)
        self.assertEqual(summary["receipt_fill_candidate_router_trace_missing_count"], 1)
        self.assertEqual(summary["receipt_fill_managed_output_entry_count"], 1)
        self.assertEqual(summary["receipt_fill_managed_output_ready_count"], 0)
        self.assertEqual(summary["receipt_fill_managed_output_missing_count"], 1)
        self.assertEqual(summary["receipt_fill_dense_output_entry_count"], 0)
        self.assertEqual(summary["receipt_fill_dense_output_ready_count"], 0)
        self.assertEqual(summary["receipt_fill_dense_output_missing_count"], 0)
        self.assertEqual(summary["receipt_fill_key_count"], 2)
        self.assertEqual(summary["receipt_fill_command_key_count"], 2)
        self.assertEqual(summary["capture_queue_artifact_key_count"], 2)
        self.assertEqual(summary["receipt_fill_command_validator_command_count"], 2)
        self.assertEqual(summary["capture_queue_validator_command_count"], 2)
        self.assertEqual(summary["receipt_fill_command_missing_validator_count"], 0)
        self.assertEqual(summary["receipt_fill_command_declared_count_mismatch_count"], 0)
        self.assertTrue(summary["receipt_fill_command_key_parity_ready"])
        self.assertTrue(summary["receipt_fill_capture_queue_key_parity_ready"])
        self.assertTrue(summary["receipt_fill_command_validator_coverage_ready"])
        self.assertEqual(summary["downstream_handoff_request_count"], 1)
        self.assertEqual(summary["downstream_handoff_policy_ready_count"], 1)
        self.assertEqual(summary["downstream_handoff_dense_ready_count"], 1)
        self.assertEqual(summary["downstream_handoff_live_ready_count"], 1)
        self.assertEqual(summary["downstream_handoff_all_ready_count"], 1)
        self.assertEqual(summary["downstream_handoff_missing_section_count"], 0)
        self.assertEqual(summary["downstream_handoff_missing_path_count"], 0)
        self.assertEqual(summary["downstream_handoff_missing_validator_count"], 0)
        self.assertTrue(summary["downstream_handoff_manifest_coverage_ready"])
        self.assertTrue(summary["recommended_approval_command_ready"])
        self.assertEqual(summary["all_request_approval_command_count"], 1)
        self.assertEqual(summary["all_request_approval_command_ready_count"], 1)
        self.assertTrue(summary["all_request_approval_commands_ready"])
        self.assertEqual(summary["all_request_approval_command_error_count"], 0)
        self.assertEqual(summary["approval_command_manifest_count"], 1)
        self.assertEqual(summary["approval_command_manifest_ready_count"], 1)
        self.assertTrue(summary["approval_command_manifest_coverage_ready"])
        self.assertEqual(summary["approval_command_manifest_error_count"], 0)
        self.assertTrue(summary["recommended_capture_queue_ready"])
        self.assertEqual(summary["launch_card_task_key_count"], 2)
        self.assertEqual(summary["worksheet_recommended_task_key_count"], 2)
        self.assertEqual(summary["capture_queue_recommended_task_key_count"], 2)
        self.assertEqual(summary["binding_worksheet_queue_task_key_count"], 2)
        self.assertEqual(summary["capture_queue_task_key_count"], 2)
        self.assertTrue(summary["launch_card_worksheet_task_parity_ready"])
        self.assertTrue(summary["launch_card_capture_queue_task_parity_ready"])
        self.assertTrue(summary["binding_worksheet_capture_queue_task_parity_ready"])
        self.assertEqual(summary["model_plane_contract_request_task_key_count"], 2)
        self.assertTrue(summary["model_plane_contract_request_parity_ready"])
        self.assertTrue(summary["launch_card_saved_handoff_artifacts_ready"])
        self.assertEqual(summary["launch_card_saved_handoff_artifact_missing_count"], 0)
        self.assertEqual(summary["launch_card_saved_handoff_artifact_drifted_count"], 0)
        self.assertTrue(summary["launch_card_saved_handoff_parity_ready"])
        self.assertEqual(summary["launch_card_saved_handoff_error_count"], 0)
        self.assertGreater(summary["repo_relative_path_count"], 0)
        self.assertTrue(summary["repo_path_safety_ready"])
        self.assertEqual(summary["repo_path_unsafe_count"], 0)
        self.assertEqual(summary["blocker_closure_reason_id_count"], 1)
        self.assertEqual(summary["blocker_closure_unmapped_count"], 0)
        self.assertEqual(summary["blocker_closure_missing_action_count"], 0)
        self.assertEqual(summary["blocker_closure_missing_primary_path_count"], 0)
        self.assertEqual(summary["blocker_closure_runtime_capture_required_count"], 1)
        self.assertEqual(summary["blocker_closure_future_adapter_required_count"], 0)
        self.assertEqual(summary["blocker_closure_validator_command_count"], 3)
        self.assertEqual(summary["blocker_closure_missing_evidence_count"], 1)
        self.assertTrue(summary["blocker_closure_manifest_coverage_ready"])
        self.assertTrue(summary["blocker_resolution_queue_ready"])
        self.assertEqual(summary["blocker_resolution_queue_work_package_count"], 1)
        self.assertEqual(summary["blocker_resolution_queue_row_count"], 1)
        self.assertEqual(summary["blocker_resolution_queue_expected_ledger_row_count"], 1)
        self.assertEqual(summary["blocker_resolution_queue_runtime_capture_package_count"], 1)
        self.assertEqual(summary["blocker_resolution_queue_future_adapter_package_count"], 0)
        self.assertEqual(summary["blocker_resolution_queue_runtime_capture_required_row_count"], 1)
        self.assertEqual(summary["blocker_resolution_queue_future_adapter_required_row_count"], 0)
        self.assertEqual(summary["blocker_resolution_queue_receipt_bound_row_count"], 0)
        self.assertEqual(summary["blocker_resolution_queue_validator_command_count"], 3)
        self.assertEqual(summary["blocker_resolution_queue_completion_gate_count"], 4)
        self.assertEqual(summary["blocker_resolution_queue_dependency_edge_count"], 0)
        self.assertEqual(summary["blocker_resolution_queue_missing_item_count"], 0)
        self.assertFalse(summary["blocker_resolution_queue_runtime_actuator_proof_handoff_ready"])
        self.assertFalse(summary["blocker_resolution_queue_runtime_actuator_live_spike_ready"])
        self.assertEqual(summary["blocker_resolution_queue_runtime_actuator_proof_requirement_count"], 0)
        self.assertEqual(summary["blocker_resolution_queue_runtime_actuator_proof_requirement_id_count"], 0)
        self.assertEqual(summary["blocker_resolution_queue_runtime_actuator_proof_artifact_count"], 0)
        self.assertEqual(summary["blocker_resolution_queue_runtime_actuator_dependency_edge_count"], 0)
        self.assertEqual(summary["blocker_resolution_queue_runtime_actuator_blocking_capability_count"], 0)
        self.assertEqual(summary["blocker_resolution_queue_runtime_actuator_control_blocker_count"], 0)
        self.assertTrue(summary["blocker_resolution_queue_coverage_ready"])
        self.assertEqual(summary["blocker_resolution_queue_error_count"], 0)
        self.assertEqual(summary["remaining_blockers"], ["no_replay_policy_candidate"])

    def test_reuse_evidence_capture_plan_drift_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            reuse_path = root / "reuse-evidence-capture-plan.json"
            reuse_plan = json.loads(reuse_path.read_text(encoding="utf-8"))
            reuse_plan["prompt_identity_metadata_missing_count"] = 0
            write_json(reuse_path, reuse_plan)

            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertIn("reuse_evidence_capture_plan_packet_mismatch", summary["errors"])
        self.assertIn("reuse_evidence_capture_handoff_prompt_identity_missing_count_mismatch", summary["errors"])
        self.assertGreaterEqual(summary["reuse_evidence_capture_error_count"], 1)

    def test_reuse_evidence_manifest_recommendation_drift_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            manifest_path = root / "operator-handoff-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["recommended_reuse_capture"]["candidate_trace_path"] = (
                "memory-moe-mvp/phase3-real-evidence/other/candidate-router-events.jsonl"
            )
            write_json(manifest_path, manifest)

            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["reuse_evidence_capture_manifest_recommendation_ready"])
        self.assertIn("reuse_evidence_capture_manifest_candidate_trace_path_mismatch", summary["errors"])
        self.assertIn("readme_reuse_evidence_capture_trace_path_missing", summary["errors"])
        self.assertGreaterEqual(summary["reuse_evidence_capture_error_count"], 1)

    def test_receipt_fill_missing_count_uses_ready_after_approval_not_raw_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            receipt_path = root / "receipt-fill-manifest.json"
            receipt_manifest = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt_manifest[0]["receipt_ready"] = True
            receipt_manifest[0]["ready_after_approval"] = False
            receipt_manifest[0]["fill_status_after_approval"] = "blocked_after_approval"
            receipt_manifest[0]["blockers_after_approval"] = ["candidate_trace_not_policy_candidate_ready"]
            write_json(receipt_path, receipt_manifest)
            packet_path = root / "phase3-evidence-packet.json"
            packet = json.loads(packet_path.read_text(encoding="utf-8"))
            packet["all_request_receipt_fill_manifest"] = receipt_manifest
            write_json(packet_path, packet)

            summary = planner.build_summary(root)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["receipt_fill_missing_count"], 2)
        self.assertNotIn("receipt_fill_missing_count_mismatch", summary["errors"])

    def test_receipt_fill_artifact_class_count_drift_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            manifest_path = root / "operator-handoff-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["handoff_counts"]["receipt_fill_candidate_router_trace_missing_count"] = 0
            write_json(manifest_path, manifest)

            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertIn("receipt_fill_candidate_router_trace_missing_count_mismatch", summary["errors"])
    def test_manual_capture_runbook_task_contract_drift_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            runbook_path = root / "manual-capture-runbook.json"
            runbook = json.loads(runbook_path.read_text(encoding="utf-8"))
            runbook["requests"][0]["manual_tasks"][0]["prompt_set_path"] = ""
            write_json(runbook_path, runbook)
            packet_path = root / "phase3-evidence-packet.json"
            packet = json.loads(packet_path.read_text(encoding="utf-8"))
            packet["all_request_manual_capture_runbook_manifest"] = runbook
            write_json(packet_path, packet)

            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertIn(
            "manual_capture_runbook_task_missing_prompt_set_path_count_mismatch",
            summary["errors"],
        )
        self.assertGreaterEqual(summary["manual_capture_runbook_error_count"], 1)

    def test_post_capture_intake_runbook_gate_contract_drift_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            runbook_path = root / "post-capture-intake-runbook.json"
            runbook = json.loads(runbook_path.read_text(encoding="utf-8"))
            runbook["requests"][0]["artifact_gates"][0]["prompt_set_path"] = ""
            write_json(runbook_path, runbook)
            packet_path = root / "phase3-evidence-packet.json"
            packet = json.loads(packet_path.read_text(encoding="utf-8"))
            packet["all_request_post_capture_intake_runbook_manifest"] = runbook
            write_json(packet_path, packet)

            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertIn(
            "post_capture_intake_runbook_gate_missing_prompt_set_path_count_mismatch",
            summary["errors"],
        )
        self.assertGreaterEqual(summary["post_capture_intake_runbook_error_count"], 1)

    def test_recommended_manual_capture_runbook_selection_drift_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            runbook_path = root / "recommended-manual-capture-runbook.json"
            runbook = json.loads(runbook_path.read_text(encoding="utf-8"))
            runbook["request_path"] = "memory-moe-mvp/phase3-real-evidence/wrong.runtime-capture-request.json"
            write_json(runbook_path, runbook)

            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertIn(
            "recommended_manual_capture_runbook_request_path_mismatch",
            summary["errors"],
        )
        self.assertGreaterEqual(summary["recommended_manual_capture_runbook_error_count"], 1)
    def test_recommended_runtime_capture_work_order_task_drift_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            work_order_path = root / "recommended-runtime-capture-work-order.json"
            work_order = json.loads(work_order_path.read_text(encoding="utf-8"))
            work_order["capture_steps"][0]["artifact_path"] = "memory-moe-mvp/phase3-real-evidence/wrong-artifact.jsonl"
            write_json(work_order_path, work_order)

            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertIn(
            "recommended_runtime_capture_work_order_manual_task_mismatch",
            summary["errors"],
        )
        self.assertGreaterEqual(summary["recommended_runtime_capture_work_order_error_count"], 1)

    def test_recommended_runtime_capture_work_order_sequence_drift_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            work_order_path = root / "recommended-runtime-capture-work-order.json"
            work_order = json.loads(work_order_path.read_text(encoding="utf-8"))
            sequence = work_order["post_capture_sequence"]
            sequence[3], sequence[4] = sequence[4], sequence[3]
            write_json(work_order_path, work_order)

            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertIn(
            "recommended_runtime_capture_work_order_completion_validation_not_before_intake",
            summary["errors"],
        )
        self.assertGreaterEqual(summary["recommended_runtime_capture_work_order_error_count"], 1)

    def test_recommended_completion_receipt_template_task_drift_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            template_path = root / "recommended-runtime-capture-completion-receipt.template.json"
            template = json.loads(template_path.read_text(encoding="utf-8"))
            template["capture_receipts"][0]["artifact_path"] = "memory-moe-mvp/phase3-real-evidence/wrong-artifact.jsonl"
            write_json(template_path, template)

            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertIn(
            "recommended_completion_receipt_template_work_order_task_mismatch",
            summary["errors"],
        )
        self.assertGreaterEqual(summary["recommended_runtime_capture_completion_receipt_error_count"], 1)

    def test_recommended_completion_receipt_validation_command_drift_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            template_path = root / "recommended-runtime-capture-completion-receipt.template.json"
            template = json.loads(template_path.read_text(encoding="utf-8"))
            template["completion_receipt_validation_step"]["command"] = []
            write_json(template_path, template)

            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertIn(
            "recommended_completion_receipt_validation_command_missing",
            summary["errors"],
        )
        self.assertGreaterEqual(summary["recommended_runtime_capture_completion_receipt_error_count"], 1)

    def test_next_unblocked_operator_handoff_packet_drift_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            handoff_path = root / "next-unblocked-operator-handoff.json"
            handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
            handoff["work_order_artifact"] = "wrong-work-order.json"
            write_json(handoff_path, handoff)

            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertEqual(summary["next_unblocked_operator_handoff_error_count"], 5)
        self.assertIn("next_unblocked_operator_handoff_packet_mismatch", summary["errors"])
        self.assertIn("next_unblocked_operator_handoff_manifest_mismatch", summary["errors"])
        self.assertIn("next_unblocked_operator_handoff_work_order_artifact_mismatch", summary["errors"])
        self.assertIn("next_unblocked_operator_handoff_gap_work_order_artifact_mismatch", summary["errors"])
        self.assertIn("next_unblocked_operator_handoff_count_next_unblocked_operator_handoff_work_order_artifact_mismatch", summary["errors"])

    def test_next_unblocked_operator_handoff_work_order_binding_drift_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            handoff_path = root / "next-unblocked-operator-handoff.json"
            handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
            handoff["work_order_next_artifact_id"] = "wrong_artifact"
            handoff["work_order_advances_next_package"] = False
            write_json(handoff_path, handoff)
            packet_path = root / "phase3-evidence-packet.json"
            packet = json.loads(packet_path.read_text(encoding="utf-8"))
            packet["phase3_next_unblocked_operator_handoff_summary"] = handoff
            packet["remaining_gaps"]["phase3_next_unblocked_operator_handoff_work_order_advances_next_package"] = False
            write_json(packet_path, packet)
            manifest_path = root / "operator-handoff-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["next_unblocked_operator_handoff"] = handoff
            manifest["handoff_counts"]["next_unblocked_operator_handoff_work_order_next_artifact_id"] = "wrong_artifact"
            manifest["handoff_counts"]["next_unblocked_operator_handoff_work_order_advances_next_package"] = False
            write_json(manifest_path, manifest)

            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["next_unblocked_operator_handoff_work_order_advances_next_package"])
        self.assertIn("next_unblocked_operator_handoff_work_order_next_artifact_mismatch", summary["errors"])
        self.assertIn("next_unblocked_operator_handoff_work_order_does_not_advance_package", summary["errors"])
        self.assertIn("next_unblocked_operator_handoff_policy_candidate_artifact_mismatch", summary["errors"])
    def test_runtime_actuator_proof_handoff_package_is_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            add_runtime_actuator_package(root)
            summary = planner.build_summary(root)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertTrue(summary["blocker_resolution_queue_coverage_ready"])
        self.assertTrue(summary["blocker_resolution_queue_runtime_actuator_proof_handoff_ready"])
        self.assertFalse(summary["blocker_resolution_queue_runtime_actuator_live_spike_ready"])
        self.assertEqual(summary["blocker_resolution_queue_runtime_actuator_proof_requirement_count"], 8)
        self.assertEqual(summary["blocker_resolution_queue_runtime_actuator_proof_requirement_id_count"], 8)
        self.assertEqual(summary["blocker_resolution_queue_runtime_actuator_proof_artifact_count"], 20)
        self.assertEqual(summary["blocker_resolution_queue_runtime_actuator_dependency_edge_count"], 14)
        self.assertEqual(summary["blocker_resolution_queue_runtime_actuator_blocking_capability_count"], 7)
        self.assertEqual(summary["blocker_resolution_queue_runtime_actuator_control_blocker_count"], 3)
        self.assertEqual(summary["blocker_resolution_queue_error_count"], 0)

    def test_runtime_actuator_residency_control_dependency_drift_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            add_runtime_actuator_package(root)
            queue_path = root / "blocker-resolution-queue.json"
            queue = json.loads(queue_path.read_text(encoding="utf-8"))
            proof_requirements = queue["work_packages"][1]["proof_handoff"]["proof_requirements"]
            for requirement in proof_requirements:
                if requirement["capability_id"] == "residency_control":
                    requirement["dependency_ids"] = ["residency_observation", "dense_fallback"]
                    requirement["dependency_count"] = 2
            queue["work_packages"][1]["proof_handoff"]["dependency_edge_count"] = 13
            write_json(queue_path, queue)
            packet_path = root / "phase3-evidence-packet.json"
            packet = json.loads(packet_path.read_text(encoding="utf-8"))
            packet["phase3_blocker_resolution_queue_manifest"] = queue
            packet["remaining_gaps"]["runtime_actuator_spike_dependency_edge_count"] = 13
            write_json(packet_path, packet)
            manifest_path = root / "operator-handoff-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["handoff_counts"]["blocker_resolution_queue_runtime_actuator_dependency_edge_count"] = 13
            write_json(manifest_path, manifest)
            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["blocker_resolution_queue_coverage_ready"])
        self.assertIn(
            "blocker_resolution_queue_runtime_actuator_control_dependencies_mismatch:runtime_actuator_spike:residency_control",
            summary["errors"],
        )

    def test_missing_required_file_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            (root / "README.md").unlink()
            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["handoff_package_ready"])
        self.assertIn("missing_required_file:README.md", summary["errors"])

    def test_safety_flag_drift_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            manifest_path = root / "operator-handoff-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["sends_prompt_traffic"] = True
            write_json(manifest_path, manifest)
            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertIn("operator_handoff_safety_flag_mismatch:sends_prompt_traffic", summary["errors"])

    def test_recommended_approval_command_drift_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            manifest_path = root / "operator-handoff-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["recommended_request"]["records_approval_keys"] = []
            write_json(manifest_path, manifest)
            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["recommended_approval_command_ready"])
        self.assertIn("recommended_approval_recorded_key_mismatch", summary["errors"])

    def test_all_request_approval_command_drift_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            queue_path = root / "capture-queue.json"
            queue = json.loads(queue_path.read_text(encoding="utf-8"))
            queue[0]["approval_rebuild_command"]["records_approval_keys"] = []
            write_json(queue_path, queue)
            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertTrue(summary["recommended_approval_command_ready"])
        self.assertEqual(summary["all_request_approval_command_count"], 1)
        self.assertEqual(summary["all_request_approval_command_ready_count"], 0)
        self.assertFalse(summary["all_request_approval_commands_ready"])
        self.assertGreater(summary["all_request_approval_command_error_count"], 0)
        self.assertTrue(any(error.startswith("all_request_approval_recorded_key_mismatch:") for error in summary["errors"]))

    def test_runtime_capture_preflight_drift_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            preflight_path = root / "recommended-runtime-capture-preflight.json"
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            preflight["artifact_checks"][0]["artifact_path"] = "memory-moe-mvp/phase3-real-evidence/wrong-router-events.jsonl"
            write_json(preflight_path, preflight)
            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["handoff_package_ready"])
        self.assertEqual(summary["runtime_capture_preflight_error_count"], 1)
        self.assertIn("runtime_capture_preflight_packet_mismatch", summary["errors"])

    def test_runtime_capture_command_contract_drift_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            contract_path = root / "recommended-runtime-capture-command-contract.json"
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract["capture_tasks"][0]["artifact_path"] = "memory-moe-mvp/phase3-real-evidence/wrong-router-events.jsonl"
            write_json(contract_path, contract)
            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["handoff_package_ready"])
        self.assertGreaterEqual(summary["runtime_capture_command_contract_error_count"], 1)
        self.assertIn("runtime_capture_command_contract_packet_mismatch", summary["errors"])
    def test_runtime_capture_execution_coverage_drift_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            execution_path = root / "recommended-runtime-capture-execution-coverage.json"
            execution = json.loads(execution_path.read_text(encoding="utf-8"))
            execution["artifact_execution"][0]["artifact_path"] = "memory-moe-mvp/phase3-real-evidence/wrong-router-events.jsonl"
            write_json(execution_path, execution)
            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["handoff_package_ready"])
        self.assertGreaterEqual(summary["runtime_capture_execution_error_count"], 1)
        self.assertIn("runtime_capture_execution_coverage_packet_mismatch", summary["errors"])

    def test_all_request_runtime_capture_execution_coverage_drift_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            execution_path = root / "all-request-runtime-capture-execution-coverage.json"
            execution = json.loads(execution_path.read_text(encoding="utf-8"))
            execution["requests"][0]["artifact_execution"][0]["artifact_path"] = "memory-moe-mvp/phase3-real-evidence/wrong-router-events.jsonl"
            write_json(execution_path, execution)
            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["handoff_package_ready"])
        self.assertGreaterEqual(summary["all_request_runtime_capture_execution_error_count"], 1)
        self.assertIn("all_request_runtime_capture_execution_coverage_packet_mismatch", summary["errors"])

    def test_runtime_capture_launch_card_directory_drift_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            directory_path = root / "runtime-capture-launch-card-directory.json"
            directory = json.loads(directory_path.read_text(encoding="utf-8"))
            directory["matched_card_count"] = 1
            write_json(directory_path, directory)
            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["handoff_package_ready"])
        self.assertGreaterEqual(summary["runtime_capture_launch_card_dir_error_count"], 1)
        self.assertIn("runtime_capture_launch_card_directory_packet_mismatch", summary["errors"])

    def test_approval_command_manifest_drift_is_invalid(self) -> None:


        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            approval_path = root / "approval-command-manifest.json"
            approval_manifest = json.loads(approval_path.read_text(encoding="utf-8"))
            approval_manifest[0]["writes_request_path"] = "memory-moe-mvp/phase3-real-evidence/wrong.runtime-capture-request.json"
            write_json(approval_path, approval_manifest)
            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertEqual(summary["approval_command_manifest_count"], 1)
        self.assertFalse(summary["approval_command_manifest_coverage_ready"])
        self.assertGreater(summary["approval_command_manifest_error_count"], 0)
        self.assertTrue(
            any(error.startswith("approval_command_manifest_embedded_writes_request_path_mismatch:") for error in summary["errors"])
        )

    def test_recommended_capture_queue_drift_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            queue_path = root / "capture-queue.json"
            queue = json.loads(queue_path.read_text(encoding="utf-8"))
            queue[0]["request_path"] = "memory-moe-mvp/phase3-real-evidence/wrong.runtime-capture-request.json"
            write_json(queue_path, queue)
            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["recommended_capture_queue_ready"])
        self.assertIn("recommended_capture_queue_request_mismatch", summary["errors"])

    def test_repo_escape_path_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            launch_card_path = root / "recommended-runtime-capture-launch-card.template.json"
            launch_card = json.loads(launch_card_path.read_text(encoding="utf-8"))
            launch_card["tasks"][0]["artifact_output_path"] = "../outside/candidate-router-events.jsonl"
            write_json(launch_card_path, launch_card)
            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["repo_path_safety_ready"])
        self.assertGreater(summary["repo_path_unsafe_count"], 0)
        self.assertTrue(any(error.startswith("repo_path_unsafe:") for error in summary["errors"]))

    def test_saved_launch_card_handoff_artifact_drift_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            manifest_path = root / "operator-handoff-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["handoff_counts"]["launch_card_saved_handoff_artifact_drifted_count"] = 1
            write_json(manifest_path, manifest)
            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["launch_card_saved_handoff_parity_ready"])
        self.assertEqual(summary["launch_card_saved_handoff_error_count"], 1)
        self.assertIn("operator_handoff_launch_card_saved_handoff_artifact_drifted_count_mismatch", summary["errors"])

    def test_launch_card_worksheet_task_drift_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            worksheet_path = root / "launch-card-binding-worksheet.json"
            worksheet = json.loads(worksheet_path.read_text(encoding="utf-8"))
            worksheet["tasks"][0]["artifact_output_path"] = "memory-moe-mvp/phase3-real-evidence/wrong.jsonl"
            write_json(worksheet_path, worksheet)
            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["launch_card_worksheet_task_parity_ready"])
        self.assertIn("launch_card_worksheet_task_mismatch", summary["errors"])

    def test_launch_card_capture_queue_task_drift_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            queue_path = root / "capture-queue.json"
            queue = json.loads(queue_path.read_text(encoding="utf-8"))
            queue[0]["capture_fill_steps"][0]["receipt_path"] = "memory-moe-mvp/phase3-real-evidence/wrong-receipt.json"
            write_json(queue_path, queue)
            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["launch_card_capture_queue_task_parity_ready"])
        self.assertIn("launch_card_capture_queue_task_mismatch", summary["errors"])

    def test_non_recommended_binding_worksheet_task_drift_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            other_request_path = "memory-moe-mvp/phase3-real-evidence/other.runtime-capture-request.json"
            other_launch_card_path = "memory-moe-mvp/phase3-real-evidence/other.runtime-capture-launch-card.template.json"
            other_prompt_set_path = "memory-moe-mvp/phase3-real-evidence/other.prompt-set.json"
            other_artifact_path = "memory-moe-mvp/phase3-real-evidence/other/candidate-router-events.jsonl"
            other_receipt_path = "memory-moe-mvp/phase3-real-evidence/other/candidate-router-events.capture-receipt.json"

            manifest_path = root / "operator-handoff-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["handoff_counts"]["capture_queue_request_count"] = 2
            manifest["handoff_counts"]["launch_card_task_count"] = 3
            manifest["handoff_counts"]["launch_card_unbound_task_count"] = 3
            manifest["handoff_counts"]["receipt_fill_entry_count"] = 3
            manifest["handoff_counts"]["receipt_fill_missing_count"] = 3
            write_json(manifest_path, manifest)

            worksheet_path = root / "launch-card-binding-worksheet.json"
            worksheet = json.loads(worksheet_path.read_text(encoding="utf-8"))
            worksheet["task_count"] = 3
            worksheet["unbound_task_count"] = 3
            worksheet["tasks"].append(
                {
                    "request_path": other_request_path,
                    "launch_card_path": other_launch_card_path,
                    "task_id": "phase3_capture_other_candidate_router_trace",
                    "artifact_id": "candidate_router_trace",
                    "artifact_output_path": "memory-moe-mvp/phase3-real-evidence/other/wrong-router-events.jsonl",
                    "receipt_output_path": other_receipt_path,
                    "prompt_set_path": other_prompt_set_path,
                    "approval_keys": ["runtime_prompt_traffic_approved", "router_trace_capture_approved"],
                }
            )
            write_json(worksheet_path, worksheet)

            queue_path = root / "capture-queue.json"
            queue = json.loads(queue_path.read_text(encoding="utf-8"))
            queue.append(
                {
                    "request_path": other_request_path,
                    "status": "approval_required",
                    "rank": 2,
                    "next_artifact_id": "candidate_router_trace",
                    "missing_approval_keys": ["runtime_prompt_traffic_approved"],
                    "approval_rebuild_command": {
                        "command": [
                            "uv",
                            "run",
                            "--managed-python",
                            "--python",
                            "3.13",
                            "scripts/build_phase3_runtime_capture_request.py",
                            "memory-moe-mvp/phase3-real-evidence/other.json",
                            "--output",
                            other_request_path,
                            "--json",
                            "--runtime-prompt-traffic-approved",
                        ],
                        "command_class": "phase3_runtime_capture_request_approval_rebuild",
                        "metadata_only": True,
                        "records_approval_keys": ["runtime_prompt_traffic_approved"],
                        "requires_explicit_user_approval": True,
                        "writes_request_path": other_request_path,
                    },
                    "capture_fill_steps": [
                        {
                            "artifact_id": "candidate_router_trace",
                            "artifact_path": other_artifact_path,
                            "receipt_path": other_receipt_path,
                            "validator_command_count": 1,
                        }
                    ],
                }
            )
            write_json(queue_path, queue)

            receipt_path = root / "receipt-fill-manifest.json"
            receipt_manifest = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt_manifest.append({"request_path": other_request_path, "artifact_id": "candidate_router_trace", "receipt_ready": False})
            write_json(receipt_path, receipt_manifest)

            command_manifest_path = root / "receipt-fill-command-manifest.json"
            command_manifest = json.loads(command_manifest_path.read_text(encoding="utf-8"))
            command_manifest.append(
                {
                    "request_path": other_request_path,
                    "artifact_id": "candidate_router_trace",
                    "validator_command_count": 1,
                    "validator_commands": ["uv run --managed-python --python 3.13 scripts/other.py"],
                }
            )
            write_json(command_manifest_path, command_manifest)

            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["handoff_package_ready"])
        self.assertEqual(summary["binding_worksheet_queue_task_key_count"], 3)
        self.assertEqual(summary["capture_queue_task_key_count"], 3)
        self.assertFalse(summary["binding_worksheet_capture_queue_task_parity_ready"])
        self.assertIn("binding_worksheet_capture_queue_task_mismatch", summary["errors"])

    def test_downstream_handoff_missing_section_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            downstream_path = root / "downstream-handoff-manifest.json"
            downstream_manifest = json.loads(downstream_path.read_text(encoding="utf-8"))
            downstream_manifest[0].pop("dense_fallback_capture")
            write_json(downstream_path, downstream_manifest)
            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["downstream_handoff_manifest_coverage_ready"])
        self.assertEqual(summary["downstream_handoff_missing_section_count"], 1)
        self.assertTrue(
            any(error.startswith("downstream_handoff_section_missing:") for error in summary["errors"])
        )
    def test_blocker_closure_missing_next_action_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            blocker_path = root / "blocker-closure-manifest.json"
            blocker_manifest = json.loads(blocker_path.read_text(encoding="utf-8"))
            blocker_manifest[0]["next_action"] = ""
            write_json(blocker_path, blocker_manifest)
            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["blocker_closure_manifest_coverage_ready"])
        self.assertEqual(summary["blocker_closure_missing_action_count"], 1)
        self.assertTrue(
            any(error.startswith("blocker_closure_next_action_missing:") for error in summary["errors"])
        )

    def test_blocker_closure_remaining_blocker_mismatch_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            blocker_path = root / "blocker-closure-manifest.json"
            blocker_manifest = json.loads(blocker_path.read_text(encoding="utf-8"))
            blocker_manifest[0]["reason_id"] = "different_blocker"
            write_json(blocker_path, blocker_manifest)
            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["blocker_closure_manifest_coverage_ready"])
        self.assertIn("blocker_closure_remaining_blocker_mismatch", summary["errors"])
    def test_validator_manifest_runtime_entry_missing_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            validator_manifest_path = root / "validator-command-manifest.json"
            validator_manifest = json.loads(validator_manifest_path.read_text(encoding="utf-8"))
            validator_manifest = [
                item for item in validator_manifest
                if item.get("artifact_stage") != "runtime_capture" or item.get("artifact_id") != "candidate_router_trace"
            ]
            write_json(validator_manifest_path, validator_manifest)
            manifest_path = root / "operator-handoff-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["handoff_counts"]["validator_command_count"] = 2
            write_json(manifest_path, manifest)
            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["validator_manifest_coverage_ready"])
        self.assertEqual(summary["validator_manifest_runtime_artifact_count"], 1)
        self.assertTrue(
            any(error.startswith("validator_manifest_runtime_entry_missing_for_receipt:") for error in summary["errors"])
        )
        self.assertTrue(
            any(
                error.startswith("validator_manifest_runtime_entry_missing_for_capture_queue:")
                for error in summary["errors"]
            )
        )

    def test_validator_manifest_runtime_validator_missing_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            validator_manifest_path = root / "validator-command-manifest.json"
            validator_manifest = json.loads(validator_manifest_path.read_text(encoding="utf-8"))
            validator_manifest[0]["validator_commands"] = []
            validator_manifest[0]["validator_command_count"] = 0
            write_json(validator_manifest_path, validator_manifest)
            manifest_path = root / "operator-handoff-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["handoff_counts"]["validator_command_count"] = 2
            write_json(manifest_path, manifest)
            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["validator_manifest_coverage_ready"])
        self.assertEqual(summary["validator_manifest_runtime_missing_validator_count"], 1)
        self.assertTrue(
            any(error.startswith("validator_manifest_runtime_commands_missing:") for error in summary["errors"])
        )
        self.assertTrue(
            any(error.startswith("validator_manifest_receipt_command_count_mismatch:") for error in summary["errors"])
        )
        self.assertTrue(
            any(error.startswith("validator_manifest_capture_queue_count_mismatch:") for error in summary["errors"])
        )
    def test_validator_manifest_future_entry_missing_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            validator_manifest_path = root / "validator-command-manifest.json"
            validator_manifest = json.loads(validator_manifest_path.read_text(encoding="utf-8"))
            validator_manifest = [item for item in validator_manifest if item.get("artifact_stage") != "future_adapter"]
            write_json(validator_manifest_path, validator_manifest)
            manifest_path = root / "operator-handoff-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["handoff_counts"]["validator_command_count"] = 2
            write_json(manifest_path, manifest)
            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["validator_manifest_coverage_ready"])
        self.assertEqual(summary["validator_manifest_future_artifact_count"], 0)
        self.assertTrue(
            any(error.startswith("validator_manifest_future_entry_missing_for_request:") for error in summary["errors"])
        )

    def test_receipt_command_validator_missing_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            command_manifest_path = root / "receipt-fill-command-manifest.json"
            command_manifest = json.loads(command_manifest_path.read_text(encoding="utf-8"))
            command_manifest[0]["validator_commands"] = []
            command_manifest[0]["validator_command_count"] = 0
            write_json(command_manifest_path, command_manifest)
            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertEqual(summary["receipt_fill_command_missing_validator_count"], 1)
        self.assertFalse(summary["receipt_fill_command_validator_coverage_ready"])
        self.assertTrue(
            any(error.startswith("receipt_fill_command_validator_commands_missing:") for error in summary["errors"])
        )
        self.assertTrue(
            any(
                error.startswith("receipt_fill_command_capture_queue_validator_count_mismatch:")
                for error in summary["errors"]
            )
        )

    def test_receipt_command_key_drift_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            command_manifest_path = root / "receipt-fill-command-manifest.json"
            command_manifest = json.loads(command_manifest_path.read_text(encoding="utf-8"))
            command_manifest[0]["artifact_id"] = "wrong_artifact"
            command_manifest[0]["command_manifest_key"] = f"{command_manifest[0]['request_path']}::wrong_artifact"
            write_json(command_manifest_path, command_manifest)
            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["receipt_fill_command_key_parity_ready"])
        self.assertIn("receipt_fill_command_key_mismatch", summary["errors"])

    def test_capture_queue_key_drift_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            queue_path = root / "capture-queue.json"
            queue = json.loads(queue_path.read_text(encoding="utf-8"))
            queue[0]["capture_fill_steps"][0]["artifact_id"] = "wrong_artifact"
            write_json(queue_path, queue)
            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["receipt_fill_capture_queue_key_parity_ready"])
        self.assertIn("receipt_fill_capture_queue_key_mismatch", summary["errors"])

    def test_blocker_resolution_queue_missing_completion_gates_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            queue_path = root / "blocker-resolution-queue.json"
            queue = json.loads(queue_path.read_text(encoding="utf-8"))
            queue["work_packages"][0]["completion_gates"] = []
            queue["work_packages"][0]["completion_gate_count"] = 0
            queue["completion_gate_count"] = 0
            write_json(queue_path, queue)
            packet_path = root / "phase3-evidence-packet.json"
            packet = json.loads(packet_path.read_text(encoding="utf-8"))
            packet["phase3_blocker_resolution_queue_manifest"] = queue
            packet["remaining_gaps"]["phase3_blocker_resolution_queue_completion_gate_count"] = 0
            write_json(packet_path, packet)
            manifest_path = root / "operator-handoff-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["handoff_counts"]["blocker_resolution_queue_completion_gate_count"] = 0
            write_json(manifest_path, manifest)
            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["blocker_resolution_queue_coverage_ready"])
        self.assertIn("blocker_resolution_queue_completion_gates_missing:policy_candidate_trace_capture", summary["errors"])

    def test_blocker_resolution_queue_unknown_dependency_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            queue_path = root / "blocker-resolution-queue.json"
            queue = json.loads(queue_path.read_text(encoding="utf-8"))
            queue["work_packages"][0]["depends_on_work_package_ids"] = ["missing_package"]
            queue["work_packages"][0]["dependency_count"] = 1
            queue["dependency_edge_count"] = 1
            write_json(queue_path, queue)
            packet_path = root / "phase3-evidence-packet.json"
            packet = json.loads(packet_path.read_text(encoding="utf-8"))
            packet["phase3_blocker_resolution_queue_manifest"] = queue
            packet["remaining_gaps"]["phase3_blocker_resolution_queue_dependency_edge_count"] = 1
            write_json(packet_path, packet)
            manifest_path = root / "operator-handoff-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["handoff_counts"]["blocker_resolution_queue_dependency_edge_count"] = 1
            write_json(manifest_path, manifest)
            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["blocker_resolution_queue_coverage_ready"])
        self.assertIn(
            "blocker_resolution_queue_dependency_unknown:policy_candidate_trace_capture:missing_package",
            summary["errors"],
        )

    def test_blocker_resolution_queue_self_dependency_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            queue_path = root / "blocker-resolution-queue.json"
            queue = json.loads(queue_path.read_text(encoding="utf-8"))
            queue["work_packages"][0]["depends_on_work_package_ids"] = ["policy_candidate_trace_capture"]
            queue["work_packages"][0]["dependency_count"] = 1
            queue["dependency_edge_count"] = 1
            write_json(queue_path, queue)
            packet_path = root / "phase3-evidence-packet.json"
            packet = json.loads(packet_path.read_text(encoding="utf-8"))
            packet["phase3_blocker_resolution_queue_manifest"] = queue
            packet["remaining_gaps"]["phase3_blocker_resolution_queue_dependency_edge_count"] = 1
            write_json(packet_path, packet)
            manifest_path = root / "operator-handoff-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["handoff_counts"]["blocker_resolution_queue_dependency_edge_count"] = 1
            write_json(manifest_path, manifest)
            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["blocker_resolution_queue_coverage_ready"])
        self.assertIn(
            "blocker_resolution_queue_dependency_order_invalid:policy_candidate_trace_capture:policy_candidate_trace_capture",
            summary["errors"],
        )
    def test_blocker_resolution_queue_row_mismatch_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_minimal_handoff(root)
            queue_path = root / "blocker-resolution-queue.json"
            queue = json.loads(queue_path.read_text(encoding="utf-8"))
            queue["work_packages"][0]["row_count"] = 2
            queue["queue_row_count"] = 2
            write_json(queue_path, queue)
            packet_path = root / "phase3-evidence-packet.json"
            packet = json.loads(packet_path.read_text(encoding="utf-8"))
            packet["phase3_blocker_resolution_queue_manifest"] = queue
            write_json(packet_path, packet)
            summary = planner.build_summary(root)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["blocker_resolution_queue_coverage_ready"])
        self.assertIn("blocker_resolution_queue_ledger_row_count_mismatch", summary["errors"])
    def test_markdown_report_names_errors(self) -> None:
        summary = {
            "valid": False,
            "handoff_package_ready": False,
            "metadata_only": True,
            "phase3_complete": False,
            "decision": "no_go_live_spike",
            "required_file_count": 27,
            "missing_file_count": 1,
            "extra_file_count": 0,
            "capture_queue_request_count": 0,
            "validator_command_count": 0,
            "receipt_fill_missing_count": 0,
            "remaining_blockers": ["no_replay_policy_candidate"],
            "errors": ["missing_required_file:README.md"],
            "safety_contract": ["operator handoff validation reads local files only"],
        }
        report = planner.format_markdown_report(summary)

        self.assertIn("# Phase 3 Operator Handoff Validation", report)
        self.assertIn("`missing_required_file:README.md`", report)
        self.assertIn("`no_replay_policy_candidate`", report)


if __name__ == "__main__":
    unittest.main()
