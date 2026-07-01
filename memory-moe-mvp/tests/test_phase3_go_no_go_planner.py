import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_phase3_go_no_go.py"
SPEC = importlib.util.spec_from_file_location("plan_phase3_go_no_go", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)

TRACE_FIXTURE = ROOT / "memory-moe-mvp" / "data" / "llama_cpp_router_trace.fixture.jsonl"
INVENTORY_FIXTURE = ROOT / "memory-moe-mvp" / "data" / "expert_inventory_manifest.fixture.json"
POLICIES_FIXTURE = ROOT / "memory-moe-mvp" / "data" / "baseline_replay_policies.json"
MANAGED_PLAN = ROOT / "memory-moe-mvp" / "data" / "managed_expert_loading_plan.json"


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


class Phase3GoNoGoPlannerTests(unittest.TestCase):
    def test_default_decision_is_valid_no_go_with_explicit_reasons(self) -> None:
        summary = planner.build_decision_summary(
            TRACE_FIXTURE,
            INVENTORY_FIXTURE,
            POLICIES_FIXTURE,
            MANAGED_PLAN,
        )

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["mode"], "phase3_go_no_go_decision")
        self.assertEqual(summary["decision"], "no_go_live_spike")
        self.assertFalse(summary["ready_for_phase4_adapter_spike"])
        self.assertFalse(summary["ready_for_live_spike"])
        reason_ids = {item["id"] for item in summary["no_go_reasons"]}
        self.assertIn("no_replay_policy_candidate", reason_ids)
        policy_reason = next(item for item in summary["no_go_reasons"] if item["id"] == "no_replay_policy_candidate")
        self.assertEqual(policy_reason["evidence"]["source"], "repo_real_evidence_matrix")
        self.assertEqual(policy_reason["evidence"]["blocked_bundle_count"], 6)
        self.assertEqual(policy_reason["evidence"]["no_reuse_distance_observation_count"], 6)
        self.assertEqual(policy_reason["evidence"]["next_operator_step"], "capture_candidate_router_trace_with_reuse_distance")
        self.assertEqual(policy_reason["evidence"]["blocker_counts"]["no_replay_policy_candidate"], 6)
        self.assertNotIn("real_model_trace_inventory_pairing_not_ready", reason_ids)
        self.assertIn("reuse_evidence_capture_not_ready", reason_ids)
        reuse_reason = next(item for item in summary["no_go_reasons"] if item["id"] == "reuse_evidence_capture_not_ready")
        self.assertEqual(reuse_reason["evidence"]["bundle_count"], 6)
        self.assertEqual(reuse_reason["evidence"]["reuse_ready_count"], 0)
        self.assertEqual(reuse_reason["evidence"]["candidate_trace_receipt_ready_count"], 6)
        self.assertEqual(reuse_reason["evidence"]["recommended_next_capture"]["bundle_name"], "PC Mixtral Phase 3 real evidence")
        self.assertIn("dense_fallback_comparison_not_ready", reason_ids)
        self.assertIn("capture_result_intake_not_ready", reason_ids)
        intake_reason = next(item for item in summary["no_go_reasons"] if item["id"] == "capture_result_intake_not_ready")
        self.assertEqual(intake_reason["evidence"]["request_ready_for_operator_capture_flag_count"], 0)
        self.assertEqual(intake_reason["evidence"]["approved_but_capture_incomplete_request_count"], 0)
        self.assertEqual(intake_reason["evidence"]["runtime_approval_missing_request_count"], 6)
        self.assertEqual(intake_reason["evidence"]["approval_rebuild_command_available_request_count"], 6)
        self.assertEqual(intake_reason["evidence"]["approval_rebuild_command_manifest_count"], 6)
        self.assertEqual(intake_reason["evidence"]["approved_runtime_capture_pending_request_count"], 0)
        self.assertEqual(intake_reason["evidence"]["candidate_trace_receipt_ready_count"], 6)
        self.assertEqual(intake_reason["evidence"]["managed_output_receipt_ready_count"], 0)
        self.assertEqual(intake_reason["evidence"]["dense_output_receipt_ready_count"], 0)
        self.assertEqual(intake_reason["evidence"]["live_capability_proof_ready_count"], 0)
        self.assertFalse(intake_reason["evidence"]["all_capture_receipts_ready"])
        self.assertFalse(intake_reason["evidence"]["all_output_receipt_bindings_ready"])
        self.assertEqual(intake_reason["evidence"]["receipt_fill_candidate_router_trace_entry_count"], 6)
        self.assertEqual(intake_reason["evidence"]["receipt_fill_candidate_router_trace_ready_count"], 0)
        self.assertEqual(intake_reason["evidence"]["receipt_fill_candidate_router_trace_missing_count"], 6)
        self.assertEqual(intake_reason["evidence"]["receipt_fill_managed_output_entry_count"], 6)
        self.assertEqual(intake_reason["evidence"]["receipt_fill_managed_output_ready_count"], 0)
        self.assertEqual(intake_reason["evidence"]["receipt_fill_managed_output_missing_count"], 6)
        self.assertEqual(intake_reason["evidence"]["receipt_fill_dense_output_entry_count"], 6)
        self.assertEqual(intake_reason["evidence"]["receipt_fill_dense_output_ready_count"], 0)
        self.assertEqual(intake_reason["evidence"]["receipt_fill_dense_output_missing_count"], 6)
        self.assertIn("live_capability_proof_not_ready", reason_ids)
        self.assertIn("live_actuator_missing", reason_ids)
        self.assertIn("live_actuator_capabilities_unavailable", reason_ids)
        self.assertTrue(summary["gate_status"]["policy_replay_metrics_ready"])
        self.assertTrue(summary["gate_status"]["real_model_pair_ready"])
        self.assertEqual(summary["gate_status"]["real_evidence_bundle_count"], 6)
        self.assertEqual(summary["gate_status"]["real_evidence_policy_candidate_ready_count"], 0)
        self.assertEqual(summary["gate_status"]["policy_candidate_trace_receipt_source"], "repo_real_evidence_matrix")
        self.assertFalse(summary["gate_status"]["policy_candidate_trace_receipt_ready"])
        self.assertEqual(summary["gate_status"]["policy_candidate_trace_receipt_required_count"], 0)
        self.assertEqual(summary["gate_status"]["policy_candidate_trace_receipt_ready_count"], 0)
        self.assertEqual(summary["gate_status"]["policy_candidate_trace_receipt_attached_count"], 0)
        self.assertEqual(summary["gate_status"]["policy_candidate_trace_receipt_blocked_bundle_count"], 0)
        self.assertTrue(summary["gate_status"]["policy_candidate_trace_receipt_scaffold_ready"])
        self.assertEqual(summary["gate_status"]["policy_candidate_trace_receipt_scaffold_source"], "repo_reuse_evidence_capture")
        self.assertEqual(summary["gate_status"]["policy_candidate_trace_receipt_scaffold_required_count"], 6)
        self.assertEqual(summary["gate_status"]["policy_candidate_trace_receipt_scaffold_ready_count"], 6)
        self.assertEqual(summary["gate_status"]["policy_candidate_trace_receipt_scaffold_valid_count"], 6)
        self.assertEqual(summary["gate_status"]["policy_candidate_trace_receipt_scaffold_blocked_count"], 0)
        self.assertEqual(summary["gate_status"]["policy_candidate_blocked_bundle_count"], 6)
        self.assertEqual(summary["gate_status"]["policy_candidate_no_reuse_distance_observation_count"], 6)
        self.assertEqual(summary["policy_candidate_evidence"]["blocked_bundle_count"], 6)
        self.assertEqual(summary["policy_candidate_evidence"]["blocker_counts"]["no_reuse_distance_observations"], 6)
        self.assertFalse(summary["gate_status"]["reuse_evidence_capture_ready"])
        self.assertEqual(summary["gate_status"]["reuse_evidence_reuse_ready_count"], 0)
        self.assertEqual(summary["gate_status"]["reuse_evidence_reuse_blocked_count"], 6)
        self.assertEqual(summary["gate_status"]["reuse_evidence_candidate_trace_valid_count"], 6)
        self.assertEqual(summary["gate_status"]["reuse_evidence_candidate_trace_receipt_ready_count"], 6)
        self.assertEqual(summary["gate_status"]["reuse_evidence_no_reuse_distance_observation_count"], 6)
        self.assertEqual(summary["gate_status"]["reuse_evidence_prompt_identity_ready_count"], 0)
        self.assertEqual(summary["gate_status"]["reuse_evidence_prompt_identity_metadata_missing_count"], 6)
        self.assertFalse(summary["gate_status"]["capture_result_intake_ready"])
        self.assertEqual(summary["gate_status"]["capture_result_ready_for_operator_capture_flag_count"], 0)
        self.assertEqual(summary["gate_status"]["capture_result_approved_but_capture_incomplete_request_count"], 0)
        self.assertEqual(summary["gate_status"]["capture_result_runtime_approval_missing_request_count"], 6)
        self.assertEqual(summary["gate_status"]["capture_result_approval_rebuild_command_available_request_count"], 6)
        self.assertEqual(summary["gate_status"]["capture_result_approval_rebuild_command_manifest_count"], 6)
        self.assertEqual(summary["gate_status"]["capture_result_approved_runtime_capture_pending_request_count"], 0)
        self.assertEqual(summary["gate_status"]["capture_result_capture_receipts_required_count"], 18)
        self.assertEqual(summary["gate_status"]["capture_result_capture_receipts_ready_count"], 6)
        self.assertEqual(summary["gate_status"]["capture_result_missing_receipt_gate_request_count"], 6)
        self.assertEqual(summary["gate_status"]["capture_result_output_receipt_binding_ready_count"], 0)
        self.assertEqual(summary["gate_status"]["capture_result_candidate_trace_receipt_ready_count"], 6)
        self.assertEqual(summary["gate_status"]["capture_result_managed_output_receipt_ready_count"], 0)
        self.assertEqual(summary["gate_status"]["capture_result_dense_output_receipt_ready_count"], 0)
        self.assertEqual(summary["gate_status"]["capture_result_live_capability_proof_ready_count"], 0)
        self.assertFalse(summary["gate_status"]["capture_result_all_capture_receipts_ready"])
        self.assertFalse(summary["gate_status"]["capture_result_all_output_receipt_bindings_ready"])
        self.assertEqual(summary["gate_status"]["capture_result_receipt_fill_candidate_router_trace_entry_count"], 6)
        self.assertEqual(summary["gate_status"]["capture_result_receipt_fill_candidate_router_trace_ready_count"], 0)
        self.assertEqual(summary["gate_status"]["capture_result_receipt_fill_candidate_router_trace_missing_count"], 6)
        self.assertEqual(summary["gate_status"]["capture_result_receipt_fill_managed_output_entry_count"], 6)
        self.assertEqual(summary["gate_status"]["capture_result_receipt_fill_managed_output_ready_count"], 0)
        self.assertEqual(summary["gate_status"]["capture_result_receipt_fill_managed_output_missing_count"], 6)
        self.assertEqual(summary["gate_status"]["capture_result_receipt_fill_dense_output_entry_count"], 6)
        self.assertEqual(summary["gate_status"]["capture_result_receipt_fill_dense_output_ready_count"], 0)
        self.assertEqual(summary["gate_status"]["capture_result_receipt_fill_dense_output_missing_count"], 6)
        self.assertEqual(summary["gate_status"]["capture_result_ready_to_update_bundle_count"], 0)
        self.assertTrue(summary["gate_status"]["handoff_scaffold_coverage_ready"])
        self.assertEqual(summary["gate_status"]["handoff_bundle_count"], 6)
        self.assertEqual(summary["gate_status"]["handoff_scaffold_ready_count"], 6)
        self.assertEqual(summary["gate_status"]["handoff_runtime_capture_launch_card_template_ready_count"], 6)
        self.assertEqual(summary["gate_status"]["handoff_runtime_capture_launch_card_binding_ready_count"], 0)
        self.assertEqual(summary["gate_status"]["handoff_runtime_capture_launch_card_runtime_command_ready_count"], 0)
        self.assertEqual(summary["gate_status"]["handoff_runtime_capture_launch_card_command_option_count"], 0)
        self.assertEqual(summary["gate_status"]["handoff_runtime_capture_launch_card_missing_runtime_command_count"], 18)
        self.assertTrue(summary["gate_status"]["launch_card_library_ready"])
        self.assertEqual(summary["gate_status"]["launch_card_library_card_count"], 6)
        self.assertEqual(summary["gate_status"]["launch_card_library_task_count"], 18)
        self.assertTrue(summary["gate_status"]["launch_card_binding_handoff_ready"])
        self.assertEqual(summary["gate_status"]["launch_card_binding_handoff_task_count"], 18)
        self.assertTrue(summary["gate_status"]["launch_card_model_plane_contract_request_ready"])
        self.assertEqual(summary["gate_status"]["launch_card_model_plane_contract_request_task_count"], 18)
        self.assertTrue(summary["gate_status"]["launch_card_saved_handoff_artifacts_ready"])
        self.assertEqual(summary["gate_status"]["launch_card_saved_handoff_artifact_missing_count"], 0)
        self.assertEqual(summary["gate_status"]["launch_card_saved_handoff_artifact_drifted_count"], 0)
        self.assertEqual(summary["gate_status"]["launch_card_missing_runtime_command_count"], 18)
        self.assertTrue(summary["gate_status"]["runtime_capture_request_audit_ready"])
        self.assertEqual(summary["gate_status"]["runtime_capture_request_count"], 6)
        self.assertEqual(summary["gate_status"]["runtime_capture_drifted_request_count"], 0)
        self.assertEqual(summary["gate_status"]["runtime_capture_validator_command_missing_request_count"], 0)
        self.assertFalse(summary["gate_status"]["dense_fallback_comparison_ready"])
        self.assertFalse(summary["gate_status"]["live_capability_proof_ready"])
        self.assertTrue(summary["gate_status"]["runtime_actuator_spike_handoff_ready"])
        self.assertFalse(summary["gate_status"]["runtime_actuator_spike_live_ready"])
        self.assertEqual(summary["gate_status"]["runtime_actuator_spike_proof_requirement_count"], 8)
        self.assertEqual(summary["gate_status"]["runtime_actuator_spike_proof_artifact_count"], 20)
        self.assertEqual(summary["gate_status"]["runtime_actuator_spike_dependency_edge_count"], 14)
        self.assertEqual(summary["gate_status"]["runtime_actuator_spike_blocking_capability_count"], 7)
        self.assertEqual(summary["gate_status"]["runtime_actuator_spike_control_blocker_count"], 3)
        self.assertEqual(summary["input_summaries"]["pairing"]["decision_pairing_basis"], "repo_real_evidence_matrix")
        self.assertTrue(summary["input_summaries"]["pairing"]["decision_real_model_pair_ready"])
        policy_candidate = summary["input_summaries"]["policy_candidate"]
        receipt = summary["input_summaries"]["policy_candidate_trace_receipt"]
        self.assertEqual(receipt["source"], "repo_real_evidence_matrix")
        self.assertEqual(receipt["required_count"], 0)
        self.assertEqual(receipt["ready_count"], 0)
        self.assertTrue(receipt["scaffold_ready"])
        self.assertEqual(receipt["scaffold_source"], "repo_reuse_evidence_capture")
        self.assertEqual(receipt["scaffold_required_count"], 6)
        self.assertEqual(receipt["scaffold_ready_count"], 6)
        self.assertEqual(receipt["scaffold_valid_count"], 6)
        self.assertEqual(receipt["scaffold_blocked_count"], 0)
        self.assertEqual(policy_candidate["blocked_bundle_count"], 6)
        self.assertEqual(policy_candidate["no_reuse_distance_observation_count"], 6)
        self.assertEqual(policy_candidate["next_operator_step"], "capture_candidate_router_trace_with_reuse_distance")
        reuse = summary["input_summaries"]["reuse_evidence_capture"]
        self.assertEqual(reuse["bundle_count"], 6)
        self.assertEqual(reuse["reuse_ready_count"], 0)
        self.assertEqual(reuse["reuse_blocked_count"], 6)
        self.assertEqual(reuse["candidate_trace_receipt_ready_count"], 6)
        self.assertEqual(reuse["prompt_identity_ready_count"], 0)
        self.assertEqual(reuse["recommended_next_capture"]["required_result"], "capture_candidate_router_trace_with_reuse_distance")
        intake = summary["input_summaries"]["capture_result_intake"]
        self.assertEqual(intake["request_count"], 6)
        self.assertEqual(intake["request_ready_for_operator_capture_flag_count"], 0)
        self.assertEqual(intake["approved_but_capture_incomplete_request_count"], 0)
        self.assertEqual(intake["runtime_approval_missing_request_count"], 6)
        self.assertEqual(intake["approval_rebuild_command_available_request_count"], 6)
        self.assertEqual(intake["approval_rebuild_command_manifest_count"], 6)
        self.assertEqual(intake["approved_runtime_capture_pending_request_count"], 0)
        self.assertEqual(intake["capture_receipt_required_count"], 18)
        self.assertEqual(intake["capture_receipt_ready_count"], 6)
        self.assertEqual(intake["capture_receipt_missing_request_count"], 6)
        self.assertEqual(intake["output_receipt_binding_ready_count"], 0)
        self.assertEqual(intake["candidate_trace_receipt_ready_count"], 6)
        self.assertEqual(intake["managed_output_receipt_ready_count"], 0)
        self.assertEqual(intake["dense_output_receipt_ready_count"], 0)
        self.assertEqual(intake["live_capability_proof_ready_count"], 0)
        self.assertFalse(intake["all_capture_receipts_ready"])
        self.assertFalse(intake["all_output_receipt_bindings_ready"])
        self.assertEqual(intake["receipt_fill_candidate_router_trace_entry_count"], 6)
        self.assertEqual(intake["receipt_fill_candidate_router_trace_ready_count"], 0)
        self.assertEqual(intake["receipt_fill_candidate_router_trace_missing_count"], 6)
        self.assertEqual(intake["receipt_fill_managed_output_entry_count"], 6)
        self.assertEqual(intake["receipt_fill_managed_output_ready_count"], 0)
        self.assertEqual(intake["receipt_fill_managed_output_missing_count"], 6)
        self.assertEqual(intake["receipt_fill_dense_output_entry_count"], 6)
        self.assertEqual(intake["receipt_fill_dense_output_ready_count"], 0)
        self.assertEqual(intake["receipt_fill_dense_output_missing_count"], 6)
        handoff = summary["input_summaries"]["handoff_scaffold_coverage"]
        self.assertEqual(handoff["bundle_count"], 6)
        self.assertEqual(handoff["handoff_scaffold_ready_count"], 6)
        self.assertTrue(handoff["all_handoff_scaffolds_ready"])
        self.assertEqual(handoff["runtime_capture_launch_card_template_ready_count"], 6)
        self.assertEqual(handoff["runtime_capture_launch_card_binding_ready_count"], 0)
        self.assertEqual(handoff["runtime_capture_launch_card_runtime_command_ready_count"], 0)
        self.assertEqual(handoff["runtime_capture_launch_card_command_option_count"], 0)
        self.assertEqual(handoff["runtime_capture_launch_card_missing_runtime_command_count"], 18)
        launch_card = summary["input_summaries"]["launch_card_library"]
        self.assertTrue(launch_card["library_ready"])
        self.assertFalse(launch_card["execution_ready"])
        self.assertTrue(launch_card["binding_handoff_ready"])
        self.assertTrue(launch_card["model_plane_artifact_writer_contract_request_ready"])
        self.assertTrue(launch_card["saved_handoff_artifacts_ready"])
        self.assertEqual(launch_card["saved_handoff_artifact_missing_count"], 0)
        self.assertEqual(launch_card["saved_handoff_artifact_drifted_count"], 0)
        self.assertEqual(launch_card["missing_runtime_command_count"], 18)
        self.assertEqual(
            handoff["runtime_capture_launch_card_missing_runtime_command_artifact_ids"],
            ["candidate_router_trace", "dense_output_summary_fill", "managed_output_summary_fill"],
        )
        runtime = summary["input_summaries"]["runtime_capture_request_audit"]
        self.assertEqual(runtime["request_count"], 6)
        self.assertEqual(runtime["drifted_request_count"], 0)
        self.assertTrue(runtime["all_required_validator_commands_present"])
        spike = summary["input_summaries"]["runtime_actuator_spike"]
        self.assertTrue(spike["spike_handoff_ready"])
        self.assertFalse(spike["live_spike_ready"])
        self.assertEqual(spike["proof_requirement_count"], 8)
        self.assertEqual(spike["proof_artifact_count"], 20)
        self.assertEqual(spike["dependency_edge_count"], 14)
        self.assertEqual(spike["blocking_capability_count"], 7)
        self.assertEqual(spike["control_blocker_count"], 3)

    def test_matrix_receipt_counts_control_policy_candidate_receipt_reason(self) -> None:
        replay_summary = {"valid": True, "candidate_policy_ids": []}
        pairing_summary = {"valid": True, "real_model_pair_ready": True}
        fallback_summary = {"valid": True, "comparison_ready": True}
        managed_summary = {"valid": True, "live_expert_loading_implemented": False, "capability_statuses": {}}
        live_proof_summary = {"valid": True, "proof_ready": False, "blockers": []}
        real_evidence_summary = {
            "valid": True,
            "bundle_count": 1,
            "policy_candidate_ready_count": 1,
            "policy_candidate_trace_receipt_required_count": 1,
            "policy_candidate_trace_receipt_ready_count": 0,
            "policy_candidate_trace_receipt_attached_count": 0,
            "policy_candidate_trace_receipt_blocked_bundle_count": 1,
            "bundles": [
                {
                    "name": "candidate bundle",
                    "path": "memory-moe-mvp/phase3-real-evidence/candidate.json",
                    "model_id": "candidate-mixtral.gguf",
                    "candidate_policy_ids": ["preload_shortlist"],
                    "policy_candidate_ready": True,
                    "policy_candidate_blockers": [],
                    "reuse_distance_observations": 8,
                    "replay_valid": True,
                }
            ],
        }

        blocked_receipt = planner.policy_candidate_trace_receipt_gate_summary({}, real_evidence_summary)
        blocked_reasons = planner.collect_no_go_reasons(
            replay_summary,
            pairing_summary,
            fallback_summary,
            managed_summary,
            live_proof_summary,
            real_evidence_summary,
            None,
            None,
            None,
            blocked_receipt,
        )
        self.assertIn("policy_candidate_trace_receipt_not_ready", {item["id"] for item in blocked_reasons})

        real_evidence_summary["policy_candidate_trace_receipt_ready_count"] = 1
        real_evidence_summary["policy_candidate_trace_receipt_attached_count"] = 1
        real_evidence_summary["policy_candidate_trace_receipt_blocked_bundle_count"] = 0
        ready_receipt = planner.policy_candidate_trace_receipt_gate_summary({}, real_evidence_summary)
        ready_reasons = planner.collect_no_go_reasons(
            replay_summary,
            pairing_summary,
            fallback_summary,
            managed_summary,
            live_proof_summary,
            real_evidence_summary,
            None,
            None,
            None,
            ready_receipt,
        )
        self.assertNotIn("policy_candidate_trace_receipt_not_ready", {item["id"] for item in ready_reasons})
        self.assertTrue(ready_receipt["ready"])
        self.assertEqual(ready_receipt["source"], "repo_real_evidence_matrix")

    def test_valid_fallback_artifact_clears_fallback_no_go_reason_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fallback_path = Path(temp_dir) / "fallback.json"
            fallback_path.write_text(json.dumps(fallback_artifact()), encoding="utf-8")
            summary = planner.build_decision_summary(
                TRACE_FIXTURE,
                INVENTORY_FIXTURE,
                POLICIES_FIXTURE,
                MANAGED_PLAN,
                fallback_artifact_path=fallback_path,
                real_evidence_root=None,
            )

        self.assertTrue(summary["valid"], summary["errors"])
        reason_ids = {item["id"] for item in summary["no_go_reasons"]}
        self.assertNotIn("dense_fallback_comparison_not_ready", reason_ids)
        self.assertIn("real_model_trace_inventory_pairing_not_ready", reason_ids)
        self.assertNotIn("capture_result_intake_not_ready", reason_ids)
        self.assertIn("live_capability_proof_not_ready", reason_ids)
        self.assertFalse(summary["ready_for_live_spike"])
        self.assertTrue(summary["gate_status"]["dense_fallback_comparison_ready"])
        self.assertEqual(summary["input_summaries"]["pairing"]["decision_pairing_basis"], "default_trace_inventory_pairing")
        self.assertIsNone(summary["gate_status"]["capture_result_intake_ready"])
        self.assertIsNone(summary["gate_status"]["handoff_scaffold_coverage_ready"])
        self.assertIsNone(summary["gate_status"]["launch_card_library_ready"])
        self.assertIsNone(summary["gate_status"]["runtime_capture_request_audit_ready"])

    def test_invalid_fallback_artifact_makes_decision_invalid(self) -> None:
        artifact = fallback_artifact()
        artifact["schema_version"] = "old"
        with tempfile.TemporaryDirectory() as temp_dir:
            fallback_path = Path(temp_dir) / "fallback.json"
            fallback_path.write_text(json.dumps(artifact), encoding="utf-8")
            status, summary, error = planner.plan_paths(
                TRACE_FIXTURE,
                INVENTORY_FIXTURE,
                POLICIES_FIXTURE,
                MANAGED_PLAN,
                fallback_artifact_path=fallback_path,
                real_evidence_root=None,
            )

        self.assertEqual(status, 2)
        self.assertIsNone(error)
        assert summary is not None
        self.assertFalse(summary["valid"])
        self.assertTrue(any("fallback comparison" in item for item in summary["errors"]))

    def test_bad_json_returns_clean_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bad_policies = Path(temp_dir) / "policies.json"
            bad_policies.write_text("{", encoding="utf-8")
            status, summary, error = planner.plan_paths(
                TRACE_FIXTURE,
                INVENTORY_FIXTURE,
                bad_policies,
                MANAGED_PLAN,
                real_evidence_root=None,
            )

        self.assertEqual(status, 2)
        self.assertIsNone(summary)
        assert error is not None
        self.assertIn("Could not build Phase 3 go/no-go decision", error)

    def test_markdown_report_contains_decision_reasons_and_safety(self) -> None:
        summary = planner.build_decision_summary(
            TRACE_FIXTURE,
            INVENTORY_FIXTURE,
            POLICIES_FIXTURE,
            MANAGED_PLAN,
        )

        report = planner.format_markdown_report(summary)

        self.assertIn("# Phase 3 Go/No-Go Decision", report)
        self.assertIn("Decision: `no_go_live_spike`", report)
        self.assertIn("## Gate Status", report)
        self.assertIn("`no_replay_policy_candidate`", report)
        self.assertIn("`reuse_evidence_capture_not_ready`", report)
        self.assertIn("`reuse_evidence_capture_ready` | `False`", report)
        self.assertIn("`reuse_evidence_reuse_ready_count` | `0`", report)
        self.assertIn("`reuse_evidence_candidate_trace_receipt_ready_count` | `6`", report)
        self.assertIn("`policy_candidate_trace_receipt_scaffold_ready` | `True`", report)
        self.assertIn("`policy_candidate_trace_receipt_scaffold_ready_count` | `6`", report)
        self.assertIn("`policy_candidate_trace_receipt_scaffold_valid_count` | `6`", report)
        self.assertIn("`capture_result_intake_not_ready`", report)
        self.assertIn("`capture_result_intake_ready` | `False`", report)
        self.assertIn("`capture_result_runtime_approval_missing_request_count` | `6`", report)
        self.assertIn("`capture_result_approval_rebuild_command_available_request_count` | `6`", report)
        self.assertIn("`capture_result_approval_rebuild_command_manifest_count` | `6`", report)
        self.assertIn("`capture_result_approved_runtime_capture_pending_request_count` | `0`", report)
        self.assertIn("`capture_result_candidate_trace_receipt_ready_count` | `6`", report)
        self.assertIn("`capture_result_managed_output_receipt_ready_count` | `0`", report)
        self.assertIn("`capture_result_dense_output_receipt_ready_count` | `0`", report)
        self.assertIn("`capture_result_all_capture_receipts_ready` | `False`", report)
        self.assertIn("`capture_result_receipt_fill_candidate_router_trace_ready_count` | `0`", report)
        self.assertIn("`capture_result_receipt_fill_managed_output_ready_count` | `0`", report)
        self.assertIn("`capture_result_receipt_fill_dense_output_ready_count` | `0`", report)
        self.assertIn("`handoff_scaffold_coverage_ready` | `True`", report)
        self.assertIn("`handoff_runtime_capture_launch_card_template_ready_count` | `6`", report)
        self.assertIn("`handoff_runtime_capture_launch_card_binding_ready_count` | `0`", report)
        self.assertIn("`handoff_runtime_capture_launch_card_runtime_command_ready_count` | `0`", report)
        self.assertIn("`handoff_runtime_capture_launch_card_missing_runtime_command_count` | `18`", report)
        self.assertIn("`launch_card_library_ready` | `True`", report)
        self.assertIn("`launch_card_saved_handoff_artifacts_ready` | `True`", report)
        self.assertIn("`launch_card_saved_handoff_artifact_drifted_count` | `0`", report)
        self.assertIn("`runtime_capture_request_audit_ready` | `True`", report)
        self.assertIn("`runtime_actuator_spike_handoff_ready` | `True`", report)
        self.assertIn("`runtime_actuator_spike_live_ready` | `False`", report)
        self.assertIn("`runtime_actuator_spike_proof_requirement_count` | `8`", report)
        self.assertIn("runtime_actuator_spike", report)
        self.assertIn("`policy_candidate_no_reuse_distance_observation_count` | `6`", report)
        self.assertIn("capture_candidate_router_trace_with_reuse_distance", report)
        self.assertIn("## Evidence Summary", report)
        self.assertIn("repo_real_evidence_matrix", report)
        self.assertIn("planner does not launch model servers", report)

    def test_write_markdown_report_writes_decision_packet(self) -> None:
        summary = planner.build_decision_summary(
            TRACE_FIXTURE,
            INVENTORY_FIXTURE,
            POLICIES_FIXTURE,
            MANAGED_PLAN,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "reports" / "go-no-go.md"
            planner.write_markdown_report(summary, output_path)
            report = output_path.read_text(encoding="utf-8")

        self.assertIn("# Phase 3 Go/No-Go Decision", report)
        self.assertIn("Ready for live spike: `False`", report)
        self.assertIn("Keep the current decision as no-go", report)

if __name__ == "__main__":
    unittest.main()
