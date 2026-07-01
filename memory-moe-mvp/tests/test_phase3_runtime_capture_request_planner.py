import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_phase3_runtime_capture_request.py"
SPEC = importlib.util.spec_from_file_location("plan_phase3_runtime_capture_request", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)

import build_phase3_runtime_capture_request


BUNDLE = ROOT / "memory-moe-mvp" / "phase3-real-evidence" / "pc_mixtral_phase3_real_evidence_bundle.json"
REQUEST = ROOT / "memory-moe-mvp" / "phase3-real-evidence" / "pc_mixtral_phase3_real_evidence_bundle.runtime-capture-request.json"


def minimal_audit_summary() -> dict:
    return {
        "valid": True,
        "request_count": 1,
        "valid_request_count": 1,
        "ready_for_operator_capture_count": 0,
        "capture_complete_count": 0,
        "drifted_request_count": 0,
        "validator_command_count": 1,
        "validator_command_missing_request_count": 1,
        "all_required_validator_commands_present": False,
        "approval_queue_count": 1,
        "approval_rebuild_command_manifest_count": 1,
        "approval_rebuild_command_manifest": [
            {
                "queue_rank": 1,
                "request_name": "Fixture request",
                "request_path": "memory-moe-mvp/phase3-real-evidence/fixture.runtime-capture-request.json",
                "bundle_path": "memory-moe-mvp/phase3-real-evidence/fixture-bundle.json",
                "status": "approval_required",
                "next_artifact_id": "candidate_router_trace",
                "command_class": "phase3_runtime_capture_request_approval_rebuild",
                "writes_request_path": "memory-moe-mvp/phase3-real-evidence/fixture.runtime-capture-request.json",
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
                    "memory-moe-mvp/phase3-real-evidence/fixture-bundle.json",
                    "--output",
                    "memory-moe-mvp/phase3-real-evidence/fixture.runtime-capture-request.json",
                    "--json",
                    "--runtime-prompt-traffic-approved",
                ],
            }
        ],
        "recommended_runtime_capture_request": {
            "request_name": "Fixture request",
            "status": "approval_required",
            "bundle_path": "memory-moe-mvp/phase3-real-evidence/fixture-bundle.json",
            "model_id": "fixture/model",
            "prompt_set_path": "memory-moe-mvp/phase3-real-evidence/fixture.prompt-set.json",
            "prompt_count": 2,
            "missing_approval_keys": ["runtime_prompt_traffic_approved"],
            "pending_artifact_ids": ["candidate_router_trace"],
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
                    "memory-moe-mvp/phase3-real-evidence/fixture-bundle.json",
                    "--output",
                    "memory-moe-mvp/phase3-real-evidence/fixture.runtime-capture-request.json",
                    "--json",
                    "--runtime-prompt-traffic-approved",
                ],
                "records_approval_keys": ["runtime_prompt_traffic_approved"],
                "writes_request_path": "memory-moe-mvp/phase3-real-evidence/fixture.runtime-capture-request.json",
                "requires_explicit_user_approval": True,
                "metadata_only": True,
            },
            "capture_sequence": [
                {
                    "id": "record_runtime_approvals",
                    "stage": "approval",
                    "status": "approval_required",
                    "approval_keys": ["runtime_prompt_traffic_approved"],
                },
                {
                    "id": "capture_candidate_router_trace",
                    "stage": "runtime_capture",
                    "status": "approval_required",
                    "path": "memory-moe-mvp/phase3-real-evidence/fixture/candidate-router-events.jsonl",
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
            "request_name": "Fixture request",
            "request_path": "memory-moe-mvp/phase3-real-evidence/fixture.runtime-capture-request.json",
            "status": "ready_for_operator_capture",
            "ready_for_operator_capture": True,
            "capture_complete": False,
            "missing_approval_keys_after_preview": [],
            "records_approval_keys": ["runtime_prompt_traffic_approved"],
            "requested_status_counts": {"ready_for_operator_capture": 1},
            "pending_artifact_ids": ["candidate_router_trace"],
            "pending_artifact_count": 1,
            "next_artifact_id": "candidate_router_trace",
            "next_artifact_path": "memory-moe-mvp/phase3-real-evidence/fixture/candidate-router-events.jsonl",
            "mutates_request": False,
            "still_requires_capture_artifacts": True,
            "artifact_statuses": [
                {
                    "id": "candidate_router_trace",
                    "status": "ready_for_operator_capture",
                    "path": "memory-moe-mvp/phase3-real-evidence/fixture/candidate-router-events.jsonl",
                }
            ],
            "safety_contract": ["preview does not send prompt traffic"],
        },
        "capture_queue_summary": {
            "queue_count": 1,
            "status_counts": {"approval_required": 1},
            "approval_required_count": 1,
            "ready_for_operator_capture_count": 0,
            "artifact_capture_pending_count": 0,
            "approved_but_capture_incomplete_count": 0,
            "blocked_by_drift_count": 0,
            "backend_family_counts": {"llama_cpp": 1},
            "prompt_count_min": 2,
            "prompt_count_max": 2,
            "recommended_rank": 1,
            "selection_contract": ["Prefer the known Mixtral sparse baseline first when all capture requests are otherwise equivalent."],
            "ranked_requests": [
                {
                    "rank": 1,
                    "request_name": "Fixture request",
                    "status": "approval_required",
                    "selection_rationale": "mixtral_first_known_sparse_baseline",
                    "next_artifact_id": "candidate_router_trace",
                    "pending_artifact_count": 1,
                    "missing_approval_count": 1,
                    "prompt_count": 2,
                    "backend_family": "llama_cpp",
                }
            ],
        },
        "requests": [
            {
                "name": "Fixture request",
                "bundle_path": "memory-moe-mvp/phase3-real-evidence/fixture-bundle.json",
                "model_id": "fixture/model",
                "backend_family": "llama_cpp",
                "prompt_family": "fixture-prompts",
                "ready_for_operator_capture": False,
                "capture_complete": False,
                "drift_count": 0,
                "validator_command_coverage": {
                    "artifact_count": 4,
                    "covered_artifact_count": 1,
                    "validator_command_count": 1,
                    "missing_validator_command_artifact_ids": [
                        "managed_output_summary_fill",
                        "dense_output_summary_fill",
                        "live_capability_proof_fill",
                    ],
                    "all_required_validator_commands_present": False,
                    "artifacts": [],
                },
                "operator_handoff": {
                    "approval_summary": "runtime_prompt_traffic_approved=false",
                    "prompt_set_path": "memory-moe-mvp/phase3-real-evidence/fixture.prompt-set.json",
                    "prompt_count": 2,
                    "requested_artifacts": [
                        {
                            "id": "candidate_router_trace",
                            "status": "approval_required",
                            "approval_required": True,
                            "path": "memory-moe-mvp/phase3-real-evidence/fixture/candidate-router-events.jsonl",
                            "source": {
                                "prompt_set_path": "memory-moe-mvp/phase3-real-evidence/fixture.prompt-set.json",
                                "capture_receipt_required": True,
                                "capture_receipt_path": "memory-moe-mvp/phase3-real-evidence/fixture/candidate-router-events.capture-receipt.json",
                                "capture_receipt_ready": False,
                                "capture_receipt_blockers": ["candidate_trace_capture_receipt_not_ready"],
                                "receipt_fill_note": "trace receipt note",
                            },
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
                        }
                    ],
                    "future_artifacts": [
                        {
                            "id": "live_capability_proof_fill",
                            "status": "future_adapter_required",
                            "approval_required": True,
                            "path": "memory-moe-mvp/phase3-real-evidence/fixture.live-capability-proof.template.json",
                            "validator_commands": [],
                        }
                    ],
                    "next_actions": ["Capture only after explicit approval."],
                },
            }
        ],
        "safety_contract": ["metadata only"],
    }


class Phase3RuntimeCaptureRequestPlannerTests(unittest.TestCase):
    def test_current_repo_requests_validate_without_runtime_side_effects(self) -> None:
        summary = planner.build_root_summary()

        self.assertTrue(summary["valid"], json.dumps(summary["errors"], indent=2))
        self.assertEqual(summary["request_count"], 6)
        self.assertEqual(summary["valid_request_count"], 6)
        self.assertEqual(summary["ready_for_operator_capture_count"], 0)
        self.assertEqual(summary["capture_complete_count"], 0)
        self.assertEqual(summary["drifted_request_count"], 0)
        self.assertEqual(summary["validator_command_count"], 36)
        self.assertEqual(summary["validator_command_missing_request_count"], 0)
        self.assertTrue(summary["all_required_validator_commands_present"])
        self.assertEqual(summary["validator_command_coverage"]["artifact_count"], 24)
        self.assertEqual(summary["validator_command_coverage"]["covered_artifact_count"], 24)
        self.assertEqual(summary["approval_queue_count"], 6)
        self.assertEqual(summary["approval_rebuild_command_manifest_count"], 6)
        approval_manifest = summary["approval_rebuild_command_manifest"]
        self.assertEqual(len(approval_manifest), 6)
        self.assertEqual(summary["next_artifact_approval_rebuild_command_manifest_count"], 6)
        next_artifact_manifest = summary["next_artifact_approval_rebuild_command_manifest"]
        self.assertEqual(len(next_artifact_manifest), 6)
        self.assertEqual(approval_manifest[0]["request_name"], "PC Mixtral Phase 3 real evidence")
        self.assertEqual(approval_manifest[0]["queue_rank"], 1)
        self.assertEqual(approval_manifest[0]["command_class"], "phase3_runtime_capture_request_approval_rebuild")
        self.assertTrue(approval_manifest[0]["metadata_only"])
        self.assertTrue(approval_manifest[0]["requires_explicit_user_approval"])
        self.assertEqual(approval_manifest[0]["approval_scope"], "all_missing_runtime_approvals")
        self.assertTrue(approval_manifest[0]["records_all_missing_approvals"])
        self.assertIn("--runtime-prompt-traffic-approved", approval_manifest[0]["command"])
        self.assertEqual(next_artifact_manifest[0]["request_name"], "PC Mixtral Phase 3 real evidence")
        self.assertEqual(next_artifact_manifest[0]["approval_scope"], "next_runtime_artifact")
        self.assertEqual(next_artifact_manifest[0]["approval_scope_artifact_id"], "candidate_router_trace")
        self.assertFalse(next_artifact_manifest[0]["records_all_missing_approvals"])
        self.assertEqual(
            next_artifact_manifest[0]["records_approval_keys"],
            ["runtime_prompt_traffic_approved", "router_trace_capture_approved"],
        )
        self.assertIn("--runtime-prompt-traffic-approved", next_artifact_manifest[0]["command"])
        self.assertIn("--router-trace-capture-approved", next_artifact_manifest[0]["command"])
        self.assertNotIn("--managed-output-capture-approved", next_artifact_manifest[0]["command"])
        self.assertNotIn("--dense-output-capture-approved", next_artifact_manifest[0]["command"])
        queue_summary = summary["capture_queue_summary"]
        self.assertEqual(queue_summary["queue_count"], 6)
        self.assertEqual(queue_summary["recommended_rank"], 1)
        self.assertEqual(queue_summary["status_counts"], {"approval_required": 6})
        self.assertEqual(queue_summary["approval_required_count"], 6)
        self.assertEqual(queue_summary["ready_for_operator_capture_count"], 0)
        self.assertEqual(queue_summary["approved_but_capture_incomplete_count"], 0)
        self.assertEqual(queue_summary["blocked_by_drift_count"], 0)
        self.assertEqual(queue_summary["backend_family_counts"], {"llama_cpp": 6})
        self.assertEqual(queue_summary["prompt_count_min"], 8)
        self.assertEqual(queue_summary["prompt_count_max"], 8)
        self.assertEqual(queue_summary["ranked_requests"][0]["request_name"], "PC Mixtral Phase 3 real evidence")
        self.assertEqual(queue_summary["ranked_requests"][0]["selection_rationale"], "mixtral_first_known_sparse_baseline")
        self.assertEqual(queue_summary["ranked_requests"][-1]["rank"], 6)
        recommended = summary["recommended_runtime_capture_request"]
        self.assertEqual(recommended["request_name"], "PC Mixtral Phase 3 real evidence")
        self.assertEqual(recommended["status"], "approval_required")
        self.assertEqual(recommended["next_artifact_id"], "candidate_router_trace")
        self.assertEqual(recommended["queue_rank"], 1)
        self.assertEqual(recommended["selection_rationale"], "mixtral_first_known_sparse_baseline")
        approval_command = recommended["approval_rebuild_command"]
        self.assertEqual(approval_command["command_class"], "phase3_runtime_capture_request_approval_rebuild")
        self.assertTrue(approval_command["metadata_only"])
        self.assertTrue(approval_command["requires_explicit_user_approval"])
        self.assertEqual(approval_command["writes_request_path"], recommended["path"])
        self.assertEqual(approval_command["records_approval_keys"], recommended["missing_approval_keys"])
        self.assertEqual(approval_command["approval_scope"], "all_missing_runtime_approvals")
        self.assertTrue(approval_command["records_all_missing_approvals"])
        next_approval_command = recommended["next_artifact_approval_rebuild_command"]
        self.assertEqual(next_approval_command["approval_scope"], "next_runtime_artifact")
        self.assertEqual(next_approval_command["approval_scope_artifact_id"], "candidate_router_trace")
        self.assertEqual(
            next_approval_command["records_approval_keys"],
            ["runtime_prompt_traffic_approved", "router_trace_capture_approved"],
        )
        self.assertFalse(next_approval_command["records_all_missing_approvals"])
        self.assertIn("scripts/build_phase3_runtime_capture_request.py", approval_command["command"])
        self.assertIn("--output", approval_command["command"])
        self.assertIn(recommended["path"], approval_command["command"])
        self.assertIn("--runtime-prompt-traffic-approved", approval_command["command"])
        self.assertIn("--router-trace-capture-approved", approval_command["command"])
        self.assertIn("--managed-output-capture-approved", approval_command["command"])
        self.assertIn("--dense-output-capture-approved", approval_command["command"])
        self.assertEqual(
            recommended["missing_approval_keys"],
            [
                "runtime_prompt_traffic_approved",
                "router_trace_capture_approved",
                "managed_output_capture_approved",
                "dense_output_capture_approved",
            ],
        )
        post_approval = summary["recommended_post_approval_preview"]
        self.assertTrue(post_approval["preview_only"])
        self.assertTrue(post_approval["valid"], post_approval["errors"])
        self.assertFalse(post_approval["mutates_request"])
        self.assertEqual(post_approval["status"], "ready_for_operator_capture")
        self.assertTrue(post_approval["ready_for_operator_capture"])
        self.assertFalse(post_approval["capture_complete"])
        self.assertEqual(post_approval["missing_approval_keys_after_preview"], [])
        self.assertEqual(post_approval["requested_status_counts"], {"ready_for_operator_capture": 3})
        self.assertEqual(
            post_approval["pending_artifact_ids"],
            ["candidate_router_trace", "managed_output_summary_fill", "dense_output_summary_fill"],
        )
        self.assertEqual(post_approval["next_artifact_id"], "candidate_router_trace")
        self.assertTrue(post_approval["still_requires_capture_artifacts"])
        self.assertEqual(summary["approval_queue"][0]["request_name"], recommended["request_name"])
        self.assertEqual(
            [step["id"] for step in recommended["capture_sequence"]],
            [
                "record_runtime_approvals",
                "capture_candidate_router_trace",
                "fill_managed_output_summary",
                "fill_dense_output_summary",
                "run_capture_result_intake",
                "defer_live_capability_proof",
            ],
        )
        self.assertEqual(recommended["capture_sequence"][0]["status"], "approval_required")
        self.assertEqual(recommended["capture_sequence"][1]["validator_command_count"], 3)
        self.assertIn("capture_receipt", recommended["capture_sequence"][2]["action"])
        self.assertIn("capture_receipt", recommended["capture_sequence"][3]["action"])
        self.assertEqual(recommended["capture_sequence"][4]["status"], "pending_artifacts")
        self.assertEqual(recommended["capture_sequence"][5]["status"], "future_adapter_required")

    def test_saved_request_detects_status_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            request_path = temp_path / REQUEST.name
            payload = json.loads(REQUEST.read_text(encoding="utf-8"))
            drifted = copy.deepcopy(payload)
            drifted["requested_artifacts"][0]["status"] = "already_satisfied"
            request_path.write_text(json.dumps(drifted), encoding="utf-8")

            summary = planner.summarize_request(request_path)

        self.assertFalse(summary["valid"])
        self.assertEqual(summary["drift_count"], 1)
        self.assertTrue(any("candidate_router_trace.status drifted" in error for error in summary["errors"]))

    def test_saved_request_detects_source_summary_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            request_path = temp_path / REQUEST.name
            payload = json.loads(REQUEST.read_text(encoding="utf-8"))
            drifted = copy.deepcopy(payload)
            drifted["requested_artifacts"][0]["source"]["capture_receipt_valid"] = False
            request_path.write_text(json.dumps(drifted), encoding="utf-8")

            summary = planner.summarize_request(request_path)

        self.assertFalse(summary["valid"])
        self.assertEqual(summary["drift_count"], 1)
        self.assertTrue(any("candidate_router_trace.source drifted" in error for error in summary["errors"]))

    def test_saved_request_detects_embedded_prompt_summary_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            request_path = temp_path / REQUEST.name
            payload = json.loads(REQUEST.read_text(encoding="utf-8"))
            drifted = copy.deepcopy(payload)
            drifted["prompt_set"]["prompt_count"] = 999
            request_path.write_text(json.dumps(drifted), encoding="utf-8")

            summary = planner.summarize_request(request_path)

        self.assertFalse(summary["valid"])
        self.assertEqual(summary["drift_count"], 1)
        self.assertTrue(any("prompt_set drifted" in error for error in summary["errors"]))

    def test_builder_output_round_trips_through_saved_request_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            request_path = Path(temp_dir) / "request.json"
            summary = build_phase3_runtime_capture_request.build_request(BUNDLE)
            build_phase3_runtime_capture_request.write_request(request_path, summary)

            audit = planner.summarize_request(request_path)

        self.assertTrue(audit["valid"], audit["errors"])
        self.assertEqual(audit["requested_artifact_ids"], ["candidate_router_trace", "dense_output_summary_fill", "managed_output_summary_fill"])
        self.assertEqual(audit["future_artifact_ids"], ["live_capability_proof_fill"])
        self.assertIn("operator_handoff", audit)
        self.assertTrue(audit["validator_command_coverage"]["all_required_validator_commands_present"])
        self.assertEqual(audit["validator_command_coverage"]["validator_command_count"], 6)
        self.assertEqual(audit["operator_handoff"]["prompt_set_path"], "memory-moe-mvp/phase3-real-evidence/pc_mixtral_phase3_real_evidence_bundle.prompt-set.json")

    def test_approved_request_audit_enters_ready_capture_queue_without_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            request_path = Path(temp_dir) / REQUEST.name
            payload = build_phase3_runtime_capture_request.build_request(
                BUNDLE,
                router_trace_capture_approved=True,
                managed_output_capture_approved=True,
                dense_output_capture_approved=True,
                runtime_prompt_traffic_approved=True,
            )
            build_phase3_runtime_capture_request.write_request(request_path, payload)

            audit = planner.summarize_request(request_path)
            single_status, single_summary, single_error = planner.plan_paths(request_path)
            queue = planner.build_approval_queue([audit])
            queue_summary = planner.capture_queue_summary(queue)

        self.assertTrue(audit["valid"], json.dumps(audit["errors"], indent=2))
        self.assertTrue(audit["ready_for_operator_capture"])
        self.assertFalse(audit["capture_complete"])
        self.assertEqual(audit["drift_count"], 0)
        self.assertEqual(audit["requested_status_counts"], {"ready_for_operator_capture": 3})
        self.assertEqual(planner.missing_runtime_approval_keys(audit), [])
        self.assertEqual(single_status, 0)
        self.assertIsNone(single_error)
        assert single_summary is not None
        self.assertTrue(single_summary["valid"])
        self.assertEqual(single_summary["ready_for_operator_capture_count"], 1)
        self.assertEqual(single_summary["capture_complete_count"], 0)
        self.assertEqual(single_summary["capture_queue_summary"]["ready_for_operator_capture_count"], 1)
        self.assertEqual(single_summary["capture_queue_summary"]["approved_but_capture_incomplete_count"], 1)
        self.assertEqual(single_summary["approval_rebuild_command_manifest_count"], 1)
        self.assertEqual(single_summary["approval_rebuild_command_manifest"][0]["records_approval_keys"], list(planner.RUNTIME_APPROVAL_KEYS))
        self.assertEqual(len(queue), 1)
        recommended = queue[0]
        self.assertEqual(recommended["status"], "ready_for_operator_capture")
        self.assertFalse(recommended["approval_required"])
        self.assertEqual(recommended["missing_approval_keys"], [])
        self.assertEqual(
            recommended["pending_artifact_ids"],
            ["candidate_router_trace", "managed_output_summary_fill", "dense_output_summary_fill"],
        )
        self.assertEqual(recommended["next_artifact_id"], "candidate_router_trace")
        self.assertEqual(recommended["capture_sequence"][0]["status"], "satisfied")
        self.assertEqual(
            [step["status"] for step in recommended["capture_sequence"][1:4]],
            ["ready_for_operator_capture", "ready_for_operator_capture", "ready_for_operator_capture"],
        )
        self.assertEqual(recommended["capture_sequence"][4]["status"], "pending_artifacts")
        self.assertEqual(recommended["approval_rebuild_command"]["records_approval_keys"], list(planner.RUNTIME_APPROVAL_KEYS))
        self.assertEqual(single_summary["recommended_post_approval_preview"]["status"], "ready_for_operator_capture")
        self.assertTrue(single_summary["recommended_post_approval_preview"]["ready_for_operator_capture"])
        self.assertFalse(single_summary["recommended_post_approval_preview"]["capture_complete"])
        self.assertEqual(queue_summary["approval_required_count"], 0)
        self.assertEqual(queue_summary["ready_for_operator_capture_count"], 1)
        self.assertEqual(queue_summary["approved_but_capture_incomplete_count"], 1)
        self.assertEqual(queue_summary["blocked_by_drift_count"], 0)

    def test_markdown_report_contains_operator_handoff(self) -> None:
        report = planner.format_markdown_report(minimal_audit_summary())

        self.assertIn("# Phase 3 Runtime-Capture Request Audit", report)
        self.assertIn("## Recommended First Capture", report)
        self.assertIn("Missing approvals: `runtime_prompt_traffic_approved`", report)
        self.assertIn("Next artifact: `candidate_router_trace`", report)
        self.assertIn("Queue rank: `1`", report)
        self.assertIn("Selection rationale: `mixtral_first_known_sparse_baseline`", report)
        self.assertIn("### Approval Metadata Rebuild", report)
        self.assertIn("phase3_runtime_capture_request_approval_rebuild", report)
        self.assertIn("Records approvals: `runtime_prompt_traffic_approved`", report)
        self.assertIn("scripts/build_phase3_runtime_capture_request.py", report)
        self.assertIn("--runtime-prompt-traffic-approved", report)
        self.assertIn("## Capture Queue Selection", report)
        self.assertIn("## Approval Rebuild Command Manifest", report)
        self.assertIn("| 1 | Fixture request | approval_required | memory-moe-mvp/phase3-real-evidence/fixture.runtime-capture-request.json | runtime_prompt_traffic_approved | True | True |", report)
        self.assertIn('Status counts: `{"approval_required": 1}`', report)
        self.assertIn("Approval required: `1`", report)
        self.assertIn("Ready for operator capture: `0`", report)
        self.assertIn("Approved but capture incomplete: `0`", report)
        self.assertIn("| 1 | Fixture request | approval_required | mixtral_first_known_sparse_baseline | candidate_router_trace | 1 | 1 |", report)
        self.assertIn("### Recommended Capture Sequence", report)
        self.assertIn("### Post-Approval Preview", report)
        self.assertIn("Status after approval: `ready_for_operator_capture`", report)
        self.assertIn("Ready for operator capture after approval: `True`", report)
        self.assertIn("Capture complete after approval: `False`", report)
        self.assertIn("Pending artifacts after approval: `candidate_router_trace`", report)
        self.assertIn("Mutates request: `False`", report)
        self.assertIn("| record_runtime_approvals | approval | approval_required | runtime_prompt_traffic_approved |", report)
        self.assertIn("| run_capture_result_intake | intake | pending_artifacts | none |", report)
        self.assertIn("Validator commands: `1`", report)
        self.assertIn("Requests missing validator commands: `1`", report)
        self.assertIn("All required validator commands present: `False`", report)
        self.assertIn("| Request | Ready | Complete | Drift | Validators | Approvals | Prompt Set |", report)
        self.assertIn("Required validator commands present: `False`", report)
        self.assertIn("candidate_router_trace", report)
        self.assertIn("### Capture Receipt Requirements", report)
        self.assertIn("fixture/candidate-router-events.capture-receipt.json", report)
        self.assertIn("candidate_trace_capture_receipt_not_ready", report)
        self.assertIn("trace receipt note", report)
        self.assertIn("live_capability_proof_fill", report)
        self.assertIn("uv run --managed-python --python 3.13 scripts/validate_llama_cpp_router_trace.py fixture.jsonl", report)
        self.assertIn("metadata only", report)

    def test_markdown_report_writes_requested_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "handoff" / "runtime-capture.md"

            planner.write_markdown_report(minimal_audit_summary(), output_path)
            written = output_path.read_text(encoding="utf-8")

        self.assertIn("# Phase 3 Runtime-Capture Request Audit", written)
        self.assertIn("Fixture request", written)


if __name__ == "__main__":
    unittest.main()
