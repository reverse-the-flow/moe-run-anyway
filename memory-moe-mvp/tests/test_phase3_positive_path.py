import copy
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

plan_phase3_evidence_packet = importlib.import_module("plan_phase3_evidence_packet")
plan_phase3_go_no_go = importlib.import_module("plan_phase3_go_no_go")
plan_phase3_real_evidence_bundle = importlib.import_module("plan_phase3_real_evidence_bundle")
plan_phase3_real_evidence_capture = importlib.import_module("plan_phase3_real_evidence_capture")
plan_real_model_trace_inventory_pairing = importlib.import_module("plan_real_model_trace_inventory_pairing")
plan_baseline_policy_replay = importlib.import_module("plan_baseline_policy_replay")
build_phase3_real_evidence_bundle = importlib.import_module("build_phase3_real_evidence_bundle")

POLICIES_FIXTURE = ROOT / "memory-moe-mvp" / "data" / "baseline_replay_policies.json"
MANAGED_PLAN = ROOT / "memory-moe-mvp" / "data" / "managed_expert_loading_plan.json"
INVENTORY_FIXTURE = ROOT / "memory-moe-mvp" / "data" / "expert_inventory_manifest.fixture.json"
MODEL_ID = "local-mixtral.gguf"


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def local_trace_events():
    base = {
        "backend_family": "llama_cpp",
        "contract_version": "memory-moe-bridge-v1",
        "event_type": "llama_cpp_moe_router_tensor",
        "layer_id": 0,
        "model": MODEL_ID,
        "source": "llama_cpp_eval_callback",
        "timestamp": "2026-06-25T00:00:00Z",
        "values_truncated": False,
        "prompt_id": "case-001",
        "prompt_group_id": "phase3-positive-group",
        "repeat": 0,
        "repeat_index": 0,
        "request_id": "phase3-positive-request",
        "capture_run_id": "phase3-positive-capture-run",
    }
    return [
        {
            **base,
            "ggml_type": "i32",
            "shape": [8, 1, 1, 1],
            "tensor_kind": "selected_experts",
            "tensor_name": "ffn_moe_topk-0",
            "value_count": 8,
            "values": [0, 0, 0, 1, 0, 0, 0, 0],
        },
        {
            **base,
            "ggml_type": "f32",
            "shape": [8, 1, 1, 1],
            "tensor_kind": "selected_weights",
            "tensor_name": "ffn_moe_weights-0",
            "value_count": 8,
            "values": [1.0, 1.0, 1.0, 0.99, 1.0, 1.0, 1.0, 1.0],
        },
        {
            **base,
            "ggml_type": "f32",
            "shape": [8, 1, 1, 1],
            "tensor_kind": "selected_weights_norm",
            "tensor_name": "ffn_moe_weights_norm-0",
            "value_count": 8,
            "values": [1.0, 1.0, 1.0, 0.99, 1.0, 1.0, 1.0, 1.0],
        },
    ]


def local_inventory():
    manifest = json.loads(INVENTORY_FIXTURE.read_text(encoding="utf-8"))
    manifest["name"] = "Local Mixtral Expert Inventory"
    manifest["model_id"] = MODEL_ID
    manifest["inventory_scope"] = "scanner_derived_metadata"
    manifest["source_files"] = [
        {
            "path": MODEL_ID,
            "source_format": "gguf",
            "byte_length": 6291456,
            "note": "Metadata-only scan from local GGUF header.",
        }
    ]
    manifest["expected"] = {
        "layer_ids": [0],
        "required_components": ["gate_proj", "up_proj", "down_proj"],
        "expert_count": 2,
        "component_count": 6,
        "total_estimated_residency_bytes": 6291456,
        "routing_top_k": 1,
    }
    entries = []
    byte_offset = 0
    for entry in manifest["entries"]:
        if entry["layer_id"] != 0 or entry["expert_id"] not in {0, 1}:
            continue
        copied = copy.deepcopy(entry)
        copied["source_file"] = MODEL_ID
        copied["byte_offset"] = byte_offset
        copied["byte_length"] = 1048576
        copied["stride_bytes"] = 1048576
        copied["estimated_residency_bytes"] = 1048576
        copied["coverage_status"] = "complete"
        byte_offset += 1048576
        entries.append(copied)
    manifest["entries"] = entries
    manifest["safety_contract"] = [
        "Inventory was generated from metadata only.",
        "Tensor values were not loaded.",
        "Inventory does not claim live expert paging.",
    ]
    manifest["next_actions"] = [
        "Pair with semantic router trace.",
        "Replay baseline residency policies.",
    ]
    return manifest


def fallback_artifact():
    return {
        "schema_version": "moe-dense-fallback-comparison-v1",
        "model_id": MODEL_ID,
        "prompt_family": "phase3-positive-path",
        "managed_policy_id": "preload_shortlist",
        "managed_artifact": "memory-moe-mvp/phase3-real-evidence/managed-output.json",
        "dense_artifact": "memory-moe-mvp/phase3-real-evidence/dense-output.json",
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


def trace_receipt(trace_path: Path, request_path: Path, prompt_path: Path):
    return {
        "receipt_ready": True,
        "source_request_path": str(request_path),
        "source_prompt_set_path": str(prompt_path),
        "candidate_trace_path": str(trace_path),
        "router_trace_capture_approved": True,
        "runtime_prompt_traffic_approved": True,
        "captured_at": "2026-06-30T00:00:00Z",
        "capture_host": "positive-path-fixture",
        "runtime_backend": "llama_cpp",
        "model_id": MODEL_ID,
        "prompt_family": "phase3-positive-path",
        "operator_notes": "Positive-path test artifact only.",
    }


class Phase3PositivePathTests(unittest.TestCase):
    def test_positive_artifacts_promote_phase3_to_phase4_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            trace_path = temp_path / "router-events.jsonl"
            inventory_path = temp_path / "expert-inventory.json"
            fallback_path = temp_path / "fallback-comparison.json"
            trace_receipt_path = temp_path / "router-events.capture-receipt.json"
            request_source_path = temp_path / "positive.runtime-capture-request.json"
            prompt_source_path = temp_path / "positive.prompt-set.json"
            bundle_path = temp_path / "phase3-bundle.json"

            write_jsonl(trace_path, local_trace_events())
            write_json(inventory_path, local_inventory())
            write_json(fallback_path, fallback_artifact())
            write_json(request_source_path, {"schema_version": "positive-runtime-capture-request"})
            write_json(prompt_source_path, {"schema_version": "positive-prompt-set", "model_id": MODEL_ID})
            write_json(trace_receipt_path, trace_receipt(trace_path, request_source_path, prompt_source_path))

            pairing = plan_real_model_trace_inventory_pairing.build_pairing_summary(trace_path, inventory_path)
            self.assertTrue(pairing["valid"], pairing["errors"])
            self.assertTrue(pairing["real_model_pair_ready"])
            self.assertFalse(pairing["fixture_only_pair"])

            replay = plan_baseline_policy_replay.build_replay_summary(
                trace_path,
                inventory_path,
                POLICIES_FIXTURE,
            )
            self.assertTrue(replay["valid"], replay["errors"])
            self.assertTrue(replay["phase_3_gate"]["policy_candidate_ready"])
            self.assertIn("preload_shortlist", replay["candidate_policy_ids"])
            self.assertNotIn("no_replay_policy_candidate", replay["phase_3_gate"]["missing_for_phase_3_completion"])

            missing_receipt_decision = plan_phase3_go_no_go.build_decision_summary(
                trace_path,
                inventory_path,
                POLICIES_FIXTURE,
                MANAGED_PLAN,
                fallback_artifact_path=fallback_path,
                real_evidence_root=None,
            )
            self.assertTrue(missing_receipt_decision["valid"], missing_receipt_decision["errors"])
            self.assertFalse(missing_receipt_decision["ready_for_phase4_adapter_spike"])
            missing_receipt_reason_ids = {item["id"] for item in missing_receipt_decision["no_go_reasons"]}
            self.assertIn("policy_candidate_trace_receipt_not_ready", missing_receipt_reason_ids)
            self.assertFalse(missing_receipt_decision["gate_status"]["policy_candidate_trace_receipt_ready"])

            decision = plan_phase3_go_no_go.build_decision_summary(
                trace_path,
                inventory_path,
                POLICIES_FIXTURE,
                MANAGED_PLAN,
                fallback_artifact_path=fallback_path,
                policy_candidate_trace_receipt_path=trace_receipt_path,
                real_evidence_root=None,
                expected_live_proof_model_id=MODEL_ID,
                expected_live_proof_backend_family="llama_cpp",
                expected_live_proof_prompt_family="phase3-positive-path",
            )
            self.assertTrue(decision["valid"], decision["errors"])
            self.assertTrue(decision["ready_for_phase4_adapter_spike"])
            self.assertTrue(decision["gate_status"]["policy_candidate_trace_receipt_ready"])
            self.assertEqual(decision["gate_status"]["policy_candidate_trace_receipt_path"], str(trace_receipt_path))
            self.assertFalse(decision["ready_for_live_spike"])
            reason_ids = {item["id"] for item in decision["no_go_reasons"]}
            self.assertNotIn("no_replay_policy_candidate", reason_ids)
            self.assertNotIn("real_model_trace_inventory_pairing_not_ready", reason_ids)
            self.assertNotIn("dense_fallback_comparison_not_ready", reason_ids)
            self.assertIn("live_capability_proof_not_ready", reason_ids)
            self.assertIn("live_actuator_missing", reason_ids)

            packet = plan_phase3_evidence_packet.build_packet_summary(
                trace_path,
                inventory_path,
                POLICIES_FIXTURE,
                MANAGED_PLAN,
                fallback_artifact_path=fallback_path,
                policy_candidate_trace_receipt_path=trace_receipt_path,
                include_repo_gates=False,
                expected_model_id=MODEL_ID,
                expected_backend_family="llama_cpp",
                expected_prompt_family="phase3-positive-path",
            )
            self.assertTrue(packet["valid"], packet["errors"])
            self.assertTrue(packet["phase3_complete"])
            self.assertEqual(packet["promotion_checklist"], [])
            self.assertIsNone(packet["recommended_runtime_capture_request"])
            self.assertEqual(packet["status_counts"], {"future_phase_4": 1, "proven": 12})
            by_item = {item["id"]: item for item in packet["evidence_items"]}
            self.assertEqual(by_item["live_capability_proof"]["status"], "future_phase_4")
            self.assertEqual(by_item["phase3_blocker_closure_manifest"]["status"], "proven")
            self.assertEqual(by_item["phase3_blocker_evidence_ledger"]["status"], "proven")
            self.assertEqual(by_item["phase3_blocker_resolution_queue"]["status"], "proven")
            blocker_queue = packet["phase3_blocker_resolution_queue_summary"]
            self.assertTrue(blocker_queue["ready"])
            self.assertEqual(blocker_queue["work_package_count"], 2)
            self.assertEqual(blocker_queue["dependency_edge_count"], 1)
            self.assertEqual(blocker_queue["missing_item_count"], 0)
            packages = {
                item["work_package_id"]: item
                for item in packet["phase3_blocker_resolution_queue_manifest"]["work_packages"]
            }
            self.assertEqual(packages["runtime_actuator_spike"]["depends_on_work_package_ids"], [])
            self.assertEqual(
                packages["live_capability_proof_fill"]["depends_on_work_package_ids"],
                ["runtime_actuator_spike"],
            )
            self.assertTrue(packet["remaining_gaps"]["ready_for_phase4_adapter_spike"])
            self.assertTrue(packet["remaining_gaps"]["policy_candidate_trace_receipt_ready"])
            self.assertTrue(packet["remaining_gaps"]["policy_candidate_trace_decision_receipt_ready"])
            self.assertFalse(packet["remaining_gaps"]["live_capability_proof_ready"])
            self.assertFalse(packet["remaining_gaps"]["ready_for_live_spike"])

            manifest = build_phase3_real_evidence_bundle.build_manifest(
                name="Positive Phase 3 bundle",
                model_id=MODEL_ID,
                source_format="gguf",
                backend_family="llama_cpp",
                prompt_family="phase3-positive-path",
                trace_path=trace_path,
                trace_receipt_path=trace_receipt_path,
                inventory_path=inventory_path,
                policies_path=POLICIES_FIXTURE,
                managed_plan_path=MANAGED_PLAN,
                fallback_artifact_path=fallback_path,
                real_model_trace_capture_approved=True,
                dense_fallback_capture_approved=True,
                runtime_prompt_traffic_approved=True,
                approval_notes=["Positive-path test artifact only."],
            )
            write_json(bundle_path, manifest)
            bundle = plan_phase3_real_evidence_bundle.build_summary(bundle_path)
            self.assertTrue(bundle["valid"], bundle["errors"])
            self.assertTrue(bundle["phase3_complete"])
            self.assertTrue(bundle["bundle_ready_for_phase4"])
            self.assertEqual(bundle["artifact_statuses"]["trace_receipt_path"]["status"], "present")
            self.assertNotIn("policy_candidate_trace_receipt_not_ready", bundle["no_go_reason_ids"])
            self.assertNotIn("fallback_artifact_not_attached", bundle["no_go_reason_ids"])

            capture = plan_phase3_real_evidence_capture.build_capture_plan(
                trace_path,
                inventory_path,
                POLICIES_FIXTURE,
                MANAGED_PLAN,
                fallback_artifact_path=fallback_path,
                policy_candidate_trace_receipt_path=trace_receipt_path,
                model_path=Path(MODEL_ID),
                model_id=MODEL_ID,
                source_format="gguf",
                output_dir=temp_path / "capture-plan-output",
            )
            self.assertTrue(capture["valid"], capture["errors"])
            self.assertTrue(capture["phase3_complete"])
            self.assertEqual(capture["blocking_request_ids"], [])
            self.assertEqual(capture["future_request_ids"], ["live_residency_capability_proof"])


if __name__ == "__main__":
    unittest.main()
