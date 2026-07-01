import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_phase3_real_evidence_matrix.py"
SPEC = importlib.util.spec_from_file_location("plan_phase3_real_evidence_matrix", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)

TRACE_FIXTURE = ROOT / "memory-moe-mvp" / "data" / "llama_cpp_router_trace.fixture.jsonl"
INVENTORY_FIXTURE = ROOT / "memory-moe-mvp" / "data" / "expert_inventory_manifest.fixture.json"
POLICIES_FIXTURE = ROOT / "memory-moe-mvp" / "data" / "baseline_replay_policies.json"
MANAGED_PLAN = ROOT / "memory-moe-mvp" / "data" / "managed_expert_loading_plan.json"


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_real_trace_and_inventory(root: Path, model_id: str) -> tuple[Path, Path]:
    trace_rows = []
    for line in TRACE_FIXTURE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["model"] = model_id
        trace_rows.append(row)
    trace_path = root / "router-events.jsonl"
    trace_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in trace_rows) + "\n", encoding="utf-8")

    inventory = json.loads(INVENTORY_FIXTURE.read_text(encoding="utf-8"))
    inventory["name"] = "Local Mixtral Expert Inventory"
    inventory["model_id"] = model_id
    inventory["inventory_scope"] = "scanner_derived_metadata"
    inventory["source_files"] = [
        {
            "path": model_id,
            "source_format": "gguf",
            "byte_length": inventory["source_files"][0]["byte_length"],
            "note": "Metadata-only scan from local GGUF header.",
        }
    ]
    for entry in inventory["entries"]:
        entry["source_file"] = model_id
    inventory_path = root / "local-mixtral.expert_inventory.json"
    write_json(inventory_path, inventory)
    return trace_path, inventory_path



def make_candidate_trace_and_inventory(root: Path, model_id: str) -> tuple[Path, Path]:
    base = {
        "backend_family": "llama_cpp",
        "contract_version": "memory-moe-bridge-v1",
        "event_type": "llama_cpp_moe_router_tensor",
        "layer_id": 0,
        "model": model_id,
        "source": "llama_cpp_eval_callback",
        "timestamp": "2026-06-30T00:00:00Z",
        "values_truncated": False,
    }
    trace_rows = [
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
    trace_path = root / "candidate-router-events.jsonl"
    trace_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in trace_rows), encoding="utf-8")

    inventory = json.loads(INVENTORY_FIXTURE.read_text(encoding="utf-8"))
    inventory["name"] = "Local Candidate Mixtral Expert Inventory"
    inventory["model_id"] = model_id
    inventory["inventory_scope"] = "scanner_derived_metadata"
    inventory["source_files"] = [
        {
            "path": model_id,
            "source_format": "gguf",
            "byte_length": 6291456,
            "note": "Metadata-only scan from local GGUF header.",
        }
    ]
    inventory["expected"] = {
        "layer_ids": [0],
        "required_components": ["gate_proj", "up_proj", "down_proj"],
        "expert_count": 2,
        "component_count": 6,
        "total_estimated_residency_bytes": 6291456,
        "routing_top_k": 1,
    }
    entries = []
    byte_offset = 0
    for entry in inventory["entries"]:
        if entry["layer_id"] != 0 or entry["expert_id"] not in {0, 1}:
            continue
        copied = copy.deepcopy(entry)
        copied["source_file"] = model_id
        copied["byte_offset"] = byte_offset
        copied["byte_length"] = 1048576
        copied["stride_bytes"] = 1048576
        copied["estimated_residency_bytes"] = 1048576
        copied["coverage_status"] = "complete"
        byte_offset += 1048576
        entries.append(copied)
    inventory["entries"] = entries
    inventory_path = root / "candidate-mixtral.expert_inventory.json"
    write_json(inventory_path, inventory)
    return trace_path, inventory_path


def bundle_manifest(trace_path: Path, inventory_path: Path, model_id: str) -> dict:
    return {
        "schema_version": "moe-phase3-real-evidence-bundle-v1",
        "name": "Local Mixtral Matrix Test",
        "model_id": model_id,
        "source_format": "gguf",
        "backend_family": "llama_cpp",
        "prompt_family": "matrix-test",
        "artifact_paths": {
            "trace_path": str(trace_path),
            "inventory_path": str(inventory_path),
            "policies_path": str(POLICIES_FIXTURE),
            "managed_plan_path": str(MANAGED_PLAN),
            "fallback_artifact_path": None,
            "trace_receipt_path": None,
        },
        "approvals": {
            "real_model_trace_capture_approved": True,
            "dense_fallback_capture_approved": False,
            "runtime_prompt_traffic_approved": False,
            "notes": ["Unit test bundle only."],
        },
        "safety_contract": ["Bundle validation reads local artifacts only."],
        "next_actions": ["Attach dense fallback comparison before Phase 4."],
    }


class Phase3RealEvidenceMatrixPlannerTests(unittest.TestCase):
    def test_matrix_reports_real_pair_and_unreferenced_scanner_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_id = "local-mixtral.gguf"
            trace_path, inventory_path = make_real_trace_and_inventory(root, model_id)
            bundle_path = root / "local_phase3_real_evidence_bundle.json"
            write_json(bundle_path, bundle_manifest(trace_path, inventory_path, model_id))

            invalid_inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            invalid_inventory["entries"] = [
                entry for entry in invalid_inventory["entries"] if entry["component_name"] != "gate_proj"
            ]
            invalid_inventory["expected"] = copy.deepcopy(invalid_inventory["expected"])
            invalid_inventory["expected"]["component_count"] = len(invalid_inventory["entries"])
            invalid_path = root / "invalid-gap.expert_inventory.json"
            write_json(invalid_path, invalid_inventory)

            summary = planner.build_matrix(root)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["bundle_count"], 1)
        self.assertEqual(summary["real_model_pair_ready_count"], 1)
        self.assertEqual(summary["replay_valid_count"], 1)
        self.assertEqual(summary["policy_candidate_ready_count"], 0)
        self.assertEqual(summary["policy_candidate_trace_receipt_attached_count"], 0)
        self.assertEqual(summary["policy_candidate_trace_receipt_required_count"], 0)
        self.assertEqual(summary["policy_candidate_trace_receipt_ready_count"], 0)
        self.assertEqual(summary["policy_candidate_trace_receipt_blocked_bundle_count"], 0)
        self.assertEqual(summary["policy_candidate_blocked_bundle_count"], 1)
        self.assertEqual(summary["policy_candidate_no_reuse_distance_observation_count"], 1)
        self.assertEqual(summary["policy_candidate_blocker_counts"]["no_replay_policy_candidate"], 1)
        self.assertEqual(summary["policy_candidate_blocker_counts"]["no_reuse_distance_observations"], 1)
        self.assertEqual(summary["candidate_policy_ids"], [])
        self.assertEqual(summary["phase4_ready_bundle_count"], 0)
        self.assertEqual(summary["valid_inventory_count"], 1)
        self.assertEqual(summary["invalid_inventory_count"], 1)
        self.assertEqual(len(summary["scanner_coverage_gap_inventories"]), 1)
        self.assertFalse(summary["scanner_coverage_gap_inventories"][0]["referenced_by_bundle"])
        self.assertIn("no_replay_policy_candidate", summary["remaining_blockers"])
        self.assertIn("dense_fallback_output_artifact_for_quality_bounds", summary["remaining_blockers"])
        bundle = summary["bundles"][0]
        self.assertTrue(bundle["real_model_pair_ready"])
        self.assertTrue(bundle["replay_valid"])
        self.assertEqual(bundle["reuse_distance_observations"], 0)
        self.assertFalse(bundle["policy_candidate_ready"])
        self.assertFalse(bundle["policy_candidate_trace_receipt_required"])
        self.assertFalse(bundle["policy_candidate_trace_receipt_ready"])
        self.assertFalse(bundle["policy_candidate_trace_receipt_blocked"])
        blocker_ids = {item["id"] for item in bundle["policy_candidate_blockers"]}
        self.assertIn("no_replay_policy_candidate", blocker_ids)
        self.assertIn("no_reuse_distance_observations", bundle["policy_candidate_blocker_ids"])
        self.assertFalse(bundle["bundle_ready_for_phase4"])

    def test_policy_candidate_requires_ready_trace_receipt_in_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_id = "candidate-mixtral.gguf"
            trace_path, inventory_path = make_candidate_trace_and_inventory(root, model_id)
            bundle_path = root / "candidate_phase3_real_evidence_bundle.json"
            write_json(bundle_path, bundle_manifest(trace_path, inventory_path, model_id))

            summary = planner.build_matrix(root)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["bundle_count"], 1)
        self.assertEqual(summary["policy_candidate_ready_count"], 1)
        self.assertEqual(summary["policy_candidate_blocked_bundle_count"], 0)
        self.assertEqual(summary["policy_candidate_trace_receipt_attached_count"], 0)
        self.assertEqual(summary["policy_candidate_trace_receipt_required_count"], 1)
        self.assertEqual(summary["policy_candidate_trace_receipt_ready_count"], 0)
        self.assertEqual(summary["policy_candidate_trace_receipt_blocked_bundle_count"], 1)
        self.assertEqual(summary["no_go_reason_counts"]["policy_candidate_trace_receipt_not_ready"], 1)
        self.assertIn("preload_shortlist", summary["candidate_policy_ids"])
        bundle = summary["bundles"][0]
        self.assertTrue(bundle["policy_candidate_ready"])
        self.assertFalse(bundle["trace_receipt_attached"])
        self.assertEqual(bundle["trace_receipt_status"], "not_provided")
        self.assertTrue(bundle["policy_candidate_trace_receipt_required"])
        self.assertFalse(bundle["policy_candidate_trace_receipt_ready"])
        self.assertTrue(bundle["policy_candidate_trace_receipt_blocked"])
        self.assertIn("policy_candidate_trace_receipt_not_ready", bundle["no_go_reason_ids"])
        self.assertFalse(bundle["bundle_ready_for_phase4"])

    def test_matrix_merges_reuse_prompt_identity_diagnostics(self) -> None:
        original_reuse_summary = planner.plan_phase3_reuse_evidence_capture.build_root_summary
        planner.plan_phase3_reuse_evidence_capture.build_root_summary = lambda _root: {
            "prompt_identity_ready_count": 1,
            "prompt_identity_metadata_missing_count": 2,
            "blocker_counts": {
                "no_reuse_distance_observations": 1,
                "prompt_identity_metadata_missing": 2,
            },
        }
        planner.build_matrix.cache_clear()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                model_id = "local-mixtral.gguf"
                trace_path, inventory_path = make_real_trace_and_inventory(root, model_id)
                bundle_path = root / "local_phase3_real_evidence_bundle.json"
                write_json(bundle_path, bundle_manifest(trace_path, inventory_path, model_id))

                summary = planner.build_matrix(root)
        finally:
            planner.plan_phase3_reuse_evidence_capture.build_root_summary = original_reuse_summary
            planner.build_matrix.cache_clear()

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["policy_candidate_prompt_identity_ready_count"], 1)
        self.assertEqual(summary["policy_candidate_prompt_identity_metadata_missing_count"], 2)
        self.assertEqual(summary["policy_candidate_blocker_counts"]["no_reuse_distance_observations"], 1)
        self.assertEqual(summary["policy_candidate_blocker_counts"]["prompt_identity_metadata_missing"], 2)
    def test_missing_root_is_invalid_but_side_effect_free(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = planner.build_matrix(Path(temp_dir) / "missing")

        self.assertFalse(summary["valid"])
        self.assertEqual(summary["bundle_count"], 0)
        self.assertIn("matrix does not run Docker", summary["safety_contract"])


if __name__ == "__main__":
    unittest.main()