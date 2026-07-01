import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_phase3_real_evidence_capture.py"
SPEC = importlib.util.spec_from_file_location("plan_phase3_real_evidence_capture", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)

TRACE_FIXTURE = ROOT / "memory-moe-mvp" / "data" / "llama_cpp_router_trace.fixture.jsonl"
INVENTORY_FIXTURE = ROOT / "memory-moe-mvp" / "data" / "expert_inventory_manifest.fixture.json"
POLICIES_FIXTURE = ROOT / "memory-moe-mvp" / "data" / "baseline_replay_policies.json"
MANAGED_PLAN = ROOT / "memory-moe-mvp" / "data" / "managed_expert_loading_plan.json"


def request_by_id(summary):
    return {request["id"]: request for request in summary["artifact_requests"]}


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
            "evidence_fields": ["resident_expert_count", "resident_bytes"],
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
class Phase3RealEvidenceCapturePlannerTests(unittest.TestCase):
    def test_default_plan_lists_remaining_real_evidence_requests(self) -> None:
        summary = planner.build_capture_plan(
            TRACE_FIXTURE,
            INVENTORY_FIXTURE,
            POLICIES_FIXTURE,
            MANAGED_PLAN,
        )

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertFalse(summary["phase3_complete"])
        self.assertEqual(summary["packet_decision"], "no_go_live_spike")
        self.assertFalse(summary["ready_to_execute_without_approval"])
        self.assertFalse(summary["metadata_only_steps_ready"])
        self.assertTrue(summary["prompt_or_runtime_approval_required"])
        by_id = request_by_id(summary)
        self.assertEqual(by_id["real_semantic_trace_artifact"]["status"], "needed")
        self.assertEqual(by_id["scanner_derived_inventory"]["status"], "needed")
        self.assertEqual(by_id["real_trace_inventory_pairing"]["status"], "blocked_by_required_artifacts")
        self.assertEqual(by_id["dense_fallback_comparison_artifact"]["status"], "needed")
        self.assertEqual(by_id["policy_replay_candidate_window"]["status"], "blocked_by_real_model_pairing")
        self.assertEqual(by_id["live_residency_capability_proof"]["status"], "future_phase_4")
        self.assertNotIn("live_residency_capability_proof", summary["blocking_request_ids"])
        self.assertIn("live_residency_capability_proof", summary["future_request_ids"])
        self.assertIn("phase3_real_evidence_bundle_manifest", summary["blocking_request_ids"])
        scanner_formats = {
            option["source_format"]
            for option in by_id["scanner_derived_inventory"]["command_options"]
        }
        self.assertEqual(scanner_formats, {"gguf", "safetensors"})
        fallback_builder_command = by_id["dense_fallback_comparison_artifact"]["command_options"][0]["command"]
        self.assertIn("scripts/build_dense_fallback_comparison.py", fallback_builder_command)
        self.assertIn("--managed-artifact", fallback_builder_command)
        self.assertIn("--dense-artifact", fallback_builder_command)
        bundle_builder = by_id["phase3_real_evidence_bundle_manifest"]["command_options"][0]
        self.assertEqual(bundle_builder["command_class"], "local_phase3_bundle_builder")
        bundle_builder_command = bundle_builder["command"]
        self.assertIn("scripts/build_phase3_real_evidence_bundle.py", bundle_builder_command)
        self.assertIn("--trace-path", bundle_builder_command)
        self.assertIn("--inventory-path", bundle_builder_command)
        self.assertIn("--output", bundle_builder_command)
        self.assertIn("planner does not send prompt traffic", summary["safety_contract"])

    def test_gguf_model_inputs_make_metadata_scan_ready(self) -> None:
        summary = planner.build_capture_plan(
            TRACE_FIXTURE,
            INVENTORY_FIXTURE,
            POLICIES_FIXTURE,
            MANAGED_PLAN,
            model_path=Path("models/local-mixtral.gguf"),
            model_id="local-mixtral.gguf",
            source_format="gguf",
        )

        scanner = request_by_id(summary)["scanner_derived_inventory"]
        self.assertEqual(scanner["status"], "ready_metadata_only")
        self.assertTrue(summary["metadata_only_steps_ready"])
        self.assertEqual(scanner["details"]["inferred_source_format"], "gguf")
        command = scanner["command_options"][0]["command"]
        self.assertEqual(command[:5], ["uv", "run", "--managed-python", "--python", "3.13"])
        self.assertIn("scripts/scan_gguf_expert_inventory.py", command)
        self.assertIn("models/local-mixtral.gguf", command)
        self.assertIn("--model-id", command)
        self.assertIn("local-mixtral.gguf", command)

    def test_source_format_can_be_inferred_from_gguf_suffix(self) -> None:
        summary = planner.build_capture_plan(
            TRACE_FIXTURE,
            INVENTORY_FIXTURE,
            POLICIES_FIXTURE,
            MANAGED_PLAN,
            model_path=Path("models/local-mixtral.gguf"),
            model_id="local-mixtral.gguf",
        )

        scanner = request_by_id(summary)["scanner_derived_inventory"]
        self.assertEqual(scanner["status"], "ready_metadata_only")
        self.assertEqual(scanner["command_options"][0]["source_format"], "gguf")

    def test_safetensors_model_inputs_emit_safetensors_scanner_command(self) -> None:
        summary = planner.build_capture_plan(
            TRACE_FIXTURE,
            INVENTORY_FIXTURE,
            POLICIES_FIXTURE,
            MANAGED_PLAN,
            model_path=Path("models/local-moe/model-00001-of-00002.safetensors"),
            model_id="local-moe",
            source_format="safetensors",
        )

        scanner = request_by_id(summary)["scanner_derived_inventory"]
        self.assertEqual(scanner["status"], "ready_metadata_only")
        self.assertEqual(scanner["command_options"][0]["source_format"], "safetensors")
        command = scanner["command_options"][0]["command"]
        self.assertIn("scripts/scan_safetensors_expert_inventory.py", command)
        self.assertIn("models/local-moe/model-00001-of-00002.safetensors", command)

    def test_valid_fallback_artifact_satisfies_fallback_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fallback_path = Path(temp_dir) / "fallback.json"
            fallback_path.write_text(json.dumps(fallback_artifact()), encoding="utf-8")
            summary = planner.build_capture_plan(
                TRACE_FIXTURE,
                INVENTORY_FIXTURE,
                POLICIES_FIXTURE,
                MANAGED_PLAN,
                fallback_artifact_path=fallback_path,
            )

        fallback = request_by_id(summary)["dense_fallback_comparison_artifact"]
        self.assertEqual(fallback["status"], "already_satisfied")
        self.assertFalse(fallback["approval_required"])
        bundle_builder_command = request_by_id(summary)["phase3_real_evidence_bundle_manifest"]["command_options"][0]["command"]
        self.assertIn("--fallback-artifact-path", bundle_builder_command)
        self.assertIn(str(fallback_path), bundle_builder_command)

    def test_valid_live_proof_artifact_satisfies_future_live_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            live_path = Path(temp_dir) / "live-proof.json"
            export_path = Path(temp_dir) / "live-proof-export.json"
            export_path.write_text("{}\n", encoding="utf-8")
            live_path.write_text(json.dumps(live_proof_artifact(artifact_paths=[str(export_path)])), encoding="utf-8")
            summary = planner.build_capture_plan(
                TRACE_FIXTURE,
                INVENTORY_FIXTURE,
                POLICIES_FIXTURE,
                MANAGED_PLAN,
                live_proof_artifact_path=live_path,
            )

        live = request_by_id(summary)["live_residency_capability_proof"]
        self.assertEqual(live["status"], "already_satisfied")
        self.assertFalse(live["approval_required"])
        self.assertTrue(live["details"]["current_live_proof_ready"])
        self.assertEqual(summary["live_proof_artifact_path"], str(live_path))
    def test_markdown_report_contains_capture_requests_and_commands(self) -> None:
        summary = planner.build_capture_plan(
            TRACE_FIXTURE,
            INVENTORY_FIXTURE,
            POLICIES_FIXTURE,
            MANAGED_PLAN,
        )

        report = planner.format_markdown_report(summary)

        self.assertIn("# Phase 3 Real-Evidence Capture Plan", report)
        self.assertIn("## Artifact Requests", report)
        self.assertIn("real_semantic_trace_artifact", report)
        self.assertIn("scanner_derived_inventory", report)
        self.assertIn("dense_fallback_comparison_artifact", report)
        self.assertIn("live_residency_capability_proof", report)
        self.assertIn("## Commands", report)
        self.assertIn("scripts/build_dense_fallback_comparison.py", report)
        self.assertIn("planner does not send prompt traffic", report)

    def test_write_markdown_report_writes_capture_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "real-evidence-capture.md"
            summary = planner.build_capture_plan(
                TRACE_FIXTURE,
                INVENTORY_FIXTURE,
                POLICIES_FIXTURE,
                MANAGED_PLAN,
            )

            planner.write_markdown_report(summary, output_path)

            report = output_path.read_text(encoding="utf-8")

        self.assertTrue(report.startswith("# Phase 3 Real-Evidence Capture Plan"))
        self.assertIn("## Phase 3 Blocking Requests", report)
        self.assertIn("phase3_real_evidence_bundle_manifest", report)
    def test_bad_input_returns_plan_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bad_inventory = Path(temp_dir) / "bad_inventory.json"
            bad_inventory.write_text("{", encoding="utf-8")
            status, summary, error = planner.plan_paths(
                TRACE_FIXTURE,
                bad_inventory,
                POLICIES_FIXTURE,
                MANAGED_PLAN,
            )

        self.assertEqual(status, 2)
        self.assertIsNone(summary)
        assert error is not None
        self.assertIn("Could not build Phase 3 real-evidence capture plan", error)



if __name__ == "__main__":
    unittest.main()
