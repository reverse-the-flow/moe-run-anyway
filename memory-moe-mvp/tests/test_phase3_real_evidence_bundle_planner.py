import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_phase3_real_evidence_bundle.py"
SPEC = importlib.util.spec_from_file_location("plan_phase3_real_evidence_bundle", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


def load_fixture_manifest():
    return json.loads(planner.DEFAULT_BUNDLE_PATH.read_text(encoding="utf-8"))


def write_manifest(path: Path, manifest) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


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


def live_proof_artifact(*, source_bundle_path: Path | str = "memory-moe-mvp/data/phase3_real_evidence_bundle.fixture.json", artifact_paths: list[str] | None = None):
    return {
        "schema_version": "moe-phase3-live-capability-proof-v1",
        "name": "Fixture live capability proof",
        "model_id": "fixture-mixtral.gguf",
        "backend_family": "llama_cpp",
        "prompt_family": "fixture",
        "source_bundle_path": str(source_bundle_path),
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


class Phase3RealEvidenceBundlePlannerTests(unittest.TestCase):
    def test_default_fixture_bundle_is_valid_but_not_ready(self) -> None:
        summary = planner.build_summary(planner.DEFAULT_BUNDLE_PATH)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertTrue(summary["packet_ready"])
        self.assertFalse(summary["phase3_complete"])
        self.assertFalse(summary["bundle_ready_for_phase4"])
        self.assertEqual(summary["artifact_statuses"]["trace_path"]["status"], "present")
        self.assertEqual(summary["artifact_statuses"]["fallback_artifact_path"]["status"], "not_provided")
        self.assertEqual(summary["artifact_statuses"]["trace_receipt_path"]["status"], "not_provided")
        self.assertEqual(summary["artifact_statuses"]["live_proof_artifact_path"]["status"], "not_provided")
        self.assertIn("real_model_trace_inventory_pairing_not_ready", summary["no_go_reason_ids"])
        self.assertIn("dense_fallback_comparison_not_ready", summary["no_go_reason_ids"])
        self.assertIn("live_capability_proof_not_ready", summary["no_go_reason_ids"])
        self.assertIn("bundle validator does not send prompt traffic", summary["safety_contract"])

    def test_valid_fallback_artifact_removes_fallback_no_go_reason(self) -> None:
        manifest = load_fixture_manifest()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fallback_path = temp_path / "fallback.json"
            fallback_path.write_text(json.dumps(fallback_artifact()), encoding="utf-8")
            manifest["artifact_paths"]["fallback_artifact_path"] = str(fallback_path)
            bundle_path = temp_path / "bundle.json"
            write_manifest(bundle_path, manifest)

            summary = planner.build_summary(bundle_path)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["artifact_statuses"]["fallback_artifact_path"]["status"], "present")
        self.assertNotIn("dense_fallback_comparison_not_ready", summary["no_go_reason_ids"])
        self.assertIn("real_model_trace_inventory_pairing_not_ready", summary["no_go_reason_ids"])
        self.assertIn("live_capability_proof_not_ready", summary["no_go_reason_ids"])
        self.assertFalse(summary["bundle_ready_for_phase4"])

    def test_valid_live_proof_artifact_removes_live_proof_no_go_reason(self) -> None:
        manifest = load_fixture_manifest()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            live_path = temp_path / "live-proof.json"
            export_path = temp_path / "live-proof-export.json"
            bundle_path = temp_path / "bundle.json"
            export_path.write_text("{}\n", encoding="utf-8")
            live_path.write_text(
                json.dumps(live_proof_artifact(source_bundle_path=bundle_path, artifact_paths=[str(export_path)])),
                encoding="utf-8",
            )
            manifest["artifact_paths"]["live_proof_artifact_path"] = str(live_path)
            write_manifest(bundle_path, manifest)

            summary = planner.build_summary(bundle_path)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["artifact_statuses"]["live_proof_artifact_path"]["status"], "present")
        self.assertNotIn("live_capability_proof_not_ready", summary["no_go_reason_ids"])
        self.assertNotIn("live_actuator_missing", summary["no_go_reason_ids"])
        self.assertFalse(summary["bundle_ready_for_phase4"])

    def test_live_proof_from_different_bundle_is_invalid(self) -> None:
        manifest = load_fixture_manifest()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            live_path = temp_path / "live-proof.json"
            live_path.write_text(json.dumps(live_proof_artifact(source_bundle_path=temp_path / "other-bundle.json")), encoding="utf-8")
            manifest["artifact_paths"]["live_proof_artifact_path"] = str(live_path)
            bundle_path = temp_path / "bundle.json"
            write_manifest(bundle_path, manifest)

            summary = planner.build_summary(bundle_path)

        self.assertFalse(summary["valid"])
        self.assertEqual(summary["artifact_statuses"]["live_proof_artifact_path"]["status"], "present")
        self.assertTrue(any("live capability proof" in error for error in summary["errors"]))
        self.assertIn("live_capability_proof_not_ready", summary["no_go_reason_ids"])

    def test_missing_optional_trace_receipt_path_is_invalid(self) -> None:
        manifest = load_fixture_manifest()
        manifest["artifact_paths"]["trace_receipt_path"] = "memory-moe-mvp/data/missing-trace-receipt.json"
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_path = Path(temp_dir) / "bundle.json"
            write_manifest(bundle_path, manifest)
            summary = planner.build_summary(bundle_path)

        self.assertFalse(summary["valid"])
        self.assertEqual(summary["artifact_statuses"]["trace_receipt_path"]["status"], "missing")
        self.assertTrue(any("trace_receipt_path" in error for error in summary["errors"]))
    def test_missing_optional_live_proof_artifact_path_is_invalid(self) -> None:
        manifest = load_fixture_manifest()
        manifest["artifact_paths"]["live_proof_artifact_path"] = "memory-moe-mvp/data/missing-live-proof.json"
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_path = Path(temp_dir) / "bundle.json"
            write_manifest(bundle_path, manifest)
            summary = planner.build_summary(bundle_path)

        self.assertFalse(summary["valid"])
        self.assertEqual(summary["artifact_statuses"]["live_proof_artifact_path"]["status"], "missing")
        self.assertTrue(any("live_proof_artifact_path" in error for error in summary["errors"]))

    def test_missing_required_artifact_path_is_invalid(self) -> None:
        manifest = load_fixture_manifest()
        manifest["artifact_paths"]["inventory_path"] = "memory-moe-mvp/data/missing-inventory.json"
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_path = Path(temp_dir) / "bundle.json"
            write_manifest(bundle_path, manifest)
            summary = planner.build_summary(bundle_path)

        self.assertFalse(summary["valid"])
        self.assertEqual(summary["artifact_statuses"]["inventory_path"]["status"], "missing")
        self.assertTrue(any("inventory_path" in error for error in summary["errors"]))

    def test_bad_schema_is_invalid(self) -> None:
        manifest = load_fixture_manifest()
        manifest["schema_version"] = "old"
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_path = Path(temp_dir) / "bundle.json"
            write_manifest(bundle_path, manifest)
            summary = planner.build_summary(bundle_path)

        self.assertFalse(summary["valid"])
        self.assertTrue(any("schema_version" in error for error in summary["errors"]))

    def test_bad_json_returns_clean_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_path = Path(temp_dir) / "bundle.json"
            bundle_path.write_text("{", encoding="utf-8")
            status, summary, error = planner.plan_path(bundle_path)

        self.assertEqual(status, 2)
        self.assertIsNone(summary)
        assert error is not None
        self.assertIn("Could not build Phase 3 real-evidence bundle summary", error)


if __name__ == "__main__":
    unittest.main()
