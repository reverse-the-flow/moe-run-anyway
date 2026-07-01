import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "build_phase3_real_evidence_bundle.py"
SPEC = importlib.util.spec_from_file_location("build_phase3_real_evidence_bundle", SCRIPT_PATH)
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def args(**overrides):
    values = {
        "name": "Phase 3 real-evidence bundle",
        "model_id": "fixture-mixtral.gguf",
        "source_format": "gguf",
        "backend_family": "llama_cpp",
        "prompt_family": "fixture",
        "trace_path": builder.DEFAULT_TRACE_PATH,
        "trace_receipt_path": None,
        "inventory_path": builder.DEFAULT_INVENTORY_PATH,
        "policies_path": builder.DEFAULT_POLICIES_PATH,
        "managed_plan_path": builder.DEFAULT_MANAGED_PLAN_PATH,
        "fallback_artifact_path": None,
        "live_proof_artifact_path": None,
        "real_model_trace_capture_approved": False,
        "dense_fallback_capture_approved": False,
        "runtime_prompt_traffic_approved": False,
        "approval_note": [],
        "output": None,
        "manifest_json": False,
        "json": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class Phase3RealEvidenceBundleBuilderTests(unittest.TestCase):
    def test_default_build_is_valid_without_writing(self) -> None:
        status, summary, manifest, error = builder.plan_build(args())

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        assert manifest is not None
        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(manifest["schema_version"], "moe-phase3-real-evidence-bundle-v1")
        self.assertFalse(summary["fallback_artifact_attached"])
        self.assertFalse(summary["trace_receipt_attached"])
        self.assertFalse(summary["live_proof_artifact_attached"])
        self.assertIsNone(manifest["artifact_paths"]["trace_receipt_path"])
        self.assertIsNone(manifest["artifact_paths"]["live_proof_artifact_path"])
        self.assertEqual(summary["artifact_statuses"]["trace_path"]["status"], "present")
        self.assertEqual(summary["artifact_statuses"]["trace_receipt_path"]["status"], "not_provided")
        self.assertEqual(summary["artifact_statuses"]["live_proof_artifact_path"]["status"], "not_provided")
        self.assertIn("bundle builder does not send prompt traffic", summary["safety_contract"])

    def test_output_manifest_validates_with_bundle_planner(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "bundle.json"
            status, summary, manifest, error = builder.plan_build(args(output=output_path))
            written = json.loads(output_path.read_text(encoding="utf-8"))
            bundle_summary = builder.plan_phase3_real_evidence_bundle.build_summary(output_path)

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        assert manifest is not None
        self.assertEqual(manifest, written)
        self.assertTrue(bundle_summary["valid"], bundle_summary["errors"])
        self.assertFalse(bundle_summary["bundle_ready_for_phase4"])

    def test_trace_receipt_path_is_recorded_when_provided(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            receipt_path = Path(temp_dir) / "trace-receipt.json"
            receipt_path.write_text("{}", encoding="utf-8")
            status, summary, manifest, error = builder.plan_build(args(trace_receipt_path=receipt_path))

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        assert manifest is not None
        self.assertTrue(summary["trace_receipt_attached"])
        self.assertEqual(summary["artifact_statuses"]["trace_receipt_path"]["status"], "present")
        self.assertEqual(manifest["artifact_paths"]["trace_receipt_path"], str(receipt_path))
    def test_live_proof_path_is_recorded_when_provided(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            live_path = Path(temp_dir) / "live-proof.json"
            live_path.write_text("{}", encoding="utf-8")
            status, summary, manifest, error = builder.plan_build(args(live_proof_artifact_path=live_path))

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        assert manifest is not None
        self.assertTrue(summary["live_proof_artifact_attached"])
        self.assertEqual(summary["artifact_statuses"]["live_proof_artifact_path"]["status"], "present")
        self.assertEqual(manifest["artifact_paths"]["live_proof_artifact_path"], str(live_path))

    def test_approval_flags_and_notes_are_recorded(self) -> None:
        status, summary, manifest, error = builder.plan_build(
            args(
                real_model_trace_capture_approved=True,
                dense_fallback_capture_approved=True,
                approval_note=["approved trace capture from external run log"],
            )
        )

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        assert manifest is not None
        approvals = manifest["approvals"]
        self.assertTrue(approvals["real_model_trace_capture_approved"])
        self.assertTrue(approvals["dense_fallback_capture_approved"])
        self.assertFalse(approvals["runtime_prompt_traffic_approved"])
        self.assertEqual(approvals["notes"], ["approved trace capture from external run log"])

    def test_missing_required_artifact_makes_builder_summary_invalid(self) -> None:
        status, summary, manifest, error = builder.plan_build(
            args(trace_path=Path("memory-moe-mvp/data/missing-router-trace.jsonl"))
        )

        self.assertEqual(status, 2)
        self.assertIsNone(error)
        assert summary is not None
        assert manifest is not None
        self.assertFalse(summary["valid"])
        self.assertEqual(summary["artifact_statuses"]["trace_path"]["status"], "missing")
        self.assertTrue(any("trace_path" in item for item in summary["errors"]))


if __name__ == "__main__":
    unittest.main()
