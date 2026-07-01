import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_dense_fallback_comparison.py"
SPEC = importlib.util.spec_from_file_location("plan_dense_fallback_comparison", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


def valid_artifact():
    return {
        "schema_version": planner.SUPPORTED_SCHEMA_VERSION,
        "model_id": "fixture-mixtral.gguf",
        "prompt_family": "fixture",
        "managed_policy_id": "preload_shortlist",
        "managed_artifact": "memory-moe-mvp/replay/fixture-managed.json",
        "dense_artifact": "memory-moe-mvp/replay/fixture-dense.json",
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
            },
            {
                "prompt_id": "case-002",
                "managed_output_present": True,
                "dense_output_present": True,
                "quality_delta_label": "minor_delta",
            },
        ],
    }


class DenseFallbackComparisonPlannerTests(unittest.TestCase):
    def test_missing_artifact_records_exact_blocker_but_is_valid_plan(self) -> None:
        status, summary, error = planner.plan_path()

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        self.assertTrue(summary["valid"], summary["errors"])
        self.assertFalse(summary["comparison_available"])
        self.assertFalse(summary["comparison_ready"])
        self.assertEqual(summary["blocker"]["id"], "fallback_output_missing_for_quality_comparison")
        self.assertFalse(summary["phase_3_gate"]["ready_for_live_spike"])
        self.assertIn("planner does not send prompt traffic", summary["safety_contract"])

    def test_valid_artifact_marks_comparison_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fallback.json"
            path.write_text(json.dumps(valid_artifact()), encoding="utf-8")
            status, summary, error = planner.plan_path(path)

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        self.assertTrue(summary["valid"], summary["errors"])
        self.assertTrue(summary["comparison_available"])
        self.assertTrue(summary["comparison_ready"])
        self.assertTrue(summary["comparison_provenance_ready"])
        self.assertEqual(summary["comparison_count"], 2)
        self.assertEqual(summary["quality_delta_counts"]["same"], 1)
        self.assertEqual(summary["quality_delta_counts"]["minor_delta"], 1)
        self.assertIsNone(summary["blocker"])

    def test_missing_builder_provenance_keeps_shape_valid_but_not_ready(self) -> None:
        artifact = valid_artifact()
        artifact.pop("builder")

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fallback.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            status, summary, error = planner.plan_path(path)

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        self.assertTrue(summary["valid"], summary["errors"])
        self.assertTrue(summary["comparison_shape_ready"])
        self.assertFalse(summary["comparison_provenance_ready"])
        self.assertFalse(summary["comparison_ready"])
        self.assertEqual(summary["blocker"]["id"], "fallback_comparison_missing_capture_receipt_provenance")
        self.assertTrue(any("artifact.builder" in item for item in summary["provenance_blockers"]))


    def test_inconsistent_receipt_pair_keeps_shape_valid_but_not_ready(self) -> None:
        artifact = valid_artifact()
        artifact["builder"]["input_receipts"]["receipt_pair_consistent"] = False

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fallback.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            status, summary, error = planner.plan_path(path)

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        self.assertTrue(summary["valid"], summary["errors"])
        self.assertTrue(summary["comparison_shape_ready"])
        self.assertFalse(summary["comparison_provenance_ready"])
        self.assertFalse(summary["comparison_ready"])
        self.assertEqual(summary["blocker"]["id"], "fallback_comparison_missing_capture_receipt_provenance")
        self.assertTrue(any("receipt_pair_consistent" in item for item in summary["provenance_blockers"]))


    def test_major_or_unknown_delta_keeps_comparison_not_ready(self) -> None:
        artifact = valid_artifact()
        artifact["comparisons"][0]["quality_delta_label"] = "major_delta"

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fallback.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            status, summary, error = planner.plan_path(path)

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        self.assertTrue(summary["valid"], summary["errors"])
        self.assertFalse(summary["comparison_ready"])
        self.assertEqual(summary["blocker"]["id"], "fallback_comparison_not_ready")

    def test_missing_dense_output_keeps_comparison_not_ready(self) -> None:
        artifact = valid_artifact()
        artifact["comparisons"][0]["dense_output_present"] = False

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fallback.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            status, summary, error = planner.plan_path(path)

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        self.assertTrue(summary["valid"], summary["errors"])
        self.assertFalse(summary["comparison_ready"])
        self.assertEqual(summary["missing_dense_output_count"], 1)

    def test_invalid_schema_is_rejected(self) -> None:
        artifact = valid_artifact()
        artifact["schema_version"] = "old"

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fallback.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            status, summary, error = planner.plan_path(path)

        self.assertEqual(status, 2)
        self.assertIsNone(error)
        assert summary is not None
        self.assertFalse(summary["valid"])
        self.assertTrue(any("schema_version" in item for item in summary["errors"]))

    def test_duplicate_prompt_ids_are_rejected(self) -> None:
        artifact = valid_artifact()
        artifact["comparisons"].append(copy.deepcopy(artifact["comparisons"][0]))

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fallback.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            status, summary, error = planner.plan_path(path)

        self.assertEqual(status, 2)
        self.assertIsNone(error)
        assert summary is not None
        self.assertTrue(any("duplicates" in item for item in summary["errors"]))

    def test_bad_json_returns_clean_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fallback.json"
            path.write_text("{", encoding="utf-8")
            status, summary, error = planner.plan_path(path)

        self.assertEqual(status, 2)
        self.assertIsNone(summary)
        assert error is not None
        self.assertIn("Could not build dense fallback comparison plan", error)


if __name__ == "__main__":
    unittest.main()
