import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_expert_inventory.py"
SPEC = importlib.util.spec_from_file_location("plan_expert_inventory", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


class ExpertInventoryPlannerTests(unittest.TestCase):
    def load_fixture(self):
        return planner.load_manifest(planner.DEFAULT_MANIFEST_PATH)

    def test_default_manifest_validates_and_reports_replay_ready(self) -> None:
        manifest = self.load_fixture()
        summary = planner.build_summary(manifest, planner.DEFAULT_MANIFEST_PATH)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["schema_version"], planner.SUPPORTED_SCHEMA_VERSION)
        self.assertEqual(summary["source_format"], "gguf")
        self.assertEqual(summary["backend_family"], "llama_cpp")
        self.assertTrue(summary["phase_3_replay_ready"])
        self.assertEqual(summary["expert_count"], 8)
        self.assertEqual(summary["component_count"], 24)
        self.assertEqual(summary["total_estimated_residency_bytes"], 25165824)
        self.assertEqual(summary["coverage_counts"]["complete"], 24)
        self.assertIn({"layer_id": 1, "expert_id": 7}, summary["trace_join_keys"])

    def test_validation_rejects_duplicate_component_entry(self) -> None:
        manifest = self.load_fixture()
        manifest["entries"].append(copy.deepcopy(manifest["entries"][0]))

        errors = planner.validate_manifest(manifest)

        self.assertTrue(any("duplicates component" in error for error in errors), errors)

    def test_validation_reports_missing_required_component(self) -> None:
        manifest = self.load_fixture()
        manifest["entries"] = [
            entry
            for entry in manifest["entries"]
            if not (
                entry["layer_id"] == 0
                and entry["expert_id"] == 0
                and entry["component_name"] == "down_proj"
            )
        ]

        errors = planner.validate_manifest(manifest)

        self.assertTrue(any("missing required components: down_proj" in error for error in errors), errors)

    def test_validation_rejects_expected_byte_total_drift(self) -> None:
        manifest = self.load_fixture()
        manifest["expected"]["total_estimated_residency_bytes"] += 1

        errors = planner.validate_manifest(manifest)

        self.assertTrue(
            any("expected.total_estimated_residency_bytes" in error for error in errors),
            errors,
        )

    def test_validation_rejects_overlapping_source_ranges(self) -> None:
        manifest = self.load_fixture()
        manifest["entries"][1]["byte_offset"] = manifest["entries"][0]["byte_offset"]

        errors = planner.validate_manifest(manifest)

        self.assertTrue(any("overlaps previous tensor" in error for error in errors), errors)

    def test_cli_json_summary_shape(self) -> None:
        manifest = self.load_fixture()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "inventory.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            status, summary, error = planner.plan_manifest_path(path)

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["mode"], "expert_inventory_manifest_plan")
        self.assertIn("trace_join_keys", summary)
        self.assertIn("phase_3_replay_ready", summary)
        self.assertIn("next_actions", summary)

    def test_cli_returns_nonzero_for_invalid_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "inventory.json"
            path.write_text("{}", encoding="utf-8")

            self.assertEqual(planner.main_from_test_path(path), 2)

    def test_cli_returns_nonzero_for_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "inventory.json"
            path.write_text("{", encoding="utf-8")

            status, summary, error = planner.plan_manifest_path(path)

        self.assertEqual(status, 2)
        self.assertIsNone(summary)
        assert error is not None
        self.assertIn("Could not load expert inventory manifest", error)


if __name__ == "__main__":
    unittest.main()
