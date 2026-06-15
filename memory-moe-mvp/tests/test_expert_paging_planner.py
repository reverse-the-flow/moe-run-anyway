import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_expert_paging.py"
SPEC = importlib.util.spec_from_file_location("plan_expert_paging", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


class ExpertPagingPlannerTests(unittest.TestCase):
    def test_default_roadmap_validates_and_reports_phase_one(self) -> None:
        roadmap = planner.load_roadmap(planner.DEFAULT_ROADMAP_PATH)
        summary = planner.build_summary(roadmap, planner.DEFAULT_ROADMAP_PATH)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["schema_version"], planner.SUPPORTED_SCHEMA_VERSION)
        self.assertEqual(summary["current_phase"]["id"], "phase_1")
        self.assertEqual(roadmap["phases"][0]["status"], "complete")
        self.assertEqual(summary["planned_stage"], "harness_run_request")
        self.assertIn("planner does not send prompt traffic", summary["safety_contract"])

    def test_validation_requires_runtime_actuator_truth_table_entry(self) -> None:
        roadmap = planner.load_roadmap(planner.DEFAULT_ROADMAP_PATH)
        roadmap["current_truth_table"] = [
            row for row in roadmap["current_truth_table"] if row["surface"] != "runtime_actuator"
        ]

        errors = planner.validate_roadmap(roadmap)

        self.assertIn("current_truth_table must include runtime_actuator", errors)

    def test_validation_requires_actuator_spike_proof_ids(self) -> None:
        roadmap = planner.load_roadmap(planner.DEFAULT_ROADMAP_PATH)
        roadmap["actuator_spike_checklist"] = [
            item
            for item in roadmap["actuator_spike_checklist"]
            if item["id"] != "tensor_residency_control"
        ]

        errors = planner.validate_roadmap(roadmap)

        self.assertTrue(
            any("tensor_residency_control" in error for error in errors),
            errors,
        )

    def test_cli_json_summary_shape(self) -> None:
        roadmap = planner.load_roadmap(planner.DEFAULT_ROADMAP_PATH)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "roadmap.json"
            path.write_text(json.dumps(roadmap), encoding="utf-8")
            status, summary, error = planner.plan_roadmap_path(path)

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["mode"], "expert_paging_roadmap_plan")

    def test_cli_returns_nonzero_for_invalid_roadmap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "roadmap.json"
            path.write_text("{}", encoding="utf-8")

            self.assertEqual(planner.main_from_test_path(path), 2)


if __name__ == "__main__":
    unittest.main()
