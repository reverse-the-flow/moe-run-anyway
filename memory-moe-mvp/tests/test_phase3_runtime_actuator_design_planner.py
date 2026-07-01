import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_phase3_runtime_actuator_design.py"
SPEC = importlib.util.spec_from_file_location("plan_phase3_runtime_actuator_design", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


class Phase3RuntimeActuatorDesignPlannerTests(unittest.TestCase):
    def test_default_llama_cpp_design_is_handoff_ready_but_not_live_ready(self) -> None:
        plan = planner.load_plan(planner.DEFAULT_PLAN_PATH)
        summary = planner.build_summary(plan, planner.DEFAULT_PLAN_PATH)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["schema_version"], planner.SUPPORTED_SCHEMA_VERSION)
        self.assertEqual(summary["backend_family"], "llama_cpp")
        self.assertTrue(summary["design_handoff_ready"])
        self.assertFalse(summary["live_actuator_ready"])
        self.assertFalse(summary["may_mutate_runtime"])
        self.assertIn("residency_observation", summary["blocking_capabilities"])
        self.assertIn("residency_control", summary["missing_actuator_capabilities"])
        self.assertIn("cleanup_restore", summary["control_blockers"])
        self.assertIn("design handoff readiness does not imply live actuator readiness", summary["safety_contract"])

    def test_unknown_backend_is_invalid(self) -> None:
        plan = planner.load_plan(planner.DEFAULT_PLAN_PATH)
        summary = planner.build_summary(plan, planner.DEFAULT_PLAN_PATH, backend_family="missing_backend")

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["design_handoff_ready"])
        self.assertTrue(any("missing_backend" in error for error in summary["errors"]), summary["errors"])

    def test_unlisted_control_gap_is_invalid(self) -> None:
        plan = planner.load_plan(planner.DEFAULT_PLAN_PATH)
        adapter = next(item for item in plan["backend_adapters"] if item["id"] == "llama_cpp")
        adapter["missing_actuator_capabilities"] = [
            item for item in adapter["missing_actuator_capabilities"] if item != "residency_control"
        ]
        summary = planner.build_summary(plan, planner.DEFAULT_PLAN_PATH)

        self.assertFalse(summary["valid"])
        self.assertTrue(any("residency_control" in error for error in summary["errors"]), summary["errors"])

    def test_cli_json_summary_shape(self) -> None:
        plan = planner.load_plan(planner.DEFAULT_PLAN_PATH)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "managed.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            status, summary, error = planner.plan_path(path, backend_family="hookable_pytorch")

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        self.assertEqual(summary["mode"], "phase3_runtime_actuator_design_plan")
        self.assertEqual(summary["backend_family"], "hookable_pytorch")
        self.assertIn("blocking_capabilities", summary)
        self.assertIn("next_actions", summary)

    def test_markdown_report_names_design_boundary(self) -> None:
        plan = planner.load_plan(planner.DEFAULT_PLAN_PATH)
        summary = planner.build_summary(plan, planner.DEFAULT_PLAN_PATH)
        report = planner.format_markdown_report(summary)

        self.assertIn("# Phase 3 Runtime Actuator Design", report)
        self.assertIn("Design handoff ready: `True`", report)
        self.assertIn("Live actuator ready: `False`", report)
        self.assertIn("residency_control", report)

    def test_cli_returns_nonzero_for_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "managed.json"
            path.write_text("{", encoding="utf-8")

            status, summary, error = planner.plan_path(path)

        self.assertEqual(status, 2)
        self.assertIsNone(summary)
        assert error is not None
        self.assertIn("Could not load managed expert loading plan", error)


if __name__ == "__main__":
    unittest.main()