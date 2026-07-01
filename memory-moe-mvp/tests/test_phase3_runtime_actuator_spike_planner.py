import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_phase3_runtime_actuator_spike.py"
SPEC = importlib.util.spec_from_file_location("plan_phase3_runtime_actuator_spike", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


class Phase3RuntimeActuatorSpikePlannerTests(unittest.TestCase):
    def test_default_llama_cpp_spike_is_handoff_ready_but_not_live_ready(self) -> None:
        plan = planner.load_plan(planner.DEFAULT_PLAN_PATH)
        summary = planner.build_summary(plan, planner.DEFAULT_PLAN_PATH)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["schema_version"], planner.SUPPORTED_SCHEMA_VERSION)
        self.assertEqual(summary["backend_family"], "llama_cpp")
        self.assertTrue(summary["spike_handoff_ready"])
        self.assertFalse(summary["live_spike_ready"])
        self.assertFalse(summary["live_actuator_ready"])
        self.assertFalse(summary["may_mutate_runtime"])
        self.assertEqual(summary["required_capability_count"], 8)
        self.assertEqual(summary["proof_requirement_count"], 8)
        self.assertGreaterEqual(summary["proof_artifact_count"], 16)
        self.assertIn("residency_observation", summary["blocking_capabilities"])
        self.assertIn("residency_control", summary["control_blockers"])
        self.assertIn("cleanup_restore", summary["control_blockers"])

    def test_proof_requirements_cover_all_live_capabilities_with_gates(self) -> None:
        plan = planner.load_plan(planner.DEFAULT_PLAN_PATH)
        summary = planner.build_summary(plan, planner.DEFAULT_PLAN_PATH)

        required = set(summary["required_capabilities"])
        requirements = {item["capability_id"]: item for item in summary["proof_requirements"]}

        self.assertEqual(set(requirements), required)
        for capability_id, requirement in requirements.items():
            self.assertTrue(requirement["probe_or_patch_boundary"], capability_id)
            self.assertTrue(requirement["proof_artifacts"], capability_id)
            self.assertTrue(requirement["completion_gate"], capability_id)
            self.assertFalse(requirement["may_mutate_runtime"], capability_id)
        control = requirements["residency_control"]
        self.assertIn("residency_observation", control["dependency_ids"])
        self.assertTrue(control["requires_explicit_runtime_approval_before_live"])

    def test_unknown_backend_is_invalid(self) -> None:
        plan = planner.load_plan(planner.DEFAULT_PLAN_PATH)
        summary = planner.build_summary(plan, planner.DEFAULT_PLAN_PATH, backend_family="missing_backend")

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["spike_handoff_ready"])
        self.assertTrue(any("missing_backend" in error for error in summary["errors"]), summary["errors"])

    def test_unmapped_live_capability_is_invalid(self) -> None:
        plan = planner.load_plan(planner.DEFAULT_PLAN_PATH)
        plan["capability_requirements"].append(
            {
                "id": "mystery_residency_teleporter",
                "required_for_live_actuator": True,
                "current_status": "missing",
                "description": "Synthetic test capability.",
            }
        )
        summary = planner.build_summary(plan, planner.DEFAULT_PLAN_PATH)

        self.assertFalse(summary["valid"])
        self.assertTrue(
            any("mystery_residency_teleporter" in error for error in summary["errors"]),
            summary["errors"],
        )

    def test_markdown_report_names_spike_handoff_and_control_gate(self) -> None:
        plan = planner.load_plan(planner.DEFAULT_PLAN_PATH)
        summary = planner.build_summary(plan, planner.DEFAULT_PLAN_PATH)
        report = planner.format_markdown_report(summary)

        self.assertIn("# Phase 3 Runtime Actuator Spike Plan", report)
        self.assertIn("Spike handoff ready: `True`", report)
        self.assertIn("Live spike ready: `False`", report)
        self.assertIn("residency_control", report)
        self.assertIn("cleanup proof", report)

    def test_cli_returns_nonzero_for_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "managed.json"
            path.write_text("{", encoding="utf-8")

            status, summary, error = planner.plan_path(path)

        self.assertEqual(status, 2)
        self.assertIsNone(summary)
        assert error is not None
        self.assertIn("Could not load managed expert loading plan", error)

    def test_cli_json_summary_shape_for_hookable_backend(self) -> None:
        plan = planner.load_plan(planner.DEFAULT_PLAN_PATH)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "managed.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            status, summary, error = planner.plan_path(path, backend_family="hookable_pytorch")

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        self.assertEqual(summary["mode"], "phase3_runtime_actuator_spike_plan")
        self.assertEqual(summary["backend_family"], "hookable_pytorch")
        self.assertIn("proof_requirements", summary)
        self.assertFalse(summary["live_spike_ready"])


if __name__ == "__main__":
    unittest.main()