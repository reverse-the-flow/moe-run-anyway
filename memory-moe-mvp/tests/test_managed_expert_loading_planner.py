import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_managed_expert_loading.py"
SPEC = importlib.util.spec_from_file_location("plan_managed_expert_loading", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


class ManagedExpertLoadingPlannerTests(unittest.TestCase):
    def test_default_plan_validates_and_reports_no_live_actuator(self) -> None:
        plan = planner.load_plan(planner.DEFAULT_PLAN_PATH)
        summary = planner.build_summary(plan, planner.DEFAULT_PLAN_PATH)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["schema_version"], planner.SUPPORTED_SCHEMA_VERSION)
        self.assertFalse(summary["live_expert_loading_implemented"])
        self.assertEqual(
            summary["backend_adapters"],
            [
                "hookable_pytorch",
                "llama_cpp",
                "moe_infinity_style",
                "prototype_offload_system",
                "vllm_openai_compatible",
            ],
        )
        self.assertIn("residency_control", summary["adapter_gaps"]["llama_cpp"])
        self.assertIn("Live expert loading is not implemented yet.", summary["safety_contract"])

    def test_validation_requires_known_state_vocabulary(self) -> None:
        plan = planner.load_plan(planner.DEFAULT_PLAN_PATH)
        plan["state_vocab"] = [item for item in plan["state_vocab"] if item["id"] != "loading"]

        errors = planner.validate_plan(plan)

        self.assertTrue(any("loading" in error for error in errors), errors)

    def test_validation_rejects_unknown_state_vocabulary(self) -> None:
        plan = planner.load_plan(planner.DEFAULT_PLAN_PATH)
        plan["state_vocab"].append({"id": "gpu_magic", "description": "unsupported"})

        errors = planner.validate_plan(plan)

        self.assertTrue(any("gpu_magic" in error for error in errors), errors)

    def test_validation_requires_known_policy_actions(self) -> None:
        plan = planner.load_plan(planner.DEFAULT_PLAN_PATH)
        plan["policy_actions"] = [
            item for item in plan["policy_actions"] if item["id"] != "abort_run"
        ]

        errors = planner.validate_plan(plan)

        self.assertTrue(any("abort_run" in error for error in errors), errors)

    def test_validation_requires_backend_adapters(self) -> None:
        plan = planner.load_plan(planner.DEFAULT_PLAN_PATH)
        plan["backend_adapters"] = [
            item for item in plan["backend_adapters"] if item["id"] != "moe_infinity_style"
        ]

        errors = planner.validate_plan(plan)

        self.assertTrue(any("moe_infinity_style" in error for error in errors), errors)

    def test_validation_requires_capability_requirements(self) -> None:
        plan = planner.load_plan(planner.DEFAULT_PLAN_PATH)
        plan["capability_requirements"] = [
            item for item in plan["capability_requirements"] if item["id"] != "cleanup_restore"
        ]

        errors = planner.validate_plan(plan)

        self.assertTrue(any("cleanup_restore" in error for error in errors), errors)

    def test_cli_json_summary_shape(self) -> None:
        plan = planner.load_plan(planner.DEFAULT_PLAN_PATH)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "managed.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            status, summary, error = planner.plan_path(path)

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["mode"], "managed_expert_loading_plan")
        self.assertIn("capability_statuses", summary)
        self.assertIn("adapter_gaps", summary)
        self.assertIn("next_actions", summary)

    def test_cli_returns_nonzero_for_invalid_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "managed.json"
            path.write_text("{}", encoding="utf-8")

            self.assertEqual(planner.main_from_test_path(path), 2)

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
