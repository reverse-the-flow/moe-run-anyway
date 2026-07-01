import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_baseline_replay_policies.py"
SPEC = importlib.util.spec_from_file_location("plan_baseline_replay_policies", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


class BaselineReplayPoliciesPlannerTests(unittest.TestCase):
    def load_fixture(self):
        return planner.load_policies(planner.DEFAULT_POLICIES_PATH)

    def test_default_policies_validate_and_do_not_mutate_runtime(self) -> None:
        plan = self.load_fixture()
        summary = planner.build_summary(plan, planner.DEFAULT_POLICIES_PATH)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["schema_version"], planner.SUPPORTED_SCHEMA_VERSION)
        self.assertEqual(set(summary["policy_ids"]), planner.REQUIRED_POLICY_IDS)
        self.assertFalse(summary["may_mutate_runtime"])
        self.assertIn("rejected_policy_reasons", summary["required_replay_metrics"])
        self.assertIn("fallback_dense", summary["managed_loading_actions"])

    def test_validation_requires_all_baseline_policy_ids(self) -> None:
        plan = self.load_fixture()
        plan["policies"] = [policy for policy in plan["policies"] if policy["id"] != "evict_cold"]

        errors = planner.validate_policies(plan)

        self.assertTrue(any("evict_cold" in error for error in errors), errors)

    def test_validation_rejects_runtime_mutation(self) -> None:
        plan = self.load_fixture()
        plan["policies"][0]["may_mutate_runtime"] = True

        errors = planner.validate_policies(plan)

        self.assertTrue(any("may_mutate_runtime must be false" in error for error in errors), errors)

    def test_validation_rejects_unknown_managed_loading_action(self) -> None:
        plan = self.load_fixture()
        plan["policies"][0]["managed_loading_actions"].append("teleport")

        errors = planner.validate_policies(plan)

        self.assertTrue(any("teleport" in error for error in errors), errors)

    def test_cli_json_summary_shape(self) -> None:
        plan = self.load_fixture()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "policies.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            status, summary, error = planner.plan_path(path)

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["mode"], "baseline_replay_policies_plan")
        self.assertIn("policy_ids", summary)
        self.assertIn("next_actions", summary)

    def test_duplicate_policy_ids_are_rejected(self) -> None:
        plan = self.load_fixture()
        plan["policies"].append(copy.deepcopy(plan["policies"][0]))

        errors = planner.validate_policies(plan)

        self.assertTrue(any("duplicates" in error for error in errors), errors)

    def test_invalid_json_returns_clean_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "policies.json"
            path.write_text("{", encoding="utf-8")
            status, summary, error = planner.plan_path(path)

        self.assertEqual(status, 2)
        self.assertIsNone(summary)
        assert error is not None
        self.assertIn("Could not load baseline replay policies", error)


if __name__ == "__main__":
    unittest.main()
