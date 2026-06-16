import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_edge_hardware_quickstart.py"
SPEC = importlib.util.spec_from_file_location("plan_edge_hardware_quickstart", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


class EdgeHardwareQuickstartPlannerTests(unittest.TestCase):
    def test_default_matrix_validates_and_reports_required_shape(self) -> None:
        matrix = planner.load_matrix(planner.DEFAULT_MATRIX_PATH)
        summary = planner.build_summary(matrix, planner.DEFAULT_MATRIX_PATH)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["schema_version"], planner.SUPPORTED_SCHEMA_VERSION)
        self.assertEqual(summary["tier_count"], 9)
        self.assertIn("Planner does not download models.", summary["safety_contract"])
        self.assertIn("Planner does not inspect devices.", summary["safety_contract"])
        self.assertIn("Planner does not send prompt traffic.", summary["safety_contract"])

    def test_required_hardware_tiers_are_present(self) -> None:
        matrix = planner.load_matrix(planner.DEFAULT_MATRIX_PATH)
        tier_ids = {tier["id"] for tier in matrix["hardware_tiers"]}

        self.assertEqual(tier_ids, planner.REQUIRED_TIER_IDS)

    def test_each_tier_includes_small_dense_routing_class(self) -> None:
        matrix = planner.load_matrix(planner.DEFAULT_MATRIX_PATH)

        for tier in matrix["hardware_tiers"]:
            self.assertIn(
                "small_dense_routable_experts",
                tier["candidate_model_classes"],
                tier["id"],
            )

    def test_android_emulator_caveat_distinguishes_real_device_testing(self) -> None:
        matrix = planner.load_matrix(planner.DEFAULT_MATRIX_PATH)
        android_tier = next(
            tier for tier in matrix["hardware_tiers"] if tier["id"] == "android_phone_emulator"
        )
        joined_risks = " ".join(android_tier["risk_notes"])
        joined_notes = " ".join(android_tier["context_kv_cache_notes"])

        self.assertIn("Emulator", joined_risks)
        self.assertIn("real mobile", joined_risks)
        self.assertIn("Mobile KV cache", joined_notes)

    def test_validation_rejects_missing_android_emulator_tier(self) -> None:
        matrix = planner.load_matrix(planner.DEFAULT_MATRIX_PATH)
        matrix["hardware_tiers"] = [
            tier for tier in matrix["hardware_tiers"] if tier["id"] != "android_phone_emulator"
        ]

        errors = planner.validate_matrix(matrix)

        self.assertTrue(any("android_phone_emulator" in error for error in errors), errors)

    def test_validation_requires_semantic_claim_gate(self) -> None:
        matrix = planner.load_matrix(planner.DEFAULT_MATRIX_PATH)
        matrix["recommended_test_path"] = [
            step for step in matrix["recommended_test_path"] if step["id"] != "semantic_claim_gate"
        ]

        errors = planner.validate_matrix(matrix)

        self.assertTrue(any("semantic_claim_gate" in error for error in errors), errors)

    def test_cli_json_summary_shape(self) -> None:
        matrix = planner.load_matrix(planner.DEFAULT_MATRIX_PATH)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "edge.json"
            path.write_text(json.dumps(matrix), encoding="utf-8")
            status, summary, error = planner.plan_path(path)

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["mode"], "edge_hardware_quickstart_plan")
        self.assertIn("hardware_tiers", summary)
        self.assertIn("recommended_test_path", summary)
        self.assertEqual(summary["tier_count"], len(planner.REQUIRED_TIER_IDS))

    def test_cli_returns_nonzero_for_invalid_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "edge.json"
            path.write_text("{}", encoding="utf-8")

            self.assertEqual(planner.main_from_test_path(path), 2)

    def test_cli_returns_nonzero_for_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "edge.json"
            path.write_text("{", encoding="utf-8")

            status, summary, error = planner.plan_path(path)

        self.assertEqual(status, 2)
        self.assertIsNone(summary)
        assert error is not None
        self.assertIn("Could not load edge hardware quickstart matrix", error)


if __name__ == "__main__":
    unittest.main()
