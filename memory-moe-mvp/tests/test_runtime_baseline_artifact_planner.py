import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_runtime_baseline_artifacts.py"
SPEC = importlib.util.spec_from_file_location("plan_runtime_baseline_artifacts", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


class RuntimeBaselineArtifactPlannerTests(unittest.TestCase):
    def test_default_contract_validates_and_separates_evidence_labels(self) -> None:
        contract = planner.load_contract(planner.DEFAULT_CONTRACT_PATH)
        summary = planner.build_summary(contract, planner.DEFAULT_CONTRACT_PATH)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["phase"], "phase_1")
        self.assertIn("runtime_baseline_preflight_bundle", summary["artifact_classes"])
        self.assertIn("runtime_baseline_probe_bundle", summary["artifact_classes"])
        self.assertIn("runtime_request_telemetry", summary["runtime_evidence_labels"])
        self.assertIn("semantic_router_trace", summary["semantic_routing_evidence_labels"])

    def test_runtime_artifacts_cannot_claim_semantic_expert_ids(self) -> None:
        contract = planner.load_contract(planner.DEFAULT_CONTRACT_PATH)
        contract["artifact_classes"][0]["may_claim_semantic_expert_ids"] = True

        errors = planner.validate_contract(contract)

        self.assertTrue(
            any("may_claim_semantic_expert_ids must be false" in error for error in errors),
            errors,
        )

    def test_cli_returns_nonzero_for_invalid_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "contract.json"
            path.write_text(json.dumps({}), encoding="utf-8")

            self.assertEqual(planner.main_from_test_path(path), 2)


if __name__ == "__main__":
    unittest.main()
