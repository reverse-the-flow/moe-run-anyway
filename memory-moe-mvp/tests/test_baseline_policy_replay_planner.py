import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_baseline_policy_replay.py"
SPEC = importlib.util.spec_from_file_location("plan_baseline_policy_replay", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)

TRACE_FIXTURE = ROOT / "memory-moe-mvp" / "data" / "llama_cpp_router_trace.fixture.jsonl"
INVENTORY_FIXTURE = ROOT / "memory-moe-mvp" / "data" / "expert_inventory_manifest.fixture.json"
POLICIES_FIXTURE = ROOT / "memory-moe-mvp" / "data" / "baseline_replay_policies.json"


class BaselinePolicyReplayPlannerTests(unittest.TestCase):
    def test_default_replay_reports_required_policy_metrics_without_live_claim(self) -> None:
        summary = planner.build_replay_summary(TRACE_FIXTURE, INVENTORY_FIXTURE, POLICIES_FIXTURE)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["mode"], "baseline_policy_replay_plan")
        self.assertEqual(summary["route_count"], 8)
        self.assertEqual(summary["joined_route_count"], 8)
        self.assertEqual(summary["unique_expert_count"], 8)
        self.assertFalse(summary["phase_3_gate"]["ready_for_live_spike"])
        self.assertFalse(summary["phase_3_gate"]["dense_fallback_comparison_ready"])
        self.assertFalse(summary["phase_3_gate"]["policy_candidate_ready"])
        self.assertIn("no_replay_policy_candidate", summary["phase_3_gate"]["missing_for_phase_3_completion"])
        self.assertFalse(summary["policy_candidate_diagnostics"]["candidate_ready"])
        blocker_ids = {item["id"] for item in summary["policy_candidate_diagnostics"]["blockers"]}
        self.assertIn("no_replay_policy_candidate", blocker_ids)
        self.assertIn("no_reuse_distance_observations", blocker_ids)
        self.assertIn("planner does not claim live expert paging", summary["safety_contract"])

        by_policy = {item["policy_id"]: item for item in summary["policy_results"]}
        self.assertEqual(set(by_policy), set(planner.plan_baseline_replay_policies.REQUIRED_POLICY_IDS))
        for policy_id, result in by_policy.items():
            with self.subTest(policy_id=policy_id):
                self.assertIn("miss_rate", result)
                self.assertIn("warm_hit_rate", result)
                self.assertIn("churn", result)
                self.assertIn("fallback_frequency", result)
                self.assertIn("rejected_policy_reasons", result)

        self.assertEqual(by_policy["observe_only"]["miss_rate"], 1.0)
        self.assertGreater(by_policy["preload_shortlist"]["warm_hit_rate"], by_policy["keep_hot"]["warm_hit_rate"])
        self.assertIn(
            "fallback_output_missing_for_quality_comparison",
            by_policy["fallback_dense"]["rejected_policy_reasons"],
        )

    def test_real_model_pair_removes_real_pair_completion_gap(self) -> None:
        model_id = "local-mixtral.gguf"
        trace_rows = []
        for line in TRACE_FIXTURE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["model"] = model_id
            trace_rows.append(row)

        inventory = json.loads(INVENTORY_FIXTURE.read_text(encoding="utf-8"))
        inventory["name"] = "Local Mixtral Expert Inventory"
        inventory["model_id"] = model_id
        inventory["inventory_scope"] = "scanner_derived_metadata"
        inventory["source_files"] = [
            {
                "path": model_id,
                "source_format": "gguf",
                "byte_length": inventory["source_files"][0]["byte_length"],
                "note": "Metadata-only scan from local GGUF header.",
            }
        ]
        for entry in inventory["entries"]:
            entry["source_file"] = model_id

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trace_path = root / "router-events.jsonl"
            inventory_path = root / "expert-inventory.json"
            trace_path.write_text("\n".join(json.dumps(row) for row in trace_rows) + "\n", encoding="utf-8")
            inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

            summary = planner.build_replay_summary(trace_path, inventory_path, POLICIES_FIXTURE)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertTrue(summary["phase_3_gate"]["real_model_pair_ready"])
        self.assertFalse(summary["phase_3_gate"]["fixture_only_pair"])
        missing = summary["phase_3_gate"]["missing_for_phase_3_completion"]
        self.assertNotIn("real_model_trace_inventory_pairing_ready", missing)
        self.assertIn("no_replay_policy_candidate", missing)
        self.assertIn("dense_fallback_output_artifact_for_quality_bounds", missing)

    def test_repeated_routes_make_keep_hot_warm_hits_visible_with_full_budget(self) -> None:
        rows = []
        for line in TRACE_FIXTURE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        rows.append(copy.deepcopy(rows[0]))

        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "trace.jsonl"
            trace_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            summary = planner.build_replay_summary(
                trace_path,
                INVENTORY_FIXTURE,
                POLICIES_FIXTURE,
                resident_budget_bytes=25165824,
            )

        self.assertTrue(summary["valid"], summary["errors"])
        by_policy = {item["policy_id"]: item for item in summary["policy_results"]}
        self.assertGreater(by_policy["keep_hot"]["hit_count"], 0)
        self.assertLess(by_policy["keep_hot"]["miss_rate"], by_policy["observe_only"]["miss_rate"])
        self.assertGreater(by_policy["keep_hot"]["warm_hit_rate"], 0.0)
        self.assertTrue(summary["phase_3_gate"]["policy_candidate_ready"])
        self.assertTrue(summary["policy_candidate_diagnostics"]["candidate_ready"])
        self.assertNotIn("no_replay_policy_candidate", summary["phase_3_gate"]["missing_for_phase_3_completion"])

    def test_tiny_budget_reports_budget_rejections(self) -> None:
        summary = planner.build_replay_summary(
            TRACE_FIXTURE,
            INVENTORY_FIXTURE,
            POLICIES_FIXTURE,
            resident_budget_bytes=1,
        )

        self.assertTrue(summary["valid"], summary["errors"])
        by_policy = {item["policy_id"]: item for item in summary["policy_results"]}
        self.assertEqual(by_policy["keep_hot"]["fallback_frequency"], 1.0)
        self.assertIn("resident_budget_exceeded", by_policy["keep_hot"]["rejected_policy_reasons"])
        self.assertIn("shortlist_bytes_exceed_budget", by_policy["preload_shortlist"]["rejected_policy_reasons"])

    def test_missing_inventory_route_makes_replay_invalid(self) -> None:
        inventory = json.loads(INVENTORY_FIXTURE.read_text(encoding="utf-8"))
        removed_bytes = 0
        kept_entries = []
        for entry in inventory["entries"]:
            if entry["layer_id"] == 1 and entry["expert_id"] == 7:
                removed_bytes += entry["estimated_residency_bytes"]
                continue
            kept_entries.append(entry)
        inventory["entries"] = kept_entries
        inventory["expected"]["expert_count"] -= 1
        inventory["expected"]["component_count"] -= 3
        inventory["expected"]["total_estimated_residency_bytes"] -= removed_bytes

        with tempfile.TemporaryDirectory() as temp_dir:
            inventory_path = Path(temp_dir) / "inventory.json"
            inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
            status, summary, error = planner.plan_paths(TRACE_FIXTURE, inventory_path, POLICIES_FIXTURE)

        self.assertEqual(status, 2)
        self.assertIsNone(error)
        assert summary is not None
        self.assertFalse(summary["valid"])
        self.assertEqual(summary["policy_results"], [])
        self.assertTrue(any("missing from inventory" in item for item in summary["errors"]))

    def test_invalid_policy_contract_returns_clean_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            policies_path = Path(temp_dir) / "policies.json"
            policies_path.write_text("{", encoding="utf-8")
            status, summary, error = planner.plan_paths(TRACE_FIXTURE, INVENTORY_FIXTURE, policies_path)

        self.assertEqual(status, 2)
        self.assertIsNone(summary)
        assert error is not None
        self.assertIn("Could not build baseline policy replay plan", error)

    def test_negative_budget_is_rejected_before_replay(self) -> None:
        status, summary, error = planner.plan_paths(
            TRACE_FIXTURE,
            INVENTORY_FIXTURE,
            POLICIES_FIXTURE,
            resident_budget_bytes=-1,
        )

        self.assertEqual(status, 2)
        self.assertIsNone(summary)
        self.assertEqual(error, "resident budget bytes must be >= 0")


if __name__ == "__main__":
    unittest.main()
