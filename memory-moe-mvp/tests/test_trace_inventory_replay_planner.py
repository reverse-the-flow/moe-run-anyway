import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_trace_inventory_replay.py"
SPEC = importlib.util.spec_from_file_location("plan_trace_inventory_replay", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)

TRACE_FIXTURE = ROOT / "memory-moe-mvp" / "data" / "llama_cpp_router_trace.fixture.jsonl"
INVENTORY_FIXTURE = ROOT / "memory-moe-mvp" / "data" / "expert_inventory_manifest.fixture.json"


class TraceInventoryReplayPlannerTests(unittest.TestCase):
    def test_default_trace_and_inventory_join_to_byte_estimates(self) -> None:
        summary = planner.build_join_summary(TRACE_FIXTURE, INVENTORY_FIXTURE)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["trace_event_count"], 6)
        self.assertEqual(summary["selected_expert_event_count"], 2)
        self.assertEqual(summary["route_count"], 8)
        self.assertEqual(summary["joined_route_count"], 8)
        self.assertEqual(summary["missing_route_count"], 0)
        self.assertEqual(summary["unique_expert_count"], 8)
        self.assertEqual(summary["unique_estimated_residency_bytes"], 25165824)
        self.assertEqual(summary["route_estimated_residency_bytes"], 25165824)
        self.assertEqual(summary["reuse_distance"]["observations"], 0)
        self.assertTrue(summary["policy_replay_prerequisites"]["inventory_join_ready"])
        self.assertIn("keep_hot", summary["policy_replay_prerequisites"]["baseline_policy_names"])

    def test_missing_inventory_route_is_reported_without_claiming_ready(self) -> None:
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
            status, summary, error = planner.plan_paths(TRACE_FIXTURE, inventory_path)

        self.assertEqual(status, 2)
        self.assertIsNone(error)
        assert summary is not None
        self.assertFalse(summary["valid"])
        self.assertEqual(summary["missing_route_count"], 1)
        self.assertFalse(summary["policy_replay_prerequisites"]["inventory_join_ready"])
        self.assertTrue(any("missing from inventory" in item for item in summary["errors"]))

    def test_trace_without_selected_experts_is_invalid(self) -> None:
        rows = []
        for line in TRACE_FIXTURE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["tensor_kind"] != "selected_experts":
                rows.append(row)

        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "trace.jsonl"
            trace_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            status, summary, error = planner.plan_paths(trace_path, INVENTORY_FIXTURE)

        self.assertEqual(status, 2)
        self.assertIsNone(error)
        assert summary is not None
        self.assertFalse(summary["valid"])
        self.assertTrue(any("selected_experts" in item for item in summary["errors"]))

    def test_reuse_distance_is_reported_for_repeated_routes(self) -> None:
        rows = []
        for line in TRACE_FIXTURE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        duplicate_rows = rows + [copy.deepcopy(rows[0])]

        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "trace.jsonl"
            trace_path.write_text("\n".join(json.dumps(row) for row in duplicate_rows) + "\n", encoding="utf-8")
            summary = planner.build_join_summary(trace_path, INVENTORY_FIXTURE)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["route_count"], 12)
        self.assertEqual(summary["unique_expert_count"], 8)
        self.assertGreater(summary["reuse_distance"]["observations"], 0)
        self.assertEqual(summary["reuse_distance"]["first_touch_count"], 8)

    def test_cli_returns_nonzero_for_bad_trace_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "trace.jsonl"
            trace_path.write_text("{", encoding="utf-8")
            status, summary, error = planner.plan_paths(trace_path, INVENTORY_FIXTURE)

        self.assertEqual(status, 2)
        self.assertIsNone(summary)
        assert error is not None
        self.assertIn("Could not build trace inventory replay plan", error)


if __name__ == "__main__":
    unittest.main()
