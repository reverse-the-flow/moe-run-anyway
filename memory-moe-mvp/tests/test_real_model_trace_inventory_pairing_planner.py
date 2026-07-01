import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_real_model_trace_inventory_pairing.py"
SPEC = importlib.util.spec_from_file_location("plan_real_model_trace_inventory_pairing", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)

TRACE_FIXTURE = ROOT / "memory-moe-mvp" / "data" / "llama_cpp_router_trace.fixture.jsonl"
INVENTORY_FIXTURE = ROOT / "memory-moe-mvp" / "data" / "expert_inventory_manifest.fixture.json"


def load_trace_rows():
    return [
        json.loads(line)
        for line in TRACE_FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def make_real_like_pair(temp_dir: Path) -> tuple[Path, Path]:
    rows = load_trace_rows()
    for row in rows:
        row["model"] = "models/local-mixtral.gguf"

    inventory = json.loads(INVENTORY_FIXTURE.read_text(encoding="utf-8"))
    inventory["name"] = "Local Mixtral Expert Inventory"
    inventory["model_id"] = "local-mixtral.gguf"
    inventory["inventory_scope"] = "scanner_output_for_phase_3_pairing_test"
    inventory["source_files"][0]["path"] = "models/local-mixtral.gguf"
    inventory["source_files"][0]["note"] = "Header-derived inventory from scanner output."
    for entry in inventory["entries"]:
        entry["source_file"] = "models/local-mixtral.gguf"

    trace_path = temp_dir / "trace.jsonl"
    inventory_path = temp_dir / "inventory.json"
    trace_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    return trace_path, inventory_path


class RealModelTraceInventoryPairingPlannerTests(unittest.TestCase):
    def test_default_fixture_pair_validates_but_is_not_real_model_evidence(self) -> None:
        summary = planner.build_pairing_summary(TRACE_FIXTURE, INVENTORY_FIXTURE)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertTrue(summary["trace_inventory_join_valid"])
        self.assertEqual(summary["joined_route_count"], 8)
        self.assertTrue(summary["model_match"])
        self.assertTrue(summary["backend_match"])
        self.assertTrue(summary["contract_match"])
        self.assertTrue(summary["fixture_only_pair"])
        self.assertFalse(summary["real_model_pair_ready"])
        self.assertTrue(any(item["id"] == "fixture_only_trace_inventory_pair" for item in summary["blockers"]))

    def test_require_real_model_rejects_fixture_pair(self) -> None:
        status, summary, error = planner.plan_paths(
            TRACE_FIXTURE,
            INVENTORY_FIXTURE,
            require_real_model=True,
        )

        self.assertEqual(status, 2)
        self.assertIsNone(error)
        assert summary is not None
        self.assertFalse(summary["valid"])
        self.assertTrue(any("real-model trace/inventory pair is required" in item for item in summary["errors"]))

    def test_real_like_pair_can_satisfy_strict_gate_without_live_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path, inventory_path = make_real_like_pair(Path(temp_dir))
            status, summary, error = planner.plan_paths(
                trace_path,
                inventory_path,
                require_real_model=True,
            )

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        self.assertTrue(summary["valid"], summary["errors"])
        self.assertFalse(summary["fixture_only_pair"])
        self.assertTrue(summary["real_model_pair_ready"])
        self.assertEqual(summary["blockers"], [])
        self.assertTrue(summary["phase_3_gate"]["real_model_inventory_for_traced_model"])

    def test_model_mismatch_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path, inventory_path = make_real_like_pair(Path(temp_dir))
            rows = load_trace_rows()
            for row in rows:
                row["model"] = "other-model.gguf"
            trace_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            status, summary, error = planner.plan_paths(trace_path, inventory_path)

        self.assertEqual(status, 2)
        self.assertIsNone(error)
        assert summary is not None
        self.assertFalse(summary["model_match"])
        self.assertTrue(any("trace model must match inventory" in item for item in summary["errors"]))

    def test_partial_component_coverage_is_a_blocker_not_a_parser_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path, inventory_path = make_real_like_pair(Path(temp_dir))
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            inventory["entries"][0]["coverage_status"] = "partial"
            inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
            status, summary, error = planner.plan_paths(trace_path, inventory_path)

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        self.assertTrue(summary["valid"], summary["errors"])
        self.assertFalse(summary["real_model_pair_ready"])
        self.assertTrue(any(item["id"] == "inventory_component_coverage_incomplete" for item in summary["blockers"]))

    def test_multiple_trace_models_are_invalid(self) -> None:
        rows = load_trace_rows()
        mixed = copy.deepcopy(rows)
        mixed[0]["model"] = "model-a.gguf"
        mixed[1]["model"] = "model-b.gguf"

        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "trace.jsonl"
            trace_path.write_text("\n".join(json.dumps(row) for row in mixed) + "\n", encoding="utf-8")
            status, summary, error = planner.plan_paths(trace_path, INVENTORY_FIXTURE)

        self.assertEqual(status, 2)
        self.assertIsNone(error)
        assert summary is not None
        self.assertTrue(any("exactly one model" in item for item in summary["errors"]))

    def test_bad_json_returns_clean_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "trace.jsonl"
            trace_path.write_text("{", encoding="utf-8")
            status, summary, error = planner.plan_paths(trace_path, INVENTORY_FIXTURE)

        self.assertEqual(status, 2)
        self.assertIsNone(summary)
        assert error is not None
        self.assertIn("Could not build real-model trace/inventory pairing plan", error)


if __name__ == "__main__":
    unittest.main()
