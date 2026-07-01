import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_expert_store_layout.py"
SPEC = importlib.util.spec_from_file_location("plan_expert_store_layout", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)

INVENTORY_FIXTURE = ROOT / "memory-moe-mvp" / "data" / "expert_inventory_manifest.fixture.json"


class ExpertStoreLayoutPlannerTests(unittest.TestCase):
    def test_default_inventory_builds_streaming_layout_without_writes(self) -> None:
        summary = planner.build_layout_summary(INVENTORY_FIXTURE)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["mode"], "expert_store_layout_plan")
        self.assertEqual(summary["expert_count"], 8)
        self.assertEqual(summary["component_count"], 24)
        self.assertEqual(summary["complete_expert_count"], 8)
        self.assertEqual(summary["total_packed_bytes"], 25165824)
        self.assertGreater(summary["required_disk_bytes"], summary["total_packed_bytes"])
        self.assertTrue(summary["can_stream_without_full_model_load"])
        self.assertFalse(summary["write_path_implemented"])
        self.assertEqual(summary["missing_components"], [])
        self.assertEqual(summary["coverage_counts"]["complete"], 24)
        self.assertIn("planner does not write packed expert stores", summary["safety_contract"])

    def test_expert_layouts_have_stable_target_offsets(self) -> None:
        summary = planner.build_layout_summary(INVENTORY_FIXTURE, store_name="packed-fixture")
        first = summary["expert_layouts"][0]

        self.assertEqual(first["target_file"], "packed-fixture/layers/layer-0/expert-0.bin")
        self.assertEqual(first["packed_bytes"], 3145728)
        self.assertEqual([item["component_name"] for item in first["components"]], ["gate_proj", "up_proj", "down_proj"])
        self.assertEqual([item["target_byte_offset"] for item in first["components"]], [0, 1048576, 2097152])
        self.assertEqual([item["target_byte_end"] for item in first["components"]], [1048576, 2097152, 3145728])

    def test_source_read_plan_summarizes_ranges_without_reading_values(self) -> None:
        summary = planner.build_layout_summary(INVENTORY_FIXTURE)

        self.assertEqual(len(summary["source_read_plan"]), 1)
        source = summary["source_read_plan"][0]
        self.assertEqual(source["source_file"], "fixture-mixtral.gguf")
        self.assertEqual(source["range_count"], 24)
        self.assertEqual(source["bytes_to_read"], 25165824)
        self.assertEqual(source["first_byte_offset"], 0)
        self.assertEqual(source["last_byte_end"], 25165824)

    def test_missing_component_is_reported_as_invalid_layout(self) -> None:
        inventory = json.loads(INVENTORY_FIXTURE.read_text(encoding="utf-8"))
        inventory["entries"] = [
            entry
            for entry in inventory["entries"]
            if not (
                entry["layer_id"] == 0
                and entry["expert_id"] == 1
                and entry["component_name"] == "down_proj"
            )
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "inventory.json"
            path.write_text(json.dumps(inventory), encoding="utf-8")
            status, summary, error = planner.plan_path(path)

        self.assertEqual(status, 2)
        self.assertIsNone(error)
        assert summary is not None
        self.assertFalse(summary["valid"])
        self.assertFalse(summary["can_stream_without_full_model_load"])
        self.assertTrue(any("missing required components" in item for item in summary["errors"]))
        self.assertEqual(summary["missing_components"][0]["layer_id"], 0)
        self.assertEqual(summary["missing_components"][0]["expert_id"], 1)
        self.assertEqual(summary["missing_components"][0]["missing_components"], ["down_proj"])

    def test_invalid_store_name_is_rejected_before_layout(self) -> None:
        status, summary, error = planner.plan_path(INVENTORY_FIXTURE, store_name="   ")

        self.assertEqual(status, 2)
        self.assertIsNone(summary)
        self.assertEqual(error, "store name must be a non-empty string")

    def test_bad_json_returns_clean_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "inventory.json"
            path.write_text("{", encoding="utf-8")
            status, summary, error = planner.plan_path(path)

        self.assertEqual(status, 2)
        self.assertIsNone(summary)
        assert error is not None
        self.assertIn("Could not build expert store layout plan", error)


if __name__ == "__main__":
    unittest.main()
