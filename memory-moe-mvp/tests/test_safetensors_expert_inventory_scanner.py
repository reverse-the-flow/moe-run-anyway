import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "scan_safetensors_expert_inventory.py"
SPEC = importlib.util.spec_from_file_location("scan_safetensors_expert_inventory", SCRIPT_PATH)
scanner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = scanner
SPEC.loader.exec_module(scanner)


def write_safetensors_header(path: Path, tensors: list[tuple[str, str, list[int], int]]) -> None:
    header = {}
    offset = 0
    for tensor_name, dtype, shape, byte_length in tensors:
        header[tensor_name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + byte_length],
        }
        offset += byte_length
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(len(header_bytes).to_bytes(8, byteorder="little") + header_bytes + (b"\0" * offset))


def expert_tensors(*, missing_down_for: tuple[int, int] | None = None) -> list[tuple[str, str, list[int], int]]:
    tensors: list[tuple[str, str, list[int], int]] = [
        ("model.embed_tokens.weight", "F16", [8, 4], 64),
    ]
    for layer_id in (0, 1):
        for expert_id in (0, 1):
            for component in ("w1", "w2", "w3"):
                if missing_down_for == (layer_id, expert_id) and component == "w2":
                    continue
                tensors.append(
                    (
                        f"model.layers.{layer_id}.block_sparse_moe.experts.{expert_id}.{component}.weight",
                        "F16",
                        [2, 4],
                        16,
                    )
                )
    return tensors


class SafetensorsExpertInventoryScannerTests(unittest.TestCase):
    def test_single_safetensors_file_emits_valid_inventory_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.safetensors"
            write_safetensors_header(path, expert_tensors())

            summary = scanner.build_scan_summary(path, model_id="fixture/hf-mixtral")

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["scan_stats"]["source_file_count"], 1)
        self.assertEqual(summary["scan_stats"]["recognized_expert_tensor_count"], 12)
        self.assertEqual(summary["scan_stats"]["ignored_tensor_count"], 1)
        inventory = summary["inventory_summary"]
        self.assertEqual(inventory["source_format"], "safetensors")
        self.assertEqual(inventory["backend_family"], "hookable_pytorch")
        self.assertEqual(inventory["expert_count"], 4)
        self.assertEqual(inventory["component_count"], 12)
        self.assertEqual(inventory["total_estimated_residency_bytes"], 192)
        self.assertEqual(inventory["required_components"], ["gate_proj", "up_proj", "down_proj"])

    def test_model_index_limits_scan_to_indexed_safetensors_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            shard_path = model_dir / "model-00001-of-00001.safetensors"
            write_safetensors_header(shard_path, expert_tensors())
            write_safetensors_header(model_dir / "unused.safetensors", expert_tensors())
            index = {
                "metadata": {"total_size": 192},
                "weight_map": {
                    "model.layers.0.block_sparse_moe.experts.0.w1.weight": shard_path.name,
                },
            }
            (model_dir / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

            summary = scanner.build_scan_summary(model_dir)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["scan_stats"]["source_file_count"], 1)
        self.assertEqual(summary["manifest"]["source_files"][0]["path"], "model-00001-of-00001.safetensors")

    def test_missing_component_is_reported_by_inventory_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.safetensors"
            write_safetensors_header(path, expert_tensors(missing_down_for=(0, 0)))

            summary = scanner.build_scan_summary(path)

        self.assertFalse(summary["valid"])
        self.assertTrue(
            any("missing required components: down_proj" in error for error in summary["errors"]),
            summary["errors"],
        )

    def test_cli_can_write_manifest_without_tensor_payload_reads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.safetensors"
            output_path = Path(temp_dir) / "inventory.json"
            write_safetensors_header(model_path, expert_tensors())

            status, summary, error = scanner.scan_path(model_path, output=output_path)

            self.assertEqual(status, 0)
            self.assertIsNone(error)
            assert summary is not None
            self.assertTrue(output_path.exists())
            written = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(written["schema_version"], scanner.plan_expert_inventory.SUPPORTED_SCHEMA_VERSION)
            self.assertEqual(len(written["entries"]), 12)

    def test_invalid_safetensors_header_returns_clean_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.safetensors"
            path.write_bytes(b"short")

            status, summary, error = scanner.scan_path(path)

        self.assertEqual(status, 2)
        self.assertIsNone(summary)
        assert error is not None
        self.assertIn("Could not scan safetensors expert inventory", error)


if __name__ == "__main__":
    unittest.main()
