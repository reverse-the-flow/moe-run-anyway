import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "scan_gguf_expert_inventory.py"
SPEC = importlib.util.spec_from_file_location("scan_gguf_expert_inventory", SCRIPT_PATH)
scanner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = scanner
SPEC.loader.exec_module(scanner)


def pack_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def pack_metadata_value(value_type: int, value) -> bytes:
    if value_type == scanner.GGUF_VALUE_TYPES["UINT32"]:
        return struct.pack("<I", value)
    if value_type == scanner.GGUF_VALUE_TYPES["STRING"]:
        return pack_string(value)
    raise AssertionError(f"unsupported test value type {value_type}")


def pack_metadata(key: str, value_type: int, value) -> bytes:
    return pack_string(key) + struct.pack("<I", value_type) + pack_metadata_value(value_type, value)


def align_offset(offset: int, alignment: int) -> int:
    remainder = offset % alignment
    return offset if remainder == 0 else offset + alignment - remainder


def write_gguf(
    path: Path,
    *,
    metadata: list[tuple[str, int, object]],
    tensors: list[tuple[str, list[int], int, int]],
    alignment: int = 32,
) -> None:
    header = bytearray()
    header += b"GGUF"
    header += struct.pack("<I", 3)
    header += struct.pack("<Q", len(tensors))
    header += struct.pack("<Q", len(metadata))
    for key, value_type, value in metadata:
        header += pack_metadata(key, value_type, value)

    offset = 0
    payload = bytearray()
    for tensor_name, dimensions, ggml_type, byte_length in tensors:
        header += pack_string(tensor_name)
        header += struct.pack("<I", len(dimensions))
        for dimension in dimensions:
            header += struct.pack("<Q", dimension)
        header += struct.pack("<I", ggml_type)
        header += struct.pack("<Q", offset)
        payload += bytes([len(payload) % 251]) * byte_length
        offset += byte_length

    data_start = align_offset(len(header), alignment)
    path.write_bytes(bytes(header) + (b"\0" * (data_start - len(header))) + bytes(payload))


def default_metadata(*, expert_count: int | None = 4, expert_used_count: int = 2):
    metadata = [
        ("general.architecture", scanner.GGUF_VALUE_TYPES["STRING"], "llama"),
        ("general.alignment", scanner.GGUF_VALUE_TYPES["UINT32"], 32),
        ("llama.expert_used_count", scanner.GGUF_VALUE_TYPES["UINT32"], expert_used_count),
    ]
    if expert_count is not None:
        metadata.append(("llama.expert_count", scanner.GGUF_VALUE_TYPES["UINT32"], expert_count))
    return metadata


def stacked_tensors(*, include_down: bool = True) -> list[tuple[str, list[int], int, int]]:
    tensors = [("token_embd.weight", [8, 4], 1, 32)]
    components = ["ffn_gate_exps", "ffn_up_exps"]
    if include_down:
        components.append("ffn_down_exps")
    for layer_id in (0, 1):
        for component in components:
            tensors.append((f"blk.{layer_id}.{component}.weight", [4, 2, 4], 1, 64))
    return tensors


def nemotron_metadata(*, expert_count: int = 4, expert_used_count: int = 2):
    return [
        ("general.architecture", scanner.GGUF_VALUE_TYPES["STRING"], "nemotron_h_moe"),
        ("general.alignment", scanner.GGUF_VALUE_TYPES["UINT32"], 32),
        ("nemotron_h_moe.expert_count", scanner.GGUF_VALUE_TYPES["UINT32"], expert_count),
        ("nemotron_h_moe.expert_used_count", scanner.GGUF_VALUE_TYPES["UINT32"], expert_used_count),
    ]


def nemotron_stacked_tensors() -> list[tuple[str, list[int], int, int]]:
    tensors = [("token_embd.weight", [8, 4], 1, 32)]
    for layer_id in (1, 3):
        tensors.append((f"blk.{layer_id}.ffn_gate_inp.weight", [4, 4], 0, 64))
        tensors.append((f"blk.{layer_id}.ffn_up_exps.weight", [4, 2, 4], 1, 64))
        tensors.append((f"blk.{layer_id}.ffn_down_exps.weight", [2, 4, 4], 1, 64))
        tensors.append((f"blk.{layer_id}.ffn_up_shexp.weight", [4, 2], 1, 32))
    return tensors


def gemma_metadata(*, expert_count: int = 4, expert_used_count: int = 2):
    return [
        ("general.architecture", scanner.GGUF_VALUE_TYPES["STRING"], "gemma4"),
        ("general.alignment", scanner.GGUF_VALUE_TYPES["UINT32"], 32),
        ("gemma4.expert_count", scanner.GGUF_VALUE_TYPES["UINT32"], expert_count),
        ("gemma4.expert_used_count", scanner.GGUF_VALUE_TYPES["UINT32"], expert_used_count),
    ]


def gemma_fused_tensors() -> list[tuple[str, list[int], int, int]]:
    return [
        ("token_embd.weight", [8, 4], 1, 32),
        ("blk.0.ffn_gate_up_exps.weight", [4, 4, 4], 1, 128),
        ("blk.0.ffn_down_exps.weight", [2, 4, 4], 1, 64),
        ("blk.0.ffn_down_exps.scale", [4], 0, 16),
        ("blk.0.ffn_gate_inp.weight", [4, 4], 0, 64),
    ]


def per_expert_tensors() -> list[tuple[str, list[int], int, int]]:
    tensors = [("token_embd.weight", [8, 4], 1, 32)]
    for expert_id in (0, 1):
        for component in ("ffn_gate", "ffn_up", "ffn_down"):
            tensors.append((f"blk.0.{component}.{expert_id}.weight", [2, 4], 1, 16))
    return tensors


class GGUFExpertInventoryScannerTests(unittest.TestCase):
    def test_stacked_expert_tensors_emit_valid_inventory_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mixtral.gguf"
            write_gguf(path, metadata=default_metadata(expert_count=4), tensors=stacked_tensors())

            summary = scanner.build_scan_summary(path, model_id="fixture/mixtral-gguf")

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["scan_stats"]["stacked_expert_tensor_count"], 6)
        self.assertEqual(summary["scan_stats"]["per_expert_tensor_count"], 0)
        inventory = summary["inventory_summary"]
        self.assertEqual(inventory["source_format"], "gguf")
        self.assertEqual(inventory["backend_family"], "llama_cpp")
        self.assertEqual(inventory["expert_count"], 8)
        self.assertEqual(inventory["component_count"], 24)
        self.assertEqual(inventory["total_estimated_residency_bytes"], 384)
        self.assertEqual(inventory["required_components"], ["gate_proj", "up_proj", "down_proj"])
        self.assertIn({"layer_id": 1, "expert_id": 3}, inventory["trace_join_keys"])

    def test_nemotron_up_down_profile_is_valid_without_per_expert_gate_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nemotron.gguf"
            write_gguf(path, metadata=nemotron_metadata(expert_count=4), tensors=nemotron_stacked_tensors())

            summary = scanner.build_scan_summary(path, model_id="fixture/nemotron-h-gguf")

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["scan_stats"]["stacked_expert_tensor_count"], 4)
        self.assertEqual(summary["scan_stats"]["fused_stacked_expert_tensor_count"], 0)
        self.assertEqual(summary["scan_stats"]["component_profile"]["profile_id"], "nemotron_h_up_down")
        inventory = summary["inventory_summary"]
        self.assertEqual(inventory["required_components"], ["up_proj", "down_proj"])
        self.assertEqual(inventory["expert_count"], 8)
        self.assertEqual(inventory["component_count"], 16)

    def test_gemma_fused_gate_up_profile_splits_metadata_slices(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "gemma4.gguf"
            write_gguf(path, metadata=gemma_metadata(expert_count=4), tensors=gemma_fused_tensors())

            summary = scanner.build_scan_summary(path, model_id="fixture/gemma4-gguf")

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["scan_stats"]["stacked_expert_tensor_count"], 1)
        self.assertEqual(summary["scan_stats"]["fused_stacked_expert_tensor_count"], 1)
        self.assertEqual(summary["scan_stats"]["stacked_expert_scale_tensor_count"], 1)
        self.assertEqual(summary["scan_stats"]["component_profile"]["profile_id"], "gate_up_down")
        inventory = summary["inventory_summary"]
        self.assertEqual(
            inventory["required_components"],
            ["gate_proj", "up_proj", "down_proj", "down_proj_scale"],
        )
        self.assertEqual(inventory["expert_count"], 4)
        self.assertEqual(inventory["component_count"], 16)
        self.assertEqual(inventory["total_estimated_residency_bytes"], 208)
        manifest_entries = summary["manifest"]["entries"]
        first_expert = [
            entry for entry in manifest_entries if entry["layer_id"] == 0 and entry["expert_id"] == 0
        ]
        self.assertEqual([entry["component_name"] for entry in first_expert], [
            "gate_proj",
            "up_proj",
            "down_proj",
            "down_proj_scale",
        ])
        self.assertEqual(first_expert[0]["source_layout"], "fused_stacked_expert_tensor")
        self.assertEqual(first_expert[1]["source_layout"], "fused_stacked_expert_tensor")
        self.assertEqual(first_expert[3]["source_layout"], "stacked_expert_scale_tensor")

    def test_per_expert_tensor_layout_does_not_require_expert_count_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "per-expert.gguf"
            write_gguf(path, metadata=default_metadata(expert_count=None), tensors=per_expert_tensors())

            summary = scanner.build_scan_summary(path)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["scan_stats"]["per_expert_tensor_count"], 6)
        self.assertEqual(summary["scan_stats"]["stacked_expert_tensor_count"], 0)
        inventory = summary["inventory_summary"]
        self.assertEqual(inventory["expert_count"], 2)
        self.assertEqual(inventory["component_count"], 6)
        self.assertEqual(inventory["total_estimated_residency_bytes"], 96)

    def test_stacked_tensor_without_expert_count_returns_clean_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing-expert-count.gguf"
            write_gguf(path, metadata=default_metadata(expert_count=None), tensors=stacked_tensors())

            status, summary, error = scanner.scan_path(path)

        self.assertEqual(status, 2)
        self.assertIsNone(summary)
        assert error is not None
        self.assertIn("expert_count is unavailable", error)

    def test_missing_component_is_reported_by_inventory_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing-down.gguf"
            write_gguf(path, metadata=default_metadata(expert_count=4), tensors=stacked_tensors(include_down=False))

            summary = scanner.build_scan_summary(path)

        self.assertFalse(summary["valid"])
        self.assertTrue(
            any("missing required components: down_proj" in error for error in summary["errors"]),
            summary["errors"],
        )

    def test_cli_can_write_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "mixtral.gguf"
            output_path = Path(temp_dir) / "inventory.json"
            write_gguf(model_path, metadata=default_metadata(expert_count=4), tensors=stacked_tensors())

            status, summary, error = scanner.scan_path(model_path, output=output_path)

            self.assertEqual(status, 0)
            self.assertIsNone(error)
            assert summary is not None
            self.assertTrue(output_path.exists())
            written = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(written["schema_version"], scanner.plan_expert_inventory.SUPPORTED_SCHEMA_VERSION)
            self.assertEqual(len(written["entries"]), 24)

    def test_invalid_gguf_returns_clean_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.gguf"
            path.write_bytes(b"nope")

            status, summary, error = scanner.scan_path(path)

        self.assertEqual(status, 2)
        self.assertIsNone(summary)
        assert error is not None
        self.assertIn("Could not scan GGUF expert inventory", error)


if __name__ == "__main__":
    unittest.main()
