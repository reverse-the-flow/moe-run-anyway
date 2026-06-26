import importlib.util
import struct
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "gguf_moe_preflight.py"
SPEC = importlib.util.spec_from_file_location("gguf_moe_preflight", SCRIPT_PATH)
preflight_mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = preflight_mod
SPEC.loader.exec_module(preflight_mod)


def write_string(handle, value: str) -> None:
    data = value.encode("utf-8")
    handle.write(struct.pack("<Q", len(data)))
    handle.write(data)


def write_metadata_string(handle, key: str, value: str) -> None:
    write_string(handle, key)
    handle.write(struct.pack("<I", 8))
    write_string(handle, value)


def write_metadata_u32(handle, key: str, value: int) -> None:
    write_string(handle, key)
    handle.write(struct.pack("<I", 4))
    handle.write(struct.pack("<I", value))


def write_metadata_string_array(handle, key: str, values: list[str]) -> None:
    write_string(handle, key)
    handle.write(struct.pack("<I", 9))
    handle.write(struct.pack("<I", 8))
    handle.write(struct.pack("<Q", len(values)))
    for value in values:
        write_string(handle, value)


def write_tensor(handle, name: str, dimensions: list[int], tensor_type: int = 0, offset: int = 0) -> None:
    write_string(handle, name)
    handle.write(struct.pack("<I", len(dimensions)))
    for dimension in dimensions:
        handle.write(struct.pack("<Q", dimension))
    handle.write(struct.pack("<I", tensor_type))
    handle.write(struct.pack("<Q", offset))


class GGUFMoEPreflightTests(unittest.TestCase):
    def test_reads_metadata_and_moe_tensor_summary_without_tensor_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fixture.gguf"
            with path.open("wb") as handle:
                handle.write(b"GGUF")
                handle.write(struct.pack("<I", 3))
                handle.write(struct.pack("<Q", 3))
                handle.write(struct.pack("<Q", 5))
                write_metadata_string(handle, "general.architecture", "fixture_moe")
                write_metadata_u32(handle, "fixture.block_count", 2)
                write_metadata_u32(handle, "fixture.expert_count", 8)
                write_metadata_u32(handle, "fixture.expert_used_count", 2)
                write_metadata_string_array(handle, "tokenizer.ggml.tokens", ["a", "b", "c"])
                write_tensor(handle, "blk.0.ffn_gate_inp.weight", [8, 16])
                write_tensor(handle, "blk.0.ffn_down_exps.weight", [8, 16, 32])
                write_tensor(handle, "output.weight", [16, 16])

            result = preflight_mod.preflight(path)

        self.assertEqual(result["version"], 3)
        self.assertEqual(result["metadata_kv_count"], 5)
        self.assertEqual(result["tensor_count"], 3)
        self.assertEqual(result["metadata"]["general.architecture"], "fixture_moe")
        self.assertEqual(result["metadata"]["fixture.expert_count"], 8)
        self.assertEqual(result["metadata"]["fixture.expert_used_count"], 2)
        self.assertEqual(result["moe_layer_count"], 1)
        self.assertEqual(result["moe_layers"], [0])
        self.assertEqual(result["moe_tensor_counts"]["ffn_gate_inp"], 1)
        self.assertEqual(result["moe_tensor_counts"]["ffn_down_exps"], 1)


if __name__ == "__main__":
    unittest.main()
