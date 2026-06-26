#!/usr/bin/env python3
"""Read GGUF MoE metadata without loading tensor data."""

from __future__ import annotations

import argparse
import json
import re
import struct
from collections import Counter
from pathlib import Path
from typing import BinaryIO, Any


GGUF_MAGIC = b"GGUF"

VALUE_TYPES = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}

FIXED_FORMATS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}

FIXED_SIZES = {value_type: struct.calcsize(fmt) for value_type, fmt in FIXED_FORMATS.items()}

MOE_PATTERNS = (
    "ffn_gate_inp",
    "ffn_down_exps",
    "ffn_gate_exps",
    "ffn_up_exps",
    "ffn_down_shexp",
    "ffn_gate_shexp",
    "ffn_up_shexp",
    "exp_probs_b",
)

LAYER_RE = re.compile(r"^blk\.(\d+)\.")


class GGUFReadError(RuntimeError):
    """Raised when a GGUF file cannot be read."""


def read_exact(handle: BinaryIO, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise GGUFReadError(f"unexpected EOF while reading {size} bytes")
    return data


def read_u32(handle: BinaryIO) -> int:
    return struct.unpack("<I", read_exact(handle, 4))[0]


def read_u64(handle: BinaryIO) -> int:
    return struct.unpack("<Q", read_exact(handle, 8))[0]


def read_string(handle: BinaryIO) -> str:
    length = read_u64(handle)
    data = read_exact(handle, length)
    return data.decode("utf-8", errors="replace")


def skip_string(handle: BinaryIO) -> None:
    length = read_u64(handle)
    handle.seek(length, 1)


def read_value(handle: BinaryIO, value_type: int, *, keep_arrays: bool = False) -> Any:
    if value_type in FIXED_FORMATS:
        return struct.unpack(FIXED_FORMATS[value_type], read_exact(handle, FIXED_SIZES[value_type]))[0]
    if value_type == 8:
        return read_string(handle)
    if value_type == 9:
        item_type = read_u32(handle)
        length = read_u64(handle)
        type_name = VALUE_TYPES.get(item_type, f"unknown:{item_type}")
        if keep_arrays and item_type in FIXED_FORMATS and length <= 64:
            return {
                "array_type": type_name,
                "length": length,
                "values": [
                    struct.unpack(FIXED_FORMATS[item_type], read_exact(handle, FIXED_SIZES[item_type]))[0]
                    for _ in range(length)
                ],
            }
        skip_array(handle, item_type, length)
        return {"array_type": type_name, "length": length}
    raise GGUFReadError(f"unsupported GGUF value type {value_type}")


def skip_array(handle: BinaryIO, item_type: int, length: int) -> None:
    if item_type in FIXED_SIZES:
        handle.seek(FIXED_SIZES[item_type] * length, 1)
        return
    if item_type == 8:
        for _ in range(length):
            skip_string(handle)
        return
    if item_type == 9:
        for _ in range(length):
            nested_type = read_u32(handle)
            nested_length = read_u64(handle)
            skip_array(handle, nested_type, nested_length)
        return
    raise GGUFReadError(f"unsupported GGUF array value type {item_type}")


def read_metadata(handle: BinaryIO, kv_count: int) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for _ in range(kv_count):
        key = read_string(handle)
        value_type = read_u32(handle)
        metadata[key] = read_value(handle, value_type)
    return metadata


def read_tensor_summary(handle: BinaryIO, tensor_count: int) -> dict[str, Any]:
    by_prefix: Counter[str] = Counter()
    moe_tensor_counts: Counter[str] = Counter()
    moe_layers: set[int] = set()
    examples: dict[str, str] = {}

    for _ in range(tensor_count):
        name = read_string(handle)
        n_dimensions = read_u32(handle)
        dimensions = [read_u64(handle) for _ in range(n_dimensions)]
        tensor_type = read_u32(handle)
        offset = read_u64(handle)

        prefix = name.split(".", 1)[0]
        by_prefix[prefix] += 1

        layer_match = LAYER_RE.match(name)
        layer_id = int(layer_match.group(1)) if layer_match else None
        for pattern in MOE_PATTERNS:
            if pattern in name:
                moe_tensor_counts[pattern] += 1
                examples.setdefault(pattern, name)
                if layer_id is not None:
                    moe_layers.add(layer_id)

        # Keep variables referenced so malformed tensor records fail during read.
        _ = dimensions, tensor_type, offset

    return {
        "tensor_prefix_counts": dict(sorted(by_prefix.items())),
        "moe_tensor_counts": dict(sorted(moe_tensor_counts.items())),
        "moe_tensor_examples": dict(sorted(examples.items())),
        "moe_layer_count": len(moe_layers),
        "moe_layers": sorted(moe_layers),
    }


def summarize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    interesting_suffixes = (
        "architecture",
        "block_count",
        "leading_dense_block_count",
        "expert_count",
        "expert_used_count",
        "expert_shared_count",
        "interleave_moe_layer_step",
        "context_length",
        "embedding_length",
        "file_type",
        "quantization_version",
    )
    return {
        key: value
        for key, value in sorted(metadata.items())
        if key.endswith(interesting_suffixes) or key in {"general.name", "general.basename", "general.size_label"}
    }


def preflight(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        magic = read_exact(handle, 4)
        if magic != GGUF_MAGIC:
            raise GGUFReadError(f"{path} is not a GGUF file")
        version = read_u32(handle)
        tensor_count = read_u64(handle)
        kv_count = read_u64(handle)
        metadata = read_metadata(handle, kv_count)
        tensor_summary = read_tensor_summary(handle, tensor_count)

    return {
        "path": str(path),
        "version": version,
        "metadata_kv_count": kv_count,
        "tensor_count": tensor_count,
        "metadata": summarize_metadata(metadata),
        **tensor_summary,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf_path", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    result = preflight(args.gguf_path)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"GGUF: {result['path']}")
        print(f"version: {result['version']}")
        print(f"metadata_kv_count: {result['metadata_kv_count']}")
        print(f"tensor_count: {result['tensor_count']}")
        print(f"moe_layer_count: {result['moe_layer_count']}")
        print("metadata:")
        for key, value in result["metadata"].items():
            print(f"  {key}: {value}")
        print("moe_tensor_counts:")
        for key, value in result["moe_tensor_counts"].items():
            print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
