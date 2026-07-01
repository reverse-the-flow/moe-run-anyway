#!/usr/bin/env python3
"""Dry-run a GGUF MoE expert inventory scan.

This scanner reads only the GGUF metadata and tensor table needed to build the
offline expert inventory manifest used by Phase 3 replay work. It does not
load tensor values, write packed stores, download models, authenticate, run
Docker, launch runtimes, or send prompt traffic.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import plan_expert_inventory


ROOT = Path(__file__).resolve().parents[1]
GGUF_MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32
COMPONENT_ORDER = ("gate_proj", "up_proj", "down_proj")
UP_DOWN_COMPONENT_ORDER = ("up_proj", "down_proj")
EXTRA_COMPONENT_ORDER = ("down_proj_scale",)
COMPONENT_SORT_ORDER = {
    component: index for index, component in enumerate((*COMPONENT_ORDER, *EXTRA_COMPONENT_ORDER))
}
COMPONENT_ALIASES = {
    "ffn_gate": "gate_proj",
    "ffn_up": "up_proj",
    "ffn_down": "down_proj",
    "ffn_gate_exps": "gate_proj",
    "ffn_up_exps": "up_proj",
    "ffn_down_exps": "down_proj",
}
GGUF_VALUE_TYPES = {
    "UINT8": 0,
    "INT8": 1,
    "UINT16": 2,
    "INT16": 3,
    "UINT32": 4,
    "INT32": 5,
    "FLOAT32": 6,
    "BOOL": 7,
    "STRING": 8,
    "ARRAY": 9,
    "UINT64": 10,
    "INT64": 11,
    "FLOAT64": 12,
}
GGML_TYPE_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    31: "Q4_0_4_4",
    32: "Q4_0_4_8",
    33: "Q4_0_8_8",
}
PER_EXPERT_PATTERNS = (
    re.compile(
        r"^blk\.(?P<layer_id>\d+)\."
        r"(?P<component>ffn_gate|ffn_up|ffn_down)\."
        r"(?P<expert_id>\d+)\.weight$"
    ),
)
STACKED_EXPERT_PATTERNS = (
    re.compile(
        r"^blk\.(?P<layer_id>\d+)\."
        r"(?P<component>ffn_gate_exps|ffn_up_exps|ffn_down_exps)\.weight$"
    ),
)
FUSED_STACKED_EXPERT_PATTERNS = (
    re.compile(
        r"^blk\.(?P<layer_id>\d+)\."
        r"(?P<component>ffn_gate_up_exps)\.weight$"
    ),
)
FUSED_COMPONENT_ALIASES = {
    "ffn_gate_up_exps": ("gate_proj", "up_proj"),
}
STACKED_EXPERT_SCALE_PATTERNS = (
    re.compile(
        r"^blk\.(?P<layer_id>\d+)\."
        r"(?P<component>ffn_down_exps)\.scale$"
    ),
)
SCALE_COMPONENT_ALIASES = {
    "ffn_down_exps": "down_proj_scale",
}

JSONDict = dict[str, Any]


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dimensions: list[int]
    ggml_type: int
    offset: int
    byte_length: int


class GGUFReader:
    def __init__(self, handle: BinaryIO, path: Path) -> None:
        self.handle = handle
        self.path = path

    def read_exact(self, byte_count: int) -> bytes:
        data = self.handle.read(byte_count)
        if len(data) != byte_count:
            raise ValueError(f"{self.path} ended unexpectedly while reading {byte_count} bytes")
        return data

    def read_u8(self) -> int:
        return struct.unpack("<B", self.read_exact(1))[0]

    def read_i8(self) -> int:
        return struct.unpack("<b", self.read_exact(1))[0]

    def read_u16(self) -> int:
        return struct.unpack("<H", self.read_exact(2))[0]

    def read_i16(self) -> int:
        return struct.unpack("<h", self.read_exact(2))[0]

    def read_u32(self) -> int:
        return struct.unpack("<I", self.read_exact(4))[0]

    def read_i32(self) -> int:
        return struct.unpack("<i", self.read_exact(4))[0]

    def read_u64(self) -> int:
        return struct.unpack("<Q", self.read_exact(8))[0]

    def read_i64(self) -> int:
        return struct.unpack("<q", self.read_exact(8))[0]

    def read_f32(self) -> float:
        return struct.unpack("<f", self.read_exact(4))[0]

    def read_f64(self) -> float:
        return struct.unpack("<d", self.read_exact(8))[0]

    def read_string(self) -> str:
        length = self.read_u64()
        data = self.read_exact(length)
        return data.decode("utf-8")

    def tell(self) -> int:
        return self.handle.tell()


def align_offset(offset: int, alignment: int) -> int:
    if alignment <= 0:
        alignment = DEFAULT_ALIGNMENT
    remainder = offset % alignment
    return offset if remainder == 0 else offset + alignment - remainder


def read_metadata_value(reader: GGUFReader, value_type: int) -> Any:
    if value_type == GGUF_VALUE_TYPES["UINT8"]:
        return reader.read_u8()
    if value_type == GGUF_VALUE_TYPES["INT8"]:
        return reader.read_i8()
    if value_type == GGUF_VALUE_TYPES["UINT16"]:
        return reader.read_u16()
    if value_type == GGUF_VALUE_TYPES["INT16"]:
        return reader.read_i16()
    if value_type == GGUF_VALUE_TYPES["UINT32"]:
        return reader.read_u32()
    if value_type == GGUF_VALUE_TYPES["INT32"]:
        return reader.read_i32()
    if value_type == GGUF_VALUE_TYPES["FLOAT32"]:
        return reader.read_f32()
    if value_type == GGUF_VALUE_TYPES["BOOL"]:
        return bool(reader.read_u8())
    if value_type == GGUF_VALUE_TYPES["STRING"]:
        return reader.read_string()
    if value_type == GGUF_VALUE_TYPES["UINT64"]:
        return reader.read_u64()
    if value_type == GGUF_VALUE_TYPES["INT64"]:
        return reader.read_i64()
    if value_type == GGUF_VALUE_TYPES["FLOAT64"]:
        return reader.read_f64()
    if value_type == GGUF_VALUE_TYPES["ARRAY"]:
        array_type = reader.read_u32()
        array_length = reader.read_u64()
        return [read_metadata_value(reader, array_type) for _ in range(array_length)]
    raise ValueError(f"unsupported GGUF metadata value type {value_type}")


def read_gguf_tensor_table(path: Path) -> tuple[JSONDict, list[TensorInfo], int]:
    with path.open("rb") as handle:
        reader = GGUFReader(handle, path)
        magic = reader.read_exact(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"{path} is not a GGUF file")
        version = reader.read_u32()
        if version not in {2, 3}:
            raise ValueError(f"{path} has unsupported GGUF version {version}")
        tensor_count = reader.read_u64()
        metadata_count = reader.read_u64()

        metadata: JSONDict = {}
        for _ in range(metadata_count):
            key = reader.read_string()
            value_type = reader.read_u32()
            metadata[key] = read_metadata_value(reader, value_type)

        raw_tensor_infos: list[tuple[str, list[int], int, int]] = []
        for _ in range(tensor_count):
            name = reader.read_string()
            dimension_count = reader.read_u32()
            dimensions = [reader.read_u64() for _ in range(dimension_count)]
            ggml_type = reader.read_u32()
            offset = reader.read_u64()
            raw_tensor_infos.append((name, dimensions, ggml_type, offset))

        alignment = metadata.get("general.alignment", DEFAULT_ALIGNMENT)
        if not isinstance(alignment, int) or isinstance(alignment, bool):
            alignment = DEFAULT_ALIGNMENT
        data_start = align_offset(reader.tell(), alignment)

    file_size = path.stat().st_size
    sorted_infos = sorted(raw_tensor_infos, key=lambda item: item[3])
    tensors: list[TensorInfo] = []
    for index, (name, dimensions, ggml_type, offset) in enumerate(sorted_infos):
        next_offset = sorted_infos[index + 1][3] if index + 1 < len(sorted_infos) else file_size - data_start
        byte_length = next_offset - offset
        if byte_length <= 0:
            raise ValueError(f"{path}:{name} has invalid tensor byte range")
        tensors.append(
            TensorInfo(
                name=name,
                dimensions=dimensions,
                ggml_type=ggml_type,
                offset=data_start + offset,
                byte_length=byte_length,
            )
        )
    metadata["_gguf_version"] = version
    metadata["_tensor_count"] = tensor_count
    return metadata, tensors, data_start


def metadata_int(metadata: JSONDict, key: str) -> int | None:
    value = metadata.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def infer_expert_count(metadata: JSONDict) -> int | None:
    for key in ("llama.expert_count", "qwen3moe.expert_count", "mixtral.expert_count"):
        value = metadata_int(metadata, key)
        if value and value > 0:
            return value
    for key, value in metadata.items():
        if key.endswith(".expert_count") and isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def infer_routing_top_k(metadata: JSONDict, fallback: int) -> int:
    for key in ("llama.expert_used_count", "qwen3moe.expert_used_count", "mixtral.expert_used_count"):
        value = metadata_int(metadata, key)
        if value and value > 0:
            return value
    for key, value in metadata.items():
        if key.endswith(".expert_used_count") and isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return fallback


def parse_per_expert_tensor(name: str) -> tuple[int, int, str] | None:
    for pattern in PER_EXPERT_PATTERNS:
        match = pattern.match(name)
        if match is None:
            continue
        return (
            int(match.group("layer_id")),
            int(match.group("expert_id")),
            COMPONENT_ALIASES[match.group("component")],
        )
    return None


def parse_stacked_expert_tensor(name: str) -> tuple[int, str] | None:
    for pattern in STACKED_EXPERT_PATTERNS:
        match = pattern.match(name)
        if match is None:
            continue
        return int(match.group("layer_id")), COMPONENT_ALIASES[match.group("component")]
    return None


def parse_fused_stacked_expert_tensor(name: str) -> tuple[int, tuple[str, ...]] | None:
    for pattern in FUSED_STACKED_EXPERT_PATTERNS:
        match = pattern.match(name)
        if match is None:
            continue
        return int(match.group("layer_id")), FUSED_COMPONENT_ALIASES[match.group("component")]
    return None


def parse_stacked_expert_scale_tensor(name: str) -> tuple[int, str] | None:
    for pattern in STACKED_EXPERT_SCALE_PATTERNS:
        match = pattern.match(name)
        if match is None:
            continue
        return int(match.group("layer_id")), SCALE_COMPONENT_ALIASES[match.group("component")]
    return None


def component_sort_key(component_name: str) -> tuple[int, str]:
    return COMPONENT_SORT_ORDER.get(component_name, len(COMPONENT_SORT_ORDER)), component_name


def infer_component_profile(metadata: JSONDict, entries: list[JSONDict]) -> JSONDict:
    if not entries:
        return {
            "profile_id": "none",
            "required_components": [],
            "observed_components": [],
            "notes": ["no expert tensors were recognized"],
        }

    architecture = metadata.get("general.architecture")
    architecture_name = architecture if isinstance(architecture, str) else ""
    observed = sorted({str(entry["component_name"]) for entry in entries}, key=component_sort_key)
    observed_set = set(observed)
    extras = [component for component in observed if component not in COMPONENT_ORDER]

    if set(COMPONENT_ORDER).issubset(observed_set):
        profile_id = "gate_up_down"
        required = [*COMPONENT_ORDER, *extras]
        notes = ["scanner found gate/up/down routed expert components"]
    elif architecture_name == "nemotron_h_moe" and set(UP_DOWN_COMPONENT_ORDER).issubset(observed_set):
        profile_id = "nemotron_h_up_down"
        required = [*UP_DOWN_COMPONENT_ORDER, *extras]
        notes = [
            "nemotron_h_moe GGUF exposes routed expert up/down tensors without per-expert gate projection tensors",
            "router and shared expert tensors are intentionally left resident-side for this inventory",
        ]
    else:
        profile_id = "expected_gate_up_down_incomplete"
        required = list(COMPONENT_ORDER)
        notes = ["scanner did not find a complete supported routed expert component profile"]

    return {
        "profile_id": profile_id,
        "required_components": required,
        "observed_components": observed,
        "notes": notes,
    }


def tensor_dtype_name(ggml_type: int) -> str:
    return GGML_TYPE_NAMES.get(ggml_type, f"GGML_TYPE_{ggml_type}")


def build_inventory_manifest(
    model_path: Path,
    *,
    model_id: str | None,
    generated_for: str,
    backend_family: str,
    routing_top_k: int,
    experts_per_layer: int | None,
) -> tuple[JSONDict, JSONDict]:
    metadata, tensors, _ = read_gguf_tensor_table(model_path)
    inferred_experts = experts_per_layer or infer_expert_count(metadata)
    effective_top_k = infer_routing_top_k(metadata, routing_top_k)
    entries: list[JSONDict] = []
    ignored_tensor_count = 0
    stacked_tensor_count = 0
    fused_stacked_tensor_count = 0
    stacked_scale_tensor_count = 0
    per_expert_tensor_count = 0

    for tensor in tensors:
        parsed_per_expert = parse_per_expert_tensor(tensor.name)
        if parsed_per_expert is not None:
            layer_id, expert_id, component_name = parsed_per_expert
            per_expert_tensor_count += 1
            entries.append(
                {
                    "layer_id": layer_id,
                    "expert_id": expert_id,
                    "component_name": component_name,
                    "tensor_name": tensor.name,
                    "source_file": model_path.name,
                    "byte_offset": tensor.offset,
                    "byte_length": tensor.byte_length,
                    "stride_bytes": tensor.byte_length,
                    "dtype": tensor_dtype_name(tensor.ggml_type),
                    "shape": tensor.dimensions,
                    "estimated_residency_bytes": tensor.byte_length,
                    "coverage_status": "complete",
                    "source_layout": "per_expert_tensor",
                }
            )
            continue

        parsed_stacked = parse_stacked_expert_tensor(tensor.name)
        if parsed_stacked is not None:
            if inferred_experts is None:
                raise ValueError(
                    f"{model_path}:{tensor.name} is a stacked expert tensor, but expert_count is unavailable"
                )
            if tensor.byte_length % inferred_experts != 0:
                raise ValueError(
                    f"{model_path}:{tensor.name} byte length {tensor.byte_length} is not divisible "
                    f"by expert_count {inferred_experts}"
                )
            layer_id, component_name = parsed_stacked
            per_expert_bytes = tensor.byte_length // inferred_experts
            stacked_tensor_count += 1
            for expert_id in range(inferred_experts):
                entries.append(
                    {
                        "layer_id": layer_id,
                        "expert_id": expert_id,
                        "component_name": component_name,
                        "tensor_name": tensor.name,
                        "source_file": model_path.name,
                        "byte_offset": tensor.offset + (expert_id * per_expert_bytes),
                        "byte_length": per_expert_bytes,
                        "stride_bytes": per_expert_bytes,
                        "dtype": tensor_dtype_name(tensor.ggml_type),
                        "shape": tensor.dimensions,
                        "estimated_residency_bytes": per_expert_bytes,
                        "coverage_status": "complete",
                        "source_layout": "stacked_expert_tensor",
                        "source_tensor_name": tensor.name,
                    }
                )
            continue

        parsed_fused = parse_fused_stacked_expert_tensor(tensor.name)
        if parsed_fused is not None:
            if inferred_experts is None:
                raise ValueError(
                    f"{model_path}:{tensor.name} is a fused stacked expert tensor, but expert_count is unavailable"
                )
            layer_id, component_names = parsed_fused
            if tensor.byte_length % inferred_experts != 0:
                raise ValueError(
                    f"{model_path}:{tensor.name} byte length {tensor.byte_length} is not divisible "
                    f"by expert_count {inferred_experts}"
                )
            per_expert_bytes = tensor.byte_length // inferred_experts
            if per_expert_bytes % len(component_names) != 0:
                raise ValueError(
                    f"{model_path}:{tensor.name} per-expert byte length {per_expert_bytes} is not divisible "
                    f"by fused component count {len(component_names)}"
                )
            per_component_bytes = per_expert_bytes // len(component_names)
            fused_stacked_tensor_count += 1
            for expert_id in range(inferred_experts):
                expert_offset = tensor.offset + (expert_id * per_expert_bytes)
                for component_index, component_name in enumerate(component_names):
                    entries.append(
                        {
                            "layer_id": layer_id,
                            "expert_id": expert_id,
                            "component_name": component_name,
                            "tensor_name": tensor.name,
                            "source_file": model_path.name,
                            "byte_offset": expert_offset + (component_index * per_component_bytes),
                            "byte_length": per_component_bytes,
                            "stride_bytes": per_expert_bytes,
                            "dtype": tensor_dtype_name(tensor.ggml_type),
                            "shape": tensor.dimensions,
                            "estimated_residency_bytes": per_component_bytes,
                            "coverage_status": "complete",
                            "source_layout": "fused_stacked_expert_tensor",
                            "source_tensor_name": tensor.name,
                            "source_tensor_component_index": component_index,
                        }
                    )
            continue

        parsed_scale = parse_stacked_expert_scale_tensor(tensor.name)
        if parsed_scale is not None:
            if inferred_experts is None:
                raise ValueError(
                    f"{model_path}:{tensor.name} is a stacked expert scale tensor, but expert_count is unavailable"
                )
            if tensor.byte_length % inferred_experts != 0:
                raise ValueError(
                    f"{model_path}:{tensor.name} byte length {tensor.byte_length} is not divisible "
                    f"by expert_count {inferred_experts}"
                )
            layer_id, component_name = parsed_scale
            per_expert_bytes = tensor.byte_length // inferred_experts
            stacked_scale_tensor_count += 1
            for expert_id in range(inferred_experts):
                entries.append(
                    {
                        "layer_id": layer_id,
                        "expert_id": expert_id,
                        "component_name": component_name,
                        "tensor_name": tensor.name,
                        "source_file": model_path.name,
                        "byte_offset": tensor.offset + (expert_id * per_expert_bytes),
                        "byte_length": per_expert_bytes,
                        "stride_bytes": per_expert_bytes,
                        "dtype": tensor_dtype_name(tensor.ggml_type),
                        "shape": tensor.dimensions,
                        "estimated_residency_bytes": per_expert_bytes,
                        "coverage_status": "complete",
                        "source_layout": "stacked_expert_scale_tensor",
                        "source_tensor_name": tensor.name,
                    }
                )
            continue

        ignored_tensor_count += 1

    layer_ids = sorted({entry["layer_id"] for entry in entries})
    component_profile = infer_component_profile(metadata, entries)
    required_components = component_profile["required_components"]
    expert_keys = {(entry["layer_id"], entry["expert_id"]) for entry in entries}
    manifest = {
        "schema_version": plan_expert_inventory.SUPPORTED_SCHEMA_VERSION,
        "name": "GGUF Expert Inventory Dry Run",
        "generated_for": generated_for,
        "model_id": model_id or model_path.name,
        "source_format": "gguf",
        "backend_family": backend_family,
        "contract_version": "memory-moe-bridge-v1",
        "inventory_scope": "gguf_tensor_table_dry_run",
        "source_files": [
            {
                "path": model_path.name,
                "source_format": "gguf",
                "byte_length": model_path.stat().st_size,
                "gguf_version": metadata.get("_gguf_version"),
            }
        ],
        "expected": {
            "layer_ids": layer_ids,
            "required_components": required_components,
            "expert_count": len(expert_keys),
            "component_count": len(entries),
            "total_estimated_residency_bytes": sum(entry["estimated_residency_bytes"] for entry in entries),
            "routing_top_k": effective_top_k,
            "component_profile": component_profile,
        },
        "entries": sorted(
            entries,
            key=lambda entry: (entry["layer_id"], entry["expert_id"], component_sort_key(entry["component_name"])),
        ),
        "safety_contract": [
            "Scanner reads GGUF metadata and tensor table only.",
            "Scanner does not load tensor values.",
            "Scanner does not write packed expert stores.",
            "Scanner does not claim live expert paging.",
        ],
        "next_actions": [
            "Validate scanner output with scripts/plan_expert_inventory.py.",
            "Join validated GGUF inventory to llama.cpp semantic routing traces for Phase 3 replay metrics.",
            "Add a dry-run expert-store layout planner after trace+inventory replay basics are stable.",
        ],
    }
    scan_stats = {
        "model_path": str(model_path),
        "tensor_count": len(tensors),
        "recognized_expert_tensor_count": (
            per_expert_tensor_count
            + stacked_tensor_count
            + fused_stacked_tensor_count
            + stacked_scale_tensor_count
        ),
        "per_expert_tensor_count": per_expert_tensor_count,
        "stacked_expert_tensor_count": stacked_tensor_count,
        "fused_stacked_expert_tensor_count": fused_stacked_tensor_count,
        "stacked_expert_scale_tensor_count": stacked_scale_tensor_count,
        "ignored_tensor_count": ignored_tensor_count,
        "inferred_experts_per_layer": inferred_experts,
        "routing_top_k": effective_top_k,
        "component_profile": component_profile,
    }
    return manifest, scan_stats


def build_scan_summary(
    model_path: Path,
    *,
    model_id: str | None = None,
    generated_for: str = "memory-moe-mvp",
    backend_family: str = "llama_cpp",
    routing_top_k: int = 2,
    experts_per_layer: int | None = None,
) -> JSONDict:
    manifest, scan_stats = build_inventory_manifest(
        model_path,
        model_id=model_id,
        generated_for=generated_for,
        backend_family=backend_family,
        routing_top_k=routing_top_k,
        experts_per_layer=experts_per_layer,
    )
    inventory_summary = plan_expert_inventory.build_summary(manifest, model_path)
    return {
        "mode": "gguf_expert_inventory_scan",
        "model_path": str(model_path),
        "valid": inventory_summary["valid"],
        "errors": inventory_summary["errors"],
        "scan_stats": scan_stats,
        "inventory_summary": inventory_summary,
        "manifest": manifest,
        "safety_contract": [
            "dry run only",
            "no tensor values loaded",
            "no packed stores written",
            "no model runtime launched",
        ],
    }


def print_human_summary(summary: JSONDict) -> None:
    print("MoE Run Anyway GGUF expert inventory scan")
    print(f"Model path: {summary['model_path']}")
    print(f"Valid inventory: {summary['valid']}")
    if summary["errors"]:
        print("Errors:")
        for error in summary["errors"]:
            print(f"  - {error}")
    stats = summary["scan_stats"]
    print(f"Tensors: {stats['tensor_count']}")
    print(f"Recognized expert tensors: {stats['recognized_expert_tensor_count']}")
    print(f"Stacked expert tensors: {stats['stacked_expert_tensor_count']}")
    print(f"Fused stacked expert tensors: {stats['fused_stacked_expert_tensor_count']}")
    print(f"Stacked expert scale tensors: {stats['stacked_expert_scale_tensor_count']}")
    print(f"Per-expert tensors: {stats['per_expert_tensor_count']}")
    print(f"Ignored tensors: {stats['ignored_tensor_count']}")
    print(f"Component profile: {stats['component_profile']['profile_id']}")
    inventory = summary["inventory_summary"]
    print(f"Experts: {inventory['expert_count']}")
    print(f"Components: {inventory['component_count']}")
    print(f"Estimated residency bytes: {inventory['total_estimated_residency_bytes']}")
    print("Safety contract:")
    for item in summary["safety_contract"]:
        print(f"  - {item}")


def write_manifest(path: Path, manifest: JSONDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path, help="GGUF model file")
    parser.add_argument("--model-id", help="model id to write into the inventory manifest")
    parser.add_argument("--generated-for", default="memory-moe-mvp", help="inventory manifest generated_for value")
    parser.add_argument("--backend-family", default="llama_cpp", help="inventory manifest backend_family value")
    parser.add_argument("--routing-top-k", type=int, default=2, help="fallback MoE routing top-k")
    parser.add_argument("--experts-per-layer", type=int, help="fallback expert count for stacked GGUF expert tensors")
    parser.add_argument("--output", type=Path, help="optional path to write the generated inventory manifest JSON")
    parser.add_argument("--json", action="store_true", help="emit machine-readable scan summary")
    parser.add_argument("--manifest-json", action="store_true", help="emit only the generated inventory manifest JSON")
    return parser


def scan_path(
    model_path: Path,
    *,
    model_id: str | None = None,
    generated_for: str = "memory-moe-mvp",
    backend_family: str = "llama_cpp",
    routing_top_k: int = 2,
    experts_per_layer: int | None = None,
    output: Path | None = None,
) -> tuple[int, JSONDict | None, str | None]:
    try:
        summary = build_scan_summary(
            model_path,
            model_id=model_id,
            generated_for=generated_for,
            backend_family=backend_family,
            routing_top_k=routing_top_k,
            experts_per_layer=experts_per_layer,
        )
        if output is not None:
            write_manifest(output, summary["manifest"])
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, struct.error, ValueError) as exc:
        return 2, None, f"Could not scan GGUF expert inventory: {exc}"
    return (0 if summary["valid"] else 2), summary, None


def main_from_test_path(path: Path) -> int:
    status, _, _ = scan_path(path)
    return status


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary, error_message = scan_path(
        args.model_path,
        model_id=args.model_id,
        generated_for=args.generated_for,
        backend_family=args.backend_family,
        routing_top_k=args.routing_top_k,
        experts_per_layer=args.experts_per_layer,
        output=args.output,
    )
    if error_message:
        print(error_message, file=sys.stderr)
        return status

    assert summary is not None
    if args.manifest_json:
        print(json.dumps(summary["manifest"], indent=2, sort_keys=False))
    elif args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_human_summary(summary)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
