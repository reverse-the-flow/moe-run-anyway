#!/usr/bin/env python3
"""Dry-run a safetensors MoE expert inventory scan.

This scanner reads safetensors headers and optional Hugging Face shard indexes
to emit the offline expert inventory manifest used by Phase 3 replay work. It
does not load tensor values, write packed stores, download models, authenticate,
run Docker, launch runtimes, or send prompt traffic.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import plan_expert_inventory


ROOT = Path(__file__).resolve().parents[1]
INDEX_FILENAMES = ("model.safetensors.index.json", "pytorch_model.safetensors.index.json")
COMPONENT_ALIASES = {
    "w1": "gate_proj",
    "w2": "down_proj",
    "w3": "up_proj",
    "gate_proj": "gate_proj",
    "up_proj": "up_proj",
    "down_proj": "down_proj",
}
COMPONENT_ORDER = ("gate_proj", "up_proj", "down_proj")
EXPERT_TENSOR_PATTERNS = (
    re.compile(
        r"(?:^|\.)layers\.(?P<layer_id>\d+)\..*?"
        r"\.experts\.(?P<expert_id>\d+)\."
        r"(?P<component>w[123]|gate_proj|up_proj|down_proj)\.weight$"
    ),
)
JSONDict = dict[str, Any]


def read_json_object(path: Path) -> JSONDict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def read_safetensors_header(path: Path) -> tuple[JSONDict, int]:
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"{path} is too small to contain a safetensors header length")
        header_length = int.from_bytes(length_bytes, byteorder="little", signed=False)
        if header_length <= 0:
            raise ValueError(f"{path} has an empty safetensors header")
        header_bytes = handle.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError(f"{path} ended before the declared safetensors header")

    header = json.loads(header_bytes.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError(f"{path} safetensors header must be a JSON object")
    return header, 8 + header_length


def discover_safetensors_files(model_path: Path) -> tuple[Path, list[Path], dict[str, str]]:
    if model_path.is_file():
        if model_path.suffix != ".safetensors":
            raise ValueError(f"{model_path} is not a .safetensors file")
        return model_path.parent, [model_path], {}

    if not model_path.is_dir():
        raise ValueError(f"{model_path} is not a file or directory")

    for index_name in INDEX_FILENAMES:
        index_path = model_path / index_name
        if not index_path.exists():
            continue
        index = read_json_object(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"{index_path} weight_map must be a non-empty object")
        files = sorted({model_path / str(path) for path in weight_map.values()})
        return model_path, files, {str(name): str(path) for name, path in weight_map.items()}

    files = sorted(model_path.glob("*.safetensors"))
    if not files:
        raise ValueError(f"{model_path} does not contain .safetensors files")
    return model_path, files, {}


def source_label(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def parse_expert_tensor_name(tensor_name: str) -> tuple[int, int, str] | None:
    for pattern in EXPERT_TENSOR_PATTERNS:
        match = pattern.search(tensor_name)
        if not match:
            continue
        component = COMPONENT_ALIASES[match.group("component")]
        return int(match.group("layer_id")), int(match.group("expert_id")), component
    return None


def tensor_offsets(tensor_meta: Any, *, tensor_name: str, source_file: Path) -> tuple[int, int]:
    if not isinstance(tensor_meta, dict):
        raise ValueError(f"{source_file}:{tensor_name} metadata must be an object")
    offsets = tensor_meta.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError(f"{source_file}:{tensor_name} data_offsets must contain two integers")
    start, end = offsets
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or start < 0
        or end <= start
    ):
        raise ValueError(f"{source_file}:{tensor_name} data_offsets must be increasing non-negative integers")
    return start, end


def tensor_shape(tensor_meta: Any, *, tensor_name: str, source_file: Path) -> list[int]:
    if not isinstance(tensor_meta, dict):
        raise ValueError(f"{source_file}:{tensor_name} metadata must be an object")
    shape = tensor_meta.get("shape")
    if not isinstance(shape, list) or not shape:
        raise ValueError(f"{source_file}:{tensor_name} shape must be a non-empty list")
    normalized: list[int] = []
    for index, dimension in enumerate(shape):
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
            raise ValueError(f"{source_file}:{tensor_name} shape[{index}] must be an integer > 0")
        normalized.append(dimension)
    return normalized


def tensor_dtype(tensor_meta: Any, *, tensor_name: str, source_file: Path) -> str:
    if not isinstance(tensor_meta, dict):
        raise ValueError(f"{source_file}:{tensor_name} metadata must be an object")
    dtype = tensor_meta.get("dtype")
    if not isinstance(dtype, str) or not dtype.strip():
        raise ValueError(f"{source_file}:{tensor_name} dtype must be a non-empty string")
    return dtype


def build_inventory_manifest(
    model_path: Path,
    *,
    model_id: str | None,
    generated_for: str,
    backend_family: str,
    routing_top_k: int,
) -> tuple[JSONDict, JSONDict]:
    root, files, weight_map = discover_safetensors_files(model_path)
    source_files: list[JSONDict] = []
    entries: list[JSONDict] = []
    ignored_tensor_count = 0

    for file_path in files:
        if not file_path.exists():
            raise ValueError(f"indexed safetensors shard does not exist: {file_path}")
        header, data_start = read_safetensors_header(file_path)
        label = source_label(file_path, root)
        source_files.append(
            {
                "path": label,
                "source_format": "safetensors",
                "byte_length": file_path.stat().st_size,
            }
        )
        for tensor_name, tensor_meta in sorted(header.items()):
            if tensor_name == "__metadata__":
                continue
            parsed = parse_expert_tensor_name(tensor_name)
            if parsed is None:
                ignored_tensor_count += 1
                continue
            layer_id, expert_id, component_name = parsed
            start, end = tensor_offsets(tensor_meta, tensor_name=tensor_name, source_file=file_path)
            byte_length = end - start
            entries.append(
                {
                    "layer_id": layer_id,
                    "expert_id": expert_id,
                    "component_name": component_name,
                    "tensor_name": tensor_name,
                    "source_file": label,
                    "byte_offset": data_start + start,
                    "byte_length": byte_length,
                    "stride_bytes": byte_length,
                    "dtype": tensor_dtype(tensor_meta, tensor_name=tensor_name, source_file=file_path),
                    "shape": tensor_shape(tensor_meta, tensor_name=tensor_name, source_file=file_path),
                    "estimated_residency_bytes": byte_length,
                    "coverage_status": "complete",
                }
            )

    layer_ids = sorted({entry["layer_id"] for entry in entries})
    required_components = [component for component in COMPONENT_ORDER if any(entry["component_name"] == component for entry in entries)]
    expert_keys = {(entry["layer_id"], entry["expert_id"]) for entry in entries}
    manifest = {
        "schema_version": plan_expert_inventory.SUPPORTED_SCHEMA_VERSION,
        "name": "Safetensors Expert Inventory Dry Run",
        "generated_for": generated_for,
        "model_id": model_id or model_path.name,
        "source_format": "safetensors",
        "backend_family": backend_family,
        "contract_version": "memory-moe-bridge-v1",
        "inventory_scope": "safetensors_header_dry_run",
        "source_files": source_files,
        "expected": {
            "layer_ids": layer_ids,
            "required_components": required_components,
            "expert_count": len(expert_keys),
            "component_count": len(entries),
            "total_estimated_residency_bytes": sum(entry["estimated_residency_bytes"] for entry in entries),
            "routing_top_k": routing_top_k,
        },
        "entries": sorted(entries, key=lambda entry: (entry["layer_id"], entry["expert_id"], entry["component_name"])),
        "safety_contract": [
            "Scanner reads safetensors headers only.",
            "Scanner does not load tensor values.",
            "Scanner does not write packed expert stores.",
            "Scanner does not claim live expert paging.",
        ],
        "next_actions": [
            "Validate scanner output with scripts/plan_expert_inventory.py.",
            "Join validated inventory to semantic routing traces for Phase 3 replay metrics.",
            "Use scripts/scan_gguf_expert_inventory.py for llama.cpp trace targets.",
        ],
    }
    scan_stats = {
        "model_root": str(root),
        "indexed_tensor_count": len(weight_map),
        "source_file_count": len(source_files),
        "recognized_expert_tensor_count": len(entries),
        "ignored_tensor_count": ignored_tensor_count,
    }
    return manifest, scan_stats


def build_scan_summary(
    model_path: Path,
    *,
    model_id: str | None = None,
    generated_for: str = "memory-moe-mvp",
    backend_family: str = "hookable_pytorch",
    routing_top_k: int = 2,
) -> JSONDict:
    manifest, scan_stats = build_inventory_manifest(
        model_path,
        model_id=model_id,
        generated_for=generated_for,
        backend_family=backend_family,
        routing_top_k=routing_top_k,
    )
    inventory_summary = plan_expert_inventory.build_summary(manifest, model_path)
    return {
        "mode": "safetensors_expert_inventory_scan",
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
    print("MoE Run Anyway safetensors expert inventory scan")
    print(f"Model path: {summary['model_path']}")
    print(f"Valid inventory: {summary['valid']}")
    if summary["errors"]:
        print("Errors:")
        for error in summary["errors"]:
            print(f"  - {error}")
    stats = summary["scan_stats"]
    print(f"Source files: {stats['source_file_count']}")
    print(f"Indexed tensors: {stats['indexed_tensor_count']}")
    print(f"Recognized expert tensors: {stats['recognized_expert_tensor_count']}")
    print(f"Ignored tensors: {stats['ignored_tensor_count']}")
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
    parser.add_argument("model_path", type=Path, help="safetensors file or model directory")
    parser.add_argument("--model-id", help="model id to write into the inventory manifest")
    parser.add_argument("--generated-for", default="memory-moe-mvp", help="inventory manifest generated_for value")
    parser.add_argument("--backend-family", default="hookable_pytorch", help="inventory manifest backend_family value")
    parser.add_argument("--routing-top-k", type=int, default=2, help="expected MoE routing top-k")
    parser.add_argument("--output", type=Path, help="optional path to write the generated inventory manifest JSON")
    parser.add_argument("--json", action="store_true", help="emit machine-readable scan summary")
    parser.add_argument("--manifest-json", action="store_true", help="emit only the generated inventory manifest JSON")
    return parser


def scan_path(
    model_path: Path,
    *,
    model_id: str | None = None,
    generated_for: str = "memory-moe-mvp",
    backend_family: str = "hookable_pytorch",
    routing_top_k: int = 2,
    output: Path | None = None,
) -> tuple[int, JSONDict | None, str | None]:
    try:
        summary = build_scan_summary(
            model_path,
            model_id=model_id,
            generated_for=generated_for,
            backend_family=backend_family,
            routing_top_k=routing_top_k,
        )
        if output is not None:
            write_manifest(output, summary["manifest"])
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not scan safetensors expert inventory: {exc}"
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
