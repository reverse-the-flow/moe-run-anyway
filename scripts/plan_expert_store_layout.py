#!/usr/bin/env python3
"""Plan a dry-run packed expert-store layout from an inventory manifest.

This Phase 3 planner turns inventory rows into a proposed packed-file layout
without reading tensor values or writing the store. It exists to expose disk
requirements, source byte ranges, target offsets, and missing components before
any streaming packer write path exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import plan_expert_inventory


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = ROOT / "memory-moe-mvp" / "data" / "expert_inventory_manifest.fixture.json"
DEFAULT_STORE_NAME = "expert-store"
DEFAULT_METADATA_BASE_BYTES = 4096
ESTIMATED_METADATA_BYTES_PER_EXPERT = 128
ESTIMATED_METADATA_BYTES_PER_COMPONENT = 256

JSONDict = dict[str, Any]
ExpertKey = tuple[int, int]


def component_order(manifest: JSONDict) -> dict[str, int]:
    expected = manifest.get("expected")
    if not isinstance(expected, dict):
        return {}
    required = expected.get("required_components")
    if not isinstance(required, list):
        return {}
    return {
        str(component): index
        for index, component in enumerate(required)
        if isinstance(component, str)
    }


def expected_component_names(manifest: JSONDict) -> list[str]:
    expected = manifest.get("expected")
    if not isinstance(expected, dict):
        return []
    required = expected.get("required_components")
    if not isinstance(required, list):
        return []
    return [str(component) for component in required if isinstance(component, str)]


def entry_sort_key(entry: JSONDict, order: dict[str, int]) -> tuple[int, int, int, str]:
    component = str(entry.get("component_name", ""))
    return (
        int(entry.get("layer_id", 0)),
        int(entry.get("expert_id", 0)),
        order.get(component, len(order)),
        component,
    )


def group_entries(manifest: JSONDict) -> dict[ExpertKey, list[JSONDict]]:
    grouped: dict[ExpertKey, list[JSONDict]] = defaultdict(list)
    for entry in manifest.get("entries", []):
        if not isinstance(entry, dict):
            continue
        layer_id = entry.get("layer_id")
        expert_id = entry.get("expert_id")
        if isinstance(layer_id, int) and not isinstance(layer_id, bool) and isinstance(expert_id, int) and not isinstance(expert_id, bool):
            grouped[(layer_id, expert_id)].append(entry)
    return dict(grouped)


def build_source_read_plan(layout_components: list[JSONDict]) -> list[JSONDict]:
    by_source: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for component in layout_components:
        by_source[str(component["source_file"])].append(
            (
                int(component["source_byte_offset"]),
                int(component["source_byte_offset"]) + int(component["byte_length"]),
            )
        )

    plan = []
    for source_file, ranges in sorted(by_source.items()):
        sorted_ranges = sorted(ranges)
        plan.append(
            {
                "source_file": source_file,
                "range_count": len(sorted_ranges),
                "bytes_to_read": sum(end - start for start, end in sorted_ranges),
                "first_byte_offset": sorted_ranges[0][0],
                "last_byte_end": max(end for _, end in sorted_ranges),
            }
        )
    return plan


def build_expert_layouts(manifest: JSONDict, store_name: str) -> tuple[list[JSONDict], list[JSONDict], list[JSONDict]]:
    order = component_order(manifest)
    required_components = set(expected_component_names(manifest))
    grouped = group_entries(manifest)
    layouts: list[JSONDict] = []
    all_components: list[JSONDict] = []
    missing_components: list[JSONDict] = []

    for layer_id, expert_id in sorted(grouped):
        entries = sorted(grouped[(layer_id, expert_id)], key=lambda entry: entry_sort_key(entry, order))
        seen_components = {
            str(entry.get("component_name"))
            for entry in entries
            if isinstance(entry.get("component_name"), str)
        }
        missing = sorted(required_components - seen_components)
        if missing:
            missing_components.append(
                {
                    "layer_id": layer_id,
                    "expert_id": expert_id,
                    "missing_components": missing,
                }
            )

        target_offset = 0
        components: list[JSONDict] = []
        for entry in entries:
            byte_length = int(entry.get("byte_length", 0))
            component = {
                "component_name": str(entry.get("component_name")),
                "tensor_name": str(entry.get("tensor_name")),
                "source_file": str(entry.get("source_file")),
                "source_byte_offset": int(entry.get("byte_offset", 0)),
                "byte_length": byte_length,
                "target_byte_offset": target_offset,
                "target_byte_end": target_offset + byte_length,
                "dtype": str(entry.get("dtype")),
                "shape": entry.get("shape", []),
                "coverage_status": str(entry.get("coverage_status")),
            }
            components.append(component)
            all_components.append(component)
            target_offset += byte_length

        layouts.append(
            {
                "layer_id": layer_id,
                "expert_id": expert_id,
                "target_file": f"{store_name}/layers/layer-{layer_id}/expert-{expert_id}.bin",
                "target_index_file": f"{store_name}/layers/layer-{layer_id}/expert-{expert_id}.json",
                "packed_bytes": target_offset,
                "component_count": len(components),
                "complete": not missing,
                "components": components,
            }
        )
    return layouts, all_components, missing_components


def build_layout_summary(manifest_path: Path, *, store_name: str = DEFAULT_STORE_NAME) -> JSONDict:
    manifest = plan_expert_inventory.load_manifest(manifest_path)
    validation_errors = plan_expert_inventory.validate_manifest(manifest)
    expert_layouts, layout_components, missing_components = build_expert_layouts(manifest, store_name)
    total_packed_bytes = sum(int(expert["packed_bytes"]) for expert in expert_layouts)
    estimated_metadata_bytes = (
        DEFAULT_METADATA_BASE_BYTES
        + len(expert_layouts) * ESTIMATED_METADATA_BYTES_PER_EXPERT
        + len(layout_components) * ESTIMATED_METADATA_BYTES_PER_COMPONENT
    )
    coverage_counts: dict[str, int] = defaultdict(int)
    for component in layout_components:
        coverage_counts[str(component["coverage_status"])] += 1

    errors = list(validation_errors)
    if missing_components:
        errors.append(f"{len(missing_components)} expert layouts are missing required components")

    return {
        "mode": "expert_store_layout_plan",
        "manifest_path": str(manifest_path),
        "store_name": store_name,
        "valid": not errors,
        "errors": errors,
        "schema_version": manifest.get("schema_version"),
        "model_id": manifest.get("model_id"),
        "source_format": manifest.get("source_format"),
        "backend_family": manifest.get("backend_family"),
        "expert_count": len(expert_layouts),
        "component_count": len(layout_components),
        "complete_expert_count": sum(1 for expert in expert_layouts if expert["complete"]),
        "missing_components": missing_components,
        "total_packed_bytes": total_packed_bytes,
        "estimated_metadata_bytes": estimated_metadata_bytes,
        "required_disk_bytes": total_packed_bytes + estimated_metadata_bytes,
        "source_read_plan": build_source_read_plan(layout_components),
        "coverage_counts": dict(sorted(coverage_counts.items())),
        "expert_layouts": expert_layouts,
        "can_stream_without_full_model_load": not errors and bool(expert_layouts),
        "write_path_implemented": False,
        "safety_contract": [
            "planner validates local inventory only",
            "planner does not read tensor values",
            "planner does not write packed expert stores",
            "planner does not mutate model files or caches",
            "planner does not claim live expert paging",
        ],
        "next_actions": [
            "Run this planner against scanner-derived real-model inventory.",
            "Add a streaming write path only after source byte ranges and disk budget validate.",
            "Validate sampled packed byte ranges against the source checkpoint after a write path exists.",
        ],
    }


def print_human_summary(summary: JSONDict) -> None:
    print("MoE Run Anyway expert store layout plan")
    print(f"Manifest: {summary['manifest_path']}")
    print(f"Store name: {summary['store_name']}")
    print(f"Valid: {summary['valid']}")
    if summary["errors"]:
        print("Errors:")
        for error in summary["errors"]:
            print(f"  - {error}")
    print(f"Experts: {summary['expert_count']}")
    print(f"Components: {summary['component_count']}")
    print(f"Complete experts: {summary['complete_expert_count']}")
    print(f"Packed bytes: {summary['total_packed_bytes']}")
    print(f"Estimated metadata bytes: {summary['estimated_metadata_bytes']}")
    print(f"Required disk bytes: {summary['required_disk_bytes']}")
    print(f"Can stream without full model load: {summary['can_stream_without_full_model_load']}")
    print(f"Write path implemented: {summary['write_path_implemented']}")
    print("Safety contract:")
    for item in summary["safety_contract"]:
        print(f"  - {item}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_path", nargs="?", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--store-name", default=DEFAULT_STORE_NAME)
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    return parser


def plan_path(manifest_path: Path, *, store_name: str = DEFAULT_STORE_NAME) -> tuple[int, JSONDict | None, str | None]:
    if not store_name.strip():
        return 2, None, "store name must be a non-empty string"
    try:
        summary = build_layout_summary(manifest_path, store_name=store_name.strip().strip("/\\"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not build expert store layout plan: {exc}"
    return (0 if summary["valid"] else 2), summary, None


def main_from_test_path(manifest_path: Path) -> int:
    status, _, _ = plan_path(manifest_path)
    return status


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary, error_message = plan_path(args.manifest_path, store_name=args.store_name)
    if error_message:
        print(error_message, file=sys.stderr)
        return status
    assert summary is not None
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_human_summary(summary)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
