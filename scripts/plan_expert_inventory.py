#!/usr/bin/env python3
"""Validate and summarize an offline MoE expert inventory manifest.

The inventory manifest is the Phase 3 bridge between semantic routing traces
and later replay/paging policy work. This planner is deliberately offline: it
does not load tensor values, scan model files, download models, authenticate,
run Docker, launch runtimes, or send prompt traffic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = ROOT / "memory-moe-mvp" / "data" / "expert_inventory_manifest.fixture.json"
SUPPORTED_SCHEMA_VERSION = "moe-expert-inventory-manifest-v1"
SUPPORTED_SOURCE_FORMATS = {"gguf", "safetensors", "fixture"}
SUPPORTED_BACKEND_FAMILIES = {
    "llama_cpp",
    "hookable_pytorch",
    "moe_infinity_style",
    "prototype_offload_system",
    "vllm_openai_compatible",
}
SUPPORTED_COVERAGE_STATUSES = {"complete", "partial", "missing"}

REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "name",
    "generated_for",
    "model_id",
    "source_format",
    "backend_family",
    "contract_version",
    "inventory_scope",
    "source_files",
    "expected",
    "entries",
    "safety_contract",
    "next_actions",
}
REQUIRED_EXPECTED_FIELDS = {
    "layer_ids",
    "required_components",
    "expert_count",
    "component_count",
    "total_estimated_residency_bytes",
    "routing_top_k",
}
REQUIRED_ENTRY_FIELDS = {
    "layer_id",
    "expert_id",
    "component_name",
    "tensor_name",
    "source_file",
    "byte_offset",
    "byte_length",
    "stride_bytes",
    "dtype",
    "shape",
    "estimated_residency_bytes",
    "coverage_status",
}

JSONDict = dict[str, Any]


def load_manifest(path: Path) -> JSONDict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("manifest root must be a JSON object")
    return data


def require_string(obj: JSONDict, field: str, errors: list[str], *, context: str) -> None:
    value = obj.get(field)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{context}.{field} must be a non-empty string")


def require_int(obj: JSONDict, field: str, errors: list[str], *, context: str, minimum: int = 0) -> None:
    value = obj.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        errors.append(f"{context}.{field} must be an integer >= {minimum}")


def require_string_list(obj: JSONDict, field: str, errors: list[str], *, context: str) -> list[str]:
    value = obj.get(field)
    if not isinstance(value, list) or not value:
        errors.append(f"{context}.{field} must be a non-empty list")
        return []

    strings: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            errors.append(f"{context}.{field}[{index}] must be a non-empty string")
        else:
            strings.append(item)
    return strings


def require_int_list(obj: JSONDict, field: str, errors: list[str], *, context: str) -> list[int]:
    value = obj.get(field)
    if not isinstance(value, list) or not value:
        errors.append(f"{context}.{field} must be a non-empty list")
        return []

    ints: list[int] = []
    for index, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            errors.append(f"{context}.{field}[{index}] must be an integer >= 0")
        else:
            ints.append(item)
    return ints


def validate_source_files(manifest: JSONDict, errors: list[str]) -> dict[str, int | None]:
    source_files = manifest.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        errors.append("source_files must be a non-empty list")
        return {}

    by_path: dict[str, int | None] = {}
    for index, source_file in enumerate(source_files):
        context = f"source_files[{index}]"
        if not isinstance(source_file, dict):
            errors.append(f"{context} must be an object")
            continue
        require_string(source_file, "path", errors, context=context)
        require_string(source_file, "source_format", errors, context=context)
        if source_file.get("source_format") not in SUPPORTED_SOURCE_FORMATS:
            errors.append(
                f"{context}.source_format has unsupported value {source_file.get('source_format')!r}"
            )
        byte_length = source_file.get("byte_length")
        if byte_length is not None and (
            not isinstance(byte_length, int) or isinstance(byte_length, bool) or byte_length <= 0
        ):
            errors.append(f"{context}.byte_length must be an integer > 0 when present")
            byte_length = None
        path = source_file.get("path")
        if isinstance(path, str):
            if path in by_path:
                errors.append(f"{context}.path duplicates {path!r}")
            by_path[path] = byte_length if isinstance(byte_length, int) else None
    return by_path


def validate_expected(manifest: JSONDict, errors: list[str]) -> tuple[set[int], list[str]]:
    expected = manifest.get("expected")
    if not isinstance(expected, dict):
        errors.append("expected must be an object")
        return set(), []

    missing = sorted(REQUIRED_EXPECTED_FIELDS - set(expected))
    if missing:
        errors.append(f"expected missing required fields: {', '.join(missing)}")

    layer_ids = require_int_list(expected, "layer_ids", errors, context="expected")
    required_components = require_string_list(expected, "required_components", errors, context="expected")
    for field in ("expert_count", "component_count", "total_estimated_residency_bytes", "routing_top_k"):
        require_int(expected, field, errors, context="expected", minimum=1)
    if len(set(layer_ids)) != len(layer_ids):
        errors.append("expected.layer_ids must not contain duplicates")
    if len(set(required_components)) != len(required_components):
        errors.append("expected.required_components must not contain duplicates")
    return set(layer_ids), required_components


def validate_entry_shape(
    entry: JSONDict,
    *,
    context: str,
    source_files: dict[str, int | None],
    expected_layers: set[int],
    required_components: list[str],
    errors: list[str],
) -> tuple[int, int, str] | None:
    missing = sorted(REQUIRED_ENTRY_FIELDS - set(entry))
    if missing:
        errors.append(f"{context} missing required fields: {', '.join(missing)}")

    require_int(entry, "layer_id", errors, context=context)
    require_int(entry, "expert_id", errors, context=context)
    require_string(entry, "component_name", errors, context=context)
    require_string(entry, "tensor_name", errors, context=context)
    require_string(entry, "source_file", errors, context=context)
    require_int(entry, "byte_offset", errors, context=context)
    require_int(entry, "byte_length", errors, context=context, minimum=1)
    require_int(entry, "stride_bytes", errors, context=context, minimum=1)
    require_string(entry, "dtype", errors, context=context)
    require_int(entry, "estimated_residency_bytes", errors, context=context, minimum=1)
    require_string(entry, "coverage_status", errors, context=context)

    layer_id = entry.get("layer_id")
    expert_id = entry.get("expert_id")
    component_name = entry.get("component_name")
    if not isinstance(layer_id, int) or isinstance(layer_id, bool):
        return None
    if not isinstance(expert_id, int) or isinstance(expert_id, bool):
        return None
    if not isinstance(component_name, str):
        return None

    if expected_layers and layer_id not in expected_layers:
        errors.append(f"{context}.layer_id {layer_id!r} is not listed in expected.layer_ids")
    if required_components and component_name not in required_components:
        errors.append(f"{context}.component_name has unsupported value {component_name!r}")

    source_file = entry.get("source_file")
    if isinstance(source_file, str) and source_file not in source_files:
        errors.append(f"{context}.source_file references unknown source file {source_file!r}")
    coverage_status = entry.get("coverage_status")
    if coverage_status not in SUPPORTED_COVERAGE_STATUSES:
        errors.append(f"{context}.coverage_status has unsupported value {coverage_status!r}")

    shape = entry.get("shape")
    if not isinstance(shape, list) or not shape:
        errors.append(f"{context}.shape must be a non-empty list")
    else:
        for index, dimension in enumerate(shape):
            if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
                errors.append(f"{context}.shape[{index}] must be an integer > 0")

    byte_length = entry.get("byte_length")
    stride_bytes = entry.get("stride_bytes")
    if isinstance(byte_length, int) and isinstance(stride_bytes, int) and stride_bytes < byte_length:
        errors.append(f"{context}.stride_bytes must be >= byte_length")

    return (layer_id, expert_id, component_name)


def validate_entries(
    manifest: JSONDict,
    *,
    source_files: dict[str, int | None],
    expected_layers: set[int],
    required_components: list[str],
    errors: list[str],
) -> JSONDict:
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        errors.append("entries must be a non-empty list")
        return {
            "expert_count": 0,
            "component_count": 0,
            "total_estimated_residency_bytes": 0,
            "coverage_counts": {},
            "join_keys": [],
        }

    by_expert: dict[tuple[int, int], set[str]] = {}
    seen_components: set[tuple[int, int, str]] = set()
    file_ranges: dict[str, list[tuple[int, int, str]]] = {}
    coverage_counts = {status: 0 for status in sorted(SUPPORTED_COVERAGE_STATUSES)}
    total_estimated_bytes = 0

    for index, entry in enumerate(entries):
        context = f"entries[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{context} must be an object")
            continue
        key = validate_entry_shape(
            entry,
            context=context,
            source_files=source_files,
            expected_layers=expected_layers,
            required_components=required_components,
            errors=errors,
        )
        if key is None:
            continue

        if key in seen_components:
            layer_id, expert_id, component_name = key
            errors.append(
                f"{context} duplicates component {component_name!r} for expert "
                f"(layer_id={layer_id}, expert_id={expert_id})"
            )
        seen_components.add(key)
        by_expert.setdefault((key[0], key[1]), set()).add(key[2])

        estimated = entry.get("estimated_residency_bytes")
        if isinstance(estimated, int) and not isinstance(estimated, bool):
            total_estimated_bytes += estimated

        coverage = entry.get("coverage_status")
        if isinstance(coverage, str) and coverage in coverage_counts:
            coverage_counts[coverage] += 1

        source_file = entry.get("source_file")
        byte_offset = entry.get("byte_offset")
        byte_length = entry.get("byte_length")
        tensor_name = entry.get("tensor_name")
        if (
            isinstance(source_file, str)
            and isinstance(byte_offset, int)
            and not isinstance(byte_offset, bool)
            and isinstance(byte_length, int)
            and not isinstance(byte_length, bool)
            and isinstance(tensor_name, str)
        ):
            file_ranges.setdefault(source_file, []).append((byte_offset, byte_offset + byte_length, tensor_name))

    required = set(required_components)
    for layer_id, expert_id in sorted(by_expert):
        missing = sorted(required - by_expert[(layer_id, expert_id)])
        extra = sorted(by_expert[(layer_id, expert_id)] - required)
        if missing:
            errors.append(
                f"expert (layer_id={layer_id}, expert_id={expert_id}) missing required components: "
                f"{', '.join(missing)}"
            )
        if extra:
            errors.append(
                f"expert (layer_id={layer_id}, expert_id={expert_id}) has unsupported components: "
                f"{', '.join(extra)}"
            )

    validate_file_ranges(file_ranges, source_files, errors)

    return {
        "expert_count": len(by_expert),
        "component_count": len(seen_components),
        "total_estimated_residency_bytes": total_estimated_bytes,
        "coverage_counts": coverage_counts,
        "join_keys": [
            {"layer_id": layer_id, "expert_id": expert_id}
            for layer_id, expert_id in sorted(by_expert)
        ],
    }


def validate_file_ranges(
    file_ranges: dict[str, list[tuple[int, int, str]]],
    source_files: dict[str, int | None],
    errors: list[str],
) -> None:
    for source_file, ranges in sorted(file_ranges.items()):
        previous_end = 0
        previous_tensor = ""
        for start, end, tensor_name in sorted(ranges):
            if start < previous_end:
                errors.append(
                    f"source_file {source_file!r} tensor {tensor_name!r} overlaps previous tensor "
                    f"{previous_tensor!r}"
                )
            previous_end = max(previous_end, end)
            previous_tensor = tensor_name
        byte_length = source_files.get(source_file)
        if byte_length is not None and previous_end > byte_length:
            errors.append(
                f"source_file {source_file!r} byte ranges end at {previous_end}, "
                f"past declared byte_length {byte_length}"
            )


def validate_expected_totals(manifest: JSONDict, computed: JSONDict, errors: list[str]) -> None:
    expected = manifest.get("expected")
    if not isinstance(expected, dict):
        return
    for field in ("expert_count", "component_count", "total_estimated_residency_bytes"):
        expected_value = expected.get(field)
        computed_value = computed.get(field)
        if isinstance(expected_value, int) and expected_value != computed_value:
            errors.append(f"expected.{field} is {expected_value}, computed {computed_value}")


def validate_manifest(manifest: JSONDict) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_TOP_LEVEL_FIELDS - set(manifest))
    if missing:
        errors.append(f"manifest missing required fields: {', '.join(missing)}")
    if manifest.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        errors.append(
            "schema_version must be "
            f"{SUPPORTED_SCHEMA_VERSION!r}, got {manifest.get('schema_version')!r}"
        )

    for field in ("name", "generated_for", "model_id", "contract_version", "inventory_scope"):
        require_string(manifest, field, errors, context="manifest")
    if manifest.get("source_format") not in SUPPORTED_SOURCE_FORMATS:
        errors.append(f"manifest.source_format has unsupported value {manifest.get('source_format')!r}")
    if manifest.get("backend_family") not in SUPPORTED_BACKEND_FAMILIES:
        errors.append(f"manifest.backend_family has unsupported value {manifest.get('backend_family')!r}")
    require_string_list(manifest, "safety_contract", errors, context="manifest")
    require_string_list(manifest, "next_actions", errors, context="manifest")

    source_files = validate_source_files(manifest, errors)
    expected_layers, required_components = validate_expected(manifest, errors)
    computed = validate_entries(
        manifest,
        source_files=source_files,
        expected_layers=expected_layers,
        required_components=required_components,
        errors=errors,
    )
    validate_expected_totals(manifest, computed, errors)
    return errors


def build_summary(manifest: JSONDict, path: Path) -> JSONDict:
    errors = validate_manifest(manifest)
    source_files = validate_source_files(manifest, []) if isinstance(manifest, dict) else {}
    expected_layers, required_components = validate_expected(manifest, []) if isinstance(manifest, dict) else (set(), [])
    computed = validate_entries(
        manifest,
        source_files=source_files,
        expected_layers=expected_layers,
        required_components=required_components,
        errors=[],
    )
    return {
        "mode": "expert_inventory_manifest_plan",
        "manifest_path": str(path),
        "valid": not errors,
        "errors": errors,
        "schema_version": manifest.get("schema_version"),
        "model_id": manifest.get("model_id"),
        "source_format": manifest.get("source_format"),
        "backend_family": manifest.get("backend_family"),
        "contract_version": manifest.get("contract_version"),
        "layer_ids": sorted(expected_layers),
        "required_components": required_components,
        "expert_count": computed["expert_count"],
        "component_count": computed["component_count"],
        "total_estimated_residency_bytes": computed["total_estimated_residency_bytes"],
        "coverage_counts": computed["coverage_counts"],
        "trace_join_keys": computed["join_keys"],
        "phase_3_replay_ready": not errors and computed["expert_count"] > 0,
        "safety_contract": manifest.get("safety_contract", []),
        "next_actions": manifest.get("next_actions", []),
    }


def print_human_summary(summary: JSONDict) -> None:
    print("MoE Run Anyway expert inventory manifest")
    print(f"Manifest: {summary['manifest_path']}")
    print(f"Valid: {summary['valid']}")
    if summary["errors"]:
        print("Errors:")
        for error in summary["errors"]:
            print(f"  - {error}")
        return
    print(f"Model: {summary['model_id']}")
    print(f"Source format: {summary['source_format']}")
    print(f"Backend family: {summary['backend_family']}")
    print(f"Layers: {', '.join(str(layer) for layer in summary['layer_ids'])}")
    print(f"Experts: {summary['expert_count']}")
    print(f"Components: {summary['component_count']}")
    print(f"Estimated residency bytes: {summary['total_estimated_residency_bytes']}")
    print(f"Phase 3 replay ready: {summary['phase_3_replay_ready']}")
    print("Coverage counts:")
    for status, count in summary["coverage_counts"].items():
        print(f"  - {status}: {count}")
    print("Next actions:")
    for action in summary["next_actions"]:
        print(f"  - {action}")
    print("Safety contract:")
    for item in summary["safety_contract"]:
        print(f"  - {item}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest_path",
        nargs="?",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="expert inventory manifest JSON path",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    return parser


def plan_manifest_path(path: Path) -> tuple[int, JSONDict | None, str | None]:
    try:
        manifest = load_manifest(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not load expert inventory manifest: {exc}"

    summary = build_summary(manifest, path)
    return (0 if summary["valid"] else 2), summary, None


def main_from_test_path(path: Path) -> int:
    status, _, _ = plan_manifest_path(path)
    return status


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary, error_message = plan_manifest_path(args.manifest_path)
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
