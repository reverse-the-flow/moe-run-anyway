#!/usr/bin/env python3
"""Validate patched llama.cpp MoE router trace JSONL files."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CONTRACT_VERSION = "memory-moe-bridge-v1"
EVENT_TYPE = "llama_cpp_moe_router_tensor"
BACKEND_FAMILY = "llama_cpp"
SOURCE = "llama_cpp_eval_callback"
SUPPORTED_KINDS = {
    "selected_experts",
    "selected_weights",
    "selected_weights_norm",
}
WEIGHT_KINDS = {"selected_weights", "selected_weights_norm"}

JSONDict = dict[str, Any]


def load_jsonl(path: Path) -> list[JSONDict]:
    events: list[JSONDict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number}: event must be a JSON object")
            payload["_line_number"] = line_number
            events.append(payload)
    return events


def validate_shape(event_id: str, value: Any, errors: list[str]) -> list[int]:
    if not isinstance(value, list) or len(value) != 4:
        errors.append(f"{event_id}: shape must be a four-item list")
        return []
    shape: list[int] = []
    for item in value:
        if not isinstance(item, int) or item <= 0:
            errors.append(f"{event_id}: shape entries must be positive integers")
            return []
        shape.append(item)
    return shape


def validate_values(event_id: str, kind: str, ggml_type: str, values: Any, errors: list[str]) -> None:
    if not isinstance(values, list) or not values:
        errors.append(f"{event_id}: values must be a non-empty list")
        return
    if kind == "selected_experts":
        if ggml_type != "i32":
            errors.append(f"{event_id}: selected_experts must use ggml_type i32")
        for value in values:
            if not isinstance(value, int) or value < 0:
                errors.append(f"{event_id}: selected expert ids must be non-negative integers")
                return
    elif kind in WEIGHT_KINDS:
        if ggml_type not in {"f32", "f16"}:
            errors.append(f"{event_id}: selected weights must use ggml_type f32 or f16")
        for value in values:
            if not isinstance(value, (int, float)):
                errors.append(f"{event_id}: selected weights must be numeric")
                return


def validate_event(event: JSONDict) -> list[str]:
    errors: list[str] = []
    line = event.get("_line_number", "?")
    event_id = f"line {line}"

    required_string_fields = {
        "event_type": EVENT_TYPE,
        "contract_version": CONTRACT_VERSION,
        "backend_family": BACKEND_FAMILY,
        "source": SOURCE,
    }
    for field, expected in required_string_fields.items():
        if event.get(field) != expected:
            errors.append(f"{event_id}: {field} must be {expected!r}")

    for field in ("timestamp", "model", "tensor_name", "tensor_kind", "ggml_type"):
        if not isinstance(event.get(field), str) or not event[field].strip():
            errors.append(f"{event_id}: {field} must be a non-empty string")

    kind = event.get("tensor_kind")
    if kind not in SUPPORTED_KINDS:
        errors.append(f"{event_id}: unsupported tensor_kind {kind!r}")

    layer_id = event.get("layer_id")
    if not isinstance(layer_id, int) or layer_id < 0:
        errors.append(f"{event_id}: layer_id must be a non-negative integer")

    validate_shape(event_id, event.get("shape"), errors)
    validate_values(event_id, str(kind), str(event.get("ggml_type")), event.get("values"), errors)

    value_count = event.get("value_count")
    if not isinstance(value_count, int) or value_count < len(event.get("values", [])):
        errors.append(f"{event_id}: value_count must be an integer >= emitted values length")

    if not isinstance(event.get("values_truncated"), bool):
        errors.append(f"{event_id}: values_truncated must be a boolean")

    return errors


def summarize_events(events: list[JSONDict]) -> JSONDict:
    by_kind = Counter(str(event.get("tensor_kind")) for event in events)
    by_model = Counter(str(event.get("model")) for event in events)
    layers_by_kind: dict[str, set[int]] = defaultdict(set)
    top_k_by_layer: dict[int, int] = {}
    token_count_by_layer: dict[int, int] = {}
    unique_experts: set[int] = set()

    for event in events:
        kind = str(event.get("tensor_kind"))
        layer_id = event.get("layer_id")
        if isinstance(layer_id, int) and layer_id >= 0:
            layers_by_kind[kind].add(layer_id)
            shape = event.get("shape")
            if kind == "selected_experts" and isinstance(shape, list) and len(shape) >= 2:
                top_k_by_layer[layer_id] = int(shape[0])
                token_count_by_layer[layer_id] = int(shape[1])
                values = event.get("values")
                if isinstance(values, list):
                    unique_experts.update(value for value in values if isinstance(value, int))

    all_layers = set().union(*layers_by_kind.values()) if layers_by_kind else set()
    return {
        "event_count": len(events),
        "by_tensor_kind": dict(sorted(by_kind.items())),
        "by_model": dict(sorted(by_model.items())),
        "layer_count": len(all_layers),
        "layer_min": min(all_layers) if all_layers else None,
        "layer_max": max(all_layers) if all_layers else None,
        "layers_by_kind": {
            kind: sorted(layers)
            for kind, layers in sorted(layers_by_kind.items())
        },
        "top_k_values": sorted(set(top_k_by_layer.values())),
        "token_count_values": sorted(set(token_count_by_layer.values())),
        "unique_experts_seen": len(unique_experts),
        "unique_expert_sample": sorted(unique_experts)[:32],
    }


def validate_trace(
    events: list[JSONDict],
    *,
    expected_layers: int | None = None,
    require_kinds: set[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    if not events:
        return ["trace must contain at least one event"]

    for event in events:
        errors.extend(validate_event(event))

    summary = summarize_events(events)
    by_kind = summary["by_tensor_kind"]
    if "selected_experts" not in by_kind:
        errors.append("trace must contain selected_experts events")
    if not any(kind in by_kind for kind in WEIGHT_KINDS):
        errors.append("trace must contain selected weight events")

    if require_kinds:
        missing = sorted(kind for kind in require_kinds if kind not in by_kind)
        if missing:
            errors.append(f"trace missing required tensor kinds: {missing}")

        selected_layers = set(summary["layers_by_kind"].get("selected_experts", []))
        for kind in require_kinds:
            layers = set(summary["layers_by_kind"].get(kind, []))
            if selected_layers and layers and layers != selected_layers:
                errors.append(f"{kind} layers do not match selected_experts layers")

    if expected_layers is not None and summary["layer_count"] != expected_layers:
        errors.append(
            f"layer_count {summary['layer_count']} does not match expected_layers {expected_layers}"
        )

    return errors


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_path", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--expected-layers", type=int)
    parser.add_argument(
        "--require-kind",
        action="append",
        choices=sorted(SUPPORTED_KINDS),
        default=[],
        help="tensor kind that must be present; may be repeated",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    events = load_jsonl(args.trace_path)
    required_kinds = set(args.require_kind)
    errors = validate_trace(
        events,
        expected_layers=args.expected_layers,
        require_kinds=required_kinds if required_kinds else None,
    )
    payload = {
        "valid": not errors,
        "errors": errors,
        "trace_path": str(args.trace_path),
        **summarize_events(events),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"llama.cpp MoE router trace: {args.trace_path}")
        print(f"Valid: {payload['valid']}")
        print(f"Events: {payload['event_count']}")
        print(f"Layer count: {payload['layer_count']}")
        print(f"By tensor kind: {payload['by_tensor_kind']}")
        print(f"Top-k values: {payload['top_k_values']}")
        print(f"Unique experts seen: {payload['unique_experts_seen']}")
        if errors:
            print("Errors:")
            for error in errors:
                print(f"  - {error}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
