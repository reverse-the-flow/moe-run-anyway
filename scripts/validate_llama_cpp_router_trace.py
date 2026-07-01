#!/usr/bin/env python3
"""Validate patched llama.cpp MoE router trace JSONL files."""

from __future__ import annotations

import argparse
import json
import statistics
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
PROMPT_IDENTITY_FIELDS = (
    "prompt_id",
    "prompt_group_id",
    "group_id",
    "repeat",
    "repeat_index",
    "request_id",
    "capture_run_id",
    "capture_sequence",
)
PROMPT_IDENTITY_REQUIRED_FIELD_GROUPS = (
    {
        "id": "prompt_boundary",
        "fields": ("prompt_id", "request_id"),
        "description": "each event must name the prompt/request boundary it came from",
    },
    {
        "id": "capture_order",
        "fields": ("capture_sequence", "repeat_index", "repeat"),
        "description": "each event must carry enough ordering metadata to reconstruct repeated captures",
    },
)
PROMPT_IDENTITY_RUN_FIELDS = ("capture_run_id",)

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


def event_has_any_field(event: JSONDict, fields: tuple[str, ...]) -> bool:
    return any(event.get(field) is not None and event.get(field) != "" for field in fields)


def prompt_identity_summary(events: list[JSONDict]) -> JSONDict:
    field_counts: Counter[str] = Counter()
    for event in events:
        for field in PROMPT_IDENTITY_FIELDS:
            value = event.get(field)
            if value is not None and value != "":
                field_counts[field] += 1

    group_results: list[JSONDict] = []
    missing_group_ids: list[str] = []
    for group in PROMPT_IDENTITY_REQUIRED_FIELD_GROUPS:
        fields = tuple(str(field) for field in group["fields"])
        missing_count = sum(1 for event in events if not event_has_any_field(event, fields))
        group_ready = bool(events) and missing_count == 0
        if not group_ready:
            missing_group_ids.append(str(group["id"]))
        group_results.append(
            {
                "id": group["id"],
                "description": group["description"],
                "fields": list(fields),
                "ready": group_ready,
                "missing_event_count": missing_count,
            }
        )

    run_field_present = any(event_has_any_field(event, PROMPT_IDENTITY_RUN_FIELDS) for event in events)
    if not run_field_present:
        missing_group_ids.append("capture_run")

    return {
        "ready": bool(events) and not missing_group_ids,
        "event_count": len(events),
        "fields_present": sorted(field_counts),
        "field_counts": dict(sorted(field_counts.items())),
        "required_field_groups": group_results,
        "run_fields": list(PROMPT_IDENTITY_RUN_FIELDS),
        "run_field_present": run_field_present,
        "missing_required_group_ids": sorted(set(missing_group_ids)),
        "events_with_any_identity_count": sum(
            1 for event in events if event_has_any_field(event, PROMPT_IDENTITY_FIELDS)
        ),
    }


def selected_expert_events(events: list[JSONDict], errors: list[str]) -> list[JSONDict]:
    selected: list[JSONDict] = []
    for index, event in enumerate(events):
        if event.get("tensor_kind") != "selected_experts":
            continue
        values = event.get("values")
        if not isinstance(values, list) or not values:
            errors.append(f"events[{index}].values must be a non-empty list for selected_experts")
            continue
        layer_id = event.get("layer_id")
        if not isinstance(layer_id, int) or isinstance(layer_id, bool):
            errors.append(f"events[{index}].layer_id must be an integer")
            continue
        selected.append(event)
    if not selected:
        errors.append("trace must include at least one selected_experts event")
    return selected


def build_route_rows(events: list[JSONDict]) -> list[JSONDict]:
    rows: list[JSONDict] = []
    route_index = 0
    for event_index, event in enumerate(events):
        layer_id = str(event["layer_id"])
        for rank, raw_expert_id in enumerate(event["values"]):
            if not isinstance(raw_expert_id, int) or isinstance(raw_expert_id, bool):
                continue
            rows.append(
                {
                    "route_index": route_index,
                    "event_index": event_index,
                    "layer_id": layer_id,
                    "expert_id": str(raw_expert_id),
                    "rank": rank,
                    "tensor_name": event.get("tensor_name"),
                    "model": event.get("model"),
                }
            )
            route_index += 1
    return rows


def route_repetition_summary(routes: list[JSONDict]) -> JSONDict:
    counts: Counter[tuple[str, str]] = Counter((str(row["layer_id"]), str(row["expert_id"])) for row in routes)
    repeated = {key: count for key, count in counts.items() if count > 1}
    samples = [
        {"layer_id": layer_id, "expert_id": expert_id, "route_count": count}
        for (layer_id, expert_id), count in sorted(repeated.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))[:12]
    ]
    return {
        "route_count": len(routes),
        "unique_route_key_count": len(counts),
        "repeated_route_key_count": len(repeated),
        "repeated_route_observation_count": sum(count - 1 for count in repeated.values()),
        "repeated_route_key_samples": samples,
    }


def reuse_distance_summary(routes: list[JSONDict]) -> JSONDict:
    last_seen: dict[tuple[str, str], int] = {}
    distances: list[int] = []
    first_touch_count = 0
    for route in routes:
        key = (str(route["layer_id"]), str(route["expert_id"]))
        route_index = int(route["route_index"])
        if key in last_seen:
            distances.append(route_index - last_seen[key])
        else:
            first_touch_count += 1
        last_seen[key] = route_index
    if not distances:
        return {
            "observations": 0,
            "first_touch_count": first_touch_count,
            "min": None,
            "mean": None,
            "max": None,
        }
    return {
        "observations": len(distances),
        "first_touch_count": first_touch_count,
        "min": min(distances),
        "mean": round(float(statistics.mean(distances)), 3),
        "max": max(distances),
    }


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
    route_errors: list[str] = []
    selected_events = selected_expert_events(events, route_errors)
    routes = build_route_rows(selected_events) if not route_errors else []
    prompt_identity = prompt_identity_summary(events)
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
        "prompt_identity_ready": prompt_identity["ready"],
        "prompt_identity_fields_present": prompt_identity["fields_present"],
        "prompt_identity": prompt_identity,
        "route_repetition": route_repetition_summary(routes),
        "reuse_distance": reuse_distance_summary(routes),
    }


def validate_trace(
    events: list[JSONDict],
    *,
    expected_layers: int | None = None,
    require_kinds: set[str] | None = None,
    require_prompt_identity: bool = False,
    min_reuse_distance_observations: int | None = None,
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

    if require_prompt_identity:
        prompt_identity = summary.get("prompt_identity", {})
        if not isinstance(prompt_identity, dict) or prompt_identity.get("ready") is not True:
            missing = []
            if isinstance(prompt_identity, dict):
                missing = [str(item) for item in prompt_identity.get("missing_required_group_ids", [])]
            suffix = ": " + ",".join(missing) if missing else ""
            errors.append(f"prompt_identity_metadata_missing{suffix}")

    if min_reuse_distance_observations is not None:
        if min_reuse_distance_observations < 0:
            errors.append("min_reuse_distance_observations must be non-negative")
        reuse_distance = summary.get("reuse_distance", {}) if isinstance(summary.get("reuse_distance"), dict) else {}
        observations = int(reuse_distance.get("observations", 0) or 0)
        if observations < min_reuse_distance_observations:
            errors.append(
                f"reuse_distance_observations {observations} below required {min_reuse_distance_observations}"
            )
        repetition = summary.get("route_repetition", {}) if isinstance(summary.get("route_repetition"), dict) else {}
        if min_reuse_distance_observations > 0 and int(repetition.get("repeated_route_key_count", 0) or 0) == 0:
            errors.append("trace missing repeated routed expert keys")

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
    parser.add_argument(
        "--require-prompt-identity",
        action="store_true",
        help="require per-event prompt/request identity and capture ordering metadata",
    )
    parser.add_argument(
        "--min-reuse-distance-observations",
        type=int,
        default=None,
        help="minimum routed expert reuse-distance observations required for policy-candidate capture",
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
        require_prompt_identity=args.require_prompt_identity,
        min_reuse_distance_observations=args.min_reuse_distance_observations,
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
        print(f"Prompt identity ready: {payload['prompt_identity_ready']}")
        print(f"Reuse-distance observations: {payload['reuse_distance']['observations']}")
        if errors:
            print("Errors:")
            for error in errors:
                print(f"  - {error}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
