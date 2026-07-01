#!/usr/bin/env python3
"""Join semantic MoE router traces to expert inventory byte estimates.

This Phase 3 planner consumes validated router trace JSONL and an offline
expert inventory manifest, then reports the byte and reuse-distance facts a
future policy replay needs. It does not load tensor values, mutate residency,
launch runtimes, run Docker, download models, authenticate, or send prompt
traffic.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import plan_expert_inventory


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_PATH = ROOT / "memory-moe-mvp" / "data" / "llama_cpp_router_trace.fixture.jsonl"
DEFAULT_INVENTORY_PATH = ROOT / "memory-moe-mvp" / "data" / "expert_inventory_manifest.fixture.json"
SUPPORTED_TRACE_CONTRACT = "memory-moe-bridge-v1"

JSONDict = dict[str, Any]


def load_jsonl(path: Path) -> list[JSONDict]:
    rows: list[JSONDict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        rows.append(row)
    return rows


def load_inventory(path: Path) -> JSONDict:
    manifest = plan_expert_inventory.load_manifest(path)
    errors = plan_expert_inventory.validate_manifest(manifest)
    if errors:
        raise ValueError(f"inventory manifest is invalid: {'; '.join(errors)}")
    return manifest


def expert_key(layer_id: Any, expert_id: Any) -> tuple[str, str]:
    return str(layer_id), str(expert_id)


def build_inventory_index(manifest: JSONDict) -> dict[tuple[str, str], JSONDict]:
    by_expert: dict[tuple[str, str], JSONDict] = {}
    for entry in manifest.get("entries", []):
        if not isinstance(entry, dict):
            continue
        key = expert_key(entry.get("layer_id"), entry.get("expert_id"))
        item = by_expert.setdefault(
            key,
            {
                "layer_id": key[0],
                "expert_id": key[1],
                "component_count": 0,
                "estimated_residency_bytes": 0,
                "components": [],
                "source_files": set(),
            },
        )
        item["component_count"] += 1
        item["estimated_residency_bytes"] += int(entry.get("estimated_residency_bytes", 0))
        item["components"].append(str(entry.get("component_name")))
        item["source_files"].add(str(entry.get("source_file")))

    normalized: dict[tuple[str, str], JSONDict] = {}
    for key, item in by_expert.items():
        item["components"] = sorted(set(item["components"]))
        item["source_files"] = sorted(item["source_files"])
        normalized[key] = item
    return normalized


def selected_expert_events(events: list[JSONDict], errors: list[str]) -> list[JSONDict]:
    selected: list[JSONDict] = []
    for index, event in enumerate(events):
        if event.get("tensor_kind") != "selected_experts":
            continue
        if event.get("contract_version") != SUPPORTED_TRACE_CONTRACT:
            errors.append(
                f"events[{index}].contract_version must be {SUPPORTED_TRACE_CONTRACT!r}, "
                f"got {event.get('contract_version')!r}"
            )
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


def reuse_distance_summary(routes: list[JSONDict]) -> JSONDict:
    last_seen: dict[tuple[str, str], int] = {}
    distances: list[int] = []
    first_touch_count = 0
    for route in routes:
        key = (route["layer_id"], route["expert_id"])
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


def build_join_summary(trace_path: Path, inventory_path: Path) -> JSONDict:
    errors: list[str] = []
    trace_events = load_jsonl(trace_path)
    inventory = load_inventory(inventory_path)
    inventory_index = build_inventory_index(inventory)
    selected_events = selected_expert_events(trace_events, errors)
    routes = build_route_rows(selected_events)

    joined_routes: list[JSONDict] = []
    missing_routes: list[JSONDict] = []
    route_counter: Counter[tuple[str, str]] = Counter()
    layer_counter: Counter[str] = Counter()
    layer_bytes: defaultdict[str, int] = defaultdict(int)
    unique_experts: set[tuple[str, str]] = set()
    total_route_bytes = 0

    for route in routes:
        key = (route["layer_id"], route["expert_id"])
        inventory_item = inventory_index.get(key)
        if inventory_item is None:
            missing_routes.append(route)
            continue
        route_bytes = int(inventory_item["estimated_residency_bytes"])
        joined = dict(route)
        joined["estimated_residency_bytes"] = route_bytes
        joined["component_count"] = inventory_item["component_count"]
        joined["components"] = inventory_item["components"]
        joined["source_files"] = inventory_item["source_files"]
        joined_routes.append(joined)
        route_counter[key] += 1
        layer_counter[route["layer_id"]] += 1
        layer_bytes[route["layer_id"]] += route_bytes
        unique_experts.add(key)
        total_route_bytes += route_bytes

    unique_bytes = sum(int(inventory_index[key]["estimated_residency_bytes"]) for key in unique_experts)
    if missing_routes:
        errors.append(f"{len(missing_routes)} routed experts were missing from inventory")

    per_layer = []
    for layer_id in sorted({route["layer_id"] for route in routes}, key=lambda value: int(value) if value.isdigit() else value):
        layer_unique = {key for key in unique_experts if key[0] == layer_id}
        per_layer.append(
            {
                "layer_id": layer_id,
                "route_count": layer_counter[layer_id],
                "unique_expert_count": len(layer_unique),
                "unique_estimated_residency_bytes": sum(
                    int(inventory_index[key]["estimated_residency_bytes"]) for key in layer_unique
                ),
                "route_estimated_residency_bytes": layer_bytes[layer_id],
                "missing_route_count": sum(1 for route in missing_routes if route["layer_id"] == layer_id),
            }
        )

    return {
        "mode": "trace_inventory_replay_plan",
        "trace_path": str(trace_path),
        "inventory_path": str(inventory_path),
        "valid": not errors,
        "errors": errors,
        "trace_event_count": len(trace_events),
        "selected_expert_event_count": len(selected_events),
        "route_count": len(routes),
        "joined_route_count": len(joined_routes),
        "missing_route_count": len(missing_routes),
        "unique_expert_count": len(unique_experts),
        "unique_estimated_residency_bytes": unique_bytes,
        "route_estimated_residency_bytes": total_route_bytes,
        "reuse_distance": reuse_distance_summary(joined_routes),
        "per_layer": per_layer,
        "top_requested_experts": [
            {
                "layer_id": layer_id,
                "expert_id": expert_id,
                "route_count": count,
                "estimated_residency_bytes": int(inventory_index[(layer_id, expert_id)]["estimated_residency_bytes"]),
            }
            for (layer_id, expert_id), count in route_counter.most_common()
        ],
        "missing_routes": missing_routes,
        "policy_replay_prerequisites": {
            "inventory_join_ready": not errors and bool(joined_routes),
            "baseline_policy_names": [
                "observe_only",
                "keep_hot",
                "preload_shortlist",
                "evict_cold",
                "fallback_dense",
                "no_live_actuator",
            ],
            "live_actuator_available": False,
        },
        "safety_contract": [
            "planner validates local artifacts only",
            "planner does not load tensor values",
            "planner does not mutate runtime residency",
            "planner does not launch model servers",
            "planner does not send prompt traffic",
        ],
    }


def print_human_summary(summary: JSONDict) -> None:
    print("MoE Run Anyway trace inventory replay plan")
    print(f"Trace: {summary['trace_path']}")
    print(f"Inventory: {summary['inventory_path']}")
    print(f"Valid: {summary['valid']}")
    if summary["errors"]:
        print("Errors:")
        for error in summary["errors"]:
            print(f"  - {error}")
        return
    print(f"Selected expert events: {summary['selected_expert_event_count']}")
    print(f"Joined routes: {summary['joined_route_count']}")
    print(f"Unique experts: {summary['unique_expert_count']}")
    print(f"Unique estimated residency bytes: {summary['unique_estimated_residency_bytes']}")
    print(f"Route estimated residency bytes: {summary['route_estimated_residency_bytes']}")
    print(f"Reuse distance observations: {summary['reuse_distance']['observations']}")
    print("Policy replay prerequisites:")
    for name in summary["policy_replay_prerequisites"]["baseline_policy_names"]:
        print(f"  - {name}")
    print("Safety contract:")
    for item in summary["safety_contract"]:
        print(f"  - {item}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_path", nargs="?", type=Path, default=DEFAULT_TRACE_PATH)
    parser.add_argument("inventory_path", nargs="?", type=Path, default=DEFAULT_INVENTORY_PATH)
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    return parser


def plan_paths(trace_path: Path, inventory_path: Path) -> tuple[int, JSONDict | None, str | None]:
    try:
        summary = build_join_summary(trace_path, inventory_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not build trace inventory replay plan: {exc}"
    return (0 if summary["valid"] else 2), summary, None


def main_from_test_paths(trace_path: Path, inventory_path: Path) -> int:
    status, _, _ = plan_paths(trace_path, inventory_path)
    return status


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary, error_message = plan_paths(args.trace_path, args.inventory_path)
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
