#!/usr/bin/env python3
"""Audit whether a semantic trace is paired to real-model inventory evidence.

This Phase 3 gate sits between fixture replay and real managed-loading claims.
It validates the trace/inventory join, checks model/backend/contract agreement,
and reports whether the pair is fixture-only or real-model evidence. It does
not read tensor values, check endpoints, launch runtimes, or send prompt traffic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import plan_trace_inventory_replay


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_PATH = ROOT / "memory-moe-mvp" / "data" / "llama_cpp_router_trace.fixture.jsonl"
DEFAULT_INVENTORY_PATH = ROOT / "memory-moe-mvp" / "data" / "expert_inventory_manifest.fixture.json"
SUPPORTED_TRACE_CONTRACT = "memory-moe-bridge-v1"

JSONDict = dict[str, Any]


def is_fixture_text(value: Any) -> bool:
    return isinstance(value, str) and ("fixture" in value.lower() or "synthetic" in value.lower())


def trace_model_names(events: list[JSONDict]) -> list[str]:
    names = {
        str(event["model"])
        for event in events
        if isinstance(event.get("model"), str) and event["model"].strip()
    }
    return sorted(names)


def trace_backend_families(events: list[JSONDict]) -> list[str]:
    families = {
        str(event["backend_family"])
        for event in events
        if isinstance(event.get("backend_family"), str) and event["backend_family"].strip()
    }
    return sorted(families)


def trace_contract_versions(events: list[JSONDict]) -> list[str]:
    versions = {
        str(event["contract_version"])
        for event in events
        if isinstance(event.get("contract_version"), str) and event["contract_version"].strip()
    }
    return sorted(versions)


def inventory_source_names(manifest: JSONDict) -> list[str]:
    names: set[str] = set()
    model_id = manifest.get("model_id")
    if isinstance(model_id, str) and model_id.strip():
        names.add(model_id)
        names.add(Path(model_id).name)
    for source in manifest.get("source_files", []):
        if not isinstance(source, dict):
            continue
        path = source.get("path")
        if isinstance(path, str) and path.strip():
            names.add(path)
            names.add(Path(path).name)
    return sorted(names)


def names_match(trace_names: list[str], inventory_names: list[str]) -> bool:
    if len(trace_names) != 1:
        return False
    trace_name = trace_names[0]
    candidates = set(inventory_names)
    return trace_name in candidates or Path(trace_name).name in candidates


def manifest_fixture_signals(manifest: JSONDict) -> list[str]:
    signals: list[str] = []
    for field in ("model_id", "source_format", "inventory_scope", "name"):
        if is_fixture_text(manifest.get(field)):
            signals.append(f"manifest.{field}")
    for index, source in enumerate(manifest.get("source_files", [])):
        if not isinstance(source, dict):
            continue
        for field in ("path", "note"):
            if is_fixture_text(source.get(field)):
                signals.append(f"source_files[{index}].{field}")
    return signals


def trace_fixture_signals(events: list[JSONDict], trace_path: Path) -> list[str]:
    signals: list[str] = []
    if is_fixture_text(trace_path.name):
        signals.append("trace_path")
    for index, event in enumerate(events):
        for field in ("model", "tensor_name"):
            if is_fixture_text(event.get(field)):
                signals.append(f"events[{index}].{field}")
    return sorted(set(signals))


def coverage_summary(manifest: JSONDict) -> JSONDict:
    counts: dict[str, int] = {}
    for entry in manifest.get("entries", []):
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("coverage_status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return {
        "coverage_counts": dict(sorted(counts.items())),
        "all_components_complete": bool(counts) and set(counts) == {"complete"},
    }


def build_pairing_summary(trace_path: Path, inventory_path: Path, *, require_real_model: bool = False) -> JSONDict:
    trace_events = plan_trace_inventory_replay.load_jsonl(trace_path)
    inventory = plan_trace_inventory_replay.load_inventory(inventory_path)
    join_summary = plan_trace_inventory_replay.build_join_summary(trace_path, inventory_path)

    trace_names = trace_model_names(trace_events)
    inventory_names = inventory_source_names(inventory)
    trace_backends = trace_backend_families(trace_events)
    trace_contracts = trace_contract_versions(trace_events)
    manifest_fixture = manifest_fixture_signals(inventory)
    trace_fixture = trace_fixture_signals(trace_events, trace_path)
    coverage = coverage_summary(inventory)

    errors: list[str] = []
    blockers: list[JSONDict] = []
    if not join_summary.get("valid"):
        errors.extend(f"trace inventory join: {error}" for error in join_summary.get("errors", []))
    if len(trace_names) != 1:
        errors.append(f"trace must name exactly one model, got {trace_names}")
    if len(trace_backends) != 1:
        errors.append(f"trace must name exactly one backend family, got {trace_backends}")
    if trace_contracts != [SUPPORTED_TRACE_CONTRACT]:
        errors.append(f"trace contract versions must be [{SUPPORTED_TRACE_CONTRACT!r}], got {trace_contracts}")

    model_match = names_match(trace_names, inventory_names)
    if not model_match:
        errors.append(
            "trace model must match inventory model_id or source file name "
            f"(trace={trace_names}, inventory={inventory_names})"
        )

    inventory_backend = inventory.get("backend_family")
    backend_match = len(trace_backends) == 1 and trace_backends[0] == inventory_backend
    if not backend_match:
        errors.append(f"trace backend {trace_backends} must match inventory backend {inventory_backend!r}")

    inventory_contract = inventory.get("contract_version")
    contract_match = trace_contracts == [inventory_contract]
    if not contract_match:
        errors.append(f"trace contract {trace_contracts} must match inventory contract {inventory_contract!r}")

    if not coverage["all_components_complete"]:
        blockers.append(
            {
                "id": "inventory_component_coverage_incomplete",
                "description": "Inventory contains partial, missing, or unknown component coverage.",
            }
        )

    fixture_only_pair = bool(manifest_fixture or trace_fixture)
    if fixture_only_pair:
        blockers.append(
            {
                "id": "fixture_only_trace_inventory_pair",
                "description": "Trace and inventory pair is valid for schema replay but is not real-model evidence.",
                "manifest_fixture_signals": manifest_fixture,
                "trace_fixture_signals": trace_fixture,
            }
        )

    real_model_pair_ready = not errors and not fixture_only_pair and coverage["all_components_complete"]
    if require_real_model and not real_model_pair_ready:
        errors.append("real-model trace/inventory pair is required but not ready")

    return {
        "mode": "real_model_trace_inventory_pairing_plan",
        "trace_path": str(trace_path),
        "inventory_path": str(inventory_path),
        "valid": not errors,
        "errors": errors,
        "trace_inventory_join_valid": bool(join_summary.get("valid")),
        "joined_route_count": join_summary.get("joined_route_count", 0),
        "missing_route_count": join_summary.get("missing_route_count", 0),
        "unique_expert_count": join_summary.get("unique_expert_count", 0),
        "unique_estimated_residency_bytes": join_summary.get("unique_estimated_residency_bytes", 0),
        "trace_model_names": trace_names,
        "inventory_model_id": inventory.get("model_id"),
        "inventory_source_names": inventory_names,
        "model_match": model_match,
        "trace_backend_families": trace_backends,
        "inventory_backend_family": inventory_backend,
        "backend_match": backend_match,
        "trace_contract_versions": trace_contracts,
        "inventory_contract_version": inventory_contract,
        "contract_match": contract_match,
        "coverage": coverage,
        "fixture_only_pair": fixture_only_pair,
        "real_model_pair_ready": real_model_pair_ready,
        "blockers": blockers,
        "phase_3_gate": {
            "real_model_inventory_for_traced_model": real_model_pair_ready,
            "ready_for_live_spike": False,
        },
        "safety_contract": [
            "planner validates local trace and inventory metadata only",
            "planner does not load tensor values",
            "planner does not check endpoints",
            "planner does not launch model servers",
            "planner does not send prompt traffic",
            "planner does not claim live expert paging",
        ],
        "next_actions": [
            "Run a scanner against the same model that produced the semantic trace.",
            "Validate this pairing planner with --require-real-model before treating replay metrics as real-model evidence.",
            "Keep fixture-only pairs useful for schema tests but out of live-spike decisions.",
        ],
    }


def print_human_summary(summary: JSONDict) -> None:
    print("MoE Run Anyway real-model trace/inventory pairing")
    print(f"Trace: {summary['trace_path']}")
    print(f"Inventory: {summary['inventory_path']}")
    print(f"Valid: {summary['valid']}")
    if summary["errors"]:
        print("Errors:")
        for error in summary["errors"]:
            print(f"  - {error}")
    print(f"Joined routes: {summary['joined_route_count']}")
    print(f"Model match: {summary['model_match']}")
    print(f"Backend match: {summary['backend_match']}")
    print(f"Contract match: {summary['contract_match']}")
    print(f"Fixture-only pair: {summary['fixture_only_pair']}")
    print(f"Real-model pair ready: {summary['real_model_pair_ready']}")
    if summary["blockers"]:
        print("Blockers:")
        for blocker in summary["blockers"]:
            print(f"  - {blocker['id']}")
    print("Safety contract:")
    for item in summary["safety_contract"]:
        print(f"  - {item}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_path", nargs="?", type=Path, default=DEFAULT_TRACE_PATH)
    parser.add_argument("inventory_path", nargs="?", type=Path, default=DEFAULT_INVENTORY_PATH)
    parser.add_argument(
        "--require-real-model",
        action="store_true",
        help="return nonzero unless the pair is real-model evidence rather than fixture evidence",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    return parser


def plan_paths(
    trace_path: Path,
    inventory_path: Path,
    *,
    require_real_model: bool = False,
) -> tuple[int, JSONDict | None, str | None]:
    try:
        summary = build_pairing_summary(
            trace_path,
            inventory_path,
            require_real_model=require_real_model,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not build real-model trace/inventory pairing plan: {exc}"
    return (0 if summary["valid"] else 2), summary, None


def main_from_test_paths(trace_path: Path, inventory_path: Path) -> int:
    status, _, _ = plan_paths(trace_path, inventory_path)
    return status


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary, error_message = plan_paths(
        args.trace_path,
        args.inventory_path,
        require_real_model=args.require_real_model,
    )
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
