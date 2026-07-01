#!/usr/bin/env python3
"""Replay Phase 3 baseline residency policies over trace+inventory artifacts.

This planner is an offline audit step. It validates the trace/inventory join and
baseline policy contract, then simulates policy behavior over the joined route
stream. It does not load tensor values, launch runtimes, mutate residency, write
expert stores, use secrets, run Docker, or send prompt traffic.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import plan_baseline_replay_policies
import plan_real_model_trace_inventory_pairing
import plan_trace_inventory_replay


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_PATH = ROOT / "memory-moe-mvp" / "data" / "llama_cpp_router_trace.fixture.jsonl"
DEFAULT_INVENTORY_PATH = ROOT / "memory-moe-mvp" / "data" / "expert_inventory_manifest.fixture.json"
DEFAULT_POLICIES_PATH = ROOT / "memory-moe-mvp" / "data" / "baseline_replay_policies.json"
DEFAULT_RESIDENT_BUDGET_FRACTION = 0.5
MAX_CHURN_PER_ROUTE = 1.0
MAX_FALLBACK_FREQUENCY = 0.25

JSONDict = dict[str, Any]
ExpertKey = tuple[str, str]


def rate(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def expert_key(route: JSONDict) -> ExpertKey:
    return str(route["layer_id"]), str(route["expert_id"])


def expert_key_id(key: ExpertKey) -> str:
    return f"{key[0]}:{key[1]}"


def build_joined_routes(trace_path: Path, inventory_path: Path) -> tuple[list[JSONDict], list[str]]:
    errors: list[str] = []
    trace_events = plan_trace_inventory_replay.load_jsonl(trace_path)
    inventory = plan_trace_inventory_replay.load_inventory(inventory_path)
    inventory_index = plan_trace_inventory_replay.build_inventory_index(inventory)
    selected_events = plan_trace_inventory_replay.selected_expert_events(trace_events, errors)
    routes = plan_trace_inventory_replay.build_route_rows(selected_events)

    joined_routes: list[JSONDict] = []
    missing_routes: list[JSONDict] = []
    for route in routes:
        key = expert_key(route)
        inventory_item = inventory_index.get(key)
        if inventory_item is None:
            missing_routes.append(route)
            continue
        joined = dict(route)
        joined["expert_key"] = expert_key_id(key)
        joined["estimated_residency_bytes"] = int(inventory_item["estimated_residency_bytes"])
        joined["component_count"] = int(inventory_item["component_count"])
        joined["components"] = inventory_item["components"]
        joined["source_files"] = inventory_item["source_files"]
        joined_routes.append(joined)

    if missing_routes:
        errors.append(f"{len(missing_routes)} routed experts were missing from inventory")
    return joined_routes, errors


def resolve_resident_budget(
    join_summary: JSONDict,
    *,
    resident_budget_bytes: int | None,
    resident_budget_fraction: float,
) -> JSONDict:
    unique_bytes = int(join_summary.get("unique_estimated_residency_bytes", 0))
    if resident_budget_bytes is not None:
        budget = resident_budget_bytes
        source = "explicit_bytes"
    else:
        budget = int(unique_bytes * resident_budget_fraction)
        source = "fraction_of_unique_estimated_residency"
    return {
        "resident_budget_bytes": max(0, budget),
        "resident_budget_fraction": resident_budget_fraction,
        "budget_source": source,
        "unique_estimated_residency_bytes": unique_bytes,
    }


def expert_byte_map(routes: list[JSONDict]) -> dict[ExpertKey, int]:
    sizes: dict[ExpertKey, int] = {}
    for route in routes:
        key = expert_key(route)
        sizes[key] = int(route["estimated_residency_bytes"])
    return sizes


def rank_preload_shortlist(routes: list[JSONDict]) -> list[ExpertKey]:
    counts: Counter[ExpertKey] = Counter(expert_key(route) for route in routes)
    first_seen: dict[ExpertKey, int] = {}
    bytes_by_key = expert_byte_map(routes)
    for index, route in enumerate(routes):
        first_seen.setdefault(expert_key(route), index)
    return sorted(
        counts,
        key=lambda key: (-counts[key], first_seen[key], bytes_by_key[key], key[0], key[1]),
    )


def preload_until_budget(
    *,
    resident: dict[ExpertKey, JSONDict],
    expert_bytes: dict[ExpertKey, int],
    resident_budget_bytes: int,
    preload_keys: list[ExpertKey],
) -> JSONDict:
    resident_bytes = 0
    preloaded: list[str] = []
    rejected: list[str] = []
    actions = 0
    for key in preload_keys:
        size = expert_bytes[key]
        if size > resident_budget_bytes:
            rejected.append(expert_key_id(key))
            continue
        if resident_bytes + size > resident_budget_bytes:
            continue
        resident[key] = {"bytes": size, "last_used_route": -1}
        resident_bytes += size
        preloaded.append(expert_key_id(key))
        actions += 1
    return {
        "resident_bytes": resident_bytes,
        "preloaded_experts": preloaded,
        "preload_rejected_experts": rejected,
        "preload_action_count": actions,
    }


def evict_lru_until_fits(
    *,
    resident: dict[ExpertKey, JSONDict],
    resident_bytes: int,
    needed_bytes: int,
    resident_budget_bytes: int,
) -> tuple[int, int, list[str]]:
    evictions = 0
    evicted: list[str] = []
    while resident and resident_bytes + needed_bytes > resident_budget_bytes:
        evict_key = min(resident, key=lambda key: (int(resident[key]["last_used_route"]), key[0], key[1]))
        resident_bytes -= int(resident[evict_key]["bytes"])
        del resident[evict_key]
        evictions += 1
        evicted.append(expert_key_id(evict_key))
    return resident_bytes, evictions, evicted


def simulate_residency_policy(
    policy_id: str,
    routes: list[JSONDict],
    *,
    resident_budget_bytes: int,
    preload_keys: list[ExpertKey] | None = None,
    evict_cold_after_routes: int | None = None,
) -> JSONDict:
    expert_bytes = expert_byte_map(routes)
    resident: dict[ExpertKey, JSONDict] = {}
    preload = preload_until_budget(
        resident=resident,
        expert_bytes=expert_bytes,
        resident_budget_bytes=resident_budget_bytes,
        preload_keys=preload_keys or [],
    )
    resident_bytes = int(preload["resident_bytes"])
    preload_action_count = int(preload["preload_action_count"])
    hit_count = 0
    miss_count = 0
    fallback_count = 0
    load_count = 0
    eviction_count = 0
    estimated_byte_misses = 0
    max_resident_bytes = resident_bytes
    event_samples: list[JSONDict] = []

    for route in routes:
        route_index = int(route["route_index"])
        key = expert_key(route)
        size = int(route["estimated_residency_bytes"])
        if key in resident:
            hit_count += 1
            resident[key]["last_used_route"] = route_index
            action = "hit"
        else:
            miss_count += 1
            estimated_byte_misses += size
            if size > resident_budget_bytes:
                fallback_count += 1
                action = "fallback_expert_exceeds_budget"
            else:
                resident_bytes, evictions, evicted = evict_lru_until_fits(
                    resident=resident,
                    resident_bytes=resident_bytes,
                    needed_bytes=size,
                    resident_budget_bytes=resident_budget_bytes,
                )
                eviction_count += evictions
                if evicted and len(event_samples) < 12:
                    event_samples.append(
                        {
                            "route_index": route_index,
                            "action": "evict_for_load",
                            "experts": evicted,
                        }
                    )
                if resident_bytes + size <= resident_budget_bytes:
                    resident[key] = {"bytes": size, "last_used_route": route_index}
                    resident_bytes += size
                    load_count += 1
                    action = "load_after_miss"
                else:
                    fallback_count += 1
                    action = "fallback_budget_unavailable"

        if evict_cold_after_routes is not None:
            cold_cutoff = route_index - evict_cold_after_routes
            cold_keys = [
                candidate_key
                for candidate_key, item in resident.items()
                if int(item["last_used_route"]) <= cold_cutoff and candidate_key != key
            ]
            for cold_key in sorted(cold_keys, key=lambda item: (resident[item]["last_used_route"], item[0], item[1])):
                resident_bytes -= int(resident[cold_key]["bytes"])
                del resident[cold_key]
                eviction_count += 1
                if len(event_samples) < 12:
                    event_samples.append(
                        {
                            "route_index": route_index,
                            "action": "evict_cold",
                            "experts": [expert_key_id(cold_key)],
                        }
                    )

        max_resident_bytes = max(max_resident_bytes, resident_bytes)
        if len(event_samples) < 12:
            event_samples.append(
                {
                    "route_index": route_index,
                    "expert": expert_key_id(key),
                    "action": action,
                    "resident_bytes": resident_bytes,
                }
            )

    route_count = len(routes)
    residency_action_count = preload_action_count + load_count + eviction_count
    return {
        "policy_id": policy_id,
        "route_count": route_count,
        "hit_count": hit_count,
        "miss_count": miss_count,
        "miss_rate": rate(miss_count, route_count),
        "warm_hit_rate": rate(hit_count, route_count),
        "fallback_count": fallback_count,
        "fallback_frequency": rate(fallback_count, route_count),
        "estimated_byte_misses": estimated_byte_misses,
        "resident_budget_bytes": resident_budget_bytes,
        "max_resident_bytes": max_resident_bytes,
        "final_resident_expert_count": len(resident),
        "preload_count": preload_action_count,
        "load_count": load_count,
        "eviction_count": eviction_count,
        "residency_action_count": residency_action_count,
        "churn": rate(residency_action_count, route_count),
        "preloaded_experts": preload["preloaded_experts"],
        "preload_rejected_experts": preload["preload_rejected_experts"],
        "event_samples": event_samples,
    }


def observe_only_result(routes: list[JSONDict]) -> JSONDict:
    route_count = len(routes)
    return {
        "policy_id": "observe_only",
        "route_count": route_count,
        "hit_count": 0,
        "miss_count": route_count,
        "miss_rate": rate(route_count, route_count),
        "warm_hit_rate": 0.0,
        "fallback_count": 0,
        "fallback_frequency": 0.0,
        "estimated_byte_misses": sum(int(route["estimated_residency_bytes"]) for route in routes),
        "resident_budget_bytes": 0,
        "max_resident_bytes": 0,
        "final_resident_expert_count": 0,
        "preload_count": 0,
        "load_count": 0,
        "eviction_count": 0,
        "residency_action_count": 0,
        "churn": 0.0,
        "preloaded_experts": [],
        "preload_rejected_experts": [],
        "event_samples": [],
        "rejected_policy_reasons": [],
        "candidate_for_live_spike": False,
        "decision": "observation_baseline_only",
    }


def fallback_dense_result(routes: list[JSONDict]) -> JSONDict:
    route_count = len(routes)
    return {
        "policy_id": "fallback_dense",
        "route_count": route_count,
        "hit_count": route_count,
        "miss_count": 0,
        "miss_rate": 0.0,
        "warm_hit_rate": rate(route_count, route_count),
        "fallback_count": route_count,
        "fallback_frequency": rate(route_count, route_count),
        "estimated_byte_misses": 0,
        "resident_budget_bytes": 0,
        "max_resident_bytes": 0,
        "final_resident_expert_count": 0,
        "preload_count": 0,
        "load_count": 0,
        "eviction_count": 0,
        "residency_action_count": 0,
        "churn": 0.0,
        "preloaded_experts": [],
        "preload_rejected_experts": [],
        "event_samples": [],
        "rejected_policy_reasons": ["fallback_output_missing_for_quality_comparison"],
        "candidate_for_live_spike": False,
        "decision": "dense_fallback_reference_missing_output_artifact",
    }


def no_live_actuator_result(routes: list[JSONDict], policy_results: list[JSONDict]) -> JSONDict:
    route_count = len(routes)
    benefit_proven = any(
        result.get("candidate_for_live_spike") is True
        for result in policy_results
        if result.get("policy_id") not in {"observe_only", "fallback_dense"}
    )
    reasons = ["no_residency_control", "cleanup_restore_unproven"]
    if not benefit_proven:
        reasons.append("policy_benefit_unproven")
    return {
        "policy_id": "no_live_actuator",
        "route_count": route_count,
        "hit_count": 0,
        "miss_count": route_count,
        "miss_rate": rate(route_count, route_count),
        "warm_hit_rate": 0.0,
        "fallback_count": 0,
        "fallback_frequency": 0.0,
        "estimated_byte_misses": sum(int(route["estimated_residency_bytes"]) for route in routes),
        "resident_budget_bytes": 0,
        "max_resident_bytes": 0,
        "final_resident_expert_count": 0,
        "preload_count": 0,
        "load_count": 0,
        "eviction_count": 0,
        "residency_action_count": 0,
        "churn": 0.0,
        "preloaded_experts": [],
        "preload_rejected_experts": [],
        "event_samples": [],
        "rejected_policy_reasons": reasons,
        "candidate_for_live_spike": False,
        "decision": "recommend_no_live_actuator",
    }


def attach_policy_decision(result: JSONDict, observe: JSONDict) -> JSONDict:
    policy_id = str(result["policy_id"])
    reasons: list[str] = []
    if result["fallback_frequency"] > MAX_FALLBACK_FREQUENCY:
        reasons.append("fallback_frequency_above_threshold")
    if result["churn"] > MAX_CHURN_PER_ROUTE:
        reasons.append("churn_above_threshold")

    if policy_id == "keep_hot":
        if result["warm_hit_rate"] <= observe["warm_hit_rate"]:
            reasons.append("warm_hit_rate_not_better_than_observe_only")
        if result["miss_rate"] >= observe["miss_rate"]:
            reasons.append("resident_budget_exceeded")
    elif policy_id == "preload_shortlist":
        if result["preload_rejected_experts"]:
            reasons.append("shortlist_bytes_exceed_budget")
        if result["miss_rate"] >= observe["miss_rate"]:
            reasons.append("preload_misses_exceed_observe_only")
    elif policy_id == "evict_cold":
        if result["miss_rate"] >= observe["miss_rate"]:
            reasons.append("eviction_causes_repeat_misses")
        if result["fallback_frequency"] > 0:
            reasons.append("dense_fallback_unavailable")

    result["rejected_policy_reasons"] = sorted(set(reasons))
    result["candidate_for_live_spike"] = (
        policy_id in {"keep_hot", "preload_shortlist", "evict_cold"}
        and not result["rejected_policy_reasons"]
        and result["warm_hit_rate"] > observe["warm_hit_rate"]
        and result["miss_rate"] < observe["miss_rate"]
    )
    result["decision"] = "candidate_replay_policy" if result["candidate_for_live_spike"] else "rejected_or_observational"
    return result


def replay_policy_results(routes: list[JSONDict], resident_budget_bytes: int) -> list[JSONDict]:
    observe = observe_only_result(routes)
    results: list[JSONDict] = [observe]
    keep_hot = simulate_residency_policy(
        "keep_hot",
        routes,
        resident_budget_bytes=resident_budget_bytes,
    )
    results.append(attach_policy_decision(keep_hot, observe))
    preload_shortlist = simulate_residency_policy(
        "preload_shortlist",
        routes,
        resident_budget_bytes=resident_budget_bytes,
        preload_keys=rank_preload_shortlist(routes),
    )
    results.append(attach_policy_decision(preload_shortlist, observe))
    evict_cold = simulate_residency_policy(
        "evict_cold",
        routes,
        resident_budget_bytes=resident_budget_bytes,
        evict_cold_after_routes=1,
    )
    results.append(attach_policy_decision(evict_cold, observe))
    results.append(fallback_dense_result(routes))
    results.append(no_live_actuator_result(routes, results))
    return results


def policy_candidate_diagnostics(policy_results: list[JSONDict], reuse_distance: JSONDict) -> JSONDict:
    candidate_policy_ids = [
        str(result["policy_id"])
        for result in policy_results
        if result.get("candidate_for_live_spike") is True
    ]
    blockers: list[JSONDict] = []
    rejected_by_policy = {
        str(result["policy_id"]): result.get("rejected_policy_reasons", [])
        for result in policy_results
        if result.get("candidate_for_live_spike") is not True
        and result.get("policy_id") not in {"observe_only", "no_live_actuator"}
    }
    if not candidate_policy_ids:
        blockers.append(
            {
                "id": "no_replay_policy_candidate",
                "description": "No replayed residency policy passed the candidate gate.",
            }
        )
    if int(reuse_distance.get("observations", 0) or 0) == 0:
        blockers.append(
            {
                "id": "no_reuse_distance_observations",
                "description": "Trace has no repeated routed expert observations, so warm-hit policy benefit is hard to prove.",
            }
        )
    return {
        "candidate_ready": bool(candidate_policy_ids),
        "candidate_policy_ids": candidate_policy_ids,
        "blockers": blockers,
        "rejected_policy_reasons_by_policy": rejected_by_policy,
        "reuse_distance": reuse_distance,
    }


def validate_replay_inputs(policy_summary: JSONDict, join_summary: JSONDict, route_errors: list[str]) -> list[str]:
    errors: list[str] = []
    if not policy_summary.get("valid"):
        errors.extend(f"policy contract: {error}" for error in policy_summary.get("errors", []))
    if not join_summary.get("valid"):
        errors.extend(f"trace inventory join: {error}" for error in join_summary.get("errors", []))
    errors.extend(f"route reconstruction: {error}" for error in route_errors)
    return errors


def build_replay_summary(
    trace_path: Path,
    inventory_path: Path,
    policies_path: Path,
    *,
    resident_budget_bytes: int | None = None,
    resident_budget_fraction: float = DEFAULT_RESIDENT_BUDGET_FRACTION,
) -> JSONDict:
    policy_plan = plan_baseline_replay_policies.load_policies(policies_path)
    policy_summary = plan_baseline_replay_policies.build_summary(policy_plan, policies_path)
    join_summary = plan_trace_inventory_replay.build_join_summary(trace_path, inventory_path)
    pairing_summary = plan_real_model_trace_inventory_pairing.build_pairing_summary(trace_path, inventory_path)
    joined_routes, route_errors = build_joined_routes(trace_path, inventory_path)
    errors = validate_replay_inputs(policy_summary, join_summary, route_errors)
    if not pairing_summary.get("valid"):
        errors.extend(f"trace/inventory pairing: {error}" for error in pairing_summary.get("errors", []))

    budget = resolve_resident_budget(
        join_summary,
        resident_budget_bytes=resident_budget_bytes,
        resident_budget_fraction=resident_budget_fraction,
    )
    policy_results: list[JSONDict] = []
    if not errors and joined_routes:
        policy_results = replay_policy_results(
            joined_routes,
            int(budget["resident_budget_bytes"]),
        )
    elif not joined_routes:
        errors.append("trace inventory join produced no replayable routes")

    reuse_distance = join_summary.get("reuse_distance", {})
    candidate_diagnostics = policy_candidate_diagnostics(policy_results, reuse_distance)
    candidate_policy_ids = candidate_diagnostics["candidate_policy_ids"]
    real_model_pair_ready = pairing_summary.get("real_model_pair_ready") is True
    missing_for_phase_3_completion = [
        "dense_fallback_output_artifact_for_quality_bounds",
        "live_residency_observation_and_control",
        "cleanup_restore_proof",
    ]
    if not candidate_policy_ids:
        missing_for_phase_3_completion.insert(0, "no_replay_policy_candidate")
    if not real_model_pair_ready:
        missing_for_phase_3_completion.insert(1, "real_model_trace_inventory_pairing_ready")
    return {
        "mode": "baseline_policy_replay_plan",
        "trace_path": str(trace_path),
        "inventory_path": str(inventory_path),
        "policies_path": str(policies_path),
        "valid": not errors,
        "errors": errors,
        "budget": budget,
        "policy_ids": [policy.get("id") for policy in policy_plan.get("policies", []) if isinstance(policy, dict)],
        "route_count": join_summary.get("route_count", 0),
        "joined_route_count": join_summary.get("joined_route_count", 0),
        "unique_expert_count": join_summary.get("unique_expert_count", 0),
        "unique_estimated_residency_bytes": join_summary.get("unique_estimated_residency_bytes", 0),
        "route_estimated_residency_bytes": join_summary.get("route_estimated_residency_bytes", 0),
        "reuse_distance": reuse_distance,
        "policy_results": policy_results,
        "candidate_policy_ids": candidate_policy_ids,
        "policy_candidate_diagnostics": candidate_diagnostics,
        "phase_3_gate": {
            "policy_replay_metrics_ready": not errors and bool(policy_results),
            "policy_candidate_ready": bool(candidate_policy_ids),
            "real_model_pair_ready": real_model_pair_ready,
            "fixture_only_pair": pairing_summary.get("fixture_only_pair"),
            "dense_fallback_comparison_ready": False,
            "live_actuator_available": False,
            "ready_for_live_spike": False,
            "missing_for_phase_3_completion": missing_for_phase_3_completion,
        },
        "safety_contract": [
            "planner validates local artifacts only",
            "planner does not load tensor values",
            "planner does not mutate runtime residency",
            "planner does not write packed expert stores",
            "planner does not launch model servers",
            "planner does not send prompt traffic",
            "planner does not claim live expert paging",
        ],
    }


def print_human_summary(summary: JSONDict) -> None:
    print("MoE Run Anyway baseline policy replay")
    print(f"Trace: {summary['trace_path']}")
    print(f"Inventory: {summary['inventory_path']}")
    print(f"Policies: {summary['policies_path']}")
    print(f"Valid: {summary['valid']}")
    if summary["errors"]:
        print("Errors:")
        for error in summary["errors"]:
            print(f"  - {error}")
        return
    print(f"Joined routes: {summary['joined_route_count']}")
    print(f"Unique experts: {summary['unique_expert_count']}")
    print(f"Resident budget bytes: {summary['budget']['resident_budget_bytes']}")
    print("Policy results:")
    for result in summary["policy_results"]:
        reasons = ", ".join(result["rejected_policy_reasons"]) or "none"
        print(
            "  - "
            f"{result['policy_id']}: miss_rate={result['miss_rate']} "
            f"warm_hit_rate={result['warm_hit_rate']} churn={result['churn']} "
            f"fallback_frequency={result['fallback_frequency']} "
            f"reasons={reasons}"
        )
    print(f"Policy candidate ready: {summary['phase_3_gate']['policy_candidate_ready']}")
    print(f"Ready for live spike: {summary['phase_3_gate']['ready_for_live_spike']}")
    print("Still missing for Phase 3 completion:")
    for item in summary["phase_3_gate"]["missing_for_phase_3_completion"]:
        print(f"  - {item}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_path", nargs="?", type=Path, default=DEFAULT_TRACE_PATH)
    parser.add_argument("inventory_path", nargs="?", type=Path, default=DEFAULT_INVENTORY_PATH)
    parser.add_argument("policies_path", nargs="?", type=Path, default=DEFAULT_POLICIES_PATH)
    parser.add_argument("--resident-budget-bytes", type=int, default=None)
    parser.add_argument("--resident-budget-fraction", type=float, default=DEFAULT_RESIDENT_BUDGET_FRACTION)
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    return parser


def plan_paths(
    trace_path: Path,
    inventory_path: Path,
    policies_path: Path,
    *,
    resident_budget_bytes: int | None = None,
    resident_budget_fraction: float = DEFAULT_RESIDENT_BUDGET_FRACTION,
) -> tuple[int, JSONDict | None, str | None]:
    if resident_budget_bytes is not None and resident_budget_bytes < 0:
        return 2, None, "resident budget bytes must be >= 0"
    if resident_budget_fraction < 0:
        return 2, None, "resident budget fraction must be >= 0"
    try:
        summary = build_replay_summary(
            trace_path,
            inventory_path,
            policies_path,
            resident_budget_bytes=resident_budget_bytes,
            resident_budget_fraction=resident_budget_fraction,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not build baseline policy replay plan: {exc}"
    return (0 if summary["valid"] else 2), summary, None


def main_from_test_paths(trace_path: Path, inventory_path: Path, policies_path: Path) -> int:
    status, _, _ = plan_paths(trace_path, inventory_path, policies_path)
    return status


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary, error_message = plan_paths(
        args.trace_path,
        args.inventory_path,
        args.policies_path,
        resident_budget_bytes=args.resident_budget_bytes,
        resident_budget_fraction=args.resident_budget_fraction,
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
