#!/usr/bin/env python3
"""Minimal memory-aware MoE runtime simulator.

This MVP treats memory-aware MoE as a runtime orchestration problem:

- a fixed core stays resident
- experts are loaded on demand
- unused experts can be evicted
- freed memory can increase effective KV-cache context capacity

The simulator writes inspectable run artifacts so policy decisions can be
reviewed after the fact.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from moe_shared_contract import attach_shared_contract, build_shared_contract, contract_shell


JSONDict = dict[str, Any]


@dataclass(frozen=True)
class ExpertSpec:
    expert_id: str
    memory_mb: int
    load_ms: float


@dataclass(frozen=True)
class MemoryConfig:
    total_budget_mb: int
    core_memory_mb: int
    kv_cache_mb_per_token: float
    hard_context_limit_tokens: int
    top_k: int
    bootstrap_resident_experts: tuple[str, ...]


@dataclass(frozen=True)
class LatencyModel:
    base_ms: float
    ms_per_token: float
    ms_per_selected_expert: float
    eviction_ms: float


@dataclass(frozen=True)
class QualityModel:
    expert_weight: float
    context_weight: float


@dataclass(frozen=True)
class RequestCase:
    request_id: str
    prompt_tokens: int
    router_scores: dict[str, float]
    quality_scores: dict[str, float]
    note: str | None
    prompt_family: str | None = None


@dataclass(frozen=True)
class Scenario:
    scenario_name: str
    origin: str
    why: str
    memory: MemoryConfig
    latency: LatencyModel
    quality: QualityModel
    experts: dict[str, ExpertSpec]
    requests: list[RequestCase]


@dataclass(frozen=True)
class RoutingPolicyConfig:
    name: str
    resident_bonus: float = 0.0
    recency_bonus: float = 0.0
    cold_load_penalty: float = 0.0


@dataclass(frozen=True)
class ResidencyPolicyConfig:
    name: str
    proactive_context_eviction: bool = False


@dataclass(frozen=True)
class PolicyProfile:
    name: str
    routing: RoutingPolicyConfig
    residency: ResidencyPolicyConfig


@dataclass
class ResidentExpertState:
    expert_id: str
    loaded_at_step: int
    last_used_step: int
    load_count: int = 1


@dataclass(frozen=True)
class RouteEntry:
    expert_id: str
    router_score: float
    adjusted_score: float
    resident: bool
    age_steps: int | None
    resident_bonus: float
    recency_bonus: float
    cold_penalty: float


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: JSONDict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def round3(value: float) -> float:
    return round(value, 3)


def default_profiles() -> dict[str, PolicyProfile]:
    return {
        "baseline": PolicyProfile(
            name="baseline",
            routing=RoutingPolicyConfig(name="top_k"),
            residency=ResidencyPolicyConfig(name="lru"),
        ),
        "reuse_bias": PolicyProfile(
            name="reuse_bias",
            routing=RoutingPolicyConfig(
                name="reuse_bias",
                resident_bonus=0.02,
                recency_bonus=0.05,
                cold_load_penalty=0.04,
            ),
            residency=ResidencyPolicyConfig(name="lru"),
        ),
        "context_reserve": PolicyProfile(
            name="context_reserve",
            routing=RoutingPolicyConfig(
                name="reuse_bias",
                resident_bonus=0.02,
                recency_bonus=0.05,
                cold_load_penalty=0.04,
            ),
            residency=ResidencyPolicyConfig(
                name="lru_context_reserve",
                proactive_context_eviction=True,
            ),
        ),
    }


def validate_score_map(expert_ids: set[str], raw: Any, field_name: str) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise ValueError(f"{field_name} must be an object keyed by expert id")
    values: dict[str, float] = {}
    missing = expert_ids.difference(raw.keys())
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{field_name} is missing scores for experts: {missing_text}")
    extra = set(raw.keys()).difference(expert_ids)
    if extra:
        extra_text = ", ".join(sorted(extra))
        raise ValueError(f"{field_name} contains unknown experts: {extra_text}")
    for expert_id, value in raw.items():
        if not isinstance(value, (int, float)):
            raise ValueError(f"{field_name}.{expert_id} must be numeric")
        values[expert_id] = float(value)
    return values


def load_scenario(path: Path) -> Scenario:
    raw = json.loads(path.read_text(encoding="utf-8"))
    for required in ["scenario_name", "origin", "why", "memory", "latency", "quality", "experts", "requests"]:
        if required not in raw:
            raise ValueError(f"Scenario is missing required field: {required}")

    experts_list = raw["experts"]
    if not isinstance(experts_list, list) or not experts_list:
        raise ValueError("experts must be a non-empty list")
    experts: dict[str, ExpertSpec] = {}
    for item in experts_list:
        if not isinstance(item, dict):
            raise ValueError("Each expert entry must be an object")
        expert_id = item.get("expert_id")
        memory_mb = item.get("memory_mb")
        load_ms = item.get("load_ms")
        if not isinstance(expert_id, str) or not expert_id:
            raise ValueError("Each expert must include a non-empty expert_id")
        if expert_id in experts:
            raise ValueError(f"Duplicate expert_id: {expert_id}")
        if not isinstance(memory_mb, int) or memory_mb <= 0:
            raise ValueError(f"expert {expert_id} must include positive integer memory_mb")
        if not isinstance(load_ms, (int, float)) or load_ms < 0:
            raise ValueError(f"expert {expert_id} must include non-negative load_ms")
        experts[expert_id] = ExpertSpec(
            expert_id=expert_id,
            memory_mb=memory_mb,
            load_ms=float(load_ms),
        )

    memory_raw = raw["memory"]
    bootstrap = tuple(memory_raw.get("bootstrap_resident_experts", []))
    for expert_id in bootstrap:
        if expert_id not in experts:
            raise ValueError(f"Unknown bootstrap expert: {expert_id}")
    memory = MemoryConfig(
        total_budget_mb=int(memory_raw["total_budget_mb"]),
        core_memory_mb=int(memory_raw["core_memory_mb"]),
        kv_cache_mb_per_token=float(memory_raw["kv_cache_mb_per_token"]),
        hard_context_limit_tokens=int(memory_raw["hard_context_limit_tokens"]),
        top_k=int(memory_raw["top_k"]),
        bootstrap_resident_experts=bootstrap,
    )
    if memory.total_budget_mb <= memory.core_memory_mb:
        raise ValueError("total_budget_mb must be larger than core_memory_mb")
    if memory.kv_cache_mb_per_token <= 0:
        raise ValueError("kv_cache_mb_per_token must be positive")
    if memory.top_k <= 0:
        raise ValueError("top_k must be positive")

    latency_raw = raw["latency"]
    latency = LatencyModel(
        base_ms=float(latency_raw["base_ms"]),
        ms_per_token=float(latency_raw["ms_per_token"]),
        ms_per_selected_expert=float(latency_raw["ms_per_selected_expert"]),
        eviction_ms=float(latency_raw["eviction_ms"]),
    )

    quality_raw = raw["quality"]
    quality = QualityModel(
        expert_weight=float(quality_raw["expert_weight"]),
        context_weight=float(quality_raw["context_weight"]),
    )

    expert_ids = set(experts)
    requests: list[RequestCase] = []
    for item in raw["requests"]:
        if not isinstance(item, dict):
            raise ValueError("Each request must be an object")
        request_id = item.get("request_id")
        prompt_tokens = item.get("prompt_tokens")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("Each request must include a non-empty request_id")
        if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
            raise ValueError(f"request {request_id} must include positive integer prompt_tokens")
        requests.append(
            RequestCase(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                router_scores=validate_score_map(expert_ids, item.get("router_scores"), f"{request_id}.router_scores"),
                quality_scores=validate_score_map(expert_ids, item.get("quality_scores"), f"{request_id}.quality_scores"),
                note=str(item["note"]) if "note" in item and item["note"] is not None else None,
                prompt_family=str(item["prompt_family"]) if item.get("prompt_family") else None,
            )
        )

    return Scenario(
        scenario_name=str(raw["scenario_name"]),
        origin=str(raw["origin"]),
        why=str(raw["why"]),
        memory=memory,
        latency=latency,
        quality=quality,
        experts=experts,
        requests=requests,
    )


class RuntimeState:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.resident: dict[str, ResidentExpertState] = {}
        for expert_id in scenario.memory.bootstrap_resident_experts:
            self.resident[expert_id] = ResidentExpertState(
                expert_id=expert_id,
                loaded_at_step=0,
                last_used_step=0,
                load_count=1,
            )

    def resident_ids(self) -> list[str]:
        return sorted(self.resident)

    def resident_memory_mb(self) -> int:
        return sum(self.scenario.experts[expert_id].memory_mb for expert_id in self.resident)

    def free_budget_without_kv_mb(self) -> int:
        return self.scenario.memory.total_budget_mb - self.scenario.memory.core_memory_mb - self.resident_memory_mb()

    def context_capacity_tokens(self) -> int:
        available_mb = max(self.free_budget_without_kv_mb(), 0)
        by_memory = int(available_mb / self.scenario.memory.kv_cache_mb_per_token)
        return max(0, min(by_memory, self.scenario.memory.hard_context_limit_tokens))

    def select_lru_victim(self, protected: set[str]) -> str | None:
        candidates = [item for item in self.resident.values() if item.expert_id not in protected]
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item.last_used_step, item.loaded_at_step, item.expert_id))
        return candidates[0].expert_id

    def evict(self, expert_id: str) -> None:
        del self.resident[expert_id]

    def load(self, expert_id: str, step_index: int) -> None:
        current = self.resident.get(expert_id)
        if current is None:
            self.resident[expert_id] = ResidentExpertState(
                expert_id=expert_id,
                loaded_at_step=step_index,
                last_used_step=step_index,
                load_count=1,
            )
            return
        current.last_used_step = step_index
        current.load_count += 1

    def mark_used(self, expert_id: str, step_index: int) -> None:
        self.resident[expert_id].last_used_step = step_index


def rank_experts(
    scenario: Scenario,
    state: RuntimeState,
    request: RequestCase,
    profile: PolicyProfile,
    step_index: int,
) -> list[RouteEntry]:
    entries: list[RouteEntry] = []
    for expert_id in sorted(scenario.experts):
        router_score = request.router_scores[expert_id]
        resident = expert_id in state.resident
        age_steps: int | None = None
        resident_bonus = 0.0
        recency_bonus = 0.0
        cold_penalty = 0.0
        if resident:
            resident_bonus = profile.routing.resident_bonus
            age_steps = max(1, step_index - state.resident[expert_id].last_used_step)
            recency_bonus = profile.routing.recency_bonus / age_steps
        else:
            cold_penalty = profile.routing.cold_load_penalty
        adjusted_score = router_score + resident_bonus + recency_bonus - cold_penalty
        entries.append(
            RouteEntry(
                expert_id=expert_id,
                router_score=round3(router_score),
                adjusted_score=round3(adjusted_score),
                resident=resident,
                age_steps=age_steps,
                resident_bonus=round3(resident_bonus),
                recency_bonus=round3(recency_bonus),
                cold_penalty=round3(cold_penalty),
            )
        )
    entries.sort(key=lambda item: (-item.adjusted_score, -item.router_score, item.expert_id))
    return entries


def ensure_selected_experts(
    scenario: Scenario,
    state: RuntimeState,
    selected_experts: list[str],
    step_index: int,
) -> tuple[list[JSONDict], list[JSONDict], list[JSONDict], list[str]]:
    loads: list[JSONDict] = []
    evictions: list[JSONDict] = []
    dropped: list[JSONDict] = []
    active_selected: list[str] = []
    protected: set[str] = set(selected_experts)

    for expert_id in selected_experts:
        if expert_id in state.resident:
            active_selected.append(expert_id)
            continue

        expert = scenario.experts[expert_id]
        while expert.memory_mb > state.free_budget_without_kv_mb():
            victim = state.select_lru_victim(protected)
            if victim is None:
                break
            victim_memory = scenario.experts[victim].memory_mb
            evictions.append(
                {
                    "expert_id": victim,
                    "reason": "free_budget_for_selected_expert",
                    "memory_freed_mb": victim_memory,
                }
            )
            state.evict(victim)

        if expert.memory_mb > state.free_budget_without_kv_mb():
            dropped.append(
                {
                    "expert_id": expert_id,
                    "reason": "insufficient_budget_even_after_eviction",
                }
            )
            protected.discard(expert_id)
            continue

        state.load(expert_id, step_index)
        active_selected.append(expert_id)
        loads.append(
            {
                "expert_id": expert_id,
                "reason": "selected_by_router",
                "memory_mb": expert.memory_mb,
                "load_ms": round3(expert.load_ms),
            }
        )

    return loads, evictions, dropped, active_selected


def reserve_context_capacity(
    scenario: Scenario,
    state: RuntimeState,
    protected: set[str],
    requested_tokens: int,
) -> list[JSONDict]:
    evictions: list[JSONDict] = []
    while state.context_capacity_tokens() < requested_tokens:
        victim = state.select_lru_victim(protected)
        if victim is None:
            break
        victim_memory = scenario.experts[victim].memory_mb
        evictions.append(
            {
                "expert_id": victim,
                "reason": "free_budget_for_context_capacity",
                "memory_freed_mb": victim_memory,
            }
        )
        state.evict(victim)
    return evictions


def compute_quality_metrics(
    scenario: Scenario,
    request: RequestCase,
    selected_experts: list[str],
    admitted_tokens: int,
) -> JSONDict:
    top_k = scenario.memory.top_k
    selected_scores = sorted((request.quality_scores[expert_id] for expert_id in selected_experts), reverse=True)
    while len(selected_scores) < top_k:
        selected_scores.append(0.0)
    selected_expert_quality = safe_div(sum(selected_scores[:top_k]), top_k)

    oracle_scores = sorted(request.quality_scores.values(), reverse=True)[:top_k]
    oracle_expert_quality = safe_div(sum(oracle_scores), top_k)

    context_retention = safe_div(admitted_tokens, request.prompt_tokens)
    quality_proxy = (
        scenario.quality.expert_weight * selected_expert_quality
        + scenario.quality.context_weight * context_retention
    )
    oracle_quality = (
        scenario.quality.expert_weight * oracle_expert_quality
        + scenario.quality.context_weight * 1.0
    )
    return {
        "selected_expert_quality": round3(selected_expert_quality),
        "oracle_expert_quality": round3(oracle_expert_quality),
        "context_retention": round3(context_retention),
        "quality_proxy": round3(quality_proxy),
        "oracle_quality": round3(oracle_quality),
        "quality_ratio": round3(safe_div(quality_proxy, oracle_quality)),
    }


def simulate_request(
    scenario: Scenario,
    state: RuntimeState,
    profile: PolicyProfile,
    request: RequestCase,
    step_index: int,
) -> JSONDict:
    resident_before = state.resident_ids()
    resident_before_set = set(resident_before)
    ranking = rank_experts(scenario, state, request, profile, step_index)
    requested_selected = [item.expert_id for item in ranking[: scenario.memory.top_k]]
    warm_before = set(resident_before)

    loads, load_evictions, dropped, active_selected = ensure_selected_experts(
        scenario=scenario,
        state=state,
        selected_experts=requested_selected,
        step_index=step_index,
    )
    context_evictions: list[JSONDict] = []
    if profile.residency.proactive_context_eviction:
        context_evictions = reserve_context_capacity(
            scenario=scenario,
            state=state,
            protected=set(active_selected),
            requested_tokens=request.prompt_tokens,
        )

    for expert_id in active_selected:
        state.mark_used(expert_id, step_index)

    admitted_tokens = min(request.prompt_tokens, state.context_capacity_tokens())
    truncated_tokens = max(0, request.prompt_tokens - admitted_tokens)
    resident_memory_mb = state.resident_memory_mb()
    kv_cache_mb = admitted_tokens * scenario.memory.kv_cache_mb_per_token
    peak_memory_mb = scenario.memory.core_memory_mb + resident_memory_mb + kv_cache_mb
    quality = compute_quality_metrics(scenario, request, active_selected, admitted_tokens)

    load_ms = sum(item["load_ms"] for item in loads)
    eviction_count = len(load_evictions) + len(context_evictions)
    compute_ms = (
        scenario.latency.base_ms
        + admitted_tokens * scenario.latency.ms_per_token
        + len(active_selected) * scenario.latency.ms_per_selected_expert
    )
    eviction_ms = eviction_count * scenario.latency.eviction_ms
    total_latency_ms = compute_ms + load_ms + eviction_ms
    resident_after = state.resident_ids()
    resident_after_set = set(resident_after)
    warm_selected_count = sum(1 for expert_id in active_selected if expert_id in warm_before)
    candidate_set_size = len(requested_selected)
    fallback_used = bool(dropped)
    changed_residents = resident_before_set.symmetric_difference(resident_after_set)

    return attach_shared_contract(
        {
        "timestamp": now_iso(),
        "event_type": "request_result",
        "profile": profile.name,
        "step": step_index,
        "request_id": request.request_id,
        "prompt_family": request.prompt_family,
        "note": request.note,
        "routing_policy": asdict(profile.routing),
        "residency_policy": asdict(profile.residency),
        "resident_before": resident_before,
        "resident_after": resident_after,
        "router_ranking": [asdict(item) for item in ranking],
        "selected_experts_requested": requested_selected,
        "selected_experts_active": active_selected,
        "dropped_selected_experts": dropped,
        "warm_selected_count": warm_selected_count,
        "loads": loads,
        "evictions": load_evictions + context_evictions,
        "memory": {
            "budget_mb": scenario.memory.total_budget_mb,
            "core_memory_mb": scenario.memory.core_memory_mb,
            "resident_expert_memory_mb": resident_memory_mb,
            "context_capacity_tokens": state.context_capacity_tokens(),
            "requested_tokens": request.prompt_tokens,
            "admitted_tokens": admitted_tokens,
            "truncated_tokens": truncated_tokens,
            "active_kv_cache_mb": round3(kv_cache_mb),
            "peak_used_mb": round3(peak_memory_mb),
        },
        "latency_ms": {
            "load_ms": round3(load_ms),
            "eviction_ms": round3(eviction_ms),
            "compute_ms": round3(compute_ms),
            "total_ms": round3(total_latency_ms),
        },
        "quality": quality,
        },
        build_shared_contract(
            probe_tier="simulator",
            backend_family="logical_residency_simulator",
            prompt_family=request.prompt_family,
            policy_name=profile.name,
            baseline_kind="budgeted_simulation",
            candidate_set_size=candidate_set_size,
            fallback_used=fallback_used,
            warm_hit_rate=safe_div(warm_selected_count, len(active_selected)),
            miss_rate=safe_div(len(dropped), candidate_set_size),
            churn=safe_div(len(changed_residents), max(1, len(resident_before_set | resident_after_set))),
            context_retention=quality["context_retention"],
        ),
    )


def summarize_profile(
    scenario: Scenario,
    profile: PolicyProfile,
    outcomes: list[JSONDict],
) -> JSONDict:
    latencies = [float(item["latency_ms"]["total_ms"]) for item in outcomes]
    load_latencies = [float(item["latency_ms"]["load_ms"]) for item in outcomes]
    quality_scores = [float(item["quality"]["quality_proxy"]) for item in outcomes]
    quality_ratios = [float(item["quality"]["quality_ratio"]) for item in outcomes]
    context_retention = [float(item["quality"]["context_retention"]) for item in outcomes]
    truncated_requests = sum(1 for item in outcomes if int(item["memory"]["truncated_tokens"]) > 0)
    load_count = sum(len(item["loads"]) for item in outcomes)
    eviction_count = sum(len(item["evictions"]) for item in outcomes)
    selected_count = sum(len(item["selected_experts_active"]) for item in outcomes)
    warm_selected_count = sum(int(item["warm_selected_count"]) for item in outcomes)
    miss_rate_values = [
        float(item["shared_contract"]["miss_rate"])
        for item in outcomes
        if item.get("shared_contract", {}).get("miss_rate") is not None
    ]
    churn_values = [
        float(item["shared_contract"]["churn"])
        for item in outcomes
        if item.get("shared_contract", {}).get("churn") is not None
    ]
    peak_memory_mb = max(float(item["memory"]["peak_used_mb"]) for item in outcomes) if outcomes else 0.0
    peak_resident_experts = max(len(item["resident_after"]) for item in outcomes) if outcomes else 0

    return {
        "profile": profile.name,
        "shared_contract_rollup": build_shared_contract(
            probe_tier="simulator",
            backend_family="logical_residency_simulator",
            policy_name=profile.name,
            baseline_kind="budgeted_simulation",
            warm_hit_rate=safe_div(warm_selected_count, selected_count),
            miss_rate=mean(miss_rate_values),
            churn=mean(churn_values),
            context_retention=mean(context_retention),
        ),
        "routing_policy": asdict(profile.routing),
        "residency_policy": asdict(profile.residency),
        "request_count": len(outcomes),
        "mean_latency_ms": round3(mean(latencies)),
        "total_latency_ms": round3(sum(latencies)),
        "mean_load_latency_ms": round3(mean(load_latencies)),
        "mean_quality_proxy": round3(mean(quality_scores)),
        "mean_quality_ratio": round3(mean(quality_ratios)),
        "mean_context_retention": round3(mean(context_retention)),
        "truncated_request_count": truncated_requests,
        "load_count": load_count,
        "eviction_count": eviction_count,
        "warm_selection_rate": round3(safe_div(warm_selected_count, selected_count)),
        "peak_memory_mb": round3(peak_memory_mb),
        "peak_resident_experts": peak_resident_experts,
        "final_resident_experts": outcomes[-1]["resident_after"] if outcomes else [],
        "interpretation": {
            "primary_objective": "maximize quality proxy while staying within the memory budget",
            "secondary_objective": "minimize latency once quality stays acceptable",
            "budget_mb": scenario.memory.total_budget_mb,
        },
    }


def run_profiles(
    scenario: Scenario,
    profiles: list[PolicyProfile],
    output_dir: Path,
) -> JSONDict:
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{scenario.scenario_name}"
    run_dir = output_dir / run_id
    ensure_dir(run_dir)
    events_path = run_dir / "events.jsonl"
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.json"

    manifest = {
        "created_at": now_iso(),
        "run_id": run_id,
        "shared_contract_template": contract_shell(
            probe_tier="simulator",
            backend_family="logical_residency_simulator",
            default_baseline_kind="budgeted_simulation",
        ),
        "scenario": {
            "scenario_name": scenario.scenario_name,
            "origin": scenario.origin,
            "why": scenario.why,
            "memory": asdict(scenario.memory),
            "latency": asdict(scenario.latency),
            "quality": asdict(scenario.quality),
            "experts": [asdict(expert) for expert in scenario.experts.values()],
            "request_count": len(scenario.requests),
        },
        "profiles": [
            {
                "name": profile.name,
                "routing": asdict(profile.routing),
                "residency": asdict(profile.residency),
            }
            for profile in profiles
        ],
    }
    write_json(manifest_path, manifest)

    by_profile: list[JSONDict] = []
    for profile in profiles:
        state = RuntimeState(scenario)
        outcomes: list[JSONDict] = []
        for step_index, request in enumerate(scenario.requests, start=1):
            outcome = simulate_request(scenario, state, profile, request, step_index)
            append_jsonl(events_path, outcome)
            outcomes.append(outcome)
        by_profile.append(
            {
                "profile_summary": summarize_profile(scenario, profile, outcomes),
                "requests": [
                    {
                        "request_id": item["request_id"],
                        "latency_ms": item["latency_ms"]["total_ms"],
                        "quality_ratio": item["quality"]["quality_ratio"],
                        "context_retention": item["quality"]["context_retention"],
                        "selected_experts": item["selected_experts_active"],
                        "loads": len(item["loads"]),
                        "evictions": len(item["evictions"]),
                    }
                    for item in outcomes
                ],
            }
        )

    ranked = sorted(
        by_profile,
        key=lambda item: (
            -item["profile_summary"]["mean_quality_ratio"],
            item["profile_summary"]["mean_latency_ms"],
        ),
    )
    summary = {
        "created_at": now_iso(),
        "run_id": run_id,
        "scenario_name": scenario.scenario_name,
        "shared_contract_template": contract_shell(
            probe_tier="simulator",
            backend_family="logical_residency_simulator",
            default_baseline_kind="budgeted_simulation",
        ),
        "ranking_rule": "higher mean_quality_ratio first, lower mean_latency_ms as tiebreak",
        "profiles": ranked,
    }
    write_json(summary_path, summary)
    return {
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "summary": summary,
    }


def print_plan(scenario: Scenario, profiles: list[PolicyProfile]) -> None:
    print(f"Scenario: {scenario.scenario_name}")
    print(f"Origin: {scenario.origin}")
    print(f"Why: {scenario.why}")
    print()
    print("Memory")
    print(f"  total budget: {scenario.memory.total_budget_mb} MB")
    print(f"  core resident: {scenario.memory.core_memory_mb} MB")
    print(f"  KV per token: {scenario.memory.kv_cache_mb_per_token} MB")
    print(f"  hard context limit: {scenario.memory.hard_context_limit_tokens} tokens")
    print(f"  router top-k: {scenario.memory.top_k}")
    print()
    print("Experts")
    for expert in sorted(scenario.experts.values(), key=lambda item: item.expert_id):
        print(f"  {expert.expert_id}: {expert.memory_mb} MB, load {expert.load_ms} ms")
    print()
    print("Profiles")
    for profile in profiles:
        print(f"  {profile.name}")
        print(f"    routing: {profile.routing.name}")
        print(f"    residency: {profile.residency.name}")
    print()
    print(f"Requests: {len(scenario.requests)}")


def print_run_summary(result: JSONDict) -> None:
    print(f"Run folder: {result['run_dir']}")
    for item in result["summary"]["profiles"]:
        profile = item["profile_summary"]
        print(
            f"{profile['profile']}: "
            f"quality={profile['mean_quality_ratio']:.3f}, "
            f"latency_ms={profile['mean_latency_ms']:.3f}, "
            f"context={profile['mean_context_retention']:.3f}, "
            f"loads={profile['load_count']}, "
            f"evictions={profile['eviction_count']}"
        )


def resolve_profiles(profile_names: list[str] | None) -> list[PolicyProfile]:
    catalog = default_profiles()
    if not profile_names:
        return list(catalog.values())
    resolved: list[PolicyProfile] = []
    for name in profile_names:
        if name not in catalog:
            choices = ", ".join(sorted(catalog))
            raise ValueError(f"Unknown profile {name!r}. Choose from: {choices}")
        resolved.append(catalog[name])
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a minimal memory-aware MoE runtime simulation.")
    default_scenario = Path(__file__).resolve().parent / "data" / "toy_workload.json"
    default_output = Path(__file__).resolve().parent / "runs"
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--scenario", type=Path, default=default_scenario)
    common.add_argument(
        "--profile",
        action="append",
        help="Limit the run to one or more named profiles. Defaults to all built-in profiles.",
    )

    plan_parser = subparsers.add_parser("plan", parents=[common], help="Print scenario and profile details.")
    plan_parser.set_defaults(command="plan")

    run_parser = subparsers.add_parser("run", parents=[common], help="Execute the simulation and write run artifacts.")
    run_parser.add_argument("--output-dir", type=Path, default=default_output)
    run_parser.set_defaults(command="run")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    scenario = load_scenario(args.scenario)
    profiles = resolve_profiles(args.profile)

    if args.command == "plan":
        print_plan(scenario, profiles)
        return

    result = run_profiles(scenario, profiles, args.output_dir)
    print_run_summary(result)


if __name__ == "__main__":
    main()
