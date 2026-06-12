#!/usr/bin/env python3
"""Trace replay, advisor analysis, and controller demo for memory-aware MoE work."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from moe_shared_contract import (
    DEFAULT_WINDOW_SIZE_TOKENS,
    attach_shared_contract,
    build_shared_contract,
    contract_shell,
)


JSONDict = dict[str, Any]


@dataclass(frozen=True)
class ReplayExpertSpec:
    expert_key: str
    layer_id: str
    expert_id: str
    resident_cost: int
    load_ms: float


@dataclass(frozen=True)
class ReplayWindow:
    window_id: str
    prompt_family: str
    prompt_id: str
    window_index: int
    token_start: int
    token_end: int
    token_count: int
    layer_expert_mass: dict[str, dict[str, float]]
    runtime_state: dict[str, float]
    note: str | None = None


@dataclass(frozen=True)
class ReplayDataset:
    trace_name: str
    origin: str
    source_kind: str
    window_size_tokens: int
    top_k: int
    experts: dict[str, ReplayExpertSpec]
    windows: list[ReplayWindow]


@dataclass(frozen=True)
class ReplayPolicyConfig:
    name: str
    shortlist_extra: int = 0
    reuse_bonus: float = 0.0
    recency_weight: float = 0.0
    frequency_weight: float = 0.0
    predictive_mass_weight: float = 0.0
    family_prior_weight: float = 0.0
    cold_load_penalty: float = 0.0
    hysteresis_windows: int = 0
    burst_pool_size: int = 0
    per_layer_budget: bool = True


@dataclass
class ResidentExpertState:
    expert_key: str
    layer_id: str
    loaded_at_window: int
    last_used_window: int
    use_count: int = 0
    sticky_until_window: int = 0
    in_burst_pool: bool = False


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, rows: Iterable[JSONDict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def round3(value: float | int | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 3)


def compose_expert_key(layer_id: str, expert_id: str) -> str:
    return f"{layer_id}:{expert_id}"


def split_expert_key(expert_key: str) -> tuple[str, str]:
    layer_id, expert_id = expert_key.split(":", 1)
    return layer_id, expert_id


def read_jsonl(path: Path) -> list[JSONDict]:
    rows: list[JSONDict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def normalize_layer_expert_mass(raw: Any) -> dict[str, dict[str, float]]:
    normalized: dict[str, dict[str, float]] = {}
    if not isinstance(raw, dict):
        return normalized
    for layer_id, raw_experts in raw.items():
        if not isinstance(raw_experts, dict):
            continue
        experts: dict[str, float] = {}
        for expert_id, raw_value in raw_experts.items():
            try:
                experts[str(expert_id)] = float(raw_value)
            except (TypeError, ValueError):
                continue
        if experts:
            normalized[str(layer_id)] = experts
    return normalized


def flatten_window_mass(window: ReplayWindow) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for layer_id, experts in window.layer_expert_mass.items():
        for expert_id, value in experts.items():
            flattened[compose_expert_key(layer_id, expert_id)] = float(value)
    return flattened


def requested_experts_for_window(window: ReplayWindow, top_k: int) -> dict[str, float]:
    requested: dict[str, float] = {}
    for layer_id, experts in window.layer_expert_mass.items():
        ranked = sorted(experts.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        for expert_id, value in ranked:
            requested[compose_expert_key(layer_id, expert_id)] = float(value)
    return requested


def infer_experts_from_windows(windows: list[dict[str, Any]], default_load_ms: float = 10.0) -> dict[str, ReplayExpertSpec]:
    experts: dict[str, ReplayExpertSpec] = {}
    for item in windows:
        layer_expert_mass = normalize_layer_expert_mass(
            item.get("layer_expert_mass") or item.get("layer_expert_hits") or {}
        )
        for layer_id, raw_experts in layer_expert_mass.items():
            for expert_id in raw_experts:
                expert_key = compose_expert_key(layer_id, expert_id)
                if expert_key in experts:
                    continue
                try:
                    layer_number = int(layer_id)
                except ValueError:
                    layer_number = 0
                load_ms = default_load_ms + max(0, layer_number) * 0.2
                experts[expert_key] = ReplayExpertSpec(
                    expert_key=expert_key,
                    layer_id=layer_id,
                    expert_id=expert_id,
                    resident_cost=1,
                    load_ms=round(float(load_ms), 3),
                )
    return experts


def load_trace_dataset(path: Path) -> ReplayDataset:
    raw = json.loads(path.read_text(encoding="utf-8"))
    windows_raw = raw.get("windows")
    if not isinstance(windows_raw, list) or not windows_raw:
        raise ValueError("Replay trace JSON must include a non-empty windows list")

    experts: dict[str, ReplayExpertSpec] = {}
    for item in raw.get("experts", []):
        layer_id = str(item["layer_id"])
        expert_id = str(item["expert_id"])
        expert_key = compose_expert_key(layer_id, expert_id)
        experts[expert_key] = ReplayExpertSpec(
            expert_key=expert_key,
            layer_id=layer_id,
            expert_id=expert_id,
            resident_cost=int(item.get("resident_cost", 1)),
            load_ms=float(item.get("load_ms", 10.0)),
        )
    if not experts:
        experts = infer_experts_from_windows(windows_raw)

    windows: list[ReplayWindow] = []
    for index, item in enumerate(windows_raw, start=1):
        layer_expert_mass = normalize_layer_expert_mass(item.get("layer_expert_mass"))
        token_count = int(item.get("token_count") or item.get("window_token_count") or raw.get("window_size_tokens", DEFAULT_WINDOW_SIZE_TOKENS))
        token_start = int(item.get("token_start", 0))
        token_end = int(item.get("token_end", token_start + max(0, token_count - 1)))
        runtime_state = {
            str(key): float(value)
            for key, value in (item.get("runtime_state") or {}).items()
            if isinstance(value, (int, float))
        }
        windows.append(
            ReplayWindow(
                window_id=str(item.get("window_id") or f"window-{index:03d}"),
                prompt_family=str(item.get("prompt_family") or "unknown"),
                prompt_id=str(item.get("prompt_id") or item.get("probe_id") or f"prompt-{index:03d}"),
                window_index=int(item.get("window_index") or index),
                token_start=token_start,
                token_end=token_end,
                token_count=token_count,
                layer_expert_mass=layer_expert_mass,
                runtime_state=runtime_state,
                note=str(item["note"]) if item.get("note") else None,
            )
        )

    return ReplayDataset(
        trace_name=str(raw.get("trace_name") or path.stem),
        origin=str(raw.get("origin") or f"Replay trace loaded from {path}"),
        source_kind=str(raw.get("source_kind") or "synthetic_trace"),
        window_size_tokens=int(raw.get("window_size_tokens", DEFAULT_WINDOW_SIZE_TOKENS)),
        top_k=int(raw.get("top_k", 2)),
        experts=experts,
        windows=windows,
    )


def load_forward_probe_dataset(run_dir: Path) -> ReplayDataset:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    windows_raw = read_jsonl(run_dir / "window_summaries.jsonl")
    if not windows_raw:
        raise ValueError(f"No window summaries found in {run_dir}")
    experts = infer_experts_from_windows(windows_raw)
    windows: list[ReplayWindow] = []
    for index, item in enumerate(windows_raw, start=1):
        layer_expert_mass = normalize_layer_expert_mass(item.get("layer_expert_mass"))
        if not layer_expert_mass:
            layer_expert_mass = normalize_layer_expert_mass(item.get("layer_expert_hits"))
        token_count = int(item.get("window_token_count") or manifest.get("config", {}).get("window_size_tokens", DEFAULT_WINDOW_SIZE_TOKENS))
        runtime_state = {}
        windows.append(
            ReplayWindow(
                window_id=f"{run_dir.name}-window-{index:03d}",
                prompt_family=str(item.get("prompt_family") or "unknown"),
                prompt_id=str(item.get("probe_id") or f"prompt-{index:03d}"),
                window_index=int(item.get("window_index") or index),
                token_start=int(item.get("token_start", 0)),
                token_end=int(item.get("token_end", max(0, token_count - 1))),
                token_count=token_count,
                layer_expert_mass=layer_expert_mass,
                runtime_state=runtime_state,
                note="Loaded from forward-hook probe artifacts.",
            )
        )
    label = manifest.get("config", {}).get("label") or run_dir.name
    source_kind = "forward_hook_demo" if "demo" in str(label).lower() else "forward_hook_run"
    return ReplayDataset(
        trace_name=str(label),
        origin=f"Forward-hook run loaded from {run_dir}",
        source_kind=source_kind,
        window_size_tokens=int(manifest.get("config", {}).get("window_size_tokens", DEFAULT_WINDOW_SIZE_TOKENS)),
        top_k=int(manifest.get("config", {}).get("default_top_k", 2)),
        experts=experts,
        windows=windows,
    )


def default_policies() -> dict[str, ReplayPolicyConfig]:
    return {
        "reactive_lru": ReplayPolicyConfig(
            name="reactive_lru",
            recency_weight=0.05,
        ),
        "weighted_lru": ReplayPolicyConfig(
            name="weighted_lru",
            shortlist_extra=0,
            reuse_bonus=0.2,
            recency_weight=0.18,
            frequency_weight=0.07,
            predictive_mass_weight=0.18,
            cold_load_penalty=0.08,
            hysteresis_windows=2,
            burst_pool_size=1,
        ),
        "window_predictive": ReplayPolicyConfig(
            name="window_predictive",
            shortlist_extra=2,
            reuse_bonus=0.22,
            recency_weight=0.2,
            frequency_weight=0.08,
            predictive_mass_weight=0.5,
            family_prior_weight=0.24,
            cold_load_penalty=0.12,
            hysteresis_windows=2,
            burst_pool_size=1,
        ),
    }


class ReplayRuntimeState:
    def __init__(self, dataset: ReplayDataset) -> None:
        self.dataset = dataset
        self.residents: dict[str, ResidentExpertState] = {}
        self.recent_window_mass: deque[dict[str, float]] = deque(maxlen=3)
        self.family_mass_totals: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self.family_window_counts: defaultdict[str, int] = defaultdict(int)

    def resident_keys(self) -> list[str]:
        return sorted(self.residents)

    def resident_keys_by_layer(self, layer_id: str, include_burst: bool = True) -> list[str]:
        return sorted(
            key
            for key, item in self.residents.items()
            if item.layer_id == layer_id and (include_burst or not item.in_burst_pool)
        )

    def update_history(self, window: ReplayWindow, actual_mass: dict[str, float]) -> None:
        self.recent_window_mass.append(actual_mass)
        self.family_window_counts[window.prompt_family] += 1
        family_counter = self.family_mass_totals[window.prompt_family]
        for expert_key, value in actual_mass.items():
            family_counter[expert_key] += value

    def recent_mass_scores(self) -> Counter[str]:
        scores: Counter[str] = Counter()
        history = list(self.recent_window_mass)
        for index, mass_map in enumerate(reversed(history), start=1):
            weight = 1.0 / index
            for expert_key, value in mass_map.items():
                scores[expert_key] += value * weight
        return scores

    def family_prior_scores(self, prompt_family: str) -> Counter[str]:
        count = self.family_window_counts.get(prompt_family, 0)
        if count <= 0:
            return Counter()
        return Counter(
            {
                expert_key: value / count
                for expert_key, value in self.family_mass_totals[prompt_family].items()
            }
        )


def compute_layer_budgets(dataset: ReplayDataset, resident_budget_fraction: float, per_layer_budget: bool) -> dict[str, int]:
    experts_by_layer: defaultdict[str, list[str]] = defaultdict(list)
    for expert_key, expert in dataset.experts.items():
        experts_by_layer[expert.layer_id].append(expert_key)
    if not per_layer_budget:
        total_budget = max(dataset.top_k, int(math.ceil(len(dataset.experts) * resident_budget_fraction)))
        return {"*": total_budget}
    budgets: dict[str, int] = {}
    for layer_id, expert_keys in experts_by_layer.items():
        budgets[layer_id] = max(dataset.top_k, int(math.ceil(len(expert_keys) * resident_budget_fraction)))
    return budgets


def context_retention_for_fraction(resident_budget_fraction: float) -> float:
    return max(0.55, min(1.0, 1.0 - 0.35 * resident_budget_fraction))


def selection_score(
    expert_key: str,
    state: ReplayRuntimeState,
    window_index: int,
    prompt_family: str,
    policy: ReplayPolicyConfig,
    runtime_pressure: float,
) -> float:
    recent_mass = state.recent_mass_scores().get(expert_key, 0.0)
    family_prior = state.family_prior_scores(prompt_family).get(expert_key, 0.0)
    resident = state.residents.get(expert_key)
    recency_score = 0.0
    frequency_score = 0.0
    resident_bonus = 0.0
    cold_penalty = 0.0
    if resident is not None:
        age = max(1, window_index - resident.last_used_window)
        recency_score = policy.recency_weight / age
        frequency_score = policy.frequency_weight * resident.use_count
        resident_bonus = policy.reuse_bonus
    else:
        cold_penalty = policy.cold_load_penalty * runtime_pressure
    return (
        policy.predictive_mass_weight * recent_mass
        + policy.family_prior_weight * family_prior
        + recency_score
        + frequency_score
        + resident_bonus
        - cold_penalty
    )


def eviction_keep_score(
    resident: ResidentExpertState,
    state: ReplayRuntimeState,
    window_index: int,
    policy: ReplayPolicyConfig,
) -> float:
    age = max(1, window_index - resident.last_used_window)
    recent_mass = state.recent_mass_scores().get(resident.expert_key, 0.0)
    return (
        policy.reuse_bonus
        + policy.recency_weight / age
        + policy.frequency_weight * resident.use_count
        + 0.1 * recent_mass
    )


def choose_target_shortlist(
    dataset: ReplayDataset,
    state: ReplayRuntimeState,
    window: ReplayWindow,
    policy: ReplayPolicyConfig,
    budgets: dict[str, int],
) -> set[str]:
    if policy.name == "reactive_lru":
        return set(state.resident_keys())

    runtime_pressure = float(window.runtime_state.get("memory_pressure", 1.0))
    shortlist: set[str] = set()
    experts_by_layer: defaultdict[str, list[str]] = defaultdict(list)
    for expert_key, expert in dataset.experts.items():
        experts_by_layer[expert.layer_id].append(expert_key)

    for layer_id, expert_keys in sorted(experts_by_layer.items()):
        layer_budget = budgets.get(layer_id, budgets.get("*", len(expert_keys)))
        ranked = sorted(
            expert_keys,
            key=lambda expert_key: (
                -selection_score(
                    expert_key=expert_key,
                    state=state,
                    window_index=window.window_index,
                    prompt_family=window.prompt_family,
                    policy=policy,
                    runtime_pressure=runtime_pressure,
                ),
                expert_key,
            ),
        )
        shortlist.update(ranked[: layer_budget + policy.shortlist_extra])
    return shortlist


def candidate_victims(
    state: ReplayRuntimeState,
    layer_id: str,
    protected: set[str],
    window_index: int,
    policy: ReplayPolicyConfig,
) -> list[ResidentExpertState]:
    candidates = [
        item
        for item in state.residents.values()
        if item.layer_id == layer_id and item.expert_key not in protected
    ]
    non_sticky = [item for item in candidates if item.sticky_until_window < window_index]
    if non_sticky:
        candidates = non_sticky
    candidates.sort(
        key=lambda item: (
            item.in_burst_pool,
            eviction_keep_score(item, state, window_index, policy),
            item.last_used_window,
            item.expert_key,
        )
    )
    return candidates


def ensure_resident(
    *,
    dataset: ReplayDataset,
    state: ReplayRuntimeState,
    policy: ReplayPolicyConfig,
    budgets: dict[str, int],
    window_index: int,
    expert_key: str,
    protected: set[str],
    reason: str,
    allow_burst: bool,
) -> tuple[list[JSONDict], list[JSONDict]]:
    if expert_key in state.residents:
        return [], []

    layer_id, _ = split_expert_key(expert_key)
    loads: list[JSONDict] = []
    evictions: list[JSONDict] = []
    base_residents = state.resident_keys_by_layer(layer_id, include_burst=False)
    layer_budget = budgets.get(layer_id, budgets.get("*", len(base_residents) + 1))

    while len(base_residents) >= layer_budget:
        victims = candidate_victims(state, layer_id, protected, window_index, policy)
        if not victims:
            break
        victim = victims[0]
        evictions.append(
            {
                "expert_key": victim.expert_key,
                "reason": "evict_for_load",
                "source_reason": reason,
                "was_burst_pool": victim.in_burst_pool,
            }
        )
        del state.residents[victim.expert_key]
        base_residents = state.resident_keys_by_layer(layer_id, include_burst=False)

    use_burst = False
    if len(base_residents) >= layer_budget:
        burst_in_use = sum(1 for item in state.residents.values() if item.in_burst_pool)
        if allow_burst and burst_in_use < policy.burst_pool_size:
            use_burst = True
        else:
            return loads, evictions

    spec = dataset.experts[expert_key]
    state.residents[expert_key] = ResidentExpertState(
        expert_key=expert_key,
        layer_id=layer_id,
        loaded_at_window=window_index,
        last_used_window=window_index,
        use_count=0,
        sticky_until_window=window_index + policy.hysteresis_windows,
        in_burst_pool=use_burst,
    )
    loads.append(
        {
            "expert_key": expert_key,
            "reason": reason,
            "load_ms": round3(spec.load_ms),
            "burst_pool": use_burst,
        }
    )
    return loads, evictions


def mark_used_residents(state: ReplayRuntimeState, expert_keys: Iterable[str], window_index: int, hysteresis_windows: int) -> None:
    for expert_key in expert_keys:
        resident = state.residents.get(expert_key)
        if resident is None:
            continue
        resident.last_used_window = window_index
        resident.use_count += 1
        resident.sticky_until_window = max(resident.sticky_until_window, window_index + hysteresis_windows)


def release_burst_residents(state: ReplayRuntimeState, protected: set[str]) -> list[JSONDict]:
    evictions: list[JSONDict] = []
    for expert_key in sorted(list(state.residents)):
        resident = state.residents.get(expert_key)
        if resident is None or not resident.in_burst_pool or expert_key in protected:
            continue
        evictions.append(
            {
                "expert_key": expert_key,
                "reason": "burst_pool_reset",
                "source_reason": "post_window_cleanup",
                "was_burst_pool": True,
            }
        )
        del state.residents[expert_key]
    return evictions


def build_run_event(
    *,
    run_kind: str,
    window: ReplayWindow,
    policy: ReplayPolicyConfig,
    resident_budget_fraction: float,
    resident_before: list[str],
    resident_after: list[str],
    candidate_set: set[str],
    requested_experts: dict[str, float],
    active_experts: list[str],
    loads: list[JSONDict],
    evictions: list[JSONDict],
    fallback_used: bool,
    fallback_kind: str | None,
    controller_action: str,
    latency_ms: dict[str, float],
    quality_proxy: float,
    context_retention: float,
    dense_baseline_delta: float | None,
    runtime_state: dict[str, float],
) -> JSONDict:
    resident_before_set = set(resident_before)
    resident_after_set = set(resident_after)
    warm_requested = sum(1 for expert_key in requested_experts if expert_key in resident_before_set)
    miss_count = sum(1 for expert_key in requested_experts if expert_key not in candidate_set)
    churn = safe_div(len(resident_before_set.symmetric_difference(resident_after_set)), max(1, len(resident_before_set | resident_after_set)))
    event = {
        "timestamp": now_iso(),
        "event_type": "replay_window_result",
        "run_kind": run_kind,
        "policy_name": policy.name,
        "resident_budget_fraction": round3(resident_budget_fraction),
        "window": asdict(window),
        "runtime_state": runtime_state,
        "resident_before": resident_before,
        "resident_after": resident_after,
        "candidate_set": sorted(candidate_set),
        "candidate_set_size": len(candidate_set),
        "requested_experts": dict(sorted(requested_experts.items())),
        "active_experts": active_experts,
        "loads": loads,
        "evictions": evictions,
        "fallback_used": fallback_used,
        "fallback_kind": fallback_kind,
        "controller_action": controller_action,
        "latency_ms": latency_ms,
        "quality": {
            "quality_proxy": round3(quality_proxy),
            "context_retention": round3(context_retention),
            "dense_baseline_delta": round3(dense_baseline_delta),
        },
        "audit": {
            "warm_requested_count": warm_requested,
            "requested_count": len(requested_experts),
            "miss_count": miss_count,
            "dense_baseline_delta": round3(dense_baseline_delta),
        },
    }
    return attach_shared_contract(
        event,
        build_shared_contract(
            probe_tier="controller_replay" if run_kind == "controller_demo" else "replay_analysis",
            backend_family="trace_replay",
            prompt_family=window.prompt_family,
            window_size_tokens=window.token_count,
            policy_name=policy.name,
            resident_budget_fraction=resident_budget_fraction,
            baseline_kind=run_kind,
            candidate_set_size=len(candidate_set),
            fallback_used=fallback_used,
            warm_hit_rate=safe_div(warm_requested, len(requested_experts)),
            miss_rate=safe_div(miss_count, len(requested_experts)),
            churn=churn,
            eviction_regret=None,
            context_retention=context_retention,
            dense_baseline_delta=dense_baseline_delta,
        ),
    )


def summarize_run(events: list[JSONDict], policy: ReplayPolicyConfig, run_kind: str, resident_budget_fraction: float) -> JSONDict:
    latencies = [float(item["latency_ms"]["total_ms"]) for item in events]
    load_latencies = [float(item["latency_ms"]["load_ms"]) for item in events]
    quality_values = [float(item["quality"]["quality_proxy"]) for item in events]
    dense_deltas = [
        float(item["quality"]["dense_baseline_delta"])
        for item in events
        if item["quality"].get("dense_baseline_delta") is not None
    ]
    warm_hit_rates = [
        float(item["shared_contract"]["warm_hit_rate"])
        for item in events
        if item["shared_contract"].get("warm_hit_rate") is not None
    ]
    miss_rates = [
        float(item["shared_contract"]["miss_rate"])
        for item in events
        if item["shared_contract"].get("miss_rate") is not None
    ]
    churn_values = [
        float(item["shared_contract"]["churn"])
        for item in events
        if item["shared_contract"].get("churn") is not None
    ]
    eviction_regrets = [
        float(item["shared_contract"]["eviction_regret"])
        for item in events
        if item["shared_contract"].get("eviction_regret") is not None
    ]
    context_values = [
        float(item["quality"]["context_retention"])
        for item in events
        if item["quality"].get("context_retention") is not None
    ]
    fallback_rate = safe_div(sum(1 for item in events if item.get("fallback_used")), len(events))
    dense_fallback_rate = safe_div(
        sum(1 for item in events if item.get("fallback_kind") == "dense_fallback"),
        len(events),
    )
    by_family: defaultdict[str, list[JSONDict]] = defaultdict(list)
    for item in events:
        by_family[item["window"]["prompt_family"]].append(item)
    family_metrics = {
        family_id: {
            "window_count": len(rows),
            "mean_latency_ms": round3(mean([float(row["latency_ms"]["total_ms"]) for row in rows])),
            "mean_quality_proxy": round3(mean([float(row["quality"]["quality_proxy"]) for row in rows])),
            "mean_dense_baseline_delta": round3(
                mean(
                    [
                        float(row["quality"]["dense_baseline_delta"])
                        for row in rows
                        if row["quality"].get("dense_baseline_delta") is not None
                    ]
                )
            ),
            "fallback_rate": round3(safe_div(sum(1 for row in rows if row.get("fallback_used")), len(rows))),
        }
        for family_id, rows in sorted(by_family.items())
    }
    return {
        "policy_name": policy.name,
        "run_kind": run_kind,
        "resident_budget_fraction": round3(resident_budget_fraction),
        "shared_contract_rollup": build_shared_contract(
            probe_tier="controller_replay" if run_kind == "controller_demo" else "replay_analysis",
            backend_family="trace_replay",
            policy_name=policy.name,
            resident_budget_fraction=resident_budget_fraction,
            baseline_kind=run_kind,
            fallback_used=fallback_rate > 0.0,
            warm_hit_rate=mean(warm_hit_rates),
            miss_rate=mean(miss_rates),
            churn=mean(churn_values),
            eviction_regret=mean(eviction_regrets),
            context_retention=mean(context_values),
            dense_baseline_delta=mean(dense_deltas),
        ),
        "window_count": len(events),
        "mean_latency_ms": round3(mean(latencies)),
        "mean_load_latency_ms": round3(mean(load_latencies)),
        "mean_quality_proxy": round3(mean(quality_values)),
        "mean_dense_baseline_delta": round3(mean(dense_deltas)),
        "mean_warm_hit_rate": round3(mean(warm_hit_rates)),
        "mean_miss_rate": round3(mean(miss_rates)),
        "mean_churn": round3(mean(churn_values)),
        "mean_eviction_regret": round3(mean(eviction_regrets)),
        "mean_context_retention": round3(mean(context_values)),
        "fallback_rate": round3(fallback_rate),
        "dense_fallback_rate": round3(dense_fallback_rate),
        "load_count": sum(len(item["loads"]) for item in events),
        "eviction_count": sum(len(item["evictions"]) for item in events),
        "by_family": family_metrics,
    }


def annotate_eviction_regret(events: list[JSONDict]) -> None:
    for index, event in enumerate(events):
        next_requested = events[index + 1]["requested_experts"] if index + 1 < len(events) else {}
        total_next_mass = sum(float(value) for value in next_requested.values())
        regret_mass = 0.0
        for eviction in event.get("evictions", []):
            expert_key = eviction.get("expert_key")
            if expert_key in next_requested:
                regret_mass += float(next_requested[expert_key])
        regret = safe_div(regret_mass, total_next_mass)
        event["audit"]["eviction_regret"] = round3(regret)
        event["shared_contract"]["eviction_regret"] = round3(regret)


def run_policy(
    *,
    dataset: ReplayDataset,
    policy: ReplayPolicyConfig,
    resident_budget_fraction: float,
    run_kind: str,
    dense_quality_by_window: dict[str, float] | None = None,
) -> list[JSONDict]:
    state = ReplayRuntimeState(dataset)
    budgets = compute_layer_budgets(dataset, resident_budget_fraction, policy.per_layer_budget)
    events: list[JSONDict] = []

    for window in dataset.windows:
        actual_mass = flatten_window_mass(window)
        requested = requested_experts_for_window(window, dataset.top_k)
        requested_count = len(requested)
        requested_mass_total = sum(requested.values())
        runtime_pressure = float(window.runtime_state.get("memory_pressure", 1.0))

        resident_before = state.resident_keys()
        protected = set(requested)
        loads: list[JSONDict] = []
        evictions: list[JSONDict] = []
        shortlist = choose_target_shortlist(dataset, state, window, policy, budgets)

        for expert_key in sorted(shortlist):
            preload_loads, preload_evictions = ensure_resident(
                dataset=dataset,
                state=state,
                policy=policy,
                budgets=budgets,
                window_index=window.window_index,
                expert_key=expert_key,
                protected=protected,
                reason="preload_shortlist",
                allow_burst=False,
            )
            loads.extend(preload_loads)
            evictions.extend(preload_evictions)

        candidate_set = set(state.resident_keys())
        planned_candidate_set = set(candidate_set)
        missing = [expert_key for expert_key in requested if expert_key not in planned_candidate_set]
        active_experts: list[str] = []
        fallback_used = False
        fallback_kind: str | None = None
        controller_action = "keep_preload_evict"
        effective_fraction = resident_budget_fraction

        if run_kind == "dense_baseline":
            active_experts = sorted(requested)
            candidate_set = set(dataset.experts)
            planned_candidate_set = set(dataset.experts)
            effective_fraction = 1.0
        elif run_kind == "controller_demo":
            dense_fallback = bool(missing) and (
                len(missing) > max(1, requested_count // 2) or runtime_pressure >= 1.4
            )
            if dense_fallback:
                active_experts = sorted(requested)
                fallback_used = True
                fallback_kind = "dense_fallback"
                controller_action = "dense_fallback"
                effective_fraction = 1.0
            else:
                active_experts = sorted(expert_key for expert_key in requested if expert_key in candidate_set)
                fallback_used = bool(missing)
                fallback_kind = "resident_only" if missing else None
                controller_action = "shortlist_execute"
        else:
            for expert_key in missing:
                miss_loads, miss_evictions = ensure_resident(
                    dataset=dataset,
                    state=state,
                    policy=policy,
                    budgets=budgets,
                    window_index=window.window_index,
                    expert_key=expert_key,
                    protected=protected,
                    reason="load_on_miss",
                    allow_burst=True,
                )
                loads.extend(miss_loads)
                evictions.extend(miss_evictions)
            active_experts = sorted(requested)

        if run_kind == "controller_demo":
            captured_mass_ratio = safe_div(
                sum(requested[expert_key] for expert_key in active_experts if expert_key in requested),
                requested_mass_total,
            )
        else:
            captured_mass_ratio = 1.0 if requested else 0.0

        context_retention = context_retention_for_fraction(effective_fraction)
        quality_proxy = 0.7 * captured_mass_ratio + 0.3 * context_retention
        dense_baseline_delta = None
        if dense_quality_by_window is not None:
            dense_baseline_delta = quality_proxy - dense_quality_by_window[window.window_id]

        mark_used_residents(
            state=state,
            expert_keys=active_experts,
            window_index=window.window_index,
            hysteresis_windows=policy.hysteresis_windows,
        )
        burst_cleanup = release_burst_residents(state, protected=set(active_experts))
        evictions.extend(burst_cleanup)
        resident_after = state.resident_keys()

        load_ms = sum(float(item.get("load_ms") or 0.0) for item in loads)
        eviction_ms = len(evictions) * 2.5
        dense_fallback_penalty_ms = 18.0 if fallback_kind == "dense_fallback" else 0.0
        compute_ms = 10.0 + window.token_count * 0.7 + len(active_experts) * 2.0
        total_latency_ms = compute_ms + load_ms + eviction_ms + dense_fallback_penalty_ms

        event = build_run_event(
            run_kind=run_kind,
            window=window,
            policy=policy,
            resident_budget_fraction=resident_budget_fraction,
            resident_before=resident_before,
            resident_after=resident_after,
            candidate_set=planned_candidate_set,
            requested_experts=requested,
            active_experts=active_experts,
            loads=loads,
            evictions=evictions,
            fallback_used=fallback_used,
            fallback_kind=fallback_kind,
            controller_action=controller_action,
            latency_ms={
                "load_ms": round3(load_ms) or 0.0,
                "eviction_ms": round3(eviction_ms) or 0.0,
                "compute_ms": round3(compute_ms) or 0.0,
                "dense_fallback_penalty_ms": round3(dense_fallback_penalty_ms) or 0.0,
                "total_ms": round3(total_latency_ms) or 0.0,
            },
            quality_proxy=quality_proxy,
            context_retention=context_retention,
            dense_baseline_delta=dense_baseline_delta,
            runtime_state=window.runtime_state,
        )
        events.append(event)
        state.update_history(window, actual_mass)

    annotate_eviction_regret(events)
    return events


def effective_working_set_count(expert_mass: dict[str, float], coverage_fraction: float = 0.8) -> int:
    ranked = sorted(expert_mass.values(), reverse=True)
    total = sum(ranked)
    if total <= 0:
        return 0
    running = 0.0
    for index, value in enumerate(ranked, start=1):
        running += value
        if running / total >= coverage_fraction:
            return index
    return len(ranked)


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return safe_div(len(left & right), len(left | right))


def build_advisor_report(dataset: ReplayDataset) -> JSONDict:
    by_family: defaultdict[str, list[ReplayWindow]] = defaultdict(list)
    for window in dataset.windows:
        by_family[window.prompt_family].append(window)

    family_findings: dict[str, JSONDict] = {}
    viable_families = 0
    chaotic_families = 0
    for family_id, windows in sorted(by_family.items()):
        overlap_values: list[float] = []
        working_set_values: list[float] = []
        previous_requested: set[str] | None = None
        for window in windows:
            requested = set(requested_experts_for_window(window, dataset.top_k))
            flattened = flatten_window_mass(window)
            working_set_values.append(float(effective_working_set_count(flattened)))
            if previous_requested is not None:
                overlap_values.append(jaccard_similarity(previous_requested, requested))
            previous_requested = requested
        mean_overlap = mean(overlap_values)
        mean_working_set = mean(working_set_values)
        if mean_overlap >= 0.6 and mean_working_set <= 6.0:
            status = "compact"
            viable_families += 1
        elif mean_overlap <= 0.35 or mean_working_set >= 7.5:
            status = "chaotic"
            chaotic_families += 1
        else:
            status = "mixed"
        family_findings[family_id] = {
            "window_count": len(windows),
            "mean_overlap_prev": round3(mean_overlap),
            "mean_effective_working_set_80": round3(mean_working_set),
            "status": status,
            "predictive_residency_viable": status != "chaotic",
        }

    viability = "predictive_residency_viable" if viable_families >= max(2, chaotic_families) else "probe_more_first"
    return {
        "trace_name": dataset.trace_name,
        "window_size_tokens": dataset.window_size_tokens,
        "family_findings": family_findings,
        "overall": {
            "family_count": len(by_family),
            "compact_family_count": viable_families,
            "chaotic_family_count": chaotic_families,
            "viability_assessment": viability,
        },
    }


def best_constrained_summary(
    sweep_summaries: list[JSONDict],
    budget_fraction: float,
) -> JSONDict | None:
    candidates = [
        item
        for item in sweep_summaries
        if item["resident_budget_fraction"] == round3(budget_fraction) and item["policy_name"] != "reactive_lru"
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            item["mean_miss_rate"],
            item["mean_churn"],
            item["mean_eviction_regret"],
            -item["mean_dense_baseline_delta"],
        )
    )
    return candidates[0]


def find_summary(summaries: list[JSONDict], policy_name: str, resident_budget_fraction: float) -> JSONDict:
    target_fraction = round3(resident_budget_fraction)
    for item in summaries:
        if item["policy_name"] == policy_name and item["resident_budget_fraction"] == target_fraction:
            return item
    raise ValueError(f"Missing summary for {policy_name} at budget {resident_budget_fraction}")


def build_failure_mode_matrix(
    *,
    advisor_report: JSONDict,
    sweep_summaries: list[JSONDict],
    controller_summary: JSONDict,
) -> JSONDict:
    reactive_40 = find_summary(sweep_summaries, "reactive_lru", 0.4)
    best_40 = best_constrained_summary(sweep_summaries, 0.4) or reactive_40
    router_chaos = advisor_report["overall"]["chaotic_family_count"] >= 1
    latency_thrash = best_40["mean_load_latency_ms"] > 12.0 or best_40["mean_churn"] > 0.45
    false_economy = best_40["mean_dense_baseline_delta"] < -0.05
    policy_overfit = any(
        metrics["mean_dense_baseline_delta"] is not None and metrics["mean_dense_baseline_delta"] < -0.08
        for metrics in controller_summary["by_family"].values()
    )
    hidden_coupling = advisor_report["overall"]["compact_family_count"] >= 2 and best_40["mean_miss_rate"] > 0.25
    return {
        "router_chaos": {
            "active": router_chaos,
            "signal": "Chaotic prompt families detected in advisor report." if router_chaos else "Most families stay compact or mixed.",
        },
        "latency_thrash": {
            "active": latency_thrash,
            "signal": "Load or churn cost still dominates constrained runs." if latency_thrash else "Load and churn remain controlled.",
        },
        "false_economy": {
            "active": false_economy,
            "signal": "Latency gains come with more than 5% dense-baseline quality loss." if false_economy else "Quality stays inside the default guardrail.",
        },
        "policy_overfit": {
            "active": policy_overfit,
            "signal": "At least one family regresses sharply under the controller demo." if policy_overfit else "No major family-specific cliffs detected.",
        },
        "hidden_coupling": {
            "active": hidden_coupling,
            "signal": "Compact families still miss too often, suggesting coupling or poor separability." if hidden_coupling else "Observed misses mostly align with expected budget pressure.",
        },
    }


def build_next_stage_branch(
    *,
    advisor_report: JSONDict,
    sweep_summaries: list[JSONDict],
    bridge_status: JSONDict,
) -> JSONDict:
    reactive_40 = find_summary(sweep_summaries, "reactive_lru", 0.4)
    best_40 = best_constrained_summary(sweep_summaries, 0.4) or reactive_40
    improvement_count = sum(
        [
            best_40["mean_miss_rate"] < reactive_40["mean_miss_rate"],
            best_40["mean_churn"] < reactive_40["mean_churn"],
            best_40["mean_eviction_regret"] < reactive_40["mean_eviction_regret"],
        ]
    )
    quality_ok = best_40["mean_dense_baseline_delta"] >= -0.05
    if improvement_count >= 2 and quality_ok:
        status = "green"
        summary = "Proceed to one bounded PyTorch target before Triton."
    elif quality_ok:
        status = "yellow"
        summary = "Stay at the advisor/controller layer and tighten traces before deeper descent."
    else:
        status = "red"
        summary = "Do not descend yet; current constrained policies lose too much quality."
    return {
        "status": status,
        "summary": summary,
        "bridge_status": bridge_status,
        "recommended_next_target": (
            "Inspect a hookable PyTorch MoE layer with explicit router logits and expert modules."
            if status == "green"
            else "Refine prompt coverage, trace quality, and model choice above the engine layer."
        ),
        "inspect_first": [
            "router output tensors and top-k selection boundaries",
            "expert weight layout and whether experts remain individually addressable",
            "dispatch packing or fusion boundaries before any Triton descent",
        ]
        if status == "green"
        else [
            "prompt-family coverage gaps",
            "advisor false positives on compact vs chaotic windows",
            "controller fallback frequency and miss causes",
        ],
        "rationale": {
            "compact_family_count": advisor_report["overall"]["compact_family_count"],
            "chaotic_family_count": advisor_report["overall"]["chaotic_family_count"],
            "improvement_count_vs_reactive_40": improvement_count,
            "quality_guardrail_met": quality_ok,
        },
    }


def build_bridge_status(dataset: ReplayDataset) -> JSONDict:
    if dataset.source_kind in {"synthetic_trace", "forward_hook_demo"}:
        return {
            "status": "paused",
            "reason": (
                "No real hookable MoE backend is wired yet, so the bridge track stays on synthetic or demo artifacts "
                "while the schema, advisor, and controller work continue."
            ),
        }
    return {
        "status": "active",
        "reason": "Replay data came from a hookable forward-probe run.",
    }


def execute_bridge_plan(
    *,
    dataset: ReplayDataset,
    output_dir: Path,
    label: str,
) -> JSONDict:
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{label}"
    run_dir = output_dir / run_id
    ensure_dir(run_dir)
    events_path = run_dir / "events.jsonl"
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.json"

    policies = default_policies()
    budget_fractions = [1.0, 0.4, 0.25]
    manifest = {
        "created_at": now_iso(),
        "run_id": run_id,
        "mode": "moe_controller_demo",
        "shared_contract_template": contract_shell(
            probe_tier="controller_replay",
            backend_family="trace_replay",
            default_baseline_kind="controller_demo",
            default_window_size_tokens=dataset.window_size_tokens,
        ),
        "dataset": {
            "trace_name": dataset.trace_name,
            "origin": dataset.origin,
            "source_kind": dataset.source_kind,
            "window_size_tokens": dataset.window_size_tokens,
            "top_k": dataset.top_k,
            "window_count": len(dataset.windows),
            "expert_count": len(dataset.experts),
        },
        "policy_names": list(policies),
        "budget_fractions": budget_fractions,
    }
    write_json(manifest_path, manifest)

    dense_policy = ReplayPolicyConfig(name="dense_baseline")
    dense_events = run_policy(
        dataset=dataset,
        policy=dense_policy,
        resident_budget_fraction=1.0,
        run_kind="dense_baseline",
    )
    dense_quality_by_window = {
        item["window"]["window_id"]: float(item["quality"]["quality_proxy"])
        for item in dense_events
    }
    dense_summary = summarize_run(dense_events, dense_policy, "dense_baseline", 1.0)

    sweep_summaries: list[JSONDict] = []
    sweep_events: list[JSONDict] = []
    for budget_fraction in [0.4, 0.25]:
        for policy_name in ["reactive_lru", "weighted_lru", "window_predictive"]:
            policy = policies[policy_name]
            events = run_policy(
                dataset=dataset,
                policy=policy,
                resident_budget_fraction=budget_fraction,
                run_kind="budget_sweep",
                dense_quality_by_window=dense_quality_by_window,
            )
            sweep_events.extend(events)
            sweep_summaries.append(summarize_run(events, policy, "budget_sweep", budget_fraction))

    controller_policy = policies["window_predictive"]
    controller_events = run_policy(
        dataset=dataset,
        policy=controller_policy,
        resident_budget_fraction=0.4,
        run_kind="controller_demo",
        dense_quality_by_window=dense_quality_by_window,
    )
    controller_summary = summarize_run(controller_events, controller_policy, "controller_demo", 0.4)
    advisor_report = build_advisor_report(dataset)
    bridge_status = build_bridge_status(dataset)
    failure_mode_matrix = build_failure_mode_matrix(
        advisor_report=advisor_report,
        sweep_summaries=sweep_summaries,
        controller_summary=controller_summary,
    )
    next_stage_branch = build_next_stage_branch(
        advisor_report=advisor_report,
        sweep_summaries=sweep_summaries,
        bridge_status=bridge_status,
    )

    append_jsonl(events_path, dense_events + sweep_events + controller_events)
    summary = {
        "created_at": now_iso(),
        "run_id": run_id,
        "trace_name": dataset.trace_name,
        "shared_contract_template": contract_shell(
            probe_tier="controller_replay",
            backend_family="trace_replay",
            default_baseline_kind="controller_demo",
            default_window_size_tokens=dataset.window_size_tokens,
        ),
        "bridge_status": bridge_status,
        "dense_baseline": dense_summary,
        "advisor_report": advisor_report,
        "budget_sweeps": sweep_summaries,
        "controller_demo": controller_summary,
        "failure_mode_matrix": failure_mode_matrix,
        "next_stage_branch": next_stage_branch,
    }
    write_json(summary_path, summary)
    return {
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "summary": summary,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-path", type=Path)
    parser.add_argument("--forward-run-dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "controller-runs",
    )
    parser.add_argument("--label", default="moe-controller-demo")
    return parser


def resolve_dataset(args: argparse.Namespace) -> ReplayDataset:
    if args.trace_path:
        return load_trace_dataset(args.trace_path)
    if args.forward_run_dir:
        return load_forward_probe_dataset(args.forward_run_dir)
    default_trace = Path(__file__).resolve().parent / "data" / "synthetic_controller_trace.json"
    return load_trace_dataset(default_trace)


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    dataset = resolve_dataset(args)
    result = execute_bridge_plan(
        dataset=dataset,
        output_dir=args.output_dir,
        label=args.label,
    )
    print(json.dumps({"run_dir": result["run_dir"], "summary_path": result["summary_path"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
