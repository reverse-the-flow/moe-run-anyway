#!/usr/bin/env python3
"""Shared trace and report contract for MoE probe and controller runs."""

from __future__ import annotations

from typing import Any


JSONDict = dict[str, Any]
CONTRACT_VERSION = "memory-moe-bridge-v1"
DEFAULT_WINDOW_SIZE_TOKENS = 32
SHARED_CONTRACT_FIELDS = (
    "probe_tier",
    "backend_family",
    "prompt_family",
    "window_size_tokens",
    "policy_name",
    "resident_budget_fraction",
    "baseline_kind",
    "candidate_set_size",
    "fallback_used",
    "warm_hit_rate",
    "miss_rate",
    "churn",
    "eviction_regret",
    "context_retention",
    "dense_baseline_delta",
)


def round_or_none(value: float | int | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def build_shared_contract(
    *,
    probe_tier: str,
    backend_family: str,
    prompt_family: str | None = None,
    window_size_tokens: int | None = DEFAULT_WINDOW_SIZE_TOKENS,
    policy_name: str | None = None,
    resident_budget_fraction: float | None = None,
    baseline_kind: str | None = None,
    candidate_set_size: int | None = None,
    fallback_used: bool | None = None,
    warm_hit_rate: float | None = None,
    miss_rate: float | None = None,
    churn: float | None = None,
    eviction_regret: float | None = None,
    context_retention: float | None = None,
    dense_baseline_delta: float | None = None,
) -> JSONDict:
    payload: JSONDict = {field: None for field in SHARED_CONTRACT_FIELDS}
    payload.update(
        {
            "probe_tier": probe_tier,
            "backend_family": backend_family,
            "prompt_family": prompt_family,
            "window_size_tokens": window_size_tokens,
            "policy_name": policy_name,
            "resident_budget_fraction": round_or_none(resident_budget_fraction),
            "baseline_kind": baseline_kind,
            "candidate_set_size": candidate_set_size,
            "fallback_used": fallback_used,
            "warm_hit_rate": round_or_none(warm_hit_rate),
            "miss_rate": round_or_none(miss_rate),
            "churn": round_or_none(churn),
            "eviction_regret": round_or_none(eviction_regret),
            "context_retention": round_or_none(context_retention),
            "dense_baseline_delta": round_or_none(dense_baseline_delta),
        }
    )
    return payload


def attach_shared_contract(payload: JSONDict, shared_contract: JSONDict) -> JSONDict:
    result = dict(payload)
    result["contract_version"] = CONTRACT_VERSION
    result["shared_contract"] = shared_contract
    return result


def contract_shell(
    *,
    probe_tier: str,
    backend_family: str,
    default_policy_name: str | None = None,
    default_baseline_kind: str | None = None,
    default_window_size_tokens: int | None = DEFAULT_WINDOW_SIZE_TOKENS,
) -> JSONDict:
    return {
        "contract_version": CONTRACT_VERSION,
        "required_fields": list(SHARED_CONTRACT_FIELDS),
        "defaults": build_shared_contract(
            probe_tier=probe_tier,
            backend_family=backend_family,
            policy_name=default_policy_name,
            baseline_kind=default_baseline_kind,
            window_size_tokens=default_window_size_tokens,
        ),
    }
