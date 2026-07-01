#!/usr/bin/env python3
"""Validate and summarize Phase 3 baseline replay policy definitions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICIES_PATH = ROOT / "memory-moe-mvp" / "data" / "baseline_replay_policies.json"
SUPPORTED_SCHEMA_VERSION = "moe-baseline-replay-policies-v1"
REQUIRED_POLICY_IDS = {
    "observe_only",
    "keep_hot",
    "preload_shortlist",
    "evict_cold",
    "fallback_dense",
    "no_live_actuator",
}
SUPPORTED_MODES = {"observe_only", "replay_simulate"}
REQUIRED_ACTION_IDS = {
    "observe_only",
    "preload",
    "pin",
    "keep",
    "evict",
    "demote",
    "fallback_dense",
    "abort_run",
}
REQUIRED_METRICS = {
    "unique_expert_count",
    "unique_estimated_residency_bytes",
    "route_estimated_residency_bytes",
    "reuse_distance",
    "miss_rate",
    "warm_hit_rate",
    "churn",
    "fallback_frequency",
    "rejected_policy_reasons",
}

JSONDict = dict[str, Any]


def load_policies(path: Path) -> JSONDict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("policy root must be a JSON object")
    return data


def require_string(obj: JSONDict, field: str, errors: list[str], *, context: str) -> None:
    value = obj.get(field)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{context}.{field} must be a non-empty string")


def require_bool(obj: JSONDict, field: str, errors: list[str], *, context: str) -> None:
    if not isinstance(obj.get(field), bool):
        errors.append(f"{context}.{field} must be a boolean")


def require_string_list(obj: JSONDict, field: str, errors: list[str], *, context: str) -> list[str]:
    value = obj.get(field)
    if not isinstance(value, list) or not value:
        errors.append(f"{context}.{field} must be a non-empty list")
        return []
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            errors.append(f"{context}.{field}[{index}] must be a non-empty string")
        else:
            result.append(item)
    return result


def validate_top_level(plan: JSONDict, errors: list[str]) -> tuple[set[str], set[str]]:
    if plan.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        errors.append(
            "schema_version must be "
            f"{SUPPORTED_SCHEMA_VERSION!r}, got {plan.get('schema_version')!r}"
        )
    for field in ("name", "generated_for", "phase"):
        require_string(plan, field, errors, context="plan")
    if plan.get("phase") != "phase_3":
        errors.append("plan.phase must be 'phase_3'")
    source_artifacts = require_string_list(plan, "source_artifacts", errors, context="plan")
    if not any("plan_trace_inventory_replay.py" in item for item in source_artifacts):
        errors.append("plan.source_artifacts must include scripts/plan_trace_inventory_replay.py")

    actions = set(require_string_list(plan, "managed_loading_actions", errors, context="plan"))
    missing_actions = sorted(REQUIRED_ACTION_IDS - actions)
    if missing_actions:
        errors.append(f"plan.managed_loading_actions missing: {', '.join(missing_actions)}")
    unsupported_actions = sorted(actions - REQUIRED_ACTION_IDS)
    if unsupported_actions:
        errors.append(f"plan.managed_loading_actions unsupported: {', '.join(unsupported_actions)}")

    metrics = set(require_string_list(plan, "required_replay_metrics", errors, context="plan"))
    missing_metrics = sorted(REQUIRED_METRICS - metrics)
    if missing_metrics:
        errors.append(f"plan.required_replay_metrics missing: {', '.join(missing_metrics)}")
    require_string_list(plan, "safety_contract", errors, context="plan")
    require_string_list(plan, "next_actions", errors, context="plan")
    return actions, metrics


def validate_policy(
    policy: JSONDict,
    *,
    context: str,
    allowed_actions: set[str],
    allowed_metrics: set[str],
    errors: list[str],
) -> str | None:
    for field in ("id", "mode", "description"):
        require_string(policy, field, errors, context=context)
    require_bool(policy, "may_mutate_runtime", errors, context=context)
    if policy.get("may_mutate_runtime") is not False:
        errors.append(f"{context}.may_mutate_runtime must be false for Phase 3 replay policies")
    if policy.get("mode") not in SUPPORTED_MODES:
        errors.append(f"{context}.mode has unsupported value {policy.get('mode')!r}")

    actions = set(require_string_list(policy, "managed_loading_actions", errors, context=context))
    unsupported_actions = sorted(actions - allowed_actions)
    if unsupported_actions:
        errors.append(f"{context}.managed_loading_actions unsupported: {', '.join(unsupported_actions)}")
    if not actions:
        errors.append(f"{context}.managed_loading_actions must not be empty")

    requires = set(require_string_list(policy, "requires_metrics", errors, context=context))
    reports = set(require_string_list(policy, "reports_metrics", errors, context=context))
    rejects = require_string_list(policy, "rejects_when", errors, context=context)
    unsupported_requires = sorted(requires - allowed_metrics)
    unsupported_reports = sorted(reports - allowed_metrics)
    if unsupported_requires:
        errors.append(f"{context}.requires_metrics unsupported: {', '.join(unsupported_requires)}")
    if unsupported_reports:
        errors.append(f"{context}.reports_metrics unsupported: {', '.join(unsupported_reports)}")
    missing_reports = sorted(reports - requires - {"rejected_policy_reasons"})
    if missing_reports:
        errors.append(f"{context}.reports_metrics should be required or rejected_policy_reasons: {', '.join(missing_reports)}")
    if not rejects:
        errors.append(f"{context}.rejects_when must name at least one rejection reason")

    policy_id = policy.get("id")
    return str(policy_id) if isinstance(policy_id, str) else None


def validate_policies(plan: JSONDict) -> list[str]:
    errors: list[str] = []
    allowed_actions, allowed_metrics = validate_top_level(plan, errors)
    policies = plan.get("policies")
    if not isinstance(policies, list) or not policies:
        errors.append("policies must be a non-empty list")
        return errors

    seen: set[str] = set()
    for index, policy in enumerate(policies):
        context = f"policies[{index}]"
        if not isinstance(policy, dict):
            errors.append(f"{context} must be an object")
            continue
        policy_id = validate_policy(
            policy,
            context=context,
            allowed_actions=allowed_actions,
            allowed_metrics=allowed_metrics,
            errors=errors,
        )
        if policy_id is None:
            continue
        if policy_id in seen:
            errors.append(f"{context}.id duplicates {policy_id!r}")
        seen.add(policy_id)

    missing = sorted(REQUIRED_POLICY_IDS - seen)
    unsupported = sorted(seen - REQUIRED_POLICY_IDS)
    if missing:
        errors.append(f"policies missing required ids: {', '.join(missing)}")
    if unsupported:
        errors.append(f"policies unsupported ids: {', '.join(unsupported)}")
    return errors


def build_summary(plan: JSONDict, path: Path) -> JSONDict:
    errors = validate_policies(plan)
    policies = plan.get("policies") if isinstance(plan.get("policies"), list) else []
    return {
        "mode": "baseline_replay_policies_plan",
        "policies_path": str(path),
        "valid": not errors,
        "errors": errors,
        "schema_version": plan.get("schema_version"),
        "phase": plan.get("phase"),
        "policy_ids": [item.get("id") for item in policies if isinstance(item, dict)],
        "required_replay_metrics": plan.get("required_replay_metrics", []),
        "managed_loading_actions": plan.get("managed_loading_actions", []),
        "may_mutate_runtime": any(
            bool(item.get("may_mutate_runtime")) for item in policies if isinstance(item, dict)
        ),
        "safety_contract": plan.get("safety_contract", []),
        "next_actions": plan.get("next_actions", []),
    }


def print_human_summary(summary: JSONDict) -> None:
    print("MoE Run Anyway baseline replay policies")
    print(f"Policies: {summary['policies_path']}")
    print(f"Valid: {summary['valid']}")
    if summary["errors"]:
        print("Errors:")
        for error in summary["errors"]:
            print(f"  - {error}")
        return
    print(f"Phase: {summary['phase']}")
    print("Policies:")
    for policy_id in summary["policy_ids"]:
        print(f"  - {policy_id}")
    print(f"May mutate runtime: {summary['may_mutate_runtime']}")
    print("Next actions:")
    for action in summary["next_actions"]:
        print(f"  - {action}")
    print("Safety contract:")
    for item in summary["safety_contract"]:
        print(f"  - {item}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policies_path", nargs="?", type=Path, default=DEFAULT_POLICIES_PATH)
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    return parser


def plan_path(path: Path) -> tuple[int, JSONDict | None, str | None]:
    try:
        plan = load_policies(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not load baseline replay policies: {exc}"
    summary = build_summary(plan, path)
    return (0 if summary["valid"] else 2), summary, None


def main_from_test_path(path: Path) -> int:
    status, _, _ = plan_path(path)
    return status


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary, error_message = plan_path(args.policies_path)
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
