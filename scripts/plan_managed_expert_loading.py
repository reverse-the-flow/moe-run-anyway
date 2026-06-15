#!/usr/bin/env python3
"""Validate and summarize the managed expert loading contract.

This planner reads a local JSON contract and reports whether MoE Run Anyway has
the evidence needed to move from observe-only/replay work toward a future live
expert-loading actuator. It does not start models, download files, authenticate,
run Docker, launch servers, or send prompt traffic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN_PATH = ROOT / "memory-moe-mvp" / "data" / "managed_expert_loading_plan.json"
SUPPORTED_SCHEMA_VERSION = "moe-managed-expert-loading-plan-v1"

REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "name",
    "generated_for",
    "artifact_compatibility",
    "state_vocab",
    "policy_actions",
    "backend_adapters",
    "capability_requirements",
    "current_capability_gaps",
    "safety_contract",
    "next_actions",
}
REQUIRED_STATE_IDS = {
    "resident",
    "offloaded_cpu",
    "offloaded_disk",
    "loading",
    "evicting",
    "unknown",
}
REQUIRED_POLICY_ACTION_IDS = {
    "observe_only",
    "preload",
    "pin",
    "keep",
    "evict",
    "demote",
    "fallback_dense",
    "abort_run",
}
REQUIRED_BACKEND_ADAPTER_IDS = {
    "llama_cpp",
    "vllm_openai_compatible",
    "hookable_pytorch",
    "moe_infinity_style",
    "prototype_offload_system",
}
REQUIRED_CAPABILITY_IDS = {
    "expert_inventory",
    "routing_visibility",
    "residency_observation",
    "residency_control",
    "policy_application",
    "dense_fallback",
    "cleanup_restore",
    "artifact_export",
}
REQUIRED_MODES = {"observe_only", "replay_simulate", "live_actuator"}
REQUIRED_BOUNDARY_OWNERS = {"Model Plane", "MoE Run Anyway"}
CAPABILITY_STATUSES = {"available", "partial", "missing", "planned", "blocked"}
ADAPTER_MODES = {"observe_only", "replay_simulate", "live_actuator_missing"}

JSONDict = dict[str, Any]


def load_plan(path: Path) -> JSONDict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("plan root must be a JSON object")
    return data


def require_string(obj: JSONDict, field: str, errors: list[str], *, context: str) -> None:
    value = obj.get(field)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{context}.{field} must be a non-empty string")


def require_bool(obj: JSONDict, field: str, errors: list[str], *, context: str) -> None:
    if not isinstance(obj.get(field), bool):
        errors.append(f"{context}.{field} must be a boolean")


def require_string_list(obj: JSONDict, field: str, errors: list[str], *, context: str) -> None:
    value = obj.get(field)
    if not isinstance(value, list) or not value:
        errors.append(f"{context}.{field} must be a non-empty list")
        return
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            errors.append(f"{context}.{field}[{index}] must be a non-empty string")


def collect_ids(value: Any, *, context: str, errors: list[str]) -> set[str]:
    if not isinstance(value, list) or not value:
        errors.append(f"{context} must be a non-empty list")
        return set()

    ids: set[str] = set()
    for index, item in enumerate(value):
        item_context = f"{context}[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{item_context} must be an object")
            continue
        require_string(item, "id", errors, context=item_context)
        require_string(item, "description", errors, context=item_context)
        item_id = item.get("id")
        if isinstance(item_id, str):
            if item_id in ids:
                errors.append(f"{item_context}.id duplicates {item_id!r}")
            ids.add(item_id)
    return ids


def require_exact_ids(
    actual_ids: set[str],
    required_ids: set[str],
    *,
    context: str,
    errors: list[str],
) -> None:
    missing = sorted(required_ids - actual_ids)
    extra = sorted(actual_ids - required_ids)
    if missing:
        errors.append(f"{context} missing required ids: {', '.join(missing)}")
    if extra:
        errors.append(f"{context} has unsupported ids: {', '.join(extra)}")


def validate_artifact_compatibility(plan: JSONDict, errors: list[str]) -> None:
    artifact = plan.get("artifact_compatibility")
    if not isinstance(artifact, dict):
        errors.append("artifact_compatibility must be an object")
        return

    require_string(artifact, "contract_version", errors, context="artifact_compatibility")
    require_string(artifact, "dense_fallback_field", errors, context="artifact_compatibility")
    require_string_list(artifact, "extends_fields", errors, context="artifact_compatibility")
    require_string_list(artifact, "managed_loading_fields", errors, context="artifact_compatibility")

    required_fields = {
        "mode",
        "expert_inventory",
        "routing_visibility_source",
        "residency_state_before",
        "policy_action",
        "residency_state_after",
        "capability_gaps",
        "dense_fallback_available",
        "cleanup_restore_status",
        "live_actuator_implemented",
    }
    managed_fields = set(artifact.get("managed_loading_fields", []))
    missing = sorted(required_fields - managed_fields)
    if missing:
        errors.append(f"artifact_compatibility.managed_loading_fields missing: {', '.join(missing)}")


def validate_boundary(plan: JSONDict, errors: list[str]) -> None:
    boundary = plan.get("boundary")
    if not isinstance(boundary, dict):
        errors.append("boundary must be an object")
        return

    owners = boundary.get("owners")
    if not isinstance(owners, list) or not owners:
        errors.append("boundary.owners must be a non-empty list")
        return

    seen: set[str] = set()
    for index, owner in enumerate(owners):
        context = f"boundary.owners[{index}]"
        if not isinstance(owner, dict):
            errors.append(f"{context} must be an object")
            continue
        require_string(owner, "owner", errors, context=context)
        require_string_list(owner, "owns", errors, context=context)
        require_string_list(owner, "does_not_own", errors, context=context)
        owner_id = owner.get("owner")
        if isinstance(owner_id, str):
            seen.add(owner_id)

    require_exact_ids(seen, REQUIRED_BOUNDARY_OWNERS, context="boundary.owners", errors=errors)


def validate_modes(plan: JSONDict, errors: list[str]) -> None:
    modes = plan.get("modes")
    mode_ids = collect_ids(modes, context="modes", errors=errors)
    require_exact_ids(mode_ids, REQUIRED_MODES, context="modes", errors=errors)

    if not isinstance(modes, list):
        return
    for index, mode in enumerate(modes):
        if not isinstance(mode, dict):
            continue
        context = f"modes[{index}]"
        require_bool(mode, "may_mutate_runtime", errors, context=context)
        require_string_list(mode, "allowed_outputs", errors, context=context)


def validate_state_vocab(plan: JSONDict, errors: list[str]) -> None:
    ids = collect_ids(plan.get("state_vocab"), context="state_vocab", errors=errors)
    require_exact_ids(ids, REQUIRED_STATE_IDS, context="state_vocab", errors=errors)


def validate_policy_actions(plan: JSONDict, errors: list[str]) -> None:
    actions = plan.get("policy_actions")
    ids = collect_ids(actions, context="policy_actions", errors=errors)
    require_exact_ids(ids, REQUIRED_POLICY_ACTION_IDS, context="policy_actions", errors=errors)

    if not isinstance(actions, list):
        return
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        context = f"policy_actions[{index}]"
        require_string(action, "requires_capability", errors, context=context)
        capability = action.get("requires_capability")
        if capability not in REQUIRED_CAPABILITY_IDS:
            errors.append(f"{context}.requires_capability has unsupported value {capability!r}")


def validate_capability_requirements(plan: JSONDict, errors: list[str]) -> None:
    capabilities = plan.get("capability_requirements")
    ids = collect_ids(capabilities, context="capability_requirements", errors=errors)
    require_exact_ids(ids, REQUIRED_CAPABILITY_IDS, context="capability_requirements", errors=errors)

    if not isinstance(capabilities, list):
        return
    for index, capability in enumerate(capabilities):
        if not isinstance(capability, dict):
            continue
        context = f"capability_requirements[{index}]"
        require_bool(capability, "required_for_live_actuator", errors, context=context)
        require_string(capability, "current_status", errors, context=context)
        if capability.get("current_status") not in CAPABILITY_STATUSES:
            errors.append(f"{context}.current_status has unsupported value {capability.get('current_status')!r}")


def validate_backend_adapters(plan: JSONDict, errors: list[str]) -> None:
    adapters = plan.get("backend_adapters")
    ids = collect_ids(adapters, context="backend_adapters", errors=errors)
    require_exact_ids(ids, REQUIRED_BACKEND_ADAPTER_IDS, context="backend_adapters", errors=errors)

    if not isinstance(adapters, list):
        return
    for index, adapter in enumerate(adapters):
        if not isinstance(adapter, dict):
            continue
        context = f"backend_adapters[{index}]"
        for field in ("display_name", "class", "mode", "adapter_boundary"):
            require_string(adapter, field, errors, context=context)
        if adapter.get("mode") not in ADAPTER_MODES:
            errors.append(f"{context}.mode has unsupported value {adapter.get('mode')!r}")
        require_string_list(adapter, "evidence_sources", errors, context=context)
        require_string_list(adapter, "missing_actuator_capabilities", errors, context=context)

        status = adapter.get("capability_status")
        if not isinstance(status, dict):
            errors.append(f"{context}.capability_status must be an object")
            continue
        missing = sorted(REQUIRED_CAPABILITY_IDS - set(status))
        extra = sorted(set(status) - REQUIRED_CAPABILITY_IDS)
        if missing:
            errors.append(f"{context}.capability_status missing ids: {', '.join(missing)}")
        if extra:
            errors.append(f"{context}.capability_status has unsupported ids: {', '.join(extra)}")
        for capability_id, capability_status in status.items():
            if capability_status not in CAPABILITY_STATUSES:
                errors.append(
                    f"{context}.capability_status[{capability_id!r}] "
                    f"has unsupported value {capability_status!r}"
                )


def validate_current_capability_gaps(plan: JSONDict, errors: list[str]) -> None:
    gaps = plan.get("current_capability_gaps")
    if not isinstance(gaps, list) or not gaps:
        errors.append("current_capability_gaps must be a non-empty list")
        return

    for index, gap in enumerate(gaps):
        context = f"current_capability_gaps[{index}]"
        if not isinstance(gap, dict):
            errors.append(f"{context} must be an object")
            continue
        require_string(gap, "capability_id", errors, context=context)
        require_string(gap, "impact", errors, context=context)
        require_string(gap, "needed_before_live", errors, context=context)
        if gap.get("capability_id") not in REQUIRED_CAPABILITY_IDS:
            errors.append(f"{context}.capability_id has unsupported value {gap.get('capability_id')!r}")


def validate_plan(plan: JSONDict) -> list[str]:
    errors: list[str] = []
    if plan.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        errors.append(
            "schema_version must be "
            f"{SUPPORTED_SCHEMA_VERSION!r}, got {plan.get('schema_version')!r}"
        )

    missing_fields = sorted(REQUIRED_TOP_LEVEL_FIELDS - set(plan))
    if missing_fields:
        errors.append(f"plan missing required top-level fields: {', '.join(missing_fields)}")

    require_string(plan, "name", errors, context="plan")
    require_string(plan, "generated_for", errors, context="plan")
    require_bool(plan, "live_expert_loading_implemented", errors, context="plan")
    if plan.get("live_expert_loading_implemented") is not False:
        errors.append("plan.live_expert_loading_implemented must be false for this pass")

    validate_boundary(plan, errors)
    validate_modes(plan, errors)
    validate_artifact_compatibility(plan, errors)
    validate_state_vocab(plan, errors)
    validate_policy_actions(plan, errors)
    validate_capability_requirements(plan, errors)
    validate_backend_adapters(plan, errors)
    validate_current_capability_gaps(plan, errors)
    require_string_list(plan, "safety_contract", errors, context="plan")
    require_string_list(plan, "next_actions", errors, context="plan")
    return errors


def adapter_gaps(plan: JSONDict) -> dict[str, list[str]]:
    adapters = plan.get("backend_adapters")
    if not isinstance(adapters, list):
        return {}

    result: dict[str, list[str]] = {}
    for adapter in adapters:
        if not isinstance(adapter, dict) or not isinstance(adapter.get("id"), str):
            continue
        gaps = adapter.get("missing_actuator_capabilities")
        result[adapter["id"]] = list(gaps) if isinstance(gaps, list) else []
    return result


def capability_statuses(plan: JSONDict) -> dict[str, str]:
    capabilities = plan.get("capability_requirements")
    if not isinstance(capabilities, list):
        return {}

    result: dict[str, str] = {}
    for item in capabilities:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            status = item.get("current_status")
            result[item["id"]] = status if isinstance(status, str) else "unknown"
    return result


def build_summary(plan: JSONDict, path: Path) -> JSONDict:
    errors = validate_plan(plan)
    return {
        "mode": "managed_expert_loading_plan",
        "plan_path": str(path),
        "valid": not errors,
        "errors": errors,
        "schema_version": plan.get("schema_version"),
        "name": plan.get("name"),
        "generated_for": plan.get("generated_for"),
        "live_expert_loading_implemented": plan.get("live_expert_loading_implemented"),
        "backend_adapters": sorted(adapter_gaps(plan)),
        "adapter_gaps": adapter_gaps(plan),
        "capability_statuses": capability_statuses(plan),
        "current_capability_gaps": plan.get("current_capability_gaps", []),
        "safety_contract": plan.get("safety_contract", []),
        "next_actions": plan.get("next_actions", []),
    }


def print_human_summary(summary: JSONDict) -> None:
    print("MoE Run Anyway managed expert loading contract")
    print(f"Plan: {summary['plan_path']}")
    print(f"Valid: {summary['valid']}")
    if summary["errors"]:
        print("Errors:")
        for error in summary["errors"]:
            print(f"  - {error}")
        return

    print(f"Schema: {summary['schema_version']}")
    print(f"Live expert loading implemented: {summary['live_expert_loading_implemented']}")
    print("Capability status:")
    for capability_id, status in summary["capability_statuses"].items():
        print(f"  - {capability_id}: {status}")
    print("Backend actuator gaps:")
    for adapter_id, gaps in summary["adapter_gaps"].items():
        print(f"  - {adapter_id}: {', '.join(gaps) if gaps else 'none'}")
    print("Next actions:")
    for action in summary["next_actions"]:
        print(f"  - {action}")
    print("Safety contract:")
    for item in summary["safety_contract"]:
        print(f"  - {item}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "plan_path",
        nargs="?",
        type=Path,
        default=DEFAULT_PLAN_PATH,
        help="managed expert loading plan JSON path",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    return parser


def plan_path(path: Path) -> tuple[int, JSONDict | None, str | None]:
    try:
        plan = load_plan(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not load managed expert loading plan: {exc}"

    summary = build_summary(plan, path)
    return (0 if summary["valid"] else 2), summary, None


def main_from_test_path(path: Path) -> int:
    status, _, _ = plan_path(path)
    return status


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary, error_message = plan_path(args.plan_path)
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
