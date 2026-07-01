#!/usr/bin/env python3
"""Validate Phase 3 runtime actuator design metadata.

This planner makes the future runtime-control blocker inspectable before any
live mutation work. It reads the managed expert-loading contract, selects one
backend adapter, and reports whether the adapter boundary, capability gaps,
and safety contract are explicit enough for operator handoff. It does not start
models, run Docker, inspect endpoints, read secrets, send prompt traffic, load
tensor values, or mutate runtime residency.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import plan_managed_expert_loading


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN_PATH = ROOT / "memory-moe-mvp" / "data" / "managed_expert_loading_plan.json"
SUPPORTED_SCHEMA_VERSION = "moe-phase3-runtime-actuator-design-v1"
DEFAULT_BACKEND_FAMILY = "llama_cpp"
CONTROL_CAPABILITIES = {"residency_observation", "residency_control", "cleanup_restore"}

JSONDict = dict[str, Any]


def load_plan(path: Path) -> JSONDict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("managed loading plan root must be a JSON object")
    return data


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def find_adapter(plan: JSONDict, backend_family: str) -> JSONDict | None:
    adapters = plan.get("backend_adapters")
    if not isinstance(adapters, list):
        return None
    for adapter in adapters:
        if isinstance(adapter, dict) and adapter.get("id") == backend_family:
            return adapter
    return None


def live_required_capabilities(plan: JSONDict) -> list[str]:
    capabilities = plan.get("capability_requirements")
    if not isinstance(capabilities, list):
        return []
    result: list[str] = []
    for capability in capabilities:
        if not isinstance(capability, dict):
            continue
        if capability.get("required_for_live_actuator") is True and isinstance(capability.get("id"), str):
            result.append(str(capability["id"]))
    return sorted(result)


def adapter_capability_status(adapter: JSONDict | None) -> dict[str, str]:
    if not isinstance(adapter, dict) or not isinstance(adapter.get("capability_status"), dict):
        return {}
    result: dict[str, str] = {}
    for key, value in adapter["capability_status"].items():
        if isinstance(key, str):
            result[key] = value if isinstance(value, str) else "unknown"
    return result


def build_summary(plan: JSONDict, path: Path, *, backend_family: str = DEFAULT_BACKEND_FAMILY) -> JSONDict:
    managed_errors = plan_managed_expert_loading.validate_plan(plan)
    errors = list(managed_errors)
    adapter = find_adapter(plan, backend_family)
    if adapter is None:
        errors.append(f"backend adapter {backend_family!r} is not defined")
        adapter = {}

    status = adapter_capability_status(adapter)
    required_capabilities = live_required_capabilities(plan)
    missing_actuator_capabilities = string_list(adapter.get("missing_actuator_capabilities"))
    evidence_sources = string_list(adapter.get("evidence_sources"))
    adapter_boundary = adapter.get("adapter_boundary") if isinstance(adapter.get("adapter_boundary"), str) else ""
    mode = adapter.get("mode") if isinstance(adapter.get("mode"), str) else "unknown"

    blocking_capabilities = sorted(
        capability
        for capability in required_capabilities
        if status.get(capability) != "available"
    )
    explicit_gap_set = set(missing_actuator_capabilities)
    control_blockers = sorted(capability for capability in CONTROL_CAPABILITIES if status.get(capability) != "available")
    missing_explicit_gaps = sorted(
        capability for capability in control_blockers if capability not in explicit_gap_set
    )
    if missing_explicit_gaps:
        errors.append(
            "adapter missing_actuator_capabilities must name unavailable control gaps: "
            + ", ".join(missing_explicit_gaps)
        )

    design_handoff_ready = (
        not errors
        and bool(adapter_boundary.strip())
        and bool(evidence_sources)
        and bool(missing_actuator_capabilities)
        and plan.get("live_expert_loading_implemented") is False
    )
    residency_control_ready = status.get("residency_control") == "available"
    residency_observation_ready = status.get("residency_observation") == "available"
    cleanup_restore_ready = status.get("cleanup_restore") == "available"
    live_actuator_ready = False
    live_actuator_blockers = sorted(
        set(blocking_capabilities)
        | ({"live_expert_loading_not_implemented"} if plan.get("live_expert_loading_implemented") is not True else set())
    )

    return {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "mode": "phase3_runtime_actuator_design_plan",
        "metadata_only": True,
        "valid": not errors,
        "errors": errors,
        "managed_plan_path": display_path(path),
        "managed_plan_schema_version": plan.get("schema_version"),
        "managed_plan_valid": not managed_errors,
        "backend_family": backend_family,
        "backend_adapter_found": bool(adapter),
        "backend_display_name": adapter.get("display_name"),
        "backend_class": adapter.get("class"),
        "backend_mode": mode,
        "adapter_boundary": adapter_boundary,
        "evidence_sources": evidence_sources,
        "evidence_source_count": len(evidence_sources),
        "required_live_capabilities": required_capabilities,
        "required_live_capability_count": len(required_capabilities),
        "capability_status": status,
        "blocking_capabilities": blocking_capabilities,
        "blocking_capability_count": len(blocking_capabilities),
        "missing_actuator_capabilities": missing_actuator_capabilities,
        "missing_actuator_capability_count": len(missing_actuator_capabilities),
        "control_blockers": control_blockers,
        "control_blocker_count": len(control_blockers),
        "design_handoff_ready": design_handoff_ready,
        "live_actuator_ready": live_actuator_ready,
        "live_actuator_blockers": live_actuator_blockers,
        "live_actuator_blocker_count": len(live_actuator_blockers),
        "residency_observation_ready": residency_observation_ready,
        "residency_control_ready": residency_control_ready,
        "cleanup_restore_ready": cleanup_restore_ready,
        "may_mutate_runtime": False,
        "safety_contract": [
            "runtime actuator design validation is metadata only",
            "runtime actuator design validation does not launch runtimes or send prompt traffic",
            "runtime actuator design validation does not mutate residency or model caches",
            "design handoff readiness does not imply live actuator readiness",
        ],
        "next_actions": [
            "Bind the selected backend to an auditable residency observation surface.",
            "Define guarded residency control actions before live mutation.",
            "Prove cleanup/restore before any live managed-loading claim.",
            "Fill the live capability proof artifact only after the adapter can produce before/after evidence.",
        ],
    }


def format_markdown_report(summary: JSONDict) -> str:
    lines = [
        "# Phase 3 Runtime Actuator Design",
        "",
        f"- Valid: `{summary.get('valid')}`",
        f"- Backend: `{summary.get('backend_family')}`",
        f"- Design handoff ready: `{summary.get('design_handoff_ready')}`",
        f"- Live actuator ready: `{summary.get('live_actuator_ready')}`",
        f"- Blocking capabilities: `{', '.join(summary.get('blocking_capabilities', [])) or 'none'}`",
        f"- Missing actuator capabilities: `{', '.join(summary.get('missing_actuator_capabilities', [])) or 'none'}`",
        f"- Evidence sources: `{summary.get('evidence_source_count')}`",
        "",
        "## Safety Contract",
    ]
    for item in summary.get("safety_contract", []):
        lines.append(f"- {item}")
    if summary.get("errors"):
        lines.extend(["", "## Errors"])
        for error in summary["errors"]:
            lines.append(f"- `{error}`")
    lines.extend(["", "## Next Actions"])
    for item in summary.get("next_actions", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def write_markdown_report(summary: JSONDict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_markdown_report(summary), encoding="utf-8")


def print_human_summary(summary: JSONDict) -> None:
    print("MoE Run Anyway Phase 3 runtime actuator design")
    print(f"Plan: {summary['managed_plan_path']}")
    print(f"Valid: {summary['valid']}")
    print(f"Backend: {summary['backend_family']}")
    if summary["errors"]:
        print("Errors:")
        for error in summary["errors"]:
            print(f"  - {error}")
        return
    print(f"Design handoff ready: {summary['design_handoff_ready']}")
    print(f"Live actuator ready: {summary['live_actuator_ready']}")
    print(f"Blocking capabilities: {', '.join(summary['blocking_capabilities']) if summary['blocking_capabilities'] else 'none'}")
    print(f"Missing actuator capabilities: {', '.join(summary['missing_actuator_capabilities']) if summary['missing_actuator_capabilities'] else 'none'}")
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
    parser.add_argument("--backend", default=DEFAULT_BACKEND_FAMILY, help="backend adapter id to inspect")
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    parser.add_argument("--output-md", type=Path, help="write a Markdown runtime actuator design report")
    return parser


def plan_path(path: Path, *, backend_family: str = DEFAULT_BACKEND_FAMILY) -> tuple[int, JSONDict | None, str | None]:
    try:
        plan = load_plan(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not load managed expert loading plan: {exc}"

    summary = build_summary(plan, path, backend_family=backend_family)
    return (0 if summary["valid"] else 2), summary, None


def main_from_test_path(path: Path, *, backend_family: str = DEFAULT_BACKEND_FAMILY) -> int:
    status, _, _ = plan_path(path, backend_family=backend_family)
    return status


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary, error_message = plan_path(args.plan_path, backend_family=args.backend)
    if error_message:
        print(error_message, file=sys.stderr)
        return status

    assert summary is not None
    if args.output_md:
        write_markdown_report(summary, args.output_md)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_human_summary(summary)
    return status


if __name__ == "__main__":
    raise SystemExit(main())