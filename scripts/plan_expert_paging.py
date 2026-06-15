#!/usr/bin/env python3
"""Validate and summarize the expert-paging roadmap.

This planner reads a local roadmap JSON artifact and reports current phase
status plus next actions. It does not start models, download files,
authenticate, run Docker, or send prompt traffic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROADMAP_PATH = ROOT / "memory-moe-mvp" / "data" / "expert_paging_roadmap.json"
SUPPORTED_SCHEMA_VERSION = "moe-expert-paging-roadmap-v1"
PHASE_STATUSES = {"complete", "in_progress", "planned", "blocked"}
SURFACE_STATUSES = {"implemented", "implemented_for_hookable_runtimes", "missing", "planned"}

JSONDict = dict[str, Any]


def load_roadmap(path: Path) -> JSONDict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("roadmap root must be a JSON object")
    return data


def require_string(obj: JSONDict, field: str, errors: list[str], *, context: str) -> None:
    value = obj.get(field)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{context}.{field} must be a non-empty string")


def require_string_list(obj: JSONDict, field: str, errors: list[str], *, context: str) -> None:
    value = obj.get(field)
    if not isinstance(value, list) or not value:
        errors.append(f"{context}.{field} must be a non-empty list")
        return
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            errors.append(f"{context}.{field}[{index}] must be a non-empty string")


def validate_truth_table(roadmap: JSONDict, errors: list[str]) -> None:
    truth_table = roadmap.get("current_truth_table")
    if not isinstance(truth_table, list) or not truth_table:
        errors.append("current_truth_table must be a non-empty list")
        return

    surfaces = set()
    for index, row in enumerate(truth_table):
        context = f"current_truth_table[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{context} must be an object")
            continue
        require_string(row, "surface", errors, context=context)
        require_string(row, "can_do", errors, context=context)
        require_string(row, "cannot_do", errors, context=context)
        status = row.get("status")
        if status not in SURFACE_STATUSES:
            errors.append(f"{context}.status has unsupported value {status!r}")
        if isinstance(row.get("surface"), str):
            surfaces.add(row["surface"])

    if "runtime_actuator" not in surfaces:
        errors.append("current_truth_table must include runtime_actuator")


def validate_phases(roadmap: JSONDict, errors: list[str]) -> None:
    phases = roadmap.get("phases")
    if not isinstance(phases, list) or not phases:
        errors.append("phases must be a non-empty list")
        return

    expected_ids = [f"phase_{index}" for index in range(len(phases))]
    seen_ids: list[str] = []
    for index, phase in enumerate(phases):
        context = f"phases[{index}]"
        if not isinstance(phase, dict):
            errors.append(f"{context} must be an object")
            continue
        require_string(phase, "id", errors, context=context)
        require_string(phase, "title", errors, context=context)
        require_string_list(phase, "deliverables", errors, context=context)
        require_string_list(phase, "evidence_gates", errors, context=context)
        require_string_list(phase, "next_actions", errors, context=context)
        status = phase.get("status")
        if status not in PHASE_STATUSES:
            errors.append(f"{context}.status has unsupported value {status!r}")
        if isinstance(phase.get("id"), str):
            seen_ids.append(phase["id"])

    if seen_ids != expected_ids:
        errors.append(f"phase ids must be sequential {expected_ids!r}, got {seen_ids!r}")


def validate_actuator_spike_checklist(roadmap: JSONDict, errors: list[str]) -> None:
    checklist = roadmap.get("actuator_spike_checklist")
    if not isinstance(checklist, list) or not checklist:
        errors.append("actuator_spike_checklist must be a non-empty list")
        return

    required_ids = {
        "routing_visibility",
        "tensor_residency_control",
        "dense_fallback",
        "artifact_shape",
        "cleanup",
    }
    seen = set()
    for index, item in enumerate(checklist):
        context = f"actuator_spike_checklist[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{context} must be an object")
            continue
        require_string(item, "id", errors, context=context)
        require_string(item, "question", errors, context=context)
        require_string(item, "required_evidence", errors, context=context)
        if isinstance(item.get("id"), str):
            seen.add(item["id"])

    missing = sorted(required_ids - seen)
    if missing:
        errors.append(f"actuator_spike_checklist missing required ids: {', '.join(missing)}")


def validate_model_plane_handoff(roadmap: JSONDict, errors: list[str]) -> None:
    handoff = roadmap.get("model_plane_handoff")
    if not isinstance(handoff, dict):
        errors.append("model_plane_handoff must be an object")
        return
    for field in ("source", "artifact", "planned_stage", "contract"):
        require_string(handoff, field, errors, context="model_plane_handoff")
    if handoff.get("planned_stage") != "harness_run_request":
        errors.append("model_plane_handoff.planned_stage must be 'harness_run_request'")
    require_string_list(handoff, "safety", errors, context="model_plane_handoff")


def validate_roadmap(roadmap: JSONDict) -> list[str]:
    errors: list[str] = []
    if roadmap.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        errors.append(
            "schema_version must be "
            f"{SUPPORTED_SCHEMA_VERSION!r}, got {roadmap.get('schema_version')!r}"
        )
    require_string(roadmap, "name", errors, context="roadmap")
    require_string(roadmap, "generated_for", errors, context="roadmap")
    validate_truth_table(roadmap, errors)
    validate_phases(roadmap, errors)
    validate_model_plane_handoff(roadmap, errors)
    validate_actuator_spike_checklist(roadmap, errors)
    require_string_list(roadmap, "risks", errors, context="roadmap")
    require_string_list(roadmap, "non_goals", errors, context="roadmap")
    return errors


def current_phase(phases: list[JSONDict]) -> JSONDict | None:
    for phase in phases:
        if phase.get("status") in {"in_progress", "blocked"}:
            return phase
    for phase in phases:
        if phase.get("status") == "planned":
            return phase
    return phases[-1] if phases else None


def build_summary(roadmap: JSONDict, path: Path) -> JSONDict:
    errors = validate_roadmap(roadmap)
    phases = roadmap.get("phases") if isinstance(roadmap.get("phases"), list) else []
    phase = current_phase([item for item in phases if isinstance(item, dict)])
    return {
        "mode": "expert_paging_roadmap_plan",
        "roadmap_path": str(path),
        "valid": not errors,
        "errors": errors,
        "schema_version": roadmap.get("schema_version"),
        "current_phase": phase,
        "phase_statuses": [
            {
                "id": item.get("id"),
                "title": item.get("title"),
                "status": item.get("status"),
            }
            for item in phases
            if isinstance(item, dict)
        ],
        "planned_stage": (
            roadmap.get("model_plane_handoff", {}).get("planned_stage")
            if isinstance(roadmap.get("model_plane_handoff"), dict)
            else None
        ),
        "safety_contract": [
            "planner does not start containers or model servers",
            "planner does not download models",
            "planner does not authenticate or inspect private tokens",
            "planner does not run Docker",
            "planner does not send prompt traffic",
        ],
    }


def print_human_summary(summary: JSONDict) -> None:
    print("MoE Run Anyway expert-paging roadmap")
    print(f"Roadmap: {summary['roadmap_path']}")
    print(f"Valid: {summary['valid']}")
    if summary["errors"]:
        print("Errors:")
        for error in summary["errors"]:
            print(f"  - {error}")
        return

    phase = summary["current_phase"]
    if phase:
        print(f"Current phase: {phase['id']} - {phase['title']} ({phase['status']})")
        print("Evidence gates:")
        for gate in phase["evidence_gates"]:
            print(f"  - {gate}")
        print("Next actions:")
        for action in phase["next_actions"]:
            print(f"  - {action}")
    print(f"Planned handoff stage: {summary['planned_stage']}")
    print("Safety contract:")
    for item in summary["safety_contract"]:
        print(f"  - {item}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roadmap_path",
        nargs="?",
        type=Path,
        default=DEFAULT_ROADMAP_PATH,
        help="roadmap JSON path",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    return parser


def plan_roadmap_path(path: Path) -> tuple[int, JSONDict | None, str | None]:
    try:
        roadmap = load_roadmap(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not load roadmap: {exc}"

    summary = build_summary(roadmap, path)
    return (0 if summary["valid"] else 2), summary, None


def main_from_test_path(path: Path) -> int:
    status, _, _ = plan_roadmap_path(path)
    return status


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary, error_message = plan_roadmap_path(args.roadmap_path)
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
