#!/usr/bin/env python3
"""Validate Phase 3 live residency and cleanup proof metadata.

This planner makes the live-residency blocker inspectable. It validates an
already-captured proof artifact for residency observation, residency control,
cleanup/restore, and artifact export. It does not launch runtimes, run Docker,
load tensor values, mutate residency, or send prompt traffic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_SCHEMA_VERSION = "moe-phase3-live-capability-proof-v1"
SUPPORTED_STATUSES = {"available", "partial", "planned", "missing"}
SUPPORTED_ACTIONS = {
    "observe",
    "preload",
    "pin",
    "evict",
    "demote",
    "fallback_dense",
    "restore_dense",
    "abort_run",
}
CONTROL_ACTIONS = {"preload", "pin", "evict", "demote"}
REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "name",
    "model_id",
    "backend_family",
    "prompt_family",
    "source_bundle_path",
    "proof_scope",
    "residency_observation",
    "residency_control",
    "cleanup_restore",
    "artifact_export",
    "safety_contract",
    "next_actions",
}
REQUIRED_OBSERVATION_FIELDS = {"status", "before_state_captured", "after_state_captured", "evidence_fields"}
REQUIRED_CONTROL_FIELDS = {"status", "supported_actions", "actuator_boundary", "dry_run_only"}
REQUIRED_CLEANUP_FIELDS = {"status", "restore_verified", "cleanup_actions", "failure_path_tested"}
REQUIRED_EXPORT_FIELDS = {"status", "artifact_paths"}

JSONDict = dict[str, Any]


def load_artifact(path: Path) -> JSONDict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("live capability proof root must be a JSON object")
    return data


def display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_repo_path(value: Any) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def normalized_path_text(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().replace("\\", "/")


def resolved_path_text(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def path_matches(value: Any, expected_path: Path | None) -> bool:
    if expected_path is None:
        return False
    actual = normalized_path_text(value)
    if actual is None:
        return False
    expected_display = normalized_path_text(display_path(expected_path))
    if expected_display is not None and actual == expected_display:
        return True
    actual_path = resolve_repo_path(actual)
    if actual_path is None:
        return False
    return resolved_path_text(actual_path) == resolved_path_text(expected_path)


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
    strings: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            errors.append(f"{context}.{field}[{index}] must be a non-empty string")
        else:
            strings.append(item)
    return strings


def validate_status(section: JSONDict, errors: list[str], *, context: str) -> str | None:
    status = section.get("status")
    if status not in SUPPORTED_STATUSES:
        errors.append(f"{context}.status has unsupported value {status!r}")
        return None
    return str(status)


def validate_observation(section: Any, errors: list[str]) -> JSONDict:
    context = "residency_observation"
    if not isinstance(section, dict):
        errors.append(f"{context} must be an object")
        return {"ready": False, "blockers": ["residency_observation_section_invalid"]}
    missing = sorted(REQUIRED_OBSERVATION_FIELDS - set(section))
    if missing:
        errors.append(f"{context} missing required fields: {', '.join(missing)}")
    status = validate_status(section, errors, context=context)
    require_bool(section, "before_state_captured", errors, context=context)
    require_bool(section, "after_state_captured", errors, context=context)
    evidence_fields = require_string_list(section, "evidence_fields", errors, context=context)
    blockers: list[str] = []
    if status != "available":
        blockers.append("residency_observation_not_available")
    if section.get("before_state_captured") is not True:
        blockers.append("before_residency_state_missing")
    if section.get("after_state_captured") is not True:
        blockers.append("after_residency_state_missing")
    if not evidence_fields:
        blockers.append("residency_observation_fields_missing")
    return {"ready": not blockers, "blockers": blockers}


def validate_control(section: Any, errors: list[str]) -> JSONDict:
    context = "residency_control"
    if not isinstance(section, dict):
        errors.append(f"{context} must be an object")
        return {"ready": False, "blockers": ["residency_control_section_invalid"]}
    missing = sorted(REQUIRED_CONTROL_FIELDS - set(section))
    if missing:
        errors.append(f"{context} missing required fields: {', '.join(missing)}")
    status = validate_status(section, errors, context=context)
    require_string(section, "actuator_boundary", errors, context=context)
    require_bool(section, "dry_run_only", errors, context=context)
    supported_actions = require_string_list(section, "supported_actions", errors, context=context)
    unsupported_actions = sorted(set(supported_actions) - SUPPORTED_ACTIONS)
    for action in unsupported_actions:
        errors.append(f"{context}.supported_actions has unsupported value {action!r}")
    blockers: list[str] = []
    if status != "available":
        blockers.append("residency_control_not_available")
    if not (CONTROL_ACTIONS & set(supported_actions)):
        blockers.append("residency_control_actions_missing")
    if section.get("dry_run_only") is True:
        blockers.append("residency_control_is_dry_run_only")
    return {"ready": not blockers, "blockers": blockers}


def validate_cleanup(section: Any, errors: list[str]) -> JSONDict:
    context = "cleanup_restore"
    if not isinstance(section, dict):
        errors.append(f"{context} must be an object")
        return {"ready": False, "blockers": ["cleanup_restore_section_invalid"]}
    missing = sorted(REQUIRED_CLEANUP_FIELDS - set(section))
    if missing:
        errors.append(f"{context} missing required fields: {', '.join(missing)}")
    status = validate_status(section, errors, context=context)
    require_bool(section, "restore_verified", errors, context=context)
    require_bool(section, "failure_path_tested", errors, context=context)
    cleanup_actions = require_string_list(section, "cleanup_actions", errors, context=context)
    blockers: list[str] = []
    if status != "available":
        blockers.append("cleanup_restore_not_available")
    if section.get("restore_verified") is not True:
        blockers.append("cleanup_restore_not_verified")
    if section.get("failure_path_tested") is not True:
        blockers.append("cleanup_failure_path_not_tested")
    if not cleanup_actions:
        blockers.append("cleanup_actions_missing")
    return {"ready": not blockers, "blockers": blockers}


def path_exists(value: Any) -> bool:
    path = resolve_repo_path(value)
    return path is not None and path.exists()


def validate_export(section: Any, errors: list[str]) -> JSONDict:
    context = "artifact_export"
    if not isinstance(section, dict):
        errors.append(f"{context} must be an object")
        return {"ready": False, "blockers": ["artifact_export_section_invalid"]}
    missing = sorted(REQUIRED_EXPORT_FIELDS - set(section))
    if missing:
        errors.append(f"{context} missing required fields: {', '.join(missing)}")
    status = validate_status(section, errors, context=context)
    artifact_paths = require_string_list(section, "artifact_paths", errors, context=context)
    missing_artifact_paths: list[str] = []
    if status == "available":
        for index, artifact_path in enumerate(artifact_paths):
            if not path_exists(artifact_path):
                missing_artifact_paths.append(artifact_path)
                errors.append(f"{context}.artifact_paths[{index}] must point to an existing file")
    blockers: list[str] = []
    if status != "available":
        blockers.append("artifact_export_not_available")
    if not artifact_paths:
        blockers.append("artifact_paths_missing")
    if missing_artifact_paths:
        blockers.append("artifact_export_paths_missing")
    return {
        "ready": not blockers,
        "blockers": blockers,
        "artifact_path_count": len(artifact_paths),
        "artifact_path_exists_count": len(artifact_paths) - len(missing_artifact_paths),
        "missing_artifact_paths": missing_artifact_paths,
    }


def validate_expected_context(
    artifact: JSONDict,
    errors: list[str],
    *,
    expected_model_id: str | None = None,
    expected_backend_family: str | None = None,
    expected_prompt_family: str | None = None,
    expected_source_bundle_path: Path | None = None,
) -> JSONDict:
    checks = {
        "model_id": expected_model_id,
        "backend_family": expected_backend_family,
        "prompt_family": expected_prompt_family,
    }
    matched: JSONDict = {}
    for field, expected in checks.items():
        if expected is None:
            matched[f"{field}_matches"] = None
            continue
        actual = artifact.get(field)
        field_matches = actual == expected
        matched[f"{field}_matches"] = field_matches
        if not field_matches:
            errors.append(f"artifact.{field} must match expected {field} {expected!r}")

    source_bundle_path = resolve_repo_path(artifact.get("source_bundle_path"))
    source_bundle_path_exists = source_bundle_path is not None and source_bundle_path.exists()
    matched["source_bundle_path_exists"] = source_bundle_path_exists
    if not source_bundle_path_exists:
        errors.append("artifact.source_bundle_path must point to an existing file")
    if expected_source_bundle_path is None:
        matched["source_bundle_path_matches"] = None
    else:
        bundle_matches = path_matches(artifact.get("source_bundle_path"), expected_source_bundle_path)
        matched["source_bundle_path_matches"] = bundle_matches
        if not bundle_matches:
            errors.append("artifact.source_bundle_path must match expected source bundle path")
    return matched


def validate_artifact(
    artifact: JSONDict,
    *,
    expected_model_id: str | None = None,
    expected_backend_family: str | None = None,
    expected_prompt_family: str | None = None,
    expected_source_bundle_path: Path | None = None,
) -> tuple[list[str], JSONDict]:
    errors: list[str] = []
    missing = sorted(REQUIRED_TOP_LEVEL_FIELDS - set(artifact))
    if missing:
        errors.append(f"artifact missing required fields: {', '.join(missing)}")
    if artifact.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        errors.append(
            "schema_version must be "
            f"{SUPPORTED_SCHEMA_VERSION!r}, got {artifact.get('schema_version')!r}"
        )
    for field in ("name", "model_id", "backend_family", "prompt_family", "source_bundle_path", "proof_scope"):
        require_string(artifact, field, errors, context="artifact")
    context_binding = validate_expected_context(
        artifact,
        errors,
        expected_model_id=expected_model_id,
        expected_backend_family=expected_backend_family,
        expected_prompt_family=expected_prompt_family,
        expected_source_bundle_path=expected_source_bundle_path,
    )
    require_string_list(artifact, "safety_contract", errors, context="artifact")
    require_string_list(artifact, "next_actions", errors, context="artifact")

    section_results = {
        "residency_observation": validate_observation(artifact.get("residency_observation"), errors),
        "residency_control": validate_control(artifact.get("residency_control"), errors),
        "cleanup_restore": validate_cleanup(artifact.get("cleanup_restore"), errors),
        "artifact_export": validate_export(artifact.get("artifact_export"), errors),
    }
    section_results["context_binding"] = {"ready": not any(value is False for value in context_binding.values()), "blockers": []}
    if any(value is False for value in context_binding.values()):
        section_results["context_binding"]["blockers"] = ["live_proof_context_mismatch"]
    section_results["context_binding"]["checks"] = context_binding
    return errors, section_results


def missing_artifact_summary(path: Path | None) -> JSONDict:
    return {
        "mode": "phase3_live_capability_proof_plan",
        "proof_artifact_path": str(path) if path is not None else None,
        "valid": True,
        "errors": [],
        "proof_available": False,
        "proof_ready": False,
        "section_status": {},
        "blockers": [
            {
                "id": "live_capability_proof_artifact_missing",
                "description": "No live residency observation/control and cleanup/restore proof artifact was provided.",
                "required_artifact_schema": SUPPORTED_SCHEMA_VERSION,
            }
        ],
        "phase_3_gate": {
            "live_residency_observation_and_control_ready": False,
            "cleanup_restore_proof_ready": False,
            "ready_for_live_spike": False,
        },
        "safety_contract": safety_contract(),
        "next_actions": [
            "Capture before/after residency state from a selected backend adapter after approval.",
            "Record which residency actions were available and whether they were dry-run only.",
            "Record cleanup/restore and failure-path proof before treating live mutation as safe.",
        ],
    }


def safety_contract() -> list[str]:
    return [
        "planner validates local proof metadata only",
        "planner does not launch model servers",
        "planner does not run Docker",
        "planner does not load tensor values",
        "planner does not mutate runtime residency",
        "planner does not send prompt traffic",
        "planner does not claim live expert paging",
    ]


def section_status_summary(section_results: dict[str, JSONDict]) -> JSONDict:
    summary: JSONDict = {}
    detail_keys = ("artifact_path_count", "artifact_path_exists_count", "missing_artifact_paths")
    for section_id, result in section_results.items():
        item: JSONDict = {
            "ready": result.get("ready") is True,
            "blockers": result.get("blockers", []),
        }
        for detail_key in detail_keys:
            if detail_key in result:
                item[detail_key] = result[detail_key]
        summary[section_id] = item
    return summary


def build_summary(
    path: Path | None,
    *,
    expected_model_id: str | None = None,
    expected_backend_family: str | None = None,
    expected_prompt_family: str | None = None,
    expected_source_bundle_path: Path | None = None,
) -> JSONDict:
    if path is None:
        return missing_artifact_summary(path)

    artifact = load_artifact(path)
    errors, section_results = validate_artifact(
        artifact,
        expected_model_id=expected_model_id,
        expected_backend_family=expected_backend_family,
        expected_prompt_family=expected_prompt_family,
        expected_source_bundle_path=expected_source_bundle_path,
    )
    blockers = [
        {"id": blocker_id, "section": section_id}
        for section_id, result in section_results.items()
        for blocker_id in result.get("blockers", [])
    ]
    proof_ready = not errors and not blockers
    return {
        "mode": "phase3_live_capability_proof_plan",
        "proof_artifact_path": str(path),
        "valid": not errors,
        "errors": errors,
        "proof_available": True,
        "proof_ready": proof_ready,
        "schema_version": artifact.get("schema_version"),
        "name": artifact.get("name"),
        "model_id": artifact.get("model_id"),
        "backend_family": artifact.get("backend_family"),
        "prompt_family": artifact.get("prompt_family"),
        "source_bundle_path": artifact.get("source_bundle_path"),
        "proof_scope": artifact.get("proof_scope"),
        "context_binding": section_results.get("context_binding", {}),
        "section_status": section_status_summary(section_results),
        "blockers": blockers,
        "phase_3_gate": {
            "live_residency_observation_and_control_ready": (
                section_results["residency_observation"].get("ready") is True
                and section_results["residency_control"].get("ready") is True
            ),
            "cleanup_restore_proof_ready": section_results["cleanup_restore"].get("ready") is True,
            "ready_for_live_spike": proof_ready,
        },
        "safety_contract": safety_contract(),
        "next_actions": [
            "Use this proof with go/no-go and bundle evidence before a guarded live spike.",
            "Keep proof paired to the same backend, model, prompt family, and adapter boundary as the replay bundle.",
        ],
    }


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def blocker_text(blockers: Any) -> str:
    if not isinstance(blockers, list) or not blockers:
        return "none"
    values: list[str] = []
    for blocker in blockers:
        if isinstance(blocker, dict):
            blocker_id = blocker.get("id") or "unknown"
            section = blocker.get("section")
            values.append(f"{section}:{blocker_id}" if section else str(blocker_id))
        else:
            values.append(str(blocker))
    return ", ".join(values)


def format_markdown_report(summary: JSONDict) -> str:
    phase_gate = summary.get("phase_3_gate") if isinstance(summary.get("phase_3_gate"), dict) else {}
    context_binding = summary.get("context_binding") if isinstance(summary.get("context_binding"), dict) else {}
    context_checks = context_binding.get("checks") if isinstance(context_binding.get("checks"), dict) else {}
    section_status = summary.get("section_status") if isinstance(summary.get("section_status"), dict) else {}
    lines = [
        "# Phase 3 Live Capability Proof",
        "",
        f"- Artifact: `{markdown_escape(summary.get('proof_artifact_path'))}`",
        f"- Valid: `{summary.get('valid')}`",
        f"- Proof available: `{summary.get('proof_available')}`",
        f"- Proof ready: `{summary.get('proof_ready')}`",
        f"- Model: `{markdown_escape(summary.get('model_id', 'unknown'))}`",
        f"- Backend: `{markdown_escape(summary.get('backend_family', 'unknown'))}`",
        f"- Prompt family: `{markdown_escape(summary.get('prompt_family', 'unknown'))}`",
        f"- Source bundle: `{markdown_escape(summary.get('source_bundle_path', 'unknown'))}`",
        f"- Live residency observation/control ready: `{phase_gate.get('live_residency_observation_and_control_ready')}`",
        f"- Cleanup/restore proof ready: `{phase_gate.get('cleanup_restore_proof_ready')}`",
        f"- Ready for live spike: `{phase_gate.get('ready_for_live_spike')}`",
        "",
        "## Required Proof Sections",
        "",
        "- `residency_observation`: before/after residency state and evidence fields",
        "- `residency_control`: non-dry-run preload, pin, evict, or demote boundary",
        "- `cleanup_restore`: verified dense/full-residency restore and failure-path cleanup",
        "- `artifact_export`: saved proof artifacts that can be attached to a bundle",
    ]
    if section_status:
        lines.extend(
            [
                "",
                "## Section Status",
                "",
                "| Section | Ready | Blockers |",
                "| --- | --- | --- |",
            ]
        )
        for section, result in section_status.items():
            if not isinstance(result, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    markdown_escape(value)
                    for value in (
                        section,
                        result.get("ready", "unknown"),
                        blocker_text(result.get("blockers")),
                    )
                )
                + " |"
            )
    else:
        lines.extend(["", "## Section Status", "", "No proof section status is available until a proof artifact is supplied."])
    if context_binding:
        lines.extend(
            [
                "",
                "## Context Binding",
                "",
                f"- Ready: `{context_binding.get('ready')}`",
            ]
        )
        for check, value in context_checks.items():
            lines.append(f"- `{markdown_escape(check)}`: `{value}`")
    blockers = summary.get("blockers") if isinstance(summary.get("blockers"), list) else []
    if blockers:
        lines.extend(["", "## Blockers", ""])
        for blocker in blockers:
            if isinstance(blocker, dict):
                section = blocker.get("section")
                suffix = f" ({section})" if section else ""
                lines.append(f"- `{markdown_escape(blocker.get('id') or 'unknown')}`{suffix}")
            else:
                lines.append(f"- `{markdown_escape(blocker)}`")
    next_actions = summary.get("next_actions") if isinstance(summary.get("next_actions"), list) else []
    if next_actions:
        lines.extend(["", "## Next Actions", ""])
        for action in next_actions:
            lines.append(f"- {markdown_escape(action)}")
    lines.extend(["", "## Safety Contract", ""])
    for item in summary.get("safety_contract", []):
        lines.append(f"- {markdown_escape(item)}")
    return "\n".join(lines) + "\n"


def write_markdown_report(summary: JSONDict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_markdown_report(summary), encoding="utf-8")

def print_human_summary(summary: JSONDict) -> None:
    print("MoE Run Anyway Phase 3 live capability proof")
    print(f"Artifact: {summary['proof_artifact_path']}")
    print(f"Valid: {summary['valid']}")
    print(f"Proof available: {summary['proof_available']}")
    print(f"Proof ready: {summary['proof_ready']}")
    if summary["errors"]:
        print("Errors:")
        for error in summary["errors"]:
            print(f"  - {error}")
    if summary["blockers"]:
        print("Blockers:")
        for blocker in summary["blockers"]:
            print(f"  - {blocker['id']}")
    print("Safety contract:")
    for item in summary["safety_contract"]:
        print(f"  - {item}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proof_artifact_path", nargs="?", type=Path, default=None)
    parser.add_argument("--expected-model-id")
    parser.add_argument("--expected-backend-family")
    parser.add_argument("--expected-prompt-family")
    parser.add_argument("--expected-source-bundle-path", type=Path)
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    parser.add_argument("--output-md", type=Path, help="write a Markdown live-capability proof handoff report")
    return parser


def plan_path(path: Path | None = None, **kwargs: Any) -> tuple[int, JSONDict | None, str | None]:
    try:
        summary = build_summary(path, **kwargs)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not build Phase 3 live capability proof plan: {exc}"
    return (0 if summary["valid"] else 2), summary, None


def main_from_test_path(path: Path | None = None) -> int:
    status, _, _ = plan_path(path)
    return status


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary, error_message = plan_path(
        args.proof_artifact_path,
        expected_model_id=args.expected_model_id,
        expected_backend_family=args.expected_backend_family,
        expected_prompt_family=args.expected_prompt_family,
        expected_source_bundle_path=args.expected_source_bundle_path,
    )
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