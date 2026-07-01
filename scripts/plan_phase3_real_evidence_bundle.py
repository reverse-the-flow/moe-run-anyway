#!/usr/bin/env python3
"""Validate a Phase 3 real-evidence bundle manifest.

The bundle manifest binds the trace, scanner-derived inventory, policies,
managed-loading plan, optional candidate trace capture receipt, optional dense
fallback comparison artifact, and optional live capability proof artifact into one auditable unit. This validator reads
local files and reuses the Phase 3 evidence
packet; it does not launch runtimes, download models, inspect secrets, mutate
residency, or send prompt traffic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import plan_phase3_evidence_packet


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_PATH = ROOT / "memory-moe-mvp" / "data" / "phase3_real_evidence_bundle.fixture.json"
SUPPORTED_SCHEMA_VERSION = "moe-phase3-real-evidence-bundle-v1"
REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "name",
    "model_id",
    "source_format",
    "backend_family",
    "prompt_family",
    "artifact_paths",
    "approvals",
    "safety_contract",
    "next_actions",
}
REQUIRED_ARTIFACT_PATH_FIELDS = {
    "trace_path",
    "inventory_path",
    "policies_path",
    "managed_plan_path",
}
OPTIONAL_ARTIFACT_PATH_FIELDS = {
    "trace_receipt_path",
    "fallback_artifact_path",
    "live_proof_artifact_path",
}
KNOWN_ARTIFACT_PATH_FIELDS = REQUIRED_ARTIFACT_PATH_FIELDS | OPTIONAL_ARTIFACT_PATH_FIELDS
REQUIRED_APPROVAL_FIELDS = {
    "real_model_trace_capture_approved",
    "dense_fallback_capture_approved",
    "runtime_prompt_traffic_approved",
}

JSONDict = dict[str, Any]


def load_manifest(path: Path) -> JSONDict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("bundle manifest root must be a JSON object")
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
    if path.is_absolute():
        return path
    return ROOT / path


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


def validate_manifest_shape(manifest: JSONDict) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_TOP_LEVEL_FIELDS - set(manifest))
    if missing:
        errors.append(f"manifest missing required top-level fields: {', '.join(missing)}")
    if manifest.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        errors.append(
            "schema_version must be "
            f"{SUPPORTED_SCHEMA_VERSION!r}, got {manifest.get('schema_version')!r}"
        )
    for field in ("name", "model_id", "source_format", "backend_family", "prompt_family"):
        require_string(manifest, field, errors, context="manifest")

    artifact_paths = manifest.get("artifact_paths")
    if not isinstance(artifact_paths, dict):
        errors.append("artifact_paths must be an object")
    else:
        missing_artifacts = sorted(REQUIRED_ARTIFACT_PATH_FIELDS - set(artifact_paths))
        if missing_artifacts:
            errors.append(f"artifact_paths missing required fields: {', '.join(missing_artifacts)}")
        for field in sorted(REQUIRED_ARTIFACT_PATH_FIELDS):
            require_string(artifact_paths, field, errors, context="artifact_paths")
        for field in sorted(OPTIONAL_ARTIFACT_PATH_FIELDS):
            value = artifact_paths.get(field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                errors.append(f"artifact_paths.{field} must be null or a non-empty string")

    approvals = manifest.get("approvals")
    if not isinstance(approvals, dict):
        errors.append("approvals must be an object")
    else:
        missing_approvals = sorted(REQUIRED_APPROVAL_FIELDS - set(approvals))
        if missing_approvals:
            errors.append(f"approvals missing required fields: {', '.join(missing_approvals)}")
        for field in REQUIRED_APPROVAL_FIELDS:
            if field in approvals and not isinstance(approvals.get(field), bool):
                errors.append(f"approvals.{field} must be a boolean")
        if "notes" in approvals:
            require_string_list(approvals, "notes", errors, context="approvals")

    require_string_list(manifest, "safety_contract", errors, context="manifest")
    require_string_list(manifest, "next_actions", errors, context="manifest")
    return errors


def artifact_statuses(manifest: JSONDict) -> tuple[dict[str, JSONDict], list[str], dict[str, Path | None]]:
    artifact_paths = manifest.get("artifact_paths") if isinstance(manifest.get("artifact_paths"), dict) else {}
    resolved: dict[str, Path | None] = {}
    statuses: dict[str, JSONDict] = {}
    errors: list[str] = []
    for field in sorted(KNOWN_ARTIFACT_PATH_FIELDS):
        path = resolve_repo_path(artifact_paths.get(field))
        resolved[field] = path
        required = field in REQUIRED_ARTIFACT_PATH_FIELDS
        if path is None:
            status = "not_provided"
            exists = False
            if required:
                errors.append(f"artifact_paths.{field} is required")
        else:
            exists = path.exists()
            status = "present" if exists else "missing"
            if required and not exists:
                errors.append(f"artifact_paths.{field} does not exist: {display_path(path)}")
            if not required and not exists:
                errors.append(f"artifact_paths.{field} does not exist: {display_path(path)}")
        statuses[field] = {
            "path": display_path(path),
            "required": required,
            "exists": exists,
            "status": status,
        }
    return statuses, errors, resolved


def reason_ids(packet: JSONDict | None) -> list[str]:
    if packet is None:
        return []
    reasons = packet.get("remaining_gaps", {}).get("no_go_reasons", [])
    if not isinstance(reasons, list):
        return []
    return [
        str(reason["id"])
        for reason in reasons
        if isinstance(reason, dict) and isinstance(reason.get("id"), str)
    ]


def build_summary(bundle_path: Path) -> JSONDict:
    manifest = load_manifest(bundle_path)
    shape_errors = validate_manifest_shape(manifest)
    statuses, path_errors, resolved = artifact_statuses(manifest)
    errors = [*shape_errors, *path_errors]
    required_paths_present = not path_errors and all(
        statuses[field]["exists"]
        for field in ("trace_path", "inventory_path", "policies_path", "managed_plan_path")
    )

    packet: JSONDict | None = None
    packet_error: str | None = None
    if required_paths_present:
        try:
            packet = plan_phase3_evidence_packet.build_packet_summary(
                resolved["trace_path"],  # type: ignore[arg-type]
                resolved["inventory_path"],  # type: ignore[arg-type]
                resolved["policies_path"],  # type: ignore[arg-type]
                resolved["managed_plan_path"],  # type: ignore[arg-type]
                fallback_artifact_path=resolved.get("fallback_artifact_path"),
                policy_candidate_trace_receipt_path=resolved.get("trace_receipt_path"),
                live_proof_artifact_path=resolved.get("live_proof_artifact_path"),
                include_repo_gates=False,
                expected_model_id=manifest.get("model_id"),
                expected_backend_family=manifest.get("backend_family"),
                expected_prompt_family=manifest.get("prompt_family"),
                expected_source_bundle_path=bundle_path,
            )
            if packet.get("valid") is not True:
                errors.extend(
                    f"phase3 evidence packet: {error}"
                    for error in packet.get("errors", [])
                    if isinstance(error, str)
                )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            packet_error = str(exc)
            errors.append(f"phase3 evidence packet failed: {exc}")

    valid = not errors
    remaining_gaps = packet.get("remaining_gaps", {}) if packet else {}
    bundle_ready_for_phase4 = bool(packet and packet.get("phase3_complete") is True)
    blockers: list[JSONDict] = []
    for error in errors:
        blockers.append({"id": "bundle_manifest_invalid", "description": error})
    if packet is not None:
        for reason in packet.get("remaining_gaps", {}).get("no_go_reasons", []):
            if isinstance(reason, dict) and reason.get("id"):
                blockers.append(reason)
    if statuses.get("fallback_artifact_path", {}).get("status") == "not_provided":
        blockers.append(
            {
                "id": "fallback_artifact_not_attached",
                "description": "Bundle does not attach a dense/full-runtime fallback comparison artifact.",
            }
        )

    return {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "mode": "phase3_real_evidence_bundle",
        "bundle_path": display_path(bundle_path),
        "valid": valid,
        "errors": errors,
        "packet_error": packet_error,
        "name": manifest.get("name"),
        "model_id": manifest.get("model_id"),
        "source_format": manifest.get("source_format"),
        "backend_family": manifest.get("backend_family"),
        "prompt_family": manifest.get("prompt_family"),
        "artifact_statuses": statuses,
        "approvals": manifest.get("approvals", {}),
        "packet_ready": packet.get("packet_ready") if packet else False,
        "phase3_complete": packet.get("phase3_complete") if packet else False,
        "bundle_ready_for_phase4": bundle_ready_for_phase4,
        "remaining_gaps": remaining_gaps,
        "no_go_reason_ids": reason_ids(packet),
        "blockers": blockers,
        "safety_contract": [
            "bundle validator reads local artifact manifests only",
            "bundle validator does not launch model servers",
            "bundle validator does not download models",
            "bundle validator does not inspect private tokens",
            "bundle validator does not mutate runtime residency",
            "bundle validator does not send prompt traffic",
            "bundle validator does not claim live expert paging",
        ],
        "next_actions": [
            "Keep the bundle valid-but-not-ready until real-model pairing, trace receipt, and fallback comparison are proven.",
            "Use this bundle as the Phase 3 handoff artifact before Phase 4 adapter work.",
            "Attach a live capability proof artifact before promoting an adapter-ready bundle to a live-spike bundle.",
            "Do not treat approvals metadata as permission to run; live runtime work still needs explicit user approval.",
        ],
    }


def print_human_summary(summary: JSONDict) -> None:
    print("MoE Run Anyway Phase 3 real-evidence bundle")
    print(f"Valid: {summary['valid']}")
    print(f"Bundle ready for Phase 4: {summary['bundle_ready_for_phase4']}")
    print(f"Phase 3 complete: {summary['phase3_complete']}")
    print("Artifacts:")
    for field, status in summary["artifact_statuses"].items():
        print(f"  - {field}: {status['status']} ({status['path']})")
    if summary["errors"]:
        print("Errors:")
        for error in summary["errors"]:
            print(f"  - {error}")
    if summary["no_go_reason_ids"]:
        print("No-go reasons:")
        for reason_id in summary["no_go_reason_ids"]:
            print(f"  - {reason_id}")
    print("Safety contract:")
    for item in summary["safety_contract"]:
        print(f"  - {item}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_path", nargs="?", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    return parser


def plan_path(bundle_path: Path = DEFAULT_BUNDLE_PATH) -> tuple[int, JSONDict | None, str | None]:
    try:
        summary = build_summary(bundle_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not build Phase 3 real-evidence bundle summary: {exc}"
    return (0 if summary["valid"] else 2), summary, None


def main_from_test_path(bundle_path: Path = DEFAULT_BUNDLE_PATH) -> int:
    status, _, _ = plan_path(bundle_path)
    return status


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary, error_message = plan_path(args.bundle_path)
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
