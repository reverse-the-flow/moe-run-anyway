#!/usr/bin/env python3
"""Plan or validate Phase 3 dense/full-runtime fallback comparison evidence.

The fallback comparison is the last audit step before a replay policy can be
treated as more than an offline suggestion. This planner never launches a model
or sends prompt traffic. With no fallback artifact, it records the exact blocker.
With an artifact, it validates the comparison shape and summarizes coverage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_SCHEMA_VERSION = "moe-dense-fallback-comparison-v1"
SUPPORTED_QUALITY_LABELS = {"same", "minor_delta", "major_delta", "unknown"}
REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "model_id",
    "prompt_family",
    "managed_policy_id",
    "managed_artifact",
    "dense_artifact",
    "comparisons",
}
REQUIRED_COMPARISON_FIELDS = {
    "prompt_id",
    "managed_output_present",
    "dense_output_present",
    "quality_delta_label",
}

JSONDict = dict[str, Any]


def load_artifact(path: Path) -> JSONDict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("fallback comparison root must be a JSON object")
    return data


def require_string(obj: JSONDict, field: str, errors: list[str], *, context: str) -> None:
    value = obj.get(field)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{context}.{field} must be a non-empty string")


def require_bool(obj: JSONDict, field: str, errors: list[str], *, context: str) -> None:
    if not isinstance(obj.get(field), bool):
        errors.append(f"{context}.{field} must be a boolean")


def validate_comparison_artifact(artifact: JSONDict) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_TOP_LEVEL_FIELDS - set(artifact))
    if missing:
        errors.append(f"artifact missing required fields: {', '.join(missing)}")
    if artifact.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        errors.append(
            "schema_version must be "
            f"{SUPPORTED_SCHEMA_VERSION!r}, got {artifact.get('schema_version')!r}"
        )
    for field in ("model_id", "prompt_family", "managed_policy_id", "managed_artifact", "dense_artifact"):
        require_string(artifact, field, errors, context="artifact")

    comparisons = artifact.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        errors.append("artifact.comparisons must be a non-empty list")
        return errors

    seen_prompt_ids: set[str] = set()
    for index, comparison in enumerate(comparisons):
        context = f"comparisons[{index}]"
        if not isinstance(comparison, dict):
            errors.append(f"{context} must be an object")
            continue
        missing_comparison = sorted(REQUIRED_COMPARISON_FIELDS - set(comparison))
        if missing_comparison:
            errors.append(f"{context} missing required fields: {', '.join(missing_comparison)}")
        require_string(comparison, "prompt_id", errors, context=context)
        require_bool(comparison, "managed_output_present", errors, context=context)
        require_bool(comparison, "dense_output_present", errors, context=context)
        quality_delta = comparison.get("quality_delta_label")
        if quality_delta not in SUPPORTED_QUALITY_LABELS:
            errors.append(f"{context}.quality_delta_label has unsupported value {quality_delta!r}")
        prompt_id = comparison.get("prompt_id")
        if isinstance(prompt_id, str):
            if prompt_id in seen_prompt_ids:
                errors.append(f"{context}.prompt_id duplicates {prompt_id!r}")
            seen_prompt_ids.add(prompt_id)
    return errors


def comparison_provenance_summary(artifact: JSONDict) -> JSONDict:
    blockers: list[str] = []
    builder = artifact.get("builder")
    input_receipts: JSONDict = {}
    builder_mode: str | None = None
    if not isinstance(builder, dict):
        blockers.append("artifact.builder must be an object with input_receipts")
    else:
        mode = builder.get("mode")
        builder_mode = mode if isinstance(mode, str) else None
        if builder_mode != "paired_output_summary":
            blockers.append("artifact.builder.mode must be paired_output_summary for ready fallback evidence")
        receipts = builder.get("input_receipts")
        if not isinstance(receipts, dict):
            blockers.append("artifact.builder.input_receipts must be an object")
        else:
            input_receipts = receipts
            for field in ("managed_capture_receipt_ready", "dense_capture_receipt_ready", "receipt_pair_consistent"):
                if receipts.get(field) is not True:
                    blockers.append(f"artifact.builder.input_receipts.{field} must be true")
            if receipts.get("receipt_gate") != "required_before_write":
                blockers.append("artifact.builder.input_receipts.receipt_gate must be required_before_write")
    return {
        "comparison_provenance_ready": not blockers,
        "provenance_blockers": blockers,
        "builder_mode": builder_mode,
        "input_receipts": input_receipts,
    }


def comparison_counts(artifact: JSONDict) -> JSONDict:
    comparisons = artifact.get("comparisons")
    rows = comparisons if isinstance(comparisons, list) else []
    quality_counts = {label: 0 for label in sorted(SUPPORTED_QUALITY_LABELS)}
    missing_managed = 0
    missing_dense = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = row.get("quality_delta_label")
        if isinstance(label, str) and label in quality_counts:
            quality_counts[label] += 1
        if row.get("managed_output_present") is not True:
            missing_managed += 1
        if row.get("dense_output_present") is not True:
            missing_dense += 1
    return {
        "comparison_count": len(rows),
        "quality_delta_counts": quality_counts,
        "missing_managed_output_count": missing_managed,
        "missing_dense_output_count": missing_dense,
    }


def missing_artifact_summary(fallback_artifact_path: Path | None) -> JSONDict:
    return {
        "mode": "dense_fallback_comparison_plan",
        "fallback_artifact_path": str(fallback_artifact_path) if fallback_artifact_path is not None else None,
        "valid": True,
        "errors": [],
        "comparison_available": False,
        "comparison_ready": False,
        "comparison_provenance_ready": False,
        "provenance_blockers": [],
        "builder_mode": None,
        "input_receipts": {},
        "blocker": {
            "id": "fallback_output_missing_for_quality_comparison",
            "description": "No dense or full-runtime fallback comparison artifact was provided.",
            "required_artifact_schema": SUPPORTED_SCHEMA_VERSION,
            "required_command_class": "approved_dense_or_full_runtime_baseline_run",
        },
        "phase_3_gate": {
            "dense_fallback_comparison_ready": False,
            "ready_for_live_spike": False,
        },
        "safety_contract": safety_contract(),
        "next_actions": [
            "Capture dense or full-runtime output for the same prompt set after approval.",
            "Validate the saved comparison artifact with this planner.",
            "Keep no_live_actuator as the decision until fallback behavior is bounded.",
        ],
    }


def safety_contract() -> list[str]:
    return [
        "planner validates local comparison metadata only",
        "planner does not launch model servers",
        "planner does not send prompt traffic",
        "planner does not inspect private tokens",
        "planner requires ready input capture-receipt provenance before comparison_ready",
        "planner does not mutate runtime residency",
        "planner does not claim live expert paging",
    ]


def build_summary(fallback_artifact_path: Path | None) -> JSONDict:
    if fallback_artifact_path is None:
        return missing_artifact_summary(fallback_artifact_path)

    artifact = load_artifact(fallback_artifact_path)
    errors = validate_comparison_artifact(artifact)
    provenance = comparison_provenance_summary(artifact)
    counts = comparison_counts(artifact)
    comparison_shape_ready = (
        not errors
        and counts["comparison_count"] > 0
        and counts["missing_managed_output_count"] == 0
        and counts["missing_dense_output_count"] == 0
        and counts["quality_delta_counts"].get("major_delta", 0) == 0
        and counts["quality_delta_counts"].get("unknown", 0) == 0
    )
    comparison_ready = comparison_shape_ready and provenance["comparison_provenance_ready"]
    if comparison_ready:
        blocker = None
    elif not provenance["comparison_provenance_ready"]:
        blocker = {
            "id": "fallback_comparison_missing_capture_receipt_provenance",
            "description": "Fallback comparison artifact lacks ready managed/dense capture-receipt provenance.",
        }
    else:
        blocker = {
            "id": "fallback_comparison_not_ready",
            "description": "Fallback comparison artifact exists but does not yet bound behavior well enough for a live spike.",
        }
    return {
        "mode": "dense_fallback_comparison_plan",
        "fallback_artifact_path": str(fallback_artifact_path),
        "valid": not errors,
        "errors": errors,
        "comparison_available": True,
        "comparison_shape_ready": comparison_shape_ready,
        "comparison_provenance_ready": provenance["comparison_provenance_ready"],
        "provenance_blockers": provenance["provenance_blockers"],
        "builder_mode": provenance["builder_mode"],
        "input_receipts": provenance["input_receipts"],
        "comparison_ready": comparison_ready,
        "schema_version": artifact.get("schema_version"),
        "model_id": artifact.get("model_id"),
        "prompt_family": artifact.get("prompt_family"),
        "managed_policy_id": artifact.get("managed_policy_id"),
        "managed_artifact": artifact.get("managed_artifact"),
        "dense_artifact": artifact.get("dense_artifact"),
        **counts,
        "blocker": blocker,
        "phase_3_gate": {
            "dense_fallback_comparison_ready": comparison_ready,
            "ready_for_live_spike": False,
        },
        "safety_contract": safety_contract(),
        "next_actions": [
            "Use this comparison with replay metrics when choosing no-go or a guarded backend spike.",
            "Keep dense fallback output paired to the same prompt set and model target.",
            "Treat comparison artifacts without ready input receipts as not Phase 3-ready.",
        ],
    }


def print_human_summary(summary: JSONDict) -> None:
    print("MoE Run Anyway dense fallback comparison")
    print(f"Artifact: {summary['fallback_artifact_path']}")
    print(f"Valid: {summary['valid']}")
    print(f"Comparison available: {summary['comparison_available']}")
    print(f"Comparison ready: {summary['comparison_ready']}")
    if summary["errors"]:
        print("Errors:")
        for error in summary["errors"]:
            print(f"  - {error}")
    if summary.get("blocker"):
        print(f"Blocker: {summary['blocker']['id']}")
    if summary.get("comparison_count") is not None:
        print(f"Comparisons: {summary['comparison_count']}")
        print(f"Missing dense outputs: {summary['missing_dense_output_count']}")
        print(f"Missing managed outputs: {summary['missing_managed_output_count']}")
    print("Safety contract:")
    for item in summary["safety_contract"]:
        print(f"  - {item}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fallback_artifact_path", nargs="?", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    return parser


def plan_path(fallback_artifact_path: Path | None = None) -> tuple[int, JSONDict | None, str | None]:
    try:
        summary = build_summary(fallback_artifact_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not build dense fallback comparison plan: {exc}"
    return (0 if summary["valid"] else 2), summary, None


def main_from_test_path(fallback_artifact_path: Path | None = None) -> int:
    status, _, _ = plan_path(fallback_artifact_path)
    return status


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary, error_message = plan_path(args.fallback_artifact_path)
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
