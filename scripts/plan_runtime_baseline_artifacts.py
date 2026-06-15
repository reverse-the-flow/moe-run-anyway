#!/usr/bin/env python3
"""Validate and summarize the Phase 1 runtime baseline artifact contract.

This planner is deliberately offline. It checks the expected artifact classes
and evidence labels for baseline runs, but it does not inspect endpoints, start
servers, download models, authenticate, run Docker, or send prompt traffic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT_PATH = ROOT / "memory-moe-mvp" / "data" / "runtime_baseline_artifact_contract.json"
SUPPORTED_SCHEMA_VERSION = "moe-runtime-baseline-artifact-contract-v1"
REQUIRED_ARTIFACT_CLASS_IDS = {
    "runtime_baseline_preflight_bundle",
    "runtime_baseline_probe_bundle",
}
REQUIRED_EVIDENCE_LABEL_IDS = {
    "runtime_readiness_evidence",
    "runtime_request_telemetry",
    "semantic_router_trace",
}
EVIDENCE_TYPES = {"runtime_evidence", "semantic_routing_evidence"}

JSONDict = dict[str, Any]


def load_contract(path: Path) -> JSONDict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("contract root must be a JSON object")
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


def validate_evidence_labels(contract: JSONDict, errors: list[str]) -> dict[str, JSONDict]:
    labels = contract.get("evidence_labels")
    if not isinstance(labels, list) or not labels:
        errors.append("evidence_labels must be a non-empty list")
        return {}

    by_id: dict[str, JSONDict] = {}
    for index, label in enumerate(labels):
        context = f"evidence_labels[{index}]"
        if not isinstance(label, dict):
            errors.append(f"{context} must be an object")
            continue
        require_string(label, "id", errors, context=context)
        require_string(label, "evidence_type", errors, context=context)
        require_string(label, "description", errors, context=context)
        require_bool(label, "may_claim_semantic_expert_ids", errors, context=context)
        if label.get("evidence_type") not in EVIDENCE_TYPES:
            errors.append(f"{context}.evidence_type has unsupported value {label.get('evidence_type')!r}")
        label_id = label.get("id")
        if isinstance(label_id, str):
            if label_id in by_id:
                errors.append(f"{context}.id duplicates {label_id!r}")
            by_id[label_id] = label

    missing = sorted(REQUIRED_EVIDENCE_LABEL_IDS - set(by_id))
    if missing:
        errors.append(f"evidence_labels missing required ids: {', '.join(missing)}")

    for label_id in ("runtime_readiness_evidence", "runtime_request_telemetry"):
        label = by_id.get(label_id)
        if label and label.get("may_claim_semantic_expert_ids") is not False:
            errors.append(f"evidence_labels[{label_id!r}] must not claim semantic expert ids")
    semantic = by_id.get("semantic_router_trace")
    if semantic and semantic.get("evidence_type") != "semantic_routing_evidence":
        errors.append("evidence_labels['semantic_router_trace'] must be semantic_routing_evidence")
    return by_id


def validate_artifact_classes(
    contract: JSONDict,
    evidence_labels: dict[str, JSONDict],
    errors: list[str],
) -> None:
    classes = contract.get("artifact_classes")
    if not isinstance(classes, list) or not classes:
        errors.append("artifact_classes must be a non-empty list")
        return

    seen: set[str] = set()
    for index, artifact_class in enumerate(classes):
        context = f"artifact_classes[{index}]"
        if not isinstance(artifact_class, dict):
            errors.append(f"{context} must be an object")
            continue
        for field in ("id", "description", "evidence_label_id"):
            require_string(artifact_class, field, errors, context=context)
        require_string_list(artifact_class, "required_fields", errors, context=context)
        require_string_list(artifact_class, "required_files", errors, context=context)
        require_bool(artifact_class, "may_include_prompt_traffic", errors, context=context)
        require_bool(artifact_class, "may_claim_semantic_expert_ids", errors, context=context)

        artifact_id = artifact_class.get("id")
        if isinstance(artifact_id, str):
            if artifact_id in seen:
                errors.append(f"{context}.id duplicates {artifact_id!r}")
            seen.add(artifact_id)

        label_id = artifact_class.get("evidence_label_id")
        label = evidence_labels.get(label_id) if isinstance(label_id, str) else None
        if label is None:
            errors.append(f"{context}.evidence_label_id references unknown label {label_id!r}")
        elif label.get("evidence_type") != "runtime_evidence":
            errors.append(f"{context}.evidence_label_id must reference runtime evidence")
        if artifact_class.get("may_claim_semantic_expert_ids") is not False:
            errors.append(f"{context}.may_claim_semantic_expert_ids must be false for Phase 1 baselines")

    missing = sorted(REQUIRED_ARTIFACT_CLASS_IDS - seen)
    if missing:
        errors.append(f"artifact_classes missing required ids: {', '.join(missing)}")


def validate_contract(contract: JSONDict) -> list[str]:
    errors: list[str] = []
    if contract.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        errors.append(
            "schema_version must be "
            f"{SUPPORTED_SCHEMA_VERSION!r}, got {contract.get('schema_version')!r}"
        )
    for field in ("name", "generated_for", "phase", "boundary"):
        require_string(contract, field, errors, context="contract")
    if contract.get("phase") != "phase_1":
        errors.append("contract.phase must be 'phase_1'")
    labels = validate_evidence_labels(contract, errors)
    validate_artifact_classes(contract, labels, errors)
    require_string_list(contract, "capability_labels", errors, context="contract")
    require_string_list(contract, "safety_contract", errors, context="contract")
    require_string_list(contract, "next_actions", errors, context="contract")
    return errors


def build_summary(contract: JSONDict, path: Path) -> JSONDict:
    errors = validate_contract(contract)
    labels = contract.get("evidence_labels") if isinstance(contract.get("evidence_labels"), list) else []
    artifact_classes = (
        contract.get("artifact_classes") if isinstance(contract.get("artifact_classes"), list) else []
    )
    return {
        "mode": "runtime_baseline_artifact_contract_plan",
        "contract_path": str(path),
        "valid": not errors,
        "errors": errors,
        "schema_version": contract.get("schema_version"),
        "phase": contract.get("phase"),
        "artifact_classes": [
            item.get("id") for item in artifact_classes if isinstance(item, dict)
        ],
        "runtime_evidence_labels": [
            item.get("id")
            for item in labels
            if isinstance(item, dict) and item.get("evidence_type") == "runtime_evidence"
        ],
        "semantic_routing_evidence_labels": [
            item.get("id")
            for item in labels
            if isinstance(item, dict) and item.get("evidence_type") == "semantic_routing_evidence"
        ],
        "safety_contract": contract.get("safety_contract", []),
        "next_actions": contract.get("next_actions", []),
    }


def print_human_summary(summary: JSONDict) -> None:
    print("MoE Run Anyway runtime baseline artifact contract")
    print(f"Contract: {summary['contract_path']}")
    print(f"Valid: {summary['valid']}")
    if summary["errors"]:
        print("Errors:")
        for error in summary["errors"]:
            print(f"  - {error}")
        return
    print(f"Phase: {summary['phase']}")
    print("Artifact classes:")
    for artifact_class in summary["artifact_classes"]:
        print(f"  - {artifact_class}")
    print("Runtime evidence labels:")
    for label in summary["runtime_evidence_labels"]:
        print(f"  - {label}")
    print("Semantic routing evidence labels:")
    for label in summary["semantic_routing_evidence_labels"]:
        print(f"  - {label}")
    print("Next actions:")
    for action in summary["next_actions"]:
        print(f"  - {action}")
    print("Safety contract:")
    for item in summary["safety_contract"]:
        print(f"  - {item}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "contract_path",
        nargs="?",
        type=Path,
        default=DEFAULT_CONTRACT_PATH,
        help="runtime baseline artifact contract JSON path",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    return parser


def plan_contract_path(path: Path) -> tuple[int, JSONDict | None, str | None]:
    try:
        contract = load_contract(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not load runtime baseline artifact contract: {exc}"

    summary = build_summary(contract, path)
    return (0 if summary["valid"] else 2), summary, None


def main_from_test_path(path: Path) -> int:
    status, _, _ = plan_contract_path(path)
    return status


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary, error_message = plan_contract_path(args.contract_path)
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
