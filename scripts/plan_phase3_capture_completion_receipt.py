#!/usr/bin/env python3
"""Validate a filled Phase 3 runtime-capture completion receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


JSONDict = dict[str, Any]

COMPLETION_RECEIPT_SCHEMA = "moe-phase3-recommended-runtime-capture-completion-receipt-template-v1"
WORK_ORDER_SCHEMA = "moe-phase3-recommended-runtime-capture-work-order-v1"
SAFETY_FLAGS = (
    ("selected", True),
    ("metadata_only", True),
    ("launches_runtimes", False),
    ("runs_docker", False),
    ("sends_prompt_traffic", False),
    ("reads_private_tokens", False),
    ("mutates_runtime_residency", False),
    ("execution_requires_explicit_approval", True),
)
ROW_READY_FIELDS = (
    "capture_complete",
    "receipt_filled",
    "validator_passed",
    "ready_for_intake",
)
READY_ROW_TEXT_FIELDS = (
    "capture_step_id",
    "artifact_id",
    "artifact_path",
    "receipt_path",
    "receipt_kind",
    "request_path",
    "source_request_path",
    "source_prompt_set_path",
    "observed_at",
)


def string_or_none(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def normalized_manifest_path(value: Any) -> str | None:
    text = string_or_none(value)
    return text.replace("\\", "/") if text is not None else None


def int_count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def command_tokens(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)


def _task_tuple(request_path: Any, artifact_id: Any, artifact_path: Any, receipt_path: Any) -> tuple[str, str, str, str] | None:
    request = normalized_manifest_path(request_path)
    artifact = string_or_none(artifact_id)
    artifact_file = normalized_manifest_path(artifact_path)
    receipt_file = normalized_manifest_path(receipt_path)
    if not request or not artifact or not artifact_file or not receipt_file:
        return None
    return (request, artifact, artifact_file, receipt_file)


def capture_rows(receipt: Any) -> list[JSONDict]:
    if not isinstance(receipt, dict) or not isinstance(receipt.get("capture_receipts"), list):
        return []
    return [row for row in receipt["capture_receipts"] if isinstance(row, dict)]


def work_order_steps(work_order: Any) -> list[JSONDict]:
    if not isinstance(work_order, dict) or not isinstance(work_order.get("capture_steps"), list):
        return []
    return [step for step in work_order["capture_steps"] if isinstance(step, dict)]


def completion_receipt_task_keys(receipt: Any) -> set[tuple[str, str, str, str]]:
    if not isinstance(receipt, dict):
        return set()
    keys: set[tuple[str, str, str, str]] = set()
    for row in capture_rows(receipt):
        key = _task_tuple(
            row.get("request_path") or receipt.get("request_path"),
            row.get("artifact_id"),
            row.get("artifact_path"),
            row.get("receipt_path"),
        )
        if key:
            keys.add(key)
    return keys


def work_order_task_keys(work_order: Any) -> set[tuple[str, str, str, str]]:
    if not isinstance(work_order, dict):
        return set()
    keys: set[tuple[str, str, str, str]] = set()
    for step in work_order_steps(work_order):
        key = _task_tuple(
            step.get("request_path") or work_order.get("request_path"),
            step.get("artifact_id"),
            step.get("artifact_path"),
            step.get("receipt_path"),
        )
        if key:
            keys.add(key)
    return keys


def completion_receipt_template_from_work_order(work_order: JSONDict) -> JSONDict:
    rows: list[JSONDict] = []
    for step in work_order_steps(work_order):
        rows.append(
            {
                "receipt_rank": len(rows) + 1,
                "capture_step_id": step.get("step_id"),
                "artifact_id": step.get("artifact_id"),
                "artifact_path": normalized_manifest_path(step.get("artifact_path")),
                "receipt_path": normalized_manifest_path(step.get("receipt_path")),
                "receipt_kind": step.get("receipt_kind"),
                "request_path": normalized_manifest_path(step.get("request_path") or work_order.get("request_path")),
                "source_request_path": normalized_manifest_path(step.get("source_request_path")),
                "prompt_set_path": normalized_manifest_path(step.get("prompt_set_path")),
                "source_prompt_set_path": normalized_manifest_path(step.get("source_prompt_set_path")),
                "records_approval_keys": list_of_strings(step.get("records_approval_keys")),
                "requires_explicit_user_approval": step.get("requires_explicit_user_approval") is True,
                "approval_records_prompt_traffic": step.get("approval_records_prompt_traffic") is True,
                "may_send_prompt_traffic_after_approval": step.get("may_send_prompt_traffic_after_approval") is True,
                "expected_validator_command_count": int_count(step.get("validator_command_count")),
                "expected_validator_commands": step.get("validator_commands") if isinstance(step.get("validator_commands"), list) else [],
                "capture_complete": False,
                "receipt_filled": False,
                "validator_passed": False,
                "ready_for_intake": False,
                "observed_at": None,
                "operator_notes": "",
            }
        )
    validator_command_count = sum(int_count(row.get("expected_validator_command_count")) for row in rows)
    missing_items: list[str] = []
    if not normalized_manifest_path(work_order.get("request_path")):
        missing_items.append("completion_receipt_request_path_missing")
    if work_order.get("ready") is not True:
        missing_items.append("completion_receipt_work_order_not_ready")
    if int_count(work_order.get("capture_step_count")) != len(rows):
        missing_items.append("completion_receipt_capture_step_count_mismatch")
    intake_step = work_order.get("intake_step") if isinstance(work_order.get("intake_step"), dict) else {}
    if not command_tokens(intake_step.get("command")):
        missing_items.append("completion_receipt_intake_command_missing")
    ready = not missing_items
    return {
        "schema_version": COMPLETION_RECEIPT_SCHEMA,
        "selected": True,
        "metadata_only": True,
        "launches_runtimes": False,
        "runs_docker": False,
        "sends_prompt_traffic": False,
        "reads_private_tokens": False,
        "mutates_runtime_residency": False,
        "template_ready": ready,
        "ready": ready,
        "receipt_complete": False,
        "ready_for_capture_result_intake": False,
        "execution_requires_explicit_approval": True,
        "request_name": work_order.get("request_name"),
        "request_path": normalized_manifest_path(work_order.get("request_path")),
        "queue_rank": int_count(work_order.get("queue_rank")),
        "source_work_order_schema_version": work_order.get("schema_version"),
        "source_work_order_ready": work_order.get("ready") is True,
        "capture_receipt_count": len(rows),
        "expected_capture_step_count": int_count(work_order.get("capture_step_count")),
        "validator_command_count": validator_command_count,
        "missing_item_count": len(missing_items),
        "missing_items": missing_items,
        "approval_step": work_order.get("approval_step") if isinstance(work_order.get("approval_step"), dict) else {},
        "capture_receipts": rows,
        "intake_step": intake_step,
    }


def _row_is_ready(row: JSONDict) -> bool:
    return all(row.get(field) is True for field in ROW_READY_FIELDS)


def validate_completion_receipt(receipt: Any, work_order: Any | None = None) -> JSONDict:
    errors: list[str] = []
    if not isinstance(receipt, dict):
        return {
            "mode": "phase3_capture_completion_receipt",
            "valid": False,
            "receipt_complete": False,
            "ready_for_capture_result_intake": False,
            "errors": ["completion_receipt must be an object"],
        }

    rows = capture_rows(receipt)
    claimed_receipt_complete = receipt.get("receipt_complete") is True
    claimed_ready = receipt.get("ready_for_capture_result_intake") is True
    missing_items = receipt.get("missing_items") if isinstance(receipt.get("missing_items"), list) else []
    missing_item_count = int_count(receipt.get("missing_item_count"))
    row_ready_count = sum(1 for row in rows if _row_is_ready(row))
    capture_complete_count = sum(1 for row in rows if row.get("capture_complete") is True)
    receipt_filled_count = sum(1 for row in rows if row.get("receipt_filled") is True)
    validator_passed_count = sum(1 for row in rows if row.get("validator_passed") is True)
    ready_for_intake_row_count = sum(1 for row in rows if row.get("ready_for_intake") is True)
    actual_validator_command_count = sum(int_count(row.get("expected_validator_command_count")) for row in rows)
    all_rows_ready = bool(rows) and row_ready_count == len(rows)

    if receipt.get("schema_version") != COMPLETION_RECEIPT_SCHEMA:
        errors.append("completion_receipt_schema_mismatch")
    for key, expected in SAFETY_FLAGS:
        if receipt.get(key) is not expected:
            errors.append(f"completion_receipt_safety_flag_mismatch:{key}")
    if receipt.get("template_ready") is not True or receipt.get("ready") is not True:
        errors.append("completion_receipt_template_not_ready")
    if not isinstance(receipt.get("receipt_complete"), bool):
        errors.append("completion_receipt_complete_flag_not_boolean")
    if not isinstance(receipt.get("ready_for_capture_result_intake"), bool):
        errors.append("completion_receipt_ready_for_intake_flag_not_boolean")
    if int_count(receipt.get("capture_receipt_count")) != len(rows):
        errors.append("completion_receipt_row_count_mismatch")
    if int_count(receipt.get("validator_command_count")) != actual_validator_command_count:
        errors.append("completion_receipt_validator_count_mismatch")
    if missing_item_count != len(missing_items):
        errors.append("completion_receipt_missing_item_count_mismatch")
    if missing_item_count and claimed_ready:
        errors.append("completion_receipt_ready_with_missing_items")

    for index, row in enumerate(rows, start=1):
        row_prefix = f"completion_receipt_row_{index}"
        if row.get("ready_for_intake") is True and not _row_is_ready(row):
            errors.append(f"{row_prefix}_ready_without_prereqs")
        if _row_is_ready(row):
            for field in READY_ROW_TEXT_FIELDS:
                if string_or_none(row.get(field)) is None:
                    errors.append(f"{row_prefix}_{field}_missing")
            for field in ("requires_explicit_user_approval", "approval_records_prompt_traffic", "may_send_prompt_traffic_after_approval"):
                if row.get(field) is not True:
                    errors.append(f"{row_prefix}_{field}_not_confirmed")
            if int_count(row.get("expected_validator_command_count")) < 1:
                errors.append(f"{row_prefix}_validator_command_count_missing")

    if claimed_receipt_complete and not all_rows_ready:
        errors.append("completion_receipt_complete_without_all_rows_ready")
    if claimed_ready and not claimed_receipt_complete:
        errors.append("completion_receipt_ready_without_complete_receipt")
    if claimed_ready and not all_rows_ready:
        errors.append("completion_receipt_ready_without_all_rows_ready")
    if all_rows_ready and not claimed_receipt_complete:
        errors.append("completion_receipt_complete_flag_missing")
    if all_rows_ready and claimed_receipt_complete and not claimed_ready:
        errors.append("completion_receipt_ready_for_intake_flag_missing")

    if work_order is not None:
        if not isinstance(work_order, dict):
            errors.append("work_order_not_object")
        else:
            if work_order.get("schema_version") != WORK_ORDER_SCHEMA:
                errors.append("work_order_schema_mismatch")
            if work_order.get("ready") is not True:
                errors.append("work_order_not_ready")
            if receipt.get("source_work_order_schema_version") != work_order.get("schema_version"):
                errors.append("completion_receipt_work_order_schema_binding_mismatch")
            if normalized_manifest_path(receipt.get("request_path")) != normalized_manifest_path(work_order.get("request_path")):
                errors.append("completion_receipt_request_path_mismatch")
            if int_count(receipt.get("expected_capture_step_count")) != int_count(work_order.get("capture_step_count")):
                errors.append("completion_receipt_expected_step_count_mismatch")
            if int_count(receipt.get("capture_receipt_count")) != int_count(work_order.get("capture_step_count")):
                errors.append("completion_receipt_work_order_receipt_count_mismatch")
            if int_count(receipt.get("validator_command_count")) != int_count(work_order.get("validator_command_count")):
                errors.append("completion_receipt_work_order_validator_count_mismatch")
            receipt_intake = receipt.get("intake_step") if isinstance(receipt.get("intake_step"), dict) else {}
            work_order_intake = work_order.get("intake_step") if isinstance(work_order.get("intake_step"), dict) else {}
            if command_tokens(receipt_intake.get("command")) != command_tokens(work_order_intake.get("command")):
                errors.append("completion_receipt_intake_command_mismatch")
            if work_order_task_keys(work_order) and completion_receipt_task_keys(receipt) != work_order_task_keys(work_order):
                errors.append("completion_receipt_work_order_task_mismatch")

    structurally_valid = not errors
    complete = claimed_receipt_complete and all_rows_ready and structurally_valid
    ready_for_intake = complete and claimed_ready and missing_item_count == 0
    return {
        "mode": "phase3_capture_completion_receipt",
        "valid": structurally_valid,
        "receipt_complete": complete,
        "ready_for_capture_result_intake": ready_for_intake,
        "template_ready": receipt.get("template_ready") is True,
        "row_count": len(rows),
        "complete_row_count": row_ready_count,
        "capture_complete_count": capture_complete_count,
        "receipt_filled_count": receipt_filled_count,
        "validator_passed_count": validator_passed_count,
        "ready_for_intake_row_count": ready_for_intake_row_count,
        "capture_receipt_count": int_count(receipt.get("capture_receipt_count")),
        "expected_capture_step_count": int_count(receipt.get("expected_capture_step_count")),
        "validator_command_count": int_count(receipt.get("validator_command_count")),
        "computed_validator_command_count": actual_validator_command_count,
        "missing_item_count": missing_item_count,
        "errors": errors,
    }


def fixture_work_order() -> JSONDict:
    request_path = "memory-moe-mvp/phase3-real-evidence/fixture/fixture.runtime-capture-request.json"
    return {
        "schema_version": WORK_ORDER_SCHEMA,
        "selected": True,
        "metadata_only": True,
        "launches_runtimes": False,
        "runs_docker": False,
        "sends_prompt_traffic": False,
        "reads_private_tokens": False,
        "mutates_runtime_residency": False,
        "execution_requires_explicit_approval": True,
        "ready": True,
        "request_name": "Fixture request",
        "request_path": request_path,
        "queue_rank": 1,
        "capture_step_count": 2,
        "validator_command_count": 2,
        "capture_steps": [
            {
                "step_id": "capture_candidate_router_trace",
                "request_path": request_path,
                "source_request_path": request_path,
                "prompt_set_path": "memory-moe-mvp/phase3-real-evidence/fixture/fixture.prompt-set.json",
                "source_prompt_set_path": "memory-moe-mvp/phase3-real-evidence/fixture/fixture.prompt-set.json",
                "artifact_id": "candidate_router_trace",
                "artifact_path": "memory-moe-mvp/phase3-real-evidence/fixture/candidate-router-events.jsonl",
                "receipt_path": "memory-moe-mvp/phase3-real-evidence/fixture/candidate-router-events.capture-receipt.json",
                "receipt_kind": "trace_capture_receipt",
                "requires_explicit_user_approval": True,
                "approval_records_prompt_traffic": True,
                "may_send_prompt_traffic_after_approval": True,
                "records_approval_keys": ["approved_for_runtime_capture"],
                "validator_command_count": 1,
                "validator_commands": [["uv", "run", "--managed-python", "--python", "3.13", "scripts/phase3_trace_receipts.py"]],
            },
            {
                "step_id": "capture_managed_output_summary",
                "request_path": request_path,
                "source_request_path": request_path,
                "prompt_set_path": "memory-moe-mvp/phase3-real-evidence/fixture/fixture.prompt-set.json",
                "source_prompt_set_path": "memory-moe-mvp/phase3-real-evidence/fixture/fixture.prompt-set.json",
                "artifact_id": "managed_output_summary_fill",
                "artifact_path": "memory-moe-mvp/phase3-real-evidence/fixture/managed-output-summary.json",
                "receipt_path": "memory-moe-mvp/phase3-real-evidence/fixture/managed-output-summary.json",
                "receipt_kind": "embedded_output_capture_receipt",
                "requires_explicit_user_approval": True,
                "approval_records_prompt_traffic": True,
                "may_send_prompt_traffic_after_approval": True,
                "records_approval_keys": ["approved_for_runtime_capture"],
                "validator_command_count": 1,
                "validator_commands": [["uv", "run", "--managed-python", "--python", "3.13", "scripts/phase3_output_receipts.py"]],
            },
        ],
        "intake_step": {
            "stage": "capture_result_intake",
            "command": ["uv", "run", "--managed-python", "--python", "3.13", "scripts/plan_phase3_capture_result_intake.py", "--json"],
            "metadata_only": True,
            "requires_completed_capture_artifacts": True,
        },
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plan_validation(args: argparse.Namespace) -> tuple[int, JSONDict, str | None]:
    try:
        if args.receipt is None:
            work_order = fixture_work_order()
            receipt = completion_receipt_template_from_work_order(work_order)
        else:
            receipt = load_json(Path(args.receipt))
            work_order = load_json(Path(args.work_order)) if args.work_order else None
        summary = validate_completion_receipt(receipt, work_order)
        if args.output:
            write_json(Path(args.output), summary)
        return (0 if summary["valid"] else 2), summary, None
    except (OSError, json.JSONDecodeError) as exc:
        return 2, {
            "mode": "phase3_capture_completion_receipt",
            "valid": False,
            "receipt_complete": False,
            "ready_for_capture_result_intake": False,
            "errors": [str(exc)],
        }, str(exc)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", nargs="?", help="Completion receipt JSON to validate. With no path, validates the built-in template fixture.")
    parser.add_argument("--work-order", help="Recommended runtime-capture work order JSON to bind against.")
    parser.add_argument("--output", help="Optional path for the validation summary JSON.")
    parser.add_argument("--json", action="store_true", help="Print the validation summary as JSON.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    status, summary, error = plan_validation(args)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif error:
        print(error)
    else:
        print(f"valid: {summary['valid']}")
        print(f"receipt complete: {summary['receipt_complete']}")
        print(f"ready for capture-result intake: {summary['ready_for_capture_result_intake']}")
        if summary["errors"]:
            print("errors:")
            for item in summary["errors"]:
                print(f"  - {item}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())