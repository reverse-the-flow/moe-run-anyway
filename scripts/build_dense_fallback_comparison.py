#!/usr/bin/env python3
"""Build a Phase 3 dense/full-runtime fallback comparison artifact.

The builder consumes already-saved managed and dense/full-runtime output
summaries, pairs rows by prompt id, and emits the schema validated by
plan_dense_fallback_comparison.py. It does not launch runtimes, call endpoints,
send prompt traffic, or judge semantic quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import plan_dense_fallback_comparison
import phase3_output_receipts


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = ROOT / "memory-moe-mvp" / "phase3-real-evidence" / "dense_fallback_comparison.json"
SUPPORTED_SCHEMA_VERSION = "moe-dense-fallback-comparison-builder-v1"
TEXT_FIELDS = ("output", "text", "response_text", "generated_text", "content")
ROW_LIST_FIELDS = ("outputs", "responses", "events", "items", "records", "comparisons")
JSONDict = dict[str, Any]


def display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def load_json_or_jsonl(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(item)
        return rows
    return json.loads(path.read_text(encoding="utf-8"))


def rows_from_payload(payload: Any, *, path: Path) -> list[JSONDict]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = []
        for field in ROW_LIST_FIELDS:
            candidate = payload.get(field)
            if isinstance(candidate, list):
                rows = candidate
                break
        if not rows and any(field in payload for field in (*TEXT_FIELDS, "prompt_id", "probe_id", "case", "response")):
            rows = [payload]
    else:
        rows = []

    if not rows:
        raise ValueError(f"{path} does not contain a supported output row list")
    result: list[JSONDict] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: row {index} must be a JSON object")
        result.append(row)
    return result


def prompt_id_for_row(row: JSONDict) -> str | None:
    for field in ("prompt_id", "probe_id", "id"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            prompt_id = value.strip()
            break
    else:
        case = row.get("case")
        if isinstance(case, dict):
            value = case.get("prompt_id") or case.get("probe_id") or case.get("id")
            prompt_id = value.strip() if isinstance(value, str) and value.strip() else ""
        else:
            prompt_id = ""

    if not prompt_id:
        return None
    repeat = row.get("repeat")
    if repeat is None and isinstance(row.get("case"), dict):
        repeat = row["case"].get("repeat")
    if repeat is not None:
        return f"{prompt_id}#repeat-{repeat}"
    return prompt_id


def nested_response_summary(row: JSONDict) -> JSONDict:
    response = row.get("response")
    if isinstance(response, dict):
        summary = response.get("summary")
        if isinstance(summary, dict):
            return summary
    summary = row.get("summary")
    return summary if isinstance(summary, dict) else {}


def output_text(row: JSONDict) -> tuple[str | None, str | None]:
    for field in TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str):
            return value, field

    summary = nested_response_summary(row)
    for field in ("response_text", "message_content", "raw_body"):
        value = summary.get(field)
        if isinstance(value, str):
            return value, f"response.summary.{field}"
    for field in ("response_preview", "message_content_preview"):
        value = summary.get(field)
        if isinstance(value, str):
            return value, f"response.summary.{field}"
    return None, None


def output_char_count(row: JSONDict, text: str | None) -> int:
    if text is not None:
        return len(text)
    summary = nested_response_summary(row)
    for field in ("response_chars", "message_content_chars", "body_bytes"):
        value = summary.get(field)
        if isinstance(value, int) and value > 0:
            return value
    return 0


def output_present(row: JSONDict) -> bool:
    text, _ = output_text(row)
    return output_char_count(row, text) > 0 and row.get("error") in (None, "")


def output_fingerprint(row: JSONDict) -> JSONDict:
    text, source = output_text(row)
    chars = output_char_count(row, text)
    result: JSONDict = {
        "present": output_present(row),
        "char_count": chars,
        "text_source": source,
    }
    if text is not None:
        result["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    summary = nested_response_summary(row)
    finish_reason = summary.get("finish_reason")
    if isinstance(finish_reason, str):
        result["finish_reason"] = finish_reason
    return result


def index_rows(rows: list[JSONDict], *, label: str) -> tuple[dict[str, JSONDict], list[str]]:
    indexed: dict[str, JSONDict] = {}
    errors: list[str] = []
    for index, row in enumerate(rows):
        prompt_id = prompt_id_for_row(row)
        if prompt_id is None:
            errors.append(f"{label}[{index}] missing prompt_id/probe_id")
            continue
        if prompt_id in indexed:
            errors.append(f"{label}[{index}] duplicates prompt id {prompt_id!r}")
            continue
        indexed[prompt_id] = row
    return indexed, errors


def quality_label(
    managed_row: JSONDict | None,
    dense_row: JSONDict | None,
    *,
    default_quality_label: str,
    auto_label_exact: bool,
) -> str:
    if managed_row is None or dense_row is None:
        return "unknown"
    if not auto_label_exact:
        return default_quality_label
    managed_text, managed_source = output_text(managed_row)
    dense_text, dense_source = output_text(dense_row)
    if (
        managed_text is not None
        and dense_text is not None
        and managed_source in TEXT_FIELDS
        and dense_source in TEXT_FIELDS
        and managed_text.strip() == dense_text.strip()
    ):
        return "same"
    return default_quality_label


def comparison_rows(
    managed_rows: list[JSONDict],
    dense_rows: list[JSONDict],
    *,
    default_quality_label: str,
    auto_label_exact: bool,
) -> tuple[list[JSONDict], list[str]]:
    managed_index, managed_errors = index_rows(managed_rows, label="managed")
    dense_index, dense_errors = index_rows(dense_rows, label="dense")
    comparisons: list[JSONDict] = []
    for prompt_id in sorted(set(managed_index) | set(dense_index)):
        managed_row = managed_index.get(prompt_id)
        dense_row = dense_index.get(prompt_id)
        item: JSONDict = {
            "prompt_id": prompt_id,
            "managed_output_present": managed_row is not None and output_present(managed_row),
            "dense_output_present": dense_row is not None and output_present(dense_row),
            "quality_delta_label": quality_label(
                managed_row,
                dense_row,
                default_quality_label=default_quality_label,
                auto_label_exact=auto_label_exact,
            ),
        }
        if managed_row is not None:
            item["managed_output_fingerprint"] = output_fingerprint(managed_row)
        if dense_row is not None:
            item["dense_output_fingerprint"] = output_fingerprint(dense_row)
        comparisons.append(item)
    return comparisons, [*managed_errors, *dense_errors]


def build_template_artifact(
    *,
    model_id: str,
    prompt_family: str,
    managed_policy_id: str,
    managed_artifact: str,
    dense_artifact: str,
) -> JSONDict:
    return {
        "schema_version": plan_dense_fallback_comparison.SUPPORTED_SCHEMA_VERSION,
        "model_id": model_id,
        "prompt_family": prompt_family,
        "managed_policy_id": managed_policy_id,
        "managed_artifact": managed_artifact,
        "dense_artifact": dense_artifact,
        "comparisons": [
            {
                "prompt_id": "case-001",
                "managed_output_present": False,
                "dense_output_present": False,
                "quality_delta_label": "unknown",
                "notes": "replace this row with paired managed and dense/full-runtime output evidence",
            }
        ],
        "builder": {
            "schema_version": SUPPORTED_SCHEMA_VERSION,
            "mode": "template",
            "quality_note": "unknown labels are valid but keep the Phase 3 gate not ready",
        },
    }


def receipt_validation_for_payload(payload: Any, *, label: str) -> JSONDict:
    if not isinstance(payload, dict):
        return {
            "shape_valid": False,
            "receipt_ready": False,
            "errors": [f"{label} output summary must be a JSON object with capture_receipt"],
            "source_request_path_exists": False,
            "source_prompt_set_path_exists": False,
            "source_request_path_matches": False,
            "source_prompt_set_path_matches": False,
        }
    return phase3_output_receipts.validate_capture_receipt_source_binding(
        payload.get("capture_receipt"),
        repo_root=ROOT,
        expected_label=label,
    )


def receipt_errors_for_payload(payload: Any, *, label: str) -> list[str]:
    validation = receipt_validation_for_payload(payload, label=label)
    if validation["receipt_ready"]:
        return []
    errors = list(validation["errors"])
    if not errors:
        errors.append("capture_receipt.receipt_ready must be true before fallback comparison build")
    return [f"{label} output summary: {error}" for error in errors]


RECEIPT_PAIR_FIELDS = ("source_request_path", "source_prompt_set_path", "model_id", "prompt_family")


def capture_receipt_from_payload(payload: Any) -> JSONDict:
    if isinstance(payload, dict) and isinstance(payload.get("capture_receipt"), dict):
        return payload["capture_receipt"]
    return {}


def receipt_string(receipt: JSONDict, field: str) -> str | None:
    return phase3_output_receipts.string_or_none(receipt.get(field))


def receipt_pair_errors(
    managed_payload: Any,
    dense_payload: Any,
    *,
    model_id: str,
    prompt_family: str,
) -> list[str]:
    managed_receipt = capture_receipt_from_payload(managed_payload)
    dense_receipt = capture_receipt_from_payload(dense_payload)
    if not managed_receipt or not dense_receipt:
        return []
    errors: list[str] = []
    for field in RECEIPT_PAIR_FIELDS:
        managed_value = receipt_string(managed_receipt, field)
        dense_value = receipt_string(dense_receipt, field)
        if managed_value is not None and dense_value is not None and managed_value != dense_value:
            errors.append(f"managed and dense capture_receipt.{field} must match")
    for label, receipt in (("managed", managed_receipt), ("dense", dense_receipt)):
        receipt_model_id = receipt_string(receipt, "model_id")
        receipt_prompt_family = receipt_string(receipt, "prompt_family")
        if receipt_model_id is not None and receipt_model_id != model_id:
            errors.append(f"{label} capture_receipt.model_id must match comparison model_id")
        if receipt_prompt_family is not None and receipt_prompt_family != prompt_family:
            errors.append(f"{label} capture_receipt.prompt_family must match comparison prompt_family")
    return errors


def input_receipt_summary(managed_payload: Any, dense_payload: Any, *, pair_consistent: bool) -> JSONDict:
    managed_receipt = capture_receipt_from_payload(managed_payload)
    dense_receipt = capture_receipt_from_payload(dense_payload)
    managed_validation = receipt_validation_for_payload(managed_payload, label="managed")
    dense_validation = receipt_validation_for_payload(dense_payload, label="dense")
    source_receipt = managed_receipt or dense_receipt
    return {
        "managed_capture_receipt_ready": managed_validation["receipt_ready"] and pair_consistent,
        "dense_capture_receipt_ready": dense_validation["receipt_ready"] and pair_consistent,
        "receipt_pair_consistent": pair_consistent,
        "receipt_gate": "required_before_write",
        "source_request_path": receipt_string(source_receipt, "source_request_path"),
        "source_prompt_set_path": receipt_string(source_receipt, "source_prompt_set_path"),
        "source_request_path_exists": managed_validation.get("source_request_path_exists") if managed_receipt else dense_validation.get("source_request_path_exists"),
        "source_prompt_set_path_exists": managed_validation.get("source_prompt_set_path_exists") if managed_receipt else dense_validation.get("source_prompt_set_path_exists"),
        "model_id": receipt_string(source_receipt, "model_id"),
        "prompt_family": receipt_string(source_receipt, "prompt_family"),
    }

def build_artifact(
    *,
    managed_artifact_path: Path,
    dense_artifact_path: Path,
    model_id: str,
    prompt_family: str,
    managed_policy_id: str,
    default_quality_label: str,
    auto_label_exact: bool,
) -> tuple[JSONDict, list[str]]:
    managed_payload = load_json_or_jsonl(managed_artifact_path)
    dense_payload = load_json_or_jsonl(dense_artifact_path)
    managed_rows = rows_from_payload(managed_payload, path=managed_artifact_path)
    dense_rows = rows_from_payload(dense_payload, path=dense_artifact_path)
    managed_receipt_errors = receipt_errors_for_payload(managed_payload, label="managed")
    dense_receipt_errors = receipt_errors_for_payload(dense_payload, label="dense")
    pair_errors = receipt_pair_errors(
        managed_payload,
        dense_payload,
        model_id=model_id,
        prompt_family=prompt_family,
    )
    pair_consistent = not managed_receipt_errors and not dense_receipt_errors and not pair_errors
    comparisons, row_errors = comparison_rows(
        managed_rows,
        dense_rows,
        default_quality_label=default_quality_label,
        auto_label_exact=auto_label_exact,
    )
    artifact: JSONDict = {
        "schema_version": plan_dense_fallback_comparison.SUPPORTED_SCHEMA_VERSION,
        "model_id": model_id,
        "prompt_family": prompt_family,
        "managed_policy_id": managed_policy_id,
        "managed_artifact": display_path(managed_artifact_path) or str(managed_artifact_path),
        "dense_artifact": display_path(dense_artifact_path) or str(dense_artifact_path),
        "comparisons": comparisons,
        "builder": {
            "schema_version": SUPPORTED_SCHEMA_VERSION,
            "mode": "paired_output_summary",
            "auto_label_exact": auto_label_exact,
            "default_quality_label": default_quality_label,
            "input_receipts": input_receipt_summary(
                managed_payload,
                dense_payload,
                pair_consistent=pair_consistent,
            ),
            "safety_contract": safety_contract(),
        },
    }
    return artifact, [*managed_receipt_errors, *dense_receipt_errors, *pair_errors, *row_errors]


def safety_contract() -> list[str]:
    return [
        "builder reads local saved output summaries only",
        "builder does not launch model servers",
        "builder does not call endpoints",
        "builder does not send prompt traffic",
        "builder does not inspect private tokens",
        "builder requires ready capture receipts before writing paired comparison artifacts",
        "builder requires managed and dense receipts to share request, prompt set, model, and prompt family",
        "builder does not judge semantic quality automatically",
    ]


def build_summary(artifact: JSONDict, *, output_path: Path | None, build_errors: list[str]) -> JSONDict:
    validation_errors = plan_dense_fallback_comparison.validate_comparison_artifact(artifact)
    counts = plan_dense_fallback_comparison.comparison_counts(artifact)
    comparison_ready_if_validated = (
        not build_errors
        and not validation_errors
        and counts["comparison_count"] > 0
        and counts["missing_managed_output_count"] == 0
        and counts["missing_dense_output_count"] == 0
        and counts["quality_delta_counts"].get("major_delta", 0) == 0
        and counts["quality_delta_counts"].get("unknown", 0) == 0
    )
    return {
        "mode": "dense_fallback_comparison_builder",
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "valid": not build_errors and not validation_errors,
        "errors": [*build_errors, *validation_errors],
        "output_path": display_path(output_path),
        "artifact_schema_version": artifact.get("schema_version"),
        "model_id": artifact.get("model_id"),
        "prompt_family": artifact.get("prompt_family"),
        "managed_policy_id": artifact.get("managed_policy_id"),
        **counts,
        "comparison_ready_if_validated": comparison_ready_if_validated,
        "safety_contract": safety_contract(),
        "next_actions": [
            "Review unknown or major_delta rows before treating fallback behavior as bounded.",
            "Confirm managed and dense input capture_receipts are ready before building fallback comparison evidence.",
            "Validate the written artifact with scripts/plan_dense_fallback_comparison.py.",
            "Keep this paired to the same model target and prompt family as the managed replay trace.",
        ],
    }


def write_artifact(path: Path, artifact: JSONDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_human_summary(summary: JSONDict) -> None:
    print("MoE Run Anyway dense fallback comparison builder")
    print(f"Valid: {summary['valid']}")
    print(f"Output path: {summary['output_path']}")
    print(f"Comparisons: {summary['comparison_count']}")
    print(f"Missing managed outputs: {summary['missing_managed_output_count']}")
    print(f"Missing dense outputs: {summary['missing_dense_output_count']}")
    print(f"Comparison ready if validated: {summary['comparison_ready_if_validated']}")
    if summary["errors"]:
        print("Errors:")
        for error in summary["errors"]:
            print(f"  - {error}")
    print("Safety contract:")
    for item in summary["safety_contract"]:
        print(f"  - {item}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--managed-artifact", type=Path)
    parser.add_argument("--dense-artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-id", default="replace-with-model-id")
    parser.add_argument("--prompt-family", default="replace-with-prompt-family")
    parser.add_argument("--managed-policy-id", default="preload_shortlist")
    parser.add_argument(
        "--default-quality-label",
        choices=sorted(plan_dense_fallback_comparison.SUPPORTED_QUALITY_LABELS),
        default="unknown",
    )
    parser.add_argument(
        "--auto-label-exact",
        action="store_true",
        help="label rows as same only when both inputs expose identical top-level output text",
    )
    parser.add_argument("--template", action="store_true", help="emit a valid not-ready template artifact")
    parser.add_argument("--artifact-json", action="store_true", help="print the built artifact JSON")
    parser.add_argument("--json", action="store_true", help="emit machine-readable builder summary")
    return parser


def plan_build(args: argparse.Namespace) -> tuple[int, JSONDict | None, JSONDict | None, str | None]:
    try:
        if args.template:
            artifact = build_template_artifact(
                model_id=args.model_id,
                prompt_family=args.prompt_family,
                managed_policy_id=args.managed_policy_id,
                managed_artifact=display_path(args.managed_artifact) or "replace-with-managed-artifact.json",
                dense_artifact=display_path(args.dense_artifact) or "replace-with-dense-artifact.json",
            )
            build_errors: list[str] = []
        else:
            if args.managed_artifact is None or args.dense_artifact is None:
                raise ValueError("--managed-artifact and --dense-artifact are required unless --template is used")
            artifact, build_errors = build_artifact(
                managed_artifact_path=args.managed_artifact,
                dense_artifact_path=args.dense_artifact,
                model_id=args.model_id,
                prompt_family=args.prompt_family,
                managed_policy_id=args.managed_policy_id,
                default_quality_label=args.default_quality_label,
                auto_label_exact=args.auto_label_exact,
            )
        summary = build_summary(artifact, output_path=args.output, build_errors=build_errors)
        if args.output and summary["valid"]:
            write_artifact(args.output, artifact)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, None, f"Could not build dense fallback comparison artifact: {exc}"
    return (0 if summary["valid"] else 2), summary, artifact, None


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary, artifact, error_message = plan_build(args)
    if error_message:
        print(error_message, file=sys.stderr)
        return status
    assert summary is not None
    assert artifact is not None
    if args.artifact_json:
        print(json.dumps(artifact, indent=2, sort_keys=True))
    elif args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_human_summary(summary)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
