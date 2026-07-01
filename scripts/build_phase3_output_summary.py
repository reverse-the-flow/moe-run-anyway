#!/usr/bin/env python3
"""Build or validate Phase 3 saved-output summary artifacts.

These summaries are the input to dense/full-runtime fallback comparison. The
builder can emit prompt-set-backed templates or validate filled summaries. It
only reads/writes local JSON metadata; it does not launch runtimes, call
endpoints, inspect secrets, mutate residency, or send prompt traffic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import build_dense_fallback_comparison
import plan_phase3_dense_fallback_capture
import phase3_output_receipts


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_SET_PATH = ROOT / "memory-moe-mvp" / "phase3-real-evidence" / "pc_mixtral_phase3_real_evidence_bundle.prompt-set.json"
SUPPORTED_SCHEMA_VERSION = "moe-phase3-output-summary-v1"
BUILDER_SCHEMA_VERSION = "moe-phase3-output-summary-builder-v1"
OUTPUT_LABELS = ("managed", "dense")
JSONDict = dict[str, Any]


def display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def load_json(path: Path) -> JSONDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{display_path(path) or path} must be a JSON object")
    return payload


def load_prompt_rows(prompt_set_path: Path) -> tuple[JSONDict, list[JSONDict]]:
    prompt_set = load_json(prompt_set_path)
    rows = plan_phase3_dense_fallback_capture.rows_from_prompt_payload(prompt_set, path=prompt_set_path)
    return prompt_set, rows


def prompt_id(row: JSONDict) -> str:
    value = plan_phase3_dense_fallback_capture.prompt_id_from_row(row)
    if value is None:
        raise ValueError("prompt row missing prompt_id/probe_id/id")
    return value


def default_output_path(prompt_set_path: Path, output_label: str) -> Path:
    name = prompt_set_path.name
    suffix = ".prompt-set.json"
    if name.endswith(suffix):
        stem = name[: -len(suffix)]
    else:
        stem = prompt_set_path.stem
    output_dir = prompt_set_path.with_name(f"{stem}-fallback")
    return output_dir / f"{output_label}-output-summary.json"


def template_rows(prompt_rows: list[JSONDict], *, output_label: str) -> list[JSONDict]:
    rows: list[JSONDict] = []
    for row in prompt_rows:
        rows.append(
            {
                "prompt_id": prompt_id(row),
                "group_id": row.get("group_id"),
                "prompt_repeat": row.get("repeat"),
                "output": "",
                "error": "output_missing",
                "output_label": output_label,
                "ready": False,
                "notes": "replace output and clear error after an approved runtime capture",
            }
        )
    return rows


def build_template_artifact(prompt_set_path: Path, *, output_label: str) -> JSONDict:
    prompt_set, rows = load_prompt_rows(prompt_set_path)
    source_prompt_set_path = display_path(prompt_set_path)
    return {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "source_prompt_set_path": source_prompt_set_path,
        "model_id": prompt_set.get("model_id"),
        "backend_family": prompt_set.get("backend_family"),
        "prompt_family": prompt_set.get("prompt_family"),
        "output_label": output_label,
        "output_ready": False,
        "capture_receipt": phase3_output_receipts.default_capture_receipt(
            prompt_set=prompt_set,
            source_prompt_set_path=source_prompt_set_path,
            output_label=output_label,
        ),
        "outputs": template_rows(rows, output_label=output_label),
        "safety_contract": safety_contract(),
        "next_actions": [
            "Fill output rows only from an approved runtime capture using the same prompt ids.",
            "Fill capture_receipt with the approved request path, capture host, backend, timestamp, and approval flags.",
            "Clear row error values when output text or response summaries are present.",
            "Validate this summary before building the dense fallback comparison artifact.",
        ],
    }


def prompt_ids_from_set(prompt_set_path: Path) -> set[str]:
    _, rows = load_prompt_rows(prompt_set_path)
    return {prompt_id(row) for row in rows}


def validate_output_summary(artifact: JSONDict, *, prompt_set_path: Path | None = None, expected_label: str | None = None) -> JSONDict:
    errors: list[str] = []
    if artifact.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        errors.append(f"schema_version must be {SUPPORTED_SCHEMA_VERSION!r}")
    output_label = artifact.get("output_label")
    if output_label not in OUTPUT_LABELS:
        errors.append(f"output_label must be one of {', '.join(OUTPUT_LABELS)}")
    if expected_label is not None and output_label != expected_label:
        errors.append(f"output_label must be {expected_label!r}")
    receipt_validation = phase3_output_receipts.validate_capture_receipt_source_binding(
        artifact.get("capture_receipt"),
        repo_root=ROOT,
        expected_label=str(output_label) if output_label in OUTPUT_LABELS else expected_label,
        expected_source_prompt_set_path=prompt_set_path,
    )
    errors.extend(receipt_validation["errors"])


    rows_payload = artifact.get("outputs")
    rows: list[JSONDict] = []
    if not isinstance(rows_payload, list) or not rows_payload:
        errors.append("outputs must be a non-empty list")
    else:
        for index, row in enumerate(rows_payload):
            if not isinstance(row, dict):
                errors.append(f"outputs[{index}] must be an object")
                continue
            rows.append(row)

    indexed: dict[str, JSONDict] = {}
    row_errors: list[str] = []
    if rows:
        indexed, row_errors = build_dense_fallback_comparison.index_rows(rows, label=str(output_label or "outputs"))
        errors.extend(row_errors)

    prompt_ids = set(indexed)
    expected_prompt_ids: set[str] = set()
    if prompt_set_path is not None:
        expected_prompt_ids = prompt_ids_from_set(prompt_set_path)
        missing = sorted(expected_prompt_ids - prompt_ids)
        extra = sorted(prompt_ids - expected_prompt_ids)
        if missing:
            errors.append(f"outputs missing prompt ids: {', '.join(missing)}")
        if extra:
            errors.append(f"outputs contain prompt ids not in prompt set: {', '.join(extra)}")

    output_present_count = sum(1 for row in indexed.values() if build_dense_fallback_comparison.output_present(row))
    missing_output_count = len(indexed) - output_present_count
    shape_valid = not errors
    summary_ready = (
        shape_valid
        and bool(indexed)
        and missing_output_count == 0
        and receipt_validation["receipt_ready"]
    )
    return {
        "shape_valid": shape_valid,
        "summary_ready": summary_ready,
        "errors": errors,
        "capture_receipt_ready": receipt_validation["receipt_ready"],
        "capture_receipt_errors": receipt_validation["errors"],
        "capture_receipt_source_request_path_exists": receipt_validation.get("source_request_path_exists"),
        "capture_receipt_source_prompt_set_path_exists": receipt_validation.get("source_prompt_set_path_exists"),
        "capture_receipt_source_prompt_set_path_matches": receipt_validation.get("source_prompt_set_path_matches"),
        "row_count": len(indexed),
        "prompt_count": len(expected_prompt_ids) if expected_prompt_ids else len(indexed),
        "output_present_count": output_present_count,
        "missing_output_count": missing_output_count,
        "prompt_ids": sorted(indexed),
    }


def safety_contract() -> list[str]:
    return [
        "output-summary builder reads and writes local JSON metadata only",
        "output-summary builder does not launch model servers",
        "output-summary builder does not call endpoints",
        "output-summary builder does not run Docker",
        "output-summary builder does not download models",
        "output-summary builder does not inspect private tokens",
        "output-summary builder does not send prompt traffic",
        "output-summary builder does not mutate runtime residency",
        "output-summary builder does not claim live expert paging",
    ]


def write_artifact(path: Path, artifact: JSONDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_summary(
    artifact: JSONDict,
    *,
    prompt_set_path: Path | None,
    output_path: Path | None,
    expected_label: str | None,
) -> JSONDict:
    validation = validate_output_summary(artifact, prompt_set_path=prompt_set_path, expected_label=expected_label)
    return {
        "mode": "phase3_output_summary_builder",
        "schema_version": BUILDER_SCHEMA_VERSION,
        "artifact_schema_version": artifact.get("schema_version"),
        "valid": validation["shape_valid"],
        "summary_ready": validation["summary_ready"],
        "errors": validation["errors"],
        "capture_receipt_ready": validation["capture_receipt_ready"],
        "capture_receipt_errors": validation["capture_receipt_errors"],
        "capture_receipt_source_request_path_exists": validation.get("capture_receipt_source_request_path_exists"),
        "capture_receipt_source_prompt_set_path_exists": validation.get("capture_receipt_source_prompt_set_path_exists"),
        "capture_receipt_source_prompt_set_path_matches": validation.get("capture_receipt_source_prompt_set_path_matches"),
        "output_path": display_path(output_path),
        "source_prompt_set_path": display_path(prompt_set_path),
        "model_id": artifact.get("model_id"),
        "prompt_family": artifact.get("prompt_family"),
        "output_label": artifact.get("output_label"),
        "row_count": validation["row_count"],
        "prompt_count": validation["prompt_count"],
        "output_present_count": validation["output_present_count"],
        "missing_output_count": validation["missing_output_count"],
        "prompt_ids": validation["prompt_ids"],
        "safety_contract": safety_contract(),
        "next_actions": [
            "Fill missing outputs from an approved runtime capture using the same prompt ids.",
            "Mark capture_receipt ready only after recording the approved request path, host/backend, timestamp, and approval flags.",
            "Use managed and dense summaries together with build_dense_fallback_comparison.py.",
        ],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt_set_path", nargs="?", type=Path, default=DEFAULT_PROMPT_SET_PATH)
    parser.add_argument("--output-label", choices=OUTPUT_LABELS, default="managed")
    parser.add_argument("--input-summary", type=Path, help="validate an existing output summary instead of building a template")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--default-output", action="store_true", help="write to <prompt-set-stem>-fallback/<label>-output-summary.json")
    parser.add_argument("--artifact-json", action="store_true", help="print the output-summary artifact")
    parser.add_argument("--json", action="store_true", help="emit machine-readable builder summary")
    return parser


def plan_build(args: argparse.Namespace) -> tuple[int, JSONDict | None, JSONDict | None, str | None]:
    try:
        output_path = args.output
        if args.default_output:
            if output_path is not None:
                return 2, None, None, "--output and --default-output cannot be used together"
            output_path = default_output_path(args.prompt_set_path, args.output_label)
        if args.input_summary is not None:
            artifact = load_json(args.input_summary)
            output_path = output_path or args.input_summary
        else:
            artifact = build_template_artifact(args.prompt_set_path, output_label=args.output_label)
        summary = build_summary(
            artifact,
            prompt_set_path=args.prompt_set_path,
            output_path=output_path,
            expected_label=args.output_label,
        )
        if output_path is not None and args.input_summary is None and summary["valid"]:
            write_artifact(output_path, artifact)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, None, f"Could not build Phase 3 output summary: {exc}"
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
        print("MoE Run Anyway Phase 3 output-summary builder")
        print(f"Valid: {summary['valid']}")
        print(f"Summary ready: {summary['summary_ready']}")
        print(f"Capture receipt ready: {summary['capture_receipt_ready']}")
        print(f"Output path: {summary['output_path']}")
        print(f"Output label: {summary['output_label']}")
        print(f"Rows: {summary['row_count']}")
        print(f"Missing outputs: {summary['missing_output_count']}")
        if summary["errors"]:
            print("Errors:")
            for error in summary["errors"]:
                print(f"  - {error}")
        print("Safety contract:")
        for item in summary["safety_contract"]:
            print(f"  - {item}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
