#!/usr/bin/env python3
"""Build or validate Phase 3 candidate router-trace receipt artifacts.

The receipt is the sidecar that binds a saved candidate router trace to the
approved runtime-capture request and shared prompt set. This builder creates a
fillable not-ready template or validates a filled receipt. It only reads/writes
local JSON metadata; it does not launch runtimes, call endpoints, inspect
secrets, mutate residency, or send prompt traffic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import phase3_trace_receipts


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_PATH = ROOT / "memory-moe-mvp" / "phase3-real-evidence" / "pc_mixtral_phase3_real_evidence_bundle.json"
ARTIFACT_SCHEMA_VERSION = "moe-phase3-trace-capture-receipt-v1"
BUILDER_SCHEMA_VERSION = "moe-phase3-trace-receipt-builder-v1"
JSONDict = dict[str, Any]


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


def load_json(path: Path) -> JSONDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{display_path(path) or path} must be a JSON object")
    return payload


def canonical_prompt_set_path(bundle_path: Path) -> Path:
    return bundle_path.with_name(f"{bundle_path.stem}.prompt-set.json")


def default_candidate_trace_path(bundle_path: Path) -> Path:
    return bundle_path.with_name(f"{bundle_path.stem}-policy-candidate") / "candidate-router-events.jsonl"


def default_trace_receipt_path(candidate_trace_path: Path) -> Path:
    return candidate_trace_path.with_name(f"{candidate_trace_path.stem}.capture-receipt.json")


def normalized_path_text(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().replace("\\", "/")


def resolved_path_text(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def receipt_path_matches(value: Any, expected_path: Path | None) -> bool:
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


def safety_contract() -> list[str]:
    return [
        "trace-receipt builder reads and writes local JSON metadata only",
        "trace-receipt builder does not launch model servers",
        "trace-receipt builder does not call endpoints",
        "trace-receipt builder does not run Docker",
        "trace-receipt builder does not download models",
        "trace-receipt builder does not inspect private tokens",
        "trace-receipt builder does not send prompt traffic",
        "trace-receipt builder does not mutate runtime residency",
        "trace-receipt builder does not claim live expert paging",
    ]


def build_template_artifact(
    bundle_path: Path,
    *,
    prompt_set_path: Path | None = None,
    candidate_trace_path: Path | None = None,
) -> JSONDict:
    prompt_path = prompt_set_path or canonical_prompt_set_path(bundle_path)
    candidate_path = candidate_trace_path or default_candidate_trace_path(bundle_path)
    prompt_set = load_json(prompt_path)
    receipt = phase3_trace_receipts.default_trace_receipt(
        prompt_set=prompt_set,
        source_prompt_set_path=display_path(prompt_path),
        candidate_trace_path=display_path(candidate_path),
    )
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "source_bundle_path": display_path(bundle_path),
        "receipt_scope": "phase3-candidate-router-trace",
        **receipt,
        "safety_contract": safety_contract(),
        "next_actions": [
            "Capture the candidate router trace only after explicit runtime approval.",
            "Fill source_request_path with the saved runtime-capture request that approved the trace.",
            "Mark receipt_ready true only after recording request, prompt set, trace path, host/backend, timestamp, and approval flags.",
            "Validate this receipt before replay-based bundle promotion.",
        ],
    }


def validate_trace_receipt_artifact(
    artifact: JSONDict,
    *,
    bundle_path: Path | None = None,
    prompt_set_path: Path | None = None,
    candidate_trace_path: Path | None = None,
    expected_model_id: str | None = None,
    expected_backend_family: str | None = None,
    expected_prompt_family: str | None = None,
) -> JSONDict:
    errors: list[str] = []
    if artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        errors.append(f"schema_version must be {ARTIFACT_SCHEMA_VERSION!r}")
    validation = phase3_trace_receipts.validate_trace_receipt_source_binding(
        artifact,
        repo_root=ROOT,
        expected_source_prompt_set_path=prompt_set_path,
        expected_candidate_trace_path=candidate_trace_path,
    )
    errors.extend(validation["errors"])

    if bundle_path is not None and artifact.get("source_bundle_path") is not None:
        if not receipt_path_matches(artifact.get("source_bundle_path"), bundle_path):
            errors.append("source_bundle_path must match bundle path")
    if expected_model_id is not None and artifact.get("model_id") != expected_model_id:
        errors.append("model_id must match prompt set model_id")
    if expected_backend_family is not None and artifact.get("runtime_backend") != expected_backend_family:
        errors.append("runtime_backend must match prompt set backend_family")
    if expected_prompt_family is not None and artifact.get("prompt_family") != expected_prompt_family:
        errors.append("prompt_family must match prompt set prompt_family")

    return {
        "shape_valid": not errors,
        "receipt_ready": validation["receipt_ready"] and not errors,
        "errors": errors,
    }


def write_artifact(path: Path, artifact: JSONDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_summary(
    artifact: JSONDict,
    *,
    bundle_path: Path,
    prompt_set_path: Path,
    candidate_trace_path: Path,
    output_path: Path | None,
) -> JSONDict:
    prompt_set = load_json(prompt_set_path)
    validation = validate_trace_receipt_artifact(
        artifact,
        bundle_path=bundle_path,
        prompt_set_path=prompt_set_path,
        candidate_trace_path=candidate_trace_path,
        expected_model_id=str(prompt_set.get("model_id")) if prompt_set.get("model_id") is not None else None,
        expected_backend_family=str(prompt_set.get("backend_family")) if prompt_set.get("backend_family") is not None else None,
        expected_prompt_family=str(prompt_set.get("prompt_family")) if prompt_set.get("prompt_family") is not None else None,
    )
    return {
        "mode": "phase3_trace_receipt_builder",
        "schema_version": BUILDER_SCHEMA_VERSION,
        "artifact_schema_version": artifact.get("schema_version"),
        "valid": validation["shape_valid"],
        "receipt_ready": validation["receipt_ready"],
        "errors": validation["errors"],
        "output_path": display_path(output_path),
        "source_bundle_path": display_path(bundle_path),
        "source_prompt_set_path": display_path(prompt_set_path),
        "candidate_trace_path": display_path(candidate_trace_path),
        "runtime_backend": artifact.get("runtime_backend"),
        "model_id": artifact.get("model_id"),
        "prompt_family": artifact.get("prompt_family"),
        "safety_contract": safety_contract(),
        "next_actions": [
            "Fill the receipt from an approved candidate router trace capture.",
            "Keep the receipt paired to the same request, prompt set, trace path, model, backend, and prompt family.",
            "Run the policy-candidate trace planner with this receipt before bundle promotion.",
        ],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_path", nargs="?", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--prompt-set-path", type=Path)
    parser.add_argument("--candidate-trace-path", type=Path)
    parser.add_argument("--input-receipt", type=Path, help="validate an existing receipt instead of building a template")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--default-output", action="store_true", help="write beside the default candidate trace path")
    parser.add_argument("--artifact-json", action="store_true", help="print the trace receipt artifact")
    parser.add_argument("--json", action="store_true", help="emit machine-readable builder summary")
    return parser


def plan_build(args: argparse.Namespace) -> tuple[int, JSONDict | None, JSONDict | None, str | None]:
    try:
        prompt_path = args.prompt_set_path or canonical_prompt_set_path(args.bundle_path)
        candidate_path = args.candidate_trace_path or default_candidate_trace_path(args.bundle_path)
        output_path = args.output
        if args.default_output:
            if output_path is not None:
                return 2, None, None, "--output and --default-output cannot be used together"
            output_path = default_trace_receipt_path(candidate_path)
        if args.input_receipt is not None:
            artifact = load_json(args.input_receipt)
            output_path = output_path or args.input_receipt
        else:
            artifact = build_template_artifact(
                args.bundle_path,
                prompt_set_path=prompt_path,
                candidate_trace_path=candidate_path,
            )
        summary = build_summary(
            artifact,
            bundle_path=args.bundle_path,
            prompt_set_path=prompt_path,
            candidate_trace_path=candidate_path,
            output_path=output_path,
        )
        if output_path is not None and args.input_receipt is None and summary["valid"]:
            write_artifact(output_path, artifact)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, None, f"Could not build Phase 3 trace receipt: {exc}"
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
        print("MoE Run Anyway Phase 3 trace-receipt builder")
        print(f"Valid: {summary['valid']}")
        print(f"Receipt ready: {summary['receipt_ready']}")
        print(f"Output path: {summary['output_path']}")
        print(f"Candidate trace: {summary['candidate_trace_path']}")
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
