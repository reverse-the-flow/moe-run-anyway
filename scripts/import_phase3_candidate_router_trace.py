#!/usr/bin/env python3
"""Import an already-captured llama.cpp router trace into Phase 3 evidence.

This importer fills the gap between an operator-saved router trace JSONL file
and the canonical Phase 3 candidate-trace artifact expected by the replay and
go/no-go planners. It validates and copies local files only; it does not launch
runtimes, call endpoints, inspect secrets, mutate residency, or send prompt
traffic.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import build_phase3_trace_receipt
import phase3_trace_receipts
import plan_phase3_reuse_evidence_capture
import validate_llama_cpp_router_trace


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_SCHEMA_VERSION = "moe-phase3-candidate-router-trace-import-v1"
DEFAULT_REQUIRED_KINDS = ("selected_experts", "selected_weights", "selected_weights_norm")
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


def write_json(path: Path, payload: JSONDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def artifact_items_by_id(request: JSONDict) -> dict[str, JSONDict]:
    items = request.get("requested_artifacts")
    if not isinstance(items, list):
        return {}
    result: dict[str, JSONDict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        if isinstance(item_id, str) and item_id:
            result[item_id] = item
    return result


def candidate_artifact(request: JSONDict) -> JSONDict:
    return artifact_items_by_id(request).get("candidate_router_trace", {})


def source_path_from_candidate(candidate: JSONDict, key: str) -> Path | None:
    source = candidate.get("source")
    if not isinstance(source, dict):
        return None
    return resolve_repo_path(source.get(key))


def default_candidate_trace_path(request: JSONDict) -> Path | None:
    return resolve_repo_path(candidate_artifact(request).get("path"))


def default_candidate_receipt_path(request: JSONDict, candidate_trace_path: Path | None) -> Path | None:
    from_request = source_path_from_candidate(candidate_artifact(request), "capture_receipt_path")
    if from_request is not None:
        return from_request
    if candidate_trace_path is None:
        return None
    return candidate_trace_path.with_name(f"{candidate_trace_path.stem}.capture-receipt.json")


def default_prompt_set_path(request: JSONDict) -> Path | None:
    return source_path_from_candidate(candidate_artifact(request), "prompt_set_path")


def default_bundle_path(request: JSONDict) -> Path | None:
    return resolve_repo_path(request.get("bundle_path"))


def request_approval(request: JSONDict, key: str) -> bool:
    approvals = request.get("approvals")
    return isinstance(approvals, dict) and approvals.get(key) is True


def safety_contract() -> list[str]:
    return [
        "candidate-router-trace importer reads and writes local artifacts only",
        "candidate-router-trace importer validates llama.cpp router trace JSONL before copying",
        "candidate-router-trace importer does not launch model servers",
        "candidate-router-trace importer does not call endpoints",
        "candidate-router-trace importer does not run Docker",
        "candidate-router-trace importer does not download models",
        "candidate-router-trace importer does not inspect private tokens",
        "candidate-router-trace importer does not send prompt traffic",
        "candidate-router-trace importer does not mutate runtime residency",
        "candidate-router-trace importer does not claim live expert paging",
    ]


def utc_now_text() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_required_kinds(values: list[str] | None) -> set[str]:
    if not values:
        return set(DEFAULT_REQUIRED_KINDS)
    return set(values)


def trace_model_errors(prompt_set: JSONDict, trace_summary: JSONDict) -> list[str]:
    model_id = prompt_set.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        return ["prompt set model_id is required"]
    by_model = trace_summary.get("by_model")
    if not isinstance(by_model, dict) or not by_model:
        return ["trace summary must include at least one model"]
    trace_models = sorted(str(model) for model in by_model)
    if trace_models != [model_id]:
        return [f"trace model ids must match prompt set model_id; got {trace_models!r}"]
    return []


def build_ready_receipt(
    *,
    bundle_path: Path,
    request_path: Path,
    prompt_set_path: Path,
    candidate_trace_path: Path,
    source_trace_path: Path,
    prompt_set: JSONDict,
    captured_at: str,
    capture_host: str,
    operator_notes: str,
    trace_summary: JSONDict,
) -> JSONDict:
    receipt = build_phase3_trace_receipt.build_template_artifact(
        bundle_path,
        prompt_set_path=prompt_set_path,
        candidate_trace_path=candidate_trace_path,
    )
    receipt.update(
        {
            "receipt_ready": True,
            "source_request_path": display_path(request_path),
            "source_prompt_set_path": display_path(prompt_set_path),
            "candidate_trace_path": display_path(candidate_trace_path),
            "router_trace_capture_approved": True,
            "runtime_prompt_traffic_approved": True,
            "captured_at": captured_at,
            "capture_host": capture_host,
            "runtime_backend": prompt_set.get("backend_family"),
            "model_id": prompt_set.get("model_id"),
            "prompt_family": prompt_set.get("prompt_family"),
            "operator_notes": operator_notes,
            "source_trace_path": display_path(source_trace_path),
            "trace_summary": trace_summary,
            "next_actions": [
                "Run the Phase 3 policy-candidate trace planner against this canonical candidate trace.",
                "Run baseline policy replay before rebuilding the Phase 3 bundle around this candidate trace.",
                "Keep managed and dense/full-runtime output summaries paired to the same request and prompt set.",
            ],
        }
    )
    return receipt


def existing_ready_receipt(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return payload.get("receipt_ready") is True


def validate_inputs(
    *,
    request_path: Path,
    source_trace_path: Path,
    bundle_path: Path | None,
    prompt_set_path: Path | None,
    candidate_trace_path: Path | None,
    trace_receipt_path: Path | None,
    router_trace_capture_approved: bool,
    runtime_prompt_traffic_approved: bool,
    overwrite: bool,
) -> list[str]:
    errors: list[str] = []
    if not router_trace_capture_approved:
        errors.append("--approved-router-trace-capture is required")
    if not runtime_prompt_traffic_approved:
        errors.append("--approved-runtime-prompt-traffic is required")
    if not request_path.exists():
        errors.append(f"source request does not exist: {display_path(request_path) or request_path}")
    if not source_trace_path.exists():
        errors.append(f"source trace does not exist: {display_path(source_trace_path) or source_trace_path}")
    if bundle_path is None:
        errors.append("bundle path could not be resolved from request or --bundle-path")
    elif not bundle_path.exists():
        errors.append(f"bundle does not exist: {display_path(bundle_path) or bundle_path}")
    if prompt_set_path is None:
        errors.append("prompt set path could not be resolved from request or --prompt-set-path")
    elif not prompt_set_path.exists():
        errors.append(f"prompt set does not exist: {display_path(prompt_set_path) or prompt_set_path}")
    if candidate_trace_path is None:
        errors.append("candidate trace path could not be resolved from request or --candidate-trace-path")
    elif candidate_trace_path.exists() and not overwrite:
        errors.append(f"candidate trace already exists; pass --overwrite to replace it: {display_path(candidate_trace_path) or candidate_trace_path}")
    if trace_receipt_path is None:
        errors.append("trace receipt path could not be resolved from request or --trace-receipt-path")
    elif existing_ready_receipt(trace_receipt_path) and not overwrite:
        errors.append(f"ready trace receipt already exists; pass --overwrite to replace it: {display_path(trace_receipt_path) or trace_receipt_path}")
    return errors


def build_summary(
    *,
    request_path: Path,
    source_trace_path: Path,
    bundle_path: Path | None,
    prompt_set_path: Path | None,
    candidate_trace_path: Path | None,
    trace_receipt_path: Path | None,
    prompt_set: JSONDict | None,
    trace_summary: JSONDict | None,
    trace_errors: list[str],
    receipt_validation: JSONDict | None,
    prompt_identity: JSONDict | None,
    dry_run: bool,
    write_performed: bool,
    errors: list[str],
) -> JSONDict:
    return {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "mode": "phase3_candidate_router_trace_import",
        "valid": not errors,
        "ready_to_import": not errors,
        "dry_run": dry_run,
        "write_performed": write_performed,
        "errors": errors,
        "source_request_path": display_path(request_path),
        "source_trace_path": display_path(source_trace_path),
        "source_bundle_path": display_path(bundle_path),
        "source_prompt_set_path": display_path(prompt_set_path),
        "candidate_trace_path": display_path(candidate_trace_path),
        "trace_receipt_path": display_path(trace_receipt_path),
        "runtime_backend": prompt_set.get("backend_family") if prompt_set else None,
        "model_id": prompt_set.get("model_id") if prompt_set else None,
        "prompt_family": prompt_set.get("prompt_family") if prompt_set else None,
        "trace_validation": {
            "valid": not trace_errors,
            "errors": trace_errors,
            **(trace_summary or {}),
            "prompt_identity_ready": prompt_identity.get("ready") if isinstance(prompt_identity, dict) else None,
            "prompt_identity": prompt_identity,
        },
        "receipt_validation": receipt_validation,
        "safety_contract": safety_contract(),
        "next_actions": [
            "Run scripts/plan_phase3_policy_candidate_trace.py against the canonical candidate trace.",
            "Run scripts/plan_baseline_policy_replay.py before treating the trace as a policy candidate.",
            "Do not promote to live paging until replay, dense/full-runtime output, and managed-output evidence are all ready.",
        ],
    }


def plan_import(args: argparse.Namespace) -> tuple[int, JSONDict, JSONDict | None]:
    request_path = resolve_repo_path(args.request_path) or args.request_path
    source_trace_path = resolve_repo_path(args.source_trace_path) or args.source_trace_path
    errors: list[str] = []
    request: JSONDict = {}
    prompt_set: JSONDict | None = None
    receipt: JSONDict | None = None
    trace_summary: JSONDict | None = None
    prompt_identity: JSONDict | None = None
    trace_errors: list[str] = []
    receipt_validation: JSONDict | None = None
    write_performed = False

    try:
        if request_path.exists():
            request = load_json(request_path)
        bundle_path = args.bundle_path or default_bundle_path(request)
        prompt_set_path = args.prompt_set_path or default_prompt_set_path(request)
        candidate_trace_path = args.candidate_trace_path or default_candidate_trace_path(request)
        trace_receipt_path = args.trace_receipt_path or default_candidate_receipt_path(request, candidate_trace_path)

        errors.extend(
            validate_inputs(
                request_path=request_path,
                source_trace_path=source_trace_path,
                bundle_path=bundle_path,
                prompt_set_path=prompt_set_path,
                candidate_trace_path=candidate_trace_path,
                trace_receipt_path=trace_receipt_path,
                router_trace_capture_approved=args.approved_router_trace_capture,
                runtime_prompt_traffic_approved=args.approved_runtime_prompt_traffic,
                overwrite=args.overwrite,
            )
        )

        request_warnings = []
        if request and not request_approval(request, "router_trace_capture_approved"):
            request_warnings.append("source_request_router_trace_capture_approval_false")
        if request and not request_approval(request, "runtime_prompt_traffic_approved"):
            request_warnings.append("source_request_runtime_prompt_traffic_approval_false")

        if prompt_set_path is not None and prompt_set_path.exists():
            prompt_set = load_json(prompt_set_path)

        if source_trace_path.exists():
            try:
                events = validate_llama_cpp_router_trace.load_jsonl(source_trace_path)
                required_kinds = normalize_required_kinds(args.require_kind)
                trace_errors.extend(
                    validate_llama_cpp_router_trace.validate_trace(events, require_kinds=required_kinds)
                )
                trace_summary = validate_llama_cpp_router_trace.summarize_events(events)
                prompt_identity = plan_phase3_reuse_evidence_capture.prompt_identity_summary(events)
                if prompt_identity.get("ready") is not True:
                    missing_groups = ",".join(prompt_identity.get("missing_required_group_ids", []))
                    trace_errors.append(f"prompt_identity_metadata_missing:{missing_groups}")
            except ValueError as exc:
                trace_errors.append(str(exc))
        errors.extend(trace_errors)

        if prompt_set is not None and trace_summary is not None:
            errors.extend(trace_model_errors(prompt_set, trace_summary))

        if not errors and all(
            path is not None
            for path in (bundle_path, prompt_set_path, candidate_trace_path, trace_receipt_path)
        ):
            assert bundle_path is not None
            assert prompt_set_path is not None
            assert candidate_trace_path is not None
            assert trace_receipt_path is not None
            assert prompt_set is not None
            trace_summary_for_receipt = {
                **(trace_summary or {}),
                "prompt_identity_ready": prompt_identity.get("ready") if isinstance(prompt_identity, dict) else None,
                "prompt_identity": prompt_identity,
            }
            receipt = build_ready_receipt(
                bundle_path=bundle_path,
                request_path=request_path,
                prompt_set_path=prompt_set_path,
                candidate_trace_path=candidate_trace_path,
                source_trace_path=source_trace_path,
                prompt_set=prompt_set,
                captured_at=args.captured_at or utc_now_text(),
                capture_host=args.capture_host or platform.node() or "unknown-host",
                operator_notes=args.operator_notes,
                trace_summary=trace_summary_for_receipt,
            )
            receipt["operator_warnings"] = request_warnings
            if args.dry_run:
                receipt_validation = phase3_trace_receipts.validate_trace_receipt(receipt)
            else:
                candidate_trace_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_trace_path, candidate_trace_path)
                receipt_validation = build_phase3_trace_receipt.validate_trace_receipt_artifact(
                    receipt,
                    bundle_path=bundle_path,
                    prompt_set_path=prompt_set_path,
                    candidate_trace_path=candidate_trace_path,
                    expected_model_id=str(prompt_set.get("model_id")) if prompt_set.get("model_id") is not None else None,
                    expected_backend_family=str(prompt_set.get("backend_family")) if prompt_set.get("backend_family") is not None else None,
                    expected_prompt_family=str(prompt_set.get("prompt_family")) if prompt_set.get("prompt_family") is not None else None,
                )
                if not receipt_validation["shape_valid"]:
                    errors.extend(receipt_validation["errors"])
                else:
                    write_json(trace_receipt_path, receipt)
                    write_performed = True
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(str(exc))

    summary = build_summary(
        request_path=request_path,
        source_trace_path=source_trace_path,
        bundle_path=bundle_path if "bundle_path" in locals() else None,
        prompt_set_path=prompt_set_path if "prompt_set_path" in locals() else None,
        candidate_trace_path=candidate_trace_path if "candidate_trace_path" in locals() else None,
        trace_receipt_path=trace_receipt_path if "trace_receipt_path" in locals() else None,
        prompt_set=prompt_set,
        trace_summary=trace_summary,
        trace_errors=trace_errors,
        receipt_validation=receipt_validation,
        prompt_identity=prompt_identity,
        dry_run=args.dry_run,
        write_performed=write_performed,
        errors=errors,
    )
    return (0 if not errors else 2), summary, receipt


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request_path", type=Path)
    parser.add_argument("source_trace_path", type=Path)
    parser.add_argument("--bundle-path", type=Path)
    parser.add_argument("--prompt-set-path", type=Path)
    parser.add_argument("--candidate-trace-path", type=Path)
    parser.add_argument("--trace-receipt-path", type=Path)
    parser.add_argument("--capture-host")
    parser.add_argument("--captured-at")
    parser.add_argument(
        "--operator-notes",
        default="imported from existing approved llama.cpp router trace artifact",
    )
    parser.add_argument("--require-kind", choices=sorted(validate_llama_cpp_router_trace.SUPPORTED_KINDS), action="append")
    parser.add_argument("--approved-router-trace-capture", action="store_true")
    parser.add_argument("--approved-runtime-prompt-traffic", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--artifact-json", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary, receipt = plan_import(args)
    if args.artifact_json and receipt is not None:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    elif args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Phase 3 candidate router trace import: {summary['valid']}")
        print(f"Source trace: {summary['source_trace_path']}")
        print(f"Candidate trace: {summary['candidate_trace_path']}")
        print(f"Receipt: {summary['trace_receipt_path']}")
        print(f"Dry run: {summary['dry_run']}")
        print(f"Write performed: {summary['write_performed']}")
        for error in summary["errors"]:
            print(f"Error: {error}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
