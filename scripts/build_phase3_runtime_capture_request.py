#!/usr/bin/env python3
"""Build a Phase 3 runtime-capture request artifact.

The request binds the shared prompt set, candidate router-trace target, managed
output summary template, dense/full-runtime output summary template, and live-capability proof template into one
operator-facing plan. It records approvals and validator commands only; it does
not launch runtimes, run Docker, call endpoints, inspect secrets, mutate
residency, or send prompt traffic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import build_phase3_trace_receipt
import plan_phase3_dense_fallback_capture
import plan_phase3_live_capability_proof
import plan_phase3_real_evidence_bundle
import plan_phase3_reuse_evidence_capture


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_PATH = ROOT / "memory-moe-mvp" / "phase3-real-evidence" / "pc_mixtral_phase3_real_evidence_bundle.json"
SUPPORTED_SCHEMA_VERSION = "moe-phase3-runtime-capture-request-v1"
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


def uv_command(*args: str) -> list[str]:
    return ["uv", "run", "--managed-python", "--python", "3.13", *args]


def canonical_prompt_set_path(bundle_path: Path) -> Path:
    return bundle_path.with_name(f"{bundle_path.stem}.prompt-set.json")


def default_policy_candidate_trace_path(bundle_path: Path) -> Path:
    return bundle_path.with_name(f"{bundle_path.stem}-policy-candidate") / "candidate-router-events.jsonl"


def default_policy_candidate_trace_receipt_path(candidate_trace_path: Path) -> Path:
    return candidate_trace_path.with_name(f"{candidate_trace_path.stem}.capture-receipt.json")


def default_fallback_dir(bundle_path: Path) -> Path:
    return bundle_path.with_name(f"{bundle_path.stem}-fallback")


def default_output_summary_path(bundle_path: Path, label: str) -> Path:
    return default_fallback_dir(bundle_path) / f"{label}-output-summary.json"


def default_request_path(bundle_path: Path) -> Path:
    return bundle_path.with_name(f"{bundle_path.stem}.runtime-capture-request.json")


def default_live_proof_template_path(bundle_path: Path) -> Path:
    return bundle_path.with_name(f"{bundle_path.stem}.live-capability-proof.template.json")


def artifact_path(manifest: JSONDict, key: str) -> Path | None:
    paths = manifest.get("artifact_paths")
    if not isinstance(paths, dict):
        return None
    return resolve_repo_path(paths.get(key))


def request_status(*, approved: bool, source_ready: bool, target_ready: bool) -> str:
    if target_ready:
        return "already_satisfied"
    if not source_ready:
        return "blocked_by_missing_template"
    return "ready_for_operator_capture" if approved else "approval_required"


def live_proof_status(summary: JSONDict) -> str:
    if summary.get("proof_ready") is True:
        return "already_satisfied"
    if summary.get("proof_available") is True and summary.get("valid") is True:
        return "future_adapter_required"
    return "blocked_by_missing_template"


def load_manifest(bundle_path: Path) -> JSONDict:
    return plan_phase3_real_evidence_bundle.load_manifest(bundle_path)


def load_json(path: Path) -> JSONDict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{display_path(path) or path} must be a JSON object")
    return data


def summarize_trace_receipt(
    path: Path,
    *,
    bundle_path: Path,
    prompt_set_path: Path,
    candidate_trace_path: Path,
    prompt_set: JSONDict,
) -> JSONDict:
    if not path.exists():
        return {
            "path": display_path(path),
            "exists": False,
            "valid": False,
            "ready": False,
            "errors": [],
            "blockers": ["candidate_trace_capture_receipt_missing"],
        }
    receipt = load_json(path)
    validation = build_phase3_trace_receipt.validate_trace_receipt_artifact(
        receipt,
        bundle_path=bundle_path,
        prompt_set_path=prompt_set_path,
        candidate_trace_path=candidate_trace_path,
        expected_model_id=str(prompt_set.get("model_id")) if prompt_set.get("model_id") is not None else None,
        expected_backend_family=str(prompt_set.get("backend_family")) if prompt_set.get("backend_family") is not None else None,
        expected_prompt_family=str(prompt_set.get("prompt_family")) if prompt_set.get("prompt_family") is not None else None,
    )
    blockers: list[str] = []
    if validation["shape_valid"] and validation["receipt_ready"] is not True:
        blockers.append("candidate_trace_capture_receipt_not_ready")
    if validation["shape_valid"] is not True:
        blockers.append("candidate_trace_capture_receipt_invalid")
    return {
        "path": display_path(path),
        "exists": True,
        "valid": validation["shape_valid"],
        "ready": validation["receipt_ready"] is True,
        "errors": validation["errors"],
        "blockers": blockers,
    }


def build_request(
    bundle_path: Path,
    *,
    prompt_set_path: Path | None = None,
    candidate_trace_path: Path | None = None,
    candidate_trace_receipt_path: Path | None = None,
    managed_output_path: Path | None = None,
    dense_output_path: Path | None = None,
    live_proof_template_path: Path | None = None,
    router_trace_capture_approved: bool = False,
    managed_output_capture_approved: bool = False,
    dense_output_capture_approved: bool = False,
    runtime_prompt_traffic_approved: bool = False,
) -> JSONDict:
    manifest = load_manifest(bundle_path)
    prompt_path = prompt_set_path or canonical_prompt_set_path(bundle_path)
    candidate_path = candidate_trace_path or default_policy_candidate_trace_path(bundle_path)
    candidate_receipt_path = candidate_trace_receipt_path or default_policy_candidate_trace_receipt_path(candidate_path)
    managed_path = managed_output_path or default_output_summary_path(bundle_path, "managed")
    dense_path = dense_output_path or default_output_summary_path(bundle_path, "dense")
    live_proof_path = live_proof_template_path or default_live_proof_template_path(bundle_path)

    prompt_payload = load_json(prompt_path) if prompt_path.exists() else {}
    prompt_summary = plan_phase3_dense_fallback_capture.summarize_prompt_set(prompt_path)
    managed_summary = plan_phase3_dense_fallback_capture.summarize_output_summary(managed_path if managed_path.exists() else None, label="managed")
    dense_summary = plan_phase3_dense_fallback_capture.summarize_output_summary(dense_path if dense_path.exists() else None, label="dense")
    managed_coverage = plan_phase3_dense_fallback_capture.coverage_against_prompt_set(prompt_summary, managed_summary, label="managed")
    dense_coverage = plan_phase3_dense_fallback_capture.coverage_against_prompt_set(prompt_summary, dense_summary, label="dense")
    if live_proof_path.exists():
        live_proof_summary = plan_phase3_live_capability_proof.build_summary(live_proof_path)
    else:
        live_proof_summary = plan_phase3_live_capability_proof.missing_artifact_summary(live_proof_path)

    prompt_ready = prompt_summary.get("ready") is True
    managed_template_ready = managed_summary.get("valid") is True and managed_summary.get("exists") is True and managed_coverage.get("ready") is True
    dense_template_ready = dense_summary.get("valid") is True and dense_summary.get("exists") is True and dense_coverage.get("ready") is True
    managed_output_ready = managed_summary.get("ready") is True and managed_coverage.get("ready") is True
    dense_output_ready = dense_summary.get("ready") is True and dense_coverage.get("ready") is True
    approvals = {
        "router_trace_capture_approved": router_trace_capture_approved,
        "managed_output_capture_approved": managed_output_capture_approved,
        "dense_output_capture_approved": dense_output_capture_approved,
        "runtime_prompt_traffic_approved": runtime_prompt_traffic_approved,
    }
    router_approved = router_trace_capture_approved and runtime_prompt_traffic_approved
    managed_approved = managed_output_capture_approved and runtime_prompt_traffic_approved
    dense_approved = dense_output_capture_approved and runtime_prompt_traffic_approved
    candidate_trace_exists = candidate_path.exists()
    candidate_receipt_summary = summarize_trace_receipt(
        candidate_receipt_path,
        bundle_path=bundle_path,
        prompt_set_path=prompt_path,
        candidate_trace_path=candidate_path,
        prompt_set=prompt_payload if isinstance(prompt_payload, dict) else {},
    )
    candidate_reuse_summary = plan_phase3_reuse_evidence_capture.trace_summary(
        candidate_path,
        receipt_path=candidate_receipt_path,
        request_path=default_request_path(bundle_path),
        prompt_set_path=prompt_path,
    )
    candidate_trace_ready = candidate_trace_exists and candidate_reuse_summary.get("reuse_ready") is True

    trace_path = artifact_path(manifest, "trace_path")
    inventory_path = artifact_path(manifest, "inventory_path")
    policies_path = artifact_path(manifest, "policies_path")

    future_artifacts = [
        {
            "id": "live_capability_proof_fill",
            "status": live_proof_status(live_proof_summary),
            "approval_required": live_proof_summary.get("proof_ready") is not True,
            "path": display_path(live_proof_path),
            "description": "Fill the live-capability proof with approved residency observation/control and cleanup/restore evidence.",
            "source": {
                "proof_available": live_proof_summary.get("proof_available"),
                "proof_ready": live_proof_summary.get("proof_ready"),
                "phase_3_gate": live_proof_summary.get("phase_3_gate", {}),
                "blockers": live_proof_summary.get("blockers", []),
            },
            "validator_commands": [
                uv_command(
                    "scripts/plan_phase3_live_capability_proof.py",
                    display_path(live_proof_path) or str(live_proof_path),
                    "--json",
                )
            ],
        }
    ]

    requested_artifacts = [
        {
            "id": "candidate_router_trace",
            "status": request_status(approved=router_approved, source_ready=prompt_ready, target_ready=candidate_trace_ready),
            "approval_required": not candidate_trace_ready,
            "path": display_path(candidate_path),
            "description": "Capture semantic llama.cpp router events with per-event prompt identity metadata for the shared repeated prompt set and mark the trace capture_receipt ready.",
            "source": {
                "prompt_set_path": display_path(prompt_path),
                "prompt_set_ready": prompt_ready,
                "capture_receipt_required": True,
                "capture_receipt_path": display_path(candidate_receipt_path),
                "capture_receipt_ready": candidate_receipt_summary.get("ready") is True,
                "capture_receipt_valid": candidate_receipt_summary.get("valid") is True,
                "capture_receipt_errors": candidate_receipt_summary.get("errors", []),
                "capture_receipt_blockers": candidate_receipt_summary.get("blockers", []),
                "candidate_trace_exists": candidate_trace_exists,
                "candidate_trace_valid": candidate_reuse_summary.get("valid") is True,
                "candidate_trace_reuse_ready": candidate_reuse_summary.get("reuse_ready") is True,
                "candidate_trace_prompt_identity_ready": candidate_reuse_summary.get("prompt_identity_ready") is True,
                "candidate_trace_prompt_identity_fields_present": candidate_reuse_summary.get("prompt_identity_fields_present", []),
                "candidate_trace_classification": candidate_reuse_summary.get("classification"),
                "candidate_trace_reuse_distance_observations": candidate_reuse_summary.get("reuse_distance", {}).get("observations", 0) if isinstance(candidate_reuse_summary.get("reuse_distance"), dict) else 0,
                "candidate_trace_repeated_route_key_count": candidate_reuse_summary.get("route_repetition", {}).get("repeated_route_key_count", 0) if isinstance(candidate_reuse_summary.get("route_repetition"), dict) else 0,
                "candidate_trace_blockers": candidate_reuse_summary.get("blockers", []),
                "candidate_trace_warnings": candidate_reuse_summary.get("warnings", []),
                "receipt_fill_note": "trace capture_receipt must record approved request path, prompt set path, trace path, model/backend, host, timestamp, approval flags, and preserve per-event prompt identity metadata in the trace",
            },
            "validator_commands": [
                uv_command(
                    "scripts/validate_llama_cpp_router_trace.py",
                    display_path(candidate_path) or str(candidate_path),
                    "--require-kind",
                    "selected_experts",
                    "--require-kind",
                    "selected_weights",
                    "--require-kind",
                    "selected_weights_norm",
                    "--require-prompt-identity",
                    "--min-reuse-distance-observations",
                    str(plan_phase3_reuse_evidence_capture.MIN_REUSE_DISTANCE_OBSERVATIONS),
                    "--json",
                ),
                uv_command(
                    "scripts/plan_baseline_policy_replay.py",
                    display_path(candidate_path) or str(candidate_path),
                    display_path(inventory_path) or "<inventory_path>",
                    display_path(policies_path) or "<policies_path>",
                    "--json",
                ),
                uv_command(
                    "scripts/plan_phase3_policy_candidate_trace.py",
                    display_path(bundle_path) or str(bundle_path),
                    "--candidate-prompt-set-path",
                    display_path(prompt_path) or str(prompt_path),
                    "--candidate-trace-path",
                    display_path(candidate_path) or str(candidate_path),
                    "--candidate-trace-receipt-path",
                    display_path(candidate_receipt_path) or str(candidate_receipt_path),
                    "--json",
                ),
            ],
        },
        {
            "id": "managed_output_summary_fill",
            "status": request_status(approved=managed_approved, source_ready=managed_template_ready, target_ready=managed_output_ready),
            "approval_required": not managed_output_ready,
            "path": display_path(managed_path),
            "description": "Fill the managed-path output summary template and mark capture_receipt ready using the same approved request.",
            "source": {
                "template_valid": managed_template_ready,
                "prompt_coverage": managed_coverage,
                "capture_receipt_required": True,
                "capture_receipt_ready": managed_summary.get("capture_receipt_ready") is True,
                "receipt_fill_note": "capture_receipt must record approved request path, host/backend, timestamp, and approval flags",
            },
            "validator_commands": [
                uv_command(
                    "scripts/build_phase3_output_summary.py",
                    display_path(prompt_path) or str(prompt_path),
                    "--output-label",
                    "managed",
                    "--input-summary",
                    display_path(managed_path) or str(managed_path),
                    "--json",
                )
            ],
        },
        {
            "id": "dense_output_summary_fill",
            "status": request_status(approved=dense_approved, source_ready=dense_template_ready, target_ready=dense_output_ready),
            "approval_required": not dense_output_ready,
            "path": display_path(dense_path),
            "description": "Fill the dense/full-runtime output summary template and mark capture_receipt ready using the same approved request.",
            "source": {
                "template_valid": dense_template_ready,
                "prompt_coverage": dense_coverage,
                "capture_receipt_required": True,
                "capture_receipt_ready": dense_summary.get("capture_receipt_ready") is True,
                "receipt_fill_note": "capture_receipt must record approved request path, host/backend, timestamp, and approval flags",
            },
            "validator_commands": [
                uv_command(
                    "scripts/build_phase3_output_summary.py",
                    display_path(prompt_path) or str(prompt_path),
                    "--output-label",
                    "dense",
                    "--input-summary",
                    display_path(dense_path) or str(dense_path),
                    "--json",
                )
            ],
        },
    ]

    errors: list[str] = []
    errors.extend(prompt_summary.get("errors", []))
    errors.extend(f"managed output summary: {error}" for error in managed_summary.get("errors", []))
    errors.extend(f"dense output summary: {error}" for error in dense_summary.get("errors", []))
    if not prompt_ready:
        errors.append("shared prompt set is not ready")
    if not managed_template_ready:
        errors.append("managed output-summary template is not valid or does not cover the prompt set")
    if not dense_template_ready:
        errors.append("dense output-summary template is not valid or does not cover the prompt set")

    return {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "mode": "phase3_runtime_capture_request",
        "valid": not errors,
        "errors": errors,
        "bundle_path": display_path(bundle_path),
        "name": manifest.get("name"),
        "model_id": manifest.get("model_id"),
        "backend_family": manifest.get("backend_family"),
        "prompt_family": manifest.get("prompt_family"),
        "trace_path": display_path(trace_path),
        "inventory_path": display_path(inventory_path),
        "policies_path": display_path(policies_path),
        "prompt_set": prompt_summary,
        "managed_output_summary": managed_summary,
        "dense_output_summary": dense_summary,
        "live_capability_proof": live_proof_summary,
        "prompt_coverage": {"managed": managed_coverage, "dense": dense_coverage},
        "approvals": approvals,
        "runtime_prompt_traffic_approved": runtime_prompt_traffic_approved,
        "candidate_trace_receipt": candidate_receipt_summary,
        "candidate_trace_reuse": candidate_reuse_summary,
        "ready_for_operator_capture": not errors and all(
            item["status"] in {"ready_for_operator_capture", "already_satisfied"}
            for item in requested_artifacts
        ),
        "capture_complete": not errors and all(item["status"] == "already_satisfied" for item in requested_artifacts),
        "requested_artifacts": requested_artifacts,
        "future_artifacts": future_artifacts,
        "safety_contract": [
            "runtime-capture request reads and writes local metadata only",
            "runtime-capture request does not launch model servers",
            "runtime-capture request does not run Docker",
            "runtime-capture request does not call endpoints",
            "runtime-capture request does not download models",
            "runtime-capture request does not inspect private tokens",
            "runtime-capture request does not send prompt traffic",
            "runtime-capture request does not mutate runtime residency",
            "runtime-capture request does not claim live expert paging",
        ],
        "next_actions": [
            "Grant runtime approval only when the target host is quiet enough for capture.",
            "Capture the candidate router trace with the shared prompt set and per-event prompt identity metadata, fill its trace capture_receipt, then run the listed replay validators.",
            "Fill the managed and dense output summaries from approved runtime outputs, mark their capture_receipts ready, then build the fallback comparison artifact.",
            "Fill the live-capability proof template only after an approved backend adapter can observe/control residency and prove cleanup.",
        ],
    }


def validate_request(summary: JSONDict) -> list[str]:
    errors: list[str] = []
    if summary.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        errors.append(f"schema_version must be {SUPPORTED_SCHEMA_VERSION!r}")
    if not isinstance(summary.get("requested_artifacts"), list) or not summary["requested_artifacts"]:
        errors.append("requested_artifacts must be a non-empty list")
    requested_by_id = {item.get("id"): item for item in summary.get("requested_artifacts", []) if isinstance(item, dict)}
    ids = list(requested_by_id)
    for required_id in ("candidate_router_trace", "managed_output_summary_fill", "dense_output_summary_fill"):
        if required_id not in ids:
            errors.append(f"requested_artifacts missing {required_id}")
    candidate_source = requested_by_id.get("candidate_router_trace", {}).get("source", {})
    if not isinstance(candidate_source, dict) or candidate_source.get("capture_receipt_required") is not True:
        errors.append("requested_artifacts.candidate_router_trace.source.capture_receipt_required must be true")
    if isinstance(candidate_source, dict) and not candidate_source.get("capture_receipt_path"):
        errors.append("requested_artifacts.candidate_router_trace.source.capture_receipt_path is required")
    future_ids = [item.get("id") for item in summary.get("future_artifacts", []) if isinstance(item, dict)]
    if "live_capability_proof_fill" not in future_ids:
        errors.append("future_artifacts missing live_capability_proof_fill")
    return errors


def write_request(path: Path, summary: JSONDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_path", nargs="?", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--prompt-set-path", type=Path)
    parser.add_argument("--candidate-trace-path", type=Path)
    parser.add_argument("--candidate-trace-receipt-path", type=Path)
    parser.add_argument("--managed-output-path", type=Path)
    parser.add_argument("--dense-output-path", type=Path)
    parser.add_argument("--live-proof-template-path", type=Path)
    parser.add_argument("--router-trace-capture-approved", action="store_true")
    parser.add_argument("--managed-output-capture-approved", action="store_true")
    parser.add_argument("--dense-output-capture-approved", action="store_true")
    parser.add_argument("--runtime-prompt-traffic-approved", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--default-output", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    return parser


def plan_build(args: argparse.Namespace) -> tuple[int, JSONDict | None, str | None]:
    try:
        output_path = args.output
        if args.default_output:
            if output_path is not None:
                return 2, None, "--output and --default-output cannot be used together"
            output_path = default_request_path(args.bundle_path)
        summary = build_request(
            args.bundle_path,
            prompt_set_path=args.prompt_set_path,
            candidate_trace_path=args.candidate_trace_path,
            candidate_trace_receipt_path=getattr(args, "candidate_trace_receipt_path", None),
            managed_output_path=args.managed_output_path,
            dense_output_path=args.dense_output_path,
            live_proof_template_path=args.live_proof_template_path,
            router_trace_capture_approved=args.router_trace_capture_approved,
            managed_output_capture_approved=args.managed_output_capture_approved,
            dense_output_capture_approved=args.dense_output_capture_approved,
            runtime_prompt_traffic_approved=args.runtime_prompt_traffic_approved,
        )
        validation_errors = validate_request(summary)
        summary["errors"] = [*summary.get("errors", []), *validation_errors]
        summary["valid"] = not summary["errors"]
        if output_path is not None and summary["valid"]:
            write_request(output_path, summary)
            summary["output_path"] = display_path(output_path)
        else:
            summary["output_path"] = display_path(output_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not build Phase 3 runtime-capture request: {exc}"
    return (0 if summary["valid"] else 2), summary, None


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary, error_message = plan_build(args)
    if error_message:
        print(error_message, file=sys.stderr)
        return status
    assert summary is not None
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("MoE Run Anyway Phase 3 runtime-capture request")
        print(f"Valid: {summary['valid']}")
        print(f"Ready for operator capture: {summary['ready_for_operator_capture']}")
        print(f"Capture complete: {summary['capture_complete']}")
        print(f"Output path: {summary['output_path']}")
        if summary["errors"]:
            print("Errors:")
            for error in summary["errors"]:
                print(f"  - {error}")
        print("Requested artifacts:")
        for item in summary["requested_artifacts"]:
            print(f"  - {item['id']}: {item['status']}")
        print("Safety contract:")
        for item in summary["safety_contract"]:
            print(f"  - {item}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
