#!/usr/bin/env python3
"""Build a Phase 3 real-evidence bundle manifest.

The builder writes or prints the manifest consumed by
plan_phase3_real_evidence_bundle.py. It records artifact paths, capture receipts, and approval
metadata only; it does not launch runtimes, download models, inspect secrets,
mutate residency, or send prompt traffic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import plan_phase3_real_evidence_bundle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_PATH = ROOT / "memory-moe-mvp" / "data" / "llama_cpp_router_trace.fixture.jsonl"
DEFAULT_INVENTORY_PATH = ROOT / "memory-moe-mvp" / "data" / "expert_inventory_manifest.fixture.json"
DEFAULT_POLICIES_PATH = ROOT / "memory-moe-mvp" / "data" / "baseline_replay_policies.json"
DEFAULT_MANAGED_PLAN_PATH = ROOT / "memory-moe-mvp" / "data" / "managed_expert_loading_plan.json"
DEFAULT_OUTPUT_PATH = ROOT / "memory-moe-mvp" / "phase3-real-evidence" / "phase3_real_evidence_bundle.json"
SUPPORTED_SCHEMA_VERSION = "moe-phase3-real-evidence-bundle-builder-v1"

JSONDict = dict[str, Any]


def display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def safety_contract() -> list[str]:
    return [
        "bundle builder writes manifest metadata only",
        "bundle builder does not launch model servers",
        "bundle builder does not download models",
        "bundle builder does not inspect private tokens",
        "bundle builder does not mutate runtime residency",
        "bundle builder does not send prompt traffic",
        "bundle builder does not claim live expert paging",
    ]


def build_manifest(
    *,
    name: str,
    model_id: str,
    source_format: str,
    backend_family: str,
    prompt_family: str,
    trace_path: Path,
    trace_receipt_path: Path | None = None,
    inventory_path: Path,
    policies_path: Path,
    managed_plan_path: Path,
    fallback_artifact_path: Path | None,
    real_model_trace_capture_approved: bool,
    dense_fallback_capture_approved: bool,
    runtime_prompt_traffic_approved: bool,
    approval_notes: list[str],
    live_proof_artifact_path: Path | None = None,
) -> JSONDict:
    notes = approval_notes or [
        "Builder records approval metadata only; it does not grant permission to run live work."
    ]
    return {
        "schema_version": plan_phase3_real_evidence_bundle.SUPPORTED_SCHEMA_VERSION,
        "name": name,
        "model_id": model_id,
        "source_format": source_format,
        "backend_family": backend_family,
        "prompt_family": prompt_family,
        "artifact_paths": {
            "trace_path": display_path(trace_path),
            "trace_receipt_path": display_path(trace_receipt_path),
            "inventory_path": display_path(inventory_path),
            "policies_path": display_path(policies_path),
            "managed_plan_path": display_path(managed_plan_path),
            "fallback_artifact_path": display_path(fallback_artifact_path),
            "live_proof_artifact_path": display_path(live_proof_artifact_path),
        },
        "approvals": {
            "real_model_trace_capture_approved": real_model_trace_capture_approved,
            "dense_fallback_capture_approved": dense_fallback_capture_approved,
            "runtime_prompt_traffic_approved": runtime_prompt_traffic_approved,
            "notes": notes,
        },
        "safety_contract": [
            "Bundle validation reads local artifacts only.",
            "Bundle validation does not launch model servers.",
            "Bundle validation does not download models.",
            "Bundle validation does not inspect private tokens.",
            "Bundle validation does not send prompt traffic.",
            "Bundle validation does not claim live expert paging.",
        ],
        "next_actions": [
            "Validate this bundle with scripts/plan_phase3_real_evidence_bundle.py.",
            "Keep the bundle valid-but-not-ready until real-model pairing and fallback comparison are proven.",
            "Use the bundle as the Phase 3 handoff artifact before Phase 4 adapter work.",
        ],
    }


def validate_built_manifest(manifest: JSONDict) -> tuple[list[str], dict[str, JSONDict]]:
    shape_errors = plan_phase3_real_evidence_bundle.validate_manifest_shape(manifest)
    statuses, path_errors, _ = plan_phase3_real_evidence_bundle.artifact_statuses(manifest)
    return [*shape_errors, *path_errors], statuses


def write_manifest(path: Path, manifest: JSONDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_summary(manifest: JSONDict, *, output_path: Path | None, errors: list[str], statuses: dict[str, JSONDict]) -> JSONDict:
    validator_command = [
        "uv",
        "run",
        "--managed-python",
        "--python",
        "3.13",
        "scripts/plan_phase3_real_evidence_bundle.py",
        display_path(output_path) or "<phase3_bundle_manifest_path>",
        "--json",
    ]
    return {
        "mode": "phase3_real_evidence_bundle_builder",
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "valid": not errors,
        "errors": errors,
        "output_path": display_path(output_path),
        "manifest_schema_version": manifest.get("schema_version"),
        "model_id": manifest.get("model_id"),
        "source_format": manifest.get("source_format"),
        "backend_family": manifest.get("backend_family"),
        "prompt_family": manifest.get("prompt_family"),
        "artifact_statuses": statuses,
        "fallback_artifact_attached": manifest.get("artifact_paths", {}).get("fallback_artifact_path") is not None,
        "trace_receipt_attached": manifest.get("artifact_paths", {}).get("trace_receipt_path") is not None,
        "live_proof_artifact_attached": manifest.get("artifact_paths", {}).get("live_proof_artifact_path") is not None,
        "approvals": manifest.get("approvals", {}),
        "validator_command": validator_command,
        "safety_contract": safety_contract(),
        "next_actions": [
            "Write the manifest with --output once artifact paths are selected.",
            "Run the validator command before treating the bundle as Phase 3 handoff evidence.",
            "Keep approval flags false unless the corresponding runtime capture was explicitly approved.",
        ],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="Phase 3 real-evidence bundle")
    parser.add_argument("--model-id", default="fixture-mixtral.gguf")
    parser.add_argument("--source-format", default="gguf")
    parser.add_argument("--backend-family", default="llama_cpp")
    parser.add_argument("--prompt-family", default="fixture")
    parser.add_argument("--trace-path", type=Path, default=DEFAULT_TRACE_PATH)
    parser.add_argument("--trace-receipt-path", type=Path)
    parser.add_argument("--inventory-path", type=Path, default=DEFAULT_INVENTORY_PATH)
    parser.add_argument("--policies-path", type=Path, default=DEFAULT_POLICIES_PATH)
    parser.add_argument("--managed-plan-path", type=Path, default=DEFAULT_MANAGED_PLAN_PATH)
    parser.add_argument("--fallback-artifact-path", type=Path)
    parser.add_argument("--live-proof-artifact-path", type=Path)
    parser.add_argument("--real-model-trace-capture-approved", action="store_true")
    parser.add_argument("--dense-fallback-capture-approved", action="store_true")
    parser.add_argument("--runtime-prompt-traffic-approved", action="store_true")
    parser.add_argument(
        "--approval-note",
        action="append",
        default=[],
        help="approval/provenance note to include in the bundle manifest",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest-json", action="store_true", help="print the built bundle manifest")
    parser.add_argument("--json", action="store_true", help="emit machine-readable builder summary")
    return parser


def plan_build(args: argparse.Namespace) -> tuple[int, JSONDict | None, JSONDict | None, str | None]:
    try:
        manifest = build_manifest(
            name=args.name,
            model_id=args.model_id,
            source_format=args.source_format,
            backend_family=args.backend_family,
            prompt_family=args.prompt_family,
            trace_path=args.trace_path,
            trace_receipt_path=args.trace_receipt_path,
            inventory_path=args.inventory_path,
            policies_path=args.policies_path,
            managed_plan_path=args.managed_plan_path,
            fallback_artifact_path=args.fallback_artifact_path,
            real_model_trace_capture_approved=args.real_model_trace_capture_approved,
            dense_fallback_capture_approved=args.dense_fallback_capture_approved,
            runtime_prompt_traffic_approved=args.runtime_prompt_traffic_approved,
            approval_notes=args.approval_note,
            live_proof_artifact_path=args.live_proof_artifact_path,
        )
        if args.output:
            write_manifest(args.output, manifest)
        errors, statuses = validate_built_manifest(manifest)
        summary = build_summary(manifest, output_path=args.output, errors=errors, statuses=statuses)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, None, f"Could not build Phase 3 real-evidence bundle manifest: {exc}"
    return (0 if summary["valid"] else 2), summary, manifest, None


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary, manifest, error_message = plan_build(args)
    if error_message:
        print(error_message, file=sys.stderr)
        return status
    assert summary is not None
    assert manifest is not None
    if args.manifest_json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    elif args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("MoE Run Anyway Phase 3 real-evidence bundle builder")
        print(f"Valid: {summary['valid']}")
        print(f"Output path: {summary['output_path']}")
        print(f"Fallback attached: {summary['fallback_artifact_attached']}")
        print(f"Trace receipt attached: {summary['trace_receipt_attached']}")
        print(f"Live proof attached: {summary['live_proof_artifact_attached']}")
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
