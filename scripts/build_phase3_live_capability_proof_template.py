#!/usr/bin/env python3
"""Build a Phase 3 live-capability proof template.

The template is a valid but not-ready proof artifact for live residency
observation, residency control, cleanup/restore, and artifact export. It gives
an operator a concrete file to fill after approved runtime work, but it does
not launch runtimes, run Docker, call endpoints, inspect secrets, mutate
residency, or send prompt traffic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import plan_phase3_live_capability_proof
import plan_phase3_real_evidence_bundle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_PATH = ROOT / "memory-moe-mvp" / "phase3-real-evidence" / "pc_mixtral_phase3_real_evidence_bundle.json"
BUILDER_SCHEMA_VERSION = "moe-phase3-live-capability-proof-template-builder-v1"
JSONDict = dict[str, Any]


def display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def uv_command(*args: str) -> list[str]:
    return ["uv", "run", "--managed-python", "--python", "3.13", *args]


def default_output_path(bundle_path: Path) -> Path:
    return bundle_path.with_name(f"{bundle_path.stem}.live-capability-proof.template.json")


def safety_contract() -> list[str]:
    return [
        "live-capability proof template builder reads and writes local JSON metadata only",
        "live-capability proof template builder does not launch model servers",
        "live-capability proof template builder does not call endpoints",
        "live-capability proof template builder does not run Docker",
        "live-capability proof template builder does not download models",
        "live-capability proof template builder does not inspect private tokens",
        "live-capability proof template builder does not send prompt traffic",
        "live-capability proof template builder does not mutate runtime residency",
        "live-capability proof template builder does not claim live expert paging",
    ]


def build_template_artifact(
    bundle_path: Path,
    *,
    proof_scope: str | None = None,
    output_path: Path | None = None,
) -> JSONDict:
    manifest = plan_phase3_real_evidence_bundle.load_manifest(bundle_path)
    artifact_path = output_path or default_output_path(bundle_path)
    scope = proof_scope or "phase3-live-capability-proof-template"
    return {
        "schema_version": plan_phase3_live_capability_proof.SUPPORTED_SCHEMA_VERSION,
        "name": f"{manifest.get('name', 'Phase 3 bundle')} live capability proof template",
        "model_id": manifest.get("model_id"),
        "backend_family": manifest.get("backend_family"),
        "prompt_family": manifest.get("prompt_family"),
        "source_bundle_path": display_path(bundle_path),
        "proof_scope": scope,
        "residency_observation": {
            "status": "planned",
            "before_state_captured": False,
            "after_state_captured": False,
            "evidence_fields": [
                "resident_expert_count",
                "resident_bytes",
                "layer_expert_keys",
                "backend_session_id",
                "observation_timestamp",
            ],
            "notes": "fill from approved before/after residency observations",
        },
        "residency_control": {
            "status": "planned",
            "supported_actions": ["observe", "preload", "pin", "evict", "restore_dense", "abort_run"],
            "actuator_boundary": "fill after selecting a concrete backend adapter or patch point",
            "dry_run_only": True,
            "notes": "set dry_run_only false only after a real non-dry-run actuator proves control",
        },
        "cleanup_restore": {
            "status": "planned",
            "restore_verified": False,
            "cleanup_actions": ["restore_dense", "clear_policy_state", "abort_run"],
            "failure_path_tested": False,
            "notes": "record the cleanup/rollback path and a failure-path test before live mutation",
        },
        "artifact_export": {
            "status": "planned",
            "artifact_paths": [display_path(artifact_path)],
            "notes": "replace with exported proof artifacts from the approved runtime capture",
        },
        "safety_contract": safety_contract(),
        "next_actions": [
            "Fill this artifact only after approved live residency observation and control work.",
            "Validate the filled artifact before attaching it to a Phase 3 bundle or go/no-go decision.",
            "Keep live-spike readiness false until observation, control, cleanup, and export all validate.",
        ],
    }


def write_artifact(path: Path, artifact: JSONDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_summary(artifact: JSONDict, *, bundle_path: Path, output_path: Path | None) -> JSONDict:
    errors, section_results = plan_phase3_live_capability_proof.validate_artifact(artifact)
    blockers = [
        {"id": blocker_id, "section": section_id}
        for section_id, result in section_results.items()
        for blocker_id in result.get("blockers", [])
    ]
    proof_ready = not errors and not blockers
    return {
        "mode": "phase3_live_capability_proof_template_builder",
        "schema_version": BUILDER_SCHEMA_VERSION,
        "artifact_schema_version": artifact.get("schema_version"),
        "valid": not errors,
        "proof_ready": proof_ready,
        "template_ready": not errors and not proof_ready,
        "errors": errors,
        "blockers": blockers,
        "source_bundle_path": display_path(bundle_path),
        "output_path": display_path(output_path),
        "model_id": artifact.get("model_id"),
        "backend_family": artifact.get("backend_family"),
        "prompt_family": artifact.get("prompt_family"),
        "proof_scope": artifact.get("proof_scope"),
        "phase_3_gate": {
            "live_residency_observation_and_control_ready": (
                section_results["residency_observation"].get("ready") is True
                and section_results["residency_control"].get("ready") is True
            ),
            "cleanup_restore_proof_ready": section_results["cleanup_restore"].get("ready") is True,
            "ready_for_live_spike": proof_ready,
        },
        "validator_command": uv_command(
            "scripts/plan_phase3_live_capability_proof.py",
            display_path(output_path) or "<live_capability_proof_path>",
            "--json",
        ),
        "safety_contract": safety_contract(),
        "next_actions": [
            "Treat this as a fillable proof artifact, not as live paging evidence.",
            "Fill before/after residency state, non-dry-run control, cleanup, and exported proof paths after approval.",
            "Run the validator command before attaching the proof to a bundle.",
        ],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_path", nargs="?", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--proof-scope")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--default-output", action="store_true")
    parser.add_argument("--artifact-json", action="store_true", help="print the live-capability proof template artifact")
    parser.add_argument("--json", action="store_true", help="emit machine-readable builder summary")
    return parser


def plan_build(args: argparse.Namespace) -> tuple[int, JSONDict | None, JSONDict | None, str | None]:
    try:
        output_path = args.output
        if args.default_output:
            if output_path is not None:
                return 2, None, None, "--output and --default-output cannot be used together"
            output_path = default_output_path(args.bundle_path)
        artifact = build_template_artifact(args.bundle_path, proof_scope=args.proof_scope, output_path=output_path)
        summary = build_summary(artifact, bundle_path=args.bundle_path, output_path=output_path)
        if output_path is not None and summary["valid"]:
            write_artifact(output_path, artifact)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, None, f"Could not build Phase 3 live-capability proof template: {exc}"
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
        print("MoE Run Anyway Phase 3 live-capability proof template")
        print(f"Valid: {summary['valid']}")
        print(f"Template ready: {summary['template_ready']}")
        print(f"Proof ready: {summary['proof_ready']}")
        print(f"Output path: {summary['output_path']}")
        if summary["errors"]:
            print("Errors:")
            for error in summary["errors"]:
                print(f"  - {error}")
        if summary["blockers"]:
            print("Blockers:")
            for blocker in summary["blockers"]:
                print(f"  - {blocker['id']} ({blocker['section']})")
        print("Safety contract:")
        for item in summary["safety_contract"]:
            print(f"  - {item}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
