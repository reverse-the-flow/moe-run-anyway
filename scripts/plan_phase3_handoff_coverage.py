#!/usr/bin/env python3
"""Summarize Phase 3 operator-handoff scaffold coverage.

This planner checks whether each repo-local real-evidence bundle has the
metadata artifacts needed before approved runtime capture: prompt set, candidate
trace receipt template, managed and dense output-summary templates,
runtime-capture request, planned-only runtime-capture launch-card template, and live-capability proof template. It reads local
metadata only; it does not launch runtimes, run
Docker, call endpoints, inspect secrets, mutate residency, or send prompt
traffic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import build_phase3_runtime_capture_request
import build_phase3_trace_receipt
import plan_phase3_dense_fallback_capture
import plan_phase3_live_capability_proof
import plan_phase3_real_evidence_bundle
import plan_phase3_runtime_capture_commands


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "memory-moe-mvp" / "phase3-real-evidence"
BUNDLE_GLOB = "*phase3_real_evidence_bundle.json"
SUPPORTED_SCHEMA_VERSION = "moe-phase3-handoff-coverage-v1"
JSONDict = dict[str, Any]


def display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def canonical_prompt_set_path(bundle_path: Path) -> Path:
    return bundle_path.with_name(f"{bundle_path.stem}.prompt-set.json")


def fallback_dir(bundle_path: Path) -> Path:
    return bundle_path.with_name(f"{bundle_path.stem}-fallback")


def output_summary_path(bundle_path: Path, label: str) -> Path:
    return fallback_dir(bundle_path) / f"{label}-output-summary.json"


def candidate_trace_path(bundle_path: Path) -> Path:
    return build_phase3_trace_receipt.default_candidate_trace_path(bundle_path)


def candidate_trace_receipt_path(bundle_path: Path) -> Path:
    return build_phase3_trace_receipt.default_trace_receipt_path(candidate_trace_path(bundle_path))


def runtime_request_path(bundle_path: Path) -> Path:
    return bundle_path.with_name(f"{bundle_path.stem}.runtime-capture-request.json")


def live_proof_template_path(bundle_path: Path) -> Path:
    return bundle_path.with_name(f"{bundle_path.stem}.live-capability-proof.template.json")

def runtime_launch_card_template_path(bundle_path: Path) -> Path:
    return bundle_path.with_name(f"{bundle_path.stem}.runtime-capture-launch-card.template.json")


def uv_command(*args: str) -> list[str]:
    return ["uv", "run", "--managed-python", "--python", "3.13", *args]


def load_json(path: Path) -> JSONDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{display_path(path) or path} must be a JSON object")
    return payload


def runtime_request_summary(bundle_path: Path, request_path: Path) -> JSONDict:
    if not request_path.exists():
        return {
            "path": display_path(request_path),
            "exists": False,
            "valid": False,
            "ready": False,
            "errors": [f"runtime-capture request does not exist: {display_path(request_path)}"],
            "blocker": "runtime_capture_request_missing",
            "future_artifact_ids": [],
            "requested_artifact_ids": [],
        }

    errors: list[str] = []
    payload = load_json(request_path)
    errors.extend(build_phase3_runtime_capture_request.validate_request(payload))
    if payload.get("bundle_path") != display_path(bundle_path):
        errors.append("runtime-capture request bundle_path does not match bundle")
    requested_artifact_ids = [
        item.get("id")
        for item in payload.get("requested_artifacts", [])
        if isinstance(item, dict)
    ]
    future_artifact_ids = [
        item.get("id")
        for item in payload.get("future_artifacts", [])
        if isinstance(item, dict)
    ]
    return {
        "path": display_path(request_path),
        "exists": True,
        "valid": not errors,
        "ready": not errors,
        "errors": errors,
        "blocker": None if not errors else "runtime_capture_request_invalid",
        "requested_artifact_ids": requested_artifact_ids,
        "future_artifact_ids": future_artifact_ids,
        "ready_for_operator_capture": payload.get("ready_for_operator_capture"),
        "capture_complete": payload.get("capture_complete"),
    }


def trace_receipt_template_summary(
    path: Path,
    *,
    bundle_path: Path,
    prompt_set_path: Path,
    candidate_trace_path: Path,
) -> JSONDict:
    if not path.exists():
        return {
            "path": display_path(path),
            "exists": False,
            "valid": False,
            "ready": False,
            "receipt_ready": False,
            "errors": [f"trace receipt template does not exist: {display_path(path)}"],
            "blocker": "candidate_trace_receipt_template_missing",
        }
    artifact = load_json(path)
    prompt_set = load_json(prompt_set_path) if prompt_set_path.exists() else {}
    validation = build_phase3_trace_receipt.validate_trace_receipt_artifact(
        artifact,
        bundle_path=bundle_path,
        prompt_set_path=prompt_set_path,
        candidate_trace_path=candidate_trace_path,
        expected_model_id=str(prompt_set.get("model_id")) if prompt_set.get("model_id") is not None else None,
        expected_backend_family=str(prompt_set.get("backend_family")) if prompt_set.get("backend_family") is not None else None,
        expected_prompt_family=str(prompt_set.get("prompt_family")) if prompt_set.get("prompt_family") is not None else None,
    )
    return {
        "path": display_path(path),
        "exists": True,
        "valid": validation["shape_valid"],
        "ready": validation["shape_valid"],
        "receipt_ready": validation["receipt_ready"],
        "errors": validation["errors"],
        "blocker": None if validation["shape_valid"] else "candidate_trace_receipt_template_invalid",
    }


def live_template_summary(path: Path) -> JSONDict:
    if path.exists():
        summary = plan_phase3_live_capability_proof.build_summary(path)
    else:
        summary = plan_phase3_live_capability_proof.missing_artifact_summary(path)
    return {
        "path": display_path(path),
        "exists": summary.get("proof_available") is True,
        "valid": summary.get("valid") is True,
        "ready": summary.get("proof_available") is True and summary.get("valid") is True,
        "proof_ready": summary.get("proof_ready") is True,
        "blockers": summary.get("blockers", []),
        "blocker": None
        if summary.get("proof_available") is True and summary.get("valid") is True
        else "live_capability_proof_template_missing",
    }
def launch_card_template_summary(request_path: Path, launch_card_path: Path) -> JSONDict:
    if not launch_card_path.exists():
        return {
            "path": display_path(launch_card_path),
            "exists": False,
            "valid": False,
            "ready": False,
            "template_ready": False,
            "planned_only": False,
            "runtime_capture_command_ready": False,
            "binding_valid": False,
            "binding_ready": False,
            "task_count": 0,
            "command_option_count": 0,
            "command_ready_count": 0,
            "missing_runtime_command_count": 0,
            "errors": [f"runtime-capture launch-card template does not exist: {display_path(launch_card_path)}"],
            "blocker": "runtime_capture_launch_card_template_missing",
            "binding_blockers": [],
        }
    if not request_path.exists():
        return {
            "path": display_path(launch_card_path),
            "exists": True,
            "valid": False,
            "ready": False,
            "template_ready": False,
            "planned_only": False,
            "runtime_capture_command_ready": False,
            "binding_valid": False,
            "binding_ready": False,
            "task_count": 0,
            "command_option_count": 0,
            "command_ready_count": 0,
            "missing_runtime_command_count": 0,
            "errors": [f"runtime-capture request does not exist: {display_path(request_path)}"],
            "blocker": "runtime_capture_request_missing",
            "binding_blockers": [],
        }

    launch_card = plan_phase3_runtime_capture_commands.load_json(launch_card_path)
    command_summary = plan_phase3_runtime_capture_commands.build_summary(
        request_path,
        launch_card_path=launch_card_path,
    )
    template_summary = command_summary.get("launch_card_template_summary") if isinstance(command_summary.get("launch_card_template_summary"), dict) else {}
    binding_summary = command_summary.get("launch_card_binding_summary") if isinstance(command_summary.get("launch_card_binding_summary"), dict) else {}
    errors = list(command_summary.get("errors", [])) if isinstance(command_summary.get("errors"), list) else []
    blockers: list[str] = []
    if launch_card.get("schema_version") != plan_phase3_runtime_capture_commands.LAUNCH_CARD_SCHEMA_VERSION:
        errors.append("runtime_capture_launch_card_schema_version_mismatch")
    if launch_card.get("status") != "planned_only":
        blockers.append("runtime_capture_launch_card_not_planned_only")
    if launch_card.get("executable") is not False:
        blockers.append("runtime_capture_launch_card_must_be_non_executable")
    if launch_card.get("request_path") != display_path(request_path):
        errors.append("runtime_capture_launch_card_request_path_mismatch")
    if template_summary.get("template_ready") is not True:
        blockers.append("runtime_capture_launch_card_template_not_ready")
    if binding_summary.get("valid") is not True:
        errors.extend(str(error) for error in binding_summary.get("errors", []) if isinstance(error, str))
    task_count = binding_summary.get("task_count") if isinstance(binding_summary.get("task_count"), int) else launch_card.get("task_count")
    if task_count != len(plan_phase3_runtime_capture_commands.RUNTIME_ARTIFACT_IDS):
        blockers.append("runtime_capture_launch_card_task_count_mismatch")

    ready = not errors and not blockers
    return {
        "path": display_path(launch_card_path),
        "exists": True,
        "valid": not errors,
        "ready": ready,
        "template_ready": template_summary.get("template_ready") is True,
        "planned_only": launch_card.get("status") == "planned_only",
        "executable": launch_card.get("executable") is True,
        "model_plane_binding_ready": template_summary.get("model_plane_binding_ready") is True,
        "runtime_capture_command_ready": command_summary.get("runtime_capture_command_ready") is True,
        "binding_valid": binding_summary.get("valid") is True,
        "binding_ready": binding_summary.get("binding_ready") is True,
        "task_count": task_count if isinstance(task_count, int) else 0,
        "bound_task_count": binding_summary.get("bound_task_count", 0),
        "command_option_count": binding_summary.get("command_option_count", 0),
        "command_ready_count": binding_summary.get("command_ready_count", 0),
        "missing_runtime_command_count": binding_summary.get("missing_runtime_command_count", 0),
        "missing_runtime_command_artifact_ids": binding_summary.get("missing_runtime_command_artifact_ids", []),
        "binding_blockers": binding_summary.get("blockers", []),
        "errors": errors,
        "blockers": blockers,
        "blocker": None if ready else "runtime_capture_launch_card_template_invalid",
    }


def summarize_bundle(bundle_path: Path) -> JSONDict:
    errors: list[str] = []
    try:
        manifest = plan_phase3_real_evidence_bundle.load_manifest(bundle_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {
            "path": display_path(bundle_path),
            "valid": False,
            "errors": [str(exc)],
            "handoff_scaffold_ready": False,
            "missing_artifact_ids": ["bundle_manifest"],
        }

    prompt_path = canonical_prompt_set_path(bundle_path)
    managed_path = output_summary_path(bundle_path, "managed")
    dense_path = output_summary_path(bundle_path, "dense")
    trace_path = candidate_trace_path(bundle_path)
    receipt_path = candidate_trace_receipt_path(bundle_path)
    request_path = runtime_request_path(bundle_path)
    proof_path = live_proof_template_path(bundle_path)
    launch_card_path = runtime_launch_card_template_path(bundle_path)

    prompt = plan_phase3_dense_fallback_capture.summarize_prompt_set(prompt_path)
    managed = plan_phase3_dense_fallback_capture.summarize_output_summary(
        managed_path if managed_path.exists() else None,
        label="managed",
    )
    dense = plan_phase3_dense_fallback_capture.summarize_output_summary(
        dense_path if dense_path.exists() else None,
        label="dense",
    )
    managed_coverage = plan_phase3_dense_fallback_capture.coverage_against_prompt_set(
        prompt,
        managed,
        label="managed",
    )
    dense_coverage = plan_phase3_dense_fallback_capture.coverage_against_prompt_set(
        prompt,
        dense,
        label="dense",
    )
    trace_receipt = trace_receipt_template_summary(
        receipt_path,
        bundle_path=bundle_path,
        prompt_set_path=prompt_path,
        candidate_trace_path=trace_path,
    )
    request = runtime_request_summary(bundle_path, request_path)
    proof = live_template_summary(proof_path)
    launch_card = launch_card_template_summary(request_path, launch_card_path)

    artifact_checks = {
        "prompt_set": prompt.get("ready") is True,
        "managed_output_summary_template": (
            managed.get("valid") is True
            and managed.get("exists") is True
            and managed_coverage.get("ready") is True
        ),
        "dense_output_summary_template": (
            dense.get("valid") is True
            and dense.get("exists") is True
            and dense_coverage.get("ready") is True
        ),
        "candidate_trace_receipt_template": trace_receipt.get("ready") is True,
        "runtime_capture_request": request.get("ready") is True,
        "runtime_capture_launch_card_template": launch_card.get("ready") is True,
        "live_capability_proof_template": proof.get("ready") is True,
    }
    missing_artifact_ids = [
        artifact_id
        for artifact_id, ready in artifact_checks.items()
        if ready is not True
    ]
    for section in (prompt, managed, dense, trace_receipt, request, launch_card):
        if section.get("exists") is True and section.get("valid") is not True:
            errors.extend(section.get("errors", []))
    handoff_ready = not missing_artifact_ids and not errors

    return {
        "path": display_path(bundle_path),
        "name": manifest.get("name"),
        "model_id": manifest.get("model_id"),
        "prompt_family": manifest.get("prompt_family"),
        "valid": not errors,
        "errors": errors,
        "handoff_scaffold_ready": handoff_ready,
        "missing_artifact_ids": missing_artifact_ids,
        "artifact_checks": artifact_checks,
        "prompt_set": prompt,
        "managed_output_summary": managed,
        "dense_output_summary": dense,
        "candidate_trace_receipt_template": trace_receipt,
        "prompt_coverage": {"managed": managed_coverage, "dense": dense_coverage},
        "runtime_capture_request": request,
        "runtime_capture_launch_card_template": launch_card,
        "live_capability_proof_template": proof,
        "build_commands": {
            "prompt_set": uv_command(
                "scripts/build_phase3_prompt_set.py",
                display_path(bundle_path) or str(bundle_path),
                "--default-output",
                "--json",
            ),
            "managed_output_summary": uv_command(
                "scripts/build_phase3_output_summary.py",
                display_path(prompt_path) or str(prompt_path),
                "--output-label",
                "managed",
                "--default-output",
                "--json",
            ),
            "dense_output_summary": uv_command(
                "scripts/build_phase3_output_summary.py",
                display_path(prompt_path) or str(prompt_path),
                "--output-label",
                "dense",
                "--default-output",
                "--json",
            ),
            "candidate_trace_receipt_template": uv_command(
                "scripts/build_phase3_trace_receipt.py",
                display_path(bundle_path) or str(bundle_path),
                "--default-output",
                "--json",
            ),
            "live_capability_proof_template": uv_command(
                "scripts/build_phase3_live_capability_proof_template.py",
                display_path(bundle_path) or str(bundle_path),
                "--default-output",
                "--json",
            ),
            "runtime_capture_request": uv_command(
                "scripts/build_phase3_runtime_capture_request.py",
                display_path(bundle_path) or str(bundle_path),
                "--default-output",
                "--json",
            ),
            "runtime_capture_launch_card_template": uv_command(
                "scripts/plan_phase3_runtime_capture_commands.py",
                display_path(request_path) or str(request_path),
                "--output-launch-card",
                display_path(launch_card_path) or str(launch_card_path),
                "--json",
            ),
        },
    }


def build_coverage(root: Path = DEFAULT_ROOT) -> JSONDict:
    errors: list[str] = []
    if not root.exists():
        errors.append(f"handoff root does not exist: {display_path(root)}")
        bundle_paths: list[Path] = []
    else:
        bundle_paths = sorted(root.glob(BUNDLE_GLOB))

    bundles = [summarize_bundle(path) for path in bundle_paths]
    ready = [bundle for bundle in bundles if bundle.get("handoff_scaffold_ready") is True]
    missing_counts: dict[str, int] = {}
    for bundle in bundles:
        for artifact_id in bundle.get("missing_artifact_ids", []):
            missing_counts[artifact_id] = missing_counts.get(artifact_id, 0) + 1

    return {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "mode": "phase3_handoff_coverage",
        "root": display_path(root),
        "valid": not errors and all(bundle.get("valid") is True for bundle in bundles),
        "errors": errors,
        "bundle_count": len(bundles),
        "handoff_scaffold_ready_count": len(ready),
        "all_handoff_scaffolds_ready": len(bundles) > 0 and len(ready) == len(bundles),
        "missing_artifact_counts": dict(sorted(missing_counts.items())),
        "bundles": bundles,
        "safety_contract": [
            "handoff coverage reads local metadata only",
            "handoff coverage does not launch model servers",
            "handoff coverage does not run Docker",
            "handoff coverage does not call endpoints",
            "handoff coverage does not download models",
            "handoff coverage does not inspect private tokens",
            "handoff coverage does not send prompt traffic",
            "handoff coverage does not mutate runtime residency",
            "handoff coverage does not claim live expert paging",
        ],
    }


def command_to_text(command: Any) -> str:
    if isinstance(command, list):
        return " ".join(str(part) for part in command)
    return str(command)


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def missing_artifacts_text(bundle: JSONDict) -> str:
    missing = bundle.get("missing_artifact_ids") if isinstance(bundle.get("missing_artifact_ids"), list) else []
    return ", ".join(str(item) for item in missing) if missing else "none"


def format_markdown_report(summary: JSONDict) -> str:
    lines = [
        "# Phase 3 Handoff Coverage",
        "",
        f"- Valid: `{summary.get('valid')}`",
        f"- Root: `{markdown_escape(summary.get('root'))}`",
        f"- Bundles: `{summary.get('bundle_count')}`",
        f"- Handoff scaffolds ready: `{summary.get('handoff_scaffold_ready_count')}`",
        f"- All handoff scaffolds ready: `{summary.get('all_handoff_scaffolds_ready')}`",
        "",
        "## Missing Artifact Counts",
        "",
    ]
    missing_counts = summary.get("missing_artifact_counts") if isinstance(summary.get("missing_artifact_counts"), dict) else {}
    if missing_counts:
        for artifact_id, count in missing_counts.items():
            lines.append(f"- `{markdown_escape(artifact_id)}`: `{count}`")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Bundle Coverage",
            "",
            "| Bundle | Ready | Missing | Runtime Request | Launch Card Template | Live Proof Template |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for bundle in summary.get("bundles", []):
        if not isinstance(bundle, dict):
            continue
        runtime_request = bundle.get("runtime_capture_request") if isinstance(bundle.get("runtime_capture_request"), dict) else {}
        launch_card = bundle.get("runtime_capture_launch_card_template") if isinstance(bundle.get("runtime_capture_launch_card_template"), dict) else {}
        live_template = bundle.get("live_capability_proof_template") if isinstance(bundle.get("live_capability_proof_template"), dict) else {}
        lines.append(
            "| "
            + " | ".join(
                markdown_escape(value)
                for value in (
                    bundle.get("name") or bundle.get("path") or "unknown",
                    bundle.get("handoff_scaffold_ready"),
                    missing_artifacts_text(bundle),
                    runtime_request.get("path") or "missing",
                    launch_card.get("path") or "missing",
                    live_template.get("path") or "missing",
                )
            )
            + " |"
        )
    bundles_with_missing = [
        bundle
        for bundle in summary.get("bundles", [])
        if isinstance(bundle, dict) and bundle.get("missing_artifact_ids")
    ]
    if bundles_with_missing:
        lines.extend(["", "## Missing Scaffold Build Commands", ""])
        for bundle in bundles_with_missing:
            lines.append(f"### {markdown_escape(bundle.get('name') or bundle.get('path') or 'unknown')}")
            build_commands = bundle.get("build_commands") if isinstance(bundle.get("build_commands"), dict) else {}
            for artifact_id in bundle.get("missing_artifact_ids", []):
                lines.append(f"- `{markdown_escape(artifact_id)}`")
                command = build_commands.get(artifact_id)
                if command:
                    lines.append("```sh")
                    lines.append(command_to_text(command))
                    lines.append("```")
                else:
                    lines.append("  - no build command available")
    lines.extend(["", "## Safety Contract", ""])
    for item in summary.get("safety_contract", []):
        lines.append(f"- {markdown_escape(item)}")
    return "\n".join(lines) + "\n"


def write_markdown_report(summary: JSONDict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_markdown_report(summary), encoding="utf-8")

def print_human_summary(summary: JSONDict) -> None:
    print("MoE Run Anyway Phase 3 handoff coverage")
    print(f"Valid: {summary['valid']}")
    print(f"Bundles: {summary['bundle_count']}")
    print(f"Handoff scaffolds ready: {summary['handoff_scaffold_ready_count']}")
    print(f"All handoff scaffolds ready: {summary['all_handoff_scaffolds_ready']}")
    if summary["missing_artifact_counts"]:
        print("Missing artifact counts:")
        for artifact_id, count in summary["missing_artifact_counts"].items():
            print(f"  - {artifact_id}: {count}")
    print("Bundles:")
    for bundle in summary["bundles"]:
        missing = ", ".join(bundle.get("missing_artifact_ids", [])) or "none"
        print(f"  - {bundle.get('name')}: handoff={bundle.get('handoff_scaffold_ready')} missing={missing}")
    print("Safety contract:")
    for item in summary["safety_contract"]:
        print(f"  - {item}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    parser.add_argument("--output-md", type=Path, help="write a Markdown handoff coverage report")
    return parser


def plan_paths(root: Path = DEFAULT_ROOT) -> tuple[int, JSONDict | None, str | None]:
    try:
        summary = build_coverage(root)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not build Phase 3 handoff coverage: {exc}"
    return (0 if summary["valid"] else 2), summary, None


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary, error_message = plan_paths(args.root)
    if error_message:
        print(error_message, file=sys.stderr)
        return status
    assert summary is not None
    if args.output_md:
        write_markdown_report(summary, args.output_md)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_human_summary(summary)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
