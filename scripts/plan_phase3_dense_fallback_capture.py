#!/usr/bin/env python3
"""Plan dense/full-runtime fallback capture for a Phase 3 real-evidence bundle.

This planner turns a selected Phase 3 bundle into concrete artifact requests for
the dense fallback blocker: prompt set, managed output summary, dense/full-runtime
output summary, comparison artifact, and updated bundle manifest. It only reads
local metadata and saved summaries; it does not launch runtimes, run Docker,
inspect secrets, mutate residency, or send prompt traffic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import build_dense_fallback_comparison
import plan_baseline_policy_replay
import plan_dense_fallback_comparison
import plan_phase3_real_evidence_bundle
import phase3_output_receipts


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_PATH = ROOT / "memory-moe-mvp" / "phase3-real-evidence" / "pc_mixtral_phase3_real_evidence_bundle.json"
SUPPORTED_SCHEMA_VERSION = "moe-phase3-dense-fallback-capture-plan-v1"
PROMPT_TEXT_FIELDS = ("prompt", "input", "text", "user_prompt")
PROMPT_ROW_LIST_FIELDS = ("prompts", "cases", "items", "records")

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
    if path.is_absolute():
        return path
    return ROOT / path


def uv_command(*args: str) -> list[str]:
    return ["uv", "run", "--managed-python", "--python", "3.13", *args]


def canonical_prompt_set_path(bundle_path: Path) -> Path:
    return bundle_path.with_name(f"{bundle_path.stem}.prompt-set.json")


def load_json_or_jsonl(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{display_path(path) or path}:{line_number} must be a JSON object")
            rows.append(row)
        return rows
    return json.loads(path.read_text(encoding="utf-8"))


def prompt_id_from_row(row: JSONDict) -> str | None:
    for field in ("prompt_id", "probe_id", "id"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    case = row.get("case")
    if isinstance(case, dict):
        for field in ("prompt_id", "probe_id", "id"):
            value = case.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def rows_from_prompt_payload(payload: Any, *, path: Path) -> list[JSONDict]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = []
        for field in PROMPT_ROW_LIST_FIELDS:
            candidate = payload.get(field)
            if isinstance(candidate, list):
                rows = candidate
                break
        if not rows and prompt_id_from_row(payload) is not None:
            rows = [payload]
    else:
        rows = []

    if not rows:
        raise ValueError(f"{display_path(path) or path} does not contain a supported prompt row list")
    result: list[JSONDict] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{display_path(path) or path}: prompt row {index} must be a JSON object")
        result.append(row)
    return result


def row_has_prompt_text(row: JSONDict) -> bool:
    for field in PROMPT_TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return True
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        return all(isinstance(message, dict) for message in messages)
    case = row.get("case")
    if isinstance(case, dict):
        return row_has_prompt_text(case)
    return False


def summarize_prompt_set(path: Path | None) -> JSONDict:
    if path is None:
        return {
            "path": None,
            "provided": False,
            "exists": False,
            "valid": True,
            "ready": False,
            "prompt_count": 0,
            "prompt_ids": [],
            "errors": [],
            "blocker": "prompt_set_artifact_missing",
        }
    if not path.exists():
        return {
            "path": display_path(path),
            "provided": True,
            "exists": False,
            "valid": False,
            "ready": False,
            "prompt_count": 0,
            "prompt_ids": [],
            "errors": [f"prompt set artifact does not exist: {display_path(path)}"],
            "blocker": "prompt_set_artifact_missing",
        }
    payload = load_json_or_jsonl(path)
    rows = rows_from_prompt_payload(payload, path=path)
    errors: list[str] = []
    prompt_ids: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        prompt_id = prompt_id_from_row(row)
        if prompt_id is None:
            errors.append(f"prompt rows[{index}] missing prompt_id/probe_id/id")
            continue
        if prompt_id in seen:
            errors.append(f"prompt rows[{index}] duplicates prompt id {prompt_id!r}")
            continue
        seen.add(prompt_id)
        prompt_ids.append(prompt_id)
        if not row_has_prompt_text(row):
            errors.append(f"prompt rows[{index}] missing prompt text or messages")
    return {
        "path": display_path(path),
        "provided": True,
        "exists": True,
        "valid": not errors,
        "ready": not errors and bool(prompt_ids),
        "prompt_count": len(prompt_ids),
        "prompt_ids": prompt_ids,
        "errors": errors,
        "blocker": None if not errors and prompt_ids else "prompt_set_artifact_invalid",
    }


def summarize_output_summary(path: Path | None, *, label: str) -> JSONDict:
    if path is None:
        return {
            "path": None,
            "provided": False,
            "exists": False,
            "valid": True,
            "ready": False,
            "row_count": 0,
            "prompt_ids": [],
            "errors": [],
            "capture_receipt_ready": False,
            "capture_receipt_errors": [],
            "blocker": f"{label}_output_summary_missing",
        }
    if not path.exists():
        return {
            "path": display_path(path),
            "provided": True,
            "exists": False,
            "valid": False,
            "ready": False,
            "row_count": 0,
            "prompt_ids": [],
            "errors": [f"{label} output summary does not exist: {display_path(path)}"],
            "capture_receipt_ready": False,
            "capture_receipt_errors": [],
            "blocker": f"{label}_output_summary_missing",
        }
    payload = build_dense_fallback_comparison.load_json_or_jsonl(path)
    rows = build_dense_fallback_comparison.rows_from_payload(payload, path=path)
    indexed, row_errors = build_dense_fallback_comparison.index_rows(rows, label=label)
    if isinstance(payload, dict):
        receipt_validation = phase3_output_receipts.validate_capture_receipt_source_binding(
            payload.get("capture_receipt"),
            repo_root=ROOT,
            expected_label=label,
        )
    else:
        receipt_validation = {
            "shape_valid": False,
            "receipt_ready": False,
            "errors": ["capture_receipt must be an object"],
            "source_request_path_exists": False,
            "source_prompt_set_path_exists": False,
            "source_request_path_matches": False,
            "source_prompt_set_path_matches": False,
        }
    errors = [*row_errors, *receipt_validation["errors"]]
    prompt_ids = sorted(indexed)
    present_count = sum(1 for row in indexed.values() if build_dense_fallback_comparison.output_present(row))
    missing_output_count = len(indexed) - present_count
    ready = not errors and bool(indexed) and missing_output_count == 0 and receipt_validation["receipt_ready"]
    if ready:
        blocker = None
    elif errors:
        blocker = f"{label}_output_summary_invalid"
    elif missing_output_count > 0:
        blocker = f"{label}_output_summary_missing_outputs"
    else:
        blocker = f"{label}_output_summary_capture_receipt_not_ready"
    return {
        "path": display_path(path),
        "provided": True,
        "exists": True,
        "valid": not errors,
        "ready": ready,
        "row_count": len(indexed),
        "output_present_count": present_count,
        "missing_output_count": missing_output_count,
        "prompt_ids": prompt_ids,
        "errors": errors,
        "capture_receipt_ready": receipt_validation["receipt_ready"],
        "capture_receipt_errors": receipt_validation["errors"],
        "capture_receipt_source_request_path_exists": receipt_validation.get("source_request_path_exists"),
        "capture_receipt_source_prompt_set_path_exists": receipt_validation.get("source_prompt_set_path_exists"),
        "blocker": blocker,
    }


def output_covers_prompt(output_ids: set[str], prompt_id: str) -> bool:
    return prompt_id in output_ids or any(output_id.startswith(f"{prompt_id}#") for output_id in output_ids)


def coverage_against_prompt_set(prompt_summary: JSONDict, output_summary: JSONDict, *, label: str) -> JSONDict:
    prompt_ids = set(prompt_summary.get("prompt_ids", []))
    output_ids = set(output_summary.get("prompt_ids", []))
    if not prompt_ids or not output_ids:
        return {
            "label": label,
            "ready": False,
            "missing_prompt_ids": sorted(prompt_ids),
            "extra_output_ids": sorted(output_ids),
        }
    missing = sorted(prompt_id for prompt_id in prompt_ids if not output_covers_prompt(output_ids, prompt_id))
    extras = sorted(
        output_id
        for output_id in output_ids
        if output_id.split("#", 1)[0] not in prompt_ids and output_id not in prompt_ids
    )
    return {
        "label": label,
        "ready": not missing,
        "missing_prompt_ids": missing,
        "extra_output_ids": extras,
    }


def artifact_request(request_id: str, *, status: str, approval_required: bool, description: str, details: JSONDict | None = None) -> JSONDict:
    item: JSONDict = {
        "id": request_id,
        "status": status,
        "approval_required": approval_required,
        "description": description,
    }
    if details is not None:
        item["details"] = details
    return item


def status_for_saved_summary(summary: JSONDict, *, prompt_ready: bool, approval_granted: bool) -> str:
    if summary.get("ready") is True:
        return "already_satisfied"
    if not prompt_ready:
        return "blocked_by_prompt_set"
    return "needed" if approval_granted else "approval_required"


def build_capture_plan(
    bundle_path: Path,
    *,
    output_dir: Path | None = None,
    prompt_set_path: Path | None = None,
    managed_output_path: Path | None = None,
    dense_output_path: Path | None = None,
    fallback_artifact_path: Path | None = None,
    updated_bundle_path: Path | None = None,
    managed_policy_id: str = "preload_shortlist",
    default_quality_label: str = "unknown",
    auto_label_exact: bool = False,
    dense_fallback_capture_approved: bool = False,
    runtime_prompt_traffic_approved: bool = False,
) -> JSONDict:
    manifest = plan_phase3_real_evidence_bundle.load_manifest(bundle_path)
    bundle_summary = plan_phase3_real_evidence_bundle.build_summary(bundle_path)
    artifact_paths = manifest.get("artifact_paths") if isinstance(manifest.get("artifact_paths"), dict) else {}
    trace_path = resolve_repo_path(artifact_paths.get("trace_path"))
    inventory_path = resolve_repo_path(artifact_paths.get("inventory_path"))
    policies_path = resolve_repo_path(artifact_paths.get("policies_path"))
    managed_plan_path = resolve_repo_path(artifact_paths.get("managed_plan_path"))
    existing_fallback_path = resolve_repo_path(artifact_paths.get("fallback_artifact_path"))
    output_root = output_dir or (ROOT / "memory-moe-mvp" / "phase3-real-evidence" / f"{bundle_path.stem}-fallback")
    shared_prompt_set = canonical_prompt_set_path(bundle_path)
    planned_prompt_set = prompt_set_path or (shared_prompt_set if shared_prompt_set.exists() else output_root / "prompt-set.json")
    planned_managed_output = managed_output_path or (output_root / "managed-output-summary.json")
    planned_dense_output = dense_output_path or (output_root / "dense-output-summary.json")
    planned_fallback_artifact = fallback_artifact_path or existing_fallback_path or (output_root / "dense-fallback-comparison.json")
    planned_updated_bundle = updated_bundle_path or (output_root / f"{bundle_path.stem}.with-fallback.json")

    prompt_summary_path = prompt_set_path or (planned_prompt_set if planned_prompt_set.exists() else None)
    managed_summary_path = managed_output_path or (planned_managed_output if planned_managed_output.exists() else None)
    dense_summary_path = dense_output_path or (planned_dense_output if planned_dense_output.exists() else None)
    prompt_summary = summarize_prompt_set(prompt_summary_path)
    managed_summary = summarize_output_summary(managed_summary_path, label="managed")
    dense_summary = summarize_output_summary(dense_summary_path, label="dense")
    managed_coverage = coverage_against_prompt_set(prompt_summary, managed_summary, label="managed")
    dense_coverage = coverage_against_prompt_set(prompt_summary, dense_summary, label="dense")

    fallback_summary = plan_dense_fallback_comparison.build_summary(
        planned_fallback_artifact if planned_fallback_artifact.exists() else None
    )
    replay_summary: JSONDict | None = None
    replay_error: str | None = None
    if trace_path is not None and inventory_path is not None and policies_path is not None:
        try:
            replay_summary = plan_baseline_policy_replay.build_replay_summary(trace_path, inventory_path, policies_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            replay_error = str(exc)

    prompt_ready = prompt_summary.get("ready") is True
    saved_outputs_ready = (
        managed_summary.get("ready") is True
        and dense_summary.get("ready") is True
        and managed_coverage.get("ready") is True
        and dense_coverage.get("ready") is True
    )
    comparison_ready = fallback_summary.get("comparison_ready") is True
    metadata_ready_to_build_comparison = prompt_ready and saved_outputs_ready
    approvals_ready = dense_fallback_capture_approved and runtime_prompt_traffic_approved

    candidate_policy_ids = replay_summary.get("candidate_policy_ids", []) if replay_summary else []
    policy_warning = None
    if replay_summary is not None and not candidate_policy_ids:
        policy_warning = {
            "id": "no_replay_policy_candidate_after_trace_inventory_replay",
            "description": "This bundle currently has no candidate residency policy; fallback comparison can clear quality bounds but may not make the bundle Phase 4-ready by itself.",
        }

    artifact_requests = [
        artifact_request(
            "prompt_set_artifact",
            status="already_satisfied" if prompt_ready else "needed",
            approval_required=False,
            description="Exact prompt set used for both managed and dense/full-runtime output summaries.",
            details={"path": display_path(planned_prompt_set), "current": prompt_summary},
        ),
        artifact_request(
            "managed_output_summary",
            status=status_for_saved_summary(managed_summary, prompt_ready=prompt_ready, approval_granted=approvals_ready),
            approval_required=not managed_summary.get("ready", False),
            description="Saved output summary for the managed/replay policy path, keyed by prompt id.",
            details={"path": display_path(planned_managed_output), "current": managed_summary, "coverage": managed_coverage},
        ),
        artifact_request(
            "dense_output_summary",
            status=status_for_saved_summary(dense_summary, prompt_ready=prompt_ready, approval_granted=approvals_ready),
            approval_required=not dense_summary.get("ready", False),
            description="Saved dense or full-runtime output summary for the same prompt set, keyed by prompt id.",
            details={"path": display_path(planned_dense_output), "current": dense_summary, "coverage": dense_coverage},
        ),
        artifact_request(
            "dense_fallback_comparison_artifact",
            status="already_satisfied" if comparison_ready else ("ready_to_build" if metadata_ready_to_build_comparison else "blocked_by_saved_outputs"),
            approval_required=False,
            description="Comparison artifact built from saved managed and dense/full-runtime outputs.",
            details={"path": display_path(planned_fallback_artifact), "current": fallback_summary},
        ),
        artifact_request(
            "phase3_bundle_with_fallback",
            status="ready_to_build" if comparison_ready else "blocked_by_comparison",
            approval_required=False,
            description="Updated Phase 3 bundle manifest that attaches the validated fallback comparison artifact.",
            details={"path": display_path(planned_updated_bundle)},
        ),
    ]

    commands = [
        {
            "command_class": "managed_output_summary_template_builder",
            "requires_runtime": False,
            "command": uv_command(
                "scripts/build_phase3_output_summary.py",
                display_path(planned_prompt_set) or str(planned_prompt_set),
                "--output-label",
                "managed",
                "--output",
                display_path(planned_managed_output) or str(planned_managed_output),
                "--json",
            ),
        },
        {
            "command_class": "dense_output_summary_template_builder",
            "requires_runtime": False,
            "command": uv_command(
                "scripts/build_phase3_output_summary.py",
                display_path(planned_prompt_set) or str(planned_prompt_set),
                "--output-label",
                "dense",
                "--output",
                display_path(planned_dense_output) or str(planned_dense_output),
                "--json",
            ),
        },
        {
            "command_class": "dense_fallback_comparison_builder",
            "requires_runtime": False,
            "command": uv_command(
                "scripts/build_dense_fallback_comparison.py",
                "--managed-artifact",
                display_path(planned_managed_output) or str(planned_managed_output),
                "--dense-artifact",
                display_path(planned_dense_output) or str(planned_dense_output),
                "--model-id",
                str(manifest.get("model_id", "replace-with-model-id")),
                "--prompt-family",
                str(manifest.get("prompt_family", "replace-with-prompt-family")),
                "--managed-policy-id",
                managed_policy_id,
                "--default-quality-label",
                default_quality_label,
                "--output",
                display_path(planned_fallback_artifact) or str(planned_fallback_artifact),
                "--json",
                *( ["--auto-label-exact"] if auto_label_exact else [] ),
            ),
        },
        {
            "command_class": "dense_fallback_comparison_validator",
            "requires_runtime": False,
            "command": uv_command(
                "scripts/plan_dense_fallback_comparison.py",
                display_path(planned_fallback_artifact) or str(planned_fallback_artifact),
                "--json",
            ),
        },
    ]
    bundle_command = uv_command(
        "scripts/build_phase3_real_evidence_bundle.py",
        "--name",
        str(manifest.get("name", "Phase 3 real-evidence bundle")),
        "--model-id",
        str(manifest.get("model_id", "replace-with-model-id")),
        "--source-format",
        str(manifest.get("source_format", "gguf")),
        "--backend-family",
        str(manifest.get("backend_family", "llama_cpp")),
        "--prompt-family",
        str(manifest.get("prompt_family", "replace-with-prompt-family")),
        "--trace-path",
        display_path(trace_path) or "<trace_path>",
        "--inventory-path",
        display_path(inventory_path) or "<inventory_path>",
        "--policies-path",
        display_path(policies_path) or "<policies_path>",
        "--managed-plan-path",
        display_path(managed_plan_path) or "<managed_plan_path>",
        "--fallback-artifact-path",
        display_path(planned_fallback_artifact) or str(planned_fallback_artifact),
        "--output",
        display_path(planned_updated_bundle) or str(planned_updated_bundle),
        "--json",
    )
    if dense_fallback_capture_approved:
        bundle_command.insert(-3, "--dense-fallback-capture-approved")
    if runtime_prompt_traffic_approved:
        bundle_command.insert(-3, "--runtime-prompt-traffic-approved")
    commands.extend(
        [
            {
                "command_class": "phase3_bundle_builder_with_fallback",
                "requires_runtime": False,
                "command": bundle_command,
            },
            {
                "command_class": "phase3_bundle_validator",
                "requires_runtime": False,
                "command": uv_command(
                    "scripts/plan_phase3_real_evidence_bundle.py",
                    display_path(planned_updated_bundle) or str(planned_updated_bundle),
                    "--json",
                ),
            },
        ]
    )

    errors = [*prompt_summary.get("errors", []), *managed_summary.get("errors", []), *dense_summary.get("errors", [])]
    if replay_error:
        errors.append(f"policy replay failed: {replay_error}")
    if managed_coverage.get("missing_prompt_ids") and managed_summary.get("exists"):
        errors.append("managed output summary does not cover every prompt id")
    if dense_coverage.get("missing_prompt_ids") and dense_summary.get("exists"):
        errors.append("dense output summary does not cover every prompt id")

    return {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "mode": "phase3_dense_fallback_capture_plan",
        "valid": not errors,
        "errors": errors,
        "bundle_path": display_path(bundle_path),
        "bundle_valid": bundle_summary.get("valid"),
        "bundle_ready_for_phase4": bundle_summary.get("bundle_ready_for_phase4"),
        "model_id": manifest.get("model_id"),
        "backend_family": manifest.get("backend_family"),
        "prompt_family": manifest.get("prompt_family"),
        "managed_policy_id": managed_policy_id,
        "output_dir": display_path(output_root),
        "prompt_set_path": display_path(planned_prompt_set),
        "managed_output_path": display_path(planned_managed_output),
        "dense_output_path": display_path(planned_dense_output),
        "fallback_artifact_path": display_path(planned_fallback_artifact),
        "updated_bundle_path": display_path(planned_updated_bundle),
        "approvals": {
            "dense_fallback_capture_approved": dense_fallback_capture_approved,
            "runtime_prompt_traffic_approved": runtime_prompt_traffic_approved,
        },
        "prompt_set": prompt_summary,
        "managed_output_summary": managed_summary,
        "dense_output_summary": dense_summary,
        "prompt_coverage": {
            "managed": managed_coverage,
            "dense": dense_coverage,
        },
        "metadata_ready_to_build_comparison": metadata_ready_to_build_comparison,
        "comparison_ready": comparison_ready,
        "phase3_fallback_ready": comparison_ready,
        "candidate_policy_ids": candidate_policy_ids,
        "policy_warning": policy_warning,
        "artifact_requests": artifact_requests,
        "commands": commands,
        "runtime_capture_contract": {
            "requires_explicit_approval": not approvals_ready,
            "must_use_same_prompt_set": True,
            "must_write_saved_output_summaries_only": True,
            "must_not_store_private_tokens": True,
            "required_output_row_fields": ["prompt_id", "output or response.summary.response_chars", "capture_receipt.receipt_ready"],
        },
        "safety_contract": [
            "planner reads local metadata and saved summaries only",
            "planner does not launch model servers",
            "planner does not run Docker",
            "planner does not download models",
            "planner does not inspect private tokens",
            "planner does not send prompt traffic",
            "planner does not mutate runtime residency",
            "planner does not claim live expert paging",
        ],
        "next_actions": [
            "Attach or reconstruct the exact prompt set used for the selected real bundle.",
            "After approval, capture managed and dense/full-runtime output summaries for that prompt set.",
            "Fill each output summary capture_receipt from the same approved runtime-capture request before building fallback comparison.",
            "Build and validate the dense fallback comparison artifact from saved summaries.",
            "Rebuild the Phase 3 bundle with the fallback artifact attached, then rerun the matrix.",
        ],
    }


def command_to_text(command: Any) -> str:
    if isinstance(command, list):
        return " ".join(str(part) for part in command)
    return str(command)


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def artifact_detail_value(request: JSONDict, key: str) -> Any:
    details = request.get("details") if isinstance(request.get("details"), dict) else {}
    current = details.get("current") if isinstance(details.get("current"), dict) else {}
    if key == "path":
        return details.get("path") or current.get("path") or "missing"
    if key == "ready":
        return current.get("ready", "unknown")
    if key == "blocker":
        blocker = current.get("blocker")
        if isinstance(blocker, dict):
            return blocker.get("id") or "unknown"
        return blocker or "none"
    return current.get(key, "unknown")


def format_markdown_report(summary: JSONDict) -> str:
    contract = summary.get("runtime_capture_contract") if isinstance(summary.get("runtime_capture_contract"), dict) else {}
    lines = [
        "# Phase 3 Dense Fallback Capture",
        "",
        f"- Valid: `{summary.get('valid')}`",
        f"- Bundle: `{markdown_escape(summary.get('bundle_path'))}`",
        f"- Prompt family: `{markdown_escape(summary.get('prompt_family'))}`",
        f"- Prompt set ready: `{summary.get('prompt_set', {}).get('ready') if isinstance(summary.get('prompt_set'), dict) else 'unknown'}`",
        f"- Managed output ready: `{summary.get('managed_output_summary', {}).get('ready') if isinstance(summary.get('managed_output_summary'), dict) else 'unknown'}`",
        f"- Dense output ready: `{summary.get('dense_output_summary', {}).get('ready') if isinstance(summary.get('dense_output_summary'), dict) else 'unknown'}`",
        f"- Metadata ready to build comparison: `{summary.get('metadata_ready_to_build_comparison')}`",
        f"- Comparison ready: `{summary.get('comparison_ready')}`",
        f"- Runtime approval required: `{contract.get('requires_explicit_approval')}`",
        "",
        "## Artifact Requests",
        "",
        "| Artifact | Status | Approval | Ready | Path | Blocker |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for request in summary.get("artifact_requests", []):
        if not isinstance(request, dict):
            continue
        approval = "required" if request.get("approval_required") is True else "not required"
        lines.append(
            "| "
            + " | ".join(
                markdown_escape(value)
                for value in (
                    request.get("id") or "unknown",
                    request.get("status") or "unknown",
                    approval,
                    artifact_detail_value(request, "ready"),
                    artifact_detail_value(request, "path"),
                    artifact_detail_value(request, "blocker"),
                )
            )
            + " |"
        )
    warning = summary.get("policy_warning")
    if isinstance(warning, dict):
        lines.extend(["", "## Policy Warning", "", f"- `{markdown_escape(warning.get('id') or 'unknown')}`: {markdown_escape(warning.get('description') or '')}"])
    if contract:
        fields = contract.get("required_output_row_fields") if isinstance(contract.get("required_output_row_fields"), list) else []
        lines.extend(
            [
                "",
                "## Runtime Capture Contract",
                "",
                f"- Requires explicit approval: `{contract.get('requires_explicit_approval')}`",
                f"- Must use same prompt set: `{contract.get('must_use_same_prompt_set')}`",
                f"- Saved summaries only: `{contract.get('must_write_saved_output_summaries_only')}`",
                f"- Required output row fields: `{markdown_escape(', '.join(str(item) for item in fields))}`",
            ]
        )
    commands = summary.get("commands") if isinstance(summary.get("commands"), list) else []
    if commands:
        lines.extend(["", "## Commands", ""])
        for item in commands:
            if not isinstance(item, dict):
                continue
            lines.append(f"- `{markdown_escape(item.get('command_class') or 'unknown')}`")
            lines.append("```sh")
            lines.append(command_to_text(item.get("command")))
            lines.append("```")
    next_actions = summary.get("next_actions") if isinstance(summary.get("next_actions"), list) else []
    if next_actions:
        lines.extend(["", "## Next Actions", ""])
        for action in next_actions:
            lines.append(f"- {markdown_escape(action)}")
    lines.extend(["", "## Safety Contract", ""])
    for item in summary.get("safety_contract", []):
        lines.append(f"- {markdown_escape(item)}")
    return "\n".join(lines) + "\n"


def write_markdown_report(summary: JSONDict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_markdown_report(summary), encoding="utf-8")


def print_human_summary(summary: JSONDict) -> None:
    print("MoE Run Anyway Phase 3 dense fallback capture plan")
    print(f"Valid: {summary['valid']}")
    print(f"Bundle: {summary['bundle_path']}")
    print(f"Prompt family: {summary['prompt_family']}")
    print(f"Prompt set ready: {summary['prompt_set']['ready']}")
    print(f"Managed output ready: {summary['managed_output_summary']['ready']}")
    print(f"Dense output ready: {summary['dense_output_summary']['ready']}")
    print(f"Comparison ready: {summary['comparison_ready']}")
    print(f"Metadata ready to build comparison: {summary['metadata_ready_to_build_comparison']}")
    if summary.get("policy_warning"):
        print(f"Policy warning: {summary['policy_warning']['id']}")
    if summary["errors"]:
        print("Errors:")
        for error in summary["errors"]:
            print(f"  - {error}")
    print("Artifact requests:")
    for request in summary["artifact_requests"]:
        print(f"  - {request['id']}: {request['status']}")
    print("Safety contract:")
    for item in summary["safety_contract"]:
        print(f"  - {item}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_path", nargs="?", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prompt-set-path", type=Path)
    parser.add_argument("--managed-output-path", type=Path)
    parser.add_argument("--dense-output-path", type=Path)
    parser.add_argument("--fallback-artifact-path", type=Path)
    parser.add_argument("--updated-bundle-path", type=Path)
    parser.add_argument("--managed-policy-id", default="preload_shortlist")
    parser.add_argument("--default-quality-label", choices=sorted(plan_dense_fallback_comparison.SUPPORTED_QUALITY_LABELS), default="unknown")
    parser.add_argument("--auto-label-exact", action="store_true")
    parser.add_argument("--dense-fallback-capture-approved", action="store_true")
    parser.add_argument("--runtime-prompt-traffic-approved", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    parser.add_argument("--output-md", type=Path, help="write a Markdown dense fallback handoff report")
    return parser


def plan_path(
    bundle_path: Path = DEFAULT_BUNDLE_PATH,
    **kwargs: Any,
) -> tuple[int, JSONDict | None, str | None]:
    try:
        summary = build_capture_plan(bundle_path, **kwargs)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not build Phase 3 dense fallback capture plan: {exc}"
    return (0 if summary["valid"] else 2), summary, None


def main_from_test_path(bundle_path: Path = DEFAULT_BUNDLE_PATH) -> int:
    status, _, _ = plan_path(bundle_path)
    return status


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary, error_message = plan_path(
        args.bundle_path,
        output_dir=args.output_dir,
        prompt_set_path=args.prompt_set_path,
        managed_output_path=args.managed_output_path,
        dense_output_path=args.dense_output_path,
        fallback_artifact_path=args.fallback_artifact_path,
        updated_bundle_path=args.updated_bundle_path,
        managed_policy_id=args.managed_policy_id,
        default_quality_label=args.default_quality_label,
        auto_label_exact=args.auto_label_exact,
        dense_fallback_capture_approved=args.dense_fallback_capture_approved,
        runtime_prompt_traffic_approved=args.runtime_prompt_traffic_approved,
    )
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
