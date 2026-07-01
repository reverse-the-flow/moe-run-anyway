#!/usr/bin/env python3
"""Plan the next llama.cpp runtime-actuator spike without mutating runtime state.

This planner turns the managed expert-loading contract into a concrete proof
handoff for the selected backend. It names the read-only proof artifacts,
patch/probe boundaries, completion gates, and dependency order required before
any residency-control work can become a live actuator. It does not launch
models, run Docker, inspect endpoints, send prompt traffic, read tensor values,
or mutate runtime residency.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import plan_phase3_runtime_actuator_design


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN_PATH = plan_phase3_runtime_actuator_design.DEFAULT_PLAN_PATH
DEFAULT_BACKEND_FAMILY = plan_phase3_runtime_actuator_design.DEFAULT_BACKEND_FAMILY
SUPPORTED_SCHEMA_VERSION = "moe-phase3-runtime-actuator-spike-v1"

CAPABILITY_ORDER = [
    "expert_inventory",
    "routing_visibility",
    "residency_observation",
    "policy_application",
    "dense_fallback",
    "artifact_export",
    "residency_control",
    "cleanup_restore",
]

CAPABILITY_SPIKE_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "expert_inventory": {
        "probe_or_patch_boundary": "GGUF or runtime scanner emits layer/expert/component inventory with byte and backend identity fields.",
        "proof_artifacts": [
            "scanner-derived expert inventory manifest",
            "model/backend/profile identity record",
        ],
        "completion_gate": "inventory manifest validates and matches the runtime-capture request model/backend",
        "dependency_ids": [],
    },
    "routing_visibility": {
        "probe_or_patch_boundary": "llama.cpp MoE gate selection emits semantic layer/expert ids, scores, ranks, and prompt/window metadata.",
        "proof_artifacts": [
            "candidate router trace JSONL",
            "trace capture receipt bound to the approved request",
        ],
        "completion_gate": "router trace validates and replay observes routed experts from an approved request",
        "dependency_ids": ["expert_inventory"],
    },
    "residency_observation": {
        "probe_or_patch_boundary": "backend exposes read-only before/after residency state for addressed experts without changing cache state.",
        "proof_artifacts": [
            "before-residency snapshot",
            "after-residency snapshot",
            "residency observation fields in the live capability proof",
        ],
        "completion_gate": "observation artifact proves current residency tier before and after a no-op request",
        "dependency_ids": ["expert_inventory", "routing_visibility"],
    },
    "policy_application": {
        "probe_or_patch_boundary": "controller converts replay policy decisions into adapter intents and rejects mutation intents when required capabilities are unavailable.",
        "proof_artifacts": [
            "policy decision trace",
            "adapter intent audit log",
        ],
        "completion_gate": "policy adapter emits keep/fallback/abort decisions without requiring residency mutation",
        "dependency_ids": ["routing_visibility", "residency_observation"],
    },
    "dense_fallback": {
        "probe_or_patch_boundary": "same prompt set can run through normal dense/full-residency execution as the comparison and escape path.",
        "proof_artifacts": [
            "managed output summary",
            "dense output summary",
            "dense fallback comparison artifact",
        ],
        "completion_gate": "managed and dense outputs have ready receipts and pair-consistent comparison provenance",
        "dependency_ids": ["policy_application"],
    },
    "artifact_export": {
        "probe_or_patch_boundary": "runtime and controller export memory-moe-bridge-v1-compatible records for every actuator spike attempt.",
        "proof_artifacts": [
            "live capability proof artifact",
            "managed-loading bridge record",
        ],
        "completion_gate": "proof artifact validates with matching source bundle/model/backend/prompt context",
        "dependency_ids": ["policy_application", "dense_fallback"],
    },
    "residency_control": {
        "probe_or_patch_boundary": "guarded adapter exposes preload, pin, evict, and demote as auditable actions behind explicit approval and before/after observation.",
        "proof_artifacts": [
            "dry-run control intent log",
            "guarded mutation acceptance log",
            "before/after residency transition proof",
        ],
        "completion_gate": "control action changes only the addressed expert residency state and records a reversible before/after transition",
        "dependency_ids": ["residency_observation", "dense_fallback", "artifact_export"],
    },
    "cleanup_restore": {
        "probe_or_patch_boundary": "runtime restores dense/full-residency behavior or proves no mutation occurred after abort, failure, or interruption.",
        "proof_artifacts": [
            "cleanup action log",
            "restore verification artifact",
            "failure-path test receipt",
        ],
        "completion_gate": "cleanup proof shows restore verified and failure path tested after a guarded control attempt",
        "dependency_ids": ["residency_control", "dense_fallback", "artifact_export"],
    },
}

JSONDict = dict[str, Any]


def load_plan(path: Path) -> JSONDict:
    return plan_phase3_runtime_actuator_design.load_plan(path)


def display_path(path: Path) -> str:
    return plan_phase3_runtime_actuator_design.display_path(path)


def status_for_capability(adapter: JSONDict, capability_id: str) -> str:
    status = adapter.get("capability_status") if isinstance(adapter.get("capability_status"), dict) else {}
    value = status.get(capability_id)
    return value if isinstance(value, str) and value.strip() else "unknown"


def live_required_capabilities(plan: JSONDict) -> list[str]:
    required = plan_phase3_runtime_actuator_design.live_required_capabilities(plan)
    return sorted(required, key=lambda capability: CAPABILITY_ORDER.index(capability) if capability in CAPABILITY_ORDER else 999)


def build_proof_requirement(capability_id: str, *, status: str, backend_family: str) -> JSONDict | None:
    template = CAPABILITY_SPIKE_REQUIREMENTS.get(capability_id)
    if template is None:
        return None
    dependency_ids = [str(item) for item in template.get("dependency_ids", [])]
    return {
        "capability_id": capability_id,
        "backend_family": backend_family,
        "current_status": status,
        "live_ready": status == "available",
        "spike_stage": "prove_available" if status != "available" else "verify_available",
        "probe_or_patch_boundary": template["probe_or_patch_boundary"],
        "proof_artifacts": list(template["proof_artifacts"]),
        "proof_artifact_count": len(template["proof_artifacts"]),
        "completion_gate": template["completion_gate"],
        "dependency_ids": dependency_ids,
        "dependency_count": len(dependency_ids),
        "may_mutate_runtime": False,
        "requires_explicit_runtime_approval_before_live": capability_id in {"residency_control", "cleanup_restore"},
    }


def build_summary(plan: JSONDict, path: Path, *, backend_family: str = DEFAULT_BACKEND_FAMILY) -> JSONDict:
    design_summary = plan_phase3_runtime_actuator_design.build_summary(plan, path, backend_family=backend_family)
    adapter = plan_phase3_runtime_actuator_design.find_adapter(plan, backend_family) or {}
    required = live_required_capabilities(plan)
    errors = list(design_summary.get("errors", []))

    proof_requirements: list[JSONDict] = []
    missing_requirement_mappings: list[str] = []
    for capability_id in required:
        requirement = build_proof_requirement(
            capability_id,
            status=status_for_capability(adapter, capability_id),
            backend_family=backend_family,
        )
        if requirement is None:
            missing_requirement_mappings.append(capability_id)
            continue
        proof_requirements.append(requirement)

    if missing_requirement_mappings:
        errors.append(
            "missing actuator spike requirement mappings: " + ", ".join(sorted(missing_requirement_mappings))
        )

    requirement_ids = [str(item["capability_id"]) for item in proof_requirements]
    missing_required_capabilities = sorted(set(required) - set(requirement_ids))
    if missing_required_capabilities:
        errors.append("proof requirements missing required live capabilities: " + ", ".join(missing_required_capabilities))

    dependency_edges: list[JSONDict] = []
    for requirement in proof_requirements:
        capability_id = str(requirement["capability_id"])
        for dependency_id in requirement.get("dependency_ids", []):
            dependency_edges.append({"from": dependency_id, "to": capability_id})
            if dependency_id not in requirement_ids:
                errors.append(f"{capability_id} depends on unknown capability {dependency_id!r}")

    blocking_capabilities = [
        str(item["capability_id"])
        for item in proof_requirements
        if item.get("current_status") != "available"
    ]
    control_blockers = [
        capability_id
        for capability_id in ("residency_observation", "residency_control", "cleanup_restore")
        if capability_id in blocking_capabilities
    ]
    proof_artifact_count = sum(int(item.get("proof_artifact_count", 0)) for item in proof_requirements)
    live_spike_ready = (
        not errors
        and bool(proof_requirements)
        and not blocking_capabilities
        and plan.get("live_expert_loading_implemented") is True
    )
    spike_handoff_ready = (
        not errors
        and design_summary.get("design_handoff_ready") is True
        and bool(proof_requirements)
        and proof_artifact_count >= len(proof_requirements)
        and not live_spike_ready
    )

    return {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "mode": "phase3_runtime_actuator_spike_plan",
        "metadata_only": True,
        "valid": not errors,
        "errors": errors,
        "managed_plan_path": display_path(path),
        "backend_family": backend_family,
        "backend_display_name": adapter.get("display_name"),
        "backend_class": adapter.get("class"),
        "design_handoff_ready": design_summary.get("design_handoff_ready") is True,
        "runtime_actuator_design_schema_version": design_summary.get("schema_version"),
        "spike_handoff_ready": spike_handoff_ready,
        "live_spike_ready": live_spike_ready,
        "live_actuator_ready": False,
        "may_mutate_runtime": False,
        "required_capabilities": required,
        "required_capability_count": len(required),
        "proof_requirements": proof_requirements,
        "proof_requirement_count": len(proof_requirements),
        "proof_artifact_count": proof_artifact_count,
        "blocking_capabilities": blocking_capabilities,
        "blocking_capability_count": len(blocking_capabilities),
        "control_blockers": control_blockers,
        "control_blocker_count": len(control_blockers),
        "dependency_edges": dependency_edges,
        "dependency_edge_count": len(dependency_edges),
        "implementation_sequence": [
            "Verify scanner-derived expert inventory against the selected runtime profile.",
            "Capture semantic routing traces from the llama.cpp gate boundary.",
            "Add read-only residency observation before any control action exists.",
            "Run policy-to-adapter intent dry-runs and keep dense fallback available.",
            "Export live-proof artifacts for observation, fallback, and cleanup sections.",
            "Only after those artifacts validate, attempt a guarded residency-control spike.",
            "Prove cleanup/restore after the guarded spike before considering live paging.",
        ],
        "safety_contract": [
            "runtime actuator spike planning is metadata only",
            "runtime actuator spike planning does not launch runtimes or send prompt traffic",
            "runtime actuator spike planning does not mutate residency or model caches",
            "spike handoff readiness does not imply live actuator readiness",
            "residency control remains blocked until observation, fallback, artifact export, and cleanup proof validate",
        ],
        "next_actions": [
            "Use this handoff to choose the smallest llama.cpp patch/probe boundary for read-only residency observation.",
            "Do not add guarded residency-control mutation until observation, fallback, and artifact-export proofs exist.",
            "Keep cleanup/restore proof as a release gate for any later live actuator branch.",
        ],
    }


def format_markdown_report(summary: JSONDict) -> str:
    lines = [
        "# Phase 3 Runtime Actuator Spike Plan",
        "",
        f"- Valid: `{summary.get('valid')}`",
        f"- Backend: `{summary.get('backend_family')}`",
        f"- Spike handoff ready: `{summary.get('spike_handoff_ready')}`",
        f"- Live spike ready: `{summary.get('live_spike_ready')}`",
        f"- Blocking capabilities: `{', '.join(summary.get('blocking_capabilities', [])) or 'none'}`",
        f"- Proof requirements: `{summary.get('proof_requirement_count')}`",
        f"- Proof artifacts: `{summary.get('proof_artifact_count')}`",
        "",
        "## Requirements",
        "",
        "| Capability | Status | Proof Artifacts | Completion Gate |",
        "| --- | --- | ---: | --- |",
    ]
    for requirement in summary.get("proof_requirements", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(requirement.get("capability_id")),
                    str(requirement.get("current_status")),
                    str(requirement.get("proof_artifact_count")),
                    str(requirement.get("completion_gate")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Safety Contract"])
    for item in summary.get("safety_contract", []):
        lines.append(f"- {item}")
    if summary.get("errors"):
        lines.extend(["", "## Errors"])
        for error in summary["errors"]:
            lines.append(f"- `{error}`")
    lines.extend(["", "## Next Actions"])
    for item in summary.get("next_actions", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def write_markdown_report(summary: JSONDict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_markdown_report(summary), encoding="utf-8")


def print_human_summary(summary: JSONDict) -> None:
    print("MoE Run Anyway Phase 3 runtime actuator spike plan")
    print(f"Plan: {summary['managed_plan_path']}")
    print(f"Valid: {summary['valid']}")
    print(f"Backend: {summary['backend_family']}")
    if summary["errors"]:
        print("Errors:")
        for error in summary["errors"]:
            print(f"  - {error}")
        return
    print(f"Spike handoff ready: {summary['spike_handoff_ready']}")
    print(f"Live spike ready: {summary['live_spike_ready']}")
    print(f"Proof requirements: {summary['proof_requirement_count']}")
    print(f"Proof artifacts: {summary['proof_artifact_count']}")
    print(f"Blocking capabilities: {', '.join(summary['blocking_capabilities']) if summary['blocking_capabilities'] else 'none'}")
    print("Implementation sequence:")
    for item in summary["implementation_sequence"]:
        print(f"  - {item}")
    print("Safety contract:")
    for item in summary["safety_contract"]:
        print(f"  - {item}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "plan_path",
        nargs="?",
        type=Path,
        default=DEFAULT_PLAN_PATH,
        help="managed expert loading plan JSON path",
    )
    parser.add_argument("--backend", default=DEFAULT_BACKEND_FAMILY, help="backend adapter id to inspect")
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    parser.add_argument("--output-md", type=Path, help="write a Markdown actuator-spike report")
    return parser


def plan_path(path: Path, *, backend_family: str = DEFAULT_BACKEND_FAMILY) -> tuple[int, JSONDict | None, str | None]:
    try:
        plan = load_plan(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not load managed expert loading plan: {exc}"

    summary = build_summary(plan, path, backend_family=backend_family)
    return (0 if summary["valid"] else 2), summary, None


def main_from_test_path(path: Path, *, backend_family: str = DEFAULT_BACKEND_FAMILY) -> int:
    status, _, _ = plan_path(path, backend_family=backend_family)
    return status


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary, error_message = plan_path(args.plan_path, backend_family=args.backend)
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