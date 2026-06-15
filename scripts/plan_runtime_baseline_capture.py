#!/usr/bin/env python3
"""Plan Phase 1 runtime baseline capture from a saved Model Plane manifest.

This planner is intentionally offline. It reads a saved Model Plane MoE probe
manifest and emits the dry-run/preflight capture packet required before any
approved prompt traffic. It does not inspect endpoints, start servers, download
models, authenticate, read tokens, run Docker, mutate caches, or send prompt
traffic.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = (
    ROOT / "memory-moe-mvp" / "data" / "model_plane_moe_probe_manifest.runtime_baseline.fixture.json"
)
DEFAULT_SUITE_ARG = "memory-moe-mvp/data/mixtral_probe_prompts.json"
DEFAULT_OUTPUT_DIR_ARG = "memory-moe-mvp/runtime-probe-runs"
SUPPORTED_SCHEMA_VERSION = "model-plane-moe-probe-manifest-v1"
RUNTIME_BASELINE_HINT = "runtime_baseline"
LLAMA_CPP_BACKEND_FAMILY = "llama_cpp"
OPENAI_COMPATIBLE_BACKEND_FAMILIES = {
    "vllm_openai_compatible",
    "ollama_openai_compatible",
    "openai_compatible",
}
SUPPORTED_BACKEND_FAMILIES = {LLAMA_CPP_BACKEND_FAMILY, *OPENAI_COMPATIBLE_BACKEND_FAMILIES}
SUPPORTED_SEMANTIC_EXPERT_ID_STATES = {
    "not_exposed",
    "runtime_dependent",
    "expected_when_router_outputs_are_exposed",
}

JSONDict = dict[str, Any]


def quote_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def safe_label(value: Any) -> str:
    raw = str(value or "runtime-baseline").strip().lower()
    chars = []
    for char in raw:
        if char.isalnum() or char in {"-", "_"}:
            chars.append(char)
        elif char.isspace() or char in {"/", "."}:
            chars.append("-")
    label = "".join(chars).strip("-_")
    return label or "runtime-baseline"


def load_manifest(path: Path) -> JSONDict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("manifest root must be a JSON object")
    return data


def require_string(manifest: JSONDict, field: str, errors: list[str]) -> None:
    value = manifest.get(field)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{field} must be a non-empty string")


def validate_manifest(manifest: JSONDict) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        errors.append(
            "schema_version must be "
            f"{SUPPORTED_SCHEMA_VERSION!r}, got {manifest.get('schema_version')!r}"
        )

    for field in ("profile_id", "model_id", "backend_family", "base_url", "primary_probe_hint"):
        require_string(manifest, field, errors)

    backend_family = manifest.get("backend_family")
    if isinstance(backend_family, str) and backend_family not in SUPPORTED_BACKEND_FAMILIES:
        errors.append(f"backend_family must be one of {sorted(SUPPORTED_BACKEND_FAMILIES)!r}")

    if manifest.get("primary_probe_hint") != RUNTIME_BASELINE_HINT:
        errors.append("primary_probe_hint must be 'runtime_baseline' for Phase 1 capture planning")

    semantic_expert_ids = manifest.get("semantic_expert_ids")
    if semantic_expert_ids not in SUPPORTED_SEMANTIC_EXPERT_ID_STATES:
        errors.append("semantic_expert_ids has an unsupported value")
    elif semantic_expert_ids == "expected_when_router_outputs_are_exposed":
        errors.append(
            "runtime baseline capture cannot claim semantic expert ids; use a semantic router trace planner"
        )

    hookable = manifest.get("hookable_runtime_available")
    if hookable is not None and not isinstance(hookable, bool):
        errors.append("hookable_runtime_available must be a boolean when present")

    for section_name in ("runtime_observability",):
        section = manifest.get(section_name)
        if section is not None and not isinstance(section, dict):
            errors.append(f"{section_name} must be an object when present")

    return errors


def optional_log_args(manifest: JSONDict) -> list[str]:
    log_file_path = manifest.get("log_file_path")
    if isinstance(log_file_path, str) and log_file_path.strip():
        return ["--log-file-path", log_file_path]
    return []


def runtime_label(manifest: JSONDict) -> str:
    label = safe_label(manifest.get("profile_id"))
    if manifest.get("backend_family") in OPENAI_COMPATIBLE_BACKEND_FAMILIES:
        return f"{label}-openai-compatible-runtime"
    return f"{label}-runtime"


def base_runtime_command(manifest: JSONDict) -> list[str]:
    return [
        "python3",
        "scripts/run_live_baseline.py",
        "--base-url",
        normalize_base_url(str(manifest["base_url"])),
        "--backend-family",
        str(manifest["backend_family"]),
        "--model",
        str(manifest["model_id"]),
        "--output-dir",
        DEFAULT_OUTPUT_DIR_ARG,
        "--label",
        runtime_label(manifest),
        "--suite-path",
        DEFAULT_SUITE_ARG,
        "--max-prompts",
        "4",
        "--repeats",
        "2",
        "--preflight-timeout-seconds",
        "2",
        *optional_log_args(manifest),
    ]


def build_command_plan(manifest: JSONDict) -> JSONDict:
    common = base_runtime_command(manifest)
    dry_run = [*common, "--dry-run", "--json"]
    preflight = [*common, "--preflight-only", "--json"]
    approved_probe = [*common, "--json"]
    return {
        "safe_commands": [
            {
                "command_class": "dry_run_plan",
                "artifact_class_id": "runtime_baseline_preflight_bundle",
                "evidence_label_id": "runtime_readiness_evidence",
                "may_include_prompt_traffic": False,
                "requires_explicit_user_approval": False,
                "command": quote_command(dry_run),
            },
            {
                "command_class": "preflight_only_readiness_gate",
                "artifact_class_id": "runtime_baseline_preflight_bundle",
                "evidence_label_id": "runtime_readiness_evidence",
                "may_include_prompt_traffic": False,
                "requires_explicit_user_approval": False,
                "command": quote_command(preflight),
            },
        ],
        "deferred_prompt_traffic_commands": [
            {
                "command_class": "approved_runtime_probe",
                "artifact_class_id": "runtime_baseline_probe_bundle",
                "evidence_label_id": "runtime_request_telemetry",
                "may_include_prompt_traffic": True,
                "requires_explicit_user_approval": True,
                "requires_prior_command_classes": [
                    "dry_run_plan",
                    "preflight_only_readiness_gate",
                ],
                "command": quote_command(approved_probe),
            }
        ],
        "required_pre_prompt_command_classes": [
            "dry_run_plan",
            "preflight_only_readiness_gate",
        ],
    }


def readiness_paths(manifest: JSONDict) -> list[str]:
    observability = manifest.get("runtime_observability")
    if isinstance(observability, dict) and isinstance(observability.get("readiness_paths"), list):
        return [path for path in observability["readiness_paths"] if isinstance(path, str)]
    if manifest.get("backend_family") in OPENAI_COMPATIBLE_BACKEND_FAMILIES:
        return ["/v1/models", "/models"]
    return ["/models"]


def observability_paths(manifest: JSONDict) -> list[str]:
    observability = manifest.get("runtime_observability")
    if isinstance(observability, dict) and isinstance(observability.get("expected_paths"), list):
        return [path for path in observability["expected_paths"] if isinstance(path, str)]
    if manifest.get("backend_family") == LLAMA_CPP_BACKEND_FAMILY:
        return ["/props", "/metrics", "/slots"]
    return []


def build_capture_packet(manifest: JSONDict) -> JSONDict:
    commands = build_command_plan(manifest)
    return {
        "phase": "phase_1",
        "stage": "runtime_baseline_capture",
        "status": "planned_only",
        "source": "model_plane_moe_probe_manifest",
        "profile_id": manifest.get("profile_id"),
        "model_id": manifest.get("model_id"),
        "backend_family": manifest.get("backend_family"),
        "base_url": normalize_base_url(str(manifest.get("base_url", ""))),
        "health_url": manifest.get("health_url"),
        "container_name": manifest.get("container_name"),
        "log_file_path": manifest.get("log_file_path"),
        "runtime_observability": {
            "expected_paths": observability_paths(manifest),
            "readiness_paths": readiness_paths(manifest),
        },
        "artifact_evidence": [
            {
                "artifact_class_id": "runtime_baseline_preflight_bundle",
                "evidence_label_id": "runtime_readiness_evidence",
                "evidence_type": "runtime_evidence",
                "may_include_prompt_traffic": False,
                "may_claim_semantic_expert_ids": False,
            },
            {
                "artifact_class_id": "runtime_baseline_probe_bundle",
                "evidence_label_id": "runtime_request_telemetry",
                "evidence_type": "runtime_evidence",
                "may_include_prompt_traffic": True,
                "may_claim_semantic_expert_ids": False,
            },
        ],
        "semantic_expert_ids_claim": False,
        "semantic_routing_evidence_label": "semantic_router_trace",
        "semantic_routing_evidence_required_for_expert_ids": True,
        "stock_endpoint_telemetry_note": (
            "Stock endpoint telemetry is runtime evidence and cannot claim semantic expert ids."
        ),
        **commands,
    }


def build_plan(manifest: JSONDict, manifest_path: Path) -> JSONDict:
    errors = validate_manifest(manifest)
    capture_packet = build_capture_packet(manifest) if not errors else None
    return {
        "mode": "runtime_baseline_capture_plan",
        "manifest_path": str(manifest_path),
        "valid": not errors,
        "errors": errors,
        "schema_version": manifest.get("schema_version"),
        "profile_id": manifest.get("profile_id"),
        "model_id": manifest.get("model_id"),
        "backend_family": manifest.get("backend_family"),
        "primary_probe_hint": manifest.get("primary_probe_hint"),
        "semantic_expert_ids": manifest.get("semantic_expert_ids"),
        "capture_packet": capture_packet,
        "safety_contract": [
            "planner does not inspect endpoints",
            "planner does not start containers or model servers",
            "planner does not download models",
            "planner does not authenticate, inspect, read, or print private tokens",
            "planner does not run Docker",
            "planner does not mutate model caches",
            "planner does not send prompt traffic",
            "safe commands are dry-run or preflight-only until explicit user approval",
            "planned artifacts are runtime evidence, not semantic routing evidence",
        ],
    }


def print_human_plan(plan: JSONDict) -> None:
    print("MoE Run Anyway runtime baseline capture plan")
    print(f"Manifest: {plan['manifest_path']}")
    print(f"Valid: {plan['valid']}")
    if plan["errors"]:
        print("Errors:")
        for error in plan["errors"]:
            print(f"  - {error}")
        return

    packet = plan["capture_packet"]
    assert packet is not None
    print(f"Phase: {packet['phase']}")
    print(f"Profile: {packet['profile_id']}")
    print(f"Model: {packet['model_id']}")
    print(f"Backend family: {packet['backend_family']}")
    print("Artifact evidence:")
    for item in packet["artifact_evidence"]:
        print(
            "  - "
            f"{item['artifact_class_id']} -> {item['evidence_label_id']} "
            f"({item['evidence_type']})"
        )
    print(f"Semantic expert ids claim: {packet['semantic_expert_ids_claim']}")
    print(f"Note: {packet['stock_endpoint_telemetry_note']}")
    print("Required pre-prompt command classes:")
    for command_class in packet["required_pre_prompt_command_classes"]:
        print(f"  - {command_class}")
    print("Safe commands:")
    for item in packet["safe_commands"]:
        print(f"  {item['command']}")
    print("Deferred prompt-traffic commands:")
    for item in packet["deferred_prompt_traffic_commands"]:
        print(f"  {item['command_class']}: requires explicit user approval")
    print("Safety contract:")
    for item in plan["safety_contract"]:
        print(f"  - {item}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest_path",
        nargs="?",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="saved Model Plane MoE runtime-baseline manifest JSON path",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable capture plan")
    return parser


def plan_manifest_path(path: Path) -> tuple[int, JSONDict | None, str | None]:
    try:
        manifest = load_manifest(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not load runtime baseline manifest: {exc}"

    plan = build_plan(manifest, path)
    return (0 if plan["valid"] else 2), plan, None


def main_from_test_path(path: Path) -> int:
    status, _, _ = plan_manifest_path(path)
    return status


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, plan, error_message = plan_manifest_path(args.manifest_path)
    if error_message:
        print(error_message, file=sys.stderr)
        return status

    assert plan is not None
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        print_human_plan(plan)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
