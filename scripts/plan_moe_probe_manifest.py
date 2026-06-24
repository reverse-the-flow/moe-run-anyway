#!/usr/bin/env python3
"""Plan MoE Run Anyway probes from a Model Plane MoE probe manifest.

The manifest is intentionally small and file-friendly. This planner validates
only the bridge contract, then prints commands for existing safe probe paths. It
does not start model servers, download models, authenticate, inspect tokens, run
Docker, or send prompt traffic.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE_ARG = "memory-moe-mvp/data/mixtral_probe_prompts.json"
SUPPORTED_SCHEMA_VERSION = "model-plane-moe-probe-manifest-v1"
PASSIVE_PROBE_HINTS = {"passive_sidecar", "passive_sidecar_proxy"}
HOOKABLE_PROBE_HINTS = {"hookable_pytorch", "hookable_pytorch_moe", "small_local_moe"}
OPENAI_COMPATIBLE_BACKEND_FAMILIES = {
    "vllm_openai_compatible",
    "ollama_openai_compatible",
    "openai_compatible",
}

JSONDict = dict[str, Any]


def quote_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def safe_label(value: Any) -> str:
    raw = str(value or "model-plane-profile").strip().lower()
    chars = []
    for char in raw:
        if char.isalnum() or char in {"-", "_"}:
            chars.append(char)
        elif char.isspace() or char in {"/", "."}:
            chars.append("-")
    label = "".join(chars).strip("-_")
    return label or "model-plane-profile"


def load_manifest(path: Path) -> JSONDict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("manifest root must be a JSON object")
    return data


def validate_manifest(manifest: JSONDict) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        errors.append(
            "schema_version must be "
            f"{SUPPORTED_SCHEMA_VERSION!r}, got {manifest.get('schema_version')!r}"
        )

    for field in ("profile_id", "model_id", "backend_family", "base_url", "primary_probe_hint"):
        if not isinstance(manifest.get(field), str) or not str(manifest.get(field)).strip():
            errors.append(f"{field} must be a non-empty string")

    semantic_status = manifest.get("semantic_expert_ids")
    if semantic_status not in {
        "not_exposed",
        "expected_when_router_outputs_are_exposed",
        "runtime_dependent",
    }:
        errors.append("semantic_expert_ids has an unsupported value")

    hint = str(manifest.get("primary_probe_hint", ""))
    hookable = bool(manifest.get("hookable_runtime_available"))
    if hint in HOOKABLE_PROBE_HINTS and not hookable:
        errors.append("hookable probe hints require hookable_runtime_available=true")

    return errors


def selected_target_class(manifest: JSONDict) -> str:
    hint = str(manifest.get("primary_probe_hint", "runtime_baseline"))
    backend_family = str(manifest.get("backend_family", ""))
    if hint in PASSIVE_PROBE_HINTS:
        return "passive_sidecar_proxy"
    if hint in HOOKABLE_PROBE_HINTS and bool(manifest.get("hookable_runtime_available")):
        return "hookable_pytorch_moe"
    if backend_family in OPENAI_COMPATIBLE_BACKEND_FAMILIES:
        return "openai_compatible_runtime"
    return "stock_llama_cpp_openai_compatible"


def optional_log_args(manifest: JSONDict) -> list[str]:
    log_file_path = manifest.get("log_file_path")
    if isinstance(log_file_path, str) and log_file_path.strip():
        return ["--log-file-path", log_file_path]
    return []


def runtime_baseline_commands(manifest: JSONDict, label: str) -> list[str]:
    base_url = normalize_base_url(str(manifest["base_url"]))
    model_id = str(manifest["model_id"])
    backend_family = str(manifest["backend_family"])
    runtime_label = (
        f"{label}-openai-compatible-runtime"
        if backend_family in OPENAI_COMPATIBLE_BACKEND_FAMILIES
        else f"{label}-runtime"
    )
    common = [
        "--base-url",
        base_url,
        "--backend-family",
        backend_family,
        "--model",
        model_id,
        "--output-dir",
        "memory-moe-mvp/runtime-probe-runs",
        "--label",
        runtime_label,
        "--suite-path",
        DEFAULT_SUITE_ARG,
        "--max-prompts",
        "4",
        "--repeats",
        "2",
        "--preflight-timeout-seconds",
        "2",
    ] + optional_log_args(manifest)
    return [
        quote_command(["python3", "scripts/run_live_baseline.py", *common, "--dry-run"]),
        quote_command(["python3", "scripts/run_live_baseline.py", *common, "--preflight-only"]),
    ]


def passive_sidecar_commands(manifest: JSONDict, label: str) -> list[str]:
    return [
        "cd memory-moe-mvp && "
        + quote_command(
            [
                "python3",
                "llama_sidecar.py",
                "--listen-port",
                "8091",
                "--upstream-base-url",
                normalize_base_url(str(manifest["base_url"])),
                "--output-dir",
                "sidecar-runs",
                "--label",
                f"{label}-sidecar",
                "--capture-upstream-observability",
            ]
        )
    ]


def hookable_pytorch_commands(manifest: JSONDict, label: str) -> list[str]:
    commands = [
        "cd memory-moe-mvp && "
        + quote_command(
            [
                "python3",
                "run_forward_probe_demo.py",
                "--output-dir",
                "forward-probe-runs",
                "--suite-path",
                "data/mixtral_probe_prompts.json",
                "--max-prompts",
                "2",
                "--window-size-events",
                "2",
                "--label",
                f"{label}-synthetic-hook-smoke",
            ]
        )
    ]
    model_path = manifest.get("model_path")
    if isinstance(model_path, str) and model_path.strip():
        common = [
            "python3",
            "run_transformers_forward_probe.py",
            "--model-path",
            model_path,
            "--output-dir",
            "forward-probe-runs",
            "--suite-path",
            "data/mixtral_probe_prompts.json",
            "--label",
            f"{label}-transformers-hook-trace",
            "--max-prompts",
            "4",
            "--repeats",
            "2",
            "--window-size-events",
            "2",
        ]
        commands.append(
            "cd memory-moe-mvp && "
            + quote_command([*common, "--dry-run"])
        )
        commands.append(
            "cd memory-moe-mvp && "
            + quote_command(common)
        )
    return commands


def honesty_note(manifest: JSONDict, target_class: str) -> str:
    semantic_status = str(manifest.get("semantic_expert_ids"))
    if target_class == "hookable_pytorch_moe":
        return "Semantic expert ids are expected only if the local hookable runtime exposes router outputs."
    if semantic_status == "not_exposed":
        return "This path can collect runtime/request telemetry, not semantic expert ids."
    return "Semantic expert ids remain runtime-dependent; do not infer them from stock endpoint telemetry."


def planned_harness_run_request(
    manifest: JSONDict,
    target_class: str,
    commands: list[str],
    deferred_live_commands: list[str],
) -> JSONDict:
    artifact_class = {
        "passive_sidecar_proxy": "sidecar_manifest_events_summary",
        "hookable_pytorch_moe": "forward_hook_trace_or_synthetic_smoke",
        "openai_compatible_runtime": "runtime_baseline_preflight_or_probe_bundle",
        "stock_llama_cpp_openai_compatible": "runtime_baseline_preflight_or_probe_bundle",
    }.get(target_class, "unknown")
    return {
        "stage": "harness_run_request",
        "status": "planned_only",
        "source": "model_plane_moe_probe_manifest",
        "profile_id": manifest.get("profile_id"),
        "target_class": target_class,
        "approved_command_class_required": True,
        "safe_commands": commands,
        "deferred_live_commands": deferred_live_commands,
        "expected_artifact_class": artifact_class,
        "missing_runtime_actuator": [
            "expert_tensor_pin",
            "expert_tensor_evict",
            "expert_tensor_preload",
            "load_only_routed_experts",
        ],
        "notes": [
            "This is a planning artifact, not an execution API.",
            "Runtime actuator work requires a future backend hook, patch, or fork.",
        ],
    }


def build_plan(manifest: JSONDict, manifest_path: Path) -> JSONDict:
    errors = validate_manifest(manifest)
    target_class = selected_target_class(manifest) if not errors else "invalid_manifest"
    label = safe_label(manifest.get("profile_id"))

    commands: list[str] = []
    deferred_live_commands: list[str] = []
    if not errors:
        if target_class == "passive_sidecar_proxy":
            commands = passive_sidecar_commands(manifest, label)
        elif target_class == "hookable_pytorch_moe":
            hookable_commands = hookable_pytorch_commands(manifest, label)
            commands = hookable_commands[:2]
            deferred_live_commands = hookable_commands[2:]
        else:
            commands = runtime_baseline_commands(manifest, label)

    return {
        "mode": "model_plane_moe_probe_manifest_plan",
        "manifest_path": str(manifest_path),
        "valid": not errors,
        "errors": errors,
        "profile_id": manifest.get("profile_id"),
        "model_id": manifest.get("model_id"),
        "backend_family": manifest.get("backend_family"),
        "base_url": normalize_base_url(str(manifest.get("base_url", ""))),
        "health_url": manifest.get("health_url"),
        "container_name": manifest.get("container_name"),
        "log_file_path": manifest.get("log_file_path"),
        "primary_probe_hint": manifest.get("primary_probe_hint"),
        "target_class": target_class,
        "semantic_expert_ids": manifest.get("semantic_expert_ids"),
        "honesty_note": honesty_note(manifest, target_class) if not errors else None,
        "safe_commands": commands,
        "deferred_live_commands": deferred_live_commands,
        "planned_harness_run_request": (
            planned_harness_run_request(manifest, target_class, commands, deferred_live_commands)
            if not errors
            else None
        ),
        "safety_contract": [
            "planner does not start containers or model servers",
            "planner does not download models",
            "planner does not authenticate or inspect private tokens",
            "runtime baseline commands are dry-run or preflight-only",
            "stock endpoint telemetry is runtime evidence, not semantic expert ids",
            "hookable PyTorch semantic probing is shown only when the manifest declares hookable_runtime_available=true",
        ],
    }


def print_human_plan(plan: JSONDict) -> None:
    print("Model Plane MoE probe plan")
    print(f"Manifest: {plan['manifest_path']}")
    print(f"Valid: {plan['valid']}")
    if plan["errors"]:
        print("Errors:")
        for error in plan["errors"]:
            print(f"  - {error}")
        return

    print(f"Profile: {plan['profile_id']}")
    print(f"Model: {plan['model_id']}")
    print(f"Backend family: {plan['backend_family']}")
    print(f"Base URL: {plan['base_url']}")
    print(f"Target class: {plan['target_class']}")
    print(f"Semantic expert ids: {plan['semantic_expert_ids']}")
    print(f"Note: {plan['honesty_note']}")
    print("Safe commands:")
    for command in plan["safe_commands"]:
        print(f"  {command}")
    if plan["deferred_live_commands"]:
        print("Deferred live commands:")
        for command in plan["deferred_live_commands"]:
            print(f"  {command}")
    request = plan["planned_harness_run_request"]
    print("Planned harness run request:")
    print(f"  Stage: {request['stage']}")
    print(f"  Status: {request['status']}")
    print(f"  Expected artifact class: {request['expected_artifact_class']}")
    print("  Missing runtime actuator:")
    for item in request["missing_runtime_actuator"]:
        print(f"    - {item}")
    print("Safety contract:")
    for item in plan["safety_contract"]:
        print(f"  - {item}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_path", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def plan_manifest_path(path: Path) -> tuple[int, JSONDict | None, str | None]:
    try:
        manifest = load_manifest(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not load manifest: {exc}"

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
