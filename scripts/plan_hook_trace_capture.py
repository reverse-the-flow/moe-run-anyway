#!/usr/bin/env python3
"""Plan hook-level semantic router trace capture.

This planner is intentionally offline. It reads a Model Plane MoE probe
manifest, or a direct local model path, and emits the command sequence for the
existing forward-hook path. It does not import torch or transformers, download
models, authenticate, start servers, run Docker, or send prompt traffic.
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_SCHEMA_VERSION = "model-plane-moe-probe-manifest-v1"
HOOKABLE_PROBE_HINTS = {"hookable_pytorch", "hookable_pytorch_moe", "small_local_moe"}
DEFAULT_SUITE_ARG = "data/mixtral_probe_prompts.json"
DEFAULT_OUTPUT_DIR_ARG = "forward-probe-runs"
DEFAULT_MAX_PROMPTS = 4
DEFAULT_REPEATS = 2
DEFAULT_WINDOW_SIZE_EVENTS = 2
DEFAULT_MAX_NEW_TOKENS = 16

JSONDict = dict[str, Any]


def quote_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def safe_label(value: Any) -> str:
    raw = str(value or "hook-trace").strip().lower()
    chars: list[str] = []
    for char in raw:
        if char.isalnum() or char in {"-", "_"}:
            chars.append(char)
        elif char.isspace() or char in {"/", "."}:
            chars.append("-")
    label = "".join(chars).strip("-_")
    return label or "hook-trace"


def is_uri_like(raw_path: str) -> bool:
    if len(raw_path) >= 3 and raw_path[1] == ":" and raw_path[0].isalpha() and raw_path[2] in {"/", "\\"}:
        return False
    parsed = urlparse(raw_path)
    return bool(parsed.scheme and parsed.scheme not in {"", "file"})


def local_model_path_report(raw_path: str) -> JSONDict:
    expanded = Path(raw_path).expanduser()
    resolved = expanded.resolve() if expanded.exists() else expanded.absolute()
    hints = []
    if expanded.is_dir():
        hints = [
            name
            for name in (
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "model.safetensors.index.json",
                "pytorch_model.bin.index.json",
            )
            if (expanded / name).exists()
        ]
    return {
        "input": raw_path,
        "path": str(resolved),
        "exists": expanded.exists(),
        "is_dir": expanded.is_dir(),
        "is_file": expanded.is_file(),
        "is_uri_like": is_uri_like(raw_path),
        "config_hints": hints,
    }


def load_json(path: Path) -> JSONDict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("manifest root must be a JSON object")
    return data


def manifest_from_model_path(model_path: Path, label: str | None) -> JSONDict:
    return {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "profile_id": label or model_path.name or "local-hookable-model",
        "model_id": label or model_path.name or "local-hookable-model",
        "model_path": str(model_path),
        "backend_family": "pytorch_transformers",
        "primary_probe_hint": "hookable_pytorch",
        "semantic_expert_ids": "expected_when_router_outputs_are_exposed",
        "hookable_runtime_available": True,
    }


def validate_manifest(manifest: JSONDict, model_path_report: JSONDict) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if manifest.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        errors.append(
            "schema_version must be "
            f"{SUPPORTED_SCHEMA_VERSION!r}, got {manifest.get('schema_version')!r}"
        )

    for field in ("profile_id", "model_id", "backend_family", "primary_probe_hint"):
        if not isinstance(manifest.get(field), str) or not str(manifest.get(field)).strip():
            errors.append(f"{field} must be a non-empty string")

    if manifest.get("primary_probe_hint") not in HOOKABLE_PROBE_HINTS:
        errors.append(f"primary_probe_hint must be one of {sorted(HOOKABLE_PROBE_HINTS)!r}")
    if manifest.get("hookable_runtime_available") is not True:
        errors.append("hookable_runtime_available must be true for hook trace capture")
    if manifest.get("semantic_expert_ids") not in {
        "expected_when_router_outputs_are_exposed",
        "runtime_dependent",
    }:
        errors.append("semantic_expert_ids must indicate expected or runtime-dependent router outputs")

    model_path = manifest.get("model_path")
    if not isinstance(model_path, str) or not model_path.strip():
        errors.append("model_path must be a non-empty local filesystem path")
    elif model_path_report["is_uri_like"]:
        errors.append("model_path must be a local filesystem path, not a URI or remote model id")
    elif not model_path_report["exists"]:
        errors.append("model_path does not exist")
    elif not model_path_report["is_dir"]:
        errors.append("model_path must be a local Transformers model directory")

    if model_path_report["exists"] and model_path_report["is_dir"]:
        if "config.json" not in model_path_report["config_hints"]:
            warnings.append("model directory does not contain config.json")
        if not any(
            hint in model_path_report["config_hints"]
            for hint in ("tokenizer.json", "tokenizer_config.json")
        ):
            warnings.append("model directory does not contain tokenizer metadata hints")

    return errors, warnings


def base_transformers_command(
    *,
    model_path: str,
    label: str,
    max_prompts: int,
    repeats: int,
    window_size_events: int,
    max_new_tokens: int,
) -> list[str]:
    return [
        "python3",
        "run_transformers_forward_probe.py",
        "--model-path",
        model_path,
        "--output-dir",
        DEFAULT_OUTPUT_DIR_ARG,
        "--suite-path",
        DEFAULT_SUITE_ARG,
        "--label",
        label,
        "--max-prompts",
        str(max_prompts),
        "--repeats",
        str(repeats),
        "--window-size-events",
        str(window_size_events),
        "--max-new-tokens",
        str(max_new_tokens),
    ]


def build_command_plan(manifest: JSONDict, label: str) -> JSONDict:
    model_path = str(manifest.get("model_path") or "")
    common = base_transformers_command(
        model_path=model_path,
        label=f"{label}-transformers-hook-trace",
        max_prompts=DEFAULT_MAX_PROMPTS,
        repeats=DEFAULT_REPEATS,
        window_size_events=DEFAULT_WINDOW_SIZE_EVENTS,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    )
    synthetic = [
        "python3",
        "run_forward_probe_demo.py",
        "--output-dir",
        DEFAULT_OUTPUT_DIR_ARG,
        "--suite-path",
        DEFAULT_SUITE_ARG,
        "--max-prompts",
        "2",
        "--window-size-events",
        str(DEFAULT_WINDOW_SIZE_EVENTS),
        "--label",
        f"{label}-synthetic-hook-smoke",
    ]
    return {
        "safe_commands": [
            {
                "command_class": "synthetic_hook_smoke",
                "artifact_class_id": "forward_hook_synthetic_smoke_bundle",
                "evidence_label_id": "hook_pipeline_readiness",
                "may_include_prompt_traffic": False,
                "requires_explicit_user_approval": False,
                "command": "cd memory-moe-mvp && " + quote_command(synthetic),
            },
            {
                "command_class": "local_transformers_hook_dry_run",
                "artifact_class_id": "semantic_router_trace_preflight",
                "evidence_label_id": "hookable_model_readiness",
                "may_include_prompt_traffic": False,
                "requires_explicit_user_approval": False,
                "command": "cd memory-moe-mvp && " + quote_command([*common, "--dry-run"]),
            },
        ],
        "deferred_prompt_traffic_commands": [
            {
                "command_class": "approved_local_transformers_hook_trace",
                "artifact_class_id": "forward_hook_router_trace_bundle",
                "evidence_label_id": "semantic_router_trace",
                "may_include_prompt_traffic": True,
                "requires_explicit_user_approval": True,
                "requires_prior_command_classes": [
                    "synthetic_hook_smoke",
                    "local_transformers_hook_dry_run",
                ],
                "command": "cd memory-moe-mvp && " + quote_command(common),
            }
        ],
        "required_pre_prompt_command_classes": [
            "synthetic_hook_smoke",
            "local_transformers_hook_dry_run",
        ],
    }


def build_plan(manifest: JSONDict, source_path: Path | None = None) -> JSONDict:
    model_path_value = str(manifest.get("model_path") or "")
    model_path_report = local_model_path_report(model_path_value) if model_path_value else {}
    errors, warnings = validate_manifest(manifest, model_path_report)
    label = safe_label(manifest.get("profile_id") or manifest.get("model_id"))
    commands = build_command_plan(manifest, label) if not errors else {
        "safe_commands": [],
        "deferred_prompt_traffic_commands": [],
        "required_pre_prompt_command_classes": [],
    }
    return {
        "mode": "hook_trace_capture_plan",
        "phase": "phase_2",
        "stage": "semantic_router_trace_capture",
        "status": "planned_only",
        "source": str(source_path) if source_path else "direct_model_path",
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "profile_id": manifest.get("profile_id"),
        "model_id": manifest.get("model_id"),
        "backend_family": manifest.get("backend_family"),
        "primary_probe_hint": manifest.get("primary_probe_hint"),
        "semantic_expert_ids": manifest.get("semantic_expert_ids"),
        "model_path": model_path_report,
        "artifact_evidence": [
            {
                "artifact_class_id": "semantic_router_trace_preflight",
                "evidence_label_id": "hookable_model_readiness",
                "evidence_type": "readiness_evidence",
                "may_include_prompt_traffic": False,
                "may_claim_semantic_expert_ids": False,
            },
            {
                "artifact_class_id": "forward_hook_router_trace_bundle",
                "evidence_label_id": "semantic_router_trace",
                "evidence_type": "semantic_routing_evidence",
                "may_include_prompt_traffic": True,
                "may_claim_semantic_expert_ids": True,
                "claim_preconditions": [
                    "manifest.json records hook_count greater than zero",
                    "router_events.jsonl contains at least one router event",
                    "summary.json totals.router_event_count is greater than zero",
                    "router event source is expert_indices or router_logits",
                ],
            },
        ],
        "semantic_expert_ids_claim": False,
        "semantic_expert_ids_claim_note": (
            "This plan does not claim semantic expert ids. Only completed hook trace artifacts "
            "that meet claim_preconditions may support that claim."
        ),
        "missing_runtime_actuator": [
            "expert_tensor_pin",
            "expert_tensor_evict",
            "expert_tensor_preload",
            "load_only_routed_experts",
        ],
        "safety_contract": [
            "planner does not import torch or transformers",
            "planner does not download models",
            "planner does not authenticate or inspect token values",
            "planner does not start model servers or Docker",
            "dry-run command uses the guarded local-only Transformers runner",
            "approved hook trace command sends prompts only to a local model path",
        ],
        **commands,
    }


def plan_from_args(args: argparse.Namespace) -> tuple[int, JSONDict | None, str | None]:
    try:
        if args.manifest_path:
            manifest = load_json(args.manifest_path)
            source = args.manifest_path
        else:
            manifest = manifest_from_model_path(args.model_path, args.label)
            source = None
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not load hook trace input: {exc}"

    plan = build_plan(manifest, source)
    return (0 if plan["valid"] else 2), plan, None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest-path", type=Path)
    source.add_argument("--model-path", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--json", action="store_true")
    return parser


def print_human_plan(plan: JSONDict) -> None:
    print("Hook trace capture plan")
    print(f"Valid: {plan['valid']}")
    if plan["errors"]:
        print("Errors:")
        for error in plan["errors"]:
            print(f"  - {error}")
        return
    print(f"Profile: {plan['profile_id']}")
    print(f"Model: {plan['model_id']}")
    print(f"Model path: {plan['model_path']['path']}")
    print("Safe commands:")
    for command in plan["safe_commands"]:
        print(f"  {command['command_class']}: {command['command']}")
    print("Deferred prompt-traffic commands:")
    for command in plan["deferred_prompt_traffic_commands"]:
        print(f"  {command['command_class']}: {command['command']}")


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, plan, error_message = plan_from_args(args)
    if error_message:
        print(error_message)
        return status
    assert plan is not None
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        print_human_plan(plan)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
