#!/usr/bin/env python3
"""Build a reusable Phase 3 prompt-set artifact for a real-evidence bundle.

The prompt set is a provenance artifact used by policy-candidate trace capture
and dense/full-runtime fallback comparison. This builder writes or prints prompt
metadata only; it does not launch runtimes, download models, inspect secrets,
mutate residency, or send prompt traffic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import plan_phase3_dense_fallback_capture
import plan_phase3_real_evidence_bundle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_PATH = ROOT / "memory-moe-mvp" / "phase3-real-evidence" / "pc_mixtral_phase3_real_evidence_bundle.json"
SUPPORTED_SCHEMA_VERSION = "moe-phase3-prompt-set-v1"
DEFAULT_REPEAT_COUNT = 2

JSONDict = dict[str, Any]

PROMPT_TEMPLATES = [
    {
        "group_id": "expert-cache-benefit",
        "prompt": "Give a concise explanation of how expert caching could reduce repeated MoE routing work. Keep it under 120 words.",
        "intent": "policy_candidate_reuse_probe",
    },
    {
        "group_id": "loading-risk-summary",
        "prompt": "Summarize the operational risks of loading only routed experts in a MoE model. Include cleanup and fallback in the answer.",
        "intent": "fallback_quality_probe",
    },
    {
        "group_id": "trace-validation-checklist",
        "prompt": "Write a short checklist for validating a MoE router trace before using it for managed loading decisions.",
        "intent": "artifact_quality_probe",
    },
    {
        "group_id": "observe-vs-managed",
        "prompt": "Compare observe-only routing logs with a managed expert-loading policy in two short paragraphs.",
        "intent": "policy_comparison_probe",
    },
]


def display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def canonical_prompt_set_path(bundle_path: Path) -> Path:
    return bundle_path.with_name(f"{bundle_path.stem}.prompt-set.json")


def build_prompts(*, repeat_count: int, purpose: str) -> list[JSONDict]:
    prompts: list[JSONDict] = []
    for template in PROMPT_TEMPLATES:
        for repeat_index in range(1, repeat_count + 1):
            prompts.append(
                {
                    "prompt_id": f"{purpose}-{template['group_id']}-r{repeat_index}",
                    "group_id": template["group_id"],
                    "repeat": repeat_index,
                    "prompt": template["prompt"],
                    "intent": template["intent"],
                    "expected_reuse_group": template["group_id"],
                }
            )
    return prompts


def validate_prompt_set_artifact(artifact: JSONDict) -> list[str]:
    errors: list[str] = []
    if artifact.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        errors.append(f"schema_version must be {SUPPORTED_SCHEMA_VERSION!r}")
    prompts = artifact.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        errors.append("prompts must be a non-empty list")
    else:
        seen: set[str] = set()
        groups: dict[str, set[int]] = {}
        for index, row in enumerate(prompts):
            if not isinstance(row, dict):
                errors.append(f"prompts[{index}] must be an object")
                continue
            prompt_id = row.get("prompt_id")
            if not isinstance(prompt_id, str) or not prompt_id.strip():
                errors.append(f"prompts[{index}].prompt_id must be a non-empty string")
            elif prompt_id in seen:
                errors.append(f"prompts[{index}] duplicates prompt id {prompt_id!r}")
            else:
                seen.add(prompt_id)
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                errors.append(f"prompts[{index}].prompt must be a non-empty string")
            group_id = row.get("group_id")
            repeat = row.get("repeat")
            if not isinstance(group_id, str) or not group_id.strip():
                errors.append(f"prompts[{index}].group_id must be a non-empty string")
            elif not isinstance(repeat, int) or repeat < 1:
                errors.append(f"prompts[{index}].repeat must be a positive integer")
            else:
                groups.setdefault(group_id, set()).add(repeat)
        repeated_groups = [group_id for group_id, repeats in groups.items() if len(repeats) >= 2]
        if not repeated_groups:
            errors.append("prompt set must include at least one group with two or more repeats")
    required_strings = ("model_id", "backend_family", "prompt_family", "purpose", "source_bundle_path")
    for field in required_strings:
        value = artifact.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field} must be a non-empty string")
    return errors


def build_artifact(bundle_path: Path, *, purpose: str, repeat_count: int) -> JSONDict:
    if repeat_count < 2:
        raise ValueError("repeat_count must be >= 2 so reuse-distance observations are possible")
    manifest = plan_phase3_real_evidence_bundle.load_manifest(bundle_path)
    prompts = build_prompts(repeat_count=repeat_count, purpose=purpose)
    return {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "source_bundle_path": display_path(bundle_path),
        "model_id": manifest.get("model_id"),
        "backend_family": manifest.get("backend_family"),
        "prompt_family": manifest.get("prompt_family"),
        "purpose": purpose,
        "repeat_count_per_group": repeat_count,
        "prompt_count": len(prompts),
        "prompts": prompts,
        "expected_trace_properties": {
            "min_reuse_distance_observations": 1,
            "requires_router_trace": True,
            "requires_selected_experts": True,
            "requires_selected_weights": True,
            "note": "Repeated prompts make reuse observations possible; only replay can prove a policy candidate.",
        },
        "safety_contract": [
            "prompt-set builder writes metadata only",
            "prompt-set builder does not launch model servers",
            "prompt-set builder does not run Docker",
            "prompt-set builder does not download models",
            "prompt-set builder does not inspect private tokens",
            "prompt-set builder does not send prompt traffic",
            "prompt-set builder does not mutate runtime residency",
            "prompt-set builder does not claim live expert paging",
        ],
        "next_actions": [
            "Use this exact prompt set for policy-candidate router trace capture.",
            "Use the same prompt ids for managed and dense/full-runtime output summaries.",
            "Run replay and fallback comparison validators before updating the Phase 3 bundle.",
        ],
    }


def build_summary(artifact: JSONDict, *, output_path: Path | None, errors: list[str]) -> JSONDict:
    prompt_validation: JSONDict
    if output_path is not None and output_path.exists():
        prompt_validation = plan_phase3_dense_fallback_capture.summarize_prompt_set(output_path)
    else:
        prompt_validation = {
            "ready": not errors,
            "prompt_count": artifact.get("prompt_count", 0),
            "prompt_ids": [row.get("prompt_id") for row in artifact.get("prompts", []) if isinstance(row, dict)],
            "errors": errors,
        }
    return {
        "mode": "phase3_prompt_set_builder",
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "valid": not errors,
        "errors": errors,
        "output_path": display_path(output_path),
        "source_bundle_path": artifact.get("source_bundle_path"),
        "model_id": artifact.get("model_id"),
        "prompt_family": artifact.get("prompt_family"),
        "purpose": artifact.get("purpose"),
        "prompt_count": artifact.get("prompt_count"),
        "repeat_count_per_group": artifact.get("repeat_count_per_group"),
        "prompt_validation": prompt_validation,
        "policy_candidate_trace_command": [
            "uv",
            "run",
            "--managed-python",
            "--python",
            "3.13",
            "scripts/plan_phase3_policy_candidate_trace.py",
            str(artifact.get("source_bundle_path")),
            "--candidate-prompt-set-path",
            display_path(output_path) or "<prompt_set_path>",
        ],
        "dense_fallback_capture_command": [
            "uv",
            "run",
            "--managed-python",
            "--python",
            "3.13",
            "scripts/plan_phase3_dense_fallback_capture.py",
            str(artifact.get("source_bundle_path")),
            "--prompt-set-path",
            display_path(output_path) or "<prompt_set_path>",
        ],
        "safety_contract": artifact.get("safety_contract", []),
        "next_actions": artifact.get("next_actions", []),
    }


def write_artifact(path: Path, artifact: JSONDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_path", nargs="?", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--purpose", choices=("policy_candidate", "dense_fallback", "shared_phase3"), default="shared_phase3")
    parser.add_argument("--repeat-count", type=int, default=DEFAULT_REPEAT_COUNT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--default-output", action="store_true", help="write to <bundle>.prompt-set.json")
    parser.add_argument("--artifact-json", action="store_true", help="print the prompt-set artifact")
    parser.add_argument("--json", action="store_true", help="emit machine-readable builder summary")
    return parser


def plan_build(args: argparse.Namespace) -> tuple[int, JSONDict | None, JSONDict | None, str | None]:
    try:
        output_path = args.output
        if args.default_output:
            if output_path is not None:
                return 2, None, None, "--output and --default-output cannot be used together"
            output_path = canonical_prompt_set_path(args.bundle_path)
        artifact = build_artifact(args.bundle_path, purpose=args.purpose, repeat_count=args.repeat_count)
        errors = validate_prompt_set_artifact(artifact)
        if output_path is not None and not errors:
            write_artifact(output_path, artifact)
            errors = [*errors, *plan_phase3_dense_fallback_capture.summarize_prompt_set(output_path).get("errors", [])]
        summary = build_summary(artifact, output_path=output_path, errors=errors)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, None, f"Could not build Phase 3 prompt-set artifact: {exc}"
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
        print("MoE Run Anyway Phase 3 prompt-set builder")
        print(f"Valid: {summary['valid']}")
        print(f"Output path: {summary['output_path']}")
        print(f"Purpose: {summary['purpose']}")
        print(f"Prompts: {summary['prompt_count']}")
        print(f"Repeat count per group: {summary['repeat_count_per_group']}")
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