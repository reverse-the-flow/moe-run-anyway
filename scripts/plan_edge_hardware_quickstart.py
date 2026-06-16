#!/usr/bin/env python3
"""Validate and summarize the edge hardware quickstart matrix.

This planner reads a local JSON matrix and reports feasibility tiers for edge
hardware and small-model routing. It does not download models, authenticate,
run Docker, launch servers, inspect devices, or send prompt traffic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX_PATH = ROOT / "memory-moe-mvp" / "data" / "edge_hardware_quickstart.json"
SUPPORTED_SCHEMA_VERSION = "moe-edge-hardware-quickstart-v1"

REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "name",
    "generated_for",
    "scope",
    "safety_contract",
    "routing_strategy",
    "runtime_families",
    "model_classes",
    "hardware_tiers",
    "recommended_test_path",
}
REQUIRED_TIER_IDS = {
    "cpu_high_ram",
    "vram_8gb",
    "vram_12gb",
    "vram_16gb",
    "vram_24gb",
    "vram_48gb_plus",
    "apple_silicon_unified_memory",
    "jetson_arm_edge",
    "android_phone_emulator",
}
REQUIRED_MODEL_CLASS_IDS = {
    "tiny_smoke_models",
    "small_dense_routable_experts",
    "medium_dense_models",
    "small_moe_smoke_models",
    "qwen3_30b_a3b",
    "deepseek_v2_lite",
    "mixtral_class",
}
REQUIRED_RUNTIME_FAMILY_IDS = {
    "llama_cpp",
    "openai_compatible",
    "ollama",
    "vllm",
    "transformers_hookable",
    "mlc_llm",
    "executorch",
    "coreml_mlx",
    "tensorrt_llm",
    "android_emulator_client",
}
REQUIRED_TEST_STEP_IDS = {
    "baseline_readiness",
    "tiny_smoke_route",
    "router_artifact_compare",
    "semantic_claim_gate",
}
REQUIRED_SAFETY_LINES = {
    "Planner does not download models.",
    "Planner does not authenticate.",
    "Planner does not run Docker.",
    "Planner does not launch model servers.",
    "Planner does not inspect devices.",
    "Planner does not send prompt traffic.",
}

JSONDict = dict[str, Any]


def load_matrix(path: Path) -> JSONDict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("matrix root must be a JSON object")
    return data


def require_string(obj: JSONDict, field: str, errors: list[str], *, context: str) -> None:
    value = obj.get(field)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{context}.{field} must be a non-empty string")


def require_string_list(obj: JSONDict, field: str, errors: list[str], *, context: str) -> None:
    value = obj.get(field)
    if not isinstance(value, list) or not value:
        errors.append(f"{context}.{field} must be a non-empty list")
        return
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            errors.append(f"{context}.{field}[{index}] must be a non-empty string")


def collect_ids(value: Any, *, context: str, errors: list[str]) -> set[str]:
    if not isinstance(value, list) or not value:
        errors.append(f"{context} must be a non-empty list")
        return set()

    ids: set[str] = set()
    for index, item in enumerate(value):
        item_context = f"{context}[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{item_context} must be an object")
            continue
        require_string(item, "id", errors, context=item_context)
        item_id = item.get("id")
        if isinstance(item_id, str):
            if item_id in ids:
                errors.append(f"{item_context}.id duplicates {item_id!r}")
            ids.add(item_id)
    return ids


def require_exact_ids(
    actual_ids: set[str],
    required_ids: set[str],
    *,
    context: str,
    errors: list[str],
) -> None:
    missing = sorted(required_ids - actual_ids)
    extra = sorted(actual_ids - required_ids)
    if missing:
        errors.append(f"{context} missing required ids: {', '.join(missing)}")
    if extra:
        errors.append(f"{context} has unsupported ids: {', '.join(extra)}")


def validate_runtime_families(matrix: JSONDict, errors: list[str]) -> None:
    families = matrix.get("runtime_families")
    ids = collect_ids(families, context="runtime_families", errors=errors)
    require_exact_ids(ids, REQUIRED_RUNTIME_FAMILY_IDS, context="runtime_families", errors=errors)

    if not isinstance(families, list):
        return
    for index, family in enumerate(families):
        if not isinstance(family, dict):
            continue
        context = f"runtime_families[{index}]"
        require_string(family, "display_name", errors, context=context)
        require_string(family, "notes", errors, context=context)


def validate_model_classes(matrix: JSONDict, errors: list[str]) -> None:
    classes = matrix.get("model_classes")
    ids = collect_ids(classes, context="model_classes", errors=errors)
    require_exact_ids(ids, REQUIRED_MODEL_CLASS_IDS, context="model_classes", errors=errors)

    if not isinstance(classes, list):
        return
    for index, model_class in enumerate(classes):
        if not isinstance(model_class, dict):
            continue
        context = f"model_classes[{index}]"
        require_string(model_class, "display_name", errors, context=context)
        require_string(model_class, "routing_role", errors, context=context)
        require_string_list(model_class, "examples", errors, context=context)


def validate_routing_strategy(matrix: JSONDict, errors: list[str]) -> None:
    strategy = matrix.get("routing_strategy")
    if not isinstance(strategy, dict):
        errors.append("routing_strategy must be an object")
        return

    require_string_list(strategy, "near_term", errors, context="routing_strategy")
    require_string_list(strategy, "later", errors, context="routing_strategy")

    near_term = " ".join(strategy.get("near_term", [])) if isinstance(strategy.get("near_term"), list) else ""
    if "small dense" not in near_term:
        errors.append("routing_strategy.near_term must describe small dense model routing")


def validate_hardware_tiers(matrix: JSONDict, errors: list[str]) -> None:
    tiers = matrix.get("hardware_tiers")
    tier_ids = collect_ids(tiers, context="hardware_tiers", errors=errors)
    require_exact_ids(tier_ids, REQUIRED_TIER_IDS, context="hardware_tiers", errors=errors)

    runtime_ids = collect_ids(matrix.get("runtime_families"), context="runtime_families", errors=[])
    model_class_ids = collect_ids(matrix.get("model_classes"), context="model_classes", errors=[])

    if not isinstance(tiers, list):
        return
    for index, tier in enumerate(tiers):
        if not isinstance(tier, dict):
            continue
        context = f"hardware_tiers[{index}]"
        for field in (
            "display_name",
            "target_type",
            "minimum_memory",
            "comfortable_memory",
            "recommended_first_test",
        ):
            require_string(tier, field, errors, context=context)
        for field in (
            "storage_notes",
            "context_kv_cache_notes",
            "candidate_runtime_families",
            "candidate_model_classes",
            "risk_notes",
        ):
            require_string_list(tier, field, errors, context=context)

        runtimes = set(tier.get("candidate_runtime_families", []))
        unknown_runtimes = sorted(runtimes - runtime_ids)
        if unknown_runtimes:
            errors.append(f"{context}.candidate_runtime_families has unknown ids: {', '.join(unknown_runtimes)}")

        classes = set(tier.get("candidate_model_classes", []))
        unknown_classes = sorted(classes - model_class_ids)
        if unknown_classes:
            errors.append(f"{context}.candidate_model_classes has unknown ids: {', '.join(unknown_classes)}")

        if "small_dense_routable_experts" not in classes:
            errors.append(f"{context}.candidate_model_classes must include small_dense_routable_experts")

        joined_risks = " ".join(tier.get("risk_notes", [])) if isinstance(tier.get("risk_notes"), list) else ""
        if tier.get("id") == "android_phone_emulator":
            if "Emulator" not in joined_risks or "real mobile" not in joined_risks:
                errors.append("hardware_tiers.android_phone_emulator must distinguish emulator smoke from real mobile testing")


def validate_recommended_test_path(matrix: JSONDict, errors: list[str]) -> None:
    steps = matrix.get("recommended_test_path")
    step_ids = collect_ids(steps, context="recommended_test_path", errors=errors)
    require_exact_ids(step_ids, REQUIRED_TEST_STEP_IDS, context="recommended_test_path", errors=errors)

    if not isinstance(steps, list):
        return
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        require_string(step, "description", errors, context=f"recommended_test_path[{index}]")


def validate_matrix(matrix: JSONDict) -> list[str]:
    errors: list[str] = []
    if matrix.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        errors.append(
            "schema_version must be "
            f"{SUPPORTED_SCHEMA_VERSION!r}, got {matrix.get('schema_version')!r}"
        )

    missing_fields = sorted(REQUIRED_TOP_LEVEL_FIELDS - set(matrix))
    if missing_fields:
        errors.append(f"matrix missing required top-level fields: {', '.join(missing_fields)}")

    require_string(matrix, "name", errors, context="matrix")
    require_string(matrix, "generated_for", errors, context="matrix")
    require_string_list(matrix, "scope", errors, context="matrix")
    require_string_list(matrix, "safety_contract", errors, context="matrix")
    missing_safety = sorted(REQUIRED_SAFETY_LINES - set(matrix.get("safety_contract", [])))
    if missing_safety:
        errors.append(f"safety_contract missing required lines: {'; '.join(missing_safety)}")

    validate_routing_strategy(matrix, errors)
    validate_runtime_families(matrix, errors)
    validate_model_classes(matrix, errors)
    validate_hardware_tiers(matrix, errors)
    validate_recommended_test_path(matrix, errors)
    return errors


def tier_summary(matrix: JSONDict) -> list[JSONDict]:
    tiers = matrix.get("hardware_tiers")
    if not isinstance(tiers, list):
        return []

    result: list[JSONDict] = []
    for tier in tiers:
        if not isinstance(tier, dict):
            continue
        result.append(
            {
                "id": tier.get("id"),
                "display_name": tier.get("display_name"),
                "target_type": tier.get("target_type"),
                "candidate_runtime_families": tier.get("candidate_runtime_families", []),
                "candidate_model_classes": tier.get("candidate_model_classes", []),
                "recommended_first_test": tier.get("recommended_first_test"),
            }
        )
    return result


def build_summary(matrix: JSONDict, path: Path) -> JSONDict:
    errors = validate_matrix(matrix)
    return {
        "mode": "edge_hardware_quickstart_plan",
        "matrix_path": str(path),
        "valid": not errors,
        "errors": errors,
        "schema_version": matrix.get("schema_version"),
        "name": matrix.get("name"),
        "generated_for": matrix.get("generated_for"),
        "tier_count": len(matrix.get("hardware_tiers", [])) if isinstance(matrix.get("hardware_tiers"), list) else 0,
        "hardware_tiers": tier_summary(matrix),
        "recommended_test_path": matrix.get("recommended_test_path", []),
        "safety_contract": matrix.get("safety_contract", []),
    }


def print_human_summary(summary: JSONDict) -> None:
    print("MoE Run Anyway edge hardware quickstart")
    print(f"Matrix: {summary['matrix_path']}")
    print(f"Valid: {summary['valid']}")
    if summary["errors"]:
        print("Errors:")
        for error in summary["errors"]:
            print(f"  - {error}")
        return

    print(f"Schema: {summary['schema_version']}")
    print(f"Hardware tiers: {summary['tier_count']}")
    for tier in summary["hardware_tiers"]:
        classes = ", ".join(tier["candidate_model_classes"])
        runtimes = ", ".join(tier["candidate_runtime_families"])
        print(f"  - {tier['id']}: {tier['display_name']}")
        print(f"    runtimes: {runtimes}")
        print(f"    model classes: {classes}")
        print(f"    first test: {tier['recommended_first_test']}")
    print("Recommended test path:")
    for step in summary["recommended_test_path"]:
        print(f"  - {step['id']}: {step['description']}")
    print("Safety contract:")
    for item in summary["safety_contract"]:
        print(f"  - {item}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "matrix_path",
        nargs="?",
        type=Path,
        default=DEFAULT_MATRIX_PATH,
        help="edge hardware quickstart JSON path",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    return parser


def plan_path(path: Path) -> tuple[int, JSONDict | None, str | None]:
    try:
        matrix = load_matrix(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not load edge hardware quickstart matrix: {exc}"

    summary = build_summary(matrix, path)
    return (0 if summary["valid"] else 2), summary, None


def main_from_test_path(path: Path) -> int:
    status, _, _ = plan_path(path)
    return status


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary, error_message = plan_path(args.matrix_path)
    if error_message:
        print(error_message, file=sys.stderr)
        return status

    assert summary is not None
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_human_summary(summary)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
