#!/usr/bin/env python3
"""Validate the local MoE model-target registry.

This module is intentionally dependency-free. It validates the matrix that
connects model/runtime families to the probes already present in this repo.
It does not start servers, download models, import torch, or contact Docker.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


JSONDict = dict[str, Any]
REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "required_target_classes",
    "targets",
}
REQUIRED_TARGET_FIELDS = {
    "target_id",
    "target_class",
    "backend_family",
    "probe_tier",
    "model_family",
    "semantic_expert_ids",
    "local_only_status",
    "primary_probe",
    "observable_signals",
    "deferred_requirements",
    "next_live_commands",
}


def default_registry_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "model_target_registry.json"


def load_registry(path: Path) -> JSONDict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_registry(registry: JSONDict) -> list[str]:
    errors: list[str] = []
    missing_top = sorted(REQUIRED_TOP_LEVEL_FIELDS - set(registry))
    if missing_top:
        errors.append(f"missing top-level fields: {', '.join(missing_top)}")

    required_classes = registry.get("required_target_classes")
    if not isinstance(required_classes, list) or not required_classes:
        errors.append("required_target_classes must be a non-empty list")
        required_classes = []

    targets = registry.get("targets")
    if not isinstance(targets, list) or not targets:
        errors.append("targets must be a non-empty list")
        return errors

    seen_ids: set[str] = set()
    covered_classes: set[str] = set()
    for index, target in enumerate(targets):
        prefix = f"targets[{index}]"
        if not isinstance(target, dict):
            errors.append(f"{prefix} must be an object")
            continue

        missing_target = sorted(REQUIRED_TARGET_FIELDS - set(target))
        if missing_target:
            errors.append(f"{prefix} missing fields: {', '.join(missing_target)}")

        target_id = target.get("target_id")
        if not isinstance(target_id, str) or not target_id:
            errors.append(f"{prefix}.target_id must be a non-empty string")
        elif target_id in seen_ids:
            errors.append(f"duplicate target_id: {target_id}")
        else:
            seen_ids.add(target_id)

        target_class = target.get("target_class")
        if isinstance(target_class, str) and target_class:
            covered_classes.add(target_class)
        else:
            errors.append(f"{prefix}.target_class must be a non-empty string")

        for field in ("observable_signals", "deferred_requirements", "next_live_commands"):
            value = target.get(field)
            if not isinstance(value, list) or not value:
                errors.append(f"{prefix}.{field} must be a non-empty list")
            elif not all(isinstance(item, str) and item for item in value):
                errors.append(f"{prefix}.{field} must contain only non-empty strings")

    missing_classes = sorted(set(required_classes) - covered_classes)
    if missing_classes:
        errors.append(f"missing required target classes: {', '.join(missing_classes)}")

    return errors


def summarize_registry(registry: JSONDict) -> JSONDict:
    targets = registry.get("targets", [])
    by_class: dict[str, int] = {}
    by_backend: dict[str, int] = {}
    for target in targets:
        if not isinstance(target, dict):
            continue
        target_class = str(target.get("target_class", "unknown"))
        backend_family = str(target.get("backend_family", "unknown"))
        by_class[target_class] = by_class.get(target_class, 0) + 1
        by_backend[backend_family] = by_backend.get(backend_family, 0) + 1
    return {
        "schema_version": registry.get("schema_version"),
        "target_count": len(targets) if isinstance(targets, list) else 0,
        "by_class": dict(sorted(by_class.items())),
        "by_backend": dict(sorted(by_backend.items())),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-path", type=Path, default=default_registry_path())
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    registry = load_registry(args.registry_path)
    errors = validate_registry(registry)
    payload = summarize_registry(registry)
    payload["valid"] = not errors
    payload["errors"] = errors
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
