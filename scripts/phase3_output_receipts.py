#!/usr/bin/env python3
"""Shared Phase 3 saved-output capture receipt helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


JSONDict = dict[str, Any]

READY_TEXT_FIELDS = (
    "source_request_path",
    "source_prompt_set_path",
    "captured_at",
    "capture_host",
    "runtime_backend",
    "model_id",
    "prompt_family",
    "output_label",
)
READY_BOOLEAN_FIELDS = (
    "runtime_capture_approved",
    "runtime_prompt_traffic_approved",
)


def string_or_none(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def normalized_path_text(value: Any) -> str | None:
    text = string_or_none(value)
    return text.replace("\\", "/") if text is not None else None


def resolve_repo_path(value: Any, repo_root: Path) -> Path | None:
    text = normalized_path_text(value)
    if text is None:
        return None
    path = Path(text)
    return path if path.is_absolute() else repo_root / path


def resolved_path_text(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return resolved_path_text(path)


def receipt_path_matches(value: Any, expected_path: Path | None, repo_root: Path) -> bool:
    if expected_path is None:
        return True
    actual = normalized_path_text(value)
    if actual is None:
        return False
    if actual == display_path(expected_path, repo_root):
        return True
    actual_path = resolve_repo_path(actual, repo_root)
    return actual_path is not None and resolved_path_text(actual_path) == resolved_path_text(expected_path)


def receipt_path_exists(value: Any, repo_root: Path) -> bool:
    path = resolve_repo_path(value, repo_root)
    if path is None:
        return False
    try:
        return path.exists()
    except OSError:
        return False


def default_capture_receipt(
    *,
    prompt_set: JSONDict,
    source_prompt_set_path: str | None,
    output_label: str,
) -> JSONDict:
    return {
        "receipt_ready": False,
        "source_request_path": None,
        "source_prompt_set_path": source_prompt_set_path,
        "runtime_capture_approved": False,
        "runtime_prompt_traffic_approved": False,
        "captured_at": None,
        "capture_host": None,
        "runtime_backend": prompt_set.get("backend_family"),
        "model_id": prompt_set.get("model_id"),
        "prompt_family": prompt_set.get("prompt_family"),
        "output_label": output_label,
        "operator_notes": "fill after approved runtime capture",
    }


def validate_capture_receipt(receipt: Any, *, expected_label: str | None = None) -> JSONDict:
    errors: list[str] = []
    if not isinstance(receipt, dict):
        return {
            "shape_valid": False,
            "receipt_ready": False,
            "errors": ["capture_receipt must be an object"],
        }

    ready_value = receipt.get("receipt_ready")
    if not isinstance(ready_value, bool):
        errors.append("capture_receipt.receipt_ready must be a boolean")
        receipt_ready = False
    else:
        receipt_ready = ready_value

    output_label = receipt.get("output_label")
    if expected_label is not None and output_label != expected_label:
        errors.append(f"capture_receipt.output_label must be {expected_label!r}")

    if receipt_ready:
        for field in READY_TEXT_FIELDS:
            if string_or_none(receipt.get(field)) is None:
                errors.append(f"capture_receipt.{field} must be a non-empty string when receipt_ready is true")
        for field in READY_BOOLEAN_FIELDS:
            if receipt.get(field) is not True:
                errors.append(f"capture_receipt.{field} must be true when receipt_ready is true")

    return {
        "shape_valid": not errors,
        "receipt_ready": receipt_ready and not errors,
        "errors": errors,
    }


def validate_capture_receipt_source_binding(
    receipt: Any,
    *,
    repo_root: Path,
    expected_label: str | None = None,
    expected_source_request_path: Path | None = None,
    expected_source_prompt_set_path: Path | None = None,
    require_existing_sources: bool = True,
) -> JSONDict:
    validation = validate_capture_receipt(receipt, expected_label=expected_label)
    errors = list(validation["errors"])
    receipt_ready = validation["receipt_ready"] is True
    fields = {
        "source_request_path": expected_source_request_path,
        "source_prompt_set_path": expected_source_prompt_set_path,
    }
    exists: dict[str, bool] = {field: False for field in fields}
    matches: dict[str, bool] = {field: expected is None for field, expected in fields.items()}

    if isinstance(receipt, dict):
        for field, expected_path in fields.items():
            value_present = string_or_none(receipt.get(field)) is not None
            exists[field] = receipt_path_exists(receipt.get(field), repo_root)
            matches[field] = receipt_path_matches(receipt.get(field), expected_path, repo_root)
            if expected_path is not None and value_present and not matches[field]:
                errors.append(f"capture_receipt.{field} must match expected {field}")
            if receipt_ready and require_existing_sources and not exists[field]:
                errors.append(f"capture_receipt.{field} must point to an existing file when receipt_ready is true")

    return {
        "shape_valid": not errors,
        "receipt_ready": receipt_ready and not errors,
        "errors": errors,
        "source_request_path_exists": exists["source_request_path"],
        "source_prompt_set_path_exists": exists["source_prompt_set_path"],
        "source_request_path_matches": matches["source_request_path"],
        "source_prompt_set_path_matches": matches["source_prompt_set_path"],
    }