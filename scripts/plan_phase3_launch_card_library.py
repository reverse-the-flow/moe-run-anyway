#!/usr/bin/env python3
"""Summarize the Phase 3 runtime-capture launch-card library.

This planner gives Model Plane, operators, and agents a repo-level view of the
planned-only runtime-capture launch cards for every Phase 3 real-evidence
bundle. It reads local JSON metadata only. It does not launch runtimes, run
Docker, call endpoints, inspect secrets, mutate residency, or send prompt
traffic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import plan_phase3_handoff_coverage
import plan_phase3_runtime_capture_commands


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "memory-moe-mvp" / "phase3-real-evidence"
BUNDLE_GLOB = "*phase3_real_evidence_bundle.json"
SUPPORTED_SCHEMA_VERSION = "moe-phase3-launch-card-library-v1"
BINDING_WORKSHEET_SCHEMA_VERSION = "moe-phase3-launch-card-binding-worksheet-v1"
FILLED_LAUNCH_CARD_PACKAGE_SCHEMA_VERSION = "moe-phase3-filled-launch-card-package-v1"
MODEL_PLANE_ARTIFACT_WRITER_REQUEST_SCHEMA_VERSION = "moe-phase3-model-plane-artifact-writer-request-v1"
MODEL_PLANE_ARTIFACT_WRITER_FULFILLMENT_SCHEMA_VERSION = "moe-phase3-model-plane-artifact-writer-fulfillment-v1"
MODEL_PLANE_ARTIFACT_WRITER_FULFILLMENT_VALIDATION_SCHEMA_VERSION = "moe-phase3-model-plane-artifact-writer-fulfillment-validation-v1"
DEFAULT_BINDING_WORKSHEET_NAME = "phase3-launch-card-binding-worksheet.json"
DEFAULT_MODEL_PLANE_CONTRACT_REQUEST_NAME = "phase3-model-plane-artifact-writer-contract-request.json"
JSONDict = dict[str, Any]


def display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def int_count(value: Any) -> int:
    return value if isinstance(value, int) else 0


def list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item.strip()]


def load_model_plane_card_catalog(path: Path | None) -> list[JSONDict]:
    if path is None:
        return []
    payload = plan_phase3_runtime_capture_commands.load_json(path)
    cards = payload.get("cards") if isinstance(payload.get("cards"), list) else payload.get("items")
    if isinstance(cards, list):
        return [item for item in cards if isinstance(item, dict)]
    if payload.get("card_id"):
        return [payload]
    raise ValueError(f"{display_path(path) or path} must be a Model Plane card catalog object or card object")


def text_tokens(*values: Any) -> set[str]:
    text = " ".join(str(value or "").lower() for value in values)
    for char in "/\\_.:-()[]{}":
        text = text.replace(char, " ")
    return {
        token
        for token in text.split()
        if len(token) >= 3 and not token.startswith("sha256")
    }


def command_record_has_argv(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    argv = record.get("argv")
    return isinstance(argv, list) and any(isinstance(item, str) and item.strip() for item in argv)


def model_plane_backend_matches_task(card: JSONDict, task: JSONDict) -> bool:
    card_backend = card.get("backend_family")
    task_backend = task.get("backend_family")
    if card_backend == task_backend:
        return True
    # Ollama serves GGUF/llama.cpp-family models well enough for dense/full-runtime
    # output capture, while router-trace and managed-output readiness remain gated
    # by each artifact writer descriptor.
    if task_backend == "llama_cpp" and card_backend == "ollama_openai_compatible":
        return True
    return False

def model_plane_card_matches_task(card: JSONDict, task: JSONDict) -> bool:
    if not model_plane_backend_matches_task(card, task):
        return False
    task_tokens = text_tokens(task.get("request_name"), task.get("prompt_family"))
    card_tokens = text_tokens(
        card.get("card_id"),
        card.get("title"),
        card.get("model"),
        card.get("model_class"),
        card.get("profile_id"),
        card.get("label"),
    )
    return bool(task_tokens & card_tokens)


def model_plane_card_runtime_launch_available(card: JSONDict) -> bool:
    return (
        card.get("execution_mode") == "runner"
        and (
            command_record_has_argv(card.get("launch_command"))
            or command_record_has_argv(card.get("preflight_command"))
            or command_record_has_argv(card.get("smoke_command"))
        )
    )


def compact_model_plane_card(card: JSONDict) -> JSONDict:
    return {
        "card_id": card.get("card_id"),
        "title": card.get("title"),
        "model": card.get("model"),
        "model_class": card.get("model_class"),
        "profile_id": card.get("profile_id"),
        "backend_family": card.get("backend_family"),
        "card_type": card.get("card_type"),
        "evidence_level": card.get("evidence_level"),
        "probe_tier": card.get("probe_tier"),
        "execution_mode": card.get("execution_mode"),
        "runtime_launch_available": model_plane_card_runtime_launch_available(card),
        "has_launch_command": command_record_has_argv(card.get("launch_command")),
        "has_preflight_command": command_record_has_argv(card.get("preflight_command")),
        "has_smoke_command": command_record_has_argv(card.get("smoke_command")),
        "expected_artifacts": list_of_strings(card.get("expected_artifacts")),
        "limitations": list_of_strings(card.get("limitations")),
    }


def phase3_artifact_writer_contract_matches_task(writer: JSONDict, task: JSONDict) -> bool:
    expected_approval_keys = set(list_of_strings(task.get("approval_keys")))
    writer_approval_keys = set(list_of_strings(writer.get("approval_keys")))
    return (
        writer.get("artifact_id") == task.get("artifact_id")
        and writer.get("capture_kind") == task.get("capture_kind")
        and writer.get("receipt_kind") == task.get("receipt_kind")
        and writer.get("accepts_runtime_capture_request_path") is True
        and writer.get("accepts_prompt_set_path") is True
        and writer.get("writes_explicit_artifact_path") is True
        and writer.get("writes_explicit_receipt_path") is True
        and expected_approval_keys.issubset(writer_approval_keys)
    )


def phase3_artifact_writer_matches_task(writer: JSONDict, task: JSONDict) -> bool:
    return (
        phase3_artifact_writer_contract_matches_task(writer, task)
        and writer.get("runtime_execution_ready") is True
        and writer.get("writes_runtime_artifacts") is True
    )


def phase3_artifact_writer_contract_candidates(card: JSONDict, task: JSONDict) -> list[JSONDict]:
    writers = card.get("phase3_artifact_writers")
    if not isinstance(writers, list):
        return []
    return [
        writer
        for writer in writers
        if isinstance(writer, dict) and phase3_artifact_writer_contract_matches_task(writer, task)
    ]


def phase3_artifact_writer_candidates(card: JSONDict, task: JSONDict) -> list[JSONDict]:
    writers = card.get("phase3_artifact_writers")
    if not isinstance(writers, list):
        return []
    return [writer for writer in writers if isinstance(writer, dict) and phase3_artifact_writer_matches_task(writer, task)]


def model_plane_bridge_task(task: JSONDict, card_catalog: list[JSONDict]) -> JSONDict:
    matching_cards = [card for card in card_catalog if model_plane_card_matches_task(card, task)]
    runtime_candidates = [compact_model_plane_card(card) for card in matching_cards if model_plane_card_runtime_launch_available(card)]
    contract_cards = [card for card in matching_cards if phase3_artifact_writer_contract_candidates(card, task)]
    contract_candidates = [compact_model_plane_card(card) for card in contract_cards]
    writer_cards = [card for card in matching_cards if phase3_artifact_writer_candidates(card, task)]
    writer_candidates = [compact_model_plane_card(card) for card in writer_cards]
    limitations = [
        limitation
        for card in runtime_candidates
        for limitation in list_of_strings(card.get("limitations"))
    ]
    blockers: list[str] = []
    if not runtime_candidates:
        blockers.append("model_plane_runtime_card_missing")
    if not contract_candidates:
        blockers.append("model_plane_phase3_artifact_writer_missing")
    elif not writer_candidates:
        blockers.append("model_plane_phase3_artifact_writer_runtime_execution_pending")
    if runtime_candidates and not contract_candidates:
        blockers.append("model_plane_runtime_card_not_artifact_writer")
    if task.get("artifact_id") == "candidate_router_trace" and any("semantic expert" in item.lower() and "not" in item.lower() for item in limitations):
        blockers.append("model_plane_candidate_card_runtime_evidence_only")
    return {
        "request_name": task.get("request_name"),
        "request_path": task.get("request_path"),
        "launch_card_path": task.get("launch_card_path"),
        "artifact_id": task.get("artifact_id"),
        "capture_kind": task.get("capture_kind"),
        "receipt_kind": task.get("receipt_kind"),
        "approval_keys": task.get("approval_keys", []),
        "runtime_launch_candidate_count": len(runtime_candidates),
        "runtime_launch_available": bool(runtime_candidates),
        "artifact_writer_contract_candidate_count": len(contract_candidates),
        "artifact_writer_contract_available": bool(contract_candidates),
        "artifact_writer_candidate_count": len(writer_candidates),
        "artifact_writer_ready": bool(writer_candidates),
        "runtime_candidates": runtime_candidates,
        "artifact_writer_contract_candidates": contract_candidates,
        "artifact_writer_candidates": writer_candidates,
        "blockers": blockers,
    }


def build_model_plane_bridge_summary(cards: list[JSONDict], card_catalog: list[JSONDict], *, provided: bool) -> JSONDict:
    tasks = [
        task
        for card in cards
        for task in card.get("binding_tasks", [])
        if isinstance(task, dict)
    ]
    task_results = [model_plane_bridge_task(task, card_catalog) for task in tasks] if provided else []
    runtime_launch_candidate_task_count = sum(1 for item in task_results if item.get("runtime_launch_available") is True)
    artifact_writer_contract_count = sum(1 for item in task_results if item.get("artifact_writer_contract_available") is True)
    artifact_writer_ready_count = sum(1 for item in task_results if item.get("artifact_writer_ready") is True)
    unique_runtime_card_ids = sorted({
        str(card.get("card_id"))
        for item in task_results
        for card in item.get("runtime_candidates", [])
        if card.get("card_id")
    })
    blockers = sorted({
        blocker
        for item in task_results
        for blocker in list_of_strings(item.get("blockers"))
    })
    if not provided:
        blockers = ["model_plane_card_catalog_not_provided"]
    elif not card_catalog:
        blockers.append("model_plane_card_catalog_empty")
    bridge_ready = provided and bool(tasks) and artifact_writer_ready_count == len(tasks) and not blockers
    return {
        "provided": provided,
        "card_catalog_count": len(card_catalog),
        "task_count": len(tasks),
        "runtime_launch_candidate_task_count": runtime_launch_candidate_task_count,
        "runtime_launch_candidate_card_count": len(unique_runtime_card_ids),
        "runtime_launch_candidate_card_ids": unique_runtime_card_ids,
        "artifact_writer_contract_count": artifact_writer_contract_count,
        "missing_artifact_writer_contract_count": max(len(tasks) - artifact_writer_contract_count, 0),
        "artifact_writer_ready_count": artifact_writer_ready_count,
        "missing_artifact_writer_count": max(len(tasks) - artifact_writer_ready_count, 0),
        "bridge_ready": bridge_ready,
        "task_results": task_results,
        "blockers": blockers,
        "next_actions": [
            "Use Model Plane runtime cards as launch context only until a Phase 3 artifact-writer callable contract exists.",
            "Promote artifact-writer contracts to ready only after they execute and write exact artifact and receipt paths.",
            "Do not mark launch-card tasks command_ready from runtime/probe cards unless they write the exact Phase 3 artifact and receipt paths.",
        ],
    }


def load_launch_card(path: Path) -> JSONDict:
    if not path.exists():
        return {}
    return plan_phase3_runtime_capture_commands.load_json(path)


def resolve_repo_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def string_value(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def binding_payload_for_task(task: JSONDict) -> JSONDict:
    binding = task.get("binding")
    if isinstance(binding, dict):
        return binding
    template = task.get("binding_template")
    return template if isinstance(template, dict) else {}


def launch_command_for_binding(binding: JSONDict) -> list[str]:
    command = binding.get("launch_command")
    return list_of_strings(command)


def binding_has_command_option(binding: JSONDict) -> bool:
    return (
        string_value(binding.get("model_plane_callable_id")) is not None
        or plan_phase3_runtime_capture_commands.launch_command_ready(launch_command_for_binding(binding))
    )


def binding_command_ready(binding: JSONDict) -> bool:
    return binding_has_command_option(binding) and binding.get("command_ready") is True


def canonical_json(payload: JSONDict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def default_binding_worksheet_path(root: Path = DEFAULT_ROOT) -> Path:
    return root / DEFAULT_BINDING_WORKSHEET_NAME


def default_model_plane_contract_request_path(root: Path = DEFAULT_ROOT) -> Path:
    return root / DEFAULT_MODEL_PLANE_CONTRACT_REQUEST_NAME


def build_validation_command(request_path: Path, launch_card_path: Path) -> list[str]:
    return [
        "uv",
        "run",
        "--managed-python",
        "--python",
        "3.13",
        "scripts/plan_phase3_runtime_capture_commands.py",
        display_path(request_path) or str(request_path),
        "--launch-card",
        display_path(launch_card_path) or str(launch_card_path),
        "--json",
    ]


def build_binding_handoff_task(
    *,
    launch_card: JSONDict,
    request_path: Path,
    launch_card_path: Path,
    task: JSONDict,
    validation_command: list[str],
) -> JSONDict:
    binding = binding_payload_for_task(task)
    approval_keys = list_of_strings(task.get("approval_keys"))
    binding_approval_keys = set(list_of_strings(binding.get("approval_keys")))
    missing_fields: list[str] = []
    required_fields = {
        "task_id": task.get("task_id"),
        "artifact_id": task.get("artifact_id"),
        "capture_kind": task.get("capture_kind"),
        "receipt_kind": task.get("receipt_kind"),
        "artifact_output_path": task.get("artifact_output_path"),
        "receipt_output_path": task.get("receipt_output_path"),
        "prompt_set_path": task.get("prompt_set_path"),
    }
    for field, value in required_fields.items():
        if string_value(value) is None:
            missing_fields.append(field)
    if not approval_keys:
        missing_fields.append("approval_keys")
    if binding.get("binding_kind") != "model_plane_callable_or_launch_command":
        missing_fields.append("binding_template.binding_kind")
    if binding.get("requires_explicit_user_approval") is not True:
        missing_fields.append("binding_template.requires_explicit_user_approval")
    if binding.get("may_send_prompt_traffic") is not True:
        missing_fields.append("binding_template.may_send_prompt_traffic")
    if binding.get("runtime_capture_request_path") != display_path(request_path):
        missing_fields.append("binding_template.runtime_capture_request_path")
    if binding.get("reads_prompt_set_path") != task.get("prompt_set_path"):
        missing_fields.append("binding_template.reads_prompt_set_path")
    if not set(approval_keys).issubset(binding_approval_keys):
        missing_fields.append("binding_template.approval_keys")
    if binding.get("writes_artifact_path") != task.get("artifact_output_path"):
        missing_fields.append("binding_template.writes_artifact_path")
    if binding.get("writes_receipt_path") != task.get("receipt_output_path"):
        missing_fields.append("binding_template.writes_receipt_path")

    command_option_ready = binding_has_command_option(binding)
    command_ready = binding_command_ready(binding)
    return {
        "request_name": launch_card.get("request_name"),
        "request_path": display_path(request_path),
        "launch_card_path": display_path(launch_card_path),
        "model_id": launch_card.get("model_id"),
        "backend_family": launch_card.get("backend_family"),
        "prompt_family": launch_card.get("prompt_family"),
        "task_id": task.get("task_id"),
        "artifact_id": task.get("artifact_id"),
        "capture_kind": task.get("capture_kind"),
        "receipt_kind": task.get("receipt_kind"),
        "artifact_output_path": task.get("artifact_output_path"),
        "receipt_output_path": task.get("receipt_output_path"),
        "prompt_set_path": task.get("prompt_set_path"),
        "approval_keys": approval_keys,
        "requires_explicit_user_approval": task.get("requires_explicit_user_approval") is True,
        "requires_prompt_traffic": task.get("requires_prompt_traffic") is True,
        "requires_runtime": task.get("requires_runtime") is True,
        "validator_command_count": int_count(task.get("validator_command_count")),
        "binding_handoff_ready": not missing_fields,
        "binding_ready": task.get("command_binding_ready") is True,
        "command_option_ready": command_option_ready,
        "command_ready": command_ready,
        "model_plane_callable_id": string_value(binding.get("model_plane_callable_id")),
        "launch_command": launch_command_for_binding(binding),
        "missing_fields": missing_fields,
        "binding_template": binding,
        "fill_instruction": "Fill either binding.model_plane_callable_id or binding.launch_command in the launch card task; keep approval keys and exact output paths unchanged.",
        "validation_command": validation_command,
    }


def binding_task_key(task: JSONDict) -> str:
    return "::".join(
        str(task.get(field) or "")
        for field in ("request_path", "launch_card_path", "artifact_id")
    )


def build_binding_worksheet(summary: JSONDict) -> JSONDict:
    rows: list[JSONDict] = []
    for task in summary.get("binding_tasks", []):
        if not isinstance(task, dict):
            continue
        template = task.get("binding_template") if isinstance(task.get("binding_template"), dict) else {}
        rows.append(
            {
                "request_name": task.get("request_name"),
                "request_path": task.get("request_path"),
                "launch_card_path": task.get("launch_card_path"),
                "model_id": task.get("model_id"),
                "backend_family": task.get("backend_family"),
                "prompt_family": task.get("prompt_family"),
                "task_id": task.get("task_id"),
                "artifact_id": task.get("artifact_id"),
                "capture_kind": task.get("capture_kind"),
                "receipt_kind": task.get("receipt_kind"),
                "artifact_output_path": task.get("artifact_output_path"),
                "receipt_output_path": task.get("receipt_output_path"),
                "prompt_set_path": task.get("prompt_set_path"),
                "approval_keys": task.get("approval_keys", []),
                "requires_explicit_user_approval": task.get("requires_explicit_user_approval") is True,
                "requires_prompt_traffic": task.get("requires_prompt_traffic") is True,
                "requires_runtime": task.get("requires_runtime") is True,
                "fill_instruction": task.get("fill_instruction"),
                "validation_command": task.get("validation_command", []),
                "binding": {
                    "binding_kind": template.get("binding_kind") or "model_plane_callable_or_launch_command",
                    "model_plane_callable_id": task.get("model_plane_callable_id") or "",
                    "launch_command": task.get("launch_command", []),
                    "command_ready": task.get("command_ready") is True,
                    "requires_explicit_user_approval": True,
                    "may_send_prompt_traffic": task.get("requires_prompt_traffic") is True,
                    "runtime_capture_request_path": task.get("request_path"),
                    "reads_prompt_set_path": task.get("prompt_set_path"),
                    "approval_keys": task.get("approval_keys", []),
                    "writes_artifact_path": task.get("artifact_output_path"),
                    "writes_receipt_path": task.get("receipt_output_path"),
                },
            }
        )
    command_option_count = sum(1 for row in rows if binding_has_command_option(row.get("binding", {})))
    command_ready_count = sum(1 for row in rows if binding_command_ready(row.get("binding", {})))
    return {
        "schema_version": BINDING_WORKSHEET_SCHEMA_VERSION,
        "mode": "phase3_launch_card_binding_worksheet",
        "valid": True,
        "worksheet_ready": summary.get("binding_handoff_ready") is True,
        "execution_ready": False,
        "root": summary.get("root"),
        "card_count": summary.get("card_count"),
        "task_count": len(rows),
        "command_option_count": command_option_count,
        "command_ready_count": command_ready_count,
        "unbound_task_count": len(rows) - command_ready_count,
        "tasks": rows,
        "next_actions": [
            "Fill binding.model_plane_callable_id or binding.launch_command for every task after explicit approval.",
            "Keep artifact_output_path, receipt_output_path, prompt_set_path, and approval_keys unchanged.",
            "Run this planner with --binding-worksheet and --output-filled-card-dir to validate filled cards offline.",
        ],
        "safety_contract": [
            "binding worksheet is metadata only",
            "binding worksheet validation does not launch model servers",
            "binding worksheet validation does not run Docker",
            "binding worksheet validation does not call endpoints",
            "binding worksheet validation does not inspect private tokens",
            "binding worksheet validation does not send prompt traffic",
            "binding worksheet validation does not mutate runtime residency",
            "binding worksheet validation does not claim live expert paging",
        ],
    }



def model_plane_artifact_writer_contract_for_task(task: JSONDict) -> JSONDict:
    return {
        "artifact_id": task.get("artifact_id"),
        "capture_kind": task.get("capture_kind"),
        "receipt_kind": task.get("receipt_kind"),
        "approval_keys": list_of_strings(task.get("approval_keys")),
        "accepts_runtime_capture_request_path": True,
        "accepts_prompt_set_path": True,
        "writes_explicit_artifact_path": True,
        "writes_explicit_receipt_path": True,
        "runtime_execution_ready": False,
        "writes_runtime_artifacts": False,
    }


def build_model_plane_artifact_writer_contract_request(summary: JSONDict) -> JSONDict:
    tasks = [task for task in summary.get("binding_tasks", []) if isinstance(task, dict)]
    rows: list[JSONDict] = []
    for task in tasks:
        rows.append(
            {
                "request_name": task.get("request_name"),
                "request_path": task.get("request_path"),
                "launch_card_path": task.get("launch_card_path"),
                "model_id": task.get("model_id"),
                "backend_family": task.get("backend_family"),
                "prompt_family": task.get("prompt_family"),
                "task_id": task.get("task_id"),
                "artifact_id": task.get("artifact_id"),
                "capture_kind": task.get("capture_kind"),
                "receipt_kind": task.get("receipt_kind"),
                "artifact_output_path": task.get("artifact_output_path"),
                "receipt_output_path": task.get("receipt_output_path"),
                "prompt_set_path": task.get("prompt_set_path"),
                "approval_keys": list_of_strings(task.get("approval_keys")),
                "requires_explicit_user_approval": task.get("requires_explicit_user_approval") is True,
                "requires_prompt_traffic": task.get("requires_prompt_traffic") is True,
                "requires_runtime": task.get("requires_runtime") is True,
                "required_phase3_artifact_writer": model_plane_artifact_writer_contract_for_task(task),
                "runtime_binding_template": task.get("binding_template") if isinstance(task.get("binding_template"), dict) else {},
                "model_plane_card_catalog_hint": {
                    "put_under": "card.phase3_artifact_writers[]",
                    "contract_only_until_runtime_execution_ready": True,
                    "must_write_exact_artifact_path": task.get("artifact_output_path"),
                    "must_write_exact_receipt_path": task.get("receipt_output_path"),
                    "must_accept_runtime_capture_request_path": task.get("request_path"),
                    "must_accept_prompt_set_path": task.get("prompt_set_path"),
                },
            }
        )
    return {
        "schema_version": MODEL_PLANE_ARTIFACT_WRITER_REQUEST_SCHEMA_VERSION,
        "mode": "phase3_model_plane_artifact_writer_contract_request",
        "valid": summary.get("valid") is True and summary.get("binding_handoff_ready") is True,
        "request_ready": summary.get("binding_handoff_ready") is True,
        "execution_ready": False,
        "root": summary.get("root"),
        "card_count": summary.get("card_count"),
        "task_count": len(rows),
        "phase3_artifact_writer_contract_count": len(rows),
        "phase3_artifact_writer_ready_count": 0,
        "tasks": rows,
        "next_actions": [
            "Create or update Model Plane cards so each requested writer appears under card.phase3_artifact_writers[].",
            "Keep runtime_execution_ready false until the callable actually writes the exact artifact and receipt paths after approval.",
            "Rerun scripts/plan_phase3_launch_card_library.py --model-plane-card-catalog with the exported Model Plane catalog before marking bridge readiness.",
            "Validate returned Model Plane bindings with --model-plane-contract-fulfillment before writing filled launch cards.",
        ],
        "safety_contract": [
            "artifact-writer contract request is metadata only",
            "artifact-writer contract request does not launch model servers",
            "artifact-writer contract request does not run Docker",
            "artifact-writer contract request does not call endpoints",
            "artifact-writer contract request does not inspect private tokens",
            "artifact-writer contract request does not send prompt traffic",
            "artifact-writer contract request does not mutate runtime residency",
            "artifact-writer contract request does not claim live expert paging",
        ],
    }

def model_plane_fulfillment_task_rows(fulfillment: JSONDict) -> list[JSONDict]:
    tasks = fulfillment.get("tasks")
    if not isinstance(tasks, list):
        tasks = fulfillment.get("artifact_writers")
    if not isinstance(tasks, list):
        return []
    return [task for task in tasks if isinstance(task, dict)]


def model_plane_fulfillment_binding_payload(row: JSONDict, expected: JSONDict) -> JSONDict:
    binding = row.get("binding") if isinstance(row.get("binding"), dict) else {}
    runtime_binding = row.get("runtime_binding") if isinstance(row.get("runtime_binding"), dict) else {}
    if runtime_binding:
        binding = {**runtime_binding, **binding}
    launch_command = binding.get("launch_command") if "launch_command" in binding else row.get("launch_command", [])
    return {
        "binding_kind": binding.get("binding_kind") or row.get("binding_kind") or "model_plane_callable_or_launch_command",
        "model_plane_callable_id": string_value(binding.get("model_plane_callable_id")) or string_value(row.get("model_plane_callable_id")) or string_value(row.get("callable_id")) or "",
        "launch_command": list_of_strings(launch_command),
        "command_ready": binding.get("command_ready") is True or row.get("command_ready") is True,
        "requires_explicit_user_approval": binding.get("requires_explicit_user_approval") is True or row.get("requires_explicit_user_approval") is True,
        "may_send_prompt_traffic": binding.get("may_send_prompt_traffic") is True or row.get("may_send_prompt_traffic") is True,
        "runtime_capture_request_path": binding.get("runtime_capture_request_path") or row.get("runtime_capture_request_path") or row.get("request_path"),
        "reads_prompt_set_path": binding.get("reads_prompt_set_path") or row.get("reads_prompt_set_path") or row.get("prompt_set_path"),
        "approval_keys": list_of_strings(binding.get("approval_keys")) or list_of_strings(row.get("approval_keys")) or list_of_strings(expected.get("approval_keys")),
        "writes_artifact_path": binding.get("writes_artifact_path") or row.get("writes_artifact_path") or row.get("artifact_output_path"),
        "writes_receipt_path": binding.get("writes_receipt_path") or row.get("writes_receipt_path") or row.get("receipt_output_path"),
    }


def build_binding_worksheet_from_model_plane_fulfillment(summary: JSONDict, fulfillment: JSONDict) -> JSONDict:
    worksheet = build_binding_worksheet(summary)
    rows = worksheet.get("tasks") if isinstance(worksheet.get("tasks"), list) else []
    worksheet_rows_by_key = {binding_task_key(row): row for row in rows if isinstance(row, dict)}
    expected_tasks = [task for task in summary.get("binding_tasks", []) if isinstance(task, dict)]
    expected_by_key = {binding_task_key(task): task for task in expected_tasks}
    errors: list[str] = []
    if fulfillment.get("schema_version") != MODEL_PLANE_ARTIFACT_WRITER_FULFILLMENT_SCHEMA_VERSION:
        errors.append("model_plane_fulfillment_schema_version_mismatch")
    request_schema_version = fulfillment.get("request_schema_version") or fulfillment.get("contract_request_schema_version")
    if request_schema_version != MODEL_PLANE_ARTIFACT_WRITER_REQUEST_SCHEMA_VERSION:
        errors.append("model_plane_fulfillment_request_schema_version_mismatch")
    task_rows = model_plane_fulfillment_task_rows(fulfillment)
    if not task_rows:
        errors.append("model_plane_fulfillment_tasks_missing")

    seen: set[str] = set()
    matched_keys: set[str] = set()
    task_results: list[JSONDict] = []
    identity_fields = (
        "task_id",
        "artifact_id",
        "capture_kind",
        "receipt_kind",
        "artifact_output_path",
        "receipt_output_path",
        "prompt_set_path",
    )
    for index, row in enumerate(task_rows):
        key = binding_task_key(row)
        if key in seen:
            errors.append(f"{key}: model_plane_fulfillment_task_duplicate")
            continue
        seen.add(key)
        expected = expected_by_key.get(key)
        item_errors: list[str] = []
        if expected is None:
            item_errors.append("model_plane_fulfillment_task_unexpected")
        else:
            matched_keys.add(key)
            for field in identity_fields:
                if row.get(field) != expected.get(field):
                    item_errors.append(f"{field}_mismatch")
            expected_approval_keys = set(list_of_strings(expected.get("approval_keys")))
            row_approval_keys = set(list_of_strings(row.get("approval_keys")))
            if not expected_approval_keys.issubset(row_approval_keys):
                item_errors.append("approval_keys_mismatch")
            worksheet_row = worksheet_rows_by_key.get(key)
            if isinstance(worksheet_row, dict):
                worksheet_row["binding"] = model_plane_fulfillment_binding_payload(row, expected)
        if key == "::":
            item_errors.append(f"row_{index}: model_plane_fulfillment_task_key_missing")
        errors.extend(f"{key or 'unknown'}: {error}" for error in item_errors)
        task_results.append(
            {
                "key": key,
                "request_name": row.get("request_name"),
                "request_path": row.get("request_path"),
                "launch_card_path": row.get("launch_card_path"),
                "artifact_id": row.get("artifact_id"),
                "matched_expected_task": expected is not None,
                "errors": item_errors,
            }
        )

    missing_keys = sorted(set(expected_by_key) - matched_keys)
    errors.extend(f"{key}: model_plane_fulfillment_task_missing" for key in missing_keys)
    worksheet_validation = validate_binding_worksheet(summary, worksheet)
    fulfillment_ready = not errors and len(matched_keys) == len(expected_by_key)
    binding_ready = fulfillment_ready and worksheet_validation.get("binding_ready") is True
    return {
        "schema_version": MODEL_PLANE_ARTIFACT_WRITER_FULFILLMENT_VALIDATION_SCHEMA_VERSION,
        "valid": not errors and worksheet_validation.get("valid") is True,
        "fulfillment_ready": fulfillment_ready,
        "binding_ready": binding_ready,
        "expected_task_count": len(expected_by_key),
        "fulfillment_task_count": len(task_rows),
        "matched_task_count": len(matched_keys),
        "missing_task_count": len(missing_keys),
        "command_option_count": worksheet_validation.get("command_option_count"),
        "command_ready_count": worksheet_validation.get("command_ready_count"),
        "missing_runtime_command_count": worksheet_validation.get("missing_runtime_command_count"),
        "missing_task_keys": missing_keys,
        "errors": errors + list_of_strings(worksheet_validation.get("errors")),
        "blockers": sorted(set(list_of_strings(worksheet_validation.get("blockers")) + (["model_plane_fulfillment_invalid"] if errors else []))),
        "task_results": task_results,
        "worksheet_validation": worksheet_validation,
        "binding_worksheet": worksheet,
        "safety_contract": [
            "artifact-writer fulfillment validation reads local metadata only",
            "artifact-writer fulfillment validation does not launch model servers",
            "artifact-writer fulfillment validation does not run Docker",
            "artifact-writer fulfillment validation does not call endpoints",
            "artifact-writer fulfillment validation does not inspect private tokens",
            "artifact-writer fulfillment validation does not send prompt traffic",
            "artifact-writer fulfillment validation does not mutate runtime residency",
            "artifact-writer fulfillment validation does not claim live expert paging",
        ],
    }


def compact_model_plane_artifact_writer_fulfillment_validation(validation: JSONDict) -> JSONDict:
    return {
        "valid": validation.get("valid"),
        "fulfillment_ready": validation.get("fulfillment_ready"),
        "binding_ready": validation.get("binding_ready"),
        "expected_task_count": validation.get("expected_task_count"),
        "fulfillment_task_count": validation.get("fulfillment_task_count"),
        "matched_task_count": validation.get("matched_task_count"),
        "missing_task_count": validation.get("missing_task_count"),
        "command_option_count": validation.get("command_option_count"),
        "command_ready_count": validation.get("command_ready_count"),
        "missing_runtime_command_count": validation.get("missing_runtime_command_count"),
        "errors": validation.get("errors", []),
        "blockers": validation.get("blockers", []),
    }


def build_filled_launch_card_package_from_model_plane_fulfillment(summary: JSONDict, fulfillment: JSONDict) -> JSONDict:
    fulfillment_validation = build_binding_worksheet_from_model_plane_fulfillment(summary, fulfillment)
    package = build_filled_launch_card_package(summary, fulfillment_validation["binding_worksheet"])
    package["model_plane_artifact_writer_fulfillment_validation"] = compact_model_plane_artifact_writer_fulfillment_validation(fulfillment_validation)
    if fulfillment_validation.get("valid") is not True:
        package["valid"] = False
        package["binding_ready"] = False
        package["errors"] = list_of_strings(fulfillment_validation.get("errors")) + list_of_strings(package.get("errors"))
        package["blockers"] = sorted(set(list_of_strings(package.get("blockers")) + ["model_plane_fulfillment_invalid"]))
    return package

def worksheet_binding_payload(row: JSONDict) -> JSONDict:
    binding = row.get("binding")
    if isinstance(binding, dict):
        return binding
    return {
        "binding_kind": "model_plane_callable_or_launch_command",
        "model_plane_callable_id": row.get("model_plane_callable_id") or "",
        "launch_command": row.get("launch_command", []),
        "command_ready": False,
        "requires_explicit_user_approval": row.get("requires_explicit_user_approval") is True,
        "may_send_prompt_traffic": row.get("requires_prompt_traffic") is True,
        "runtime_capture_request_path": row.get("request_path"),
        "reads_prompt_set_path": row.get("prompt_set_path"),
        "approval_keys": row.get("approval_keys", []),
        "writes_artifact_path": row.get("artifact_output_path"),
        "writes_receipt_path": row.get("receipt_output_path"),
    }


def validate_binding_worksheet(summary: JSONDict, worksheet: JSONDict) -> JSONDict:
    errors: list[str] = []
    if worksheet.get("schema_version") != BINDING_WORKSHEET_SCHEMA_VERSION:
        errors.append("binding_worksheet_schema_version_mismatch")
    rows_raw = worksheet.get("tasks")
    if not isinstance(rows_raw, list):
        errors.append("binding_worksheet_tasks_missing")
        rows_raw = []

    expected_tasks = [task for task in summary.get("binding_tasks", []) if isinstance(task, dict)]
    expected_by_key = {binding_task_key(task): task for task in expected_tasks}
    seen: set[str] = set()
    row_by_key: dict[str, JSONDict] = {}
    task_results: list[JSONDict] = []
    for index, raw_row in enumerate(rows_raw):
        if not isinstance(raw_row, dict):
            errors.append(f"row_{index}: binding_worksheet_task_must_be_object")
            continue
        key = binding_task_key(raw_row)
        if key in seen:
            errors.append(f"{key}: binding_worksheet_task_duplicate")
            continue
        seen.add(key)
        row_by_key[key] = raw_row
        expected = expected_by_key.get(key)
        item_errors: list[str] = []
        blockers: list[str] = []
        if expected is None:
            item_errors.append("binding_worksheet_task_unexpected")
        else:
            for field in ("artifact_output_path", "receipt_output_path", "prompt_set_path"):
                if raw_row.get(field) != expected.get(field):
                    item_errors.append(f"{field}_mismatch")
            expected_approval_keys = set(list_of_strings(expected.get("approval_keys")))
            row_approval_keys = set(list_of_strings(raw_row.get("approval_keys")))
            if not expected_approval_keys.issubset(row_approval_keys):
                item_errors.append("approval_keys_mismatch")
        binding = worksheet_binding_payload(raw_row)
        launch_command = launch_command_for_binding(binding)
        command_option_ready = binding_has_command_option(binding)
        command_ready = binding_command_ready(binding)
        if not command_option_ready:
            blockers.append("runtime_command_binding_missing")
        elif binding.get("command_ready") is not True:
            blockers.append("runtime_command_ready_flag_missing")
        if binding.get("requires_explicit_user_approval") is not True:
            blockers.append("explicit_user_approval_gate_missing")
        if binding.get("may_send_prompt_traffic") is not True:
            blockers.append("prompt_traffic_ack_missing")
        if expected is not None:
            if binding.get("runtime_capture_request_path") != expected.get("request_path"):
                blockers.append("runtime_capture_request_path_missing")
            if binding.get("reads_prompt_set_path") != expected.get("prompt_set_path"):
                blockers.append("reads_prompt_set_path_missing")
            expected_approval_keys = set(list_of_strings(expected.get("approval_keys")))
            binding_approval_keys = set(list_of_strings(binding.get("approval_keys")))
            if not expected_approval_keys.issubset(binding_approval_keys):
                blockers.append("approval_keys_missing")
            if binding.get("writes_artifact_path") != expected.get("artifact_output_path"):
                blockers.append("writes_artifact_path_missing")
            if binding.get("writes_receipt_path") != expected.get("receipt_output_path"):
                blockers.append("writes_receipt_path_missing")
        errors.extend(f"{key}: {error}" for error in item_errors)
        task_results.append(
            {
                "key": key,
                "request_name": raw_row.get("request_name"),
                "request_path": raw_row.get("request_path"),
                "launch_card_path": raw_row.get("launch_card_path"),
                "artifact_id": raw_row.get("artifact_id"),
                "command_option_ready": command_option_ready,
                "command_ready": command_ready,
                "model_plane_callable_id": string_value(binding.get("model_plane_callable_id")),
                "launch_command": launch_command,
                "errors": item_errors,
                "blockers": blockers,
            }
        )

    missing_keys = sorted(set(expected_by_key) - set(row_by_key))
    errors.extend(f"{key}: binding_worksheet_task_missing" for key in missing_keys)
    matched_task_count = len(expected_by_key) - len(missing_keys)
    command_option_count = sum(1 for item in task_results if item.get("command_option_ready") is True)
    command_ready_count = sum(1 for item in task_results if item.get("command_ready") is True)
    blockers = sorted({blocker for item in task_results for blocker in item.get("blockers", [])})
    if missing_keys:
        blockers.append("binding_worksheet_tasks_missing")
    worksheet_ready = not errors and matched_task_count == len(expected_by_key)
    binding_ready = worksheet_ready and command_ready_count == len(expected_by_key) and not blockers
    return {
        "schema_version": BINDING_WORKSHEET_SCHEMA_VERSION,
        "valid": not errors,
        "worksheet_ready": worksheet_ready,
        "binding_ready": binding_ready,
        "expected_task_count": len(expected_by_key),
        "worksheet_task_count": len(rows_raw),
        "matched_task_count": matched_task_count,
        "command_option_count": command_option_count,
        "command_ready_count": command_ready_count,
        "command_option_without_ready_count": max(command_option_count - command_ready_count, 0),
        "missing_runtime_command_count": max(len(expected_by_key) - command_ready_count, 0),
        "missing_task_count": len(missing_keys),
        "missing_task_keys": missing_keys,
        "errors": errors,
        "blockers": blockers,
        "task_bindings": task_results,
    }


def filled_binding_for_task(card_task: JSONDict, row: JSONDict) -> JSONDict:
    template = card_task.get("binding_template") if isinstance(card_task.get("binding_template"), dict) else {}
    binding = dict(template)
    worksheet_binding = worksheet_binding_payload(row)
    binding["binding_kind"] = worksheet_binding.get("binding_kind") or "model_plane_callable_or_launch_command"
    binding["model_plane_callable_id"] = string_value(worksheet_binding.get("model_plane_callable_id")) or ""
    binding["launch_command"] = launch_command_for_binding(worksheet_binding)
    binding["command_ready"] = binding_has_command_option(binding) and worksheet_binding.get("command_ready") is True
    binding["requires_explicit_user_approval"] = worksheet_binding.get("requires_explicit_user_approval") is True
    binding["may_send_prompt_traffic"] = worksheet_binding.get("may_send_prompt_traffic") is True
    binding["runtime_capture_request_path"] = worksheet_binding.get("runtime_capture_request_path")
    binding["reads_prompt_set_path"] = worksheet_binding.get("reads_prompt_set_path")
    binding["approval_keys"] = list_of_strings(worksheet_binding.get("approval_keys"))
    binding["writes_artifact_path"] = worksheet_binding.get("writes_artifact_path")
    binding["writes_receipt_path"] = worksheet_binding.get("writes_receipt_path")
    return binding


def build_filled_launch_card_package(summary: JSONDict, worksheet: JSONDict) -> JSONDict:
    worksheet_validation = validate_binding_worksheet(summary, worksheet)
    worksheet_tasks = worksheet.get("tasks") if isinstance(worksheet.get("tasks"), list) else []
    rows_by_key = {
        binding_task_key(row): row
        for row in worksheet_tasks
        if isinstance(row, dict)
    }
    cards: list[JSONDict] = []
    package_errors: list[str] = []
    for card_entry in summary.get("cards", []):
        if not isinstance(card_entry, dict):
            continue
        launch_card_path = resolve_repo_path(card_entry.get("launch_card_path"))
        request_path = resolve_repo_path(card_entry.get("request_path"))
        if launch_card_path is None or request_path is None:
            package_errors.append(f"{card_entry.get('request_name')}: card_or_request_path_missing")
            continue
        try:
            launch_card = load_launch_card(launch_card_path)
            filled_card = json.loads(json.dumps(launch_card))
            for card_task in filled_card.get("tasks", []):
                if not isinstance(card_task, dict):
                    continue
                key = "::".join(
                    str(value or "")
                    for value in (
                        card_entry.get("request_path"),
                        card_entry.get("launch_card_path"),
                        card_task.get("artifact_id"),
                    )
                )
                row = rows_by_key.get(key)
                if isinstance(row, dict):
                    card_task["binding"] = filled_binding_for_task(card_task, row)
            command_summary = plan_phase3_runtime_capture_commands.build_summary(request_path)
            binding_summary = plan_phase3_runtime_capture_commands.validate_launch_card_bindings(
                command_summary,
                filled_card,
                launch_card_path=launch_card_path,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            package_errors.append(f"{card_entry.get('request_name')}: {exc}")
            continue
        cards.append(
            {
                "request_name": card_entry.get("request_name"),
                "request_path": card_entry.get("request_path"),
                "launch_card_path": card_entry.get("launch_card_path"),
                "binding_ready": binding_summary.get("binding_ready") is True,
                "valid": binding_summary.get("valid") is True,
                "task_count": binding_summary.get("task_count"),
                "bound_task_count": binding_summary.get("bound_task_count"),
                "command_option_count": binding_summary.get("command_option_count"),
                "command_ready_count": binding_summary.get("command_ready_count"),
                "missing_runtime_command_count": binding_summary.get("missing_runtime_command_count"),
                "missing_runtime_command_artifact_ids": binding_summary.get("missing_runtime_command_artifact_ids", []),
                "errors": binding_summary.get("errors", []),
                "blockers": binding_summary.get("blockers", []),
                "launch_card": filled_card,
            }
        )
    card_errors = [error for card in cards for error in list_of_strings(card.get("errors"))]
    valid = worksheet_validation.get("valid") is True and not package_errors and not card_errors and all(card.get("valid") is True for card in cards)
    binding_ready = valid and len(cards) == int_count(summary.get("card_count")) and all(card.get("binding_ready") is True for card in cards)
    return {
        "schema_version": FILLED_LAUNCH_CARD_PACKAGE_SCHEMA_VERSION,
        "mode": "phase3_filled_launch_card_package",
        "valid": valid,
        "binding_ready": binding_ready,
        "card_count": len(cards),
        "binding_ready_card_count": sum(1 for card in cards if card.get("binding_ready") is True),
        "task_count": sum(int_count(card.get("task_count")) for card in cards),
        "bound_task_count": sum(int_count(card.get("bound_task_count")) for card in cards),
        "command_option_count": sum(int_count(card.get("command_option_count")) for card in cards),
        "command_ready_count": sum(int_count(card.get("command_ready_count")) for card in cards),
        "missing_runtime_command_count": sum(int_count(card.get("missing_runtime_command_count")) for card in cards),
        "errors": package_errors + card_errors,
        "blockers": sorted({blocker for card in cards for blocker in list_of_strings(card.get("blockers"))}),
        "worksheet_validation": worksheet_validation,
        "cards": cards,
        "safety_contract": [
            "filled launch-card package is built from local metadata only",
            "filled launch-card package validation does not launch model servers",
            "filled launch-card package validation does not run Docker",
            "filled launch-card package validation does not call endpoints",
            "filled launch-card package validation does not inspect private tokens",
            "filled launch-card package validation does not send prompt traffic",
            "filled launch-card package validation does not mutate runtime residency",
            "filled launch-card package validation does not claim live expert paging",
        ],
    }


def compact_filled_launch_card_package(package: JSONDict) -> JSONDict:
    return {
        "valid": package.get("valid"),
        "binding_ready": package.get("binding_ready"),
        "card_count": package.get("card_count"),
        "binding_ready_card_count": package.get("binding_ready_card_count"),
        "task_count": package.get("task_count"),
        "bound_task_count": package.get("bound_task_count"),
        "command_option_count": package.get("command_option_count"),
        "command_ready_count": package.get("command_ready_count"),
        "missing_runtime_command_count": package.get("missing_runtime_command_count"),
        "errors": package.get("errors", []),
        "blockers": package.get("blockers", []),
        "worksheet_validation": package.get("worksheet_validation", {}),
    }


def write_binding_worksheet(summary: JSONDict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_json(build_binding_worksheet(summary)), encoding="utf-8")



def write_model_plane_artifact_writer_contract_request(summary: JSONDict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_json(build_model_plane_artifact_writer_contract_request(summary)), encoding="utf-8")


def saved_handoff_artifact_status(
    *,
    artifact_id: str,
    path: Path,
    expected_payload: JSONDict,
    write_command: list[str],
) -> JSONDict:
    display = display_path(path) or str(path)
    if not path.exists():
        return {
            "artifact_id": artifact_id,
            "path": display,
            "exists": False,
            "valid": False,
            "ready": False,
            "drifted": False,
            "errors": [f"saved handoff artifact missing: {display}"],
            "blockers": ["saved_handoff_artifact_missing"],
            "write_command": write_command,
        }
    try:
        actual = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "artifact_id": artifact_id,
            "path": display,
            "exists": True,
            "valid": False,
            "ready": False,
            "drifted": False,
            "errors": [str(exc)],
            "blockers": ["saved_handoff_artifact_invalid"],
            "write_command": write_command,
        }
    drifted = actual != expected_payload
    return {
        "artifact_id": artifact_id,
        "path": display,
        "exists": True,
        "valid": isinstance(actual, dict),
        "ready": isinstance(actual, dict) and not drifted,
        "drifted": drifted,
        "errors": [] if isinstance(actual, dict) else ["saved handoff artifact must be a JSON object"],
        "blockers": ["saved_handoff_artifact_drifted"] if drifted else ([] if isinstance(actual, dict) else ["saved_handoff_artifact_invalid"]),
        "write_command": write_command,
    }


def summary_for_saved_handoff_artifacts(summary: JSONDict) -> JSONDict:
    normalized = dict(summary)
    if "library_metadata_valid" in normalized:
        normalized["valid"] = normalized["library_metadata_valid"] is True
    if "library_scaffold_ready" in normalized:
        normalized["library_ready"] = normalized["library_scaffold_ready"] is True
    return normalized


def build_saved_handoff_artifacts_summary(summary: JSONDict, *, root: Path = DEFAULT_ROOT) -> JSONDict:
    source_summary = summary_for_saved_handoff_artifacts(summary)
    worksheet_path = default_binding_worksheet_path(root)
    contract_request_path = default_model_plane_contract_request_path(root)
    artifacts = [
        saved_handoff_artifact_status(
            artifact_id="launch_card_binding_worksheet",
            path=worksheet_path,
            expected_payload=build_binding_worksheet(source_summary),
            write_command=[
                "uv",
                "run",
                "--managed-python",
                "--python",
                "3.13",
                "scripts/plan_phase3_launch_card_library.py",
                "--write-default-handoff-artifacts",
                "--json",
            ],
        ),
        saved_handoff_artifact_status(
            artifact_id="model_plane_artifact_writer_contract_request",
            path=contract_request_path,
            expected_payload=build_model_plane_artifact_writer_contract_request(source_summary),
            write_command=[
                "uv",
                "run",
                "--managed-python",
                "--python",
                "3.13",
                "scripts/plan_phase3_launch_card_library.py",
                "--write-default-handoff-artifacts",
                "--json",
            ],
        ),
    ]
    errors = [error for item in artifacts for error in list_of_strings(item.get("errors"))]
    blockers = sorted({blocker for item in artifacts for blocker in list_of_strings(item.get("blockers"))})
    ready_count = sum(1 for item in artifacts if item.get("ready") is True)
    return {
        "schema_version": "moe-phase3-saved-handoff-artifacts-v1",
        "mode": "phase3_saved_handoff_artifacts",
        "valid": not errors,
        "ready": ready_count == len(artifacts) and not blockers,
        "artifact_count": len(artifacts),
        "ready_count": ready_count,
        "missing_count": sum(1 for item in artifacts if item.get("exists") is not True),
        "drifted_count": sum(1 for item in artifacts if item.get("drifted") is True),
        "invalid_count": sum(1 for item in artifacts if item.get("valid") is not True),
        "errors": errors,
        "blockers": blockers,
        "artifacts": artifacts,
        "next_actions": [
            "Regenerate saved handoff artifacts after any launch-card, prompt-set, or capture-request change.",
            "Give Model Plane the saved artifact-writer contract request before filling runtime callables.",
            "Validate any returned fulfillment before writing filled launch cards.",
        ],
        "safety_contract": [
            "saved handoff artifact audit reads local metadata only",
            "saved handoff artifact audit does not launch model servers",
            "saved handoff artifact audit does not run Docker",
            "saved handoff artifact audit does not call endpoints",
            "saved handoff artifact audit does not inspect private tokens",
            "saved handoff artifact audit does not send prompt traffic",
            "saved handoff artifact audit does not mutate runtime residency",
            "saved handoff artifact audit does not claim live expert paging",
        ],
    }


def write_default_handoff_artifacts(summary: JSONDict, *, root: Path = DEFAULT_ROOT) -> list[JSONDict]:
    source_summary = summary_for_saved_handoff_artifacts(summary)
    outputs: list[JSONDict] = []
    payloads = [
        ("launch_card_binding_worksheet", default_binding_worksheet_path(root), build_binding_worksheet(source_summary)),
        (
            "model_plane_artifact_writer_contract_request",
            default_model_plane_contract_request_path(root),
            build_model_plane_artifact_writer_contract_request(source_summary),
        ),
    ]
    for artifact_id, output_path, payload in payloads:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(canonical_json(payload), encoding="utf-8")
        outputs.append({"artifact_id": artifact_id, "path": display_path(output_path) or str(output_path)})
    return outputs

def write_filled_launch_cards(package: JSONDict, output_dir: Path) -> list[JSONDict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[JSONDict] = []
    for card in package.get("cards", []):
        if not isinstance(card, dict) or not isinstance(card.get("launch_card"), dict):
            continue
        source_path = Path(str(card.get("launch_card_path") or "launch-card.json"))
        output_path = output_dir / source_path.name
        output_path.write_text(json.dumps(card["launch_card"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(
            {
                "request_name": card.get("request_name"),
                "source_launch_card_path": card.get("launch_card_path"),
                "output_path": display_path(output_path) or str(output_path),
                "binding_ready": card.get("binding_ready"),
                "missing_runtime_command_count": card.get("missing_runtime_command_count"),
            }
        )
    return written


def error_card_entry(bundle_path: Path, request_path: Path, launch_card_path: Path, exc: Exception) -> JSONDict:
    return {
        "bundle_path": display_path(bundle_path),
        "request_path": display_path(request_path),
        "launch_card_path": display_path(launch_card_path),
        "request_name": None,
        "model_id": None,
        "backend_family": None,
        "prompt_family": None,
        "valid": False,
        "template_ready": False,
        "planned_only": False,
        "executable": False,
        "model_plane_binding_ready": False,
        "binding_valid": False,
        "binding_ready": False,
        "runtime_capture_command_ready": False,
        "task_count": 0,
        "bound_task_count": 0,
        "command_option_count": 0,
        "command_ready_count": 0,
        "missing_runtime_command_count": 0,
        "missing_runtime_command_artifact_ids": [],
        "binding_handoff_task_count": 0,
        "binding_handoff_ready_count": 0,
        "binding_handoff_missing_field_count": 0,
        "unbound_task_count": 0,
        "binding_tasks": [],
        "blockers": ["launch_card_library_entry_invalid"],
        "errors": [str(exc)],
    }


def build_card_entry(bundle_path: Path) -> JSONDict:
    request_path = plan_phase3_handoff_coverage.runtime_request_path(bundle_path)
    launch_card_path = plan_phase3_handoff_coverage.runtime_launch_card_template_path(bundle_path)
    try:
        launch_card = load_launch_card(launch_card_path)
        card_summary = plan_phase3_handoff_coverage.launch_card_template_summary(request_path, launch_card_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return error_card_entry(bundle_path, request_path, launch_card_path, exc)

    blockers = set(list_of_strings(card_summary.get("blockers")))
    blockers.update(list_of_strings(card_summary.get("binding_blockers")))
    if card_summary.get("exists") is not True:
        blockers.add("runtime_capture_launch_card_template_missing")
    if card_summary.get("template_ready") is not True:
        blockers.add("runtime_capture_launch_card_template_not_ready")
    if card_summary.get("model_plane_binding_ready") is not True:
        blockers.add("model_plane_binding_context_missing")
    if card_summary.get("binding_ready") is not True:
        blockers.add("launch_card_runtime_command_bindings_missing")
    if card_summary.get("runtime_capture_command_ready") is not True:
        blockers.add("runtime_capture_commands_unbound")

    validation_command = build_validation_command(request_path, launch_card_path)
    raw_tasks = launch_card.get("tasks") if isinstance(launch_card.get("tasks"), list) else []
    binding_tasks = [
        build_binding_handoff_task(
            launch_card=launch_card,
            request_path=request_path,
            launch_card_path=launch_card_path,
            task=task,
            validation_command=validation_command,
        )
        for task in raw_tasks
        if isinstance(task, dict)
    ]
    binding_handoff_task_count = len(binding_tasks)
    binding_handoff_ready_count = sum(1 for task in binding_tasks if task.get("binding_handoff_ready") is True)
    binding_handoff_missing_field_count = sum(len(task.get("missing_fields", [])) for task in binding_tasks)
    command_ready_count = sum(1 for task in binding_tasks if task.get("command_ready") is True)
    unbound_task_count = sum(1 for task in binding_tasks if task.get("command_ready") is not True)
    if binding_handoff_task_count != int_count(card_summary.get("task_count")):
        blockers.add("launch_card_binding_handoff_task_count_mismatch")
    if binding_handoff_missing_field_count:
        blockers.add("launch_card_binding_handoff_incomplete")
    if unbound_task_count:
        blockers.add("launch_card_runtime_bindings_unfilled")

    return {
        "bundle_path": display_path(bundle_path),
        "request_path": display_path(request_path),
        "launch_card_path": display_path(launch_card_path),
        "request_name": launch_card.get("request_name"),
        "model_id": launch_card.get("model_id"),
        "backend_family": launch_card.get("backend_family"),
        "prompt_family": launch_card.get("prompt_family"),
        "valid": card_summary.get("valid") is True,
        "template_ready": card_summary.get("template_ready") is True,
        "planned_only": card_summary.get("planned_only") is True,
        "executable": card_summary.get("executable") is True,
        "model_plane_binding_ready": card_summary.get("model_plane_binding_ready") is True,
        "binding_valid": card_summary.get("binding_valid") is True,
        "binding_ready": card_summary.get("binding_ready") is True,
        "runtime_capture_command_ready": card_summary.get("runtime_capture_command_ready") is True,
        "task_count": int_count(card_summary.get("task_count")),
        "bound_task_count": int_count(card_summary.get("bound_task_count")),
        "command_option_count": int_count(card_summary.get("command_option_count")),
        "command_ready_count": command_ready_count,
        "missing_runtime_command_count": int_count(card_summary.get("missing_runtime_command_count")),
        "missing_runtime_command_artifact_ids": list_of_strings(card_summary.get("missing_runtime_command_artifact_ids")),
        "binding_handoff_task_count": binding_handoff_task_count,
        "binding_handoff_ready_count": binding_handoff_ready_count,
        "binding_handoff_missing_field_count": binding_handoff_missing_field_count,
        "unbound_task_count": unbound_task_count,
        "binding_tasks": binding_tasks,
        "blockers": sorted(blockers),
        "errors": list_of_strings(card_summary.get("errors")),
    }


def build_library(root: Path = DEFAULT_ROOT, *, model_plane_card_catalog: list[JSONDict] | None = None) -> JSONDict:
    errors: list[str] = []
    if not root.exists():
        errors.append(f"launch-card library root does not exist: {display_path(root)}")
        bundle_paths: list[Path] = []
    else:
        bundle_paths = sorted(root.glob(BUNDLE_GLOB))

    cards = [build_card_entry(path) for path in bundle_paths]
    model_plane_bridge = build_model_plane_bridge_summary(
        cards,
        model_plane_card_catalog or [],
        provided=model_plane_card_catalog is not None,
    )
    card_count = len(cards)
    template_ready_count = sum(1 for card in cards if card.get("template_ready") is True)
    model_plane_binding_ready_count = sum(1 for card in cards if card.get("model_plane_binding_ready") is True)
    binding_ready_count = sum(1 for card in cards if card.get("binding_ready") is True)
    runtime_capture_command_ready_count = sum(1 for card in cards if card.get("runtime_capture_command_ready") is True)
    task_count = sum(int_count(card.get("task_count")) for card in cards)
    bound_task_count = sum(int_count(card.get("bound_task_count")) for card in cards)
    command_option_count = sum(int_count(card.get("command_option_count")) for card in cards)
    command_ready_count = sum(int_count(card.get("command_ready_count")) for card in cards)
    missing_runtime_command_count = sum(int_count(card.get("missing_runtime_command_count")) for card in cards)
    binding_tasks = [
        task
        for card in cards
        for task in card.get("binding_tasks", [])
        if isinstance(task, dict)
    ]
    binding_handoff_task_count = sum(int_count(card.get("binding_handoff_task_count")) for card in cards)
    binding_handoff_ready_count = sum(int_count(card.get("binding_handoff_ready_count")) for card in cards)
    binding_handoff_missing_field_count = sum(int_count(card.get("binding_handoff_missing_field_count")) for card in cards)
    unbound_task_count = sum(int_count(card.get("unbound_task_count")) for card in cards)
    card_errors = [error for card in cards for error in list_of_strings(card.get("errors"))]

    library_ready = card_count > 0 and template_ready_count == card_count and not errors and not card_errors
    binding_handoff_ready = (
        library_ready
        and binding_handoff_task_count == task_count
        and binding_handoff_task_count > 0
        and binding_handoff_ready_count == binding_handoff_task_count
        and binding_handoff_missing_field_count == 0
    )
    execution_ready = (
        library_ready
        and model_plane_binding_ready_count == card_count
        and binding_ready_count == card_count
        and runtime_capture_command_ready_count == card_count
        and missing_runtime_command_count == 0
    )
    blockers: set[str] = set()
    if card_count == 0:
        blockers.add("phase3_launch_card_library_empty")
    if template_ready_count != card_count:
        blockers.add("runtime_capture_launch_card_templates_missing_or_invalid")
    if model_plane_binding_ready_count != card_count:
        blockers.add("model_plane_binding_context_missing")
    if binding_ready_count != card_count:
        blockers.add("launch_card_runtime_command_bindings_missing")
    if runtime_capture_command_ready_count != card_count:
        blockers.add("runtime_capture_commands_unbound")
    if not binding_handoff_ready:
        blockers.add("launch_card_binding_handoff_incomplete")
    if unbound_task_count:
        blockers.add("launch_card_runtime_bindings_unfilled")
    for card in cards:
        blockers.update(list_of_strings(card.get("blockers")))
    if model_plane_card_catalog is not None and model_plane_bridge.get("bridge_ready") is not True:
        blockers.update(list_of_strings(model_plane_bridge.get("blockers")))

    summary = {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "mode": "phase3_launch_card_library",
        "root": display_path(root),
        "valid": not errors and not card_errors and all(card.get("valid") is True for card in cards),
        "library_ready": library_ready,
        "execution_ready": execution_ready,
        "errors": errors + card_errors,
        "library_metadata_valid": not errors and not card_errors and all(card.get("valid") is True for card in cards),
        "library_scaffold_ready": library_ready,
        "card_count": card_count,
        "template_ready_count": template_ready_count,
        "model_plane_binding_ready_count": model_plane_binding_ready_count,
        "binding_ready_count": binding_ready_count,
        "runtime_capture_command_ready_count": runtime_capture_command_ready_count,
        "task_count": task_count,
        "bound_task_count": bound_task_count,
        "command_option_count": command_option_count,
        "command_ready_count": command_ready_count,
        "missing_runtime_command_count": missing_runtime_command_count,
        "binding_handoff_ready": binding_handoff_ready,
        "binding_handoff_task_count": binding_handoff_task_count,
        "binding_handoff_ready_count": binding_handoff_ready_count,
        "binding_handoff_missing_field_count": binding_handoff_missing_field_count,
        "unbound_task_count": unbound_task_count,
        "binding_tasks": binding_tasks,
        "model_plane_bridge": model_plane_bridge,
        "model_plane_bridge_ready": model_plane_bridge.get("bridge_ready") is True,
        "model_plane_runtime_launch_candidate_task_count": model_plane_bridge.get("runtime_launch_candidate_task_count"),
        "model_plane_phase3_artifact_writer_contract_count": model_plane_bridge.get("artifact_writer_contract_count"),
        "model_plane_phase3_artifact_writer_ready_count": model_plane_bridge.get("artifact_writer_ready_count"),
        "model_plane_artifact_writer_contract_request_ready": binding_handoff_ready,
        "model_plane_artifact_writer_contract_request_task_count": binding_handoff_task_count,
        "cards": cards,
        "blockers": sorted(blockers),
        "next_actions": [
            "Use binding_tasks as the repo-wide worksheet for filling Model Plane callable ids or launch commands.",
            "Bind each launch card to a real Model Plane callable id or launch command after explicit approval.",
            "Keep planned-only cards non-executable until every task has a ready command binding.",
            "After approved capture, fill receipts and run capture-result intake before bundle promotion.",
            "If Model Plane card catalog is provided, use model_plane_bridge to avoid treating runtime/probe cards as Phase 3 artifact writers.",
            "Export the Model Plane artifact-writer contract request when Model Plane needs exact Phase 3 writer requirements.",
            "Validate a returned Model Plane artifact-writer fulfillment with --model-plane-contract-fulfillment before using filled cards.",
            "Keep the saved binding worksheet and Model Plane artifact-writer contract request drift-free.",
        ],
        "safety_contract": [
            "launch-card library planner reads local metadata only",
            "launch-card library planner does not launch model servers",
            "launch-card library planner does not run Docker",
            "launch-card library planner does not call endpoints",
            "launch-card library planner does not inspect private tokens",
            "launch-card library planner does not send prompt traffic",
            "launch-card library planner does not mutate runtime residency",
            "launch-card library planner does not claim live expert paging",
        ],
    }
    saved_handoff_artifacts = build_saved_handoff_artifacts_summary(summary, root=root)
    summary["saved_handoff_artifacts"] = saved_handoff_artifacts
    summary["saved_handoff_artifacts_ready"] = saved_handoff_artifacts.get("ready") is True
    summary["saved_handoff_artifact_missing_count"] = int_count(saved_handoff_artifacts.get("missing_count"))
    summary["saved_handoff_artifact_drifted_count"] = int_count(saved_handoff_artifacts.get("drifted_count"))
    if saved_handoff_artifacts.get("ready") is not True:
        summary["valid"] = False
        summary["library_ready"] = False
        summary["errors"] = list_of_strings(summary.get("errors")) + list_of_strings(saved_handoff_artifacts.get("errors"))
        summary["blockers"] = sorted(
            set(list_of_strings(summary.get("blockers")) + list_of_strings(saved_handoff_artifacts.get("blockers")))
        )
    return summary


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def format_markdown_report(summary: JSONDict) -> str:
    lines = [
        "# Phase 3 Launch-Card Library",
        "",
        f"- Valid: `{summary.get('valid')}`",
        f"- Library ready: `{summary.get('library_ready')}`",
        f"- Execution ready: `{summary.get('execution_ready')}`",
        f"- Cards: `{summary.get('card_count')}`",
        f"- Template-ready cards: `{summary.get('template_ready_count')}`",
        f"- Model Plane binding-ready cards: `{summary.get('model_plane_binding_ready_count')}`",
        f"- Runtime-command-ready cards: `{summary.get('runtime_capture_command_ready_count')}`",
        f"- Tasks: `{summary.get('task_count')}`",
        f"- Runtime command options: `{summary.get('command_option_count')}`",
        f"- Runtime commands marked ready: `{summary.get('command_ready_count')}`",
        f"- Missing runtime commands: `{summary.get('missing_runtime_command_count')}`",
        f"- Binding handoff ready: `{summary.get('binding_handoff_ready')}`",
        f"- Binding handoff tasks: `{summary.get('binding_handoff_task_count')}`",
        f"- Binding handoff missing fields: `{summary.get('binding_handoff_missing_field_count')}`",
        f"- Unbound tasks: `{summary.get('unbound_task_count')}`",
        f"- Model Plane artifact-writer contract request ready: `{summary.get('model_plane_artifact_writer_contract_request_ready')}`",
        f"- Model Plane artifact-writer contract request tasks: `{summary.get('model_plane_artifact_writer_contract_request_task_count')}`",
        f"- Saved handoff artifacts ready: `{summary.get('saved_handoff_artifacts_ready')}`",
        f"- Saved handoff artifacts missing: `{summary.get('saved_handoff_artifact_missing_count')}`",
        f"- Saved handoff artifacts drifted: `{summary.get('saved_handoff_artifact_drifted_count')}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = list_of_strings(summary.get("blockers"))
    if blockers:
        for blocker in blockers:
            lines.append(f"- `{markdown_escape(blocker)}`")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Cards",
            "",
            "| Card | Template | Model Plane | Binding | Runtime Commands | Missing Commands | Path |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for card in summary.get("cards", []):
        if not isinstance(card, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                markdown_escape(value)
                for value in (
                    card.get("request_name") or card.get("bundle_path") or "unknown",
                    card.get("template_ready"),
                    card.get("model_plane_binding_ready"),
                    card.get("binding_ready"),
                    card.get("runtime_capture_command_ready"),
                    card.get("missing_runtime_command_count"),
                    card.get("launch_card_path") or "missing",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Binding Handoff Tasks",
            "",
            "| Card | Artifact | Capture Kind | Approval Keys | Command Option | Artifact Path | Receipt Path |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for task in summary.get("binding_tasks", []):
        if not isinstance(task, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                markdown_escape(value)
                for value in (
                    task.get("request_name") or "unknown",
                    task.get("artifact_id") or "unknown",
                    task.get("capture_kind") or "unknown",
                    ", ".join(list_of_strings(task.get("approval_keys"))) or "missing",
                    task.get("command_option_ready"),
                    task.get("artifact_output_path") or "missing",
                    task.get("receipt_output_path") or "missing",
                )
            )
            + " |"
        )
    print(f"Saved handoff artifacts ready: {summary.get('saved_handoff_artifacts_ready')}")
    print(f"Saved handoff artifacts missing: {summary.get('saved_handoff_artifact_missing_count')}")
    print(f"Saved handoff artifacts drifted: {summary.get('saved_handoff_artifact_drifted_count')}")
    bridge = summary.get("model_plane_bridge") if isinstance(summary.get("model_plane_bridge"), dict) else {}
    if bridge:
        lines.extend(
            [
                "",
                "## Model Plane Bridge",
                "",
                f"- Catalog provided: `{bridge.get('provided')}`",
                f"- Runtime launch candidate tasks: `{bridge.get('runtime_launch_candidate_task_count')}`",
                f"- Phase 3 artifact-writer-contract tasks: `{bridge.get('artifact_writer_contract_count')}`",
                f"- Phase 3 artifact-writer-ready tasks: `{bridge.get('artifact_writer_ready_count')}`",
                f"- Missing artifact writers: `{bridge.get('missing_artifact_writer_count')}`",
                f"- Bridge ready: `{bridge.get('bridge_ready')}`",
            ]
        )
        bridge_blockers = list_of_strings(bridge.get("blockers"))
        if bridge_blockers:
            lines.append(f"- Bridge blockers: `{markdown_escape(', '.join(bridge_blockers))}`")
    lines.extend(["", "## Safety Contract", ""])
    for item in summary.get("safety_contract", []):
        lines.append(f"- {markdown_escape(item)}")
    return "\n".join(lines) + "\n"


def write_markdown_report(summary: JSONDict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_markdown_report(summary), encoding="utf-8")


def print_human_summary(summary: JSONDict) -> None:
    print("MoE Run Anyway Phase 3 launch-card library")
    print(f"Valid: {summary['valid']}")
    print(f"Library ready: {summary['library_ready']}")
    print(f"Execution ready: {summary['execution_ready']}")
    print(f"Cards: {summary['card_count']}")
    print(f"Template-ready cards: {summary['template_ready_count']}")
    print(f"Model Plane binding-ready cards: {summary['model_plane_binding_ready_count']}")
    print(f"Runtime-command-ready cards: {summary['runtime_capture_command_ready_count']}")
    print(f"Tasks: {summary['task_count']}")
    print(f"Runtime command options: {summary['command_option_count']}")
    print(f"Runtime commands marked ready: {summary['command_ready_count']}")
    print(f"Missing runtime commands: {summary['missing_runtime_command_count']}")
    print(f"Binding handoff ready: {summary['binding_handoff_ready']}")
    print(f"Binding handoff tasks: {summary['binding_handoff_task_count']}")
    print(f"Binding handoff missing fields: {summary['binding_handoff_missing_field_count']}")
    print(f"Unbound tasks: {summary['unbound_task_count']}")
    print(f"Model Plane artifact-writer contract request ready: {summary['model_plane_artifact_writer_contract_request_ready']}")
    print(f"Model Plane artifact-writer contract request tasks: {summary['model_plane_artifact_writer_contract_request_task_count']}")
    print(f"Saved handoff artifacts ready: {summary.get('saved_handoff_artifacts_ready')}")
    print(f"Saved handoff artifacts missing: {summary.get('saved_handoff_artifact_missing_count')}")
    print(f"Saved handoff artifacts drifted: {summary.get('saved_handoff_artifact_drifted_count')}")
    bridge = summary.get("model_plane_bridge") if isinstance(summary.get("model_plane_bridge"), dict) else {}
    if bridge:
        print(
            "Model Plane bridge: "
            f"provided={bridge.get('provided')} "
            f"runtime_tasks={bridge.get('runtime_launch_candidate_task_count')} "
            f"artifact_writer_contracts={bridge.get('artifact_writer_contract_count')} "
            f"artifact_writers={bridge.get('artifact_writer_ready_count')} "
            f"ready={bridge.get('bridge_ready')}"
        )
    blockers = ", ".join(summary.get("blockers", [])) or "none"
    print(f"Blockers: {blockers}")
    print("Cards:")
    for card in summary["cards"]:
        print(
            "  - "
            f"{card.get('request_name')}: "
            f"template={card.get('template_ready')} "
            f"binding={card.get('binding_ready')} "
            f"runtime={card.get('runtime_capture_command_ready')} "
            f"missing={card.get('missing_runtime_command_count')}"
        )
    print("Safety contract:")
    for item in summary["safety_contract"]:
        print(f"  - {item}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    parser.add_argument("--output-md", type=Path, help="write a Markdown launch-card library report")
    parser.add_argument("--output-binding-worksheet", type=Path, help="write a fillable repo-wide binding worksheet JSON")
    parser.add_argument("--output-model-plane-contract-request", type=Path, help="write a Model Plane Phase 3 artifact-writer contract request JSON")
    parser.add_argument("--write-default-handoff-artifacts", action="store_true", help="write the default saved binding worksheet and Model Plane contract request")
    parser.add_argument("--model-plane-contract-fulfillment", type=Path, help="validate a Model Plane artifact-writer fulfillment JSON without executing it")
    parser.add_argument("--binding-worksheet", type=Path, help="validate a filled binding worksheet JSON without executing it")
    parser.add_argument("--output-filled-card-dir", type=Path, help="write filled launch cards built from --binding-worksheet")
    parser.add_argument("--model-plane-card-catalog", type=Path, help="read a Model Plane /moe-test-cards catalog for bridge compatibility checks")
    return parser


def plan_paths(root: Path = DEFAULT_ROOT, *, model_plane_card_catalog_path: Path | None = None) -> tuple[int, JSONDict | None, str | None]:
    try:
        model_plane_card_catalog = load_model_plane_card_catalog(model_plane_card_catalog_path) if model_plane_card_catalog_path else None
        summary = build_library(root, model_plane_card_catalog=model_plane_card_catalog)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not build Phase 3 launch-card library: {exc}"
    return (0 if summary["valid"] else 2), summary, None


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary, error_message = plan_paths(args.root, model_plane_card_catalog_path=args.model_plane_card_catalog)
    if error_message:
        print(error_message, file=sys.stderr)
        return status
    assert summary is not None
    if args.output_md:
        write_markdown_report(summary, args.output_md)
    if args.output_binding_worksheet:
        write_binding_worksheet(summary, args.output_binding_worksheet)
    if args.output_model_plane_contract_request:
        write_model_plane_artifact_writer_contract_request(summary, args.output_model_plane_contract_request)
    if args.write_default_handoff_artifacts:
        written_outputs = write_default_handoff_artifacts(summary, root=args.root)
        status, refreshed_summary, refresh_error = plan_paths(args.root, model_plane_card_catalog_path=args.model_plane_card_catalog)
        if refresh_error:
            print(refresh_error, file=sys.stderr)
            return status
        assert refreshed_summary is not None
        summary = refreshed_summary
        summary["written_default_handoff_artifacts"] = written_outputs
        status = 0 if summary.get("valid") is True else 2
    if args.binding_worksheet and args.model_plane_contract_fulfillment:
        print("Use either --binding-worksheet or --model-plane-contract-fulfillment, not both.", file=sys.stderr)
        return 2
    if args.binding_worksheet:
        worksheet = plan_phase3_runtime_capture_commands.load_json(args.binding_worksheet)
        package = build_filled_launch_card_package(summary, worksheet)
        summary["filled_launch_card_package_summary"] = compact_filled_launch_card_package(package)
        if args.output_filled_card_dir:
            summary["filled_launch_card_outputs"] = write_filled_launch_cards(package, args.output_filled_card_dir)
        if package.get("valid") is not True:
            status = 2
    if args.model_plane_contract_fulfillment:
        fulfillment = plan_phase3_runtime_capture_commands.load_json(args.model_plane_contract_fulfillment)
        package = build_filled_launch_card_package_from_model_plane_fulfillment(summary, fulfillment)
        summary["model_plane_artifact_writer_fulfillment_validation"] = package.get("model_plane_artifact_writer_fulfillment_validation", {})
        summary["filled_launch_card_package_summary"] = compact_filled_launch_card_package(package)
        if args.output_filled_card_dir:
            summary["filled_launch_card_outputs"] = write_filled_launch_cards(package, args.output_filled_card_dir)
        if package.get("valid") is not True:
            status = 2
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_human_summary(summary)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
