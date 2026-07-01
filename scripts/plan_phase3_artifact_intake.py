#!/usr/bin/env python3
"""Plan intake of generated Phase 3 real-evidence artifacts.

This planner scans local/generated artifact folders for semantic router traces,
optionally matches them to a GGUF MoE inventory summary, and optionally checks a
scanner-derived expert inventory manifest plus dense fallback comparison. It
never launches runtimes, reads tensor values, runs Docker, downloads models,
uses secrets, mutates residency, or sends prompt traffic.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import plan_dense_fallback_comparison
import plan_real_model_trace_inventory_pairing


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "memory-moe-mvp" / "phase3-real-evidence"
DEFAULT_TRACE_GLOB = "**/router-events.jsonl"
SUPPORTED_SCHEMA_VERSION = "moe-phase3-artifact-intake-plan-v1"
SUPPORTED_TRACE_CONTRACT = "memory-moe-bridge-v1"
REQUIRED_TRACE_KINDS = {"selected_experts", "selected_weights", "selected_weights_norm"}
DEFAULT_DOCKER_IMAGE = "ghcr.io/astral-sh/uv:python3.13-bookworm"
DEFAULT_DOCKER_REPO_MOUNT = "/repo"
DEFAULT_DOCKER_OLLAMA_MOUNT = "/ollama"

JSONDict = dict[str, Any]


def display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def uv_command(*parts: Path | str) -> list[str]:
    return [
        "uv",
        "run",
        "--managed-python",
        "--python",
        "3.13",
        *[display_path(part) if isinstance(part, Path) else part for part in parts],
    ]


def safe_artifact_stem(value: str) -> str:
    stem = Path(value).name or "model"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return cleaned or "model"


def clean_container_mount(value: str) -> str:
    cleaned = value.strip() or "/mount"
    if not cleaned.startswith("/"):
        cleaned = f"/{cleaned}"
    return cleaned.rstrip("/") or "/"


def container_repo_path(path: Path, docker_repo_mount: str) -> str:
    mount = clean_container_mount(docker_repo_mount)
    try:
        relative = path.resolve().relative_to(ROOT)
    except (OSError, ValueError):
        return str(path)
    if relative.as_posix() == ".":
        return mount
    return f"{mount}/{relative.as_posix()}"


def docker_ollama_model_path(model_id: str, docker_ollama_mount: str) -> str | None:
    mount = clean_container_mount(docker_ollama_mount)
    normalized = model_id.replace("\\", "/")
    known_prefixes = (
        "/root/.ollama/",
        "/home/ollama/.ollama/",
        "/usr/share/ollama/.ollama/",
        "/ollama/",
    )
    for prefix in known_prefixes:
        if normalized.startswith(prefix):
            return f"{mount}/{normalized[len(prefix):]}"
    return None


def docker_ollama_scanner_command(
    *,
    model_id: str,
    inventory_output: Path,
    docker_ollama_volume: str | None,
    docker_image: str,
    docker_repo_mount: str,
    docker_ollama_mount: str,
) -> JSONDict | None:
    if not docker_ollama_volume:
        return None
    model_path = docker_ollama_model_path(model_id, docker_ollama_mount)
    if model_path is None:
        return None

    repo_mount = clean_container_mount(docker_repo_mount)
    ollama_mount = clean_container_mount(docker_ollama_mount)
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{docker_ollama_volume}:{ollama_mount}:ro",
        "-v",
        f"{ROOT}:{repo_mount}",
        "-w",
        repo_mount,
        docker_image,
        "uv",
        "run",
        "--managed-python",
        "--python",
        "3.13",
        "scripts/scan_gguf_expert_inventory.py",
        model_path,
        "--model-id",
        model_id,
        "--output",
        container_repo_path(inventory_output, repo_mount),
        "--json",
    ]
    return {
        "command_class": "docker_ollama_scanner_inventory_manifest_builder",
        "command": command,
        "notes": [
            "mounts the Ollama Docker volume read-only so the scanner can read the traced GGUF blob",
            "mounts the repo so the scanner can write the requested expert inventory manifest",
            "planner only emits this command; it does not run Docker",
        ],
    }


def read_text_any(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-16")


def load_jsonl(path: Path) -> list[JSONDict]:
    rows: list[JSONDict] = []
    for line_number, line in enumerate(read_text_any(path).splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        rows.append(row)
    return rows


def load_inventory_summary(path: Path | None) -> list[JSONDict]:
    if path is None:
        return []
    return load_jsonl(path)


def trace_paths(root: Path, trace_glob: str) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.glob(trace_glob) if path.is_file())


def unique_strings(events: list[JSONDict], field: str) -> list[str]:
    values = {
        str(event[field])
        for event in events
        if isinstance(event.get(field), str) and str(event[field]).strip()
    }
    return sorted(values)


def trace_fixture_signals(path: Path, events: list[JSONDict]) -> list[str]:
    signals: list[str] = []
    if "fixture" in path.name.lower() or "synthetic" in path.name.lower():
        signals.append("trace_path")
    for index, event in enumerate(events):
        for field in ("model", "tensor_name"):
            value = event.get(field)
            if isinstance(value, str) and ("fixture" in value.lower() or "synthetic" in value.lower()):
                signals.append(f"events[{index}].{field}")
    return sorted(set(signals))


def summarize_trace(path: Path) -> JSONDict:
    errors: list[str] = []
    try:
        events = load_jsonl(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {
            "path": display_path(path),
            "valid": False,
            "errors": [str(exc)],
            "event_count": 0,
        }

    kind_counts: dict[str, int] = {}
    routed_layers: set[int] = set()
    top_k_values: set[int] = set()
    for index, event in enumerate(events):
        kind = event.get("tensor_kind")
        if isinstance(kind, str):
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
        if kind == "selected_experts":
            layer_id = event.get("layer_id")
            if isinstance(layer_id, int) and not isinstance(layer_id, bool):
                routed_layers.add(layer_id)
            else:
                errors.append(f"events[{index}].layer_id must be an integer")
            values = event.get("values")
            if not isinstance(values, list) or not values:
                errors.append(f"events[{index}].values must be a non-empty list")
            shape = event.get("shape")
            if isinstance(shape, list) and shape and isinstance(shape[0], int) and not isinstance(shape[0], bool):
                top_k_values.add(shape[0])
        contract = event.get("contract_version")
        if contract != SUPPORTED_TRACE_CONTRACT:
            errors.append(f"events[{index}].contract_version must be {SUPPORTED_TRACE_CONTRACT!r}")

    missing_kinds = sorted(REQUIRED_TRACE_KINDS - set(kind_counts))
    if missing_kinds:
        errors.append(f"trace missing tensor kinds: {', '.join(missing_kinds)}")
    if not events:
        errors.append("trace file is empty")

    model_names = unique_strings(events, "model")
    backend_families = unique_strings(events, "backend_family")
    contract_versions = unique_strings(events, "contract_version")
    if len(model_names) != 1:
        errors.append(f"trace must name exactly one model, got {model_names}")
    if len(backend_families) != 1:
        errors.append(f"trace must name exactly one backend family, got {backend_families}")

    return {
        "path": display_path(path),
        "valid": not errors,
        "errors": errors,
        "event_count": len(events),
        "kind_counts": dict(sorted(kind_counts.items())),
        "model_names": model_names,
        "backend_families": backend_families,
        "contract_versions": contract_versions,
        "routed_layer_count": len(routed_layers),
        "routed_layers": sorted(routed_layers),
        "top_k_values": sorted(top_k_values),
        "fixture_signals": trace_fixture_signals(path, events),
    }


def inventory_summary_names(row: JSONDict) -> set[str]:
    names: set[str] = set()
    for field in ("blob", "model"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            names.add(value)
            names.add(Path(value).name)
    return names


def find_inventory_summary_match(trace_summary: JSONDict, rows: list[JSONDict]) -> JSONDict | None:
    trace_names = set(trace_summary.get("model_names", []))
    trace_names.update(Path(name).name for name in trace_summary.get("model_names", []))
    for row in rows:
        if trace_names & inventory_summary_names(row):
            return row
    return None


def summarize_inventory_manifest(trace_path: Path, inventory_manifest_path: Path | None) -> JSONDict:
    if inventory_manifest_path is None:
        return {
            "path": None,
            "provided": False,
            "valid": False,
            "real_model_pair_ready": False,
            "errors": [],
            "blockers": ["scanner_inventory_manifest_missing"],
        }
    try:
        pairing = plan_real_model_trace_inventory_pairing.build_pairing_summary(
            trace_path,
            inventory_manifest_path,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {
            "path": display_path(inventory_manifest_path),
            "provided": True,
            "valid": False,
            "real_model_pair_ready": False,
            "errors": [str(exc)],
            "blockers": ["scanner_inventory_manifest_invalid"],
        }
    return {
        "path": display_path(inventory_manifest_path),
        "provided": True,
        "valid": pairing.get("valid") is True,
        "real_model_pair_ready": pairing.get("real_model_pair_ready") is True,
        "fixture_only_pair": pairing.get("fixture_only_pair"),
        "errors": pairing.get("errors", []),
        "blockers": [item.get("id") for item in pairing.get("blockers", []) if isinstance(item, dict)],
    }


def summarize_fallback(fallback_artifact_path: Path | None) -> JSONDict:
    summary = plan_dense_fallback_comparison.build_summary(fallback_artifact_path)
    return {
        "path": display_path(fallback_artifact_path) if fallback_artifact_path is not None else None,
        "provided": fallback_artifact_path is not None,
        "valid": summary.get("valid") is True,
        "comparison_ready": summary.get("comparison_ready") is True,
        "errors": summary.get("errors", []),
        "blocker": summary.get("blocker"),
    }


def candidate_commands(
    *,
    trace_path: Path,
    trace_summary: JSONDict,
    inventory_manifest_path: Path | None,
    fallback_artifact_path: Path | None,
    output_dir: Path,
    docker_ollama_volume: str | None,
    docker_image: str,
    docker_repo_mount: str,
    docker_ollama_mount: str,
) -> list[JSONDict]:
    model_id = trace_summary.get("model_names", ["<model_id>"])[0] if trace_summary.get("model_names") else "<model_id>"
    inventory_output = output_dir / f"{safe_artifact_stem(model_id)}.expert_inventory.json"
    bundle_output = output_dir / "phase3_real_evidence_bundle.json"
    commands = [
        {
            "command_class": "trace_contract_validator",
            "command": uv_command(
                "scripts/validate_llama_cpp_router_trace.py",
                trace_path,
                "--require-kind",
                "selected_experts",
                "--require-kind",
                "selected_weights",
                "--require-kind",
                "selected_weights_norm",
                "--json",
            ),
        },
        {
            "command_class": "scanner_inventory_manifest_builder",
            "command": uv_command(
                "scripts/scan_gguf_expert_inventory.py",
                model_id,
                "--model-id",
                model_id,
                "--output",
                inventory_manifest_path or inventory_output,
                "--json",
            ),
            "notes": [
                "run on the host/container where the GGUF path from the trace is readable",
                "inventory summary JSONL is not enough; Phase 3 needs a scanner-derived expert inventory manifest",
            ],
        },
    ]
    docker_command = docker_ollama_scanner_command(
        model_id=model_id,
        inventory_output=inventory_manifest_path or inventory_output,
        docker_ollama_volume=docker_ollama_volume,
        docker_image=docker_image,
        docker_repo_mount=docker_repo_mount,
        docker_ollama_mount=docker_ollama_mount,
    )
    if docker_command is not None:
        commands.append(docker_command)
    if inventory_manifest_path is not None:
        command = uv_command(
            "scripts/build_phase3_real_evidence_bundle.py",
            "--trace-path",
            trace_path,
            "--inventory-path",
            inventory_manifest_path,
            "--policies-path",
            "memory-moe-mvp/data/baseline_replay_policies.json",
            "--managed-plan-path",
            "memory-moe-mvp/data/managed_expert_loading_plan.json",
            "--model-id",
            model_id,
            "--source-format",
            "gguf",
            "--backend-family",
            "llama_cpp",
            "--prompt-family",
            "<prompt_family>",
        )
        if fallback_artifact_path is not None:
            command.extend(["--fallback-artifact-path", display_path(fallback_artifact_path) or str(fallback_artifact_path)])
        command.extend(["--output", bundle_output, "--json"])
        commands.append({"command_class": "phase3_bundle_builder", "command": command})
    return commands


def build_candidate(
    *,
    trace_path: Path,
    trace_summary: JSONDict,
    inventory_rows: list[JSONDict],
    inventory_manifest_path: Path | None,
    fallback_summary: JSONDict,
    fallback_artifact_path: Path | None,
    output_dir: Path,
    docker_ollama_volume: str | None,
    docker_image: str,
    docker_repo_mount: str,
    docker_ollama_mount: str,
) -> JSONDict:
    inventory_match = find_inventory_summary_match(trace_summary, inventory_rows)
    manifest_summary = summarize_inventory_manifest(trace_path, inventory_manifest_path)
    trace_ready = trace_summary.get("valid") is True and not trace_summary.get("fixture_signals")
    handoff_ready = (
        trace_ready
        and manifest_summary.get("real_model_pair_ready") is True
        and fallback_summary.get("comparison_ready") is True
    )
    missing: list[str] = []
    if not trace_ready:
        missing.append("non_fixture_valid_router_trace")
    if inventory_match is None and manifest_summary.get("provided") is not True:
        missing.append("matching_inventory_summary_or_model_metadata")
    if manifest_summary.get("real_model_pair_ready") is not True:
        missing.append("scanner_inventory_manifest_for_trace_model")
    if fallback_summary.get("comparison_ready") is not True:
        missing.append("dense_fallback_comparison_artifact")
    return {
        "trace": trace_summary,
        "inventory_summary_match": inventory_match,
        "inventory_manifest": manifest_summary,
        "fallback": fallback_summary,
        "phase3_handoff_ready": handoff_ready,
        "missing_for_phase3_handoff": missing,
        "commands": candidate_commands(
            trace_path=trace_path,
            trace_summary=trace_summary,
            inventory_manifest_path=inventory_manifest_path,
            fallback_artifact_path=fallback_artifact_path,
            output_dir=output_dir,
            docker_ollama_volume=docker_ollama_volume,
            docker_image=docker_image,
            docker_repo_mount=docker_repo_mount,
            docker_ollama_mount=docker_ollama_mount,
        ),
    }


def build_intake_summary(
    *,
    root: Path,
    trace_glob: str = DEFAULT_TRACE_GLOB,
    inventory_summary_path: Path | None = None,
    inventory_manifest_path: Path | None = None,
    fallback_artifact_path: Path | None = None,
    output_dir: Path = DEFAULT_ROOT,
    docker_ollama_volume: str | None = None,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    docker_repo_mount: str = DEFAULT_DOCKER_REPO_MOUNT,
    docker_ollama_mount: str = DEFAULT_DOCKER_OLLAMA_MOUNT,
) -> JSONDict:
    errors: list[str] = []
    try:
        inventory_rows = load_inventory_summary(inventory_summary_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        inventory_rows = []
        errors.append(f"inventory summary: {exc}")
    try:
        fallback_summary = summarize_fallback(fallback_artifact_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        fallback_summary = {
            "path": display_path(fallback_artifact_path) if fallback_artifact_path is not None else None,
            "provided": fallback_artifact_path is not None,
            "valid": False,
            "comparison_ready": False,
            "errors": [str(exc)],
            "blocker": {"id": "fallback_artifact_invalid"},
        }

    paths = trace_paths(root, trace_glob)
    candidates = [
        build_candidate(
            trace_path=path,
            trace_summary=summarize_trace(path),
            inventory_rows=inventory_rows,
            inventory_manifest_path=inventory_manifest_path,
            fallback_summary=fallback_summary,
            fallback_artifact_path=fallback_artifact_path,
            output_dir=output_dir,
            docker_ollama_volume=docker_ollama_volume,
            docker_image=docker_image,
            docker_repo_mount=docker_repo_mount,
            docker_ollama_mount=docker_ollama_mount,
        )
        for path in paths
    ]
    ready_count = sum(1 for candidate in candidates if candidate["phase3_handoff_ready"])
    trace_valid_count = sum(1 for candidate in candidates if candidate["trace"].get("valid") is True)
    inventory_summary_match_count = sum(1 for candidate in candidates if candidate.get("inventory_summary_match") is not None)
    return {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "mode": "phase3_artifact_intake_plan",
        "valid": not errors,
        "errors": errors,
        "root": display_path(root),
        "trace_glob": trace_glob,
        "trace_candidate_count": len(candidates),
        "trace_valid_count": trace_valid_count,
        "inventory_summary_match_count": inventory_summary_match_count,
        "phase3_handoff_ready_count": ready_count,
        "inventory_summary_path": display_path(inventory_summary_path) if inventory_summary_path is not None else None,
        "inventory_manifest_path": display_path(inventory_manifest_path) if inventory_manifest_path is not None else None,
        "fallback_artifact_path": display_path(fallback_artifact_path) if fallback_artifact_path is not None else None,
        "docker_ollama_volume": docker_ollama_volume,
        "docker_image": docker_image if docker_ollama_volume else None,
        "candidates": candidates,
        "safety_contract": [
            "planner reads local generated artifacts only",
            "planner does not launch model servers",
            "planner does not run Docker",
            "planner does not download models",
            "planner does not inspect private tokens",
            "planner does not read tensor values",
            "planner does not mutate runtime residency",
            "planner does not send prompt traffic",
            "planner does not claim live expert paging",
        ],
        "next_actions": [
            "Use discovered trace candidates as inputs to scanner-derived inventory manifests.",
            "Treat inventory summary matches as hints, not as Phase 3 inventory evidence.",
            "Use --docker-ollama-volume when traced model paths live under an Ollama Docker volume.",
            "Attach a dense/full-runtime fallback comparison before marking the handoff ready.",
            "Build and validate a Phase 3 real-evidence bundle only after pairing and fallback are ready.",
        ],
    }


def print_human_summary(summary: JSONDict) -> None:
    print("MoE Run Anyway Phase 3 artifact intake")
    print(f"Valid: {summary['valid']}")
    print(f"Trace candidates: {summary['trace_candidate_count']}")
    print(f"Valid traces: {summary['trace_valid_count']}")
    print(f"Inventory summary matches: {summary['inventory_summary_match_count']}")
    print(f"Phase 3 handoff-ready candidates: {summary['phase3_handoff_ready_count']}")
    if summary["errors"]:
        print("Errors:")
        for error in summary["errors"]:
            print(f"  - {error}")
    for candidate in summary["candidates"]:
        trace = candidate["trace"]
        print(f"  - {trace.get('path')}: ready={candidate['phase3_handoff_ready']}")
        missing = candidate.get("missing_for_phase3_handoff", [])
        if missing:
            print(f"    missing: {', '.join(missing)}")
    print("Safety contract:")
    for item in summary["safety_contract"]:
        print(f"  - {item}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--trace-glob", default=DEFAULT_TRACE_GLOB)
    parser.add_argument("--inventory-summary", type=Path)
    parser.add_argument("--inventory-manifest", type=Path)
    parser.add_argument("--fallback-artifact-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--docker-ollama-volume", help="optional Docker volume name containing Ollama's .ollama directory")
    parser.add_argument("--docker-image", default=DEFAULT_DOCKER_IMAGE, help="image used only in emitted Docker scanner commands")
    parser.add_argument("--docker-repo-mount", default=DEFAULT_DOCKER_REPO_MOUNT, help="container path for the repo in emitted Docker commands")
    parser.add_argument("--docker-ollama-mount", default=DEFAULT_DOCKER_OLLAMA_MOUNT, help="container path for the Ollama volume in emitted Docker commands")
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    return parser


def plan_paths(
    *,
    root: Path = DEFAULT_ROOT,
    trace_glob: str = DEFAULT_TRACE_GLOB,
    inventory_summary_path: Path | None = None,
    inventory_manifest_path: Path | None = None,
    fallback_artifact_path: Path | None = None,
    output_dir: Path = DEFAULT_ROOT,
    docker_ollama_volume: str | None = None,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    docker_repo_mount: str = DEFAULT_DOCKER_REPO_MOUNT,
    docker_ollama_mount: str = DEFAULT_DOCKER_OLLAMA_MOUNT,
) -> tuple[int, JSONDict | None, str | None]:
    try:
        summary = build_intake_summary(
            root=root,
            trace_glob=trace_glob,
            inventory_summary_path=inventory_summary_path,
            inventory_manifest_path=inventory_manifest_path,
            fallback_artifact_path=fallback_artifact_path,
            output_dir=output_dir,
            docker_ollama_volume=docker_ollama_volume,
            docker_image=docker_image,
            docker_repo_mount=docker_repo_mount,
            docker_ollama_mount=docker_ollama_mount,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not build Phase 3 artifact intake plan: {exc}"
    return (0 if summary["valid"] else 2), summary, None


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary, error_message = plan_paths(
        root=args.root,
        trace_glob=args.trace_glob,
        inventory_summary_path=args.inventory_summary,
        inventory_manifest_path=args.inventory_manifest,
        fallback_artifact_path=args.fallback_artifact_path,
        output_dir=args.output_dir,
        docker_ollama_volume=args.docker_ollama_volume,
        docker_image=args.docker_image,
        docker_repo_mount=args.docker_repo_mount,
        docker_ollama_mount=args.docker_ollama_mount,
    )
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