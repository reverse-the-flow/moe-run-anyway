#!/usr/bin/env python3
"""Plan the next Phase 3 capture needed to prove routed-expert reuse.

The Phase 3 policy replay gate needs more than a valid router trace: it needs a
route stream where at least one routed expert key appears again later, so a
managed residency policy can demonstrate warm-hit behavior. This planner audits
local candidate traces and explains the next capture requirements. It does not
launch runtimes, call endpoints, inspect secrets, mutate residency, or send
prompt traffic.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import phase3_trace_receipts
import plan_trace_inventory_replay
import validate_llama_cpp_router_trace


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REAL_EVIDENCE_ROOT = ROOT / "memory-moe-mvp" / "phase3-real-evidence"
SUPPORTED_SCHEMA_VERSION = "moe-phase3-reuse-evidence-capture-plan-v1"
MIN_REUSE_DISTANCE_OBSERVATIONS = 1
REQUIRED_TRACE_KINDS = {"selected_experts", "selected_weights", "selected_weights_norm"}
PROMPT_IDENTITY_FIELDS = (
    "prompt_id",
    "prompt_group_id",
    "group_id",
    "repeat",
    "repeat_index",
    "request_id",
    "capture_run_id",
    "capture_sequence",
)
PROMPT_IDENTITY_REQUIRED_FIELD_GROUPS = (
    {
        "id": "prompt_boundary",
        "fields": ("prompt_id", "request_id"),
        "description": "each event must name the prompt/request boundary it came from",
    },
    {
        "id": "capture_order",
        "fields": ("capture_sequence", "repeat_index", "repeat"),
        "description": "each event must carry enough ordering metadata to reconstruct repeated captures",
    },
)
PROMPT_IDENTITY_RUN_FIELDS = ("capture_run_id",)

JSONDict = dict[str, Any]


def display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_repo_path(value: Any) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> JSONDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{display_path(path) or path} must be a JSON object")
    return payload


def bundle_paths(root: Path) -> list[Path]:
    return sorted(root.glob("*_phase3_real_evidence_bundle.json"), key=lambda path: path.name)


def canonical_prompt_set_path(bundle_path: Path) -> Path:
    return bundle_path.with_name(f"{bundle_path.stem}.prompt-set.json")


def canonical_request_path(bundle_path: Path) -> Path:
    return bundle_path.with_name(f"{bundle_path.stem}.runtime-capture-request.json")


def canonical_candidate_trace_path(bundle_path: Path) -> Path:
    return bundle_path.with_name(f"{bundle_path.stem}-policy-candidate") / "candidate-router-events.jsonl"


def canonical_candidate_receipt_path(candidate_trace_path: Path) -> Path:
    return candidate_trace_path.with_name(f"{candidate_trace_path.stem}.capture-receipt.json")


def manifest_artifact_path(manifest: JSONDict, key: str) -> Path | None:
    paths = manifest.get("artifact_paths")
    if not isinstance(paths, dict):
        return None
    return resolve_repo_path(paths.get(key))


def prompt_identity_fields(events: list[JSONDict]) -> list[str]:
    present: set[str] = set()
    for event in events:
        for field in PROMPT_IDENTITY_FIELDS:
            value = event.get(field)
            if value is not None and value != "":
                present.add(field)
    return sorted(present)


def event_has_any_field(event: JSONDict, fields: tuple[str, ...]) -> bool:
    return any(event.get(field) is not None and event.get(field) != "" for field in fields)


def prompt_identity_summary(events: list[JSONDict]) -> JSONDict:
    field_counts: Counter[str] = Counter()
    for event in events:
        for field in PROMPT_IDENTITY_FIELDS:
            value = event.get(field)
            if value is not None and value != "":
                field_counts[field] += 1

    group_results: list[JSONDict] = []
    missing_group_ids: list[str] = []
    for group in PROMPT_IDENTITY_REQUIRED_FIELD_GROUPS:
        fields = tuple(str(field) for field in group["fields"])
        missing_count = sum(1 for event in events if not event_has_any_field(event, fields))
        group_ready = bool(events) and missing_count == 0
        if not group_ready:
            missing_group_ids.append(str(group["id"]))
        group_results.append(
            {
                "id": group["id"],
                "description": group["description"],
                "fields": list(fields),
                "ready": group_ready,
                "missing_event_count": missing_count,
            }
        )

    run_field_present = any(event_has_any_field(event, PROMPT_IDENTITY_RUN_FIELDS) for event in events)
    if not run_field_present:
        missing_group_ids.append("capture_run")

    ready = bool(events) and not missing_group_ids
    return {
        "ready": ready,
        "event_count": len(events),
        "fields_present": sorted(field_counts),
        "field_counts": dict(sorted(field_counts.items())),
        "required_field_groups": group_results,
        "run_fields": list(PROMPT_IDENTITY_RUN_FIELDS),
        "run_field_present": run_field_present,
        "missing_required_group_ids": sorted(set(missing_group_ids)),
        "events_with_any_identity_count": sum(
            1 for event in events if event_has_any_field(event, PROMPT_IDENTITY_FIELDS)
        ),
    }


def route_repetition_summary(routes: list[JSONDict]) -> JSONDict:
    counts: Counter[tuple[str, str]] = Counter((str(row["layer_id"]), str(row["expert_id"])) for row in routes)
    repeated = {key: count for key, count in counts.items() if count > 1}
    repeated_observations = sum(count - 1 for count in repeated.values())
    samples = [
        {"layer_id": layer_id, "expert_id": expert_id, "route_count": count}
        for (layer_id, expert_id), count in sorted(repeated.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))[:12]
    ]
    return {
        "route_count": len(routes),
        "unique_route_key_count": len(counts),
        "repeated_route_key_count": len(repeated),
        "repeated_route_observation_count": repeated_observations,
        "repeated_route_key_samples": samples,
    }


def receipt_summary(
    receipt_path: Path,
    *,
    request_path: Path,
    prompt_set_path: Path,
    candidate_trace_path: Path,
) -> JSONDict:
    if not receipt_path.exists():
        return {
            "path": display_path(receipt_path),
            "exists": False,
            "ready": False,
            "errors": [],
            "blockers": ["candidate_trace_capture_receipt_missing"],
        }
    receipt = load_json(receipt_path)
    validation = phase3_trace_receipts.validate_trace_receipt_source_binding(
        receipt,
        repo_root=ROOT,
        expected_source_request_path=request_path,
        expected_source_prompt_set_path=prompt_set_path,
        expected_candidate_trace_path=candidate_trace_path,
    )
    blockers: list[str] = []
    if validation["receipt_ready"] is not True:
        blockers.append("candidate_trace_capture_receipt_not_ready")
    if validation["shape_valid"] is not True:
        blockers.append("candidate_trace_capture_receipt_invalid")
    return {
        "path": display_path(receipt_path),
        "exists": True,
        "ready": validation["receipt_ready"] is True,
        "valid": validation["shape_valid"],
        "errors": validation["errors"],
        "blockers": blockers,
        "candidate_trace_path_matches": validation["candidate_trace_path_matches"],
        "source_prompt_set_path_matches": validation["source_prompt_set_path_matches"],
        "source_request_path_matches": validation["source_request_path_matches"],
    }


def trace_summary(
    trace_path: Path,
    *,
    receipt_path: Path,
    request_path: Path,
    prompt_set_path: Path,
) -> JSONDict:
    receipt = receipt_summary(
        receipt_path,
        request_path=request_path,
        prompt_set_path=prompt_set_path,
        candidate_trace_path=trace_path,
    )
    if not trace_path.exists():
        return {
            "path": display_path(trace_path),
            "exists": False,
            "valid": False,
            "reuse_ready": False,
            "errors": [],
            "blockers": ["candidate_trace_missing"],
            "receipt": receipt,
        }

    errors: list[str] = []
    events: list[JSONDict] = []
    routes: list[JSONDict] = []
    try:
        events = validate_llama_cpp_router_trace.load_jsonl(trace_path)
        errors.extend(validate_llama_cpp_router_trace.validate_trace(events, require_kinds=REQUIRED_TRACE_KINDS))
        selected_errors: list[str] = []
        selected_events = plan_trace_inventory_replay.selected_expert_events(events, selected_errors)
        errors.extend(f"selected expert events: {error}" for error in selected_errors)
        routes = plan_trace_inventory_replay.build_route_rows(selected_events)
    except ValueError as exc:
        errors.append(str(exc))

    reuse = plan_trace_inventory_replay.reuse_distance_summary(routes) if routes else {
        "observations": 0,
        "first_touch_count": 0,
        "min": None,
        "mean": None,
        "max": None,
    }
    repetition = route_repetition_summary(routes)
    identity = prompt_identity_summary(events)
    blockers: list[str] = []
    if errors:
        blockers.append("candidate_trace_contract_invalid")
    if receipt.get("ready") is not True:
        blockers.append("candidate_trace_capture_receipt_not_ready")
    if int(reuse.get("observations", 0) or 0) < MIN_REUSE_DISTANCE_OBSERVATIONS:
        blockers.append("no_reuse_distance_observations")
    if repetition["repeated_route_key_count"] == 0:
        blockers.append("no_repeated_route_keys")
    if identity["ready"] is not True:
        blockers.append("prompt_identity_metadata_missing")

    reuse_ready = (
        not errors
        and receipt.get("ready") is True
        and int(reuse.get("observations", 0) or 0) >= MIN_REUSE_DISTANCE_OBSERVATIONS
        and identity["ready"] is True
    )
    classification = "reuse_evidence_ready" if reuse_ready else "valid_single_pass_or_no_reuse"
    if errors:
        classification = "invalid_trace_contract"
    elif not trace_path.exists():
        classification = "missing_trace"
    elif identity["ready"] is not True:
        classification = "trace_missing_prompt_identity"

    return {
        "path": display_path(trace_path),
        "exists": True,
        "valid": not errors,
        "reuse_ready": reuse_ready,
        "classification": classification,
        "errors": errors,
        "blockers": blockers,
        "warnings": [],
        "receipt": receipt,
        "event_count": len(events),
        "selected_expert_event_count": sum(1 for event in events if event.get("tensor_kind") == "selected_experts"),
        "route_repetition": repetition,
        "reuse_distance": reuse,
        "prompt_identity_ready": identity["ready"],
        "prompt_identity": identity,
        "prompt_identity_fields_present": identity["fields_present"],
    }


def summarize_bundle(bundle_path: Path) -> JSONDict:
    manifest = load_json(bundle_path)
    prompt_set_path = canonical_prompt_set_path(bundle_path)
    request_path = canonical_request_path(bundle_path)
    candidate_trace_path = canonical_candidate_trace_path(bundle_path)
    candidate_receipt_path = canonical_candidate_receipt_path(candidate_trace_path)
    inventory_path = manifest_artifact_path(manifest, "inventory_path")
    summary = trace_summary(
        candidate_trace_path,
        receipt_path=candidate_receipt_path,
        request_path=request_path,
        prompt_set_path=prompt_set_path,
    )
    return {
        "name": manifest.get("name"),
        "bundle_path": display_path(bundle_path),
        "model_id": manifest.get("model_id"),
        "backend_family": manifest.get("backend_family"),
        "prompt_family": manifest.get("prompt_family"),
        "prompt_set_path": display_path(prompt_set_path),
        "runtime_capture_request_path": display_path(request_path),
        "inventory_path": display_path(inventory_path),
        "candidate_trace_path": display_path(candidate_trace_path),
        "candidate_trace_receipt_path": display_path(candidate_receipt_path),
        "candidate_trace": summary,
    }


def rank_next_capture(bundle_summaries: list[JSONDict]) -> JSONDict | None:
    not_ready = [item for item in bundle_summaries if item["candidate_trace"].get("reuse_ready") is not True]
    if not not_ready:
        return None
    def rank_key(item: JSONDict) -> tuple[int, str]:
        name = str(item.get("name") or "")
        return (0 if "Mixtral" in name else 1, name)
    selected = sorted(not_ready, key=rank_key)[0]
    return {
        "bundle_name": selected.get("name"),
        "bundle_path": selected.get("bundle_path"),
        "model_id": selected.get("model_id"),
        "candidate_trace_path": selected.get("candidate_trace_path"),
        "prompt_set_path": selected.get("prompt_set_path"),
        "runtime_capture_request_path": selected.get("runtime_capture_request_path"),
        "selection_rationale": "mixtral_first_known_sparse_baseline" if "Mixtral" in str(selected.get("name")) else "stable_name_order",
        "required_result": "capture_candidate_router_trace_with_reuse_distance",
    }


def capture_contract() -> JSONDict:
    return {
        "minimum_reuse_distance_observations": MIN_REUSE_DISTANCE_OBSERVATIONS,
        "must_preserve_trace_kinds": sorted(REQUIRED_TRACE_KINDS),
        "must_bind_ready_receipt": True,
        "must_use_same_model_backend_and_inventory": True,
        "must_bind_prompt_identity_metadata": True,
        "recommended_prompt_identity_fields": list(PROMPT_IDENTITY_FIELDS),
        "required_prompt_identity_field_groups": [
            {
                "id": group["id"],
                "fields": list(group["fields"]),
                "description": group["description"],
            }
            for group in PROMPT_IDENTITY_REQUIRED_FIELD_GROUPS
        ],
        "required_prompt_identity_run_fields": list(PROMPT_IDENTITY_RUN_FIELDS),
        "capture_shape": [
            "Use the exact Phase 3 prompt set for the selected bundle.",
            "Run at least two repeated prompt cases into one saved candidate router-trace JSONL stream, or preserve a deterministic route order across appended traces.",
            "Keep selected_experts, selected_weights, and selected_weights_norm events for every captured layer.",
            "Add prompt_id or request_id plus repeat/order metadata to every emitted event, and include a capture_run_id for the trace.",
            "After capture, import with scripts/import_phase3_candidate_router_trace.py and rerun the policy-candidate planner.",
        ],
    }


def safety_contract() -> list[str]:
    return [
        "reuse-evidence planner reads local metadata and JSONL traces only",
        "reuse-evidence planner does not launch model servers",
        "reuse-evidence planner does not call endpoints",
        "reuse-evidence planner does not run Docker",
        "reuse-evidence planner does not download models",
        "reuse-evidence planner does not inspect private tokens",
        "reuse-evidence planner does not send prompt traffic",
        "reuse-evidence planner does not mutate runtime residency",
        "reuse-evidence planner does not claim live expert paging",
    ]


def build_root_summary(root: Path = DEFAULT_REAL_EVIDENCE_ROOT) -> JSONDict:
    bundles = [summarize_bundle(path) for path in bundle_paths(root)]
    errors = [
        f"{bundle['bundle_path']}: {error}"
        for bundle in bundles
        for error in bundle["candidate_trace"].get("errors", [])
    ]
    blocker_counts: Counter[str] = Counter(
        blocker
        for bundle in bundles
        for blocker in bundle["candidate_trace"].get("blockers", [])
    )
    warning_counts: Counter[str] = Counter(
        warning
        for bundle in bundles
        for warning in bundle["candidate_trace"].get("warnings", [])
    )
    reuse_ready_count = sum(1 for bundle in bundles if bundle["candidate_trace"].get("reuse_ready") is True)
    receipt_ready_count = sum(1 for bundle in bundles if bundle["candidate_trace"].get("receipt", {}).get("ready") is True)
    return {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "mode": "phase3_reuse_evidence_capture_plan",
        "valid": not errors,
        "errors": errors,
        "root": display_path(root),
        "bundle_count": len(bundles),
        "reuse_ready_count": reuse_ready_count,
        "reuse_blocked_count": len(bundles) - reuse_ready_count,
        "candidate_trace_receipt_ready_count": receipt_ready_count,
        "candidate_trace_valid_count": sum(1 for bundle in bundles if bundle["candidate_trace"].get("valid") is True),
        "no_reuse_distance_observation_count": blocker_counts.get("no_reuse_distance_observations", 0),
        "prompt_identity_ready_count": sum(1 for bundle in bundles if bundle["candidate_trace"].get("prompt_identity_ready") is True),
        "prompt_identity_metadata_missing_count": blocker_counts.get("prompt_identity_metadata_missing", 0),
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
        "recommended_next_capture": rank_next_capture(bundles),
        "capture_contract": capture_contract(),
        "bundles": bundles,
        "safety_contract": safety_contract(),
        "next_actions": [
            "Treat the current candidate traces as hook-valid but not reuse-ready unless reuse_ready_count increases.",
            "Recapture the recommended bundle with repeated prompt cases and selected expert ids/weights in one ordered route stream.",
            "Import the new trace with scripts/import_phase3_candidate_router_trace.py, then rerun policy replay and go/no-go.",
            "Capture managed and dense/full-runtime output summaries for the same prompt ids after explicit approval.",
        ],
    }


def build_markdown_report(summary: JSONDict) -> str:
    lines = [
        "# Phase 3 Reuse Evidence Capture Plan",
        "",
        f"- Valid: `{summary['valid']}`",
        f"- Bundle count: `{summary['bundle_count']}`",
        f"- Reuse-ready traces: `{summary['reuse_ready_count']}`",
        f"- Receipt-ready candidate traces: `{summary['candidate_trace_receipt_ready_count']}`",
        f"- No-reuse-distance blockers: `{summary['no_reuse_distance_observation_count']}`",
        f"- Prompt-identity-ready traces: `{summary['prompt_identity_ready_count']}`",
        f"- Prompt-identity blockers: `{summary['prompt_identity_metadata_missing_count']}`",
        "",
        "## Recommended Next Capture",
        "",
    ]
    recommendation = summary.get("recommended_next_capture")
    if isinstance(recommendation, dict):
        lines.extend(
            [
                f"- Bundle: `{recommendation.get('bundle_name')}`",
                f"- Required result: `{recommendation.get('required_result')}`",
                f"- Trace path: `{recommendation.get('candidate_trace_path')}`",
                f"- Prompt set: `{recommendation.get('prompt_set_path')}`",
                "",
            ]
        )
    else:
        lines.extend(["- None; all candidate traces are reuse-ready.", ""])

    lines.extend([
        "## Bundle Status",
        "",
        "| Bundle | Trace valid | Receipt ready | Prompt identity | Reuse ready | Reuse observations | Repeated keys | Blockers |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for bundle in summary["bundles"]:
        trace = bundle["candidate_trace"]
        repetition = trace.get("route_repetition", {}) if isinstance(trace, dict) else {}
        reuse = trace.get("reuse_distance", {}) if isinstance(trace, dict) else {}
        blockers = ", ".join(trace.get("blockers", [])) if isinstance(trace, dict) else ""
        lines.append(
            "| "
            f"`{bundle.get('name')}` | "
            f"`{trace.get('valid')}` | "
            f"`{trace.get('receipt', {}).get('ready')}` | "
            f"`{trace.get('prompt_identity_ready')}` | "
            f"`{trace.get('reuse_ready')}` | "
            f"`{reuse.get('observations', 0)}` | "
            f"`{repetition.get('repeated_route_key_count', 0)}` | "
            f"`{blockers}` |"
        )

    lines.extend([
        "",
        "## Capture Contract",
        "",
        f"- Minimum reuse-distance observations: `{summary['capture_contract']['minimum_reuse_distance_observations']}`",
        f"- Required trace kinds: `{', '.join(summary['capture_contract']['must_preserve_trace_kinds'])}`",
        f"- Prompt identity required: `{summary['capture_contract']['must_bind_prompt_identity_metadata']}`",
        f"- Recommended prompt identity fields: `{', '.join(summary['capture_contract']['recommended_prompt_identity_fields'])}`",
        "",
        "## Safety Contract",
        "",
    ])
    lines.extend(f"- {item}" for item in summary["safety_contract"])
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_REAL_EVIDENCE_ROOT)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output-md", type=Path)
    return parser


def plan_paths(root: Path) -> tuple[int, JSONDict | None, str | None]:
    try:
        summary = build_root_summary(root)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not build Phase 3 reuse-evidence capture plan: {exc}"
    return (0 if summary["valid"] else 2), summary, None


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary, error_message = plan_paths(args.root)
    if error_message:
        print(error_message, file=sys.stderr)
        return status
    assert summary is not None
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(build_markdown_report(summary), encoding="utf-8")
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("MoE Run Anyway Phase 3 reuse-evidence capture plan")
        print(f"Valid: {summary['valid']}")
        print(f"Reuse-ready traces: {summary['reuse_ready_count']} / {summary['bundle_count']}")
        print(f"Receipt-ready candidate traces: {summary['candidate_trace_receipt_ready_count']} / {summary['bundle_count']}")
        print(f"Prompt-identity-ready traces: {summary['prompt_identity_ready_count']} / {summary['bundle_count']}")
        print(f"Prompt-identity blockers: {summary['prompt_identity_metadata_missing_count']}")
        recommendation = summary.get("recommended_next_capture")
        if isinstance(recommendation, dict):
            print(f"Recommended next capture: {recommendation['bundle_name']}")
            print(f"Required result: {recommendation['required_result']}")
        if summary["errors"]:
            print("Errors:")
            for error in summary["errors"]:
                print(f"  - {error}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())