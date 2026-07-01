#!/usr/bin/env python3
"""Summarize repo-local Phase 3 real-evidence artifacts.

This matrix reads existing bundle and scanner inventory artifacts, validates the
real trace/inventory pairing and offline replay status, and reports scanner,
policy-candidate trace-receipt, and promotion coverage gaps. It does not launch runtimes, run Docker, download models, load
tensor values, mutate residency, or send prompt traffic.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from functools import lru_cache
import sys
from pathlib import Path
from typing import Any

import plan_baseline_policy_replay
import plan_expert_inventory
import plan_phase3_real_evidence_bundle
import plan_phase3_reuse_evidence_capture
import plan_real_model_trace_inventory_pairing


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "memory-moe-mvp" / "phase3-real-evidence"
SUPPORTED_SCHEMA_VERSION = "moe-phase3-real-evidence-matrix-v1"
BUNDLE_GLOB = "*phase3_real_evidence_bundle.json"
INVENTORY_GLOB = "*.expert_inventory.json"

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


def artifact_path(manifest: JSONDict, key: str) -> Path | None:
    paths = manifest.get("artifact_paths")
    if not isinstance(paths, dict):
        return None
    return resolve_repo_path(paths.get(key))


def int_count(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def blocker_id(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("id") or "unknown")
    return str(item)


def summarize_bundle(bundle_path: Path) -> JSONDict:
    try:
        manifest = plan_phase3_real_evidence_bundle.load_manifest(bundle_path)
        bundle = plan_phase3_real_evidence_bundle.build_summary(bundle_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {
            "path": display_path(bundle_path),
            "valid": False,
            "errors": [str(exc)],
            "real_model_pair_ready": False,
            "replay_valid": False,
            "bundle_ready_for_phase4": False,
            "phase3_complete": False,
        }

    trace_path = artifact_path(manifest, "trace_path")
    inventory_path = artifact_path(manifest, "inventory_path")
    policies_path = artifact_path(manifest, "policies_path")
    pairing: JSONDict | None = None
    replay: JSONDict | None = None
    errors = list(bundle.get("errors", []))

    if trace_path is not None and inventory_path is not None and trace_path.exists() and inventory_path.exists():
        try:
            pairing = plan_real_model_trace_inventory_pairing.build_pairing_summary(
                trace_path,
                inventory_path,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"pairing: {exc}")

    if (
        trace_path is not None
        and inventory_path is not None
        and policies_path is not None
        and trace_path.exists()
        and inventory_path.exists()
        and policies_path.exists()
    ):
        try:
            replay = plan_baseline_policy_replay.build_replay_summary(
                trace_path,
                inventory_path,
                policies_path,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"replay: {exc}")

    reuse_distance = replay.get("reuse_distance", {}) if replay else {}
    candidate_blockers = replay.get("policy_candidate_diagnostics", {}).get("blockers", []) if replay else []
    artifact_statuses = bundle.get("artifact_statuses", {}) if isinstance(bundle.get("artifact_statuses"), dict) else {}
    trace_receipt_status = artifact_statuses.get("trace_receipt_path", {}) if isinstance(artifact_statuses.get("trace_receipt_path"), dict) else {}
    remaining_gaps = bundle.get("remaining_gaps", {}) if isinstance(bundle.get("remaining_gaps"), dict) else {}
    no_go_reason_ids = [str(item) for item in bundle.get("no_go_reason_ids", [])]
    policy_candidate_ready = replay.get("phase_3_gate", {}).get("policy_candidate_ready") is True if replay else False
    policy_candidate_trace_receipt_ready = remaining_gaps.get("policy_candidate_trace_receipt_ready") is True
    policy_candidate_trace_receipt_required = policy_candidate_ready

    return {
        "path": display_path(bundle_path),
        "name": manifest.get("name"),
        "model_id": manifest.get("model_id"),
        "prompt_family": manifest.get("prompt_family"),
        "valid": bundle.get("valid") is True and not errors,
        "errors": errors,
        "bundle_ready_for_phase4": bundle.get("bundle_ready_for_phase4") is True,
        "phase3_complete": bundle.get("phase3_complete") is True,
        "no_go_reason_ids": no_go_reason_ids,
        "trace_path": display_path(trace_path),
        "inventory_path": display_path(inventory_path),
        "fallback_attached": artifact_statuses.get("fallback_artifact_path", {}).get("exists") is True,
        "trace_receipt_attached": trace_receipt_status.get("exists") is True,
        "trace_receipt_path": trace_receipt_status.get("path"),
        "trace_receipt_status": trace_receipt_status.get("status"),
        "policy_candidate_trace_receipt_required": policy_candidate_trace_receipt_required,
        "policy_candidate_trace_receipt_ready": policy_candidate_trace_receipt_ready,
        "policy_candidate_trace_receipt_blocked": (
            policy_candidate_trace_receipt_required and not policy_candidate_trace_receipt_ready
        ),
        "real_model_pair_ready": pairing.get("real_model_pair_ready") is True if pairing else False,
        "fixture_only_pair": pairing.get("fixture_only_pair") if pairing else None,
        "joined_route_count": pairing.get("joined_route_count", 0) if pairing else 0,
        "unique_expert_count": replay.get("unique_expert_count", 0) if replay else 0,
        "unique_estimated_residency_bytes": replay.get("unique_estimated_residency_bytes", 0) if replay else 0,
        "replay_valid": replay.get("valid") is True if replay else False,
        "reuse_distance_observations": int_count(reuse_distance.get("observations")) if isinstance(reuse_distance, dict) else 0,
        "candidate_policy_ids": replay.get("candidate_policy_ids", []) if replay else [],
        "policy_candidate_ready": policy_candidate_ready,
        "policy_candidate_blockers": candidate_blockers,
        "policy_candidate_blocker_ids": sorted({blocker_id(item) for item in candidate_blockers}),
        "missing_for_phase3_completion": replay.get("phase_3_gate", {}).get("missing_for_phase_3_completion", []) if replay else [],
    }


def summarize_inventory(path: Path, referenced_paths: set[Path]) -> JSONDict:
    try:
        manifest = plan_expert_inventory.load_manifest(path)
        summary = plan_expert_inventory.build_summary(manifest, path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {
            "path": display_path(path),
            "valid": False,
            "errors": [str(exc)],
            "error_count": 1,
            "referenced_by_bundle": path.resolve() in referenced_paths,
        }
    return {
        "path": display_path(path),
        "model_id": manifest.get("model_id"),
        "inventory_scope": manifest.get("inventory_scope"),
        "valid": summary.get("valid") is True,
        "errors": summary.get("errors", []),
        "error_count": len(summary.get("errors", [])),
        "expert_count": summary.get("expert_count", 0),
        "component_count": summary.get("component_count", 0),
        "total_estimated_residency_bytes": summary.get("total_estimated_residency_bytes", 0),
        "referenced_by_bundle": path.resolve() in referenced_paths,
    }


@lru_cache(maxsize=4)
def build_matrix(root: Path = DEFAULT_ROOT) -> JSONDict:
    errors: list[str] = []
    if not root.exists():
        errors.append(f"evidence root does not exist: {display_path(root)}")
        bundle_paths: list[Path] = []
        inventory_paths: list[Path] = []
    else:
        bundle_paths = sorted(root.glob(BUNDLE_GLOB))
        inventory_paths = sorted(root.glob(INVENTORY_GLOB))

    bundles = [summarize_bundle(path) for path in bundle_paths]
    referenced_paths: set[Path] = set()
    for bundle_path in bundle_paths:
        try:
            manifest = plan_phase3_real_evidence_bundle.load_manifest(bundle_path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        inventory_path = artifact_path(manifest, "inventory_path")
        if inventory_path is not None:
            referenced_paths.add(inventory_path.resolve())

    inventories = [summarize_inventory(path, referenced_paths) for path in inventory_paths]
    ready_pairs = [bundle for bundle in bundles if bundle.get("real_model_pair_ready") is True]
    replay_ready = [bundle for bundle in bundles if bundle.get("replay_valid") is True]
    invalid_inventories = [item for item in inventories if item.get("valid") is not True]
    phase4_ready = [bundle for bundle in bundles if bundle.get("bundle_ready_for_phase4") is True]
    policy_candidate_ready = [bundle for bundle in bundles if bundle.get("policy_candidate_ready") is True]
    policy_candidate_blocked = [bundle for bundle in bundles if bundle.get("policy_candidate_ready") is not True]
    trace_receipt_attached = [bundle for bundle in bundles if bundle.get("trace_receipt_attached") is True]
    trace_receipt_required = [bundle for bundle in bundles if bundle.get("policy_candidate_trace_receipt_required") is True]
    trace_receipt_ready = [bundle for bundle in bundles if bundle.get("policy_candidate_trace_receipt_ready") is True]
    trace_receipt_blocked = [bundle for bundle in trace_receipt_required if bundle.get("policy_candidate_trace_receipt_ready") is not True]
    policy_blocker_counts = Counter(
        blocker_id(blocker)
        for bundle in bundles
        for blocker in bundle.get("policy_candidate_blockers", [])
    )
    candidate_policy_ids = sorted(
        {
            policy_id
            for bundle in bundles
            for policy_id in bundle.get("candidate_policy_ids", [])
        }
    )
    no_go_reason_counts = Counter(
        reason_id
        for bundle in bundles
        for reason_id in bundle.get("no_go_reason_ids", [])
    )
    reuse_summary = plan_phase3_reuse_evidence_capture.build_root_summary(root) if root.exists() else {
        "prompt_identity_ready_count": 0,
        "prompt_identity_metadata_missing_count": 0,
        "blocker_counts": {},
    }
    reuse_blocker_counts = reuse_summary.get("blocker_counts", {})
    if isinstance(reuse_blocker_counts, dict):
        for blocker, count in reuse_blocker_counts.items():
            blocker_key = str(blocker)
            policy_blocker_counts[blocker_key] = max(policy_blocker_counts[blocker_key], int_count(count))

    return {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "mode": "phase3_real_evidence_matrix",
        "root": display_path(root),
        "valid": not errors and all(bundle.get("valid") is True for bundle in bundles),
        "errors": errors,
        "bundle_count": len(bundles),
        "real_model_pair_ready_count": len(ready_pairs),
        "replay_valid_count": len(replay_ready),
        "policy_candidate_ready_count": len(policy_candidate_ready),
        "policy_candidate_trace_receipt_attached_count": len(trace_receipt_attached),
        "policy_candidate_trace_receipt_required_count": len(trace_receipt_required),
        "policy_candidate_trace_receipt_ready_count": len(trace_receipt_ready),
        "policy_candidate_trace_receipt_blocked_bundle_count": len(trace_receipt_blocked),
        "policy_candidate_blocked_bundle_count": len(policy_candidate_blocked),
        "policy_candidate_no_reuse_distance_observation_count": sum(
            1
            for bundle in policy_candidate_blocked
            if "no_reuse_distance_observations" in bundle.get("policy_candidate_blocker_ids", [])
        ),
        "policy_candidate_prompt_identity_ready_count": int_count(
            reuse_summary.get("prompt_identity_ready_count")
        ),
        "policy_candidate_prompt_identity_metadata_missing_count": int_count(
            reuse_summary.get("prompt_identity_metadata_missing_count")
        ),
        "policy_candidate_blocker_counts": dict(sorted(policy_blocker_counts.items())),
        "no_go_reason_counts": dict(sorted(no_go_reason_counts.items())),
        "candidate_policy_ids": candidate_policy_ids,
        "phase4_ready_bundle_count": len(phase4_ready),
        "inventory_count": len(inventories),
        "valid_inventory_count": sum(1 for item in inventories if item.get("valid") is True),
        "invalid_inventory_count": len(invalid_inventories),
        "bundles": bundles,
        "inventories": inventories,
        "remaining_blockers": sorted(
            {
                blocker
                for bundle in bundles
                for blocker in bundle.get("missing_for_phase3_completion", [])
            }
        ),
        "scanner_coverage_gap_inventories": [
            {
                "path": item.get("path"),
                "error_count": item.get("error_count"),
                "first_error": item.get("errors", [None])[0] if item.get("errors") else None,
                "referenced_by_bundle": item.get("referenced_by_bundle"),
            }
            for item in invalid_inventories
        ],
        "safety_contract": [
            "matrix reads local evidence artifacts only",
            "matrix does not launch model servers",
            "matrix does not run Docker",
            "matrix does not download models",
            "matrix does not load tensor values",
            "matrix does not mutate runtime residency",
            "matrix does not send prompt traffic",
            "matrix does not claim live expert paging",
        ],
    }


def print_human_summary(summary: JSONDict) -> None:
    print("MoE Run Anyway Phase 3 real-evidence matrix")
    print(f"Valid: {summary['valid']}")
    print(f"Bundles: {summary['bundle_count']}")
    print(f"Real-model pairs ready: {summary['real_model_pair_ready_count']}")
    print(f"Replay-valid bundles: {summary['replay_valid_count']}")
    print(f"Policy-candidate-ready bundles: {summary['policy_candidate_ready_count']}")
    print(f"Policy-candidate prompt-identity-ready traces: {summary['policy_candidate_prompt_identity_ready_count']}")
    print(f"Policy-candidate prompt-identity blockers: {summary['policy_candidate_prompt_identity_metadata_missing_count']}")
    print(
        "Policy-candidate trace receipts: "
        f"{summary['policy_candidate_trace_receipt_ready_count']} ready / "
        f"{summary['policy_candidate_trace_receipt_required_count']} required "
        f"({summary['policy_candidate_trace_receipt_attached_count']} attached)"
    )
    print(f"Phase 4-ready bundles: {summary['phase4_ready_bundle_count']}")
    print(f"Inventories: {summary['valid_inventory_count']} valid, {summary['invalid_inventory_count']} invalid")
    if summary["remaining_blockers"]:
        print("Remaining blockers:")
        for blocker in summary["remaining_blockers"]:
            print(f"  - {blocker}")
    if summary["scanner_coverage_gap_inventories"]:
        print("Scanner coverage gaps:")
        for item in summary["scanner_coverage_gap_inventories"]:
            print(f"  - {item['path']}: {item['first_error']}")
    print("Bundles:")
    for bundle in summary["bundles"]:
        print(
            "  - "
            f"{bundle.get('name')}: pair={bundle.get('real_model_pair_ready')} "
            f"replay={bundle.get('replay_valid')} candidate={bundle.get('policy_candidate_ready')} "
            f"receipt={bundle.get('policy_candidate_trace_receipt_ready')} "
            f"routes={bundle.get('joined_route_count')} unique_experts={bundle.get('unique_expert_count')} "
            f"phase4={bundle.get('bundle_ready_for_phase4')}"
        )
    print("Safety contract:")
    for item in summary["safety_contract"]:
        print(f"  - {item}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    return parser


def plan_paths(root: Path = DEFAULT_ROOT) -> tuple[int, JSONDict | None, str | None]:
    try:
        summary = build_matrix(root)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return 2, None, f"Could not build Phase 3 real-evidence matrix: {exc}"
    return (0 if summary["valid"] else 2), summary, None


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    status, summary, error_message = plan_paths(args.root)
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
