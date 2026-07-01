#!/usr/bin/env python3
"""Run dependency-free local readiness checks for the uploadable project."""

from __future__ import annotations

import argparse
import py_compile
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MVP_DIR = ROOT / "memory-moe-mvp"
WINDOWS_ABSOLUTE_PATH = re.compile(r"\b[A-Za-z]:[\\/]")

PRIMARY_DOCS = [
    ROOT / "README.md",
    ROOT / "MODEL_CARD.md",
    ROOT / "MODEL_TESTS_STATUS.md",
    ROOT / "GIT_BASELINE_STATUS.md",
    ROOT / "UPLOAD_READINESS.md",
    MVP_DIR / "README.md",
    MVP_DIR / "docs" / "live-model-readiness-planner.md",
    MVP_DIR / "docs" / "live-baseline-runner.md",
    MVP_DIR / "docs" / "model-plane-manifest-consumption.md",
    MVP_DIR / "docs" / "expert-paging-roadmap.md",
    MVP_DIR / "docs" / "expert-paging-step-breakdown.md",
    MVP_DIR / "docs" / "hookable-progression.md",
    MVP_DIR / "docs" / "llama-cpp-engine-hook-track.md",
    MVP_DIR / "docs" / "llama-cpp-engine-hook-traces-2026-06-25.md",
    MVP_DIR / "docs" / "phase-2-repeat-protocol.md",
    MVP_DIR / "docs" / "flash-moe-inspiration-next-step.md",
    MVP_DIR / "phase3-real-evidence" / "README.md",
    MVP_DIR / "docs" / "pc-llama-debug-mixtral-hook-smoke-2026-06-26.md",
    MVP_DIR / "docs" / "hookable-moe-attempts-pc-gx10-2026-06-25.md",
    MVP_DIR / "docs" / "managed-expert-loading.md",
    MVP_DIR / "docs" / "edge-hardware-quickstart.md",
    MVP_DIR / "docs" / "model-target-test-plan.md",
    MVP_DIR / "docs" / "portability-and-gpu-hosts.md",
    MVP_DIR / "docs" / "probe-observability-notes.md",
    MVP_DIR / "docs" / "controller-architecture.md",
]


def run_command(command: list[str], *, cwd: Path = ROOT) -> int:
    print(f"$ {' '.join(command)}")
    completed = subprocess.run(command, cwd=cwd)
    return completed.returncode


def validate_model_target_registry() -> int:
    return run_command([sys.executable, str(MVP_DIR / "model_target_registry.py")])


def validate_expert_paging_roadmap() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "plan_expert_paging.py")])


def validate_managed_expert_loading_plan() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "plan_managed_expert_loading.py")])


def validate_expert_inventory_manifest() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "plan_expert_inventory.py")])


def validate_expert_store_layout() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "plan_expert_store_layout.py")])



def validate_trace_inventory_replay_plan() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "plan_trace_inventory_replay.py")])


def validate_real_model_trace_inventory_pairing() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "plan_real_model_trace_inventory_pairing.py")])



def validate_baseline_replay_policies() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "plan_baseline_replay_policies.py")])


def validate_baseline_policy_replay() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "plan_baseline_policy_replay.py")])


def validate_dense_fallback_comparison() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "plan_dense_fallback_comparison.py")])


def validate_dense_fallback_comparison_builder() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "build_dense_fallback_comparison.py"), "--template", "--json"])


def validate_phase3_prompt_set_builder() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "build_phase3_prompt_set.py"), "--json"])


def validate_phase3_output_summary_builder() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "build_phase3_output_summary.py"), "--json"])


def validate_phase3_trace_receipt_builder() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "build_phase3_trace_receipt.py"), "--json"])


def validate_phase3_runtime_capture_request_builder() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "build_phase3_runtime_capture_request.py"), "--json"])


def validate_phase3_runtime_capture_request_audit() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "plan_phase3_runtime_capture_request.py")])


def validate_phase3_runtime_capture_command_contract() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "plan_phase3_runtime_capture_commands.py")])


def validate_phase3_launch_card_library() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "plan_phase3_launch_card_library.py")])


def validate_phase3_capture_result_intake() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "plan_phase3_capture_result_intake.py")])

def validate_phase3_live_capability_proof_template_builder() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "build_phase3_live_capability_proof_template.py"), "--json"])

def validate_phase3_go_no_go() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "plan_phase3_go_no_go.py")])


def validate_phase3_live_capability_proof() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "plan_phase3_live_capability_proof.py")])


def validate_phase3_runtime_actuator_design() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "plan_phase3_runtime_actuator_design.py")])


def validate_phase3_runtime_actuator_spike() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "plan_phase3_runtime_actuator_spike.py")])


def validate_phase3_evidence_packet() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "plan_phase3_evidence_packet.py")])


def validate_phase3_operator_handoff() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "plan_phase3_operator_handoff.py")])


def validate_phase3_capture_completion_receipt() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "plan_phase3_capture_completion_receipt.py"), "--json"])


def validate_phase3_dense_fallback_capture() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "plan_phase3_dense_fallback_capture.py")])


def validate_phase3_policy_candidate_trace() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "plan_phase3_policy_candidate_trace.py")])


def validate_phase3_real_evidence_capture() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "plan_phase3_real_evidence_capture.py")])

def validate_phase3_artifact_intake() -> int:
    return run_command([
        sys.executable,
        str(ROOT / "scripts" / "plan_phase3_artifact_intake.py"),
        "--root",
        str(MVP_DIR / "data"),
        "--trace-glob",
        "llama_cpp_router_trace.fixture.jsonl",
        "--json",
    ])


def validate_phase3_real_evidence_bundle() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "plan_phase3_real_evidence_bundle.py")])

def validate_phase3_real_evidence_bundle_builder() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "build_phase3_real_evidence_bundle.py"), "--json"])


def validate_phase3_real_evidence_matrix() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "plan_phase3_real_evidence_matrix.py")])


def validate_phase3_handoff_coverage() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "plan_phase3_handoff_coverage.py")])


def validate_runtime_baseline_artifact_contract() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "plan_runtime_baseline_artifacts.py")])


def validate_runtime_baseline_capture_plan() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "plan_runtime_baseline_capture.py")])


def validate_edge_hardware_quickstart() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "plan_edge_hardware_quickstart.py")])


def validate_llama_cpp_router_trace_fixture() -> int:
    return run_command(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_llama_cpp_router_trace.py"),
            str(MVP_DIR / "data" / "llama_cpp_router_trace.fixture.jsonl"),
            "--expected-layers",
            "2",
            "--require-kind",
            "selected_experts",
            "--require-kind",
            "selected_weights",
            "--require-kind",
            "selected_weights_norm",
        ]
    )


def run_unit_tests() -> int:
    return run_command([sys.executable, "-m", "unittest", "discover", "-s", str(MVP_DIR / "tests")])


def compile_python_sources() -> int:
    failures: list[str] = []
    python_files = sorted(MVP_DIR.rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
    for path in python_files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            failures.append(f"{path.relative_to(ROOT)}: {exc.msg}")

    if failures:
        print("py_compile failures:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print(f"py_compile: ok ({len(python_files)} files)")
    return 0


def check_docs_portability() -> int:
    failures: list[str] = []
    for path in PRIMARY_DOCS:
        if not path.exists():
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if WINDOWS_ABSOLUTE_PATH.search(line):
                failures.append(f"{path.relative_to(ROOT)}:{line_number}: {line.strip()}")

    if failures:
        print("stale absolute Windows paths found in primary docs:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("docs portability: ok")
    return 0


def require_clean_git_status() -> int:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        print(completed.stderr.strip())
        return completed.returncode
    if completed.stdout.strip():
        print("git status is not clean:")
        print(completed.stdout.rstrip())
        return 1
    print("git status: clean")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="fail if git status has tracked or untracked changes",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    checks = [
        ("model target registry", validate_model_target_registry),
        ("expert paging roadmap", validate_expert_paging_roadmap),
        ("managed expert loading plan", validate_managed_expert_loading_plan),
        ("expert inventory manifest", validate_expert_inventory_manifest),
        ("expert store layout", validate_expert_store_layout),
        ("trace inventory replay plan", validate_trace_inventory_replay_plan),
        ("real-model trace inventory pairing", validate_real_model_trace_inventory_pairing),
        ("baseline replay policies", validate_baseline_replay_policies),
        ("baseline policy replay", validate_baseline_policy_replay),
        ("dense fallback comparison", validate_dense_fallback_comparison),
        ("dense fallback comparison builder", validate_dense_fallback_comparison_builder),
        ("phase 3 prompt-set builder", validate_phase3_prompt_set_builder),
        ("phase 3 output-summary builder", validate_phase3_output_summary_builder),
        ("phase 3 trace-receipt builder", validate_phase3_trace_receipt_builder),
        ("phase 3 runtime-capture request builder", validate_phase3_runtime_capture_request_builder),
        ("phase 3 runtime-capture request audit", validate_phase3_runtime_capture_request_audit),
        ("phase 3 runtime-capture command contract", validate_phase3_runtime_capture_command_contract),
        ("phase 3 launch-card library", validate_phase3_launch_card_library),
        ("phase 3 capture-result intake", validate_phase3_capture_result_intake),
        ("phase 3 live-capability proof template builder", validate_phase3_live_capability_proof_template_builder),
        ("phase 3 go/no-go decision", validate_phase3_go_no_go),
        ("phase 3 live capability proof", validate_phase3_live_capability_proof),
        ("phase 3 runtime actuator design", validate_phase3_runtime_actuator_design),
        ("phase 3 runtime actuator spike", validate_phase3_runtime_actuator_spike),
        ("phase 3 evidence packet", validate_phase3_evidence_packet),
        ("phase 3 operator handoff package", validate_phase3_operator_handoff),
        ("phase 3 capture-completion receipt", validate_phase3_capture_completion_receipt),
        ("phase 3 dense fallback capture plan", validate_phase3_dense_fallback_capture),
        ("phase 3 policy-candidate trace plan", validate_phase3_policy_candidate_trace),
        ("phase 3 real-evidence capture plan", validate_phase3_real_evidence_capture),
        ("phase 3 artifact intake", validate_phase3_artifact_intake),
        ("phase 3 real-evidence bundle", validate_phase3_real_evidence_bundle),
        ("phase 3 real-evidence bundle builder", validate_phase3_real_evidence_bundle_builder),
        ("phase 3 real-evidence matrix", validate_phase3_real_evidence_matrix),
        ("phase 3 handoff coverage", validate_phase3_handoff_coverage),
        ("runtime baseline artifact contract", validate_runtime_baseline_artifact_contract),
        ("runtime baseline capture plan", validate_runtime_baseline_capture_plan),
        ("edge hardware quickstart", validate_edge_hardware_quickstart),
        ("llama.cpp router trace fixture", validate_llama_cpp_router_trace_fixture),
        ("unit tests", run_unit_tests),
        ("py_compile", compile_python_sources),
        ("docs portability", check_docs_portability),
    ]
    if args.require_clean:
        checks.append(("git clean", require_clean_git_status))

    failed = False
    for name, check in checks:
        print(f"\n== {name} ==")
        if check() != 0:
            failed = True

    if failed:
        print("\nreadiness: failed")
        return 1

    print("\nreadiness: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
