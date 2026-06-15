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
    MVP_DIR / "docs" / "managed-expert-loading.md",
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


def validate_runtime_baseline_artifact_contract() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "plan_runtime_baseline_artifacts.py")])


def validate_runtime_baseline_capture_plan() -> int:
    return run_command([sys.executable, str(ROOT / "scripts" / "plan_runtime_baseline_capture.py")])


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
        ("runtime baseline artifact contract", validate_runtime_baseline_artifact_contract),
        ("runtime baseline capture plan", validate_runtime_baseline_capture_plan),
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
