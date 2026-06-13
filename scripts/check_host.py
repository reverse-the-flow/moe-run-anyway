#!/usr/bin/env python3
"""Inspect whether this host is ready for portable MoE probe work.

This is a lightweight preflight. It does not authenticate, download models,
start Docker, start model servers, load checkpoints, or run GPU workloads.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COMMAND_TIMEOUT_SECONDS = 8


def run_probe(command: list[str]) -> dict[str, Any]:
    executable = shutil.which(command[0])
    result: dict[str, Any] = {
        "command": command[0],
        "path": executable,
        "available": executable is not None,
        "returncode": None,
        "summary": None,
        "error": None,
    }
    if executable is None:
        return result

    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        result["error"] = f"timed out after {COMMAND_TIMEOUT_SECONDS}s"
        return result
    except OSError as exc:
        result["error"] = str(exc)
        return result

    result["returncode"] = completed.returncode
    output = "\n".join(
        line.strip()
        for line in (completed.stdout + "\n" + completed.stderr).splitlines()
        if line.strip()
    )
    if output:
        result["summary"] = output[:600]
    return result


def command_presence(command: str) -> dict[str, Any]:
    path = shutil.which(command)
    return {
        "command": command,
        "path": path,
        "available": path is not None,
    }


def collect_host_info() -> dict[str, Any]:
    git_probe = run_probe(["git", "--version"])
    nvidia_probe = run_probe(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
    )
    rocm_smi_probe = run_probe(["rocm-smi", "--showproductname"])
    rocminfo_probe = run_probe(["rocminfo"])

    gpu_tooling = {
        "nvidia_cuda": nvidia_probe,
        "amd_rocm": {
            "rocm_smi": rocm_smi_probe,
            "rocminfo": rocminfo_probe,
        },
    }
    has_nvidia = bool(nvidia_probe["available"] and nvidia_probe["returncode"] == 0)
    has_rocm = bool(
        (rocm_smi_probe["available"] and rocm_smi_probe["returncode"] == 0)
        or (rocminfo_probe["available"] and rocminfo_probe["returncode"] == 0)
    )

    return {
        "project_root": str(ROOT),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
        },
        "git": git_probe,
        "gpu_tooling": gpu_tooling,
        "gpu_detected": {
            "nvidia_cuda": has_nvidia,
            "amd_rocm": has_rocm,
            "any": has_nvidia or has_rocm,
        },
        "optional_tooling": {
            "llama-server": command_presence("llama-server"),
            "docker": command_presence("docker"),
            "huggingface-cli": command_presence("huggingface-cli"),
        },
    }


def print_command_probe(label: str, probe: dict[str, Any]) -> None:
    if not probe["available"]:
        print(f"{label}: not found")
        return
    print(f"{label}: found at {probe['path']}")
    if probe.get("returncode") is not None:
        print(f"  exit: {probe['returncode']}")
    if probe.get("summary"):
        for line in str(probe["summary"]).splitlines()[:6]:
            print(f"  {line}")
    if probe.get("error"):
        print(f"  {probe['error']}")


def print_human(info: dict[str, Any], *, require_gpu: bool) -> None:
    print("Host portability preflight")
    print(f"Project root: {info['project_root']}")
    print(
        "Platform: "
        f"{info['platform']['system']} {info['platform']['release']} "
        f"({info['platform']['machine']})"
    )
    print(f"Python: {info['python']['executable']} ({info['python']['version']})")
    print_command_probe("git", info["git"])

    print("\nGPU tooling")
    print_command_probe("nvidia-smi", info["gpu_tooling"]["nvidia_cuda"])
    print_command_probe("rocm-smi", info["gpu_tooling"]["amd_rocm"]["rocm_smi"])
    print_command_probe("rocminfo", info["gpu_tooling"]["amd_rocm"]["rocminfo"])

    gpu_detected = info["gpu_detected"]["any"]
    if gpu_detected:
        families = []
        if info["gpu_detected"]["nvidia_cuda"]:
            families.append("NVIDIA/CUDA")
        if info["gpu_detected"]["amd_rocm"]:
            families.append("AMD/ROCm")
        print(f"GPU preflight: detected {' and '.join(families)} tooling")
    elif require_gpu:
        print("GPU preflight: failed; --require-gpu was set and no NVIDIA/ROCm tooling was detected")
    else:
        print("GPU preflight: no GPU tooling detected; allowed for Tier 0 checks")

    print("\nOptional local model/server tooling")
    for name, probe in info["optional_tooling"].items():
        status = f"found at {probe['path']}" if probe["available"] else "not found"
        print(f"{name}: {status}")

    print("\nNo services were started, no credentials were used, and no models were loaded.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="fail if neither NVIDIA/CUDA nor AMD/ROCm tooling is detected",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit JSON instead of human-readable output",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    info = collect_host_info()

    if args.json:
        print(json.dumps(info, indent=2, sort_keys=True))
    else:
        print_human(info, require_gpu=args.require_gpu)

    if args.require_gpu and not info["gpu_detected"]["any"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
