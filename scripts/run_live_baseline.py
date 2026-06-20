#!/usr/bin/env python3
"""Run or plan the first live llama.cpp/OpenAI-compatible baseline.

This is a thin orchestration layer around `memory-moe-mvp/llama_runtime_probe.py`.
It intentionally does not start model servers, download models, authenticate,
or run Docker. A user supplies an already-running local backend, and this script
preflights the observable endpoints before launching the runtime probe.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


ROOT = Path(__file__).resolve().parents[1]
MVP_DIR = ROOT / "memory-moe-mvp"
DEFAULT_SUITE = MVP_DIR / "data" / "mixtral_probe_prompts.json"
DEFAULT_OUTPUT_DIR = MVP_DIR / "runtime-probe-runs"


JSONDict = dict[str, Any]
LLAMA_CPP_BACKEND_FAMILY = "llama_cpp"
OPENAI_COMPATIBLE_BACKEND_FAMILIES = {
    "vllm_openai_compatible",
    "ollama_openai_compatible",
    "openai_compatible",
}
OBSERVABILITY_PATHS = ("/props", "/metrics", "/slots")
READINESS_PATHS = ("/v1/models", "/models")


@dataclass(frozen=True)
class EndpointProbe:
    path: str
    available: bool
    status_code: int | None = None
    content_type: str | None = None
    bytes_read: int = 0
    error: str | None = None

    def as_dict(self) -> JSONDict:
        return {
            "path": self.path,
            "available": self.available,
            "status_code": self.status_code,
            "content_type": self.content_type,
            "bytes_read": self.bytes_read,
            "error": self.error,
        }


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def probe_endpoint(base_url: str, path: str, timeout_seconds: float) -> EndpointProbe:
    url = f"{normalize_base_url(base_url)}{path}"
    req = request.Request(url, method="GET")
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            body = response.read(4096)
            return EndpointProbe(
                path=path,
                available=True,
                status_code=response.status,
                content_type=response.headers.get("Content-Type"),
                bytes_read=len(body),
            )
    except (error.HTTPError, error.URLError, TimeoutError, OSError) as exc:
        return EndpointProbe(path=path, available=False, error=str(exc))


def preflight_observability(base_url: str, timeout_seconds: float) -> JSONDict:
    probes = [probe_endpoint(base_url, path, timeout_seconds) for path in OBSERVABILITY_PATHS]
    available = [probe.path for probe in probes if probe.available]
    return {
        "base_url": normalize_base_url(base_url),
        "observability_available": bool(available),
        "available_paths": available,
        "endpoints": [probe.as_dict() for probe in probes],
    }


def preflight_runtime(base_url: str, backend_family: str, timeout_seconds: float) -> JSONDict:
    observability_probes = [
        probe_endpoint(base_url, path, timeout_seconds) for path in OBSERVABILITY_PATHS
    ]
    readiness_probes = [probe_endpoint(base_url, path, timeout_seconds) for path in READINESS_PATHS]
    observability_paths = [probe.path for probe in observability_probes if probe.available]
    readiness_paths = [probe.path for probe in readiness_probes if probe.available]
    observability_available = bool(observability_paths)
    readiness_available = bool(readiness_paths)

    if backend_family == LLAMA_CPP_BACKEND_FAMILY:
        traffic_allowed = observability_available
        traffic_gate = "observability" if traffic_allowed else "unavailable"
    elif backend_family in OPENAI_COMPATIBLE_BACKEND_FAMILIES:
        traffic_allowed = readiness_available or observability_available
        if readiness_available:
            traffic_gate = "readiness"
        elif observability_available:
            traffic_gate = "observability"
        else:
            traffic_gate = "unavailable"
    else:
        traffic_allowed = observability_available
        traffic_gate = "observability" if traffic_allowed else "unavailable"

    return {
        "base_url": normalize_base_url(base_url),
        "backend_family": backend_family,
        "observability_available": observability_available,
        "readiness_available": readiness_available,
        "traffic_gate": traffic_gate,
        "traffic_allowed": traffic_allowed,
        "observability_paths": list(OBSERVABILITY_PATHS),
        "readiness_paths": list(READINESS_PATHS),
        "available_observability_paths": observability_paths,
        "available_readiness_paths": readiness_paths,
        "available_paths": observability_paths,
        "observability_endpoints": [probe.as_dict() for probe in observability_probes],
        "readiness_endpoints": [probe.as_dict() for probe in readiness_probes],
        "endpoints": [probe.as_dict() for probe in [*observability_probes, *readiness_probes]],
    }


def build_runtime_probe_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(MVP_DIR / "llama_runtime_probe.py"),
        "--base-url",
        normalize_base_url(args.base_url),
        "--backend-family",
        args.backend_family,
        "--output-dir",
        str(args.output_dir),
        "--label",
        args.label,
        "--model",
        args.model,
        "--suite-path",
        str(args.suite_path),
        "--max-prompts",
        str(args.max_prompts),
        "--repeats",
        str(args.repeats),
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]
    if args.request_max_tokens:
        command.extend(["--request-max-tokens", str(args.request_max_tokens)])
    if args.log_file_path:
        command.extend(["--log-file-path", str(args.log_file_path)])
    return command


def build_plan(args: argparse.Namespace, preflight: JSONDict | None = None) -> JSONDict:
    return {
        "mode": "live_runtime_baseline",
        "base_url": normalize_base_url(args.base_url),
        "backend_family": args.backend_family,
        "model": args.model,
        "suite_path": str(args.suite_path),
        "output_dir": str(args.output_dir),
        "label": args.label,
        "max_prompts": args.max_prompts,
        "repeats": args.repeats,
        "timeout_seconds": args.timeout_seconds,
        "request_max_tokens": args.request_max_tokens,
        "preflight_timeout_seconds": args.preflight_timeout_seconds,
        "log_file_path": str(args.log_file_path) if args.log_file_path else None,
        "runtime_probe_command": build_runtime_probe_command(args),
        "preflight": preflight,
        "notes": [
            "Requires a user-started local llama-server or OpenAI-compatible backend.",
            "Does not start servers, download models, authenticate, or run Docker.",
            "Stock OpenAI-compatible telemetry is runtime/request evidence, not semantic expert ids.",
        ],
    }


def print_human_plan(plan: JSONDict) -> None:
    print("Live baseline plan")
    print(f"Base URL: {plan['base_url']}")
    print(f"Backend family: {plan['backend_family']}")
    print(f"Model: {plan['model']}")
    print(f"Prompt suite: {plan['suite_path']}")
    print(f"Output directory: {plan['output_dir']}")
    print("Runtime probe command:")
    print("  " + " ".join(plan["runtime_probe_command"]))
    if plan.get("preflight"):
        preflight = plan["preflight"]
        print("Preflight:")
        print(f"  observability_available: {preflight['observability_available']}")
        print(f"  readiness_available: {preflight.get('readiness_available', False)}")
        print(f"  traffic_gate: {preflight.get('traffic_gate', 'unknown')}")
        print(
            "  available_observability_paths: "
            f"{', '.join(preflight.get('available_observability_paths', preflight['available_paths'])) or 'none'}"
        )
        print(
            "  available_readiness_paths: "
            f"{', '.join(preflight.get('available_readiness_paths', [])) or 'none'}"
        )
        for endpoint in preflight["endpoints"]:
            status = "ok" if endpoint["available"] else f"missing ({endpoint['error']})"
            print(f"  {endpoint['path']}: {status}")


def run_runtime_probe(command: list[str]) -> int:
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument(
        "--backend-family",
        default=LLAMA_CPP_BACKEND_FAMILY,
        choices=sorted({LLAMA_CPP_BACKEND_FAMILY, *OPENAI_COMPATIBLE_BACKEND_FAMILIES}),
        help="runtime family used for preflight gating and artifact metadata",
    )
    parser.add_argument("--model", default="dolphin-mixtral")
    parser.add_argument("--suite-path", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="live-baseline")
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--request-max-tokens",
        type=int,
        help="override the prompt suite max_tokens for this run",
    )
    parser.add_argument("--preflight-timeout-seconds", type=float, default=2.0)
    parser.add_argument("--log-file-path", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the command plan without touching the network or running the probe",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="check runtime readiness and stop before sending prompt traffic",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="run the runtime probe without checking /props, /metrics, or /slots first",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON output for plan/preflight modes")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.dry_run:
        plan = build_plan(args)
        if args.json:
            print(json.dumps(plan, indent=2, sort_keys=True))
        else:
            print_human_plan(plan)
        return 0

    preflight = None
    if not args.skip_preflight:
        preflight = preflight_runtime(
            args.base_url,
            args.backend_family,
            args.preflight_timeout_seconds,
        )
        if args.preflight_only:
            plan = build_plan(args, preflight=preflight)
            if args.json:
                print(json.dumps(plan, indent=2, sort_keys=True))
            else:
                print_human_plan(plan)
            return 0 if preflight["traffic_allowed"] else 1
        if not preflight["traffic_allowed"]:
            plan = build_plan(args, preflight=preflight)
            if args.json:
                print(json.dumps(plan, indent=2, sort_keys=True))
            else:
                print_human_plan(plan)
                print(
                    "Refusing to run prompt traffic until a required observability "
                    "or readiness endpoint is reachable."
                )
            return 1

    return run_runtime_probe(build_runtime_probe_command(args))


if __name__ == "__main__":
    raise SystemExit(main())
