#!/usr/bin/env python3
"""Plan the next live-model readiness steps without using secrets.

This script inspects local host capability, cached model hints, optional backend
tools, and the model-target registry. It prints concrete commands for each
supported target class, but it does not start servers, download models,
authenticate, run Docker, load checkpoints, or send prompt traffic.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib import error, request


ROOT = Path(__file__).resolve().parents[1]
MVP_DIR = ROOT / "memory-moe-mvp"
DEFAULT_REGISTRY_PATH = MVP_DIR / "data" / "model_target_registry.json"
DEFAULT_SUITE = MVP_DIR / "data" / "mixtral_probe_prompts.json"
ROOT_SUITE_ARG = "memory-moe-mvp/data/mixtral_probe_prompts.json"
DEFAULT_BASE_URL = "http://127.0.0.1:18080"
DEFAULT_ENDPOINTS = ("/props", "/metrics", "/slots")
DEFAULT_CACHE_DIR_LIMIT = 2500

JSONDict = dict[str, Any]


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def quote_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def command_presence(command: str) -> JSONDict:
    path = shutil.which(command)
    return {
        "command": command,
        "path": path,
        "available": path is not None,
    }


def module_presence(module_name: str) -> JSONDict:
    spec = importlib.util.find_spec(module_name)
    return {
        "module": module_name,
        "available": spec is not None,
        "origin": spec.origin if spec is not None else None,
    }


def run_probe(command: list[str], timeout_seconds: float = 5.0) -> JSONDict:
    executable = shutil.which(command[0])
    result: JSONDict = {
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
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        result["error"] = f"timed out after {timeout_seconds}s"
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


def detect_gpu_tooling() -> JSONDict:
    nvidia = run_probe(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
    )
    rocm_smi = run_probe(["rocm-smi", "--showproductname"])
    rocminfo = run_probe(["rocminfo"])
    has_nvidia = bool(nvidia["available"] and nvidia["returncode"] == 0)
    has_rocm = bool(
        (rocm_smi["available"] and rocm_smi["returncode"] == 0)
        or (rocminfo["available"] and rocminfo["returncode"] == 0)
    )
    families = []
    if has_nvidia:
        families.append("nvidia_cuda")
    if has_rocm:
        families.append("amd_rocm")
    return {
        "nvidia_cuda": nvidia,
        "amd_rocm": {
            "rocm_smi": rocm_smi,
            "rocminfo": rocminfo,
        },
        "detected": {
            "any": has_nvidia or has_rocm,
            "families": families,
        },
    }


def probe_endpoint(base_url: str, path: str, timeout_seconds: float) -> JSONDict:
    url = f"{normalize_base_url(base_url)}{path}"
    req = request.Request(url, method="GET")
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            body = response.read(4096)
            return {
                "path": path,
                "available": True,
                "status_code": response.status,
                "content_type": response.headers.get("Content-Type"),
                "bytes_read": len(body),
                "error": None,
            }
    except (error.HTTPError, error.URLError, TimeoutError, OSError) as exc:
        return {
            "path": path,
            "available": False,
            "status_code": None,
            "content_type": None,
            "bytes_read": 0,
            "error": str(exc),
        }


def preflight_backend(base_url: str, timeout_seconds: float) -> JSONDict:
    endpoints = [probe_endpoint(base_url, path, timeout_seconds) for path in DEFAULT_ENDPOINTS]
    available_paths = [endpoint["path"] for endpoint in endpoints if endpoint["available"]]
    return {
        "base_url": normalize_base_url(base_url),
        "observability_available": bool(available_paths),
        "available_paths": available_paths,
        "endpoints": endpoints,
    }


def load_registry(path: Path) -> JSONDict:
    return json.loads(path.read_text(encoding="utf-8"))


def decode_huggingface_model_dir(name: str) -> str | None:
    if not name.startswith("models--"):
        return None
    parts = name.split("--")
    if len(parts) < 3:
        return None
    owner = parts[1]
    repo = "--".join(parts[2:])
    if not owner or not repo:
        return None
    return f"{owner}/{repo}"


def is_huggingface_cache_model_dir(path: Path) -> bool:
    if ".locks" in path.parts:
        return False
    if decode_huggingface_model_dir(path.name) is None:
        return False
    return any((path / child).exists() for child in ("snapshots", "refs", "blobs"))


def default_cache_roots() -> list[Path]:
    candidates: list[Path] = []
    for name in ("MODEL_PATH", "LLAMA_MODEL_PATH"):
        value = os.environ.get(name)
        if value:
            candidates.append(Path(value).expanduser())

    for name in ("HF_HOME", "TRANSFORMERS_CACHE"):
        value = os.environ.get(name)
        if value:
            path = Path(value).expanduser()
            candidates.append(path / "hub" if name == "HF_HOME" else path)

    candidates.extend(
        [
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / ".cache" / "llama.cpp",
            Path.home() / "models",
            ROOT / "models",
        ]
    )

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved not in seen:
            seen.add(resolved)
            unique.append(candidate)
    return unique


def walk_existing_roots(roots: list[Path], max_dirs: int) -> tuple[list[Path], bool]:
    directories: list[Path] = []
    truncated = False
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            directories.append(root.parent)
            continue
        for current, dirnames, _ in os.walk(root):
            directories.append(Path(current))
            if len(directories) >= max_dirs:
                truncated = True
                dirnames[:] = []
                break
        if truncated:
            break
    return directories, truncated


def scan_cached_model_hints(
    search_roots: list[Path],
    *,
    max_hints: int,
    max_dirs: int = DEFAULT_CACHE_DIR_LIMIT,
) -> JSONDict:
    existing_roots = [root for root in search_roots if root.exists()]
    directories, truncated = walk_existing_roots(existing_roots, max_dirs=max_dirs)
    hf_models: list[JSONDict] = []
    gguf_files: list[JSONDict] = []
    seen_hf: set[str] = set()
    seen_gguf: set[Path] = set()

    for directory in directories:
        if not is_huggingface_cache_model_dir(directory):
            continue
        model_id = decode_huggingface_model_dir(directory.name)
        if model_id and model_id not in seen_hf:
            seen_hf.add(model_id)
            hf_models.append({"model_id": model_id, "path": str(directory)})
            if len(hf_models) >= max_hints:
                break

    for root in existing_roots:
        if root.is_file() and root.suffix.lower() == ".gguf" and root not in seen_gguf:
            seen_gguf.add(root)
            gguf_files.append({"path": str(root), "size_bytes": root.stat().st_size})
            continue
        if not root.is_dir():
            continue
        for directory in directories:
            if not is_relative_to(directory, root):
                continue
            try:
                entries = list(directory.iterdir())
            except OSError:
                continue
            for entry in entries:
                if entry.is_file() and entry.suffix.lower() == ".gguf" and entry not in seen_gguf:
                    seen_gguf.add(entry)
                    gguf_files.append({"path": str(entry), "size_bytes": entry.stat().st_size})
                    if len(gguf_files) >= max_hints:
                        break
            if len(gguf_files) >= max_hints:
                break
        if len(gguf_files) >= max_hints:
            break

    env_model_paths = []
    for name in ("MODEL_PATH", "LLAMA_MODEL_PATH"):
        value = os.environ.get(name)
        if value:
            path = Path(value).expanduser()
            env_model_paths.append({"name": name, "path": str(path), "exists": path.exists()})

    return {
        "search_roots": [str(root) for root in search_roots],
        "existing_roots": [str(root) for root in existing_roots],
        "truncated": truncated,
        "environment_model_paths": env_model_paths,
        "huggingface_models": hf_models,
        "gguf_files": gguf_files,
    }


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def collect_capabilities(
    *,
    base_url: str,
    preflight_timeout_seconds: float,
    skip_network: bool,
    search_roots: list[Path],
    max_cache_hints: int,
) -> JSONDict:
    backend_tools = {
        name: command_presence(name)
        for name in (
            "llama-server",
            "llama-cli",
            "docker",
            "docker-compose",
            "huggingface-cli",
            "nvidia-smi",
            "rocm-smi",
            "rocminfo",
        )
    }
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
            "modules": {
                "torch": module_presence("torch"),
                "transformers": module_presence("transformers"),
            },
        },
        "gpu_tooling": detect_gpu_tooling(),
        "backend_tools": backend_tools,
        "cached_model_hints": scan_cached_model_hints(search_roots, max_hints=max_cache_hints),
        "live_backend": None if skip_network else preflight_backend(base_url, preflight_timeout_seconds),
    }


def first_gguf_path(capabilities: JSONDict) -> str | None:
    ggufs = capabilities["cached_model_hints"]["gguf_files"]
    if ggufs:
        return str(ggufs[0]["path"])
    for env_path in capabilities["cached_model_hints"]["environment_model_paths"]:
        path = env_path["path"]
        if path.endswith(".gguf") and env_path["exists"]:
            return path
    return None


def local_model_path_hint(capabilities: JSONDict) -> str:
    for env_path in capabilities["cached_model_hints"]["environment_model_paths"]:
        if env_path["exists"]:
            return env_path["path"]
    hf_models = capabilities["cached_model_hints"]["huggingface_models"]
    if hf_models:
        return hf_models[0]["path"]
    return "/path/to/local/model"


def readiness_state_for_target(target: JSONDict, capabilities: JSONDict) -> tuple[str, list[str]]:
    target_class = target["target_class"]
    live_backend = capabilities.get("live_backend")
    observability = bool(live_backend and live_backend["observability_available"])
    has_llama_server = capabilities["backend_tools"]["llama-server"]["available"]
    has_local_model = bool(
        capabilities["cached_model_hints"]["gguf_files"]
        or capabilities["cached_model_hints"]["huggingface_models"]
        or any(item["exists"] for item in capabilities["cached_model_hints"]["environment_model_paths"])
    )
    has_torch = capabilities["python"]["modules"]["torch"]["available"]
    has_transformers = capabilities["python"]["modules"]["transformers"]["available"]

    if target_class == "stock_llama_cpp_openai_compatible":
        if observability:
            return "ready_for_guarded_runtime_probe", []
        blockers = ["no reachable /props, /metrics, or /slots endpoint"]
        if not has_llama_server:
            blockers.append("llama-server not found on PATH")
        if not first_gguf_path(capabilities):
            blockers.append("no local GGUF hint found")
        return "needs_user_started_observable_backend", blockers

    if target_class == "passive_sidecar_proxy":
        if observability:
            return "ready_for_passive_sidecar", []
        return "needs_user_started_upstream", ["no reachable upstream observability endpoint"]

    if target_class in {"hookable_pytorch_moe", "small_local_moe"}:
        blockers = []
        if not has_torch:
            blockers.append("torch module not detected")
        if not has_transformers:
            blockers.append("transformers module not detected")
        if not has_local_model:
            blockers.append("no local checkpoint/cache hint found")
        return ("candidate_hookable_runtime" if not blockers else "needs_local_hookable_runtime", blockers)

    if target_class == "mixtral_style":
        if observability:
            return "ready_for_opaque_mixtral_observation", []
        return "matrix_anchor_pending_backend", ["run opaque llama.cpp or hookable PyTorch path when available"]

    return "planned", []


def commands_for_target(target: JSONDict, capabilities: JSONDict, base_url: str) -> list[str]:
    target_class = target["target_class"]
    normalized_base_url = normalize_base_url(base_url)
    gguf_path = first_gguf_path(capabilities) or "$LLAMA_MODEL_PATH"
    model_path = local_model_path_hint(capabilities)

    guarded_runtime = quote_command(
        [
            "python3",
            "scripts/run_live_baseline.py",
            "--base-url",
            normalized_base_url,
            "--model",
            "local-moe",
            "--output-dir",
            "memory-moe-mvp/runtime-probe-runs",
            "--label",
            "live-readiness-baseline",
            "--suite-path",
            ROOT_SUITE_ARG,
            "--max-prompts",
            "4",
            "--repeats",
            "2",
            "--preflight-timeout-seconds",
            "2",
            "--log-file-path",
            "llama-server.log",
        ]
    )
    preflight = quote_command(
        [
            "python3",
            "scripts/run_live_baseline.py",
            "--base-url",
            normalized_base_url,
            "--model",
            "local-moe",
            "--preflight-only",
            "--preflight-timeout-seconds",
            "2",
        ]
    )
    sidecar = quote_command(
        [
            "python3",
            "llama_sidecar.py",
            "--listen-port",
            "8091",
            "--upstream-base-url",
            normalized_base_url,
            "--output-dir",
            "sidecar-runs",
            "--label",
            "live-readiness-sidecar",
            "--capture-upstream-observability",
        ]
    )
    hook_demo = quote_command(
        [
            "python3",
            "run_forward_probe_demo.py",
            "--output-dir",
            "forward-probe-runs",
            "--suite-path",
            "data/mixtral_probe_prompts.json",
            "--max-prompts",
            "2",
            "--window-size-events",
            "2",
            "--label",
            "synthetic-hook-smoke",
        ]
    )

    if target_class == "stock_llama_cpp_openai_compatible":
        commands = []
        if capabilities["backend_tools"]["llama-server"]["available"]:
            commands.append(
                " ".join(
                    [
                        "llama-server",
                        "--model",
                        shlex.quote(gguf_path),
                        "--host",
                        "127.0.0.1",
                        "--port",
                        "18080",
                        "--metrics",
                        "--slots",
                        "--props",
                        "--perf",
                        ">",
                        "llama-server.log",
                        "2>&1",
                    ]
                )
            )
        else:
            commands.append(
                "llama-server --model "
                + shlex.quote(gguf_path)
                + " --host 127.0.0.1 --port 18080 --metrics --slots --props --perf > llama-server.log 2>&1"
            )
        commands.extend([preflight, guarded_runtime])
        return commands

    if target_class == "passive_sidecar_proxy":
        return [f"cd memory-moe-mvp && {sidecar}"]

    if target_class in {"hookable_pytorch_moe", "small_local_moe"}:
        return [
            f"cd memory-moe-mvp && {hook_demo}",
            "MODEL_PATH="
            + shlex.quote(model_path)
            + " python3 path/to/future_transformers_runner.py --model-path \"$MODEL_PATH\" "
            "--output-dir forward-probe-runs --suite-path data/mixtral_probe_prompts.json "
            "--max-prompts 4 --repeats 2",
        ]

    if target_class == "mixtral_style":
        return [
            preflight,
            guarded_runtime,
            f"cd memory-moe-mvp && {hook_demo}",
        ]

    return list(target.get("next_live_commands", []))


def build_target_plan(target: JSONDict, capabilities: JSONDict, base_url: str) -> JSONDict:
    state, blockers = readiness_state_for_target(target, capabilities)
    semantic_status = target.get("semantic_expert_ids", "runtime_dependent")
    return {
        "target_id": target["target_id"],
        "target_class": target["target_class"],
        "backend_family": target["backend_family"],
        "probe_tier": target["probe_tier"],
        "primary_probe": target["primary_probe"],
        "readiness_state": state,
        "blockers": blockers,
        "semantic_expert_ids": semantic_status,
        "honesty_note": honesty_note_for_semantics(semantic_status),
        "commands": commands_for_target(target, capabilities, base_url),
        "observable_signals": target.get("observable_signals", []),
    }


def honesty_note_for_semantics(status: str) -> str:
    if status == "not_exposed":
        return "This path can collect runtime/request telemetry, not semantic expert ids."
    if status == "expected_when_router_outputs_are_exposed":
        return "Semantic expert traces require a hookable local runtime exposing router outputs."
    return "Semantic expert traces depend on which backend path is actually used."


def build_plan(
    registry: JSONDict,
    capabilities: JSONDict,
    *,
    base_url: str,
    target_class: str | None,
) -> JSONDict:
    targets = registry.get("targets", [])
    if target_class:
        targets = [target for target in targets if target.get("target_class") == target_class]
    return {
        "mode": "live_model_readiness_plan",
        "registry_schema_version": registry.get("schema_version"),
        "base_url": normalize_base_url(base_url),
        "capabilities": capabilities,
        "target_plans": [
            build_target_plan(target, capabilities, base_url)
            for target in targets
            if isinstance(target, dict)
        ],
        "safety_contract": [
            "does not start model servers",
            "does not download models",
            "does not authenticate or inspect private tokens",
            "does not send prompt traffic",
            "does not claim semantic expert ids for stock llama.cpp observation",
        ],
    }


def summarize_tool(probe: JSONDict) -> str:
    return f"found at {probe['path']}" if probe["available"] else "not found"


def print_human_plan(plan: JSONDict) -> None:
    capabilities = plan["capabilities"]
    print("Live model readiness plan")
    print(f"Project root: {capabilities['project_root']}")
    print(
        "Platform: "
        f"{capabilities['platform']['system']} {capabilities['platform']['release']} "
        f"({capabilities['platform']['machine']})"
    )
    print(f"Python: {capabilities['python']['executable']} ({capabilities['python']['version']})")

    gpu = capabilities["gpu_tooling"]["detected"]
    print(f"GPU tooling: {', '.join(gpu['families']) if gpu['families'] else 'none detected'}")
    print("Backend tools:")
    for name in sorted(capabilities["backend_tools"]):
        print(f"  {name}: {summarize_tool(capabilities['backend_tools'][name])}")

    cache = capabilities["cached_model_hints"]
    print("Cached model hints:")
    print(f"  roots checked: {len(cache['existing_roots'])} existing of {len(cache['search_roots'])}")
    print(f"  GGUF files: {len(cache['gguf_files'])}")
    for item in cache["gguf_files"][:3]:
        print(f"    {item['path']}")
    print(f"  Hugging Face model dirs: {len(cache['huggingface_models'])}")
    for item in cache["huggingface_models"][:3]:
        print(f"    {item['model_id']} at {item['path']}")

    live_backend = capabilities.get("live_backend")
    if live_backend is None:
        print("Live backend: skipped")
    else:
        available = ", ".join(live_backend["available_paths"]) or "none"
        print(f"Live backend: observability_available={live_backend['observability_available']} ({available})")

    print("\nTarget plans")
    for target in plan["target_plans"]:
        print(f"- {target['target_class']} ({target['readiness_state']})")
        if target["blockers"]:
            print(f"  blockers: {'; '.join(target['blockers'])}")
        print(f"  semantic expert ids: {target['semantic_expert_ids']}")
        print(f"  note: {target['honesty_note']}")
        print("  next commands:")
        for command in target["commands"]:
            print(f"    {command}")

    print("\nSafety contract:")
    for item in plan["safety_contract"]:
        print(f"  - {item}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--search-root", type=Path, action="append", default=[])
    parser.add_argument("--max-cache-hints", type=int, default=8)
    parser.add_argument("--preflight-timeout-seconds", type=float, default=0.5)
    parser.add_argument("--skip-network", action="store_true")
    parser.add_argument("--target-class")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    search_roots = args.search_root or default_cache_roots()
    registry = load_registry(args.registry_path)
    capabilities = collect_capabilities(
        base_url=args.base_url,
        preflight_timeout_seconds=args.preflight_timeout_seconds,
        skip_network=args.skip_network,
        search_roots=search_roots,
        max_cache_hints=args.max_cache_hints,
    )
    plan = build_plan(
        registry,
        capabilities,
        base_url=args.base_url,
        target_class=args.target_class,
    )
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        print_human_plan(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
