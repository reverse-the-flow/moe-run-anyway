#!/usr/bin/env python3
"""Guarded local Transformers runner for ForwardHookMoEProbe.

This is the first real-target bridge after the synthetic hook smoke. The dry-run
path is dependency-free and does not import torch or transformers. The run path
uses only a user-provided local model path, forces offline/local loading, and
wraps generation with ForwardHookMoEProbe spans.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import moe_forward_probe


JSONDict = dict[str, Any]
TOKEN_ENV_NAMES = (
    "HF_TOKEN",
    "HF_HUB_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
)
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}
MODEL_CONFIG_HINTS = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
REMOTE_CODE_AUTOMAP_KEYS = (
    "AutoConfig",
    "AutoModel",
    "AutoModelForCausalLM",
    "AutoModelForVision2Seq",
    "AutoProcessor",
    "AutoTokenizer",
)


def module_presence(module_name: str) -> JSONDict:
    spec = importlib.util.find_spec(module_name)
    return {
        "module": module_name,
        "available": spec is not None,
        "origin": spec.origin if spec is not None else None,
    }


def quote_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def is_uri_like(raw_path: str) -> bool:
    parsed = urlparse(raw_path)
    return bool(parsed.scheme and parsed.scheme not in {"", "file"})


def local_model_path_report(raw_path: str) -> JSONDict:
    expanded = Path(raw_path).expanduser()
    resolved = expanded.resolve() if expanded.exists() else expanded.absolute()
    report: JSONDict = {
        "input": raw_path,
        "path": str(resolved),
        "exists": expanded.exists(),
        "is_dir": expanded.is_dir(),
        "is_file": expanded.is_file(),
        "is_uri_like": is_uri_like(raw_path),
        "config_hints": [],
    }
    if expanded.is_dir():
        report["config_hints"] = [
            name for name in MODEL_CONFIG_HINTS if (expanded / name).exists()
        ]
    return report


def load_probe_suite(path: Path) -> JSONDict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config_json(model_path: Path) -> JSONDict:
    config_path = model_path / "config.json"
    if not config_path.exists():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def module_ref_to_file(model_path: Path, module_ref: str) -> Path | None:
    module_part = str(module_ref).split(":", 1)[0].rsplit(".", 1)[0]
    if not module_part:
        return None
    candidate = model_path / Path(*module_part.split(".")).with_suffix(".py")
    return candidate if candidate.exists() else None


def auto_map_code_files(model_path: Path, config: JSONDict) -> list[Path]:
    auto_map = config.get("auto_map")
    if not isinstance(auto_map, dict):
        return []
    files: list[Path] = []
    for key in REMOTE_CODE_AUTOMAP_KEYS:
        value = auto_map.get(key)
        refs = value if isinstance(value, list) else [value]
        for ref in refs:
            if isinstance(ref, str):
                path = module_ref_to_file(model_path, ref)
                if path is not None and path not in files:
                    files.append(path)
    return files


def relative_import_file(current_file: Path, module_name: str | None, level: int) -> Path | None:
    if level <= 0:
        return None
    base = current_file.parent
    for _ in range(max(0, level - 1)):
        base = base.parent
    if not module_name:
        return None
    candidate = base / Path(*module_name.split(".")).with_suffix(".py")
    return candidate if candidate.exists() else None


def import_root(name: str) -> str:
    return name.split(".", 1)[0]


def is_stdlib_module(root: str) -> bool:
    stdlib_names = getattr(sys, "stdlib_module_names", set())
    return root in stdlib_names or root in sys.builtin_module_names


def collect_trusted_code_imports(model_path: Path) -> JSONDict:
    config = load_config_json(model_path)
    pending = auto_map_code_files(model_path, config)
    visited: set[Path] = set()
    absolute_imports: dict[str, list[str]] = {}
    parse_errors: list[str] = []

    while pending:
        path = pending.pop(0)
        if path in visited:
            continue
        visited.add(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            parse_errors.append(f"{path.name}: {exc}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = import_root(alias.name)
                    if root:
                        absolute_imports.setdefault(root, []).append(f"{path.name}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    relative = relative_import_file(path, node.module, node.level)
                    if relative is not None and relative not in visited and relative not in pending:
                        pending.append(relative)
                    continue
                if node.module:
                    root = import_root(node.module)
                    absolute_imports.setdefault(root, []).append(f"{path.name}:{node.lineno}")

    missing: list[JSONDict] = []
    present: list[str] = []
    ignored: list[str] = []
    for root, locations in sorted(absolute_imports.items()):
        if is_stdlib_module(root):
            ignored.append(root)
            continue
        if (model_path / f"{root}.py").exists() or (model_path / root).is_dir():
            ignored.append(root)
            continue
        if importlib.util.find_spec(root) is not None:
            present.append(root)
            continue
        missing.append({"module": root, "locations": sorted(set(locations))})

    return {
        "checked": bool(visited),
        "entry_files": [str(path) for path in auto_map_code_files(model_path, config)],
        "visited_files": [str(path) for path in sorted(visited)],
        "present_modules": sorted(set(present)),
        "ignored_modules": sorted(set(ignored)),
        "missing_modules": missing,
        "parse_errors": parse_errors,
    }


def iter_selected_prompts(suite: JSONDict, max_prompts: int) -> list[JSONDict]:
    selected: list[JSONDict] = []
    prompt_budget = max_prompts if max_prompts > 0 else 10**9
    for family in suite.get("families", []):
        for prompt in family.get("prompts", []):
            if len(selected) >= prompt_budget:
                return selected
            selected.append(
                {
                    "family_id": family.get("family_id"),
                    "routing_hypothesis": family.get("routing_hypothesis"),
                    **prompt,
                }
            )
    return selected


def prompt_to_text(tokenizer: Any, prompt: JSONDict) -> str:
    messages = prompt.get("messages")
    if isinstance(messages, list) and messages:
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return "\n".join(
            f"{message.get('role', 'user')}: {message.get('content', '')}"
            for message in messages
            if isinstance(message, dict)
        )
    raw_prompt = prompt.get("prompt")
    return str(raw_prompt) if raw_prompt is not None else json.dumps(prompt, sort_keys=True)


def install_offline_environment() -> None:
    for name, value in OFFLINE_ENV.items():
        os.environ.setdefault(name, value)


def build_next_command(
    *,
    model_path: Path,
    output_dir: Path,
    suite_path: Path,
    label: str,
    max_prompts: int,
    repeats: int,
    window_size_events: int | None,
) -> str:
    parts = [
        "python3",
        "run_transformers_forward_probe.py",
        "--model-path",
        str(model_path),
        "--output-dir",
        str(output_dir),
        "--suite-path",
        str(suite_path),
        "--label",
        label,
        "--max-prompts",
        str(max_prompts),
        "--repeats",
        str(repeats),
    ]
    if window_size_events is not None:
        parts.extend(["--window-size-events", str(window_size_events)])
    return quote_command(parts)


def plan_transformers_forward_probe(args: argparse.Namespace) -> JSONDict:
    model_path = local_model_path_report(str(args.model_path))
    token_env_names_present = [name for name in TOKEN_ENV_NAMES if os.environ.get(name)]
    optional_dependencies = {
        "torch": module_presence("torch"),
        "transformers": module_presence("transformers"),
    }
    blockers: list[str] = []
    warnings: list[str] = []

    if model_path["is_uri_like"]:
        blockers.append("model path must be a local filesystem path, not a URI or remote id")
    if not model_path["exists"]:
        blockers.append("model path does not exist")
    elif not model_path["is_dir"]:
        blockers.append("model path must be a local Transformers model directory")

    for name, presence in optional_dependencies.items():
        if not presence["available"]:
            blockers.append(f"{name} module not detected")

    if token_env_names_present:
        blockers.append(
            "Hugging Face token environment variables are present; unset them for this guarded local-only runner"
        )

    if args.trust_remote_code:
        warnings.append("trust_remote_code was requested explicitly; default is disabled")
    if model_path["exists"] and model_path["is_dir"] and "config.json" not in model_path["config_hints"]:
        warnings.append("model directory does not contain config.json")

    trusted_code_dependency_check: JSONDict = {
        "checked": False,
        "entry_files": [],
        "visited_files": [],
        "present_modules": [],
        "ignored_modules": [],
        "missing_modules": [],
        "parse_errors": [],
    }
    if args.trust_remote_code and model_path["exists"] and model_path["is_dir"]:
        trusted_code_dependency_check = collect_trusted_code_imports(Path(model_path["path"]))
        for missing in trusted_code_dependency_check["missing_modules"]:
            blockers.append(
                "trusted remote-code dependency not detected: "
                f"{missing['module']} referenced by {', '.join(missing['locations'])}"
            )
        if not trusted_code_dependency_check["checked"]:
            warnings.append("trust_remote_code requested, but no local auto_map Python files were found")
        for parse_error in trusted_code_dependency_check["parse_errors"]:
            warnings.append(f"could not parse trusted code file: {parse_error}")

    suite_prompt_count = None
    if args.suite_path.exists():
        suite_prompt_count = len(iter_selected_prompts(load_probe_suite(args.suite_path), args.max_prompts))
    else:
        blockers.append("suite path does not exist")

    next_command = build_next_command(
        model_path=Path(model_path["path"]),
        output_dir=args.output_dir,
        suite_path=args.suite_path,
        label=args.label,
        max_prompts=args.max_prompts,
        repeats=args.repeats,
        window_size_events=args.window_size_events,
    )

    return {
        "mode": "transformers_forward_probe_plan",
        "dry_run": bool(args.dry_run),
        "safety_contract": [
            "local model directory only",
            "local_files_only=True",
            "offline Hugging Face and Transformers environment flags",
            "no Hugging Face token use",
            "no downloads",
            "no model server launch",
            "no Docker",
            "no prompt traffic to external services",
            "trust_remote_code disabled by default",
        ],
        "model_path": model_path,
        "suite_path": str(args.suite_path),
        "selected_prompt_count": suite_prompt_count,
        "optional_dependencies": optional_dependencies,
        "trusted_code_dependency_check": trusted_code_dependency_check,
        "token_env_names_present": token_env_names_present,
        "trust_remote_code": bool(args.trust_remote_code),
        "blockers": blockers,
        "warnings": warnings,
        "ready_to_run": not blockers,
        "next_command": next_command,
    }


def load_local_transformers_model(model_path: Path, *, trust_remote_code: bool) -> tuple[Any, Any]:
    install_offline_environment()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return tokenizer, model


def run_transformers_probe(args: argparse.Namespace) -> Path:
    plan = plan_transformers_forward_probe(args)
    if plan["blockers"]:
        print(json.dumps(plan, indent=2))
        raise SystemExit(2)

    import torch

    tokenizer, model = load_local_transformers_model(
        Path(plan["model_path"]["path"]),
        trust_remote_code=args.trust_remote_code,
    )
    if args.device:
        model.to(args.device)

    config = moe_forward_probe.ForwardHookProbeConfig(
        output_dir=args.output_dir,
        label=args.label,
        default_top_k=args.default_top_k,
        token_sample_limit=args.token_sample_limit,
        window_size_tokens=args.window_size_tokens,
        window_size_events=args.window_size_events,
    )
    probe = moe_forward_probe.ForwardHookMoEProbe(config=config)
    hook_count = probe.attach(model)
    suite = load_probe_suite(args.suite_path)
    selected_prompts = iter_selected_prompts(suite, args.max_prompts)

    with torch.no_grad():
        for repeat_index in range(args.repeats):
            for prompt in selected_prompts:
                text = prompt_to_text(tokenizer, prompt)
                inputs = tokenizer(text, return_tensors="pt")
                if args.device:
                    inputs = {key: value.to(args.device) for key, value in inputs.items()}
                span_metadata = {
                    "family_id": prompt.get("family_id"),
                    "probe_id": prompt.get("probe_id"),
                    "title": prompt.get("title"),
                    "repeat_index": repeat_index,
                    "expected_surface_features": prompt.get("expected_surface_features"),
                }
                with probe.span(span_metadata):
                    if hasattr(model, "generate"):
                        model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            pad_token_id=getattr(tokenizer, "eos_token_id", None),
                        )
                    else:
                        model(**inputs)

    probe.close()
    manifest = json.loads(probe.manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "hook_count": hook_count,
            "suite_name": suite.get("suite_name"),
            "selected_prompt_count": len(selected_prompts),
            "repeats": args.repeats,
            "model_path": plan["model_path"]["path"],
            "runner": "run_transformers_forward_probe.py",
            "local_files_only": True,
            "trust_remote_code": bool(args.trust_remote_code),
        }
    )
    probe.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return probe.run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("forward-probe-runs"))
    parser.add_argument("--suite-path", type=Path, default=Path("data/mixtral_probe_prompts.json"))
    parser.add_argument("--label", default="transformers-hookable-probe")
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--default-top-k", type=int, default=2)
    parser.add_argument("--token-sample-limit", type=int, default=8)
    parser.add_argument("--window-size-tokens", type=int, default=32)
    parser.add_argument("--window-size-events", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(plan_transformers_forward_probe(args), indent=2, sort_keys=True))
        return 0

    run_dir = run_transformers_probe(args)
    print(json.dumps({"run_dir": str(run_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
