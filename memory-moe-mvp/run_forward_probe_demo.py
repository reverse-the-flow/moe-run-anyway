#!/usr/bin/env python3
"""Synthetic driver for the forward-hook MoE probe.

This is not a model benchmark. Its purpose is narrower:
- produce real forward-probe artifacts without requiring torch/transformers
- exercise the hook path, span metadata, and window summaries
- give us a concrete run folder while we wait for a hookable MoE backend
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import moe_forward_probe


class FakeHookHandle:
    def __init__(self, hooks: list[Any], hook: Any) -> None:
        self._hooks = hooks
        self._hook = hook

    def remove(self) -> None:
        if self._hook in self._hooks:
            self._hooks.remove(self._hook)


class FakeRouterModule:
    def __init__(self, scripted_outputs: list[dict[str, Any]]) -> None:
        self._hooks: list[Any] = []
        self._scripted_outputs = scripted_outputs
        self._cursor = 0

    def register_forward_hook(self, hook: Any) -> FakeHookHandle:
        self._hooks.append(hook)
        return FakeHookHandle(self._hooks, hook)

    def forward(self) -> dict[str, Any]:
        output = self._scripted_outputs[self._cursor % len(self._scripted_outputs)]
        self._cursor += 1
        for hook in list(self._hooks):
            hook(self, tuple(), output)
        return output


class FakeModel:
    def __init__(self, modules: list[tuple[str, FakeRouterModule]]) -> None:
        self._modules = modules

    def named_modules(self):
        yield "", self
        for name, module in self._modules:
            yield name, module

    def run_case(self) -> None:
        for _, module in self._modules:
            module.forward()


def load_probe_suite(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_demo_model() -> FakeModel:
    modules = [
        (
            "layers.3.moe.router",
            FakeRouterModule(
                [
                    {
                        "expert_indices": [[0, 1], [0, 2], [1, 2]],
                        "expert_weights": [[0.72, 0.28], [0.68, 0.32], [0.6, 0.4]],
                    },
                    {
                        "expert_indices": [[2, 3], [2, 4], [3, 4]],
                        "expert_weights": [[0.74, 0.26], [0.66, 0.34], [0.61, 0.39]],
                    },
                    {
                        "expert_indices": [[4, 5], [5, 6], [4, 6]],
                        "expert_weights": [[0.71, 0.29], [0.58, 0.42], [0.63, 0.37]],
                    },
                ]
            ),
        ),
        (
            "layers.9.gate",
            FakeRouterModule(
                [
                    {
                        "router_logits": [
                            [2.2, 0.4, 0.1, -0.6],
                            [1.8, 0.5, 0.2, -0.4],
                            [2.0, 0.2, 0.1, -0.3],
                        ]
                    },
                    {
                        "router_logits": [
                            [0.1, 2.0, 0.3, -0.5],
                            [0.2, 2.3, 0.4, -0.2],
                            [0.3, 1.8, 0.5, -0.4],
                        ]
                    },
                    {
                        "router_logits": [
                            [0.2, 0.5, 2.1, -0.4],
                            [0.1, 0.4, 2.2, -0.3],
                            [0.2, 0.3, 1.9, -0.6],
                        ]
                    },
                ]
            ),
        ),
    ]
    return FakeModel(modules)


def run_demo(
    output_dir: Path,
    label: str,
    suite_path: Path,
    max_prompts: int,
    window_size_tokens: int,
    window_size_events: int | None,
) -> Path:
    suite = load_probe_suite(suite_path)
    config = moe_forward_probe.ForwardHookProbeConfig(
        output_dir=output_dir,
        label=label,
        window_size_tokens=window_size_tokens,
        window_size_events=window_size_events,
    )
    probe = moe_forward_probe.ForwardHookMoEProbe(config=config)
    model = build_demo_model()
    hook_count = probe.attach(model)

    prompt_budget = max_prompts if max_prompts > 0 else 10**9
    seen = 0
    for family in suite["families"]:
        for prompt in family["prompts"]:
            if seen >= prompt_budget:
                break
            span_metadata = {
                "family_id": family["family_id"],
                "probe_id": prompt["probe_id"],
                "title": prompt["title"],
                "expected_surface_features": prompt["expected_surface_features"],
            }
            with probe.span(span_metadata):
                model.run_case()
            seen += 1
        if seen >= prompt_budget:
            break

    probe.close()

    manifest = json.loads(probe.manifest_path.read_text(encoding="utf-8"))
    manifest["hook_count"] = hook_count
    manifest["suite_name"] = suite["suite_name"]
    manifest["selected_prompt_count"] = seen
    probe.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return probe.run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("X:/Experiments/memory-moe-mvp/forward-probe-runs"),
    )
    parser.add_argument("--label", default="forward-probe-demo")
    parser.add_argument(
        "--suite-path",
        type=Path,
        default=Path("X:/Experiments/memory-moe-mvp/data/mixtral_probe_prompts.json"),
    )
    parser.add_argument("--max-prompts", type=int, default=6)
    parser.add_argument("--window-size-tokens", type=int, default=32)
    parser.add_argument("--window-size-events", type=int)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_dir = run_demo(
        output_dir=args.output_dir,
        label=args.label,
        suite_path=args.suite_path,
        max_prompts=args.max_prompts,
        window_size_tokens=args.window_size_tokens,
        window_size_events=args.window_size_events,
    )
    print(json.dumps({"run_dir": str(run_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
