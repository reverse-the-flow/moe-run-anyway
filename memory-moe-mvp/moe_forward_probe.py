#!/usr/bin/env python3
"""Forward-hook MoE probe for PyTorch-style model runtimes.

This probe is intentionally separate from the passive llama.cpp sidecar.

The sidecar observes the process boundary:
- request and response metadata
- latency and usage exposed by the backend
- coarse memory telemetry

This probe targets models that still expose Python modules and forward hooks.
Its job is to collect semantic routing traces such as:
- routed expert ids
- selected expert weights or probabilities
- per-layer expert hit counts
- per-window routing summaries

It does not depend on torch at import time. Anything that behaves enough like a
PyTorch module graph is accepted:
- `named_modules()` on the model
- `register_forward_hook()` on hookable submodules
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import threading
import time
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Protocol

from moe_shared_contract import attach_shared_contract, build_shared_contract, contract_shell


JSONDict = dict[str, Any]
LAYER_ID_PATTERN = re.compile(r"(?<!\d)(\d+)(?!\d)")
DEFAULT_ROUTER_HINTS = (
    "router",
    "gate",
    "gating",
    "moe",
    "switch",
    "expert",
)
EXPERT_INDEX_KEYS = (
    "expert_indices",
    "topk_idx",
    "top_k_indices",
    "selected_experts",
    "expert_ids",
    "indices",
)
EXPERT_WEIGHT_KEYS = (
    "expert_weights",
    "topk_weights",
    "top_k_weights",
    "routing_weights",
    "selected_probs",
    "top_probs",
    "weights",
)
ROUTER_LOGIT_KEYS = (
    "router_logits",
    "gate_logits",
    "gating_logits",
    "routing_scores",
    "router_scores",
    "scores",
    "logits",
)


class ForwardHookHandle(Protocol):
    def remove(self) -> None: ...


class HookableModule(Protocol):
    def register_forward_hook(self, hook: Any) -> ForwardHookHandle: ...


class NamedModuleGraph(Protocol):
    def named_modules(self) -> Iterator[tuple[str, Any]]: ...


@dataclass(frozen=True)
class ForwardHookProbeConfig:
    output_dir: Path
    label: str
    default_top_k: int = 2
    token_sample_limit: int = 8
    window_size_tokens: int = 32
    window_size_events: int | None = None
    store_span_metadata: bool = True


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: JSONDict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def round3(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 3)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def preview_text(text: str, limit: int = 160) -> str:
    squashed = " ".join(text.split())
    if len(squashed) <= limit:
        return squashed
    return f"{squashed[: limit - 3]}..."


def guess_layer_id(module_name: str) -> int | None:
    matches = LAYER_ID_PATTERN.findall(module_name)
    if not matches:
        return None
    return int(matches[-1])


def has_router_hint(module_name: str, module_type_name: str) -> bool:
    haystack = f"{module_name} {module_type_name}".lower()
    return any(hint in haystack for hint in DEFAULT_ROUTER_HINTS)


def maybe_to_list(value: Any) -> Any:
    current = value
    for attr in ("detach", "cpu"):
        if hasattr(current, attr):
            try:
                current = getattr(current, attr)()
            except Exception:
                return value
    if hasattr(current, "tolist"):
        try:
            return current.tolist()
        except Exception:
            return value
    return current


def lookup_named_value(source: Any, keys: tuple[str, ...], depth: int = 2) -> Any | None:
    current = maybe_to_list(source)

    if depth < 0:
        return None

    if isinstance(current, dict):
        for key in keys:
            if key in current:
                return current[key]
        for value in current.values():
            found = lookup_named_value(value, keys, depth=depth - 1)
            if found is not None:
                return found
        return None

    for key in keys:
        if hasattr(current, key):
            try:
                return getattr(current, key)
            except Exception:
                pass

    if isinstance(current, (list, tuple)):
        for item in current:
            found = lookup_named_value(item, keys, depth=depth - 1)
            if found is not None:
                return found

    return None


def normalize_token_rows(value: Any) -> list[list[Any]]:
    current = maybe_to_list(value)
    if current is None:
        return []
    if not isinstance(current, list):
        return []
    if not current:
        return []
    if isinstance(current[0], list):
        return [list(row) for row in current if isinstance(row, list)]
    return [list(current)]


def coerce_int_rows(value: Any) -> list[list[int]]:
    rows = normalize_token_rows(value)
    coerced: list[list[int]] = []
    for row in rows:
        new_row = []
        for item in row:
            try:
                new_row.append(int(item))
            except (TypeError, ValueError):
                continue
        if new_row:
            coerced.append(new_row)
    return coerced


def coerce_float_rows(value: Any) -> list[list[float]]:
    rows = normalize_token_rows(value)
    coerced: list[list[float]] = []
    for row in rows:
        new_row = []
        for item in row:
            try:
                new_row.append(float(item))
            except (TypeError, ValueError):
                continue
        if new_row:
            coerced.append(new_row)
    return coerced


def stable_softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    shifted = [math.exp(value - max_value) for value in values]
    total = sum(shifted)
    if total == 0:
        return [0.0 for _ in shifted]
    return [value / total for value in shifted]


def compute_entropy(probabilities: list[float]) -> float:
    if not probabilities:
        return 0.0
    entropy = 0.0
    for probability in probabilities:
        if probability > 0:
            entropy -= probability * math.log(probability)
    return entropy


def summarize_indices_and_weights(
    expert_indices: list[list[int]],
    expert_weights: list[list[float]] | None,
    token_sample_limit: int,
) -> JSONDict:
    hit_counts: Counter[int] = Counter()
    mass_by_expert: defaultdict[int, float] = defaultdict(float)
    token_samples: list[JSONDict] = []
    entropy_values: list[float] = []

    for token_index, index_row in enumerate(expert_indices):
        if not index_row:
            continue
        weight_row = expert_weights[token_index] if expert_weights and token_index < len(expert_weights) else []
        if len(weight_row) != len(index_row):
            weight_row = []

        selected_probs: list[float] = []
        for position, expert_id in enumerate(index_row):
            hit_counts[expert_id] += 1
            probability = None
            if weight_row:
                probability = float(weight_row[position])
                mass_by_expert[expert_id] += probability
                selected_probs.append(probability)
            if token_index < token_sample_limit:
                if len(token_samples) <= token_index:
                    token_samples.append(
                        {
                            "token_index": token_index,
                            "selected_experts": [],
                            "selected_probs": [],
                        }
                    )
                token_samples[token_index]["selected_experts"].append(expert_id)
                token_samples[token_index]["selected_probs"].append(probability)
        if selected_probs:
            total = sum(selected_probs)
            normalized = [value / total for value in selected_probs] if total > 0 else selected_probs
            entropy_values.append(compute_entropy(normalized))

    return {
        "token_count": len(expert_indices),
        "top_k": max((len(row) for row in expert_indices), default=0),
        "hit_counts": dict(sorted(hit_counts.items())),
        "mass_by_expert": {str(key): round3(value) for key, value in sorted(mass_by_expert.items())},
        "mean_selected_entropy": round3(sum(entropy_values) / len(entropy_values)) if entropy_values else None,
        "token_samples": token_samples,
        "source": "expert_indices",
    }


def summarize_router_logits(
    router_logits: list[list[float]],
    default_top_k: int,
    token_sample_limit: int,
) -> JSONDict:
    hit_counts: Counter[int] = Counter()
    mass_by_expert: defaultdict[int, float] = defaultdict(float)
    token_samples: list[JSONDict] = []
    entropy_values: list[float] = []

    for token_index, logit_row in enumerate(router_logits):
        if not logit_row:
            continue
        probabilities = stable_softmax(logit_row)
        entropy_values.append(compute_entropy(probabilities))
        ranked = sorted(range(len(probabilities)), key=lambda idx: probabilities[idx], reverse=True)
        selected = ranked[:default_top_k]
        selected_probs = [probabilities[idx] for idx in selected]
        for expert_id, probability in zip(selected, selected_probs):
            hit_counts[expert_id] += 1
            mass_by_expert[expert_id] += probability
        if token_index < token_sample_limit:
            token_samples.append(
                {
                    "token_index": token_index,
                    "selected_experts": selected,
                    "selected_probs": [round3(value) for value in selected_probs],
                }
            )

    return {
        "token_count": len(router_logits),
        "top_k": default_top_k,
        "hit_counts": dict(sorted(hit_counts.items())),
        "mass_by_expert": {str(key): round3(value) for key, value in sorted(mass_by_expert.items())},
        "mean_selected_entropy": round3(sum(entropy_values) / len(entropy_values)) if entropy_values else None,
        "token_samples": token_samples,
        "source": "router_logits",
    }


class GenericMoEForwardAdapter:
    """Heuristic adapter for MoE router-like modules."""

    def should_hook(self, module_name: str, module: Any) -> bool:
        return has_router_hint(module_name, type(module).__name__)

    def extract_event(
        self,
        module_name: str,
        module: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
        config: ForwardHookProbeConfig,
    ) -> JSONDict | None:
        del args, kwargs

        expert_indices = coerce_int_rows(lookup_named_value(output, EXPERT_INDEX_KEYS))
        expert_weights = coerce_float_rows(lookup_named_value(output, EXPERT_WEIGHT_KEYS))
        if expert_indices:
            summary = summarize_indices_and_weights(
                expert_indices=expert_indices,
                expert_weights=expert_weights if expert_weights else None,
                token_sample_limit=config.token_sample_limit,
            )
            return self._event_payload(module_name, module, summary)

        router_logits = coerce_float_rows(lookup_named_value(output, ROUTER_LOGIT_KEYS))
        if router_logits:
            summary = summarize_router_logits(
                router_logits=router_logits,
                default_top_k=config.default_top_k,
                token_sample_limit=config.token_sample_limit,
            )
            return self._event_payload(module_name, module, summary)

        return None

    def _event_payload(self, module_name: str, module: Any, summary: JSONDict) -> JSONDict:
        return {
            "module_name": module_name,
            "module_type": type(module).__name__,
            "layer_id": guess_layer_id(module_name),
            "token_count": summary["token_count"],
            "top_k": summary["top_k"],
            "source": summary["source"],
            "hit_counts": summary["hit_counts"],
            "mass_by_expert": summary["mass_by_expert"],
            "mean_selected_entropy": summary["mean_selected_entropy"],
            "token_samples": summary["token_samples"],
        }


class ProbeAccumulator:
    def __init__(self, run_id: str, config: ForwardHookProbeConfig) -> None:
        self.run_id = run_id
        self.config = config
        self.created_at = now_iso()
        self.updated_at = self.created_at
        self.router_event_count = 0
        self.token_count = 0
        self.by_module: Counter[str] = Counter()
        self.by_layer: Counter[str] = Counter()
        self.by_source: Counter[str] = Counter()
        self.expert_hits: Counter[str] = Counter()
        self.routing_entropies: list[float] = []
        self._window_events: list[JSONDict] = []
        self._window_token_count = 0
        self._window_span_id: str | None = None
        self._span_token_offsets: dict[str, int] = {}
        self._window_counter = 0

    def ingest_event(self, event: JSONDict) -> list[JSONDict]:
        pending_windows: list[JSONDict] = []
        self.updated_at = now_iso()
        self.router_event_count += 1
        token_count = int(event.get("token_count") or 0)
        self.token_count += token_count
        self.by_module[event["module_name"]] += 1
        layer_id = event.get("layer_id")
        self.by_layer[str(layer_id) if layer_id is not None else "unknown"] += 1
        self.by_source[event.get("source", "unknown")] += 1
        if event.get("mean_selected_entropy") is not None:
            self.routing_entropies.append(float(event["mean_selected_entropy"]))

        for expert_id, count in (event.get("hit_counts") or {}).items():
            self.expert_hits[str(expert_id)] += int(count)

        incoming_span_id = str(event.get("span_id") or "unscoped")
        if self._window_events and self._window_span_id not in {None, incoming_span_id}:
            pending_windows.append(self.flush_window())

        self._window_span_id = incoming_span_id
        self._window_events.append(event)
        self._window_token_count += token_count

        reached_event_threshold = bool(
            self.config.window_size_events and len(self._window_events) >= self.config.window_size_events
        )
        reached_token_threshold = self._window_token_count >= self.config.window_size_tokens
        if reached_event_threshold or reached_token_threshold:
            pending_windows.append(self.flush_window())

        return [item for item in pending_windows if item]

    def flush_window(self) -> JSONDict:
        if not self._window_events:
            return {}
        events = list(self._window_events)
        self._window_events.clear()
        self._window_counter += 1

        combined_hits: Counter[str] = Counter()
        combined_mass: defaultdict[str, float] = defaultdict(float)
        by_layer_hits: defaultdict[str, Counter[str]] = defaultdict(Counter)
        by_layer_mass: defaultdict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
        prompt_families: Counter[str] = Counter()
        probe_ids: Counter[str] = Counter()
        span_id = str(events[0].get("span_id") or "unscoped")
        token_count = 0
        for event in events:
            token_count += int(event.get("token_count") or 0)
            span_metadata = event.get("span_metadata") if isinstance(event.get("span_metadata"), dict) else {}
            family_id = span_metadata.get("family_id")
            probe_id = span_metadata.get("probe_id")
            if isinstance(family_id, str) and family_id:
                prompt_families[family_id] += 1
            if isinstance(probe_id, str) and probe_id:
                probe_ids[probe_id] += 1
            for expert_id, count in (event.get("hit_counts") or {}).items():
                combined_hits[str(expert_id)] += int(count)
                layer_id = str(event.get("layer_id") if event.get("layer_id") is not None else "unknown")
                by_layer_hits[layer_id][str(expert_id)] += int(count)
            for expert_id, mass in (event.get("mass_by_expert") or {}).items():
                try:
                    mass_value = float(mass)
                except (TypeError, ValueError):
                    continue
                combined_mass[str(expert_id)] += mass_value
                layer_id = str(event.get("layer_id") if event.get("layer_id") is not None else "unknown")
                by_layer_mass[layer_id][str(expert_id)] += mass_value

        token_start = self._span_token_offsets.get(span_id, 0)
        token_end = token_start + max(0, token_count - 1)
        self._span_token_offsets[span_id] = token_end + 1
        self._window_token_count = 0
        self._window_span_id = None

        prompt_family = prompt_families.most_common(1)[0][0] if prompt_families else None
        probe_id = probe_ids.most_common(1)[0][0] if probe_ids else None
        top_experts_source = (
            sorted(combined_mass.items(), key=lambda item: (-item[1], item[0]))
            if combined_mass
            else combined_hits.most_common(12)
        )
        layer_top_source = by_layer_mass if by_layer_mass else by_layer_hits

        summary = {
            "event_type": "router_window_summary",
            "timestamp": now_iso(),
            "window_index": self._window_counter,
            "span_id": span_id,
            "prompt_family": prompt_family,
            "probe_id": probe_id,
            "window_event_count": len(events),
            "window_token_count": token_count,
            "window_size_tokens": self.config.window_size_tokens,
            "token_start": token_start,
            "token_end": token_end,
            "top_experts": {
                expert_id: round3(value) if combined_mass else value
                for expert_id, value in top_experts_source[:12]
            },
            "layer_top_experts": {
                layer_id: (
                    {
                        expert_id: round3(value)
                        for expert_id, value in sorted(experts.items(), key=lambda item: (-item[1], item[0]))[:8]
                    }
                    if by_layer_mass
                    else dict(experts.most_common(8))
                )
                for layer_id, experts in sorted(layer_top_source.items())
            },
            "layer_expert_mass": {
                layer_id: {expert_id: round3(value) for expert_id, value in sorted(experts.items())}
                for layer_id, experts in sorted(by_layer_mass.items())
            },
            "layer_expert_hits": {
                layer_id: dict(counter)
                for layer_id, counter in sorted(by_layer_hits.items())
            },
            "mean_routing_entropy": round3(
                sum(
                    float(event["mean_selected_entropy"])
                    for event in events
                    if event.get("mean_selected_entropy") is not None
                )
                / max(
                    1,
                    sum(1 for event in events if event.get("mean_selected_entropy") is not None),
                )
            ),
        }
        return attach_shared_contract(
            summary,
            build_shared_contract(
                probe_tier="internal_hook",
                backend_family="hookable_pytorch",
                prompt_family=prompt_family,
                window_size_tokens=self.config.window_size_tokens,
                baseline_kind="observational",
            ),
        )

    def snapshot(self) -> JSONDict:
        return {
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "run_id": self.run_id,
            "mode": "moe_forward_hook_probe",
            "shared_contract_template": contract_shell(
                probe_tier="internal_hook",
                backend_family="hookable_pytorch",
                default_baseline_kind="observational",
                default_window_size_tokens=self.config.window_size_tokens,
            ),
            "config": {
                "label": self.config.label,
                "default_top_k": self.config.default_top_k,
                "token_sample_limit": self.config.token_sample_limit,
                "window_size_tokens": self.config.window_size_tokens,
                "window_size_events": self.config.window_size_events,
            },
            "totals": {
                "router_event_count": self.router_event_count,
                "token_count": self.token_count,
                "unique_experts_seen": len(self.expert_hits),
                "mean_routing_entropy": round3(
                    sum(self.routing_entropies) / len(self.routing_entropies)
                )
                if self.routing_entropies
                else None,
            },
            "breakdowns": {
                "by_module": dict(self.by_module),
                "by_layer": dict(self.by_layer),
                "by_source": dict(self.by_source),
                "top_experts": dict(self.expert_hits.most_common(20)),
            },
        }


class ForwardHookMoEProbe:
    def __init__(
        self,
        config: ForwardHookProbeConfig,
        adapter: GenericMoEForwardAdapter | None = None,
    ) -> None:
        self.config = config
        self.adapter = adapter or GenericMoEForwardAdapter()
        self.run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{config.label}"
        self.run_dir = config.output_dir / self.run_id
        ensure_dir(self.run_dir)
        self.manifest_path = self.run_dir / "manifest.json"
        self.router_events_path = self.run_dir / "router_events.jsonl"
        self.window_summaries_path = self.run_dir / "window_summaries.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.accumulator = ProbeAccumulator(run_id=self.run_id, config=config)
        self._handles: list[ForwardHookHandle] = []
        self._thread_local = threading.local()
        self._event_index = 0
        self._lock = threading.Lock()
        self._write_manifest()

    def _write_manifest(self) -> None:
        write_json(
            self.manifest_path,
            {
                "created_at": now_iso(),
                "run_id": self.run_id,
                "mode": "moe_forward_hook_probe",
                "shared_contract_template": contract_shell(
                    probe_tier="internal_hook",
                    backend_family="hookable_pytorch",
                    default_baseline_kind="observational",
                    default_window_size_tokens=self.config.window_size_tokens,
                ),
                "config": {
                    "label": self.config.label,
                    "default_top_k": self.config.default_top_k,
                    "token_sample_limit": self.config.token_sample_limit,
                    "window_size_tokens": self.config.window_size_tokens,
                    "window_size_events": self.config.window_size_events,
                },
            },
        )

    @contextmanager
    def span(self, metadata: JSONDict | None = None, span_id: str | None = None) -> Iterator[str]:
        previous_span_id = getattr(self._thread_local, "span_id", None)
        previous_metadata = getattr(self._thread_local, "span_metadata", None)
        next_span_id = span_id or f"span-{uuid.uuid4().hex[:12]}"
        self._thread_local.span_id = next_span_id
        self._thread_local.span_metadata = metadata or {}
        try:
            yield next_span_id
        finally:
            self._thread_local.span_id = previous_span_id
            self._thread_local.span_metadata = previous_metadata

    def attach(self, model: NamedModuleGraph) -> int:
        hook_count = 0
        for module_name, module in model.named_modules():
            if not self.adapter.should_hook(module_name, module):
                continue
            if not hasattr(module, "register_forward_hook"):
                continue
            handle = module.register_forward_hook(self._make_hook(module_name))
            self._handles.append(handle)
            hook_count += 1
        self._update_summary()
        return hook_count

    def close(self) -> None:
        while self.accumulator._window_events:
            window_summary = self.accumulator.flush_window()
            if window_summary:
                append_jsonl(self.window_summaries_path, window_summary)
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._update_summary()

    def _make_hook(self, module_name: str):
        def hook(module: Any, args: tuple[Any, ...], output: Any) -> None:
            self._record_hook_event(module_name, module, args, {}, output)

        return hook

    def _record_hook_event(
        self,
        module_name: str,
        module: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        extracted = self.adapter.extract_event(
            module_name=module_name,
            module=module,
            args=args,
            kwargs=kwargs,
            output=output,
            config=self.config,
        )
        if extracted is None:
            return

        with self._lock:
            self._event_index += 1
            span_id = getattr(self._thread_local, "span_id", None) or "unscoped"
            span_metadata = getattr(self._thread_local, "span_metadata", {}) or {}
            event = {
                "event_type": "router_event",
                "event_index": self._event_index,
                "timestamp": now_iso(),
                "span_id": span_id,
                "span_metadata": span_metadata if self.config.store_span_metadata else {},
                "span_metadata_sha256": sha256_text(json.dumps(span_metadata, sort_keys=True))
                if span_metadata
                else None,
                **extracted,
            }
            event = attach_shared_contract(
                event,
                build_shared_contract(
                    probe_tier="internal_hook",
                    backend_family="hookable_pytorch",
                    prompt_family=span_metadata.get("family_id") if isinstance(span_metadata, dict) else None,
                    window_size_tokens=self.config.window_size_tokens,
                    baseline_kind="observational",
                ),
            )
            append_jsonl(self.router_events_path, event)
            window_summaries = self.accumulator.ingest_event(event)
            for summary in window_summaries:
                if summary:
                    append_jsonl(self.window_summaries_path, summary)
            self._update_summary()

    def _update_summary(self) -> None:
        write_json(self.summary_path, self.accumulator.snapshot())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--label", default="moe-forward-probe")
    parser.add_argument("--default-top-k", type=int, default=2)
    parser.add_argument("--token-sample-limit", type=int, default=8)
    parser.add_argument("--window-size-tokens", type=int, default=32)
    parser.add_argument("--window-size-events", type=int)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = ForwardHookProbeConfig(
        output_dir=args.output_dir,
        label=args.label,
        default_top_k=args.default_top_k,
        token_sample_limit=args.token_sample_limit,
        window_size_tokens=args.window_size_tokens,
        window_size_events=args.window_size_events,
    )
    probe = ForwardHookMoEProbe(config=config)
    print(json.dumps({"run_dir": str(probe.run_dir), "run_id": probe.run_id}, indent=2))
    probe.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
