import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import moe_forward_probe  # noqa: E402


class FakeHookHandle:
    def __init__(self, hooks, hook) -> None:
        self._hooks = hooks
        self._hook = hook

    def remove(self) -> None:
        if self._hook in self._hooks:
            self._hooks.remove(self._hook)


class FakeModule:
    def __init__(self, output=None) -> None:
        self._hooks = []
        self.output = output

    def register_forward_hook(self, hook):
        self._hooks.append(hook)
        return FakeHookHandle(self._hooks, hook)

    def forward(self):
        for hook in list(self._hooks):
            hook(self, tuple(), self.output)
        return self.output


class FakeModel:
    def __init__(self, modules):
        self._modules = modules

    def named_modules(self):
        yield "", self
        for name, module in self._modules:
            yield name, module


class MoEForwardProbeTests(unittest.TestCase):
    def test_collects_explicit_expert_indices(self) -> None:
        router = FakeModule(
            output={
                "expert_indices": [[1, 3], [1, 2]],
                "expert_weights": [[0.7, 0.3], [0.8, 0.2]],
            }
        )
        model = FakeModel([("layers.3.moe.router", router)])

        with tempfile.TemporaryDirectory() as temp_dir:
            probe = moe_forward_probe.ForwardHookMoEProbe(
                config=moe_forward_probe.ForwardHookProbeConfig(
                    output_dir=Path(temp_dir),
                    label="test-indices",
                    window_size_events=4,
                )
            )
            hook_count = probe.attach(model)
            self.assertEqual(hook_count, 1)

            with probe.span({"prompt_id": "case-1"}):
                router.forward()

            probe.close()

            summary = json_load(probe.summary_path)
            self.assertEqual(summary["totals"]["router_event_count"], 1)
            self.assertEqual(summary["totals"]["token_count"], 2)
            self.assertEqual(summary["breakdowns"]["top_experts"]["1"], 2)

            events = read_jsonl(probe.router_events_path)
            self.assertEqual(events[0]["span_metadata"]["prompt_id"], "case-1")
            self.assertEqual(events[0]["layer_id"], 3)
            self.assertEqual(events[0]["source"], "expert_indices")

    def test_falls_back_to_router_logits(self) -> None:
        router = FakeModule(
            output={
                "router_logits": [
                    [0.1, 2.0, 0.3],
                    [1.5, 0.4, 0.2],
                ]
            }
        )
        model = FakeModel([("blocks.7.gate", router)])

        with tempfile.TemporaryDirectory() as temp_dir:
            probe = moe_forward_probe.ForwardHookMoEProbe(
                config=moe_forward_probe.ForwardHookProbeConfig(
                    output_dir=Path(temp_dir),
                    label="test-logits",
                    default_top_k=2,
                    window_size_events=4,
                )
            )
            probe.attach(model)
            router.forward()
            probe.close()

            events = read_jsonl(probe.router_events_path)
            self.assertEqual(events[0]["source"], "router_logits")
            self.assertEqual(events[0]["layer_id"], 7)
            self.assertEqual(events[0]["token_count"], 2)
            self.assertEqual(events[0]["top_k"], 2)
            self.assertIn("1", events[0]["hit_counts"])

    def test_writes_window_summary_when_threshold_reached(self) -> None:
        router = FakeModule(
            output={
                "expert_indices": [[0, 1]],
                "expert_weights": [[0.9, 0.1]],
            }
        )
        model = FakeModel([("layers.1.moe.router", router)])

        with tempfile.TemporaryDirectory() as temp_dir:
            probe = moe_forward_probe.ForwardHookMoEProbe(
                config=moe_forward_probe.ForwardHookProbeConfig(
                    output_dir=Path(temp_dir),
                    label="test-window",
                    window_size_events=2,
                )
            )
            probe.attach(model)
            router.forward()
            router.forward()
            probe.close()

            windows = read_jsonl(probe.window_summaries_path)
            self.assertEqual(len(windows), 1)
            self.assertEqual(windows[0]["window_event_count"], 2)
            self.assertEqual(windows[0]["window_token_count"], 2)
            self.assertEqual(windows[0]["top_experts"]["0"], 1.8)
            self.assertEqual(windows[0]["shared_contract"]["window_size_tokens"], 32)


def json_load(path: Path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    import json

    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            lines.append(json.loads(line))
    return lines


if __name__ == "__main__":
    unittest.main()
