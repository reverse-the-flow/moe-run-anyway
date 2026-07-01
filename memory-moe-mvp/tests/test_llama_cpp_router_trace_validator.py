import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "validate_llama_cpp_router_trace.py"
FIXTURE_PATH = ROOT / "memory-moe-mvp" / "data" / "llama_cpp_router_trace.fixture.jsonl"
SPEC = importlib.util.spec_from_file_location("validate_llama_cpp_router_trace", SCRIPT_PATH)
validator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def events_with_prompt_identity(repeats: int = 2) -> list[dict]:
    base_events = [json.loads(line) for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    events: list[dict] = []
    capture_sequence = 0
    for repeat in range(repeats):
        for event in base_events:
            enriched = dict(event)
            enriched.update(
                {
                    "prompt_id": "fixture-repeat-prompt",
                    "prompt_group_id": "fixture-repeat-group",
                    "repeat": repeat,
                    "repeat_index": repeat,
                    "capture_run_id": "unit-test-capture-run",
                    "capture_sequence": capture_sequence,
                }
            )
            events.append(enriched)
            capture_sequence += 1
    return events


class LlamaCppRouterTraceValidatorTests(unittest.TestCase):
    def test_fixture_validates_and_summarizes_layers(self) -> None:
        events = validator.load_jsonl(FIXTURE_PATH)
        errors = validator.validate_trace(
            events,
            expected_layers=2,
            require_kinds={"selected_experts", "selected_weights", "selected_weights_norm"},
        )
        self.assertEqual(errors, [])

        summary = validator.summarize_events(events)
        self.assertEqual(summary["event_count"], 6)
        self.assertEqual(summary["layer_count"], 2)
        self.assertEqual(summary["top_k_values"], [2])
        self.assertEqual(summary["token_count_values"], [2])
        self.assertEqual(summary["by_tensor_kind"]["selected_experts"], 2)
        self.assertFalse(summary["prompt_identity_ready"])
        self.assertEqual(summary["reuse_distance"]["observations"], 0)

    def test_strict_policy_candidate_flags_reject_trace_without_prompt_identity_or_reuse(self) -> None:
        events = validator.load_jsonl(FIXTURE_PATH)

        errors = validator.validate_trace(
            events,
            require_kinds={"selected_experts", "selected_weights", "selected_weights_norm"},
            require_prompt_identity=True,
            min_reuse_distance_observations=1,
        )

        self.assertTrue(any("prompt_identity_metadata_missing" in error for error in errors), errors)
        self.assertTrue(any("reuse_distance_observations 0 below required 1" in error for error in errors), errors)
        self.assertTrue(any("repeated routed expert keys" in error for error in errors), errors)

    def test_strict_policy_candidate_flags_accept_repeated_prompt_identity_trace(self) -> None:
        events = events_with_prompt_identity(repeats=2)

        errors = validator.validate_trace(
            events,
            require_kinds={"selected_experts", "selected_weights", "selected_weights_norm"},
            require_prompt_identity=True,
            min_reuse_distance_observations=1,
        )

        self.assertEqual(errors, [])
        summary = validator.summarize_events(events)
        self.assertTrue(summary["prompt_identity_ready"])
        self.assertGreaterEqual(summary["reuse_distance"]["observations"], 1)
        self.assertGreater(summary["route_repetition"]["repeated_route_key_count"], 0)

    def test_rejects_wrong_contract_version(self) -> None:
        events = validator.load_jsonl(FIXTURE_PATH)
        events[0]["contract_version"] = "wrong"

        errors = validator.validate_trace(events)

        self.assertTrue(any("contract_version" in error for error in errors), errors)

    def test_rejects_expert_event_without_i32_values(self) -> None:
        events = validator.load_jsonl(FIXTURE_PATH)
        events[0]["ggml_type"] = "f32"
        events[0]["values"] = [1.5]

        errors = validator.validate_trace(events)

        self.assertTrue(any("selected_experts must use ggml_type i32" in error for error in errors), errors)
        self.assertTrue(any("selected expert ids" in error for error in errors), errors)

    def test_rejects_missing_required_kind(self) -> None:
        events = [
            event
            for event in validator.load_jsonl(FIXTURE_PATH)
            if event["tensor_kind"] != "selected_weights_norm"
        ]

        errors = validator.validate_trace(events, require_kinds={"selected_weights_norm"})

        self.assertTrue(any("missing required tensor kinds" in error for error in errors), errors)

    def test_loader_rejects_invalid_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.jsonl"
            path.write_text("{bad\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                validator.load_jsonl(path)


if __name__ == "__main__":
    unittest.main()
