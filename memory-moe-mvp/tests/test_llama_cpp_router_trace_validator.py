import importlib.util
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
