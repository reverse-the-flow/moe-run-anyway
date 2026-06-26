import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_hookable_moe_attempts.py"
MATRIX_PATH = ROOT / "memory-moe-mvp" / "data" / "hookable_moe_attempt_matrix.json"
SPEC = importlib.util.spec_from_file_location("plan_hookable_moe_attempts", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


class HookableMoEAttemptMatrixTests(unittest.TestCase):
    def test_default_matrix_validates_and_summarizes_pc_and_gx10(self) -> None:
        matrix = planner.load_matrix(MATRIX_PATH)
        errors = planner.validate_matrix(matrix)
        self.assertEqual(errors, [])

        summary = planner.summarize_matrix(matrix)
        self.assertEqual(summary["by_host"]["pc"], 7)
        self.assertEqual(summary["by_host"]["gx10"], 5)
        self.assertEqual(summary["real_model_semantic_hook_success_count"], 5)
        self.assertEqual(
            summary["by_status"]["blocked_missing_hook_runtime_dependencies"],
            1,
        )
        self.assertEqual(summary["by_status"]["blocked_host_memory_preflight"], 2)
        self.assertEqual(summary["by_status"]["blocked_llama_cpp_runtime_compatibility"], 1)
        self.assertEqual(summary["by_status"].get("engine_hook_candidate_uninstrumented", 0), 0)
        self.assertEqual(summary["by_status"].get("engine_debug_route_tensors_captured", 0), 0)
        self.assertEqual(summary["by_status"]["semantic_trace_captured"], 5)
        self.assertEqual(
            set(summary["real_model_semantic_hook_successes"]),
            {
                "gx10-gguf-dolphin-mixtral-8x7b",
                "gx10-gguf-qwen3-30b",
                "pc-ollama-dolphin-mixtral-8x7b",
                "pc-ollama-nemotron-cascade-30b-a3b-gguf",
                "pc-ollama-qwen3-coder-30b-a3b-gguf",
            },
        )
        blocker_ids = {
            item["attempt_id"]
            for item in summary["hookable_transformers_moe_blockers"]
        }
        self.assertEqual(
            blocker_ids,
            {"gx10-hf-nemotron-nano-omni-30b-a3b-nvfp4"},
        )

    def test_matrix_rejects_runtime_only_attempt_claiming_semantic_trace(self) -> None:
        matrix = planner.load_matrix(MATRIX_PATH)
        matrix["attempts"][5]["real_model_semantic_trace_captured"] = True

        errors = planner.validate_matrix(matrix)

        self.assertTrue(
            any("real_model_semantic_trace_captured=false" in error for error in errors),
            errors,
        )

    def test_matrix_rejects_non_moe_screen_without_false_moe_metadata(self) -> None:
        matrix = planner.load_matrix(MATRIX_PATH)
        matrix["attempts"][0]["moe_metadata"]["is_moe"] = True

        errors = planner.validate_matrix(matrix)

        self.assertTrue(
            any("screened_non_moe attempts must set moe_metadata.is_moe=false" in error for error in errors),
            errors,
        )

    def test_cli_reports_invalid_matrix(self) -> None:
        matrix = planner.load_matrix(MATRIX_PATH)
        matrix["schema_version"] = "bad"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "matrix.json"
            path.write_text(json.dumps(matrix), encoding="utf-8")
            loaded = planner.load_matrix(path)
            errors = planner.validate_matrix(loaded)

        self.assertIn("schema_version", errors[0])


if __name__ == "__main__":
    unittest.main()
