import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "build_dense_fallback_comparison.py"
SPEC = importlib.util.spec_from_file_location("build_dense_fallback_comparison", SCRIPT_PATH)
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def args(**overrides):
    values = {
        "managed_artifact": None,
        "dense_artifact": None,
        "output": None,
        "model_id": "fixture-mixtral.gguf",
        "prompt_family": "fixture",
        "managed_policy_id": "preload_shortlist",
        "default_quality_label": "unknown",
        "auto_label_exact": False,
        "template": False,
        "artifact_json": False,
        "json": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def ready_receipt(label: str) -> dict:
    return {
        "receipt_ready": True,
        "source_request_path": "memory-moe-mvp/phase3-real-evidence/pc_mixtral_phase3_real_evidence_bundle.runtime-capture-request.json",
        "source_prompt_set_path": "memory-moe-mvp/phase3-real-evidence/pc_mixtral_phase3_real_evidence_bundle.prompt-set.json",
        "runtime_capture_approved": True,
        "runtime_prompt_traffic_approved": True,
        "captured_at": "2026-06-28T21:00:00Z",
        "capture_host": "builder-test-host",
        "runtime_backend": "llama_cpp",
        "model_id": "fixture-mixtral.gguf",
        "prompt_family": "fixture",
        "output_label": label,
        "operator_notes": "test receipt",
    }


def output_summary(label: str, rows: list[dict] | None = None) -> dict:
    return {
        "schema_version": "moe-phase3-output-summary-v1",
        "model_id": "fixture-mixtral.gguf",
        "backend_family": "llama_cpp",
        "prompt_family": "fixture",
        "output_label": label,
        "output_ready": True,
        "capture_receipt": ready_receipt(label),
        "outputs": rows or [{"prompt_id": "case-001", "output": "same answer", "error": None}],
    }


class DenseFallbackComparisonBuilderTests(unittest.TestCase):
    def test_template_is_valid_but_not_ready(self) -> None:
        status, summary, artifact, error = builder.plan_build(args(template=True))

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        assert artifact is not None
        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(artifact["schema_version"], "moe-dense-fallback-comparison-v1")
        self.assertFalse(summary["comparison_ready_if_validated"])
        self.assertEqual(summary["missing_managed_output_count"], 1)
        self.assertEqual(summary["quality_delta_counts"]["unknown"], 1)

    def test_build_from_receipted_paired_json_rows_with_exact_auto_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            managed_path = Path(temp_dir) / "managed.json"
            dense_path = Path(temp_dir) / "dense.json"
            output_path = Path(temp_dir) / "fallback.json"
            managed_path.write_text(json.dumps(output_summary("managed")), encoding="utf-8")
            dense_path.write_text(json.dumps(output_summary("dense")), encoding="utf-8")

            status, summary, artifact, error = builder.plan_build(
                args(
                    managed_artifact=managed_path,
                    dense_artifact=dense_path,
                    output=output_path,
                    auto_label_exact=True,
                )
            )
            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        assert artifact is not None
        self.assertTrue(summary["comparison_ready_if_validated"], summary["errors"])
        self.assertEqual(artifact, written)
        self.assertTrue(artifact["builder"]["input_receipts"]["managed_capture_receipt_ready"])
        self.assertTrue(artifact["builder"]["input_receipts"]["dense_capture_receipt_ready"])
        self.assertTrue(artifact["builder"]["input_receipts"]["receipt_pair_consistent"])
        comparison = artifact["comparisons"][0]
        self.assertEqual(comparison["quality_delta_label"], "same")
        self.assertTrue(comparison["managed_output_present"])
        self.assertTrue(comparison["dense_output_present"])
        self.assertIn("sha256", comparison["managed_output_fingerprint"])

    def test_unreceipted_output_summaries_fail_and_do_not_write(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            managed_path = Path(temp_dir) / "managed.json"
            dense_path = Path(temp_dir) / "dense.json"
            output_path = Path(temp_dir) / "fallback.json"
            managed_path.write_text(json.dumps({"outputs": [{"prompt_id": "case-001", "output": "same answer"}]}), encoding="utf-8")
            dense_path.write_text(json.dumps({"outputs": [{"prompt_id": "case-001", "output": "same answer"}]}), encoding="utf-8")

            status, summary, artifact, error = builder.plan_build(
                args(managed_artifact=managed_path, dense_artifact=dense_path, output=output_path, auto_label_exact=True)
            )

        self.assertEqual(status, 2)
        self.assertIsNone(error)
        assert summary is not None
        assert artifact is not None
        self.assertFalse(summary["valid"])
        self.assertFalse(output_path.exists())
        self.assertTrue(any("capture_receipt must be an object" in item for item in summary["errors"]))
        self.assertFalse(artifact["builder"]["input_receipts"]["managed_capture_receipt_ready"])
        self.assertFalse(artifact["builder"]["input_receipts"]["dense_capture_receipt_ready"])

    def test_mismatched_receipt_pair_fails_and_does_not_write(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            managed_path = Path(temp_dir) / "managed.json"
            dense_path = Path(temp_dir) / "dense.json"
            output_path = Path(temp_dir) / "fallback.json"
            managed_payload = output_summary("managed")
            dense_payload = output_summary("dense")
            dense_payload["capture_receipt"]["source_request_path"] = "memory-moe-mvp/phase3-real-evidence/other.runtime-capture-request.json"
            managed_path.write_text(json.dumps(managed_payload), encoding="utf-8")
            dense_path.write_text(json.dumps(dense_payload), encoding="utf-8")

            status, summary, artifact, error = builder.plan_build(
                args(managed_artifact=managed_path, dense_artifact=dense_path, output=output_path, auto_label_exact=True)
            )

        self.assertEqual(status, 2)
        self.assertIsNone(error)
        assert summary is not None
        assert artifact is not None
        self.assertFalse(summary["valid"])
        self.assertFalse(output_path.exists())
        self.assertTrue(any("source_request_path" in item for item in summary["errors"]))
        self.assertFalse(artifact["builder"]["input_receipts"]["receipt_pair_consistent"])
        self.assertFalse(artifact["builder"]["input_receipts"]["managed_capture_receipt_ready"])
        self.assertFalse(artifact["builder"]["input_receipts"]["dense_capture_receipt_ready"])


    def test_ready_boolean_with_missing_source_prompt_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            managed_path = Path(temp_dir) / "managed.json"
            dense_path = Path(temp_dir) / "dense.json"
            output_path = Path(temp_dir) / "fallback.json"
            managed_payload = output_summary("managed")
            dense_payload = output_summary("dense")
            managed_payload["capture_receipt"]["source_prompt_set_path"] = str(Path(temp_dir) / "missing.prompt-set.json")
            dense_payload["capture_receipt"]["source_prompt_set_path"] = str(Path(temp_dir) / "missing.prompt-set.json")
            managed_path.write_text(json.dumps(managed_payload), encoding="utf-8")
            dense_path.write_text(json.dumps(dense_payload), encoding="utf-8")

            status, summary, artifact, error = builder.plan_build(
                args(managed_artifact=managed_path, dense_artifact=dense_path, output=output_path, auto_label_exact=True)
            )

        self.assertEqual(status, 2)
        self.assertIsNone(error)
        assert summary is not None
        assert artifact is not None
        self.assertFalse(summary["valid"])
        self.assertFalse(output_path.exists())
        self.assertTrue(any("source_prompt_set_path" in item and "existing file" in item for item in summary["errors"]))
        self.assertFalse(artifact["builder"]["input_receipts"]["managed_capture_receipt_ready"])
        self.assertFalse(artifact["builder"]["input_receipts"]["dense_capture_receipt_ready"])

    def test_runtime_probe_event_rows_are_supported_without_quality_overclaim(self) -> None:
        row = {
            "case": {"probe_id": "prose-summary-01", "repeat": 1},
            "response": {"summary": {"response_chars": 12, "finish_reason": "stop"}},
            "error": None,
        }
        comparisons, errors = builder.comparison_rows(
            [row],
            [row],
            default_quality_label="unknown",
            auto_label_exact=True,
        )

        self.assertEqual(errors, [])
        self.assertEqual(comparisons[0]["prompt_id"], "prose-summary-01#repeat-1")
        self.assertTrue(comparisons[0]["managed_output_present"])
        self.assertTrue(comparisons[0]["dense_output_present"])
        self.assertEqual(comparisons[0]["quality_delta_label"], "unknown")

    def test_missing_dense_row_is_marked_not_present(self) -> None:
        comparisons, errors = builder.comparison_rows(
            [{"prompt_id": "case-001", "output": "managed"}],
            [],
            default_quality_label="unknown",
            auto_label_exact=True,
        )

        self.assertEqual(errors, [])
        self.assertEqual(comparisons[0]["prompt_id"], "case-001")
        self.assertTrue(comparisons[0]["managed_output_present"])
        self.assertFalse(comparisons[0]["dense_output_present"])
        self.assertEqual(comparisons[0]["quality_delta_label"], "unknown")

    def test_duplicate_prompt_ids_fail_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            managed_path = Path(temp_dir) / "managed.json"
            dense_path = Path(temp_dir) / "dense.json"
            managed_rows = [
                {"prompt_id": "case-001", "output": "a", "error": None},
                {"prompt_id": "case-001", "output": "b", "error": None},
            ]
            managed_path.write_text(json.dumps(output_summary("managed", managed_rows)), encoding="utf-8")
            dense_path.write_text(json.dumps(output_summary("dense")), encoding="utf-8")

            status, summary, artifact, error = builder.plan_build(
                args(managed_artifact=managed_path, dense_artifact=dense_path)
            )

        self.assertEqual(status, 2)
        self.assertIsNone(error)
        assert summary is not None
        assert artifact is not None
        self.assertFalse(summary["valid"])
        self.assertTrue(any("duplicates prompt id" in item for item in summary["errors"]))


if __name__ == "__main__":
    unittest.main()
