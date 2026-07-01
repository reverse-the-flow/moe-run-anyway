import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "build_phase3_output_summary.py"
SPEC = importlib.util.spec_from_file_location("build_phase3_output_summary", SCRIPT_PATH)
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)

import plan_phase3_dense_fallback_capture

PROMPT_SET = ROOT / "memory-moe-mvp" / "phase3-real-evidence" / "pc_mixtral_phase3_real_evidence_bundle.prompt-set.json"


def args(**overrides):
    values = {
        "prompt_set_path": PROMPT_SET,
        "output_label": "managed",
        "input_summary": None,
        "output": None,
        "default_output": False,
        "artifact_json": False,
        "json": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def fill_outputs(artifact) -> None:
    for row in artifact["outputs"]:
        row["output"] = f"answer for {row['prompt_id']}"
        row["error"] = None
        row["ready"] = True


def mark_receipt_ready(artifact, *, output_label: str) -> None:
    artifact["capture_receipt"].update(
        {
            "receipt_ready": True,
            "source_request_path": "memory-moe-mvp/phase3-real-evidence/pc_mixtral_phase3_real_evidence_bundle.runtime-capture-request.json",
            "runtime_capture_approved": True,
            "runtime_prompt_traffic_approved": True,
            "captured_at": "2026-06-28T21:00:00Z",
            "capture_host": "pc-smoke-test",
            "runtime_backend": "llama_cpp",
            "model_id": "mixtral-test-model",
            "prompt_family": "pc-router-trace-mixtral-jsonl",
            "output_label": output_label,
        }
    )


class Phase3OutputSummaryBuilderTests(unittest.TestCase):
    def test_template_has_prompt_coverage_but_is_not_ready(self) -> None:
        status, summary, artifact, error = builder.plan_build(args())

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        assert artifact is not None
        self.assertTrue(summary["valid"], summary["errors"])
        self.assertFalse(summary["summary_ready"])
        self.assertFalse(summary["capture_receipt_ready"])
        self.assertEqual(summary["row_count"], 8)
        self.assertEqual(summary["missing_output_count"], 8)
        self.assertEqual(artifact["schema_version"], "moe-phase3-output-summary-v1")
        self.assertEqual(artifact["output_label"], "managed")
        self.assertFalse(artifact["capture_receipt"]["receipt_ready"])

    def test_written_template_is_valid_not_ready_for_dense_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "managed-output-summary.json"
            status, summary, artifact, error = builder.plan_build(args(output=output))
            capture_summary = plan_phase3_dense_fallback_capture.summarize_output_summary(output, label="managed")

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        assert artifact is not None
        self.assertTrue(summary["valid"], summary["errors"])
        self.assertFalse(summary["summary_ready"])
        self.assertTrue(capture_summary["valid"], capture_summary["errors"])
        self.assertFalse(capture_summary["ready"])
        self.assertFalse(capture_summary["capture_receipt_ready"])
        self.assertEqual(capture_summary["missing_output_count"], 8)

    def test_filled_summary_without_ready_receipt_is_not_ready(self) -> None:
        artifact = builder.build_template_artifact(PROMPT_SET, output_label="dense")
        fill_outputs(artifact)
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "dense-output-summary.json"
            input_path.write_text(json.dumps(artifact), encoding="utf-8")
            status, summary, _, error = builder.plan_build(args(output_label="dense", input_summary=input_path))

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        self.assertTrue(summary["valid"], summary["errors"])
        self.assertFalse(summary["summary_ready"])
        self.assertFalse(summary["capture_receipt_ready"])
        self.assertEqual(summary["missing_output_count"], 0)

    def test_filled_summary_is_ready_when_receipt_is_ready(self) -> None:
        artifact = builder.build_template_artifact(PROMPT_SET, output_label="dense")
        fill_outputs(artifact)
        mark_receipt_ready(artifact, output_label="dense")
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "dense-output-summary.json"
            input_path.write_text(json.dumps(artifact), encoding="utf-8")
            status, summary, _, error = builder.plan_build(args(output_label="dense", input_summary=input_path))

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        self.assertTrue(summary["valid"], summary["errors"])
        self.assertTrue(summary["summary_ready"])
        self.assertTrue(summary["capture_receipt_ready"])
        self.assertEqual(summary["missing_output_count"], 0)

    def test_ready_receipt_with_missing_source_prompt_is_not_ready(self) -> None:
        artifact = builder.build_template_artifact(PROMPT_SET, output_label="dense")
        fill_outputs(artifact)
        mark_receipt_ready(artifact, output_label="dense")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "dense-output-summary.json"
            artifact["capture_receipt"]["source_prompt_set_path"] = str(temp_path / "missing.prompt-set.json")
            input_path.write_text(json.dumps(artifact), encoding="utf-8")
            status, summary, _, error = builder.plan_build(args(output_label="dense", input_summary=input_path))

        self.assertEqual(status, 2)
        self.assertIsNone(error)
        assert summary is not None
        self.assertFalse(summary["valid"])
        self.assertFalse(summary["summary_ready"])
        self.assertFalse(summary["capture_receipt_ready"])
        self.assertTrue(any("source_prompt_set_path" in item and "existing file" in item for item in summary["errors"]))

    def test_wrong_label_fails_cleanly(self) -> None:
        artifact = builder.build_template_artifact(PROMPT_SET, output_label="dense")
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "dense-output-summary.json"
            input_path.write_text(json.dumps(artifact), encoding="utf-8")
            status, summary, _, error = builder.plan_build(args(output_label="managed", input_summary=input_path))

        self.assertEqual(status, 2)
        self.assertIsNone(error)
        assert summary is not None
        self.assertFalse(summary["valid"])
        self.assertTrue(any("output_label must be 'managed'" in item for item in summary["errors"]))


if __name__ == "__main__":
    unittest.main()