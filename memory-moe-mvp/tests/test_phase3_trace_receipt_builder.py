import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "build_phase3_trace_receipt.py"
SPEC = importlib.util.spec_from_file_location("build_phase3_trace_receipt", SCRIPT_PATH)
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)

BUNDLE = ROOT / "memory-moe-mvp" / "phase3-real-evidence" / "pc_mixtral_phase3_real_evidence_bundle.json"


def args(**overrides):
    values = {
        "bundle_path": BUNDLE,
        "prompt_set_path": None,
        "candidate_trace_path": None,
        "input_receipt": None,
        "output": None,
        "default_output": False,
        "artifact_json": False,
        "json": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def mark_receipt_ready(artifact, *, output_path: Path) -> None:
    artifact.update(
        {
            "receipt_ready": True,
            "source_request_path": "memory-moe-mvp/phase3-real-evidence/pc_mixtral_phase3_real_evidence_bundle.runtime-capture-request.json",
            "router_trace_capture_approved": True,
            "runtime_prompt_traffic_approved": True,
            "captured_at": "2026-06-28T21:00:00Z",
            "capture_host": "pc-smoke-test",
            "candidate_trace_path": str(output_path.with_name("candidate-router-events.jsonl")),
        }
    )


class Phase3TraceReceiptBuilderTests(unittest.TestCase):
    def test_template_is_valid_but_not_ready(self) -> None:
        status, summary, artifact, error = builder.plan_build(args())

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        assert artifact is not None
        self.assertTrue(summary["valid"], summary["errors"])
        self.assertFalse(summary["receipt_ready"])
        self.assertEqual(artifact["schema_version"], "moe-phase3-trace-capture-receipt-v1")
        self.assertFalse(artifact["receipt_ready"])
        self.assertEqual(artifact["runtime_backend"], "llama_cpp")
        self.assertTrue(artifact["candidate_trace_path"].endswith("candidate-router-events.jsonl"))

    def test_default_output_writes_valid_template(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / BUNDLE.name
            bundle.write_text(BUNDLE.read_text(encoding="utf-8"), encoding="utf-8")
            prompt_path = builder.canonical_prompt_set_path(bundle)
            prompt_path.write_text(
                (ROOT / "memory-moe-mvp" / "phase3-real-evidence" / "pc_mixtral_phase3_real_evidence_bundle.prompt-set.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            status, summary, artifact, error = builder.plan_build(args(bundle_path=bundle, default_output=True))
            output_path = Path(summary["output_path"]) if Path(summary["output_path"]).is_absolute() else ROOT / summary["output_path"]

            self.assertEqual(status, 0)
            self.assertIsNone(error)
            assert artifact is not None
            self.assertTrue(summary["valid"], summary["errors"])
            self.assertTrue(output_path.exists())
            written = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(written["schema_version"], "moe-phase3-trace-capture-receipt-v1")

    def test_filled_receipt_is_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            candidate_path = temp_path / "candidate-router-events.jsonl"
            output_path = temp_path / "candidate-router-events.capture-receipt.json"
            candidate_path.write_text("{}\n", encoding="utf-8")
            artifact = builder.build_template_artifact(BUNDLE, candidate_trace_path=candidate_path)
            mark_receipt_ready(artifact, output_path=output_path)
            output_path.write_text(json.dumps(artifact), encoding="utf-8")

            status, summary, _, error = builder.plan_build(
                args(
                    candidate_trace_path=candidate_path,
                    input_receipt=output_path,
                )
            )

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        self.assertTrue(summary["valid"], summary["errors"])
        self.assertTrue(summary["receipt_ready"])

    def test_ready_receipt_with_missing_source_request_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            candidate_path = temp_path / "candidate-router-events.jsonl"
            output_path = temp_path / "candidate-router-events.capture-receipt.json"
            candidate_path.write_text("{}\n", encoding="utf-8")
            artifact = builder.build_template_artifact(BUNDLE, candidate_trace_path=candidate_path)
            mark_receipt_ready(artifact, output_path=output_path)
            artifact["source_request_path"] = str(temp_path / "missing.runtime-capture-request.json")
            output_path.write_text(json.dumps(artifact), encoding="utf-8")

            status, summary, _, error = builder.plan_build(
                args(
                    candidate_trace_path=candidate_path,
                    input_receipt=output_path,
                )
            )

        self.assertEqual(status, 2)
        self.assertIsNone(error)
        assert summary is not None
        self.assertFalse(summary["valid"])
        self.assertFalse(summary["receipt_ready"])
        self.assertTrue(any("source_request_path" in item and "existing file" in item for item in summary["errors"]))

    def test_wrong_candidate_trace_path_fails_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            candidate_path = temp_path / "candidate-router-events.jsonl"
            output_path = temp_path / "candidate-router-events.capture-receipt.json"
            artifact = builder.build_template_artifact(BUNDLE, candidate_trace_path=candidate_path)
            artifact["candidate_trace_path"] = str(temp_path / "other-router-events.jsonl")
            output_path.write_text(json.dumps(artifact), encoding="utf-8")

            status, summary, _, error = builder.plan_build(
                args(
                    candidate_trace_path=candidate_path,
                    input_receipt=output_path,
                )
            )

        self.assertEqual(status, 2)
        self.assertIsNone(error)
        assert summary is not None
        self.assertFalse(summary["valid"])
        self.assertTrue(any("candidate_trace_path" in item for item in summary["errors"]))


if __name__ == "__main__":
    unittest.main()
