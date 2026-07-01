import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "build_phase3_runtime_capture_request.py"
SPEC = importlib.util.spec_from_file_location("build_phase3_runtime_capture_request", SCRIPT_PATH)
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)

import build_phase3_trace_receipt

BUNDLE = ROOT / "memory-moe-mvp" / "phase3-real-evidence" / "pc_mixtral_phase3_real_evidence_bundle.json"
FIXTURE_TRACE = ROOT / "memory-moe-mvp" / "data" / "llama_cpp_router_trace.fixture.jsonl"


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def trace_text_with_prompt_identity(repeats: int = 2) -> str:
    lines: list[str] = []
    capture_sequence = 0
    fixture_events = [json.loads(line) for line in FIXTURE_TRACE.read_text(encoding="utf-8").splitlines() if line.strip()]
    for repeat in range(repeats):
        for event in fixture_events:
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
            lines.append(json.dumps(enriched, sort_keys=True))
            capture_sequence += 1
    return "\n".join(lines) + "\n"

def write_ready_trace_receipt(
    receipt_path: Path,
    *,
    candidate_path: Path,
    overrides: dict | None = None,
) -> None:
    receipt = build_phase3_trace_receipt.build_template_artifact(BUNDLE, candidate_trace_path=candidate_path)
    receipt.update(
        {
            "receipt_ready": True,
            "source_request_path": "memory-moe-mvp/phase3-real-evidence/pc_mixtral_phase3_real_evidence_bundle.runtime-capture-request.json",
            "router_trace_capture_approved": True,
            "runtime_prompt_traffic_approved": True,
            "captured_at": "2026-06-28T21:00:00Z",
            "capture_host": "pc-request-test",
        }
    )
    if overrides:
        receipt.update(overrides)
    write_json(receipt_path, receipt)


def args(**overrides):
    values = {
        "bundle_path": BUNDLE,
        "prompt_set_path": None,
        "candidate_trace_path": None,
        "candidate_trace_receipt_path": None,
        "managed_output_path": None,
        "dense_output_path": None,
        "live_proof_template_path": None,
        "router_trace_capture_approved": False,
        "managed_output_capture_approved": False,
        "dense_output_capture_approved": False,
        "runtime_prompt_traffic_approved": False,
        "output": None,
        "default_output": False,
        "json": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class Phase3RuntimeCaptureRequestBuilderTests(unittest.TestCase):
    def test_default_request_is_valid_but_requires_approval(self) -> None:
        status, summary, error = builder.plan_build(args())

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        self.assertTrue(summary["valid"], summary["errors"])
        self.assertFalse(summary["ready_for_operator_capture"])
        self.assertFalse(summary["capture_complete"])
        by_id = {item["id"]: item for item in summary["requested_artifacts"]}
        self.assertEqual(by_id["candidate_router_trace"]["status"], "approval_required")
        self.assertIn("capture_receipt", by_id["candidate_router_trace"]["description"])
        self.assertTrue(by_id["candidate_router_trace"]["source"]["capture_receipt_required"])
        self.assertTrue(by_id["candidate_router_trace"]["source"]["capture_receipt_ready"])
        self.assertTrue(by_id["candidate_router_trace"]["source"]["capture_receipt_path"].endswith("candidate-router-events.capture-receipt.json"))
        self.assertFalse(by_id["candidate_router_trace"]["source"]["candidate_trace_reuse_ready"])
        self.assertEqual(by_id["candidate_router_trace"]["source"]["candidate_trace_reuse_distance_observations"], 0)
        self.assertIn("no_reuse_distance_observations", by_id["candidate_router_trace"]["source"]["candidate_trace_blockers"])
        self.assertEqual(len(by_id["candidate_router_trace"]["validator_commands"]), 3)
        trace_validator = by_id["candidate_router_trace"]["validator_commands"][0]
        self.assertIn("--require-prompt-identity", trace_validator)
        self.assertIn("--min-reuse-distance-observations", trace_validator)
        self.assertEqual(
            trace_validator[trace_validator.index("--min-reuse-distance-observations") + 1],
            str(builder.plan_phase3_reuse_evidence_capture.MIN_REUSE_DISTANCE_OBSERVATIONS),
        )
        self.assertEqual(by_id["managed_output_summary_fill"]["status"], "approval_required")
        self.assertEqual(by_id["dense_output_summary_fill"]["status"], "approval_required")
        self.assertIn("capture_receipt", by_id["managed_output_summary_fill"]["description"])
        self.assertIn("capture_receipt", by_id["dense_output_summary_fill"]["description"])
        self.assertTrue(by_id["managed_output_summary_fill"]["source"]["capture_receipt_required"])
        self.assertTrue(by_id["dense_output_summary_fill"]["source"]["capture_receipt_required"])
        self.assertFalse(by_id["managed_output_summary_fill"]["source"]["capture_receipt_ready"])
        self.assertFalse(by_id["dense_output_summary_fill"]["source"]["capture_receipt_ready"])
        future_by_id = {item["id"]: item for item in summary["future_artifacts"]}
        self.assertEqual(future_by_id["live_capability_proof_fill"]["status"], "future_adapter_required")
        self.assertTrue(summary["live_capability_proof"]["proof_available"])
        self.assertFalse(summary["live_capability_proof"]["proof_ready"])
        self.assertTrue(summary["prompt_set"]["ready"])
        self.assertTrue(summary["prompt_coverage"]["managed"]["ready"])
        self.assertTrue(summary["prompt_coverage"]["dense"]["ready"])

    def test_approval_marks_request_ready_for_operator_capture(self) -> None:
        status, summary, error = builder.plan_build(
            args(
                router_trace_capture_approved=True,
                managed_output_capture_approved=True,
                dense_output_capture_approved=True,
                runtime_prompt_traffic_approved=True,
            )
        )

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        self.assertTrue(summary["ready_for_operator_capture"], summary["errors"])
        by_id = {item["id"]: item for item in summary["requested_artifacts"]}
        self.assertEqual(by_id["candidate_router_trace"]["status"], "ready_for_operator_capture")
        self.assertEqual(by_id["managed_output_summary_fill"]["status"], "ready_for_operator_capture")
        self.assertEqual(by_id["dense_output_summary_fill"]["status"], "ready_for_operator_capture")
        future_by_id = {item["id"]: item for item in summary["future_artifacts"]}
        self.assertEqual(future_by_id["live_capability_proof_fill"]["status"], "future_adapter_required")

    def test_default_output_writes_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "runtime-capture-request.json"
            status, summary, error = builder.plan_build(args(output=output))
            written = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        self.assertEqual(written["schema_version"], "moe-phase3-runtime-capture-request-v1")
        self.assertEqual(written["future_artifacts"][0]["id"], "live_capability_proof_fill")
        self.assertEqual(summary["output_path"], str(output))


    def test_ready_repeated_trace_receipt_marks_candidate_trace_satisfied(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            candidate_path = temp_path / "candidate-router-events.jsonl"
            receipt_path = temp_path / "candidate-router-events.capture-receipt.json"
            candidate_path.write_text(trace_text_with_prompt_identity(repeats=2), encoding="utf-8")
            write_ready_trace_receipt(receipt_path, candidate_path=candidate_path)

            status, summary, error = builder.plan_build(
                args(
                    candidate_trace_path=candidate_path,
                    candidate_trace_receipt_path=receipt_path,
                )
            )

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        by_id = {item["id"]: item for item in summary["requested_artifacts"]}
        self.assertEqual(by_id["candidate_router_trace"]["status"], "already_satisfied")
        self.assertTrue(by_id["candidate_router_trace"]["source"]["capture_receipt_ready"])
        self.assertTrue(by_id["candidate_router_trace"]["source"]["capture_receipt_valid"])
        self.assertTrue(by_id["candidate_router_trace"]["source"]["candidate_trace_reuse_ready"])
        self.assertGreaterEqual(by_id["candidate_router_trace"]["source"]["candidate_trace_reuse_distance_observations"], 1)
        self.assertEqual(by_id["candidate_router_trace"]["source"]["capture_receipt_errors"], [])

    def test_ready_trace_receipt_with_wrong_model_does_not_satisfy_candidate_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            candidate_path = temp_path / "candidate-router-events.jsonl"
            receipt_path = temp_path / "candidate-router-events.capture-receipt.json"
            candidate_path.write_text("{}\n", encoding="utf-8")
            write_ready_trace_receipt(
                receipt_path,
                candidate_path=candidate_path,
                overrides={"model_id": "wrong-model"},
            )

            status, summary, error = builder.plan_build(
                args(
                    candidate_trace_path=candidate_path,
                    candidate_trace_receipt_path=receipt_path,
                    router_trace_capture_approved=True,
                    runtime_prompt_traffic_approved=True,
                )
            )

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        self.assertFalse(summary["capture_complete"])
        by_id = {item["id"]: item for item in summary["requested_artifacts"]}
        self.assertEqual(by_id["candidate_router_trace"]["status"], "ready_for_operator_capture")
        self.assertFalse(by_id["candidate_router_trace"]["source"]["capture_receipt_ready"])
        self.assertFalse(by_id["candidate_router_trace"]["source"]["capture_receipt_valid"])
        self.assertIn("candidate_trace_capture_receipt_invalid", by_id["candidate_router_trace"]["source"]["capture_receipt_blockers"])
        self.assertTrue(
            any("model_id" in error for error in by_id["candidate_router_trace"]["source"]["capture_receipt_errors"])
        )
if __name__ == "__main__":
    unittest.main()

