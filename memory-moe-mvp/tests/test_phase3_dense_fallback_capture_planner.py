import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_phase3_dense_fallback_capture.py"
SPEC = importlib.util.spec_from_file_location("plan_phase3_dense_fallback_capture", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)

MIXTRAL_BUNDLE = ROOT / "memory-moe-mvp" / "phase3-real-evidence" / "pc_mixtral_phase3_real_evidence_bundle.json"
MODEL_ID = "/root/.ollama/models/blobs/sha256-5041ba4278429fe475782b889471b5ff065a6cce5c3a539bd61d1e457f1961de"


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prompt_set():
    return {
        "schema_version": "moe-phase3-fallback-prompt-set-v1",
        "model_id": MODEL_ID,
        "backend_family": "llama_cpp",
        "prompt_family": "pc-router-trace-mixtral-jsonl",
        "prompts": [
            {"prompt_id": "case-001", "prompt": "Summarize why fallback comparison matters."},
            {"prompt_id": "case-002", "messages": [{"role": "user", "content": "Give a short answer."}]},
        ],
    }


def ready_receipt(label: str, *, request_path: Path | str | None = None, prompt_path: Path | str | None = None):
    return {
        "receipt_ready": True,
        "source_request_path": str(request_path or "memory-moe-mvp/phase3-real-evidence/pc_mixtral_phase3_real_evidence_bundle.runtime-capture-request.json"),
        "source_prompt_set_path": str(prompt_path or "memory-moe-mvp/phase3-real-evidence/pc_mixtral_phase3_real_evidence_bundle.prompt-set.json"),
        "runtime_capture_approved": True,
        "runtime_prompt_traffic_approved": True,
        "captured_at": "2026-06-28T21:00:00Z",
        "capture_host": "pc-smoke-test",
        "runtime_backend": "llama_cpp",
        "model_id": MODEL_ID,
        "prompt_family": "pc-router-trace-mixtral-jsonl",
        "output_label": label,
        "operator_notes": "test receipt",
    }


def output_summary(label: str, *, request_path: Path | str | None = None, prompt_path: Path | str | None = None):
    return {
        "schema_version": "moe-phase3-output-summary-v1",
        "source_prompt_set_path": "prompt-set.json",
        "model_id": MODEL_ID,
        "backend_family": "llama_cpp",
        "prompt_family": "pc-router-trace-mixtral-jsonl",
        "output_label": label,
        "output_ready": True,
        "capture_receipt": ready_receipt(label, request_path=request_path, prompt_path=prompt_path),
        "outputs": [
            {"prompt_id": "case-001", "output": f"{label} answer one", "error": None},
            {"prompt_id": "case-002", "output": f"{label} answer two", "error": None},
        ],
    }


def output_summary_without_receipt(label: str):
    return {
        "outputs": [
            {"prompt_id": "case-001", "output": f"{label} answer one"},
            {"prompt_id": "case-002", "output": f"{label} answer two"},
        ],
    }


def fallback_artifact():
    return {
        "schema_version": "moe-dense-fallback-comparison-v1",
        "model_id": MODEL_ID,
        "prompt_family": "pc-router-trace-mixtral-jsonl",
        "managed_policy_id": "preload_shortlist",
        "managed_artifact": "managed-output-summary.json",
        "dense_artifact": "dense-output-summary.json",
        "builder": {
            "schema_version": "moe-dense-fallback-comparison-builder-v1",
            "mode": "paired_output_summary",
            "input_receipts": {
                "managed_capture_receipt_ready": True,
                "dense_capture_receipt_ready": True,
                "receipt_pair_consistent": True,
                "receipt_gate": "required_before_write",
            },
        },
        "comparisons": [
            {
                "prompt_id": "case-001",
                "managed_output_present": True,
                "dense_output_present": True,
                "quality_delta_label": "same",
            },
            {
                "prompt_id": "case-002",
                "managed_output_present": True,
                "dense_output_present": True,
                "quality_delta_label": "minor_delta",
            },
        ],
    }


class Phase3DenseFallbackCapturePlannerTests(unittest.TestCase):
    def test_default_mixtral_plan_uses_shared_prompt_set_and_names_output_blockers(self) -> None:
        summary = planner.build_capture_plan(MIXTRAL_BUNDLE)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertTrue(summary["prompt_set"]["ready"])
        self.assertFalse(summary["metadata_ready_to_build_comparison"])
        self.assertFalse(summary["comparison_ready"])
        by_request = {item["id"]: item for item in summary["artifact_requests"]}
        self.assertEqual(by_request["prompt_set_artifact"]["status"], "already_satisfied")
        self.assertEqual(by_request["managed_output_summary"]["status"], "approval_required")
        self.assertEqual(by_request["dense_output_summary"]["status"], "approval_required")
        self.assertEqual(by_request["dense_fallback_comparison_artifact"]["status"], "blocked_by_saved_outputs")
        self.assertEqual(summary["policy_warning"]["id"], "no_replay_policy_candidate_after_trace_inventory_replay")
        command_classes = {item["command_class"] for item in summary["commands"]}
        self.assertIn("dense_fallback_comparison_builder", command_classes)
        self.assertIn("phase3_bundle_builder_with_fallback", command_classes)

    def test_markdown_report_contains_artifact_requests_contract_and_commands(self) -> None:
        summary = planner.build_capture_plan(MIXTRAL_BUNDLE)

        report = planner.format_markdown_report(summary)

        self.assertIn("# Phase 3 Dense Fallback Capture", report)
        self.assertIn("## Artifact Requests", report)
        self.assertIn("managed_output_summary", report)
        self.assertIn("managed_output_summary_missing_outputs", report)
        self.assertIn("## Runtime Capture Contract", report)
        self.assertIn("Requires explicit approval: `True`", report)
        self.assertIn("## Commands", report)
        self.assertIn("dense_fallback_comparison_builder", report)
        self.assertIn("uv run --managed-python --python 3.13 scripts/build_dense_fallback_comparison.py", report)
        self.assertIn("phase3_bundle_builder_with_fallback", report)
        self.assertIn("planner does not send prompt traffic", report)

    def test_markdown_report_writes_requested_path(self) -> None:
        summary = planner.build_capture_plan(MIXTRAL_BUNDLE)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "reports" / "dense-fallback.md"
            planner.write_markdown_report(summary, output_path)
            written = output_path.read_text(encoding="utf-8")

        self.assertIn("# Phase 3 Dense Fallback Capture", written)
        self.assertIn("dense_fallback_comparison_builder", written)

    def test_saved_prompt_and_output_summaries_make_comparison_ready_to_build(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            prompt_path = temp_path / "prompt-set.json"
            managed_path = temp_path / "managed.json"
            dense_path = temp_path / "dense.json"
            request_path = temp_path / "phase3-bundle.runtime-capture-request.json"
            write_json(prompt_path, prompt_set())
            write_json(request_path, {"schema_version": "test-runtime-capture-request"})
            write_json(managed_path, output_summary("managed", request_path=request_path, prompt_path=prompt_path))
            write_json(dense_path, output_summary("dense", request_path=request_path, prompt_path=prompt_path))

            summary = planner.build_capture_plan(
                MIXTRAL_BUNDLE,
                output_dir=temp_path,
                prompt_set_path=prompt_path,
                managed_output_path=managed_path,
                dense_output_path=dense_path,
                dense_fallback_capture_approved=True,
                runtime_prompt_traffic_approved=True,
            )

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertTrue(summary["prompt_set"]["ready"])
        self.assertTrue(summary["managed_output_summary"]["ready"])
        self.assertTrue(summary["dense_output_summary"]["ready"])
        self.assertTrue(summary["managed_output_summary"]["capture_receipt_ready"])
        self.assertTrue(summary["dense_output_summary"]["capture_receipt_ready"])
        self.assertTrue(summary["prompt_coverage"]["managed"]["ready"])
        self.assertTrue(summary["prompt_coverage"]["dense"]["ready"])
        self.assertTrue(summary["metadata_ready_to_build_comparison"])
        self.assertFalse(summary["comparison_ready"])
        by_request = {item["id"]: item for item in summary["artifact_requests"]}
        self.assertEqual(by_request["dense_fallback_comparison_artifact"]["status"], "ready_to_build")
        self.assertFalse(summary["runtime_capture_contract"]["requires_explicit_approval"])

    def test_saved_outputs_without_receipts_do_not_make_comparison_ready_to_build(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            prompt_path = temp_path / "prompt-set.json"
            managed_path = temp_path / "managed.json"
            dense_path = temp_path / "dense.json"
            write_json(prompt_path, prompt_set())
            write_json(managed_path, output_summary_without_receipt("managed"))
            write_json(dense_path, output_summary_without_receipt("dense"))

            summary = planner.build_capture_plan(
                MIXTRAL_BUNDLE,
                output_dir=temp_path,
                prompt_set_path=prompt_path,
                managed_output_path=managed_path,
                dense_output_path=dense_path,
                dense_fallback_capture_approved=True,
                runtime_prompt_traffic_approved=True,
            )

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["metadata_ready_to_build_comparison"])
        self.assertEqual(summary["managed_output_summary"]["blocker"], "managed_output_summary_invalid")
        self.assertTrue(any("capture_receipt must be an object" in item for item in summary["errors"]))

    def test_existing_fallback_artifact_marks_comparison_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            prompt_path = temp_path / "prompt-set.json"
            managed_path = temp_path / "managed.json"
            dense_path = temp_path / "dense.json"
            fallback_path = temp_path / "fallback.json"
            request_path = temp_path / "phase3-bundle.runtime-capture-request.json"
            write_json(prompt_path, prompt_set())
            write_json(request_path, {"schema_version": "test-runtime-capture-request"})
            write_json(managed_path, output_summary("managed", request_path=request_path, prompt_path=prompt_path))
            write_json(dense_path, output_summary("dense", request_path=request_path, prompt_path=prompt_path))
            write_json(fallback_path, fallback_artifact())

            summary = planner.build_capture_plan(
                MIXTRAL_BUNDLE,
                output_dir=temp_path,
                prompt_set_path=prompt_path,
                managed_output_path=managed_path,
                dense_output_path=dense_path,
                fallback_artifact_path=fallback_path,
            )

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertTrue(summary["comparison_ready"])
        by_request = {item["id"]: item for item in summary["artifact_requests"]}
        self.assertEqual(by_request["dense_fallback_comparison_artifact"]["status"], "already_satisfied")
        self.assertEqual(by_request["phase3_bundle_with_fallback"]["status"], "ready_to_build")

    def test_invalid_prompt_set_returns_clean_plan_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prompt_path = Path(temp_dir) / "prompt-set.json"
            write_json(prompt_path, {"prompts": [{"prompt": "missing id"}]})
            status, summary, error = planner.plan_path(MIXTRAL_BUNDLE, prompt_set_path=prompt_path)

        self.assertEqual(status, 2)
        self.assertIsNone(error)
        assert summary is not None
        self.assertFalse(summary["valid"])
        self.assertTrue(any("missing prompt_id" in item for item in summary["errors"]))


if __name__ == "__main__":
    unittest.main()
