import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = ROOT / "scripts" / "plan_phase3_policy_candidate_trace.py"
SPEC = importlib.util.spec_from_file_location("plan_phase3_policy_candidate_trace", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TESTS_DIR))
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)

import test_phase3_positive_path as positive

MIXTRAL_BUNDLE = ROOT / "memory-moe-mvp" / "phase3-real-evidence" / "pc_mixtral_phase3_real_evidence_bundle.json"
POLICIES_FIXTURE = ROOT / "memory-moe-mvp" / "data" / "baseline_replay_policies.json"


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def candidate_prompt_set():
    return {
        "schema_version": "moe-phase3-policy-candidate-prompt-set-v1",
        "prompts": [
            {"prompt_id": "repeat-001-a", "prompt": "Explain expert caching in one paragraph."},
            {"prompt_id": "repeat-001-b", "prompt": "Explain expert caching in one paragraph."},
        ],
    }


def ready_trace_receipt(
    candidate_path: Path,
    prompt_path: Path,
    *,
    request_path: Path | None = None,
    overrides: dict | None = None,
) -> dict:
    source_request_path = request_path or prompt_path.with_name("phase3-bundle.runtime-capture-request.json")
    receipt = {
        "receipt_ready": True,
        "source_request_path": str(source_request_path),
        "source_prompt_set_path": str(prompt_path),
        "candidate_trace_path": str(candidate_path),
        "router_trace_capture_approved": True,
        "runtime_prompt_traffic_approved": True,
        "captured_at": "2026-06-28T21:00:00Z",
        "capture_host": "policy-trace-test",
        "runtime_backend": "llama_cpp",
        "model_id": positive.MODEL_ID,
        "prompt_family": "phase3-policy-test",
        "operator_notes": "test trace receipt",
    }
    if overrides:
        receipt.update(overrides)
    return receipt


class Phase3PolicyCandidateTracePlannerTests(unittest.TestCase):
    def test_default_mixtral_plan_names_policy_candidate_blocker(self) -> None:
        summary = planner.build_capture_plan(MIXTRAL_BUNDLE)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertFalse(summary["current_replay"]["policy_candidate_ready"])
        self.assertEqual(summary["current_replay"]["reuse_distance_observations"], 0)
        self.assertFalse(summary["policy_candidate_ready_after_plan"])
        blocker_ids = {item["id"] for item in summary["current_replay"]["blockers"]}
        self.assertIn("no_replay_policy_candidate", blocker_ids)
        self.assertIn("no_reuse_distance_observations", blocker_ids)
        by_request = {item["id"]: item for item in summary["artifact_requests"]}
        self.assertEqual(by_request["candidate_prompt_set_artifact"]["status"], "already_satisfied")
        self.assertEqual(by_request["candidate_trace_artifact"]["status"], "present_but_not_candidate_ready")
        self.assertEqual(by_request["candidate_policy_replay"]["status"], "no_candidate_found")
        self.assertEqual(by_request["phase3_bundle_with_candidate_trace"]["status"], "blocked_by_candidate_replay")
        command_classes = {item["command_class"] for item in summary["commands"]}
        self.assertIn("candidate_trace_receipt_validator", command_classes)
        self.assertIn("candidate_trace_contract_validator", command_classes)
        self.assertIn("candidate_policy_replay", command_classes)
        self.assertIn("phase3_bundle_builder_with_candidate_trace", command_classes)

    def test_prompt_set_and_approval_make_candidate_trace_needed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            prompt_path = temp_path / "candidate-prompt-set.json"
            trace_path = temp_path / "candidate-router-events.jsonl"
            receipt_path = temp_path / "candidate-router-events.capture-receipt.json"
            write_json(prompt_path, candidate_prompt_set())
            summary = planner.build_capture_plan(
                MIXTRAL_BUNDLE,
                candidate_prompt_set_path=prompt_path,
                candidate_trace_path=trace_path,
                candidate_trace_receipt_path=receipt_path,
                policy_candidate_trace_capture_approved=True,
                runtime_prompt_traffic_approved=True,
            )

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertTrue(summary["prompt_set"]["ready"])
        by_request = {item["id"]: item for item in summary["artifact_requests"]}
        self.assertEqual(by_request["candidate_prompt_set_artifact"]["status"], "already_satisfied")
        self.assertEqual(by_request["candidate_trace_artifact"]["status"], "needed")
        self.assertFalse(summary["runtime_capture_contract"]["requires_explicit_approval"])

    def test_explicit_missing_candidate_trace_is_valid_planned_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            prompt_path = temp_path / "candidate-prompt-set.json"
            trace_path = temp_path / "planned-candidate-router-events.jsonl"
            receipt_path = temp_path / "planned-candidate-router-events.capture-receipt.json"
            write_json(prompt_path, candidate_prompt_set())

            summary = planner.build_capture_plan(
                MIXTRAL_BUNDLE,
                candidate_prompt_set_path=prompt_path,
                candidate_trace_path=trace_path,
                candidate_trace_receipt_path=receipt_path,
                policy_candidate_trace_capture_approved=True,
                runtime_prompt_traffic_approved=True,
            )

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["candidate_trace_path"], str(trace_path))
        self.assertEqual(summary["candidate_trace_receipt_path"], str(receipt_path))
        self.assertFalse(summary["candidate_trace"]["exists"])
        self.assertTrue(summary["candidate_trace"]["valid"])
        self.assertEqual(summary["candidate_trace"]["errors"], [])
        self.assertFalse(summary["candidate_trace"]["policy_candidate_ready"])
        self.assertFalse(summary["runtime_capture_contract"]["requires_explicit_approval"])
        by_request = {item["id"]: item for item in summary["artifact_requests"]}
        self.assertEqual(by_request["candidate_trace_artifact"]["status"], "needed")
        self.assertEqual(by_request["candidate_policy_replay"]["status"], "blocked_by_candidate_trace")

    def test_invalid_prompt_set_returns_clean_plan_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prompt_path = Path(temp_dir) / "candidate-prompt-set.json"
            write_json(prompt_path, {"prompts": [{"prompt": "missing id"}]})
            status, summary, error = planner.plan_path(MIXTRAL_BUNDLE, candidate_prompt_set_path=prompt_path)

        self.assertEqual(status, 2)
        self.assertIsNone(error)
        assert summary is not None
        self.assertFalse(summary["valid"])
        self.assertTrue(any("missing prompt_id" in item for item in summary["errors"]))


    def test_candidate_trace_replay_without_receipt_does_not_promote(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            trace_path = temp_path / "candidate-router-events.jsonl"
            inventory_path = temp_path / "expert-inventory.json"
            prompt_path = temp_path / "candidate-prompt-set.json"
            receipt_path = temp_path / "candidate-router-events.capture-receipt.json"
            write_jsonl(trace_path, positive.local_trace_events())
            write_json(inventory_path, positive.local_inventory())
            write_json(prompt_path, candidate_prompt_set())

            summary = planner.summarize_candidate_trace(
                trace_path,
                inventory_path=inventory_path,
                policies_path=POLICIES_FIXTURE,
                trace_receipt_path=receipt_path,
                expected_prompt_set_path=prompt_path,
                expected_model_id=positive.MODEL_ID,
                expected_backend_family="llama_cpp",
                expected_prompt_family="phase3-policy-test",
            )

        self.assertTrue(summary["replay_valid"], summary["errors"])
        self.assertFalse(summary["valid"])
        self.assertFalse(summary["policy_candidate_ready"])
        self.assertEqual(summary["blocker"], "candidate_trace_capture_receipt_not_ready")
        blocker_ids = {item["id"] for item in summary["blockers"]}
        self.assertIn("candidate_trace_capture_receipt_missing", blocker_ids)

    def test_candidate_trace_ready_receipt_allows_policy_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            trace_path = temp_path / "candidate-router-events.jsonl"
            inventory_path = temp_path / "expert-inventory.json"
            prompt_path = temp_path / "candidate-prompt-set.json"
            receipt_path = temp_path / "candidate-router-events.capture-receipt.json"
            write_jsonl(trace_path, positive.local_trace_events())
            write_json(inventory_path, positive.local_inventory())
            write_json(prompt_path, candidate_prompt_set())
            write_json(temp_path / "phase3-bundle.runtime-capture-request.json", {"schema_version": "test-runtime-capture-request"})
            write_json(receipt_path, ready_trace_receipt(trace_path, prompt_path))

            summary = planner.summarize_candidate_trace(
                trace_path,
                inventory_path=inventory_path,
                policies_path=POLICIES_FIXTURE,
                trace_receipt_path=receipt_path,
                expected_prompt_set_path=prompt_path,
                expected_model_id=positive.MODEL_ID,
                expected_backend_family="llama_cpp",
                expected_prompt_family="phase3-policy-test",
            )

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertTrue(summary["policy_candidate_ready"])
        self.assertTrue(summary["capture_receipt"]["ready"])
        self.assertIn("preload_shortlist", summary["candidate_policy_ids"])

    def test_candidate_trace_receipt_prompt_mismatch_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            trace_path = temp_path / "candidate-router-events.jsonl"
            inventory_path = temp_path / "expert-inventory.json"
            prompt_path = temp_path / "candidate-prompt-set.json"
            receipt_path = temp_path / "candidate-router-events.capture-receipt.json"
            write_jsonl(trace_path, positive.local_trace_events())
            write_json(inventory_path, positive.local_inventory())
            write_json(prompt_path, candidate_prompt_set())
            write_json(temp_path / "phase3-bundle.runtime-capture-request.json", {"schema_version": "test-runtime-capture-request"})
            write_json(
                receipt_path,
                ready_trace_receipt(
                    trace_path,
                    prompt_path,
                    overrides={"source_prompt_set_path": str(temp_path / "other.prompt-set.json")},
                ),
            )

            summary = planner.summarize_candidate_trace(
                trace_path,
                inventory_path=inventory_path,
                policies_path=POLICIES_FIXTURE,
                trace_receipt_path=receipt_path,
                expected_prompt_set_path=prompt_path,
                expected_model_id=positive.MODEL_ID,
                expected_backend_family="llama_cpp",
                expected_prompt_family="phase3-policy-test",
            )

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["policy_candidate_ready"])
        self.assertTrue(any("source_prompt_set_path" in error for error in summary["errors"]))
    def test_markdown_report_names_trace_receipt_contract_and_commands(self) -> None:
        summary = planner.build_capture_plan(MIXTRAL_BUNDLE)

        report = planner.format_markdown_report(summary)

        self.assertIn("# Phase 3 Policy-Candidate Trace", report)
        self.assertIn("## Candidate Trace Contract", report)
        self.assertIn("## Runtime Capture Contract", report)
        self.assertIn("candidate_trace_artifact", report)
        self.assertIn("candidate_trace_receipt_validator", report)
        self.assertIn("Trace capture receipt path", report)
        self.assertIn("no_replay_policy_candidate", report)
        self.assertIn("planner does not send prompt traffic", report)

    def test_write_markdown_report_writes_policy_candidate_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "policy-candidate-handoff.md"
            summary = planner.build_capture_plan(MIXTRAL_BUNDLE)

            planner.write_markdown_report(summary, output_path)

            report = output_path.read_text(encoding="utf-8")

        self.assertTrue(report.startswith("# Phase 3 Policy-Candidate Trace"))
        self.assertIn("## Commands", report)
        self.assertIn("phase3_bundle_builder_with_candidate_trace", report)

if __name__ == "__main__":
    unittest.main()
