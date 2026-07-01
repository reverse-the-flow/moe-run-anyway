import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_phase3_reuse_evidence_capture.py"
FIXTURE_TRACE = ROOT / "memory-moe-mvp" / "data" / "llama_cpp_router_trace.fixture.jsonl"
SPEC = importlib.util.spec_from_file_location("plan_phase3_reuse_evidence_capture", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


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


class Phase3ReuseEvidenceCapturePlannerTests(unittest.TestCase):
    def test_repeated_route_trace_is_reuse_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            trace = temp_path / "candidate-router-events.jsonl"
            receipt = temp_path / "candidate-router-events.capture-receipt.json"
            request = temp_path / "bundle.runtime-capture-request.json"
            prompt = temp_path / "bundle.prompt-set.json"
            trace.write_text(trace_text_with_prompt_identity(repeats=2), encoding="utf-8")
            request.write_text("{}\n", encoding="utf-8")
            prompt.write_text("{}\n", encoding="utf-8")
            receipt.write_text(
                json.dumps(
                    {
                        "receipt_ready": True,
                        "source_request_path": str(request),
                        "source_prompt_set_path": str(prompt),
                        "candidate_trace_path": str(trace),
                        "router_trace_capture_approved": True,
                        "runtime_prompt_traffic_approved": True,
                        "captured_at": "2026-06-28T21:00:00Z",
                        "capture_host": "reuse-test-host",
                        "runtime_backend": "llama_cpp",
                        "model_id": "fixture-mixtral.gguf",
                        "prompt_family": "fixture-router-trace",
                    }
                ),
                encoding="utf-8",
            )

            summary = planner.trace_summary(
                trace,
                receipt_path=receipt,
                request_path=request,
                prompt_set_path=prompt,
            )

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertTrue(summary["receipt"]["ready"], summary["receipt"].get("errors"))
        self.assertTrue(summary["reuse_ready"], summary["blockers"])
        self.assertTrue(summary["prompt_identity_ready"], summary["prompt_identity"])
        self.assertGreaterEqual(summary["reuse_distance"]["observations"], 1)
        self.assertGreater(summary["route_repetition"]["repeated_route_key_count"], 0)
        self.assertNotIn("prompt_identity_metadata_missing", summary["blockers"])

    def test_repeated_route_trace_without_prompt_identity_is_not_reuse_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            trace = temp_path / "candidate-router-events.jsonl"
            receipt = temp_path / "candidate-router-events.capture-receipt.json"
            request = temp_path / "bundle.runtime-capture-request.json"
            prompt = temp_path / "bundle.prompt-set.json"
            fixture_text = FIXTURE_TRACE.read_text(encoding="utf-8")
            trace.write_text(fixture_text + fixture_text, encoding="utf-8")
            request.write_text("{}\n", encoding="utf-8")
            prompt.write_text("{}\n", encoding="utf-8")
            receipt.write_text(
                json.dumps(
                    {
                        "receipt_ready": True,
                        "source_request_path": str(request),
                        "source_prompt_set_path": str(prompt),
                        "candidate_trace_path": str(trace),
                        "router_trace_capture_approved": True,
                        "runtime_prompt_traffic_approved": True,
                        "captured_at": "2026-06-28T21:00:00Z",
                        "capture_host": "reuse-test-host",
                        "runtime_backend": "llama_cpp",
                        "model_id": "fixture-mixtral.gguf",
                        "prompt_family": "fixture-router-trace",
                    }
                ),
                encoding="utf-8",
            )

            summary = planner.trace_summary(
                trace,
                receipt_path=receipt,
                request_path=request,
                prompt_set_path=prompt,
            )

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertTrue(summary["receipt"]["ready"], summary["receipt"].get("errors"))
        self.assertGreaterEqual(summary["reuse_distance"]["observations"], 1)
        self.assertGreater(summary["route_repetition"]["repeated_route_key_count"], 0)
        self.assertFalse(summary["reuse_ready"])
        self.assertFalse(summary["prompt_identity_ready"])
        self.assertEqual(summary["classification"], "trace_missing_prompt_identity")
        self.assertIn("prompt_identity_metadata_missing", summary["blockers"])

    def test_missing_candidate_trace_is_actionable_not_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            summary = planner.trace_summary(
                temp_path / "missing.jsonl",
                receipt_path=temp_path / "missing.capture-receipt.json",
                request_path=temp_path / "request.json",
                prompt_set_path=temp_path / "prompt.json",
            )

        self.assertFalse(summary["exists"])
        self.assertFalse(summary["reuse_ready"])
        self.assertIn("candidate_trace_missing", summary["blockers"])

    def test_current_repo_traces_are_valid_but_not_reuse_ready(self) -> None:
        summary = planner.build_root_summary()

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["bundle_count"], 6)
        self.assertEqual(summary["candidate_trace_valid_count"], 6)
        self.assertEqual(summary["candidate_trace_receipt_ready_count"], 6)
        self.assertEqual(summary["reuse_ready_count"], 0)
        self.assertEqual(summary["no_reuse_distance_observation_count"], 6)
        self.assertEqual(summary["prompt_identity_ready_count"], 0)
        self.assertEqual(summary["prompt_identity_metadata_missing_count"], 6)
        self.assertEqual(summary["blocker_counts"]["no_repeated_route_keys"], 6)
        self.assertEqual(summary["blocker_counts"]["prompt_identity_metadata_missing"], 6)
        self.assertEqual(summary["recommended_next_capture"]["bundle_name"], "PC Mixtral Phase 3 real evidence")
        self.assertEqual(
            summary["recommended_next_capture"]["required_result"],
            "capture_candidate_router_trace_with_reuse_distance",
        )

    def test_markdown_report_names_reuse_blocker_and_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir) / "phase3-real-evidence"
            temp_root.mkdir()
            bundle = temp_root / "fixture_phase3_real_evidence_bundle.json"
            prompt = temp_root / "fixture_phase3_real_evidence_bundle.prompt-set.json"
            request = temp_root / "fixture_phase3_real_evidence_bundle.runtime-capture-request.json"
            candidate_dir = temp_root / "fixture_phase3_real_evidence_bundle-policy-candidate"
            candidate_dir.mkdir()
            trace = candidate_dir / "candidate-router-events.jsonl"
            receipt = candidate_dir / "candidate-router-events.capture-receipt.json"
            shutil.copyfile(FIXTURE_TRACE, trace)
            bundle.write_text(
                json.dumps(
                    {
                        "schema_version": "moe-phase3-real-evidence-bundle-v1",
                        "name": "Fixture Phase 3 real evidence",
                        "model_id": "fixture-mixtral.gguf",
                        "backend_family": "llama_cpp",
                        "prompt_family": "fixture-router-trace",
                        "artifact_paths": {"inventory_path": None},
                    }
                ),
                encoding="utf-8",
            )
            prompt.write_text("{}\n", encoding="utf-8")
            request.write_text("{}\n", encoding="utf-8")
            receipt.write_text(
                json.dumps(
                    {
                        "receipt_ready": True,
                        "source_request_path": str(request),
                        "source_prompt_set_path": str(prompt),
                        "candidate_trace_path": str(trace),
                        "router_trace_capture_approved": True,
                        "runtime_prompt_traffic_approved": True,
                        "captured_at": "2026-06-28T21:00:00Z",
                        "capture_host": "reuse-test-host",
                        "runtime_backend": "llama_cpp",
                        "model_id": "fixture-mixtral.gguf",
                        "prompt_family": "fixture-router-trace",
                    }
                ),
                encoding="utf-8",
            )
            summary = planner.build_root_summary(temp_root)
            report = planner.build_markdown_report(summary)

        self.assertIn("No-reuse-distance blockers: `1`", report)
        self.assertIn("Prompt-identity blockers: `1`", report)
        self.assertIn("Minimum reuse-distance observations: `1`", report)
        self.assertIn("Prompt identity required: `True`", report)
        self.assertIn("selected_experts", report)


if __name__ == "__main__":
    unittest.main()