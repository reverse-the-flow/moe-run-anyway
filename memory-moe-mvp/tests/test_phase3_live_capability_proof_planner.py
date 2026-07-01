import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_phase3_live_capability_proof.py"
SPEC = importlib.util.spec_from_file_location("plan_phase3_live_capability_proof", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ready_artifact(
    *,
    model_id: str = "fixture-mixtral.gguf",
    backend_family: str = "llama_cpp",
    prompt_family: str = "fixture",
    source_bundle_path: str = "memory-moe-mvp/data/phase3_real_evidence_bundle.fixture.json",
):
    return {
        "schema_version": "moe-phase3-live-capability-proof-v1",
        "name": "Fixture live capability proof",
        "model_id": model_id,
        "backend_family": backend_family,
        "prompt_family": prompt_family,
        "source_bundle_path": source_bundle_path,
        "proof_scope": "fixture-adapter-smoke",
        "residency_observation": {
            "status": "available",
            "before_state_captured": True,
            "after_state_captured": True,
            "evidence_fields": ["resident_expert_count", "resident_bytes", "layer_expert_keys"],
        },
        "residency_control": {
            "status": "available",
            "supported_actions": ["observe", "preload", "evict", "restore_dense"],
            "actuator_boundary": "fixture-adapter",
            "dry_run_only": False,
        },
        "cleanup_restore": {
            "status": "available",
            "restore_verified": True,
            "cleanup_actions": ["restore_dense", "clear_policy_state"],
            "failure_path_tested": True,
        },
        "artifact_export": {
            "status": "available",
            "artifact_paths": ["memory-moe-mvp/data/phase3_real_evidence_bundle.fixture.json"],
        },
        "safety_contract": ["Fixture validates metadata only."],
        "next_actions": ["Use a real backend proof before live mutation."],
    }


class Phase3LiveCapabilityProofPlannerTests(unittest.TestCase):
    def test_missing_artifact_is_valid_blocker(self) -> None:
        summary = planner.build_summary(None)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertFalse(summary["proof_available"])
        self.assertFalse(summary["proof_ready"])
        self.assertEqual(summary["blockers"][0]["id"], "live_capability_proof_artifact_missing")
        self.assertFalse(summary["phase_3_gate"]["ready_for_live_spike"])

    def test_ready_artifact_satisfies_observation_control_cleanup_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "live-proof.json"
            write_json(path, ready_artifact())

            summary = planner.build_summary(path)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertTrue(summary["proof_available"])
        self.assertTrue(summary["proof_ready"])
        self.assertEqual(summary["blockers"], [])
        self.assertTrue(summary["phase_3_gate"]["live_residency_observation_and_control_ready"])
        self.assertTrue(summary["phase_3_gate"]["cleanup_restore_proof_ready"])
        self.assertTrue(summary["phase_3_gate"]["ready_for_live_spike"])

    def test_expected_context_mismatch_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "live-proof.json"
            write_json(path, ready_artifact(prompt_family="wrong-family"))

            summary = planner.build_summary(
                path,
                expected_model_id="fixture-mixtral.gguf",
                expected_backend_family="llama_cpp",
                expected_prompt_family="fixture",
                expected_source_bundle_path=ROOT / "memory-moe-mvp" / "data" / "phase3_real_evidence_bundle.fixture.json",
            )

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["proof_ready"])
        self.assertIn("live_proof_context_mismatch", {item["id"] for item in summary["blockers"]})
        self.assertFalse(summary["context_binding"]["ready"])
        self.assertFalse(summary["context_binding"]["checks"]["prompt_family_matches"])

    def test_available_export_requires_existing_artifact_paths(self) -> None:
        artifact = ready_artifact()
        artifact["artifact_export"]["artifact_paths"] = [
            "memory-moe-mvp/phase3-real-evidence/missing-live-proof.json"
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing-export-live-proof.json"
            write_json(path, artifact)

            summary = planner.build_summary(path)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["proof_ready"])
        self.assertIn("artifact_export.artifact_paths[0] must point to an existing file", summary["errors"])
        blocker_ids = {item["id"] for item in summary["blockers"]}
        self.assertIn("artifact_export_paths_missing", blocker_ids)
        export_status = summary["section_status"]["artifact_export"]
        self.assertEqual(export_status["artifact_path_count"], 1)
        self.assertEqual(export_status["artifact_path_exists_count"], 0)
        self.assertEqual(
            export_status["missing_artifact_paths"],
            ["memory-moe-mvp/phase3-real-evidence/missing-live-proof.json"],
        )

    def test_missing_source_bundle_path_is_invalid_context_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing-source-bundle-live-proof.json"
            write_json(path, ready_artifact(source_bundle_path="memory-moe-mvp/data/missing-bundle.json"))

            summary = planner.build_summary(path)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["proof_ready"])
        self.assertIn("artifact.source_bundle_path must point to an existing file", summary["errors"])
        self.assertFalse(summary["context_binding"]["ready"])
        self.assertFalse(summary["context_binding"]["checks"]["source_bundle_path_exists"])
        self.assertIn("live_proof_context_mismatch", {item["id"] for item in summary["blockers"]})

    def test_dry_run_only_control_and_unverified_cleanup_are_not_ready(self) -> None:
        artifact = ready_artifact()
        artifact["residency_control"]["dry_run_only"] = True
        artifact["cleanup_restore"]["restore_verified"] = False
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "partial-live-proof.json"
            write_json(path, artifact)

            summary = planner.build_summary(path)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertFalse(summary["proof_ready"])
        blocker_ids = {item["id"] for item in summary["blockers"]}
        self.assertIn("residency_control_is_dry_run_only", blocker_ids)
        self.assertIn("cleanup_restore_not_verified", blocker_ids)

    def test_bad_json_returns_clean_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.json"
            path.write_text("{", encoding="utf-8")

            status, summary, error = planner.plan_path(path)

        self.assertEqual(status, 2)
        self.assertIsNone(summary)
        assert error is not None
        self.assertIn("Could not build Phase 3 live capability proof plan", error)


    def test_markdown_report_names_live_proof_requirements_and_blockers(self) -> None:
        summary = planner.build_summary(None)

        report = planner.format_markdown_report(summary)

        self.assertIn("# Phase 3 Live Capability Proof", report)
        self.assertIn("## Required Proof Sections", report)
        self.assertIn("residency_observation", report)
        self.assertIn("residency_control", report)
        self.assertIn("cleanup_restore", report)
        self.assertIn("live_capability_proof_artifact_missing", report)
        self.assertIn("planner does not mutate runtime residency", report)

    def test_write_markdown_report_writes_live_proof_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "live-proof.json"
            output_path = Path(temp_dir) / "live-proof-handoff.md"
            write_json(path, ready_artifact())
            summary = planner.build_summary(path)

            planner.write_markdown_report(summary, output_path)

            report = output_path.read_text(encoding="utf-8")

        self.assertTrue(report.startswith("# Phase 3 Live Capability Proof"))
        self.assertIn("## Section Status", report)
        self.assertIn("Ready for live spike: `True`", report)

    def test_plan_path_accepts_expected_context_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "live-proof.json"
            bundle_path = ROOT / "memory-moe-mvp" / "data" / "phase3_real_evidence_bundle.fixture.json"
            write_json(path, ready_artifact())

            status, summary, error = planner.plan_path(
                path,
                expected_model_id="fixture-mixtral.gguf",
                expected_backend_family="llama_cpp",
                expected_prompt_family="fixture",
                expected_source_bundle_path=bundle_path,
            )

        self.assertEqual(status, 0, error)
        assert summary is not None
        self.assertTrue(summary["valid"], summary["errors"])
        self.assertTrue(summary["context_binding"]["ready"])
        self.assertTrue(summary["proof_ready"])

if __name__ == "__main__":
    unittest.main()