import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_phase3_handoff_coverage.py"
SPEC = importlib.util.spec_from_file_location("plan_phase3_handoff_coverage", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)

import build_phase3_live_capability_proof_template
import build_phase3_output_summary
import build_phase3_prompt_set
import build_phase3_runtime_capture_request
import plan_phase3_runtime_capture_commands
import build_phase3_trace_receipt


BUNDLE = ROOT / "memory-moe-mvp" / "phase3-real-evidence" / "pc_mixtral_phase3_real_evidence_bundle.json"


class Phase3HandoffCoveragePlannerTests(unittest.TestCase):
    def test_current_repo_reports_mixtral_handoff_ready(self) -> None:
        summary = planner.build_coverage()

        self.assertTrue(summary["valid"], summary["errors"])
        by_name = {item["name"]: item for item in summary["bundles"]}
        mixtral = by_name["PC Mixtral Phase 3 real evidence"]
        self.assertTrue(mixtral["handoff_scaffold_ready"], mixtral["missing_artifact_ids"])
        self.assertEqual(mixtral["runtime_capture_request"]["future_artifact_ids"], ["live_capability_proof_fill"])
        self.assertTrue(mixtral["candidate_trace_receipt_template"]["exists"])
        self.assertTrue(mixtral["candidate_trace_receipt_template"]["valid"])
        self.assertTrue(mixtral["candidate_trace_receipt_template"]["receipt_ready"])
        launch_card = mixtral["runtime_capture_launch_card_template"]
        self.assertTrue(launch_card["exists"])
        self.assertTrue(launch_card["ready"], launch_card)
        self.assertTrue(launch_card["binding_valid"], launch_card)
        self.assertFalse(launch_card["binding_ready"])
        self.assertFalse(launch_card["runtime_capture_command_ready"])
        self.assertEqual(launch_card["task_count"], 3)

    def test_missing_artifacts_are_reported_with_build_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = root / BUNDLE.name
            bundle.write_text(BUNDLE.read_text(encoding="utf-8"), encoding="utf-8")

            summary = planner.build_coverage(root)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["handoff_scaffold_ready_count"], 0)
        self.assertEqual(summary["missing_artifact_counts"]["prompt_set"], 1)
        self.assertEqual(summary["missing_artifact_counts"]["runtime_capture_request"], 1)
        self.assertEqual(summary["missing_artifact_counts"]["runtime_capture_launch_card_template"], 1)
        self.assertEqual(summary["missing_artifact_counts"]["candidate_trace_receipt_template"], 1)
        self.assertIn("build_commands", summary["bundles"][0])

    def test_generated_scaffold_for_temp_bundle_becomes_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = root / BUNDLE.name
            bundle.write_text(BUNDLE.read_text(encoding="utf-8"), encoding="utf-8")
            prompt_path = planner.canonical_prompt_set_path(bundle)

            prompt_args = type(
                "Args",
                (),
                {
                    "bundle_path": bundle,
                    "purpose": "shared_phase3",
                    "repeat_count": 2,
                    "output": prompt_path,
                    "default_output": False,
                    "artifact_json": False,
                    "json": False,
                },
            )()
            build_phase3_prompt_set.plan_build(prompt_args)
            for label in ("managed", "dense"):
                output_args = type(
                    "Args",
                    (),
                    {
                        "prompt_set_path": prompt_path,
                        "output_label": label,
                        "input_summary": None,
                        "output": planner.output_summary_path(bundle, label),
                        "default_output": False,
                        "artifact_json": False,
                        "json": False,
                    },
                )()
                build_phase3_output_summary.plan_build(output_args)
            live_args = type(
                "Args",
                (),
                {
                    "bundle_path": bundle,
                    "proof_scope": None,
                    "output": planner.live_proof_template_path(bundle),
                    "default_output": False,
                    "artifact_json": False,
                    "json": False,
                },
            )()
            build_phase3_live_capability_proof_template.plan_build(live_args)
            receipt_args = type(
                "Args",
                (),
                {
                    "bundle_path": bundle,
                    "prompt_set_path": None,
                    "candidate_trace_path": None,
                    "input_receipt": None,
                    "output": planner.candidate_trace_receipt_path(bundle),
                    "default_output": False,
                    "artifact_json": False,
                    "json": False,
                },
            )()
            build_phase3_trace_receipt.plan_build(receipt_args)
            request_args = type(
                "Args",
                (),
                {
                    "bundle_path": bundle,
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
                    "output": planner.runtime_request_path(bundle),
                    "default_output": False,
                    "json": False,
                },
            )()
            build_phase3_runtime_capture_request.plan_build(request_args)
            command_summary = plan_phase3_runtime_capture_commands.build_summary(planner.runtime_request_path(bundle))
            plan_phase3_runtime_capture_commands.write_launch_card_template(
                command_summary,
                planner.runtime_launch_card_template_path(bundle),
            )

            summary = planner.build_coverage(root)

        self.assertTrue(summary["valid"], json.dumps(summary["errors"], indent=2))
        self.assertEqual(summary["handoff_scaffold_ready_count"], 1)
        self.assertEqual(summary["missing_artifact_counts"], {})
        runtime_launch_card = summary["bundles"][0]["runtime_capture_launch_card_template"]
        self.assertTrue(runtime_launch_card["ready"], runtime_launch_card)
        self.assertFalse(runtime_launch_card["binding_ready"])
        self.assertFalse(runtime_launch_card["runtime_capture_command_ready"])


    def test_receipt_template_with_wrong_model_blocks_handoff_scaffold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = root / BUNDLE.name
            bundle.write_text(BUNDLE.read_text(encoding="utf-8"), encoding="utf-8")
            prompt_path = planner.canonical_prompt_set_path(bundle)

            prompt_args = type(
                "Args",
                (),
                {
                    "bundle_path": bundle,
                    "purpose": "shared_phase3",
                    "repeat_count": 2,
                    "output": prompt_path,
                    "default_output": False,
                    "artifact_json": False,
                    "json": False,
                },
            )()
            build_phase3_prompt_set.plan_build(prompt_args)
            receipt_args = type(
                "Args",
                (),
                {
                    "bundle_path": bundle,
                    "prompt_set_path": None,
                    "candidate_trace_path": None,
                    "input_receipt": None,
                    "output": planner.candidate_trace_receipt_path(bundle),
                    "default_output": False,
                    "artifact_json": False,
                    "json": False,
                },
            )()
            build_phase3_trace_receipt.plan_build(receipt_args)
            receipt_path = planner.candidate_trace_receipt_path(bundle)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["model_id"] = "wrong-model"
            receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            summary = planner.build_coverage(root)

        bundle_summary = summary["bundles"][0]
        self.assertFalse(bundle_summary["handoff_scaffold_ready"])
        self.assertIn("candidate_trace_receipt_template", bundle_summary["missing_artifact_ids"])
        self.assertFalse(bundle_summary["candidate_trace_receipt_template"]["valid"])
        self.assertTrue(
            any("model_id" in error for error in bundle_summary["candidate_trace_receipt_template"]["errors"])
        )
    def test_markdown_report_summarizes_current_handoff_coverage(self) -> None:
        summary = planner.build_coverage()

        report = planner.format_markdown_report(summary)

        self.assertIn("# Phase 3 Handoff Coverage", report)
        self.assertIn("## Bundle Coverage", report)
        self.assertIn("Launch Card Template", report)
        self.assertIn("PC Mixtral Phase 3 real evidence", report)
        self.assertIn("Handoff scaffolds ready", report)
        self.assertIn("- none", report)
        self.assertIn("handoff coverage does not send prompt traffic", report)

    def test_markdown_report_includes_missing_scaffold_build_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = root / BUNDLE.name
            bundle.write_text(BUNDLE.read_text(encoding="utf-8"), encoding="utf-8")

            summary = planner.build_coverage(root)
            report = planner.format_markdown_report(summary)

        self.assertIn("## Missing Scaffold Build Commands", report)
        self.assertIn("prompt_set", report)
        self.assertIn("runtime_capture_request", report)
        self.assertIn("scripts/build_phase3_prompt_set.py", report)
        self.assertIn("scripts/build_phase3_runtime_capture_request.py", report)
        self.assertIn("scripts/plan_phase3_runtime_capture_commands.py", report)

    def test_write_markdown_report_writes_handoff_coverage_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "handoff-coverage.md"
            summary = planner.build_coverage()

            planner.write_markdown_report(summary, output_path)

            report = output_path.read_text(encoding="utf-8")

        self.assertTrue(report.startswith("# Phase 3 Handoff Coverage"))
        self.assertIn("## Safety Contract", report)
if __name__ == "__main__":
    unittest.main()
