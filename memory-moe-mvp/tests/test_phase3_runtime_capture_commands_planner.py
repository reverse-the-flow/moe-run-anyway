import io
import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_phase3_runtime_capture_commands.py"
SPEC = importlib.util.spec_from_file_location("plan_phase3_runtime_capture_commands", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)

REQUEST = ROOT / "memory-moe-mvp" / "phase3-real-evidence" / "pc_mixtral_phase3_real_evidence_bundle.runtime-capture-request.json"
MODEL_PLANE_MANIFEST = ROOT / "memory-moe-mvp" / "data" / "model_plane_moe_probe_manifest.runtime_baseline.fixture.json"


class Phase3RuntimeCaptureCommandsPlannerTests(unittest.TestCase):
    def build_filled_launch_card(self) -> dict[str, object]:
        summary = planner.build_summary(REQUEST, model_plane_manifest_path=MODEL_PLANE_MANIFEST)
        launch_card = planner.build_launch_card_template(summary)
        for task in launch_card["tasks"]:
            binding = dict(task["binding_template"])
            binding["binding_kind"] = "model_plane_callable"
            binding["model_plane_callable_id"] = f"callable-{task['artifact_id']}"
            binding["command_ready"] = True
            task["binding"] = binding
        return launch_card
    def test_default_mixtral_request_has_planned_contract_without_executable_commands(self) -> None:
        summary = planner.build_summary(REQUEST)

        self.assertTrue(summary["valid"], json.dumps(summary["errors"], indent=2))
        self.assertTrue(summary["command_contract_ready"])
        self.assertFalse(summary["runtime_capture_command_ready"])
        self.assertEqual(summary["request_name"], "PC Mixtral Phase 3 real evidence")
        self.assertEqual(summary["backend_family"], "llama_cpp")
        self.assertEqual(summary["planned_capture_count"], 3)
        self.assertEqual(summary["runtime_command_option_count"], 0)
        self.assertEqual(summary["missing_runtime_command_count"], 3)
        self.assertEqual(
            summary["missing_runtime_command_artifact_ids"],
            ["candidate_router_trace", "managed_output_summary_fill", "dense_output_summary_fill"],
        )
        by_artifact = {task["artifact_id"]: task for task in summary["capture_tasks"]}
        candidate = by_artifact["candidate_router_trace"]
        self.assertEqual(candidate["capture_kind"], "llama_cpp_router_trace_jsonl")
        self.assertTrue(candidate["artifact_path"].endswith("candidate-router-events.jsonl"))
        self.assertTrue(candidate["receipt_path"].endswith("candidate-router-events.capture-receipt.json"))
        self.assertEqual(candidate["approval_keys"], ["runtime_prompt_traffic_approved", "router_trace_capture_approved"])
        self.assertEqual(candidate["validator_command_count"], 3)
        binding = candidate["planned_command_binding"]
        self.assertEqual(binding["command_class"], "phase3_model_plane_launch_card_runtime_capture")
        self.assertTrue(binding["planned_only"])
        self.assertTrue(binding["requires_runtime"])
        self.assertTrue(binding["requires_prompt_traffic"])
        self.assertTrue(binding["requires_explicit_user_approval"])
        self.assertFalse(binding["metadata_only"])
        self.assertFalse(binding["command_ready"])
        self.assertEqual(binding["command"], [])
        self.assertIn("--artifact-id", binding["command_shape"])
        self.assertEqual(by_artifact["managed_output_summary_fill"]["capture_kind"], "managed_output_summary_json")
        self.assertEqual(by_artifact["dense_output_summary_fill"]["capture_kind"], "dense_output_summary_json")
        launch_summary = summary["launch_card_template_summary"]
        self.assertTrue(launch_summary["template_ready"])
        self.assertFalse(launch_summary["model_plane_binding_ready"])
        self.assertFalse(launch_summary["runtime_capture_command_ready"])
        self.assertEqual(launch_summary["task_count"], 3)
        self.assertEqual(launch_summary["missing_runtime_command_count"], 3)
        binding_summary = summary["launch_card_binding_summary"]
        self.assertTrue(binding_summary["valid"])
        self.assertFalse(binding_summary["binding_ready"])
        self.assertEqual(binding_summary["blockers"], ["launch_card_not_provided"])
        self.assertEqual(binding_summary["task_bindings"], [])

    def test_model_plane_manifest_is_recorded_as_planning_context_not_executable_capture(self) -> None:
        summary = planner.build_summary(REQUEST, model_plane_manifest_path=MODEL_PLANE_MANIFEST)

        model_plane = summary["model_plane_manifest"]
        self.assertTrue(model_plane["provided"])
        self.assertTrue(model_plane["valid"])
        self.assertTrue(model_plane["ready_for_binding"])
        self.assertEqual(model_plane["target_class"], "stock_llama_cpp_openai_compatible")
        self.assertEqual(model_plane["planned_stage"], "harness_run_request")
        self.assertEqual(model_plane["planned_status"], "planned_only")
        self.assertEqual(model_plane["safe_command_count"], 2)
        self.assertEqual(model_plane["base_url"], "http://127.0.0.1:18080")
        self.assertEqual(model_plane["model_path"], "/models/fixture-mixtral.gguf")
        self.assertFalse(summary["runtime_capture_command_ready"])
        self.assertEqual(summary["runtime_command_option_count"], 0)
        launch_card = planner.build_launch_card_template(summary)
        self.assertTrue(launch_card["template_ready"])
        self.assertTrue(launch_card["model_plane_binding_ready"])
        self.assertFalse(launch_card["runtime_capture_command_ready"])
        self.assertFalse(launch_card["executable"])
        self.assertEqual(launch_card["schema_version"], "moe-phase3-runtime-capture-launch-card-v1")
        self.assertEqual(launch_card["model_plane_profile"]["profile_id"], "fixture-llama-cpp-runtime-baseline")
        self.assertEqual(launch_card["model_plane_profile"]["base_url"], "http://127.0.0.1:18080")
        self.assertEqual(launch_card["task_count"], 3)
        self.assertIn("binding_template", launch_card["tasks"][0])
        candidate_task = {task["artifact_id"]: task for task in launch_card["tasks"]}["candidate_router_trace"]
        self.assertEqual(candidate_task["task_id"], "phase3_capture_candidate_router_trace")
        self.assertEqual(candidate_task["capture_kind"], "llama_cpp_router_trace_jsonl")
        self.assertFalse(candidate_task["command_binding_ready"])
        self.assertIn("--artifact-id", candidate_task["command_placeholder"])
        self.assertEqual(candidate_task["binding_template"]["binding_kind"], "model_plane_callable_or_launch_command")
        self.assertEqual(candidate_task["binding_template"]["writes_artifact_path"], candidate_task["artifact_output_path"])
        self.assertEqual(candidate_task["binding_template"]["writes_receipt_path"], candidate_task["receipt_output_path"])
        self.assertEqual(candidate_task["binding_template"]["runtime_capture_request_path"], launch_card["request_path"])
        self.assertEqual(candidate_task["binding_template"]["reads_prompt_set_path"], candidate_task["prompt_set_path"])
        self.assertTrue(candidate_task["binding_template"]["requires_explicit_user_approval"])
        self.assertTrue(candidate_task["binding_template"]["may_send_prompt_traffic"])
        self.assertIn("explicit_user_approval_for_runtime_prompt_traffic", launch_card["preflight_gates"])
        self.assertIn("runtime_capture_commands_unbound", launch_card["blockers"])

    def test_unfilled_launch_card_validates_as_unbound_not_invalid(self) -> None:
        summary = planner.build_summary(REQUEST, model_plane_manifest_path=MODEL_PLANE_MANIFEST)
        launch_card = planner.build_launch_card_template(summary)

        binding_summary = planner.validate_launch_card_bindings(summary, launch_card)

        self.assertTrue(binding_summary["valid"], json.dumps(binding_summary["errors"], indent=2))
        self.assertFalse(binding_summary["binding_ready"])
        self.assertEqual(binding_summary["missing_runtime_command_count"], 3)
        self.assertEqual(
            binding_summary["missing_runtime_command_artifact_ids"],
            ["candidate_router_trace", "managed_output_summary_fill", "dense_output_summary_fill"],
        )
        self.assertIn("runtime_command_binding_missing", binding_summary["blockers"])
        self.assertEqual(binding_summary["errors"], [])

    def test_filled_launch_card_promotes_runtime_command_readiness_without_execution(self) -> None:
        launch_card = self.build_filled_launch_card()
        with tempfile.TemporaryDirectory() as tmp:
            launch_card_path = Path(tmp) / "filled-launch-card.json"
            launch_card_path.write_text(json.dumps(launch_card), encoding="utf-8")
            summary = planner.build_summary(
                REQUEST,
                model_plane_manifest_path=MODEL_PLANE_MANIFEST,
                launch_card_path=launch_card_path,
            )

        self.assertTrue(summary["valid"], json.dumps(summary["errors"], indent=2))
        self.assertTrue(summary["runtime_capture_command_ready"])
        self.assertEqual(summary["runtime_command_option_count"], 3)
        self.assertEqual(summary["missing_runtime_command_count"], 0)
        self.assertEqual(summary["missing_runtime_command_artifact_ids"], [])
        binding_summary = summary["launch_card_binding_summary"]
        self.assertTrue(binding_summary["valid"], json.dumps(binding_summary["errors"], indent=2))
        self.assertTrue(binding_summary["binding_ready"])
        self.assertEqual(binding_summary["bound_task_count"], 3)
        self.assertEqual(binding_summary["command_option_count"], 3)
        self.assertEqual(binding_summary["command_ready_count"], 3)
        launch_summary = summary["launch_card_template_summary"]
        self.assertTrue(launch_summary["runtime_capture_command_ready"])
        self.assertEqual(launch_summary["missing_runtime_command_count"], 0)
        by_artifact = {task["artifact_id"]: task for task in summary["capture_tasks"]}
        candidate = by_artifact["candidate_router_trace"]
        self.assertTrue(candidate["command_binding_ready"])
        self.assertEqual(candidate["planned_command_binding"]["model_plane_callable_id"], "callable-candidate_router_trace")
        self.assertIsNone(candidate["planned_command_binding"]["missing_binding"])

    def test_filled_launch_card_requires_command_ready_flag(self) -> None:
        launch_card = self.build_filled_launch_card()
        for task in launch_card["tasks"]:
            task["binding"]["command_ready"] = False
        with tempfile.TemporaryDirectory() as tmp:
            launch_card_path = Path(tmp) / "callable-without-ready-launch-card.json"
            launch_card_path.write_text(json.dumps(launch_card), encoding="utf-8")
            summary = planner.build_summary(
                REQUEST,
                model_plane_manifest_path=MODEL_PLANE_MANIFEST,
                launch_card_path=launch_card_path,
            )

        self.assertTrue(summary["valid"], json.dumps(summary["errors"], indent=2))
        self.assertFalse(summary["runtime_capture_command_ready"])
        binding_summary = summary["launch_card_binding_summary"]
        self.assertFalse(binding_summary["binding_ready"])
        self.assertEqual(binding_summary["command_option_count"], 3)
        self.assertEqual(binding_summary["command_ready_count"], 0)
        self.assertEqual(binding_summary["missing_runtime_command_count"], 3)
        self.assertIn("runtime_command_ready_flag_missing", binding_summary["blockers"])

    def test_filled_launch_card_requires_prompt_traffic_ack_and_request_path(self) -> None:
        launch_card = self.build_filled_launch_card()
        launch_card["tasks"][0]["binding"]["may_send_prompt_traffic"] = False
        launch_card["tasks"][1]["binding"]["runtime_capture_request_path"] = "memory-moe-mvp/phase3-real-evidence/wrong.runtime-capture-request.json"
        launch_card["tasks"][2]["binding"]["reads_prompt_set_path"] = "memory-moe-mvp/phase3-real-evidence/wrong.prompt-set.json"
        with tempfile.TemporaryDirectory() as tmp:
            launch_card_path = Path(tmp) / "unsafe-filled-launch-card.json"
            launch_card_path.write_text(json.dumps(launch_card), encoding="utf-8")
            summary = planner.build_summary(
                REQUEST,
                model_plane_manifest_path=MODEL_PLANE_MANIFEST,
                launch_card_path=launch_card_path,
            )

        self.assertTrue(summary["valid"], json.dumps(summary["errors"], indent=2))
        self.assertFalse(summary["runtime_capture_command_ready"])
        binding_summary = summary["launch_card_binding_summary"]
        self.assertFalse(binding_summary["binding_ready"])
        self.assertEqual(binding_summary["command_option_count"], 3)
        self.assertIn("prompt_traffic_ack_missing", binding_summary["blockers"])
        self.assertIn("runtime_capture_request_path_missing", binding_summary["blockers"])
        self.assertIn("reads_prompt_set_path_missing", binding_summary["blockers"])
        by_artifact = {item["artifact_id"]: item for item in binding_summary["task_bindings"]}
        self.assertFalse(by_artifact["candidate_router_trace"]["binding_ready"])
        self.assertFalse(by_artifact["candidate_router_trace"]["may_send_prompt_traffic"])
        self.assertFalse(by_artifact["managed_output_summary_fill"]["binding_ready"])
        self.assertFalse(by_artifact["dense_output_summary_fill"]["binding_ready"])
        self.assertEqual(
            by_artifact["managed_output_summary_fill"]["runtime_capture_request_path"],
            "memory-moe-mvp/phase3-real-evidence/wrong.runtime-capture-request.json",
        )
        self.assertEqual(
            by_artifact["dense_output_summary_fill"]["reads_prompt_set_path"],
            "memory-moe-mvp/phase3-real-evidence/wrong.prompt-set.json",
        )

    def test_launch_card_input_accepts_utf8_bom(self) -> None:
        summary = planner.build_summary(REQUEST, model_plane_manifest_path=MODEL_PLANE_MANIFEST)
        launch_card = planner.build_launch_card_template(summary)
        with tempfile.TemporaryDirectory() as tmp:
            launch_card_path = Path(tmp) / "launch-card-with-bom.json"
            launch_card_path.write_text(json.dumps(launch_card), encoding="utf-8-sig")
            with_card = planner.build_summary(
                REQUEST,
                model_plane_manifest_path=MODEL_PLANE_MANIFEST,
                launch_card_path=launch_card_path,
            )

        binding_summary = with_card["launch_card_binding_summary"]
        self.assertTrue(binding_summary["valid"], json.dumps(binding_summary["errors"], indent=2))
        self.assertFalse(binding_summary["binding_ready"])
        self.assertEqual(binding_summary["missing_runtime_command_count"], 3)
    def test_markdown_and_human_summary_report_missing_runtime_commands(self) -> None:
        summary = planner.build_summary(REQUEST)
        report = planner.format_markdown_report(summary)

        self.assertIn("# Phase 3 Runtime-Capture Command Contract", report)
        self.assertIn("Contract ready: `True`", report)
        self.assertIn("Runtime capture commands ready: `False`", report)
        self.assertIn("Missing runtime commands: `3`", report)
        self.assertIn("| candidate_router_trace | llama_cpp_router_trace_jsonl |", report)
        self.assertIn("## Launch-Card Template", report)
        self.assertIn("Template ready: `True`", report)
        self.assertIn("Model Plane binding ready: `False`", report)
        self.assertIn("Runtime capture commands ready: `False`", report)
        self.assertIn("## Launch-Card Binding Intake", report)
        self.assertIn("Binding ready: `False`", report)
        self.assertIn("Blockers: `launch_card_not_provided`", report)
        self.assertIn("Model Plane profile or launch card selected for the target host.", report)

        output = io.StringIO()
        with redirect_stdout(output):
            planner.print_human_summary(summary)
        human = output.getvalue()
        self.assertIn("Contract ready: True", human)
        self.assertIn("Runtime capture commands ready: False", human)
        self.assertIn("Missing runtime commands: 3", human)
        self.assertIn("Launch-card binding intake: ready=False valid=True bound=0 command_options=0 missing=0", human)

    def test_output_md_writes_report(self) -> None:
        summary = planner.build_summary(REQUEST)
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "runtime-command-contract.md"
            launch_card_path = Path(tmp) / "runtime-launch-card.json"
            planner.write_markdown_report(summary, output_path)
            planner.write_launch_card_template(summary, launch_card_path)
            report = output_path.read_text(encoding="utf-8")
            launch_card = json.loads(launch_card_path.read_text(encoding="utf-8"))
        self.assertIn("Phase 3 Runtime-Capture Command Contract", report)
        self.assertEqual(launch_card["schema_version"], "moe-phase3-runtime-capture-launch-card-v1")
        self.assertEqual(launch_card["status"], "planned_only")
        self.assertFalse(launch_card["executable"])
        self.assertEqual(launch_card["task_count"], 3)
        self.assertIn("binding_template", launch_card["tasks"][0])


if __name__ == "__main__":
    unittest.main()