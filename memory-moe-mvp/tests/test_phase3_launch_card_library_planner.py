import io
import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_phase3_launch_card_library.py"
SPEC = importlib.util.spec_from_file_location("plan_phase3_launch_card_library", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


def phase3_writer_fixture(
    artifact_id: str,
    capture_kind: str,
    receipt_kind: str,
    approval_keys: list[str],
    *,
    ready: bool,
) -> dict[str, object]:
    return {
        "artifact_id": artifact_id,
        "capture_kind": capture_kind,
        "receipt_kind": receipt_kind,
        "approval_keys": approval_keys,
        "accepts_runtime_capture_request_path": True,
        "accepts_prompt_set_path": True,
        "writes_explicit_artifact_path": True,
        "writes_explicit_receipt_path": True,
        "runtime_execution_ready": ready,
        "writes_runtime_artifacts": ready,
    }


def mixtral_phase3_writers(*, ready: bool) -> list[dict[str, object]]:
    return [
        phase3_writer_fixture(
            "candidate_router_trace",
            "llama_cpp_router_trace_jsonl",
            "trace_capture_receipt_json",
            ["runtime_prompt_traffic_approved", "router_trace_capture_approved"],
            ready=ready,
        ),
        phase3_writer_fixture(
            "managed_output_summary_fill",
            "managed_output_summary_json",
            "embedded_output_capture_receipt",
            ["runtime_prompt_traffic_approved", "managed_output_capture_approved"],
            ready=ready,
        ),
        phase3_writer_fixture(
            "dense_output_summary_fill",
            "dense_output_summary_json",
            "embedded_output_capture_receipt",
            ["runtime_prompt_traffic_approved", "dense_output_capture_approved"],
            ready=ready,
        ),
    ]


def mixtral_model_plane_card_with_phase3_writers(*, ready: bool, limitations: list[str] | None = None) -> dict[str, object]:
    return {
        "card_id": "llama-cpp-dolphin-mixtral-8x7b-sidecar",
        "title": "Dolphin Mixtral llama.cpp Sidecar",
        "model": "dolphin-mixtral-8x7b.gguf",
        "model_class": "mixtral_style",
        "profile_id": "dolphin-mixtral-8x7b-llama-sidecar",
        "backend_family": "llama_cpp",
        "card_type": "launch_plus_probe",
        "evidence_level": "stock_llama_cpp_observability_with_passive_sidecar",
        "probe_tier": "passive_external_plus_internal_runtime",
        "execution_mode": "runner",
        "launch_command": {"argv": ["docker", "run", "memory-moe-llama-sidecar:latest"]},
        "preflight_command": {"argv": ["python", "scripts/run_live_baseline.py", "--preflight-only"]},
        "smoke_command": {"argv": ["python", "scripts/run_live_baseline.py"]},
        "phase3_artifact_writers": mixtral_phase3_writers(ready=ready),
        "limitations": limitations or [],
    }




def model_plane_fulfillment_from_summary(summary: dict[str, object], *, command_ready: bool = True) -> dict[str, object]:
    tasks = []
    for index, task in enumerate(summary["binding_tasks"]):
        tasks.append(
            {
                "request_name": task["request_name"],
                "request_path": task["request_path"],
                "launch_card_path": task["launch_card_path"],
                "model_id": task["model_id"],
                "backend_family": task["backend_family"],
                "prompt_family": task["prompt_family"],
                "task_id": task["task_id"],
                "artifact_id": task["artifact_id"],
                "capture_kind": task["capture_kind"],
                "receipt_kind": task["receipt_kind"],
                "artifact_output_path": task["artifact_output_path"],
                "receipt_output_path": task["receipt_output_path"],
                "prompt_set_path": task["prompt_set_path"],
                "approval_keys": task["approval_keys"],
                "binding": {
                    "binding_kind": "model_plane_callable_or_launch_command",
                    "model_plane_callable_id": f"fixture-callable-{index:02d}",
                    "launch_command": [],
                    "command_ready": command_ready,
                    "requires_explicit_user_approval": True,
                    "may_send_prompt_traffic": True,
                    "runtime_capture_request_path": task["request_path"],
                    "reads_prompt_set_path": task["prompt_set_path"],
                    "approval_keys": task["approval_keys"],
                    "writes_artifact_path": task["artifact_output_path"],
                    "writes_receipt_path": task["receipt_output_path"],
                },
            }
        )
    return {
        "schema_version": planner.MODEL_PLANE_ARTIFACT_WRITER_FULFILLMENT_SCHEMA_VERSION,
        "request_schema_version": planner.MODEL_PLANE_ARTIFACT_WRITER_REQUEST_SCHEMA_VERSION,
        "mode": "phase3_model_plane_artifact_writer_fulfillment",
        "tasks": tasks,
    }
class Phase3LaunchCardLibraryPlannerTests(unittest.TestCase):
    def test_default_library_indexes_all_planned_cards_without_execution_readiness(self) -> None:
        summary = planner.build_library()

        self.assertTrue(summary["valid"], json.dumps(summary["errors"], indent=2))
        self.assertTrue(summary["library_ready"])
        self.assertFalse(summary["execution_ready"])
        self.assertEqual(summary["card_count"], 6)
        self.assertEqual(summary["template_ready_count"], 6)
        self.assertEqual(summary["model_plane_binding_ready_count"], 0)
        self.assertEqual(summary["binding_ready_count"], 0)
        self.assertEqual(summary["runtime_capture_command_ready_count"], 0)
        self.assertEqual(summary["task_count"], 18)
        self.assertEqual(summary["bound_task_count"], 0)
        self.assertEqual(summary["command_option_count"], 0)
        self.assertEqual(summary["missing_runtime_command_count"], 18)
        self.assertTrue(summary["binding_handoff_ready"])
        self.assertEqual(summary["binding_handoff_task_count"], 18)
        self.assertEqual(summary["binding_handoff_ready_count"], 18)
        self.assertEqual(summary["binding_handoff_missing_field_count"], 0)
        self.assertEqual(summary["unbound_task_count"], 18)
        self.assertTrue(summary["model_plane_artifact_writer_contract_request_ready"])
        self.assertEqual(summary["model_plane_artifact_writer_contract_request_task_count"], 18)
        self.assertTrue(summary["saved_handoff_artifacts_ready"])
        self.assertEqual(summary["saved_handoff_artifact_missing_count"], 0)
        self.assertEqual(summary["saved_handoff_artifact_drifted_count"], 0)
        saved_artifacts = {
            item["artifact_id"]: item
            for item in summary["saved_handoff_artifacts"]["artifacts"]
        }
        self.assertTrue(saved_artifacts["launch_card_binding_worksheet"]["ready"])
        self.assertTrue(saved_artifacts["model_plane_artifact_writer_contract_request"]["ready"])
        self.assertEqual(len(summary["binding_tasks"]), 18)
        self.assertIn("model_plane_binding_context_missing", summary["blockers"])
        self.assertIn("launch_card_runtime_command_bindings_missing", summary["blockers"])
        self.assertIn("runtime_capture_commands_unbound", summary["blockers"])
        mixtral = next(card for card in summary["cards"] if card["request_name"] == "PC Mixtral Phase 3 real evidence")
        self.assertTrue(mixtral["template_ready"])
        self.assertTrue(mixtral["planned_only"])
        self.assertFalse(mixtral["executable"])
        self.assertFalse(mixtral["model_plane_binding_ready"])
        self.assertFalse(mixtral["binding_ready"])
        self.assertFalse(mixtral["runtime_capture_command_ready"])
        self.assertEqual(mixtral["task_count"], 3)
        self.assertEqual(mixtral["missing_runtime_command_count"], 3)
        self.assertEqual(mixtral["binding_handoff_task_count"], 3)
        self.assertEqual(mixtral["binding_handoff_ready_count"], 3)
        self.assertEqual(mixtral["binding_handoff_missing_field_count"], 0)
        self.assertEqual(mixtral["unbound_task_count"], 3)
        self.assertEqual(
            mixtral["missing_runtime_command_artifact_ids"],
            ["candidate_router_trace", "managed_output_summary_fill", "dense_output_summary_fill"],
        )
        candidate_task = next(task for task in mixtral["binding_tasks"] if task["artifact_id"] == "candidate_router_trace")
        self.assertTrue(candidate_task["binding_handoff_ready"])
        self.assertFalse(candidate_task["command_option_ready"])
        self.assertEqual(candidate_task["missing_fields"], [])
        self.assertEqual(candidate_task["binding_template"]["writes_artifact_path"], candidate_task["artifact_output_path"])
        self.assertEqual(candidate_task["binding_template"]["writes_receipt_path"], candidate_task["receipt_output_path"])
        self.assertEqual(candidate_task["binding_template"]["runtime_capture_request_path"], candidate_task["request_path"])
        self.assertEqual(candidate_task["binding_template"]["reads_prompt_set_path"], candidate_task["prompt_set_path"])
        self.assertTrue(candidate_task["binding_template"]["may_send_prompt_traffic"])
        self.assertEqual(
            candidate_task["validation_command"][:6],
            ["uv", "run", "--managed-python", "--python", "3.13", "scripts/plan_phase3_runtime_capture_commands.py"],
        )

    def test_saved_handoff_artifact_audit_detects_stale_artifact(self) -> None:
        summary = planner.build_library()
        source_summary = planner.summary_for_saved_handoff_artifacts(summary)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worksheet_path = planner.default_binding_worksheet_path(root)
            contract_path = planner.default_model_plane_contract_request_path(root)
            worksheet_path.write_text(
                planner.canonical_json(planner.build_binding_worksheet(source_summary)),
                encoding="utf-8",
            )
            contract_path.write_text(json.dumps({"stale": True}), encoding="utf-8")

            audit = planner.build_saved_handoff_artifacts_summary(summary, root=root)

        self.assertFalse(audit["ready"])
        self.assertEqual(audit["missing_count"], 0)
        self.assertEqual(audit["drifted_count"], 1)
        self.assertIn("saved_handoff_artifact_drifted", audit["blockers"])
        by_id = {item["artifact_id"]: item for item in audit["artifacts"]}
        self.assertTrue(by_id["launch_card_binding_worksheet"]["ready"])
        self.assertFalse(by_id["model_plane_artifact_writer_contract_request"]["ready"])

    def test_markdown_and_human_summary_make_binding_gap_visible(self) -> None:
        summary = planner.build_library()
        report = planner.format_markdown_report(summary)

        self.assertIn("# Phase 3 Launch-Card Library", report)
        self.assertIn("Library ready: `True`", report)
        self.assertIn("Execution ready: `False`", report)
        self.assertIn("Cards: `6`", report)
        self.assertIn("Missing runtime commands: `18`", report)
        self.assertIn("Binding handoff ready: `True`", report)
        self.assertIn("Binding handoff tasks: `18`", report)
        self.assertIn("Unbound tasks: `18`", report)
        self.assertIn("Model Plane artifact-writer contract request ready: `True`", report)
        self.assertIn("Model Plane artifact-writer contract request tasks: `18`", report)
        self.assertIn("Saved handoff artifacts ready: `True`", report)
        self.assertIn("Saved handoff artifacts drifted: `0`", report)
        self.assertIn("`model_plane_binding_context_missing`", report)
        self.assertIn("`launch_card_runtime_bindings_unfilled`", report)
        self.assertIn("| PC Mixtral Phase 3 real evidence | True | False | False | False | 3 |", report)
        self.assertIn("## Binding Handoff Tasks", report)
        self.assertIn("| PC Mixtral Phase 3 real evidence | candidate_router_trace | llama_cpp_router_trace_jsonl | runtime_prompt_traffic_approved, router_trace_capture_approved | False |", report)

        output = io.StringIO()
        with redirect_stdout(output):
            planner.print_human_summary(summary)
        human = output.getvalue()
        self.assertIn("MoE Run Anyway Phase 3 launch-card library", human)
        self.assertIn("Library ready: True", human)
        self.assertIn("Execution ready: False", human)
        self.assertIn("Missing runtime commands: 18", human)
        self.assertIn("Binding handoff ready: True", human)
        self.assertIn("Binding handoff tasks: 18", human)
        self.assertIn("Unbound tasks: 18", human)
        self.assertIn("Model Plane artifact-writer contract request ready: True", human)
        self.assertIn("Model Plane artifact-writer contract request tasks: 18", human)
        self.assertIn("Saved handoff artifacts ready: True", human)
        self.assertIn("Saved handoff artifacts drifted: 0", human)
        self.assertIn("PC Mixtral Phase 3 real evidence: template=True binding=False runtime=False missing=3", human)

    def test_output_md_writes_report(self) -> None:
        summary = planner.build_library()
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "launch-card-library.md"
            planner.write_markdown_report(summary, output_path)
            report = output_path.read_text(encoding="utf-8")

        self.assertIn("Phase 3 Launch-Card Library", report)
        self.assertIn("launch-card library planner does not send prompt traffic", report)

    def test_model_plane_artifact_writer_contract_request_exports_exact_task_contracts(self) -> None:
        summary = planner.build_library()
        request = planner.build_model_plane_artifact_writer_contract_request(summary)

        self.assertEqual(request["schema_version"], planner.MODEL_PLANE_ARTIFACT_WRITER_REQUEST_SCHEMA_VERSION)
        self.assertTrue(request["valid"])
        self.assertTrue(request["request_ready"])
        self.assertFalse(request["execution_ready"])
        self.assertEqual(request["task_count"], 18)
        self.assertEqual(request["phase3_artifact_writer_contract_count"], 18)
        self.assertEqual(request["phase3_artifact_writer_ready_count"], 0)
        self.assertIn("artifact-writer contract request does not send prompt traffic", request["safety_contract"])

        mixtral_task = next(
            task
            for task in summary["binding_tasks"]
            if task["request_name"] == "PC Mixtral Phase 3 real evidence"
            and task["artifact_id"] == "candidate_router_trace"
        )
        mixtral_request = next(
            task
            for task in request["tasks"]
            if task["request_name"] == "PC Mixtral Phase 3 real evidence"
            and task["artifact_id"] == "candidate_router_trace"
        )
        writer = mixtral_request["required_phase3_artifact_writer"]
        self.assertTrue(planner.phase3_artifact_writer_contract_matches_task(writer, mixtral_task))
        self.assertFalse(writer["runtime_execution_ready"])
        self.assertFalse(writer["writes_runtime_artifacts"])
        self.assertEqual(mixtral_request["artifact_output_path"], mixtral_task["artifact_output_path"])
        self.assertEqual(mixtral_request["receipt_output_path"], mixtral_task["receipt_output_path"])
        self.assertEqual(
            mixtral_request["model_plane_card_catalog_hint"]["must_accept_runtime_capture_request_path"],
            mixtral_task["request_path"],
        )
        self.assertEqual(
            mixtral_request["runtime_binding_template"]["runtime_capture_request_path"],
            mixtral_task["request_path"],
        )
        self.assertEqual(
            mixtral_request["runtime_binding_template"]["reads_prompt_set_path"],
            mixtral_task["prompt_set_path"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "model-plane-artifact-writer-request.json"
            planner.write_model_plane_artifact_writer_contract_request(summary, output_path)
            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(written["schema_version"], planner.MODEL_PLANE_ARTIFACT_WRITER_REQUEST_SCHEMA_VERSION)
        self.assertEqual(written["task_count"], 18)
        self.assertFalse(written["execution_ready"])

    def test_model_plane_artifact_writer_fulfillment_builds_valid_command_ready_cards(self) -> None:
        summary = planner.build_library()
        fulfillment = model_plane_fulfillment_from_summary(summary)

        validation = planner.build_binding_worksheet_from_model_plane_fulfillment(summary, fulfillment)
        self.assertEqual(validation["schema_version"], planner.MODEL_PLANE_ARTIFACT_WRITER_FULFILLMENT_VALIDATION_SCHEMA_VERSION)
        self.assertTrue(validation["valid"], validation["errors"])
        self.assertTrue(validation["fulfillment_ready"], validation["blockers"])
        self.assertTrue(validation["binding_ready"], validation["blockers"])
        self.assertEqual(validation["expected_task_count"], 18)
        self.assertEqual(validation["fulfillment_task_count"], 18)
        self.assertEqual(validation["matched_task_count"], 18)
        self.assertEqual(validation["command_option_count"], 18)
        self.assertEqual(validation["command_ready_count"], 18)
        self.assertEqual(validation["missing_runtime_command_count"], 0)
        self.assertIn("artifact-writer fulfillment validation does not send prompt traffic", validation["safety_contract"])

        package = planner.build_filled_launch_card_package_from_model_plane_fulfillment(summary, fulfillment)
        self.assertTrue(package["valid"], package["errors"])
        self.assertTrue(package["binding_ready"], package["blockers"])
        self.assertEqual(package["binding_ready_card_count"], 6)
        self.assertEqual(package["bound_task_count"], 18)
        self.assertEqual(package["missing_runtime_command_count"], 0)
        self.assertTrue(package["model_plane_artifact_writer_fulfillment_validation"]["binding_ready"])

    def test_model_plane_artifact_writer_fulfillment_rejects_task_identity_drift(self) -> None:
        summary = planner.build_library()
        fulfillment = model_plane_fulfillment_from_summary(summary)
        fulfillment["tasks"][0]["artifact_output_path"] = "memory-moe-mvp/phase3-real-evidence/wrong.jsonl"
        fulfillment["tasks"][0]["binding"]["writes_artifact_path"] = "memory-moe-mvp/phase3-real-evidence/wrong.jsonl"

        validation = planner.build_binding_worksheet_from_model_plane_fulfillment(summary, fulfillment)

        self.assertFalse(validation["valid"])
        self.assertFalse(validation["fulfillment_ready"])
        self.assertFalse(validation["binding_ready"])
        self.assertIn("model_plane_fulfillment_invalid", validation["blockers"])
        self.assertIn("artifact_output_path_mismatch", "\n".join(validation["errors"]))
        package = planner.build_filled_launch_card_package_from_model_plane_fulfillment(summary, fulfillment)
        self.assertFalse(package["valid"])
        self.assertFalse(package["binding_ready"])
        self.assertIn("model_plane_fulfillment_invalid", package["blockers"])

    def test_model_plane_artifact_writer_fulfillment_cli_writes_filled_cards(self) -> None:
        summary = planner.build_library()
        fulfillment = model_plane_fulfillment_from_summary(summary)
        with tempfile.TemporaryDirectory() as tmp:
            fulfillment_path = Path(tmp) / "fulfillment.json"
            output_dir = Path(tmp) / "filled-cards"
            fulfillment_path.write_text(json.dumps(fulfillment), encoding="utf-8")
            old_argv = sys.argv
            try:
                sys.argv = [
                    "plan_phase3_launch_card_library.py",
                    "--model-plane-contract-fulfillment",
                    str(fulfillment_path),
                    "--output-filled-card-dir",
                    str(output_dir),
                    "--json",
                ]
                output = io.StringIO()
                with redirect_stdout(output):
                    status = planner.main()
            finally:
                sys.argv = old_argv
            payload = json.loads(output.getvalue())

        self.assertEqual(status, 0)
        fulfillment_validation = payload["model_plane_artifact_writer_fulfillment_validation"]
        self.assertTrue(fulfillment_validation["valid"], fulfillment_validation["errors"])
        self.assertTrue(fulfillment_validation["binding_ready"], fulfillment_validation["blockers"])
        self.assertEqual(payload["filled_launch_card_package_summary"]["missing_runtime_command_count"], 0)
        self.assertEqual(len(payload["filled_launch_card_outputs"]), 6)
    def test_binding_worksheet_template_round_trips_without_execution_readiness(self) -> None:
        summary = planner.build_library()
        worksheet = planner.build_binding_worksheet(summary)

        self.assertEqual(worksheet["schema_version"], planner.BINDING_WORKSHEET_SCHEMA_VERSION)
        self.assertTrue(worksheet["worksheet_ready"])
        self.assertFalse(worksheet["execution_ready"])
        self.assertEqual(worksheet["task_count"], 18)
        self.assertEqual(worksheet["command_option_count"], 0)
        self.assertEqual(worksheet["unbound_task_count"], 18)

        validation = planner.validate_binding_worksheet(summary, worksheet)
        self.assertTrue(validation["valid"], validation["errors"])
        self.assertTrue(validation["worksheet_ready"])
        self.assertFalse(validation["binding_ready"])
        self.assertEqual(validation["expected_task_count"], 18)
        self.assertEqual(validation["matched_task_count"], 18)
        self.assertEqual(validation["missing_runtime_command_count"], 18)
        self.assertIn("runtime_command_binding_missing", validation["blockers"])

        package = planner.build_filled_launch_card_package(summary, worksheet)
        self.assertTrue(package["valid"], package["errors"])
        self.assertFalse(package["binding_ready"])
        self.assertEqual(package["card_count"], 6)
        self.assertEqual(package["task_count"], 18)
        self.assertEqual(package["bound_task_count"], 0)
        self.assertEqual(package["missing_runtime_command_count"], 18)

    def test_filled_binding_worksheet_builds_valid_command_ready_cards(self) -> None:
        summary = planner.build_library()
        worksheet = planner.build_binding_worksheet(summary)
        for index, row in enumerate(worksheet["tasks"]):
            row["binding"]["model_plane_callable_id"] = f"fixture-callable-{index:02d}"
            row["binding"]["command_ready"] = True

        validation = planner.validate_binding_worksheet(summary, worksheet)
        self.assertTrue(validation["valid"], validation["errors"])
        self.assertTrue(validation["binding_ready"], validation["blockers"])
        self.assertEqual(validation["command_option_count"], 18)
        self.assertEqual(validation["command_ready_count"], 18)
        self.assertEqual(validation["missing_runtime_command_count"], 0)

        package = planner.build_filled_launch_card_package(summary, worksheet)
        self.assertTrue(package["valid"], package["errors"])
        self.assertTrue(package["binding_ready"], package["blockers"])
        self.assertEqual(package["binding_ready_card_count"], 6)
        self.assertEqual(package["bound_task_count"], 18)
        self.assertEqual(package["command_option_count"], 18)
        self.assertEqual(package["command_ready_count"], 18)
        self.assertEqual(package["missing_runtime_command_count"], 0)

        with tempfile.TemporaryDirectory() as tmp:
            outputs = planner.write_filled_launch_cards(package, Path(tmp))
            self.assertEqual(len(outputs), 6)
            mixtral_output = next(item for item in outputs if item["request_name"] == "PC Mixtral Phase 3 real evidence")
            request_path = planner.resolve_repo_path("memory-moe-mvp/phase3-real-evidence/pc_mixtral_phase3_real_evidence_bundle.runtime-capture-request.json")
            assert request_path is not None
            command_summary = planner.plan_phase3_runtime_capture_commands.build_summary(
                request_path,
                launch_card_path=Path(mixtral_output["output_path"]),
            )

        self.assertTrue(command_summary["runtime_capture_command_ready"], command_summary["errors"])
        binding_summary = command_summary["launch_card_binding_summary"]
        self.assertTrue(binding_summary["binding_ready"], binding_summary["blockers"])
        self.assertEqual(binding_summary["bound_task_count"], 3)
        self.assertEqual(binding_summary["command_option_count"], 3)
        self.assertEqual(binding_summary["command_ready_count"], 3)
        self.assertEqual(binding_summary["missing_runtime_command_count"], 0)

    def test_binding_worksheet_callable_id_without_ready_flag_stays_blocked(self) -> None:
        summary = planner.build_library()
        worksheet = planner.build_binding_worksheet(summary)
        for index, row in enumerate(worksheet["tasks"]):
            row["binding"]["model_plane_callable_id"] = f"fixture-callable-{index:02d}"

        validation = planner.validate_binding_worksheet(summary, worksheet)

        self.assertTrue(validation["valid"], validation["errors"])
        self.assertFalse(validation["binding_ready"])
        self.assertEqual(validation["command_option_count"], 18)
        self.assertEqual(validation["command_ready_count"], 0)
        self.assertEqual(validation["command_option_without_ready_count"], 18)
        self.assertEqual(validation["missing_runtime_command_count"], 18)
        self.assertIn("runtime_command_ready_flag_missing", validation["blockers"])

    def test_binding_worksheet_requires_prompt_traffic_ack_and_request_path(self) -> None:
        summary = planner.build_library()
        worksheet = planner.build_binding_worksheet(summary)
        for index, row in enumerate(worksheet["tasks"]):
            row["binding"]["model_plane_callable_id"] = f"fixture-callable-{index:02d}"
            row["binding"]["command_ready"] = True
        worksheet["tasks"][0]["binding"]["may_send_prompt_traffic"] = False
        worksheet["tasks"][1]["binding"]["runtime_capture_request_path"] = "memory-moe-mvp/phase3-real-evidence/wrong.runtime-capture-request.json"
        worksheet["tasks"][2]["binding"]["reads_prompt_set_path"] = "memory-moe-mvp/phase3-real-evidence/wrong.prompt-set.json"

        validation = planner.validate_binding_worksheet(summary, worksheet)

        self.assertTrue(validation["valid"], validation["errors"])
        self.assertFalse(validation["binding_ready"])
        self.assertEqual(validation["command_option_count"], 18)
        self.assertIn("prompt_traffic_ack_missing", validation["blockers"])
        self.assertIn("runtime_capture_request_path_missing", validation["blockers"])
        self.assertIn("reads_prompt_set_path_missing", validation["blockers"])
        by_key = {item["key"]: item for item in validation["task_bindings"]}
        first_key = planner.binding_task_key(worksheet["tasks"][0])
        second_key = planner.binding_task_key(worksheet["tasks"][1])
        third_key = planner.binding_task_key(worksheet["tasks"][2])
        self.assertTrue(by_key[first_key]["command_option_ready"])
        self.assertIn("prompt_traffic_ack_missing", by_key[first_key]["blockers"])
        self.assertIn("runtime_capture_request_path_missing", by_key[second_key]["blockers"])
        self.assertIn("reads_prompt_set_path_missing", by_key[third_key]["blockers"])

    def test_model_plane_artifact_writer_fulfillment_rejects_prompt_set_binding_drift(self) -> None:
        summary = planner.build_library()
        fulfillment = model_plane_fulfillment_from_summary(summary)
        fulfillment["tasks"][0]["binding"]["reads_prompt_set_path"] = "memory-moe-mvp/phase3-real-evidence/wrong.prompt-set.json"

        validation = planner.build_binding_worksheet_from_model_plane_fulfillment(summary, fulfillment)

        self.assertTrue(validation["valid"], validation["errors"])
        self.assertFalse(validation["binding_ready"])
        self.assertIn("reads_prompt_set_path_missing", validation["blockers"])
        worksheet_validation = validation["worksheet_validation"]
        self.assertIn("reads_prompt_set_path_missing", worksheet_validation["blockers"])

    def test_binding_worksheet_path_drift_is_invalid(self) -> None:
        summary = planner.build_library()
        worksheet = planner.build_binding_worksheet(summary)
        worksheet["tasks"][0]["artifact_output_path"] = "memory-moe-mvp/phase3-real-evidence/wrong.jsonl"

        validation = planner.validate_binding_worksheet(summary, worksheet)

        self.assertFalse(validation["valid"])
        self.assertFalse(validation["worksheet_ready"])
        self.assertIn("artifact_output_path_mismatch", "\n".join(validation["errors"]))
    def test_model_plane_runtime_card_catalog_does_not_overclaim_phase3_artifact_writer(self) -> None:
        runtime_card = {
            "card_id": "llama-cpp-dolphin-mixtral-8x7b-sidecar",
            "title": "Dolphin Mixtral llama.cpp Sidecar",
            "model": "dolphin-mixtral-8x7b.gguf",
            "model_class": "mixtral_style",
            "profile_id": "dolphin-mixtral-8x7b-llama-sidecar",
            "backend_family": "llama_cpp",
            "card_type": "launch_plus_probe",
            "evidence_level": "stock_llama_cpp_observability_with_passive_sidecar",
            "probe_tier": "passive_external_plus_internal_runtime",
            "execution_mode": "runner",
            "launch_command": {"argv": ["docker", "run", "memory-moe-llama-sidecar:latest"]},
            "preflight_command": {"argv": ["python", "scripts/run_live_baseline.py", "--preflight-only"]},
            "smoke_command": {"argv": ["python", "scripts/run_live_baseline.py"]},
            "expected_artifacts": [
                "manifest.json",
                "summary.json",
                "events.jsonl",
                "llama.cpp /metrics, /slots, /props snapshots when available",
            ],
            "limitations": [
                "Stock llama.cpp observability is runtime evidence, not semantic expert id evidence.",
                "Router logits, selected expert ids, and per-layer dispatch require a later hook or fork.",
            ],
        }

        summary = planner.build_library(model_plane_card_catalog=[runtime_card])

        self.assertTrue(summary["valid"], json.dumps(summary["errors"], indent=2))
        self.assertFalse(summary["execution_ready"])
        self.assertFalse(summary["model_plane_bridge_ready"])
        self.assertEqual(summary["model_plane_runtime_launch_candidate_task_count"], 3)
        self.assertEqual(summary["model_plane_phase3_artifact_writer_contract_count"], 0)
        self.assertEqual(summary["model_plane_phase3_artifact_writer_ready_count"], 0)
        self.assertIn("model_plane_phase3_artifact_writer_missing", summary["blockers"])
        self.assertIn("model_plane_runtime_card_not_artifact_writer", summary["blockers"])
        bridge = summary["model_plane_bridge"]
        self.assertTrue(bridge["provided"])
        self.assertEqual(bridge["card_catalog_count"], 1)
        self.assertEqual(bridge["runtime_launch_candidate_task_count"], 3)
        self.assertEqual(bridge["runtime_launch_candidate_card_ids"], ["llama-cpp-dolphin-mixtral-8x7b-sidecar"])
        self.assertEqual(bridge["artifact_writer_contract_count"], 0)
        self.assertEqual(bridge["missing_artifact_writer_contract_count"], 18)
        self.assertEqual(bridge["artifact_writer_ready_count"], 0)
        self.assertEqual(bridge["missing_artifact_writer_count"], 18)
        self.assertFalse(bridge["bridge_ready"])
        self.assertIn("model_plane_candidate_card_runtime_evidence_only", bridge["blockers"])
        candidate = next(
            item
            for item in bridge["task_results"]
            if item["request_name"] == "PC Mixtral Phase 3 real evidence"
            and item["artifact_id"] == "candidate_router_trace"
        )
        self.assertTrue(candidate["runtime_launch_available"])
        self.assertEqual(candidate["runtime_launch_candidate_count"], 1)
        self.assertEqual(candidate["artifact_writer_contract_candidate_count"], 0)
        self.assertFalse(candidate["artifact_writer_contract_available"])
        self.assertFalse(candidate["artifact_writer_ready"])
        self.assertIn("model_plane_candidate_card_runtime_evidence_only", candidate["blockers"])
        self.assertEqual(candidate["runtime_candidates"][0]["card_id"], "llama-cpp-dolphin-mixtral-8x7b-sidecar")

        report = planner.format_markdown_report(summary)
        self.assertIn("## Model Plane Bridge", report)
        self.assertIn("Runtime launch candidate tasks: `3`", report)
        self.assertIn("Phase 3 artifact-writer-contract tasks: `0`", report)
        self.assertIn("Phase 3 artifact-writer-ready tasks: `0`", report)

    def test_model_plane_phase3_contract_catalog_stays_execution_pending(self) -> None:
        runtime_card = mixtral_model_plane_card_with_phase3_writers(
            ready=False,
            limitations=["Stock llama.cpp observability is runtime evidence, not semantic expert id evidence."],
        )

        summary = planner.build_library(model_plane_card_catalog=[runtime_card])

        self.assertTrue(summary["valid"], json.dumps(summary["errors"], indent=2))
        self.assertFalse(summary["execution_ready"])
        self.assertFalse(summary["model_plane_bridge_ready"])
        self.assertEqual(summary["model_plane_runtime_launch_candidate_task_count"], 3)
        self.assertEqual(summary["model_plane_phase3_artifact_writer_contract_count"], 3)
        self.assertEqual(summary["model_plane_phase3_artifact_writer_ready_count"], 0)
        self.assertIn("model_plane_phase3_artifact_writer_runtime_execution_pending", summary["blockers"])
        bridge = summary["model_plane_bridge"]
        self.assertEqual(bridge["artifact_writer_contract_count"], 3)
        self.assertEqual(bridge["missing_artifact_writer_contract_count"], 15)
        self.assertEqual(bridge["artifact_writer_ready_count"], 0)
        self.assertEqual(bridge["missing_artifact_writer_count"], 18)
        candidate = next(
            item
            for item in bridge["task_results"]
            if item["request_name"] == "PC Mixtral Phase 3 real evidence"
            and item["artifact_id"] == "candidate_router_trace"
        )
        self.assertTrue(candidate["runtime_launch_available"])
        self.assertTrue(candidate["artifact_writer_contract_available"])
        self.assertEqual(candidate["artifact_writer_contract_candidate_count"], 1)
        self.assertFalse(candidate["artifact_writer_ready"])
        self.assertIn("model_plane_phase3_artifact_writer_runtime_execution_pending", candidate["blockers"])
        self.assertNotIn("model_plane_phase3_artifact_writer_missing", candidate["blockers"])
        self.assertNotIn("model_plane_runtime_card_not_artifact_writer", candidate["blockers"])

        report = planner.format_markdown_report(summary)
        self.assertIn("Phase 3 artifact-writer-contract tasks: `3`", report)
        self.assertIn("Phase 3 artifact-writer-ready tasks: `0`", report)

    def test_model_plane_ready_artifact_writer_catalog_counts_ready_tasks(self) -> None:
        runtime_card = mixtral_model_plane_card_with_phase3_writers(ready=True)

        summary = planner.build_library(model_plane_card_catalog=[runtime_card])

        self.assertTrue(summary["valid"], json.dumps(summary["errors"], indent=2))
        self.assertFalse(summary["model_plane_bridge_ready"])
        self.assertEqual(summary["model_plane_phase3_artifact_writer_contract_count"], 3)
        self.assertEqual(summary["model_plane_phase3_artifact_writer_ready_count"], 3)
        bridge = summary["model_plane_bridge"]
        self.assertEqual(bridge["artifact_writer_contract_count"], 3)
        self.assertEqual(bridge["artifact_writer_ready_count"], 3)
        self.assertEqual(bridge["missing_artifact_writer_contract_count"], 15)
        self.assertEqual(bridge["missing_artifact_writer_count"], 15)
        managed = next(
            item
            for item in bridge["task_results"]
            if item["request_name"] == "PC Mixtral Phase 3 real evidence"
            and item["artifact_id"] == "managed_output_summary_fill"
        )
        self.assertTrue(managed["artifact_writer_contract_available"])
        self.assertTrue(managed["artifact_writer_ready"])
        self.assertEqual(managed["blockers"], [])

    def test_ollama_gguf_card_can_cover_llama_cpp_dense_output_contract(self) -> None:
        writers = mixtral_phase3_writers(ready=False)
        for writer in writers:
            if writer["artifact_id"] == "dense_output_summary_fill":
                writer["runtime_execution_ready"] = True
                writer["writes_runtime_artifacts"] = True
        runtime_card = {
            "card_id": "ollama-gemma4-31b",
            "title": "Gemma 31B Opaque Runtime Smoke",
            "model": "gemma4:31b",
            "model_class": "gemma4",
            "profile_id": "gemma4-31b-ollama",
            "backend_family": "ollama_openai_compatible",
            "card_type": "opaque_runtime_baseline",
            "execution_mode": "runner",
            "preflight_command": {"argv": ["python", "scripts/run_live_baseline.py", "--preflight-only"]},
            "smoke_command": {"argv": ["python", "scripts/run_live_baseline.py"]},
            "phase3_artifact_writers": writers,
            "limitations": ["Ollama does not expose llama.cpp hook telemetry through this card."],
        }

        summary = planner.build_library(model_plane_card_catalog=[runtime_card])

        self.assertTrue(summary["valid"], json.dumps(summary["errors"], indent=2))
        self.assertEqual(summary["model_plane_runtime_launch_candidate_task_count"], 3)
        self.assertEqual(summary["model_plane_phase3_artifact_writer_contract_count"], 3)
        self.assertEqual(summary["model_plane_phase3_artifact_writer_ready_count"], 1)
        bridge = summary["model_plane_bridge"]
        dense = next(
            item
            for item in bridge["task_results"]
            if item["request_name"] == "PC Gemma4 26B A4B Phase 3 real evidence"
            and item["artifact_id"] == "dense_output_summary_fill"
        )
        self.assertTrue(dense["runtime_launch_available"])
        self.assertTrue(dense["artifact_writer_contract_available"])
        self.assertTrue(dense["artifact_writer_ready"])
        candidate = next(
            item
            for item in bridge["task_results"]
            if item["request_name"] == "PC Gemma4 26B A4B Phase 3 real evidence"
            and item["artifact_id"] == "candidate_router_trace"
        )
        self.assertTrue(candidate["artifact_writer_contract_available"])
        self.assertFalse(candidate["artifact_writer_ready"])
        self.assertIn("model_plane_phase3_artifact_writer_runtime_execution_pending", candidate["blockers"])

    def test_model_plane_card_catalog_path_loads_for_bridge_checks(self) -> None:
        catalog = {
            "cards": [
                {
                    "card_id": "llama-cpp-dolphin-mixtral-8x7b-sidecar",
                    "title": "Dolphin Mixtral llama.cpp Sidecar",
                    "model": "dolphin-mixtral-8x7b.gguf",
                    "backend_family": "llama_cpp",
                    "execution_mode": "runner",
                    "launch_command": {"argv": ["docker", "run", "memory-moe-llama-sidecar:latest"]},
                    "limitations": ["Stock llama.cpp observability is runtime evidence, not semantic expert id evidence."],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            catalog_path = Path(tmp) / "model-plane-cards.json"
            catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
            status, summary, error = planner.plan_paths(model_plane_card_catalog_path=catalog_path)

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        self.assertEqual(summary["model_plane_bridge"]["card_catalog_count"], 1)
        self.assertEqual(summary["model_plane_bridge"]["runtime_launch_candidate_task_count"], 3)
        self.assertFalse(summary["model_plane_bridge"]["bridge_ready"])

    def test_missing_root_is_invalid_without_runtime_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing-root"
            status, summary, error = planner.plan_paths(missing)

        self.assertEqual(status, 2)
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertFalse(summary["valid"])
        self.assertFalse(summary["library_ready"])
        self.assertEqual(summary["card_count"], 0)
        self.assertFalse(summary["binding_handoff_ready"])
        self.assertEqual(summary["binding_handoff_task_count"], 0)
        self.assertIn("phase3_launch_card_library_empty", summary["blockers"])
        self.assertIsNone(error)


if __name__ == "__main__":
    unittest.main()
