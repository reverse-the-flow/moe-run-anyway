import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_runtime_baseline_capture.py"
LLAMA_FIXTURE_PATH = (
    ROOT / "memory-moe-mvp" / "data" / "model_plane_moe_probe_manifest.runtime_baseline.fixture.json"
)
VLLM_FIXTURE_PATH = (
    ROOT / "memory-moe-mvp" / "data" / "model_plane_moe_probe_manifest.vllm_runtime_baseline.fixture.json"
)
SPEC = importlib.util.spec_from_file_location("plan_runtime_baseline_capture", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


def base_manifest(**overrides):
    manifest = {
        "schema_version": "model-plane-moe-probe-manifest-v1",
        "profile_id": "baseline-example",
        "model_id": "local/example",
        "backend_family": "llama_cpp",
        "base_url": "http://127.0.0.1:18080/",
        "health_url": "http://127.0.0.1:18080/health",
        "log_file_path": "/tmp/example.log",
        "container_name": "model-plane-example",
        "primary_probe_hint": "runtime_baseline",
        "semantic_expert_ids": "not_exposed",
        "hookable_runtime_available": False,
        "runtime_observability": {
            "expected_paths": ["/props", "/metrics", "/slots"],
            "readiness_paths": ["/health"],
        },
    }
    manifest.update(overrides)
    return manifest


class RuntimeBaselineCapturePlannerTests(unittest.TestCase):
    def test_default_saved_manifest_planning_validates(self) -> None:
        status, plan, error = planner.plan_manifest_path(LLAMA_FIXTURE_PATH)

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert plan is not None
        self.assertTrue(plan["valid"], plan["errors"])
        packet = plan["capture_packet"]
        self.assertEqual(packet["stage"], "runtime_baseline_capture")
        self.assertEqual(packet["status"], "planned_only")
        self.assertEqual(packet["phase"], "phase_1")
        self.assertEqual(packet["backend_family"], "llama_cpp")
        self.assertFalse(packet["semantic_expert_ids_claim"])
        self.assertEqual(
            packet["required_pre_prompt_command_classes"],
            ["dry_run_plan", "preflight_only_readiness_gate"],
        )

    def test_vllm_manifest_is_labeled_as_runtime_evidence(self) -> None:
        status, plan, error = planner.plan_manifest_path(VLLM_FIXTURE_PATH)

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert plan is not None
        packet = plan["capture_packet"]
        self.assertEqual(packet["backend_family"], "vllm_openai_compatible")
        evidence_types = {item["evidence_type"] for item in packet["artifact_evidence"]}
        self.assertEqual(evidence_types, {"runtime_evidence"})
        self.assertIn("/v1/models", packet["runtime_observability"]["readiness_paths"])
        self.assertIn("vllm_openai_compatible", packet["safe_commands"][0]["command"])
        self.assertIn("openai-compatible-runtime", packet["safe_commands"][0]["command"])

    def test_stock_endpoint_telemetry_makes_no_semantic_expert_id_claim(self) -> None:
        plan = planner.build_plan(base_manifest(), Path("manifest.json"))

        self.assertTrue(plan["valid"], plan["errors"])
        packet = plan["capture_packet"]
        self.assertFalse(packet["semantic_expert_ids_claim"])
        self.assertTrue(packet["semantic_routing_evidence_required_for_expert_ids"])
        self.assertIn("cannot claim semantic expert ids", packet["stock_endpoint_telemetry_note"])
        for item in packet["artifact_evidence"]:
            self.assertFalse(item["may_claim_semantic_expert_ids"])

    def test_generated_safe_commands_are_dry_run_or_preflight_until_approval(self) -> None:
        plan = planner.build_plan(base_manifest(), Path("manifest.json"))
        packet = plan["capture_packet"]

        safe_commands = packet["safe_commands"]
        self.assertEqual([item["command_class"] for item in safe_commands], [
            "dry_run_plan",
            "preflight_only_readiness_gate",
        ])
        self.assertIn("--dry-run", safe_commands[0]["command"])
        self.assertIn("--preflight-only", safe_commands[1]["command"])
        self.assertTrue(all("--json" in item["command"] for item in safe_commands))
        self.assertTrue(all(not item["may_include_prompt_traffic"] for item in safe_commands))
        deferred = packet["deferred_prompt_traffic_commands"][0]
        self.assertTrue(deferred["requires_explicit_user_approval"])
        self.assertTrue(deferred["may_include_prompt_traffic"])
        self.assertEqual(
            deferred["requires_prior_command_classes"],
            ["dry_run_plan", "preflight_only_readiness_gate"],
        )

    def test_invalid_manifests_fail_cleanly(self) -> None:
        plan = planner.build_plan(
            base_manifest(
                backend_family="unknown_runtime",
                primary_probe_hint="passive_sidecar",
                semantic_expert_ids="expected_when_router_outputs_are_exposed",
            ),
            Path("manifest.json"),
        )

        self.assertFalse(plan["valid"])
        self.assertIsNone(plan["capture_packet"])
        self.assertTrue(any("backend_family" in error for error in plan["errors"]))
        self.assertTrue(any("primary_probe_hint" in error for error in plan["errors"]))
        self.assertTrue(any("cannot claim semantic expert ids" in error for error in plan["errors"]))

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            path.write_text(json.dumps({}), encoding="utf-8")
            self.assertEqual(planner.main_from_test_path(path), 2)


if __name__ == "__main__":
    unittest.main()
