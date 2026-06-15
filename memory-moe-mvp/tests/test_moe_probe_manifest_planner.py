import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_moe_probe_manifest.py"
SPEC = importlib.util.spec_from_file_location("plan_moe_probe_manifest", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


def base_manifest(**overrides):
    manifest = {
        "schema_version": "model-plane-moe-probe-manifest-v1",
        "profile_id": "llama-cpp-example",
        "model_id": "local/example-gguf",
        "model_path": "/models/example.gguf",
        "backend_family": "llama_cpp",
        "base_url": "http://127.0.0.1:18080",
        "health_url": "http://127.0.0.1:18080/health",
        "log_file_path": "/tmp/llama-server.log",
        "container_name": "dockyard-llama-cpp-example",
        "primary_probe_hint": "runtime_baseline",
        "semantic_expert_ids": "not_exposed",
        "hookable_runtime_available": False,
        "safety_notes": [],
    }
    manifest.update(overrides)
    return manifest


class MoeProbeManifestPlannerTests(unittest.TestCase):
    def test_runtime_manifest_plans_only_dry_run_and_preflight_commands(self) -> None:
        plan = planner.build_plan(base_manifest(), Path("manifest.json"))

        self.assertTrue(plan["valid"])
        self.assertEqual(plan["target_class"], "stock_llama_cpp_openai_compatible")
        self.assertEqual(len(plan["safe_commands"]), 2)
        self.assertIn("--dry-run", plan["safe_commands"][0])
        self.assertIn("--preflight-only", plan["safe_commands"][1])
        self.assertIn("--backend-family llama_cpp", plan["safe_commands"][0])
        self.assertIn("--log-file-path /tmp/llama-server.log", plan["safe_commands"][0])
        self.assertIn("not semantic expert ids", plan["honesty_note"])
        request = plan["planned_harness_run_request"]
        self.assertEqual(request["stage"], "harness_run_request")
        self.assertEqual(request["status"], "planned_only")
        self.assertEqual(request["target_class"], "stock_llama_cpp_openai_compatible")
        self.assertIn("expert_tensor_preload", request["missing_runtime_actuator"])

    def test_openai_compatible_manifest_passes_backend_family_to_runtime_plan(self) -> None:
        plan = planner.build_plan(
            base_manifest(
                backend_family="vllm_openai_compatible",
                profile_id="gemma-vllm-profile",
                model_id="local/openai-compatible",
                log_file_path="",
            ),
            Path("manifest.json"),
        )

        self.assertTrue(plan["valid"])
        self.assertEqual(plan["target_class"], "openai_compatible_runtime")
        joined = "\n".join(plan["safe_commands"])
        self.assertIn("--backend-family vllm_openai_compatible", joined)
        self.assertIn("gemma-vllm-profile-openai-compatible-runtime", joined)
        self.assertIn("--preflight-only", plan["safe_commands"][1])
        self.assertIn("not semantic expert ids", plan["honesty_note"])

    def test_passive_manifest_routes_to_sidecar(self) -> None:
        plan = planner.build_plan(
            base_manifest(primary_probe_hint="passive_sidecar"),
            Path("manifest.json"),
        )

        self.assertTrue(plan["valid"])
        self.assertEqual(plan["target_class"], "passive_sidecar_proxy")
        self.assertEqual(len(plan["safe_commands"]), 1)
        self.assertIn("llama_sidecar.py", plan["safe_commands"][0])
        self.assertIn("--upstream-base-url http://127.0.0.1:18080", plan["safe_commands"][0])

    def test_hookable_path_requires_manifest_flag(self) -> None:
        invalid = planner.build_plan(
            base_manifest(primary_probe_hint="hookable_pytorch"),
            Path("manifest.json"),
        )

        self.assertFalse(invalid["valid"])
        self.assertIn("hookable_runtime_available=true", invalid["errors"][0])

        valid = planner.build_plan(
            base_manifest(
                primary_probe_hint="hookable_pytorch",
                backend_family="pytorch_transformers",
                semantic_expert_ids="expected_when_router_outputs_are_exposed",
                hookable_runtime_available=True,
            ),
            Path("manifest.json"),
        )

        self.assertTrue(valid["valid"])
        self.assertEqual(valid["target_class"], "hookable_pytorch_moe")
        self.assertIn("run_forward_probe_demo.py", valid["safe_commands"][0])
        self.assertIn("future_transformers_runner.py", valid["deferred_live_commands"][0])

    def test_cli_returns_nonzero_for_invalid_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            path.write_text("{}", encoding="utf-8")
            self.assertEqual(planner.main_from_test_path(path), 2)


if __name__ == "__main__":
    unittest.main()
