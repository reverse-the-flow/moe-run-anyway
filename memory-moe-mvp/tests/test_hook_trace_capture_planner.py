import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_hook_trace_capture.py"
SPEC = importlib.util.spec_from_file_location("plan_hook_trace_capture", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


def make_model_dir(root: Path) -> Path:
    model_dir = root / "hookable-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "fixture_moe"}), encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return model_dir


def hook_manifest(model_dir: Path, **overrides):
    manifest = {
        "schema_version": "model-plane-moe-probe-manifest-v1",
        "profile_id": "fixture-hookable",
        "model_id": "local/fixture-hookable",
        "model_path": str(model_dir),
        "backend_family": "pytorch_transformers",
        "primary_probe_hint": "hookable_pytorch",
        "semantic_expert_ids": "expected_when_router_outputs_are_exposed",
        "hookable_runtime_available": True,
    }
    manifest.update(overrides)
    return manifest


class HookTraceCapturePlannerTests(unittest.TestCase):
    def test_valid_hookable_manifest_plans_synthetic_dry_run_and_approved_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = make_model_dir(Path(temp_dir))
            plan = planner.build_plan(hook_manifest(model_dir), Path("manifest.json"))

        self.assertTrue(plan["valid"], plan["errors"])
        self.assertEqual(plan["stage"], "semantic_router_trace_capture")
        self.assertEqual(plan["semantic_expert_ids_claim"], False)
        self.assertEqual(len(plan["safe_commands"]), 2)
        self.assertEqual(plan["safe_commands"][0]["command_class"], "synthetic_hook_smoke")
        self.assertEqual(plan["safe_commands"][1]["command_class"], "local_transformers_hook_dry_run")
        self.assertIn("run_forward_probe_demo.py", plan["safe_commands"][0]["command"])
        self.assertIn("run_transformers_forward_probe.py", plan["safe_commands"][1]["command"])
        self.assertIn("--dry-run", plan["safe_commands"][1]["command"])
        self.assertEqual(len(plan["deferred_prompt_traffic_commands"]), 1)
        live = plan["deferred_prompt_traffic_commands"][0]
        self.assertEqual(live["command_class"], "approved_local_transformers_hook_trace")
        self.assertIn("run_transformers_forward_probe.py", live["command"])
        self.assertNotIn("--dry-run", live["command"])
        trace_evidence = plan["artifact_evidence"][1]
        self.assertTrue(trace_evidence["may_claim_semantic_expert_ids"])
        self.assertIn("router_events.jsonl", " ".join(trace_evidence["claim_preconditions"]))

    def test_rejects_non_hookable_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = make_model_dir(Path(temp_dir))
            plan = planner.build_plan(
                hook_manifest(
                    model_dir,
                    primary_probe_hint="runtime_baseline",
                    hookable_runtime_available=False,
                    semantic_expert_ids="not_exposed",
                ),
                Path("manifest.json"),
            )

        self.assertFalse(plan["valid"])
        joined = "\n".join(plan["errors"])
        self.assertIn("primary_probe_hint", joined)
        self.assertIn("hookable_runtime_available", joined)

    def test_rejects_remote_model_ids_and_missing_paths(self) -> None:
        plan = planner.build_plan(
            hook_manifest(Path("https://huggingface.co/org/model")),
            Path("manifest.json"),
        )

        self.assertFalse(plan["valid"])
        self.assertIn("local filesystem path", "\n".join(plan["errors"]))

    def test_direct_model_path_args_build_manifest_without_loading_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = make_model_dir(Path(temp_dir))
            args = argparse.Namespace(
                manifest_path=None,
                model_path=model_dir,
                label="direct-fixture",
                json=True,
            )
            status, plan, error = planner.plan_from_args(args)

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert plan is not None
        self.assertTrue(plan["valid"], plan["errors"])
        self.assertEqual(plan["profile_id"], "direct-fixture")
        self.assertIn("does not import torch", " ".join(plan["safety_contract"]))


if __name__ == "__main__":
    unittest.main()
