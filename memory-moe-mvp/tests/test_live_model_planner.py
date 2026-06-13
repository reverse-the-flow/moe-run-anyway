import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_live_model.py"
SPEC = importlib.util.spec_from_file_location("plan_live_model", SCRIPT_PATH)
plan_live_model = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = plan_live_model
SPEC.loader.exec_module(plan_live_model)


def fake_capabilities(*, llama_server: bool = False, observability: bool = False) -> dict:
    return {
        "project_root": str(ROOT),
        "platform": {
            "system": "Linux",
            "release": "test",
            "machine": "x86_64",
            "platform": "Linux-test",
        },
        "python": {
            "executable": "/usr/bin/python3",
            "version": "3.12",
            "implementation": "CPython",
            "modules": {
                "torch": {"module": "torch", "available": False, "origin": None},
                "transformers": {"module": "transformers", "available": False, "origin": None},
            },
        },
        "gpu_tooling": {"detected": {"any": True, "families": ["nvidia_cuda"]}},
        "backend_tools": {
            "llama-server": {"command": "llama-server", "path": "/usr/bin/llama-server" if llama_server else None, "available": llama_server},
            "docker": {"command": "docker", "path": "/usr/bin/docker", "available": True},
            "huggingface-cli": {"command": "huggingface-cli", "path": None, "available": False},
        },
        "cached_model_hints": {
            "search_roots": [],
            "existing_roots": [],
            "truncated": False,
            "environment_model_paths": [],
            "huggingface_models": [],
            "gguf_files": [],
        },
        "live_backend": {
            "base_url": "http://127.0.0.1:18080",
            "observability_available": observability,
            "available_paths": ["/props"] if observability else [],
            "endpoints": [],
        },
    }


class LiveModelPlannerTests(unittest.TestCase):
    def test_scan_cached_model_hints_finds_hf_dirs_and_gguf_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            hf_model = root / "models--org--mixtral-test"
            hf_model.mkdir()
            (hf_model / "snapshots").mkdir()
            lock_model = root / ".locks" / "models--org--lock-only"
            lock_model.mkdir(parents=True)
            gguf = root / "weights" / "model.Q4_K_M.gguf"
            gguf.parent.mkdir()
            gguf.write_bytes(b"test")

            hints = plan_live_model.scan_cached_model_hints([root], max_hints=8)

        self.assertEqual(hints["huggingface_models"][0]["model_id"], "org/mixtral-test")
        self.assertEqual(len(hints["huggingface_models"]), 1)
        self.assertEqual(hints["gguf_files"][0]["path"], str(gguf))
        self.assertFalse(hints["truncated"])

    def test_missing_backend_plan_emits_guarded_commands_without_claiming_semantics(self) -> None:
        registry = plan_live_model.load_registry(plan_live_model.DEFAULT_REGISTRY_PATH)
        plan = plan_live_model.build_plan(
            registry,
            fake_capabilities(llama_server=False, observability=False),
            base_url="http://127.0.0.1:18080",
            target_class="stock_llama_cpp_openai_compatible",
        )

        target = plan["target_plans"][0]
        self.assertEqual(target["readiness_state"], "needs_user_started_observable_backend")
        self.assertIn("llama-server not found on PATH", target["blockers"])
        self.assertIn("not semantic expert ids", target["honesty_note"])
        joined_commands = "\n".join(target["commands"])
        self.assertIn("scripts/run_live_baseline.py", joined_commands)
        self.assertIn("--preflight-only", joined_commands)

    def test_hookable_plan_requires_local_runtime_for_semantic_traces(self) -> None:
        registry = plan_live_model.load_registry(plan_live_model.DEFAULT_REGISTRY_PATH)
        plan = plan_live_model.build_plan(
            registry,
            fake_capabilities(),
            base_url="http://127.0.0.1:18080",
            target_class="hookable_pytorch_moe",
        )

        target = plan["target_plans"][0]
        self.assertEqual(target["readiness_state"], "needs_local_hookable_runtime")
        self.assertIn("torch module not detected", target["blockers"])
        self.assertIn("router outputs", target["honesty_note"])
        self.assertIn("run_forward_probe_demo.py", "\n".join(target["commands"]))


if __name__ == "__main__":
    unittest.main()
