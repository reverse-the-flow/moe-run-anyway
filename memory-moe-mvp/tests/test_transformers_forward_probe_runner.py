import argparse
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCRIPT_PATH = ROOT / "run_transformers_forward_probe.py"
SPEC = importlib.util.spec_from_file_location("run_transformers_forward_probe", SCRIPT_PATH)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def make_args(**overrides):
    values = {
        "model_path": Path("/missing/local/model"),
        "output_dir": Path("forward-probe-runs"),
        "suite_path": ROOT / "data" / "mixtral_probe_prompts.json",
        "label": "test-transformers-hookable",
        "max_prompts": 2,
        "repeats": 1,
        "window_size_events": 2,
        "dry_run": True,
        "trust_remote_code": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def present_module(name: str) -> dict:
    return {"module": name, "available": True, "origin": f"/fake/{name}.py"}


class TransformersForwardProbeRunnerTests(unittest.TestCase):
    def test_dry_run_rejects_remote_or_missing_model_path(self) -> None:
        args = make_args(model_path=Path("https://huggingface.co/org/model"))

        with patch.object(runner, "module_presence", side_effect=present_module):
            with patch.dict(os.environ, {}, clear=True):
                plan = runner.plan_transformers_forward_probe(args)

        self.assertFalse(plan["ready_to_run"])
        self.assertIn("model path must be a local filesystem path", "\n".join(plan["blockers"]))
        self.assertIn("model path does not exist", plan["blockers"])
        self.assertIn("no downloads", plan["safety_contract"])

    def test_dry_run_reports_missing_optional_dependencies_without_importing_them(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            args = make_args(model_path=model_dir)

            with patch.object(
                runner,
                "module_presence",
                side_effect=lambda name: {"module": name, "available": False, "origin": None},
            ):
                with patch.dict(os.environ, {}, clear=True):
                    plan = runner.plan_transformers_forward_probe(args)

        self.assertFalse(plan["ready_to_run"])
        self.assertIn("torch module not detected", plan["blockers"])
        self.assertIn("transformers module not detected", plan["blockers"])
        self.assertEqual(plan["selected_prompt_count"], 2)
        self.assertIn("run_transformers_forward_probe.py", plan["next_command"])
        self.assertNotIn("future_transformers_runner", plan["next_command"])

    def test_dry_run_blocks_token_environment_by_name_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            args = make_args(model_path=model_dir)

            with patch.object(runner, "module_presence", side_effect=present_module):
                with patch.dict(os.environ, {"HF_TOKEN": "secret-value"}, clear=True):
                    plan = runner.plan_transformers_forward_probe(args)

        self.assertFalse(plan["ready_to_run"])
        self.assertEqual(plan["token_env_names_present"], ["HF_TOKEN"])
        self.assertNotIn("secret-value", str(plan))
        self.assertIn("Hugging Face token environment variables", "\n".join(plan["blockers"]))


if __name__ == "__main__":
    unittest.main()
