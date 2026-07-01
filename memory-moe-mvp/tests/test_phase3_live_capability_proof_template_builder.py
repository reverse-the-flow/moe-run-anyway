import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "build_phase3_live_capability_proof_template.py"
SPEC = importlib.util.spec_from_file_location("build_phase3_live_capability_proof_template", SCRIPT_PATH)
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def args(**overrides):
    defaults = {
        "bundle_path": builder.DEFAULT_BUNDLE_PATH,
        "proof_scope": None,
        "output": None,
        "default_output": False,
        "artifact_json": False,
        "json": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class Phase3LiveCapabilityProofTemplateBuilderTests(unittest.TestCase):
    def test_default_template_is_valid_but_not_proof_ready(self) -> None:
        status, summary, artifact, error = builder.plan_build(args())

        self.assertEqual(status, 0, error)
        assert summary is not None
        assert artifact is not None
        self.assertTrue(summary["valid"], summary["errors"])
        self.assertTrue(summary["template_ready"])
        self.assertFalse(summary["proof_ready"])
        self.assertFalse(summary["phase_3_gate"]["ready_for_live_spike"])
        blocker_ids = {item["id"] for item in summary["blockers"]}
        self.assertIn("residency_observation_not_available", blocker_ids)
        self.assertIn("residency_control_is_dry_run_only", blocker_ids)
        self.assertIn("cleanup_restore_not_verified", blocker_ids)
        self.assertEqual(artifact["schema_version"], "moe-phase3-live-capability-proof-v1")
        self.assertEqual(artifact["model_id"], "/root/.ollama/models/blobs/sha256-5041ba4278429fe475782b889471b5ff065a6cce5c3a539bd61d1e457f1961de")

    def test_default_output_writes_template_and_validator_accepts_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = builder.DEFAULT_BUNDLE_PATH
            output = Path(temp_dir) / "live-template.json"

            status, summary, artifact, error = builder.plan_build(args(bundle_path=bundle, output=output))

            self.assertEqual(status, 0, error)
            self.assertTrue(output.exists())
            written = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(written, artifact)
            self.assertEqual(written["artifact_export"]["artifact_paths"], [str(output)])
            assert summary is not None
            self.assertEqual(summary["output_path"], str(output))

            proof_summary = builder.plan_phase3_live_capability_proof.build_summary(output)

        self.assertTrue(proof_summary["valid"], proof_summary["errors"])
        self.assertTrue(proof_summary["proof_available"])
        self.assertFalse(proof_summary["proof_ready"])

    def test_rejects_output_and_default_output_together(self) -> None:
        status, summary, artifact, error = builder.plan_build(
            args(output=Path("custom.json"), default_output=True)
        )

        self.assertEqual(status, 2)
        self.assertIsNone(summary)
        self.assertIsNone(artifact)
        self.assertEqual(error, "--output and --default-output cannot be used together")


if __name__ == "__main__":
    unittest.main()
