import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "build_phase3_prompt_set.py"
SPEC = importlib.util.spec_from_file_location("build_phase3_prompt_set", SCRIPT_PATH)
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)

import plan_phase3_dense_fallback_capture
import plan_phase3_policy_candidate_trace

MIXTRAL_BUNDLE = ROOT / "memory-moe-mvp" / "phase3-real-evidence" / "pc_mixtral_phase3_real_evidence_bundle.json"


def args(**overrides):
    values = {
        "bundle_path": MIXTRAL_BUNDLE,
        "purpose": "shared_phase3",
        "repeat_count": 2,
        "output": None,
        "default_output": False,
        "artifact_json": False,
        "json": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class Phase3PromptSetBuilderTests(unittest.TestCase):
    def test_default_prompt_set_has_repeated_groups_and_valid_shape(self) -> None:
        status, summary, artifact, error = builder.plan_build(args())

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        assert artifact is not None
        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(artifact["schema_version"], "moe-phase3-prompt-set-v1")
        self.assertEqual(summary["prompt_count"], 8)
        groups = {}
        for row in artifact["prompts"]:
            groups.setdefault(row["group_id"], set()).add(row["repeat"])
        self.assertTrue(all(repeats == {1, 2} for repeats in groups.values()))

    def test_written_prompt_set_is_accepted_by_phase3_planners(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output = temp_path / "prompt-set.json"
            candidate_trace = temp_path / "candidate-router-events.jsonl"
            candidate_receipt = temp_path / "candidate-router-events.capture-receipt.json"
            status, summary, artifact, error = builder.plan_build(args(output=output))
            prompt_summary = plan_phase3_dense_fallback_capture.summarize_prompt_set(output)
            policy_plan = plan_phase3_policy_candidate_trace.build_capture_plan(
                MIXTRAL_BUNDLE,
                candidate_prompt_set_path=output,
                candidate_trace_path=candidate_trace,
                candidate_trace_receipt_path=candidate_receipt,
            )

        self.assertEqual(status, 0)
        self.assertIsNone(error)
        assert summary is not None
        assert artifact is not None
        self.assertTrue(prompt_summary["ready"], prompt_summary["errors"])
        self.assertEqual(prompt_summary["prompt_count"], 8)
        by_request = {item["id"]: item for item in policy_plan["artifact_requests"]}
        self.assertEqual(by_request["candidate_prompt_set_artifact"]["status"], "already_satisfied")
        self.assertEqual(by_request["candidate_trace_artifact"]["status"], "approval_required")

    def test_repeat_count_must_allow_reuse_observations(self) -> None:
        status, summary, artifact, error = builder.plan_build(args(repeat_count=1))

        self.assertEqual(status, 2)
        self.assertIsNone(summary)
        self.assertIsNone(artifact)
        self.assertIsNotNone(error)
        assert error is not None
        self.assertIn("repeat_count must be >= 2", error)


if __name__ == "__main__":
    unittest.main()