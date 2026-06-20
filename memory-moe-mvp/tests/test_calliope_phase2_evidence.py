import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = ROOT / "data" / "calliope_moe_phase2_evidence.json"


class CalliopePhase2EvidenceTests(unittest.TestCase):
    def load_evidence(self):
        return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))

    def test_evidence_covers_current_calliope_moe_targets(self) -> None:
        evidence = self.load_evidence()
        target_ids = {target["target_id"] for target in evidence["targets"]}

        self.assertEqual(evidence["schema_version"], "calliope-moe-phase2-evidence-v1")
        self.assertIn("dolphin_mixtral_8x7b_gguf_llama_cpp", target_ids)
        self.assertIn("qwen3_30b_a3b_gguf_llama_cpp", target_ids)
        self.assertIn("nemotron_3_super_120b_gguf_llama_cpp", target_ids)
        self.assertIn("nemotron_3_nano_omni_30b_a3b_nvfp4_vllm", target_ids)

    def test_runtime_baselines_do_not_claim_semantic_expert_ids(self) -> None:
        evidence = self.load_evidence()

        for target in evidence["targets"]:
            self.assertFalse(target["semantic_expert_ids_observed"], target["target_id"])
        self.assertIn("do not prove semantic expert routing", evidence["honesty_boundary"])

    def test_nemotron_super_records_load_failure_not_request_failure(self) -> None:
        evidence = self.load_evidence()
        target = next(
            item
            for item in evidence["targets"]
            if item["target_id"] == "nemotron_3_super_120b_gguf_llama_cpp"
        )

        self.assertEqual(target["run_summary"]["request_count"], 0)
        self.assertEqual(target["run_summary"]["finish_reason"], "model_load_failed")
        self.assertEqual(target["failure"]["kind"], "llama_cpp_tensor_shape_mismatch")

    def test_vllm_row_records_richer_metric_surface(self) -> None:
        evidence = self.load_evidence()
        by_id = {target["target_id"]: target for target in evidence["targets"]}

        self.assertGreater(
            by_id["nemotron_3_nano_omni_30b_a3b_nvfp4_vllm"]["run_summary"][
                "changed_metric_count"
            ],
            by_id["dolphin_mixtral_8x7b_gguf_llama_cpp"]["run_summary"][
                "changed_metric_count"
            ],
        )
        self.assertEqual(
            by_id["nemotron_3_nano_omni_30b_a3b_nvfp4_vllm"]["run_summary"][
                "finish_reason"
            ],
            "stop",
        )


if __name__ == "__main__":
    unittest.main()
