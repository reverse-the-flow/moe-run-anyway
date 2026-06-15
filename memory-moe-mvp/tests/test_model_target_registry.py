import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import model_target_registry  # noqa: E402


class ModelTargetRegistryTests(unittest.TestCase):
    def test_default_registry_is_valid_and_covers_required_classes(self) -> None:
        registry = model_target_registry.load_registry(model_target_registry.default_registry_path())
        errors = model_target_registry.validate_registry(registry)
        self.assertEqual(errors, [])

        targets = registry["targets"]
        covered_classes = {target["target_class"] for target in targets}
        self.assertGreaterEqual(covered_classes, set(registry["required_target_classes"]))

    def test_registry_summary_counts_classes_and_backends(self) -> None:
        registry = model_target_registry.load_registry(model_target_registry.default_registry_path())
        summary = model_target_registry.summarize_registry(registry)

        self.assertEqual(summary["schema_version"], "memory-moe-target-registry-v1")
        self.assertEqual(summary["target_count"], len(registry["targets"]))
        self.assertIn("stock_llama_cpp_openai_compatible", summary["by_class"])
        self.assertIn("openai_compatible_runtime", summary["by_class"])
        self.assertIn("openai_compatible", summary["by_backend"])
        self.assertIn("pytorch_transformers", summary["by_backend"])

    def test_validator_rejects_missing_target_class(self) -> None:
        registry = {
            "schema_version": "test",
            "required_target_classes": ["missing_class"],
            "targets": [
                {
                    "target_id": "target-1",
                    "target_class": "present_class",
                    "backend_family": "test",
                    "probe_tier": "test",
                    "model_family": "test",
                    "semantic_expert_ids": "not_exposed",
                    "local_only_status": "local",
                    "primary_probe": "probe.py",
                    "observable_signals": ["signal"],
                    "deferred_requirements": ["requirement"],
                    "next_live_commands": ["python3 probe.py"],
                }
            ],
        }

        errors = model_target_registry.validate_registry(registry)
        self.assertIn("missing required target classes: missing_class", errors)


if __name__ == "__main__":
    unittest.main()
