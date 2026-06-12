import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import moe_controller_demo  # noqa: E402
import run_forward_probe_demo  # noqa: E402


def read_jsonl(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def find_summary(summaries, policy_name, budget_fraction):
    target_fraction = round(float(budget_fraction), 3)
    for item in summaries:
        if item["policy_name"] == policy_name and item["resident_budget_fraction"] == target_fraction:
            return item
    raise AssertionError(f"Missing summary for {policy_name} at {budget_fraction}")


class ControllerDemoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = moe_controller_demo.load_trace_dataset(
            ROOT / "data" / "synthetic_controller_trace.json"
        )

    def test_execute_bridge_plan_writes_budget_sweeps_and_controller_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = moe_controller_demo.execute_bridge_plan(
                dataset=self.dataset,
                output_dir=Path(temp_dir),
                label="controller-test",
            )
            summary = result["summary"]
            self.assertEqual(summary["bridge_status"]["status"], "paused")
            self.assertEqual(len(summary["budget_sweeps"]), 6)
            self.assertIn("controller_demo", summary)
            self.assertIn("failure_mode_matrix", summary)
            self.assertIn("next_stage_branch", summary)

            events = read_jsonl(Path(result["run_dir"]) / "events.jsonl")
            controller_events = [item for item in events if item["run_kind"] == "controller_demo"]
            self.assertTrue(controller_events)
            self.assertTrue(any(event["fallback_used"] for event in controller_events))
            self.assertTrue(
                all(event["shared_contract"]["candidate_set_size"] is not None for event in controller_events)
            )
            self.assertTrue(
                all(event["quality"]["dense_baseline_delta"] is not None for event in controller_events)
            )

    def test_predictive_and_weighted_policies_reduce_miss_and_churn_vs_reactive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = moe_controller_demo.execute_bridge_plan(
                dataset=self.dataset,
                output_dir=Path(temp_dir),
                label="policy-test",
            )
            sweeps = result["summary"]["budget_sweeps"]
            reactive = find_summary(sweeps, "reactive_lru", 0.4)
            weighted = find_summary(sweeps, "weighted_lru", 0.4)
            predictive = find_summary(sweeps, "window_predictive", 0.4)

            self.assertLess(weighted["mean_miss_rate"], reactive["mean_miss_rate"])
            self.assertLess(predictive["mean_miss_rate"], reactive["mean_miss_rate"])
            self.assertLess(weighted["mean_churn"], reactive["mean_churn"])

    def test_advisor_flags_hybrid_as_chaotic_and_bridge_loader_handles_demo_runs(self) -> None:
        advisor = moe_controller_demo.build_advisor_report(self.dataset)
        self.assertEqual(advisor["family_findings"]["hybrid_mixed"]["status"], "chaotic")
        self.assertEqual(advisor["family_findings"]["english_prose"]["status"], "compact")

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = run_forward_probe_demo.run_demo(
                output_dir=Path(temp_dir),
                label="forward-probe-demo",
                suite_path=ROOT / "data" / "mixtral_probe_prompts.json",
                max_prompts=3,
                window_size_tokens=32,
                window_size_events=2,
            )
            dataset = moe_controller_demo.load_forward_probe_dataset(run_dir)
            self.assertEqual(dataset.source_kind, "forward_hook_demo")
            self.assertGreater(len(dataset.windows), 0)
            self.assertGreater(len(dataset.experts), 0)


if __name__ == "__main__":
    unittest.main()
