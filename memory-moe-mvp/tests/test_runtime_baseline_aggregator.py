import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AGGREGATOR_PATH = ROOT / "scripts" / "aggregate_runtime_baselines.py"
SPEC = importlib.util.spec_from_file_location("aggregate_runtime_baselines", AGGREGATOR_PATH)
aggregate_runtime_baselines = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(aggregate_runtime_baselines)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class RuntimeBaselineAggregatorTests(unittest.TestCase):
    def test_summarize_run_reports_latency_prompts_and_observability(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run-a"
            run_dir.mkdir()
            write_json(
                run_dir / "summary.json",
                {
                    "run_id": "run-a",
                    "config": {
                        "backend_family": "llama_cpp",
                        "model": "mixtral",
                        "label": "mixtral-coverage",
                    },
                    "totals": {
                        "request_count": 2,
                        "failure_count": 0,
                        "latency_observation_count": 2,
                        "mean_latency_ms": 15.0,
                    },
                    "breakdowns": {
                        "by_family": {"english_prose": 2},
                        "by_finish_reason": {"stop": 2},
                    },
                },
            )
            write_json(run_dir / "manifest.json", {"run_id": "run-a", "config": {}})
            write_jsonl(
                run_dir / "events.jsonl",
                [
                    {
                        "case": {
                            "family_id": "english_prose",
                            "probe_id": "prose-summary-01",
                            "repeat": 1,
                        },
                        "latency_ms": {"total": 10.0},
                        "observability": {
                            "after": {
                                "metrics": {"available": True},
                                "props": {"available": True},
                                "slots": {"available": True},
                            },
                            "metrics_delta": {
                                "changed_metric_count": 2,
                                "changed_metrics": {"metric_a": 1, "metric_b": 1},
                            },
                        },
                    },
                    {
                        "case": {
                            "family_id": "english_prose",
                            "probe_id": "prose-summary-01",
                            "repeat": 2,
                        },
                        "latency_ms": {"total": 20.0},
                        "observability": {
                            "after": {
                                "metrics": {"available": True},
                                "props": {"available": False},
                                "slots": {"available": True},
                            },
                            "metrics_delta": {
                                "changed_metric_count": 1,
                                "changed_metrics": {"metric_a": 1},
                            },
                        },
                    },
                ],
            )

            summary = aggregate_runtime_baselines.summarize_run(run_dir)

        self.assertEqual(summary["run_id"], "run-a")
        self.assertEqual(summary["request_count"], 2)
        self.assertEqual(summary["unique_prompt_count"], 1)
        self.assertEqual(summary["prompt_repeat_count_min"], 2)
        self.assertEqual(summary["prompt_repeat_count_max"], 2)
        self.assertEqual(summary["latency_ms"]["median"], 15.0)
        self.assertEqual(summary["observability_available"], {"metrics": 2, "props": 1, "slots": 2})
        self.assertEqual(summary["changed_metric_count"]["min"], 1)
        self.assertEqual(summary["changed_metric_count"]["max"], 2)
        self.assertEqual(summary["top_changed_metrics"][0], ("metric_a", 2))


if __name__ == "__main__":
    unittest.main()
