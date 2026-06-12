import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import llama_sidecar  # noqa: E402


class LlamaSidecarTests(unittest.TestCase):
    def test_parse_prometheus_metrics_summarizes_by_name(self) -> None:
        parsed = llama_sidecar.parse_prometheus_metrics(
            "# comment\n"
            "metric_a 2\n"
            "metric_b{path=\"/chat\"} 3\n"
            "metric_b{path=\"/embeddings\"} 5\n"
        )

        self.assertEqual(parsed["metric_count"], 3)
        self.assertEqual(parsed["metrics_by_name_sum"]["metric_a"], 2.0)
        self.assertEqual(parsed["metrics_by_name_sum"]["metric_b"], 8.0)

    def test_summarize_request_payload_for_chat(self) -> None:
        body = b'{"model":"mixtral","messages":[{"role":"system","content":"You are helpful."},{"role":"user","content":"Write code."}],"stream":false,"cache_prompt":true,"id_slot":2,"max_tokens":64}'
        summary = llama_sidecar.summarize_request_payload(
            path="/v1/chat/completions",
            body=body,
            store_body=False,
        )

        self.assertTrue(summary["body_json"])
        self.assertEqual(summary["endpoint_kind"], "chat_completions")
        self.assertEqual(summary["model"], "mixtral")
        self.assertEqual(summary["slot_id"], 2)
        self.assertEqual(summary["max_tokens"], 64)
        self.assertEqual(summary["text_meta"]["message_count"], 2)
        self.assertEqual(summary["text_meta"]["role_counts"]["system"], 1)
        self.assertEqual(summary["text_meta"]["role_counts"]["user"], 1)
        self.assertGreater(summary["text_meta"]["text_chars"], 0)
        self.assertIsNotNone(summary["text_meta"]["text_sha256"])

    def test_summarize_response_body_extracts_usage(self) -> None:
        body = (
            b'{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"done"}}],'
            b'"usage":{"prompt_tokens":10,"completion_tokens":4},'
            b'"timings":{"prompt_ms":12.5,"predicted_ms":3.0}}'
        )
        summary = llama_sidecar.summarize_response_body(
            body=body,
            headers={"Content-Type": "application/json"},
            streaming=False,
            store_body=False,
        )

        self.assertEqual(summary["finish_reason"], "stop")
        self.assertEqual(summary["response_chars"], 4)
        self.assertEqual(summary["usage"]["prompt_tokens"], 10)
        self.assertEqual(summary["timings"]["prompt_ms"], 12.5)

    def test_run_accumulator_updates_totals(self) -> None:
        config = llama_sidecar.SidecarConfig(
            listen_host="127.0.0.1",
            listen_port=8091,
            upstream_base_url="http://127.0.0.1:8080",
            request_timeout_seconds=60.0,
            output_dir=ROOT,
            label="test",
            store_request_body=False,
            store_response_body=False,
            capture_gpu=False,
            gpu_query_command=None,
            capture_upstream_observability=False,
            metrics_path="/metrics",
            slots_path="/slots",
            props_path="/props",
        )
        accumulator = llama_sidecar.RunAccumulator(run_id="run-1", config=config)
        accumulator.ingest(
            {
                "request": {
                    "path": "/v1/chat/completions",
                    "summary": {"endpoint_kind": "chat_completions", "model": "mixtral", "stream": False},
                },
                "response": {
                    "status_code": 200,
                    "summary": {"usage": {"prompt_tokens": 100, "completion_tokens": 20}},
                },
                "latency_ms": {"ttfb": 40.0, "total": 95.0},
                "error": None,
            }
        )
        snapshot = accumulator.snapshot()

        self.assertEqual(snapshot["totals"]["request_count"], 1)
        self.assertEqual(snapshot["totals"]["failure_count"], 0)
        self.assertEqual(snapshot["totals"]["prompt_tokens"], 100)
        self.assertEqual(snapshot["totals"]["completion_tokens"], 20)
        self.assertEqual(snapshot["breakdowns"]["by_status"]["200"], 1)
        self.assertEqual(snapshot["breakdowns"]["by_model"]["mixtral"], 1)


if __name__ == "__main__":
    unittest.main()
