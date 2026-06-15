import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import llama_runtime_probe  # noqa: E402


class StubState:
    request_count = 0
    prompt_tokens = 0
    completion_tokens = 0


class StubHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A003
        return

    def do_GET(self):  # noqa: N802
        if self.path == "/metrics":
            body = (
                "# HELP llamacpp:prompt_tokens_seconds Synthetic metric\n"
                f"llamacpp:prompt_tokens_seconds {StubState.prompt_tokens}\n"
                f"llamacpp:predicted_tokens_seconds {StubState.completion_tokens}\n"
                f"llamacpp:requests_total{{path=\"/v1/chat/completions\"}} {StubState.request_count}\n"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/slots":
            body = json.dumps(
                [
                    {"id": 0, "state": "idle"},
                    {"id": 1, "state": "busy" if StubState.request_count > 0 else "idle"},
                ]
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/props":
            body = json.dumps({"ctx_size": 8192, "n_parallel": 1}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        StubState.request_count += 1
        StubState.prompt_tokens += 11
        StubState.completion_tokens += 4

        body = json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ready"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 4,
                    "total_tokens": 15,
                },
                "timings": {
                    "prompt_ms": 12.5,
                    "predicted_ms": 18.0,
                },
                "echo_model": payload.get("model"),
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class LlamaRuntimeProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self) -> None:
        StubState.request_count = 0
        StubState.prompt_tokens = 0
        StubState.completion_tokens = 0

    def test_parse_prometheus_metrics(self) -> None:
        parsed = llama_runtime_probe.parse_prometheus_metrics(
            "# comment\n"
            "metric_a 2\n"
            "metric_b{path=\"/chat\"} 3\n"
            "metric_b{path=\"/embeddings\"} 5\n"
        )
        self.assertEqual(parsed["metric_count"], 3)
        self.assertEqual(parsed["metrics_by_name_sum"]["metric_a"], 2.0)
        self.assertEqual(parsed["metrics_by_name_sum"]["metric_b"], 8.0)

    def test_diff_metric_summaries_detects_changes(self) -> None:
        before = {"metrics_by_name_sum": {"a": 1.0, "b": 2.0}}
        after = {"metrics_by_name_sum": {"a": 3.5, "b": 2.0, "c": 1.0}}
        diff = llama_runtime_probe.diff_metric_summaries(before, after)
        self.assertEqual(diff["changed_metric_count"], 2)
        self.assertEqual(diff["changed_metrics"]["a"], 2.5)
        self.assertEqual(diff["changed_metrics"]["c"], 1.0)

    def test_probe_runs_against_stub_server(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "server.log"
            log_path.write_text("before\n", encoding="utf-8")

            config = llama_runtime_probe.RuntimeProbeConfig(
                base_url=self.base_url,
                output_dir=Path(temp_dir),
                label="stub",
                request_timeout_seconds=10.0,
                model="mixtral",
                backend_family="vllm_openai_compatible",
                log_file_path=log_path,
            )
            probe = llama_runtime_probe.LlamaRuntimeProbe(config=config)
            log_path.write_text("before\nafter\n", encoding="utf-8")
            event = probe.run_case(
                {
                    "family_id": "english_prose",
                    "probe_id": "probe-1",
                    "title": "Stub case",
                    "repeat": 1,
                    "body": {
                        "model": "mixtral",
                        "messages": [{"role": "user", "content": "Say ready."}],
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "max_tokens": 8,
                        "stream": False,
                    },
                }
            )

            self.assertIsNone(event["error"])
            self.assertEqual(event["response"]["summary"]["finish_reason"], "stop")
            self.assertEqual(event["observability"]["before"]["slots"]["summary"]["slot_count"], 2)
            self.assertEqual(
                event["observability"]["metrics_delta"]["changed_metrics"]["llamacpp:requests_total"],
                1.0,
            )
            self.assertTrue(event["observability"]["log_growth"]["available"])
            self.assertEqual(event["shared_contract"]["backend_family"], "vllm_openai_compatible")

            manifest = json.loads(probe.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["config"]["backend_family"], "vllm_openai_compatible")
            self.assertEqual(
                manifest["shared_contract_template"]["defaults"]["backend_family"],
                "vllm_openai_compatible",
            )
            summary = json.loads(probe.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["config"]["backend_family"], "vllm_openai_compatible")
            self.assertEqual(summary["totals"]["request_count"], 1)
            self.assertEqual(summary["breakdowns"]["by_family"]["english_prose"], 1)


if __name__ == "__main__":
    unittest.main()
