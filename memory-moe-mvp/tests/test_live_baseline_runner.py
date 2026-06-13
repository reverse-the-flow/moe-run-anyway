import importlib.util
import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "run_live_baseline.py"
SPEC = importlib.util.spec_from_file_location("run_live_baseline", SCRIPT_PATH)
run_live_baseline = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = run_live_baseline
SPEC.loader.exec_module(run_live_baseline)


class ObservabilityHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A003
        return

    def do_GET(self):  # noqa: N802
        if self.path == "/props":
            body = json.dumps({"ctx_size": 4096}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/metrics":
            body = b"requests_total 0\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/slots":
            body = b"[]"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()


class LiveBaselineRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), ObservabilityHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_preflight_observability_detects_stub_endpoints(self) -> None:
        result = run_live_baseline.preflight_observability(self.base_url, timeout_seconds=2.0)
        self.assertTrue(result["observability_available"])
        self.assertEqual(set(result["available_paths"]), {"/props", "/metrics", "/slots"})

    def test_preflight_observability_reports_missing_server(self) -> None:
        result = run_live_baseline.preflight_observability("http://127.0.0.1:1", timeout_seconds=0.2)
        self.assertFalse(result["observability_available"])
        self.assertEqual(result["available_paths"], [])

    def test_build_runtime_probe_command_includes_required_args(self) -> None:
        parser = run_live_baseline.build_arg_parser()
        args = parser.parse_args(
            [
                "--base-url",
                "http://127.0.0.1:18080/",
                "--model",
                "mixtral-test",
                "--max-prompts",
                "2",
                "--repeats",
                "3",
                "--preflight-timeout-seconds",
                "1.5",
                "--dry-run",
            ]
        )
        command = run_live_baseline.build_runtime_probe_command(args)
        self.assertIn("llama_runtime_probe.py", command[1])
        self.assertIn("http://127.0.0.1:18080", command)
        self.assertIn("mixtral-test", command)
        self.assertIn("2", command)
        self.assertIn("3", command)
        self.assertNotIn("1.5", command)


if __name__ == "__main__":
    unittest.main()
