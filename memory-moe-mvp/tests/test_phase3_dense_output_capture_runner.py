from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "run_phase3_dense_output_capture.py"
SPEC = importlib.util.spec_from_file_location("run_phase3_dense_output_capture", SCRIPT_PATH)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)

import build_phase3_output_summary


class FakeOpenAIHandler(BaseHTTPRequestHandler):
    requests_seen: list[dict[str, object]] = []

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.path in {"/v1/models", "/models"}:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"data": []}')
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        FakeOpenAIHandler.requests_seen.append(payload)
        prompt = payload["messages"][0]["content"]
        response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": f"captured dense answer for: {prompt}"},
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5},
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def prompt_set() -> dict[str, object]:
    return {
        "schema_version": "moe-phase3-prompt-set-v1",
        "model_id": "fixture-model-id",
        "backend_family": "llama_cpp",
        "prompt_family": "fixture-phase3",
        "prompts": [
            {
                "prompt_id": "fixture-prompt-r1",
                "group_id": "fixture",
                "repeat": 1,
                "prompt": "Explain dense fallback briefly.",
            },
            {
                "prompt_id": "fixture-prompt-r2",
                "group_id": "fixture",
                "repeat": 2,
                "prompt": "Explain dense fallback briefly.",
            },
        ],
    }


def runtime_request() -> dict[str, object]:
    return {
        "schema_version": "moe-phase3-runtime-capture-request-v1",
        "name": "fixture request",
        "backend_family": "llama_cpp",
        "prompt_family": "fixture-phase3",
    }


class FakeServer:
    def __enter__(self) -> str:
        FakeOpenAIHandler.requests_seen = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAIHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class Phase3DenseOutputCaptureRunnerTests(unittest.TestCase):
    def test_approved_capture_writes_validator_ready_dense_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, FakeServer() as base_url:
            root = Path(temp_dir)
            request_path = root / "fixture.runtime-capture-request.json"
            prompt_path = root / "fixture.prompt-set.json"
            output_path = root / "dense-output-summary.json"
            write_json(request_path, runtime_request())
            write_json(prompt_path, prompt_set())
            args = runner.build_arg_parser().parse_args(
                [
                    str(request_path),
                    "--prompt-set-path",
                    str(prompt_path),
                    "--artifact-output",
                    str(output_path),
                    "--receipt-output",
                    str(output_path),
                    "--base-url",
                    base_url,
                    "--model",
                    "fixture-model",
                    "--approved-runtime-prompt-traffic",
                    "--approved-dense-output-capture",
                ]
            )

            status, summary = runner.run_capture(args)

            self.assertEqual(status, 0, summary)
            self.assertTrue(summary["ok"], summary)
            self.assertTrue(summary["prompt_traffic_sent"])
            self.assertTrue(summary["write_performed"])
            self.assertEqual(len(FakeOpenAIHandler.requests_seen), 2)
            artifact = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(artifact["output_ready"])
            self.assertTrue(artifact["capture_receipt"]["receipt_ready"])
            self.assertEqual(artifact["capture_receipt"]["runtime_model"], "fixture-model")
            self.assertEqual({row["prompt_id"] for row in artifact["outputs"]}, {"fixture-prompt-r1", "fixture-prompt-r2"})
            validation = build_phase3_output_summary.build_summary(
                artifact,
                prompt_set_path=prompt_path,
                output_path=output_path,
                expected_label="dense",
            )
            self.assertTrue(validation["valid"], validation["errors"])
            self.assertTrue(validation["summary_ready"], validation["errors"])

    def test_missing_approval_refuses_prompt_traffic_and_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            request_path = root / "fixture.runtime-capture-request.json"
            prompt_path = root / "fixture.prompt-set.json"
            output_path = root / "dense-output-summary.json"
            write_json(request_path, runtime_request())
            write_json(prompt_path, prompt_set())
            args = runner.build_arg_parser().parse_args(
                [
                    str(request_path),
                    "--prompt-set-path",
                    str(prompt_path),
                    "--artifact-output",
                    str(output_path),
                    "--receipt-output",
                    str(output_path),
                    "--base-url",
                    "http://127.0.0.1:9",
                    "--model",
                    "fixture-model",
                    "--skip-preflight",
                ]
            )

            status, summary = runner.run_capture(args)

            self.assertEqual(status, 2)
            self.assertFalse(summary["prompt_traffic_sent"])
            self.assertFalse(summary["write_performed"])
            self.assertFalse(output_path.exists())
            self.assertIn("approved-runtime-prompt-traffic is required", summary["errors"])
            self.assertIn("approved-dense-output-capture is required", summary["errors"])

    def test_embedded_receipt_requires_same_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            request_path = root / "fixture.runtime-capture-request.json"
            prompt_path = root / "fixture.prompt-set.json"
            output_path = root / "dense-output-summary.json"
            receipt_path = root / "separate-receipt.json"
            write_json(request_path, runtime_request())
            write_json(prompt_path, prompt_set())
            args = runner.build_arg_parser().parse_args(
                [
                    str(request_path),
                    "--prompt-set-path",
                    str(prompt_path),
                    "--artifact-output",
                    str(output_path),
                    "--receipt-output",
                    str(receipt_path),
                    "--base-url",
                    "http://127.0.0.1:9",
                    "--model",
                    "fixture-model",
                    "--approved-runtime-prompt-traffic",
                    "--approved-dense-output-capture",
                    "--skip-preflight",
                ]
            )

            status, summary = runner.run_capture(args)

            self.assertEqual(status, 2)
            self.assertFalse(summary["prompt_traffic_sent"])
            self.assertIn("receipt_output_path must equal artifact_output_path", "\n".join(summary["errors"]))


if __name__ == "__main__":
    unittest.main()