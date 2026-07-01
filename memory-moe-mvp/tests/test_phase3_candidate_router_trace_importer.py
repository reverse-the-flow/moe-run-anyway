import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "import_phase3_candidate_router_trace.py"
FIXTURE_TRACE = ROOT / "memory-moe-mvp" / "data" / "llama_cpp_router_trace.fixture.jsonl"
SPEC = importlib.util.spec_from_file_location("import_phase3_candidate_router_trace", SCRIPT_PATH)
importer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = importer
SPEC.loader.exec_module(importer)


def args(**overrides):
    values = {
        "request_path": None,
        "source_trace_path": None,
        "bundle_path": None,
        "prompt_set_path": None,
        "candidate_trace_path": None,
        "trace_receipt_path": None,
        "capture_host": "phase3-test-host",
        "captured_at": "2026-06-28T21:00:00Z",
        "operator_notes": "unit test import",
        "require_kind": None,
        "approved_router_trace_capture": True,
        "approved_runtime_prompt_traffic": True,
        "overwrite": False,
        "dry_run": False,
        "artifact_json": False,
        "json": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def trace_text_with_prompt_identity(repeats: int = 1) -> str:
    lines: list[str] = []
    capture_sequence = 0
    fixture_events = [json.loads(line) for line in FIXTURE_TRACE.read_text(encoding="utf-8").splitlines() if line.strip()]
    for repeat in range(repeats):
        for event in fixture_events:
            enriched = dict(event)
            enriched.update(
                {
                    "prompt_id": "fixture-import-prompt",
                    "prompt_group_id": "fixture-import-group",
                    "repeat": repeat,
                    "repeat_index": repeat,
                    "capture_run_id": "import-test-capture-run",
                    "capture_sequence": capture_sequence,
                }
            )
            lines.append(json.dumps(enriched, sort_keys=True))
            capture_sequence += 1
    return "\n".join(lines) + "\n"


def write_source_trace_with_prompt_identity(path: Path) -> None:
    path.write_text(trace_text_with_prompt_identity(), encoding="utf-8")


def fixture_paths(temp_path: Path) -> dict[str, Path]:
    bundle = temp_path / "phase3_bundle.json"
    prompt = temp_path / "phase3_bundle.prompt-set.json"
    request = temp_path / "phase3_bundle.runtime-capture-request.json"
    source_trace = temp_path / "source-router-events.jsonl"
    candidate_trace = temp_path / "phase3_bundle-policy-candidate" / "candidate-router-events.jsonl"
    receipt = candidate_trace.with_name("candidate-router-events.capture-receipt.json")
    return {
        "bundle": bundle,
        "prompt": prompt,
        "request": request,
        "source_trace": source_trace,
        "candidate_trace": candidate_trace,
        "receipt": receipt,
    }


def write_fixture_request(paths: dict[str, Path]) -> None:
    write_json(
        paths["bundle"],
        {
            "schema_version": "fixture",
            "name": "fixture bundle",
            "backend_family": "llama_cpp",
            "model_id": "fixture-mixtral.gguf",
            "prompt_family": "fixture-router-trace",
        },
    )
    write_json(
        paths["prompt"],
        {
            "schema_version": "moe-phase3-prompt-set-v1",
            "backend_family": "llama_cpp",
            "model_id": "fixture-mixtral.gguf",
            "prompt_family": "fixture-router-trace",
            "prompts": [],
        },
    )
    write_json(
        paths["request"],
        {
            "schema_version": "moe-phase3-runtime-capture-request-v1",
            "bundle_path": str(paths["bundle"]),
            "approvals": {
                "router_trace_capture_approved": False,
                "runtime_prompt_traffic_approved": False,
            },
            "requested_artifacts": [
                {
                    "id": "candidate_router_trace",
                    "path": str(paths["candidate_trace"]),
                    "source": {
                        "prompt_set_path": str(paths["prompt"]),
                        "capture_receipt_path": str(paths["receipt"]),
                    },
                }
            ],
        },
    )
    write_source_trace_with_prompt_identity(paths["source_trace"])


class Phase3CandidateRouterTraceImporterTests(unittest.TestCase):
    def test_import_writes_canonical_trace_and_ready_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = fixture_paths(Path(temp_dir))
            write_fixture_request(paths)

            status, summary, receipt = importer.plan_import(
                args(
                    request_path=paths["request"],
                    source_trace_path=paths["source_trace"],
                )
            )

            self.assertEqual(status, 0, summary["errors"])
            self.assertTrue(summary["valid"], summary["errors"])
            self.assertTrue(summary["write_performed"])
            self.assertEqual(paths["candidate_trace"].read_text(encoding="utf-8"), paths["source_trace"].read_text(encoding="utf-8"))
            self.assertTrue(summary["trace_validation"]["prompt_identity_ready"])
            written_receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
            self.assertIsNotNone(receipt)
            self.assertTrue(written_receipt["receipt_ready"])
            self.assertTrue(written_receipt["router_trace_capture_approved"])
            self.assertTrue(written_receipt["runtime_prompt_traffic_approved"])
            self.assertEqual(written_receipt["capture_host"], "phase3-test-host")
            self.assertEqual(written_receipt["model_id"], "fixture-mixtral.gguf")
            self.assertEqual(written_receipt["operator_warnings"], [
                "source_request_router_trace_capture_approval_false",
                "source_request_runtime_prompt_traffic_approval_false",
            ])
            validation = importer.build_phase3_trace_receipt.validate_trace_receipt_artifact(
                written_receipt,
                bundle_path=paths["bundle"],
                prompt_set_path=paths["prompt"],
                candidate_trace_path=paths["candidate_trace"],
                expected_model_id="fixture-mixtral.gguf",
                expected_backend_family="llama_cpp",
                expected_prompt_family="fixture-router-trace",
            )
            self.assertTrue(validation["shape_valid"], validation["errors"])
            self.assertTrue(validation["receipt_ready"], validation["errors"])

    def test_missing_approvals_refuses_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = fixture_paths(Path(temp_dir))
            write_fixture_request(paths)

            status, summary, _ = importer.plan_import(
                args(
                    request_path=paths["request"],
                    source_trace_path=paths["source_trace"],
                    approved_router_trace_capture=False,
                    approved_runtime_prompt_traffic=False,
                )
            )

            self.assertEqual(status, 2)
            self.assertFalse(summary["valid"])
            self.assertTrue(any("--approved-router-trace-capture" in error for error in summary["errors"]))
            self.assertTrue(any("--approved-runtime-prompt-traffic" in error for error in summary["errors"]))
            self.assertFalse(paths["candidate_trace"].exists())
            self.assertFalse(paths["receipt"].exists())

    def test_missing_prompt_identity_refuses_import(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = fixture_paths(Path(temp_dir))
            write_fixture_request(paths)
            shutil.copyfile(FIXTURE_TRACE, paths["source_trace"])

            status, summary, _ = importer.plan_import(
                args(
                    request_path=paths["request"],
                    source_trace_path=paths["source_trace"],
                )
            )

            self.assertEqual(status, 2)
            self.assertFalse(summary["valid"])
            self.assertFalse(summary["trace_validation"]["prompt_identity_ready"])
            self.assertTrue(any("prompt_identity_metadata_missing" in error for error in summary["errors"]))
            self.assertFalse(paths["candidate_trace"].exists())
            self.assertFalse(paths["receipt"].exists())

    def test_invalid_trace_refuses_import(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = fixture_paths(Path(temp_dir))
            write_fixture_request(paths)
            first_line = paths["source_trace"].read_text(encoding="utf-8").splitlines()[0]
            event = json.loads(first_line)
            event["contract_version"] = "wrong"
            paths["source_trace"].write_text(json.dumps(event) + "\n", encoding="utf-8")

            status, summary, _ = importer.plan_import(
                args(
                    request_path=paths["request"],
                    source_trace_path=paths["source_trace"],
                )
            )

            self.assertEqual(status, 2)
            self.assertFalse(summary["valid"])
            self.assertTrue(any("contract_version" in error for error in summary["errors"]))
            self.assertFalse(paths["candidate_trace"].exists())
            self.assertFalse(paths["receipt"].exists())

    def test_dry_run_validates_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = fixture_paths(Path(temp_dir))
            write_fixture_request(paths)

            status, summary, receipt = importer.plan_import(
                args(
                    request_path=paths["request"],
                    source_trace_path=paths["source_trace"],
                    dry_run=True,
                )
            )

            self.assertEqual(status, 0, summary["errors"])
            self.assertTrue(summary["valid"], summary["errors"])
            self.assertFalse(summary["write_performed"])
            self.assertFalse(paths["candidate_trace"].exists())
            self.assertFalse(paths["receipt"].exists())
            self.assertIsNotNone(receipt)
            self.assertTrue(summary["trace_validation"]["prompt_identity_ready"])
            self.assertTrue(summary["receipt_validation"]["receipt_ready"])


if __name__ == "__main__":
    unittest.main()
