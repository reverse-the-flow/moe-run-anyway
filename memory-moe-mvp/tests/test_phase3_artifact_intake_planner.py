import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_phase3_artifact_intake.py"
SPEC = importlib.util.spec_from_file_location("plan_phase3_artifact_intake", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)

INVENTORY_FIXTURE = ROOT / "memory-moe-mvp" / "data" / "expert_inventory_manifest.fixture.json"
MODEL_ID = "local-mixtral.gguf"


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def trace_events(model_id: str = MODEL_ID):
    base = {
        "backend_family": "llama_cpp",
        "contract_version": "memory-moe-bridge-v1",
        "event_type": "llama_cpp_moe_router_tensor",
        "layer_id": 0,
        "model": model_id,
        "source": "llama_cpp_eval_callback",
        "timestamp": "2026-06-25T00:00:00Z",
        "values_truncated": False,
    }
    return [
        {
            **base,
            "ggml_type": "i32",
            "shape": [1, 4, 1, 1],
            "tensor_kind": "selected_experts",
            "tensor_name": "ffn_moe_topk-0",
            "value_count": 4,
            "values": [0, 1, 0, 1],
        },
        {
            **base,
            "ggml_type": "f32",
            "shape": [1, 4, 1, 1],
            "tensor_kind": "selected_weights",
            "tensor_name": "ffn_moe_weights-0",
            "value_count": 4,
            "values": [1.0, 0.99, 1.0, 0.99],
        },
        {
            **base,
            "ggml_type": "f32",
            "shape": [1, 4, 1, 1],
            "tensor_kind": "selected_weights_norm",
            "tensor_name": "ffn_moe_weights_norm-0",
            "value_count": 4,
            "values": [1.0, 0.99, 1.0, 0.99],
        },
    ]


def inventory_manifest():
    manifest = json.loads(INVENTORY_FIXTURE.read_text(encoding="utf-8"))
    manifest["name"] = "Local Mixtral Expert Inventory"
    manifest["model_id"] = MODEL_ID
    manifest["inventory_scope"] = "scanner_derived_metadata"
    manifest["source_files"] = [
        {
            "path": MODEL_ID,
            "source_format": "gguf",
            "byte_length": 6291456,
            "note": "Metadata-only scan from local GGUF header.",
        }
    ]
    manifest["expected"] = {
        "layer_ids": [0],
        "required_components": ["gate_proj", "up_proj", "down_proj"],
        "expert_count": 2,
        "component_count": 6,
        "total_estimated_residency_bytes": 6291456,
        "routing_top_k": 1,
    }
    entries = []
    byte_offset = 0
    for entry in manifest["entries"]:
        if entry["layer_id"] != 0 or entry["expert_id"] not in {0, 1}:
            continue
        copied = copy.deepcopy(entry)
        copied["source_file"] = MODEL_ID
        copied["byte_offset"] = byte_offset
        copied["byte_length"] = 1048576
        copied["stride_bytes"] = 1048576
        copied["estimated_residency_bytes"] = 1048576
        copied["coverage_status"] = "complete"
        byte_offset += 1048576
        entries.append(copied)
    manifest["entries"] = entries
    manifest["safety_contract"] = ["Inventory was generated from metadata only."]
    manifest["next_actions"] = ["Pair with semantic router trace."]
    return manifest


def fallback_artifact():
    return {
        "schema_version": "moe-dense-fallback-comparison-v1",
        "model_id": MODEL_ID,
        "prompt_family": "phase3-intake-test",
        "managed_policy_id": "preload_shortlist",
        "managed_artifact": "memory-moe-mvp/phase3-real-evidence/managed-output.json",
        "dense_artifact": "memory-moe-mvp/phase3-real-evidence/dense-output.json",
        "builder": {
            "schema_version": "moe-dense-fallback-comparison-builder-v1",
            "mode": "paired_output_summary",
            "input_receipts": {
                "managed_capture_receipt_ready": True,
                "dense_capture_receipt_ready": True,
                "receipt_pair_consistent": True,
                "receipt_gate": "required_before_write",
            },
        },
        "comparisons": [
            {
                "prompt_id": "case-001",
                "managed_output_present": True,
                "dense_output_present": True,
                "quality_delta_label": "same",
            }
        ],
    }


class Phase3ArtifactIntakePlannerTests(unittest.TestCase):
    def test_missing_root_is_valid_empty_intake(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = planner.build_intake_summary(root=Path(temp_dir) / "missing")

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["trace_candidate_count"], 0)
        self.assertEqual(summary["phase3_handoff_ready_count"], 0)
        self.assertIn("planner does not send prompt traffic", summary["safety_contract"])

    def test_trace_with_inventory_summary_still_needs_manifest_and_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trace_path = root / "run" / "router-events.jsonl"
            trace_path.parent.mkdir()
            inventory_summary_path = root / "pc-ollama-inventory.jsonl"
            write_jsonl(trace_path, trace_events())
            inventory_summary_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in
                [
                    {
                        "model": "dolphin-mixtral:8x7b",
                        "blob": MODEL_ID,
                        "architecture": "llama",
                        "expert_count": 8,
                        "expert_used_count": 2,
                        "moe_layer_count": 32,
                    }
                ]
                ),
                encoding="utf-16",
            )

            summary = planner.build_intake_summary(
                root=root,
                inventory_summary_path=inventory_summary_path,
            )

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["trace_candidate_count"], 1)
        self.assertEqual(summary["trace_valid_count"], 1)
        self.assertEqual(summary["inventory_summary_match_count"], 1)
        candidate = summary["candidates"][0]
        self.assertFalse(candidate["phase3_handoff_ready"])
        self.assertIsNotNone(candidate["inventory_summary_match"])
        self.assertIn("scanner_inventory_manifest_for_trace_model", candidate["missing_for_phase3_handoff"])
        self.assertIn("dense_fallback_comparison_artifact", candidate["missing_for_phase3_handoff"])

    def test_manifest_and_fallback_make_candidate_handoff_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trace_path = root / "run" / "router-events.jsonl"
            trace_path.parent.mkdir()
            inventory_path = root / "expert-inventory.json"
            fallback_path = root / "fallback-comparison.json"
            write_jsonl(trace_path, trace_events())
            write_json(inventory_path, inventory_manifest())
            write_json(fallback_path, fallback_artifact())

            summary = planner.build_intake_summary(
                root=root,
                inventory_manifest_path=inventory_path,
                fallback_artifact_path=fallback_path,
            )

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["phase3_handoff_ready_count"], 1)
        candidate = summary["candidates"][0]
        self.assertTrue(candidate["phase3_handoff_ready"])
        self.assertEqual(candidate["missing_for_phase3_handoff"], [])
        self.assertTrue(candidate["inventory_manifest"]["real_model_pair_ready"])
        self.assertTrue(candidate["fallback"]["comparison_ready"])
        command_classes = {command["command_class"] for command in candidate["commands"]}
        self.assertIn("trace_contract_validator", command_classes)
        self.assertIn("phase3_bundle_builder", command_classes)

    def test_docker_ollama_volume_command_maps_trace_blob_to_scanner_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trace_path = root / "run" / "router-events.jsonl"
            trace_path.parent.mkdir()
            model_id = "/root/.ollama/models/blobs/sha256-abc123"
            write_jsonl(trace_path, trace_events(model_id=model_id))

            summary = planner.build_intake_summary(
                root=root,
                docker_ollama_volume="open-webui_ollama",
            )

        self.assertTrue(summary["valid"], summary["errors"])
        candidate = summary["candidates"][0]
        docker_commands = [
            command
            for command in candidate["commands"]
            if command["command_class"] == "docker_ollama_scanner_inventory_manifest_builder"
        ]
        self.assertEqual(len(docker_commands), 1)
        command = docker_commands[0]["command"]
        self.assertIn("docker", command)
        self.assertIn("open-webui_ollama:/ollama:ro", command)
        self.assertIn("scripts/scan_gguf_expert_inventory.py", command)
        self.assertIn("/ollama/models/blobs/sha256-abc123", command)
        self.assertIn("--model-id", command)
        self.assertIn(model_id, command)
        self.assertIn("/repo/memory-moe-mvp/phase3-real-evidence/sha256-abc123.expert_inventory.json", command)
        self.assertIn("planner only emits this command; it does not run Docker", docker_commands[0]["notes"])


if __name__ == "__main__":
    unittest.main()
