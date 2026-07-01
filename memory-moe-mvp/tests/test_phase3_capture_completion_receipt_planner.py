import argparse
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "plan_phase3_capture_completion_receipt.py"
SPEC = importlib.util.spec_from_file_location("plan_phase3_capture_completion_receipt", SCRIPT_PATH)
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


def args(**overrides):
    values = {
        "receipt": None,
        "work_order": None,
        "output": None,
        "json": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def filled_receipt(template: dict) -> dict:
    receipt = copy.deepcopy(template)
    receipt["receipt_complete"] = True
    receipt["ready_for_capture_result_intake"] = True
    for row in receipt["capture_receipts"]:
        row["capture_complete"] = True
        row["receipt_filled"] = True
        row["validator_passed"] = True
        row["ready_for_intake"] = True
        row["observed_at"] = "2026-06-30T21:00:00Z"
        row["operator_notes"] = "test receipt filled after approved capture"
    return receipt


class Phase3CaptureCompletionReceiptPlannerTests(unittest.TestCase):
    def test_template_is_valid_but_not_ready_for_intake(self) -> None:
        work_order = planner.fixture_work_order()
        receipt = planner.completion_receipt_template_from_work_order(work_order)

        summary = planner.validate_completion_receipt(receipt, work_order)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertTrue(summary["template_ready"])
        self.assertFalse(summary["receipt_complete"])
        self.assertFalse(summary["ready_for_capture_result_intake"])
        self.assertEqual(summary["row_count"], 2)
        self.assertEqual(summary["complete_row_count"], 0)

    def test_filled_receipt_is_ready_for_intake(self) -> None:
        work_order = planner.fixture_work_order()
        receipt = filled_receipt(planner.completion_receipt_template_from_work_order(work_order))

        summary = planner.validate_completion_receipt(receipt, work_order)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertTrue(summary["receipt_complete"])
        self.assertTrue(summary["ready_for_capture_result_intake"])
        self.assertEqual(summary["complete_row_count"], 2)
        self.assertEqual(summary["validator_passed_count"], 2)
        self.assertEqual(summary["missing_item_count"], 0)

    def test_partial_receipt_is_valid_but_not_ready(self) -> None:
        work_order = planner.fixture_work_order()
        receipt = planner.completion_receipt_template_from_work_order(work_order)
        receipt["capture_receipts"][0]["capture_complete"] = True
        receipt["capture_receipts"][0]["receipt_filled"] = True
        receipt["capture_receipts"][0]["observed_at"] = "2026-06-30T21:00:00Z"

        summary = planner.validate_completion_receipt(receipt, work_order)

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertFalse(summary["receipt_complete"])
        self.assertFalse(summary["ready_for_capture_result_intake"])
        self.assertEqual(summary["capture_complete_count"], 1)
        self.assertEqual(summary["receipt_filled_count"], 1)
        self.assertEqual(summary["validator_passed_count"], 0)

    def test_work_order_task_drift_is_invalid(self) -> None:
        work_order = planner.fixture_work_order()
        receipt = filled_receipt(planner.completion_receipt_template_from_work_order(work_order))
        receipt["capture_receipts"][0]["artifact_path"] = "memory-moe-mvp/phase3-real-evidence/fixture/wrong-artifact.jsonl"

        summary = planner.validate_completion_receipt(receipt, work_order)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["ready_for_capture_result_intake"])
        self.assertIn("completion_receipt_work_order_task_mismatch", summary["errors"])

    def test_claimed_ready_without_validator_pass_is_invalid(self) -> None:
        work_order = planner.fixture_work_order()
        receipt = filled_receipt(planner.completion_receipt_template_from_work_order(work_order))
        receipt["capture_receipts"][0]["validator_passed"] = False

        summary = planner.validate_completion_receipt(receipt, work_order)

        self.assertFalse(summary["valid"])
        self.assertFalse(summary["receipt_complete"])
        self.assertFalse(summary["ready_for_capture_result_intake"])
        self.assertIn("completion_receipt_row_1_ready_without_prereqs", summary["errors"])
        self.assertIn("completion_receipt_complete_without_all_rows_ready", summary["errors"])

    def test_cli_validates_bound_files(self) -> None:
        work_order = planner.fixture_work_order()
        receipt = filled_receipt(planner.completion_receipt_template_from_work_order(work_order))
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            work_order_path = temp_path / "work-order.json"
            receipt_path = temp_path / "completion-receipt.json"
            output_path = temp_path / "summary.json"
            work_order_path.write_text(json.dumps(work_order), encoding="utf-8")
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

            status, summary, error = planner.plan_validation(
                args(receipt=receipt_path, work_order=work_order_path, output=output_path)
            )

            self.assertEqual(status, 0)
            self.assertIsNone(error)
            self.assertTrue(summary["ready_for_capture_result_intake"])
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(saved["valid"])
            self.assertTrue(saved["ready_for_capture_result_intake"])


if __name__ == "__main__":
    unittest.main()