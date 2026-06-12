import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import memory_moe  # noqa: E402


class MemoryMoETests(unittest.TestCase):
    def setUp(self) -> None:
        scenario_path = ROOT / "data" / "toy_workload.json"
        self.scenario = memory_moe.load_scenario(scenario_path)

    def test_reuse_bias_prefers_recent_resident_expert(self) -> None:
        baseline = memory_moe.default_profiles()["baseline"]
        reuse_bias = memory_moe.default_profiles()["reuse_bias"]
        state = memory_moe.RuntimeState(self.scenario)
        state.load("e0", step_index=1)
        state.mark_used("e0", step_index=1)

        request = memory_moe.RequestCase(
            request_id="routing-check",
            prompt_tokens=4000,
            router_scores={
                "e0": 0.87,
                "e1": 0.2,
                "e2": 0.2,
                "e3": 0.2,
                "e4": 0.91,
                "e5": 0.1
            },
            quality_scores={
                "e0": 0.85,
                "e1": 0.2,
                "e2": 0.2,
                "e3": 0.2,
                "e4": 0.94,
                "e5": 0.1
            },
            note=None
        )

        baseline_top = memory_moe.rank_experts(self.scenario, state, request, baseline, step_index=2)[0].expert_id
        reuse_top = memory_moe.rank_experts(self.scenario, state, request, reuse_bias, step_index=2)[0].expert_id

        self.assertEqual(baseline_top, "e4")
        self.assertEqual(reuse_top, "e0")

    def test_lru_eviction_frees_budget_for_selected_expert(self) -> None:
        state = memory_moe.RuntimeState(self.scenario)
        state.load("e0", step_index=1)
        state.load("e1", step_index=1)
        state.load("e2", step_index=2)
        state.load("e4", step_index=2)
        state.mark_used("e0", step_index=1)
        state.mark_used("e1", step_index=2)
        state.mark_used("e2", step_index=3)
        state.mark_used("e4", step_index=4)

        loads, evictions, dropped, active = memory_moe.ensure_selected_experts(
            scenario=self.scenario,
            state=state,
            selected_experts=["e3", "e5"],
            step_index=4
        )

        self.assertEqual(active, ["e3", "e5"])
        self.assertEqual(dropped, [])
        self.assertEqual([item["expert_id"] for item in loads], ["e3", "e5"])
        self.assertEqual([item["expert_id"] for item in evictions], ["e0"])

    def test_context_reserve_improves_context_retention(self) -> None:
        baseline = memory_moe.default_profiles()["baseline"]
        context_reserve = memory_moe.default_profiles()["context_reserve"]
        request = self.scenario.requests[4]

        baseline_state = memory_moe.RuntimeState(self.scenario)
        context_state = memory_moe.RuntimeState(self.scenario)
        for state in [baseline_state, context_state]:
            state.load("e0", step_index=1)
            state.load("e1", step_index=1)
            state.load("e2", step_index=2)
            state.load("e3", step_index=2)
            state.mark_used("e0", step_index=1)
            state.mark_used("e1", step_index=1)
            state.mark_used("e2", step_index=2)
            state.mark_used("e3", step_index=2)

        baseline_result = memory_moe.simulate_request(
            self.scenario,
            baseline_state,
            baseline,
            request,
            step_index=5
        )
        context_result = memory_moe.simulate_request(
            self.scenario,
            context_state,
            context_reserve,
            request,
            step_index=5
        )

        self.assertLess(
            baseline_result["quality"]["context_retention"],
            context_result["quality"]["context_retention"]
        )
        self.assertGreater(
            len(context_result["evictions"]),
            len(baseline_result["evictions"])
        )


if __name__ == "__main__":
    unittest.main()
