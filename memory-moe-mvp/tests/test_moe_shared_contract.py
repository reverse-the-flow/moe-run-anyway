import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import llama_runtime_probe  # noqa: E402
import llama_sidecar  # noqa: E402
import memory_moe  # noqa: E402
import moe_forward_probe  # noqa: E402
import moe_shared_contract  # noqa: E402


class FakeHookHandle:
    def __init__(self, hooks, hook) -> None:
        self._hooks = hooks
        self._hook = hook

    def remove(self) -> None:
        if self._hook in self._hooks:
            self._hooks.remove(self._hook)


class FakeModule:
    def __init__(self, output=None) -> None:
        self._hooks = []
        self.output = output

    def register_forward_hook(self, hook):
        self._hooks.append(hook)
        return FakeHookHandle(self._hooks, hook)

    def forward(self):
        for hook in list(self._hooks):
            hook(self, tuple(), self.output)
        return self.output


class FakeModel:
    def __init__(self, modules):
        self._modules = modules

    def named_modules(self):
        yield "", self
        for name, module in self._modules:
            yield name, module


def read_jsonl(path: Path):
    import json

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


class SharedContractTests(unittest.TestCase):
    def assert_has_contract_fields(self, payload) -> None:
        self.assertEqual(payload["contract_version"], moe_shared_contract.CONTRACT_VERSION)
        self.assertEqual(
            set(payload["shared_contract"].keys()),
            set(moe_shared_contract.SHARED_CONTRACT_FIELDS),
        )

    def test_sidecar_and_runtime_reports_expose_shared_contract_templates(self) -> None:
        sidecar_config = llama_sidecar.SidecarConfig(
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
        sidecar_snapshot = llama_sidecar.RunAccumulator(run_id="run-1", config=sidecar_config).snapshot()
        runtime_config = llama_runtime_probe.RuntimeProbeConfig(
            base_url="http://127.0.0.1:8080",
            output_dir=ROOT,
            label="runtime-test",
            request_timeout_seconds=60.0,
            model="mixtral",
        )
        runtime_snapshot = llama_runtime_probe.RuntimeProbeAccumulator(
            run_id="run-2",
            config=runtime_config,
        ).snapshot()

        for snapshot in [sidecar_snapshot, runtime_snapshot]:
            template = snapshot["shared_contract_template"]
            self.assertEqual(template["contract_version"], moe_shared_contract.CONTRACT_VERSION)
            self.assertEqual(
                set(template["required_fields"]),
                set(moe_shared_contract.SHARED_CONTRACT_FIELDS),
            )

    def test_forward_probe_and_simulator_events_populate_shared_contract(self) -> None:
        scenario = memory_moe.load_scenario(ROOT / "data" / "toy_workload.json")
        state = memory_moe.RuntimeState(scenario)
        simulate_event = memory_moe.simulate_request(
            scenario=scenario,
            state=state,
            profile=memory_moe.default_profiles()["baseline"],
            request=scenario.requests[0],
            step_index=1,
        )
        self.assert_has_contract_fields(simulate_event)

        router = FakeModule(
            output={
                "expert_indices": [[1, 3], [1, 2]],
                "expert_weights": [[0.7, 0.3], [0.8, 0.2]],
            }
        )
        model = FakeModel([("layers.3.moe.router", router)])
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = moe_forward_probe.ForwardHookMoEProbe(
                config=moe_forward_probe.ForwardHookProbeConfig(
                    output_dir=Path(temp_dir),
                    label="shared-contract-test",
                    window_size_events=4,
                )
            )
            probe.attach(model)
            with probe.span({"family_id": "code_python", "probe_id": "case-1"}):
                router.forward()
            probe.close()

            forward_event = read_jsonl(probe.router_events_path)[0]
            self.assert_has_contract_fields(forward_event)
            self.assertEqual(forward_event["shared_contract"]["prompt_family"], "code_python")


if __name__ == "__main__":
    unittest.main()
