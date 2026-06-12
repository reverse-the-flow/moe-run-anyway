# Memory-Aware MoE MVP

This package is a focused probe and replay harness for memory-aware Mixture-of-Experts runtime experiments. It treats the current problem as measurement and controllability first: learn what a backend exposes, score simple residency policies offline, and avoid claiming real expert paging before the runtime supports it.

## Current Scope

Implemented local surfaces:

- [memory_moe.py](memory_moe.py): synthetic logical-residency simulator.
- [moe_controller_demo.py](moe_controller_demo.py): replay/advisor/controller demo over synthetic traces or forward-probe window summaries.
- [llama_sidecar.py](llama_sidecar.py): passive sidecar/proxy around an OpenAI-compatible or `llama-server` upstream.
- [llama_runtime_probe.py](llama_runtime_probe.py): active stock `llama.cpp` runtime probe using `/metrics`, `/slots`, `/props`, response timings, and optional log growth.
- [moe_forward_probe.py](moe_forward_probe.py): forward-hook probe for hookable PyTorch-style MoE runtimes.
- [run_forward_probe_demo.py](run_forward_probe_demo.py): no-model synthetic hook driver.
- [model_target_registry.py](model_target_registry.py): dependency-free validator for [data/model_target_registry.json](data/model_target_registry.json).

Not implemented yet:

- live expert loading, eviction, pinning, or paging
- a runtime patch that emits semantic expert ids from stock `llama.cpp`
- a bundled live model server, model weights, or package installer
- GPU-heavy or Docker-required validation in the local readiness path

## Shared Contract

Probe outputs are aligned around [moe_shared_contract.py](moe_shared_contract.py), currently `memory-moe-bridge-v1`. Important fields include:

- `probe_tier`
- `backend_family`
- `prompt_family`
- `window_size_tokens`
- `policy_name`
- `resident_budget_fraction`
- `candidate_set_size`
- `fallback_used`
- `warm_hit_rate`
- `miss_rate`
- `churn`
- `eviction_regret`
- `context_retention`
- `dense_baseline_delta`

The contract is meant to keep simulator, sidecar, runtime, hook, and controller artifacts comparable without pretending they expose the same depth of model internals.

## Probe Tiers

### Tier 0: Local Replay And Shape Checks

Run without a model server:

```bash
python3 memory_moe.py plan
python3 memory_moe.py run --output-dir sim-runs
python3 moe_controller_demo.py --output-dir controller-runs --label local-replay
python3 run_forward_probe_demo.py \
  --output-dir forward-probe-runs \
  --suite-path data/mixtral_probe_prompts.json \
  --max-prompts 2 \
  --window-size-events 2 \
  --label synthetic-hook-smoke
```

### Tier 1: Passive Sidecar

Use when a backend is already running and the goal is non-invasive request-boundary telemetry:

```bash
python3 llama_sidecar.py \
  --listen-port 8091 \
  --upstream-base-url http://127.0.0.1:18080 \
  --output-dir sidecar-runs \
  --label mixtral-sidecar \
  --capture-upstream-observability
```

Point the client at `http://127.0.0.1:8091` instead of the upstream server. This writes `manifest.json`, `events.jsonl`, and `summary.json`.

### Tier 2A: Stock llama.cpp Runtime Probe

Start `llama-server` yourself with observability enabled:

```bash
llama-server \
  --metrics \
  --slots \
  --props \
  --perf \
  --log-file /path/to/llama-server.log \
  --log-prefix \
  --log-timestamps \
  --verbosity 4 \
  -m /path/to/model.gguf
```

Then run:

```bash
python3 llama_runtime_probe.py \
  --base-url http://127.0.0.1:18080 \
  --output-dir runtime-probe-runs \
  --label mixtral-runtime \
  --model dolphin-mixtral \
  --suite-path data/mixtral_probe_prompts.json \
  --max-prompts 4 \
  --repeats 2 \
  --log-file-path /path/to/llama-server.log
```

This path still does not expose semantic expert ids. It can correlate request timing, server metrics, slot state, properties, and log growth.

### Tier 2B: Forward-Hook Semantic Probe

Use this only when the model runtime still exposes Python modules and `register_forward_hook()` works:

```python
from pathlib import Path
import moe_forward_probe

config = moe_forward_probe.ForwardHookProbeConfig(
    output_dir=Path("forward-probe-runs"),
    label="local-hookable-moe",
)
probe = moe_forward_probe.ForwardHookMoEProbe(config=config)
hook_count = probe.attach(model)

with probe.span({"prompt_id": "case-001", "family_id": "code_python"}):
    _ = model(**inputs)

probe.close()
```

The forward-hook path can capture routed expert ids, expert weights or probabilities, routing entropy, per-layer hit counts, and window summaries when the backend exposes those values.

## Fixtures And Docs

- [data/mixtral_probe_prompts.json](data/mixtral_probe_prompts.json): prompt corpus for Mixtral-style routing and runtime probes.
- [data/synthetic_controller_trace.json](data/synthetic_controller_trace.json): replay fixture for controller/advisor logic.
- [data/toy_workload.json](data/toy_workload.json): simulator workload.
- [docs/model-target-test-plan.md](docs/model-target-test-plan.md): target classes and live-test command gates.
- [docs/probe-observability-notes.md](docs/probe-observability-notes.md): probe surfaces and `llama.cpp` observability knobs.
- [docs/controller-architecture.md](docs/controller-architecture.md): controller pattern and failure-mode guardrails.

Historical probe result fixtures remain under `probe-results/`.

## Local Checks

From the repository root, run the upload-readiness check:

```bash
python3 scripts/check_project.py
```

From this package directory, the direct component checks are:

```bash
python3 model_target_registry.py
python3 -m unittest discover -s tests
python3 -m py_compile \
  memory_moe.py \
  moe_shared_contract.py \
  moe_forward_probe.py \
  run_forward_probe_demo.py \
  moe_controller_demo.py \
  llama_runtime_probe.py \
  llama_sidecar.py \
  model_target_registry.py
```

## Docker Materials

Docker files are present but not part of the local readiness command:

- [docker/llama-sidecar.Dockerfile](docker/llama-sidecar.Dockerfile)
- [docker/llama-with-sidecar-entrypoint.sh](docker/llama-with-sidecar-entrypoint.sh)
- [docker/forward-hook-probe.Dockerfile](docker/forward-hook-probe.Dockerfile)
- [docker/forward-hook-probe.requirements.txt](docker/forward-hook-probe.requirements.txt)
- [docker/forward-hook-probe.compose.yaml](docker/forward-hook-probe.compose.yaml)

The forward-hook container stack is for PyTorch/Transformers-style runtimes, not stock `llama.cpp`.

## Windows Helper

[run_mixtral_probe.ps1](run_mixtral_probe.ps1) is a Windows PowerShell helper for the older direct Mixtral prompt probe. Its defaults are relative to this package directory:

```powershell
py -3 .\run_mixtral_probe.ps1
```

The maintained cross-platform live command shapes are in [docs/model-target-test-plan.md](docs/model-target-test-plan.md).

## Next Work

The grounded next step is a real shared baseline report from a user-supplied live MoE backend. After that, decide whether to stay at the probe/advisor layer, wire a hookable PyTorch bridge, or descend into a backend-specific runtime patch.
