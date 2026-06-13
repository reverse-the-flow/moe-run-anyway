# Model Target Test Plan

Date: 2026-06-12

This project is not ready for a single universal "MoE test." The observable
surface depends on how the model is served. The durable test direction is a
small target registry plus probe-specific validation tiers that reuse the
current sidecar, llama runtime probe, forward-hook probe, and shared contract.

## Target Classes

The registry in `data/model_target_registry.json` currently requires coverage
for these classes:

- `stock_llama_cpp_openai_compatible`: stock `llama-server` or another
  OpenAI-compatible server. This path can validate request handling, usage,
  timings, metrics, slots, props, and log-growth correlation. It should not
  claim semantic expert ids.
- `passive_sidecar_proxy`: sidecar observation around a running backend. This
  proves non-invasive request/memory telemetry before a runtime fork or hook is
  justified.
- `hookable_pytorch_moe`: PyTorch/Transformers-style runtime where router or
  gate modules can be hooked. This is the path for semantic expert ids, expert
  weights, routing entropy, and per-window hit counts.
- `small_local_moe`: an already cached or user-provided smaller MoE target,
  such as an OLMoE-style checkpoint, for fast iteration without downloads.
- `mixtral_style`: project anchor for the already referenced Mixtral-family
  target, tested through both opaque `llama.cpp` observation and hookable
  runtime observation when available.

## Test Tiers

Tier 0 is pure local validation:

- validate `data/model_target_registry.json` with `model_target_registry.py`
- run the unit tests
- run `py_compile` for touched Python files
- run `run_forward_probe_demo.py` only when a synthetic artifact smoke is
  useful

Tier 1 is passive live observation:

- run `python3 scripts/check_host.py` from the repository root when the host is
  new or unknown
- user starts a model server
- run `llama_sidecar.py`
- route client traffic through the sidecar
- inspect `manifest.json`, `events.jsonl`, and `summary.json`

Tier 2A is stock `llama.cpp` active observation:

- run `python3 scripts/check_host.py --require-gpu` from the repository root
  when the test depends on local GPU execution
- user starts `llama-server` with `--metrics --slots --props --perf`
- run `llama_runtime_probe.py` against `data/mixtral_probe_prompts.json`
- assert shared-contract shape and prompt-family coverage
- do not assert semantic expert ids

Tier 2B is hookable semantic routing:

- run `python3 scripts/check_host.py --require-gpu` from the repository root
  when the target checkpoint needs GPU execution
- user provides an already available local PyTorch MoE checkpoint
- attach `ForwardHookMoEProbe`
- assert router events, expert ids or inferred top-k ids, entropy, and window
  summaries

## Next Live Commands

From `memory-moe-mvp/`, with a user-started llama-server:

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

From the repository root, the guarded runner performs the same runtime probe
after checking that at least one observability endpoint is reachable:

```bash
python3 scripts/run_live_baseline.py \
  --base-url http://127.0.0.1:18080 \
  --model dolphin-mixtral \
  --output-dir memory-moe-mvp/runtime-probe-runs \
  --label mixtral-live-baseline \
  --suite-path memory-moe-mvp/data/mixtral_probe_prompts.json \
  --max-prompts 4 \
  --repeats 2 \
  --preflight-timeout-seconds 2 \
  --log-file-path /path/to/llama-server.log
```

For passive observation of the same upstream:

```bash
python3 llama_sidecar.py \
  --listen-port 8091 \
  --upstream-base-url http://127.0.0.1:18080 \
  --output-dir sidecar-runs \
  --label mixtral-sidecar \
  --capture-upstream-observability
```

For a no-model hook-shape smoke:

```bash
python3 run_forward_probe_demo.py \
  --output-dir forward-probe-runs \
  --suite-path data/mixtral_probe_prompts.json \
  --max-prompts 2 \
  --window-size-events 2 \
  --label synthetic-hook-smoke
```

For a future local hookable model runner, the command shape should stay close
to:

```bash
MODEL_PATH=/path/to/local/model \
python3 path/to/future_transformers_runner.py \
  --model-path "$MODEL_PATH" \
  --output-dir forward-probe-runs \
  --suite-path data/mixtral_probe_prompts.json \
  --max-prompts 4 \
  --repeats 2
```

The future runner should be thin: load the user-provided local model, attach
`ForwardHookMoEProbe`, execute prompt cases, and close the probe. It should not
change the shared contract or introduce a new probe artifact shape.
