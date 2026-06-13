# Live Model Readiness Planner

Date: 2026-06-13

`scripts/plan_live_model.py` is the no-secrets planning layer before live MoE
tests. It inspects the local machine, cached model hints, optional backend
tools, and `data/model_target_registry.json`, then prints next commands for each
supported target class.

It does not start a server, download a model, authenticate, inspect private
tokens, run Docker, load checkpoints, or send prompt traffic.

## Run The Planner

From the repository root:

```bash
python3 scripts/plan_live_model.py
```

For machine-readable output:

```bash
python3 scripts/plan_live_model.py --json
```

To avoid even the short local HTTP observability probe:

```bash
python3 scripts/plan_live_model.py --skip-network
```

To focus on one target class:

```bash
python3 scripts/plan_live_model.py --target-class stock_llama_cpp_openai_compatible
```

## What It Detects

The planner reports:

- OS, platform, Python executable, and Python version
- visible NVIDIA/CUDA or AMD/ROCm tooling
- optional backend tools such as `llama-server`, `docker`, and
  `huggingface-cli`
- Python module presence for `torch` and `transformers`
- local cache hints from `MODEL_PATH`, `LLAMA_MODEL_PATH`, Hugging Face cache
  directories, `~/.cache/llama.cpp`, `~/models`, and `./models`
- `.gguf` files found under those cache roots
- local observability endpoints at `/props`, `/metrics`, and `/slots`, unless
  `--skip-network` is used

Cache hints are only hints. The planner does not verify model compatibility,
license access, quantization quality, context length, or whether a checkpoint
will fit on the GPU.

## Supported Target Classes

The output follows the registry target classes:

- `stock_llama_cpp_openai_compatible`: suggests a `llama-server` start command,
  a guarded `run_live_baseline.py --preflight-only` command, and the runtime
  probe command. This path remains observational and does not claim semantic
  expert ids.
- `passive_sidecar_proxy`: suggests running `llama_sidecar.py` against a
  user-started upstream.
- `hookable_pytorch_moe`: suggests the synthetic forward-hook smoke and the
  command shape for a future local Transformers runner. Semantic expert traces
  require router outputs from a hookable local runtime.
- `small_local_moe`: follows the same hookable-runtime shape, but is intended
  for an already cached smaller MoE checkpoint.
- `mixtral_style`: keeps the Mixtral-family matrix anchored across the opaque
  llama.cpp path and the hookable PyTorch path.

## Typical GPU-Only Host Flow

On a new GPU host, run:

```bash
python3 scripts/check_project.py
python3 scripts/check_host.py --require-gpu
python3 scripts/plan_live_model.py
```

If the planner reports no live backend, start a compatible local backend
yourself with a local model file, then re-run:

```bash
python3 scripts/run_live_baseline.py \
  --base-url http://127.0.0.1:18080 \
  --model local-moe \
  --preflight-only \
  --preflight-timeout-seconds 2
```

Only after that guarded preflight sees `/props`, `/metrics`, or `/slots` should
the active runtime baseline send prompt traffic.

## Honesty Boundary

The planner improves readiness and orchestration. It is not proof of live MoE
semantics. Stock `llama.cpp` endpoints can support request/runtime telemetry,
metrics deltas, slot state, props snapshots, and logs, but not per-layer expert
ids. Semantic expert traces require a hookable runtime or future backend patch
that exposes router outputs.
