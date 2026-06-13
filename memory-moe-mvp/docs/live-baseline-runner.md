# Live Baseline Runner

Date: 2026-06-13

`scripts/run_live_baseline.py` is the one-command entrypoint for the first
real `llama.cpp` or OpenAI-compatible live baseline. It wraps
`memory-moe-mvp/llama_runtime_probe.py` with a small preflight so the project
does not accidentally send prompt traffic to a missing or misconfigured server.

The runner does not start a server, download a model, authenticate, run Docker,
or use a Hugging Face token. It assumes the user has already started a local
backend and supplied any model files outside the repository.

## Dry Run

From the repository root:

```bash
python3 scripts/run_live_baseline.py --dry-run
```

This prints the exact runtime-probe command that would run, without touching the
network.

## Preflight A Running Backend

With a user-started server:

```bash
python3 scripts/run_live_baseline.py \
  --base-url http://127.0.0.1:18080 \
  --model dolphin-mixtral \
  --preflight-only \
  --preflight-timeout-seconds 2
```

The preflight checks stock observability surfaces:

- `/props`
- `/metrics`
- `/slots`

At least one of those surfaces must be reachable before the runner will send
prompt traffic, unless `--skip-preflight` is explicitly used.

## Run The First Live Baseline

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

This launches `llama_runtime_probe.py`, which writes:

- `manifest.json`
- `events.jsonl`
- `summary.json`

The result is still observational. Stock `llama.cpp` does not expose semantic
expert ids through these endpoints, so the baseline proves request/runtime
telemetry and shared-contract shape, not per-layer expert routing.

## Automation Output

For machine-readable planning or preflight results:

```bash
python3 scripts/run_live_baseline.py --dry-run --json
python3 scripts/run_live_baseline.py --base-url http://127.0.0.1:18080 --preflight-only --preflight-timeout-seconds 2 --json
```

## Where This Fits

Use this after:

1. `python3 scripts/check_project.py`
2. `python3 scripts/check_host.py --require-gpu`
3. starting a compatible local model server yourself

Use the forward-hook path later for semantic routing traces from hookable
PyTorch-style MoE checkpoints.
