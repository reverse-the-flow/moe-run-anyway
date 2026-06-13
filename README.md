# MoE Run Anyway

This repository is a measurement-first workspace for testing whether Mixture-of-Experts runtime behavior is observable and controllable enough to justify deeper expert-residency work.

It is currently a probe and replay project, not a production MoE pager. The useful surface today is:

- simulator/controller replay for expert residency policies
- passive sidecar telemetry around an OpenAI-compatible or `llama-server` backend
- active `llama.cpp` runtime probing through stock server observability endpoints
- forward-hook probing for hookable PyTorch-style MoE runtimes
- a local model-target registry that records which target classes each probe can cover

## What This Is Not Yet

The repo does not currently load, evict, pin, or page real model experts in a live runtime. It also does not include model weights, a live model server, a Docker-run requirement, or GPU-heavy validation in the uploadable check path.

Live model work is intentionally deferred until a user supplies a running backend or a local hookable checkpoint.

## Layout

- `memory-moe-mvp/`: source package, probe scripts, fixtures, tests, Docker scaffolding, and project docs.
- `memory-moe-mvp/data/model_target_registry.json`: target matrix for stock `llama.cpp`, passive sidecar, hookable PyTorch MoE, small local MoE, and Mixtral-style paths.
- `memory-moe-mvp/docs/model-target-test-plan.md`: live-test gates and command shapes.
- `memory-moe-mvp/docs/live-model-readiness-planner.md`: no-secrets planner for host, cache, backend-tool, and target-class readiness.
- `memory-moe-mvp/docs/live-baseline-runner.md`: guarded entrypoint for the first live runtime baseline.
- `memory-moe-mvp/docs/portability-and-gpu-hosts.md`: host portability, GPU preflight, and live-machine dependency notes.
- `context-pack/`: archived conversation/context notes that explain the project direction.
- `CATCHUP_STATUS.md`, `MODEL_TESTS_STATUS.md`, `GIT_BASELINE_STATUS.md`, `UPLOAD_READINESS.md`: status trail for future agents.
- `scripts/check_project.py`: dependency-free local readiness check.
- `scripts/check_host.py`: dependency-free host preflight for Python, git, GPU tooling, and optional local model/server tools.

## Local Validation

From the repository root:

```bash
python3 scripts/check_project.py
```

The check validates the model-target registry, runs unit tests, compiles Python sources, and scans primary docs for stale absolute Windows paths.

To inspect a host before live GPU work:

```bash
python3 scripts/check_host.py
```

Use `python3 scripts/check_host.py --require-gpu` only for workflows that truly need visible NVIDIA/CUDA or AMD/ROCm tooling. Read [memory-moe-mvp/docs/portability-and-gpu-hosts.md](memory-moe-mvp/docs/portability-and-gpu-hosts.md) for the portable path and host expectations.

To plan the next live-model commands without secrets, downloads, server starts,
or prompt traffic:

```bash
python3 scripts/plan_live_model.py
```

After committing, verify the uploadable tree is clean:

```bash
python3 scripts/check_project.py --require-clean
```

## Live Model Gates

The next live checks remain deferred by design:

- user-started `llama-server` with `/metrics`, `/slots`, `/props`, and logs enabled
- passive sidecar run against that upstream
- active runtime probe against `data/mixtral_probe_prompts.json`
- hookable PyTorch MoE run only when a local checkpoint and compatible environment already exist

The first live runtime baseline now has a guarded runner:

```bash
python3 scripts/plan_live_model.py --target-class stock_llama_cpp_openai_compatible
python3 scripts/run_live_baseline.py --dry-run
python3 scripts/run_live_baseline.py --base-url http://127.0.0.1:18080 --model dolphin-mixtral --preflight-only --preflight-timeout-seconds 2
```

Read [memory-moe-mvp/docs/model-target-test-plan.md](memory-moe-mvp/docs/model-target-test-plan.md) for the exact command shapes.
Read [memory-moe-mvp/docs/live-model-readiness-planner.md](memory-moe-mvp/docs/live-model-readiness-planner.md) for the no-secrets planning layer.
Read [memory-moe-mvp/docs/live-baseline-runner.md](memory-moe-mvp/docs/live-baseline-runner.md) for the runner contract.

## Current Status

Start with [UPLOAD_READINESS.md](UPLOAD_READINESS.md), then use:

- [GIT_BASELINE_STATUS.md](GIT_BASELINE_STATUS.md) for the initial repository baseline
- [MODEL_TESTS_STATUS.md](MODEL_TESTS_STATUS.md) for the target-registry pass
- [CATCHUP_STATUS.md](CATCHUP_STATUS.md) for the architecture catch-up snapshot

The durable project docs are in [memory-moe-mvp/README.md](memory-moe-mvp/README.md) and `memory-moe-mvp/docs/`.
