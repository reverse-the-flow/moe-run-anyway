# Model Tests Status

Date: 2026-06-12

Workspace: `/home/codexlab/moe-run-anyway-project`

## Git State

- Repository: initialized git worktree.
- Branch: `main`.
- Commit state: no commits yet.
- Baseline commit: not created because local git identity is not configured.
  - `git config user.name` returned no value.
  - `git config user.email` returned no value.
- Current status after this pass: uncommitted, untracked project files plus ignored local logs/caches.
- Commit hash: none.

The repo is ready for a first commit after identity is configured, for example
by the user with local `user.name` and `user.email` settings.

## Files Changed This Pass

- `.gitignore`
  - Ignores Python caches, local test/cache output, logs, temp files, Codex
    session logs, and generated probe/controller run directories.
  - Does not ignore source, docs, tests, data fixtures, Docker files, context
    notes, prompts, or status docs.
- `memory-moe-mvp/data/model_target_registry.json`
  - Adds the durable target matrix fixture for model/runtime families.
- `memory-moe-mvp/model_target_registry.py`
  - Adds a dependency-free validator and summary CLI for the target registry.
- `memory-moe-mvp/docs/model-target-test-plan.md`
  - Documents target classes, test tiers, and exact next live-test command
    shapes.
- `memory-moe-mvp/tests/test_model_target_registry.py`
  - Adds local unit coverage for required target-class coverage and registry
    validation behavior.
- `MODEL_TESTS_STATUS.md`
  - This status report.

Pre-existing source, tests, docs, data fixtures, Docker files, context notes,
catch-up files, and run scripts were preserved.

## Model-Test Direction Added

This pass keeps the existing probe architecture and adds a disciplined target
matrix around it rather than introducing a new runtime architecture.

Covered target classes:

- `stock_llama_cpp_openai_compatible`
  - Uses `llama_runtime_probe.py`.
  - Observes stock `llama-server` or OpenAI-compatible surfaces.
  - Explicitly marks semantic expert ids as not exposed.
- `passive_sidecar_proxy`
  - Uses `llama_sidecar.py`.
  - Captures request/response, latency, memory, optional GPU, and optional
    upstream `/metrics`, `/slots`, and `/props` snapshots.
- `hookable_pytorch_moe`
  - Uses `moe_forward_probe.py`.
  - Intended for PyTorch/Transformers-style MoE runtimes where router/gate
    modules expose expert ids, weights, or logits.
- `small_local_moe`
  - Represents already cached or user-provided small MoE targets such as
    OLMoE-style checkpoints.
  - No download or package install is assumed.
- `mixtral_style`
  - Anchors the already referenced Mixtral-family path.
  - Allows both opaque `llama.cpp` observation and hookable PyTorch semantic
    routing observation when the runtime permits it.

The new registry validator makes the target coverage testable in pure local CI:

```bash
python3 memory-moe-mvp/model_target_registry.py
```

## Commands And Checks Run

From `/home/codexlab/moe-run-anyway-project`:

```bash
git status --short --branch
git rev-parse --is-inside-work-tree
git config user.name
git config user.email
git log --oneline -1
python3 memory-moe-mvp/model_target_registry.py
python3 -m unittest discover -s memory-moe-mvp/tests
python3 -m py_compile \
  memory-moe-mvp/model_target_registry.py \
  memory-moe-mvp/moe_shared_contract.py \
  memory-moe-mvp/moe_forward_probe.py \
  memory-moe-mvp/run_forward_probe_demo.py \
  memory-moe-mvp/llama_runtime_probe.py \
  memory-moe-mvp/llama_sidecar.py
git status --short --branch --ignored
```

Observed validation results:

- Registry validation: valid, 5 targets, all required target classes covered.
- Unit tests: `Ran 21 tests in 0.566s`, `OK`.
- `py_compile`: passed with no output.
- Git log: `fatal: your current branch 'main' does not have any commits yet`.

## Next Live Model Commands

These are intentionally deferred until the user supplies/runs the backend.

From `memory-moe-mvp/`, with a user-started `llama-server`:

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

Passive sidecar around a running upstream:

```bash
python3 llama_sidecar.py \
  --listen-port 8091 \
  --upstream-base-url http://127.0.0.1:18080 \
  --output-dir sidecar-runs \
  --label mixtral-sidecar \
  --capture-upstream-observability
```

No-model hook-shape smoke:

```bash
python3 run_forward_probe_demo.py \
  --output-dir forward-probe-runs \
  --suite-path data/mixtral_probe_prompts.json \
  --max-prompts 2 \
  --window-size-events 2 \
  --label synthetic-hook-smoke
```

Future hookable local model runner shape:

```bash
MODEL_PATH=/path/to/local/model \
python3 path/to/future_transformers_runner.py \
  --model-path "$MODEL_PATH" \
  --output-dir forward-probe-runs \
  --suite-path data/mixtral_probe_prompts.json \
  --max-prompts 4 \
  --repeats 2
```

## Blockers And Deferred Requirements

- Baseline commit is blocked by missing git identity.
- No live model server was started.
- No Docker command was run.
- No model was downloaded.
- No packages were installed.
- No GPU-heavy job was launched.
- Real `llama.cpp`/Mixtral testing requires a user-started server with model
  weights already available.
- Semantic expert ids remain unavailable on the stock `llama.cpp` path unless a
  `libllama` harness or runtime patch emits routing events.
- Hookable PyTorch MoE testing requires a user-provided local checkpoint and a
  compatible environment with the needed packages already installed.
- OLMoE-style or other small local MoE testing is represented in the registry
  but deferred until a local model path/runtime is provided.
