# Model Card: MoE Run Anyway

Date: 2026-06-14

## Summary

MoE Run Anyway is not a released model checkpoint. It is a measurement-first
evaluation harness for local Mixture-of-Experts runtime experiments. The
project tests whether different serving paths expose enough request, runtime,
and routing evidence to justify deeper expert-residency work.

The repository does not ship model weights, train a model, fine-tune a model,
start a model server, download gated assets, or manage private credentials.
Live tests require a user-supplied local backend or a user-provided local
checkpoint.

## Model Details

- Project name: MoE Run Anyway
- Artifact type: MoE runtime probe, planner, replay, and controller harness
- Model weights: none bundled
- Primary model families: Mixtral-style MoE, OLMoE-style small MoE, and
  OpenAI-compatible local MoE or dense backends
- Primary runtime families: stock `llama.cpp`/OpenAI-compatible servers,
  passive sidecar proxying, and hookable PyTorch/Transformers-style runtimes
- Current status: local validation and readiness planning work; live semantic
  expert traces still require a hookable local runtime or backend patch
- License: not specified by this repository; external model licenses remain
  governed by their model providers

## Supported Target Classes

| Target class | Backend family | Intended evidence | Semantic expert ids |
| --- | --- | --- | --- |
| `stock_llama_cpp_openai_compatible` | `llama_cpp` | request usage, timings, metrics, slots, props, optional logs | not exposed |
| `passive_sidecar_proxy` | `llama_cpp` or OpenAI-compatible | request boundaries, latency, usage, memory/GPU snapshots, upstream observability | not exposed |
| `hookable_pytorch_moe` | `pytorch_transformers` | router modules, inferred layer ids, expert weights or logits, routing entropy | expected when router outputs are exposed |
| `small_local_moe` | `pytorch_transformers` | faster semantic hook iteration against a smaller local MoE checkpoint | expected when router outputs are exposed |
| `mixtral_style` | mixed | matrix anchor across opaque runtime telemetry and hookable semantic probes | runtime dependent |

## Intended Use

Use this project to:

- validate that the repository and host are ready for local MoE experiments
- inspect whether a machine has GPU tooling, backend tools, cached model hints,
  and live observability endpoints
- run no-model synthetic replay and controller checks
- collect passive or active runtime telemetry from a user-started backend
- capture semantic router traces only when a hookable local runtime exposes
  router outputs
- compare probe outputs through the shared `memory-moe-bridge-v1` contract

## Out-of-Scope Use

Do not use this project as:

- a production model-serving stack
- a benchmark claiming model quality, accuracy, safety, or capability
- proof that stock `llama.cpp` exposes per-layer expert routing
- an automatic downloader or credential manager
- a tool for bypassing model licenses, access controls, or provider terms
- evidence that expert paging is implemented in a live runtime

## Inputs

The harness can consume:

- prompt suites such as `memory-moe-mvp/data/mixtral_probe_prompts.json`
- synthetic controller and workload fixtures
- OpenAI-compatible request/response JSON from a local backend
- optional `llama.cpp` observability surfaces: `/metrics`, `/slots`, `/props`,
  and bounded log growth
- hook events from PyTorch modules when `ForwardHookMoEProbe` can attach to a
  local model

## Outputs

Probe and replay runs write structured artifacts such as:

- `manifest.json`
- `events.jsonl`
- `summary.json`

The shared contract focuses on comparable runtime and controller fields,
including `probe_tier`, `backend_family`, `prompt_family`,
`resident_budget_fraction`, `fallback_used`, `warm_hit_rate`, `miss_rate`,
`churn`, `eviction_regret`, `context_retention`, and
`dense_baseline_delta`.

## Training Data

No model training or fine-tuning is performed by this repository. The bundled
data is limited to prompts, synthetic workloads, and replay fixtures for
measurement and contract validation.

External checkpoints used in live experiments may have their own training data,
licenses, and risk profiles. Those properties are not inherited by this
repository and must be reviewed separately for each selected model.

## Evaluation Data And Metrics

Current validation covers repository behavior, not model capability:

- target registry schema coverage
- unit tests for planner, probes, replay, and controller surfaces
- Python compilation checks
- docs portability checks
- optional host preflight for GPU and backend tooling
- guarded live-backend preflight before prompt traffic

The project does not currently report accuracy, helpfulness, toxicity, bias,
memorization, robustness, or benchmark scores for any model.

## Safety And Privacy

The default tools are designed to avoid secrets and external side effects:

- no model downloads
- no authentication
- no token inspection
- no server start
- no Docker run
- no prompt traffic unless the user explicitly runs a live probe against a
  reachable local backend

Prompt suites and generated probe artifacts may still contain user-provided
text if live tests are run. Treat run artifacts as local experiment data and
review them before sharing.

## Limitations

- Stock `llama.cpp` observability can support runtime telemetry, not semantic
  expert ids.
- Semantic routing evidence requires a hookable runtime or future backend patch
  that exposes router outputs.
- Cached model hints do not prove that a checkpoint is complete, compatible,
  licensed, quantized appropriately, or small enough for the machine.
- GPU presence does not prove the selected backend can use that GPU.
- The controller and simulator are advisory/replay tools, not live expert
  paging implementations.

## Operational Readiness

Before live work on a new machine:

```bash
python3 scripts/check_project.py
python3 scripts/check_host.py --require-gpu
python3 scripts/plan_live_model.py
```

Before sending prompt traffic to a local backend:

```bash
python3 scripts/run_live_baseline.py \
  --base-url http://127.0.0.1:18080 \
  --model local-moe \
  --preflight-only \
  --preflight-timeout-seconds 2
```

Only continue to an active runtime probe after the guarded preflight sees at
least one observability endpoint.

## Maintenance Notes

Update this card when the project adds:

- a real local model runner
- a backend patch that emits expert-routing events
- a new supported target class
- a bundled or recommended model checkpoint
- live benchmark results that are reproducible and clearly scoped
