# Model Plane Manifest Consumption

MoE Run Anyway consumes the Model Plane MoE probe manifest as the durable bridge
from local runtime orchestration into probe planning. The manifest should come
from Model Plane's backend API or callable functions, not from values copied out
of the human console.

The bridge flow is:

```text
Model Plane callable function or cron -> validate/launch/health/logs
  -> run-scoped MoE probe manifest -> planner
  -> planned harness_run_request -> approved probe path
```

The human UI is useful for status and inspection. The machine-readable manifest
is the contract agents should pass between repos.

Model Plane owns cron/callable orchestration, profile validation, launch/health
checks, log discovery, and run-scoped manifest export. MoE Run Anyway owns
manifest consumption, safe probe planning, artifact contracts, and controller
replay. The planner may describe a `harness_run_request`, but that request is a
planning artifact, not a runtime executor.

## Planner Entry Point

From the repository root:

```bash
python3 scripts/plan_moe_probe_manifest.py /path/to/moe-probe-manifest.json
```

Add `--json` when another agent or script needs the plan as structured output:

```bash
python3 scripts/plan_moe_probe_manifest.py /path/to/moe-probe-manifest.json --json
```

The planner validates only the bridge contract. It does not start model servers,
download models, authenticate, inspect private tokens, run Docker, or send
prompt traffic.

Structured output includes a `planned_harness_run_request` object with:

- `stage: harness_run_request`
- `status: planned_only`
- selected `target_class`
- safe command class and deferred live command class
- expected artifact class
- missing actuator capabilities such as expert pin, evict, preload, and
  load-only-routed-experts

This planned stage exists so Model Plane agents and MoE agents can agree on the
next harness step without implying that a live expert-paging actuator exists.

## Required Manifest Fields

The planner accepts `schema_version: model-plane-moe-probe-manifest-v1` and
requires these non-empty fields:

- `profile_id`
- `model_id`
- `backend_family`
- `base_url`
- `primary_probe_hint`

It also consumes:

- `health_url`
- `container_name`
- `log_file_path`
- `model_path`
- `semantic_expert_ids`
- `hookable_runtime_available`
- `passive_sidecar_requested`
- `runtime_observability.expected_paths`
- `safety_notes`

`semantic_expert_ids` must be one of:

- `not_exposed`
- `expected_when_router_outputs_are_exposed`
- `runtime_dependent`

## Decision Policy

Agents should let `scripts/plan_moe_probe_manifest.py` make the final path
selection, but the policy is intentionally simple:

| Manifest signal | Selected target class | Meaning |
| --- | --- | --- |
| `primary_probe_hint=runtime_baseline` with `backend_family=llama_cpp` or unknown safe default | `stock_llama_cpp_openai_compatible` | Use guarded llama.cpp runtime baseline planning against the existing endpoint. |
| `backend_family=vllm_openai_compatible`, `ollama_openai_compatible`, or `openai_compatible` | `openai_compatible_runtime` | Use guarded OpenAI-compatible runtime planning where `/v1/models` or `/models` can satisfy readiness. |
| `primary_probe_hint=passive_sidecar` | `passive_sidecar_proxy` | Put the passive sidecar between client traffic and the upstream endpoint. |
| `primary_probe_hint=hookable_pytorch` and `hookable_runtime_available=true` | `hookable_pytorch_moe` | Use the hookable semantic path where router outputs may be captured. |
| Hookable hint without `hookable_runtime_available=true` | invalid manifest | Refuse to plan semantic probing. |

Runtime baseline plans emit only `run_live_baseline.py --dry-run` and
`run_live_baseline.py --preflight-only` commands. These are planning and
readiness commands, not prompt-traffic runs. Runtime baseline commands pass the
manifest `backend_family` through to the runner and runtime probe so artifacts
do not claim `llama_cpp` when the profile is vLLM, Ollama, or another
OpenAI-compatible backend.

Passive sidecar plans emit the sidecar command because the sidecar itself is the
non-invasive observation layer. A separate client must still choose to send
traffic through it.

Hookable PyTorch plans emit the no-model synthetic hook smoke command as a safe
first step. A real local hookable runner remains deferred unless a compatible
checkpoint and runtime already exist.

## Semantic Honesty

Do not infer semantic expert ids from stock endpoint telemetry. For
`llama.cpp`, vLLM, Ollama, and generic OpenAI-compatible endpoints, Model Plane
can provide base URLs, health URLs, log paths, and observability hints; MoE Run
Anyway can collect runtime/request evidence from those surfaces. That evidence
is useful, but it is not a router trace.

Semantic expert ids require one of:

- a hookable PyTorch/Transformers-style runtime exposing router modules or
  router outputs
- a future backend patch that emits semantic expert ids
- a model-specific runner that writes compatible `memory-moe-bridge-v1`
  artifacts

The manifest field `semantic_expert_ids` is therefore a guardrail, not a claim
of coverage.

## Agent Handoff Shape

A safe agent loop is:

1. Ask Model Plane for `GET /profiles`.
2. Choose a profile and call `POST /profiles/{profile_id}/validate`.
3. Use Model Plane callable launch, health, log-inspection, or cron functions
   only when the user has approved runtime actions.
4. Fetch or receive the run-scoped MoE probe manifest, such as
   `GET /profiles/{profile_id}/moe-probe-manifest` or a callable-function
   export artifact.
5. Save the JSON manifest.
6. Run `python3 scripts/plan_moe_probe_manifest.py manifest.json --json`.
7. Review the returned `planned_harness_run_request`.
8. Present or execute only the returned command class the user has approved.

This keeps Model Plane as the orchestration/control layer and MoE Run Anyway as
the probe planner and harness. The user should not have to copy endpoints,
ports, model ids, or log paths between applications.

## Expert Paging Boundary

The current bridge can plan runtime baselines, passive sidecar capture,
hookable PyTorch semantic traces, and controller replay. It still cannot page
experts. Real expert paging needs a future runtime actuator, hook, or patch
that can control expert tensor residency or expert offload behavior.

For `llama.cpp`, an external controller can schedule runs, select probe suites,
consume artifacts, and recommend residency. Loading only routed experts requires
a `llama.cpp` hook, controller patch, or fork that can prove routing visibility,
tensor residency control, dense fallback, compatible artifacts, and cleanup.
Those proof requirements are tracked in
[expert-paging-roadmap.md](expert-paging-roadmap.md) and
[../data/expert_paging_roadmap.json](../data/expert_paging_roadmap.json).
