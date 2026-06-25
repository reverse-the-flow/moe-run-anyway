# Managed Expert Loading

Managed expert loading is MoE Run Anyway's first concrete bridge from expert
paging replay toward a future live actuator. The abstraction names what must be
known before any backend is allowed to move expert tensors: inventory, routing
visibility, residency state, policy action, adapter capability, dense fallback,
cleanup, and artifact export.

Live expert loading is not implemented yet. This pass is a contract and planner
only.

Validate the machine-readable plan from the repository root:

```bash
python3 scripts/plan_managed_expert_loading.py
python3 scripts/plan_managed_expert_loading.py --json
```

The planner is dependency-free and side-effect-free. It does not start models,
download files, authenticate, run Docker, launch servers, send prompt traffic,
inspect secrets, or mutate runtime state.

## Ownership Boundary

Model Plane owns state, control, cron/callable orchestration, profile
validation, launch/health/log inspection, and the run-scoped manifest handoff.
It is the system that knows when a run exists and how a backend is supervised.

MoE Run Anyway owns harness/runtime evidence and planning: consuming manifests,
classifying backend capability gaps, validating managed-loading contracts,
running replay/simulation policy audits, and emitting comparable artifacts.
MoE Run Anyway must not reach back into Model Plane internals or claim runtime
control it does not have.

## Modes

`observe_only` is the current safe default. It can summarize backend evidence,
inventory hints, and missing actuator capabilities. It cannot change expert
residency.

`replay_simulate` applies managed-loading policy actions to recorded,
synthetic, or hook-derived traces. It can score warm-hit rate, churn, fallback
frequency, and dense-baseline deltas. It still cannot mutate a live backend.

`live_actuator` is a future mode. It would be allowed to request preload, pin,
evict, demote, fallback, abort, and cleanup operations only after a backend
adapter proves the required capabilities. There is no live actuator in this
repository today.

## Contract Vocabulary

Expert inventory must identify the model/backend, layer id, expert id, expert
kind, storage or residency cost, and whether that expert can be loaded by the
selected adapter. Valid sources include runtime-exported inventory, local
manifest metadata, hookable module walks, and static fixtures. Endpoint timing
alone is not inventory.

Residency state uses this vocabulary:

- `resident`
- `offloaded_cpu`
- `offloaded_disk`
- `loading`
- `evicting`
- `unknown`

Policy actions use this vocabulary:

- `observe_only`
- `preload`
- `pin`
- `keep`
- `evict`
- `demote`
- `fallback_dense`
- `abort_run`

The required actuator capabilities are:

- `expert_inventory`
- `routing_visibility`
- `residency_observation`
- `residency_control`
- `policy_application`
- `dense_fallback`
- `cleanup_restore`
- `artifact_export`

## Backend Adapter Classes

`llama_cpp` covers stock `llama.cpp` / `llama-server` evidence and a possible
small patch or fork later. Stock observability is useful runtime evidence, but
it does not expose semantic expert routing or live residency mutation. A patched
engine hook can become a valid semantic-routing source if it emits selected
expert ids, scores, layer ids, and token/window metadata under the shared trace
contract.

The first patched direct llama.cpp hook now emits selected expert ids and
selected weights for GX10 Mixtral and Qwen3 GGUF runs. That upgrades
`llama_cpp` routing visibility for patched direct runs only. It still does not
provide residency observation, residency control, or cleanup/restore proof.

`vllm_openai_compatible` covers vLLM or compatible HTTP endpoints. Request
metadata and timing are not semantic expert ids unless a plugin or runtime
extension exposes MoE internals.

`hookable_pytorch` covers local PyTorch/Transformers-style MoE runtimes where
module walks and forward hooks can expose router outputs. This can support
semantic traces and replay, but it is not live residency control by itself.

`moe_infinity_style` covers an already installed offload-capable runtime class.
The planner does not install or launch it. Any adapter must still prove
inventory, routing visibility, residency observation/control, dense fallback,
cleanup, and artifact export.

`prototype_offload_system` covers a future local prototype adapter. It is useful
for exercising the contract shape, not for claiming production behavior.

## Fallback, Cleanup, And Safety

Dense or full-residency fallback is mandatory for every live experiment design.
Managed loading must be comparable against normal backend behavior, and
`fallback_dense` must remain available when capabilities are missing.

Cleanup is also mandatory. A failed, interrupted, or aborted live run must
restore runtime state or prove that no residency mutation occurred. Without
cleanup proof, the correct policy action is `abort_run` or `fallback_dense`.

Safety constraints remain strict: no model downloads, no Docker runs, no server
launch, no prompt traffic, no secret inspection, and no runtime mutation from
the planner or roadmap checks.

## Current Capability Gaps

The current project can validate plans, collect endpoint/runtime evidence,
capture hookable semantic traces in suitable local runtimes, and replay policy
decisions offline. It still lacks live residency observation, live residency
control, backend cleanup/restore proof, and validated live managed-loading
artifacts.

Do not claim live managed expert loading until those gaps are closed by a real
backend adapter and reproducible artifacts.
