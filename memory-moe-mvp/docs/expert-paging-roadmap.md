# Expert Paging Roadmap

MoE Run Anyway is the harness and contract layer for expert-paging work. It can
plan probes, collect comparable artifacts, replay routing traces, and audit
controller policies. It is not yet a runtime actuator.

The current objective is to move from harness/probes to real expert paging
without pretending the missing runtime control exists. The likely future path
may require a small `llama.cpp` controller patch or fork if evidence shows
`llama.cpp` is the right substrate.

## Current Truth Table

| Surface | Exists now | What it can prove | What it cannot prove |
| --- | --- | --- | --- |
| Model Plane MoE manifest handoff | Yes | Run-scoped model, backend, endpoint, log, observability, and safety hints can be exported for MoE planning. | It does not expose semantic expert ids unless the runtime does. |
| Runtime baseline planner | Yes | Safe dry-run and preflight-only commands for `llama.cpp`, vLLM, Ollama, or OpenAI-compatible endpoints. | It does not launch servers, send prompt traffic, or page experts. |
| Passive sidecar | Yes | Request-boundary telemetry and upstream observability can be captured non-invasively. | It cannot alter expert residency or infer semantic routing from endpoint timing alone. |
| Stock `llama.cpp` runtime probe | Yes | Metrics, slots, props, timings, and optional log growth can become runtime evidence. | Stock endpoint telemetry is not semantic expert routing. |
| Forward-hook PyTorch probe | Yes for hookable runtimes | Router outputs, routed expert ids, weights, entropy, and per-layer hit counts can be captured when modules expose them. | It does not apply to stock `llama.cpp` GGUF execution. |
| Controller replay/advisor | Yes | Offline residency-policy scoring and dense-fallback comparison over traces or summaries. | It cannot pin, evict, preload, or load only routed experts in a live backend. |
| Expert paging simulator | Yes | Logical policy behavior, churn, warm-hit rate, and replay shape. | It is not live tensor residency control. |
| Runtime actuator | No | Not applicable. | No live expert loading, eviction, pinning, preloading, or offload-behavior override exists yet. |

## Roadmap Phases

### Phase 0: Harness Contract And Model Plane Handoff

Goal: make the current boundary explicit and machine-checkable.

Deliverables:

- Model Plane run-scoped MoE manifests remain the durable handoff contract.
- MoE planners consume manifests without launching models, downloading weights,
  authenticating, running Docker, or sending prompt traffic.
- A planned `harness_run_request` stage describes what the harness would run
  after user approval, but remains planning-only.
- Roadmap and actuator-spike requirements are tracked as artifacts.

Evidence gate:

- `scripts/plan_moe_probe_manifest.py` and `scripts/plan_expert_paging.py`
  validate local artifacts and print next actions without side effects.
- Dependency-free unit tests cover the roadmap and planner outputs.

### Phase 1: Artifact Evidence From Runtime Baselines

Goal: collect comparable evidence from existing runtimes before designing an
actuator.

Deliverables:

- llama.cpp baseline artifacts from `/metrics`, `/slots`, `/props`, response
  timings, optional log growth, and Model Plane metadata.
- vLLM and OpenAI-compatible baseline artifacts where readiness endpoints exist.
- Artifact shape mapped to `memory-moe-bridge-v1`.
- Clear labels for runtime evidence versus semantic routing evidence.

Evidence gate:

- At least one run-scoped artifact bundle per backend class.
- Baseline artifacts show reproducible timing and observability fields.
- No artifact claims semantic expert ids unless the runtime explicitly exposes
  router outputs.

### Phase 2: Hookable PyTorch Semantic Routing Trace Runner

Goal: capture semantic routing traces in a runtime where hooks are legitimate.

Deliverables:

- A real local hookable trace runner for a PyTorch/Transformers-style MoE.
- Router output capture for layer id, expert id, score or probability, entropy,
  prompt family, and token/window metadata.
- Compatibility with replay/controller artifacts.

Evidence gate:

- Dense fallback output can be recorded or compared for the same prompt set.
- Trace artifacts validate under `memory-moe-bridge-v1`.
- Hooking failure modes are explicit and do not silently downgrade to timing
  inference.

### Phase 3: Controller Policy And Replay Audit

Goal: prove controller policies offline before touching live runtime residency.

Deliverables:

- Replay controller evaluates candidate residency policies over semantic traces.
- Dense fallback comparison is part of the audit contract.
- Policy outputs include preload, keep, evict, fallback, and confidence fields.
- Failure-mode reports cover churn, missed experts, fallback frequency, and
  quality deltas.

Evidence gate:

- Controller improves or explains warm-hit rate without unacceptable churn.
- Dense fallback comparison bounds quality or behavior drift.
- Replay artifacts are sufficient to reproduce policy decisions.

### Phase 4: llama.cpp Actuator Feasibility Spike / Fork Plan

Goal: decide whether `llama.cpp` can support guarded live expert paging with a
small patch, controller hook, or fork.

Deliverables:

- A spike document or branch plan identifying exact `llama.cpp` control points.
- Proof requirements for routing visibility, tensor residency control, dense
  fallback, artifact shape, cleanup, and rollback.
- A comparison between a controller outside `llama.cpp`, a small controller
  patch, and a fork.

Evidence gate:

- There is a concrete runtime hook or patch point for pin/evict/preload or
  expert offload behavior.
- The patch can produce auditable artifacts without breaking dense baseline
  execution.
- Cleanup and fallback are tested under failure.

### Phase 5: Guarded Runtime Actuator Experiment

Goal: run a narrowly scoped live actuator experiment only after the prior gates
have passed.

Deliverables:

- A guarded actuator mode that can preload, pin, evict, or restrict expert
  residency in a live runtime.
- Kill switch, dense fallback, artifact export, and cleanup are mandatory.
- Comparisons against baseline and replay expectations are automatic.

Evidence gate:

- Actuator behavior is externally observable and internally auditable.
- Dense fallback remains available for every run.
- The experiment demonstrates a measurable memory or latency effect without
  unacceptable quality regression.

## Decision Gates

Do not advance from a phase until its evidence gate is satisfied by artifacts,
not by intent.

- Phase 0 to Phase 1: manifest, roadmap, and planner checks pass locally.
- Phase 1 to Phase 2: runtime baselines establish artifact shape and limits.
- Phase 2 to Phase 3: semantic traces exist from a hookable runtime.
- Phase 3 to Phase 4: replay policies show useful behavior against dense
  fallback comparisons.
- Phase 4 to Phase 5: a runtime actuator point exists and has a rollback plan.

## llama.cpp Controller And Fork Options

A controller outside `llama.cpp` can schedule runs, choose probe suites, inspect
Model Plane manifests, analyze artifacts, and recommend expert residency. That
is useful, but it cannot by itself load only routed experts.

Real expert paging needs a runtime actuator, hook, or patch that can pin, evict,
preload, or otherwise alter expert tensor residency or expert offload behavior.
If `llama.cpp` remains the right substrate, prefer the smallest controller
patch or fork that exposes those controls and artifacts. A large custom runtime
is a last resort, not the starting point.

The actuator feasibility spike must prove:

- routing visibility: routed expert ids or sufficient router metadata are
  available at the decision point
- residency control: expert tensors can be pinned, evicted, preloaded, or
  offloaded according to policy
- dense fallback: normal dense/full-residency behavior remains available
- artifact shape: decisions and outcomes can be emitted in
  `memory-moe-bridge-v1`-compatible form
- cleanup: failed or interrupted runs restore runtime state
- isolation: experiments do not require global machine or model-cache mutation

## Model Plane Cron And Callable-Function Handoff

Model Plane owns orchestration surfaces such as callable cron functions, profile
validation, launch/health/log inspection, and run-scoped MoE manifest export.
MoE Run Anyway should consume those manifests and produce a planning artifact,
not reach back into Model Plane internals.

The planned handoff is:

```text
Model Plane callable function or cron -> run-scoped MoE manifest
  -> MoE manifest planner -> planned harness_run_request
  -> user-approved probe or replay command
```

`harness_run_request` is a planning stage, not an execution API yet. It should
name the target class, approved command class, expected artifacts, safety
contract, and missing actuator capabilities.

## Risks

- Timing and endpoint metrics may be overinterpreted as semantic routing.
- A runtime patch may expose routing but still lack safe tensor residency
  control.
- A controller policy can look good in replay while causing live churn or
  cleanup failures.
- Baseline artifacts from different backends may not be comparable without
  strict labels and shared fields.
- Fork maintenance cost can exceed the benefit unless the actuator patch stays
  small.

## Non-Goals

- No claim of live expert paging before an actuator exists.
- No model downloads, token use, Docker runs, model server launch, or prompt
  traffic from roadmap/planner checks.
- No custom runtime rewrite as the first actuator plan.
- No semantic expert-id inference from stock OpenAI-compatible endpoint
  telemetry.
- No destructive cleanup or mutation of user model caches.
